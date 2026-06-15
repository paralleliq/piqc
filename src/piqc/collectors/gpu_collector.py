"""
GPU metrics collection using nvidia-smi.

Collects real-time GPU metrics from pods by executing nvidia-smi
inside containers with NVIDIA GPUs.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from piqc.core.k8s_client import K8sClient
from piqc.utils.exceptions import GPUMetricsUnavailableError
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class GPUMetrics:
    """Real-time metrics for a single GPU."""
    
    gpu_index: int
    gpu_model: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: int
    memory_utilization_percent: int
    temperature_celsius: int
    power_draw_watts: int
    power_limit_watts: int
    collection_timestamp: datetime
    
    def to_memory_str(self, mb_value: int) -> str:
        """Convert MB to human-readable string."""
        if mb_value >= 1024:
            return f"{mb_value // 1024}GB"
        return f"{mb_value}MB"


class GPUCollector:
    """
    Collects GPU metrics from Kubernetes pods.
    
    Uses nvidia-smi to gather real-time GPU information including
    utilization, memory usage, temperature, and power consumption.
    
    Gracefully degrades if nvidia-smi is unavailable.
    """
    
    # nvidia-smi paths to try — GKE mounts it at /usr/local/nvidia/bin/
    NVIDIA_SMI_PATHS = [
        "nvidia-smi",
        "/usr/local/nvidia/bin/nvidia-smi",
        "/usr/bin/nvidia-smi",
    ]

    # nvidia-smi command for CSV output
    NVIDIA_SMI_COMMAND = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]

    # Alternative command if the detailed query fails
    NVIDIA_SMI_SIMPLE_COMMAND = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    
    def __init__(self, k8s_client: K8sClient, exec_timeout: int = 10) -> None:
        """
        Initialize the GPU collector.
        
        Args:
            k8s_client: Kubernetes client instance.
            exec_timeout: Timeout for nvidia-smi execution in seconds.
        """
        self.k8s_client = k8s_client
        self.exec_timeout = exec_timeout
    
    def collect(
        self,
        pod_name: str,
        namespace: str,
        container: Optional[str] = None,
    ) -> Optional[list[GPUMetrics]]:
        """
        Collect GPU metrics from a pod.
        
        Args:
            pod_name: Name of the target pod.
            namespace: Namespace of the pod.
            container: Specific container name. If None, uses default.
            
        Returns:
            List of GPUMetrics if successful, None if nvidia-smi unavailable.
        """
        _NOT_FOUND_TERMS = ["not found", "command not found", "no such file", "exec failed", "executable file"]

        for smi_path in self.NVIDIA_SMI_PATHS:
            command = [smi_path] + self.NVIDIA_SMI_COMMAND[1:]
            try:
                stdout, stderr = self.k8s_client.exec_in_pod(
                    pod_name=pod_name,
                    namespace=namespace,
                    command=command,
                    container=container,
                    timeout=self.exec_timeout,
                )

                # stderr "not found" means the binary doesn't exist at this path — try next
                if any(term in stderr.lower() for term in _NOT_FOUND_TERMS):
                    logger.debug(f"{smi_path} not found in {namespace}/{pod_name}, trying next path")
                    continue

                if not stdout.strip():
                    logger.debug(f"Empty nvidia-smi output from {namespace}/{pod_name}")
                    return self._try_simple_query(pod_name, namespace, container, smi_path)

                return self._parse_nvidia_smi_output(stdout)

            except GPUMetricsUnavailableError:
                raise
            except Exception as e:
                error_str = str(e).lower()

                if "permission denied" in error_str:
                    logger.debug(f"nvidia-smi permission denied in {namespace}/{pod_name}")
                    raise GPUMetricsUnavailableError(
                        pod_name=pod_name,
                        reason="Permission denied for nvidia-smi",
                    )

                if any(term in error_str for term in _NOT_FOUND_TERMS):
                    logger.debug(f"{smi_path} not found in {namespace}/{pod_name}, trying next path")
                    continue

                logger.warning(f"GPU metrics collection failed for {namespace}/{pod_name}: {e}")
                raise GPUMetricsUnavailableError(
                    pod_name=pod_name,
                    reason=str(e),
                )

        raise GPUMetricsUnavailableError(
            pod_name=pod_name,
            reason="nvidia-smi not found at any known path",
        )
    
    def _try_simple_query(
        self,
        pod_name: str,
        namespace: str,
        container: Optional[str],
        smi_path: str = "nvidia-smi",
    ) -> Optional[list[GPUMetrics]]:
        """Try a simpler nvidia-smi query as fallback."""
        try:
            command = [smi_path] + self.NVIDIA_SMI_SIMPLE_COMMAND[1:]
            stdout, stderr = self.k8s_client.exec_in_pod(
                pod_name=pod_name,
                namespace=namespace,
                command=command,
                container=container,
                timeout=self.exec_timeout,
            )
            
            if not stdout.strip():
                return None
            
            return self._parse_simple_output(stdout)
            
        except Exception as e:
            logger.debug(f"Simple nvidia-smi query also failed: {e}")
            return None
    
    def _parse_nvidia_smi_output(self, output: str) -> list[GPUMetrics]:
        """
        Parse full nvidia-smi CSV output.
        
        Expected format:
        index, name, memory.total, memory.used, memory.free,
        utilization.gpu, utilization.memory, temperature.gpu, power.draw, power.limit
        
        Example:
        0, NVIDIA A100-SXM4-80GB, 81920, 45120, 36800, 87, 55, 72, 320, 400
        """
        metrics: list[GPUMetrics] = []
        timestamp = datetime.utcnow()
        
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            
            parts = [p.strip() for p in line.split(",")]
            
            if len(parts) < 10:
                logger.debug(f"Skipping incomplete nvidia-smi line: {line}")
                continue
            
            try:
                metrics.append(GPUMetrics(
                    gpu_index=int(parts[0]),
                    gpu_model=self._clean_gpu_name(parts[1]),
                    memory_total_mb=int(float(parts[2])),
                    memory_used_mb=int(float(parts[3])),
                    memory_free_mb=int(float(parts[4])),
                    utilization_percent=int(float(parts[5])),
                    memory_utilization_percent=int(float(parts[6])),
                    temperature_celsius=int(float(parts[7])),
                    power_draw_watts=int(float(parts[8])),
                    power_limit_watts=int(float(parts[9])),
                    collection_timestamp=timestamp,
                ))
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse nvidia-smi line '{line}': {e}")
                continue
        
        return metrics
    
    def _parse_simple_output(self, output: str) -> list[GPUMetrics]:
        """Parse simple nvidia-smi output with limited fields."""
        metrics: list[GPUMetrics] = []
        timestamp = datetime.utcnow()
        
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            
            parts = [p.strip() for p in line.split(",")]
            
            if len(parts) < 3:
                continue
            
            try:
                memory_total = int(float(parts[2]))
                metrics.append(GPUMetrics(
                    gpu_index=int(parts[0]),
                    gpu_model=self._clean_gpu_name(parts[1]),
                    memory_total_mb=memory_total,
                    memory_used_mb=0,
                    memory_free_mb=memory_total,
                    utilization_percent=0,
                    memory_utilization_percent=0,
                    temperature_celsius=0,
                    power_draw_watts=0,
                    power_limit_watts=0,
                    collection_timestamp=timestamp,
                ))
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse simple nvidia-smi line '{line}': {e}")
                continue
        
        return metrics
    
    def _clean_gpu_name(self, name: str) -> str:
        """Clean up GPU name for display."""
        # Remove "NVIDIA " prefix if present
        name = re.sub(r"^NVIDIA\s+", "", name, flags=re.IGNORECASE)
        return name.strip()


def get_gpu_info_from_pod_spec(
    gpu_count: int,
    gpu_resource_key: str = "nvidia.com/gpu",
) -> list[dict]:
    """
    Create placeholder GPU info when nvidia-smi is unavailable.
    
    Args:
        gpu_count: Number of GPUs from pod resource requests.
        gpu_resource_key: The resource key used for GPU allocation.
        
    Returns:
        List of placeholder GPU info dictionaries.
    """
    gpu_type = "Unknown"
    if "nvidia" in gpu_resource_key.lower():
        gpu_type = "NVIDIA GPU"
    elif "amd" in gpu_resource_key.lower():
        gpu_type = "AMD GPU"
    
    return [
        {
            "index": i,
            "type": gpu_type,
            "memory_total": "Unknown",
            "memory_used": None,
            "utilization": None,
            "temperature": None,
            "power_draw": None,
        }
        for i in range(gpu_count)
    ]
