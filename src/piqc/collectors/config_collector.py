"""
Generic configuration collection.

Extracts configuration data from pods that applies to all
inference frameworks.
"""

from dataclasses import dataclass, field
from typing import Optional

from kubernetes.client import V1Pod

from piqc.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ContainerConfig:
    """Configuration extracted from a container."""
    
    image: str
    image_name: str
    image_tag: Optional[str]
    command: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    ports: list[int] = field(default_factory=list)
    
    # Resource requests/limits
    cpu_request: Optional[str] = None
    cpu_limit: Optional[str] = None
    memory_request: Optional[str] = None
    memory_limit: Optional[str] = None
    gpu_count: int = 0


@dataclass
class EndpointInfo:
    """Network endpoint information."""
    
    type: str  # "http", "grpc"
    port: int
    path: Optional[str] = None
    url: Optional[str] = None


class ConfigCollector:
    """
    Collects generic configuration from Kubernetes pods.
    
    Extracts container configuration, environment variables,
    resource requests, and network endpoints.
    """
    
    # Common inference ports
    HTTP_PORTS = {8000, 8080, 80, 5000, 3000}
    GRPC_PORTS = {8001, 50051, 9000}
    
    # GPU resource keys
    GPU_RESOURCE_KEYS = {
        "nvidia.com/gpu",
        "amd.com/gpu",
        "gpu.intel.com/i915",
    }
    
    def collect_container_config(self, pod: V1Pod) -> Optional[ContainerConfig]:
        """
        Extract configuration from the main container of a pod.
        
        Args:
            pod: Kubernetes pod object.
            
        Returns:
            ContainerConfig with extracted data, or None if no containers.
        """
        if not pod.spec or not pod.spec.containers:
            return None
        
        container = pod.spec.containers[0]
        
        # Parse image
        image = container.image or ""
        image_name, image_tag = self._parse_image(image)
        
        # Extract environment variables
        env_vars: dict[str, str] = {}
        if container.env:
            for env in container.env:
                if env.name and env.value:
                    env_vars[env.name] = env.value
        
        # Extract ports
        ports: list[int] = []
        if container.ports:
            for port in container.ports:
                if port.container_port:
                    ports.append(port.container_port)
        
        # Extract resources
        cpu_request = None
        cpu_limit = None
        memory_request = None
        memory_limit = None
        gpu_count = 0
        
        if container.resources:
            requests = container.resources.requests or {}
            limits = container.resources.limits or {}
            
            cpu_request = requests.get("cpu")
            cpu_limit = limits.get("cpu")
            memory_request = requests.get("memory")
            memory_limit = limits.get("memory")
            
            # Count GPUs
            for key in self.GPU_RESOURCE_KEYS:
                if key in limits:
                    try:
                        gpu_count += int(limits[key])
                    except (ValueError, TypeError):
                        pass
        
        return ContainerConfig(
            image=image,
            image_name=image_name,
            image_tag=image_tag,
            command=list(container.command or []),
            args=list(container.args or []),
            env_vars=env_vars,
            ports=ports,
            cpu_request=cpu_request,
            cpu_limit=cpu_limit,
            memory_request=memory_request,
            memory_limit=memory_limit,
            gpu_count=gpu_count,
        )
    
    def detect_endpoints(self, config: ContainerConfig) -> list[EndpointInfo]:
        """
        Detect network endpoints from container configuration.
        
        Args:
            config: Container configuration.
            
        Returns:
            List of detected endpoints.
        """
        endpoints: list[EndpointInfo] = []
        
        for port in config.ports:
            if port in self.HTTP_PORTS:
                endpoints.append(EndpointInfo(
                    type="http",
                    port=port,
                    path="/v1/completions" if port == 8000 else None,
                ))
            elif port in self.GRPC_PORTS:
                endpoints.append(EndpointInfo(
                    type="grpc",
                    port=port,
                ))
            else:
                # Assume HTTP for unknown ports
                endpoints.append(EndpointInfo(
                    type="http",
                    port=port,
                ))
        
        # If no ports declared, add vLLM default
        if not endpoints:
            endpoints.append(EndpointInfo(
                type="http",
                port=8000,
                path="/v1/completions",
            ))
        
        return endpoints
    
    def _parse_image(self, image: str) -> tuple[str, Optional[str]]:
        """
        Parse container image into name and tag.
        
        Args:
            image: Full image string (e.g., "vllm/vllm-openai:v0.2.0").
            
        Returns:
            Tuple of (image_name, image_tag).
        """
        if not image:
            return "", None
        
        # Handle digest format (image@sha256:...)
        if "@" in image:
            image_name = image.split("@")[0]
            return image_name, None
        
        # Handle tag format (image:tag)
        if ":" in image:
            # Handle port numbers in registry (e.g., registry:5000/image:tag)
            parts = image.rsplit(":", 1)
            if "/" in parts[-1]:
                # The last part contains a slash, so it's likely a registry port
                return image, None
            return parts[0], parts[1]
        
        return image, None


def extract_creation_timestamp(pod: V1Pod) -> Optional[str]:
    """
    Extract creation timestamp from pod metadata.
    
    Args:
        pod: Kubernetes pod object.
        
    Returns:
        ISO 8601 formatted timestamp, or None.
    """
    if pod.metadata and pod.metadata.creation_timestamp:
        return pod.metadata.creation_timestamp.isoformat()
    return None


def get_node_name(pod: V1Pod) -> Optional[str]:
    """
    Get the node name where the pod is running.
    
    Args:
        pod: Kubernetes pod object.
        
    Returns:
        Node name, or None if not scheduled.
    """
    if pod.spec:
        return pod.spec.node_name
    return None
