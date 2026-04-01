"""
Deployment discovery and framework detection.

This module is responsible for discovering AI/ML inference workloads
in Kubernetes and detecting the inference framework being used,
with a primary focus on vLLM detection.
"""

import re
from datetime import datetime
from typing import Optional

from kubernetes.client import V1Pod, V1Container

from piqc.models.modelspec import InferenceDeployment
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# vLLM Detection Patterns (PRIMARY - 80% of effort)
# =============================================================================

# Container image patterns for vLLM (weight: 0.40)
VLLM_IMAGE_PATTERNS = [
    re.compile(r"^vllm/vllm-openai:.*$", re.IGNORECASE),
    re.compile(r"^vllm/vllm:.*$", re.IGNORECASE),
    re.compile(r"^ghcr\.io/vllm-project/vllm:.*$", re.IGNORECASE),
    re.compile(r".*vllm.*:.*", re.IGNORECASE),
]

# Environment variables that indicate vLLM (weight: 0.30)
VLLM_ENV_INDICATORS = {
    "VLLM_VERSION",
    "VLLM_ENGINE",
    "SERVED_MODEL_NAME",
    "TENSOR_PARALLEL_SIZE",
    "PIPELINE_PARALLEL_SIZE",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_USE_MODELSCOPE",
}

# Environment variables that suggest ML/inference but not vLLM-specific
VLLM_ENV_HINTS = {
    "MODEL_NAME",
    "MAX_MODEL_LEN",
    "GPU_MEMORY_UTILIZATION",
    "DTYPE",
    "QUANTIZATION",
}

# CLI arguments that indicate vLLM (weight: 0.20)
VLLM_CLI_INDICATORS = {
    "--model",
    "--served-model-name",
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--max-model-len",
    "--max-num-batched-tokens",
    "--max-num-seqs",
    "--gpu-memory-utilization",
    "--quantization",
    "--disable-log-stats",
    "--trust-remote-code",
    "--download-dir",
    "--load-format",
    "--kv-cache-dtype",
    "--enforce-eager",
}

# vLLM entrypoint patterns
VLLM_ENTRYPOINT_PATTERNS = [
    re.compile(r"vllm\.entrypoints\.", re.IGNORECASE),
    re.compile(r"python\s+-m\s+vllm", re.IGNORECASE),
]

# Pod labels that indicate vLLM (weight: 0.10)
VLLM_LABEL_INDICATORS = {
    "app.kubernetes.io/name": {"vllm", "vllm-openai"},
    "app": {"vllm", "vllm-openai", "vllm-server"},
    "framework": {"vllm"},
    "inference-engine": {"vllm"},
}

# =============================================================================
# GPU Resource Keys (for ML workload detection)
# =============================================================================

# =============================================================================
# Generic ML Indicators (for filtering non-ML workloads)
# =============================================================================

ML_IMAGE_PATTERNS = [
    re.compile(r".*pytorch.*", re.IGNORECASE),
    re.compile(r".*tensorflow.*", re.IGNORECASE),
    re.compile(r".*cuda.*", re.IGNORECASE),
    re.compile(r".*nvidia.*", re.IGNORECASE),
    re.compile(r".*transformers.*", re.IGNORECASE),
    re.compile(r".*huggingface.*", re.IGNORECASE),
    re.compile(r".*llm.*", re.IGNORECASE),
    re.compile(r".*inference.*", re.IGNORECASE),
]

GPU_RESOURCE_KEYS = {
    "nvidia.com/gpu",
    "amd.com/gpu",
    "gpu.intel.com/i915",
}

# =============================================================================
# Known GPU infrastructure pods to exclude (not inference workloads)
# =============================================================================

# Image substrings that identify GPU system/monitoring daemons, not inference
INFRA_IMAGE_PATTERNS = [
    re.compile(r"dcgm.exporter", re.IGNORECASE),
    re.compile(r"nvidia.device.plugin", re.IGNORECASE),
    re.compile(r"nvidia-device-plugin", re.IGNORECASE),
    re.compile(r"gpu-feature-discovery", re.IGNORECASE),
    re.compile(r"node-feature-discovery", re.IGNORECASE),
    re.compile(r"k8s-device-plugin", re.IGNORECASE),
]

# Pod name prefixes that identify well-known GPU infrastructure DaemonSets
INFRA_NAME_PATTERNS = [
    re.compile(r"^dcgm-exporter", re.IGNORECASE),
    re.compile(r"^nvidia-gpu-device-plugin", re.IGNORECASE),
    re.compile(r"^gpu-feature-discovery", re.IGNORECASE),
    re.compile(r"^node-feature-discovery", re.IGNORECASE),
]


class FrameworkDetector:
    """
    Detects the inference framework used by a Kubernetes pod.
    
    Uses multiple signals with weighted confidence scoring to
    accurately identify the framework, with primary focus on vLLM.
    """
    
    # Confidence weights for different signals
    WEIGHT_IMAGE = 0.40
    WEIGHT_ENV = 0.30
    WEIGHT_CLI = 0.20
    WEIGHT_LABELS = 0.10
    
    def detect(self, pod: V1Pod) -> tuple[str, float]:
        """
        Detect vLLM inference framework for a pod.
        
        Args:
            pod: Kubernetes pod object.
            
        Returns:
            Tuple of (framework_name, confidence_score).
            Framework is "vllm" or "unknown".
            Confidence is a float from 0.0 to 1.0.
        """
        # Exclude known GPU infrastructure daemons (dcgm-exporter, device plugins, etc.)
        if self._is_gpu_infra_pod(pod):
            return "unknown", 0.0

        # Check vLLM
        vllm_confidence = self._calculate_vllm_confidence(pod)
        if vllm_confidence >= 0.2:  # Threshold for positive detection
            return "vllm", vllm_confidence

        # Check if it's at least an ML workload
        if self._is_ml_workload(pod):
            return "unknown", 0.1

        return "unknown", 0.0
    
    def _calculate_vllm_confidence(self, pod: V1Pod) -> float:
        """Calculate confidence score for vLLM detection."""
        confidence = 0.0
        
        containers = self._get_all_containers(pod)
        
        for container in containers:
            # Check image pattern (weight: 0.40)
            if self._matches_patterns(container.image or "", VLLM_IMAGE_PATTERNS):
                confidence += self.WEIGHT_IMAGE
                break
        
        # Check environment variables (weight: 0.30)
        env_vars = self._extract_env_var_names(pod)
        vllm_env_count = len(env_vars.intersection(VLLM_ENV_INDICATORS))
        hint_env_count = len(env_vars.intersection(VLLM_ENV_HINTS))
        
        if vllm_env_count > 0:
            # Strong indicator
            confidence += self.WEIGHT_ENV * min(1.0, vllm_env_count / 2)
        elif hint_env_count >= 2:
            # Weak indicator - partial weight
            confidence += self.WEIGHT_ENV * 0.3
        
        # Check CLI arguments (weight: 0.20)
        # Multiple CLI args strongly indicate vLLM
        cli_args = self._extract_cli_args(pod)
        vllm_cli_count = len(cli_args.intersection(VLLM_CLI_INDICATORS))
        if vllm_cli_count > 0:
            # 2+ CLI args gives full weight, enabling CLI-only detection
            confidence += self.WEIGHT_CLI * min(1.0, vllm_cli_count / 2)
        
        # Check for vLLM entrypoint in command
        for container in containers:
            command = " ".join(container.command or []) + " " + " ".join(container.args or [])
            for pattern in VLLM_ENTRYPOINT_PATTERNS:
                if pattern.search(command):
                    confidence += self.WEIGHT_CLI * 0.5
                    break
        
        # Check labels (weight: 0.10)
        if self._matches_label_indicators(pod, VLLM_LABEL_INDICATORS):
            confidence += self.WEIGHT_LABELS
        
        return min(confidence, 1.0)
    

    
    def _is_gpu_infra_pod(self, pod: V1Pod) -> bool:
        """Return True if this pod is a GPU infrastructure daemon, not an inference workload."""
        pod_name = (pod.metadata.name or "") if pod.metadata else ""
        for pattern in INFRA_NAME_PATTERNS:
            if pattern.search(pod_name):
                return True
        for container in self._get_all_containers(pod):
            for pattern in INFRA_IMAGE_PATTERNS:
                if pattern.search(container.image or ""):
                    return True
        return False

    def _is_ml_workload(self, pod: V1Pod) -> bool:
        """Check if the pod is likely an ML workload."""
        # Check for GPU resources
        containers = self._get_all_containers(pod)
        for container in containers:
            if container.resources:
                requests = container.resources.requests or {}
                limits = container.resources.limits or {}
                all_resources = set(requests.keys()) | set(limits.keys())
                if all_resources.intersection(GPU_RESOURCE_KEYS):
                    return True
        
        # Check image patterns
        for container in containers:
            if self._matches_patterns(container.image or "", ML_IMAGE_PATTERNS):
                return True
        
        return False
    
    def _get_all_containers(self, pod: V1Pod) -> list[V1Container]:
        """Get all containers from pod spec."""
        if not pod.spec:
            return []
        containers = list(pod.spec.containers or [])
        containers.extend(pod.spec.init_containers or [])
        return containers
    
    def _matches_patterns(self, value: str, patterns: list[re.Pattern]) -> bool:
        """Check if value matches any pattern."""
        for pattern in patterns:
            if pattern.search(value):
                return True
        return False
    
    def _extract_env_var_names(self, pod: V1Pod) -> set[str]:
        """Extract all environment variable names from pod."""
        env_names: set[str] = set()
        for container in self._get_all_containers(pod):
            if container.env:
                for env in container.env:
                    if env.name:
                        env_names.add(env.name)
        return env_names
    
    def _extract_cli_args(self, pod: V1Pod) -> set[str]:
        """Extract CLI argument flags from container commands."""
        cli_args: set[str] = set()
        for container in self._get_all_containers(pod):
            args = (container.command or []) + (container.args or [])
            for arg in args:
                if arg.startswith("--"):
                    # Extract just the flag name (before =)
                    flag = arg.split("=")[0]
                    cli_args.add(flag)
        return cli_args
    
    def _matches_label_indicators(
        self,
        pod: V1Pod,
        indicators: dict[str, set[str]],
    ) -> bool:
        """Check if pod labels match any indicator."""
        if not pod.metadata or not pod.metadata.labels:
            return False
        
        labels = pod.metadata.labels
        for key, values in indicators.items():
            if key in labels and labels[key].lower() in {v.lower() for v in values}:
                return True
        return False


class DeploymentDiscovery:
    """
    Discovers AI/ML inference deployments in Kubernetes.
    
    Scans namespaces for pods running inference workloads
    and groups them by their parent deployment/statefulset.
    """
    
    def __init__(self) -> None:
        """Initialize the discovery service."""
        self.detector = FrameworkDetector()
    
    def analyze_pod(self, pod: V1Pod) -> Optional[InferenceDeployment]:
        """
        Analyze a single pod for inference workload detection.
        
        Args:
            pod: Kubernetes pod object.
            
        Returns:
            InferenceDeployment if detected, None otherwise.
        """
        if not pod.metadata:
            return None
        
        # Detect framework
        framework, confidence = self.detector.detect(pod)
        
        # Skip non-inference workloads
        if framework == "unknown" and confidence == 0.0:
            return None
        
        # Extract pod information
        pod_name = pod.metadata.name or "unknown"
        namespace = pod.metadata.namespace or "default"
        labels = dict(pod.metadata.labels or {})
        
        # Determine parent deployment
        deployment_name = self._get_owner_name(pod) or pod_name
        deployment_type = self._get_owner_kind(pod) or "Pod"
        
        # Extract container details
        container = self._get_main_container(pod)
        container_image = container.image if container else None
        container_args = list((container.command or []) + (container.args or [])) if container else []
        
        # Extract environment variables
        env_vars = self._extract_env_vars(pod)
        
        # Extract resource requests
        gpu_count = self._get_gpu_count(pod)
        cpu_request = self._get_resource_request(pod, "cpu")
        memory_request = self._get_resource_request(pod, "memory")
        
        return InferenceDeployment(
            name=deployment_name,
            namespace=namespace,
            framework=framework,
            confidence=confidence,
            deployment_type=deployment_type,
            replicas=1,  # Will be updated during grouping
            pod_names=[pod_name],
            detected_at=datetime.utcnow(),
            container_image=container_image,
            container_args=container_args,
            env_vars=env_vars,
            labels=labels,
            gpu_count=gpu_count,
            cpu_request=cpu_request,
            memory_request=memory_request,
        )
    
    def group_deployments(
        self,
        deployments: list[InferenceDeployment],
    ) -> list[InferenceDeployment]:
        """
        Group inference deployments by name and namespace.
        
        Merges pods from the same deployment into a single entry
        with accurate replica counts.
        
        Args:
            deployments: List of individual deployments.
            
        Returns:
            Grouped list of deployments.
        """
        grouped: dict[tuple[str, str], InferenceDeployment] = {}
        
        for deployment in deployments:
            key = (deployment.namespace, deployment.name)
            
            if key in grouped:
                existing = grouped[key]
                # Merge pod names
                existing.pod_names.extend(deployment.pod_names)
                existing.replicas = len(existing.pod_names)
                # Use highest confidence
                if deployment.confidence > existing.confidence:
                    existing.confidence = deployment.confidence
                    existing.framework = deployment.framework
            else:
                grouped[key] = deployment
        
        return list(grouped.values())
    
    def filter_running_pods(self, pods: list[V1Pod]) -> list[V1Pod]:
        """
        Filter pods to only include running ones.
        
        Args:
            pods: List of pods to filter.
            
        Returns:
            List of pods in Running phase.
        """
        return [
            pod for pod in pods
            if pod.status and pod.status.phase == "Running"
        ]
    
    def _get_owner_name(self, pod: V1Pod) -> Optional[str]:
        """Get the name of the pod's owner deployment/statefulset."""
        if not pod.metadata or not pod.metadata.owner_references:
            return None
        
        for owner in pod.metadata.owner_references:
            if owner.kind in ("ReplicaSet", "StatefulSet"):
                # For ReplicaSet, try to get the Deployment name
                if owner.kind == "ReplicaSet" and owner.name:
                    # ReplicaSet names are usually deployment-name-hash
                    parts = owner.name.rsplit("-", 1)
                    if len(parts) == 2:
                        return parts[0]
                return owner.name
        
        return None
    
    def _get_owner_kind(self, pod: V1Pod) -> Optional[str]:
        """Get the kind of the pod's owner."""
        if not pod.metadata or not pod.metadata.owner_references:
            return None
        
        for owner in pod.metadata.owner_references:
            if owner.kind == "ReplicaSet":
                return "Deployment"
            elif owner.kind in ("StatefulSet", "DaemonSet"):
                return owner.kind
        
        return None
    
    def _get_main_container(self, pod: V1Pod) -> Optional[V1Container]:
        """Get the main container from a pod."""
        if not pod.spec or not pod.spec.containers:
            return None
        return pod.spec.containers[0]
    
    def _extract_env_vars(self, pod: V1Pod) -> dict[str, str]:
        """Extract environment variables from pod containers."""
        env_vars: dict[str, str] = {}
        
        if not pod.spec:
            return env_vars
        
        for container in pod.spec.containers or []:
            if container.env:
                for env in container.env:
                    if env.name and env.value:
                        env_vars[env.name] = env.value
        
        return env_vars
    
    def _get_gpu_count(self, pod: V1Pod) -> int:
        """Get the total GPU count requested by the pod."""
        if not pod.spec:
            return 0
        
        total_gpus = 0
        for container in pod.spec.containers or []:
            if container.resources:
                limits = container.resources.limits or {}
                for key in GPU_RESOURCE_KEYS:
                    if key in limits:
                        try:
                            total_gpus += int(limits[key])
                        except (ValueError, TypeError):
                            pass
        
        return total_gpus
    
    def _get_resource_request(
        self,
        pod: V1Pod,
        resource_type: str,
    ) -> Optional[str]:
        """Get resource request value for a pod."""
        if not pod.spec or not pod.spec.containers:
            return None
        
        container = pod.spec.containers[0]
        if not container.resources or not container.resources.requests:
            return None
        
        return container.resources.requests.get(resource_type)
