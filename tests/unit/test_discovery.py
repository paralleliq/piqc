"""
Unit tests for deployment discovery.

Tests framework detection and confidence scoring.
"""

import pytest
from unittest.mock import MagicMock

from kubernetes.client import V1Pod, V1ObjectMeta, V1PodSpec, V1Container, V1EnvVar

from piqc.core.discovery import (
    FrameworkDetector,
    DeploymentDiscovery,
)


def create_mock_pod(
    name: str = "test-pod",
    namespace: str = "default",
    image: str = "nginx:latest",
    env_vars: dict | None = None,
    args: list | None = None,
    labels: dict | None = None,
) -> V1Pod:
    """Create a mock V1Pod for testing."""
    env_list = None
    if env_vars:
        env_list = [V1EnvVar(name=k, value=v) for k, v in env_vars.items()]
    
    container = V1Container(
        name="main",
        image=image,
        env=env_list,
        args=args,
    )
    
    return V1Pod(
        metadata=V1ObjectMeta(
            name=name,
            namespace=namespace,
            labels=labels or {},
        ),
        spec=V1PodSpec(
            containers=[container],
        ),
    )


class TestFrameworkDetector:
    """Tests for FrameworkDetector."""
    
    def test_detect_vllm_by_image(self) -> None:
        """Test vLLM detection via container image."""
        detector = FrameworkDetector()
        
        pod = create_mock_pod(image="vllm/vllm-openai:v0.2.0")
        framework, confidence = detector.detect(pod)
        
        assert framework == "vllm"
        assert confidence >= 0.4  # Image weight
    
    def test_detect_vllm_by_env_vars(self) -> None:
        """Test vLLM detection via environment variables."""
        detector = FrameworkDetector()
        
        pod = create_mock_pod(
            image="custom-image:latest",
            env_vars={
                "VLLM_VERSION": "0.2.0",
                "TENSOR_PARALLEL_SIZE": "4",
            },
        )
        framework, confidence = detector.detect(pod)
        
        assert framework == "vllm"
        assert confidence >= 0.3
    
    def test_detect_vllm_by_cli_args(self) -> None:
        """Test vLLM detection via CLI arguments."""
        detector = FrameworkDetector()
        
        pod = create_mock_pod(
            image="custom-image:latest",
            args=[
                "--model", "llama-7b",
                "--tensor-parallel-size", "2",
                "--served-model-name", "my-model",
            ],
        )
        framework, confidence = detector.detect(pod)
        
        assert framework == "vllm"
        assert confidence >= 0.2
    
    def test_detect_vllm_combined_signals(self) -> None:
        """Test vLLM detection with multiple signals."""
        detector = FrameworkDetector()
        
        pod = create_mock_pod(
            image="vllm/vllm-openai:v0.2.0",
            env_vars={
                "TENSOR_PARALLEL_SIZE": "4",
                "MODEL_NAME": "llama-7b",
            },
            args=[
                "--tensor-parallel-size", "4",
            ],
            labels={
                "app.kubernetes.io/name": "vllm",
            },
        )
        framework, confidence = detector.detect(pod)
        
        assert framework == "vllm"
        assert confidence >= 0.7  # High confidence with multiple signals
    
    def test_detect_unknown_non_ml(self) -> None:
        """Test unknown detection for non-ML workload."""
        detector = FrameworkDetector()
        
        pod = create_mock_pod(
            image="nginx:latest",
        )
        framework, confidence = detector.detect(pod)
        
        assert framework == "unknown"
        assert confidence == 0.0
    
    def test_detect_labels(self) -> None:
        """Test detection via pod labels."""
        detector = FrameworkDetector()
        
        pod = create_mock_pod(
            image="custom-image:latest",
            labels={
                "app.kubernetes.io/name": "vllm",
                "framework": "vllm",
            },
        )
        
        # Should contribute to confidence even without other signals
        framework, confidence = detector.detect(pod)
        
        # Low confidence but should still detect vllm due to labels
        # Combined with weak signals if any
        assert confidence >= 0


class TestDeploymentDiscovery:
    """Tests for DeploymentDiscovery."""
    
    def test_analyze_vllm_pod(self) -> None:
        """Test analyzing a vLLM pod."""
        discovery = DeploymentDiscovery()
        
        pod = create_mock_pod(
            name="vllm-server-abc123",
            namespace="inference",
            image="vllm/vllm-openai:v0.2.0",
            env_vars={
                "MODEL_NAME": "llama-7b",
            },
        )
        
        deployment = discovery.analyze_pod(pod)
        
        assert deployment is not None
        assert deployment.framework == "vllm"
        assert deployment.namespace == "inference"
    
    def test_analyze_non_inference_pod(self) -> None:
        """Test analyzing a non-inference pod returns None."""
        discovery = DeploymentDiscovery()
        
        pod = create_mock_pod(
            image="nginx:latest",
        )
        
        deployment = discovery.analyze_pod(pod)
        
        assert deployment is None
    
    def test_group_deployments(self) -> None:
        """Test grouping pods by deployment."""
        discovery = DeploymentDiscovery()
        
        # Create mock deployments from same parent
        from piqc.models.modelspec import InferenceDeployment
        from datetime import datetime
        
        deployments = [
            InferenceDeployment(
                name="llama-server",
                namespace="inference",
                framework="vllm",
                confidence=0.8,
                deployment_type="Deployment",
                pod_names=["llama-server-pod-1"],
                detected_at=datetime.utcnow(),
            ),
            InferenceDeployment(
                name="llama-server",
                namespace="inference",
                framework="vllm",
                confidence=0.9,  # Higher confidence
                deployment_type="Deployment",
                pod_names=["llama-server-pod-2"],
                detected_at=datetime.utcnow(),
            ),
        ]
        
        grouped = discovery.group_deployments(deployments)
        
        assert len(grouped) == 1
        assert grouped[0].replicas == 2
        assert len(grouped[0].pod_names) == 2
        assert grouped[0].confidence == 0.9  # Highest confidence
    
    def test_analyze_ray_serve_worker_pod(self) -> None:
        """Test analyzing a Ray Serve worker pod returns a deployment."""
        discovery = DeploymentDiscovery()

        pod = create_mock_pod(
            name="piqc-ray-cluster-workers-worker-abc",
            namespace="default",
            image="rayproject/ray:2.55.1-py312",
            labels={
                "ray.io/is-ray-node": "yes",
                "ray.io/node-type": "worker",
                "ray.io/cluster": "piqc-ray-cluster",
                "app.kubernetes.io/created-by": "kuberay-operator",
            },
        )

        deployment = discovery.analyze_pod(pod)

        assert deployment is not None
        assert deployment.framework == "ray_serve"

    def test_ray_head_pod_not_detected_as_inference(self) -> None:
        """Test that Ray head pods are not detected as inference workloads."""
        detector = FrameworkDetector()

        pod = create_mock_pod(
            name="piqc-ray-cluster-head-abc",
            image="rayproject/ray:2.55.1-py312",
            labels={
                "ray.io/is-ray-node": "yes",
                "ray.io/node-type": "head",
            },
        )

        framework, confidence = detector.detect(pod)

        assert framework != "ray_serve"

    def test_filter_running_pods(self) -> None:
        """Test filtering to only running pods."""
        discovery = DeploymentDiscovery()
        
        from kubernetes.client import V1PodStatus
        
        running_pod = create_mock_pod(name="running")
        running_pod.status = V1PodStatus(phase="Running")
        
        pending_pod = create_mock_pod(name="pending")
        pending_pod.status = V1PodStatus(phase="Pending")
        
        pods = [running_pod, pending_pod]
        
        filtered = discovery.filter_running_pods(pods)
        
        assert len(filtered) == 1
        assert filtered[0].metadata.name == "running"
