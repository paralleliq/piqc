"""
Unit tests for ModelSpec Pydantic models.

Tests schema validation and serialization.
"""

import pytest
from datetime import datetime

from piqc.models.modelspec import (
    ModelSpec,
    MetadataInfo,
    ModelInfo,
    EngineInfo,
    InferenceConfig,
    GPUInfo,
    ResourceInfo,
    Endpoint,
    KubernetesMetadata,
    CollectionMetadata,
    DataCompleteness,
    InferenceDeployment,
)


class TestModelSpec:
    """Tests for ModelSpec model."""
    
    def test_create_minimal_modelspec(self) -> None:
        """Test creating a minimal valid ModelSpec."""
        modelspec = ModelSpec(
            metadata=MetadataInfo(
                name="test-deployment",
                namespace="default",
                collection_timestamp="2024-01-01T00:00:00Z",
                collector_version="1.0.0",
            ),
            model=ModelInfo(),
            engine=EngineInfo(name="vllm"),
            inference=InferenceConfig(),
            resources=ResourceInfo(replicas=1),
            kubernetes=KubernetesMetadata(
                deployment_type="Deployment",
                image="vllm/vllm-openai:latest",
            ),
            collection=CollectionMetadata(
                mode="remote",
                duration="10s",
                data_completeness=DataCompleteness(),
            ),
        )
        
        assert modelspec.api_version == "modelspec.paralleliq.ai/v1"
        assert modelspec.kind == "ModelSpec"
        assert modelspec.metadata.name == "test-deployment"
    
    def test_to_dict_by_alias(self) -> None:
        """Test serialization with camelCase aliases."""
        modelspec = ModelSpec(
            metadata=MetadataInfo(
                name="test",
                namespace="default",
                collection_timestamp="2024-01-01T00:00:00Z",
                collector_version="1.0.0",
            ),
            model=ModelInfo(
                name="llama-7b",
                served_name="my-llama",
            ),
            engine=EngineInfo(
                name="vllm",
                detection_confidence=0.95,
            ),
            inference=InferenceConfig(
                tensor_parallel_size=4,
                max_model_len=4096,
            ),
            resources=ResourceInfo(replicas=2),
            kubernetes=KubernetesMetadata(
                deployment_type="Deployment",
                image="vllm/vllm:latest",
            ),
            collection=CollectionMetadata(
                mode="remote",
                duration="5s",
                data_completeness=DataCompleteness(),
            ),
        )
        
        data = modelspec.to_dict(by_alias=True)
        
        # Check aliases
        assert "apiVersion" in data
        assert "collectionTimestamp" in data["metadata"]
        assert "servedName" in data["model"]
        assert "detectionConfidence" in data["engine"]
        assert "tensorParallelSize" in data["inference"]
        assert "deploymentType" in data["kubernetes"]
        assert "dataCompleteness" in data["collection"]


class TestGPUInfo:
    """Tests for GPUInfo model."""
    
    def test_create_gpu_info(self) -> None:
        """Test creating GPU info."""
        gpu = GPUInfo(
            index=0,
            type="A100-SXM4-80GB",
            memory_total="80GB",
            memory_used="45GB",
            utilization=87,
            temperature=72,
            power_draw=320,
        )
        
        assert gpu.index == 0
        assert gpu.type == "A100-SXM4-80GB"
        assert gpu.utilization == 87
    
    def test_utilization_validation(self) -> None:
        """Test utilization must be between 0 and 100."""
        with pytest.raises(ValueError):
            GPUInfo(
                index=0,
                type="A100",
                memory_total="80GB",
                utilization=150,  # Invalid
            )


class TestInferenceDeployment:
    """Tests for InferenceDeployment model."""
    
    def test_create_deployment(self) -> None:
        """Test creating an inference deployment."""
        deployment = InferenceDeployment(
            name="llama-server",
            namespace="inference",
            framework="vllm",
            confidence=0.95,
            deployment_type="Deployment",
            replicas=3,
            pod_names=["pod-1", "pod-2", "pod-3"],
        )
        
        assert deployment.name == "llama-server"
        assert deployment.framework == "vllm"
        assert deployment.replicas == 3
        assert len(deployment.pod_names) == 3
    
    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        deployment = InferenceDeployment(
            name="test",
            namespace="default",
            framework="unknown",
            deployment_type="Deployment",
        )
        
        assert deployment.confidence == 0.0
        assert deployment.replicas == 1
        assert deployment.pod_names == []
        assert deployment.gpu_count == 0


class TestEndpoint:
    """Tests for Endpoint model."""
    
    def test_create_http_endpoint(self) -> None:
        """Test creating an HTTP endpoint."""
        endpoint = Endpoint(
            type="http",
            port=8000,
            path="/v1/completions",
        )
        
        assert endpoint.type == "http"
        assert endpoint.port == 8000
    
    def test_create_grpc_endpoint(self) -> None:
        """Test creating a gRPC endpoint."""
        endpoint = Endpoint(
            type="grpc",
            port=8001,
        )
        
        assert endpoint.type == "grpc"
        assert endpoint.path is None


class TestDataCompleteness:
    """Tests for DataCompleteness model."""
    
    def test_default_values(self) -> None:
        """Test default completeness flags."""
        completeness = DataCompleteness()
        
        assert completeness.static_config is True
        assert completeness.gpu_metrics is False
        assert completeness.runtime_metrics is False
    
    def test_serialization(self) -> None:
        """Test serialization with aliases."""
        completeness = DataCompleteness(
            static_config=True,
            gpu_metrics=True,
            runtime_metrics=False,
        )
        
        data = completeness.model_dump(by_alias=True)
        
        assert "staticConfig" in data
        assert "gpuMetrics" in data
        assert "runtimeMetrics" in data
