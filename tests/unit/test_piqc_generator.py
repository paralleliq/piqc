"""
Unit tests for PIQC generator.

Tests fact extraction, provenance tracking, and factErrors generation.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from piqc.generators.piqc_generator import PIQCGenerator
from piqc.models.modelspec import (
    CollectionMetadata,
    DataCompleteness,
    Endpoint,
    EngineInfo,
    GPUInfo,
    InferenceConfig,
    KubernetesMetadata,
    MetadataInfo,
    ModelInfo,
    ModelSpec,
    ResourceInfo,
    RuntimeState,
    VLLMRuntimeState,
)
from piqc.models.piqc_schema import (
    Confidence,
    REQUIRED_FACTS,
    SourceType,
)


def create_test_modelspec(
    name: str = "test-vllm",
    namespace: str = "default",
    framework: str = "vllm",
    confidence: float = 0.9,
    model_name: str = "meta-llama/Llama-2-7b-hf",
    precision: str = "bfloat16",
    max_model_len: int = 4096,
    max_sequences: int = 256,
    tensor_parallel: int = 1,
    gpu_count: int = 1,
    gpu_type: str = "NVIDIA A100-SXM4-80GB",
    gpu_memory: str = "80GB",
    gpu_utilization: int = 75,
    include_runtime: bool = False,
) -> ModelSpec:
    """Create a test ModelSpec with configurable parameters."""
    runtime_state = None
    if include_runtime:
        runtime_state = RuntimeState(
            collection_method="vllm-api",
            vllm=VLLMRuntimeState(
                loaded_model=model_name,
                requests_running=5,
                requests_waiting=2,
                gpu_cache_usage_percent=45.5,
                generation_tokens_per_sec=150.5,
                api_available=True,
                health_status="healthy",
                collection_timestamp="2024-01-01T00:00:00Z",
            ),
        )

    gpus = []
    if gpu_count > 0:
        gpus = [
            GPUInfo(
                index=i,
                type=gpu_type,
                memory_total=gpu_memory,
                memory_used="60GB",
                utilization=gpu_utilization,
                temperature=65,
                power_draw=250,
                pod_name=f"{name}-pod-{i}",
            )
            for i in range(gpu_count)
        ]

    return ModelSpec(
        metadata=MetadataInfo(
            name=name,
            namespace=namespace,
            labels={"app": "vllm"},
            annotations={},
            collection_timestamp="2024-01-01T00:00:00Z",
            collector_version="0.1.0",
        ),
        model=ModelInfo(
            name=model_name,
            architecture="llama",
            parameters="7B",
            confidence=0.85,
            identification_method="container_args",
        ),
        engine=EngineInfo(
            name=framework,
            version="0.4.0",
            detection_confidence=confidence,
        ),
        inference=InferenceConfig(
            precision=precision,
            max_model_len=max_model_len,
            max_sequences=max_sequences,
            tensor_parallel_size=tensor_parallel,
            quantization=None,
        ),
        resources=ResourceInfo(
            replicas=1,
            gpus=gpus,
            cpu_request="4",
            memory_request="32Gi",
        ),
        endpoints=[
            Endpoint(type="http", port=8000, path="/v1/completions"),
        ],
        kubernetes=KubernetesMetadata(
            deployment_type="Deployment",
            cluster_name="test-cluster",
            image="vllm/vllm-openai:v0.4.0",
            image_tag="v0.4.0",
        ),
        runtime_state=runtime_state,
        collection=CollectionMetadata(
            mode="remote",
            duration="5s",
            warnings=[],
            errors=[],
            data_completeness=DataCompleteness(
                static_config=True,
                gpu_metrics=True,
                runtime_metrics=include_runtime,
            ),
        ),
    )


class TestPIQCGenerator:
    """Tests for PIQCGenerator class."""

    def test_generator_initialization(self) -> None:
        """Test generator can be instantiated."""
        generator = PIQCGenerator()
        assert generator is not None

    def test_generate_creates_file(self) -> None:
        """Test generate creates piqc-facts.json file."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate(
                modelspecs=[modelspec],
                output_path=tmpdir,
            )
            assert Path(output_file).exists()
            assert output_file.endswith("piqc-facts.json")

    def test_generate_valid_json(self) -> None:
        """Test generated file is valid JSON."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            assert data["schemaVersion"] == "piqc-scan.v0.1"
            assert "generatedAt" in data
            assert "tool" in data
            assert "objects" in data

    def test_generate_includes_cluster_info(self) -> None:
        """Test generated bundle includes cluster info."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate(
                modelspecs=[modelspec],
                output_path=tmpdir,
                cluster_context="gke_project_zone_cluster",
                cluster_name="production",
            )

            with open(output_file) as f:
                data = json.load(f)

            assert data["cluster"]["context"] == "gke_project_zone_cluster"
            assert data["cluster"]["name"] == "production"


class TestFactExtraction:
    """Tests for fact extraction from ModelSpec."""

    def test_extract_runtime_engine_type(self) -> None:
        """Test runtime.engineType fact extraction."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(framework="vllm", confidence=0.95)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert "runtime.engineType" in facts
            assert facts["runtime.engineType"]["value"] == "vllm"
            assert facts["runtime.engineType"]["dataConfidence"] == "high"

    def test_extract_vllm_dtype(self) -> None:
        """Test vllm.dtype fact extraction."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(precision="bfloat16")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert "vllm.dtype" in facts
            assert facts["vllm.dtype"]["value"] == "bfloat16"

    def test_extract_vllm_max_model_len(self) -> None:
        """Test vllm.maxModelLen fact extraction."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(max_model_len=8192)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert "vllm.maxModelLen" in facts
            assert facts["vllm.maxModelLen"]["value"] == 8192
            assert facts["vllm.maxModelLen"]["units"] == "tokens"

    def test_extract_hardware_gpu_count(self) -> None:
        """Test hardware.gpuCount fact extraction."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(gpu_count=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["hardware.gpuCount"]["value"] == 4

    def test_extract_hardware_gpu_type(self) -> None:
        """Test hardware.gpuType fact extraction."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(gpu_type="NVIDIA H100-SXM5-80GB")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["hardware.gpuType"]["value"] == "NVIDIA H100-SXM5-80GB"
            assert facts["hardware.gpuType"]["source"]["type"] == "pod_exec"

    def test_extract_hardware_gpu_memory(self) -> None:
        """Test hardware.gpuMemoryGB fact extraction."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(gpu_memory="80GB")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["hardware.gpuMemoryGB"]["value"] == 80.0
            assert facts["hardware.gpuMemoryGB"]["units"] == "GB"

    def test_extract_observed_gpu_utilization(self) -> None:
        """Test obs.gpu.utilAvgPct fact extraction."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(gpu_utilization=85)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["obs.gpu.utilAvgPct"]["value"] == 85
            assert facts["obs.gpu.utilAvgPct"]["units"] == "%"


class TestExtendedFacts:
    """Tests for extended facts beyond registry."""

    def test_extract_model_name(self) -> None:
        """Test model.name extended fact extraction."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(model_name="mistralai/Mistral-7B-v0.1")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["model.name"]["value"] == "mistralai/Mistral-7B-v0.1"

    def test_extract_model_architecture(self) -> None:
        """Test model.architecture extended fact."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["model.architecture"]["value"] == "llama"

    def test_extract_k8s_replicas(self) -> None:
        """Test k8s.replicas extended fact."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["k8s.replicas"]["value"] == 1

    def test_extract_endpoint_http_port(self) -> None:
        """Test endpoint.httpPort extended fact."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["endpoint.httpPort"]["value"] == 8000


class TestRuntimeMetrics:
    """Tests for vLLM runtime metrics extraction."""

    def test_extract_vllm_tokens_per_sec(self) -> None:
        """Test obs.vllm.tokensPerSec from runtime state."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(include_runtime=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["obs.vllm.tokensPerSec"]["value"] == 150.5
            assert facts["obs.vllm.tokensPerSec"]["units"] == "tokens/s"

    def test_extract_vllm_requests_running(self) -> None:
        """Test obs.vllm.requestsRunning from runtime state."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(include_runtime=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["obs.vllm.requestsRunning"]["value"] == 5

    def test_extract_vllm_kv_cache_usage(self) -> None:
        """Test obs.vllm.kvCacheUsagePct from runtime state."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(include_runtime=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["obs.vllm.kvCacheUsagePct"]["value"] == 45.5


class TestFactErrors:
    """Tests for factErrors generation for missing required facts."""

    def test_missing_precision_generates_error(self) -> None:
        """Test missing dtype generates factError."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(precision=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            errors = data.get("factErrors", [])
            dtype_errors = [e for e in errors if e["factKey"] == "vllm.dtype"]
            assert len(dtype_errors) == 1
            assert dtype_errors[0]["severity"] == "warning"

    def test_missing_gpu_metrics_generates_errors(self) -> None:
        """Test missing GPU generates factErrors."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(gpu_count=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            errors = data.get("factErrors", [])
            fact_keys = [e["factKey"] for e in errors]
            assert "hardware.gpuType" in fact_keys
            assert "hardware.gpuMemoryGB" in fact_keys

    def test_fact_errors_have_workload_id(self) -> None:
        """Test factErrors include correct workloadId."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(
            name="my-model",
            namespace="production",
            precision=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            errors = data.get("factErrors", [])
            if errors:
                assert errors[0]["workloadId"].startswith("ns/production/")


class TestProvenance:
    """Tests for source provenance tracking."""

    def test_k8s_api_source(self) -> None:
        """Test K8s API sources are tracked correctly."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["vllm.maxModelLen"]["source"]["type"] == "k8s_api"
            assert facts["vllm.maxModelLen"]["source"]["method"] == "container_args"

    def test_pod_exec_source(self) -> None:
        """Test pod exec sources are tracked correctly."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["hardware.gpuType"]["source"]["type"] == "pod_exec"
            assert facts["hardware.gpuType"]["source"]["method"] == "nvidia-smi"

    def test_http_metrics_source(self) -> None:
        """Test HTTP metrics sources are tracked correctly."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(include_runtime=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["obs.vllm.tokensPerSec"]["source"]["type"] == "http_metrics"
            assert "GET /metrics" in facts["obs.vllm.tokensPerSec"]["source"]["method"]


class TestConfidenceScoring:
    """Tests for data confidence scoring."""

    def test_high_confidence_from_args(self) -> None:
        """Test high confidence for explicit args."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["vllm.tensorParallelSize"]["dataConfidence"] == "high"

    def test_medium_confidence_for_observations(self) -> None:
        """Test medium confidence for observed metrics."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["obs.gpu.utilAvgPct"]["dataConfidence"] == "medium"

    def test_confidence_from_detection_score(self) -> None:
        """Test confidence derived from detection score."""
        generator = PIQCGenerator()

        # High detection confidence
        modelspec = create_test_modelspec(confidence=0.95)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)
            with open(output_file) as f:
                data = json.load(f)
            facts = data["objects"][0]["facts"]
            assert facts["runtime.engineType"]["dataConfidence"] == "high"

        # Low detection confidence
        modelspec = create_test_modelspec(confidence=0.35)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)
            with open(output_file) as f:
                data = json.load(f)
            facts = data["objects"][0]["facts"]
            assert facts["runtime.engineType"]["dataConfidence"] == "low"


class TestMultipleWorkloads:
    """Tests for multiple workloads in bundle."""

    def test_multiple_modelspecs_creates_multiple_objects(self) -> None:
        """Test multiple ModelSpecs create multiple workload objects."""
        generator = PIQCGenerator()
        modelspecs = [
            create_test_modelspec(name="model-a", namespace="ns-a"),
            create_test_modelspec(name="model-b", namespace="ns-b"),
            create_test_modelspec(name="model-c", namespace="ns-c"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate(modelspecs, tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            assert len(data["objects"]) == 3

    def test_workloads_have_unique_ids(self) -> None:
        """Test each workload has unique workloadId."""
        generator = PIQCGenerator()
        modelspecs = [
            create_test_modelspec(name="model-a", namespace="prod"),
            create_test_modelspec(name="model-b", namespace="prod"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate(modelspecs, tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            workload_ids = [obj["workloadId"] for obj in data["objects"]]
            assert len(workload_ids) == len(set(workload_ids))


class TestNotes:
    """Tests for informational notes generation."""

    def test_notes_when_no_runtime_metrics(self) -> None:
        """Test notes generated when runtime metrics not collected."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(include_runtime=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)

            with open(output_file) as f:
                data = json.load(f)

            notes = data.get("notes", [])
            assert len(notes) >= 1
            assert any("runtime" in note.lower() for note in notes)
