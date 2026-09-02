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

from piqc.core.orchestrator import FragmentedNodeInfo, PendingGPUPod, UnallocatedNodeInfo
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
    pipeline_parallel: int = 1,
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
            pipeline_parallel_size=pipeline_parallel,
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


class TestParallelismStrategy:
    """Tests for deployment.parallelismStrategy -- tensor_parallel_cross_node_v1
    and fragmentation_v1 both gate on this value directly."""

    def test_tensor_parallel_size_derives_tensor_strategy(self) -> None:
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(tensor_parallel=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)
            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["deployment.parallelismStrategy"]["value"] == "tensor"

    def test_pipeline_parallel_size_derives_pipeline_strategy(self) -> None:
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(tensor_parallel=1, pipeline_parallel=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)
            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert facts["deployment.parallelismStrategy"]["value"] == "pipeline"

    def test_no_parallelism_omits_the_fact(self) -> None:
        """Size of 1 (vLLM's own default) means "not using this strategy" --
        the fact shouldn't be silently reported as some other value."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec(tensor_parallel=1, pipeline_parallel=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir)
            with open(output_file) as f:
                data = json.load(f)

            facts = data["objects"][0]["facts"]
            assert "deployment.parallelismStrategy" not in facts


class TestPendingPodWorkload:
    """Tests for the pod-scoped WorkloadObject fragmentation_v1 needs, from
    _analyze_pending_gpu_pods()/_analyze_node_capacity()/
    _analyze_fragmented_nodes() -- all already computed today for the CLI's
    own table report, wired into this bundle for the first time.

    Cluster-wide fragmentation facts are folded directly onto the same
    pod-scoped object as the pod's own demand facts, not emitted as a
    separate cluster-scoped object -- fragmentation_v1's `when` clause
    evaluates one workload's fact store at a time (_evaluate_from_rules
    takes a single flat scan.facts dict), so a standalone object's facts
    would never be visible to the same evaluation as a pending job's facts.
    An earlier version of this fix emitted them separately and, despite
    every fact being technically present in the bundle, the rule never
    fired against a real /v1/ingest payload -- see test_rule_actually_fires
    below, which guards against that regression directly.
    """

    def test_pending_pod_facts_emitted(self) -> None:
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()
        pending = [
            PendingGPUPod(
                pod_name="vllm-70b",
                namespace="inference",
                gpus_needed=8,
                pending_minutes=47.3,
                parallelism_strategy="tensor",
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir, pending_gpu_pods=pending)
            with open(output_file) as f:
                data = json.load(f)

            pod_obj = next(o for o in data["objects"] if o["workloadId"] == "ns/inference/pod/vllm-70b")
            facts = pod_obj["facts"]
            assert facts["job.pendingGpuCount"]["value"] == 8
            assert facts["job.pendingSinceMinutes"]["value"] == 47.3
            assert facts["deployment.parallelismStrategy"]["value"] == "tensor"

    def test_no_parallelism_strategy_omits_the_fact(self) -> None:
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()
        pending = [
            PendingGPUPod(
                pod_name="batch-job",
                namespace="training",
                gpus_needed=2,
                pending_minutes=5.0,
                parallelism_strategy=None,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir, pending_gpu_pods=pending)
            with open(output_file) as f:
                data = json.load(f)

            pod_obj = next(o for o in data["objects"] if o["workloadId"] == "ns/training/pod/batch-job")
            assert "deployment.parallelismStrategy" not in pod_obj["facts"]
            assert "deployment.gpuType" not in pod_obj["facts"]

    def test_no_pending_pods_emits_no_pod_objects(self) -> None:
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()
        unallocated = [
            UnallocatedNodeInfo(node_name="node-a", gpu_type="H100", total_gpus=8, allocated_gpus=7, unallocated_gpus=1),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate([modelspec], tmpdir, unallocated_nodes=unallocated)
            with open(output_file) as f:
                data = json.load(f)

            assert not any(o["kind"] == "Pod" for o in data["objects"])

    def test_cluster_facts_folded_onto_the_pending_pod_object(self) -> None:
        """The core fix: totals/largest-block/fragmented-count -- computed
        cluster-wide, not per-pod -- must land on the SAME object as the
        pod's own job.pendingGpuCount, since that's the only object
        fragmentation_v1 will ever evaluate them together on."""
        generator = PIQCGenerator()
        modelspec = create_test_modelspec()
        unallocated = [
            UnallocatedNodeInfo(node_name="node-a", gpu_type="H100", total_gpus=8, allocated_gpus=7, unallocated_gpus=1),
            UnallocatedNodeInfo(node_name="node-b", gpu_type="H100", total_gpus=8, allocated_gpus=3, unallocated_gpus=5),
        ]
        fragmented = [
            FragmentedNodeInfo(node_name="node-a", gpu_type="H100", total_gpus=8, allocated_gpus=7, stranded_gpus=1, min_model_gpus_needed=2),
        ]
        pending = [
            PendingGPUPod(pod_name="vllm-70b", namespace="inference", gpus_needed=4, pending_minutes=42.5, parallelism_strategy="tensor"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate(
                [modelspec],
                tmpdir,
                unallocated_nodes=unallocated,
                fragmented_nodes=fragmented,
                pending_gpu_pods=pending,
            )
            with open(output_file) as f:
                data = json.load(f)

            pod_obj = next(o for o in data["objects"] if o["workloadId"] == "ns/inference/pod/vllm-70b")
            facts = pod_obj["facts"]
            assert facts["job.pendingGpuCount"]["value"] == 4
            assert facts["cluster.totalFreeGpus"]["value"] == 6
            assert facts["cluster.largestContiguousBlock"]["value"] == 5
            assert facts["cluster.fragmentedNodeCount"]["value"] == 1
            # No separate cluster-scoped object -- everything lives on the pod.
            assert not any(o["workloadId"].endswith("/cluster/fleet-wide") for o in data["objects"])

    def test_rule_actually_fires(self) -> None:
        """End-to-end regression guard: run the real fragmentation_v1 rule
        (not just check which facts got emitted) against a bundle shaped
        exactly like what generate() produces, using the same evaluator
        platform's /v1/ingest calls. Written after an earlier version of
        this fix passed every fact-presence test above yet never actually
        fired the rule, because the facts were split across two objects.
        """
        import sys
        from pathlib import Path

        platform_root = Path(__file__).resolve().parents[3] / "platform"
        if not platform_root.is_dir():
            pytest.skip("platform repo not checked out alongside piqc")
        sys.path.insert(0, str(platform_root / "services" / "control-plane"))
        from src.advisor_integration import evaluate_fragmentation  # type: ignore[import-not-found]

        generator = PIQCGenerator()
        modelspec = create_test_modelspec()
        # Enough total free GPUs across the cluster (6) to satisfy the
        # pending job's request (4), but split across nodes such that no
        # single node has more than 2 free -- genuinely fragmented, not
        # simply short on capacity (the rule's `ge total >= needed` /
        # `lt largest_block < needed` pair both need to hold: 6 >= 4, and
        # 2 < 4).
        unallocated = [
            UnallocatedNodeInfo(node_name="node-a", gpu_type="H100", total_gpus=8, allocated_gpus=6, unallocated_gpus=2),
            UnallocatedNodeInfo(node_name="node-b", gpu_type="H100", total_gpus=8, allocated_gpus=6, unallocated_gpus=2),
            UnallocatedNodeInfo(node_name="node-c", gpu_type="H100", total_gpus=8, allocated_gpus=6, unallocated_gpus=2),
        ]
        fragmented = [
            FragmentedNodeInfo(node_name="node-a", gpu_type="H100", total_gpus=8, allocated_gpus=6, stranded_gpus=2, min_model_gpus_needed=4),
        ]
        pending = [
            PendingGPUPod(pod_name="vllm-70b", namespace="inference", gpus_needed=4, pending_minutes=42.5, parallelism_strategy="tensor"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = generator.generate(
                [modelspec],
                tmpdir,
                unallocated_nodes=unallocated,
                fragmented_nodes=fragmented,
                pending_gpu_pods=pending,
            )
            with open(output_file) as f:
                bundle = json.load(f)

        pod_obj = next(o for o in bundle["objects"] if o["workloadId"] == "ns/inference/pod/vllm-70b")
        scan = {"facts": {k: {"value": v["value"]} for k, v in pod_obj["facts"].items()}}
        result = evaluate_fragmentation(scan)

        issue_codes = [i["issueCode"] for i in result["issues"]]
        assert "fragmentation_v1" in issue_codes
