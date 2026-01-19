"""
Unit tests for PIQC schema models.

Tests all Pydantic models for the piqc-scan.v0.1 schema.
"""

import pytest
from datetime import datetime

from piqc.models.piqc_schema import (
    Confidence,
    Severity,
    SourceType,
    Source,
    FactValue,
    ToolInfo,
    ClusterInfo,
    ScopeInfo,
    WorkloadObject,
    FactError,
    PIQCBundle,
    REQUIRED_FACTS,
    OPTIONAL_FACTS,
)


class TestEnums:
    """Tests for enum types."""

    def test_confidence_values(self) -> None:
        """Test Confidence enum values."""
        assert Confidence.HIGH.value == "high"
        assert Confidence.MEDIUM.value == "medium"
        assert Confidence.LOW.value == "low"

    def test_severity_values(self) -> None:
        """Test Severity enum values."""
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"

    def test_source_type_values(self) -> None:
        """Test SourceType enum values."""
        assert SourceType.SCANNER.value == "scanner"
        assert SourceType.K8S_API.value == "k8s_api"
        assert SourceType.POD_EXEC.value == "pod_exec"
        assert SourceType.HTTP_METRICS.value == "http_metrics"
        assert SourceType.PROMETHEUS.value == "prometheus"
        assert SourceType.OTHER.value == "other"


class TestSource:
    """Tests for Source model."""

    def test_create_minimal_source(self) -> None:
        """Test creating source with only required field."""
        source = Source(type=SourceType.SCANNER)
        assert source.type == SourceType.SCANNER
        assert source.method is None
        assert source.ref is None

    def test_create_full_source(self) -> None:
        """Test creating source with all fields."""
        source = Source(
            type=SourceType.POD_EXEC,
            method="nvidia-smi",
            ref="vllm-pod-abc123",
        )
        assert source.type == SourceType.POD_EXEC
        assert source.method == "nvidia-smi"
        assert source.ref == "vllm-pod-abc123"

    def test_source_serialization(self) -> None:
        """Test source serialization."""
        source = Source(type=SourceType.K8S_API, method="container_args")
        data = source.model_dump(by_alias=True, exclude_none=True)
        assert data == {"type": "k8s_api", "method": "container_args"}


class TestFactValue:
    """Tests for FactValue model."""

    def test_create_string_fact(self) -> None:
        """Test creating fact with string value."""
        fact = FactValue(
            value="vllm",
            source=Source(type=SourceType.SCANNER),
            data_confidence=Confidence.HIGH,
            observed_at="2024-01-01T00:00:00Z",
        )
        assert fact.value == "vllm"
        assert fact.data_confidence == Confidence.HIGH

    def test_create_numeric_fact(self) -> None:
        """Test creating fact with numeric value."""
        fact = FactValue(
            value=4096,
            source=Source(type=SourceType.K8S_API, method="container_args"),
            data_confidence=Confidence.HIGH,
            observed_at="2024-01-01T00:00:00Z",
            units="tokens",
        )
        assert fact.value == 4096
        assert fact.units == "tokens"

    def test_create_fact_with_staleness(self) -> None:
        """Test creating fact with staleness."""
        fact = FactValue(
            value=85.5,
            source=Source(type=SourceType.PROMETHEUS),
            data_confidence=Confidence.MEDIUM,
            observed_at="2024-01-01T00:00:00Z",
            units="%",
            staleness_sec=30,
        )
        assert fact.staleness_sec == 30
        assert fact.units == "%"

    def test_fact_serialization(self) -> None:
        """Test fact serialization with aliases."""
        fact = FactValue(
            value="bfloat16",
            source=Source(type=SourceType.K8S_API),
            data_confidence=Confidence.HIGH,
            observed_at="2024-01-01T00:00:00Z",
        )
        data = fact.model_dump(by_alias=True, exclude_none=True)
        assert data["dataConfidence"] == "high"
        assert data["observedAt"] == "2024-01-01T00:00:00Z"


class TestToolInfo:
    """Tests for ToolInfo model."""

    def test_create_tool_info(self) -> None:
        """Test creating tool info."""
        tool = ToolInfo(name="piqc", version="0.1.0")
        assert tool.name == "piqc"
        assert tool.version == "0.1.0"

    def test_tool_info_with_git_sha(self) -> None:
        """Test tool info with git SHA."""
        tool = ToolInfo(
            name="piqc-scan",
            version="1.0.0",
            git_sha="abc123def456",
        )
        assert tool.git_sha == "abc123def456"


class TestClusterInfo:
    """Tests for ClusterInfo model."""

    def test_create_empty_cluster(self) -> None:
        """Test creating empty cluster info."""
        cluster = ClusterInfo()
        assert cluster.context is None
        assert cluster.name is None

    def test_create_full_cluster(self) -> None:
        """Test creating full cluster info."""
        cluster = ClusterInfo(
            context="gke_project_zone_cluster",
            name="production-cluster",
            uid="abc123",
        )
        assert cluster.context == "gke_project_zone_cluster"
        assert cluster.name == "production-cluster"


class TestWorkloadObject:
    """Tests for WorkloadObject model."""

    def test_create_minimal_workload(self) -> None:
        """Test creating workload with minimal fields."""
        workload = WorkloadObject(
            workload_id="ns/default/deploy/vllm-server",
            kind="Deployment",
            name="vllm-server",
            namespace="default",
            facts={},
        )
        assert workload.workload_id == "ns/default/deploy/vllm-server"
        assert workload.kind == "Deployment"

    def test_create_workload_with_facts(self) -> None:
        """Test creating workload with facts."""
        workload = WorkloadObject(
            workload_id="ns/prod/deploy/llama-7b",
            kind="Deployment",
            name="llama-7b",
            namespace="prod",
            images=["vllm/vllm-openai:v0.4.0"],
            pods=["llama-7b-abc123"],
            facts={
                "runtime.engineType": FactValue(
                    value="vllm",
                    source=Source(type=SourceType.SCANNER),
                    data_confidence=Confidence.HIGH,
                    observed_at="2024-01-01T00:00:00Z",
                ),
            },
        )
        assert len(workload.facts) == 1
        assert "runtime.engineType" in workload.facts


class TestFactError:
    """Tests for FactError model."""

    def test_create_fact_error(self) -> None:
        """Test creating fact error."""
        error = FactError(
            workload_id="ns/default/deploy/vllm",
            fact_key="hardware.gpuType",
            error="nvidia-smi not available",
            severity=Severity.WARNING,
        )
        assert error.fact_key == "hardware.gpuType"
        assert error.severity == Severity.WARNING

    def test_fact_error_serialization(self) -> None:
        """Test fact error serialization."""
        error = FactError(
            workload_id="ns/test/deploy/model",
            fact_key="obs.gpu.utilAvgPct",
            error="GPU metrics unavailable",
            severity=Severity.ERROR,
            source=Source(type=SourceType.POD_EXEC),
            observed_at="2024-01-01T00:00:00Z",
        )
        data = error.model_dump(by_alias=True, exclude_none=True)
        assert data["workloadId"] == "ns/test/deploy/model"
        assert data["factKey"] == "obs.gpu.utilAvgPct"


class TestPIQCBundle:
    """Tests for PIQCBundle root model."""

    def test_create_minimal_bundle(self) -> None:
        """Test creating minimal PIQC bundle."""
        bundle = PIQCBundle(
            generated_at="2024-01-01T00:00:00Z",
            tool=ToolInfo(name="test", version="1.0"),
            objects=[],
        )
        assert bundle.schema_version == "piqc-scan.v0.1"
        assert len(bundle.objects) == 0

    def test_create_full_bundle(self) -> None:
        """Test creating full PIQC bundle."""
        bundle = PIQCBundle(
            generated_at="2024-01-01T00:00:00Z",
            tool=ToolInfo(name="piqc", version="0.1.0"),
            cluster=ClusterInfo(context="test-context"),
            scope=ScopeInfo(namespaces=["default", "production"]),
            objects=[
                WorkloadObject(
                    workload_id="ns/default/deploy/test",
                    kind="Deployment",
                    name="test",
                    namespace="default",
                    facts={},
                ),
            ],
            fact_errors=[
                FactError(
                    workload_id="ns/default/deploy/test",
                    fact_key="hardware.gpuType",
                    error="Unavailable",
                    severity=Severity.WARNING,
                ),
            ],
            notes=["Test bundle"],
        )
        assert len(bundle.objects) == 1
        assert len(bundle.fact_errors) == 1

    def test_bundle_to_dict(self) -> None:
        """Test bundle serialization."""
        bundle = PIQCBundle(
            generated_at="2024-01-01T00:00:00Z",
            tool=ToolInfo(name="test", version="1.0"),
            objects=[],
        )
        data = bundle.to_dict()
        assert data["schemaVersion"] == "piqc-scan.v0.1"
        assert data["generatedAt"] == "2024-01-01T00:00:00Z"
        assert "objects" in data


class TestFactsRegistry:
    """Tests for facts registry constants."""

    def test_required_facts_count(self) -> None:
        """Test required facts count."""
        assert len(REQUIRED_FACTS) == 10

    def test_required_facts_include_runtime(self) -> None:
        """Test runtime facts in required list."""
        assert "runtime.engineType" in REQUIRED_FACTS

    def test_required_facts_include_vllm(self) -> None:
        """Test vLLM facts in required list."""
        assert "vllm.dtype" in REQUIRED_FACTS
        assert "vllm.maxModelLen" in REQUIRED_FACTS
        assert "vllm.tensorParallelSize" in REQUIRED_FACTS

    def test_required_facts_include_hardware(self) -> None:
        """Test hardware facts in required list."""
        assert "hardware.gpuType" in REQUIRED_FACTS
        assert "hardware.gpuCount" in REQUIRED_FACTS

    def test_optional_facts_include_extended(self) -> None:
        """Test extended facts in optional list."""
        assert "model.name" in OPTIONAL_FACTS
        assert "obs.vllm.tokensPerSec" in OPTIONAL_FACTS
