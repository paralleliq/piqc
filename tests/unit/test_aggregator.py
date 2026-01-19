"""
Unit tests for aggregator module.

Tests multi-pod aggregation and statistical summaries.
"""

import pytest
import math

from piqc.core.aggregator import (
    MetricRange,
    AggregatedGPUMetrics,
    AggregatedRuntimeMetrics,
    AggregatedDeployment,
    PodAggregator,
)
from piqc.models.modelspec import (
    CollectionMetadata,
    DataCompleteness,
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


class TestMetricRange:
    """Tests for MetricRange statistical class."""

    def test_from_single_value(self) -> None:
        """Test creating range from single value."""
        result = MetricRange.from_values([50.0])
        assert result is not None
        assert result.min == 50.0
        assert result.max == 50.0
        assert result.avg == 50.0
        assert result.stddev == 0.0
        assert result.count == 1

    def test_from_multiple_values(self) -> None:
        """Test creating range from multiple values."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = MetricRange.from_values(values)
        assert result is not None
        assert result.min == 10.0
        assert result.max == 50.0
        assert result.avg == 30.0
        assert result.count == 5

    def test_from_empty_list(self) -> None:
        """Test creating range from empty list returns None."""
        result = MetricRange.from_values([])
        assert result is None

    def test_stddev_calculation(self) -> None:
        """Test standard deviation calculation."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = MetricRange.from_values(values)
        assert result is not None
        # Expected sample stddev is ~2.14
        assert 2.0 < result.stddev < 2.3

    def test_two_values_stddev(self) -> None:
        """Test stddev with two values."""
        values = [10.0, 20.0]
        result = MetricRange.from_values(values)
        assert result is not None
        assert result.avg == 15.0
        # Sample stddev of [10, 20] = sqrt(50) ≈ 7.07
        assert 7.0 < result.stddev < 7.2


class TestAggregatedDataclasses:
    """Tests for aggregation dataclasses."""

    def test_aggregated_gpu_metrics_defaults(self) -> None:
        """Test AggregatedGPUMetrics defaults."""
        metrics = AggregatedGPUMetrics()
        assert metrics.total_gpus == 0
        assert metrics.utilization is None
        assert metrics.memory_used_percent is None

    def test_aggregated_runtime_metrics_defaults(self) -> None:
        """Test AggregatedRuntimeMetrics defaults."""
        metrics = AggregatedRuntimeMetrics()
        assert metrics.total_requests_running == 0
        assert metrics.avg_prompt_throughput == 0.0
        assert metrics.gpu_cache_usage is None

    def test_aggregated_deployment_creation(self) -> None:
        """Test AggregatedDeployment creation."""
        agg = AggregatedDeployment(
            name="vllm-llama",
            namespace="production",
            framework="vllm",
            replicas=3,
            total_gpus=12,
        )
        assert agg.name == "vllm-llama"
        assert agg.replicas == 3
        assert agg.total_gpus == 12


def create_test_modelspec(
    name: str = "test",
    namespace: str = "default",
    gpu_utilization: int = 75,
    gpu_memory_total: str = "80GB",
    gpu_memory_used: str = "60GB",
    include_runtime: bool = False,
    confidence: float = 0.95,
) -> ModelSpec:
    """Create test ModelSpec for aggregation tests."""
    runtime_state = None
    if include_runtime:
        runtime_state = RuntimeState(
            collection_method="vllm-api",
            vllm=VLLMRuntimeState(
                requests_running=5,
                requests_waiting=2,
                prompt_tokens_per_sec=100.0,
                generation_tokens_per_sec=150.0,
                gpu_cache_usage_percent=45.0,
            ),
        )

    return ModelSpec(
        metadata=MetadataInfo(
            name=name,
            namespace=namespace,
            labels={},
            annotations={},
            collection_timestamp="2024-01-01T00:00:00Z",
            collector_version="0.1.0",
        ),
        model=ModelInfo(
            name="llama-7b",
            architecture="llama",
            confidence=0.9,
        ),
        engine=EngineInfo(
            name="vllm",
            detection_confidence=confidence,
        ),
        inference=InferenceConfig(),
        resources=ResourceInfo(
            replicas=1,
            gpus=[
                GPUInfo(
                    index=0,
                    type="A100-80GB",
                    memory_total=gpu_memory_total,
                    memory_used=gpu_memory_used,
                    utilization=gpu_utilization,
                    temperature=65,
                    power_draw=250,
                    pod_name=f"{name}-pod",
                ),
            ],
        ),
        endpoints=[],
        kubernetes=KubernetesMetadata(
            deployment_type="Deployment",
            image="vllm:latest",
        ),
        runtime_state=runtime_state,
        collection=CollectionMetadata(
            mode="remote",
            duration="5s",
            warnings=[],
            errors=[],
            data_completeness=DataCompleteness(),
        ),
    )


class TestPodAggregator:
    """Tests for PodAggregator class."""

    def test_aggregator_initialization(self) -> None:
        """Test aggregator can be created."""
        aggregator = PodAggregator()
        assert aggregator is not None

    def test_empty_modelspecs(self) -> None:
        """Test aggregating empty list."""
        aggregator = PodAggregator()
        result = aggregator.aggregate([])
        assert result == []

    def test_single_modelspec(self) -> None:
        """Test aggregating single ModelSpec."""
        aggregator = PodAggregator()
        modelspec = create_test_modelspec()

        result = aggregator.aggregate([modelspec])
        assert len(result) == 1
        assert result[0].name == "test"
        assert result[0].replicas == 1

    def test_multiple_modelspecs_same_deployment(self) -> None:
        """Test aggregating multiple pods of same deployment."""
        aggregator = PodAggregator()
        modelspecs = [
            create_test_modelspec(name="vllm-app", namespace="prod", gpu_utilization=70),
            create_test_modelspec(name="vllm-app", namespace="prod", gpu_utilization=80),
            create_test_modelspec(name="vllm-app", namespace="prod", gpu_utilization=90),
        ]

        result = aggregator.aggregate(modelspecs)
        assert len(result) == 1
        agg = result[0]
        assert agg.name == "vllm-app"
        assert agg.replicas == 3
        assert agg.total_gpus == 3

    def test_aggregation_gpu_utilization_stats(self) -> None:
        """Test GPU utilization aggregation statistics."""
        aggregator = PodAggregator()
        modelspecs = [
            create_test_modelspec(name="app", gpu_utilization=60),
            create_test_modelspec(name="app", gpu_utilization=70),
            create_test_modelspec(name="app", gpu_utilization=80),
        ]

        result = aggregator.aggregate(modelspecs)
        agg = result[0]

        assert agg.gpu_metrics.utilization is not None
        assert agg.gpu_metrics.utilization.min == 60.0
        assert agg.gpu_metrics.utilization.max == 80.0
        assert agg.gpu_metrics.utilization.avg == 70.0

    def test_aggregation_multiple_deployments(self) -> None:
        """Test aggregating multiple different deployments."""
        aggregator = PodAggregator()
        modelspecs = [
            create_test_modelspec(name="app-a", namespace="ns-a"),
            create_test_modelspec(name="app-b", namespace="ns-b"),
            create_test_modelspec(name="app-a", namespace="ns-a"),  # Second pod
        ]

        result = aggregator.aggregate(modelspecs)
        assert len(result) == 2

        # Find app-a
        app_a = next(a for a in result if a.name == "app-a")
        assert app_a.replicas == 2

        # Find app-b
        app_b = next(a for a in result if a.name == "app-b")
        assert app_b.replicas == 1

    def test_aggregation_confidence_averaging(self) -> None:
        """Test confidence scores are averaged."""
        aggregator = PodAggregator()

        modelspecs = [
            create_test_modelspec(name="app", confidence=0.90),
            create_test_modelspec(name="app", confidence=0.80),
        ]

        result = aggregator.aggregate(modelspecs)
        agg = result[0]

        assert agg.avg_confidence == pytest.approx(0.85)

    def test_aggregation_memory_percentage(self) -> None:
        """Test memory usage percentage calculation."""
        aggregator = PodAggregator()
        modelspecs = [
            create_test_modelspec(
                name="app",
                gpu_memory_total="80GB",
                gpu_memory_used="40GB",
            ),
        ]

        result = aggregator.aggregate(modelspecs)
        agg = result[0]

        assert agg.gpu_metrics.memory_used_percent is not None
        assert agg.gpu_metrics.memory_used_percent.avg == 50.0

    def test_aggregation_runtime_metrics(self) -> None:
        """Test runtime metrics aggregation."""
        aggregator = PodAggregator()
        modelspecs = [
            create_test_modelspec(name="app", include_runtime=True),
            create_test_modelspec(name="app", include_runtime=True),
        ]

        result = aggregator.aggregate(modelspecs)
        agg = result[0]

        # Each has 5 running requests
        assert agg.runtime_metrics.total_requests_running == 10
        # Each has 2 waiting requests
        assert agg.runtime_metrics.total_requests_waiting == 4

    def test_aggregation_warnings_deduplication(self) -> None:
        """Test warnings are deduplicated."""
        aggregator = PodAggregator()

        spec1 = create_test_modelspec(name="app")
        spec2 = create_test_modelspec(name="app")
        spec1.collection.warnings = ["Warning A", "Warning B"]
        spec2.collection.warnings = ["Warning A", "Warning C"]

        result = aggregator.aggregate([spec1, spec2])
        agg = result[0]

        # Should deduplicate Warning A
        assert len(agg.warnings) == 3


class TestAggregatorGrouping:
    """Tests for deployment grouping logic."""

    def test_group_by_namespace_and_name(self) -> None:
        """Test grouping uses namespace + name as key."""
        aggregator = PodAggregator()

        # Same name, different namespace should be different deployments
        modelspecs = [
            create_test_modelspec(name="app", namespace="dev"),
            create_test_modelspec(name="app", namespace="prod"),
        ]

        result = aggregator.aggregate(modelspecs)
        assert len(result) == 2

    def test_model_info_from_first_pod(self) -> None:
        """Test model info is taken from first pod."""
        aggregator = PodAggregator()

        spec1 = create_test_modelspec(name="app")
        spec2 = create_test_modelspec(name="app")
        spec1.model.name = "llama-7b"
        spec2.model.name = "llama-13b"  # Different model (shouldn't happen but test it)

        result = aggregator.aggregate([spec1, spec2])
        agg = result[0]

        assert agg.model_name == "llama-7b"  # From first spec


class TestMemoryParsing:
    """Tests for memory string parsing."""

    def test_parse_gb(self) -> None:
        """Test parsing GB memory strings."""
        aggregator = PodAggregator()
        assert aggregator._parse_memory("80GB") == 80 * 1024

    def test_parse_mb(self) -> None:
        """Test parsing MB memory strings."""
        aggregator = PodAggregator()
        assert aggregator._parse_memory("81920MB") == 81920

    def test_parse_kb(self) -> None:
        """Test parsing KB memory strings."""
        aggregator = PodAggregator()
        assert aggregator._parse_memory("1024KB") == 1.0

    def test_parse_case_insensitive(self) -> None:
        """Test memory parsing is case-insensitive."""
        aggregator = PodAggregator()
        assert aggregator._parse_memory("80gb") == 80 * 1024
        assert aggregator._parse_memory("80Gb") == 80 * 1024
