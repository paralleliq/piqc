"""
Multi-pod aggregation for deployment-level metrics.

Aggregates metrics from multiple pod replicas into a unified
deployment-level view with statistical summaries.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from piqc.models.modelspec import ModelSpec
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class MetricRange:
    """Statistical summary for a numeric metric."""

    min: float
    max: float
    avg: float
    stddev: float
    count: int

    @classmethod
    def from_values(cls, values: list[float]) -> Optional["MetricRange"]:
        """
        Create a MetricRange from a list of values.

        Args:
            values: List of numeric values.

        Returns:
            MetricRange or None if values is empty.
        """
        if not values:
            return None

        n = len(values)
        avg = sum(values) / n
        min_val = min(values)
        max_val = max(values)

        if n > 1:
            variance = sum((x - avg) ** 2 for x in values) / (n - 1)
            stddev = math.sqrt(variance)
        else:
            stddev = 0.0

        return cls(
            min=min_val,
            max=max_val,
            avg=avg,
            stddev=stddev,
            count=n,
        )


@dataclass
class AggregatedGPUMetrics:
    """Aggregated GPU metrics across all pods."""

    total_gpus: int = 0
    utilization: Optional[MetricRange] = None
    memory_used_percent: Optional[MetricRange] = None
    temperature: Optional[MetricRange] = None
    power_draw: Optional[MetricRange] = None


@dataclass
class AggregatedRuntimeMetrics:
    """Aggregated runtime metrics across all pods."""

    total_requests_running: int = 0
    total_requests_waiting: int = 0
    avg_prompt_throughput: float = 0.0
    avg_generation_throughput: float = 0.0
    gpu_cache_usage: Optional[MetricRange] = None


@dataclass
class AggregatedDeployment:
    """
    Aggregated metrics for a multi-replica deployment.

    Combines data from all pods in a deployment into a
    unified statistical view.
    """

    name: str
    namespace: str
    framework: str
    replicas: int
    total_gpus: int

    # Model info (from first pod)
    model_name: Optional[str] = None
    model_architecture: Optional[str] = None

    # Aggregated GPU metrics
    gpu_metrics: AggregatedGPUMetrics = field(default_factory=AggregatedGPUMetrics)

    # Aggregated runtime metrics
    runtime_metrics: AggregatedRuntimeMetrics = field(
        default_factory=AggregatedRuntimeMetrics
    )

    # Individual pod names
    pods: list[str] = field(default_factory=list)

    # Aggregated warnings and errors
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Detection confidence (average across pods)
    avg_confidence: float = 0.0


class PodAggregator:
    """
    Aggregates ModelSpec data from multiple pods.

    Groups pods by deployment and calculates statistical
    summaries for all numeric metrics.
    """

    def aggregate(
        self,
        modelspecs: list[ModelSpec],
    ) -> list[AggregatedDeployment]:
        """
        Aggregate ModelSpecs by deployment.

        Args:
            modelspecs: List of ModelSpec objects from individual pods.

        Returns:
            List of AggregatedDeployment objects.
        """
        # Group by deployment (namespace + name)
        grouped = self._group_by_deployment(modelspecs)

        # Aggregate each group
        aggregated = []
        for key, specs in grouped.items():
            namespace, name = key
            agg = self._aggregate_group(namespace, name, specs)
            aggregated.append(agg)

        return aggregated

    def _group_by_deployment(
        self,
        modelspecs: list[ModelSpec],
    ) -> dict[tuple[str, str], list[ModelSpec]]:
        """Group ModelSpecs by namespace and deployment name."""
        groups: dict[tuple[str, str], list[ModelSpec]] = {}

        for spec in modelspecs:
            key = (spec.metadata.namespace, spec.metadata.name)
            if key not in groups:
                groups[key] = []
            groups[key].append(spec)

        return groups

    def _aggregate_group(
        self,
        namespace: str,
        name: str,
        specs: list[ModelSpec],
    ) -> AggregatedDeployment:
        """Aggregate a group of ModelSpecs into a single AggregatedDeployment."""
        if not specs:
            return AggregatedDeployment(
                name=name,
                namespace=namespace,
                framework="unknown",
                replicas=0,
                total_gpus=0,
            )

        first_spec = specs[0]

        # Collect all values for aggregation
        gpu_utils: list[float] = []
        gpu_memory_pcts: list[float] = []
        gpu_temps: list[float] = []
        gpu_powers: list[float] = []
        cache_usages: list[float] = []
        confidences: list[float] = []
        all_warnings: list[str] = []
        all_errors: list[str] = []
        all_pods: list[str] = []
        total_gpus = 0
        total_requests_running = 0
        total_requests_waiting = 0
        prompt_throughputs: list[float] = []
        gen_throughputs: list[float] = []

        for spec in specs:
            # Collect confidence
            confidences.append(spec.engine.detection_confidence)

            # Collect GPU metrics
            if spec.resources and spec.resources.gpus:
                for gpu in spec.resources.gpus:
                    total_gpus += 1
                    if gpu.utilization is not None:
                        gpu_utils.append(float(gpu.utilization))
                    if gpu.memory_total and gpu.memory_used:
                        # Calculate memory percentage
                        try:
                            total_mb = self._parse_memory(gpu.memory_total)
                            used_mb = self._parse_memory(gpu.memory_used)
                            if total_mb > 0:
                                gpu_memory_pcts.append((used_mb / total_mb) * 100)
                        except (ValueError, TypeError):
                            pass
                    if gpu.temperature is not None:
                        gpu_temps.append(float(gpu.temperature))
                    if gpu.power_draw is not None:
                        gpu_powers.append(float(gpu.power_draw))

                    if gpu.pod_name:
                        all_pods.append(gpu.pod_name)

            # Collect runtime metrics (if available)
            if spec.runtime_state and spec.runtime_state.vllm:
                vllm = spec.runtime_state.vllm
                total_requests_running += vllm.requests_running
                total_requests_waiting += vllm.requests_waiting
                if vllm.prompt_tokens_per_sec:
                    prompt_throughputs.append(vllm.prompt_tokens_per_sec)
                if vllm.generation_tokens_per_sec:
                    gen_throughputs.append(vllm.generation_tokens_per_sec)
                if vllm.gpu_cache_usage_percent:
                    cache_usages.append(vllm.gpu_cache_usage_percent)

            # Collect warnings/errors
            if spec.collection:
                all_warnings.extend(spec.collection.warnings)
                all_errors.extend(spec.collection.errors)

        # Build aggregated result
        agg = AggregatedDeployment(
            name=name,
            namespace=namespace,
            framework=first_spec.engine.name,
            replicas=len(specs),
            total_gpus=total_gpus,
            model_name=first_spec.model.name if first_spec.model else None,
            model_architecture=first_spec.model.architecture if first_spec.model else None,
            pods=list(set(all_pods)) if all_pods else [s.metadata.name for s in specs],
            warnings=list(set(all_warnings)),
            errors=list(set(all_errors)),
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        )

        # GPU aggregations
        agg.gpu_metrics = AggregatedGPUMetrics(
            total_gpus=total_gpus,
            utilization=MetricRange.from_values(gpu_utils),
            memory_used_percent=MetricRange.from_values(gpu_memory_pcts),
            temperature=MetricRange.from_values(gpu_temps),
            power_draw=MetricRange.from_values(gpu_powers),
        )

        # Runtime aggregations
        agg.runtime_metrics = AggregatedRuntimeMetrics(
            total_requests_running=total_requests_running,
            total_requests_waiting=total_requests_waiting,
            avg_prompt_throughput=(
                sum(prompt_throughputs) / len(prompt_throughputs)
                if prompt_throughputs else 0.0
            ),
            avg_generation_throughput=(
                sum(gen_throughputs) / len(gen_throughputs)
                if gen_throughputs else 0.0
            ),
            gpu_cache_usage=MetricRange.from_values(cache_usages),
        )

        return agg

    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string like '80GB' or '45120MB' to MB."""
        memory_str = memory_str.strip().upper()

        if memory_str.endswith('GB'):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith('MB'):
            return float(memory_str[:-2])
        elif memory_str.endswith('KB'):
            return float(memory_str[:-2]) / 1024
        else:
            # Assume MB
            return float(memory_str)
