"""Unit test for ScanOrchestrator._convert_gpu_metrics -- the bridge
between GPUCollector's raw GPUMetrics (nvidia-smi output) and the GPUInfo
model piqc_generator.py reads facts from.

memory_utilization_percent (nvidia-smi's own utilization.memory column) was
parsed by GPUCollector but silently dropped at this exact conversion step
for a long time -- collected, never reaching a fact. This test guards that
regression specifically, alongside the fields that already made it through.
"""

from datetime import datetime
from unittest.mock import MagicMock

from piqc.collectors.gpu_collector import GPUMetrics
from piqc.core.orchestrator import ScanOrchestrator


def _make_orchestrator() -> ScanOrchestrator:
    return ScanOrchestrator(k8s_client=MagicMock())


def test_convert_gpu_metrics_carries_memory_bandwidth_util_pct() -> None:
    orch = _make_orchestrator()
    metrics = [
        GPUMetrics(
            gpu_index=0,
            gpu_model="NVIDIA A100-SXM4-80GB",
            memory_total_mb=81920,
            memory_used_mb=40960,
            memory_free_mb=40960,
            utilization_percent=85,
            memory_utilization_percent=22,
            temperature_celsius=65,
            power_draw_watts=250,
            power_limit_watts=400,
            collection_timestamp=datetime(2026, 1, 1),
        )
    ]

    result = orch._convert_gpu_metrics(metrics, pod_name="llama-70b-0")

    assert len(result) == 1
    gpu = result[0]
    assert gpu.utilization == 85
    assert gpu.memory_bandwidth_util_pct == 22


def test_convert_gpu_metrics_handles_missing_memory_bandwidth_util() -> None:
    """nvidia-smi returned [N/A] for utilization.memory -- must pass
    through as None, not 0 or a guessed value."""
    orch = _make_orchestrator()
    metrics = [
        GPUMetrics(
            gpu_index=0,
            gpu_model="NVIDIA A100-SXM4-80GB",
            memory_total_mb=81920,
            memory_used_mb=40960,
            memory_free_mb=40960,
            utilization_percent=85,
            memory_utilization_percent=None,
            temperature_celsius=65,
            power_draw_watts=250,
            power_limit_watts=400,
            collection_timestamp=datetime(2026, 1, 1),
        )
    ]

    result = orch._convert_gpu_metrics(metrics, pod_name="llama-70b-0")

    assert result[0].memory_bandwidth_util_pct is None
