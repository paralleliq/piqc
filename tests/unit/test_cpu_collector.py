"""
Unit tests for CPU collector.

Tests /proc/stat parsing, delta computation, and fallback behavior.
"""

import pytest
from unittest.mock import MagicMock

from piqc.collectors.cpu_collector import (
    CPUCollector,
    _compute_utilization,
    _count_cpus,
    _parse_cpu_line,
    _SNAPSHOT_MARKER,
)
from piqc.utils.exceptions import CPUMetricsUnavailableError


SNAPSHOT_A = """cpu  100 10 50 800 20 0 5 0 0 0
cpu0 50 5 25 400 10 0 2 0 0 0
cpu1 50 5 25 400 10 0 3 0 0 0
intr 12345 0 0
ctxt 54321"""

SNAPSHOT_B = """cpu  150 10 70 850 40 0 5 0 0 0
cpu0 75 5 35 425 20 0 2 0 0 0
cpu1 75 5 35 425 20 0 3 0 0 0
intr 12500 0 0
ctxt 54400"""


class TestParseCpuLine:
    """Tests for _parse_cpu_line."""

    def test_parses_named_fields(self) -> None:
        fields = _parse_cpu_line("cpu  100 10 50 800 20 0 5 0 0 0")
        assert fields["user"] == 100
        assert fields["nice"] == 10
        assert fields["system"] == 50
        assert fields["idle"] == 800
        assert fields["iowait"] == 20

    def test_rejects_non_cpu_line(self) -> None:
        with pytest.raises(ValueError):
            _parse_cpu_line("intr 12345 0 0")


class TestCountCpus:
    """Tests for _count_cpus."""

    def test_counts_per_core_lines(self) -> None:
        assert _count_cpus(SNAPSHOT_A) == 2

    def test_zero_when_no_per_core_lines(self) -> None:
        assert _count_cpus("cpu  100 10 50 800 20 0 5 0 0 0") == 0


class TestComputeUtilization:
    """Tests for _compute_utilization delta math."""

    def test_computes_utilization_and_iowait(self) -> None:
        utilization, iowait = _compute_utilization(SNAPSHOT_A, SNAPSHOT_B)
        # total_delta = (150-100)+(10-10)+(70-50)+(850-800)+(40-20)+0+0+0+0+0 = 50+0+20+50+20 = 140
        # idle_delta (idle+iowait) = 50 + 20 = 70 -> utilization = (140-70)/140*100 = 50.0
        assert utilization == pytest.approx(50.0)
        assert iowait == pytest.approx(20 / 140 * 100)

    def test_zero_delta_returns_zero(self) -> None:
        utilization, iowait = _compute_utilization(SNAPSHOT_A, SNAPSHOT_A)
        assert utilization == 0.0
        assert iowait == 0.0

    def test_missing_cpu_line_raises(self) -> None:
        with pytest.raises(StopIteration):
            _compute_utilization("intr 1 2 3", "intr 4 5 6")


class TestCPUCollectorCollect:
    """Tests for CPUCollector.collect using a mocked k8s_client."""

    def _collector(self, exec_return: tuple[str, str], sample_interval: float = 0.01) -> CPUCollector:
        k8s_client = MagicMock()
        k8s_client.exec_in_pod.return_value = exec_return
        return CPUCollector(k8s_client=k8s_client, sample_interval=sample_interval)

    def test_collect_returns_metrics(self) -> None:
        stdout = f"{SNAPSHOT_A}\n{_SNAPSHOT_MARKER}\n{SNAPSHOT_B}"
        collector = self._collector((stdout, ""))

        metrics = collector.collect(pod_name="pod-a", namespace="inference")

        assert metrics is not None
        assert metrics.utilization_percent == pytest.approx(50.0)
        assert metrics.cpu_count == 2

    def test_collect_returns_none_on_missing_marker(self) -> None:
        collector = self._collector((SNAPSHOT_A, ""))
        assert collector.collect(pod_name="pod-a", namespace="inference") is None

    def test_collect_returns_none_on_empty_snapshot(self) -> None:
        stdout = f"{_SNAPSHOT_MARKER}\n{SNAPSHOT_B}"
        collector = self._collector((stdout, ""))
        assert collector.collect(pod_name="pod-a", namespace="inference") is None

    def test_collect_raises_on_permission_denied(self) -> None:
        k8s_client = MagicMock()
        k8s_client.exec_in_pod.side_effect = Exception("permission denied")
        collector = CPUCollector(k8s_client=k8s_client)

        with pytest.raises(CPUMetricsUnavailableError):
            collector.collect(pod_name="pod-a", namespace="inference")

    def test_collect_raises_when_no_shell(self) -> None:
        k8s_client = MagicMock()
        k8s_client.exec_in_pod.side_effect = Exception("exec failed: command not found")
        collector = CPUCollector(k8s_client=k8s_client)

        with pytest.raises(CPUMetricsUnavailableError):
            collector.collect(pod_name="pod-a", namespace="inference")

    def test_collect_raises_when_stderr_indicates_no_shell(self) -> None:
        collector = self._collector((SNAPSHOT_A, "sh: not found"))
        with pytest.raises(CPUMetricsUnavailableError):
            collector.collect(pod_name="pod-a", namespace="inference")

    def test_collect_returns_none_on_malformed_snapshot(self) -> None:
        stdout = f"garbage\n{_SNAPSHOT_MARKER}\nmore garbage"
        collector = self._collector((stdout, ""))
        assert collector.collect(pod_name="pod-a", namespace="inference") is None
