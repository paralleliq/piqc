"""
Unit tests for DCGM Exporter discovery and metrics collection.

Tests the "if it is available" design throughout: a missing service, an
unreachable endpoint, and an exporter that doesn't expose the profiling
fields must all degrade to an empty/unavailable result, never an exception
and never a guessed value.
"""

from unittest.mock import MagicMock, Mock, patch

from piqc.collectors.dcgm_collector import (
    DCGMCollector,
    DCGMProfilingMetrics,
    discover_dcgm_exporter,
)


def _fake_service(name: str, cluster_ip: str = "10.0.1.5", port_name: str | None = "metrics") -> Mock:
    svc = MagicMock()
    svc.metadata.name = name
    svc.spec.cluster_ip = cluster_ip
    port = MagicMock()
    port.name = port_name
    port.port = 9400
    svc.spec.ports = [port]
    return svc


class TestDiscoverDCGMExporter:
    """Tests for discover_dcgm_exporter."""

    def test_finds_dcgm_exporter_by_name(self) -> None:
        k8s_client = Mock()
        k8s_client.list_all_services.return_value = [_fake_service("nvidia-dcgm-exporter")]

        url = discover_dcgm_exporter(k8s_client)

        assert url == "http://10.0.1.5:9400"

    def test_ignores_unrelated_services(self) -> None:
        k8s_client = Mock()
        k8s_client.list_all_services.return_value = [
            _fake_service("otel-collector"),
            _fake_service("some-other-service"),
        ]

        assert discover_dcgm_exporter(k8s_client) is None

    def test_skips_headless_service(self) -> None:
        """A headless Service (cluster_ip 'None') has no single stable
        address to scrape without per-pod IP resolution -- must be skipped,
        not guessed at."""
        k8s_client = Mock()
        k8s_client.list_all_services.return_value = [
            _fake_service("dcgm-exporter", cluster_ip="None")
        ]

        assert discover_dcgm_exporter(k8s_client) is None

    def test_falls_back_to_default_port_without_named_metrics_port(self) -> None:
        k8s_client = Mock()
        k8s_client.list_all_services.return_value = [
            _fake_service("dcgm-exporter", port_name="grpc")
        ]

        assert discover_dcgm_exporter(k8s_client) == "http://10.0.1.5:9400"

    def test_service_listing_failure_returns_none(self) -> None:
        """A failed service listing must never raise -- treated the same
        as 'not found'."""
        k8s_client = Mock()
        k8s_client.list_all_services.side_effect = Exception("connection refused")

        assert discover_dcgm_exporter(k8s_client) is None


class TestDCGMCollector:
    """Tests for DCGMCollector.collect."""

    @patch("requests.get")
    def test_collect_averages_across_gpus(self, mock_get: Mock) -> None:
        mock_get.return_value.text = """
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="0",UUID="a"} 0.80
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="1",UUID="b"} 0.60
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0",UUID="a"} 0.10
DCGM_FI_PROF_DRAM_ACTIVE{gpu="1",UUID="b"} 0.20
"""
        mock_get.return_value.raise_for_status = Mock()

        collector = DCGMCollector()
        result = collector.collect("http://10.0.1.5:9400")

        assert result.available is True
        assert result.tensor_active_pct == 70.0  # (0.80 + 0.60) / 2 * 100
        assert result.dram_active_pct == 15.0  # (0.10 + 0.20) / 2 * 100

    @patch("requests.get")
    def test_collect_unreachable_endpoint(self, mock_get: Mock) -> None:
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError()

        collector = DCGMCollector()
        result = collector.collect("http://10.0.1.5:9400")

        assert result == DCGMProfilingMetrics(available=False)

    @patch("requests.get")
    def test_collect_when_profiling_metrics_not_exposed(self, mock_get: Mock) -> None:
        """Exporter reachable, but doesn't expose the profiling fields --
        older DCGM versions or profiling disabled in config. Must not be
        confused with an unreachable endpoint, but the result is the same:
        unavailable, no guessed values."""
        mock_get.return_value.text = "DCGM_FI_DEV_GPU_TEMP{gpu=\"0\"} 65\n"
        mock_get.return_value.raise_for_status = Mock()

        collector = DCGMCollector()
        result = collector.collect("http://10.0.1.5:9400")

        assert result.available is False
        assert result.tensor_active_pct is None
        assert result.dram_active_pct is None

    @patch("requests.get")
    def test_collect_http_error_status(self, mock_get: Mock) -> None:
        import requests

        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError()

        collector = DCGMCollector()
        result = collector.collect("http://10.0.1.5:9400")

        assert result.available is False
