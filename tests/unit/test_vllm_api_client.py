"""
Unit tests for vLLM API client.

Tests API client, Prometheus metrics parsing, and service discovery.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from piqc.collectors.vllm_api_client import (
    VLLMAPIClient,
    PrometheusMetricsParser,
    VLLMRuntimeMetrics,
    VLLMRequestMetrics,
    VLLMLatencyMetrics,
    VLLMThroughputMetrics,
    VLLMCacheMetrics,
    discover_vllm_service,
)


class TestVLLMRuntimeMetricsDataclass:
    """Tests for VLLMRuntimeMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        metrics = VLLMRuntimeMetrics()
        assert metrics.loaded_model is None
        assert metrics.api_available is False
        assert metrics.health_status == "unknown"

    def test_nested_metrics_defaults(self) -> None:
        """Test nested metrics have correct defaults."""
        metrics = VLLMRuntimeMetrics()
        assert metrics.requests.running == 0
        assert metrics.requests.waiting == 0
        assert metrics.latency.ttft_p50 is None
        assert metrics.throughput.prompt_tokens_total == 0
        assert metrics.cache.gpu_cache_usage_percent == 0.0


class TestPrometheusMetricsParser:
    """Tests for PrometheusMetricsParser."""

    def test_parse_simple_gauge(self) -> None:
        """Test parsing simple gauge metric."""
        parser = PrometheusMetricsParser()
        text = "vllm_num_requests_running 5\n"
        result = parser.parse(text)
        assert result["vllm_num_requests_running"] == 5.0

    def test_parse_labeled_metric(self) -> None:
        """Test parsing metric with labels."""
        parser = PrometheusMetricsParser()
        text = 'vllm_request_success{model_name="llama-7b"} 100\n'
        result = parser.parse(text)
        assert "vllm_request_success" in result
        assert result["vllm_request_success"]["model_name=llama-7b"] == 100.0

    def test_parse_histogram_buckets(self) -> None:
        """Test parsing histogram buckets and calculating percentiles."""
        parser = PrometheusMetricsParser()
        text = """
# HELP vllm_time_to_first_token_seconds Time to first token
# TYPE vllm_time_to_first_token_seconds histogram
vllm_time_to_first_token_seconds_bucket{le="0.001"} 0
vllm_time_to_first_token_seconds_bucket{le="0.005"} 10
vllm_time_to_first_token_seconds_bucket{le="0.01"} 50
vllm_time_to_first_token_seconds_bucket{le="0.025"} 80
vllm_time_to_first_token_seconds_bucket{le="0.05"} 95
vllm_time_to_first_token_seconds_bucket{le="0.1"} 100
vllm_time_to_first_token_seconds_bucket{le="+Inf"} 100
vllm_time_to_first_token_seconds_count 100
vllm_time_to_first_token_seconds_sum 1.5
"""
        result = parser.parse(text)
        assert "vllm_time_to_first_token_seconds_percentiles" in result
        percentiles = result["vllm_time_to_first_token_seconds_percentiles"]
        assert "p50" in percentiles
        assert "p95" in percentiles

    def test_parse_skips_comments(self) -> None:
        """Test parser skips comment lines."""
        parser = PrometheusMetricsParser()
        text = """
# HELP vllm_num_requests Number of requests
# TYPE vllm_num_requests gauge
vllm_num_requests 42
"""
        result = parser.parse(text)
        assert result["vllm_num_requests"] == 42.0

    def test_parse_handles_nan(self) -> None:
        """Test parser handles NaN values."""
        parser = PrometheusMetricsParser()
        text = "some_metric NaN\n"
        result = parser.parse(text)
        import math
        assert math.isnan(result["some_metric"])

    def test_parse_empty_input(self) -> None:
        """Test parsing empty input."""
        parser = PrometheusMetricsParser()
        result = parser.parse("")
        assert result == {}


class TestVLLMAPIClient:
    """Tests for VLLMAPIClient."""

    def test_client_initialization(self) -> None:
        """Test client initialization."""
        client = VLLMAPIClient("http://localhost:8000")
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 10

    def test_client_strips_trailing_slash(self) -> None:
        """Test trailing slash is removed from base URL."""
        client = VLLMAPIClient("http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_custom_timeout(self) -> None:
        """Test custom timeout is set."""
        client = VLLMAPIClient("http://localhost:8000", timeout=30)
        assert client.timeout == 30

    @patch("requests.get")
    def test_get_health_success(self, mock_get: Mock) -> None:
        """Test health check success."""
        mock_get.return_value.status_code = 200
        client = VLLMAPIClient("http://localhost:8000")
        assert client.get_health() is True
        mock_get.assert_called_once_with(
            "http://localhost:8000/health",
            timeout=10,
        )

    @patch("requests.get")
    def test_get_health_failure(self, mock_get: Mock) -> None:
        """Test health check failure."""
        mock_get.return_value.status_code = 503
        client = VLLMAPIClient("http://localhost:8000")
        assert client.get_health() is False

    @patch("requests.get")
    def test_get_health_exception(self, mock_get: Mock) -> None:
        """Test health check with network error."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        client = VLLMAPIClient("http://localhost:8000")
        assert client.get_health() is False

    @patch("requests.get")
    def test_get_models_success(self, mock_get: Mock) -> None:
        """Test get models success."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "data": [
                {"id": "meta-llama/Llama-2-7b-hf", "object": "model"}
            ]
        }
        client = VLLMAPIClient("http://localhost:8000")
        models = client.get_models()
        assert len(models) == 1
        assert models[0]["id"] == "meta-llama/Llama-2-7b-hf"

    @patch("requests.get")
    def test_get_models_empty(self, mock_get: Mock) -> None:
        """Test get models with empty response."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": []}
        client = VLLMAPIClient("http://localhost:8000")
        models = client.get_models()
        assert models == []

    @patch("requests.get")
    def test_get_metrics_raw(self, mock_get: Mock) -> None:
        """Test getting raw metrics."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "vllm_num_requests_running 5\n"
        mock_get.return_value.raise_for_status = Mock()

        client = VLLMAPIClient("http://localhost:8000")
        raw = client.get_metrics_raw()
        assert raw == "vllm_num_requests_running 5\n"

    @patch.object(VLLMAPIClient, "get_health")
    @patch.object(VLLMAPIClient, "get_models")
    @patch.object(VLLMAPIClient, "get_metrics_raw")
    def test_get_runtime_metrics_success(
        self,
        mock_metrics: Mock,
        mock_models: Mock,
        mock_health: Mock,
    ) -> None:
        """Test complete runtime metrics collection."""
        mock_health.return_value = True
        mock_models.return_value = [{"id": "llama-7b"}]
        mock_metrics.return_value = """
vllm_num_requests_running 5
vllm_num_requests_waiting 2
vllm_gpu_cache_usage_perc 0.45
"""
        client = VLLMAPIClient("http://localhost:8000")
        metrics = client.get_runtime_metrics()

        assert metrics.api_available is True
        assert metrics.health_status == "healthy"
        assert metrics.loaded_model == "llama-7b"
        assert metrics.requests.running == 5
        assert metrics.requests.waiting == 2

    @patch.object(VLLMAPIClient, "get_health")
    def test_get_runtime_metrics_unavailable(self, mock_health: Mock) -> None:
        """Test runtime metrics when API unavailable."""
        mock_health.return_value = False
        client = VLLMAPIClient("http://localhost:8000")
        metrics = client.get_runtime_metrics()

        assert metrics.api_available is False
        assert metrics.health_status == "unavailable"


class TestDiscoverVLLMService:
    """Tests for vLLM service discovery."""

    def test_discover_service_no_pods(self) -> None:
        """Test discovery with no pods found."""
        mock_client = Mock()
        mock_client.list_pods.return_value = []

        result = discover_vllm_service(mock_client, "default", "vllm-pod")
        assert result is None

    @patch.object(VLLMAPIClient, "get_health")
    def test_discover_service_success(self, mock_health: Mock) -> None:
        """Test successful service discovery."""
        mock_health.return_value = True

        mock_pod = Mock()
        mock_pod.status = Mock()
        mock_pod.status.pod_ip = "10.0.0.1"

        mock_client = Mock()
        mock_client.list_pods.return_value = [mock_pod]

        result = discover_vllm_service(mock_client, "default", "vllm-pod")
        assert result[0] == "http://10.0.0.1:8000"

    @patch.object(VLLMAPIClient, "get_health")
    def test_discover_service_try_multiple_ports(self, mock_health: Mock) -> None:
        """Test discovery tries multiple ports."""
        # First port fails, second succeeds
        mock_health.side_effect = [False, True]

        mock_pod = Mock()
        mock_pod.status = Mock()
        mock_pod.status.pod_ip = "10.0.0.1"

        mock_client = Mock()
        mock_client.list_pods.return_value = [mock_pod]

        result = discover_vllm_service(mock_client, "default", "vllm-pod")
        assert result[0] == "http://10.0.0.1:8080"
