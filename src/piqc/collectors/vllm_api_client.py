"""
vLLM API client for runtime metrics collection.

Collects dynamic runtime data from vLLM inference servers via their
OpenAI-compatible API and Prometheus metrics endpoints.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import requests

from piqc.utils.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# Data Structures for vLLM Runtime Metrics
# =============================================================================


@dataclass
class VLLMRequestMetrics:
    """Metrics about request queue and processing."""

    running: int = 0
    waiting: int = 0
    swapped: int = 0
    success_total: int = 0


@dataclass
class VLLMLatencyMetrics:
    """Latency metrics with percentiles (in seconds)."""

    ttft_p50: Optional[float] = None  # Time to First Token
    ttft_p95: Optional[float] = None
    ttft_p99: Optional[float] = None
    tpot_p50: Optional[float] = None  # Time Per Output Token
    tpot_p95: Optional[float] = None
    e2e_p50: Optional[float] = None  # End-to-End
    e2e_p95: Optional[float] = None
    e2e_p99: Optional[float] = None


@dataclass
class VLLMThroughputMetrics:
    """Throughput and token processing metrics."""

    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    prompt_tokens_per_second: float = 0.0
    generation_tokens_per_second: float = 0.0


@dataclass
class VLLMCacheMetrics:
    """KV cache and prefix cache metrics."""

    gpu_cache_usage_percent: float = 0.0
    cpu_cache_usage_percent: float = 0.0
    prefix_cache_hit_rate: float = 0.0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0


@dataclass
class VLLMRuntimeMetrics:
    """Complete vLLM runtime metrics from API endpoints."""

    loaded_model: Optional[str] = None
    model_max_len: Optional[int] = None
    requests: VLLMRequestMetrics = field(default_factory=VLLMRequestMetrics)
    latency: VLLMLatencyMetrics = field(default_factory=VLLMLatencyMetrics)
    throughput: VLLMThroughputMetrics = field(default_factory=VLLMThroughputMetrics)
    cache: VLLMCacheMetrics = field(default_factory=VLLMCacheMetrics)
    api_available: bool = False
    health_status: str = "unknown"
    collection_timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Prometheus Metrics Parser
# =============================================================================


class PrometheusMetricsParser:
    """
    Parser for Prometheus text format metrics.

    Parses the text exposition format from vLLM's /metrics endpoint
    into structured data including histogram percentiles.
    """

    # Regex patterns for parsing Prometheus format
    METRIC_LINE_PATTERN = re.compile(
        r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
        r'(?:\{(?P<labels>[^}]*)\})?'
        r'\s+(?P<value>[^\s]+)'
        r'(?:\s+(?P<timestamp>\d+))?$'
    )

    LABEL_PATTERN = re.compile(r'(\w+)="([^"]*)"')

    def parse(self, metrics_text: str) -> dict[str, Any]:
        """
        Parse Prometheus text format into a dictionary.

        Args:
            metrics_text: Raw Prometheus metrics text.

        Returns:
            Dictionary with metric names as keys, values can be:
            - float for gauges/counters
            - dict with labels for labeled metrics
            - dict with 'buckets' for histograms
        """
        metrics: dict[str, Any] = {}
        histograms: dict[str, dict[str, list[tuple[float, float]]]] = {}

        for line in metrics_text.strip().split('\n'):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            match = self.METRIC_LINE_PATTERN.match(line)
            if not match:
                continue

            name = match.group('name')
            labels_str = match.group('labels') or ''
            value_str = match.group('value')

            # Parse value
            try:
                if value_str.lower() in ('nan', '+inf', '-inf'):
                    value = float(value_str)
                else:
                    value = float(value_str)
            except ValueError:
                continue

            # Parse labels
            labels = dict(self.LABEL_PATTERN.findall(labels_str))

            # Handle histogram buckets
            if name.endswith('_bucket'):
                base_name = name[:-7]  # Remove '_bucket'
                le = labels.get('le', '')
                if base_name not in histograms:
                    histograms[base_name] = {'buckets': []}
                try:
                    le_val = float(le) if le != '+Inf' else float('inf')
                    histograms[base_name]['buckets'].append((le_val, value))
                except ValueError:
                    pass
            elif name.endswith('_sum') or name.endswith('_count'):
                # Store histogram sum/count separately
                metrics[name] = value
            else:
                # Regular gauge or counter
                if labels:
                    if name not in metrics:
                        metrics[name] = {}
                    label_key = ','.join(f'{k}={v}' for k, v in sorted(labels.items()))
                    metrics[name][label_key] = value
                else:
                    metrics[name] = value

        # Calculate histogram percentiles
        for hist_name, hist_data in histograms.items():
            buckets = sorted(hist_data['buckets'], key=lambda x: x[0])
            if buckets:
                percentiles = self._calculate_percentiles(buckets)
                metrics[f'{hist_name}_percentiles'] = percentiles

        return metrics

    def _calculate_percentiles(
        self,
        buckets: list[tuple[float, float]],
    ) -> dict[str, float]:
        """
        Calculate percentiles from histogram buckets.

        Uses linear interpolation between bucket boundaries.
        """
        if not buckets:
            return {}

        total = buckets[-1][1]  # Total count is the last bucket (le=+Inf)
        if total == 0:
            return {}

        percentiles = {}
        target_percentiles = {'p50': 0.50, 'p95': 0.95, 'p99': 0.99}

        for name, target in target_percentiles.items():
            target_count = total * target

            # Find the bucket containing this percentile
            prev_bound = 0.0
            prev_count = 0.0

            for bound, count in buckets:
                if count >= target_count:
                    # Linear interpolation
                    if count == prev_count:
                        percentiles[name] = bound
                    else:
                        fraction = (target_count - prev_count) / (count - prev_count)
                        percentiles[name] = prev_bound + fraction * (bound - prev_bound)
                    break
                prev_bound = bound
                prev_count = count

        return percentiles




class KubectlExecVLLMClient:
    """
    vLLM API client that uses kubectl exec for remote access.
    
    This client is used when direct pod IP access is not available,
    typically when running PIQC remotely via kubectl.
    """
    
    DEFAULT_TIMEOUT = 10
    
    def __init__(
        self,
        k8s_client: Any,
        namespace: str,
        pod_name: str,
        port: int = 8000,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize kubectl exec-based vLLM client.
        
        Args:
            k8s_client: Kubernetes client instance.
            namespace: Pod namespace.
            pod_name: Pod name.
            port: vLLM API port.
            timeout: Request timeout.
        """
        self.k8s_client = k8s_client
        self.namespace = namespace
        self.pod_name = pod_name
        self.port = port
        self.timeout = timeout
        self._parser = PrometheusMetricsParser()
    
    def _exec_curl(self, endpoint: str) -> Optional[str]:
        """Execute HTTP request inside pod via kubectl exec.

        Uses python3 urllib — always available in vLLM containers since
        vLLM is Python-based. Falls back to curl if python3 is not found.
        """
        url = f"http://localhost:{self.port}{endpoint}"

        # Try python3 first — guaranteed in any vLLM container
        try:
            python_cmd = (
                "import urllib.request; "
                "print(urllib.request.urlopen("
                f'"{url}", timeout={self.timeout}'
                ").read().decode())"
            )
            stdout, stderr = self.k8s_client.exec_in_pod(
                pod_name=self.pod_name,
                namespace=self.namespace,
                command=["python3", "-c", python_cmd],
            )
            if stdout is not None and "executable file not found" not in (stderr or ""):
                return stdout
        except Exception as e:
            logger.debug(f"kubectl exec python3 failed for {endpoint}: {e}")

        # Fall back to curl
        try:
            stdout, stderr = self.k8s_client.exec_in_pod(
                pod_name=self.pod_name,
                namespace=self.namespace,
                command=["curl", "-s", "-m", str(self.timeout), url],
            )
            if stdout is not None and "executable file not found" not in (stderr or ""):
                return stdout
        except Exception as e:
            logger.debug(f"kubectl exec curl fallback failed for {endpoint}: {e}")

        return None
    
    def get_health(self) -> bool:
        """Check vLLM server health via kubectl exec."""
        result = self._exec_curl("/health")
        # Empty response is valid for /health (HTTP 200 with no body)
        # Just check that curl didn't fail (result is not None) and no error in output
        if result is None:
            return False
        return "error" not in result.lower() and "failed" not in result.lower()
    
    def get_models(self) -> list[dict[str, Any]]:
        """Get list of loaded models via kubectl exec."""
        import json
        
        result = self._exec_curl("/v1/models")
        if not result:
            return []
        
        try:
            data = json.loads(result)
            return data.get('data', [])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse models JSON: {e}")
            return []
    
    def get_metrics_raw(self) -> str:
        """Get raw Prometheus metrics via kubectl exec."""
        result = self._exec_curl("/metrics")
        if not result:
            raise Exception("Failed to get metrics via kubectl exec")
        return result
    
    def get_runtime_metrics(self) -> VLLMRuntimeMetrics:
        """Collect comprehensive runtime metrics via kubectl exec."""
        metrics = VLLMRuntimeMetrics(collection_timestamp=datetime.utcnow())
        
        # Check health first
        metrics.api_available = self.get_health()
        if not metrics.api_available:
            metrics.health_status = "unavailable"
            return metrics
        
        metrics.health_status = "healthy"
        
        # Get model info
        models = self.get_models()
        if models:
            metrics.loaded_model = models[0].get('id')
        
        # Parse Prometheus metrics
        try:
            raw_metrics = self.get_metrics_raw()
            parsed = self._parser.parse(raw_metrics)
            self._populate_metrics(metrics, parsed)
        except Exception as e:
            logger.warning(f"Failed to get Prometheus metrics via kubectl exec: {e}")
        
        return metrics
    
    def _populate_metrics(
        self,
        metrics: VLLMRuntimeMetrics,
        parsed: dict[str, Any],
    ) -> None:
        """Populate VLLMRuntimeMetrics from parsed Prometheus data."""
        # Use same logic as VLLMAPIClient
        client = VLLMAPIClient("http://dummy", timeout=self.timeout)
        client._populate_metrics(metrics, parsed)


class VLLMAPIClient:
    """
    Client for vLLM OpenAI-compatible API.

    Collects runtime metrics from vLLM servers including:
    - Model information via /v1/models
    - Health status via /health
    - Prometheus metrics via /metrics
    """

    DEFAULT_TIMEOUT = 10
    METRICS_ENDPOINT = "/metrics"
    MODELS_ENDPOINT = "/v1/models"
    HEALTH_ENDPOINT = "/health"

    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        """
        Initialize the vLLM API client.

        Args:
            base_url: Base URL of the vLLM server (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._parser = PrometheusMetricsParser()

    def get_health(self) -> bool:
        """
        Check vLLM server health.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = requests.get(
                f"{self.base_url}{self.HEALTH_ENDPOINT}",
                timeout=self.timeout,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_models(self) -> list[dict[str, Any]]:
        """
        Get list of loaded models.

        Returns:
            List of model objects from /v1/models endpoint.
        """
        try:
            response = requests.get(
                f"{self.base_url}{self.MODELS_ENDPOINT}",
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get models: {e}")
            return []

    def get_metrics_raw(self) -> str:
        """
        Get raw Prometheus metrics text.

        Returns:
            Raw metrics text from /metrics endpoint.

        Raises:
            requests.exceptions.RequestException: If request fails.
        """
        response = requests.get(
            f"{self.base_url}{self.METRICS_ENDPOINT}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    def get_runtime_metrics(self) -> VLLMRuntimeMetrics:
        """
        Collect comprehensive runtime metrics from vLLM server.

        Returns:
            VLLMRuntimeMetrics with all available metrics.
        """
        metrics = VLLMRuntimeMetrics(collection_timestamp=datetime.utcnow())

        # Check health first
        metrics.api_available = self.get_health()
        if not metrics.api_available:
            metrics.health_status = "unavailable"
            return metrics

        metrics.health_status = "healthy"

        # Get model info
        models = self.get_models()
        if models:
            metrics.loaded_model = models[0].get('id')

        # Parse Prometheus metrics
        try:
            raw_metrics = self.get_metrics_raw()
            parsed = self._parser.parse(raw_metrics)
            self._populate_metrics(metrics, parsed)
        except Exception as e:
            logger.warning(f"Failed to get Prometheus metrics via kubectl exec: {e}")

        return metrics

    @staticmethod
    def _scalar(value: Any, default: float = 0.0) -> float:
        """Extract a scalar float from a metric value that may be a labeled dict."""
        if value is None:
            return default
        if isinstance(value, dict):
            return sum(value.values()) if value else default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get(self, parsed: dict[str, Any], *keys: str) -> Any:
        """Return first non-None value from parsed dict for given keys."""
        for key in keys:
            v = parsed.get(key)
            if v is not None:
                return v
        return None

    def _populate_metrics(
        self,
        metrics: VLLMRuntimeMetrics,
        parsed: dict[str, Any],
    ) -> None:
        """Populate VLLMRuntimeMetrics from parsed Prometheus data."""

        s = self._scalar

        # Request metrics
        metrics.requests.running = int(s(self._get(parsed,
            'vllm:num_requests_running', 'vllm_num_requests_running')))
        metrics.requests.waiting = int(s(self._get(parsed,
            'vllm:num_requests_waiting', 'vllm_num_requests_waiting')))
        metrics.requests.swapped = int(s(self._get(parsed,
            'vllm:num_requests_swapped', 'vllm_num_requests_swapped')))
        metrics.requests.success_total = int(s(self._get(parsed,
            'vllm:request_success_total', 'vllm_request_success_total')))

        # Cache metrics
        metrics.cache.gpu_cache_usage_percent = s(self._get(parsed,
            'vllm:gpu_cache_usage_perc', 'vllm_gpu_cache_usage_perc'))
        metrics.cache.cpu_cache_usage_percent = s(self._get(parsed,
            'vllm:cpu_cache_usage_perc', 'vllm_cpu_cache_usage_perc'))
        metrics.cache.prefix_cache_hit_rate = s(self._get(parsed,
            'vllm:cpu_prefix_cache_hit_rate', 'vllm_cpu_prefix_cache_hit_rate'))
        metrics.cache.prefix_cache_queries = int(s(self._get(parsed,
            'vllm:prefix_cache_queries', 'vllm_prefix_cache_queries')))
        metrics.cache.prefix_cache_hits = int(s(self._get(parsed,
            'vllm:prefix_cache_hits', 'vllm_prefix_cache_hits')))

        # Throughput metrics
        metrics.throughput.prompt_tokens_total = int(s(self._get(parsed,
            'vllm:prompt_tokens_total', 'vllm_prompt_tokens_total')))
        metrics.throughput.generation_tokens_total = int(s(self._get(parsed,
            'vllm:generation_tokens_total', 'vllm_generation_tokens_total')))
        metrics.throughput.prompt_tokens_per_second = s(self._get(parsed,
            'vllm:avg_prompt_throughput_toks_per_s', 'vllm_avg_prompt_throughput_toks_per_s'))
        metrics.throughput.generation_tokens_per_second = s(self._get(parsed,
            'vllm:avg_generation_throughput_toks_per_s', 'vllm_avg_generation_throughput_toks_per_s'))

        # Latency metrics (from histogram percentiles)
        ttft_percentiles = parsed.get('vllm:time_to_first_token_seconds_percentiles', {})
        if not ttft_percentiles:
            ttft_percentiles = parsed.get('vllm_time_to_first_token_seconds_percentiles', {})

        metrics.latency.ttft_p50 = ttft_percentiles.get('p50')
        metrics.latency.ttft_p95 = ttft_percentiles.get('p95')
        metrics.latency.ttft_p99 = ttft_percentiles.get('p99')

        tpot_percentiles = parsed.get('vllm:time_per_output_token_seconds_percentiles', {})
        if not tpot_percentiles:
            tpot_percentiles = parsed.get('vllm_time_per_output_token_seconds_percentiles', {})

        metrics.latency.tpot_p50 = tpot_percentiles.get('p50')
        metrics.latency.tpot_p95 = tpot_percentiles.get('p95')

        e2e_percentiles = parsed.get('vllm:e2e_request_latency_seconds_percentiles', {})
        if not e2e_percentiles:
            e2e_percentiles = parsed.get('vllm_e2e_request_latency_seconds_percentiles', {})

        metrics.latency.e2e_p50 = e2e_percentiles.get('p50')
        metrics.latency.e2e_p95 = e2e_percentiles.get('p95')
        metrics.latency.e2e_p99 = e2e_percentiles.get('p99')


# =============================================================================
# Service Discovery
# =============================================================================


def discover_vllm_service(
    k8s_client: Any,
    namespace: str,
    pod_name: str,
) -> Optional[tuple[str, Optional[int]]]:
    """
    Discover vLLM service endpoint for a pod.

    Attempts to find an accessible vLLM API endpoint by:
    1. Trying pod IP directly (works for in-cluster access)
    2. Using kubectl exec as fallback (for remote kubectl access)

    Args:
        k8s_client: Kubernetes client instance.
        namespace: Pod namespace.
        pod_name: Pod name to find service for.

    Returns:
        Tuple of (access_method, port) where:
        - access_method: "http" with pod IP URL, or "kubectl-exec" for remote access
        - port: Port number if using kubectl-exec, None for direct HTTP
        Returns None if no accessible endpoint found.
    """
    # Common vLLM ports to try
    common_ports = [8000, 8080]

    # Try pod IP directly with common ports (works for in-cluster)
    try:
        pods = k8s_client.list_pods(
            namespace=namespace,
            field_selector=f"metadata.name={pod_name}",
        )
        if pods and pods[0].status and pods[0].status.pod_ip:
            pod_ip = pods[0].status.pod_ip

            for port in common_ports:
                url = f"http://{pod_ip}:{port}"
                try:
                    client = VLLMAPIClient(url, timeout=5)
                    if client.get_health():
                        logger.debug(f"Found vLLM API at {url} (direct access)")
                        return (url, None)
                except Exception:
                    # Direct access failed, try kubectl exec
                    pass

    except Exception as e:
        logger.debug(f"Direct service discovery failed: {e}")

    # Fallback: Use kubectl exec for remote access
    try:
        for port in common_ports:
            url = f"http://localhost:{port}/health"
            # Try python3 urllib first (always available in vLLM containers)
            python_cmd = (
                "import urllib.request; "
                f'r = urllib.request.urlopen("{url}", timeout=3); '
                "print(r.status)"
            )
            try:
                stdout, stderr = k8s_client.exec_in_pod(
                    pod_name=pod_name,
                    namespace=namespace,
                    command=["python3", "-c", python_cmd],
                )
                if stdout is not None and "executable file not found" not in (stderr or ""):
                    logger.debug(f"Found vLLM API via kubectl exec (python3) on port {port}")
                    return ("kubectl-exec", port)
            except Exception:
                pass

            # Fall back to curl
            try:
                stdout, stderr = k8s_client.exec_in_pod(
                    pod_name=pod_name,
                    namespace=namespace,
                    command=["curl", "-s", "-m", "3", url],
                )
                if stdout is not None and "error" not in stdout.lower() and "error" not in (stderr or "").lower():
                    logger.debug(f"Found vLLM API via kubectl exec (curl) on port {port}")
                    return ("kubectl-exec", port)
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"kubectl exec service discovery failed: {e}")

    return None
