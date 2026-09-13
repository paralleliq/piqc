"""
NVIDIA DCGM Exporter metrics collection.

DCGM (Data Center GPU Manager) profiling counters are the only way to get a
genuine compute-bound vs. memory-bandwidth-bound utilization split --
nvidia-smi's own utilization.gpu number (see gpu_collector.py) can't
distinguish them; it's a single undifferentiated "some kernel was running"
percentage. This is a real, previously-missing capability, not a rename of
an existing one -- see the platform repo's docs/ROADMAP.md entry
"obs.gpu.computeUtilAvgPct Doesn't Exist in piqc" for the gap this closes.

DCGM Exporter is not guaranteed to be running on every cluster -- it's an
extra piece of infrastructure some clusters have (GPU cloud providers, the
NVIDIA GPU Operator) and self-managed ones often don't, unlike nvidia-smi
which ships with every driver install. "If it is available" is the design
constraint throughout this module: discovery failure, an unreachable
endpoint, or an exporter that doesn't expose the profiling fields (older
DCGM versions, or profiling disabled in config) all degrade to "no facts,"
never a guess and never a scan-blocking error.
"""

from dataclasses import dataclass
from typing import Any, Optional

import requests

from piqc.collectors.vllm_api_client import PrometheusMetricsParser
from piqc.utils.logger import get_logger

logger = get_logger(__name__)

# Service name conventions for the NVIDIA GPU Operator's dcgm-exporter and
# common hand-rolled equivalents -- same by-name-hint approach
# core/coverage.py's _detect_otel_collector already uses for the OTel
# Collector, applied here to actually scrape the endpoint, not just report
# whether it exists.
_DCGM_SERVICE_NAME_HINTS = ("dcgm-exporter", "nvidia-dcgm-exporter", "dcgm")
_DCGM_DEFAULT_PORT = 9400

# DCGM profiling field names -- ratios in [0, 1] in the exporter's own
# output, converted to percentages here to match this codebase's other
# *Pct fact conventions (obs.gpu.utilAvgPct, etc.).
_TENSOR_ACTIVE_METRIC = "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"
_DRAM_ACTIVE_METRIC = "DCGM_FI_PROF_DRAM_ACTIVE"
_SM_ACTIVE_METRIC = "DCGM_FI_PROF_SM_ACTIVE"


@dataclass
class DCGMProfilingMetrics:
    """Node/cluster-scoped average of DCGM's profiling counters, across
    whichever GPUs the discovered exporter reports on -- DCGM Exporter
    isn't tied to one workload's pod the way vLLM's own /metrics is, so
    there's no single "this workload's GPU" to isolate without a device-ID
    cross-reference the exporter doesn't expose the same way piqc's own
    pod-to-GPU mapping does. None for any field DCGM Exporter isn't
    running, isn't reachable, or doesn't expose -- never a default guess.
    """

    available: bool = False
    tensor_active_pct: Optional[float] = None
    dram_active_pct: Optional[float] = None
    sm_active_pct: Optional[float] = None


def discover_dcgm_exporter(k8s_client: Any) -> Optional[str]:
    """Find a DCGM Exporter Service by name convention and return its
    scrape base URL (e.g. "http://10.0.1.5:9400"), or None if none is
    found. Never raises -- a failed service listing is treated the same as
    "not found."
    """
    try:
        services = k8s_client.list_all_services()
    except Exception as e:
        logger.debug(f"Failed to list services for DCGM discovery: {e}")
        return None

    for svc in services:
        name = (svc.metadata.name or "").lower() if svc.metadata else ""
        if not any(hint in name for hint in _DCGM_SERVICE_NAME_HINTS):
            continue

        cluster_ip = svc.spec.cluster_ip if svc.spec else None
        if not cluster_ip or cluster_ip == "None":
            # Headless service (cluster_ip "None") -- no single stable
            # address to scrape without per-pod IP resolution this
            # collector doesn't do. Skip rather than guess an address.
            continue

        port = _DCGM_DEFAULT_PORT
        for p in (svc.spec.ports or []) if svc.spec else []:
            if p.name and "metrics" in p.name.lower():
                port = p.port
                break

        return f"http://{cluster_ip}:{port}"

    return None


class DCGMCollector:
    """Scrapes a discovered DCGM Exporter's /metrics endpoint for the
    profiling counters nvidia-smi can't provide."""

    def __init__(self, timeout: int = 5) -> None:
        self.timeout = timeout
        self._parser = PrometheusMetricsParser()

    def collect(self, base_url: str) -> DCGMProfilingMetrics:
        """Fetch and parse DCGM's profiling metrics. Any failure --
        unreachable endpoint, bad response, or an exporter that doesn't
        expose these specific fields -- returns available=False with every
        value None, not a partial or fabricated result."""
        try:
            response = requests.get(f"{base_url}/metrics", timeout=self.timeout)
            response.raise_for_status()
        except Exception as e:
            logger.debug(f"DCGM Exporter unreachable at {base_url}: {e}")
            return DCGMProfilingMetrics(available=False)

        parsed = self._parser.parse(response.text)

        tensor_active = self._average_gauge(parsed, _TENSOR_ACTIVE_METRIC)
        dram_active = self._average_gauge(parsed, _DRAM_ACTIVE_METRIC)
        sm_active = self._average_gauge(parsed, _SM_ACTIVE_METRIC)

        if tensor_active is None and dram_active is None and sm_active is None:
            return DCGMProfilingMetrics(available=False)

        return DCGMProfilingMetrics(
            available=True,
            tensor_active_pct=tensor_active,
            dram_active_pct=dram_active,
            sm_active_pct=sm_active,
        )

    def _average_gauge(self, parsed: dict[str, Any], metric_name: str) -> Optional[float]:
        """DCGM Exporter reports one labeled sample per GPU (e.g.
        DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="0",...}) -- the same
        multi-label-gauge shape PrometheusMetricsParser already handles
        generically (see its own labeled-metric branch), so this just
        averages across whatever GPUs the exporter reported on. Converts
        DCGM's [0, 1] ratio convention to a percentage.
        """
        value = parsed.get(metric_name)
        if value is None:
            return None
        if isinstance(value, dict):
            samples = [v for v in value.values() if isinstance(v, int | float)]
            if not samples:
                return None
            return round(100.0 * (sum(samples) / len(samples)), 1)
        if isinstance(value, int | float):
            return round(100.0 * value, 1)
        return None
