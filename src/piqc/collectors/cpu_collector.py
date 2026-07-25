"""
CPU metrics collection using /proc/stat.

Collects real-time CPU utilization from pods by executing two /proc/stat
reads a short interval apart inside the target container, then computing
per-field deltas the same way top/vmstat do. No metrics-server or DCGM
dependency - just exec_in_pod(), matching the mechanism gpu_collector.py
already uses.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from piqc.core.k8s_client import K8sClient
from piqc.utils.exceptions import CPUMetricsUnavailableError
from piqc.utils.logger import get_logger


logger = get_logger(__name__)

# /proc/stat "cpu  " summary line field order (man proc_stat). Fields beyond
# guest/guest_nice exist on newer kernels but aren't needed for utilization math.
_PROC_STAT_FIELDS = [
    "user", "nice", "system", "idle", "iowait",
    "irq", "softirq", "steal", "guest", "guest_nice",
]

_SNAPSHOT_MARKER = "---PIQC-CPU-SNAPSHOT---"


@dataclass
class CPUMetrics:
    """CPU utilization computed from two /proc/stat snapshots."""

    utilization_percent: float
    iowait_percent: float
    cpu_count: Optional[int]
    collection_timestamp: datetime


def _parse_cpu_line(line: str) -> dict[str, int]:
    """Parse a '/proc/stat' aggregate 'cpu  ...' line into named jiffy counters."""
    parts = line.split()
    if not parts or parts[0] != "cpu":
        raise ValueError(f"Not a 'cpu' summary line: {line!r}")

    values = parts[1:]
    fields: dict[str, int] = {}
    for name, value in zip(_PROC_STAT_FIELDS, values):
        fields[name] = int(value)
    return fields


def _count_cpus(stat_text: str) -> int:
    """Count per-core 'cpuN' lines (cpu0, cpu1, ...) in a /proc/stat snapshot."""
    return sum(1 for line in stat_text.splitlines() if line.startswith("cpu") and line[3:4].isdigit())


def _compute_utilization(snapshot_a: str, snapshot_b: str) -> tuple[float, float]:
    """
    Compute utilization % and iowait % from two /proc/stat snapshots.

    Returns:
        Tuple of (utilization_percent, iowait_percent).
    """
    line_a = next(line for line in snapshot_a.splitlines() if line.startswith("cpu "))
    line_b = next(line for line in snapshot_b.splitlines() if line.startswith("cpu "))

    fields_a = _parse_cpu_line(line_a)
    fields_b = _parse_cpu_line(line_b)

    deltas = {name: fields_b[name] - fields_a[name] for name in fields_a}
    total_delta = sum(deltas.values())

    if total_delta <= 0:
        return 0.0, 0.0

    idle_delta = deltas["idle"] + deltas["iowait"]
    utilization_percent = max(0.0, min(100.0, (total_delta - idle_delta) / total_delta * 100))
    iowait_percent = max(0.0, min(100.0, deltas["iowait"] / total_delta * 100))

    return utilization_percent, iowait_percent


class CPUCollector:
    """
    Collects CPU utilization metrics from Kubernetes pods.

    Reads /proc/stat twice, a short interval apart, inside the target
    container and computes utilization % and I/O-wait % from the deltas.
    Gracefully degrades if /proc/stat or a shell is unavailable.
    """

    def __init__(self, k8s_client: K8sClient, exec_timeout: int = 10, sample_interval: float = 1.0) -> None:
        """
        Initialize the CPU collector.

        Args:
            k8s_client: Kubernetes client instance.
            exec_timeout: Timeout for the exec command in seconds.
            sample_interval: Seconds between the two /proc/stat reads.
        """
        self.k8s_client = k8s_client
        self.exec_timeout = exec_timeout
        self.sample_interval = sample_interval

    def collect(
        self,
        pod_name: str,
        namespace: str,
        container: Optional[str] = None,
    ) -> Optional[CPUMetrics]:
        """
        Collect CPU utilization metrics from a pod.

        Args:
            pod_name: Name of the target pod.
            namespace: Namespace of the pod.
            container: Specific container name. If None, uses default.

        Returns:
            CPUMetrics if successful, None if /proc/stat is empty/unreadable.

        Raises:
            CPUMetricsUnavailableError: If exec fails (no shell, permission denied, etc).
        """
        _NOT_FOUND_TERMS = ["not found", "command not found", "no such file", "exec failed", "executable file"]

        command = [
            "sh", "-c",
            f"cat /proc/stat && echo {_SNAPSHOT_MARKER} && sleep {self.sample_interval} && cat /proc/stat",
        ]

        try:
            stdout, stderr = self.k8s_client.exec_in_pod(
                pod_name=pod_name,
                namespace=namespace,
                command=command,
                container=container,
                timeout=self.exec_timeout,
            )
        except Exception as e:
            error_str = str(e).lower()

            if "permission denied" in error_str:
                logger.debug(f"/proc/stat read permission denied in {namespace}/{pod_name}")
                raise CPUMetricsUnavailableError(
                    pod_name=pod_name,
                    reason="Permission denied for /proc/stat",
                )

            if any(term in error_str for term in _NOT_FOUND_TERMS):
                logger.debug(f"No shell available in {namespace}/{pod_name}")
                raise CPUMetricsUnavailableError(
                    pod_name=pod_name,
                    reason="No shell available to read /proc/stat",
                )

            logger.warning(f"CPU metrics collection failed for {namespace}/{pod_name}: {e}")
            raise CPUMetricsUnavailableError(
                pod_name=pod_name,
                reason=str(e),
            )

        if any(term in stderr.lower() for term in _NOT_FOUND_TERMS):
            raise CPUMetricsUnavailableError(
                pod_name=pod_name,
                reason="No shell available to read /proc/stat",
            )

        if _SNAPSHOT_MARKER not in stdout:
            logger.debug(f"Incomplete /proc/stat output from {namespace}/{pod_name}")
            return None

        snapshot_a, snapshot_b = stdout.split(_SNAPSHOT_MARKER, 1)
        snapshot_a, snapshot_b = snapshot_a.strip(), snapshot_b.strip()

        if not snapshot_a or not snapshot_b:
            return None

        try:
            utilization_percent, iowait_percent = _compute_utilization(snapshot_a, snapshot_b)
        except (StopIteration, ValueError, KeyError) as e:
            logger.debug(f"Failed to parse /proc/stat output from {namespace}/{pod_name}: {e}")
            return None

        return CPUMetrics(
            utilization_percent=round(utilization_percent, 2),
            iowait_percent=round(iowait_percent, 2),
            cpu_count=_count_cpus(snapshot_b) or None,
            collection_timestamp=datetime.utcnow(),
        )
