"""
Coverage analysis.

Assembles a CoverageReport from data already gathered during a normal scan
(GPU infra pods, detected serving frameworks) plus a small number of
additional read-only cluster checks (Prometheus Operator presence,
ServiceMonitor coverage of vLLM, OTel Collector presence).

Every status here is deliberately three-valued (detected / partial /
absent), never a hard pass/fail — "absent" means undetected from where this
scan could look, not confirmed missing. RBAC scope or a non-standard naming
convention can produce the same result as a true absence.
"""

from typing import Any

from piqc.core.k8s_client import K8sClient
from piqc.models.coverage import CoverageCheck, CoverageReport
from piqc.models.modelspec import ModelSpec
from piqc.utils.logger import get_logger

logger = get_logger(__name__)

_INFRA_LABELS: dict[str, str] = {
    "dcgm_exporter": "DCGM exporter",
    "nvidia_device_plugin": "NVIDIA device plugin",
    "gpu_feature_discovery": "GPU feature discovery",
    "node_feature_discovery": "Node feature discovery",
}

_OTEL_SERVICE_NAME_HINTS = ("otel-collector", "opentelemetry-collector")
_OTEL_PORTS = {4317, 4318}


class CoverageAnalyzer:
    """Builds a CoverageReport from scan data plus a few extra cluster reads."""

    def __init__(self, k8s_client: K8sClient) -> None:
        self.k8s_client = k8s_client

    def analyze(
        self,
        infra_pod_counts: dict[str, int],
        modelspecs: list[ModelSpec],
        total_nodes: int,
    ) -> CoverageReport:
        """
        Args:
            infra_pod_counts: category -> pod count, gathered during the
                normal per-namespace pod scan (see orchestrator._scan_namespace).
            modelspecs: ModelSpecs already produced by this scan.
            total_nodes: Total node count, for "N/M nodes covered" detail.

        Returns:
            CoverageReport across the GPU hardware, serving framework, and
            observability layers.
        """
        return CoverageReport(
            gpu_hardware=self._gpu_hardware_layer(infra_pod_counts, total_nodes),
            serving_framework=self._serving_framework_layer(modelspecs),
            observability=self._observability_layer(modelspecs),
        )

    def _gpu_hardware_layer(
        self,
        infra_pod_counts: dict[str, int],
        total_nodes: int,
    ) -> list[CoverageCheck]:
        checks: list[CoverageCheck] = []
        for category, label in _INFRA_LABELS.items():
            count = infra_pod_counts.get(category, 0)
            if count == 0:
                checks.append(
                    CoverageCheck(
                        name=label,
                        status="absent",
                        detail=f"No {label} pod found in any scanned namespace.",
                    )
                )
            elif total_nodes and count < total_nodes:
                checks.append(
                    CoverageCheck(
                        name=label,
                        status="partial",
                        detail=f"{count}/{total_nodes} node(s) covered.",
                    )
                )
            else:
                node_str = f"{count}/{total_nodes}" if total_nodes else str(count)
                checks.append(
                    CoverageCheck(name=label, status="detected", detail=f"{node_str} node(s).")
                )
        return checks

    def _serving_framework_layer(self, modelspecs: list[ModelSpec]) -> list[CoverageCheck]:
        checks: list[CoverageCheck] = []
        vllm_specs = [s for s in modelspecs if s.engine.name == "vllm"]
        unknown_specs = [s for s in modelspecs if s.engine.name == "unknown"]

        if vllm_specs:
            avg_confidence = sum(s.engine.detection_confidence for s in vllm_specs) / len(
                vllm_specs
            )
            checks.append(
                CoverageCheck(
                    name="vLLM",
                    status="detected",
                    detail=f"{len(vllm_specs)} deployment(s), {avg_confidence:.2f} avg confidence.",
                )
            )
        else:
            checks.append(
                CoverageCheck(
                    name="vLLM",
                    status="absent",
                    detail="No vLLM deployment detected in any scanned namespace.",
                )
            )

        if unknown_specs:
            checks.append(
                CoverageCheck(
                    name=f"{len(unknown_specs)} unclassified pod(s)",
                    status="absent",
                    detail=(
                        "Claim a GPU but don't match a known serving signature — may be "
                        "a custom runtime, or outside piqc's current RBAC scope."
                    ),
                )
            )

        return checks

    def _observability_layer(self, modelspecs: list[ModelSpec]) -> list[CoverageCheck]:
        checks: list[CoverageCheck] = []

        if self.k8s_client.has_api_group("monitoring.coreos.com"):
            checks.append(
                CoverageCheck(
                    name="Prometheus Operator",
                    status="detected",
                    detail="ServiceMonitor/PodMonitor CRDs registered on this cluster.",
                )
            )
            checks.extend(self._vllm_scrape_check(modelspecs))
        else:
            checks.append(
                CoverageCheck(
                    name="Prometheus Operator",
                    status="absent",
                    detail="No monitoring.coreos.com API group found on this cluster.",
                )
            )

        otel_present = self._detect_otel_collector()
        checks.append(
            CoverageCheck(
                name="OpenTelemetry Collector",
                status="detected" if otel_present else "absent",
                detail=(
                    "Found a service matching known OTel Collector naming/ports."
                    if otel_present
                    else "No OTLP endpoint (4317/4318) or known Collector service found."
                ),
            )
        )

        return checks

    def _vllm_scrape_check(self, modelspecs: list[ModelSpec]) -> list[CoverageCheck]:
        """Whether any ServiceMonitor's label selector actually matches the
        vLLM pods this scan found — Prometheus Operator being installed
        doesn't by itself mean vLLM is being scraped."""
        vllm_specs = [s for s in modelspecs if s.engine.name == "vllm"]
        if not vllm_specs:
            return []

        service_monitors = self.k8s_client.list_service_monitors()
        selectors = [
            match_labels
            for sm in service_monitors
            if (match_labels := sm.get("spec", {}).get("selector", {}).get("matchLabels", {}))
        ]

        scraped = sum(
            1
            for spec in vllm_specs
            if any(
                all((spec.metadata.labels or {}).get(k) == v for k, v in selector.items())
                for selector in selectors
            )
        )

        status = "detected" if scraped == len(vllm_specs) else "partial"

        return [
            CoverageCheck(
                name="vLLM → Prometheus",
                status=status,
                detail=(
                    f"{scraped}/{len(vllm_specs)} vLLM deployment(s) matched by a "
                    "ServiceMonitor selector — the rest emit metrics at /metrics that "
                    "nothing is currently collecting."
                    if scraped < len(vllm_specs)
                    else f"{scraped}/{len(vllm_specs)} vLLM deployment(s) matched by a ServiceMonitor selector."
                ),
            )
        ]

    def _detect_otel_collector(self) -> bool:
        services = self.k8s_client.list_all_services()
        for svc in services:
            name = (svc.metadata.name or "").lower() if svc.metadata else ""
            if any(hint in name for hint in _OTEL_SERVICE_NAME_HINTS):
                return True
            for port in ((svc.spec.ports or []) if svc.spec else []):
                if port.port in _OTEL_PORTS or _target_port_matches(port, _OTEL_PORTS):
                    return True
        return False


def _target_port_matches(port: Any, candidates: set[int]) -> bool:
    target = getattr(port, "target_port", None)
    return isinstance(target, int) and target in candidates
