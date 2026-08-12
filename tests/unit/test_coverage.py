"""
Unit tests for coverage analysis.

Tests infra-pod classification (discovery.py) and the CoverageAnalyzer that
turns scan data plus a few extra cluster reads into a CoverageReport.
"""

from unittest.mock import MagicMock

from kubernetes.client import V1Container, V1ObjectMeta, V1Pod, V1PodSpec, V1Service, V1ServicePort, V1ServiceSpec

from piqc.core.coverage import CoverageAnalyzer
from piqc.core.discovery import classify_infra_pod
from piqc.core.k8s_client import K8sClient
from piqc.models.coverage import CoverageReport
from tests.unit.test_piqc_generator import create_test_modelspec


def _pod(name: str = "pod", image: str = "nginx:latest") -> V1Pod:
    return V1Pod(
        metadata=V1ObjectMeta(name=name, namespace="default"),
        spec=V1PodSpec(containers=[V1Container(name="main", image=image)]),
    )


class TestClassifyInfraPod:
    """classify_infra_pod() — same detection discovery.py already used to
    exclude these pods from inference classification, now surfaced."""

    def test_dcgm_exporter_by_name(self) -> None:
        assert classify_infra_pod(_pod(name="dcgm-exporter-abcde")) == "dcgm_exporter"

    def test_dcgm_exporter_by_image(self) -> None:
        pod = _pod(name="some-pod", image="nvcr.io/nvidia/k8s/dcgm-exporter:3.1.0")
        assert classify_infra_pod(pod) == "dcgm_exporter"

    def test_nvidia_device_plugin(self) -> None:
        pod = _pod(name="nvidia-gpu-device-plugin-xyz")
        assert classify_infra_pod(pod) == "nvidia_device_plugin"

    def test_gpu_feature_discovery(self) -> None:
        pod = _pod(name="gpu-feature-discovery-abc")
        assert classify_infra_pod(pod) == "gpu_feature_discovery"

    def test_node_feature_discovery(self) -> None:
        pod = _pod(name="node-feature-discovery-worker-abc")
        assert classify_infra_pod(pod) == "node_feature_discovery"

    def test_non_infra_pod_returns_none(self) -> None:
        pod = _pod(name="vllm-llama-70b-abc", image="vllm/vllm-openai:v0.4.0")
        assert classify_infra_pod(pod) is None


def _mock_k8s_client(
    has_prometheus_operator: bool = False,
    service_monitors: list[dict] | None = None,
    services: list[V1Service] | None = None,
) -> K8sClient:
    client = MagicMock(spec=K8sClient)
    client.has_api_group.return_value = has_prometheus_operator
    client.list_service_monitors.return_value = service_monitors or []
    client.list_all_services.return_value = services or []
    return client


def _service(name: str, ports: list[int] | None = None) -> V1Service:
    service_ports = [V1ServicePort(port=p) for p in (ports or [])]
    return V1Service(
        metadata=V1ObjectMeta(name=name, namespace="monitoring"),
        spec=V1ServiceSpec(ports=service_ports) if service_ports else None,
    )


class TestGpuHardwareLayer:
    def test_present_on_every_node_is_detected(self) -> None:
        analyzer = CoverageAnalyzer(_mock_k8s_client())
        report = analyzer.analyze(
            infra_pod_counts={"dcgm_exporter": 4},
            modelspecs=[],
            total_nodes=4,
        )
        check = next(c for c in report.gpu_hardware if c.name == "DCGM exporter")
        assert check.status == "detected"
        assert "4/4" in check.detail

    def test_present_on_some_nodes_is_partial(self) -> None:
        analyzer = CoverageAnalyzer(_mock_k8s_client())
        report = analyzer.analyze(
            infra_pod_counts={"dcgm_exporter": 2},
            modelspecs=[],
            total_nodes=4,
        )
        check = next(c for c in report.gpu_hardware if c.name == "DCGM exporter")
        assert check.status == "partial"
        assert "2/4" in check.detail

    def test_missing_entirely_is_absent(self) -> None:
        analyzer = CoverageAnalyzer(_mock_k8s_client())
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=[], total_nodes=4)
        check = next(c for c in report.gpu_hardware if c.name == "DCGM exporter")
        assert check.status == "absent"

    def test_all_four_categories_present(self) -> None:
        analyzer = CoverageAnalyzer(_mock_k8s_client())
        report = analyzer.analyze(
            infra_pod_counts={
                "dcgm_exporter": 4,
                "nvidia_device_plugin": 4,
                "gpu_feature_discovery": 4,
                "node_feature_discovery": 4,
            },
            modelspecs=[],
            total_nodes=4,
        )
        assert len(report.gpu_hardware) == 4
        assert all(c.status == "detected" for c in report.gpu_hardware)


class TestServingFrameworkLayer:
    def test_vllm_detected_reports_count_and_avg_confidence(self) -> None:
        analyzer = CoverageAnalyzer(_mock_k8s_client())
        specs = [
            create_test_modelspec(name="a", framework="vllm", confidence=0.9),
            create_test_modelspec(name="b", framework="vllm", confidence=0.7),
        ]
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=specs, total_nodes=0)
        check = next(c for c in report.serving_framework if c.name == "vLLM")
        assert check.status == "detected"
        assert "2 deployment(s)" in check.detail
        assert "0.80 avg confidence" in check.detail

    def test_no_vllm_is_absent(self) -> None:
        analyzer = CoverageAnalyzer(_mock_k8s_client())
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=[], total_nodes=0)
        check = next(c for c in report.serving_framework if c.name == "vLLM")
        assert check.status == "absent"

    def test_unclassified_pods_reported_separately(self) -> None:
        analyzer = CoverageAnalyzer(_mock_k8s_client())
        specs = [
            create_test_modelspec(name="a", framework="vllm", confidence=0.9),
            create_test_modelspec(name="b", framework="unknown", confidence=0.1),
        ]
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=specs, total_nodes=0)
        names = [c.name for c in report.serving_framework]
        assert "1 unclassified pod(s)" in names


class TestObservabilityLayer:
    def test_no_prometheus_operator(self) -> None:
        analyzer = CoverageAnalyzer(_mock_k8s_client(has_prometheus_operator=False))
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=[], total_nodes=0)
        check = next(c for c in report.observability if c.name == "Prometheus Operator")
        assert check.status == "absent"

    def test_prometheus_operator_present_no_servicemonitor_for_vllm(self) -> None:
        client = _mock_k8s_client(has_prometheus_operator=True, service_monitors=[])
        analyzer = CoverageAnalyzer(client)
        specs = [create_test_modelspec(name="a", framework="vllm")]
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=specs, total_nodes=0)

        op_check = next(c for c in report.observability if c.name == "Prometheus Operator")
        assert op_check.status == "detected"

        scrape_check = next(c for c in report.observability if c.name == "vLLM → Prometheus")
        assert scrape_check.status == "partial"
        assert "0/1" in scrape_check.detail

    def test_servicemonitor_selector_matches_vllm_labels(self) -> None:
        # create_test_modelspec() sets metadata.labels={"app": "vllm"}
        service_monitors = [{"spec": {"selector": {"matchLabels": {"app": "vllm"}}}}]
        client = _mock_k8s_client(has_prometheus_operator=True, service_monitors=service_monitors)
        analyzer = CoverageAnalyzer(client)
        specs = [create_test_modelspec(name="a", framework="vllm")]
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=specs, total_nodes=0)

        scrape_check = next(c for c in report.observability if c.name == "vLLM → Prometheus")
        assert scrape_check.status == "detected"
        assert "1/1" in scrape_check.detail

    def test_servicemonitor_selector_not_matching_stays_partial(self) -> None:
        service_monitors = [{"spec": {"selector": {"matchLabels": {"app": "some-other-service"}}}}]
        client = _mock_k8s_client(has_prometheus_operator=True, service_monitors=service_monitors)
        analyzer = CoverageAnalyzer(client)
        specs = [create_test_modelspec(name="a", framework="vllm")]
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=specs, total_nodes=0)

        scrape_check = next(c for c in report.observability if c.name == "vLLM → Prometheus")
        assert scrape_check.status == "partial"

    def test_no_vllm_specs_skips_scrape_check(self) -> None:
        client = _mock_k8s_client(has_prometheus_operator=True, service_monitors=[])
        analyzer = CoverageAnalyzer(client)
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=[], total_nodes=0)
        names = [c.name for c in report.observability]
        assert "vLLM → Prometheus" not in names

    def test_otel_collector_detected_by_service_name(self) -> None:
        client = _mock_k8s_client(services=[_service("otel-collector")])
        analyzer = CoverageAnalyzer(client)
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=[], total_nodes=0)
        check = next(c for c in report.observability if c.name == "OpenTelemetry Collector")
        assert check.status == "detected"

    def test_otel_collector_detected_by_port(self) -> None:
        client = _mock_k8s_client(services=[_service("some-collector-service", ports=[4317])])
        analyzer = CoverageAnalyzer(client)
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=[], total_nodes=0)
        check = next(c for c in report.observability if c.name == "OpenTelemetry Collector")
        assert check.status == "detected"

    def test_otel_collector_absent(self) -> None:
        client = _mock_k8s_client(services=[_service("nginx", ports=[80])])
        analyzer = CoverageAnalyzer(client)
        report = analyzer.analyze(infra_pod_counts={}, modelspecs=[], total_nodes=0)
        check = next(c for c in report.observability if c.name == "OpenTelemetry Collector")
        assert check.status == "absent"


class TestCoverageReportHelpers:
    def test_count_by_status(self) -> None:
        client = _mock_k8s_client(has_prometheus_operator=False)
        analyzer = CoverageAnalyzer(client)
        report: CoverageReport = analyzer.analyze(
            infra_pod_counts={"dcgm_exporter": 4},
            modelspecs=[create_test_modelspec(name="a", framework="vllm")],
            total_nodes=4,
        )
        assert report.count_by_status("detected") >= 1
        assert report.count_by_status("absent") >= 1
        assert len(report.all_checks) == (
            len(report.gpu_hardware) + len(report.serving_framework) + len(report.observability)
        )
