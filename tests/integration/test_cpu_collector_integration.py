"""
Integration tests for CPU metrics collection.

Tests /proc/stat exec and delta computation against real pods in the
kind test cluster.
"""

import pytest

from piqc.collectors.cpu_collector import CPUCollector


@pytest.mark.integration
class TestCPUCollectorIntegration:
    """Integration tests for CPU metrics collection."""

    def test_collect_cpu_metrics_vllm(self, k8s_client, wait_for_cluster):
        """Test CPU metrics collection from a vLLM pod."""
        cpu_collector = CPUCollector(k8s_client, exec_timeout=10, sample_interval=1.0)

        pods = k8s_client.list_pods(
            namespace="inference",
            label_selector="app=vllm-llama-70b",
        )

        if not pods:
            pytest.skip("No vLLM pods available")

        pod = pods[0]

        metrics = cpu_collector.collect(
            pod_name=pod.metadata.name,
            namespace="inference",
        )

        assert metrics is not None
        assert 0 <= metrics.utilization_percent <= 100
        assert 0 <= metrics.iowait_percent <= 100
        assert metrics.cpu_count is None or metrics.cpu_count > 0

    def test_collect_cpu_metrics_triton(self, k8s_client, wait_for_cluster):
        """Test CPU metrics collection from a Triton pod."""
        cpu_collector = CPUCollector(k8s_client, exec_timeout=10, sample_interval=1.0)

        pods = k8s_client.list_pods(
            namespace="inference",
            label_selector="app=triton-server",
        )

        if not pods:
            pytest.skip("No Triton pods available")

        pod = pods[0]

        metrics = cpu_collector.collect(
            pod_name=pod.metadata.name,
            namespace="inference",
        )

        assert metrics is not None
        assert 0 <= metrics.utilization_percent <= 100

    def test_collect_returns_none_for_shell_less_pod(self, k8s_client, wait_for_cluster):
        """Distroless/scratch pods with no shell should degrade gracefully, not raise unexpectedly."""
        pods = k8s_client.list_pods(
            namespace="inference",
            label_selector="app=tgi-falcon",
        )

        if not pods:
            pytest.skip("No TGI pods available")

        cpu_collector = CPUCollector(k8s_client, exec_timeout=10, sample_interval=1.0)
        pod = pods[0]

        # Either succeeds with sane bounds, or raises the documented unavailable error -
        # anything else is a bug.
        from piqc.utils.exceptions import CPUMetricsUnavailableError

        try:
            metrics = cpu_collector.collect(pod_name=pod.metadata.name, namespace="inference")
            if metrics is not None:
                assert 0 <= metrics.utilization_percent <= 100
        except CPUMetricsUnavailableError:
            pass
