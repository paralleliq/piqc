"""Unit tests for per-deployment node-spread analysis
(tensor_parallel_cross_node_v1's placement.nodeCount/node.gpuCount)."""

from datetime import datetime
from unittest.mock import MagicMock

from kubernetes.client import V1Node, V1NodeStatus, V1ObjectMeta, V1Pod, V1PodSpec, V1PodStatus

from piqc.core.orchestrator import ScanOrchestrator, WorkloadPlacement
from piqc.models.modelspec import InferenceDeployment


def _make_orchestrator() -> ScanOrchestrator:
    k8s_client = MagicMock()
    return ScanOrchestrator(k8s_client=k8s_client)


def _make_deployment(name: str, namespace: str, pod_names: list[str]) -> InferenceDeployment:
    return InferenceDeployment(
        name=name,
        namespace=namespace,
        framework="vllm",
        confidence=0.9,
        deployment_type="Deployment",
        pod_names=pod_names,
        detected_at=datetime.utcnow(),
    )


def _make_running_pod(name: str, namespace: str, node_name: str) -> V1Pod:
    return V1Pod(
        metadata=V1ObjectMeta(name=name, namespace=namespace),
        spec=V1PodSpec(containers=[], node_name=node_name),
        status=V1PodStatus(phase="Running"),
    )


def _make_node(name: str, gpu_count: int) -> V1Node:
    return V1Node(
        metadata=V1ObjectMeta(name=name),
        status=V1NodeStatus(allocatable={"nvidia.com/gpu": str(gpu_count)}),
    )


class TestAnalyzeWorkloadPlacement:
    def test_single_node_deployment(self):
        """All pods on one node -- the well-placed case, node_count == 1."""
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_running_pod("llama-70b-0", "inference", "node-a"),
            _make_running_pod("llama-70b-1", "inference", "node-a"),
        ]
        orch.k8s_client.list_nodes.return_value = [_make_node("node-a", 8)]
        deployments = [_make_deployment("llama-70b", "inference", ["llama-70b-0", "llama-70b-1"])]

        result = orch._analyze_workload_placement(deployments)

        placement = result["inference/llama-70b"]
        assert placement.node_count == 1
        assert placement.node_gpu_count == 8

    def test_cross_node_deployment(self):
        """Pods split across two nodes -- the misconfiguration
        tensor_parallel_cross_node_v1 exists to catch."""
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_running_pod("llama-70b-0", "inference", "node-a"),
            _make_running_pod("llama-70b-1", "inference", "node-b"),
        ]
        orch.k8s_client.list_nodes.return_value = [
            _make_node("node-a", 8),
            _make_node("node-b", 8),
        ]
        deployments = [_make_deployment("llama-70b", "inference", ["llama-70b-0", "llama-70b-1"])]

        result = orch._analyze_workload_placement(deployments)

        assert result["inference/llama-70b"].node_count == 2

    def test_node_gpu_count_uses_most_common_node(self):
        """3 pods on node-a, 1 on node-b -- node.gpuCount reports node-a's
        capacity, the node pool most of the deployment actually landed on."""
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_running_pod("p0", "inference", "node-a"),
            _make_running_pod("p1", "inference", "node-a"),
            _make_running_pod("p2", "inference", "node-a"),
            _make_running_pod("p3", "inference", "node-b"),
        ]
        orch.k8s_client.list_nodes.return_value = [
            _make_node("node-a", 8),
            _make_node("node-b", 4),
        ]
        deployments = [_make_deployment("mixed", "inference", ["p0", "p1", "p2", "p3"])]

        result = orch._analyze_workload_placement(deployments)

        assert result["inference/mixed"].node_count == 2
        assert result["inference/mixed"].node_gpu_count == 8

    def test_unresolvable_pod_node_excluded_from_deployment(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = []  # no pods resolvable
        orch.k8s_client.list_nodes.return_value = [_make_node("node-a", 8)]
        deployments = [_make_deployment("ghost", "inference", ["ghost-0"])]

        result = orch._analyze_workload_placement(deployments)

        assert "inference/ghost" not in result

    def test_node_with_no_gpu_capacity_data_leaves_node_gpu_count_none(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_running_pod("p0", "inference", "node-a"),
        ]
        orch.k8s_client.list_nodes.return_value = []  # node capacity unknown
        deployments = [_make_deployment("llama-70b", "inference", ["p0"])]

        result = orch._analyze_workload_placement(deployments)

        placement = result["inference/llama-70b"]
        assert placement.node_count == 1
        assert placement.node_gpu_count is None

    def test_multiple_deployments_keyed_independently(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_running_pod("a-0", "ns1", "node-a"),
            _make_running_pod("b-0", "ns2", "node-a"),
            _make_running_pod("b-1", "ns2", "node-b"),
        ]
        orch.k8s_client.list_nodes.return_value = [
            _make_node("node-a", 8),
            _make_node("node-b", 8),
        ]
        deployments = [
            _make_deployment("dep-a", "ns1", ["a-0"]),
            _make_deployment("dep-b", "ns2", ["b-0", "b-1"]),
        ]

        result = orch._analyze_workload_placement(deployments)

        assert result["ns1/dep-a"].node_count == 1
        assert result["ns2/dep-b"].node_count == 2

    def test_k8s_client_error_returns_empty_dict(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.side_effect = RuntimeError("connection refused")
        deployments = [_make_deployment("llama-70b", "inference", ["llama-70b-0"])]

        result = orch._analyze_workload_placement(deployments)

        assert result == {}

    def test_no_deployments_returns_empty_dict(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = []
        orch.k8s_client.list_nodes.return_value = []

        result = orch._analyze_workload_placement([])

        assert result == {}
