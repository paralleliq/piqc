"""Unit tests for Pending-GPU-pod detection (fragmentation_v1's demand signal)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from kubernetes.client import (
    V1Container,
    V1EnvVar,
    V1ObjectMeta,
    V1OwnerReference,
    V1Pod,
    V1PodSpec,
    V1ResourceRequirements,
)

from piqc.core.orchestrator import PendingGPUPod, ScanOrchestrator


def _make_orchestrator() -> ScanOrchestrator:
    k8s_client = MagicMock()
    return ScanOrchestrator(k8s_client=k8s_client)


def _make_pending_pod(
    name: str = "vllm-70b-abc123-xyz",
    namespace: str = "inference",
    gpu_request: str = "4",
    env_vars: dict | None = None,
    args: list | None = None,
    owner_kind: str | None = "ReplicaSet",
    owner_name: str | None = "vllm-70b-abc123",
    created_minutes_ago: float = 47.0,
) -> V1Pod:
    env_list = [V1EnvVar(name=k, value=v) for k, v in (env_vars or {}).items()] or None
    container = V1Container(
        name="main",
        image="vllm/vllm-openai:latest",
        env=env_list,
        args=args,
        resources=V1ResourceRequirements(requests={"nvidia.com/gpu": gpu_request}),
    )
    owner_references = None
    if owner_kind and owner_name:
        owner_references = [V1OwnerReference(api_version="apps/v1", kind=owner_kind, name=owner_name, uid="u")]
    created_at = datetime.now(timezone.utc) - timedelta(minutes=created_minutes_ago)
    return V1Pod(
        metadata=V1ObjectMeta(
            name=name,
            namespace=namespace,
            owner_references=owner_references,
            creation_timestamp=created_at,
        ),
        spec=V1PodSpec(containers=[container]),
        status=MagicMock(phase="Pending"),
    )


class TestAnalyzePendingGpuPods:
    def test_no_pending_pods_returns_empty(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = []

        result = orch._analyze_pending_gpu_pods()

        assert result == []

    def test_pending_pod_with_no_gpu_request_excluded(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [_make_pending_pod(gpu_request="0")]

        result = orch._analyze_pending_gpu_pods()

        assert result == []

    def test_pending_gpu_pod_detected(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_pending_pod(gpu_request="4", created_minutes_ago=47.0)
        ]

        result = orch._analyze_pending_gpu_pods()

        assert len(result) == 1
        pod = result[0]
        assert isinstance(pod, PendingGPUPod)
        assert pod.namespace == "inference"
        assert pod.gpus_needed == 4
        assert 46.0 < pod.pending_minutes < 48.0

    def test_owner_reference_name_strips_replicaset_hash(self):
        """pod_name should be the deployment name, not the individual pod's
        own unique name -- matches the same deployment-level granularity the
        rest of piqc already uses for running workloads."""
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_pending_pod(
                name="vllm-70b-abc123-xyz",
                owner_kind="ReplicaSet",
                owner_name="vllm-70b-abc123",
            )
        ]

        result = orch._analyze_pending_gpu_pods()

        assert result[0].pod_name == "vllm-70b"

    def test_no_owner_reference_falls_back_to_pod_name(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_pending_pod(name="bare-pod", owner_kind=None, owner_name=None)
        ]

        result = orch._analyze_pending_gpu_pods()

        assert result[0].pod_name == "bare-pod"

    def test_tensor_parallel_size_derives_tensor_strategy(self):
        """fragmentation_v1 is gated on deployment.parallelismStrategy ==
        "tensor" -- a Pending pod's spec (env/args) is readable even though
        it was never scheduled, the same source a running pod's parallelism
        facts already come from."""
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_pending_pod(env_vars={"TENSOR_PARALLEL_SIZE": "8"})
        ]

        result = orch._analyze_pending_gpu_pods()

        assert result[0].parallelism_strategy == "tensor"

    def test_tensor_parallel_size_via_cli_arg(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_pending_pod(args=["--tensor-parallel-size", "8"])
        ]

        result = orch._analyze_pending_gpu_pods()

        assert result[0].parallelism_strategy == "tensor"

    def test_pipeline_parallel_size_derives_pipeline_strategy(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_pending_pod(env_vars={"PIPELINE_PARALLEL_SIZE": "4"})
        ]

        result = orch._analyze_pending_gpu_pods()

        assert result[0].parallelism_strategy == "pipeline"

    def test_no_parallelism_config_leaves_strategy_none(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [_make_pending_pod()]

        result = orch._analyze_pending_gpu_pods()

        assert result[0].parallelism_strategy is None

    def test_tensor_parallel_size_of_one_is_not_tensor_strategy(self):
        """A size of 1 means "not using this strategy" (vLLM's own
        default), not "using it with a degree of 1"."""
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.return_value = [
            _make_pending_pod(env_vars={"TENSOR_PARALLEL_SIZE": "1"})
        ]

        result = orch._analyze_pending_gpu_pods()

        assert result[0].parallelism_strategy is None

    def test_k8s_client_error_returns_empty_list(self):
        orch = _make_orchestrator()
        orch.k8s_client.list_all_pods.side_effect = RuntimeError("connection refused")

        result = orch._analyze_pending_gpu_pods()

        assert result == []
