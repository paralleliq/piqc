"""
Real-GPU verification of the base telemetry collection mechanism itself --
one level below the rule-level tests (e.g. test_tier_misplacement.py).

Where test_tier_misplacement.py verifies a specific rule is correct end to
end (does piqc flag a real vLLM deployment as tier-misplaced), this module
verifies something more foundational: does GPUCollector's nvidia-smi exec
path actually work at all on a real node with real NVIDIA drivers mounted,
independent of any inference framework or business rule. If this fails,
every rule-level hardware test built on top of it is meaningless regardless
of how correct its own logic is.

Backed by manifests/gke-real-gpu-workload.yaml: a raw PyTorch container
(no vLLM, no served model) running a continuous matmul loop on a real GKE
node with an L4, deliberately keeping the GPU busy so utilization and
memory-used are non-trivial and checkable, not just "the collector didn't
crash." Originally written directly against a live GKE cluster to debug a
host-mounted-driver issue (see the manifest's own comment) and absorbed into
this suite from tests/hardware -- adjacent, ad-hoc verification -- on
2026-08-03.
"""

import pytest

from piqc.collectors.gpu_collector import GPUCollector
from piqc.core.k8s_client import K8sClient

_NAMESPACE = "production"
_LABEL_SELECTOR = "app=gpu-matmul-workload"


@pytest.mark.hardware
def test_gpu_collector_reports_real_utilization_and_memory(hardware_kubeconfig):
    k8s_client = K8sClient(kubeconfig_path=hardware_kubeconfig)

    pods = k8s_client.list_pods(namespace=_NAMESPACE, label_selector=_LABEL_SELECTOR)
    if not pods:
        pytest.skip(
            f"No pods matching '{_LABEL_SELECTOR}' in namespace '{_NAMESPACE}' -- "
            "is manifests/gke-real-gpu-workload.yaml actually deployed on the "
            "target cluster?"
        )

    pod_name = pods[0].metadata.name
    collector = GPUCollector(k8s_client, exec_timeout=10)
    metrics = collector.collect(pod_name=pod_name, namespace=_NAMESPACE)

    assert metrics, "GPUCollector returned no metrics -- nvidia-smi exec path is broken"

    gpu = metrics[0]
    # The workload continuously runs 4096x4096 matmuls, so a genuinely
    # working real-hardware collection path should see real signal, not
    # zeros or the collector's own error sentinels.
    assert gpu.memory_used_mb > 0, "Reported zero memory used -- matmul workload should be allocating VRAM"
    assert gpu.utilization_percent is not None and gpu.utilization_percent > 0, (
        "Reported zero/None utilization on a pod running a continuous matmul loop"
    )
