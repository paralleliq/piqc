"""
Real-GPU verification for tier_misplacement_v1 / gpu_overprovisioned_v1
(platform/workflows/rules/inference_reliability_v1.yaml).

Scenario: deploy meta-llama/Llama-3-8B-Instruct on a single, cheap T4 (16GB) --
a real tier mismatch, since an 8B model in fp16 needs more headroom than a
T4 comfortably gives it for production traffic. This is the cheapest possible
scenario in the rule catalog: no traffic generation needed, no failure to
force -- just confirm piqc reports the deployment's actual model identity and
GPU type correctly, since those are the two facts tier_misplacement_v1 and
gpu_overprovisioned_v1 both key off (see the rule catalog's canaryPolicy
evidence lists).

Hardware needed: 1x T4 (or any single cheap GPU). No multi-node cluster,
no load generator.

This depends on `model.id` and `deployment.gpuType`, both confirmed
implemented in piqc's fact schema (docs/runtime_fact_schema.md, "Identity &
Model Facts" and "Hardware & Infra Facts" tables) -- unlike serverless,
training, batch, and OOM-kill facts, which have no collector yet (see
tests/hardware/README.md).
"""

import json

import pytest

from piqc.core.k8s_client import K8sClient
from piqc.core.orchestrator import ScanOrchestrator
from piqc.generators.piqc_generator import PIQCGenerator

# Namespace and deployment name expected to already exist on the target
# cluster -- provisioning the actual deployment (kubectl apply) is a
# prerequisite step, not something this test does itself. See the module
# docstring for exactly what to deploy.
_NAMESPACE = "hardware-verification"
_EXPECTED_MODEL_ID = "meta-llama/Llama-3-8B-Instruct"
_EXPECTED_GPU_TYPE = "nvidia-t4"


@pytest.mark.hardware
def test_piqc_reports_correct_model_and_gpu_identity(hardware_kubeconfig, tmp_path):
    k8s_client = K8sClient(kubeconfig_path=hardware_kubeconfig)
    orchestrator = ScanOrchestrator(k8s_client, workers=1)

    result = orchestrator.scan(namespaces=[_NAMESPACE])
    assert not result.errors, f"Scan produced errors: {result.errors}"
    assert result.modelspecs, (
        f"No workloads discovered in namespace '{_NAMESPACE}' -- "
        "is the deployment from this module's docstring actually running?"
    )

    output_path = str(tmp_path / "piqc-facts.json")
    PIQCGenerator().generate(
        result.modelspecs,
        output_path=output_path,
        namespaces=[_NAMESPACE],
    )

    with open(output_path) as f:
        bundle = json.load(f)

    workloads = bundle.get("workloads", bundle.get("data", {}).get("workloads", []))
    assert workloads, "piqc-facts.json contained no workloads"

    facts = workloads[0]["facts"]
    assert facts["model.id"]["value"] == _EXPECTED_MODEL_ID
    assert facts["deployment.gpuType"]["value"] == _EXPECTED_GPU_TYPE
