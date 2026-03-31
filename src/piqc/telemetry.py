"""
Benchmark telemetry for PIQC.

Collects anonymized GPU/model performance profiles and contributes them
to the ParallelIQ benchmark dataset. No identifying information is sent —
no cluster names, namespaces, workload names, IP addresses, or model paths.

Users opt in explicitly via --contribute-benchmarks.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Optional

from piqc.models.modelspec import ModelSpec
from piqc.utils.logger import get_logger


logger = get_logger(__name__)

_TELEMETRY_ENDPOINT = "https://telemetry.paralleliq.ai/v1/benchmarks"
_TIMEOUT_SECONDS = 5


def _parse_param_billions(model_name: str) -> Optional[float]:
    """Extract parameter count from model name string."""
    import re
    if not model_name:
        return None
    moe = re.search(r"\d+x(\d+(?:\.\d+)?)b", model_name, re.IGNORECASE)
    if moe:
        return float(moe.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)b", model_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _infer_architecture(model_name: str) -> Optional[str]:
    """Infer model architecture family from name."""
    if not model_name:
        return None
    name_lower = model_name.lower()
    for keyword, arch in [
        ("llama", "llama"),
        ("mistral", "mistral"),
        ("mixtral", "mixtral"),
        ("falcon", "falcon"),
        ("gemma", "gemma"),
        ("qwen", "qwen"),
        ("phi", "phi"),
        ("codellama", "llama"),
        ("bloom", "bloom"),
        ("mpt", "mpt"),
    ]:
        if keyword in name_lower:
            return arch
    return None


def build_payload(modelspecs: list[ModelSpec]) -> list[dict]:
    """
    Build anonymized benchmark payload from a list of ModelSpecs.

    Strips all identifying information. Returns only performance profiles:
    model size, architecture, GPU type/count, observed MFU and utilization.
    """
    records = []

    for spec in modelspecs:
        gpus = spec.resources.gpus
        if not gpus:
            continue

        model_name = spec.model.name or ""
        param_billions = _parse_param_billions(model_name)
        architecture = _infer_architecture(model_name)

        # Only include if we have at least param count or architecture
        if param_billions is None and architecture is None:
            continue

        gpu_type = gpus[0].type
        gpu_count = len(gpus) * spec.resources.replicas

        utilizations = [g.utilization for g in gpus if g.utilization is not None]
        avg_util = round(sum(utilizations) / len(utilizations), 1) if utilizations else None

        record: dict = {
            "model": {},
            "hardware": {
                "gpuType": gpu_type,
                "gpuCount": gpu_count,
            },
            "observed": {},
        }

        if param_billions is not None:
            record["model"]["parametersBillions"] = param_billions
        if architecture is not None:
            record["model"]["architecture"] = architecture
        if spec.inference.tensor_parallel_size:
            record["hardware"]["tensorParallel"] = spec.inference.tensor_parallel_size
        if spec.inference.precision:
            record["hardware"]["precision"] = spec.inference.precision
        if avg_util is not None:
            record["observed"]["gpuUtilPct"] = avg_util

        # Runtime metrics if available
        if spec.runtime_state and spec.runtime_state.vllm:
            vllm = spec.runtime_state.vllm
            if vllm.generation_tokens_per_sec:
                record["observed"]["generationTokensPerSec"] = round(vllm.generation_tokens_per_sec, 1)
            if vllm.gpu_cache_usage_percent:
                record["observed"]["kvCacheUsagePct"] = round(vllm.gpu_cache_usage_percent, 1)
            if vllm.ttft_p95:
                record["observed"]["ttftP95Ms"] = round(vllm.ttft_p95 * 1000, 1)

        records.append(record)

    return records


def contribute(modelspecs: list[ModelSpec], piqc_version: str) -> tuple[int, list[dict]]:
    """
    Build and POST anonymized benchmark records to the telemetry endpoint.

    Returns (records_sent, payload) so the caller can show the user what was sent.
    Never raises — failures are logged and silently ignored.
    """
    records = build_payload(modelspecs)
    if not records:
        return 0, []

    body = json.dumps({
        "schemaVersion": "piqc-telemetry.v0.1",
        "toolVersion": piqc_version,
        "records": records,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            _TELEMETRY_ENDPOINT,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": f"piqc/{piqc_version}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS):
            pass
        return len(records), records
    except Exception as exc:
        logger.debug("Telemetry post failed (non-fatal): %s", exc)
        return 0, records
