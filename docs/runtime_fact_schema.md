# Cross-Runtime Fact Schema Reference

Maps every fact piqc collects (or has proposed collecting) across the three runtimes it's been evaluated against: vLLM, SGLang, and TGI. Built while investigating why several generic-sounding facts were silently vLLM-only (see the commit relocating `runtime.dtype`/`runtime.tensorParallel`/`model.quantization`/`model.maxModelLen`, and issue #9 for the observability facts still blocked on a schema change).

**Sourcing and confidence, by runtime:**
- **vLLM** — implemented, taken directly from `piqc_schema.py` and `piqc_generator.py`. Ground truth.
- **TGI** — proposed, not implemented. From the upstream research in issue #2 (well-researched, reviewed and agreed on by the team, deprioritized rather than abandoned).
- **SGLang** — proposed, not implemented. Researched for this document from SGLang's own docs and GitHub source. Has not been through the same team review TGI's list had — treat as a solid starting point for issue #8, not a finalized spec.

Use this doc when picking up #8 (SGLang) or #9 (generic observability facts) — it's meant to be the single place where "what does runtime X call this concept" lives, so that work doesn't require re-deriving the mapping from scratch (the exact problem Sara flagged on issue #2).

---

## Identity & Model Facts — Generic (implemented, no engine dependency)

| Concept | Fact | Notes |
|---|---|---|
| Engine type | `runtime.engineType` | Framework detection, works for any engine |
| Engine version | `runtime.engineVersion` | From container image tag |
| Model name | `model.name` | |
| Model architecture | `model.architecture` | |
| Parameter count | `model.parameters` / `model.parameterCount` | |

## Config Facts

| Concept | vLLM | SGLang (proposed) | TGI (proposed) | Category |
|---|---|---|---|---|
| Precision/dtype | `vllm.dtype` -> `runtime.dtype` | `--dtype` | `tgi.dtype` (`DTYPE`) | Generic — fixed |
| Tensor parallel size | `vllm.tensorParallelSize` -> `runtime.tensorParallel` | `--tp` / `--tp-size` | `tgi.numShard` (`NUM_SHARD`) | Generic — fixed |
| Quantization method | `inference.quantization` -> `model.quantization` | `--quantization` | `tgi.quantize` (`QUANTIZE`) | Generic — fixed |
| Max context length | `vllm.maxModelLen` -> `model.maxModelLen` | `--context-length` | `tgi.maxTotalTokens` (`MAX_TOTAL_TOKENS`) | Generic — fixed |
| Max concurrent requests | `vllm.maxNumSeqs` | `--max-running-requests` | `tgi.maxConcurrentRequests` | **No generic alias exists** — gap, folded into #9 |
| Pipeline parallel size | `vllm.pipelineParallelSize` | not found in SGLang docs | not applicable | Runtime-specific — vLLM-only concept among the three |
| GPU memory fraction | `vllm.gpuMemoryUtilization` | `--mem-fraction-static` | `tgi.cudaMemoryFraction` | **No generic alias exists** — gap, folded into #9 |

## Hardware & Infra Facts — Generic (implemented, sourced from nvidia-smi/k8s API, not the inference server)

| Concept | Fact |
|---|---|
| GPU count/type/memory | `hardware.gpuCount`, `hardware.gpuType`, `hardware.gpuMemoryGB` |
| GPU interconnect | `hardware.interconnect` |
| GPU util/mem/temp/power (observed) | `obs.gpu.utilAvgPct`, `obs.gpu.memUtilAvgPct`, `obs.gpu.temperatureC`, `obs.gpu.powerDrawW` |
| K8s replicas/CPU/memory/age | `k8s.replicas`, `k8s.cpuRequest`, `k8s.memoryRequest`, `k8s.ageHours` |
| Endpoint port, autoscaling | `endpoint.httpPort`, `autoscaling.enabled`, `autoscaling.metricType` |

## Observability — Latency & Throughput

| Concept | vLLM | SGLang metric | TGI (proposed) | Category |
|---|---|---|---|---|
| Time per output token, p95 | `obs.tpot.p95Ms` | `sglang_time_per_output_token_seconds` | `obs.tgi.interTokenLatencyMs` | Generic — blocked on #9 |
| End-to-end latency p95/p99 | `obs.requestLatency.p95Ms/p99Ms` | `sglang_e2e_request_latency_seconds` | `obs.tgi.e2eLatencyP50/95/99Ms` | Generic — blocked on #9 |
| Throughput (tokens/sec) | `obs.tps.avg` | `sglang_gen_throughput` | `obs.tgi.tokensPerSec` | Generic — blocked on #9 |
| Effective batch size | `obs.effectiveBatch.p95` | `sglang_num_running_reqs` (proxy) | `obs.tgi.batchCurrentSize` | Generic — blocked on #9 |
| Requests per second | `obs.rps.avg` | derivable from request counters | `obs.tgi.requestsTotal` (counter) | Generic — blocked on #9 |
| Time to first token (TTFT) | not in current schema | `sglang_time_to_first_token_seconds` (direct) | derivable, not direct | **Not a fact at all yet** — 7th generic candidate, folded into #9 |

## Observability — Cache & Queue (genuinely runtime-specific)

| Concept | vLLM | SGLang | TGI | Category |
|---|---|---|---|---|
| KV cache usage % | `obs.vllm.kvCacheUsagePct` | `sglang_token_usage` (KV cache fill) | **not exposed at all** | Runtime-specific — no cross-runtime equivalent for TGI. This is the core justification for the original TGI finding (#2). |
| Prefix cache hit rate | `obs.vllm.prefixCacheHitRate` | `sglang_cache_hit_rate` (RadixAttention, always-on) | **not exposed at all** | Runtime-specific — same pattern |
| Requests running/waiting (raw) | `obs.vllm.requestsRunning/Waiting` | `sglang_num_running_reqs`, `sglang_num_queue_reqs` | `obs.tgi.queueSize` | Runtime-specific raw counters — feed the generic batch/rps facts above, not generic themselves |
| Speculative decoding stats | not in current schema | `sglang_spec_num_steps`, `sglang_spec_num_draft_tokens` | not applicable | Runtime-specific — SGLang-only among these three |
| MFU (FLOPs utilization) | not in current schema | `sglang_estimated_flops_per_gpu_total` (opt-in via `--enable-mfu-metrics`) | not applicable | Runtime-specific — SGLang-only |

---

## Sources

- vLLM: `src/piqc/models/piqc_schema.py`, `src/piqc/generators/piqc_generator.py` (this repo)
- TGI: [piqc#2](https://github.com/paralleliq/piqc/issues/2)
- SGLang: [Server Arguments — SGLang Docs](https://docs.sglang.io/docs/advanced_features/server_arguments), [SGLang Prometheus Metrics Guide](https://kuncoro.io/blog/sglang-prometheus-metrics-guide/), [Prometheus Metrics — SGLang Docs](https://sgl-project-sglang-93.mintlify.app/observability/metrics)
