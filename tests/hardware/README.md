# Real-GPU Hardware Verification Tests

This directory verifies that piqc reports **accurate** facts when run against
a **real** GPU experiencing a deliberately-engineered condition. This is
different from `tests/integration/` (which verifies piqc's collection
*mechanism* — the right K8s API calls, the right exec commands, correct
parsing — against a `kind` cluster with mock containers standing in for
GPUs). Nothing in `tests/integration/` costs GPU money; everything in this
directory does, which is why it's kept separate and off by default.

## The three-layer test structure this fits into

1. **Rule logic** (`platform` repo, `services/control-plane/test_advisor_*.py`) —
   given a fact payload, does the rules engine produce the right
   recommendation? Pure Python, no cluster, no GPU, runs on every commit.
2. **Collection mechanism** (`tests/integration/`, this repo) — does piqc call
   the right APIs and parse responses correctly? Real `kind` cluster, mock
   containers, still no GPU.
3. **Hardware accuracy** (`tests/hardware/`, this directory) — when a real GPU
   actually hits the condition a rule is meant to detect, does piqc report it
   correctly? The only layer that costs real GPU-hours.

Layer 3 itself splits into two levels, both present in this directory:

- **Base telemetry, no rule involved** (`test_gpu_collector_real_hardware.py`) —
  does `GPUCollector`'s nvidia-smi exec path even work on a real node with
  real drivers mounted, independent of any inference framework or business
  rule? If this is broken, every rule-level test below is meaningless
  regardless of how correct its own logic is. Originally written directly
  against a live GKE cluster to debug a host-mounted-driver issue
  (`manifests/gke-real-gpu-workload.yaml`'s own comment); absorbed into this
  suite from an ad-hoc `k8s/` manifest on 2026-08-03.
- **Rule-level correctness** (`test_tier_misplacement.py` and future
  `test_*.py` files) — given a real, deliberately-misconfigured deployment,
  does a *specific rule* fire correctly end to end?

Layer 3 scales with the size of the rule catalog and the GPU tiers /
inference frameworks piqc needs to support — not with customer count. That
distinction is why a rule catalog test suite doesn't turn this into a
usage-scaling COGS line.

## Status note (2026-08-03)

Not every rule can be verified here yet — piqc doesn't collect the underlying
fact for several rule categories at all (serverless efficiency, training
efficiency, batch efficiency, and OOM-kill detection all confirmed to have no
collector as of this writing). Those need collector engineering work first;
no amount of GPU spend here helps them. `test_tier_misplacement.py` is
included as the first real, runnable rule-level example because `model.id`
and `deployment.gpuType` are confirmed-implemented facts today.
`test_gpu_collector_real_hardware.py` verifies the layer underneath any rule
— that nvidia-smi telemetry collection itself works on real hardware.

## Prerequisites

```bash
export PARALLELIQ_HARDWARE_TESTS=1
export PARALLELIQ_HARDWARE_KUBECONFIG=/path/to/kubeconfig  # must point at a real GPU node
```

A single cheap spot GPU (T4 or L4) is enough for most scenarios in this
directory — see each test module's docstring for the specific scenario and
hardware it needs. None of the currently-runnable scenarios need anything
larger.

## Quick Start

```bash
# Point at a cluster with a real GPU node, then:
poetry run pytest tests/hardware/ -v
```

## Directory Structure

```
tests/hardware/
├── conftest.py                                # hardware marker + PARALLELIQ_HARDWARE_TESTS gate
├── manifests/                                 # k8s manifests for each scenario's real deployment
│   ├── gke-real-gpu-workload.yaml             # raw matmul workload — base telemetry check
│   └── tier-misplacement-llama3-8b-t4.yaml    # real vLLM deployment — rule-level check
├── test_gpu_collector_real_hardware.py        # base telemetry — no rule/framework involved
├── test_tier_misplacement.py                  # first rule-level scenario — see docstring for setup
└── test_*.py                                  # one file per rule/scenario, added as collectors land
```

## Manifest coverage (2026-08-03) — sample only, not a complete set

`manifests/tier-misplacement-llama3-8b-t4.yaml` is the only manifest that
exists. It's a sample proving the pattern, not full coverage. Everything else
from the Bucket A/B scoping still needs either its own manifest or a
traffic-pattern variant of this one before it can be tested for real:

| Rule | Manifest status |
|---|---|
| `tier_misplacement_v1` / `gpu_overprovisioned_v1` (undersized case) | ✅ `tier-misplacement-llama3-8b-t4.yaml` |
| `gpu_overprovisioned_v1` (oversized case — small model, big GPU) | ❌ needs its own manifest (e.g. Llama-3-8B on an H100) |
| `idle_trickle_traffic_v1` | ❌ can reuse the existing manifest — no new YAML, just don't send traffic and wait past the 2-hour age threshold |
| `token_maxing_v1` | ❌ needs the existing manifest plus a long-context, high-concurrency load generator |
| `low_prefix_cache_hit_rate_v1` | ❌ needs the existing manifest plus a no-shared-prefix traffic script |
| `low_throughput_v1` | ❌ needs a deliberately bad vLLM config (e.g. `max_num_seqs=1`) — own manifest or variant |
| `dark_capacity_v1` | ❌ needs a multi-GPU node manifest that only claims a fraction of the GPUs |
| `fragmentation_v1` | ❌ needs a multi-node manifest set (fragmented free capacity + a pending tensor-parallel job) |

Do not assume any row past the first is covered just because this directory
and its README exist.

## Adding a new scenario

1. Confirm the fact the rule depends on actually has a piqc collector — check
   `docs/runtime_fact_schema.md` and grep `src/piqc` before writing the test.
   If there's no collector, this is an engineering ticket, not a test to
   write yet.
2. Check the table above — does an existing manifest cover this with just a
   different traffic pattern, or does it need new YAML in `manifests/`?
3. Write the scenario as a docstring first: what gets deployed, what
   deliberately-bad config triggers the condition, what GPU tier is actually
   needed (most scenarios need the cheapest tier that reproduces the
   condition, not a large one).
4. Mark the test `@pytest.mark.hardware`.
5. Update the manifest coverage table above.
