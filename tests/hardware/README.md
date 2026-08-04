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
included as the first real, runnable example because `model.id` and
`deployment.gpuType` are confirmed-implemented facts today.

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
├── conftest.py              # hardware marker + PARALLELIQ_HARDWARE_TESTS gate
├── test_tier_misplacement.py   # first real scenario — see docstring for setup
└── test_*.py                # one file per rule/scenario, added as collectors land
```

## Adding a new scenario

1. Confirm the fact the rule depends on actually has a piqc collector — check
   `docs/runtime_fact_schema.md` and grep `src/piqc` before writing the test.
   If there's no collector, this is an engineering ticket, not a test to
   write yet.
2. Write the scenario as a docstring first: what gets deployed, what
   deliberately-bad config triggers the condition, what GPU tier is actually
   needed (most scenarios need the cheapest tier that reproduces the
   condition, not a large one).
3. Mark the test `@pytest.mark.hardware`.
