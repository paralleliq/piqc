<p align="center">
  <img src="https://img.shields.io/badge/PIQC-v1.3.1-blue?style=for-the-badge&logo=kubernetes&logoColor=white" alt="PIQC Version"/>
  <img src="https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-BSL%201.1-orange?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/vLLM-Supported-purple?style=for-the-badge" alt="vLLM"/>
  <img src="https://img.shields.io/badge/Ray%20Serve-Supported-blue?style=for-the-badge" alt="Ray Serve"/>
  <img src="https://img.shields.io/github/stars/paralleliq/piqc?style=for-the-badge&logo=github&color=yellow" alt="GitHub Stars"/>
</p>

<h1 align="center">piqc — Inference Fact Collector for AI Infrastructure Optimization</h1>

<p align="center">
  <strong>Most AI clusters waste 20–40% of GPU spend. piqc finds it in one command.</strong>
  <br/><br/>
  vLLM-native · Hardware-pluggable · Read-only · No agents · No sidecars · Nothing installed permanently
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-orchestrator-integrations">Integrations</a> •
  <a href="#-commands">Commands</a> •
  <a href="#-output-formats">Output Formats</a> •
  <a href="#-installation">Installation</a>
</p>

---

## What is piqc?

piqc is a source-available inference fact collector for Kubernetes clusters. It collects model-aware facts — what is running, on what hardware, at what cost, with what waste — and surfaces them as a standardized facts bundle that feeds an optimization layer. It also prints a human-readable cost report so you can act on the results immediately without any external platform.

It is the fastest way to answer: **how much GPU spend is my Kubernetes cluster wasting right now?**

```
piqc
  └── inference collector (vLLM-native)     ← collects model, GPU, KV cache, throughput facts
  └── hardware collector (plugin)           ← vendors contribute their own telemetry
        ├── nvidia/   (DCGM, MIG state)
        ├── amd/      (ROCm metrics)
        └── your-hardware/
```

Facts flow to the [Paralleliq optimization layer](https://paralleliq.ai), which maps waste to the model level and routes remediations through human-approved workflows. piqc runs standalone too — no platform required to get value from the cost report.

piqc surfaces three types of waste that standard Kubernetes monitoring (`kubectl top`, `kube-state-metrics`, Prometheus node exporters) cannot detect on their own:
- **Idle allocation** — pods holding GPU resources with near-zero compute utilization
- **Tier misplacement** — models running on GPU tiers with far more memory or compute than they need
- **Dark capacity** — GPU nodes with no pods scheduled at all

It works with any Kubernetes cluster running GPU inference workloads — GKE, EKS, AKS, on-prem, or bare metal. vLLM is the primary supported inference framework, with Ray Serve workloads also detected (GPU type, utilization, and cost — deeper runtime metrics like KV cache and token throughput are vLLM-specific). Hardware fact collection is pluggable — see [Contributing a Hardware Plugin](#contributing-a-hardware-plugin).

---

## What you'll see

Run `piqc scan` against your cluster and get an instant cost report:

```
                                              Discovered Inference Deployments
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Deployment                  ┃ Engine  ┃ GPU                ┃ Replicas ┃ Age ┃ GPU Util ┃  MFU ┃ $/1K tokens ┃   $/hr ┃   Idle $/day ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━┩
│ meta-llama/Llama-3-70B-Inst │ vllm    │ 8xH100-SXM4-80GB ⚠ │        2 │  6h │       4% │ 3.1% │     $0.0842 │ $68.00 │    $1,566.72 │
│ mistral-7b-instruct         │ vllm    │ 1xA100-SXM4-40GB ⚠ │        1 │  2d │      11% │ 8.4% │     $0.0073 │  $2.50 │       $53.40 │
│ codellama-34b-staging       │ vllm    │ 4xH100-SXM4-80GB ⚠ │        1 │ 19d │       0% │  N/A │         N/A │ $17.00 │      $408.00 │
│ embedding-bge-large         │ vllm    │ 1xT4 ✓             │        3 │ 14h │      82% │  N/A │     $0.0002 │  $1.35 │        $5.83 │
│ unknown-runtime-7f3a2       │ unknown │ 2xA100-SXM4-80GB ? │        1 │ 31d │      N/A │  N/A │         N/A │  $7.00 │ util unknown │
└─────────────────────────────┴─────────┴────────────────────┴──────────┴─────┴──────────┴──────┴─────────────┴────────┴──────────────┘

  ⚠ tier larger than this model requires   ·   ? model size unknown — fit not checked   ·   Age running 3+ days — confirm it's still needed

╭──────────────────────────────────── Cost Summary ──────────────────────────────────────╮
│   Total GPU spend rate      : $95.85/hr                                                │
│                                                                                        │
│   Leased & idle (util <60%) : $2,033.95/day  (low utilization — may reflect traffic    │
│ patterns; worth investigating)                                                         │
│   Unallocated nodes         : $1,152.00/day  (12 GPU(s) with no pods scheduled)        │
│   Tier misplacement         :   $721.20/day  (3 model(s) on oversized GPU tier)        │
│                                                                                        │
│   Total estimated leak      : $3,907.15/day  ($1,426,110/yr at current rate)           │
│                                                                                        │
│   Confirmed waste   : unallocated nodes, tier misplacement                             │
│   Signals to investigate: low GPU utilization (verify against traffic data)            │
│                                                                                        │
│   Avg MFU (active deployments) : 15.7%  (healthy range: 30–60%)                        │
╰────────────────────────────────────────────────────────────────────────────────────────╯
  ─────────────────────────────────────────────────────────────
  → Want to know what this waste is actually costing you?
  Paralleliq turns these signals into confirmed findings with dollar impact,
  continuous monitoring, and automated remediation — so you act on facts, not guesses.
  Running proprietary models or on-prem hardware? We'll configure it for your exact costs.
  Free to get started: paralleliq.ai  ·  Questions? sam@paralleliq.ai
```

**piqc is free and source-available** (Business Source License 1.1, converting to Apache 2.0 in 2028). The scan gives you the full picture — what is running, on what hardware, at what cost, and where the waste is. For continuous monitoring, alerting across your fleet, and automated remediation workflows, see [paralleliq.ai](https://paralleliq.ai).

---

## 🚀 Quick Start

### Option 1: Run as a Kubernetes Job (recommended)

Runs inside your cluster — no Docker auth or kubeconfig wrangling:

```bash
# Step 1 — Apply RBAC permissions (one-time setup)
kubectl apply -f https://raw.githubusercontent.com/paralleliq/piqc/main/deploy/rbac.yaml

# Step 2 — Run the scan
kubectl apply -f https://raw.githubusercontent.com/paralleliq/piqc/main/deploy/scan-job.yaml

# Step 3 — View the output
kubectl logs -f job/piqc-scan -n kube-system

# Clean up when done
kubectl delete job piqc-scan -n kube-system
```

> The job auto-deletes itself after 10 minutes (`ttlSecondsAfterFinished: 600`).

**Want a trend, not just a snapshot?** A single scan tells you today's waste. To see it change over time — dark capacity climbing, tier misplacement recurring — run the scan on a schedule and push it to the platform: see [`deploy/scan-cronjob.yaml`](deploy/scan-cronjob.yaml). Same free scan, same RBAC, just recurring and authenticated with a cluster API key instead of one-shot and local-only.

---

### Option 2: Install from PyPI and run locally

```bash
# Recommended — pipx installs piqc into its own isolated environment
pipx install piqc

# Or with pip (see warning below)
pip install piqc

piqc scan --format table
```

> **Use `pipx`, not bare `pip install`, if you already have other Python CLI tools on your machine.** piqc requires pydantic v2. Tools like `dstack` pin pydantic v1, so a plain `pip install piqc` into a shared environment (e.g. your conda base) can silently up/downgrade pydantic and break them. `pipx` keeps piqc's dependencies fully isolated, so this can't happen.

---

### Option 3: Run with Docker from your laptop

```bash
# Export a static kubeconfig with embedded credentials
kubectl config view --raw --flatten > /tmp/piqc-kubeconfig.yaml

# Run the scan
docker run --rm \
  -v /tmp/piqc-kubeconfig.yaml:/root/.kube/config \
  ghcr.io/paralleliq/piqc:latest \
  scan --format table
```

Supports both `linux/amd64` and `linux/arm64`.

---

### Option 4: Install from source

```bash
git clone https://github.com/paralleliq/piqc.git
cd piqc
poetry install
poetry run piqc scan --format table
```

---

## ✨ Features

### 🔍 Intelligent Discovery
- **Auto-Detection**: Automatically discovers vLLM and Ray Serve inference deployments across all namespaces
- **Weighted Confidence Scoring**: Uses multiple signals (images, env vars, CLI args, labels) with weighted scoring
- **Framework Detection**: Identifies vLLM and Ray Serve with high accuracy using pattern matching and heuristics

### 📊 Comprehensive Metrics Collection
- **GPU Metrics**: Real-time GPU utilization, memory, temperature, and power via `nvidia-smi`
- **Runtime Metrics**: Collects vLLM API metrics including:
  - Request latency (P50, P95, P99)
  - Token throughput (prefill & decode)
  - KV cache utilization
  - Queue depth and active requests
  - Health status

### 💰 Waste Detection
- **GPU underutilization** — Deployments below 60% utilization threshold, with dollar waste per day and annualized
- **Dark capacity** — GPU nodes with no pods scheduled (paying for nodes sitting empty)
- **Tier misplacement** — Models running on an oversized GPU tier, with estimated cost delta per day
- **Fragmentation** — Nodes with free GPU slots too small to fit any running model
- **Pending GPU pods** — Workloads blocked from scheduling, shown with wait time
- **Cost Summary panel** — Total spend rate, all waste categories, total estimated leak per day and per year
- **MFU (Model FLOPS Utilization)** — Observed compute vs. theoretical GPU peak per deployment
- **Cost per 1K tokens** — GPU spend translated into a business metric comparable to API pricing

### 🔭 Coverage Report (`--coverage`)
- **GPU Hardware Layer** — DCGM exporter, NVIDIA device plugin, GPU/node feature discovery: what's actually instrumented vs. what isn't
- **Serving Framework Layer** — vLLM detection confidence, and GPU-claiming pods that don't match a known serving signature
- **Observability Stack** — Prometheus Operator presence, whether vLLM is actually being scraped, OpenTelemetry Collector presence
- Three-state reporting (detected / partially wired up / not visible from here) — never a hard pass/fail, since "not visible" can mean genuinely absent or just outside piqc's current RBAC scope
- Read-only, same as the rest of piqc — a few extra Kubernetes API discovery calls, no new write permissions

### 📄 Multiple Output Formats
| Format | Description |
|--------|-------------|
| **Table** | Cost report with MFU, $/1K tokens, idle waste (default) |
| **YAML** | Kubernetes-style inference deployment files |
| **JSON** | Machine-readable JSON output |
| **PIQC Facts** | Standardized facts bundle for control plane integration |

### 🚀 Production-Ready
- **Parallel Processing**: Multi-threaded scanning with configurable workers
- **RBAC Support**: Pre-configured ClusterRole and ServiceAccount manifests
- **Flexible Modes**: Auto-detect, remote (kubeconfig), or in-cluster execution
- **Timeout Controls**: Configurable operation timeouts
- **Docker Image**: Pre-built multi-platform image (`linux/amd64` + `linux/arm64`) on GitHub Container Registry

### 🔌 Hardware Plugins

piqc's hardware fact collection is designed to be pluggable. The inference collector (vLLM) is maintained in this repo. Hardware vendors contribute their own collectors using the same fact schema — so AMD, Intel, and custom hardware telemetry can be added without touching the core.

<table>
<tr>
<td width="50%" valign="top">

**🔴 AMD GPU Plugin**

Hardware plugin for AMD Instinct GPUs via `rocm-smi`:
- AMD Instinct MI250X/MI300X detection
- GPU utilization, memory & temperature metrics
- ROCm ecosystem integration
- Contributed by the community / AMD

</td>
<td width="50%" valign="top">

**🌐 LLM-D (LLM-Distributed)**

Discovery and documentation for distributed LLM inference:
- Distributed inference topology mapping
- Multi-node GPU coordination metrics
- Cross-node performance aggregation
- Distributed KV cache analysis

</td>
</tr>
</table>

Want to contribute a hardware plugin? See [Contributing a Hardware Plugin](#contributing-a-hardware-plugin).

---

## 🔌 Orchestrator Integrations

### dstack

[`paralleliq-dstack-plugin`](https://github.com/paralleliq/paralleliq-dstack-plugin) ([PyPI](https://pypi.org/project/paralleliq-dstack-plugin/)) hooks into [dstack](https://dstack.ai)'s plugin system. When a GPU fleet or task is applied against a dstack project on a Kubernetes backend, it surfaces the piqc scan commands so you know to check for waste on the cluster dstack just provisioned onto.

```bash
pip install paralleliq-dstack-plugin
```

dstack discovers it automatically via Python entry points — no further configuration required. See the plugin repo for what it does today and its current limitations.

### vLLM Production Stack

*Proposed, pending review.* [vLLM Production Stack](https://github.com/vllm-project/production-stack) is a Helm-based reference deployment for vLLM on Kubernetes, with its own Prometheus/Grafana observability stack and an LMCache-based KV cache offloading tutorial. piqc is complementary, not competing — it sits above that telemetry and turns it into prioritized, costed findings (unallocated GPU capacity, idle deployments, low prefix cache hit rate) rather than another dashboard. A draft tutorial written in their house style, [`docs/integrations/26-detect-gpu-waste.md`](docs/integrations/26-detect-gpu-waste.md), is ready to propose as a PR into their `tutorials/` folder.

---

## 📋 Commands

### `piqc scan`

**Scan your Kubernetes cluster for inference workloads and surface GPU waste.**

```bash
piqc scan [OPTIONS]
```

#### Scan Options

| Option | Default | Description |
|--------|---------|-------------|
| `--kubeconfig PATH` | `~/.kube/config` | Path to kubeconfig file |
| `--context TEXT` | current | Kubernetes context to use |
| `-n, --namespace TEXT` | all | Specific namespace to scan |
| `--format [yaml\|json\|table]` | `yaml` | Output format |
| `-o, --output PATH` | `./output` | Output directory for generated files |

#### Collection Options

| Option | Default | Description |
|--------|---------|-------------|
| `--collect-runtime` | `false` | Collect runtime metrics via vLLM API |
| `--no-exec` | `false` | Disable pod exec (skip GPU metrics) |
| `--no-logs` | `false` | Disable log reading |
| `--aggregate/--no-aggregate` | `aggregate` | Aggregate metrics across pod replicas |
| `--contribute-benchmarks` | `false` | Contribute anonymized GPU/model performance data to the Paralleliq benchmark dataset |
| `--coverage` | `false` | Also report what GPU infra, serving framework, and observability tooling this scan could see in your cluster (table output only) |

When a scan run at an interactive terminal (table output, results found) finishes, piqc asks a one-time, skippable question: whether to send your name/company/email plus a compact scan summary to Paralleliq for a follow-up. Declining is a clean exit — nothing is sent. Scripted, piped, or `--format json`/`yaml` invocations never show this prompt.

#### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--combined` | `false` | Generate single combined output file |
| `--output-piqc` | `false` | Generate `piqc-facts.json` (PIQC v0.1 schema) |

#### Execution Options

| Option | Default | Description |
|--------|---------|-------------|
| `--timeout INT` | `30` | Operation timeout in seconds |
| `--workers INT` | `10` | Number of parallel workers |
| `--mode [auto\|remote\|incluster\|dry-run]` | `auto` | Execution mode |
| `-v, --verbose` | `false` | Enable verbose output |
| `--debug` | `false` | Enable debug mode with detailed trace |

#### Examples

```bash
# Basic scan — discover all vLLM deployments and surface waste
piqc scan

# Scan specific namespace with JSON output
piqc scan -n production --format json

# Quick scan without GPU metrics (faster)
piqc scan --no-exec

# Collect runtime metrics from vLLM API
piqc scan --collect-runtime

# Also report GPU infra / serving framework / observability coverage
piqc scan --format table --coverage

# Generate PIQC facts bundle for control plane integration
piqc scan --output-piqc -o ./facts

# Table output to console (human-readable)
piqc scan --format table

# Custom kubeconfig and context
piqc scan --kubeconfig /path/to/config --context my-cluster

# Contribute anonymized GPU/model benchmarks to Paralleliq dataset
piqc scan --contribute-benchmarks
```

---

### `piqc test-connection`

**Test connection to Kubernetes cluster and verify required permissions.**

```bash
piqc test-connection [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--kubeconfig PATH` | `~/.kube/config` | Path to kubeconfig file |
| `--context TEXT` | current | Kubernetes context to use |

---

### `piqc version`

```bash
piqc version
```

---

## 📁 Output Formats

### Table Format (default)

Run `piqc scan --format table` — no flags required. See the [output example](#what-youll-see) above.

**GPU column markers** (tier fit, shown inline next to the GPU type):
| Symbol | Meaning |
|--------|---------|
| `✓` | Model is on an appropriate GPU tier for its size |
| `⚠` | Model is over-provisioned for an oversized GPU tier |
| `?` | Parameter count not parseable from model name — fit not checked |

**Age column:** shown as `5m` / `2h` / `19d` since the deployment's pods were created. Deployments running 3+ days are highlighted — long-running GPU allocations are easy to forget about and keep billing unnoticed.

### YAML Format

Generates individual Kubernetes-style YAML files for each deployment:

```yaml
apiVersion: piqc/v1
kind: InferenceDeployment
metadata:
  name: vllm-llama-7b
  namespace: inference
  collectionTimestamp: "2024-01-07T12:00:00Z"
  collectorVersion: "1.0.0"
model:
  name: meta-llama/Llama-2-7b-hf
  architecture: llama
  parameters: "7B"
  identificationConfidence: 0.95
engine:
  name: vllm
  version: "0.4.0"
  detectionConfidence: 0.95
inference:
  precision: float16
  tensorParallelSize: 4
  maxModelLen: 4096
  gpuMemoryUtilization: 0.90
resources:
  replicas: 2
  gpuCount: 4
  gpus:
    - type: A100-SXM4-80GB
      memoryTotal: "80GB"
      utilization: 87
      memoryUsed: 72000
runtimeState:
  vllm:
    healthStatus: healthy
    kvCacheUsagePercent: 45.2
    avgPromptThroughput: 1250.5
    avgGenerationThroughput: 85.3
```

### PIQC Facts Bundle

With `--output-piqc`, generates a standardized facts bundle for integration with the [Paralleliq optimization layer](https://paralleliq.ai):

```json
{
  "schemaVersion": "piqc-scan.v0.1",
  "generatedAt": "2026-06-20T12:00:00Z",
  "tool": {
    "name": "piqc",
    "version": "1.1.0"
  },
  "cluster": {
    "context": "my-context",
    "name": "my-cluster"
  },
  "objects": [
    {
      "workloadId": "ns/inference/deployment/vllm-llama-7b",
      "kind": "Deployment",
      "name": "vllm-llama-7b",
      "namespace": "inference",
      "facts": {
        "runtime.engineType": {"value": "vllm", "dataConfidence": "high"},
        "hardware.gpuType": {"value": "A100-SXM4-80GB", "dataConfidence": "high"},
        "hardware.gpuCount": {"value": 4, "dataConfidence": "high"},
        "obs.gpu.memUtilAvgPct": {"value": 87, "dataConfidence": "high"},
        "obs.vllm.kvCacheUsagePct": {"value": 45.2, "dataConfidence": "high", "units": "%"},
        "obs.vllm.requestsRunning": {"value": 3, "dataConfidence": "high"},
        "obs.vllm.requestsWaiting": {"value": 0, "dataConfidence": "high"},
        "k8s.ageHours": {"value": 18.5, "dataConfidence": "high", "units": "hours"}
      }
    },
    {
      "workloadId": "ns/gpu-pool/node/h100-node-07",
      "kind": "Node",
      "name": "h100-node-07",
      "namespace": "gpu-pool",
      "facts": {
        "hardware.gpuType": {"value": "nvidia-h100-80gb", "dataConfidence": "high"},
        "hardware.gpuCount": {"value": 4, "dataConfidence": "high"},
        "node.allocatedGpuCount": {"value": 2, "dataConfidence": "high"},
        "node.unallocatedGpuCount": {"value": 2, "dataConfidence": "high"}
      }
    }
  ]
}
```

The first object is a normal scanned workload. The second is a node-scoped object — emitted when a node has GPU capacity no pod has requested (the "Dark capacity" case under Waste Detection above); it carries no `runtime.*`/`model.*` facts since it isn't describing a running inference workload.

---

## 📥 Installation

### Prerequisites

- **Python**: 3.11 or higher
- **Kubernetes Access**: Valid kubeconfig with cluster access
- **Poetry**: For development installation

### Install from PyPI (recommended)

```bash
# Isolated install — won't conflict with other Python tools on your machine
pipx install piqc

piqc --version
```

`pip install piqc` also works, but installs into whatever environment is currently active. piqc requires pydantic v2; tools like `dstack` pin pydantic v1, so installing piqc with plain `pip` into an environment that already has `dstack` (or anything else pinned to pydantic v1) can break it. If you don't have `pipx`, install it first: `python3 -m pip install --user pipx`.

### Install from Source

```bash
git clone https://github.com/paralleliq/piqc.git
cd piqc
poetry install
poetry run piqc --version
```

### Install for Development

```bash
git clone https://github.com/paralleliq/piqc.git
cd piqc
poetry install --with dev
poetry run pytest tests/unit -v
```

---

## 🔐 Kubernetes RBAC Requirements

piqc is **read-only**. It never creates, modifies, or deletes any resource in your cluster. The only write permission is `pods/exec` (to run `nvidia-smi` inside pods for GPU metrics) — and that can be disabled with `--no-exec`.

```bash
kubectl apply -f https://raw.githubusercontent.com/paralleliq/piqc/main/deploy/rbac.yaml
```

| Resource | Verbs | Purpose |
|----------|-------|---------|
| `pods` | get, list | Discover inference workloads |
| `pods/exec` | create | Run nvidia-smi for GPU metrics |
| `pods/log` | get | Enhanced framework detection |
| `namespaces` | get, list | Scan multiple namespaces |
| `deployments` | get, list | Identify deployment metadata |
| `statefulsets` | get, list | Identify StatefulSet workloads |
| `services` | get, list | Endpoint detection, OTel Collector detection (`--coverage`) |
| `nodes` | get, list | GPU capacity / dark capacity analysis |
| `servicemonitors.monitoring.coreos.com` | get, list | Optional — only used by `--coverage` to check vLLM scrape coverage. Harmless to grant even without Prometheus Operator installed. |

---

## 🔧 Execution Modes

| Mode | Description |
|------|-------------|
| `auto` | Automatically detect if running in-cluster or remotely |
| `remote` | Force remote mode (uses kubeconfig) |
| `incluster` | Force in-cluster mode (uses ServiceAccount) |
| `dry-run` | Simulate scan without cluster access |

---

## 🐛 Troubleshooting

### Docker Auth Plugin Errors (GKE / EKS / AKS)

Use the in-cluster Job approach (Option 1 in Quick Start) — it runs inside the cluster and needs no auth plugins. Or export a static kubeconfig:

```bash
kubectl config view --raw --flatten > /tmp/piqc-kubeconfig.yaml
docker run --rm -v /tmp/piqc-kubeconfig.yaml:/root/.kube/config ghcr.io/paralleliq/piqc:latest scan
```

### RBAC Permission Errors

```bash
kubectl auth can-i list pods --all-namespaces
kubectl auth can-i create pods/exec -n <namespace>
kubectl apply -f https://raw.githubusercontent.com/paralleliq/piqc/main/deploy/rbac.yaml
```

### GPU Metrics Unavailable

```bash
piqc scan --no-exec
```

---

## 📚 Project Structure

```
piqc/
├── src/piqc/
│   ├── cli/                  # CLI commands (scan, test-connection, version)
│   ├── collectors/           # Data collectors (vLLM config, GPU metrics)
│   ├── core/                 # Core logic (orchestrator, discovery, k8s client)
│   ├── generators/           # Output generators (YAML, JSON, Table, PIQC)
│   ├── models/               # Pydantic data models (inference deployment, PIQC schema)
│   ├── parsers/              # Configuration parsers (vLLM)
│   └── utils/                # Utilities (logging, exceptions)
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── rbac/                     # Kubernetes RBAC manifests
├── docs/                     # Documentation
└── examples/                 # Example scan outputs and facts bundles
```

---

## What to do with the results

piqc tells you what's wrong. The [Paralleliq optimization layer](https://paralleliq.ai) closes the loop — it ingests the piqc facts bundle, maps waste to the model level, and routes remediations through human-approved workflows with a full audit trail.

→ [paralleliq.ai](https://paralleliq.ai) · [info@paralleliq.ai](mailto:info@paralleliq.ai)

---

## Contributing a Hardware Plugin

piqc's hardware fact collection is designed so hardware vendors and community contributors can add support for their own GPU or accelerator without modifying the core inference collector.

A hardware plugin is a collector that:
1. Reads telemetry from the target hardware (via `nvidia-smi`, `rocm-smi`, vendor BMC API, or equivalent)
2. Emits facts using the piqc fact schema (`hardware.gpuType`, `hardware.gpuCount`, `observed.gpuUtilization`, etc.)
3. Lives under `src/piqc/collectors/hardware/<vendor>/`

The vLLM inference collector is the reference implementation. If you represent a hardware vendor or want to contribute support for AMD, Intel Gaudi, or another accelerator, open an issue or email [info@paralleliq.ai](mailto:info@paralleliq.ai).

---

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
