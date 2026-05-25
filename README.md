<p align="center">
  <img src="https://img.shields.io/badge/PIQC-v1.0.0-blue?style=for-the-badge&logo=kubernetes&logoColor=white" alt="PIQC Version"/>
  <img src="https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-orange?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/vLLM-Supported-purple?style=for-the-badge" alt="vLLM"/>
  <img src="https://img.shields.io/github/stars/paralleliq/piqc?style=for-the-badge&logo=github&color=yellow" alt="GitHub Stars"/>
</p>

<h1 align="center">piqc — GPU Waste Scanner for Kubernetes</h1>

<p align="center">
  <strong>Most AI clusters waste 20–40% of GPU spend. piqc finds it in one command.</strong>
  <br/><br/>
  Read-only · No agents · No sidecars · Nothing installed permanently · Runs as a Job, prints results, exits.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-commands">Commands</a> •
  <a href="#-output-formats">Output Formats</a> •
  <a href="#-installation">Installation</a>
</p>

---

## What is piqc?

piqc is an open-source GPU waste scanner for Kubernetes clusters. It detects idle GPU allocations, tier misplacement, and dark capacity — and surfaces a dollar estimate of the waste — in under a minute. No agents, no sidecars, no write permissions, no permanent installation required.

It is the fastest way to answer: **how much GPU spend is my Kubernetes cluster wasting right now?**

piqc surfaces three types of waste that standard Kubernetes monitoring (`kubectl top`, `kube-state-metrics`, Prometheus node exporters) cannot detect on their own:
- **Idle allocation** — pods holding GPU resources with near-zero compute utilization
- **Tier misplacement** — models running on GPU tiers with far more memory or compute than they need
- **Dark capacity** — GPU nodes with no pods scheduled at all

It works with any Kubernetes cluster running GPU inference workloads — GKE, EKS, AKS, on-prem, or bare metal. Supports vLLM, Triton, TGI, and any deployment using `nvidia.com/gpu` resource requests.

---

## What you'll see

Run `piqc scan` against your cluster and get an instant cost report:

```
                                                    Discovered Inference Deployments
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Deployment                  ┃ Engine  ┃ GPU              ┃ Replicas ┃ GPU Util ┃  MFU ┃ $/1K tokens ┃   $/hr ┃   Idle $/day ┃   Tier Fit   ┃ Namespace       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ meta-llama/Llama-3-70B-Inst │ vllm    │ 8xH100-SXM4-80GB │        2 │       4% │ 3.1% │     $0.0842 │ $68.00 │    $1,566.72 │ ⚠ >A100-80GB │ production      │
│ mistral-7b-instruct         │ vllm    │ 1xA100-SXM4-40GB │        1 │      11% │ 8.4% │     $0.0073 │  $2.50 │       $53.40 │    ⚠ >T4     │ production      │
│ codellama-34b-staging       │ vllm    │ 4xH100-SXM4-80GB │        1 │       0% │  N/A │         N/A │ $17.00 │      $408.00 │ ⚠ >A100-40GB │ staging         │
│ embedding-bge-large         │ vllm    │ 1xT4             │        3 │      82% │  N/A │     $0.0002 │  $1.35 │        $5.83 │      ✓       │ shared-services │
│ unknown-runtime-7f3a2       │ unknown │ 2xA100-SXM4-80GB │        1 │      N/A │  N/A │         N/A │  $7.00 │ util unknown │      ?       │ ml-platform     │
└─────────────────────────────┴─────────┴──────────────────┴──────────┴──────────┴──────┴─────────────┴────────┴──────────────┴──────────────┴─────────────────┘

╭──────────────────────────────────── Cost Summary ──────────────────────────────────────╮
│   Total GPU spend rate      : $95.85/hr                                                │
│                                                                                        │
│   Leased & idle (util <60%) : $2,033.95/day  (pods running, GPUs underused)            │
│   Unallocated nodes         : $1,152.00/day  (12 GPU(s) with no pods scheduled)        │
│   Tier misplacement         :   $721.20/day  (3 model(s) on oversized GPU tier)        │
│                                                                                        │
│   Total estimated leak      : $3,907.15/day  ($1,426,110/yr)                           │
│                                                                                        │
│   Avg MFU (active deployments) : 15.7%  (healthy range: 30–60%)                        │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

piqc surfaces three types of waste:
- **Idle GPUs** — pods running, GPUs sitting near-empty
- **Tier misplacement** — a 7B model on an H100 that only needs a T4
- **Unallocated nodes** — GPU nodes with no pods scheduled at all

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

---

### Option 2: Run with Docker from your laptop

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

### Option 3: Install from source

```bash
git clone https://github.com/paralleliq/piqc.git
cd piqc
poetry install
poetry run piqc scan --format table
```

---

## ✨ Features

### 🔍 Intelligent Discovery
- **Auto-Detection**: Automatically discovers vLLM inference deployments across all namespaces
- **Weighted Confidence Scoring**: Uses multiple signals (images, env vars, CLI args, labels) with weighted scoring
- **Framework Detection**: Identifies vLLM with high accuracy using pattern matching and heuristics

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

### 🔮 Coming Soon

<table>
<tr>
<td width="50%" valign="top">

**🔴 AMD GPU Support**

Support for AMD Instinct and Radeon GPUs via `rocm-smi`:
- AMD Instinct MI250X/MI300X detection
- GPU utilization, memory & temperature metrics
- ROCm ecosystem integration
- Seamless multi-vendor GPU environments

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

**Tier Fit column:**
| Symbol | Meaning |
|--------|---------|
| `✓` | Model is on an appropriate GPU tier for its size |
| `⚠ >T4` | Model is over-provisioned — minimum sufficient tier shown |
| `?` | Parameter count not parseable from model name |

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

With `--output-piqc`, generates a standardized facts bundle for integration with the [Paralleliq control plane](https://paralleliq.ai):

```json
{
  "schemaVersion": "piqc-scan.v0.1",
  "generatedAt": "2024-01-07T12:00:00Z",
  "tool": {
    "name": "piqc",
    "version": "1.0.0"
  },
  "cluster": {
    "context": "my-context",
    "name": "my-cluster"
  },
  "objects": [
    {
      "workloadId": "ns/inference/deployment/vllm-llama-7b",
      "facts": {
        "runtime.engineType": {"value": "vllm", "dataConfidence": "high"},
        "hardware.gpuType": {"value": "A100-SXM4-80GB", "dataConfidence": "high"},
        "hardware.gpuCount": {"value": 4, "dataConfidence": "high"},
        "observed.gpuUtilization": {"value": 87, "unit": "%", "dataConfidence": "high"},
        "observed.kvCacheUsage": {"value": 45.2, "unit": "%", "dataConfidence": "high"}
      }
    }
  ]
}
```

---

## 📥 Installation

### Prerequisites

- **Python**: 3.11 or higher
- **Kubernetes Access**: Valid kubeconfig with cluster access
- **Poetry**: For development installation

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
| `services` | get, list | Endpoint detection |

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

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
