<p align="center">
  <img src="https://img.shields.io/badge/PIQC-v1.0.0-blue?style=for-the-badge&logo=kubernetes&logoColor=white" alt="PIQC Version"/>
  <img src="https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-orange?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/vLLM-Supported-purple?style=for-the-badge" alt="vLLM"/>
</p>

<h1 align="center">🔍 PIQC — GPU Revenue Leak & Inference Efficiency</h1>

<p align="center">
  <strong>Kubernetes AI/ML Introspector for vLLM Deployments</strong>
  <br/>
  <em>One command. See which GPUs are leaking money and why.</em>
  <br/><br/>
  <strong>🔒 Read-only</strong> — no agents, no sidecars, nothing installed permanently. Runs as a Job, prints results, exits.
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-commands">Commands</a> •
  <a href="#-output-formats">Output Formats</a> •
  <a href="#-installation">Installation</a>
</p>

---

## 🎯 Overview

**PIQC** (Production Inference Quality Control) is a **read-only** Kubernetes-native tool that discovers AI/ML inference deployments, measures their efficiency, and surfaces the dollar cost of idle and unallocated GPUs — in a single command.

> **Nothing is installed permanently.** PIQC runs as a Kubernetes Job using a scoped read-only service account. It collects data, prints the report, and exits. No agents, no sidecars, no cluster modifications.

```
piqc scan
```

No flags needed. PIQC connects to your current kubectl context, scans all namespaces, and immediately prints a cost report showing which models are running, at what efficiency (MFU), what they cost per 1K tokens, and how much GPU spend is being wasted today.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   🔍 PIQC Scan Flow                                                          │
│                                                                              │
│   ┌─────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐   │
│   │ K8s     │────▶│ Discovery &  │────▶│ Collect     │────▶│ Generate    │   │
│   │ Cluster │     │ Detection    │     │ Metrics     │     │ ModelSpec   │   │
│   └─────────┘     └──────────────┘     └─────────────┘     └─────────────┘   │
│                                                                              │
│   • Scans all namespaces          • GPU metrics via nvidia-smi              │
│   • Detects vLLM workloads        • Runtime metrics via vLLM API            │
│   • Weighted confidence scoring   • KV cache, latency, throughput           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
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

### 💰 Revenue Leak Detection
- **Idle GPU waste** — Deployments with utilization below 60% threshold, with dollar waste per day and annualized
- **Unallocated nodes** — GPU nodes with no pods scheduled (dark capacity still being paid for)
- **Tier misplacement** — Models running on a GPU tier beyond what their parameter count requires (e.g. a 7B model on an H100), with estimated cost delta per day
- **Cost Summary panel** — Total spend rate, all three leak categories, and total estimated leak per day / per year
- **MFU (Model FLOPS Utilization)** — Observed compute vs. theoretical GPU peak per deployment
- **Cost per 1K tokens** — Translate GPU spend into a business metric comparable to API pricing

### 📄 Multiple Output Formats
| Format | Description |
|--------|-------------|
| **Table** | Cost report with MFU, $/1K tokens, idle waste (default) |
| **YAML** | Kubernetes-style ModelSpec files |
| **JSON** | Machine-readable JSON output |
| **PIQC Facts** | Standardized facts bundle for quality assessment |

### 🚀 Production-Ready
- **Parallel Processing**: Multi-threaded scanning with configurable workers
- **RBAC Support**: Pre-configured ClusterRole and ServiceAccount manifests
- **Flexible Modes**: Auto-detect, remote (kubeconfig), or in-cluster execution
- **Timeout Controls**: Configurable operation timeouts
- **Docker Image**: Pre-built multi-platform image (`linux/amd64` + `linux/arm64`) available on GitHub Container Registry — no install required

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

## 🚀 Quick Start

### Option 1: Run as a Kubernetes Job (recommended)

The simplest way — runs inside your cluster with no Docker auth or kubeconfig wrangling:

**Step 1 — Apply RBAC permissions (one-time setup):**
```bash
kubectl apply -f https://raw.githubusercontent.com/paralleliq/piqc/main/deploy/rbac.yaml
```

**Step 2 — Run the scan:**
```bash
kubectl apply -f https://raw.githubusercontent.com/paralleliq/piqc/main/deploy/scan-job.yaml
```

**Step 3 — View the output:**
```bash
kubectl logs -f job/piqc-scan -n kube-system
```

**Clean up when done:**
```bash
kubectl delete job piqc-scan -n kube-system
```

> The job auto-deletes itself after 10 minutes (`ttlSecondsAfterFinished: 600`).

---

### Option 2: Run with Docker from your laptop

For laptops and CI pipelines. Requires exporting a static kubeconfig first (avoids cloud auth plugin issues):

```bash
# Export a static kubeconfig with embedded credentials
kubectl config view --raw --flatten > /tmp/piqc-kubeconfig.yaml

# Run the scan
docker run --rm \
  -v /tmp/piqc-kubeconfig.yaml:/root/.kube/config \
  ghcr.io/paralleliq/piqc:latest \
  scan --format table
```

The image supports both `linux/amd64` and `linux/arm64`.

---

### Option 3: Install from source

```bash
git clone https://github.com/paralleliq/piqc.git
cd piqc
poetry install
poetry run piqc scan --format table
```

---

### Test Your Connection

```bash
# Verify cluster connectivity and permissions
piqc test-connection
```

### Run Your First Scan

```bash
# Scan entire cluster with console table output
piqc scan --format table

# Scan and generate YAML ModelSpec files
piqc scan --format yaml -o ./output

# Scan with runtime metrics from vLLM API
piqc scan --collect-runtime --format json
```

### Expected Output

```
ModelSpec Introspector v1.0.0
========================================

[INFO] Connecting to cluster...
       Context: my-k8s-context
       Cluster: my-cluster

[INFO] Scanning namespaces...
       Discovered: 12 namespace(s)

[INFO] Detecting inference workloads...
       Pods analyzed: 47
       Inference deployments found: 3

Framework Distribution:
┃ Framework ┃ Count ┃
├───────────┼───────┤
│ vllm      │     3 │

[INFO] Scan completed in 8.2s
```

---

## 📋 Commands

### `piqc scan`

**Scan Kubernetes cluster for vLLM model deployments and generate ModelSpec documentation.**

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
| `--contribute-benchmarks` | `false` | Contribute anonymized GPU/model performance data to the ParallelIQ benchmark dataset |

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
# Basic scan - discover all vLLM deployments
piqc scan

# Scan specific namespace with JSON output
piqc scan -n production --format json

# Quick scan without GPU metrics (faster)
piqc scan --no-exec

# Collect runtime metrics from vLLM API
piqc scan --collect-runtime

# Generate PIQC facts bundle for quality assessment
piqc scan --output-piqc -o ./facts

# Combined output file instead of per-deployment files
piqc scan --combined -o ./output

# Table output to console (human-readable)
piqc scan --format table

# Custom kubeconfig and context
piqc scan --kubeconfig /path/to/config --context my-cluster

# Disable metric aggregation across replicas
piqc scan --no-aggregate

# Full verbose debug mode
piqc scan -v --debug

# Contribute anonymized GPU/model benchmarks to ParallelIQ dataset
piqc scan --contribute-benchmarks

# Preview what benchmark data would be sent (no identifying info)
piqc scan --contribute-benchmarks --verbose
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

#### Example Output

```
ModelSpec Introspector v1.0.0
========================================

[INFO] Testing cluster connection...

Connection successful

Context: my-context
Cluster: my-cluster
[INFO] Testing namespace access...
       Accessible namespaces: 15

All checks passed
```

---

### `piqc version`

**Display version information.**

```bash
piqc version
# Output: ModelSpec Introspector v1.0.0
```

---

## 📁 Output Formats

### YAML Format (Default)

Generates individual Kubernetes-style YAML files for each deployment:

```yaml
apiVersion: modelspec/v1
kind: ModelSpec
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
dataCompleteness:
  staticConfig: true
  gpuMetrics: true
  runtimeMetrics: true
```

### JSON Format

Same structure as YAML but in JSON format, ideal for programmatic processing.

### Table Format

Default output of `piqc scan` — no flags required:

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
│   Total GPU spend rate     : $95.85/hr                                                 │
│                                                                                        │
│   Leased & idle (util <60%) : $2,033.95/day  (pods running, GPUs underused)            │
│   Unallocated nodes        : $1,152.00/day  (12 GPU(s) with no pods scheduled)         │
│   Tier misplacement        :   $721.20/day  (3 model(s) on oversized GPU tier)         │
│                                                                                        │
│   Total estimated leak     : $3,907.15/day  ($1,426,110/yr)                            │
│                                                                                        │
│   Avg MFU (active deployments) : 15.7%  (healthy range: 30–60%)                        │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

**Tier Fit column:**
| Symbol | Meaning |
|--------|---------|
| `✓` | Model is on an appropriate GPU tier for its size |
| `⚠ >T4` | Model is over-provisioned — minimum sufficient tier shown |
| `?` | Parameter count not parseable from model name |

### PIQC Facts Bundle

With `--output-piqc`, generates a standardized facts bundle for quality assessment systems:

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
        "runtime.engineVersion": {"value": "0.4.0", "dataConfidence": "medium"},
        "hardware.gpuType": {"value": "A100-SXM4-80GB", "dataConfidence": "high"},
        "hardware.gpuCount": {"value": 4, "dataConfidence": "high"},
        "hardware.gpuMemoryTotal": {"value": 80, "unit": "GB", "dataConfidence": "high"},
        "observed.gpuUtilization": {"value": 87, "unit": "%", "dataConfidence": "high"},
        "vllm.tensorParallelSize": {"value": 4, "dataConfidence": "high"},
        "vllm.maxModelLen": {"value": 4096, "dataConfidence": "high"},
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
# Clone the repository
git clone https://github.com/paralleliq/piqc.git
cd piqc

# Install with Poetry
poetry install

# Verify installation
poetry run piqc --version
```

### Install for Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/paralleliq/piqc.git
cd piqc
poetry install --with dev

# Run tests
poetry run pytest tests/unit -v

# Run with coverage
poetry run pytest tests/unit --cov=src/piqc
```

---

## 🔐 Kubernetes RBAC Requirements

PIQC is **read-only**. It never creates, modifies, or deletes any resource in your cluster. The only write permission is `pods/exec` (to run `nvidia-smi` inside pods for GPU metrics) and `pods/exec` can be disabled with `--no-exec`.

Apply the provided RBAC manifests:

```bash
kubectl apply -f https://raw.githubusercontent.com/paralleliq/piqc/main/deploy/rbac.yaml
```

### Required Permissions

| Resource | Verbs | Purpose |
|----------|-------|---------|
| `pods` | get, list | Discover inference workloads |
| `pods/exec` | create | Run nvidia-smi for GPU metrics |
| `pods/log` | get | Enhanced framework detection |
| `namespaces` | get, list | Scan multiple namespaces |
| `deployments` | get, list | Identify deployment metadata |
| `statefulsets` | get, list | Identify StatefulSet workloads |
| `services` | get, list | Endpoint detection |

### RBAC Files

```
rbac/
├── serviceaccount.yaml    # ServiceAccount for PIQC
├── clusterrole.yaml       # ClusterRole with required permissions
└── clusterrolebinding.yaml # Binds role to service account
```

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

If you see `gke-gcloud-auth-plugin not found` or similar errors when using Docker, use the
in-cluster Job approach (Option 1 above) — it runs inside the cluster and needs no auth plugins.

Alternatively, export a static kubeconfig:
```bash
kubectl config view --raw --flatten > /tmp/piqc-kubeconfig.yaml
docker run --rm -v /tmp/piqc-kubeconfig.yaml:/root/.kube/config ghcr.io/paralleliq/piqc:latest scan
```

### Connection Issues

```bash
# Verify kubeconfig is valid
kubectl cluster-info

# Test with specific context
piqc test-connection --context my-context

# Enable debug mode for detailed errors
piqc scan --debug
```

### RBAC Permission Errors

```bash
# Check current permissions
kubectl auth can-i list pods --all-namespaces
kubectl auth can-i create pods/exec -n <namespace>

# Apply RBAC manifests
kubectl apply -f https://raw.githubusercontent.com/paralleliq/piqc/main/deploy/rbac.yaml
```

### GPU Metrics Unavailable

If `nvidia-smi` is not available in containers, use `--no-exec`:

```bash
piqc scan --no-exec
```

### Runtime Metrics Not Collected

Ensure the vLLM service is accessible. Use `--collect-runtime` and check:

```bash
# Verify vLLM health endpoint
kubectl port-forward svc/<vllm-service> 8000:8000
curl http://localhost:8000/health
```

---

## 🧪 Development

### Running Tests

```bash
# Run all unit tests
poetry run pytest tests/unit -v

# Run with coverage
poetry run pytest tests/unit --cov=src/piqc

# Run integration tests (requires cluster)
poetry run pytest tests/integration -v
```

### Code Quality

```bash
# Format code with Black
poetry run black src/ tests/

# Lint code with Ruff
poetry run ruff check src/ tests/

# Type checking with MyPy
poetry run mypy src/
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
│   ├── models/               # Pydantic data models (ModelSpec, PIQC schema)
│   ├── parsers/              # Configuration parsers (vLLM)
│   └── utils/                # Utilities (logging, exceptions)
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests (with mock containers)
├── rbac/                     # Kubernetes RBAC manifests
├── docs/                     # Documentation (LaTeX guides)
└── examples/                 # Example ModelSpec files
```

---

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with ❤️ by <a href="https://paralleliq.ai">ParallelIQ</a></strong>
  <br/>
  <sub>🚀 Model-aware GPU Control Plane</sub>
</p>


