# PIQC - Kubernetes AI/ML Model Introspector

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PIQC (Production Inference Quality Control) is a Kubernetes-native tool for discovering and documenting AI/ML inference deployments. It automatically detects vLLM workloads, collects GPU metrics, and generates standardized ModelSpec documentation.

## Features

- **Auto-Discovery**: Automatically finds vLLM inference deployments across namespaces
- **GPU Metrics**: Collects real-time GPU utilization, memory, and temperature via nvidia-smi
- **Runtime Metrics**: Optional collection of vLLM API metrics (latency, throughput, cache)
- **Multiple Formats**: Output as YAML, JSON, or console table
- **PIQC Schema**: Generate standardized facts bundles for quality assessment
- **Parallel Processing**: Multi-threaded scanning for large clusters

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Access to a Kubernetes cluster with kubeconfig configured
- Poetry (for development installation)

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd ModelSpec

# Install with Poetry
poetry install

# Verify installation
poetry run piqc --version
```

---

## Quick Start

```bash
# Test connection to your cluster
piqc test-connection

# Scan entire cluster and output to console
piqc scan --format table

# Scan and generate YAML files
piqc scan --format yaml -o ./output

# Scan specific namespace
piqc scan -n production --format json
```

---

## Command Reference

### Global Options

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--help` | Show help message and exit |

---

### `piqc scan`

Scan Kubernetes cluster for vLLM model deployments and generate ModelSpec documentation.

#### Basic Usage

```bash
piqc scan [OPTIONS]
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--kubeconfig PATH` | `~/.kube/config` | Path to kubeconfig file |
| `--context TEXT` | current | Kubernetes context to use |
| `-n, --namespace TEXT` | all | Specific namespace to scan |
| `--format [yaml\|json\|table]` | `yaml` | Output format |
| `-o, --output PATH` | `./output` | Output directory for generated files |
| `--timeout INT` | `30` | Operation timeout in seconds |
| `--no-exec` | false | Disable pod exec (skip GPU metrics) |
| `--no-logs` | false | Disable log reading |
| `--workers INT` | `5` | Number of parallel workers |
| `-v, --verbose` | false | Enable verbose output |
| `--debug` | false | Enable debug mode with detailed trace |
| `--combined` | false | Generate single combined output file |
| `--collect-runtime` | false | Collect runtime metrics via vLLM API |
| `--aggregate/--no-aggregate` | aggregate | Aggregate metrics across pod replicas |
| `--mode [auto\|remote\|incluster\|dry-run]` | `auto` | Execution mode |
| `--output-piqc` | false | Generate piqc-facts.json output |

#### Examples

```bash
# Basic scan - discover all vLLM deployments
piqc scan

# Scan specific namespace with JSON output
piqc scan -n production --format json

# Quick scan without GPU metrics (faster)
piqc scan --no-exec

# Verbose output for debugging
piqc scan -v --debug

# Collect runtime metrics from vLLM API
piqc scan --collect-runtime

# Generate PIQC facts bundle for quality assessment
piqc scan --output-piqc -o ./facts

# Combined output file instead of per-deployment files
piqc scan --combined -o ./output

# Table output to console (no files generated)
piqc scan --format table

# Custom kubeconfig and context
piqc scan --kubeconfig /path/to/config --context my-cluster

# Disable metric aggregation across replicas
piqc scan --no-aggregate
```

---

### `piqc test-connection`

Test connection to Kubernetes cluster and verify required permissions.

#### Usage

```bash
piqc test-connection [OPTIONS]
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--kubeconfig PATH` | `~/.kube/config` | Path to kubeconfig file |
| `--context TEXT` | current | Kubernetes context to use |

#### Example

```bash
# Test default connection
piqc test-connection

# Test specific context
piqc test-connection --context production-cluster
```

#### Expected Output

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

Display version information.

```bash
piqc version
```

---

## Output Formats

### YAML Format (Default)

Generates individual YAML files for each deployment:

```yaml
apiVersion: modelspec/v1
kind: ModelSpec
metadata:
  name: vllm-llama-7b
  namespace: inference
  collector_version: "1.0.0"
  collection_timestamp: "2024-01-07T12:00:00Z"
model:
  name: meta-llama/Llama-2-7b-hf
  architecture: llama
  parameters: "7B"
engine:
  name: vllm
  detection_confidence: 0.95
inference:
  precision: float16
  tensor_parallel_size: 4
  max_model_len: 4096
resources:
  replicas: 2
  gpus:
    - type: A100-SXM4-80GB
      memory_total: "80GB"
      utilization: 87
```

### JSON Format

Same structure as YAML but in JSON format.

### Table Format

Console-friendly table output:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Model Name                ┃ Engine ┃ GPU Type        ┃ Replicas ┃ GPU Util ┃ Namespace   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ meta-llama/Llama-2-7b-hf  │ vllm   │ 4xA100-SXM4-80GB│        2 │      87% │ inference   │
│ mistralai/Mistral-7B      │ vllm   │ 2xA100-40GB     │        1 │      72% │ production  │
└───────────────────────────┴────────┴─────────────────┴──────────┴──────────┴─────────────┘
```

### PIQC Facts Bundle

With `--output-piqc`, generates a standardized facts bundle:

```json
{
  "schemaVersion": "piqc-scan.v0.1",
  "generatedAt": "2024-01-07T12:00:00Z",
  "tool": {
    "name": "piqc",
    "version": "1.0.0"
  },
  "objects": [
    {
      "workloadId": "ns/inference/deployment/vllm-llama-7b",
      "facts": {
        "runtime.engineType": {"value": "vllm", "dataConfidence": "high"},
        "hardware.gpuType": {"value": "A100-SXM4-80GB", "dataConfidence": "high"},
        "hardware.gpuCount": {"value": 4, "dataConfidence": "high"}
      }
    }
  ]
}
```

---

## Kubernetes RBAC Requirements

The tool requires specific Kubernetes permissions. Apply the provided RBAC manifests:

```bash
kubectl apply -f rbac/
```

### Required Permissions

| Resource | Verbs | Purpose |
|----------|-------|---------|
| pods | get, list | Discover inference workloads |
| pods/exec | create | Run nvidia-smi for GPU metrics |
| pods/log | get | Enhanced framework detection |
| namespaces | get, list | Scan multiple namespaces |
| deployments | get, list | Identify deployment metadata |
| statefulsets | get, list | Identify StatefulSet workloads |
| services | get, list | Endpoint detection |

---

## Execution Modes

| Mode | Description |
|------|-------------|
| `auto` | Automatically detect if running in-cluster or remotely |
| `remote` | Force remote mode (uses kubeconfig) |
| `incluster` | Force in-cluster mode (uses service account) |
| `dry-run` | Simulate scan without cluster access |

---

## Troubleshooting

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
kubectl apply -f rbac/
```

### GPU Metrics Unavailable

If nvidia-smi is not available in containers, use `--no-exec`:

```bash
piqc scan --no-exec
```

---

## Development

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
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type checking
poetry run mypy src/
```

---

## License

MIT License - see LICENSE file for details.
