# Integration Testing with Kind

This directory contains the infrastructure for running realistic integration tests against a local Kubernetes cluster using [kind](https://kind.sigs.k8s.io/) (Kubernetes in Docker).

## Prerequisites

```bash
# Install Docker
# See: https://docs.docker.com/engine/install/

# Install kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/kubectl
```

## Quick Start

```bash
# Create test cluster with mock deployments
./scripts/setup-test-cluster.sh

# Run integration tests
poetry run pytest tests/integration/ -v

# Cleanup
./scripts/teardown-test-cluster.sh
```

## Test Scenarios

| Deployment | Framework | GPUs | Purpose |
|------------|-----------|------|---------|
| vllm-llama-70b | vLLM | 8 | Multi-GPU tensor parallel |
| vllm-mistral-7b | vLLM | 1 | Single GPU deployment |
| triton-server | Triton | 4 | Multi-model serving |
| tgi-falcon | TGI | 2 | HuggingFace TGI |
| generic-inference | Unknown | 1 | Fallback detection |
| nginx-web | None | 0 | Non-ML control group |

## Directory Structure

```
tests/integration/
├── kind/
│   ├── kind-config.yaml      # Kind cluster configuration
│   └── manifests/            # Kubernetes deployment manifests
├── mock-containers/          # Mock inference container images
├── scripts/                  # Setup/teardown automation
└── test_*.py                 # Integration test files
```
