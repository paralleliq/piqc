# PIQC — Complete Repository Structure

**Version:** v1.0.0
**Last Updated:** 2025-01-07

This document provides a complete map of the `paralleliq/piqc` repository, explaining the purpose of every directory and file.

---

## Top-Level Overview

```
piqc/
│
├── .github/                        # GitHub-specific configuration
├── docs/                           # Documentation (LaTeX guides, markdown)
├── examples/                       # Example ModelSpec output files
├── piqc-test-outputs/              # Sample scan outputs for reference/testing
├── rbac/                           # Kubernetes RBAC manifests
├── src/piqc/                       # All Python source code
├── tests/                          # Test suite (unit + integration)
│
├── .gitignore                      # Git ignore rules
├── CHANGELOG.md                    # Version history and release notes
├── CODE_OF_CONDUCT.md              # Community conduct standards
├── CONTRIBUTING.md                 # Contributor guide and development setup
├── GOVERNANCE.md                   # Project governance and decision-making
├── LICENSE                         # Apache License 2.0
├── README.md                       # Project overview and quick start
├── SECURITY.md                     # Security policy and vulnerability disclosure
│
├── ModelSpec_Final_Documentation.pdf  # ModelSpec reference documentation
├── piqc_Guide.pdf                  # PIQC user guide (PDF)
├── gcp_testing_guide.md.resolved   # GCP-specific testing notes
├── piqc-test-outputs.zip           # Archived test output samples
├── poetry.lock                     # Locked dependency versions
└── pyproject.toml                  # Project metadata, dependencies, tooling config
```

---

## Source Code: `src/piqc/`

The main Python package. All runtime logic lives here.

```
src/piqc/
│
├── __init__.py
│
├── cli/                            # CLI layer (entry points only — thin wrappers)
│   ├── __init__.py
│   ├── main.py                     # Click group entry point (`piqc`)
│   ├── scan.py                     # `piqc scan` command definition and options
│   ├── test_connection.py          # `piqc test-connection` command
│   └── version.py                  # `piqc version` command
│
├── collectors/                     # Data collectors — fetch facts from live cluster
│   ├── __init__.py
│   ├── gpu_collector.py            # GPU metrics via `nvidia-smi` pod exec
│   ├── runtime_collector.py        # vLLM API runtime metrics (latency, throughput, KV cache)
│   └── config_collector.py         # Static config extraction (env vars, CLI args, labels)
│
├── core/                           # Core business logic
│   ├── __init__.py
│   ├── orchestrator.py             # Scan orchestrator — coordinates discovery + collection
│   ├── discovery.py                # Inference workload discovery and confidence scoring
│   └── k8s_client.py               # Kubernetes API client wrapper
│
├── generators/                     # Output generators — transform models to output formats
│   ├── __init__.py
│   ├── yaml_generator.py           # ModelSpec YAML file generation
│   ├── json_generator.py           # ModelSpec JSON file generation
│   ├── table_generator.py          # Rich console table rendering
│   └── piqc_generator.py           # PIQC facts bundle (`piqc-facts.json`) generation
│
├── models/                         # Pydantic data models — typed representations
│   ├── __init__.py
│   ├── modelspec.py                # ModelSpec schema (model, engine, inference, resources, runtime)
│   └── piqc_facts.py               # PIQC facts schema (`piqc-scan.v0.1`)
│
├── parsers/                        # Input parsers — decode raw cluster data
│   ├── __init__.py
│   └── vllm_parser.py              # vLLM CLI argument and environment variable parser
│
└── utils/                          # Shared utilities
    ├── __init__.py
    ├── logging.py                  # Structured logging setup and helpers
    └── exceptions.py               # Custom exception types
```

---

## Tests: `tests/`

```
tests/
│
├── unit/                           # Unit tests — no Kubernetes cluster required
│   ├── __init__.py
│   ├── test_discovery.py           # Discovery engine and confidence scoring
│   ├── test_collectors.py          # GPU and runtime collector logic
│   ├── test_generators.py          # YAML, JSON, table, PIQC output generators
│   ├── test_models.py              # Pydantic model validation
│   └── test_parsers.py             # vLLM argument parser
│
└── integration/                    # Integration tests — requires cluster or mock containers
    ├── __init__.py
    ├── test_scan_flow.py            # End-to-end scan workflow
    └── test_connection.py           # Cluster connectivity validation
```

---

## Kubernetes RBAC: `rbac/`

Pre-built Kubernetes manifests for granting PIQC the minimum required permissions.

```
rbac/
├── serviceaccount.yaml             # ServiceAccount: `piqc` in target namespace
├── clusterrole.yaml                # ClusterRole with required RBAC verbs
└── clusterrolebinding.yaml         # Binds ClusterRole to the piqc ServiceAccount
```

**Required Permissions:**

| Resource | Verbs | Purpose |
|---|---|---|
| `pods` | `get`, `list` | Discover inference workloads |
| `pods/exec` | `create` | Run `nvidia-smi` for GPU metrics |
| `pods/log` | `get` | Enhanced framework detection via logs |
| `namespaces` | `get`, `list` | Multi-namespace scanning |
| `deployments` | `get`, `list` | Deployment metadata extraction |
| `statefulsets` | `get`, `list` | StatefulSet workload identification |
| `services` | `get`, `list` | vLLM endpoint detection |

---

## Documentation: `docs/`

```
docs/
├── architecture.md                 # High-level architecture and scan flow diagrams
├── rbac-guide.md                   # Detailed RBAC setup for different cloud providers
├── output-formats.md               # Complete reference for all output schema fields
└── (LaTeX source files)            # Source for the piqc_Guide.pdf
```

---

## Examples: `examples/`

Reference ModelSpec output files from real or simulated vLLM deployments.

```
examples/
├── llama2-7b-single-gpu.yaml       # Single-GPU Llama 2 7B deployment
├── mistral-7b-production.yaml      # Production Mistral 7B deployment
├── qwen2-72b-multi-gpu.yaml        # Multi-GPU Qwen2 72B deployment
└── piqc-facts-example.json         # Example PIQC facts bundle output
```

---

## GitHub Configuration: `.github/`

```
.github/
├── ISSUE_TEMPLATE/
│   ├── bug_report.md               # Bug report issue template
│   └── feature_request.md          # Feature request issue template
├── CODEOWNERS                      # Code ownership assignments for PR reviews
└── workflows/                      # GitHub Actions CI/CD pipelines
    ├── test.yml                    # Run pytest on PRs
    ├── lint.yml                    # Black + Ruff + MyPy checks
    └── release.yml                 # Automated release pipeline
```

---

## Root Configuration Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Poetry project config — dependencies, dev tools, package metadata, version |
| `poetry.lock` | Locked dependency tree — ensures reproducible installs |
| `.gitignore` | Excludes output files, venvs, `__pycache__`, `.env`, IDE configs |

---

## Key Design Principles

### Layered Architecture

```
CLI (thin)
  ↓
Core Orchestrator
  ↓
Discovery Engine + Collectors (parallel)
  ↓
Pydantic Models
  ↓
Generators (output)
```

Each layer has a single responsibility. The CLI never contains business logic; the generators never call the Kubernetes API directly.

### Confidence Scoring

PIQC uses weighted multi-signal scoring for vLLM detection:

| Signal | Weight |
|---|---|
| Container image pattern | 40% |
| Environment variables | 30% |
| CLI arguments | 20% |
| Kubernetes labels | 10% |

A deployment is reported as vLLM when the weighted score exceeds the detection threshold.

### Data Completeness Tracking

Every ModelSpec output includes a `dataCompleteness` section tracking whether static config, GPU metrics, and runtime metrics were successfully collected — enabling downstream consumers to understand data quality.

---

## Version

PIQC version is maintained in `pyproject.toml`:

```toml
[tool.poetry]
name = "piqc"
version = "1.0.0"
```

The version is surfaced via `piqc version` and the README badge:

```
[![PIQC Version](https://img.shields.io/badge/PIQC-v1.0.0-blue?...)]
```

When releasing a new version:
1. Update `version` in `pyproject.toml`
2. Update the badge in `README.md`
3. Add entries to `CHANGELOG.md`
4. Tag the release: `git tag -a v<version> -m "Release v<version>"`

---

*Part of the [PIQC](https://github.com/paralleliq/piqc) project — Maintained by [ParalleliQ](https://paralleliq.ai)*
