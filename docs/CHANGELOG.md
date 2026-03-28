# Changelog

All notable changes to **PIQC** will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

---

## [Unreleased]

### Added
- `CODE_OF_CONDUCT.md` — Community conduct standards
- `SECURITY.md` — Security policy and vulnerability disclosure process
- `GOVERNANCE.md` — Project governance model and decision-making process
- `CHANGELOG.md` — This file

### Changed
- `CONTRIBUTING.md` — Expanded with development setup, coding standards, and test guidance

---

## [1.1.0] — 2026-03-28

### Revenue Leak & Efficiency Reporting

#### New Table Columns
- **MFU (Model FLOPS Utilization)** — Computes observed FLOPS against theoretical GPU peak, displayed per deployment with color coding (green ≥30%, yellow ≥10%, red <10%)
- **$/1K tokens** — Cost per 1,000 generated tokens, derived from generation throughput and GPU spend rate
- **$/hr** — Total GPU spend rate for the deployment (GPU count × replicas × per-GPU rate)
- **Idle $/day** — Estimated daily waste for deployments with GPU utilization below 60% threshold

#### New Cost Summary Panel
Printed below the deployment table on every `piqc scan`:
- **Total GPU spend rate** across all discovered deployments
- **Leased & idle** — Dollar waste from pods running with GPU util below threshold
- **Unallocated nodes** — Dollar waste from nodes with GPU capacity but no pods scheduled
- **Total estimated leak** per day and annualized ($/yr)
- **Avg MFU** across active deployments with a healthy range indicator (30–60%)

#### Node-Level GPU Capacity Analysis
- Detects GPU nodes with unscheduled capacity (dark/unallocated GPUs)
- Resolves GPU type from node labels (`nvidia.com/gpu.product`, `cloud.google.com/gke-accelerator`)
- Reports unallocated GPU count and estimated daily cost per node

#### New CLI Flags
- `--gpu-cost DOLLARS` — Override GPU cost in $/GPU/hr (default: auto-detect by GPU type from built-in lookup table)
- `--node-cost DOLLARS` — Override cost for unallocated node GPUs separately from active deployment GPUs

#### GPU Cost & Peak FLOPS Lookup Tables
Built-in pricing estimates (USD/hr per GPU) and theoretical peak TFLOPS for:
H100 SXM5/SXM4/NVL/PCIE, A100 SXM4/PCIE (80GB/40GB), L40S, A10G, A10, L4, V100, T4

#### Unknown Runtime Detection
- Pods identified as ML workloads but with unrecognized runtimes (TGI, Triton, etc.) are surfaced with a yellow advisory prompt rather than silently dropped

### Changed
- `--format` default changed from `yaml` to `table` — `piqc scan` with no flags now shows the cost report immediately
- `--collect-runtime` now defaults to **enabled** — runtime metrics (TPS, latency, cache) are collected on every scan; use `--no-collect-runtime` to disable
- `K8sClient` gained `list_nodes()` and `list_all_pods()` methods for cluster-wide GPU capacity analysis

---

## [1.0.0] — 2026-01-30

### 🎉 Initial Release

The first stable release of **PIQC** — Production Inference Quality Control. A Kubernetes-native scanner that discovers LLMs running on vLLM and extracts their deployment and runtime facts.

### Added

#### Core Scanner
- **Cluster Discovery Engine** — Automatically discovers vLLM inference deployments across all Kubernetes namespaces
- **Weighted Confidence Scoring** — Multi-signal detection using container images, environment variables, CLI arguments, and labels
- **Framework Detection** — Identifies vLLM deployments with high accuracy via pattern matching and heuristics
- **Parallel Scanning** — Multi-threaded scanning with configurable worker count (`--workers`)

#### Collectors
- **GPU Metrics Collector** — Real-time GPU utilization, memory, temperature, and power via `nvidia-smi` pod exec
- **vLLM Runtime Collector** — Collects vLLM API metrics including P50/P95/P99 latency, token throughput, KV cache utilization, queue depth, and health status
- **Static Config Collector** — Extracts tensor parallel size, max model length, GPU memory utilization, precision, and serving configuration

#### CLI Commands
- `piqc scan` — Full cluster scan with extensive options
- `piqc test-connection` — Validate cluster connectivity and permissions
- `piqc version` — Display version information

#### Scan Options
- `--format [yaml|json|table]` — Multiple output formats
- `--collect-runtime` — Enable live vLLM API metrics collection
- `--no-exec` — Disable pod exec (skip GPU metrics for read-only environments)
- `--no-logs` — Disable log reading
- `--aggregate/--no-aggregate` — Control metrics aggregation across replicas
- `--combined` — Generate single combined output file
- `--output-piqc` — Generate standardized `piqc-facts.json` bundle
- `--namespace` — Scope scan to a specific namespace
- `--workers` — Configure parallel worker count
- `--timeout` — Configurable operation timeout
- `--mode [auto|remote|incluster|dry-run]` — Execution mode selection
- `--verbose` / `--debug` — Enhanced logging and trace output

#### Output Formats
- **YAML** — Kubernetes-style `ModelSpec` files per deployment
- **JSON** — Machine-readable structured output
- **Table** — Rich console table for human-readable viewing
- **PIQC Facts Bundle** — Standardized `piqc-scan.v0.1` schema for quality assessment systems

#### ModelSpec Schema (`modelspec/v1`)
- `metadata` — Name, namespace, collection timestamp, collector version
- `model` — Model name, architecture, parameter count, identification confidence
- `engine` — Engine name, version, detection confidence
- `inference` — Precision, tensor parallel size, max model length, GPU memory utilization
- `resources` — Replica count, GPU count, per-GPU metrics (type, memory, utilization)
- `runtimeState` — Health status, KV cache usage, prompt/generation throughput
- `dataCompleteness` — Static config, GPU metrics, runtime metrics availability flags

#### PIQC Facts Schema (`piqc-scan.v0.1`)
- Standardized fact keys: `runtime.engineType`, `runtime.engineVersion`, `hardware.gpuType`, `hardware.gpuCount`, `hardware.gpuMemoryTotal`, `observed.gpuUtilization`, `vllm.tensorParallelSize`, `vllm.maxModelLen`, `observed.kvCacheUsage`
- Per-fact `dataConfidence` scoring (`high`, `medium`, `low`)

#### Kubernetes Integration
- **RBAC Manifests** (`rbac/`) — ServiceAccount, ClusterRole, and ClusterRoleBinding
- **In-Cluster Mode** — Runs natively as a Kubernetes Job or CronJob using ServiceAccount tokens
- **Remote Mode** — Uses kubeconfig for local or CI/CD execution
- **Auto-Detection Mode** — Automatically selects in-cluster vs remote

#### Developer Tooling
- **Poetry** project setup with `pyproject.toml`
- **Unit Test Suite** — `tests/unit/` with pytest
- **Integration Tests** — `tests/integration/` for end-to-end validation
- **Black** code formatting
- **Ruff** linting
- **MyPy** type checking
- **Coverage** reporting
- **GitHub Issue Templates** — Bug report and feature request templates
- **Apache 2.0 License**

### Technical Details

- **Python 3.11+** required
- **kubernetes** Python client for cluster access
- **rich** for enhanced console output
- **click** for CLI framework
- **pydantic** for data model validation
- Scans up to **N namespaces** in parallel with configurable worker pool
- Weighted confidence scoring: image patterns (40%), environment variables (30%), CLI args (20%), labels (10%)

---

## Version History Summary

| Version | Date | Highlights |
|---|---|---|
| `1.0.0` | 2026-01-30 | Initial stable release |

---

## Upcoming (Roadmap)

See [GOVERNANCE.md](GOVERNANCE.md) and open [Issues](https://github.com/paralleliq/piqc/issues) for planned features.

| Feature | Target Version |
|---|---|
| AMD GPU Support (ROCm / `rocm-smi`) | `1.1.0` |
| LLM-D (LLM-Distributed) topology support | `1.2.0` |
| PIQC Advisor integration | `1.x` |
| Extended framework detection (Ray Serve, TGI) | `1.x` |

---

[Unreleased]: https://github.com/paralleliq/piqc/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/paralleliq/piqc/releases/tag/v1.0.0

---

*Part of the [PIQC](https://github.com/paralleliq/piqc) project — Maintained by [ParalleliQ](https://paralleliq.ai)*
