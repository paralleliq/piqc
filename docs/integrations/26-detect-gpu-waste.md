# Tutorial: Detect GPU Waste with piqc

## Introduction

This tutorial demonstrates how to scan a vLLM Production Stack deployment for GPU waste using [piqc](https://github.com/paralleliq/piqc), an open-source, vLLM-native fact collector for Kubernetes. piqc reads the same kind of signals exposed by the [observability stack](../observability/README.md) (KV cache hit rate, requests running/waiting, GPU allocation) and turns them into a prioritized list of findings with an estimated dollar cost — for example, GPU nodes with unscheduled capacity, deployments with no traffic for hours, or a low prefix cache hit rate that LMCache offloading would fix.

piqc runs *against* your cluster from outside it — it does not require any Helm chart changes, a Production Stack dependency, or cluster-admin RBAC beyond read access to pods and nodes.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Installing piqc](#step-1-installing-piqc)
3. [Step 2: Running a Scan](#step-2-running-a-scan)
4. [Step 3: Reading the Output](#step-3-reading-the-output)
5. [Conclusion](#conclusion)

## Prerequisites

- Completion of [01-minimal-helm-installation.md](../tutorials/01-minimal-helm-installation.md) (a running Production Stack deployment)
- A local `kubeconfig` with access to the cluster
- Python 3.11+ on the machine you're running the scan from (your laptop or a CI runner — not the cluster itself)

## Step 1: Installing piqc

Install piqc with [`pipx`](https://pipx.pypa.io) to keep it isolated from any other Python tooling on your machine:

```bash
pipx install piqc
```

(Plain `pip install piqc` also works, but piqc requires pydantic v2 — if you have other tools pinned to pydantic v1 in the same environment, `pipx` avoids any conflict.)

## Step 2: Running a Scan

Point piqc at the same cluster your Production Stack deployment is running on:

```bash
piqc scan --format table
```

This discovers every vLLM deployment in the cluster, pulls deployment and runtime facts (GPU type/count, KV cache usage, requests running/waiting, prefix cache hit rate), and prints a cost report — GPU spend rate, estimated daily waste, and Model FLOPS Utilization (MFU) per deployment.

## Step 3: Reading the Output

piqc flags three waste patterns most relevant to a Production Stack deployment:

- **Unallocated GPU capacity** — a node has GPUs that no pod has claimed (e.g. you scaled down a `modelSpec` replica but the node is still provisioned).
- **Idle trickle traffic** — a deployment has had zero in-flight requests for hours (a model you launched for testing and forgot about).
- **Low prefix cache hit rate** — a deployment is repeatedly recomputing context it's already seen. If you haven't already enabled KV cache offloading, this is the signal that [05-offload-kv-cache.md](../tutorials/05-offload-kv-cache.md) is worth doing.

Each finding includes the evidence (the underlying facts) and a suggested corrective action, so you can decide whether to scale down, reconfigure, or — in the prefix-cache case — turn on LMCache.

## Conclusion

piqc complements the observability stack you already have: Prometheus and Grafana show you the metrics, piqc tells you what they're costing you and what to do about it. Run it after any Production Stack deployment to catch waste before it shows up on your GPU bill.
