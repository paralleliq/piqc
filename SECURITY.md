# Security Policy

## Overview

The **PIQC** project takes security seriously. This document describes how to report security vulnerabilities, what versions are supported with security patches, and how the team handles disclosure.

Because PIQC is a **Kubernetes-native tool that executes pod-level commands and reads cluster state**, security in deployment and operation is critical.

---

## Supported Versions

Security fixes are applied to the latest stable release. We encourage all users to stay on the latest version.

| Version | Supported |
|---|---|
| Latest (`main`) | ✅ Active support |
| Previous minor  | ⚠️ Critical fixes only |
| Older versions  | ❌ Not supported |

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities as public GitHub issues, pull requests, or discussions.**

Public disclosure before a fix is available puts all PIQC users at risk.

### Private Disclosure

Report vulnerabilities directly to the maintainers:

📨 **[security@paralleliq.ai](mailto:security@paralleliq.ai)**

**Subject line:** `[PIQC SECURITY] <brief description>`

### What to Include

Please provide as much of the following as possible:

- **Description** of the vulnerability and its potential impact
- **PIQC version** affected (`piqc version`)
- **Affected component** (e.g., discovery engine, GPU collector, RBAC, CLI)
- **Steps to reproduce** the issue
- **Proof of concept** or example output (redact sensitive cluster details)
- **Suggested fix** if you have one
- **Your preferred contact method** for follow-up

### Response Timeline

| Stage | Target Timeline |
|---|---|
| Acknowledgement | Within **48 hours** |
| Initial assessment | Within **5 business days** |
| Fix or mitigation | Within **30 days** for critical, **90 days** for others |
| Public disclosure | Coordinated with reporter after fix is available |

---

## Security Considerations for PIQC Deployments

### Kubernetes RBAC

PIQC requires Kubernetes permissions to discover and inspect workloads. Follow the principle of least privilege:

- Apply only the RBAC manifests provided in `rbac/`
- Scope permissions to specific namespaces where possible using `-n <namespace>`
- Do not grant `cluster-admin` to the PIQC service account
- Rotate the ServiceAccount token regularly in production environments

```bash
# Apply scoped RBAC manifests
kubectl apply -f rbac/
```

### Pod Exec (`pods/exec`)

PIQC uses `pods/exec` to run `nvidia-smi` for GPU metrics. This permission is sensitive:

- If you do not require GPU metrics, use `--no-exec` to disable pod exec entirely
- Restrict exec permissions to only the namespaces containing inference workloads
- Audit exec activity using Kubernetes audit logs

```bash
# Disable pod exec (safer for read-only environments)
piqc scan --no-exec
```

### Output Files

PIQC output files (YAML, JSON, PIQC facts bundles) may contain sensitive operational data:

- Model names and inference endpoints
- GPU hardware inventory
- Kubernetes namespace and deployment names
- Runtime performance metrics

**Treat PIQC output files as internal operational data.** Do not commit them to public repositories or share them without review.

### In-Cluster Mode

When running PIQC in-cluster (e.g., as a Kubernetes Job or CronJob):

- Use a dedicated ServiceAccount — never use the `default` service account
- Mount only the RBAC-scoped ServiceAccount token
- Store output in a secured volume or push to an authenticated destination
- Limit the container image to the official PIQC release

### Kubeconfig Security

When running in remote mode:

- Use a dedicated kubeconfig with minimal permissions
- Do not embed cluster admin credentials in CI/CD pipelines
- Rotate credentials regularly
- Prefer short-lived tokens (e.g., GKE Workload Identity, EKS IRSA)

---

## Known Security Design Decisions

| Decision | Rationale |
|---|---|
| Read-only cluster access by default | PIQC never modifies cluster state |
| Pod exec is opt-in | `--no-exec` disables exec for sensitive environments |
| No network egress from scan results | Output is local only — nothing is sent to external services |
| No credentials stored | PIQC uses kubeconfig or in-cluster ServiceAccount — no secrets are persisted |

---

## Dependency Security

PIQC uses Poetry for dependency management. Dependencies are pinned in `poetry.lock`.

To audit dependencies for known vulnerabilities:

```bash
# Using pip-audit
poetry run pip-audit

# Or using safety
poetry run safety check
```

We recommend running dependency audits as part of your CI/CD pipeline before deploying PIQC in production environments.

---

## Coordinated Disclosure Policy

We follow **responsible disclosure**:

1. Reporter submits vulnerability privately
2. Maintainers acknowledge and assess
3. Fix is developed and tested
4. Release is prepared
5. CVE is requested if applicable
6. Reporter is credited (unless anonymity is requested)
7. Public disclosure in release notes and CHANGELOG

We ask that reporters:
- Allow reasonable time for a fix before public disclosure
- Not exploit the vulnerability against production systems
- Not access or modify other users' data

---

## Contact

📨 **Security Contact:** [security@paralleliq.ai](mailto:security@paralleliq.ai)

🌐 **Website:** [paralleliq.ai](https://paralleliq.ai)

💼 **Founder & CEO:** Sam Hosseini

---

*Part of the [PIQC](https://github.com/paralleliq/piqc) project — Maintained by [ParalleliQ](https://paralleliq.ai)*
