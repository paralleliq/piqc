# Governance

This document describes the governance model for the **PIQC** project — how decisions are made, who maintains the project, and how contributors can grow their involvement over time.

---

## Project Overview

**PIQC** (Production Inference Quality Control) is a source-available Kubernetes-native introspection tool for discovering and documenting AI/LLM inference deployments. It is part of the broader [ParalleliQ](https://paralleliq.ai) ecosystem alongside [ModelSpec](https://github.com/paralleliq/modelspec) and the [PIQC Knowledge Base](https://github.com/paralleliq/piqc-knowledge-base).

---

## Governance Philosophy

PIQC follows a **benevolent maintainer model** — a small team of core maintainers makes decisions in the open, with increasing community input as the project matures.

Our core principles:

- **Transparency** — decisions and rationale are documented publicly
- **Meritocracy** — contribution quality and consistency earns greater voice
- **Pragmatism** — we value what works for real-world AI infrastructure teams
- **Neutrality** — PIQC remains vendor-neutral and runtime-agnostic where possible
- **Community-first** — we prioritize the needs of the broader MLOps and AI platform community

---

## Roles & Responsibilities

### 1. Users

Anyone who uses PIQC in their infrastructure. Users can:

- Open bug reports and feature requests
- Ask questions via GitHub Discussions
- Participate in community conversations

No formal membership required.

---

### 2. Contributors

Anyone who submits a pull request, opens an issue, or contributes to documentation. Contributors can:

- Submit PRs for bugs, features, docs, and tests
- Participate in issue and PR discussions
- Propose changes via the issue tracker

All contributions must follow the [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

---

### 3. Reviewers

Trusted contributors who have demonstrated consistent, high-quality contributions. Reviewers can:

- Review and approve pull requests
- Triage issues and label them
- Provide authoritative feedback on design decisions
- Be listed in [CODEOWNERS](.github/CODEOWNERS) for specific modules

**How to become a Reviewer:** After 3–5 meaningful accepted contributions, a maintainer may invite you. You may also request consideration by opening a Discussion.

---

### 4. Maintainers

Core team members with write access to the repository. Maintainers:

- Merge pull requests
- Cut releases and manage versioning
- Set roadmap direction
- Manage GitHub repository settings
- Represent the project in public

**Current Maintainers:**

| Name | Role | Contact |
|---|---|---|
| Sam Hosseini | Founder & Lead Maintainer | [sam@paralleliq.ai](mailto:sam@paralleliq.ai) |

Maintainers are listed in [CODEOWNERS](.github/CODEOWNERS).

---

## Decision Making

### Day-to-Day Decisions

Routine decisions (bug fixes, minor features, doc improvements) are made by any maintainer through normal PR review and merge.

### Significant Changes

Changes that affect the public API, PIQC facts schema, ModelSpec integration, or core discovery behavior require:

1. An **Issue or Discussion** opened to explain the proposed change
2. A **comment period of at least 5 business days** for community input
3. **Consensus among active maintainers** before merging

Significant changes include:

- Breaking changes to CLI flags or output schemas
- New data collection capabilities (e.g., new GPU vendor support)
- Changes to the PIQC facts schema (`piqc-facts.json`)
- Integration changes that affect ModelSpec compatibility
- New RBAC requirements

### Disagreements

When maintainers disagree on a significant decision:

1. The disagreement is documented in the relevant Issue or Discussion
2. Both perspectives are presented clearly
3. A final decision is made by the Lead Maintainer after considering community input
4. The rationale is recorded publicly

---

## Versioning & Releases

PIQC follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`):

| Type | When |
|---|---|
| **PATCH** (`x.x.1`) | Bug fixes, docs, minor improvements |
| **MINOR** (`x.1.0`) | New features, backward-compatible changes |
| **MAJOR** (`2.0.0`) | Breaking changes to CLI, schemas, or RBAC |

### Release Process

1. Maintainer updates `CHANGELOG.md` — moves `[Unreleased]` entries to the new version
2. Version bump in `pyproject.toml` and version badge in `README.md`
3. Git tag created: `git tag -a v<version> -m "Release v<version>"`
4. GitHub Release published with release notes
5. PyPI package published (if applicable)

### Release Cadence

- **Patch releases**: As needed for bug fixes and security patches
- **Minor releases**: Roughly quarterly, or when significant features are ready
- **Major releases**: As needed for breaking changes, announced with migration guides

---

## Roadmap

The PIQC roadmap is maintained in GitHub Issues and the README. Current major planned features include:

| Feature | Status |
|---|---|
| AMD GPU Support (ROCm / rocm-smi) | 🔴 Planned |
| LLM-D (LLM-Distributed) support | 🔴 Planned |
| PIQC Advisor integration | 🔴 Planned |
| Extended framework detection | 🟡 In progress |

Community members are welcome to propose roadmap items via GitHub Issues or Discussions.

---

## Relationship to the ParalleliQ Ecosystem

PIQC is one component of a broader source-available ecosystem:

| Project | Role |
|---|---|
| [PIQC Knowledge Base](https://github.com/paralleliq/piqc-knowledge-base) | What *should* be true — best practices and operational standards |
| [ModelSpec](https://github.com/paralleliq/modelspec) | What was *intended* — declarative model deployment contracts |
| **PIQC Scan** (this repo) | What is *actually running* — live runtime inspection |

Governance decisions that affect cross-project compatibility (e.g., schema changes) are coordinated across all three repositories.

---

## Changes to This Document

Changes to this governance document are subject to the same significant-change process described above — an open discussion period and maintainer consensus before merging.

---

## Contact

📨 **Governance inquiries:** [sam@paralleliq.ai](mailto:sam@paralleliq.ai)
🌐 **Website:** [paralleliq.ai](https://paralleliq.ai)
💬 **Discussions:** [GitHub Discussions](https://github.com/paralleliq/piqc/discussions)

---

*Part of the [PIQC](https://github.com/paralleliq/piqc) project — Maintained by [ParalleliQ](https://paralleliq.ai)*
