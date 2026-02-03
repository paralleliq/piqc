# Contributing to PIQC Fact Collector

Thank you for your interest in contributing to PIQC Fact Collector - A tool to detect runtime information from deployed LLMs.

---

## Repository Structure

PIQC is part of the ParallelIQ ecosystem:
- **piqc**: A Kubernetes native tool for extracing facts about models in production.
- **modelspec**: Core specification and schema for AI model deployment
- **piqc-knowledge-base**: Documentation, examples, and knowledge resources that provide context for ModelSpec

---

## How to Contribute

### 1. Fork and Clone

```bash
# Fork the repo on GitHub, then clone YOUR fork:
git clone https://github.com/YourUsername/piqc.git
cd piqc

# Add upstream remote
git remote add upstream https://github.com/paralleliq/piqc.git

# Verify remotes
git remote -v
```

### 2. Sync and Create Branch

```bash
# Sync with latest changes
git checkout main
git pull upstream main
git push origin main

# Create feature branch from main
git checkout -b feature/your-feature-name
```

**Branch Naming:**
- `feature/` - New features or schema additions
- `fix/` - Bug fixes or spec corrections
- `docs/` - Documentation improvements

### 3. Make Your Changes

**Guidelines:**
- Provide documentation for any additional features or extensions
- Update examples in the examples directory if relevant
- Follow existing modelspec schema patterns

### 4. Commit and Push

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add GPU resource specification schema"

# Push to YOUR fork
git push origin feature/your-feature-name
```

**Commit Format:** `type: description`
- `fix:` - Bug fix or correction

### 5. Create Pull Request

1. Go to your fork on GitHub: `https://github.com/YourUsername/piqc`
2. Click **"Compare & pull request"**
3. Ensure:
   - Base: `paralleliq/piqc` `main`
   - Head: `YourUsername/piqc` `feature/your-feature-name`
4. Fill in PR description:

```markdown
## Description
Brief summary of what this PR does

## Changes Made
- Added X schema field
- Updated Y documentation
- Fixed Z validation issue

## Related Issues
Closes #123

## Testing
How you verified these changes work

## Checklist
- [ ] Schema changes are backward compatible
- [ ] Documentation updated
- [ ] Examples added/updated in piqc-knowledge-base (if applicable)
- [ ] Follows declarative principles
```

5. Submit PR

### 6. Address Review Feedback

```bash
# Make requested changes
# Edit files...

# Commit and push updates
git add .
git commit -m "fix: address review feedback on schema validation"
git push origin feature/your-feature-name
```

The PR automatically updates with new commits.

### 7. After PR Merges

```bash
# Update your main branch
git checkout main
git pull upstream main
git push origin main

# Delete feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

---

## Contribution Types

### Schema Improvements
- Add new model configuration fields
- Enhance validation rules
- Improve schema documentation

### Documentation
- Clarify existing specifications
- Add usage examples
- Improve README and guides

### Examples
- Provide real-world ModelSpec files
- Add deployment scenario examples
- Contribute to piqc-knowledge-base

### Bug Reports
- Report ambiguities or issues
- Suggest corrections
- Identify edge cases

---

## What to Contribute

### ✅ Welcome
- Schema enhancements for model deployment
- Platform-agnostic configuration options
- Clear documentation and examples
- Validation improvements
- Real-world use cases

### ❌ Avoid
- Deployment orchestration logic
- Vendor-specific implementations
- Runtime enforcement mechanisms
- Breaking changes without discussion

---

## Design Principles

1. **Declarative** - Describe "what", not "how"
2. **Platform-agnostic** - Works across any infrastructure
3. **Composable** - Modules can be combined
4. **Versioned** - Schema changes are tracked
5. **Documented** - Every field has clear purpose

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create an Issue with details
- **Ideas**: Open an Issue for discussion first
- **Docs**: Check piqc-knowledge-base repo

---

## Quick Command Reference

```bash
# Setup (once)
git clone https://github.com/YourUsername/piqc.git
cd piqc
git remote add upstream https://github.com/paralleliq/piqc.git

# Start work
git checkout main && git pull upstream main && git push origin main
git checkout -b feature/name

# During work
git add .
git commit -m "type: description"
git push origin feature/name

# After merge
git checkout main && git pull upstream main && git push origin main
git branch -d feature/name
git push origin --delete feature/name
```

---

## License

By contributing, you agree that your contributions will be licensed under the same Business Source License 1.1 that covers this project. See [LICENSE](LICENSE) for details.

---

## Community

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful, constructive, and collaborative.

Thank you for helping make ModelSpec better! 🚀
