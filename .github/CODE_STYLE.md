# Code Style Guide

## Commit Message Convention

All git commit messages MUST be written in English.

### Format
```
<type>: <short description>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring (no functional change)
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, config, etc.)

### Examples
```
feat: add multi-basin support for calibration
fix: prevent overflow in sediment flux calculation
refactor: extract validation logic to separate module
docs: update USER_GUIDE with new calibration workflow
test: add unit tests for ObjectiveEvaluator
chore: bump numpy dependency to 2.0
```

## Pre-commit Checklist

Before each commit, verify:

- [ ] Code changes are tested locally
- [ ] Documentation is updated if behavior changed
- [ ] Commit message is in English
- [ ] No sensitive data (API keys, passwords) is committed

## Documentation Updates

Update documentation when:

| Change Type | Files to Update |
|-------------|-----------------|
| API change | `docs/USER_GUIDE.md`, docstrings |
| Architecture change | `docs/ARCHITECTURE.md` |
| New feature | `docs/USER_GUIDE.md` |
| Bug fix | `docs/USER_GUIDE.md` if usage changed |
