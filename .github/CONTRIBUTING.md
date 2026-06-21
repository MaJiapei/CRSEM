# Contributing to CRSEM

Thank you for contributing to the CRSEM project! This document provides guidelines for contributors.

## Quick Start

1. **Install hooks** (required):
   ```bash
   python .github/install_hooks.py
   ```

2. **Before each commit**:
   - Code is tested
   - Documentation is updated
   - Commit message is in English

## Development Workflow

### 1. Create a branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make changes
- Write tests for new functionality
- Update documentation
- Follow existing code style

### 3. Pre-commit checks
The pre-commit hook will automatically check:
- ✗ No sensitive files (credentials, keys)
- ✗ Commit message in English
- ⚠ Documentation updates for code changes
- ⚠ Tests pass

### 4. Commit
```bash
git add <files>
git commit -m "feat: add new feature description"
```

### 5. Push and create PR
```bash
git push origin feature/your-feature-name
```

## Commit Message Format

Use the [conventional commits](https://www.conventionalcommits.org/) format:

```
<type>: <short description>
```

### Types
| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `docs` | Documentation only changes |
| `test` | Adding or updating tests |
| `chore` | Changes to build process or auxiliary tools |

### Good Examples
```
feat: add multi-basin calibration support
fix: handle NaN values in sediment flux calculation
refactor: extract validation logic to separate module
docs: update USER_GUIDE with calibration workflow
test: add unit tests for ObjectiveEvaluator
chore: update numpy dependency to 2.0
```

### Bad Examples
```
update code
修复 bug
修改了一些代码
add new feature and fix some bugs
```

## Documentation Guidelines

Update documentation when:

| Change Type | Files to Update |
|-------------|-----------------|
| API change | `docs/USER_GUIDE.md`, docstrings |
| Architecture change | `docs/ARCHITECTURE.md` |
| New feature | `docs/USER_GUIDE.md` |
| Bug fix | `docs/USER_GUIDE.md` if usage changed |

## Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Write docstrings for public APIs
- Keep functions focused and small

## Testing

Run tests before committing:
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_calibration_components.py -v

# Run with coverage
python -m pytest tests/ --cov=CRSEM
```

## Questions?

- See `.github/CODE_STYLE.md` for detailed style guidelines
- Check existing PRs for examples
- Ask in project discussions
