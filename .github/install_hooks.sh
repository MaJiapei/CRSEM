#!/usr/bin/env bash
# Install git hooks for this repository
# Run this script once to enable pre-commit checks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"
PROJECT_HOOKS_DIR="$REPO_ROOT/.git-hooks"

echo "Installing git hooks..."

# Create hooks directory if it doesn't exist
if [ ! -d "$GIT_HOOKS_DIR" ]; then
    mkdir -p "$GIT_HOOKS_DIR"
fi

# Install pre-commit hook
if [ -f "$PROJECT_HOOKS_DIR/pre-commit" ]; then
    cp "$PROJECT_HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
    chmod +x "$GIT_HOOKS_DIR/pre-commit"
    echo "✓ Installed pre-commit hook"
else
    echo "✗ pre-commit hook not found in .git-hooks/"
    exit 1
fi

echo ""
echo "Git hooks installed successfully!"
echo ""
echo "To bypass hooks in emergency, use:"
echo "  git commit --no-verify"
echo ""
echo "To update hooks after changes:"
echo "  python .github/install_hooks.py"
