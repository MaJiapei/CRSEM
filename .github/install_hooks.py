#!/usr/bin/env python3
"""Install git hooks for this repository.

Run this script to enable pre-commit checks:
    python .github/install_hooks.py

Hooks installed:
    - pre-commit: Checks sensitive files, documentation updates, tests
    - commit-msg: Checks commit message language and format
"""

import os
import shutil
import stat
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    git_hooks_dir = repo_root / ".git" / "hooks"
    project_hooks_dir = repo_root / ".git-hooks"

    print("Installing git hooks...")

    # Create hooks directory if it doesn't exist
    git_hooks_dir.mkdir(parents=True, exist_ok=True)

    # Install pre-commit hook
    pre_commit_src = project_hooks_dir / "pre-commit"
    pre_commit_dst = git_hooks_dir / "pre-commit"

    if pre_commit_src.exists():
        shutil.copy2(pre_commit_src, pre_commit_dst)
        # Make executable
        pre_commit_dst.chmod(pre_commit_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("✓ Installed pre-commit hook")
    else:
        print(f"✗ pre-commit hook not found: {pre_commit_src}")
        return 1

    # Install commit-msg hook
    commit_msg_src = project_hooks_dir / "commit-msg"
    commit_msg_dst = git_hooks_dir / "commit-msg"

    if commit_msg_src.exists():
        shutil.copy2(commit_msg_src, commit_msg_dst)
        # Make executable
        commit_msg_dst.chmod(commit_msg_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("✓ Installed commit-msg hook")
    else:
        print(f"✗ commit-msg hook not found: {commit_msg_src}")
        return 1

    print()
    print("Git hooks installed successfully!")
    print()
    print("To bypass hooks in emergency, use:")
    print("  git commit --no-verify")
    print()
    print("Hooks source files are in: .git-hooks/")
    print("Documentation: .github/CODE_STYLE.md")

    return 0


if __name__ == "__main__":
    exit(main())
