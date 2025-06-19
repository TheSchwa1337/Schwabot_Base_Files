#!/usr/bin/env python3
"""
Setup Pre-Commit Hooks for Schwabot
===================================

This script installs and configures pre-commit hooks to enforce:
- Flake8 compliance
- Type checking with mypy
- Code formatting with Black
- Import sorting with isort
- Custom Schwabot-specific validations

Based on systematic elimination of 257+ flake8 issues.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command: list[str], description: str) -> bool:
    """Run a command and log the result"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False


def check_prerequisites() -> bool:
    """Check if required tools are installed"""
    logger.info("Checking prerequisites...")

    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False

    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        logger.error("❌ pip not available")
        return False

    logger.info("✅ Prerequisites check passed")
    return True


def install_pre_commit() -> bool:
    """Install pre-commit package"""
    return run_command(
        [sys.executable, "-m", "pip", "install", "pre-commit"],
        "Installing pre-commit package"
    )


def install_hooks() -> bool:
    """Install pre-commit hooks"""
    return run_command(
        ["pre-commit", "install"],
        "Installing pre-commit hooks"
    )


def install_additional_dependencies() -> bool:
    """Install additional dependencies for hooks"""
    dependencies = [
        "black",
        "flake8",
        "flake8-annotations",
        "flake8-docstrings",
        "flake8-bugbear",
        "mypy",
        "isort",
        "types-requests",
        "types-PyYAML"
    ]

    logger.info("Installing additional dependencies...")
    for dep in dependencies:
        if not run_command(
            [sys.executable, "-m", "pip", "install", dep],
            f"Installing {dep}"
        ):
            return False

    return True


def run_initial_check() -> bool:
    """Run pre-commit on all files"""
    return run_command(
        ["pre-commit", "run", "--all-files"],
        "Running initial pre-commit check on all files"
    )


def create_gitignore_entries() -> None:
    """Add pre-commit related entries to .gitignore"""
    gitignore_path = Path(".gitignore")

    entries = [
        "",
        "# Pre-commit",
        ".pre-commit-config.yaml.bak",
        "mypy.ini.bak",
        "",
        "# Type checking",
        ".mypy_cache/",
        "",
    ]

    if gitignore_path.exists():
        content = gitignore_path.read_text()
        for entry in entries:
            if entry not in content:
                content += entry
        gitignore_path.write_text(content)
        logger.info("✅ Updated .gitignore with pre-commit entries")
    else:
        gitignore_path.write_text("\n".join(entries))
        logger.info("✅ Created .gitignore with pre-commit entries")


def main() -> None:
    """Main setup function"""
    logger.info("🚀 Setting up Pre-Commit Hooks for Schwabot")
    logger.info("=" * 50)

    # Check if we're in the right directory
    if not Path(".pre-commit-config.yaml").exists():
        logger.error("❌ .pre-commit-config.yaml not found. Run this from the project root.")
        sys.exit(1)

    # Run setup steps
    steps = [
        ("Checking prerequisites", check_prerequisites),
        ("Installing pre-commit", install_pre_commit),
        ("Installing additional dependencies", install_additional_dependencies),
        ("Installing hooks", install_hooks),
        ("Creating .gitignore entries", lambda: create_gitignore_entries() or True),
    ]

    for step_name, step_func in steps:
        logger.info(f"\n--- {step_name} ---")
        if not step_func():
            logger.error(f"❌ Setup failed at: {step_name}")
            sys.exit(1)

    # Optional: Run initial check
    logger.info("\n--- Running Initial Check ---")
    if run_initial_check():
        logger.info("🎉 Pre-commit setup completed successfully!")
        logger.info("\n📋 What's now available:")
        logger.info("   ✅ Automatic code formatting with Black")
        logger.info("   ✅ Flake8 compliance checking")
        logger.info("   ✅ Type checking with mypy")
        logger.info("   ✅ Import sorting with isort")
        logger.info("   ✅ Custom Schwabot validations")
        logger.info("\n💡 Usage:")
        logger.info("   - Hooks run automatically on commit")
        logger.info("   - Manual check: pre-commit run --all-files")
        logger.info("   - Check specific file: pre-commit run --files file.py")
    else:
        logger.warning("⚠️ Setup completed but initial check had issues")
        logger.info("💡 You can run 'pre-commit run --all-files' to see details")


if __name__ == "__main__":
    main()