# Code Quality Guide

## Overview

This project uses multiple tools to ensure high code quality, consistent formatting, and best practices.

## Prerequisites

- Python 3.11+
- pip
- virtualenv (recommended)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

## Linting and Code Quality Tools

### Flake8
- Checks code style and potential errors
- Configuration: `pyproject.toml`
- Run: `flake8 .`

### Black
- Automatic code formatter
- Ensures consistent code style
- Configuration: `pyproject.toml`
- Run: `black .`

### isort
- Sorts and organizes imports
- Configuration: `pyproject.toml`
- Run: `isort .`

### Pylint
- Analyzes code for potential issues
- Configuration: `pyproject.toml`
- Run: `pylint **/*.py`

### MyPy
- Static type checking
- Configuration: `pyproject.toml`
- Run: `mypy .`

### Bandit
- Security linting
- Configuration: `pyproject.toml`
- Run: `bandit -r .`

## Comprehensive Checks

Run all checks with:
```bash
python comprehensive_linting.py
```

## Pre-Commit Hooks

Consider using pre-commit hooks to automatically run these checks before committing:

1. Install pre-commit:
```bash
pip install pre-commit
pre-commit install
```

2. Create `.pre-commit-config.yaml`:
```yaml
repos:
-   repo: local
    hooks:
    -   id: flake8
        name: flake8
        entry: flake8
        language: system
    -   id: black
        name: black
        entry: black
        language: system
    -   id: isort
        name: isort
        entry: isort
        language: system
```

## Best Practices

- Keep line length under 100 characters
- Use type hints
- Write docstrings
- Handle exceptions explicitly
- Avoid using `assert` for runtime checks

## Troubleshooting

- If a file cannot be formatted, check its Python version compatibility
- Use `# fmt: off` and `# fmt: on` to disable formatting for specific blocks
- Use `# type: ignore` sparingly for type checking exceptions

## Contributing

1. Run linters before submitting a PR
2. Fix all linting issues
3. Ensure 100% type coverage where possible 