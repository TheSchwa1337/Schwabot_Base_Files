# Contributing to Schwabot

We welcome contributions to Schwabot! Whether you're fixing bugs, adding new features, improving documentation, or optimizing code, your help is greatly appreciated.

Please read through this document to understand our development process, coding standards, and how to submit your contributions.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Request Guidelines](#pull-request-guidelines)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Licensing](#licensing)

## Code of Conduct
This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to info@schwabot.com.

## How to Contribute

### Reporting Bugs
If you find a bug, please open an issue on our [GitHub Issues page](https://github.com/schwabot/schwabot/issues). Include:
- A clear and concise description of the bug.
- Steps to reproduce the behavior.
- Expected behavior.
- Screenshots or error logs if applicable.
- Your operating system and Python version.

### Suggesting Enhancements
For new features or improvements, open an issue on the [GitHub Issues page](https://github.com/schwabot/schwabot/issues). Provide a clear and detailed explanation of:
- The proposed enhancement.
- Why it would be useful.
- Any potential challenges or alternative solutions.

### Pull Request Guidelines
1. **Fork the Repository**: Start by forking the Schwabot repository to your GitHub account.
2. **Clone Your Fork**: Clone your forked repository to your local machine.
   ```bash
   git clone https://github.com/your-username/schwabot.git
   cd schwabot
   ```
3. **Create a New Branch**: Create a new branch for your feature or bug fix.
   ```bash
   git checkout -b feature/your-feature-name-or-bugfix/your-bugfix-name
   ```
4. **Implement Your Changes**: Make your changes, ensuring they adhere to our [Coding Standards](#coding-standards).
5. **Write Tests**: Add or update tests to cover your changes. See [Testing](#testing) section.
6. **Run Tests**: Ensure all tests pass locally.
   ```bash
   pytest
   ```
7. **Lint Your Code**: Ensure your code passes `flake8` and `mypy` checks.
   ```bash
   flake8 .
   mypy .
   ```
8. **Commit Your Changes**: Write clear and concise commit messages.
   ```bash
   git commit -m "feat: Add new awesome feature" # or "fix: Resolve critical bug"
   ```
9. **Push to Your Fork**: Push your new branch to your forked repository.
   ```bash
   git push origin feature/your-feature-name
   ```
10. **Open a Pull Request (PR)**: Go to the original Schwabot repository on GitHub and open a pull request from your new branch. Fill out the PR template with relevant details.

## Development Setup
Follow the [Installation](#installation) guide in `README.md` to set up your development environment from source.

## Coding Standards
We strive for clean, readable, and maintainable code. Please adhere to the following:
- **PEP 8**: Follow Python's official style guide.
- **Flake8**: Ensure your code passes `flake8` checks.
- **MyPy**: Use type hints and ensure your code passes `mypy` checks (configured in `mypy.ini`).
- **Docstrings**: Use Google-style docstrings for all modules, classes, and functions.
- **Comments**: Use inline comments sparingly for complex logic, and add section comments for larger code blocks.

## Testing
All new features and bug fixes should be accompanied by appropriate unit and/or integration tests. We use `pytest` for testing.

To run tests, activate your virtual environment and run:
```bash
pytest
```

## Licensing
By contributing to Schwabot, you agree that your contributions will be licensed under the MIT License, as per the [LICENSE](LICENSE) file. 