# Config Folder

This directory contains all configuration files for Schwabot, including YAML, JSON, and Python config scripts.

## Contents
- **System configs**: Main, production, and environment-specific settings.
- **API keys and secrets**: (Never commit real secrets!)
- **Validation and integration configs**: For CI, testing, and deployment.

## Usage
- Edit configs to tune system behavior, enable/disable features, or set up new environments.
- Use templates for onboarding new deployments.

---
Keep sensitive data out of version control. Use `.env` or template files for secrets. 