#!/usr/bin/env python3
"""
Schwabot Trading System - Cross-Platform Package Setup
=====================================================

This setup.py file enables packaging Schwabot as a cross-platform application
for Linux, Windows, and macOS with proper entry points and dependencies.
"""

import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

# Package configuration
PACKAGE_CONFIG = {
    "name": "schwabot",
    "version": "2.0.0",
    "description": "Hardware-scale-aware economic kernel for federated trading devices",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "author": "Schwabot Development Team",
    "author_email": "dev@schwabot.ai",
    "url": "https://github.com/schwabot/schwabot",
    "license": "MIT",
    "classifiers": [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    "python_requires": ">=3.8",
    "packages": find_packages(include=[
        "core*",
        "ui*", 
        "config*",
        "utils*",
        "mathlib*",
        "ncco_core*",
        "tests*",
        "docs*",
        "examples*",
        "scripts*",
        "tools*"
    ]),
    "include_package_data": True,
    "package_data": {
        "": [
            "*.yaml",
            "*.yml", 
            "*.json",
            "*.md",
            "*.txt",
            "*.ini",
            "*.cfg",
            "templates/*",
            "static/*",
            "config/*",
            "logs/*"
        ]
    },
    "install_requires": read_requirements("requirements.txt"),
    "extras_require": {
        "dev": [
            "pytest>=6.2.0",
            "pytest-asyncio>=0.15.0", 
            "pytest-cov>=2.12.0",
            "flake8>=3.9.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "gpu": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
            "cupy-cuda11x>=9.0.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
            "statsmodels>=0.13.0",
            "sympy>=1.9.0",
            "numba>=0.56.0",
            "cython>=0.29.0",
        ],
        "monitoring": [
            "prometheus-client>=0.11.0",
            "grafana-api>=1.0.0",
            "sentry-sdk>=1.3.0",
            "rollbar>=0.16.0",
        ],
        "database": [
            "sqlalchemy>=1.4.0",
            "redis>=3.5.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "bokeh>=2.3.0",
        ]
    },
    "entry_points": {
        "console_scripts": [
            "schwabot=run_schwabot:main",
            "schwabot-dashboard=ui.schwabot_dashboard:main",
            "schwabot-validate=system_validation:main",
            "schwabot-test=test_mathematical_integration:main",
            "schwabot-cli=core.cli:main",
        ],
        "gui_scripts": [
            "schwabot-gui=ui.schwabot_dashboard:main",
        ]
    },
    "zip_safe": False,
    "platforms": ["Linux", "Windows", "macOS"],
    "keywords": [
        "trading",
        "cryptocurrency", 
        "algorithmic-trading",
        "mathematical-trading",
        "distributed-systems",
        "ai",
        "machine-learning",
        "financial",
        "investment",
        "blockchain",
        "defi"
    ],
    "project_urls": {
        "Bug Reports": "https://github.com/schwabot/schwabot/issues",
        "Source": "https://github.com/schwabot/schwabot",
        "Documentation": "https://schwabot.readthedocs.io/",
        "Changelog": "https://github.com/schwabot/schwabot/blob/main/CHANGELOG.md",
    }
}

if __name__ == "__main__":
    setup(**PACKAGE_CONFIG) 