#!/usr/bin/env python3
"""
Setup script for Hash Recollection Trading System
================================================

Cross-platform installation script for Windows, macOS, and Linux.
"""

from setuptools import setup, find_packages
import os

# Read the README file


def read_readme():
    """Read README file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return (
        "Hash Recollection Trading System - Advanced trading bot with entropy analysis"
    )


# Read requirements


def read_requirements():
    """Read requirements from requirements.txt."""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="schwabot",
    version="1.0.0",
    author="Schwabot Team",
    author_email="team@schwabot.com",
    description="Schwabot – Universal cross-platform trading bot",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/schwabot/hash-recollection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "flake8>=4.0.0",
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
        ],
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "schwabot=schwabot.launch:run",
            "schwabot-cli=schwabot.cli:main",
            "schwabot-gui=schwabot.gui:launch",
            "schwabot-tray=schwabot.tray:run_tray",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

""""""
""""""
""""""
""""""
""""""
"""
"""
