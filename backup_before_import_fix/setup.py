import os

from setuptools import find_packages, setup

#!/usr/bin/env python3
"""
Setup script for Hash Recollection Trading System
================================================

Cross-platform installation script for Windows, macOS, and Linux.
"""


# Read the README file


def read_readme():
    """Read README file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ()
        "Hash Recollection Trading System - Advanced trading bot with entropy analysis"
    )


# Read requirements


def read_requirements():
    """Read requirements from requirements.txt."""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return []
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup()
    name="schwabot",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[]
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "ccxt>=1.50.0",
        "pyyaml>=5.4.0",
        "requests>=2.25.0",
        "python-dotenv>=0.15.0",
        "websockets>=9.1",
        "aiohttp>=3.7.4",
        "scikit-learn>=0.24.0",
        "tensorflow>=2.4.0",
        "torch>=1.8.0",
        "asyncio>=3.4.3",
        "pydantic>=1.8.0",
        "psutil>=5.8.0",
        "pynvml>=11.0.0",
    ],
    extras_require={}
        "dev": []
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "trading": []
            "ccxt>=1.50.0",
            "ta-lib>=0.4.0",
        ],
        "ai": []
            "torch>=1.8.0",
            "tensorflow>=2.4.0",
            "transformers>=4.11.0",
        ],
        "visualization": []
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "bokeh>=2.3.0",
            "streamlit>=1.0.0",
        ],
        "gpu": []
            "pynvml>=11.0.0",
            "cupy>=9.0.0",
        ]
    },
    entry_points={}
        "console_scripts": []
            "schwabot=schwabot.cli:main",
            "schwabot-hub=schwabot.integration_hub:main",
            "schwabot-tensor=schwabot.tensor_cli:main",
            "schwabot-visualizer=schwabot.visualizer.core_visualizer:main",
        ],
    },
    python_requires=">=3.8",
    description="Schwabot Trading System - Advanced AI-Driven Trading Platform with Real-time Visualization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Schwabot Team",
    author_email="team@schwabot.com",
    url="https://github.com/schwabot/schwabot",
    classifiers=[]
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)

""""""
""""""
""""""
""""""
""""""
"""
"""
