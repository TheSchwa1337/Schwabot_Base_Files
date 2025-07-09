# Requirements Update Report

## Summary
- Original requirements backed up to: requirements.txt.backup
- Updated requirements file: requirements.txt
- Platform: windows

## Added Dependencies
- abcBuilt-in module for abstract base classes
- aiohttp>=3.8.1
- apscheduler>=3.9.1
- asyncio>=3.4.3
- black>=22.3.0
- bokeh>=3.2.0
- ccxt>=2.0.0
- collectionsBuilt-in module for collection types
- contextlibBuilt-in module for context managers
- coverage>=6.3.2
- cryptography>=37.0.2
- cupy>=10.0.0
- dash>=2.12.0
- dask>=2022.5.0
- fastapi>=0.110.0
- flake8>=4.0.1
- flask>=2.2.0
- flask_cors>=3.0.10
- flask_socketio>=5.3.0
- functoolsBuilt-in module for function tools
- ioBuilt-in module for I/O operations
- isort>=5.10.1
- line-profiler>=3.5.0
- loguru>=0.6.0
- matplotlib>=3.5.2
- memory-profiler>=0.60.0
- multiprocessingBuilt-in module for multiprocessing
- mypy>=0.950
- numpy>=1.22.0
- nvidia-cuda-nvcc-cu11>=11.7.0
- nvidia-cuda-runtime-cu11>=11.7.0
- pandas>=1.4.0
- pandas-ta>=0.3.14b0
- pennylane>=0.26.0
- pkgutilBuilt-in module for package utilities
- plotly>=5.8.0
- psutil>=5.9.0
- pylint>=2.13.5
- pymongo>=4.0.0
- pytest>=7.1.2
- pytest-asyncio>=0.18.3
- python-dotenv>=0.20.0
- pyyaml>=6.0
- qiskit>=0.36.0
- redis>=4.0.0
- requests>=2.27.1
- scikit-learn>=1.1.0
- scipy>=1.8.0
- seaborn>=0.11.2
- sqlalchemy>=1.4.0
- starlette>=0.36.0
- streamlit>=1.34.0
- ta-lib>=0.4.20
- tensorflow>=2.9.0
- torch>=1.11.0
- typesBuilt-in module for type objects
- uvicorn[standard]>=0.27.0
- websockets>=10.4

## Platform-specific Dependencies
- pywin32>=228
- wmi>=1.5.1

## Setup Instructions
1. Install Python 3.8 or higher
2. Create a virtual environment: `python -m venv schwabot_env`
3. Activate the virtual environment:
   - Windows: `schwabot_env\Scripts\activate`
   - Unix/Linux/macOS: `source schwabot_env/bin/activate`
4. Install dependencies:
   - Windows: `setup_windows.bat`
   - Unix/Linux/macOS: `./setup_unix.sh`
   - Or manually: `pip install -r requirements.txt`
