#!/bin/bash
set -e

echo "Starting Code Quality Checks..."

# Install required tools
echo "[*] Installing required tools"
pip install autoflake black isort flake8 chardet

# Prepare files
echo "[*] Preparing files for code quality check"
python prepare_flake8.py

# Run comprehensive code check
echo "[*] Running comprehensive code check"
python comprehensive_code_check.py

# Run flake8 check
echo "[*] Running flake8 check"
flake8 core/ schwabot/ utils/ config/ --max-line-length=100 --count

# Check code formatting with black
echo "[*] Checking code formatting with black"
black core/ schwabot/ utils/ config/ --check --line-length=100

# Check import sorting
echo "[*] Checking import sorting"
isort core/ schwabot/ utils/ config/ --check-only

# Generate final report
echo "[*] Generating code quality report"
python final_code_quality_check.py

# Success message
echo "Code Quality Checks Passed Successfully!" 