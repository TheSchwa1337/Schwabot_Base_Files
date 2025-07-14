#!/bin/bash

# Exit on any error
set -e

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Install required tools
echo "Installing required tools..."
pip install autoflake black isort flake8

# Step 1: Prepare files
echo "Preparing files for flake8 check..."
python prepare_flake8.py

# Step 2: Run comprehensive flake8 check
echo "Running comprehensive flake8 check..."
python comprehensive_flake8_check.py

# If we get here, everything is clean
echo "✅ Flake8 check completed successfully!" 