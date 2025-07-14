#!/bin/bash
# Schwabot Runner Script

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the main launcher script
LAUNCHER_SCRIPT="schwabot_launcher.py"

# Check if the launcher script exists
if [ ! -f "$LAUNCHER_SCRIPT" ]; then
    echo "Error: Launcher script '$LAUNCHER_SCRIPT' not found."
    exit 1
fi

# Activate Python virtual environment if it exists
# This is best practice for managing dependencies
if [ -d "schwabot_env" ]; then
    echo "Activating Python virtual environment..."
    source schwabot_env/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating Python virtual environment..."
    source .venv/bin/activate
fi

# Run the launcher with all command-line arguments passed to this script
echo "🚀 Launching Schwabot..."
python "$LAUNCHER_SCRIPT" "$@"

# Deactivate virtual environment on exit, if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating virtual environment."
    deactivate
fi 