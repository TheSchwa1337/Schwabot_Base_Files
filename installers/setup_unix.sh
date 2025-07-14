#!/bin/bash
echo "Setting up Schwabot Trading System..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Platform-specific requirements
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    python3 -m pip install -r requirements-windows.txt
elif [[ "$OSTYPE" == "darwin"* ]]; then
    python3 -m pip install -r requirements-darwin.txt
else
    python3 -m pip install -r requirements-linux.txt
fi

echo "Setup complete!"
