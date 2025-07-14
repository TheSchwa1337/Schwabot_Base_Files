@echo off
echo Setting up Schwabot Trading System for Windows...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-windows.txt
echo Setup complete!
pause
