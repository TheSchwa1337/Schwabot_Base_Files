#!/bin/bash

echo "========================================"
echo "Schwabot Trading System - Linux Installer"
echo "========================================"
echo

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this installer as root."
    echo "Run as a regular user and use sudo when prompted."
    exit 1
fi

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    echo "Could not detect Linux distribution."
    exit 1
fi

echo "Detected: $OS $VER"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Installing..."
    
    case $ID in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y python3 python3-pip python3-venv
            ;;
        fedora|rhel|centos)
            sudo dnf install -y python3 python3-pip python3-venv
            ;;
        arch)
            sudo pacman -S --noconfirm python python-pip
            ;;
        *)
            echo "Unsupported distribution: $ID"
            echo "Please install Python 3.8+ manually and run this installer again."
            exit 1
            ;;
    esac
fi

echo "Python 3 found: $(python3 --version)"
echo

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv schwabot-env
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source schwabot-env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Schwabot dependencies..."
pip install -r ../../requirements.txt

# Install Schwabot
echo "Installing Schwabot..."
pip install -e ../..

# Create desktop entry
echo "Creating desktop entry..."
mkdir -p ~/.local/share/applications
cat > ~/.local/share/applications/schwabot.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Schwabot Trading System
Comment=Advanced AI-powered trading system
Exec=$(pwd)/schwabot-env/bin/python -m schwabot
Icon=$(pwd)/assets/schwabot-icon.png
Terminal=true
Categories=Finance;Office;
EOF

# Make desktop entry executable
chmod +x ~/.local/share/applications/schwabot.desktop

# Create systemd service (optional)
echo "Creating systemd service..."
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/schwabot.service << EOF
[Unit]
Description=Schwabot Trading System
After=network.target

[Service]
Type=simple
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/schwabot-env/bin/python -m schwabot
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF

# Enable systemd service
systemctl --user enable schwabot.service

echo
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo
echo "Schwabot has been installed successfully."
echo
echo "To start Schwabot:"
echo "1. Desktop: Search for 'Schwabot' in your applications menu"
echo "2. Terminal: source schwabot-env/bin/activate && python -m schwabot"
echo "3. Service: systemctl --user start schwabot"
echo
echo "For more information, see the documentation in docs/"
echo 