#!/bin/bash

echo "========================================"
echo "Schwabot Trading System - macOS Installer"
echo "========================================"
echo

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this installer as root."
    echo "Run as a regular user."
    exit 1
fi

# Check macOS version
MACOS_VERSION=$(sw_vers -productVersion)
echo "Detected: macOS $MACOS_VERSION"
echo

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Installing..."
    brew install python@3.11
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

# Create Applications folder entry
echo "Creating Applications folder entry..."
mkdir -p ~/Applications/Schwabot.app/Contents/MacOS
mkdir -p ~/Applications/Schwabot.app/Contents/Resources

# Create Info.plist
cat > ~/Applications/Schwabot.app/Contents/Info.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>schwabot</string>
    <key>CFBundleIdentifier</key>
    <string>com.schwabot.trading</string>
    <key>CFBundleName</key>
    <string>Schwabot</string>
    <key>CFBundleVersion</key>
    <string>2.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Create launcher script
cat > ~/Applications/Schwabot.app/Contents/MacOS/schwabot << EOF
#!/bin/bash
cd "$(dirname "$0")/../../../../"
source schwabot-env/bin/activate
python -m schwabot
EOF

# Make launcher executable
chmod +x ~/Applications/Schwabot.app/Contents/MacOS/schwabot

# Create LaunchAgent for auto-start
echo "Creating LaunchAgent for auto-start..."
mkdir -p ~/Library/LaunchAgents
cat > ~/Library/LaunchAgents/com.schwabot.trading.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.schwabot.trading</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(pwd)/schwabot-env/bin/python</string>
        <string>-m</string>
        <string>schwabot</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$(pwd)</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$(pwd)/logs/schwabot.log</string>
    <key>StandardErrorPath</key>
    <string>$(pwd)/logs/schwabot_error.log</string>
</dict>
</plist>
EOF

# Load LaunchAgent
launchctl load ~/Library/LaunchAgents/com.schwabot.trading.plist

echo
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo
echo "Schwabot has been installed successfully."
echo
echo "To start Schwabot:"
echo "1. Applications: Open Schwabot from Applications folder"
echo "2. Terminal: source schwabot-env/bin/activate && python -m schwabot"
echo "3. Auto-start: Schwabot will start automatically on login"
echo
echo "For more information, see the documentation in docs/"
echo 