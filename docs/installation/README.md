# 🚀 Schwabot Installation Guide

## 📋 Overview

This guide provides step-by-step instructions for installing Schwabot on Linux, Windows, and macOS. Schwabot is a hardware-scale-aware economic kernel for federated trading devices with mathematical precision.

## 🎯 System Requirements

### **Minimum Requirements**
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10+, macOS 10.15+
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM
- **Storage**: 10GB free space
- **Network**: Broadband internet connection

### **Recommended Requirements**
- **Memory**: 8GB+ RAM
- **Storage**: 20GB+ free space
- **CPU**: Multi-core processor
- **GPU**: NVIDIA GPU (optional, for acceleration)

## 🐧 Linux Installation

### **Ubuntu/Debian**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv schwabot-env
source schwabot-env/bin/activate

# Install Schwabot
pip install schwabot-2.0.0-py3-none-any.whl

# Verify installation
schwabot --version
```

### **Red Hat/Fedora/CentOS**
```bash
# Install Python and dependencies
sudo yum install -y python3 python3-pip

# Create virtual environment
python3 -m venv schwabot-env
source schwabot-env/bin/activate

# Install Schwabot
pip install schwabot-2.0.0-py3-none-any.whl

# Verify installation
schwabot --version
```

### **Using Package Managers**

#### **Debian Package (.deb)**
```bash
# Install .deb package
sudo dpkg -i schwabot-2.0.0.deb

# Install dependencies if needed
sudo apt-get install -f

# Run Schwabot
schwabot
```

#### **RPM Package (.rpm)**
```bash
# Install .rpm package
sudo rpm -i schwabot-2.0.0.rpm

# Run Schwabot
schwabot
```

#### **AppImage (Universal)**
```bash
# Make executable
chmod +x schwabot-2.0.0-x86_64.AppImage

# Run directly
./schwabot-2.0.0-x86_64.AppImage
```

## 🪟 Windows Installation

### **Using Python Package**
```cmd
# Install Python 3.8+ from python.org
# Open Command Prompt as Administrator

# Create virtual environment
python -m venv schwabot-env
schwabot-env\Scripts\activate

# Install Schwabot
pip install schwabot-2.0.0-py3-none-any.whl

# Verify installation
schwabot --version
```

### **Using Executable (.exe)**
```cmd
# Download schwabot.exe
# Run directly
schwabot.exe

# With configuration
schwabot.exe --config config.yaml
```

### **Using MSI Installer**
```cmd
# Run MSI installer
msiexec /i schwabot-2.0.0.msi

# Schwabot will be installed to Program Files
# Start from Start Menu or run: schwabot
```

### **Portable Installation**
```cmd
# Extract portable package
unzip schwabot-2.0.0-portable.zip

# Navigate to directory
cd schwabot-2.0.0-portable

# Run Schwabot
start_schwabot.bat
```

## 🍎 macOS Installation

### **Using Python Package**
```bash
# Install Python 3.8+ (recommended: use Homebrew)
brew install python

# Create virtual environment
python3 -m venv schwabot-env
source schwabot-env/bin/activate

# Install Schwabot
pip install schwabot-2.0.0-py3-none-any.whl

# Verify installation
schwabot --version
```

### **Using Application Bundle (.app)**
```bash
# Download schwabot.app
# Drag to Applications folder
cp -r schwabot.app /Applications/

# Run from Applications
open /Applications/schwabot.app
```

### **Using Disk Image (.dmg)**
```bash
# Mount disk image
hdiutil attach schwabot-2.0.0.dmg

# Copy application to Applications
cp -r /Volumes/Schwabot/schwabot.app /Applications/

# Unmount disk image
hdiutil detach /Volumes/Schwabot

# Run application
open /Applications/schwabot.app
```

### **Using Package Installer (.pkg)**
```bash
# Install package
sudo installer -pkg schwabot-2.0.0.pkg -target /

# Run Schwabot
schwabot
```

## 🐳 Docker Installation

### **Using Docker Image**
```bash
# Pull Docker image
docker pull schwabot:latest

# Run container
docker run -d -p 8080:8080 -p 8081:8081 -p 8082:8082 schwabot:latest

# Run with volume mounts
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  schwabot:latest
```

### **Using Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'
services:
  schwabot:
    image: schwabot:latest
    ports:
      - "8080:8080"
      - "8081:8081"
      - "8082:8082"
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - SCHWABOT_ENV=production
```

```bash
# Start with Docker Compose
docker-compose up -d
```

## 🔧 Post-Installation Setup

### **1. Initial Configuration**
```bash
# Create configuration directory
mkdir -p ~/.schwabot/config

# Copy default configuration
cp config/schwabot_config.yaml ~/.schwabot/config/

# Edit configuration
nano ~/.schwabot/config/schwabot_config.yaml
```

### **2. Environment Variables**
```bash
# Set environment variables
export SCHWABOT_ENV=production
export SCHWABOT_LOG_LEVEL=INFO
export SCHWABOT_CONFIG_PATH=~/.schwabot/config/schwabot_config.yaml
```

### **3. Verify Installation**
```bash
# Check system status
schwabot-validate

# Start Schwabot
schwabot --config ~/.schwabot/config/schwabot_config.yaml

# Access web dashboard
# Open http://localhost:8080 in browser
```

## 🎨 Starting Schwabot

### **Command Line Interface**
```bash
# Basic start
schwabot

# With configuration
schwabot --config config.yaml

# With specific mode
schwabot --mode coordinator --port 8080

# With monitoring
schwabot --monitor
```

### **Web Dashboard**
```bash
# Start dashboard
schwabot-dashboard

# Access at http://localhost:8080
```

### **Desktop GUI**
```bash
# Start GUI application
schwabot-gui
```

## 🔒 Security Setup

### **Firewall Configuration**
```bash
# Linux (UFW)
sudo ufw allow 8080/tcp
sudo ufw allow 8081/tcp
sudo ufw allow 8082/tcp

# Windows (PowerShell)
New-NetFirewallRule -DisplayName "Schwabot" -Direction Inbound -Protocol TCP -LocalPort 8080,8081,8082 -Action Allow

# macOS
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /Applications/schwabot.app
```

### **SSL/HTTPS Setup**
```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Start with SSL
schwabot --ssl-cert cert.pem --ssl-key key.pem
```

## 🧪 Testing Installation

### **System Validation**
```bash
# Run comprehensive validation
schwabot-validate

# Test mathematical components
schwabot-test

# Check API endpoints
curl http://localhost:8081/health
```

### **Performance Testing**
```bash
# Monitor system performance
schwabot --monitor

# Check resource usage
top -p $(pgrep -f schwabot)
```

## 🆘 Troubleshooting

### **Common Installation Issues**

#### **Python Version Issues**
```bash
# Check Python version
python --version

# Install correct version
# Download from python.org or use package manager
```

#### **Permission Issues**
```bash
# Linux/macOS
sudo pip install schwabot-*.whl

# Windows (Run as Administrator)
pip install schwabot-*.whl
```

#### **Dependency Issues**
```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### **Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8080

# Use different ports
schwabot --dashboard-port 8083 --api-port 8084
```

### **Getting Help**
- **Documentation**: Check [Troubleshooting Guide](../troubleshooting/README.md)
- **Logs**: Check `logs/schwabot.log` for error details
- **Community**: Join Schwabot community forums
- **Issues**: Report bugs in project repository

## ✅ Installation Checklist

- [ ] System requirements met
- [ ] Python 3.8+ installed
- [ ] Schwabot package installed
- [ ] Configuration created
- [ ] Environment variables set
- [ ] Firewall configured
- [ ] System validation passed
- [ ] Web dashboard accessible
- [ ] API endpoints responding
- [ ] Security measures implemented

## 🚀 Next Steps

After successful installation:

1. **Read the [Quick Start Guide](../quickstart/README.md)**
2. **Configure your trading settings**
3. **Set up monitoring and alerts**
4. **Review the [User Manual](../user-guide/README.md)**
5. **Explore the [API Documentation](../api/README.md)**

---

**🎉 Congratulations! Schwabot is now installed and ready to use.** 