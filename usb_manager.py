#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB Manager for Schwabot
========================

Automatically detects USB drives and manages secure storage for API keys and trading data.
Provides seamless .env file deployment and data offloading capabilities.
Enhanced with automatic detection and API key integration.
"""

import os
import shutil
import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import List, Dict, Any, Optional
import psutil
import platform
import logging
from datetime import datetime
import base64

logger = logging.getLogger(__name__)

class USBManager:
    """Comprehensive USB drive management for Schwabot security and data storage."""
    
    def __init__(self):
        self.detected_drives = []
        self.selected_drive = None
        self.env_file_name = ".env"
        self.schwabot_folder = "Schwabot_Data"
        self.backup_folder = "Backup_Data"
        self.registry_folder = "Registry_Files"
        self.memory_folder = "Memory_Keys"
        self.api_keys_file = "AOI_Base_Files_Schwabot/config/api_keys.json"
        
    def detect_usb_drives(self) -> List[Dict[str, Any]]:
        """Detect all available USB drives with detailed information."""
        try:
            drives = []
            
            # Get all drives
            for drive in range(ord('A'), ord('Z') + 1):
                drive_letter = chr(drive) + ":\\"
                
                if os.path.exists(drive_letter):
                    try:
                        # Get drive information
                        drive_info = psutil.disk_usage(drive_letter)
                        
                        # Check if it's a removable drive (likely USB)
                        if self._is_removable_drive(drive_letter):
                            drive_data = {
                                'letter': drive_letter,
                                'total_gb': drive_info.total / (1024**3),
                                'free_gb': drive_info.free / (1024**3),
                                'used_gb': (drive_info.total - drive_info.free) / (1024**3),
                                'label': self._get_drive_label(drive_letter),
                                'has_env': os.path.exists(os.path.join(drive_letter, self.env_file_name)),
                                'has_schwabot': os.path.exists(os.path.join(drive_letter, self.schwabot_folder)),
                                'is_writable': self._is_writable(drive_letter)
                            }
                            drives.append(drive_data)
                            
                    except (OSError, PermissionError):
                        continue
            
            self.detected_drives = drives
            logger.info(f"Detected {len(drives)} USB drives")
            return drives
            
        except Exception as e:
            logger.error(f"Error detecting USB drives: {e}")
            return []
    
    def _is_removable_drive(self, drive_letter: str) -> bool:
        """Check if drive is removable (USB)."""
        try:
            if platform.system() == "Windows":
                import win32api
                import win32file
                
                drive_type = win32file.GetDriveType(drive_letter)
                return drive_type == win32file.DRIVE_REMOVABLE
            else:
                # For non-Windows systems, check if it's not the system drive
                return drive_letter != "C:\\" and drive_letter != "/"
        except:
            # Fallback: assume any drive that's not C: is removable
            return drive_letter != "C:\\"
    
    def _get_drive_label(self, drive_letter: str) -> str:
        """Get the label of the drive."""
        try:
            if platform.system() == "Windows":
                import win32api
                return win32api.GetVolumeInformation(drive_letter)[0] or "USB Drive"
            else:
                return "USB Drive"
        except:
            return "USB Drive"
    
    def _is_writable(self, drive_letter: str) -> bool:
        """Check if drive is writable."""
        try:
            test_file = os.path.join(drive_letter, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except:
            return False
    
    def create_env_file_from_api_keys(self, drive_letter: str) -> bool:
        """Create a .env file on USB drive using existing API keys."""
        try:
            env_path = os.path.join(drive_letter, self.env_file_name)
            
            # Load existing API keys if available
            api_keys = self._load_existing_api_keys()
            
            # Create .env file with actual API keys or placeholders
            env_content = self._generate_env_content(api_keys)
            
            with open(env_path, 'w') as f:
                f.write(env_content)
            
            logger.info(f"Created .env file on {drive_letter} with API keys")
            return True
            
        except Exception as e:
            logger.error(f"Error creating .env file: {e}")
            return False
    
    def _load_existing_api_keys(self) -> Dict[str, str]:
        """Load existing API keys from the API key manager."""
        try:
            if os.path.exists(self.api_keys_file):
                with open(self.api_keys_file, 'r') as f:
                    encrypted_keys = json.load(f)
                
                # Decrypt and convert to .env format
                api_keys = {}
                for key_path, encrypted_value in encrypted_keys.items():
                    if encrypted_value:
                        try:
                            decrypted_value = base64.b64decode(encrypted_value.encode()).decode()
                            # Convert key path to .env format
                            env_key = self._convert_key_path_to_env(key_path)
                            if env_key:
                                api_keys[env_key] = decrypted_value
                        except:
                            continue
                
                return api_keys
            return {}
        except Exception as e:
            logger.error(f"Error loading existing API keys: {e}")
            return {}
    
    def _convert_key_path_to_env(self, key_path: str) -> Optional[str]:
        """Convert API key path to .env variable name."""
        mapping = {
            "binance.api_key": "BINANCE_API_KEY",
            "binance.secret_key": "BINANCE_SECRET_KEY",
            "binance.passphrase": "BINANCE_PASSPHRASE",
            "coinbase.api_key": "COINBASE_API_KEY",
            "coinbase.secret_key": "COINBASE_SECRET_KEY",
            "coinbase.passphrase": "COINBASE_PASSPHRASE",
            "kraken.api_key": "KRAKEN_API_KEY",
            "kraken.secret_key": "KRAKEN_SECRET_KEY",
            "kraken.passphrase": "KRAKEN_PASSPHRASE",
            "openai.api_key": "OPENAI_API_KEY",
            "anthropic.api_key": "ANTHROPIC_API_KEY",
            "google_gemini.api_key": "GOOGLE_GEMINI_API_KEY",
            "alpha_vantage.api_key": "ALPHA_VANTAGE_API_KEY",
            "polygon.api_key": "POLYGON_API_KEY",
            "news_api.api_key": "NEWS_API_KEY",
            "telegram.bot_token": "TELEGRAM_BOT_TOKEN",
            "telegram.chat_id": "TELEGRAM_CHAT_ID",
            "discord.webhook_url": "DISCORD_WEBHOOK_URL",
            "email.smtp_server": "SMTP_SERVER",
            "email.smtp_port": "SMTP_PORT",
            "email.email": "EMAIL_ADDRESS",
            "email.password": "EMAIL_PASSWORD"
        }
        return mapping.get(key_path)
    
    def _generate_env_content(self, api_keys: Dict[str, str]) -> str:
        """Generate .env file content with API keys."""
        env_content = """# Schwabot API Configuration
# This file contains your API keys for secure trading
# Keep this USB drive safe - it contains sensitive information
# Generated on: {timestamp}

# Trading Exchange API Keys
BINANCE_API_KEY={binance_api}
BINANCE_SECRET_KEY={binance_secret}
BINANCE_PASSPHRASE={binance_pass}

COINBASE_API_KEY={coinbase_api}
COINBASE_SECRET_KEY={coinbase_secret}

KRAKEN_API_KEY={kraken_api}
KRAKEN_SECRET_KEY={kraken_secret}

# AI Service API Keys
OPENAI_API_KEY={openai_api}
ANTHROPIC_API_KEY={anthropic_api}
GOOGLE_GEMINI_API_KEY={gemini_api}

# Market Data API Keys
ALPHA_VANTAGE_API_KEY={alpha_vantage_api}
POLYGON_API_KEY={polygon_api}
NEWS_API_KEY={news_api}

# Notification Service Keys
TELEGRAM_BOT_TOKEN={telegram_token}
TELEGRAM_CHAT_ID={telegram_chat}
DISCORD_WEBHOOK_URL={discord_webhook}

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_ADDRESS={email_address}
EMAIL_PASSWORD={email_password}

# Schwabot Configuration
SCHWABOT_MODE=demo
SCHWABOT_LOG_LEVEL=INFO
SCHWABOT_DATA_DIR=./data
""".format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            binance_api=api_keys.get('BINANCE_API_KEY', 'your_binance_api_key_here'),
            binance_secret=api_keys.get('BINANCE_SECRET_KEY', 'your_binance_secret_key_here'),
            binance_pass=api_keys.get('BINANCE_PASSPHRASE', 'your_binance_passphrase_here'),
            coinbase_api=api_keys.get('COINBASE_API_KEY', 'your_coinbase_api_key_here'),
            coinbase_secret=api_keys.get('COINBASE_SECRET_KEY', 'your_coinbase_secret_key_here'),
            kraken_api=api_keys.get('KRAKEN_API_KEY', 'your_kraken_api_key_here'),
            kraken_secret=api_keys.get('KRAKEN_SECRET_KEY', 'your_kraken_secret_key_here'),
            openai_api=api_keys.get('OPENAI_API_KEY', 'your_openai_api_key_here'),
            anthropic_api=api_keys.get('ANTHROPIC_API_KEY', 'your_anthropic_api_key_here'),
            gemini_api=api_keys.get('GOOGLE_GEMINI_API_KEY', 'your_google_gemini_api_key_here'),
            alpha_vantage_api=api_keys.get('ALPHA_VANTAGE_API_KEY', 'your_alpha_vantage_api_key_here'),
            polygon_api=api_keys.get('POLYGON_API_KEY', 'your_polygon_api_key_here'),
            news_api=api_keys.get('NEWS_API_KEY', 'your_news_api_key_here'),
            telegram_token=api_keys.get('TELEGRAM_BOT_TOKEN', 'your_telegram_bot_token_here'),
            telegram_chat=api_keys.get('TELEGRAM_CHAT_ID', 'your_telegram_chat_id_here'),
            discord_webhook=api_keys.get('DISCORD_WEBHOOK_URL', 'your_discord_webhook_url_here'),
            email_address=api_keys.get('EMAIL_ADDRESS', 'your_email_here'),
            email_password=api_keys.get('EMAIL_PASSWORD', 'your_app_password_here')
        )
        
        return env_content
    
    def create_env_file(self, drive_letter: str) -> bool:
        """Create a .env file on the specified USB drive."""
        return self.create_env_file_from_api_keys(drive_letter)
    
    def create_schwabot_folders(self, drive_letter: str) -> bool:
        """Create Schwabot data folders on USB drive."""
        try:
            base_path = os.path.join(drive_letter, self.schwabot_folder)
            
            # Create main folders
            folders = [
                self.backup_folder,
                self.registry_folder,
                self.memory_folder,
                "Trading_Logs",
                "Performance_Data",
                "Backtest_Results",
                "System_Backups"
            ]
            
            for folder in folders:
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                
                # Create README files
                readme_path = os.path.join(folder_path, "README.txt")
                readme_content = f"""Schwabot {folder.replace('_', ' ')} Directory
=====================================

This folder contains {folder.replace('_', ' ').lower()} for Schwabot trading system.

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Drive: {drive_letter}

Keep this USB drive secure - it contains sensitive trading data.
"""
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
            
            logger.info(f"Created Schwabot folders on {drive_letter}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Schwabot folders: {e}")
            return False
    
    def backup_trading_data(self, drive_letter: str, source_paths: List[str]) -> bool:
        """Backup trading data to USB drive."""
        try:
            backup_path = os.path.join(drive_letter, self.schwabot_folder, self.backup_folder)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for source_path in source_paths:
                if os.path.exists(source_path):
                    # Create timestamped backup
                    filename = os.path.basename(source_path)
                    backup_file = os.path.join(backup_path, f"{timestamp}_{filename}")
                    shutil.copy2(source_path, backup_file)
                    logger.info(f"Backed up {source_path} to {backup_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error backing up trading data: {e}")
            return False
    
    def load_env_from_usb(self, drive_letter: str) -> Dict[str, str]:
        """Load environment variables from USB .env file."""
        try:
            env_path = os.path.join(drive_letter, self.env_file_name)
            
            if not os.path.exists(env_path):
                logger.warning(f"No .env file found on {drive_letter}")
                return {}
            
            env_vars = {}
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
            
            logger.info(f"Loaded {len(env_vars)} environment variables from USB")
            return env_vars
            
        except Exception as e:
            logger.error(f"Error loading .env from USB: {e}")
            return {}
    
    def auto_detect_and_offer_setup(self) -> Optional[str]:
        """Automatically detect USB drives and offer setup if API keys are configured."""
        try:
            drives = self.detect_usb_drives()
            
            if not drives:
                return None
            
            # Check if API keys are configured
            api_keys_exist = os.path.exists(self.api_keys_file)
            
            if api_keys_exist:
                # Offer to deploy .env file to USB
                result = messagebox.askyesno(
                    "🔐 USB Security Setup Available",
                    f"Detected {len(drives)} USB drive(s) and existing API keys.\n\n"
                    f"Would you like to deploy your API keys to a USB drive for secure storage?\n\n"
                    f"This will:\n"
                    f"• Create a .env file with your API keys\n"
                    f"• Set up organized folders for trading data\n"
                    f"• Enable secure offloading of backtesting data\n\n"
                    f"Available drives:\n" + 
                    "\n".join([f"• {d['letter']} - {d['label']} ({d['free_gb']:.1f}GB free)" for d in drives])
                )
                
                if result:
                    return self.show_usb_setup_dialog()
            
            return None
            
        except Exception as e:
            logger.error(f"Error in auto detection: {e}")
            return None
    
    def show_usb_setup_dialog(self) -> Optional[str]:
        """Show USB setup dialog to user."""
        try:
            # Detect USB drives
            drives = self.detect_usb_drives()
            
            if not drives:
                result = messagebox.askyesno(
                    "No USB Drives Detected",
                    "No USB drives were detected.\n\n"
                    "Would you like to:\n"
                    "1. Insert a USB drive and try again\n"
                    "2. Continue without USB storage\n\n"
                    "USB storage is recommended for secure API key storage."
                )
                return None
            
            # Create setup window
            setup_window = tk.Toplevel()
            setup_window.title("🔐 Schwabot USB Security Setup")
            setup_window.geometry("700x600")
            setup_window.configure(bg='#2b2b2b')
            
            # Configure style
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='white', background='#2b2b2b')
            style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#00ff00', background='#2b2b2b')
            style.configure('Info.TLabel', font=('Arial', 10), foreground='#cccccc', background='#2b2b2b')
            
            # Title
            title_label = ttk.Label(setup_window, text="🔐 Schwabot USB Security Setup", style='Title.TLabel')
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(setup_window, 
                                 text="Select a USB drive to secure your API keys and store trading data.\n"
                                      "This will create a .env file and organize folders for secure storage.",
                                 style='Info.TLabel', justify='center')
            desc_label.pack(pady=5)
            
            # Drive selection frame
            drive_frame = ttk.Frame(setup_window)
            drive_frame.pack(fill='x', padx=20, pady=10)
            
            drive_label = ttk.Label(drive_frame, text="Available USB Drives:", style='Header.TLabel')
            drive_label.pack(anchor='w')
            
            # Drive listbox
            drive_listbox = tk.Listbox(drive_frame, height=6, bg='#404040', fg='white', selectmode='single')
            drive_listbox.pack(fill='x', pady=5)
            
            # Populate drive list
            for drive in drives:
                drive_info = f"{drive['letter']} - {drive['label']} ({drive['free_gb']:.1f}GB free)"
                if drive['has_env']:
                    drive_info += " [Has .env]"
                if drive['has_schwabot']:
                    drive_info += " [Has Schwabot]"
                drive_listbox.insert(tk.END, drive_info)
            
            # Setup options frame
            options_frame = ttk.Frame(setup_window)
            options_frame.pack(fill='x', padx=20, pady=10)
            
            # Checkboxes for setup options
            create_env_var = tk.BooleanVar(value=True)
            create_folders_var = tk.BooleanVar(value=True)
            backup_data_var = tk.BooleanVar(value=False)
            
            env_cb = tk.Checkbutton(options_frame, text="🔑 Create .env file for API keys", 
                                  variable=create_env_var, bg='#2b2b2b', fg='white', 
                                  selectcolor='#404040', font=('Arial', 11))
            env_cb.pack(anchor='w', pady=2)
            
            folders_cb = tk.Checkbutton(options_frame, text="📁 Create Schwabot data folders", 
                                      variable=create_folders_var, bg='#2b2b2b', fg='white', 
                                      selectcolor='#404040', font=('Arial', 11))
            folders_cb.pack(anchor='w', pady=2)
            
            backup_cb = tk.Checkbutton(options_frame, text="💾 Backup existing trading data", 
                                     variable=backup_data_var, bg='#2b2b2b', fg='white', 
                                     selectcolor='#404040', font=('Arial', 11))
            backup_cb.pack(anchor='w', pady=2)
            
            # Buttons frame
            button_frame = ttk.Frame(setup_window)
            button_frame.pack(pady=20)
            
            selected_drive = None
            
            def on_setup():
                nonlocal selected_drive
                
                # Get selected drive
                selection = drive_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Drive Selected", "Please select a USB drive.")
                    return
                
                selected_drive = drives[selection[0]]['letter']
                
                # Perform setup
                success = True
                messages = []
                
                if create_env_var.get():
                    if self.create_env_file(selected_drive):
                        messages.append("✅ .env file created with API keys")
                    else:
                        messages.append("❌ Failed to create .env file")
                        success = False
                
                if create_folders_var.get():
                    if self.create_schwabot_folders(selected_drive):
                        messages.append("✅ Schwabot folders created")
                    else:
                        messages.append("❌ Failed to create folders")
                        success = False
                
                if backup_data_var.get():
                    # Define common trading data paths
                    backup_paths = [
                        "AOI_Base_Files_Schwabot/config/api_keys.json",
                        "AOI_Base_Files_Schwabot/config/launcher_config.json",
                        "baseline_validation_report.txt"
                    ]
                    
                    if self.backup_trading_data(selected_drive, backup_paths):
                        messages.append("✅ Trading data backed up")
                    else:
                        messages.append("⚠️ Some data backup failed")
                
                # Show results
                if success:
                    messagebox.showinfo("USB Setup Complete", 
                                      f"USB setup completed successfully!\n\n"
                                      f"Drive: {selected_drive}\n\n"
                                      f"Setup completed:\n" + "\n".join(messages) + "\n\n"
                                      f"Your API keys and trading data are now secure on the USB drive.\n"
                                      f"You can now use this USB drive for secure trading and data offloading.")
                    setup_window.destroy()
                else:
                    messagebox.showerror("USB Setup Failed", 
                                       f"Some setup steps failed:\n\n" + "\n".join(messages))
            
            def on_cancel():
                setup_window.destroy()
            
            setup_btn = tk.Button(button_frame, text="🔐 Setup USB Drive", command=on_setup,
                                bg='#00aa00', fg='white', font=('Arial', 12, 'bold'),
                                padx=20, pady=5)
            setup_btn.pack(side='left', padx=5)
            
            cancel_btn = tk.Button(button_frame, text="❌ Cancel", command=on_cancel,
                                 bg='#aa0000', fg='white', font=('Arial', 12, 'bold'),
                                 padx=20, pady=5)
            cancel_btn.pack(side='left', padx=5)
            
            # Wait for window to close
            setup_window.wait_window()
            return selected_drive
            
        except Exception as e:
            logger.error(f"Error showing USB setup dialog: {e}")
            messagebox.showerror("Error", f"Failed to show USB setup dialog: {e}")
            return None
    
    def get_usb_status(self) -> Dict[str, Any]:
        """Get current USB status and configuration."""
        drives = self.detect_usb_drives()
        
        return {
            'drives_detected': len(drives),
            'drives': drives,
            'has_configured_drive': any(d['has_env'] for d in drives),
            'configured_drive': next((d['letter'] for d in drives if d['has_env']), None)
        }


# Global instance
usb_manager = USBManager()


def setup_usb_storage():
    """Setup USB storage for Schwabot."""
    return usb_manager.show_usb_setup_dialog()


def auto_detect_usb():
    """Automatically detect USB and offer setup."""
    return usb_manager.auto_detect_and_offer_setup()


def get_usb_status():
    """Get USB status information."""
    return usb_manager.get_usb_status()


if __name__ == "__main__":
    # Test USB detection
    drives = usb_manager.detect_usb_drives()
    print(f"Detected {len(drives)} USB drives:")
    for drive in drives:
        print(f"  {drive['letter']} - {drive['label']} ({drive['free_gb']:.1f}GB free)")
    
    # Show setup dialog
    selected = usb_manager.show_usb_setup_dialog()
    if selected:
        print(f"USB setup completed on {selected}")
    else:
        print("USB setup cancelled") 