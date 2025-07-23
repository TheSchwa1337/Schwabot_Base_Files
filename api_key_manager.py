#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Key Management System for Schwabot
======================================

Provides a user-friendly interface for managing all API keys used by the Schwabot system.
Each API key is clearly labeled with its purpose, usage, and configuration instructions.
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Any, Optional
import hashlib
import base64
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Comprehensive API key management system with secure storage and user-friendly interface."""
    
    def __init__(self):
        self.api_keys_file = "AOI_Base_Files_Schwabot/config/api_keys.json"
        self.api_config = self._load_api_config()
        self.encrypted_keys = self._load_encrypted_keys()
        
    def _load_api_config(self) -> Dict[str, Any]:
        """Load API configuration with descriptions and usage information."""
        return {
            "trading_exchanges": {
                "binance": {
                    "name": "Binance Exchange",
                    "description": "Primary cryptocurrency exchange for BTC/USDC trading",
                    "usage": "Used for executing buy/sell orders and fetching market data",
                    "required_fields": ["api_key", "secret_key"],
                    "optional_fields": ["passphrase"],
                    "instructions": "1. Go to Binance.com and log in\n2. Navigate to API Management\n3. Create a new API key with trading permissions\n4. Copy the API Key and Secret Key",
                    "icon": "🏦",
                    "category": "Trading"
                },
                "coinbase": {
                    "name": "Coinbase Exchange",
                    "description": "Alternative exchange for USDC-based trading",
                    "usage": "Backup exchange for trading when Binance is unavailable",
                    "required_fields": ["api_key", "secret_key"],
                    "optional_fields": ["passphrase"],
                    "instructions": "1. Go to Coinbase.com and log in\n2. Navigate to API Settings\n3. Generate new API credentials\n4. Copy the API Key and Secret Key",
                    "icon": "🪙",
                    "category": "Trading"
                },
                "kraken": {
                    "name": "Kraken Exchange",
                    "description": "High-security exchange for institutional trading",
                    "usage": "Used for advanced trading strategies and high-volume orders",
                    "required_fields": ["api_key", "secret_key"],
                    "optional_fields": ["passphrase"],
                    "instructions": "1. Go to Kraken.com and log in\n2. Navigate to Security > API\n3. Add new API key with trading permissions\n4. Copy the API Key and Secret Key",
                    "icon": "🐙",
                    "category": "Trading"
                }
            },
            "ai_services": {
                "openai": {
                    "name": "OpenAI API",
                    "description": "Advanced AI for market analysis and decision making",
                    "usage": "Used for analyzing market sentiment, generating trading insights, and pattern recognition",
                    "required_fields": ["api_key"],
                    "optional_fields": ["organization_id"],
                    "instructions": "1. Go to platform.openai.com and sign in\n2. Navigate to API Keys\n3. Create a new secret key\n4. Copy the API key (starts with 'sk-')",
                    "icon": "🤖",
                    "category": "AI Analysis"
                },
                "anthropic": {
                    "name": "Anthropic Claude",
                    "description": "AI assistant for trading strategy optimization",
                    "usage": "Used for strategy refinement, risk assessment, and market commentary",
                    "required_fields": ["api_key"],
                    "optional_fields": [],
                    "instructions": "1. Go to console.anthropic.com and sign in\n2. Navigate to API Keys\n3. Create a new API key\n4. Copy the API key (starts with 'sk-ant-')",
                    "icon": "🧠",
                    "category": "AI Analysis"
                },
                "google_gemini": {
                    "name": "Google Gemini",
                    "description": "Google's AI for market data analysis",
                    "usage": "Used for technical analysis, chart pattern recognition, and market predictions",
                    "required_fields": ["api_key"],
                    "optional_fields": [],
                    "instructions": "1. Go to makersuite.google.com/app/apikey\n2. Sign in with your Google account\n3. Create a new API key\n4. Copy the API key",
                    "icon": "💎",
                    "category": "AI Analysis"
                }
            },
            "data_services": {
                "alpha_vantage": {
                    "name": "Alpha Vantage",
                    "description": "Financial market data provider",
                    "usage": "Used for historical price data, technical indicators, and market statistics",
                    "required_fields": ["api_key"],
                    "optional_fields": [],
                    "instructions": "1. Go to alphavantage.co\n2. Sign up for a free account\n3. Get your API key from the dashboard\n4. Copy the API key",
                    "icon": "📊",
                    "category": "Market Data"
                },
                "polygon": {
                    "name": "Polygon.io",
                    "description": "Real-time market data and news",
                    "usage": "Used for real-time price feeds, news sentiment analysis, and market alerts",
                    "required_fields": ["api_key"],
                    "optional_fields": [],
                    "instructions": "1. Go to polygon.io and sign up\n2. Navigate to API Keys\n3. Generate a new API key\n4. Copy the API key",
                    "icon": "📈",
                    "category": "Market Data"
                },
                "news_api": {
                    "name": "News API",
                    "description": "Financial news and sentiment analysis",
                    "usage": "Used for news sentiment analysis and market impact assessment",
                    "required_fields": ["api_key"],
                    "optional_fields": [],
                    "instructions": "1. Go to newsapi.org and sign up\n2. Get your API key from the dashboard\n3. Copy the API key",
                    "icon": "📰",
                    "category": "Market Data"
                }
            },
            "monitoring": {
                "telegram": {
                    "name": "Telegram Bot",
                    "description": "Trading notifications and alerts",
                    "usage": "Used for sending trading alerts, portfolio updates, and system notifications",
                    "required_fields": ["bot_token", "chat_id"],
                    "optional_fields": [],
                    "instructions": "1. Message @BotFather on Telegram\n2. Create a new bot with /newbot\n3. Copy the bot token\n4. Get your chat ID by messaging @userinfobot",
                    "icon": "📱",
                    "category": "Notifications"
                },
                "discord": {
                    "name": "Discord Webhook",
                    "description": "Discord channel notifications",
                    "usage": "Used for sending trading updates to Discord channels",
                    "required_fields": ["webhook_url"],
                    "optional_fields": [],
                    "instructions": "1. Go to Discord channel settings\n2. Navigate to Integrations > Webhooks\n3. Create a new webhook\n4. Copy the webhook URL",
                    "icon": "🎮",
                    "category": "Notifications"
                },
                "email": {
                    "name": "Email Notifications",
                    "description": "Email alerts for important trading events",
                    "usage": "Used for critical alerts, daily reports, and system status updates",
                    "required_fields": ["smtp_server", "smtp_port", "email", "password"],
                    "optional_fields": ["recipient_email"],
                    "instructions": "1. Use your email provider's SMTP settings\n2. Common: Gmail (smtp.gmail.com:587)\n3. Enable 2FA and use app password\n4. Enter your email and app password",
                    "icon": "📧",
                    "category": "Notifications"
                }
            }
        }
    
    def _load_encrypted_keys(self) -> Dict[str, Any]:
        """Load encrypted API keys from file."""
        try:
            if os.path.exists(self.api_keys_file):
                with open(self.api_keys_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            return {}
    
    def _save_encrypted_keys(self):
        """Save encrypted API keys to file."""
        try:
            os.makedirs(os.path.dirname(self.api_keys_file), exist_ok=True)
            with open(self.api_keys_file, 'w') as f:
                json.dump(self.encrypted_keys, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
    
    def _encrypt_key(self, key: str) -> str:
        """Simple encryption for API keys (in production, use proper encryption)."""
        if not key:
            return ""
        # Simple base64 encoding (replace with proper encryption in production)
        return base64.b64encode(key.encode()).decode()
    
    def _decrypt_key(self, encrypted_key: str) -> str:
        """Simple decryption for API keys."""
        if not encrypted_key:
            return ""
        try:
            return base64.b64decode(encrypted_key.encode()).decode()
        except:
            return ""
    
    def show_api_configuration_gui(self):
        """Show the main API configuration GUI."""
        self.root = tk.Tk()
        self.root.title("🔑 Schwabot API Key Configuration")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='white', background='#2b2b2b')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#00ff00', background='#2b2b2b')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#cccccc', background='#2b2b2b')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TNotebook', background='#2b2b2b')
        style.configure('TNotebook.Tab', background='#404040', foreground='white', padding=[10, 5])
        
        # Main title
        title_label = ttk.Label(self.root, text="🔑 Schwabot API Key Configuration", style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Description
        desc_label = ttk.Label(self.root, 
                              text="Configure API keys for trading exchanges, AI services, and monitoring systems.\nEach section is clearly labeled with instructions on how to obtain the required keys.",
                              style='Info.TLabel', justify='center')
        desc_label.pack(pady=5)
        
        # Create notebook for different categories
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create tabs for each category
        self.create_trading_tab()
        self.create_ai_services_tab()
        self.create_data_services_tab()
        self.create_monitoring_tab()
        
        # Bottom buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        save_btn = tk.Button(button_frame, text="💾 Save All Keys", command=self.save_all_keys,
                           bg='#00aa00', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=5)
        save_btn.pack(side='left', padx=5)
        
        test_btn = tk.Button(button_frame, text="🧪 Test Connections", command=self.test_connections,
                           bg='#0066aa', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=5)
        test_btn.pack(side='left', padx=5)
        
        clear_btn = tk.Button(button_frame, text="🗑️ Clear All", command=self.clear_all_keys,
                            bg='#aa0000', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=5)
        clear_btn.pack(side='left', padx=5)
        
        self.root.mainloop()
    
    def create_trading_tab(self):
        """Create the trading exchanges configuration tab."""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text="🏦 Trading Exchanges")
        
        # Title
        title = ttk.Label(trading_frame, text="Trading Exchange API Keys", style='Header.TLabel')
        title.pack(pady=10)
        
        # Description
        desc = ttk.Label(trading_frame, 
                        text="Configure API keys for cryptocurrency exchanges.\nThese are used for executing trades and fetching market data.",
                        style='Info.TLabel')
        desc.pack(pady=5)
        
        # Create scrollable frame
        canvas = tk.Canvas(trading_frame, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(trading_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add exchange configurations
        for exchange_id, config in self.api_config["trading_exchanges"].items():
            self.create_exchange_config(scrollable_frame, exchange_id, config)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_ai_services_tab(self):
        """Create the AI services configuration tab."""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text="🤖 AI Services")
        
        title = ttk.Label(ai_frame, text="AI Service API Keys", style='Header.TLabel')
        title.pack(pady=10)
        
        desc = ttk.Label(ai_frame, 
                        text="Configure API keys for AI services.\nThese are used for market analysis, strategy optimization, and decision making.",
                        style='Info.TLabel')
        desc.pack(pady=5)
        
        canvas = tk.Canvas(ai_frame, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(ai_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for service_id, config in self.api_config["ai_services"].items():
            self.create_service_config(scrollable_frame, service_id, config)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_data_services_tab(self):
        """Create the data services configuration tab."""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 Market Data")
        
        title = ttk.Label(data_frame, text="Market Data API Keys", style='Header.TLabel')
        title.pack(pady=10)
        
        desc = ttk.Label(data_frame, 
                        text="Configure API keys for market data providers.\nThese are used for historical data, real-time feeds, and news analysis.",
                        style='Info.TLabel')
        desc.pack(pady=5)
        
        canvas = tk.Canvas(data_frame, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(data_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for service_id, config in self.api_config["data_services"].items():
            self.create_service_config(scrollable_frame, service_id, config)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_monitoring_tab(self):
        """Create the monitoring services configuration tab."""
        monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitoring_frame, text="📱 Notifications")
        
        title = ttk.Label(monitoring_frame, text="Notification Service Configuration", style='Header.TLabel')
        title.pack(pady=10)
        
        desc = ttk.Label(monitoring_frame, 
                        text="Configure notification services.\nThese are used for trading alerts, portfolio updates, and system notifications.",
                        style='Info.TLabel')
        desc.pack(pady=5)
        
        canvas = tk.Canvas(monitoring_frame, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(monitoring_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for service_id, config in self.api_config["monitoring"].items():
            self.create_service_config(scrollable_frame, service_id, config)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_exchange_config(self, parent, exchange_id: str, config: Dict[str, Any]):
        """Create configuration section for a trading exchange."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=10, pady=10)
        
        # Exchange header
        header_frame = ttk.Frame(frame)
        header_frame.pack(fill='x', pady=5)
        
        icon_label = tk.Label(header_frame, text=config['icon'], font=('Arial', 16), bg='#2b2b2b', fg='white')
        icon_label.pack(side='left', padx=5)
        
        name_label = ttk.Label(header_frame, text=config['name'], style='Header.TLabel')
        name_label.pack(side='left', padx=5)
        
        # Description
        desc_label = ttk.Label(frame, text=config['description'], style='Info.TLabel')
        desc_label.pack(anchor='w', pady=2)
        
        # Usage
        usage_label = ttk.Label(frame, text=f"Usage: {config['usage']}", style='Info.TLabel')
        usage_label.pack(anchor='w', pady=2)
        
        # Instructions
        instructions_frame = ttk.Frame(frame)
        instructions_frame.pack(fill='x', pady=5)
        
        instructions_label = ttk.Label(instructions_frame, text="📋 Setup Instructions:", style='Header.TLabel')
        instructions_label.pack(anchor='w')
        
        instructions_text = tk.Text(instructions_frame, height=4, width=80, bg='#404040', fg='white', 
                                  font=('Arial', 9), wrap='word')
        instructions_text.insert('1.0', config['instructions'])
        instructions_text.config(state='disabled')
        instructions_text.pack(fill='x', pady=2)
        
        # API Key fields
        keys_frame = ttk.Frame(frame)
        keys_frame.pack(fill='x', pady=10)
        
        self.create_key_fields(keys_frame, exchange_id, config)
    
    def create_service_config(self, parent, service_id: str, config: Dict[str, Any]):
        """Create configuration section for a service."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=10, pady=10)
        
        # Service header
        header_frame = ttk.Frame(frame)
        header_frame.pack(fill='x', pady=5)
        
        icon_label = tk.Label(header_frame, text=config['icon'], font=('Arial', 16), bg='#2b2b2b', fg='white')
        icon_label.pack(side='left', padx=5)
        
        name_label = ttk.Label(header_frame, text=config['name'], style='Header.TLabel')
        name_label.pack(side='left', padx=5)
        
        # Description
        desc_label = ttk.Label(frame, text=config['description'], style='Info.TLabel')
        desc_label.pack(anchor='w', pady=2)
        
        # Usage
        usage_label = ttk.Label(frame, text=f"Usage: {config['usage']}", style='Info.TLabel')
        usage_label.pack(anchor='w', pady=2)
        
        # Instructions
        instructions_frame = ttk.Frame(frame)
        instructions_frame.pack(fill='x', pady=5)
        
        instructions_label = ttk.Label(instructions_frame, text="📋 Setup Instructions:", style='Header.TLabel')
        instructions_label.pack(anchor='w')
        
        instructions_text = tk.Text(instructions_frame, height=3, width=80, bg='#404040', fg='white', 
                                  font=('Arial', 9), wrap='word')
        instructions_text.insert('1.0', config['instructions'])
        instructions_text.config(state='disabled')
        instructions_text.pack(fill='x', pady=2)
        
        # API Key fields
        keys_frame = ttk.Frame(frame)
        keys_frame.pack(fill='x', pady=10)
        
        self.create_key_fields(keys_frame, service_id, config)
    
    def create_key_fields(self, parent, service_id: str, config: Dict[str, Any]):
        """Create input fields for API keys."""
        # Required fields
        if config['required_fields']:
            required_label = ttk.Label(parent, text="🔴 Required Fields:", style='Header.TLabel')
            required_label.pack(anchor='w', pady=5)
            
            for field in config['required_fields']:
                self.create_field(parent, service_id, field, required=True)
        
        # Optional fields
        if config['optional_fields']:
            optional_label = ttk.Label(parent, text="🟡 Optional Fields:", style='Header.TLabel')
            optional_label.pack(anchor='w', pady=5)
            
            for field in config['optional_fields']:
                self.create_field(parent, service_id, field, required=False)
    
    def create_field(self, parent, service_id: str, field_name: str, required: bool):
        """Create a single API key input field."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        
        # Field label
        label_text = f"{field_name.replace('_', ' ').title()}:"
        if required:
            label_text += " *"
        
        label = ttk.Label(frame, text=label_text, style='Info.TLabel', width=20)
        label.pack(side='left', padx=5)
        
        # Input field
        entry = tk.Entry(frame, width=50, bg='#404040', fg='white', show='*')
        entry.pack(side='left', padx=5, fill='x', expand=True)
        
        # Show/Hide button
        show_var = tk.BooleanVar()
        show_btn = tk.Checkbutton(frame, text="👁️", variable=show_var, 
                                command=lambda: self.toggle_password_visibility(entry, show_var),
                                bg='#2b2b2b', fg='white', selectcolor='#404040')
        show_btn.pack(side='right', padx=5)
        
        # Load existing value
        key_path = f"{service_id}.{field_name}"
        if key_path in self.encrypted_keys:
            entry.insert(0, self._decrypt_key(self.encrypted_keys[key_path]))
        
        # Store reference for saving
        if not hasattr(self, 'key_entries'):
            self.key_entries = {}
        self.key_entries[key_path] = entry
    
    def toggle_password_visibility(self, entry, show_var):
        """Toggle password field visibility."""
        if show_var.get():
            entry.config(show='')
        else:
            entry.config(show='*')
    
    def save_all_keys(self):
        """Save all API keys to encrypted storage."""
        try:
            for key_path, entry in self.key_entries.items():
                value = entry.get().strip()
                if value:
                    self.encrypted_keys[key_path] = self._encrypt_key(value)
                elif key_path in self.encrypted_keys:
                    del self.encrypted_keys[key_path]
            
            self._save_encrypted_keys()
            messagebox.showinfo("Success", "✅ All API keys saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to save API keys: {e}")
    
    def test_connections(self):
        """Test API key connections."""
        messagebox.showinfo("Testing", "🧪 Testing API connections...\n\nThis feature will be implemented to verify that all API keys are working correctly.")
    
    def clear_all_keys(self):
        """Clear all API keys."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all API keys?"):
            for entry in self.key_entries.values():
                entry.delete(0, tk.END)
            self.encrypted_keys.clear()
            self._save_encrypted_keys()
            messagebox.showinfo("Cleared", "🗑️ All API keys cleared successfully!")
    
    def get_api_key(self, service_id: str, field_name: str) -> Optional[str]:
        """Get a specific API key."""
        key_path = f"{service_id}.{field_name}"
        if key_path in self.encrypted_keys:
            return self._decrypt_key(self.encrypted_keys[key_path])
        return None
    
    def has_api_key(self, service_id: str, field_name: str) -> bool:
        """Check if an API key exists."""
        return self.get_api_key(service_id, field_name) is not None


# Global instance
api_key_manager = APIKeyManager()


def show_api_configuration():
    """Show the API configuration GUI."""
    api_key_manager.show_api_configuration_gui()


if __name__ == "__main__":
    show_api_configuration() 