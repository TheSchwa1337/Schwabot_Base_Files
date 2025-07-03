#!/usr/bin/env python3
"""
Start Schwabot
Simple startup script for Schwabot Trading Dashboard and Intelligence System
"""

import os
import sys
import time
from datetime import datetime

def main():
    """Start Schwabot Trading System."""
    print("🚀 Starting Schwabot Trading System...")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import and start the main launcher
        from schwabot_main_launcher import SchwabotMainLauncher
        
        # Create launcher
        launcher = SchwabotMainLauncher()
        
        # Configuration
        config = {
            'session_id': f"schwabot_{int(time.time())}",
            'exchange_name': "coinbase",
            'sandbox_mode': True,
            'symbols': ['BTC/USDC', 'ETH/USDC', 'SOL/USDC'],
            'portfolio_value': 10000.0,
            'demo_mode': True,
            'enable_learning': True,
            'enable_automation': True
        }
        
        print("📊 Starting Schwabot Trading Dashboard...")
        print("🧠 Starting Schwabot Trading Intelligence...")
        print("🌐 Starting Web Dashboard...")
        print()
        
        # Start the system
        if launcher.start_schwabot_system(**config):
            print("✅ Schwabot Trading System Started Successfully!")
            print()
            print("🌐 Web Dashboard: http://127.0.0.1:5000")
            print(f"📊 Session ID: {config['session_id']}")
            print(f"🏦 Exchange: {config['exchange_name']}")
            print(f"📈 Trading Pairs: {', '.join(config['symbols'])}")
            print(f"💰 Portfolio Value: ${config['portfolio_value']:,.2f}")
            print()
            print("🎮 Demo Mode: Enabled")
            print("🧠 AI Learning: Enabled")
            print("🤖 Auto Trading: Enabled")
            print()
            print("🔄 System is running... Press Ctrl+C to stop")
            print("=" * 50)
            
            # Keep running
            while True:
                time.sleep(30)
                status = launcher.get_system_status()
                
                if status.get('running'):
                    print(f"📊 Status: Running | Dashboard: {status.get('dashboard_url')}")
                    
                    if 'dashboard' in status:
                        dashboard = status['dashboard']
                        print(f"💰 Portfolio: ${dashboard.get('portfolio_value', 0):,.2f} | "
                              f"Profit: ${dashboard.get('total_profit', 0):,.2f} | "
                              f"Win Rate: {dashboard.get('win_rate', 0):.1f}% | "
                              f"Active Trades: {dashboard.get('active_trades', 0)}")
                    
                    if 'intelligence' in status:
                        intelligence = status['intelligence']
                        print(f"🧠 Intelligence: {'Running' if intelligence.get('running') else 'Stopped'} | "
                              f"Learning: {'On' if intelligence.get('features_enabled', {}).get('learning') else 'Off'} | "
                              f"Automation: {'On' if intelligence.get('features_enabled', {}).get('automation') else 'Off'}")
                    print("-" * 50)
                else:
                    print("❌ System error detected")
                    break
        
        else:
            print("❌ Failed to start Schwabot Trading System")
            return 1
    
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested by user...")
    
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all required files are in the correct location.")
        return 1
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    finally:
        try:
            if 'launcher' in locals():
                launcher.stop()
            print("✅ Schwabot Trading System shutdown complete")
        except:
            pass
    
    return 0

if __name__ == "__main__":
    exit(main()) 