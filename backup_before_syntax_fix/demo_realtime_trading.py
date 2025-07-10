#!/usr/bin/env python3
"""
Schwabot Real-Time Trading Demonstration
Shows live trading with real-time feedback and progress updates
"""

import json
import threading
import time
from datetime import datetime

import requests
import socketio

# Configuration
BASE_URL = "http://localhost:5000"
API_BASE = f"{BASE_URL}/api/live"


class RealTimeTradingDemo:
    def __init__(self):
        self.sio = socketio.Client()
        self.events_received = []
        self.setup_socketio_handlers()

    def setup_socketio_handlers(self):
        """Setup SocketIO event handlers."""

        @self.sio.event
        def connect():
            print("🔌 Connected to Schwabot real-time server")
            self.events_received.append("connected")

        @self.sio.event
        def disconnect():
            print("🔌 Disconnected from Schwabot real-time server")
            self.events_received.append("disconnected")

        @self.sio.on('realtime_update')
        def on_realtime_update(data):
            event_type = data.get('type', 'unknown')
            event_data = data.get('data', {})
            timestamp = datetime.fromtimestamp(
                data.get('timestamp', time.time())).strftime('%H:%M:%S')

            print(f"📡 [{timestamp}] {event_type.upper()}: {json.dumps(event_data, indent=2)}")
            self.events_received.append(event_type)

        @self.sio.on('connected')
        def on_connected(data):
            print(f"✅ Connection confirmed: {data}")
            self.events_received.append("connection_confirmed")

    def connect_to_server(self):
        """Connect to the SocketIO server."""
        try:
            self.sio.connect(BASE_URL)
            time.sleep(2)  # Wait for connection

            # Subscribe to all updates
            self.sio.emit('subscribe_to_updates', {)}
                'types': ['all'],
                'room': 'demo'
            })

            print("✅ Successfully connected and subscribed to real-time updates")
            return True

        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

    def demo_live_trading(self):
        """Demonstrate live trading with real-time feedback."""
        print("\n🎯 Starting Live Trading Demonstration")
        print("=" * 50)

        # Test different trading scenarios
        trading_scenarios = []
            {}
                "name": "Momentum Strategy",
                "hash_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "strategy": "momentum"
            },
            {}
                "name": "Mean Reversion Strategy", 
                "hash_vector": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                "strategy": "mean_reversion"
            },
            {}
                "name": "Arbitrage Strategy",
                "hash_vector": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "strategy": "arbitrage"
            }
        ]

        for i, scenario in enumerate(trading_scenarios, 1):
            print(f"\n🔄 Scenario {i}: {scenario['name']}")
            print(f"   Strategy: {scenario['strategy']}")
            print(f"   Hash Vector: {scenario['hash_vector'][:3]}...")

            try:
                # Submit trade
                response = requests.post(f"{API_BASE}/trade/hash", json={)}
                    "hash_vector": scenario['hash_vector'],
                    "strategy_name": scenario['strategy']
                })

                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Trade submitted successfully")
                    print(f"   📊 Response: {result.get('message', 'Processing...')}")
                else:
                    print(f"   ❌ Trade submission failed: {response.status_code}")

            except Exception as e:
                print(f"   ❌ Error: {e}")

            # Wait for real-time events
            print("   ⏳ Waiting for real-time events...")
            time.sleep(4)

        print(f"\n📊 Total events received: {len(self.events_received)}")
        print(f"   Event types: {list(set(self.events_received))}")

    def demo_matrix_matching(self):
        """Demonstrate matrix matching with real-time feedback."""
        print("\n🧮 Starting Matrix Matching Demonstration")
        print("=" * 50)

        test_vectors = []
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ]

        for i, vector in enumerate(test_vectors, 1):
            print(f"\n🔍 Matrix Match {i}")
            print(f"   Vector: {vector[:3]}...")

            try:
                response = requests.post(f"{API_BASE}/matrix/match", json={)}
                    "hash_vector": vector,
                    "threshold": 0.8
                })

                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Matrix matching successful")
                    print(f"   📊 Result: {result.get('status', 'Unknown')}")
                else:
                    print(f"   ❌ Matrix matching failed: {response.status_code}")

            except Exception as e:
                print(f"   ❌ Error: {e}")

            time.sleep(2)

    def demo_full_orchestration(self):
        """Demonstrate full orchestration with progress tracking."""
        print("\n🧪 Starting Full Orchestration Demonstration")
        print("=" * 50)

        print("🔄 Submitting full orchestration test...")

        try:
            response = requests.post(f"{API_BASE}/test/route", json={)}
                "strategy_name": "momentum"
            })

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Test orchestration submitted successfully")
                print(f"📊 Response: {result.get('message', 'Processing...')}")

                # Wait for progress events
                print("⏳ Waiting for progress events...")
                time.sleep(8)

            else:
                print(f"❌ Test orchestration failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def demo_system_status(self):
        """Demonstrate system status monitoring."""
        print("\n📊 Starting System Status Demonstration")
        print("=" * 50)

        try:
            response = requests.get(f"{API_BASE}/status")

            if response.status_code == 200:
                status = response.json()
                print("✅ System status retrieved successfully")
                print(f"   Status: {status.get('status', 'Unknown')}")
                print(f"   API Version: {status.get('api_version', 'Unknown')}")
                print(f"   Components: {len(status.get('components', {}))} available")

                for component, status in status.get('components', {}).items():
                    print(f"     - {component}: {status}")

            else:
                print(f"❌ Status check failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def run_full_demo(self):
        """Run the complete demonstration."""
        print("🚀 Schwabot Real-Time Trading Demonstration")
        print("=" * 60)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Server: {BASE_URL}")
        print("=" * 60)

        # Connect to real-time server
        if not self.connect_to_server():
            print("❌ Cannot proceed without real-time connection")
            return

        try:
            # Run all demonstrations
            self.demo_system_status()
            self.demo_matrix_matching()
            self.demo_live_trading()
            self.demo_full_orchestration()

            # Summary
            print("\n" + "=" * 60)
            print("📋 DEMONSTRATION SUMMARY")
            print("=" * 60)
            print(f"📡 Total real-time events received: {len(self.events_received)}")
            print(f"🎯 Event types: {list(set(self.events_received))}")
            print(
                f"✅ Real-time functionality: {'Working' if len(self.events_received) > 1 else 'Limited'}")
            print(f"🌐 Dashboard available at: {BASE_URL}")
            print(f"🔧 API endpoints available at: {API_BASE}")
            print("=" * 60)

        finally:
            # Disconnect
            self.sio.disconnect()
            print("🔌 Disconnected from real-time server")

def main():
    """Main demonstration function."""
    demo = RealTimeTradingDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main() 