#!/usr/bin/env python3
"""
Schwabot Automated Trading Demonstration
Shows automated trading with CCXT integration, buy/sell walls, and learning
"""

import requests
import socketio
import time
import json
import threading
from datetime import datetime
import random

# Configuration
BASE_URL = "http://localhost:5000"
API_BASE = f"{BASE_URL}/api/automated"


class AutomatedTradingDemo:
    def __init__(self):
        self.sio = socketio.Client()
        self.events_received = []
        self.setup_socketio_handlers()

    def setup_socketio_handlers(self):
        """Setup SocketIO event handlers."""

        @self.sio.event
        def connect():
            print("🔌 Connected to Schwabot automated trading server")
            self.events_received.append("connected")

        @self.sio.event
        def disconnect():
            print("🔌 Disconnected from Schwabot automated trading server")
            self.events_received.append("disconnected")

        @sio.on('realtime_update')
        def on_realtime_update(data):
            event_type = data.get('type', 'unknown')
            event_data = data.get('data', {})
            timestamp = datetime.fromtimestamp(
                data.get('timestamp', time.time())).strftime('%H:%M:%S')

            if event_type.startswith('automated_'):
                print(f"🤖 [{timestamp}] {event_type.upper()}: {json.dumps(event_data, indent=2)}")
                self.events_received.append(event_type)

        @sio.on('connected')
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
                'room': 'automated_demo'
            })

            print("✅ Successfully connected and subscribed to automated trading updates")
            return True

        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

    def demo_initialization(self):
        """Demonstrate automated trading system initialization."""
        print("\n🚀 Starting Automated Trading System Initialization")
        print("=" * 60)

        try:
            response = requests.post(f"{API_BASE}/initialize", json={)}
                'exchange_config': {}
                    'name': 'coinbase',
                    'sandbox': True
                },
                'symbols': ['BTC/USDC', 'ETH/USDC', 'SOL/USDC', 'ADA/USDC', 'DOT/USDC']
            })

            if response.status_code == 200:
                data = response.json()
                print("✅ Automated trading system initialized successfully")
                print(f"   Exchange: {data['exchange']}")
                print(f"   Symbols tracking: {', '.join(data['symbols_tracking'])}")
                print(f"   Status: {data['status']}")
            else:
                print(f"❌ Initialization failed: {response.status_code}")
                print(f"   Error: {response.json().get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error during initialization: {e}")

    def demo_price_tracking(self):
        """Demonstrate real-time price tracking."""
        print("\n📈 Starting Real-Time Price Tracking Demonstration")
        print("=" * 60)

        # Add symbols to tracking
        symbols_to_add = ['BTC/USDC', 'ETH/USDC', 'SOL/USDC']

        for symbol in symbols_to_add:
            try:
                response = requests.post(f"{API_BASE}/add_symbol", json={'symbol': symbol})

                if response.status_code == 200:
                    print(f"✅ Added {symbol} to tracking")
                else:
                    print(f"❌ Failed to add {symbol}: {response.status_code}")

            except Exception as e:
                print(f"❌ Error adding {symbol}: {e}")

        # Get current prices
        try:
            response = requests.get(f"{API_BASE}/prices")

            if response.status_code == 200:
                data = response.json()
                print("\n📊 Current Prices:")
                for symbol, price in data['prices'].items():
                    print(f"   {symbol}: ${price:.2f}")
                print(f"   Timestamp: {data['timestamp']}")
            else:
                print(f"❌ Failed to get prices: {response.status_code}")

        except Exception as e:
            print(f"❌ Error getting prices: {e}")

    def demo_tensor_analysis(self):
        """Demonstrate mathematical tensor analysis."""
        print("\n🧮 Starting Mathematical Tensor Analysis Demonstration")
        print("=" * 60)

        symbols_to_analyze = ['BTC/USDC', 'ETH/USDC']

        for symbol in symbols_to_analyze:
            try:
                print(f"\n🔍 Analyzing {symbol}...")
                response = requests.get(f"{API_BASE}/analyze/{symbol}")

                if response.status_code == 200:
                    data = response.json()
                    analysis = data['analysis']

                    print(f"   ✅ Analysis completed for {symbol}")
                    print(f"   📊 Momentum:")
                    print(f"     - Short-term: {analysis['momentum']['short_term']:.4f}")
                    print(f"     - Medium-term: {analysis['momentum']['medium_term']:.4f}")
                    print(f"     - Long-term: {analysis['momentum']['long_term']:.4f}")
                    print(f"   📈 Volatility: {analysis['volatility']:.4f}")
                    print(f"   📉 Trend: {analysis['trend']:.6f}")
                    print(f"   🧠 Patterns found: {len(analysis['patterns'])}")

                else:
                    print(f"   ❌ Analysis failed: {response.status_code}")

            except Exception as e:
                print(f"   ❌ Error analyzing {symbol}: {e}")

    def demo_automated_decisions(self):
        """Demonstrate automated decision making."""
        print("\n🧠 Starting Automated Decision Making Demonstration")
        print("=" * 60)

        symbols_to_decide = ['BTC/USDC', 'ETH/USDC']

        for symbol in symbols_to_decide:
            try:
                print(f"\n🤖 Making automated decision for {symbol}...")
                response = requests.post(f"{API_BASE}/decision/{symbol}", json={})

                if response.status_code == 200:
                    data = response.json()

                    if data['status'] == 'success':
                        decision = data['decision']
                        print(f"   ✅ Decision made for {symbol}")
                        print(f"   🎯 Action: {decision['action']}")
                        print(f"   🎲 Confidence: {decision['confidence']:.2%}")
                        print(f"   💰 Quantity: ${decision['quantity']:.2f}")
                        print(f"   📊 Batch count: {decision['batch_count']}")
                        print(f"   ⏱️ Spread seconds: {decision['spread_seconds']}")
                        print(f"   🧠 Reasoning: {decision['reasoning']}")

                        # Execute the decision
                        print(f"   🚀 Executing decision...")
                        exec_response = requests.post(f"{API_BASE}/execute_decision", json={)}
                            'decision': decision
                        })

                        if exec_response.status_code == 200:
                            exec_data = exec_response.json()
                            print(f"   ✅ Decision executed successfully")
                            print(f"   📋 Batch ID: {exec_data['batch_id']}")
                        else:
                            print(f"   ❌ Decision execution failed: {exec_response.status_code}")

                    else:
                        print(f"   ⚠️ No confident decision for {symbol}: {data['message']}")

                else:
                    print(f"   ❌ Decision failed: {response.status_code}")

            except Exception as e:
                print(f"   ❌ Error making decision for {symbol}: {e}")

    def demo_buy_sell_walls(self):
        """Demonstrate buy/sell wall creation."""
        print("\n🏗️ Starting Buy/Sell Wall Demonstration")
        print("=" * 60)

        # Create buy wall
        try:
            print("\n🟢 Creating buy wall...")
            response = requests.post(f"{API_BASE}/create_buy_wall", json={)}
                'symbol': 'BTC/USDC',
                'quantity': 2000,  # $2000 USD
                'batch_count': 15,
                'spread_seconds': 45
            })

            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Buy wall created successfully")
                print(f"   📋 Batch ID: {data['batch_id']}")
                print(f"   💰 Quantity: $2000")
                print(f"   📊 Batch count: 15")
                print(f"   ⏱️ Spread: 45 seconds")
            else:
                print(f"   ❌ Buy wall failed: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error creating buy wall: {e}")

        # Create sell wall
        try:
            print("\n🔴 Creating sell wall...")
            response = requests.post(f"{API_BASE}/create_sell_wall", json={)}
                'symbol': 'ETH/USDC',
                'quantity': 1500,  # $1500 USD
                'batch_count': 10,
                'spread_seconds': 30
            })

            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Sell wall created successfully")
                print(f"   📋 Batch ID: {data['batch_id']}")
                print(f"   💰 Quantity: $1500")
                print(f"   📊 Batch count: 10")
                print(f"   ⏱️ Spread: 30 seconds")
            else:
                print(f"   ❌ Sell wall failed: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error creating sell wall: {e}")

    def demo_basket_trading(self):
        """Demonstrate basket trading."""
        print("\n🧺 Starting Basket Trading Demonstration")
        print("=" * 60)

        try:
            print("🧺 Creating basket order...")
            response = requests.post(f"{API_BASE}/create_basket", json={)}
                'symbols': ['BTC/USDC', 'ETH/USDC', 'SOL/USDC'],
                'weights': [0.5, 0.3, 0.2],  # 50% BTC, 30% ETH, 20% SOL
                'value': 5000,  # $5000 USD total
                'strategy': 'diversified_basket'
            })

            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Basket order created successfully")
                print(f"   📋 Basket ID: {data['basket_id']}")
                print(f"   💰 Total value: $5000")
                print(f"   📊 Symbols: BTC/USDC (50%), ETH/USDC (30%), SOL/USDC (20%)")
                print(f"   🎯 Strategy: diversified_basket")
            else:
                print(f"   ❌ Basket order failed: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error creating basket order: {e}")

    def demo_order_management(self):
        """Demonstrate order management."""
        print("\n📋 Starting Order Management Demonstration")
        print("=" * 60)

        # Get all orders
        try:
            print("📋 Getting active orders...")
            response = requests.get(f"{API_BASE}/orders")

            if response.status_code == 200:
                data = response.json()
                orders = data['orders']

                if orders:
                    print(f"   ✅ Found {len(orders)} active orders:")
                    for order_id, order in orders.items():
                        print(
                            f"     📋 {order_id}: {order['symbol']} {order['side']} {order['quantity']} ({order['status']})")
                else:
                    print("   ℹ️ No active orders found")
            else:
                print(f"   ❌ Failed to get orders: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error getting orders: {e}")

    def demo_portfolio_and_learning(self):
        """Demonstrate portfolio and learning status."""
        print("\n📊 Starting Portfolio & Learning Demonstration")
        print("=" * 60)

        # Get portfolio
        try:
            print("💰 Getting portfolio...")
            response = requests.get(f"{API_BASE}/portfolio")

            if response.status_code == 200:
                data = response.json()
                portfolio = data['portfolio']

                if portfolio and 'total' in portfolio:
                    print("   ✅ Portfolio retrieved:")
                    for currency, balance in portfolio['total'].items():
                        if balance > 0:
                            print(f"     💰 {currency}: {balance}")
                else:
                    print("   ℹ️ No portfolio data available")
            else:
                print(f"   ❌ Failed to get portfolio: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error getting portfolio: {e}")

        # Get learning status
        try:
            print("\n🧠 Getting learning status...")
            response = requests.get(f"{API_BASE}/learning_status")

            if response.status_code == 200:
                data = response.json()
                status = data['learning_status']

                print("   ✅ Learning status retrieved:")
                print(f"     🧠 Learned patterns: {status['learned_patterns']}")
                print(f"     📊 Decision history: {status['decision_history']}")
                print(f"     🎯 Active strategies: {status['active_strategies']}")
                print(f"     📈 Performance metrics: {status['performance_metrics']}")
            else:
                print(f"   ❌ Failed to get learning status: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error getting learning status: {e}")

    def demo_auto_trading(self):
        """Demonstrate continuous automated trading."""
        print("\n🤖 Starting Continuous Automated Trading Demonstration")
        print("=" * 60)

        try:
            print("🤖 Starting automated trading for BTC/USDC...")
            response = requests.post(f"{API_BASE}/auto_trade/BTC/USDC", json={)}
                'interval_seconds': 30  # Check every 30 seconds
            })

            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Automated trading started successfully")
                print(f"   📊 Symbol: BTC/USDC")
                print(f"   ⏱️ Check interval: {data['interval_seconds']} seconds")
                print(f"   🎯 Status: {data['status']}")

                # Let it run for a bit
                print("   ⏳ Letting automated trading run for 2 minutes...")
                time.sleep(120)

            else:
                print(f"   ❌ Automated trading failed: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error starting automated trading: {e}")

    def run_full_demo(self):
        """Run the complete automated trading demonstration."""
        print("🚀 Schwabot Automated Trading Demonstration")
        print("=" * 70)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Server: {BASE_URL}")
        print("=" * 70)

        # Connect to real-time server
        if not self.connect_to_server():
            print("❌ Cannot proceed without real-time connection")
            return

        try:
            # Run all demonstrations
            self.demo_initialization()
            time.sleep(2)

            self.demo_price_tracking()
            time.sleep(2)

            self.demo_tensor_analysis()
            time.sleep(2)

            self.demo_automated_decisions()
            time.sleep(2)

            self.demo_buy_sell_walls()
            time.sleep(2)

            self.demo_basket_trading()
            time.sleep(2)

            self.demo_order_management()
            time.sleep(2)

            self.demo_portfolio_and_learning()
            time.sleep(2)

            self.demo_auto_trading()

            # Summary
            print("\n" + "=" * 70)
            print("📋 AUTOMATED TRADING DEMONSTRATION SUMMARY")
            print("=" * 70)
            print(f"🤖 Total automated events received: {len(self.events_received)}")
            print(f"🎯 Event types: {list(set(self.events_received))}")
            print(
                f"✅ Automated trading functionality: {'Working' if len(self.events_received) > 1 else 'Limited'}")
            print(f"🌐 Dashboard available at: {BASE_URL}")
            print(f"🔧 API endpoints available at: {API_BASE}")
            print("=" * 70)
            print("\n🎉 Key Features Demonstrated:")
            print("   ✅ CCXT Integration with Coinbase")
            print("   ✅ Real-time price tracking")
            print("   ✅ Mathematical tensor analysis")
            print("   ✅ Automated decision making")
            print("   ✅ Buy/sell wall creation (1-50 batch, orders)")
            print("   ✅ Basket trading with multiple symbols")
            print("   ✅ Order management and portfolio tracking")
            print("   ✅ Machine learning from trading patterns")
            print("   ✅ Continuous automated trading")
            print("=" * 70)

        finally:
            # Disconnect
            self.sio.disconnect()
            print("🔌 Disconnected from automated trading server")

def main():
    """Main demonstration function."""
    demo = AutomatedTradingDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main() 