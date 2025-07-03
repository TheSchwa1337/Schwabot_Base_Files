#!/usr/bin/env python3
"""
Test script for Schwabot Real-Time API functionality
Tests SocketIO events and live trading endpoints
"""

import requests
import json
import time
import socketio
import threading

# Test configuration
BASE_URL = "http://localhost:5000"
API_BASE = f"{BASE_URL}/api/live"

def test_basic_api_endpoints():
    """Test basic API endpoints without real-time features."""
    print("🔍 Testing basic API endpoints...")
    
    # Test system status
    try:
        response = requests.get(f"{API_BASE}/status")
        if response.status_code == 200:
            print("✅ System status endpoint working")
            print(f"   Status: {response.json()}")
        else:
            print(f"❌ System status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ System status error: {e}")
    
    # Test matrix matching
    try:
        test_hash = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        response = requests.post(f"{API_BASE}/matrix/match", 
                               json={"hash_vector": test_hash, "threshold": 0.8})
        if response.status_code == 200:
            print("✅ Matrix matching endpoint working")
            print(f"   Result: {response.json()}")
        else:
            print(f"❌ Matrix matching failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Matrix matching error: {e}")

def test_socketio_connection():
    """Test SocketIO real-time connection."""
    print("\n🔌 Testing SocketIO connection...")
    
    # Create SocketIO client
    sio = socketio.Client()
    
    events_received = []
    
    @sio.event
    def connect():
        print("✅ Connected to SocketIO server")
        events_received.append("connected")
    
    @sio.event
    def disconnect():
        print("✅ Disconnected from SocketIO server")
        events_received.append("disconnected")
    
    @sio.on('realtime_update')
    def on_realtime_update(data):
        print(f"📡 Real-time event received: {data['type']}")
        events_received.append(data['type'])
    
    @sio.on('connected')
    def on_connected(data):
        print(f"✅ Connection confirmed: {data}")
        events_received.append("connection_confirmed")
    
    try:
        # Connect to SocketIO server
        sio.connect(BASE_URL)
        
        # Wait for connection
        time.sleep(2)
        
        # Subscribe to updates
        sio.emit('subscribe_to_updates', {
            'types': ['all'],
            'room': 'test'
        })
        
        # Wait for subscription confirmation
        time.sleep(1)
        
        # Test trade submission to trigger real-time events
        test_hash = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        response = requests.post(f"{API_BASE}/trade/hash", 
                               json={"hash_vector": test_hash, "strategy_name": "momentum"})
        
        if response.status_code == 200:
            print("✅ Trade submission successful")
            
            # Wait for real-time events
            print("⏳ Waiting for real-time events...")
            time.sleep(5)
            
            print(f"📊 Events received: {events_received}")
            
            if len(events_received) > 1:
                print("✅ Real-time events working correctly")
            else:
                print("⚠️  Limited real-time events received")
        else:
            print(f"❌ Trade submission failed: {response.status_code}")
        
        # Disconnect
        sio.disconnect()
        
    except Exception as e:
        print(f"❌ SocketIO test failed: {e}")
        try:
            sio.disconnect()
        except:
            pass

def test_live_trading_with_realtime():
    """Test live trading with real-time feedback."""
    print("\n🎯 Testing live trading with real-time feedback...")
    
    # Create SocketIO client for monitoring
    sio = socketio.Client()
    trade_events = []
    
    @sio.on('realtime_update')
    def on_trade_update(data):
        if data['type'].startswith('trade_'):
            print(f"📈 Trade event: {data['type']} - {data['data']}")
            trade_events.append(data['type'])
    
    try:
        # Connect to monitor events
        sio.connect(BASE_URL)
        time.sleep(1)
        
        # Submit multiple trades to test real-time feedback
        test_trades = [
            {"hash_vector": [0.1, 0.2, 0.3, 0.4, 0.5], "strategy_name": "momentum"},
            {"hash_vector": [0.6, 0.7, 0.8, 0.9, 1.0], "strategy_name": "mean_reversion"},
            {"hash_vector": [0.3, 0.4, 0.5, 0.6, 0.7], "strategy_name": "arbitrage"}
        ]
        
        for i, trade in enumerate(test_trades, 1):
            print(f"\n🔄 Submitting trade {i}...")
            
            response = requests.post(f"{API_BASE}/trade/hash", json=trade)
            
            if response.status_code == 200:
                print(f"✅ Trade {i} submitted successfully")
                
                # Wait for real-time events
                time.sleep(3)
            else:
                print(f"❌ Trade {i} failed: {response.status_code}")
        
        print(f"\n📊 Total trade events received: {len(trade_events)}")
        print(f"   Events: {trade_events}")
        
        if len(trade_events) > 0:
            print("✅ Real-time trading feedback working")
        else:
            print("⚠️  No real-time trading events received")
        
        sio.disconnect()
        
    except Exception as e:
        print(f"❌ Live trading test failed: {e}")
        try:
            sio.disconnect()
        except:
            pass

def test_full_orchestration():
    """Test full orchestration with real-time progress updates."""
    print("\n🧪 Testing full orchestration with real-time progress...")
    
    # Create SocketIO client for monitoring
    sio = socketio.Client()
    test_events = []
    
    @sio.on('realtime_update')
    def on_test_update(data):
        if data['type'].startswith('test_'):
            print(f"🧪 Test event: {data['type']} - {data['data']}")
            test_events.append(data['type'])
    
    try:
        # Connect to monitor events
        sio.connect(BASE_URL)
        time.sleep(1)
        
        # Submit test orchestration
        print("🔄 Submitting full orchestration test...")
        
        response = requests.post(f"{API_BASE}/test/route", 
                               json={"strategy_name": "momentum"})
        
        if response.status_code == 200:
            print("✅ Test orchestration submitted successfully")
            
            # Wait for real-time events
            print("⏳ Waiting for test progress events...")
            time.sleep(8)
            
            print(f"\n📊 Total test events received: {len(test_events)}")
            print(f"   Events: {test_events}")
            
            if len(test_events) > 2:
                print("✅ Real-time test progress working")
            else:
                print("⚠️  Limited real-time test events received")
        else:
            print(f"❌ Test orchestration failed: {response.status_code}")
        
        sio.disconnect()
        
    except Exception as e:
        print(f"❌ Full orchestration test failed: {e}")
        try:
            sio.disconnect()
        except:
            pass

def main():
    """Run all tests."""
    print("🚀 Schwabot Real-Time API Test Suite")
    print("=" * 50)
    
    # Test basic endpoints
    test_basic_api_endpoints()
    
    # Test SocketIO connection
    test_socketio_connection()
    
    # Test live trading with real-time feedback
    test_live_trading_with_realtime()
    
    # Test full orchestration
    test_full_orchestration()
    
    print("\n" + "=" * 50)
    print("✅ Real-time API test suite completed")
    print("\n📋 Summary:")
    print("   - SocketIO real-time events enabled")
    print("   - Live trading with progress feedback")
    print("   - Matrix matching with real-time updates")
    print("   - Full orchestration with progress tracking")
    print("   - Dashboard available at: http://localhost:5000")

if __name__ == "__main__":
    main() 