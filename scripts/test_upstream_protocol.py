#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST UPSTREAM TIMING PROTOCOL
================================

Test script to verify the Upstream Timing Protocol is working correctly.
"""

import requests
import time
import json
from pathlib import Path

def test_upstream_protocol():
    """Test the Upstream Timing Protocol endpoints."""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Upstream Timing Protocol...")
    print("=" * 50)
    
    # Test 1: Check if Flask server is running
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Flask server is running")
        else:
            print("❌ Flask server health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Flask server: {e}")
        return
    
    # Test 2: Check upstream status
    try:
        response = requests.get(f"{base_url}/api/upstream/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Upstream status: {data['status']}")
            print(f"   Primary executor: {data.get('primary_executor', 'None')}")
            print(f"   Total nodes: {data.get('total_nodes', 0)}")
            print(f"   Online nodes: {data.get('online_nodes', 0)}")
        else:
            print("❌ Upstream status check failed")
    except Exception as e:
        print(f"❌ Upstream status error: {e}")
    
    # Test 3: Check nodes endpoint
    try:
        response = requests.get(f"{base_url}/api/upstream/nodes", timeout=5)
        if response.status_code == 200:
            nodes = response.json()
            print(f"✅ Nodes endpoint: {len(nodes)} nodes found")
            for node in nodes:
                print(f"   - {node['node_id']}: {node['role']} (Score: {node['performance_score']:.1f})")
        else:
            print("❌ Nodes endpoint failed")
    except Exception as e:
        print(f"❌ Nodes endpoint error: {e}")
    
    # Test 4: Simulate trade execution
    try:
        trade_data = {
            'strategy_hash': 'test_strategy_123',
            'trade_data': {
                'symbol': 'BTCUSDC',
                'side': 'buy',
                'amount': 0.001
            }
        }
        
        response = requests.post(f"{base_url}/api/upstream/trade/execute", 
                               json=trade_data, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Trade execution: {data['status']}")
            print(f"   Trade ID: {data.get('trade_id', 'N/A')}")
            print(f"   Target node: {data.get('target_node', 'N/A')}")
        else:
            print(f"❌ Trade execution failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Trade execution error: {e}")
    
    print("=" * 50)
    print("🧪 Test completed!")

if __name__ == "__main__":
    test_upstream_protocol() 