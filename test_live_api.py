#!/usr/bin/env python3
"""
Test Live API - Test script for live trading API endpoints
"""
import requests
import json
import numpy as np
import time

BASE_URL = "http://localhost:5000/api/live"

def test_system_status():
    """Test the system status endpoint."""
    print("🔍 Testing system status...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ System status:", data)
            return True
        else:
            print(f"❌ Status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_trade_by_hash():
    """Test the trade by hash endpoint."""
    print("\n🔍 Testing trade by hash...")
    
    # Generate test hash vector
    hash_vector = np.random.rand(10).tolist()
    
    payload = {
        "hash_vector": hash_vector,
        "strategy_name": "momentum"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/trade/hash", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ Trade by hash successful:")
            print(f"   Strategy: {data['strategy_used']}")
            print(f"   Matrix file: {data['matrix_file']}")
            print(f"   Hash length: {data['hash_vector_length']}")
            return True
        else:
            print(f"❌ Trade by hash failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_matrix_matching():
    """Test the matrix matching endpoint."""
    print("\n🔍 Testing matrix matching...")
    
    # Generate test hash vector
    hash_vector = np.random.rand(10).tolist()
    
    payload = {
        "hash_vector": hash_vector,
        "threshold": 0.5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/matrix/match", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ Matrix matching successful:")
            print(f"   Matrix file: {data['matrix_file']}")
            print(f"   Threshold: {data['threshold']}")
            return True
        elif response.status_code == 404:
            data = response.json()
            print("⚠️  No matrix match found (expected for test data):")
            print(f"   Message: {data['message']}")
            return True
        else:
            print(f"❌ Matrix matching failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_visualization():
    """Test the matrix visualization endpoint."""
    print("\n🔍 Testing matrix visualization...")
    
    try:
        response = requests.get(f"{BASE_URL}/visualize/matrix")
        if response.status_code == 200:
            print("✅ Matrix visualization successful:")
            print(f"   Content-Type: {response.headers.get('Content-Type')}")
            print(f"   Content-Length: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Matrix visualization failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_full_orchestration():
    """Test the full orchestration endpoint."""
    print("\n🔍 Testing full orchestration...")
    
    payload = {
        "hash_vector": np.random.rand(10).tolist(),
        "strategy_name": "momentum"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/test/route", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ Full orchestration successful:")
            print(f"   Test type: {data['test_type']}")
            print(f"   Strategy tested: {data['strategy_tested']}")
            print(f"   Hash length: {data['test_hash_length']}")
            return True
        else:
            print(f"❌ Full orchestration failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all API tests."""
    print("🌀 Schwabot Live API Test Suite")
    print("=" * 50)
    
    tests = [
        test_system_status,
        test_trade_by_hash,
        test_matrix_matching,
        test_visualization,
        test_full_orchestration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Live API is operational.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main() 