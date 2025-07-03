#!/usr/bin/env python3
import requests
import json

def test_endpoint(url, method='GET', data=None):
    try:
        if method == 'GET':
            response = requests.get(url)
        elif method == 'POST':
            response = requests.post(url, json=data)
        
        print(f"{method} {url}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
        print("-" * 50)
        return response.status_code == 200
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    base_url = "http://localhost:5000"
    
    print("Testing API endpoints...")
    print("=" * 50)
    
    # Test health endpoint
    test_endpoint(f"{base_url}/api/health")
    
    # Test live status endpoint
    test_endpoint(f"{base_url}/api/live/status")
    
    # Test trade by hash endpoint
    test_data = {
        "hash_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "strategy_name": "momentum"
    }
    test_endpoint(f"{base_url}/api/live/trade/hash", method='POST', data=test_data) 