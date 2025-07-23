#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NODE STARTUP SCRIPT - UPSTREAM TIMING PROTOCOL
================================================

Script to start a node with Upstream Timing Protocol monitoring.
"""

import os
import sys
import time
import uuid
from pathlib import Path

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.node_performance_client import NodePerformanceClient

def main():
    """Start node with performance monitoring."""
    
    # Generate unique node ID
    node_id = f"node_{uuid.uuid4().hex[:8]}"
    
    # Flask server URL
    flask_server_url = os.environ.get('FLASK_SERVER_URL', 'http://localhost:5000')
    
    print(f"🚀 Starting Node: {node_id}")
    print(f"🔗 Connecting to: {flask_server_url}")
    
    # Create performance client
    client = NodePerformanceClient(node_id, flask_server_url)
    
    # Register node
    if client.register_node():
        print("✅ Node registered successfully")
    else:
        print("❌ Failed to register node")
        return
    
    # Start monitoring
    client.start_monitoring()
    print("📊 Performance monitoring started")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping node...")
        client.stop_monitoring()
        print("✅ Node stopped")

if __name__ == "__main__":
    main() 