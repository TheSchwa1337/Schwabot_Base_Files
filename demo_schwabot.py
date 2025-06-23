#!/usr/bin/env python3
"""
Demo Script for Unified Schwabot Integration
===========================================

This script demonstrates how to interact with the unified Schwabot integration system.
It shows how to test the API endpoints, monitor the system, and interact with the AI integration.

Usage:
    python demo_schwabot.py [--host localhost] [--port 5000]
"""

import argparse
import asyncio
import json
import time
import websockets
import requests
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SchwabotDemo:
    """Demo class for interacting with the Schwabot integration system."""
    
    def __init__(self, host: str = "localhost", api_port: int = 5000, ws_port: int = 8765):
        """
        Initialize the demo.
        
        Args:
            host: API server host
            api_port: API server port
            ws_port: WebSocket server port
        """
        self.host = host
        self.api_port = api_port
        self.ws_port = ws_port
        self.api_base_url = f"http://{host}:{api_port}"
        self.ws_url = f"ws://{host}:{ws_port}"
        
        logger.info(f"🎯 Demo initialized for {self.api_base_url}")
    
    def test_api_endpoints(self) -> None:
        """Test all available API endpoints."""
        logger.info("🧪 Testing API endpoints...")
        
        endpoints = [
            ("GET", "/api/system/status", "System Status"),
            ("GET", "/api/entropy/current", "Current Entropy"),
            ("GET", "/api/entropy/history?limit=10", "Entropy History"),
            ("GET", "/api/bit-positions", "16-Bit Positions"),
            ("GET", "/api/hash-commands", "Hash Commands"),
            ("GET", "/api/ai/responses?limit=5", "AI Responses"),
            ("GET", "/api/ai/consensus", "AI Consensus"),
            ("GET", "/api/market/state", "Market State"),
        ]
        
        for method, endpoint, description in endpoints:
            try:
                url = f"{self.api_base_url}{endpoint}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ {description}: {response.status_code}")
                    print(f"   📊 {description}:")
                    print(f"      {json.dumps(data, indent=2)[:200]}...")
                else:
                    logger.warning(f"⚠️ {description}: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ {description}: {e}")
            
            time.sleep(0.5)  # Brief pause between requests
    
    def test_hash_command_registration(self) -> None:
        """Test registering a new hash command."""
        logger.info("🔧 Testing hash command registration...")
        
        try:
            # Test command data
            test_command = {
                "command_id": "demo_test_command",
                "hash_pattern": "d",
                "execution_function": "trigger_ai_analysis",
                "parameters": {
                    "analysis_type": "demo_test",
                    "priority": "high"
                },
                "priority": 9
            }
            
            url = f"{self.api_base_url}/api/hash-commands"
            response = requests.post(url, json=test_command, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Hash command registered: {data.get('message', 'Success')}")
            else:
                logger.warning(f"⚠️ Hash command registration failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Hash command registration error: {e}")
    
    def test_entropy_analytics(self) -> None:
        """Test entropy analytics endpoint."""
        logger.info("📊 Testing entropy analytics...")
        
        try:
            url = f"{self.api_base_url}/api/entropy/analytics"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Entropy analytics retrieved")
                print(f"   📈 Current Entropy: {data.get('current_entropy', 'N/A')}")
                print(f"   🎯 Threshold: {data.get('entropy_threshold', 'N/A')}")
                print(f"   📚 History Size: {data.get('position_history_size', 'N/A')}")
            else:
                logger.warning(f"⚠️ Entropy analytics failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Entropy analytics error: {e}")
    
    async def test_websocket_connection(self) -> None:
        """Test WebSocket connection and real-time updates."""
        logger.info("📡 Testing WebSocket connection...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                logger.info("✅ WebSocket connected")
                
                # Subscribe to updates
                subscribe_message = {
                    "type": "subscribe",
                    "client": "demo_client",
                    "channels": ["market_data", "entropy_updates", "ai_consensus"]
                }
                
                await websocket.send(json.dumps(subscribe_message))
                logger.info("📨 Sent subscription message")
                
                # Listen for messages for a few seconds
                timeout = 10  # seconds
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        logger.info(f"📨 Received: {data.get('type', 'unknown')}")
                        print(f"   📊 {json.dumps(data, indent=2)[:200]}...")
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"❌ WebSocket message error: {e}")
                        break
                
                logger.info("📡 WebSocket test completed")
                
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
    
    def test_system_health(self) -> None:
        """Test system health monitoring."""
        logger.info("🏥 Testing system health...")
        
        try:
            url = f"{self.api_base_url}/api/system/status"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ System health retrieved")
                
                # Print key health metrics
                status = data.get('status', 'unknown')
                uptime = data.get('uptime_seconds', 0)
                components = data.get('components', {})
                metrics = data.get('metrics', {})
                
                print(f"   🟢 Status: {status}")
                print(f"   ⏱️ Uptime: {uptime:.1f} seconds")
                print(f"   🔧 Components: {len([c for c in components.values() if c])} ready")
                print(f"   📊 Total Ticks: {metrics.get('total_ticks', 0)}")
                print(f"   🤖 AI Consensus: {metrics.get('ai_consensus_count', 0)}")
                
            else:
                logger.warning(f"⚠️ System health failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ System health error: {e}")
    
    def test_ai_consensus_summary(self) -> None:
        """Test AI consensus summary."""
        logger.info("🤖 Testing AI consensus summary...")
        
        try:
            url = f"{self.api_base_url}/api/ai/consensus"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ AI consensus summary retrieved")
                
                recent_consensus = data.get('recent_consensus', [])
                model_stats = data.get('model_agreement_stats', {})
                total_count = data.get('total_consensus_count', 0)
                
                print(f"   📈 Total Consensus Count: {total_count}")
                print(f"   📊 Recent Consensus: {len(recent_consensus)} entries")
                
                if recent_consensus:
                    latest = recent_consensus[0]
                    print(f"   🎯 Latest Action: {latest.get('consensus_action', 'N/A')}")
                    print(f"   📊 Confidence: {latest.get('confidence', 0):.2f}")
                    print(f"   🤝 Agreement Level: {latest.get('agreement_level', 0):.2f}")
                
                if model_stats:
                    print(f"   🤖 Model Stats: {len(model_stats)} models")
                    for model, stats in model_stats.items():
                        agreement_rate = stats.get('agreement_rate', 0)
                        print(f"      {model}: {agreement_rate:.2f} agreement rate")
                
            else:
                logger.warning(f"⚠️ AI consensus summary failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ AI consensus summary error: {e}")
    
    def run_comprehensive_demo(self) -> None:
        """Run a comprehensive demo of all features."""
        logger.info("🚀 Starting comprehensive Schwabot demo...")
        print("\n" + "="*60)
        print("🧠 UNIFIED SCHWABOT INTEGRATION DEMO")
        print("="*60)
        
        # Test 1: System Health
        print("\n1️⃣ Testing System Health...")
        self.test_system_health()
        
        # Test 2: API Endpoints
        print("\n2️⃣ Testing API Endpoints...")
        self.test_api_endpoints()
        
        # Test 3: Hash Command Registration
        print("\n3️⃣ Testing Hash Command Registration...")
        self.test_hash_command_registration()
        
        # Test 4: Entropy Analytics
        print("\n4️⃣ Testing Entropy Analytics...")
        self.test_entropy_analytics()
        
        # Test 5: AI Consensus Summary
        print("\n5️⃣ Testing AI Consensus Summary...")
        self.test_ai_consensus_summary()
        
        # Test 6: WebSocket Connection
        print("\n6️⃣ Testing WebSocket Connection...")
        asyncio.run(self.test_websocket_connection())
        
        print("\n" + "="*60)
        print("✅ Demo completed successfully!")
        print("="*60)
        
        # Print usage instructions
        print("\n📖 Usage Instructions:")
        print("   • API Base URL: " + self.api_base_url)
        print("   • WebSocket URL: " + self.ws_url)
        print("   • Check system health: GET /api/system/status")
        print("   • View entropy: GET /api/entropy/current")
        print("   • Monitor AI consensus: GET /api/ai/consensus")
        print("   • Real-time updates: Connect to WebSocket")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Demo script for Unified Schwabot Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_schwabot.py
  python demo_schwabot.py --host 192.168.1.100 --port 5000
  python demo_schwabot.py --api-port 5000 --ws-port 8765
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="API server host (default: localhost)"
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=5000,
        help="API server port (default: 5000)"
    )
    
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick demo (system health only)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create demo instance
        demo = SchwabotDemo(
            host=args.host,
            api_port=args.api_port,
            ws_port=args.ws_port
        )
        
        if args.quick:
            # Quick demo
            logger.info("⚡ Running quick demo...")
            demo.test_system_health()
        else:
            # Comprehensive demo
            demo.run_comprehensive_demo()
        
    except KeyboardInterrupt:
        logger.info("🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main() 