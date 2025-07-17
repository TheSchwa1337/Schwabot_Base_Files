import argparse
import asyncio
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Set

import websockets

from core.galileo_tensor_bridge import GalileoTensorBridge
from utils.logging_setup import setup_logging

#!/usr/bin/env python3
"""Tensor WebSocket Server.

Real-time WebSocket server for streaming Galileo-Tensor analysis data
to React visualization clients. Integrates with BTC price feeds and
Schwabot's trading system.
"""


# Import our core modules

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Setup logging
logger = setup_logging(__name__)


class TensorWebSocketServer:
    """WebSocket server for real-time tensor analysis streaming."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the WebSocket server."""
        self.config = config or self._default_config()

        # Initialize tensor bridge
        self.tensor_bridge = GalileoTensorBridge(self.config.get("bridge_config", {}))

        # WebSocket management
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None

        # Data streaming
        self.is_streaming = False
        self.stream_interval = self.config.get("stream_interval", 1.0)
        self.btc_price_simulator_enabled = self.config.get("btc_price_simulator", True)

        # Performance tracking
        self.total_messages_sent = 0
        self.total_connections = 0
        self.active_connections = 0

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info(
            f"🌐 Tensor WebSocket Server initialized on port {self.config.get('port', 8765)}"
        )

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            "host": "localhost",
            "port": 8765,
            "stream_interval": 1.0,
            "btc_price_simulator": True,
            "btc_price_range": (40000, 70000),
            "enable_cors": True,
            "max_clients": 100,
            "bridge_config": {
                "enable_real_time_streaming": True,
                "tensor_analysis_interval": 0.1,
            },
        }

    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new WebSocket client."""
        self.clients.add(websocket)
        await self.tensor_bridge.add_websocket_client(websocket)

        self.total_connections += 1
        self.active_connections = len(self.clients)

        logger.info(
            f"🔗 Client registered. Active connections: {self.active_connections}"
        )

        # Send initial data
        try:
            initial_data = {
                "type": "connection_established",
                "timestamp": time.time(),
                "server_info": {
                    "version": "1.0.0",
                    "features": [
                        "galileo_tensor",
                        "qss2_validation",
                        "gut_bridge",
                        "sp_integration",
                    ],
                    "stream_interval": self.stream_interval,
                },
            }
            await websocket.send(json.dumps(initial_data))
        except Exception as e:
            logger.error(f"Failed to send initial data: {e}")

    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a WebSocket client."""
        self.clients.discard(websocket)
        await self.tensor_bridge.remove_websocket_client(websocket)

        self.active_connections = len(self.clients)
        logger.info(
            f"🔗 Client unregistered. Active connections: {self.active_connections}"
        )

    async def broadcast_data(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients."""
        if not self.clients:
            return

        message = json.dumps(data, default=str)
        disconnected_clients = set()

        for client in self.clients.copy():
            try:
                await client.send(message)
                self.total_messages_sent += 1
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected_clients.add(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            await self.unregister_client(client)

    async def handle_client_message(
        self, websocket: websockets.WebSocketServerProtocol, message: str
    ):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")

            logger.info(f"📨 Received message type: {message_type}")

            if message_type == "request_analysis":
                # Client requesting specific analysis
                btc_price = data.get("btc_price", 50000.0)
                analysis_result = await self.run_analysis(btc_price)

                response = {
                    "type": "analysis_response",
                    "timestamp": time.time(),
                    "data": analysis_result,
                }
                await websocket.send(json.dumps(response, default=str))

            elif message_type == "request_history":
                # Client requesting historical data
                count = data.get("count", 50)
                history = self.tensor_bridge.get_recent_history(count)

                response = {
                    "type": "history_response",
                    "timestamp": time.time(),
                    "data": history,
                }
                await websocket.send(json.dumps(response, default=str))

            elif message_type == "update_config":
                # Client updating configuration
                new_config = data.get("config", {})
                if "stream_interval" in new_config:
                    self.stream_interval = max(0.1, new_config["stream_interval"])
                    logger.info(f"Stream interval updated to {self.stream_interval}s")

            else:
                logger.warning(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            logger.error("Invalid JSON received from client")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")

    async def run_analysis(self, btc_price: float) -> Dict[str, Any]:
        """Run tensor analysis in executor to avoid blocking."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor, self.tensor_bridge.perform_complete_analysis, btc_price
        )
        return self.tensor_bridge.get_analysis_for_react()

    def simulate_btc_price(self) -> float:
        """Simulate BTC price movement."""

        price_range = self.config.get("btc_price_range", (40000, 70000))
        base_price = (price_range[0] + price_range[1]) / 2

        # Add some realistic price movement
        current_time = time.time()
        price_wave = math.sin(current_time / 100) * 5000  # Slow wave
        price_noise = random.uniform(-1000, 1000)  # Random noise

        return max(
            price_range[0], min(price_range[1], base_price + price_wave + price_noise)
        )

    async def data_streaming_loop(self):
        """Main data streaming loop."""
        logger.info("📡 Starting data streaming loop")

        while self.is_streaming:
            try:
                if self.clients:
                    # Get or simulate BTC price
                    if self.btc_price_simulator_enabled:
                        btc_price = self.simulate_btc_price()
                    else:
                        # In a real implementation, fetch from exchange API
                        btc_price = 50000.0  # Placeholder

                    # Run analysis
                    analysis_data = await self.run_analysis(btc_price)

                    # Broadcast to clients
                    stream_data = {
                        "type": "tensor_analysis_stream",
                        "timestamp": time.time(),
                        "data": analysis_data,
                    }

                    await self.broadcast_data(stream_data)

                await asyncio.sleep(self.stream_interval)

            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                await asyncio.sleep(self.stream_interval)

    async def handle_client(
        self, websocket: websockets.WebSocketServerProtocol, path: str
    ):
        """Handle individual client connection."""
        await self.register_client(websocket)

        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self.unregister_client(websocket)

    async def start_server(self):
        """Start the WebSocket server."""
        host = self.config.get("host", "localhost")
        port = self.config.get("port", 8765)

        # Add CORS headers if enabled
        extra_headers = {}
        if self.config.get("enable_cors", True):
            extra_headers = {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*",
            }

        self.server = await websockets.serve(
            self.handle_client, host, port, extra_headers=extra_headers
        )

        logger.info(f"🚀 WebSocket server started on ws://{host}:{port}")

        # Start data streaming
        self.is_streaming = True
        streaming_task = asyncio.create_task(self.data_streaming_loop())

        return streaming_task

    async def stop_server(self):
        """Stop the WebSocket server."""
        self.is_streaming = False

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients], return_exceptions=True
            )

        self.executor.shutdown(wait=True)
        logger.info("🛑 WebSocket server stopped")

    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "total_messages_sent": self.total_messages_sent,
            "is_streaming": self.is_streaming,
            "stream_interval": self.stream_interval,
            "tensor_bridge_performance": self.tensor_bridge.get_performance_summary(),
        }


async def main():
    """Main function for running the server."""

    parser = argparse.ArgumentParser(description="Tensor WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument(
        "--stream-interval", type=float, default=1.0, help="Stream interval in seconds"
    )
    parser.add_argument(
        "--simulate-btc", action="store_true", help="Enable BTC price simulation"
    )

    args = parser.parse_args()

    config = {
        "host": args.host,
        "port": args.port,
        "stream_interval": args.stream_interval,
        "btc_price_simulator": args.simulate_btc,
    }

    server = TensorWebSocketServer(config)

    try:
        streaming_task = await server.start_server()

        # Keep server running
        await asyncio.gather(
            server.server.wait_closed(), streaming_task, return_exceptions=True
        )

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await server.stop_server()
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
