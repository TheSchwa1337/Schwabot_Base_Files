# core/feeds/chain_ws.py

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import websockets

logger = logging.getLogger(__name__)


@dataclass
class BlockEvent:
    """Represents a blockchain block event."""

    height: int
    hash: str
    timestamp: float
    interval: float  # Time since last block
    size: int
    weight: int
    fee_rate: float
    ts: float  # Event timestamp


class BlockFeed:
    """"""
    WebSocket-based blockchain feed for real-time block events.
    Connects to mempool.space or similar services for live block data.
    """"""

    def __init__(self, websocket_url: str = "wss://mempool.space/api/v1/ws"):
        self.websocket_url = websocket_url
        self.connected = False
        self.last_block_height = 0
        self.last_block_time = time.time()
        self.callback: Optional[Callable[[BlockEvent], None]] = None

    async def connect(self):
        """Establishes WebSocket connection to the blockchain feed."""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            self.connected = True
            logger.info(f"Connected to blockchain feed at {self.websocket_url}")

            # Subscribe to block events
            subscribe_message = {"action": "want", "data": ["blocks"]}
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info("Subscribed to block events")

        except Exception as e:
            logger.error(f"Failed to connect to blockchain feed: {e}")
            self.connected = False
            raise

    async def disconnect(self):
        """Closes the WebSocket connection."""
        if self.connected:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from blockchain feed")

    async def _process_block_message(self, message: Dict[str, Any]) -> Optional[BlockEvent]:
        """Processes incoming block message and returns BlockEvent if valid."""
        try:
            if message.get("action") == "block":
                block_data = message.get("data", {})

                height = block_data.get("height", 0)
                block_hash = block_data.get("id", "")
                block_time = block_data.get("timestamp", time.time())

                # Calculate interval since last block
                current_time = time.time()
                interval = current_time - self.last_block_time if self.last_block_time > 0 else 0

                # Create block event
                block_event = BlockEvent()
                    height=height,
                        hash=block_hash,
                            timestamp=block_time,
                            interval=interval,
                            size=block_data.get("size", 0),
                            weight=block_data.get("weight", 0),
                            fee_rate=block_data.get("fee_rate", 0.0),
                            ts=current_time,
                            )

                # Update state
                self.last_block_height = height
                self.last_block_time = current_time

                return block_event

        except Exception as e:
            logger.error(f"Error processing block message: {e}")
            return None

    async def stream_blocks(self, callback: Callable[[BlockEvent], None]):
        """"""
        Streams block events to the provided callback function.

        Args:
            callback: Function to call with each BlockEvent
        """"""
        self.callback = callback

        if not self.connected:
            await self.connect()

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    block_event = await self._process_block_message(data)

                    if block_event and self.callback:
                        await self.callback(block_event)

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON message received: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"WebSocket stream error: {e}")
            self.connected = False
        finally:
            await self.disconnect()

    async def get_latest_block(self) -> Optional[BlockEvent]:
        """Fetches the latest block information."""
        if not self.connected:
            await self.connect()

        try:
            # Request latest block info
            request_message = {"action": "get", "data": "blocks"}
            await self.websocket.send(json.dumps(request_message))

            # Wait for response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            data = json.loads(response)

            return await self._process_block_message(data)

        except Exception as e:
            logger.error(f"Error fetching latest block: {e}")
            return None


# Example usage and testing
async def test_block_feed():
    """Test function for the BlockFeed."""
    logging.basicConfig(level=logging.INFO)

    block_feed = BlockFeed()

    async def test_callback(block_event: BlockEvent):
        logger.info(f"Received block {block_event.height}: {block_event.hash[:16]}...")
        logger.info(f"  Interval: {block_event.interval:.2f}s")
        logger.info(f"  Size: {block_event.size} bytes")
        logger.info(f"  Fee rate: {block_event.fee_rate} sat/vB")

    try:
        await block_feed.stream_blocks(test_callback)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        await block_feed.disconnect()


if __name__ == "__main__":
    asyncio.run(test_block_feed())
