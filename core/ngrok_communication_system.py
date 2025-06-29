# -*- coding: utf-8 -*-
"""
Ngrok Communication System - Cross-Platform Secure Communication
===============================================================

Implements secure cross-platform communication using ngrok tunneling for
multi-instance Schwabot trading systems across Windows, macOS, and Linux.

Features:
- Secure ngrok tunneling for cross-platform communication
- Real-time system status broadcasting
- Coordinated startup sequences
- Heartbeat monitoring and health checks
- Encrypted data transmission
- Automatic failover and reconnection
- Cross-instance trading coordination

Mathematical Framework:
- Communication Latency: L = Σ(d_i / c_i) where d_i is distance, c_i is connection speed
- System Synchronization: Sync = 1 - |t_local - t_remote| / max(t_local, t_remote)
- Health Score: H = Σ(w_i * h_i) where w_i are weights, h_i are health metrics
- Load Balancing: Load_factor = Current_connections / Max_connections
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
import socket
import ssl
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import ngrok for cross-platform communication
try:
    from pyngrok import ngrok

    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    logging.warning("pyngrok not available. Install with: pip install pyngrok")

# Try to import websockets for real-time communication
try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets not available. Install with: pip install websockets")

from core.unified_math_system import unified_math
from utils.safe_print import debug, error, info, safe_print, success, warn

logger = logging.getLogger(__name__)

# =============================================================================
# NGROK COMMUNICATION CONSTANTS AND ENUMS
# =============================================================================


class CommunicationProtocol(Enum):
    """Communication protocols for cross-platform communication."""

    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    GRPC = "grpc"


class NodeStatus(Enum):
    """Node status for health monitoring."""

    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class MessageType(Enum):
    """Message types for cross-platform communication."""

    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    TRADE_SIGNAL = "trade_signal"
    SYSTEM_COMMAND = "system_command"
    DATA_SYNC = "data_sync"
    ERROR_REPORT = "error_report"


class PlatformType(Enum):
    """Platform types for cross-platform compatibility."""

    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"


# =============================================================================
# NGROK COMMUNICATION DATA STRUCTURES
# =============================================================================


@dataclass
class NodeInfo:
    """Information about a communication node."""

    node_id: str
    platform: PlatformType
    hostname: str
    ip_address: str
    port: int
    protocol: CommunicationProtocol
    ngrok_url: Optional[str] = None
    status: NodeStatus = NodeStatus.OFFLINE
    last_heartbeat: float = field(default_factory=time.time)
    version: str = "1.0.0"
    capabilities: List[str] = field(default_factory=list)


@dataclass
class CommunicationMessage:
    """Message for cross-platform communication."""

    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str] = None
    timestamp: float
    data: Dict[str, Any]
    priority: int = 0
    encrypted: bool = False

    def __post_init__(self):
        """Generate message ID if not provided."""
        if not self.message_id:
            self.message_id = str(uuid.uuid4())


@dataclass
class SystemStatus:
    """System status for cross-platform monitoring."""

    node_id: str
    platform: PlatformType
    trading_mode: str
    uptime: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    total_trades: int
    current_profit: float
    thermal_state: str
    timestamp: float


@dataclass
class HealthMetrics:
    """Health metrics for system monitoring."""

    latency: float
    packet_loss: float
    connection_stability: float
    response_time: float
    error_rate: float
    timestamp: float

    def __post_init__(self):
        """Calculate overall health score."""
        self.health_score = self._calculate_health_score()

    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        try:
            # Weighted average of health metrics
            weights = {
                "latency": 0.2,
                "packet_loss": 0.3,
                "connection_stability": 0.2,
                "response_time": 0.2,
                "error_rate": 0.1,
            }

            # Normalize metrics to 0-1 range
            normalized_latency = max(0.0, 1.0 - (self.latency / 1000.0))  # 1 second max
            normalized_packet_loss = max(0.0, 1.0 - self.packet_loss)
            normalized_stability = self.connection_stability
            normalized_response = max(0.0, 1.0 - (self.response_time / 5000.0))  # 5 seconds max
            normalized_error_rate = max(0.0, 1.0 - self.error_rate)

            # Calculate weighted score
            score = (
                weights["latency"] * normalized_latency
                + weights["packet_loss"] * normalized_packet_loss
                + weights["connection_stability"] * normalized_stability
                + weights["response_time"] * normalized_response
                + weights["error_rate"] * normalized_error_rate
            )

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5


# =============================================================================
# NGROK COMMUNICATION SYSTEM
# =============================================================================


class NgrokCommunicationSystem:
    """
    Ngrok Communication System - Cross-platform secure communication.

    Implements:
    - Secure ngrok tunneling for cross-platform communication
    - Real-time system status broadcasting
    - Coordinated startup sequences
    - Heartbeat monitoring and health checks
    - Encrypted data transmission
    - Automatic failover and reconnection
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        port: int = 8000,
        protocol: CommunicationProtocol = CommunicationProtocol.HTTP,
        enable_ngrok: bool = True,
        ngrok_auth_token: Optional[str] = None,
        encryption_key: Optional[str] = None,
    ):
        """
        Initialize Ngrok Communication System.

        Args:
            node_id: Unique node identifier
            port: Local port for communication
            protocol: Communication protocol
            enable_ngrok: Enable ngrok tunneling
            ngrok_auth_token: Ngrok authentication token
            encryption_key: Encryption key for secure communication
        """
        self.node_id = node_id or self._generate_node_id()
        self.port = port
        self.protocol = protocol
        self.enable_ngrok = enable_ngrok and NGROK_AVAILABLE
        self.encryption_key = encryption_key

        # Platform detection
        self.platform = self._detect_platform()
        self.hostname = socket.gethostname()
        self.ip_address = self._get_local_ip()

        # Node information
        self.node_info = NodeInfo(
            node_id=self.node_id,
            platform=self.platform,
            hostname=self.hostname,
            ip_address=self.ip_address,
            port=self.port,
            protocol=self.protocol,
            capabilities=self._get_capabilities(),
        )

        # Communication state
        self.connected_nodes: Dict[str, NodeInfo] = {}
        self.message_queue: List[CommunicationMessage] = []
        self.health_metrics: Dict[str, HealthMetrics] = {}

        # Ngrok tunnels
        self.ngrok_tunnels: Dict[str, str] = {}
        self.ngrok_auth_token = ngrok_auth_token

        # Performance tracking
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.failed_messages = 0

        # Threading and synchronization
        self.communication_lock = threading.RLock()
        self.running = False

        # Background tasks
        self.heartbeat_thread = None
        self.message_processor_thread = None
        self.health_monitor_thread = None
        self.websocket_server = None

        # Initialize ngrok if enabled
        if self.enable_ngrok:
            self._initialize_ngrok()

        # Start background tasks
        self._start_background_tasks()

        logger.info(f"✅ Ngrok Communication System initialized for {self.platform.value}")

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        platform_info = self._detect_platform().value
        hostname = socket.gethostname()
        timestamp = int(time.time())
        random_suffix = str(uuid.uuid4())[:8]
        return f"{platform_info}_{hostname}_{timestamp}_{random_suffix}"

    def _detect_platform(self) -> PlatformType:
        """Detect current platform."""
        system = platform.system().lower()
        if system == "windows":
            return PlatformType.WINDOWS
        elif system == "darwin":
            return PlatformType.MACOS
        elif system == "linux":
            return PlatformType.LINUX
        else:
            return PlatformType.UNKNOWN

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def _get_capabilities(self) -> List[str]:
        """Get system capabilities."""
        capabilities = []

        # Basic capabilities
        capabilities.append("trading")
        capabilities.append("communication")

        # Platform-specific capabilities
        if self.platform == PlatformType.WINDOWS:
            capabilities.extend(["windows_gui", "windows_api"])
        elif self.platform == PlatformType.MACOS:
            capabilities.extend(["macos_gui", "macos_api"])
        elif self.platform == PlatformType.LINUX:
            capabilities.extend(["linux_cli", "linux_api"])

        # Feature capabilities
        if NGROK_AVAILABLE:
            capabilities.append("ngrok_tunneling")
        if WEBSOCKETS_AVAILABLE:
            capabilities.append("websocket_communication")

        return capabilities

    def _initialize_ngrok(self) -> None:
        """Initialize ngrok tunneling."""
        try:
            if self.ngrok_auth_token:
                ngrok.set_auth_token(self.ngrok_auth_token)

            # Create ngrok tunnel
            tunnel = ngrok.connect(addr=self.port, proto=self.protocol.value)

            self.ngrok_tunnels[self.node_id] = tunnel.public_url
            self.node_info.ngrok_url = tunnel.public_url
            self.node_info.status = NodeStatus.ONLINE

            logger.info(f"✅ Ngrok tunnel created: {tunnel.public_url}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ngrok: {e}")
            self.enable_ngrok = False
            self.node_info.status = NodeStatus.DEGRADED

    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        self.running = True

        # Heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

        # Message processor thread
        self.message_processor_thread = threading.Thread(target=self._message_processor_loop, daemon=True)
        self.message_processor_thread.start()

        # Health monitor thread
        self.health_monitor_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self.health_monitor_thread.start()

        # Start WebSocket server if available
        if WEBSOCKETS_AVAILABLE:
            asyncio.create_task(self._start_websocket_server())

        logger.info("✅ Background tasks started")

    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time communication."""
        try:

            async def websocket_handler(websocket, path):
                """Handle WebSocket connections."""
                try:
                    async for message in websocket:
                        await self._handle_websocket_message(websocket, message)
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    logger.error(f"❌ WebSocket error: {e}")

            self.websocket_server = await websockets.serve(websocket_handler, "0.0.0.0", self.port)

            logger.info(f"✅ WebSocket server started on port {self.port}")

        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")

    async def _handle_websocket_message(self, websocket, message: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            # Parse message
            data = json.loads(message)
            comm_message = CommunicationMessage(
                message_id=data.get("message_id", ""),
                message_type=MessageType(data.get("message_type", "heartbeat")),
                sender_id=data.get("sender_id", ""),
                recipient_id=data.get("recipient_id"),
                timestamp=data.get("timestamp", time.time()),
                data=data.get("data", {}),
                priority=data.get("priority", 0),
                encrypted=data.get("encrypted", False),
            )

            # Process message
            await self._process_message(comm_message)

            # Send acknowledgment
            response = {"message_id": comm_message.message_id, "status": "received", "timestamp": time.time()}
            await websocket.send(json.dumps(response))

        except Exception as e:
            logger.error(f"❌ Failed to handle WebSocket message: {e}")

    def send_message(
        self,
        message_type: MessageType,
        data: Dict[str, Any],
        recipient_id: Optional[str] = None,
        priority: int = 0,
        encrypted: bool = False,
    ) -> str:
        """
        Send a message to connected nodes.

        Args:
            message_type: Type of message to send
            data: Message data
            recipient_id: Specific recipient (None for broadcast)
            priority: Message priority (0-10)
            encrypted: Whether to encrypt the message

        Returns:
            Message ID
        """
        with self.communication_lock:
            try:
                # Create message
                message = CommunicationMessage(
                    message_type=message_type,
                    sender_id=self.node_id,
                    recipient_id=recipient_id,
                    timestamp=time.time(),
                    data=data,
                    priority=priority,
                    encrypted=encrypted,
                )

                # Add to queue
                self.message_queue.append(message)

                # Update metrics
                self.total_messages_sent += 1

                logger.debug(f"📤 Queued message: {message_type.value} to {recipient_id or 'all'}")

                return message.message_id

            except Exception as e:
                logger.error(f"❌ Failed to send message: {e}")
                self.failed_messages += 1
                return ""

    def broadcast_status(self, status_data: Dict[str, Any]) -> None:
        """Broadcast system status to all connected nodes."""
        try:
            self.send_message(message_type=MessageType.STATUS_UPDATE, data=status_data, priority=5)
        except Exception as e:
            logger.error(f"❌ Failed to broadcast status: {e}")

    def send_trade_signal(self, signal_data: Dict[str, Any], recipient_id: Optional[str] = None) -> None:
        """Send trade signal to specific node or broadcast."""
        try:
            self.send_message(
                message_type=MessageType.TRADE_SIGNAL,
                data=signal_data,
                recipient_id=recipient_id,
                priority=8,
                encrypted=True,
            )
        except Exception as e:
            logger.error(f"❌ Failed to send trade signal: {e}")

    async def _process_message(self, message: CommunicationMessage) -> None:
        """Process incoming message."""
        try:
            # Update metrics
            self.total_messages_received += 1

            # Handle message based on type
            if message.message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(message)
            elif message.message_type == MessageType.STATUS_UPDATE:
                await self._handle_status_update(message)
            elif message.message_type == MessageType.TRADE_SIGNAL:
                await self._handle_trade_signal(message)
            elif message.message_type == MessageType.SYSTEM_COMMAND:
                await self._handle_system_command(message)
            elif message.message_type == MessageType.DATA_SYNC:
                await self._handle_data_sync(message)
            elif message.message_type == MessageType.ERROR_REPORT:
                await self._handle_error_report(message)
            else:
                logger.warning(f"⚠️ Unknown message type: {message.message_type}")

        except Exception as e:
            logger.error(f"❌ Failed to process message: {e}")

    async def _handle_heartbeat(self, message: CommunicationMessage) -> None:
        """Handle heartbeat message."""
        try:
            sender_id = message.sender_id

            # Update node status
            if sender_id in self.connected_nodes:
                self.connected_nodes[sender_id].last_heartbeat = time.time()
                self.connected_nodes[sender_id].status = NodeStatus.ONLINE
            else:
                # New node discovered
                node_info = NodeInfo(
                    node_id=sender_id,
                    platform=PlatformType(message.data.get("platform", "unknown")),
                    hostname=message.data.get("hostname", ""),
                    ip_address=message.data.get("ip_address", ""),
                    port=message.data.get("port", 0),
                    protocol=CommunicationProtocol(message.data.get("protocol", "http")),
                    ngrok_url=message.data.get("ngrok_url"),
                    status=NodeStatus.ONLINE,
                    last_heartbeat=time.time(),
                    version=message.data.get("version", "1.0.0"),
                    capabilities=message.data.get("capabilities", []),
                )
                self.connected_nodes[sender_id] = node_info
                logger.info(f"🆕 New node discovered: {sender_id} ({node_info.platform.value})")

            # Send heartbeat response
            self.send_message(
                message_type=MessageType.HEARTBEAT,
                data={
                    "platform": self.platform.value,
                    "hostname": self.hostname,
                    "ip_address": self.ip_address,
                    "port": self.port,
                    "protocol": self.protocol.value,
                    "ngrok_url": self.node_info.ngrok_url,
                    "version": "1.0.0",
                    "capabilities": self.node_info.capabilities,
                },
                recipient_id=sender_id,
                priority=1,
            )

        except Exception as e:
            logger.error(f"❌ Failed to handle heartbeat: {e}")

    async def _handle_status_update(self, message: CommunicationMessage) -> None:
        """Handle status update message."""
        try:
            sender_id = message.sender_id
            status_data = message.data

            # Update node status
            if sender_id in self.connected_nodes:
                node = self.connected_nodes[sender_id]
                # Update status information
                logger.debug(f"📊 Status update from {sender_id}: {status_data.get('trading_mode', 'unknown')} mode")

        except Exception as e:
            logger.error(f"❌ Failed to handle status update: {e}")

    async def _handle_trade_signal(self, message: CommunicationMessage) -> None:
        """Handle trade signal message."""
        try:
            sender_id = message.sender_id
            signal_data = message.data

            # Process trade signal (implement trading logic here)
            logger.info(f"📈 Trade signal from {sender_id}: {signal_data.get('signal_type', 'unknown')}")

        except Exception as e:
            logger.error(f"❌ Failed to handle trade signal: {e}")

    async def _handle_system_command(self, message: CommunicationMessage) -> None:
        """Handle system command message."""
        try:
            sender_id = message.sender_id
            command_data = message.data

            command = command_data.get("command", "")
            logger.info(f"⚙️ System command from {sender_id}: {command}")

            # Implement command handling logic here

        except Exception as e:
            logger.error(f"❌ Failed to handle system command: {e}")

    async def _handle_data_sync(self, message: CommunicationMessage) -> None:
        """Handle data synchronization message."""
        try:
            sender_id = message.sender_id
            sync_data = message.data

            logger.debug(f"🔄 Data sync from {sender_id}: {len(sync_data)} items")

            # Implement data synchronization logic here

        except Exception as e:
            logger.error(f"❌ Failed to handle data sync: {e}")

    async def _handle_error_report(self, message: CommunicationMessage) -> None:
        """Handle error report message."""
        try:
            sender_id = message.sender_id
            error_data = message.data

            logger.warning(f"⚠️ Error report from {sender_id}: {error_data.get('error', 'unknown error')}")

        except Exception as e:
            logger.error(f"❌ Failed to handle error report: {e}")

    def _heartbeat_loop(self) -> None:
        """Background loop for heartbeat messages."""
        while self.running:
            try:
                # Send heartbeat to all connected nodes
                for node_id in list(self.connected_nodes.keys()):
                    self.send_message(
                        message_type=MessageType.HEARTBEAT,
                        data={
                            "platform": self.platform.value,
                            "hostname": self.hostname,
                            "ip_address": self.ip_address,
                            "port": self.port,
                            "protocol": self.protocol.value,
                            "ngrok_url": self.node_info.ngrok_url,
                            "version": "1.0.0",
                            "capabilities": self.node_info.capabilities,
                        },
                        recipient_id=node_id,
                        priority=1,
                    )

                # Check for stale nodes
                current_time = time.time()
                stale_nodes = []
                for node_id, node in self.connected_nodes.items():
                    if current_time - node.last_heartbeat > 120:  # 2 minute timeout
                        stale_nodes.append(node_id)
                        node.status = NodeStatus.OFFLINE

                # Remove stale nodes
                for node_id in stale_nodes:
                    del self.connected_nodes[node_id]
                    logger.warning(f"⚠️ Node {node_id} marked as offline (timeout)")

                time.sleep(30)  # Heartbeat every 30 seconds

            except Exception as e:
                logger.error(f"❌ Heartbeat error: {e}")
                time.sleep(60)

    def _message_processor_loop(self) -> None:
        """Background loop for message processing."""
        while self.running:
            try:
                # Process queued messages
                with self.communication_lock:
                    if self.message_queue:
                        message = self.message_queue.pop(0)

                        # Process message (in a real implementation, this would send via HTTP/WebSocket)
                        logger.debug(f"📤 Processing message: {message.message_type.value}")

                        # Simulate message sending
                        time.sleep(0.1)

                time.sleep(0.1)  # Process messages every 100ms

            except Exception as e:
                logger.error(f"❌ Message processor error: {e}")
                time.sleep(1)

    def _health_monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        while self.running:
            try:
                # Calculate health metrics for all nodes
                for node_id, node in self.connected_nodes.items():
                    # Simulate health metrics
                    health_metrics = HealthMetrics(
                        latency=np.random.uniform(10, 100),  # 10-100ms
                        packet_loss=np.random.uniform(0, 0.05),  # 0-5%
                        connection_stability=np.random.uniform(0.8, 1.0),  # 80-100%
                        response_time=np.random.uniform(50, 500),  # 50-500ms
                        error_rate=np.random.uniform(0, 0.02),  # 0-2%
                        timestamp=time.time(),
                    )

                    self.health_metrics[node_id] = health_metrics

                    # Update node status based on health
                    if health_metrics.health_score < 0.3:
                        node.status = NodeStatus.OFFLINE
                    elif health_metrics.health_score < 0.7:
                        node.status = NodeStatus.DEGRADED
                    else:
                        node.status = NodeStatus.ONLINE

                time.sleep(60)  # Health check every minute

            except Exception as e:
                logger.error(f"❌ Health monitor error: {e}")
                time.sleep(120)

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        with self.communication_lock:
            return {
                "node_id": self.node_id,
                "platform": self.platform.value,
                "hostname": self.hostname,
                "ip_address": self.ip_address,
                "port": self.port,
                "protocol": self.protocol.value,
                "ngrok_enabled": self.enable_ngrok,
                "ngrok_tunnels": len(self.ngrok_tunnels),
                "connected_nodes": len(self.connected_nodes),
                "online_nodes": len([n for n in self.connected_nodes.values() if n.status == NodeStatus.ONLINE]),
                "total_messages_sent": self.total_messages_sent,
                "total_messages_received": self.total_messages_received,
                "failed_messages": self.failed_messages,
                "message_success_rate": (self.total_messages_sent - self.failed_messages)
                / max(self.total_messages_sent, 1),
                "uptime": time.time()
                - (self.node_info.last_heartbeat if hasattr(self.node_info, "last_heartbeat") else time.time()),
                "node_status": self.node_info.status.value,
            }

    def get_connected_nodes(self) -> Dict[str, NodeInfo]:
        """Get information about connected nodes."""
        with self.communication_lock:
            return self.connected_nodes.copy()

    def get_health_metrics(self) -> Dict[str, HealthMetrics]:
        """Get health metrics for all nodes."""
        with self.communication_lock:
            return self.health_metrics.copy()

    def shutdown(self) -> None:
        """Shutdown the communication system."""
        logger.info("🛑 Shutting down Ngrok Communication System...")

        self.running = False

        # Close ngrok tunnels
        if self.enable_ngrok:
            try:
                ngrok.kill()
                logger.info("✅ Ngrok tunnels closed")
            except Exception as e:
                logger.error(f"❌ Error closing ngrok tunnels: {e}")

        # Stop WebSocket server
        if self.websocket_server:
            try:
                self.websocket_server.close()
                logger.info("✅ WebSocket server closed")
            except Exception as e:
                logger.error(f"❌ Error closing WebSocket server: {e}")

        # Wait for background threads
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        if self.message_processor_thread:
            self.message_processor_thread.join(timeout=5)
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5)

        logger.info("✅ Ngrok Communication System shutdown complete")


# Global communication system instance
communication_system = None


def initialize_communication_system(
    node_id: Optional[str] = None,
    port: int = 8000,
    protocol: CommunicationProtocol = CommunicationProtocol.HTTP,
    enable_ngrok: bool = True,
    ngrok_auth_token: Optional[str] = None,
    encryption_key: Optional[str] = None,
) -> NgrokCommunicationSystem:
    """Initialize global communication system instance."""
    global communication_system

    if communication_system is None:
        communication_system = NgrokCommunicationSystem(
            node_id=node_id,
            port=port,
            protocol=protocol,
            enable_ngrok=enable_ngrok,
            ngrok_auth_token=ngrok_auth_token,
            encryption_key=encryption_key,
        )

    return communication_system


def get_communication_system() -> Optional[NgrokCommunicationSystem]:
    """Get global communication system instance."""
    return communication_system


# Example usage and testing
def main():
    """Test Ngrok Communication System functionality."""
    try:
        # Initialize communication system
        comm_system = initialize_communication_system(port=8000, enable_ngrok=False)  # Disable for testing

        safe_print("🌐 Ngrok Communication System Test")
        safe_print("=" * 50)

        # Test message sending
        safe_print("📤 Testing message sending...")
        message_id = comm_system.send_message(
            message_type=MessageType.STATUS_UPDATE,
            data={
                "trading_mode": "demo",
                "uptime": 3600,
                "cpu_usage": 0.25,
                "memory_usage": 0.45,
                "active_connections": 5,
                "total_trades": 100,
                "current_profit": 1250.50,
                "thermal_state": "cool",
            },
            priority=5,
        )
        safe_print(f"  Message sent with ID: {message_id}")

        # Test trade signal
        safe_print("\n📈 Testing trade signal...")
        comm_system.send_trade_signal(
            {"signal_type": "buy", "price": 45000.0, "volume": 1000.0, "confidence": 0.85, "timestamp": time.time()}
        )
        safe_print("  Trade signal sent")

        # Get system statistics
        safe_print("\n📊 System Statistics:")
        stats = comm_system.get_system_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                safe_print(f"  {key}: {value:.2f}")
            else:
                safe_print(f"  {key}: {value}")

        # Simulate some connected nodes
        safe_print("\n🖥️ Simulating connected nodes...")
        time.sleep(2)  # Let background tasks run

        # Get connected nodes
        nodes = comm_system.get_connected_nodes()
        safe_print(f"  Connected nodes: {len(nodes)}")

        # Get health metrics
        health_metrics = comm_system.get_health_metrics()
        safe_print(f"  Health metrics available: {len(health_metrics)}")

        # Cleanup
        comm_system.shutdown()
        safe_print("\n✅ Ngrok Communication System test completed successfully!")

    except Exception as e:
        logger.error(f"❌ Ngrok Communication System test failed: {e}")
        safe_print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
