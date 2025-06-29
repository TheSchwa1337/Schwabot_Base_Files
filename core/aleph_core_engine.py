# -*- coding: utf-8 -*-
"""
ALEPH Core Engine - Advanced Logic Engine for Pattern Harmonization
===================================================================

Implements the ALEPH (Advanced Logic Engine for Pattern Harmonization) core
with BTC hashing integration, 32-bit/42-bit phase management, and cross-platform
communication through ngrok tunneling.

Mathematical Framework:
- ALEPH Trust: A_Trust(t) = sim(G_t, G_{t-n}) + NCCO_stability - Phase_dissonance
- BTC Hashing: SHA256(price + volume + phase + timestamp)
- Phase Drift: Δφ = φ_current - φ_previous
- Thermal State: T_state = f(CPU_load, GPU_load, memory_usage)
- Profit Maximization: P_max = Σ(w_i * ΔP_i) * e^(-risk_factor)

Features:
- 32-bit and 42-bit phase management
- BTC price hashing with SHA256
- Cross-platform ngrok communication
- Real-time thermal state monitoring
- Advanced profit optimization
- Entry/exit logic portalization
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

# Try to import ngrok for cross-platform communication
try:
    from pyngrok import ngrok

    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    logging.warning("pyngrok not available. Install with: pip install pyngrok")

from core.type_defs import Matrix, QuantumState, Temperature, Tensor, Vector
from core.unified_math_system import unified_math
from utils.safe_print import debug, error, info, safe_print, success, warn

logger = logging.getLogger(__name__)

# =============================================================================
# ALEPH CORE CONSTANTS AND ENUMS
# =============================================================================


class PhaseBitDepth(Enum):
    """Bit depth for phase calculations."""

    BIT_2 = 2
    BIT_4 = 4
    BIT_8 = 8
    BIT_16 = 16
    BIT_32 = 32
    BIT_42 = 42
    BIT_64 = 64


class ThermalState(Enum):
    """Thermal state for system monitoring."""

    COOL = "cool"  # Low thermal state (4-bit operations)
    WARM = "warm"  # Medium thermal state (8-bit operations)
    HOT = "hot"  # High thermal state (16-bit operations)
    CRITICAL = "critical"  # Critical thermal state (32-bit operations)


class TradingMode(Enum):
    """Trading mode for the system."""

    DEMO = "demo"
    LIVE = "live"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


class CommunicationProtocol(Enum):
    """Communication protocol for cross-platform communication."""

    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    GRPC = "grpc"


# =============================================================================
# ALEPH CORE DATA STRUCTURES
# =============================================================================


@dataclass
class BTCHashData:
    """BTC hashing data structure."""

    price: float
    volume: float
    phase: int
    timestamp: float
    hash_value: str
    hash_entropy: float
    thermal_state: ThermalState
    bit_depth: PhaseBitDepth

    def __post_init__(self):
        """Generate hash value after initialization."""
        if not self.hash_value:
            self.hash_value = self._generate_hash()
            self.hash_entropy = self._calculate_entropy()

    def _generate_hash(self) -> str:
        """Generate SHA256 hash from BTC data."""
        data_string = f"{self.price:.8f}_{self.volume:.8f}_{self.phase}_{self.timestamp:.8f}"
        return hashlib.sha256(data_string.encode()).hexdigest()

    def _calculate_entropy(self) -> float:
        """Calculate entropy from hash value."""
        hash_bytes = bytes.fromhex(self.hash_value)
        byte_counts = [0] * 256
        for byte in hash_bytes:
            byte_counts[byte] += 1

        entropy = 0.0
        total_bytes = len(hash_bytes)
        for count in byte_counts:
            if count > 0:
                probability = count / total_bytes
                entropy -= probability * np.log2(probability)

        return entropy


@dataclass
class PhaseState:
    """Phase state for bit-level operations."""

    bit_depth: PhaseBitDepth
    phase_value: int
    phase_angle: float
    coherence: float
    stability: float
    timestamp: float

    def __post_init__(self):
        """Calculate derived values."""
        self.phase_angle = (self.phase_value / (2**self.bit_depth.value)) * 2 * np.pi
        self.coherence = self._calculate_coherence()
        self.stability = self._calculate_stability()

    def _calculate_coherence(self) -> float:
        """Calculate phase coherence."""
        return np.cos(self.phase_angle) ** 2

    def _calculate_stability(self) -> float:
        """Calculate phase stability."""
        return 1.0 - abs(np.sin(self.phase_angle))


@dataclass
class ThermalMetrics:
    """Thermal metrics for system monitoring."""

    cpu_usage: float
    gpu_usage: float
    memory_usage: float
    temperature: float
    thermal_state: ThermalState
    timestamp: float

    def __post_init__(self):
        """Determine thermal state based on metrics."""
        self.thermal_state = self._determine_thermal_state()

    def _determine_thermal_state(self) -> ThermalState:
        """Determine thermal state based on system metrics."""
        max_usage = max(self.cpu_usage, self.gpu_usage, self.memory_usage)

        if max_usage < 0.3:
            return ThermalState.COOL
        elif max_usage < 0.6:
            return ThermalState.WARM
        elif max_usage < 0.8:
            return ThermalState.HOT
        else:
            return ThermalState.CRITICAL


@dataclass
class ProfitOptimization:
    """Profit optimization parameters."""

    entry_threshold: float
    exit_threshold: float
    risk_factor: float
    profit_weight: float
    confidence_score: float
    timestamp: float


@dataclass
class CommunicationNode:
    """Communication node for cross-platform communication."""

    node_id: str
    platform: str
    ip_address: str
    port: int
    protocol: CommunicationProtocol
    ngrok_url: Optional[str] = None
    is_connected: bool = False
    last_heartbeat: float = field(default_factory=time.time)


# =============================================================================
# ALEPH CORE ENGINE
# =============================================================================


class ALEPHCoreEngine:
    """
    ALEPH Core Engine - Advanced Logic Engine for Pattern Harmonization.

    Implements:
    - BTC price hashing with SHA256
    - 32-bit and 42-bit phase management
    - Cross-platform communication via ngrok
    - Real-time thermal state monitoring
    - Advanced profit optimization
    - Entry/exit logic portalization
    """

    def __init__(
        self,
        trading_mode: TradingMode = TradingMode.DEMO,
        bit_depth: PhaseBitDepth = PhaseBitDepth.BIT_32,
        enable_ngrok: bool = True,
        ngrok_auth_token: Optional[str] = None,
    ):
        """
        Initialize ALEPH Core Engine.

        Args:
            trading_mode: Trading mode for the system
            bit_depth: Bit depth for phase calculations
            enable_ngrok: Enable ngrok for cross-platform communication
            ngrok_auth_token: Ngrok authentication token
        """
        self.trading_mode = trading_mode
        self.bit_depth = bit_depth
        self.enable_ngrok = enable_ngrok and NGROK_AVAILABLE

        # Initialize core components
        self.btc_hash_history: List[BTCHashData] = []
        self.phase_history: List[PhaseState] = []
        self.thermal_history: List[ThermalMetrics] = []
        self.profit_history: List[ProfitOptimization] = []

        # Communication nodes
        self.communication_nodes: Dict[str, CommunicationNode] = {}
        self.ngrok_tunnels: Dict[str, str] = {}

        # Performance tracking
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0

        # Threading and synchronization
        self.engine_lock = threading.RLock()
        self.running = False

        # Background tasks
        self.thermal_monitor_thread = None
        self.communication_thread = None
        self.profit_optimization_thread = None

        # Initialize ngrok if enabled
        if self.enable_ngrok:
            self._initialize_ngrok(ngrok_auth_token)

        # Start background tasks
        self._start_background_tasks()

        logger.info(f"✅ ALEPH Core Engine initialized in {trading_mode.value} mode with {bit_depth.value}-bit depth")

    def _initialize_ngrok(self, auth_token: Optional[str] = None) -> None:
        """Initialize ngrok for cross-platform communication."""
        try:
            if auth_token:
                ngrok.set_auth_token(auth_token)

            # Create ngrok tunnel for this node
            node_id = self._generate_node_id()
            tunnel = ngrok.connect(addr=8000, proto="http")

            self.ngrok_tunnels[node_id] = tunnel.public_url
            logger.info(f"✅ Ngrok tunnel created: {tunnel.public_url}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ngrok: {e}")
            self.enable_ngrok = False

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        platform_info = platform.system().lower()
        hostname = socket.gethostname()
        timestamp = int(time.time())
        return f"{platform_info}_{hostname}_{timestamp}"

    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        self.running = True

        # Thermal monitoring thread
        self.thermal_monitor_thread = threading.Thread(target=self._thermal_monitor_loop, daemon=True)
        self.thermal_monitor_thread.start()

        # Communication thread
        self.communication_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.communication_thread.start()

        # Profit optimization thread
        self.profit_optimization_thread = threading.Thread(target=self._profit_optimization_loop, daemon=True)
        self.profit_optimization_thread.start()

        logger.info("✅ Background tasks started")

    def process_btc_data(self, price: float, volume: float, phase: Optional[int] = None) -> BTCHashData:
        """
        Process BTC data and generate hash.

        Args:
            price: BTC price
            volume: Trading volume
            phase: Optional phase value (auto-generated if None)

        Returns:
            BTCHashData with hash and entropy
        """
        with self.engine_lock:
            try:
                # Generate phase if not provided
                if phase is None:
                    phase = self._generate_phase_value()

                # Create BTC hash data
                btc_data = BTCHashData(
                    price=price,
                    volume=volume,
                    phase=phase,
                    timestamp=time.time(),
                    hash_value="",  # Will be generated in __post_init__
                    hash_entropy=0.0,  # Will be calculated in __post_init__
                    thermal_state=ThermalState.COOL,  # Will be updated
                    bit_depth=self.bit_depth,
                )

                # Update thermal state
                thermal_metrics = self._get_thermal_metrics()
                btc_data.thermal_state = thermal_metrics.thermal_state

                # Store in history
                self.btc_hash_history.append(btc_data)
                if len(self.btc_hash_history) > 10000:
                    self.btc_hash_history = self.btc_hash_history[-10000:]

                # Update performance metrics
                self.total_operations += 1
                self.successful_operations += 1

                logger.debug(
                    f"✅ Processed BTC data: price={price}, volume={volume}, phase={phase}, hash={btc_data.hash_value[:16]}..."
                )

                return btc_data

            except Exception as e:
                self.failed_operations += 1
                logger.error(f"❌ Failed to process BTC data: {e}")
                raise

    def _generate_phase_value(self) -> int:
        """Generate phase value based on bit depth."""
        max_value = 2**self.bit_depth.value
        return int(time.time() * 1000) % max_value

    def _get_thermal_metrics(self) -> ThermalMetrics:
        """Get current thermal metrics."""
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent

            # Try to get GPU usage (platform dependent)
            gpu_usage = 0.0
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except ImportError:
                pass

            # Estimate temperature (platform dependent)
            temperature = 25.0 + (cpu_usage * 0.5)  # Rough estimation

            return ThermalMetrics(
                cpu_usage=cpu_usage / 100.0,
                gpu_usage=gpu_usage / 100.0,
                memory_usage=memory_usage / 100.0,
                temperature=temperature,
                thermal_state=ThermalState.COOL,  # Will be determined in __post_init__
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"❌ Failed to get thermal metrics: {e}")
            return ThermalMetrics(
                cpu_usage=0.0,
                gpu_usage=0.0,
                memory_usage=0.0,
                temperature=25.0,
                thermal_state=ThermalState.COOL,
                timestamp=time.time(),
            )

    def calculate_aleph_trust(self, current_glyph: str, historical_glyphs: List[str] = None) -> float:
        """
        Calculate ALEPH trust score.

        ALEPH Trust Formula: A_Trust(t) = sim(G_t, G_{t-n}) + NCCO_stability - Phase_dissonance

        Args:
            current_glyph: Current glyph
            historical_glyphs: Historical glyphs for comparison

        Returns:
            ALEPH trust score (0.0 to 1.0)
        """
        try:
            # Calculate glyph similarity
            if historical_glyphs:
                similarities = []
                for hist_glyph in historical_glyphs[-10:]:  # Last 10 glyphs
                    similarity = self._calculate_glyph_similarity(current_glyph, hist_glyph)
                    similarities.append(similarity)
                glyph_similarity = np.mean(similarities) if similarities else 0.5
            else:
                glyph_similarity = 0.5

            # Calculate NCCO stability
            ncco_stability = self._calculate_ncco_stability()

            # Calculate phase dissonance
            phase_dissonance = self._calculate_phase_dissonance()

            # Calculate ALEPH trust
            aleph_trust = glyph_similarity + ncco_stability - phase_dissonance

            # Clamp to [0, 1] range
            aleph_trust = max(0.0, min(1.0, aleph_trust))

            return aleph_trust

        except Exception as e:
            logger.error(f"❌ Failed to calculate ALEPH trust: {e}")
            return 0.5

    def _calculate_glyph_similarity(self, glyph1: str, glyph2: str) -> float:
        """Calculate similarity between two glyphs."""
        try:
            # Simple hash-based similarity
            hash1 = hashlib.sha256(glyph1.encode()).hexdigest()[:8]
            hash2 = hashlib.sha256(glyph2.encode()).hexdigest()[:8]

            matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
            return matches / 8.0

        except Exception:
            return 0.5

    def _calculate_ncco_stability(self) -> float:
        """Calculate NCCO (Network Coherence and Coordination) stability."""
        try:
            if not self.btc_hash_history:
                return 0.5

            # Calculate entropy stability from recent BTC hashes
            recent_entropies = [data.hash_entropy for data in self.btc_hash_history[-100:]]
            if len(recent_entropies) < 2:
                return 0.5

            # Calculate entropy variance (lower variance = higher stability)
            entropy_variance = np.var(recent_entropies)
            max_entropy = 8.0  # Maximum entropy for SHA256 (256 bits)

            # Convert variance to stability (0 to 1)
            stability = 1.0 - min(1.0, entropy_variance / max_entropy)

            return stability

        except Exception:
            return 0.5

    def _calculate_phase_dissonance(self) -> float:
        """Calculate phase dissonance."""
        try:
            if not self.phase_history:
                return 0.0

            # Calculate phase differences
            recent_phases = self.phase_history[-10:]
            if len(recent_phases) < 2:
                return 0.0

            phase_differences = []
            for i in range(1, len(recent_phases)):
                diff = abs(recent_phases[i].phase_angle - recent_phases[i - 1].phase_angle)
                # Normalize to [0, π]
                diff = min(diff, 2 * np.pi - diff)
                phase_differences.append(diff)

            # Calculate average dissonance
            if phase_differences:
                avg_dissonance = np.mean(phase_differences) / np.pi
                return avg_dissonance

            return 0.0

        except Exception:
            return 0.0

    def optimize_profit_strategy(
        self, current_price: float, entry_price: float, volume: float, risk_tolerance: float = 0.02
    ) -> ProfitOptimization:
        """
        Optimize profit strategy using ALEPH logic.

        Args:
            current_price: Current BTC price
            entry_price: Entry price
            volume: Trading volume
            risk_tolerance: Risk tolerance (0.0 to 1.0)

        Returns:
            ProfitOptimization with optimized parameters
        """
        try:
            # Calculate basic profit metrics
            price_change = (current_price - entry_price) / entry_price
            profit_potential = max(0.0, price_change)

            # Calculate ALEPH trust for confidence
            current_glyph = f"BTC_{current_price:.2f}_{volume:.2f}"
            aleph_trust = self.calculate_aleph_trust(current_glyph)

            # Calculate risk-adjusted parameters
            risk_factor = risk_tolerance * (1.0 - aleph_trust)
            entry_threshold = profit_potential * aleph_trust
            exit_threshold = entry_threshold * 1.5  # 50% profit target

            # Calculate profit weight based on volume and confidence
            profit_weight = min(1.0, volume / 1000.0) * aleph_trust

            # Create optimization result
            optimization = ProfitOptimization(
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                risk_factor=risk_factor,
                profit_weight=profit_weight,
                confidence_score=aleph_trust,
                timestamp=time.time(),
            )

            # Store in history
            self.profit_history.append(optimization)
            if len(self.profit_history) > 1000:
                self.profit_history = self.profit_history[-1000:]

            return optimization

        except Exception as e:
            logger.error(f"❌ Failed to optimize profit strategy: {e}")
            return ProfitOptimization(
                entry_threshold=0.0,
                exit_threshold=0.0,
                risk_factor=risk_tolerance,
                profit_weight=0.0,
                confidence_score=0.0,
                timestamp=time.time(),
            )

    def _thermal_monitor_loop(self) -> None:
        """Background loop for thermal monitoring."""
        while self.running:
            try:
                thermal_metrics = self._get_thermal_metrics()
                self.thermal_history.append(thermal_metrics)

                # Keep only last 1000 entries
                if len(self.thermal_history) > 1000:
                    self.thermal_history = self.thermal_history[-1000:]

                # Log thermal state changes
                if len(self.thermal_history) > 1:
                    prev_state = self.thermal_history[-2].thermal_state
                    current_state = thermal_metrics.thermal_state
                    if prev_state != current_state:
                        logger.info(f"🌡️ Thermal state changed: {prev_state.value} → {current_state.value}")

                time.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                logger.error(f"❌ Thermal monitoring error: {e}")
                time.sleep(10)

    def _communication_loop(self) -> None:
        """Background loop for cross-platform communication."""
        while self.running:
            try:
                # Update node heartbeats
                current_time = time.time()
                for node_id, node in self.communication_nodes.items():
                    if current_time - node.last_heartbeat > 60:  # 60 second timeout
                        node.is_connected = False
                        logger.warning(f"⚠️ Node {node_id} disconnected (timeout)")

                # Broadcast system status
                if self.enable_ngrok and self.ngrok_tunnels:
                    self._broadcast_system_status()

                time.sleep(30)  # Communication every 30 seconds

            except Exception as e:
                logger.error(f"❌ Communication error: {e}")
                time.sleep(60)

    def _profit_optimization_loop(self) -> None:
        """Background loop for profit optimization."""
        while self.running:
            try:
                # Perform periodic profit optimization
                if self.btc_hash_history and self.profit_history:
                    latest_btc = self.btc_hash_history[-1]
                    latest_profit = self.profit_history[-1]

                    # Re-optimize if conditions changed significantly
                    time_diff = time.time() - latest_profit.timestamp
                    if time_diff > 300:  # 5 minutes
                        self.optimize_profit_strategy(
                            current_price=latest_btc.price,
                            entry_price=latest_btc.price * 0.99,  # Example entry price
                            volume=latest_btc.volume,
                            risk_tolerance=0.02,
                        )

                time.sleep(60)  # Optimize every minute

            except Exception as e:
                logger.error(f"❌ Profit optimization error: {e}")
                time.sleep(120)

    def _broadcast_system_status(self) -> None:
        """Broadcast system status to connected nodes."""
        try:
            status = {
                "node_id": self._generate_node_id(),
                "platform": platform.system(),
                "trading_mode": self.trading_mode.value,
                "bit_depth": self.bit_depth.value,
                "total_operations": self.total_operations,
                "success_rate": self.successful_operations / max(self.total_operations, 1),
                "thermal_state": self.thermal_history[-1].thermal_state.value if self.thermal_history else "unknown",
                "timestamp": time.time(),
            }

            # Broadcast to all connected nodes
            for node_id, node in self.communication_nodes.items():
                if node.is_connected:
                    # In a real implementation, this would send via HTTP/WebSocket
                    logger.debug(f"📡 Broadcasting status to {node_id}")

        except Exception as e:
            logger.error(f"❌ Failed to broadcast status: {e}")

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        with self.engine_lock:
            return {
                "trading_mode": self.trading_mode.value,
                "bit_depth": self.bit_depth.value,
                "total_operations": self.total_operations,
                "successful_operations": self.successful_operations,
                "failed_operations": self.failed_operations,
                "success_rate": self.successful_operations / max(self.total_operations, 1),
                "btc_hash_count": len(self.btc_hash_history),
                "phase_count": len(self.phase_history),
                "thermal_count": len(self.thermal_history),
                "profit_count": len(self.profit_history),
                "connected_nodes": len([n for n in self.communication_nodes.values() if n.is_connected]),
                "ngrok_tunnels": len(self.ngrok_tunnels),
                "current_thermal_state": (
                    self.thermal_history[-1].thermal_state.value if self.thermal_history else "unknown"
                ),
                "uptime": time.time() - (self.thermal_history[0].timestamp if self.thermal_history else time.time()),
            }

    def shutdown(self) -> None:
        """Shutdown the ALEPH core engine."""
        logger.info("🛑 Shutting down ALEPH Core Engine...")

        self.running = False

        # Close ngrok tunnels
        if self.enable_ngrok:
            try:
                ngrok.kill()
                logger.info("✅ Ngrok tunnels closed")
            except Exception as e:
                logger.error(f"❌ Error closing ngrok tunnels: {e}")

        # Wait for background threads
        if self.thermal_monitor_thread:
            self.thermal_monitor_thread.join(timeout=5)
        if self.communication_thread:
            self.communication_thread.join(timeout=5)
        if self.profit_optimization_thread:
            self.profit_optimization_thread.join(timeout=5)

        logger.info("✅ ALEPH Core Engine shutdown complete")


# Global ALEPH core engine instance
aleph_core_engine = None


def initialize_aleph_core_engine(
    trading_mode: TradingMode = TradingMode.DEMO,
    bit_depth: PhaseBitDepth = PhaseBitDepth.BIT_32,
    enable_ngrok: bool = True,
    ngrok_auth_token: Optional[str] = None,
) -> ALEPHCoreEngine:
    """Initialize global ALEPH core engine instance."""
    global aleph_core_engine

    if aleph_core_engine is None:
        aleph_core_engine = ALEPHCoreEngine(
            trading_mode=trading_mode, bit_depth=bit_depth, enable_ngrok=enable_ngrok, ngrok_auth_token=ngrok_auth_token
        )

    return aleph_core_engine


def get_aleph_core_engine() -> Optional[ALEPHCoreEngine]:
    """Get global ALEPH core engine instance."""
    return aleph_core_engine


# Example usage and testing
def main():
    """Test ALEPH core engine functionality."""
    try:
        # Initialize engine
        engine = initialize_aleph_core_engine(
            trading_mode=TradingMode.DEMO, bit_depth=PhaseBitDepth.BIT_32, enable_ngrok=False  # Disable for testing
        )

        safe_print("🧠 ALEPH Core Engine Test")
        safe_print("=" * 50)

        # Test BTC data processing
        safe_print("📊 Testing BTC data processing...")
        for i in range(5):
            price = 45000 + np.random.normal(0, 1000)
            volume = 1000 + np.random.normal(0, 200)
            btc_data = engine.process_btc_data(price, volume)
            safe_print(f"  Hash {i + 1}: {btc_data.hash_value[:16]}... (entropy: {btc_data.hash_entropy:.4f})")

        # Test ALEPH trust calculation
        safe_print("\n🔍 Testing ALEPH trust calculation...")
        current_glyph = "BTC_45000.00_1000.00"
        trust_score = engine.calculate_aleph_trust(current_glyph)
        safe_print(f"  ALEPH Trust Score: {trust_score:.4f}")

        # Test profit optimization
        safe_print("\n💰 Testing profit optimization...")
        optimization = engine.optimize_profit_strategy(
            current_price=45000.0, entry_price=44000.0, volume=1000.0, risk_tolerance=0.02
        )
        safe_print(f"  Entry Threshold: {optimization.entry_threshold:.4f}")
        safe_print(f"  Exit Threshold: {optimization.exit_threshold:.4f}")
        safe_print(f"  Confidence Score: {optimization.confidence_score:.4f}")

        # Get system statistics
        safe_print("\n📈 System Statistics:")
        stats = engine.get_system_statistics()
        for key, value in stats.items():
            safe_print(f"  {key}: {value}")

        # Cleanup
        engine.shutdown()
        safe_print("\n✅ ALEPH Core Engine test completed successfully!")

    except Exception as e:
        logger.error(f"❌ ALEPH Core Engine test failed: {e}")
        safe_print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
