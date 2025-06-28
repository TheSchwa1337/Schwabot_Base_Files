# -*- coding: utf-8 -*-
"""
Ferris RDE (Rotational Differential Engine) Core
===============================================

Provides the core Ferris wheel functionality for 3.75-minute BTC price mapping
and phase synchronization for the integrated Schwabot trading system.

Mathematical Components:
- 3.75-minute cycle rotation with phase tracking
- 16-bit BTC price mapping and hash sequencing
- Matrix basket creation for multi-asset analysis
- Wall detection and anomaly handling
- Profit routing through rotational phases

MATHEMATICAL PRESERVATION: All core mathematical logic preserved.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math
import time
import numpy as np
import hashlib

logger = logging.getLogger(__name__)

class FerrisPhase(Enum):
    """Ferris wheel phases for market cycle tracking."""
    ASCENT = "ascent"
    PEAK = "peak"
    DESCENT = "descent"
    VALLEY = "valley"
    TRANSITION = "transition"

class CurveType(Enum):
    """Mathematical curve types for Ferris wheel motion."""
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    HARMONIC = "harmonic"

@dataclass
class FerrisWheelState:
    """Current state of the Ferris wheel."""
    phase: FerrisPhase
    angle: float  # Radians (0 to 2π)
    height: float  # Normalized height (0 to 1)
    velocity: float  # Angular velocity
    cycle_time: float  # Time in current cycle
    total_rotations: int
    last_update: float = field(default_factory=time.time)

@dataclass
class PriceMapping16Bit:
    """16-bit BTC price mapping data."""
    btc_price: float
    price_16bit: int  # 16-bit representation
    hash_sequence: str
    tick_precision: float
    mapping_timestamp: float = field(default_factory=time.time)

@dataclass
class MatrixBasket:
    """Matrix basket for multi-asset analysis."""
    basket_id: str
    btc_component: float
    eth_component: float
    volume_btc: float
    volatility: float
    correlation_matrix: np.ndarray
    basket_value: float
    rebalance_needed: bool
    creation_timestamp: float = field(default_factory=time.time)

@dataclass
class TradeWall:
    """Trade wall detection data."""
    wall_type: str  # "buy" or "sell"
    price_level: float
    volume: float
    strength: float
    confidence: float
    detection_timestamp: float = field(default_factory=time.time)

class FerrisRDECore:
    """
    Ferris RDE Core for rotational market analysis.
    
    Provides 3.75-minute cycle tracking, BTC price mapping,
    and phase-based trading signal generation.
    """
    
    def __init__(self):
        """Initialize Ferris RDE Core."""
        # Core wheel state
        self.wheel_state = FerrisWheelState(
            phase=FerrisPhase.VALLEY,
            angle=0.0,
            height=0.0,
            velocity=0.0,
            cycle_time=0.0,
            total_rotations=0
        )
        
        # Cycle configuration
        self.cycle_duration = 3.75 * 60  # 3.75 minutes in seconds
        self.angular_velocity = (2 * math.pi) / self.cycle_duration  # rad/sec
        
        # Price mapping
        self.price_history: List[float] = []
        self.mapping_cache: Dict[float, PriceMapping16Bit] = {}
        
        # Matrix baskets
        self.active_baskets: List[MatrixBasket] = []
        self.basket_counter = 0
        
        # Wall detection
        self.detected_walls: List[TradeWall] = []
        
        logger.info("🎡 Ferris RDE Core initialized")
    
    def update_ferris_wheel(self, delta_time: float) -> FerrisWheelState:
        """Update Ferris wheel state with time delta (in minutes)."""
        try:
            # Convert minutes to seconds
            delta_seconds = delta_time * 60
            
            # Update angle and cycle time
            self.wheel_state.angle += self.angular_velocity * delta_seconds
            self.wheel_state.cycle_time += delta_seconds
            
            # Normalize angle to 0-2π
            if self.wheel_state.angle >= 2 * math.pi:
                self.wheel_state.angle -= 2 * math.pi
                self.wheel_state.total_rotations += 1
                self.wheel_state.cycle_time = 0.0
            
            # Calculate height (0 at bottom, 1 at top)
            self.wheel_state.height = (math.sin(self.wheel_state.angle) + 1) / 2
            
            # Update velocity
            self.wheel_state.velocity = self.angular_velocity
            
            # Determine phase
            self.wheel_state.phase = self._calculate_phase(self.wheel_state.angle)
            
            # Update timestamp
            self.wheel_state.last_update = time.time()
            
            logger.debug(f"✅ Ferris wheel: Phase = {self.wheel_state.phase.value}, Height = {self.wheel_state.height:.3f}")
            
            return self.wheel_state
            
        except Exception as e:
            logger.error(f"❌ Ferris wheel update failed: {e}")
            return self.wheel_state
    
    def map_btc_price_16bit(self, btc_price: float) -> PriceMapping16Bit:
        """Map BTC price to 16-bit representation with hash sequencing."""
        try:
            # Check cache first
            if btc_price in self.mapping_cache:
                return self.mapping_cache[btc_price]
            
            # Calculate 16-bit mapping
            # Assuming BTC price range: $10,000 - $100,000
            min_price = 10000.0
            max_price = 100000.0
            
            # Normalize price to 0-1 range
            normalized_price = (btc_price - min_price) / (max_price - min_price)
            normalized_price = max(0.0, min(1.0, normalized_price))  # Clamp to 0-1
            
            # Convert to 16-bit integer
            price_16bit = int(normalized_price * 65535)  # 2^16 - 1
            
            # Generate hash sequence
            hash_input = f"{btc_price:.2f}_{price_16bit}_{time.time()}"
            hash_sequence = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            
            # Calculate tick precision
            tick_precision = (max_price - min_price) / 65535
            
            # Create mapping
            mapping = PriceMapping16Bit(
                btc_price=btc_price,
                price_16bit=price_16bit,
                hash_sequence=hash_sequence,
                tick_precision=tick_precision
            )
            
            # Cache the mapping
            self.mapping_cache[btc_price] = mapping
            
            # Store in price history
            self.price_history.append(btc_price)
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-1000:]
            
            return mapping
            
        except Exception as e:
            logger.error(f"❌ BTC price mapping failed: {e}")
            return PriceMapping16Bit(
                btc_price=btc_price,
                price_16bit=0,
                hash_sequence="error",
                tick_precision=1.0
            )
    
    def create_matrix_basket(self, market_data: Dict[str, Any]) -> MatrixBasket:
        """Create matrix basket for multi-asset analysis."""
        try:
            basket_id = f"BASKET_{self.basket_counter}_{int(time.time())}"
            self.basket_counter += 1
            
            # Extract market data
            btc_price = market_data.get("btc_price", 50000.0)
            volatility = market_data.get("volatility", 0.5)
            volume_btc = market_data.get("volume_btc", 1000.0)
            
            # Calculate components
            btc_component = 0.7  # 70% BTC weight
            eth_component = 0.3  # 30% ETH weight
            
            # Create correlation matrix (2x2 for BTC-ETH)
            correlation_matrix = np.array([
                [1.0, 0.8],  # BTC-BTC, BTC-ETH
                [0.8, 1.0]   # ETH-BTC, ETH-ETH
            ])
            
            # Calculate basket value
            basket_value = btc_component * btc_price + eth_component * (btc_price * 0.06)  # ETH ~6% of BTC
            
            # Determine if rebalancing is needed
            rebalance_needed = volatility > 0.7 or volume_btc < 500.0
            
            basket = MatrixBasket(
                basket_id=basket_id,
                btc_component=btc_component,
                eth_component=eth_component,
                volume_btc=volume_btc,
                volatility=volatility,
                correlation_matrix=correlation_matrix,
                basket_value=basket_value,
                rebalance_needed=rebalance_needed
            )
            
            self.active_baskets.append(basket)
            
            # Keep only last 100 baskets
            if len(self.active_baskets) > 100:
                self.active_baskets = self.active_baskets[-100:]
            
            return basket
            
        except Exception as e:
            logger.error(f"❌ Matrix basket creation failed: {e}")
            return MatrixBasket(
                basket_id="ERROR",
                btc_component=0.0,
                eth_component=0.0,
                volume_btc=0.0,
                volatility=0.0,
                correlation_matrix=np.eye(2),
                basket_value=0.0,
                rebalance_needed=False
            )
    
    def detect_trade_walls(self, price_data: List[float], volume_data: List[float]) -> List[TradeWall]:
        """Detect buy/sell walls in price and volume data."""
        try:
            walls = []
            
            if len(price_data) < 5 or len(volume_data) < 5:
                return walls
            
            # Simple wall detection algorithm
            for i in range(2, len(price_data) - 2):
                current_price = price_data[i]
                current_volume = volume_data[i]
                
                # Check for buy wall (high volume at resistance)
                if current_volume > np.mean(volume_data) * 2:
                    # Look for price resistance
                    price_change = abs(price_data[i+1] - current_price) / current_price
                    
                    if price_change < 0.001:  # Less than 0.1% price movement
                        wall_type = "buy" if current_volume > np.mean(volume_data) * 3 else "sell"
                        
                        wall = TradeWall(
                            wall_type=wall_type,
                            price_level=current_price,
                            volume=current_volume,
                            strength=current_volume / np.mean(volume_data),
                            confidence=min(1.0, (current_volume / np.max(volume_data)))
                        )
                        
                        walls.append(wall)
            
            # Store detected walls
            self.detected_walls.extend(walls)
            
            # Keep only recent walls (last 50)
            if len(self.detected_walls) > 50:
                self.detected_walls = self.detected_walls[-50:]
            
            return walls
            
        except Exception as e:
            logger.error(f"❌ Trade wall detection failed: {e}")
            return []
    
    def _calculate_phase(self, angle: float) -> FerrisPhase:
        """Calculate Ferris wheel phase from angle."""
        # Normalize angle to 0-2π
        normalized_angle = angle % (2 * math.pi)
        
        # Define phase boundaries
        if 0 <= normalized_angle < math.pi / 2:
            return FerrisPhase.ASCENT
        elif math.pi / 2 <= normalized_angle < 3 * math.pi / 4:
            return FerrisPhase.PEAK
        elif 3 * math.pi / 4 <= normalized_angle < 5 * math.pi / 4:
            return FerrisPhase.DESCENT
        elif 5 * math.pi / 4 <= normalized_angle < 7 * math.pi / 4:
            return FerrisPhase.VALLEY
        else:
            return FerrisPhase.TRANSITION
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive Ferris RDE system status."""
        return {
            "wheel_state": {
                "phase": self.wheel_state.phase.value,
                "angle_degrees": math.degrees(self.wheel_state.angle),
                "height": self.wheel_state.height,
                "velocity": self.wheel_state.velocity,
                "total_rotations": self.wheel_state.total_rotations,
                "cycle_progress": self.wheel_state.cycle_time / self.cycle_duration
            },
            "price_tracking": {
                "prices_tracked": len(self.price_history),
                "mappings_cached": len(self.mapping_cache),
                "last_price": self.price_history[-1] if self.price_history else 0.0
            },
            "matrix_baskets": {
                "active_baskets": len(self.active_baskets),
                "basket_counter": self.basket_counter
            },
            "wall_detection": {
                "walls_detected": len(self.detected_walls),
                "recent_walls": len([w for w in self.detected_walls if time.time() - w.detection_timestamp < 300])
            },
            "last_update": time.time()
        }


# Global instance for external access
ferris_rde_core = FerrisRDECore()

# Export functions for external use
def update_ferris_wheel(delta_time: float) -> FerrisWheelState:
    """Update Ferris wheel for external use."""
    return ferris_rde_core.update_ferris_wheel(delta_time)

def map_btc_price_16bit(btc_price: float) -> PriceMapping16Bit:
    """Map BTC price for external use."""
    return ferris_rde_core.map_btc_price_16bit(btc_price)

def create_matrix_basket(market_data: Dict[str, Any]) -> MatrixBasket:
    """Create matrix basket for external use."""
    return ferris_rde_core.create_matrix_basket(market_data)

# Export all key components
__all__ = [
    "FerrisRDECore",
    "FerrisPhase",
    "CurveType",
    "FerrisWheelState",
    "PriceMapping16Bit",
    "MatrixBasket",
    "TradeWall",
    "ferris_rde_core",
    "update_ferris_wheel",
    "map_btc_price_16bit",
    "create_matrix_basket"
] 