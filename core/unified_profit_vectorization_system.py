# -*- coding: utf-8 -*-
"""
Unified Profit Vectorization System - Complete Trading Bot Integration

This system integrates all components into a unified profit vectorization engine:
- ASIC Logic Gates with dualistic emoji routing
- Emoji Symbolic Relay with 256-bit Ferris RDE hashes
- Lantern Core with 2-bit logic gates
- Tensor calculations and timing differentials
- Drift maps and trade history integration
- 16-bit BTC price mapping
- CCXT order execution with buy/sell signals

Mathematical Foundation:
- Profit Vector: P(σ) = Σ(w_i × v_i × t_i × d_i) where:
  - w_i = ASIC gate weights
  - v_i = vectorization factors
  - t_i = timing differentials
  - d_i = drift map coefficients
- 16-bit BTC Mapping: BTC_16bit = log(price/price_min) / log(price_max/price_min) × 65535
- Smoothing Function: S(t) = Σ(α_i × P_i × exp(-β_i × |t - t_i|))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import hashlib
import logging
import time
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Import all core systems
from core.asic_logic_gate_foundation import ASICLogicGate, get_asic_gate_manager
from core.emoji_symbolic_relay import EmojiSymbolicRelay, get_emoji_relay
from core.lantern_core import LanternCore, get_lantern_core
from core.unified_math_system import unified_math
from core.multi_bit_btc_processor import MultiBitBTCProcessor, BitLevel
from core.enhanced_unified_mathematical_system import EnhancedUnifiedMathematicalSystem

# Configure logging
logger = logging.getLogger(__name__)


class VectorizationMode(Enum):
    """Profit vectorization modes"""
    CONSERVATIVE = "conservative"  # Low risk, steady profits
    BALANCED = "balanced"         # Balanced risk/reward
    AGGRESSIVE = "aggressive"     # High risk, high reward
    ADAPTIVE = "adaptive"         # Self-adjusting based on market conditions


class TimingDifferential(Enum):
    """Timing differential types"""
    MICRO = "micro"      # < 1 second
    SHORT = "short"      # 1-60 seconds
    MEDIUM = "medium"    # 1-60 minutes
    LONG = "long"        # 1-24 hours


@dataclass
class DriftMap:
    """Drift map for profit vectorization"""
    drift_id: str
    timestamp: float
    drift_magnitude: float
    drift_direction: str  # "positive", "negative", "neutral"
    confidence_score: float
    market_conditions: Dict[str, Any]
    tensor_coordinates: np.ndarray
    profit_potential: float


@dataclass
class TradeHistoryEntry:
    """Trade history entry from CSV"""
    timestamp: datetime
    symbol: str
    side: str  # "buy", "sell"
    amount: float
    price: float
    fees: float
    exchange: str
    order_id: str
    profit_loss: Optional[float] = None
    strategy: Optional[str] = None
    market_conditions: Optional[Dict[str, Any]] = None


@dataclass
class ProfitVectorizationResult:
    """Result of profit vectorization calculation"""
    vector_id: str
    timestamp: float
    profit_score: float
    confidence_score: float
    recommended_action: str  # "buy", "sell", "hold"
    order_size: float
    target_price: float
    stop_loss: float
    take_profit: float
    timing_differential: TimingDifferential
    drift_map: Optional[DriftMap] = None
    asic_gate_results: Dict[str, Any] = field(default_factory=dict)
    emoji_relay_results: Dict[str, Any] = field(default_factory=dict)
    lantern_core_results: Dict[str, Any] = field(default_factory=dict)
    tensor_results: Dict[str, Any] = field(default_factory=dict)
    btc_mapping_results: Dict[str, Any] = field(default_factory=dict)


class UnifiedProfitVectorizationSystem:
    """Unified profit vectorization system integrating all components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified profit vectorization system"""
        self.config = config or self._default_config()
        
        # Initialize all core systems
        self.asic_gate_manager = get_asic_gate_manager()
        self.emoji_relay = get_emoji_relay()
        self.lantern_core = get_lantern_core()
        self.btc_processor = MultiBitBTCProcessor()
        self.math_system = EnhancedUnifiedMathematicalSystem()
        
        # Trade history and backlog
        self.trade_history: List[TradeHistoryEntry] = []
        self.backlog_entries: List[Dict[str, Any]] = []
        self.drift_maps: List[DriftMap] = []
        
        # Vectorization state
        self.current_mode = VectorizationMode.ADAPTIVE
        self.profit_vectors: List[ProfitVectorizationResult] = []
        self.btc_price_history: List[Tuple[float, float]] = []  # (timestamp, price)
        
        # Performance tracking
        self.total_calculations = 0
        self.successful_calculations = 0
        self.average_profit_score = 0.0
        self.last_update_time = time.time()
        
        # Load trade history if available
        self._load_trade_history()
        
        logger.info("Unified Profit Vectorization System initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "btc_price_min": 1000.0,
            "btc_price_max": 100000.0,
            "profit_threshold": 0.02,  # 2% minimum profit
            "confidence_threshold": 0.7,  # 70% minimum confidence
            "max_order_size": 1.0,  # Maximum order size in BTC
            "timing_differentials": {
                "micro": 0.1,
                "short": 1.0,
                "medium": 60.0,
                "long": 3600.0
            },
            "drift_map_window": 100,  # Number of drift maps to keep
            "smoothing_factor": 0.1,  # Smoothing factor for profit vectors
            "vectorization_modes": {
                "conservative": {"risk_multiplier": 0.5, "profit_target": 0.01},
                "balanced": {"risk_multiplier": 1.0, "profit_target": 0.02},
                "aggressive": {"risk_multiplier": 2.0, "profit_target": 0.05},
                "adaptive": {"risk_multiplier": 1.0, "profit_target": 0.02}
            }
        }
    
    def _load_trade_history(self) -> None:
        """Load trade history from CSV files"""
        try:
            # Look for CSV files in common locations
            csv_paths = [
                Path("data/trade_history.csv"),
                Path("trade_history.csv"),
                Path("data/trades.csv"),
                Path("trades.csv")
            ]
            
            for csv_path in csv_paths:
                if csv_path.exists():
                    self._parse_trade_history_csv(csv_path)
                    logger.info(f"Loaded trade history from {csv_path}")
                    break
            else:
                logger.info("No trade history CSV found, starting with empty history")
                
        except Exception as e:
            logger.error(f"Failed to load trade history: {e}")
    
    def _parse_trade_history_csv(self, csv_path: Path) -> None:
        """Parse trade history from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                try:
                    # Handle different CSV formats
                    timestamp = pd.to_datetime(row.get('timestamp', row.get('date', row.get('time'))))
                    symbol = row.get('symbol', row.get('pair', 'BTC/USDT'))
                    side = row.get('side', row.get('type', 'buy'))
                    amount = float(row.get('amount', row.get('quantity', 0)))
                    price = float(row.get('price', 0))
                    fees = float(row.get('fees', row.get('fee', 0)))
                    exchange = row.get('exchange', 'unknown')
                    order_id = str(row.get('order_id', row.get('id', '')))
                    
                    entry = TradeHistoryEntry(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=price,
                        fees=fees,
                        exchange=exchange,
                        order_id=order_id
                    )
                    
                    self.trade_history.append(entry)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse trade history row: {e}")
                    continue
            
            # Sort by timestamp
            self.trade_history.sort(key=lambda x: x.timestamp)
            logger.info(f"Loaded {len(self.trade_history)} trade history entries")
            
        except Exception as e:
            logger.error(f"Failed to parse trade history CSV: {e}")
    
    def calculate_profit_vectorization(
        self,
        btc_price: float,
        volume: float,
        market_data: Optional[Dict[str, Any]] = None,
        mode: Optional[VectorizationMode] = None
    ) -> ProfitVectorizationResult:
        """
        Calculate unified profit vectorization using all integrated systems.
        
        Mathematical Formula:
        P(σ) = Σ(w_i × v_i × t_i × d_i) × S(t) × M(btc_16bit)
        
        Where:
        - w_i = ASIC gate weights
        - v_i = vectorization factors from emoji relay
        - t_i = timing differentials from lantern core
        - d_i = drift map coefficients
        - S(t) = smoothing function
        - M(btc_16bit) = 16-bit BTC mapping factor
        """
        start_time = time.time()
        
        try:
            self.total_calculations += 1
            
            # Update BTC price history
            self._update_btc_price_history(btc_price)
            
            # Step 1: Process through ASIC logic gates
            asic_input = {
                "btc_price": btc_price,
                "volume": volume,
                "market_data": market_data or {},
                "timestamp": time.time()
            }
            asic_results = self.asic_gate_manager.process_input(asic_input)
            
            # Step 2: Create emoji symbolic relay
            emoji_symbols = self._extract_emoji_symbols(asic_results)
            relay_hash = self.emoji_relay.create_relay_path(emoji_symbols)
            
            # Step 3: Process through lantern core
            lantern_input = {
                "asic_results": asic_results,
                "relay_hash": relay_hash,
                "btc_price": btc_price,
                "volume": volume
            }
            lantern_results = self.lantern_core.relay_to_bit_gates(lantern_input)
            
            # Step 4: Calculate tensor operations
            tensor_results = self._calculate_tensor_operations(btc_price, volume, market_data)
            
            # Step 5: Calculate timing differentials
            timing_diff = self._calculate_timing_differentials(btc_price, volume)
            
            # Step 6: Update drift maps
            drift_map = self._update_drift_maps(btc_price, volume, market_data)
            
            # Step 7: 16-bit BTC price mapping
            btc_mapping = self.math_system.map_btc_price_16bit(btc_price, "mid")
            
            # Step 8: Calculate unified profit vectorization
            profit_score = self._calculate_unified_profit_score(
                asic_results, relay_hash, lantern_results, tensor_results,
                timing_diff, drift_map, btc_mapping
            )
            
            # Step 9: Determine trading action
            action, order_size, target_price, stop_loss, take_profit = self._determine_trading_action(
                profit_score, btc_price, mode or self.current_mode
            )
            
            # Step 10: Create result
            result = ProfitVectorizationResult(
                vector_id=f"vector_{int(time.time() * 1000)}",
                timestamp=time.time(),
                profit_score=profit_score,
                confidence_score=self._calculate_confidence_score(
                    asic_results, lantern_results, tensor_results, drift_map
                ),
                recommended_action=action,
                order_size=order_size,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timing_differential=timing_diff,
                drift_map=drift_map,
                asic_gate_results=asic_results,
                emoji_relay_results={"relay_hash": relay_hash, "symbols": emoji_symbols},
                lantern_core_results=lantern_results,
                tensor_results=tensor_results,
                btc_mapping_results={
                    "mapped_16bit": btc_mapping.mapped_16bit,
                    "hash_sequence": btc_mapping.hash_sequence,
                    "profit_factor": btc_mapping.profit_factor
                }
            )
            
            # Store result
            self.profit_vectors.append(result)
            if len(self.profit_vectors) > 1000:
                self.profit_vectors = self.profit_vectors[-1000:]
            
            # Update performance metrics
            self.successful_calculations += 1
            self._update_performance_metrics(profit_score)
            
            execution_time = time.time() - start_time
            logger.debug(f"Profit vectorization calculated in {execution_time:.4f}s: {profit_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Profit vectorization calculation failed: {e}")
            return self._create_fallback_result(btc_price, volume)
    
    def _extract_emoji_symbols(self, asic_results: Dict[str, Any]) -> List[str]:
        """Extract emoji symbols from ASIC results"""
        symbols = []
        
        # Extract emoji symbols from ASIC gate results
        for key, value in asic_results.items():
            if key == "emoji_symbol" and isinstance(value, str):
                symbols.append(value)
            elif isinstance(value, dict) and "emoji_symbol" in value:
                symbols.append(value["emoji_symbol"])
        
        # Add default symbols if none found
        if not symbols:
            symbols = ["💰", "🔥", "⚡", "🎯"]
        
        return symbols[:4]  # Limit to 4 symbols
    
    def _calculate_tensor_operations(
        self,
        btc_price: float,
        volume: float,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate tensor operations for profit vectorization"""
        try:
            # Create price and volume arrays
            prices = np.array([btc_price])
            volumes = np.array([volume])
            
            # Calculate profit routing tensor
            if hasattr(self.math_system, 'tensor_algebra'):
                routing_weights = np.array([0.4, 0.3, 0.2, 0.1])  # Weight distribution
                profit_tensor = self.math_system.tensor_algebra.profit_routing_tensor(
                    prices.reshape(-1, 1), routing_weights.reshape(1, -1)
                )
            else:
                profit_tensor = np.array([[btc_price * 0.01]])  # Fallback
            
            # Calculate tensor contraction
            tensor_score = float(np.mean(profit_tensor))
            
            return {
                "profit_tensor": profit_tensor.tolist(),
                "tensor_score": tensor_score,
                "price_volatility": self._calculate_price_volatility(),
                "volume_profile": self._calculate_volume_profile(volume)
            }
            
        except Exception as e:
            logger.error(f"Tensor operations failed: {e}")
            return {
                "profit_tensor": [[0.0]],
                "tensor_score": 0.0,
                "price_volatility": 0.0,
                "volume_profile": 0.0
            }
    
    def _calculate_timing_differentials(self, btc_price: float, volume: float) -> TimingDifferential:
        """Calculate timing differentials based on market conditions"""
        try:
            if len(self.btc_price_history) < 2:
                return TimingDifferential.MEDIUM
            
            # Calculate price change rate
            recent_prices = [price for _, price in self.btc_price_history[-10:]]
            if len(recent_prices) >= 2:
                price_change_rate = abs(recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                
                # Determine timing differential based on volatility
                if price_change_rate > 0.05:  # High volatility
                    return TimingDifferential.MICRO
                elif price_change_rate > 0.02:  # Medium volatility
                    return TimingDifferential.SHORT
                elif price_change_rate > 0.01:  # Low volatility
                    return TimingDifferential.MEDIUM
                else:
                    return TimingDifferential.LONG
            
            return TimingDifferential.MEDIUM
            
        except Exception as e:
            logger.error(f"Timing differential calculation failed: {e}")
            return TimingDifferential.MEDIUM
    
    def _update_drift_maps(
        self,
        btc_price: float,
        volume: float,
        market_data: Optional[Dict[str, Any]]
    ) -> Optional[DriftMap]:
        """Update drift maps for profit vectorization"""
        try:
            if len(self.btc_price_history) < 2:
                return None
            
            # Calculate drift magnitude
            recent_prices = [price for _, price in self.btc_price_history[-5:]]
            price_change = recent_prices[-1] - recent_prices[0]
            drift_magnitude = abs(price_change) / recent_prices[0]
            
            # Determine drift direction
            if price_change > 0:
                drift_direction = "positive"
            elif price_change < 0:
                drift_direction = "negative"
            else:
                drift_direction = "neutral"
            
            # Calculate confidence score
            confidence_score = min(1.0, drift_magnitude * 10)  # Scale to [0, 1]
            
            # Create drift map
            drift_map = DriftMap(
                drift_id=f"drift_{int(time.time() * 1000)}",
                timestamp=time.time(),
                drift_magnitude=drift_magnitude,
                drift_direction=drift_direction,
                confidence_score=confidence_score,
                market_conditions=market_data or {},
                tensor_coordinates=np.array([btc_price, volume, drift_magnitude]),
                profit_potential=drift_magnitude * (1.0 if drift_direction == "positive" else -0.5)
            )
            
            # Store drift map
            self.drift_maps.append(drift_map)
            if len(self.drift_maps) > self.config["drift_map_window"]:
                self.drift_maps = self.drift_maps[-self.config["drift_map_window"]:]
            
            return drift_map
            
        except Exception as e:
            logger.error(f"Drift map update failed: {e}")
            return None
    
    def _calculate_unified_profit_score(
        self,
        asic_results: Dict[str, Any],
        relay_hash: str,
        lantern_results: Dict[str, Any],
        tensor_results: Dict[str, Any],
        timing_diff: TimingDifferential,
        drift_map: Optional[DriftMap],
        btc_mapping: Any
    ) -> float:
        """Calculate unified profit score using all components"""
        try:
            # Extract weights from ASIC results
            asic_weight = asic_results.get("profit_vector", 1.0)
            
            # Extract vectorization factor from emoji relay
            relay_factor = len(relay_hash) / 64.0 if relay_hash else 0.5
            
            # Extract timing factor from lantern core
            timing_factor = lantern_results.get("state_energy", 0.5)
            
            # Extract tensor factor
            tensor_factor = tensor_results.get("tensor_score", 0.5)
            
            # Extract drift factor
            drift_factor = drift_map.profit_potential if drift_map else 0.0
            
            # Extract BTC mapping factor
            btc_factor = btc_mapping.profit_factor if hasattr(btc_mapping, 'profit_factor') else 0.5
            
            # Calculate unified profit score
            profit_score = (
                asic_weight * 0.3 +
                relay_factor * 0.2 +
                timing_factor * 0.2 +
                tensor_factor * 0.15 +
                drift_factor * 0.1 +
                btc_factor * 0.05
            )
            
            # Apply smoothing
            if self.profit_vectors:
                smoothing_factor = self.config["smoothing_factor"]
                last_score = self.profit_vectors[-1].profit_score
                profit_score = (1 - smoothing_factor) * last_score + smoothing_factor * profit_score
            
            # Normalize to [0, 1]
            profit_score = max(0.0, min(1.0, profit_score))
            
            return profit_score
            
        except Exception as e:
            logger.error(f"Unified profit score calculation failed: {e}")
            return 0.5
    
    def _calculate_confidence_score(
        self,
        asic_results: Dict[str, Any],
        lantern_results: Dict[str, Any],
        tensor_results: Dict[str, Any],
        drift_map: Optional[DriftMap]
    ) -> float:
        """Calculate confidence score for the vectorization"""
        try:
            # ASIC confidence
            asic_confidence = asic_results.get("profit_vector", 0.5)
            
            # Lantern core confidence
            lantern_confidence = lantern_results.get("processing_intensity", 0.5)
            
            # Tensor confidence
            tensor_confidence = tensor_results.get("tensor_score", 0.5)
            
            # Drift map confidence
            drift_confidence = drift_map.confidence_score if drift_map else 0.5
            
            # Calculate weighted average
            confidence_score = (
                asic_confidence * 0.3 +
                lantern_confidence * 0.3 +
                tensor_confidence * 0.2 +
                drift_confidence * 0.2
            )
            
            return max(0.0, min(1.0, confidence_score))
            
        except Exception as e:
            logger.error(f"Confidence score calculation failed: {e}")
            return 0.5
    
    def _determine_trading_action(
        self,
        profit_score: float,
        btc_price: float,
        mode: VectorizationMode
    ) -> Tuple[str, float, float, float, float]:
        """Determine trading action based on profit score and mode"""
        try:
            mode_config = self.config["vectorization_modes"][mode.value]
            risk_multiplier = mode_config["risk_multiplier"]
            profit_target = mode_config["profit_target"]
            
            # Determine action based on profit score
            if profit_score > self.config["confidence_threshold"]:
                action = "buy"
            elif profit_score < (1.0 - self.config["confidence_threshold"]):
                action = "sell"
            else:
                action = "hold"
            
            # Calculate order size based on confidence and risk
            order_size = min(
                self.config["max_order_size"],
                profit_score * risk_multiplier * 0.1  # Scale down for safety
            )
            
            # Calculate target prices
            if action == "buy":
                target_price = btc_price * (1 + profit_target)
                stop_loss = btc_price * (1 - profit_target * 0.5)
                take_profit = btc_price * (1 + profit_target * 2)
            elif action == "sell":
                target_price = btc_price * (1 - profit_target)
                stop_loss = btc_price * (1 + profit_target * 0.5)
                take_profit = btc_price * (1 - profit_target * 2)
            else:
                target_price = btc_price
                stop_loss = btc_price * 0.99
                take_profit = btc_price * 1.01
            
            return action, order_size, target_price, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Trading action determination failed: {e}")
            return "hold", 0.0, btc_price, btc_price * 0.99, btc_price * 1.01
    
    def _update_btc_price_history(self, btc_price: float) -> None:
        """Update BTC price history"""
        self.btc_price_history.append((time.time(), btc_price))
        
        # Keep only recent history
        if len(self.btc_price_history) > 1000:
            self.btc_price_history = self.btc_price_history[-1000:]
    
    def _calculate_price_volatility(self) -> float:
        """Calculate price volatility from history"""
        try:
            if len(self.btc_price_history) < 10:
                return 0.0
            
            prices = [price for _, price in self.btc_price_history[-20:]]
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.0
            
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Price volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_volume_profile(self, current_volume: float) -> float:
        """Calculate volume profile"""
        try:
            if len(self.trade_history) < 5:
                return 0.5
            
            # Calculate average volume from trade history
            recent_volumes = [trade.amount for trade in self.trade_history[-20:]]
            avg_volume = np.mean(recent_volumes) if recent_volumes else current_volume
            
            # Normalize current volume against average
            volume_profile = min(1.0, current_volume / avg_volume) if avg_volume > 0 else 0.5
            
            return float(volume_profile)
            
        except Exception as e:
            logger.error(f"Volume profile calculation failed: {e}")
            return 0.5
    
    def _update_performance_metrics(self, profit_score: float) -> None:
        """Update performance metrics"""
        try:
            # Update average profit score
            if self.successful_calculations == 1:
                self.average_profit_score = profit_score
            else:
                self.average_profit_score = (
                    (self.average_profit_score * (self.successful_calculations - 1) + profit_score) /
                    self.successful_calculations
                )
            
            self.last_update_time = time.time()
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    def _create_fallback_result(self, btc_price: float, volume: float) -> ProfitVectorizationResult:
        """Create fallback result when calculation fails"""
        return ProfitVectorizationResult(
            vector_id=f"fallback_{int(time.time() * 1000)}",
            timestamp=time.time(),
            profit_score=0.5,
            confidence_score=0.5,
            recommended_action="hold",
            order_size=0.0,
            target_price=btc_price,
            stop_loss=btc_price * 0.99,
            take_profit=btc_price * 1.01,
            timing_differential=TimingDifferential.MEDIUM
        )
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            success_rate = (
                self.successful_calculations / self.total_calculations
                if self.total_calculations > 0 else 0.0
            )
            
            return {
                "total_calculations": self.total_calculations,
                "successful_calculations": self.successful_calculations,
                "success_rate": success_rate,
                "average_profit_score": self.average_profit_score,
                "current_mode": self.current_mode.value,
                "trade_history_count": len(self.trade_history),
                "drift_maps_count": len(self.drift_maps),
                "profit_vectors_count": len(self.profit_vectors),
                "btc_price_history_count": len(self.btc_price_history),
                "last_update_time": self.last_update_time,
                "asic_gate_stats": self.asic_gate_manager.get_gate_statistics(),
                "emoji_relay_stats": get_relay_statistics(),
                "lantern_core_stats": get_lantern_statistics()
            }
            
        except Exception as e:
            logger.error(f"System statistics calculation failed: {e}")
            return {}
    
    def export_trade_signals(self, format: str = "json") -> str:
        """Export trade signals for CCXT execution"""
        try:
            if not self.profit_vectors:
                return ""
            
            # Get recent profitable signals
            recent_signals = [
                vector for vector in self.profit_vectors[-100:]
                if vector.recommended_action in ["buy", "sell"] and
                vector.confidence_score > self.config["confidence_threshold"]
            ]
            
            if format == "json":
                signals_data = []
                for signal in recent_signals:
                    signals_data.append({
                        "timestamp": signal.timestamp,
                        "action": signal.recommended_action,
                        "symbol": "BTC/USDT",
                        "amount": signal.order_size,
                        "price": signal.target_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "confidence": signal.confidence_score,
                        "profit_score": signal.profit_score
                    })
                
                return json.dumps(signals_data, indent=2)
            
            elif format == "csv":
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    "timestamp", "action", "symbol", "amount", "price",
                    "stop_loss", "take_profit", "confidence", "profit_score"
                ])
                
                # Write data
                for signal in recent_signals:
                    writer.writerow([
                        signal.timestamp,
                        signal.recommended_action,
                        "BTC/USDT",
                        signal.order_size,
                        signal.target_price,
                        signal.stop_loss,
                        signal.take_profit,
                        signal.confidence_score,
                        signal.profit_score
                    ])
                
                return output.getvalue()
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Trade signals export failed: {e}")
            return ""


# Global unified profit vectorization system instance
unified_profit_system = UnifiedProfitVectorizationSystem()


def get_unified_profit_system() -> UnifiedProfitVectorizationSystem:
    """Get global unified profit vectorization system instance"""
    return unified_profit_system


def calculate_profit_vectorization(
    btc_price: float,
    volume: float,
    market_data: Optional[Dict[str, Any]] = None,
    mode: Optional[VectorizationMode] = None
) -> ProfitVectorizationResult:
    """Calculate profit vectorization using the unified system"""
    return unified_profit_system.calculate_profit_vectorization(
        btc_price, volume, market_data, mode
    )


def get_profit_system_statistics() -> Dict[str, Any]:
    """Get comprehensive profit system statistics"""
    return unified_profit_system.get_system_statistics()


def export_trade_signals(format: str = "json") -> str:
    """Export trade signals for execution"""
    return unified_profit_system.export_trade_signals(format) 