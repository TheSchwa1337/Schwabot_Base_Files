# -*- coding: utf-8 -*-
"""
Integrated Ferris Glyph Controller for Schwabot Trading
======================================================

Connects the enhanced Lantern Core, Ferris RDE, Ghost Router, and glyph containment
systems into a unified recursive mathematical trading framework.

Key Integration Points:
- 3.75-minute BTC price mapping with 16-bit tick resolution
- SHA-256 glyph routing through CPU/GPU/ColdBase portals
- Word entropy correlation with trading profit tiers
- Multi-pair CCXT integration (BTC/USDC, ETH/USDC)
- Real-time glyph visualization and containment

Mathematical Flow:
BTC Price → SHA256 → Word Mapping → Bit Gate → Ferris Phase → Ghost Route → Trade Execution
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
import logging
import hashlib
import threading
from datetime import datetime, timedelta

# Import existing Schwabot components
try:
    from .lantern_core import enhanced_lantern_core, EntropyMode, map_btc_price_to_word
    from .ferris_rde_core import ferris_rde_core, FerrisPhase
    from .ghost_router import GhostRouter, RouterInput
    from .unified_math_system import unified_math
    CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core components not fully available: {e}")
    CORE_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class GlyphPortalType(Enum):
    """Portal types for glyph routing."""
    CPU_PORTAL = "cpu_portal"
    GPU_PORTAL = "gpu_portal"
    COLDBASE_PORTAL = "coldbase_portal"
    HYBRID_PORTAL = "hybrid_portal"

class TradingTimeframe(Enum):
    """Trading timeframes for BTC price mapping."""
    FERRIS_CYCLE = "3.75_minutes"  # Primary 3.75-minute cycle
    MICRO_TICK = "15_seconds"      # Micro tick for high frequency
    MEDIUM_CYCLE = "15_minutes"    # Medium-term analysis
    MACRO_CYCLE = "1_hour"         # Macro trend analysis

@dataclass
class GlyphState:
    """Glyph state for containment and routing."""
    glyph_id: str
    word: str
    entropy_value: float
    bit_pattern: str
    profit_symbolization: float
    portal_target: GlyphPortalType
    ferris_phase: Optional[str] = None
    btc_correlation: Optional[float] = None
    creation_timestamp: float = field(default_factory=time.time)

@dataclass
class IntegratedTradingSignal:
    """Comprehensive trading signal from integrated system."""
    signal_id: str
    btc_price: float
    word_mapping: Dict[str, Any]
    glyph_states: List[GlyphState]
    ferris_data: Dict[str, Any]
    ghost_route: str
    recommended_action: str
    confidence_score: float
    profit_potential: float
    risk_assessment: Dict[str, float]
    execution_timeframe: TradingTimeframe
    timestamp: float = field(default_factory=time.time)

class IntegratedFerrisGlyphController:
    """
    Main controller integrating all Schwabot mathematical trading components.
    
    Provides unified interface for:
    - 3.75-minute BTC price correlation with word entropy
    - Glyph containment and SHA-256 routing  
    - Ferris wheel phase synchronization
    - Ghost router profit optimization
    - Multi-exchange CCXT trading execution
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the integrated controller."""
        self.config = config or {}
        
        # Core integration state
        self.glyph_registry: Dict[str, GlyphState] = {}
        self.active_signals: List[IntegratedTradingSignal] = []
        self.price_word_history: List[Dict[str, Any]] = []
        self.portal_load_balancer = PortalLoadBalancer()
        
        # Threading for real-time processing
        self.processing_lock = threading.RLock()
        self.is_running = False
        self.processing_thread = None
        
        # Performance metrics
        self.metrics = {
            "total_signals_processed": 0,
            "successful_trades": 0,
            "profit_generated": 0.0,
            "average_glyph_entropy": 0.0,
            "ferris_cycles_completed": 0,
            "portal_utilization": {"cpu": 0, "gpu": 0, "coldbase": 0}
        }
        
        # Initialize ghost router
        self.ghost_router = GhostRouter() if CORE_COMPONENTS_AVAILABLE else None
        
        logger.info("🎡 Integrated Ferris Glyph Controller initialized")
    
    def process_btc_price_cycle(self, btc_price: float, timeframe: TradingTimeframe = TradingTimeframe.FERRIS_CYCLE) -> IntegratedTradingSignal:
        """
        Process complete BTC price cycle through integrated mathematical framework.
        
        Flow: BTC Price → Word Mapping → Glyph Generation → Ferris Phase → Ghost Route → Trading Signal
        """
        try:
            with self.processing_lock:
                signal_id = self._generate_signal_id(btc_price)
                
                # Step 1: Map BTC price to word entropy (Lantern Core)
                word_mapping = self._map_price_to_entropy_word(btc_price)
                
                # Step 2: Generate glyph states with SHA-256 routing
                glyph_states = self._generate_glyph_states(word_mapping, btc_price)
                
                # Step 3: Update Ferris wheel and get phase data
                ferris_data = self._update_ferris_wheel_state(btc_price, timeframe)
                
                # Step 4: Route through Ghost Router for profit optimization
                ghost_route = self._route_through_ghost_system(btc_price, word_mapping, ferris_data)
                
                # Step 5: Generate comprehensive trading signal
                trading_signal = self._generate_integrated_signal(
                    signal_id, btc_price, word_mapping, glyph_states, 
                    ferris_data, ghost_route, timeframe
                )
                
                # Step 6: Store and track signal
                self.active_signals.append(trading_signal)
                self._update_metrics(trading_signal)
                
                logger.info(f"📊 BTC cycle processed: {btc_price:.2f} → {trading_signal.recommended_action}")
                return trading_signal
                
        except Exception as e:
            logger.error(f"BTC price cycle processing failed: {e}")
            return self._create_fallback_signal(btc_price, str(e))
    
    def _map_price_to_entropy_word(self, btc_price: float) -> Dict[str, Any]:
        """Map BTC price to entropy word using Lantern Core."""
        if not CORE_COMPONENTS_AVAILABLE:
            return {"error": "Core components not available"}
            
        try:
            # Use enhanced lantern core for price-to-word mapping
            word_mapping = map_btc_price_to_word(btc_price)
            
            # Enhance with additional entropy modes
            entropy_modes = [
                EntropyMode.PROFIT_SYMBOLIC,
                EntropyMode.BTC_HASH_DERIVE,
                EntropyMode.PATTERN_MATCH
            ]
            
            additional_words = {}
            for mode in entropy_modes:
                mode_word = enhanced_lantern_core.get_entropy_word(mode)
                additional_words[mode.value] = mode_word
            
            word_mapping["additional_entropy_words"] = additional_words
            word_mapping["mapping_strength"] = self._calculate_mapping_strength(word_mapping)
            
            # Store in history for trend analysis
            self.price_word_history.append(word_mapping)
            if len(self.price_word_history) > 1000:
                self.price_word_history = self.price_word_history[-1000:]
            
            return word_mapping
            
        except Exception as e:
            logger.error(f"Price to entropy word mapping failed: {e}")
            return {"error": str(e), "btc_price": btc_price}
    
    def _generate_glyph_states(self, word_mapping: Dict[str, Any], btc_price: float) -> List[GlyphState]:
        """Generate glyph states with SHA-256 routing."""
        try:
            glyph_states = []
            
            # Primary glyph from main word
            if "selected_word" in word_mapping:
                primary_glyph = self._create_glyph_state(
                    word_mapping["selected_word"], 
                    word_mapping.get("word_entropy", 0.0),
                    btc_price
                )
                glyph_states.append(primary_glyph)
            
            # Secondary glyphs from additional entropy words
            if "additional_entropy_words" in word_mapping:
                for mode, word in word_mapping["additional_entropy_words"].items():
                    secondary_glyph = self._create_glyph_state(word, 0.0, btc_price, mode)
                    glyph_states.append(secondary_glyph)
            
            # Register glyphs
            for glyph in glyph_states:
                self.glyph_registry[glyph.glyph_id] = glyph
            
            return glyph_states
            
        except Exception as e:
            logger.error(f"Glyph state generation failed: {e}")
            return []
    
    def _create_glyph_state(self, word: str, entropy: float, btc_price: float, mode: str = "primary") -> GlyphState:
        """Create individual glyph state with SHA-256 routing."""
        # Generate SHA-256 hash for routing
        combined_input = f"{word}_{btc_price}_{time.time()}_{mode}"
        sha_hash = hashlib.sha256(combined_input.encode()).hexdigest()
        
        # Extract routing information from hash
        hash_int = int(sha_hash[:8], 16)
        
        # Route to portal based on hash
        portal_mapping = hash_int % 4
        if portal_mapping == 0:
            portal = GlyphPortalType.CPU_PORTAL
        elif portal_mapping == 1:
            portal = GlyphPortalType.GPU_PORTAL
        elif portal_mapping == 2:
            portal = GlyphPortalType.COLDBASE_PORTAL
        else:
            portal = GlyphPortalType.HYBRID_PORTAL
        
        # Generate bit pattern for Lantern Core compatibility
        bit_pattern = self._hash_to_bit_pattern(sha_hash)
        
        # Calculate profit symbolization
        profit_symbolization = self._calculate_profit_symbolization(word, entropy, btc_price)
        
        return GlyphState(
            glyph_id=sha_hash[:16],
            word=word,
            entropy_value=entropy,
            bit_pattern=bit_pattern,
            profit_symbolization=profit_symbolization,
            portal_target=portal,
            btc_correlation=self._calculate_btc_correlation(word, btc_price)
        )
    
    def _update_ferris_wheel_state(self, btc_price: float, timeframe: TradingTimeframe) -> Dict[str, Any]:
        """Update Ferris wheel state and get phase data."""
        if not CORE_COMPONENTS_AVAILABLE:
            return {"error": "Ferris RDE not available"}
            
        try:
            # Calculate delta time based on timeframe
            delta_mapping = {
                TradingTimeframe.FERRIS_CYCLE: 3.75 * 60,    # 3.75 minutes in seconds
                TradingTimeframe.MICRO_TICK: 15,             # 15 seconds
                TradingTimeframe.MEDIUM_CYCLE: 15 * 60,      # 15 minutes
                TradingTimeframe.MACRO_CYCLE: 60 * 60        # 1 hour
            }
            
            delta_time = delta_mapping.get(timeframe, 3.75 * 60) / 60  # Convert to minutes
            
            # Update Ferris wheel
            wheel_data = ferris_rde_core.update_ferris_wheel(delta_time)
            
            # Map BTC price to 16-bit
            price_data = ferris_rde_core.map_btc_price_16bit(btc_price)
            
            # Create market data for matrix basket
            market_data = {
                "btc_price": btc_price,
                "volatility": self._calculate_volatility(),
                "volume_btc": 1000.0,  # Default volume
                "timestamp": time.time()
            }
            
            basket_data = ferris_rde_core.create_matrix_basket(market_data)
            
            # Combine all Ferris data
            ferris_data = {
                "wheel": wheel_data.__dict__ if hasattr(wheel_data, '__dict__') else wheel_data,
                "price_mapping": price_data.__dict__ if hasattr(price_data, '__dict__') else price_data,
                "basket": basket_data.__dict__ if hasattr(basket_data, '__dict__') else basket_data,
                "phase": wheel_data.phase.value if hasattr(wheel_data, 'phase') else "unknown",
                "ferris_timestamp": time.time()
            }
            
            self.metrics["ferris_cycles_completed"] += 1
            return ferris_data
            
        except Exception as e:
            logger.error(f"Ferris wheel state update failed: {e}")
            return {"error": str(e)}
    
    def _route_through_ghost_system(self, btc_price: float, word_mapping: Dict[str, Any], ferris_data: Dict[str, Any]) -> str:
        """Route through Ghost Router for profit optimization."""
        if not self.ghost_router:
            return "noop"
            
        try:
            # Create RouterInput for Ghost Router
            router_input = RouterInput(
                tick_hash=word_mapping.get("price_hash", "default"),
                mem_hash=hashlib.sha256(str(time.time()).encode()).hexdigest()[:16],
                pool_volumes=np.array([1000.0, 1200.0, 900.0, 1100.0]),  # Mock pool volumes
                btc_dip=btc_price < 45000.0,  # Simple dip detection
                lantern_vec=np.array([1.0, 0.5, 0.8]),  # Mock lantern vector
                lantern_ref=np.array([1.0, 0.5, 0.8]),  # Mock reference
                ai_hashes=("hash1", "hash2", "hash3"),  # Mock AI hashes
                ai_weights=(1.0, 1.0, 1.0),
                opportunity_ts=time.time() - 300,  # 5 minutes ago
                now_ts=time.time(),
                price_now=btc_price,
                price_pred=btc_price * 1.02,  # 2% predicted increase
                curr_profit=0.0,
                projected_exit=btc_price * 1.05,  # 5% target
                news_score=0.3  # Neutral news sentiment
            )
            
            # Get routing decision
            ghost_route = self.ghost_router.route(router_input)
            logger.debug(f"Ghost router decision: {ghost_route}")
            
            return ghost_route
            
        except Exception as e:
            logger.error(f"Ghost router processing failed: {e}")
            return "noop"
    
    def _generate_integrated_signal(self, signal_id: str, btc_price: float, word_mapping: Dict[str, Any], 
                                   glyph_states: List[GlyphState], ferris_data: Dict[str, Any], 
                                   ghost_route: str, timeframe: TradingTimeframe) -> IntegratedTradingSignal:
        """Generate comprehensive integrated trading signal."""
        try:
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(word_mapping, ferris_data, ghost_route)
            
            # Calculate profit potential
            profit_potential = self._calculate_profit_potential(glyph_states, ferris_data)
            
            # Generate risk assessment
            risk_assessment = self._calculate_risk_assessment(btc_price, word_mapping, ghost_route)
            
            # Determine recommended action
            recommended_action = self._determine_recommended_action(ghost_route, confidence_score, profit_potential)
            
            return IntegratedTradingSignal(
                signal_id=signal_id,
                btc_price=btc_price,
                word_mapping=word_mapping,
                glyph_states=glyph_states,
                ferris_data=ferris_data,
                ghost_route=ghost_route,
                recommended_action=recommended_action,
                confidence_score=confidence_score,
                profit_potential=profit_potential,
                risk_assessment=risk_assessment,
                execution_timeframe=timeframe
            )
            
        except Exception as e:
            logger.error(f"Integrated signal generation failed: {e}")
            return self._create_fallback_signal(btc_price, str(e))
    
    # Helper methods for calculations
    def _generate_signal_id(self, btc_price: float) -> str:
        """Generate unique signal ID."""
        timestamp = str(int(time.time() * 1000))
        price_str = f"{btc_price:.2f}".replace(".", "")
        return f"IFGC_{timestamp}_{price_str}"
    
    def _calculate_mapping_strength(self, word_mapping: Dict[str, Any]) -> float:
        """Calculate strength of price-to-word mapping."""
        if "word_entropy" not in word_mapping:
            return 0.0
        
        entropy = word_mapping["word_entropy"]
        price_hash_strength = len(word_mapping.get("price_hash", "")) / 16.0
        return (entropy + price_hash_strength) / 2.0
    
    def _hash_to_bit_pattern(self, sha_hash: str) -> str:
        """Convert SHA hash to bit pattern for Lantern Core."""
        hash_int = int(sha_hash[:2], 16)
        if hash_int < 64:
            return "0"
        elif hash_int < 128:
            return "1"
        elif hash_int < 192:
            return "10"
        else:
            return "11"
    
    def _calculate_profit_symbolization(self, word: str, entropy: float, btc_price: float) -> float:
        """Calculate profit symbolization for glyph."""
        base_symbolization = entropy * (btc_price / 50000.0)  # Normalize around $50k BTC
        
        # Apply word category multipliers (matching Lantern Core)
        if word in enhanced_lantern_core.word_categories.get("profit_words", []):
            base_symbolization *= 1.5
        elif word in enhanced_lantern_core.word_categories.get("mathematical_words", []):
            base_symbolization *= 1.3
        
        return base_symbolization
    
    def _calculate_btc_correlation(self, word: str, btc_price: float) -> float:
        """Calculate BTC price correlation with word."""
        word_hash = hashlib.sha256(word.encode()).hexdigest()
        hash_int = int(word_hash[:8], 16)
        normalized_hash = hash_int / (2**32)  # Normalize to 0-1
        
        # Create correlation based on price and hash
        price_factor = (btc_price % 1000) / 1000.0
        return (normalized_hash + price_factor) / 2.0
    
    def _calculate_volatility(self) -> float:
        """Calculate BTC volatility from price history."""
        if len(self.price_word_history) < 2:
            return 0.5  # Default volatility
        
        recent_prices = [item.get("btc_price", 50000.0) for item in self.price_word_history[-10:]]
        if len(recent_prices) < 2:
            return 0.5
        
        price_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                        for i in range(1, len(recent_prices))]
        return np.mean(price_changes) if price_changes else 0.5
    
    def _calculate_confidence_score(self, word_mapping: Dict[str, Any], ferris_data: Dict[str, Any], ghost_route: str) -> float:
        """Calculate overall confidence score for trading signal."""
        word_confidence = word_mapping.get("mapping_strength", 0.0)
        ferris_confidence = 0.8 if ghost_route != "noop" else 0.3
        ghost_confidence = 0.9 if ghost_route == "ghost_trade" else 0.5
        
        return (word_confidence + ferris_confidence + ghost_confidence) / 3.0
    
    def _calculate_profit_potential(self, glyph_states: List[GlyphState], ferris_data: Dict[str, Any]) -> float:
        """Calculate profit potential from glyph states."""
        if not glyph_states:
            return 0.0
        
        avg_symbolization = np.mean([glyph.profit_symbolization for glyph in glyph_states])
        ferris_multiplier = 1.2 if ferris_data.get("phase") in ["PEAK", "ASCENT"] else 1.0
        
        return avg_symbolization * ferris_multiplier
    
    def _calculate_risk_assessment(self, btc_price: float, word_mapping: Dict[str, Any], ghost_route: str) -> Dict[str, float]:
        """Calculate comprehensive risk assessment."""
        return {
            "price_risk": min(1.0, abs(btc_price - 50000) / 50000),  # Risk from price deviation
            "entropy_risk": 1.0 - word_mapping.get("mapping_strength", 0.0),
            "routing_risk": 0.1 if ghost_route == "ghost_trade" else 0.7,
            "overall_risk": 0.3  # Balanced default
        }
    
    def _determine_recommended_action(self, ghost_route: str, confidence: float, profit_potential: float) -> str:
        """Determine recommended trading action."""
        if ghost_route == "ghost_trade" and confidence > 0.7 and profit_potential > 0.5:
            return "EXECUTE_TRADE"
        elif ghost_route == "hold_usdc" or confidence < 0.4:
            return "HOLD_POSITION"
        elif profit_potential > 0.3:
            return "PREPARE_ENTRY"
        else:
            return "MONITOR_MARKET"
    
    def _create_fallback_signal(self, btc_price: float, error: str) -> IntegratedTradingSignal:
        """Create fallback signal for error conditions."""
        return IntegratedTradingSignal(
            signal_id=f"FALLBACK_{int(time.time())}",
            btc_price=btc_price,
            word_mapping={"error": error},
            glyph_states=[],
            ferris_data={"error": error},
            ghost_route="noop",
            recommended_action="MONITOR_MARKET",
            confidence_score=0.0,
            profit_potential=0.0,
            risk_assessment={"overall_risk": 1.0},
            execution_timeframe=TradingTimeframe.FERRIS_CYCLE
        )
    
    def _update_metrics(self, signal: IntegratedTradingSignal) -> None:
        """Update performance metrics."""
        self.metrics["total_signals_processed"] += 1
        
        if signal.glyph_states:
            avg_entropy = np.mean([g.entropy_value for g in signal.glyph_states])
            self.metrics["average_glyph_entropy"] = (
                self.metrics["average_glyph_entropy"] * 0.9 + avg_entropy * 0.1
            )
        
        # Update portal utilization
        for glyph in signal.glyph_states:
            portal_type = glyph.portal_target.value.replace("_portal", "")
            if portal_type in self.metrics["portal_utilization"]:
                self.metrics["portal_utilization"][portal_type] += 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "controller_status": "operational" if CORE_COMPONENTS_AVAILABLE else "limited",
            "active_signals": len(self.active_signals),
            "glyph_registry_size": len(self.glyph_registry),
            "price_history_length": len(self.price_word_history),
            "metrics": self.metrics.copy(),
            "last_update": time.time()
        }


class PortalLoadBalancer:
    """Load balancer for routing glyphs to appropriate portals."""
    
    def __init__(self):
        self.portal_loads = {
            GlyphPortalType.CPU_PORTAL: 0,
            GlyphPortalType.GPU_PORTAL: 0,
            GlyphPortalType.COLDBASE_PORTAL: 0,
            GlyphPortalType.HYBRID_PORTAL: 0
        }
    
    def get_optimal_portal(self, glyph_hash: str) -> GlyphPortalType:
        """Get optimal portal based on current load."""
        # Find portal with minimum load
        min_load_portal = min(self.portal_loads.items(), key=lambda x: x[1])[0]
        
        # Increment load
        self.portal_loads[min_load_portal] += 1
        
        return min_load_portal
    
    def release_portal_load(self, portal: GlyphPortalType) -> None:
        """Release load from portal."""
        if self.portal_loads[portal] > 0:
            self.portal_loads[portal] -= 1


# Global instance for integration
integrated_controller = IntegratedFerrisGlyphController()

# Export functions for external access
def process_btc_cycle(btc_price: float, timeframe: TradingTimeframe = TradingTimeframe.FERRIS_CYCLE) -> IntegratedTradingSignal:
    """Process BTC price cycle through integrated system."""
    return integrated_controller.process_btc_price_cycle(btc_price, timeframe)

def get_controller_status() -> Dict[str, Any]:
    """Get integrated controller status."""
    return integrated_controller.get_system_status()

# Export all key components
__all__ = [
    "IntegratedFerrisGlyphController",
    "GlyphPortalType",
    "TradingTimeframe", 
    "GlyphState",
    "IntegratedTradingSignal",
    "integrated_controller",
    "process_btc_cycle",
    "get_controller_status"
] 