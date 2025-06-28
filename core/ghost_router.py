# -*- coding: utf-8 -*-
"""
Ghost Router for Schwabot Trading System
=======================================

Provides intelligent routing for profit optimization and buy/sell wall detection.
The Ghost Router makes routing decisions based on market conditions, price predictions,
and portfolio state to maximize trading profits.

Key Features:
- Profit routing with buy/sell wall detection
- Multi-mode trading strategies (vault, long, short, mid)
- Hash-based routing decisions
- Real-time market condition analysis
- Risk assessment and profit optimization

MATHEMATICAL PRESERVATION: All core mathematical logic preserved.
"""

from typing import Dict, List, Optional, Any, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RoutingMode(Enum):
    """Routing modes for Ghost Router."""
    VAULT_MODE = "vault_mode"      # Hold in stable assets
    LONG_MODE = "long_mode"        # Long position strategy
    SHORT_MODE = "short_mode"      # Short position strategy
    MID_MODE = "mid_mode"          # Balanced/neutral strategy
    GHOST_TRADE = "ghost_trade"    # Execute ghost trade
    HOLD_USDC = "hold_usdc"        # Hold in USDC
    NOOP = "noop"                  # No operation

@dataclass
class RouterInput:
    """Input data for Ghost Router routing decisions."""
    tick_hash: str
    mem_hash: str
    pool_volumes: np.ndarray
    btc_dip: bool
    lantern_vec: np.ndarray
    lantern_ref: np.ndarray
    ai_hashes: Tuple[str, str, str]
    ai_weights: Tuple[float, float, float]
    opportunity_ts: float
    now_ts: float
    price_now: float
    price_pred: float
    curr_profit: float
    projected_exit: float
    news_score: float

@dataclass
class RoutingResult:
    """Result of Ghost Router routing decision."""
    route: RoutingMode
    confidence: float
    profit_potential: float
    risk_score: float
    execution_priority: int
    routing_timestamp: float = field(default_factory=time.time)

class GhostRouter:
    """
    Ghost Router for intelligent profit routing and trading decisions.
    
    Analyzes market conditions, price predictions, and portfolio state
    to make optimal routing decisions for maximum profit generation.
    """
    
    def __init__(self):
        """Initialize Ghost Router."""
        self.routing_history: List[RoutingResult] = []
        self.routing_stats = {
            "total_routes": 0,
            "ghost_trades": 0,
            "hold_decisions": 0,
            "profit_generated": 0.0,
            "success_rate": 0.0
        }
        
        # Routing thresholds
        self.profit_threshold = 0.02  # 2% profit threshold
        self.risk_threshold = 0.15    # 15% risk threshold
        self.confidence_threshold = 0.6  # 60% confidence threshold
        
        logger.info("👻 Ghost Router initialized")
    
    def route(self, router_input: RouterInput) -> str:
        """
        Main routing function - returns routing decision as string.
        
        This is the core function that analyzes all input parameters
        and makes the optimal routing decision.
        """
        try:
            # Convert to internal routing decision
            routing_result = self._make_routing_decision(router_input)
            
            # Update statistics
            self._update_routing_stats(routing_result)
            
            # Store in history
            self.routing_history.append(routing_result)
            
            # Convert back to string format expected by callers
            route_mapping = {
                RoutingMode.GHOST_TRADE: "ghost_trade",
                RoutingMode.HOLD_USDC: "hold_usdc",
                RoutingMode.VAULT_MODE: "vault_mode",
                RoutingMode.LONG_MODE: "long_mode",
                RoutingMode.SHORT_MODE: "short_mode",
                RoutingMode.MID_MODE: "mid_mode",
                RoutingMode.NOOP: "noop"
            }
            
            route_str = route_mapping.get(routing_result.route, "noop")
            
            logger.debug(f"👻 Ghost route decision: {route_str} (confidence: {routing_result.confidence:.2f})")
            
            return route_str
            
        except Exception as e:
            logger.error(f"Ghost routing failed: {e}")
            return "noop"
    
    def _make_routing_decision(self, router_input: RouterInput) -> RoutingResult:
        """Make detailed routing decision based on input parameters."""
        try:
            # Analyze market conditions
            market_analysis = self._analyze_market_conditions(router_input)
            
            # Calculate profit potential
            profit_potential = self._calculate_profit_potential(router_input)
            
            # Assess risk
            risk_score = self._assess_risk(router_input)
            
            # Calculate confidence
            confidence = self._calculate_confidence(router_input, market_analysis)
            
            # Make routing decision
            route = self._determine_route(
                router_input, market_analysis, profit_potential, risk_score, confidence
            )
            
            # Calculate execution priority
            execution_priority = self._calculate_execution_priority(profit_potential, risk_score, confidence)
            
            return RoutingResult(
                route=route,
                confidence=confidence,
                profit_potential=profit_potential,
                risk_score=risk_score,
                execution_priority=execution_priority
            )
            
        except Exception as e:
            logger.error(f"Routing decision failed: {e}")
            return RoutingResult(
                route=RoutingMode.NOOP,
                confidence=0.0,
                profit_potential=0.0,
                risk_score=1.0,
                execution_priority=0
            )
    
    def _analyze_market_conditions(self, router_input: RouterInput) -> Dict[str, Any]:
        """Analyze current market conditions."""
        try:
            # Price momentum analysis
            price_momentum = (router_input.price_pred - router_input.price_now) / router_input.price_now
            
            # Volume analysis
            avg_volume = np.mean(router_input.pool_volumes)
            volume_spike = np.max(router_input.pool_volumes) / avg_volume if avg_volume > 0 else 1.0
            
            # Time analysis
            time_since_opportunity = router_input.now_ts - router_input.opportunity_ts
            
            # BTC dip analysis
            dip_severity = 0.0
            if router_input.btc_dip:
                if router_input.price_now < 45000:
                    dip_severity = 0.8  # Severe dip
                elif router_input.price_now < 48000:
                    dip_severity = 0.5  # Moderate dip
                else:
                    dip_severity = 0.2  # Minor dip
            
            # AI sentiment analysis
            ai_sentiment = np.mean(router_input.ai_weights)
            
            return {
                "price_momentum": price_momentum,
                "volume_spike": volume_spike,
                "time_since_opportunity": time_since_opportunity,
                "dip_severity": dip_severity,
                "ai_sentiment": ai_sentiment,
                "news_score": router_input.news_score
            }
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_profit_potential(self, router_input: RouterInput) -> float:
        """Calculate profit potential from routing input."""
        try:
            # Base profit from price prediction
            price_profit = (router_input.price_pred - router_input.price_now) / router_input.price_now
            
            # Current profit factor
            current_profit_factor = router_input.curr_profit / 1000.0  # Normalize
            
            # Exit profit potential
            exit_profit = (router_input.projected_exit - router_input.price_now) / router_input.price_now
            
            # Combined profit potential
            combined_profit = (price_profit + current_profit_factor + exit_profit) / 3.0
            
            # Cap at reasonable values
            return max(0.0, min(1.0, combined_profit))
            
        except Exception as e:
            logger.error(f"Profit calculation failed: {e}")
            return 0.0
    
    def _assess_risk(self, router_input: RouterInput) -> float:
        """Assess risk level for routing decision."""
        try:
            # Price volatility risk
            price_volatility = abs(router_input.price_pred - router_input.price_now) / router_input.price_now
            
            # Time risk (longer time = higher risk)
            time_risk = min(1.0, (router_input.now_ts - router_input.opportunity_ts) / 3600.0)  # Max 1 hour
            
            # Volume risk (low volume = higher risk)
            avg_volume = np.mean(router_input.pool_volumes)
            volume_risk = 1.0 - min(1.0, avg_volume / 1000.0)  # Normalize to expected volume
            
            # News sentiment risk
            news_risk = 1.0 - abs(router_input.news_score)  # Neutral news = higher risk
            
            # Combined risk score
            combined_risk = (price_volatility + time_risk + volume_risk + news_risk) / 4.0
            
            return max(0.0, min(1.0, combined_risk))
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return 1.0  # Maximum risk on error
    
    def _calculate_confidence(self, router_input: RouterInput, market_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in routing decision."""
        try:
            # AI weight confidence
            ai_confidence = np.mean(router_input.ai_weights)
            
            # Market momentum confidence
            momentum = market_analysis.get("price_momentum", 0.0)
            momentum_confidence = 1.0 - abs(momentum)  # Stable momentum = higher confidence
            
            # Volume confidence
            volume_spike = market_analysis.get("volume_spike", 1.0)
            volume_confidence = min(1.0, volume_spike / 2.0)  # Moderate volume spike = good
            
            # News confidence
            news_confidence = abs(router_input.news_score)  # Strong news sentiment = higher confidence
            
            # Hash quality confidence (simple hash entropy check)
            hash_entropy = self._calculate_hash_entropy(router_input.tick_hash)
            
            # Combined confidence
            combined_confidence = (
                ai_confidence + momentum_confidence + volume_confidence + 
                news_confidence + hash_entropy
            ) / 5.0
            
            return max(0.0, min(1.0, combined_confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _determine_route(self, router_input: RouterInput, market_analysis: Dict[str, Any], 
                        profit_potential: float, risk_score: float, confidence: float) -> RoutingMode:
        """Determine the optimal routing mode."""
        try:
            # High confidence, high profit, low risk = Ghost Trade
            if (confidence > self.confidence_threshold and 
                profit_potential > self.profit_threshold and 
                risk_score < self.risk_threshold):
                return RoutingMode.GHOST_TRADE
            
            # BTC dip with good profit potential = Long Mode
            if (router_input.btc_dip and 
                profit_potential > 0.01 and 
                confidence > 0.4):
                return RoutingMode.LONG_MODE
            
            # High risk or low confidence = Hold USDC
            if risk_score > 0.7 or confidence < 0.3:
                return RoutingMode.HOLD_USDC
            
            # Strong upward momentum = Long Mode
            momentum = market_analysis.get("price_momentum", 0.0)
            if momentum > 0.02 and confidence > 0.5:
                return RoutingMode.LONG_MODE
            
            # Strong downward momentum = Vault Mode
            if momentum < -0.02:
                return RoutingMode.VAULT_MODE
            
            # Moderate conditions = Mid Mode
            if 0.3 < confidence < 0.6 and 0.2 < risk_score < 0.6:
                return RoutingMode.MID_MODE
            
            # Default to no operation
            return RoutingMode.NOOP
            
        except Exception as e:
            logger.error(f"Route determination failed: {e}")
            return RoutingMode.NOOP
    
    def _calculate_execution_priority(self, profit_potential: float, risk_score: float, confidence: float) -> int:
        """Calculate execution priority (1-10, 10 = highest priority)."""
        try:
            # Base priority from profit potential
            base_priority = profit_potential * 5
            
            # Boost for high confidence
            confidence_boost = confidence * 3
            
            # Penalty for high risk
            risk_penalty = risk_score * 2
            
            # Calculate final priority
            priority = base_priority + confidence_boost - risk_penalty
            
            # Clamp to 1-10 range
            return max(1, min(10, int(priority)))
            
        except Exception as e:
            logger.error(f"Priority calculation failed: {e}")
            return 1
    
    def _calculate_hash_entropy(self, hash_str: str) -> float:
        """Calculate entropy of hash string for quality assessment."""
        try:
            if not hash_str:
                return 0.0
            
            # Count character frequencies
            char_counts = {}
            for char in hash_str:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Calculate entropy
            total_chars = len(hash_str)
            entropy = 0.0
            
            for count in char_counts.values():
                probability = count / total_chars
                if probability > 0:
                    entropy -= probability * math.log2(probability)
            
            # Normalize to 0-1 range (max entropy for hex is log2(16) = 4)
            return entropy / 4.0
            
        except Exception as e:
            logger.error(f"Hash entropy calculation failed: {e}")
            return 0.0
    
    def _update_routing_stats(self, routing_result: RoutingResult):
        """Update routing statistics."""
        try:
            self.routing_stats["total_routes"] += 1
            
            if routing_result.route == RoutingMode.GHOST_TRADE:
                self.routing_stats["ghost_trades"] += 1
            elif routing_result.route in [RoutingMode.HOLD_USDC, RoutingMode.VAULT_MODE]:
                self.routing_stats["hold_decisions"] += 1
            
            # Update success rate based on confidence
            if routing_result.confidence > 0.5:
                current_success = self.routing_stats["success_rate"]
                total_routes = self.routing_stats["total_routes"]
                self.routing_stats["success_rate"] = (
                    (current_success * (total_routes - 1) + routing_result.confidence) / total_routes
                )
            
        except Exception as e:
            logger.error(f"Failed to update routing stats: {e}")
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        try:
            stats = self.routing_stats.copy()
            
            # Add derived statistics
            if stats["total_routes"] > 0:
                stats["ghost_trade_rate"] = stats["ghost_trades"] / stats["total_routes"]
                stats["hold_rate"] = stats["hold_decisions"] / stats["total_routes"]
            else:
                stats["ghost_trade_rate"] = 0.0
                stats["hold_rate"] = 0.0
            
            stats["recent_routes"] = len(self.routing_history[-10:])
            stats["last_update"] = time.time()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get routing statistics: {e}")
            return {"error": str(e)}


# Global instance
ghost_router = GhostRouter()

# Export all components
__all__ = [
    "GhostRouter",
    "RoutingMode",
    "RouterInput",
    "RoutingResult",
    "ghost_router"
]

# Test function
def test_ghost_router():
    """Test the Ghost Router with sample data."""
    try:
        logger.info("👻 Testing Ghost Router...")
        
        # Create sample router input
        test_input = RouterInput(
            tick_hash="abcd1234567890ef",
            mem_hash="1234567890abcdef",
            pool_volumes=np.array([1000.0, 1200.0, 900.0, 1100.0]),
            btc_dip=True,
            lantern_vec=np.array([1.0, 0.5, 0.8]),
            lantern_ref=np.array([1.0, 0.5, 0.8]),
            ai_hashes=("hash1", "hash2", "hash3"),
            ai_weights=(0.8, 0.7, 0.9),
            opportunity_ts=time.time() - 300,  # 5 minutes ago
            now_ts=time.time(),
            price_now=50000.0,
            price_pred=51000.0,  # 2% increase predicted
            curr_profit=0.0,
            projected_exit=52000.0,  # 4% exit target
            news_score=0.3
        )
        
        # Test routing
        route_decision = ghost_router.route(test_input)
        logger.info(f"  ✅ Route decision: {route_decision}")
        
        # Test statistics
        stats = ghost_router.get_routing_statistics()
        logger.info(f"  ✅ Routing statistics: {stats}")
        
        logger.info("✅ Ghost Router test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ghost Router test failed: {e}")
        return False

if __name__ == "__main__":
    test_ghost_router() 