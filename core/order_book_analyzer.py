#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order Book Analyzer Module
===========================
Provides order book analyzer functionality for the Schwabot trading system.

Mathematical Core:
Imbalance(t) = (ΣBids_t - ΣAsks_t) / (ΣBids_t + ΣAsks_t)
- Signal Input: Used to bias strategy towards aggressiveness or stealth
- Detects buy/sell walls, liquidity cliffs, and depth imbalances

This module analyzes bid-ask spread, depth imbalance, wall formations, and
liquidity cliffs to provide trading signals.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import json

logger = logging.getLogger(__name__)

# Import mathematical infrastructure
try:
    from core.unified_mathematical_bridge import UnifiedMathematicalBridge
    from core.unified_mathematical_integration_methods import UnifiedMathematicalIntegrationMethods
    from core.unified_mathematical_performance_monitor import UnifiedMathematicalPerformanceMonitor
    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Mathematical infrastructure not available - using fallback")


class WallType(Enum):
    """Types of order book walls."""
    BUY_WALL = "buy_wall"
    SELL_WALL = "sell_wall"
    LIQUIDITY_CLIFF = "liquidity_cliff"
    SUPPORT_LEVEL = "support_level"
    RESISTANCE_LEVEL = "resistance_level"


class MarketStructure(Enum):
    """Market structure classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"
    CONSOLIDATING = "consolidating"


@dataclass
class OrderBookLevel:
    """Single order book level."""
    price: float
    quantity: float
    side: str  # 'bid' or 'ask'
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderBookWall:
    """Detected order book wall."""
    wall_type: WallType
    price_level: float
    total_quantity: float
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    mathematical_signature: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class LiquidityAnalysis:
    """Liquidity analysis results."""
    bid_liquidity: float
    ask_liquidity: float
    imbalance_ratio: float
    spread: float
    depth_score: float
    mathematical_signature: str = ""


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    timestamp: float
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    mathematical_analysis: Dict[str, Any] = field(default_factory=dict)
    market_structure: MarketStructure = MarketStructure.NEUTRAL


@dataclass
class OrderBookAnalyzerConfig:
    """Configuration for order book analyzer."""
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False
    wall_detection_threshold: float = 0.1  # Minimum wall strength
    imbalance_threshold: float = 0.05  # Minimum imbalance to consider
    max_depth_levels: int = 20
    mathematical_analysis_enabled: bool = True


class OrderBookAnalyzer:
    """
    Order Book Analyzer System
    
    Implements mathematical analysis of order book data:
    Imbalance(t) = (ΣBids_t - ΣAsks_t) / (ΣBids_t + ΣAsks_t)
    
    Analyzes bid-ask spread, depth imbalance, wall formations, and
    liquidity cliffs to provide trading signals.
    """
    
    def __init__(self, config: Optional[OrderBookAnalyzerConfig] = None):
        """Initialize the order book analyzer system."""
        self.config = config or OrderBookAnalyzerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Order book data
        self.order_book_snapshots: Dict[str, OrderBookSnapshot] = {}
        self.wall_history: Dict[str, List[OrderBookWall]] = {}
        self.liquidity_history: Dict[str, List[LiquidityAnalysis]] = {}
        
        # Mathematical infrastructure
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_bridge = UnifiedMathematicalBridge()
            self.math_integration = UnifiedMathematicalIntegrationMethods()
            self.math_monitor = UnifiedMathematicalPerformanceMonitor()
        else:
            self.math_bridge = None
            self.math_integration = None
            self.math_monitor = None
        
        # Performance tracking
        self.performance_metrics = {
            'snapshots_analyzed': 0,
            'walls_detected': 0,
            'liquidity_analyses': 0,
            'mathematical_analyses': 0,
            'average_processing_time': 0.0
        }
        
        # System state
        self.initialized = False
        self.active = False
        
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the order book analyzer system."""
        try:
            self.logger.info("Initializing Order Book Analyzer System")
            self.initialized = True
            self.logger.info("✅ Order Book Analyzer System initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing Order Book Analyzer System: {e}")
            self.initialized = False
    
    def analyze_order_book(self, symbol: str, order_book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze order book data and return comprehensive analysis."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return {}
        
        try:
            start_time = time.time()
            
            # Parse order book data
            snapshot = self._parse_order_book_data(symbol, order_book_data)
            
            # Perform mathematical analysis
            mathematical_analysis = self._perform_mathematical_analysis(snapshot)
            
            # Detect walls
            walls = self._detect_walls(snapshot)
            
            # Analyze liquidity
            liquidity_analysis = self._analyze_liquidity(snapshot)
            
            # Classify market structure
            market_structure = self._classify_market_structure(snapshot, walls, liquidity_analysis)
            
            # Update snapshot with analysis
            snapshot.mathematical_analysis = mathematical_analysis
            snapshot.market_structure = market_structure
            
            # Store snapshot
            self.order_book_snapshots[symbol] = snapshot
            
            # Update history
            if symbol not in self.wall_history:
                self.wall_history[symbol] = []
            self.wall_history[symbol].extend(walls)
            
            if symbol not in self.liquidity_history:
                self.liquidity_history[symbol] = []
            self.liquidity_history[symbol].append(liquidity_analysis)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['snapshots_analyzed'] += 1
            self.performance_metrics['walls_detected'] += len(walls)
            self.performance_metrics['liquidity_analyses'] += 1
            self.performance_metrics['mathematical_analyses'] += 1
            
            # Update average processing time
            current_avg = self.performance_metrics['average_processing_time']
            total_analyses = self.performance_metrics['snapshots_analyzed']
            self.performance_metrics['average_processing_time'] = (
                (current_avg * (total_analyses - 1) + processing_time) / total_analyses
            )
            
            # Return comprehensive analysis
            return {
                'symbol': symbol,
                'timestamp': snapshot.timestamp,
                'market_structure': market_structure.value,
                'walls': [self._wall_to_dict(wall) for wall in walls],
                'liquidity_analysis': self._liquidity_to_dict(liquidity_analysis),
                'mathematical_analysis': mathematical_analysis,
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing order book for {symbol}: {e}")
            return {}
    
    def _parse_order_book_data(self, symbol: str, order_book_data: Dict[str, Any]) -> OrderBookSnapshot:
        """Parse raw order book data into structured format."""
        try:
            timestamp = time.time()
            bids = []
            asks = []
            
            # Parse bids
            if 'bids' in order_book_data:
                for bid in order_book_data['bids'][:self.config.max_depth_levels]:
                    if len(bid) >= 2:
                        bids.append(OrderBookLevel(
                            price=float(bid[0]),
                            quantity=float(bid[1]),
                            side='bid',
                            timestamp=timestamp
                        ))
            
            # Parse asks
            if 'asks' in order_book_data:
                for ask in order_book_data['asks'][:self.config.max_depth_levels]:
                    if len(ask) >= 2:
                        asks.append(OrderBookLevel(
                            price=float(ask[0]),
                            quantity=float(ask[1]),
                            side='ask',
                            timestamp=timestamp
                        ))
            
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=timestamp,
                bids=bids,
                asks=asks
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing order book data: {e}")
            return OrderBookSnapshot(symbol=symbol, timestamp=time.time(), bids=[], asks=[])
    
    def _detect_walls(self, snapshot: OrderBookSnapshot) -> List[OrderBookWall]:
        """Detect buy and sell walls in the order book."""
        try:
            walls = []
            
            # Detect buy walls (large bid clusters)
            buy_walls = self._detect_buy_walls(snapshot.bids)
            walls.extend(buy_walls)
            
            # Detect sell walls (large ask clusters)
            sell_walls = self._detect_sell_walls(snapshot.asks)
            walls.extend(sell_walls)
            
            # Detect liquidity cliffs
            liquidity_cliffs = self._detect_liquidity_cliffs(snapshot)
            walls.extend(liquidity_cliffs)
            
            return walls
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting walls: {e}")
            return []
    
    def _detect_buy_walls(self, bids: List[OrderBookLevel]) -> List[OrderBookWall]:
        """Detect buy walls in bid side."""
        try:
            walls = []
            
            if len(bids) < 3:
                return walls
            
            # Calculate average quantity for threshold
            quantities = [bid.quantity for bid in bids]
            avg_quantity = np.mean(quantities)
            threshold = avg_quantity * 2.0  # Wall threshold
            
            # Group consecutive levels with high quantity
            current_wall_quantity = 0.0
            wall_start_price = 0.0
            wall_levels = 0
            
            for i, bid in enumerate(bids):
                if bid.quantity > threshold:
                    if current_wall_quantity == 0:
                        wall_start_price = bid.price
                        current_wall_quantity = bid.quantity
                        wall_levels = 1
                    else:
                        current_wall_quantity += bid.quantity
                        wall_levels += 1
                else:
                    # End of potential wall
                    if current_wall_quantity > 0 and wall_levels >= 2:
                        # Create wall
                        wall_price = wall_start_price
                        wall_strength = min(current_wall_quantity / (avg_quantity * 5), 1.0)
                        wall_confidence = min(wall_levels / 5, 1.0)
                        
                        wall = OrderBookWall(
                            wall_type=WallType.BUY_WALL,
                            price_level=wall_price,
                            total_quantity=current_wall_quantity,
                            strength=wall_strength,
                            confidence=wall_confidence,
                            mathematical_signature=self._create_wall_signature(
                                wall_price, current_wall_quantity, wall_strength, "buy"
                            )
                        )
                        walls.append(wall)
                    
                    current_wall_quantity = 0.0
                    wall_levels = 0
            
            return walls
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting buy walls: {e}")
            return []
    
    def _detect_sell_walls(self, asks: List[OrderBookLevel]) -> List[OrderBookWall]:
        """Detect sell walls in ask side."""
        try:
            walls = []
            
            if len(asks) < 3:
                return walls
            
            # Calculate average quantity for threshold
            quantities = [ask.quantity for ask in asks]
            avg_quantity = np.mean(quantities)
            threshold = avg_quantity * 2.0  # Wall threshold
            
            # Group consecutive levels with high quantity
            current_wall_quantity = 0.0
            wall_start_price = 0.0
            wall_levels = 0
            
            for i, ask in enumerate(asks):
                if ask.quantity > threshold:
                    if current_wall_quantity == 0:
                        wall_start_price = ask.price
                        current_wall_quantity = ask.quantity
                        wall_levels = 1
                    else:
                        current_wall_quantity += ask.quantity
                        wall_levels += 1
                else:
                    # End of potential wall
                    if current_wall_quantity > 0 and wall_levels >= 2:
                        # Create wall
                        wall_price = wall_start_price
                        wall_strength = min(current_wall_quantity / (avg_quantity * 5), 1.0)
                        wall_confidence = min(wall_levels / 5, 1.0)
                        
                        wall = OrderBookWall(
                            wall_type=WallType.SELL_WALL,
                            price_level=wall_price,
                            total_quantity=current_wall_quantity,
                            strength=wall_strength,
                            confidence=wall_confidence,
                            mathematical_signature=self._create_wall_signature(
                                wall_price, current_wall_quantity, wall_strength, "sell"
                            )
                        )
                        walls.append(wall)
                    
                    current_wall_quantity = 0.0
                    wall_levels = 0
            
            return walls
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting sell walls: {e}")
            return []
    
    def _detect_liquidity_cliffs(self, snapshot: OrderBookSnapshot) -> List[OrderBookWall]:
        """Detect liquidity cliffs (sudden drops in liquidity)."""
        try:
            cliffs = []
            
            # Analyze bid side for cliffs
            if len(snapshot.bids) >= 5:
                bid_quantities = [bid.quantity for bid in snapshot.bids]
                for i in range(1, len(bid_quantities) - 1):
                    current_qty = bid_quantities[i]
                    next_qty = bid_quantities[i + 1]
                    
                    # Check for significant drop
                    if next_qty < current_qty * 0.3:  # 70% drop
                        cliff = OrderBookWall(
                            wall_type=WallType.LIQUIDITY_CLIFF,
                            price_level=snapshot.bids[i].price,
                            total_quantity=current_qty,
                            strength=1.0 - (next_qty / current_qty),
                            confidence=0.8,
                            mathematical_signature=self._create_wall_signature(
                                snapshot.bids[i].price, current_qty, 1.0 - (next_qty / current_qty), "cliff"
                            )
                        )
                        cliffs.append(cliff)
            
            # Analyze ask side for cliffs
            if len(snapshot.asks) >= 5:
                ask_quantities = [ask.quantity for ask in snapshot.asks]
                for i in range(1, len(ask_quantities) - 1):
                    current_qty = ask_quantities[i]
                    next_qty = ask_quantities[i + 1]
                    
                    # Check for significant drop
                    if next_qty < current_qty * 0.3:  # 70% drop
                        cliff = OrderBookWall(
                            wall_type=WallType.LIQUIDITY_CLIFF,
                            price_level=snapshot.asks[i].price,
                            total_quantity=current_qty,
                            strength=1.0 - (next_qty / current_qty),
                            confidence=0.8,
                            mathematical_signature=self._create_wall_signature(
                                snapshot.asks[i].price, current_qty, 1.0 - (next_qty / current_qty), "cliff"
                            )
                        )
                        cliffs.append(cliff)
            
            return cliffs
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting liquidity cliffs: {e}")
            return []
    
    def _analyze_liquidity(self, snapshot: OrderBookSnapshot) -> LiquidityAnalysis:
        """Analyze liquidity and calculate imbalance ratio."""
        try:
            # Calculate total bid and ask liquidity
            bid_liquidity = sum(bid.quantity for bid in snapshot.bids)
            ask_liquidity = sum(ask.quantity for ask in snapshot.asks)
            
            # Calculate imbalance ratio: (ΣBids - ΣAsks) / (ΣBids + ΣAsks)
            total_liquidity = bid_liquidity + ask_liquidity
            if total_liquidity > 0:
                imbalance_ratio = (bid_liquidity - ask_liquidity) / total_liquidity
            else:
                imbalance_ratio = 0.0
            
            # Calculate spread
            if snapshot.bids and snapshot.asks:
                best_bid = max(bid.price for bid in snapshot.bids)
                best_ask = min(ask.price for ask in snapshot.asks)
                spread = best_ask - best_bid
            else:
                spread = 0.0
            
            # Calculate depth score (how deep the order book is)
            depth_score = min(len(snapshot.bids) + len(snapshot.asks), 100) / 100.0
            
            # Create mathematical signature
            mathematical_signature = self._create_liquidity_signature(
                bid_liquidity, ask_liquidity, imbalance_ratio, spread, depth_score
            )
            
            return LiquidityAnalysis(
                bid_liquidity=bid_liquidity,
                ask_liquidity=ask_liquidity,
                imbalance_ratio=imbalance_ratio,
                spread=spread,
                depth_score=depth_score,
                mathematical_signature=mathematical_signature
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing liquidity: {e}")
            return LiquidityAnalysis(
                bid_liquidity=0.0,
                ask_liquidity=0.0,
                imbalance_ratio=0.0,
                spread=0.0,
                depth_score=0.0
            )
    
    def _perform_mathematical_analysis(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Perform mathematical analysis on order book data."""
        try:
            if not self.math_bridge:
                return {}
            
            # Prepare data for mathematical analysis
            order_book_data = {
                'symbol': snapshot.symbol,
                'timestamp': snapshot.timestamp,
                'bid_count': len(snapshot.bids),
                'ask_count': len(snapshot.asks),
                'total_bid_quantity': sum(bid.quantity for bid in snapshot.bids),
                'total_ask_quantity': sum(ask.quantity for ask in snapshot.asks),
                'bid_prices': [bid.price for bid in snapshot.bids],
                'ask_prices': [ask.price for ask in snapshot.asks],
                'bid_quantities': [bid.quantity for bid in snapshot.bids],
                'ask_quantities': [ask.quantity for ask in snapshot.asks]
            }
            
            # Perform mathematical integration
            result = self.math_bridge.integrate_all_mathematical_systems(
                order_book_data, {}
            )
            
            return {
                'confidence': result.overall_confidence,
                'connections': len(result.connections),
                'performance_metrics': result.performance_metrics,
                'mathematical_signature': result.mathematical_signature
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error performing mathematical analysis: {e}")
            return {}
    
    def _classify_market_structure(self, snapshot: OrderBookSnapshot, 
                                 walls: List[OrderBookWall], 
                                 liquidity_analysis: LiquidityAnalysis) -> MarketStructure:
        """Classify market structure based on order book analysis."""
        try:
            # Analyze imbalance
            imbalance = abs(liquidity_analysis.imbalance_ratio)
            
            # Count wall types
            buy_walls = len([w for w in walls if w.wall_type == WallType.BUY_WALL])
            sell_walls = len([w for w in walls if w.wall_type == WallType.SELL_WALL])
            
            # Analyze spread
            spread_ratio = liquidity_analysis.spread / liquidity_analysis.bid_liquidity if liquidity_analysis.bid_liquidity > 0 else 0
            
            # Classify based on analysis
            if imbalance > 0.1:  # Strong imbalance
                if liquidity_analysis.imbalance_ratio > 0:
                    return MarketStructure.BULLISH
                else:
                    return MarketStructure.BEARISH
            elif buy_walls > sell_walls + 1:  # More buy walls
                return MarketStructure.BULLISH
            elif sell_walls > buy_walls + 1:  # More sell walls
                return MarketStructure.BEARISH
            elif spread_ratio > 0.01:  # Wide spread
                return MarketStructure.VOLATILE
            else:
                return MarketStructure.NEUTRAL
                
        except Exception as e:
            self.logger.error(f"❌ Error classifying market structure: {e}")
            return MarketStructure.NEUTRAL
    
    def _create_wall_signature(self, price: float, quantity: float, 
                             strength: float, wall_type: str) -> str:
        """Create mathematical signature for wall."""
        try:
            signature_components = [
                f"P:{price:.6f}",
                f"Q:{quantity:.6f}",
                f"S:{strength:.3f}",
                f"T:{wall_type}"
            ]
            return "|".join(signature_components)
        except Exception as e:
            self.logger.error(f"❌ Error creating wall signature: {e}")
            return ""
    
    def _create_liquidity_signature(self, bid_liquidity: float, ask_liquidity: float,
                                  imbalance: float, spread: float, depth: float) -> str:
        """Create mathematical signature for liquidity analysis."""
        try:
            signature_components = [
                f"B:{bid_liquidity:.6f}",
                f"A:{ask_liquidity:.6f}",
                f"I:{imbalance:.6f}",
                f"S:{spread:.6f}",
                f"D:{depth:.3f}"
            ]
            return "|".join(signature_components)
        except Exception as e:
            self.logger.error(f"❌ Error creating liquidity signature: {e}")
            return ""
    
    def _wall_to_dict(self, wall: OrderBookWall) -> Dict[str, Any]:
        """Convert wall to dictionary."""
        return {
            'type': wall.wall_type.value,
            'price_level': wall.price_level,
            'total_quantity': wall.total_quantity,
            'strength': wall.strength,
            'confidence': wall.confidence,
            'mathematical_signature': wall.mathematical_signature,
            'timestamp': wall.timestamp
        }
    
    def _liquidity_to_dict(self, liquidity: LiquidityAnalysis) -> Dict[str, Any]:
        """Convert liquidity analysis to dictionary."""
        return {
            'bid_liquidity': liquidity.bid_liquidity,
            'ask_liquidity': liquidity.ask_liquidity,
            'imbalance_ratio': liquidity.imbalance_ratio,
            'spread': liquidity.spread,
            'depth_score': liquidity.depth_score,
            'mathematical_signature': liquidity.mathematical_signature
        }
    
    def get_order_book_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest order book analysis for a symbol."""
        snapshot = self.order_book_snapshots.get(symbol)
        if not snapshot:
            return None
        
        return {
            'symbol': symbol,
            'timestamp': snapshot.timestamp,
            'market_structure': snapshot.market_structure.value,
            'bid_count': len(snapshot.bids),
            'ask_count': len(snapshot.asks),
            'mathematical_analysis': snapshot.mathematical_analysis
        }
    
    def get_wall_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Get wall detection history for a symbol."""
        walls = self.wall_history.get(symbol, [])
        return [self._wall_to_dict(wall) for wall in walls[-10:]]  # Last 10 walls
    
    def get_liquidity_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Get liquidity analysis history for a symbol."""
        liquidity = self.liquidity_history.get(symbol, [])
        return [self._liquidity_to_dict(liq) for liq in liquidity[-10:]]  # Last 10 analyses
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return self.performance_metrics.copy()
    
    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            self.logger.info("✅ Order Book Analyzer System activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating Order Book Analyzer System: {e}")
            return False
    
    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info("✅ Order Book Analyzer System deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating Order Book Analyzer System: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'symbols_tracked': len(self.order_book_snapshots),
            'performance_metrics': self.performance_metrics,
            'config': {
                'enabled': self.config.enabled,
                'wall_detection_threshold': self.config.wall_detection_threshold,
                'imbalance_threshold': self.config.imbalance_threshold,
                'mathematical_analysis_enabled': self.config.mathematical_analysis_enabled
            }
        }


def create_order_book_analyzer(config: Optional[OrderBookAnalyzerConfig] = None) -> OrderBookAnalyzer:
    """Factory function to create OrderBookAnalyzer instance."""
    return OrderBookAnalyzer(config)


def main():
    """Main function for testing."""
    # Create configuration
    config = OrderBookAnalyzerConfig(
        enabled=True,
        debug=True,
        wall_detection_threshold=0.1,
        imbalance_threshold=0.05,
        mathematical_analysis_enabled=True
    )
    
    # Create analyzer
    analyzer = create_order_book_analyzer(config)
    
    # Activate system
    analyzer.activate()
    
    # Sample order book data
    sample_order_book = {
        'bids': [
            ['50000.0', '1.5'],
            ['49999.0', '2.0'],
            ['49998.0', '1.8'],
            ['49997.0', '0.5'],
            ['49996.0', '0.3']
        ],
        'asks': [
            ['50001.0', '1.2'],
            ['50002.0', '2.5'],
            ['50003.0', '1.0'],
            ['50004.0', '0.8'],
            ['50005.0', '0.6']
        ]
    }
    
    # Analyze order book
    result = analyzer.analyze_order_book("BTCUSDT", sample_order_book)
    print(f"Analysis Result: {json.dumps(result, indent=2)}")
    
    # Get status
    status = analyzer.get_status()
    print(f"System Status: {status}")
    
    # Deactivate system
    analyzer.deactivate()


if __name__ == "__main__":
    main()
