"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order Book Analyzer for Schwabot Trading System.

Advanced order book analysis for detecting buy/sell walls, calculating optimal
entry/exit points, and providing liquidity analysis for profitable trading.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


    class WallType(Enum):
    """Class for Schwabot trading functionality."""
    """Class for Schwabot trading functionality."""
    """Types of order book walls."""

    BUY_WALL = "buy_wall"
    SELL_WALL = "sell_wall"
    SUPPORT_LEVEL = "support_level"
    RESISTANCE_LEVEL = "resistance_level"


    @dataclass
        class OrderBookWall:
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """Represents a significant order book wall."""

        wall_type: WallType
        price_level: float
        total_volume: float
        order_count: int
        strength_score: float
        impact_radius: float
        confidence: float
        timestamp: float
        metadata: Dict[str, Any] = field(default_factory=dict)


        @dataclass
            class LiquidityAnalysis:
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Liquidity analysis results."""

            bid_liquidity: float
            ask_liquidity: float
            spread: float
            depth_score: float
            imbalance_ratio: float
            optimal_entry_price: float
            optimal_exit_price: float
            market_impact_score: float


            @dataclass
                class OrderBookSnapshot:
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                """Complete order book snapshot with analysis."""

                timestamp: float
                symbol: str
                bids: List[Tuple[float, float]]  # (price, volume)
                asks: List[Tuple[float, float]]  # (price, volume)
                mid_price: float
                spread: float
                walls: List[OrderBookWall]
                liquidity_analysis: LiquidityAnalysis
                metadata: Dict[str, Any] = field(default_factory=dict)


                    class OrderBookAnalyzer:
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """
                    Advanced order book analyzer for detecting buy/sell walls and calculating
                    optimal entry/exit points for profitable trading.
                    """

                        def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
                        """Initialize the order book analyzer."""
                        self.config = config or self._default_config()

                        # Wall detection parameters
                        self.min_wall_volume = self.config.get("min_wall_volume", 1000.0)
                        self.min_wall_orders = self.config.get("min_wall_orders", 5)
                        self.wall_clustering_threshold = self.config.get("wall_clustering_threshold", 0.01)
                        self.impact_radius_multiplier = self.config.get("impact_radius_multiplier", 2.0)

                        # Liquidity analysis parameters
                        self.depth_levels = self.config.get("depth_levels", 20)
                        self.liquidity_threshold = self.config.get("liquidity_threshold", 0.1)

                        # Performance tracking
                        self.wall_history: List[OrderBookWall] = []
                        self.analysis_history: List[OrderBookSnapshot] = []

                        logger.info("OrderBookAnalyzer initialized with config: %s", self.config)

                            def _default_config(self) -> Dict[str, Any]:
                            """Default configuration for order book analysis."""
                        return {
                        "min_wall_volume": 1000.0,  # Minimum volume for wall detection
                        "min_wall_orders": 5,  # Minimum number of orders for wall
                        "wall_clustering_threshold": 0.01,  # Price clustering threshold
                        "impact_radius_multiplier": 2.0,  # Impact radius calculation
                        "depth_levels": 20,  # Number of levels to analyze
                        "liquidity_threshold": 0.1,  # Minimum liquidity threshold
                        "wall_strength_weights": {
                        "volume": 0.4,
                        "order_count": 0.2,
                        "price_level": 0.2,
                        "clustering": 0.2,
                        },
                        "liquidity_weights": {
                        "bid_depth": 0.3,
                        "ask_depth": 0.3,
                        "spread": 0.2,
                        "imbalance": 0.2,
                        },
                        }

                        def analyze_order_book(
                        self,
                        bids: List[Tuple[float, float]],
                        asks: List[Tuple[float, float]],
                        symbol: str = "BTC/USDT",
                            ) -> OrderBookSnapshot:
                            """
                            Analyze order book for walls, liquidity, and optimal entry/exit points.

                                Args:
                                bids: List of (price, volume) tuples for bids
                                asks: List of (price, volume) tuples for asks
                                symbol: Trading symbol

                                    Returns:
                                    Complete order book analysis snapshot
                                    """
                                        try:
                                        # Sort order book data
                                        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)  # Highest bid first
                                        asks_sorted = sorted(asks, key=lambda x: x[0])  # Lowest ask first

                                        # Calculate basic metrics
                                        mid_price = self._calculate_mid_price(bids_sorted, asks_sorted)
                                        spread = self._calculate_spread(bids_sorted, asks_sorted)

                                        # Detect walls
                                        buy_walls = self._detect_buy_walls(bids_sorted, mid_price)
                                        sell_walls = self._detect_sell_walls(asks_sorted, mid_price)

                                        # Combine all walls
                                        all_walls = buy_walls + sell_walls

                                        # Analyze liquidity
                                        liquidity_analysis = self._analyze_liquidity(bids_sorted, asks_sorted, mid_price)

                                        # Calculate optimal entry/exit points
                                        optimal_entry = self._calculate_optimal_entry(bids_sorted, asks_sorted, all_walls, liquidity_analysis)
                                        optimal_exit = self._calculate_optimal_exit(bids_sorted, asks_sorted, all_walls, liquidity_analysis)

                                        # Update liquidity analysis with optimal points
                                        liquidity_analysis.optimal_entry_price = optimal_entry
                                        liquidity_analysis.optimal_exit_price = optimal_exit

                                        # Create snapshot
                                        snapshot = OrderBookSnapshot(
                                        timestamp=self._get_current_timestamp(),
                                        symbol=symbol,
                                        bids=bids_sorted,
                                        asks=asks_sorted,
                                        mid_price=mid_price,
                                        spread=spread,
                                        walls=all_walls,
                                        liquidity_analysis=liquidity_analysis,
                                        )

                                        # Update history
                                        self.analysis_history.append(snapshot)
                                        self.wall_history.extend(all_walls)

                                        # Keep history manageable
                                        self._cleanup_history()

                                        logger.debug("Order book analysis completed for %s: %d walls detected", symbol, len(all_walls))

                                    return snapshot

                                        except Exception as e:
                                        logger.error("Order book analysis failed: %s", e)
                                    raise

                                        def _detect_buy_walls(self, bids: List[Tuple[float, float]], mid_price: float) -> List[OrderBookWall]:
                                        """Detect significant buy walls that could drive price up."""
                                        walls = []

                                            try:
                                            # Group bids by price levels (clustering)
                                            price_groups = self._cluster_by_price(bids, self.wall_clustering_threshold)

                                                for price_level, orders in price_groups.items():
                                                total_volume = sum(volume for _, volume in orders)
                                                order_count = len(orders)

                                                # Check if this qualifies as a wall
                                                    if total_volume >= self.min_wall_volume and order_count >= self.min_wall_orders:

                                                    # Calculate wall strength
                                                    strength_score = self._calculate_wall_strength(
                                                    total_volume, order_count, price_level, mid_price, "buy"
                                                    )

                                                    # Calculate impact radius
                                                    impact_radius = self._calculate_impact_radius(total_volume, price_level, mid_price)

                                                    # Calculate confidence
                                                    confidence = self._calculate_wall_confidence(total_volume, order_count, strength_score)

                                                    wall = OrderBookWall(
                                                    wall_type=WallType.BUY_WALL,
                                                    price_level=price_level,
                                                    total_volume=total_volume,
                                                    order_count=order_count,
                                                    strength_score=strength_score,
                                                    impact_radius=impact_radius,
                                                    confidence=confidence,
                                                    timestamp=self._get_current_timestamp(),
                                                    metadata={
                                                    "distance_from_mid": mid_price - price_level,
                                                    "volume_per_order": total_volume / order_count,
                                                    },
                                                    )

                                                    walls.append(wall)

                                                    # Sort walls by strength
                                                    walls.sort(key=lambda w: w.strength_score, reverse=True)

                                                return walls

                                                    except Exception as e:
                                                    logger.error("Buy wall detection failed: %s", e)
                                                return []

                                                    def _detect_sell_walls(self, asks: List[Tuple[float, float]], mid_price: float) -> List[OrderBookWall]:
                                                    """Detect significant sell walls that could drive price down."""
                                                    walls = []

                                                        try:
                                                        # Group asks by price levels (clustering)
                                                        price_groups = self._cluster_by_price(asks, self.wall_clustering_threshold)

                                                            for price_level, orders in price_groups.items():
                                                            total_volume = sum(volume for _, volume in orders)
                                                            order_count = len(orders)

                                                            # Check if this qualifies as a wall
                                                                if total_volume >= self.min_wall_volume and order_count >= self.min_wall_orders:

                                                                # Calculate wall strength
                                                                strength_score = self._calculate_wall_strength(
                                                                total_volume, order_count, price_level, mid_price, "sell"
                                                                )

                                                                # Calculate impact radius
                                                                impact_radius = self._calculate_impact_radius(total_volume, price_level, mid_price)

                                                                # Calculate confidence
                                                                confidence = self._calculate_wall_confidence(total_volume, order_count, strength_score)

                                                                wall = OrderBookWall(
                                                                wall_type=WallType.SELL_WALL,
                                                                price_level=price_level,
                                                                total_volume=total_volume,
                                                                order_count=order_count,
                                                                strength_score=strength_score,
                                                                impact_radius=impact_radius,
                                                                confidence=confidence,
                                                                timestamp=self._get_current_timestamp(),
                                                                metadata={
                                                                "distance_from_mid": price_level - mid_price,
                                                                "volume_per_order": total_volume / order_count,
                                                                },
                                                                )

                                                                walls.append(wall)

                                                                # Sort walls by strength
                                                                walls.sort(key=lambda w: w.strength_score, reverse=True)

                                                            return walls

                                                                except Exception as e:
                                                                logger.error("Sell wall detection failed: %s", e)
                                                            return []

                                                            def _cluster_by_price(
                                                            self, orders: List[Tuple[float, float]], threshold: float
                                                                ) -> Dict[float, List[Tuple[float, float]]]:
                                                                """Cluster orders by price levels within threshold."""
                                                                clusters = {}

                                                                    for price, volume in orders:
                                                                    # Find closest existing cluster
                                                                    clustered = False
                                                                        for cluster_price in clusters.keys():
                                                                            if abs(price - cluster_price) <= threshold:
                                                                            clusters[cluster_price].append((price, volume))
                                                                            clustered = True
                                                                        break

                                                                        # Create new cluster if no match found
                                                                            if not clustered:
                                                                            clusters[price] = [(price, volume)]

                                                                        return clusters

                                                                        def _calculate_wall_strength(
                                                                        self,
                                                                        total_volume: float,
                                                                        order_count: int,
                                                                        price_level: float,
                                                                        mid_price: float,
                                                                        wall_type: str,
                                                                            ) -> float:
                                                                            """Calculate wall strength score (0-1)."""
                                                                                try:
                                                                                weights = self.config["wall_strength_weights"]

                                                                                # Volume component (normalized)
                                                                                volume_score = min(total_volume / (self.min_wall_volume * 10), 1.0)

                                                                                # Order count component (normalized)
                                                                                order_score = min(order_count / (self.min_wall_orders * 5), 1.0)

                                                                                # Price level component (closer to mid price = stronger)
                                                                                distance = abs(price_level - mid_price)
                                                                                price_score = max(0, 1.0 - (distance / mid_price))

                                                                                # Clustering component (more orders at same price = stronger)
                                                                                clustering_score = min(order_count / 20, 1.0)

                                                                                # Calculate weighted strength
                                                                                strength = (
                                                                                weights["volume"] * volume_score
                                                                                + weights["order_count"] * order_score
                                                                                + weights["price_level"] * price_score
                                                                                + weights["clustering"] * clustering_score
                                                                                )

                                                                            return min(max(strength, 0.0), 1.0)

                                                                                except Exception as e:
                                                                                logger.error("Wall strength calculation failed: %s", e)
                                                                            return 0.0

                                                                                def _calculate_impact_radius(self, total_volume: float, price_level: float, mid_price: float) -> float:
                                                                                """Calculate the price impact radius of a wall."""
                                                                                    try:
                                                                                    # Base impact based on volume
                                                                                    base_impact = np.log10(total_volume / 1000) * 0.01

                                                                                    # Adjust for distance from mid price
                                                                                    distance_factor = 1.0 / (1.0 + abs(price_level - mid_price) / mid_price)

                                                                                    impact_radius = base_impact * distance_factor * self.impact_radius_multiplier

                                                                                return max(impact_radius, 0.001)  # Minimum 0.1% impact

                                                                                    except Exception as e:
                                                                                    logger.error("Impact radius calculation failed: %s", e)
                                                                                return 0.01

                                                                                    def _calculate_wall_confidence(self, total_volume: float, order_count: int, strength_score: float) -> float:
                                                                                    """Calculate confidence in wall detection (0-1)."""
                                                                                        try:
                                                                                        # Volume confidence
                                                                                        volume_confidence = min(total_volume / (self.min_wall_volume * 5), 1.0)

                                                                                        # Order count confidence
                                                                                        order_confidence = min(order_count / (self.min_wall_orders * 3), 1.0)

                                                                                        # Strength confidence
                                                                                        strength_confidence = strength_score

                                                                                        # Weighted confidence
                                                                                        confidence = volume_confidence * 0.4 + order_confidence * 0.3 + strength_confidence * 0.3

                                                                                    return min(max(confidence, 0.0), 1.0)

                                                                                        except Exception as e:
                                                                                        logger.error("Wall confidence calculation failed: %s", e)
                                                                                    return 0.5

                                                                                    def _analyze_liquidity(
                                                                                    self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], mid_price: float
                                                                                        ) -> LiquidityAnalysis:
                                                                                        """Analyze order book liquidity."""
                                                                                            try:
                                                                                            # Calculate bid liquidity (volume within depth levels)
                                                                                            bid_liquidity = self._calculate_depth_liquidity(bids, self.depth_levels)

                                                                                            # Calculate ask liquidity
                                                                                            ask_liquidity = self._calculate_depth_liquidity(asks, self.depth_levels)

                                                                                            # Calculate spread
                                                                                            spread = self._calculate_spread(bids, asks)

                                                                                            # Calculate depth score
                                                                                            depth_score = min((bid_liquidity + ask_liquidity) / (self.min_wall_volume * 20), 1.0)

                                                                                            # Calculate imbalance ratio
                                                                                            total_liquidity = bid_liquidity + ask_liquidity
                                                                                            imbalance_ratio = (bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity > 0 else 0.0

                                                                                            # Calculate market impact score
                                                                                            market_impact_score = self._calculate_market_impact_score(bid_liquidity, ask_liquidity, spread)

                                                                                        return LiquidityAnalysis(
                                                                                        bid_liquidity=bid_liquidity,
                                                                                        ask_liquidity=ask_liquidity,
                                                                                        spread=spread,
                                                                                        depth_score=depth_score,
                                                                                        imbalance_ratio=imbalance_ratio,
                                                                                        optimal_entry_price=mid_price,  # Will be updated later
                                                                                        optimal_exit_price=mid_price,  # Will be updated later
                                                                                        market_impact_score=market_impact_score,
                                                                                        )

                                                                                            except Exception as e:
                                                                                            logger.error("Liquidity analysis failed: %s", e)
                                                                                        return LiquidityAnalysis(
                                                                                        bid_liquidity=0.0,
                                                                                        ask_liquidity=0.0,
                                                                                        spread=0.0,
                                                                                        depth_score=0.0,
                                                                                        imbalance_ratio=0.0,
                                                                                        optimal_entry_price=mid_price,
                                                                                        optimal_exit_price=mid_price,
                                                                                        market_impact_score=0.0,
                                                                                        )

                                                                                            def _calculate_depth_liquidity(self, orders: List[Tuple[float, float]], levels: int) -> float:
                                                                                            """Calculate liquidity within specified depth levels."""
                                                                                                try:
                                                                                                total_volume = 0.0
                                                                                                    for i, (_, volume) in enumerate(orders[:levels]):
                                                                                                    total_volume += volume
                                                                                                return total_volume
                                                                                                    except Exception as e:
                                                                                                    logger.error("Depth liquidity calculation failed: %s", e)
                                                                                                return 0.0

                                                                                                    def _calculate_spread(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
                                                                                                    """Calculate bid-ask spread."""
                                                                                                        try:
                                                                                                            if not bids or not asks:
                                                                                                        return 0.0

                                                                                                        best_bid = bids[0][0]
                                                                                                        best_ask = asks[0][0]

                                                                                                        spread = (best_ask - best_bid) / best_bid
                                                                                                    return spread

                                                                                                        except Exception as e:
                                                                                                        logger.error("Spread calculation failed: %s", e)
                                                                                                    return 0.0

                                                                                                        def _calculate_market_impact_score(self, bid_liquidity: float, ask_liquidity: float, spread: float) -> float:
                                                                                                        """Calculate market impact score (0-1, lower is better)."""
                                                                                                            try:
                                                                                                            # Higher liquidity = lower impact
                                                                                                            liquidity_score = 1.0 / (1.0 + (bid_liquidity + ask_liquidity) / 10000)

                                                                                                            # Lower spread = lower impact
                                                                                                            spread_score = min(spread * 100, 1.0)

                                                                                                            # Combined impact score
                                                                                                            impact_score = liquidity_score * 0.7 + spread_score * 0.3

                                                                                                        return min(max(impact_score, 0.0), 1.0)

                                                                                                            except Exception as e:
                                                                                                            logger.error("Market impact score calculation failed: %s", e)
                                                                                                        return 0.5

                                                                                                        def _calculate_optimal_entry(
                                                                                                        self,
                                                                                                        bids: List[Tuple[float, float]],
                                                                                                        asks: List[Tuple[float, float]],
                                                                                                        walls: List[OrderBookWall],
                                                                                                        liquidity: LiquidityAnalysis,
                                                                                                            ) -> float:
                                                                                                            """Calculate optimal entry price considering walls and liquidity."""
                                                                                                                try:
                                                                                                                    if not asks:
                                                                                                                return 0.0

                                                                                                                best_ask = asks[0][0]

                                                                                                                # Find strongest buy wall below current price
                                                                                                                buy_walls = [w for w in walls if w.wall_type == WallType.BUY_WALL]
                                                                                                                buy_walls.sort(key=lambda w: w.strength_score, reverse=True)

                                                                                                                    if buy_walls:
                                                                                                                    strongest_buy_wall = buy_walls[0]

                                                                                                                    # If strong buy wall exists, enter slightly above it
                                                                                                                        if strongest_buy_wall.strength_score > 0.7:
                                                                                                                        entry_price = strongest_buy_wall.price_level * 1.001  # 0.1% above wall
                                                                                                                            else:
                                                                                                                            entry_price = best_ask
                                                                                                                                else:
                                                                                                                                entry_price = best_ask

                                                                                                                                # Adjust for liquidity imbalance
                                                                                                                                if liquidity.imbalance_ratio > 0.1:  # More bid liquidity
                                                                                                                                entry_price *= 1.001  # Slightly higher entry
                                                                                                                                elif liquidity.imbalance_ratio < -0.1:  # More ask liquidity
                                                                                                                                entry_price *= 0.999  # Slightly lower entry

                                                                                                                            return entry_price

                                                                                                                                except Exception as e:
                                                                                                                                logger.error("Optimal entry calculation failed: %s", e)
                                                                                                                            return asks[0][0] if asks else 0.0

                                                                                                                            def _calculate_optimal_exit(
                                                                                                                            self,
                                                                                                                            bids: List[Tuple[float, float]],
                                                                                                                            asks: List[Tuple[float, float]],
                                                                                                                            walls: List[OrderBookWall],
                                                                                                                            liquidity: LiquidityAnalysis,
                                                                                                                                ) -> float:
                                                                                                                                """Calculate optimal exit price considering walls and liquidity."""
                                                                                                                                    try:
                                                                                                                                        if not bids:
                                                                                                                                    return 0.0

                                                                                                                                    best_bid = bids[0][0]

                                                                                                                                    # Find strongest sell wall above current price
                                                                                                                                    sell_walls = [w for w in walls if w.wall_type == WallType.SELL_WALL]
                                                                                                                                    sell_walls.sort(key=lambda w: w.strength_score, reverse=True)

                                                                                                                                        if sell_walls:
                                                                                                                                        strongest_sell_wall = sell_walls[0]

                                                                                                                                        # If strong sell wall exists, exit slightly below it
                                                                                                                                            if strongest_sell_wall.strength_score > 0.7:
                                                                                                                                            exit_price = strongest_sell_wall.price_level * 0.999  # 0.1% below wall
                                                                                                                                                else:
                                                                                                                                                exit_price = best_bid
                                                                                                                                                    else:
                                                                                                                                                    exit_price = best_bid

                                                                                                                                                    # Adjust for liquidity imbalance
                                                                                                                                                    if liquidity.imbalance_ratio > 0.1:  # More bid liquidity
                                                                                                                                                    exit_price *= 1.001  # Slightly higher exit
                                                                                                                                                    elif liquidity.imbalance_ratio < -0.1:  # More ask liquidity
                                                                                                                                                    exit_price *= 0.999  # Slightly lower exit

                                                                                                                                                return exit_price

                                                                                                                                                    except Exception as e:
                                                                                                                                                    logger.error("Optimal exit calculation failed: %s", e)
                                                                                                                                                return bids[0][0] if bids else 0.0

                                                                                                                                                    def _calculate_mid_price(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
                                                                                                                                                    """Calculate mid price from best bid and ask."""
                                                                                                                                                        try:
                                                                                                                                                            if not bids or not asks:
                                                                                                                                                        return 0.0

                                                                                                                                                        best_bid = bids[0][0]
                                                                                                                                                        best_ask = asks[0][0]

                                                                                                                                                        mid_price = (best_bid + best_ask) / 2
                                                                                                                                                    return mid_price

                                                                                                                                                        except Exception as e:
                                                                                                                                                        logger.error("Mid price calculation failed: %s", e)
                                                                                                                                                    return 0.0

                                                                                                                                                        def _get_current_timestamp(self) -> float:
                                                                                                                                                        """Get current timestamp."""
                                                                                                                                                        import time

                                                                                                                                                    return time.time()

                                                                                                                                                        def _cleanup_history(self) -> None:
                                                                                                                                                        """Clean up analysis history to prevent memory issues."""
                                                                                                                                                        max_history = 1000

                                                                                                                                                            if len(self.analysis_history) > max_history:
                                                                                                                                                            self.analysis_history = self.analysis_history[-max_history:]

                                                                                                                                                                if len(self.wall_history) > max_history * 10:
                                                                                                                                                                self.wall_history = self.wall_history[-max_history * 10 :]

                                                                                                                                                                    def get_wall_summary(self) -> Dict[str, Any]:
                                                                                                                                                                    """Get summary of detected walls."""
                                                                                                                                                                        try:
                                                                                                                                                                        buy_walls = [w for w in self.wall_history if w.wall_type == WallType.BUY_WALL]
                                                                                                                                                                        sell_walls = [w for w in self.wall_history if w.wall_type == WallType.SELL_WALL]

                                                                                                                                                                    return {
                                                                                                                                                                    "total_walls": len(self.wall_history),
                                                                                                                                                                    "buy_walls": len(buy_walls),
                                                                                                                                                                    "sell_walls": len(sell_walls),
                                                                                                                                                                    "strongest_buy_wall": (
                                                                                                                                                                    max(buy_walls, key=lambda w: w.strength_score).strength_score if buy_walls else 0.0
                                                                                                                                                                    ),
                                                                                                                                                                    "strongest_sell_wall": (
                                                                                                                                                                    max(sell_walls, key=lambda w: w.strength_score).strength_score if sell_walls else 0.0
                                                                                                                                                                    ),
                                                                                                                                                                    "average_wall_strength": (
                                                                                                                                                                    np.mean([w.strength_score for w in self.wall_history]) if self.wall_history else 0.0
                                                                                                                                                                    ),
                                                                                                                                                                    "wall_detection_rate": len(self.analysis_history) / max(len(self.analysis_history), 1),
                                                                                                                                                                    }

                                                                                                                                                                        except Exception as e:
                                                                                                                                                                        logger.error("Wall summary generation failed: %s", e)
                                                                                                                                                                    return {}

                                                                                                                                                                        def get_liquidity_summary(self) -> Dict[str, Any]:
                                                                                                                                                                        """Get summary of liquidity analysis."""
                                                                                                                                                                            try:
                                                                                                                                                                                if not self.analysis_history:
                                                                                                                                                                            return {}

                                                                                                                                                                            recent_analyses = self.analysis_history[-100:]  # Last 100 analyses

                                                                                                                                                                        return {
                                                                                                                                                                        "average_spread": np.mean([a.liquidity_analysis.spread for a in recent_analyses]),
                                                                                                                                                                        "average_depth_score": np.mean([a.liquidity_analysis.depth_score for a in recent_analyses]),
                                                                                                                                                                        "average_imbalance": np.mean([a.liquidity_analysis.imbalance_ratio for a in recent_analyses]),
                                                                                                                                                                        "average_market_impact": np.mean([a.liquidity_analysis.market_impact_score for a in recent_analyses]),
                                                                                                                                                                        "liquidity_trend": self._calculate_liquidity_trend(recent_analyses),
                                                                                                                                                                        }

                                                                                                                                                                            except Exception as e:
                                                                                                                                                                            logger.error("Liquidity summary generation failed: %s", e)
                                                                                                                                                                        return {}

                                                                                                                                                                            def _calculate_liquidity_trend(self, analyses: List[OrderBookSnapshot]) -> str:
                                                                                                                                                                            """Calculate liquidity trend over recent analyses."""
                                                                                                                                                                                try:
                                                                                                                                                                                    if len(analyses) < 10:
                                                                                                                                                                                return "insufficient_data"

                                                                                                                                                                                recent_depth_scores = [a.liquidity_analysis.depth_score for a in analyses[-10:]]
                                                                                                                                                                                trend_slope = np.polyfit(range(len(recent_depth_scores)), recent_depth_scores, 1)[0]

                                                                                                                                                                                    if trend_slope > 0.01:
                                                                                                                                                                                return "improving"
                                                                                                                                                                                    elif trend_slope < -0.01:
                                                                                                                                                                                return "declining"
                                                                                                                                                                                    else:
                                                                                                                                                                                return "stable"

                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                    logger.error("Liquidity trend calculation failed: %s", e)
                                                                                                                                                                                return "unknown"

                                                                                                                                                                                    def scan_entropy(self, bids, asks) -> None:
                                                                                                                                                                                    """
                                                                                                                                                                                    Analyze entropy between bids and asks.
                                                                                                                                                                                    Computes the std deviation of the bid/ask spread.
                                                                                                                                                                                    """
                                                                                                                                                                                    spread_changes = np.diff([b - a for b, a in zip(bids[-5:], asks[-5:])])
                                                                                                                                                                                    entropy_sigma = np.std(spread_changes)
                                                                                                                                                                                    self.last_entropy = entropy_sigma

                                                                                                                                                                                    # Route entropy as a signal
                                                                                                                                                                                        if entropy_sigma > 0.022:
                                                                                                                                                                                    return {"signal": True, "entropy": entropy_sigma}
                                                                                                                                                                                return {"signal": False, "entropy": entropy_sigma}


                                                                                                                                                                                # Convenience functions for external use
                                                                                                                                                                                    def create_order_book_analyzer(config: Optional[Dict[str, Any]] = None) -> OrderBookAnalyzer:
                                                                                                                                                                                    """Create a new order book analyzer instance."""
                                                                                                                                                                                return OrderBookAnalyzer(config)


                                                                                                                                                                                def analyze_order_book_simple(
                                                                                                                                                                                bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], symbol: str = "BTC/USDT"
                                                                                                                                                                                    ) -> Dict[str, Any]:
                                                                                                                                                                                    """Simple order book analysis for quick insights."""
                                                                                                                                                                                    analyzer = OrderBookAnalyzer()
                                                                                                                                                                                    snapshot = analyzer.analyze_order_book(bids, asks, symbol)

                                                                                                                                                                                return {
                                                                                                                                                                                "mid_price": snapshot.mid_price,
                                                                                                                                                                                "spread": snapshot.spread,
                                                                                                                                                                                "wall_count": len(snapshot.walls),
                                                                                                                                                                                "buy_walls": len([w for w in snapshot.walls if w.wall_type == WallType.BUY_WALL]),
                                                                                                                                                                                "sell_walls": len([w for w in snapshot.walls if w.wall_type == WallType.SELL_WALL]),
                                                                                                                                                                                "optimal_entry": snapshot.liquidity_analysis.optimal_entry_price,
                                                                                                                                                                                "optimal_exit": snapshot.liquidity_analysis.optimal_exit_price,
                                                                                                                                                                                "liquidity_score": snapshot.liquidity_analysis.depth_score,
                                                                                                                                                                                "market_impact": snapshot.liquidity_analysis.market_impact_score,
                                                                                                                                                                                }
