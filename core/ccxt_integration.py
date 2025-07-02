from __future__ import annotations

"""CCXT Integration for Order Optimization
=========================================

Provides CCXT-based exchange connectivity and order optimization for:
- Multi-exchange arbitrage
- Order book analysis
- Buy/sell wall detection
- Profit vector optimization
- Decimal precision handling (8, 6, 2)

This module integrates with the Ghost Core system for strategy execution.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
import numpy as np

# CCXT imports
try:
    import ccxt
    import ccxt.async_support as ccxt_async

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None
    ccxt_async = None

logger = logging.getLogger(__name__)

# Set decimal precision for calculations
getcontext().prec = 28


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book data."""

    timestamp: float
    symbol: str
    bids: List[Tuple[float, float]]  # (price, volume)
    asks: List[Tuple[float, float]]  # (price, volume)
    spread: float
    mid_price: float
    total_bid_volume: float
    total_ask_volume: float
    granularity: int


@dataclass
class BuySellWall:
    """Represents a buy or sell wall in the order book."""

    side: str  # 'buy' or 'sell'
    price_level: float
    volume: float
    strength: float  # 0.0 to 1.0
    distance_from_mid: float
    granularity: int


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity between exchanges."""

    buy_exchange: str
    sell_exchange: str
    symbol: str
    buy_price: float
    sell_price: float
    spread: float
    volume_limit: float
    profit_potential: float
    risk_score: float
    timestamp: float


class CCXTIntegration:
    """
    CCXT integration for exchange connectivity and order optimization.

    Features:
    - Multi-exchange support
    - Order book analysis
    - Buy/sell wall detection
    - Arbitrage opportunity detection
    - Decimal precision handling
    - Profit vector optimization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CCXT integration."""
        if not CCXT_AVAILABLE:
            raise ImportError(
                "CCXT library not available. Install with: pip install ccxt"
            )

        self.config = config or {}
        self.exchanges: Dict[str, Any] = {}
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []

        # Configuration
        self.supported_exchanges = self.config.get(
            'exchanges', ['binance', 'coinbase', 'kraken']
        )
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'BTC/USD'])
        self.granularities = self.config.get('granularities', [8, 6, 2])
        self.min_spread = self.config.get('min_spread', 0.001)  # 0.1%
        self.max_risk_score = self.config.get('max_risk_score', 0.3)

        # Initialize exchanges
        self._initialize_exchanges()

        logger.info(
            "🔗 CCXT Integration initialized with %d exchanges", len(self.exchanges)
        )

    def _initialize_exchanges(self) -> None:
        """Initialize exchange connections."""
        for exchange_id in self.supported_exchanges:
            try:
                # Initialize both sync and async versions
                exchange = getattr(ccxt, exchange_id)(
                    {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
                )

                self.exchanges[exchange_id] = {
                    'sync': exchange,
                    'async': getattr(ccxt_async, exchange_id)(
                        {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
                    ),
                }

                logger.info("✅ Initialized exchange: %s", exchange_id)

            except Exception as e:
                logger.warning(
                    "❌ Failed to initialize exchange %s: %s", exchange_id, e
                )

    async def fetch_order_book(
        self, exchange_id: str, symbol: str, limit: int = 20
    ) -> Optional[OrderBookSnapshot]:
        """
        Fetch order book from exchange.

        Args:
            exchange_id: Exchange identifier
            symbol: Trading symbol
            limit: Number of orders to fetch

        Returns:
            Order book snapshot or None if failed
        """
        try:
            exchange = self.exchanges[exchange_id]['async']

            # Fetch order book
            order_book = await exchange.fetch_order_book(symbol, limit)

            # Extract data
            bids = order_book['bids'][:limit]
            asks = order_book['asks'][:limit]

            # Calculate metrics
            best_bid = bids[0][0] if bids else 0.0
            best_ask = asks[0][0] if asks else float('inf')
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2

            total_bid_volume = sum(bid[1] for bid in bids)
            total_ask_volume = sum(ask[1] for ask in asks)

            # Determine granularity based on price
            granularity = self._determine_granularity(mid_price)

            snapshot = OrderBookSnapshot(
                timestamp=order_book['timestamp'] / 1000.0,
                symbol=symbol,
                bids=bids,
                asks=asks,
                spread=spread,
                mid_price=mid_price,
                total_bid_volume=total_bid_volume,
                total_ask_volume=total_ask_volume,
                granularity=granularity,
            )

            # Store in cache
            cache_key = f"{exchange_id}:{symbol}"
            self.order_books[cache_key] = snapshot

            return snapshot

        except Exception as e:
            logger.error(
                "Failed to fetch order book for %s:%s: %s", exchange_id, symbol, e
            )
            return None

    def _determine_granularity(self, price: float) -> int:
        """Determine appropriate decimal granularity based on price."""
        if price >= 10000:  # High value assets like BTC
            return 2
        elif price >= 100:  # Medium value assets
            return 6
        else:  # Low value assets
            return 8

    def detect_buy_sell_walls(
        self, order_book: OrderBookSnapshot, min_wall_strength: float = 0.1
    ) -> List[BuySellWall]:
        """
        Detect buy and sell walls in order book.

        Args:
            order_book: Order book snapshot
            min_wall_strength: Minimum strength threshold

        Returns:
            List of detected walls
        """
        walls = []

        # Analyze bids (buy walls)
        bid_walls = self._analyze_walls(order_book.bids, 'buy', order_book.mid_price)
        walls.extend([wall for wall in bid_walls if wall.strength >= min_wall_strength])

        # Analyze asks (sell walls)
        ask_walls = self._analyze_walls(order_book.asks, 'sell', order_book.mid_price)
        walls.extend([wall for wall in ask_walls if wall.strength >= min_wall_strength])

        return walls

    def _analyze_walls(
        self, orders: List[Tuple[float, float]], side: str, mid_price: float
    ) -> List[BuySellWall]:
        """Analyze orders to detect walls."""
        walls = []

        if not orders:
            return walls

        # Group orders by price levels
        price_levels = {}
        for price, volume in orders:
            # Round to granularity
            rounded_price = round(price, 2)  # Adjust based on granularity
            if rounded_price in price_levels:
                price_levels[rounded_price] += volume
            else:
                price_levels[rounded_price] = volume

        # Calculate average volume for strength comparison
        volumes = list(price_levels.values())
        avg_volume = np.mean(volumes) if volumes else 0.0
        max_volume = np.max(volumes) if volumes else 0.0

        # Detect walls
        for price, volume in price_levels.items():
            # Calculate strength relative to average
            strength = volume / max_volume if max_volume > 0 else 0.0

            # Calculate distance from mid price
            distance = abs(price - mid_price) / mid_price

            wall = BuySellWall(
                side=side,
                price_level=price,
                volume=volume,
                strength=strength,
                distance_from_mid=distance,
                granularity=2,  # Adjust based on price
            )

            walls.append(wall)

        return walls

    async def detect_arbitrage_opportunities(
        self, symbol: str, min_spread: Optional[float] = None
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities across exchanges.

        Args:
            symbol: Trading symbol
            min_spread: Minimum spread threshold

        Returns:
            List of arbitrage opportunities
        """
        if min_spread is None:
            min_spread = self.min_spread

        opportunities = []

        # Fetch order books from all exchanges
        order_books = {}
        for exchange_id in self.exchanges.keys():
            order_book = await self.fetch_order_book(exchange_id, symbol)
            if order_book:
                order_books[exchange_id] = order_book

        # Find arbitrage opportunities
        exchanges = list(order_books.keys())
        for i, buy_exchange in enumerate(exchanges):
            for sell_exchange in exchanges[i + 1 :]:
                buy_book = order_books[buy_exchange]
                sell_book = order_books[sell_exchange]

                # Calculate potential arbitrage
                buy_price = buy_book.bids[0][0] if buy_book.bids else 0.0
                sell_price = sell_book.asks[0][0] if sell_book.asks else float('inf')

                if buy_price > 0 and sell_price < float('inf'):
                    spread = (sell_price - buy_price) / buy_price

                    if spread >= min_spread:
                        # Calculate volume limit (minimum of available volumes)
                        buy_volume = buy_book.bids[0][1] if buy_book.bids else 0.0
                        sell_volume = sell_book.asks[0][1] if sell_book.asks else 0.0
                        volume_limit = min(buy_volume, sell_volume)

                        # Calculate profit potential
                        profit_potential = spread * buy_price * volume_limit

                        # Calculate risk score
                        risk_score = self._calculate_arbitrage_risk(
                            buy_exchange, sell_exchange, spread, volume_limit
                        )

                        if risk_score <= self.max_risk_score:
                            opportunity = ArbitrageOpportunity(
                                buy_exchange=buy_exchange,
                                sell_exchange=sell_exchange,
                                symbol=symbol,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                spread=spread,
                                volume_limit=volume_limit,
                                profit_potential=profit_potential,
                                risk_score=risk_score,
                                timestamp=time.time(),
                            )

                            opportunities.append(opportunity)

        # Sort by profit potential
        opportunities.sort(key=lambda x: x.profit_potential, reverse=True)

        # Update stored opportunities
        self.arbitrage_opportunities = opportunities

        return opportunities

    def _calculate_arbitrage_risk(
        self, buy_exchange: str, sell_exchange: str, spread: float, volume: float
    ) -> float:
        """Calculate risk score for arbitrage opportunity."""
        risk_score = 0.0

        # Spread risk (lower spread = higher risk)
        if spread < 0.005:  # 0.5%
            risk_score += 0.3
        elif spread < 0.01:  # 1%
            risk_score += 0.2
        else:
            risk_score += 0.1

        # Volume risk (lower volume = higher risk)
        if volume < 0.01:  # Very small volume
            risk_score += 0.4
        elif volume < 0.1:  # Small volume
            risk_score += 0.2
        else:
            risk_score += 0.1

        # Exchange risk (different exchanges = higher risk)
        if buy_exchange != sell_exchange:
            risk_score += 0.2

        return min(1.0, risk_score)

    def optimize_order_size(
        self,
        order_book: OrderBookSnapshot,
        target_volume: float,
        side: str,
        max_slippage: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Optimize order size to minimize slippage.

        Args:
            order_book: Order book snapshot
            target_volume: Target volume to trade
            side: 'buy' or 'sell'
            max_slippage: Maximum acceptable slippage

        Returns:
            Optimized order parameters
        """
        orders = order_book.asks if side == 'buy' else order_book.bids
        if not orders:
            return {'error': 'No orders available'}

        total_volume = 0.0
        weighted_price = 0.0
        orders_to_use = []

        for price, volume in orders:
            if total_volume >= target_volume:
                break

            # Calculate how much to use from this level
            use_volume = min(volume, target_volume - total_volume)

            total_volume += use_volume
            weighted_price += price * use_volume
            orders_to_use.append((price, use_volume))

        if total_volume == 0:
            return {'error': 'Cannot fill target volume'}

        # Calculate average price
        avg_price = weighted_price / total_volume

        # Calculate slippage
        reference_price = orders[0][0]  # Best price
        slippage = abs(avg_price - reference_price) / reference_price

        # Check if slippage is acceptable
        if slippage > max_slippage:
            return {
                'error': f'Slippage {slippage:.4f} exceeds maximum {max_slippage:.4f}',
                'slippage': slippage,
                'max_slippage': max_slippage,
            }

        return {
            'side': side,
            'volume': total_volume,
            'average_price': avg_price,
            'slippage': slippage,
            'orders': orders_to_use,
            'granularity': order_book.granularity,
        }

    def calculate_profit_vector(
        self, order_book: OrderBookSnapshot, walls: List[BuySellWall]
    ) -> Dict[str, Any]:
        """
        Calculate profit vector based on order book and walls.

        Args:
            order_book: Order book snapshot
            walls: Detected buy/sell walls

        Returns:
            Profit vector analysis
        """
        # Calculate basic metrics
        spread = order_book.spread
        mid_price = order_book.mid_price

        # Analyze wall impact
        buy_walls = [w for w in walls if w.side == 'buy']
        sell_walls = [w for w in walls if w.side == 'sell']

        # Calculate wall pressure
        buy_pressure = sum(wall.strength * wall.volume for wall in buy_walls)
        sell_pressure = sum(wall.strength * wall.volume for wall in sell_walls)

        # Calculate pressure ratio
        pressure_ratio = buy_pressure / sell_pressure if sell_pressure > 0 else 1.0

        # Calculate profit potential
        base_profit = spread * mid_price
        wall_enhanced_profit = base_profit * pressure_ratio

        # Calculate volatility from order book
        all_prices = [price for price, _ in order_book.bids + order_book.asks]
        volatility = np.std(all_prices) / mid_price if all_prices else 0.0

        return {
            'base_profit': base_profit,
            'wall_enhanced_profit': wall_enhanced_profit,
            'pressure_ratio': pressure_ratio,
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'volatility': volatility,
            'spread': spread,
            'mid_price': mid_price,
            'granularity': order_book.granularity,
            'wall_count': len(walls),
            'buy_wall_count': len(buy_walls),
            'sell_wall_count': len(sell_walls),
        }

    async def get_market_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market summary."""
        summary = {
            'symbol': symbol,
            'timestamp': time.time(),
            'exchanges': {},
            'arbitrage_opportunities': [],
            'overall_metrics': {},
        }

        # Fetch data from all exchanges
        for exchange_id in self.exchanges.keys():
            try:
                order_book = await self.fetch_order_book(exchange_id, symbol)
                if order_book:
                    walls = self.detect_buy_sell_walls(order_book)
                    profit_vector = self.calculate_profit_vector(order_book, walls)

                    summary['exchanges'][exchange_id] = {
                        'order_book': {
                            'spread': order_book.spread,
                            'mid_price': order_book.mid_price,
                            'total_bid_volume': order_book.total_bid_volume,
                            'total_ask_volume': order_book.total_ask_volume,
                            'granularity': order_book.granularity,
                        },
                        'walls': {
                            'count': len(walls),
                            'buy_walls': len([w for w in walls if w.side == 'buy']),
                            'sell_walls': len([w for w in walls if w.side == 'sell']),
                        },
                        'profit_vector': profit_vector,
                    }
            except Exception as e:
                logger.error(
                    "Failed to get market data for %s:%s: %s", exchange_id, symbol, e
                )

        # Detect arbitrage opportunities
        arbitrage_opps = await self.detect_arbitrage_opportunities(symbol)
        summary['arbitrage_opportunities'] = [
            {
                'buy_exchange': opp.buy_exchange,
                'sell_exchange': opp.sell_exchange,
                'spread': opp.spread,
                'profit_potential': opp.profit_potential,
                'risk_score': opp.risk_score,
            }
            for opp in arbitrage_opps[:5]  # Top 5 opportunities
        ]

        # Calculate overall metrics
        if summary['exchanges']:
            spreads = [
                data['order_book']['spread'] for data in summary['exchanges'].values()
            ]
            mid_prices = [
                data['order_book']['mid_price']
                for data in summary['exchanges'].values()
            ]

            summary['overall_metrics'] = {
                'avg_spread': np.mean(spreads),
                'min_spread': np.min(spreads),
                'max_spread': np.max(spreads),
                'avg_mid_price': np.mean(mid_prices),
                'price_volatility': np.std(mid_prices) / np.mean(mid_prices),
                'exchange_count': len(summary['exchanges']),
            }

        return summary

    async def close_connections(self) -> None:
        """Close all exchange connections."""
        for exchange_id, exchange_data in self.exchanges.items():
            try:
                await exchange_data['async'].close()
                logger.info("Closed connection to %s", exchange_id)
            except Exception as e:
                logger.error("Failed to close connection to %s: %s", exchange_id, e)


async def demo_ccxt_integration():
    """Demonstrate CCXT integration functionality."""
    print("🔗 CCXT Integration Demo")
    print("=" * 50)

    # Initialize integration
    config = {
        'exchanges': ['binance', 'coinbase'],
        'symbols': ['BTC/USDT'],
        'min_spread': 0.001,
    }

    integration = CCXTIntegration(config)

    try:
        # Get market summary
        print("\nFetching market summary...")
        summary = await integration.get_market_summary('BTC/USDT')

        print(f"Market Summary for BTC/USDT:")
        print(f"  Exchanges: {list(summary['exchanges'].keys())}")
        print(
            f"  Average Spread: {
        summary['overall_metrics'].get(
            'avg_spread',
             0):.6f}"
        )
        print(
            f"  Price Volatility: {
        summary['overall_metrics'].get(
            'price_volatility',
             0):.4f}"
        )

        # Show exchange details
        for exchange_id, data in summary['exchanges'].items():
            print(f"\n{exchange_id.upper()}:")
            print(f"  Spread: {data['order_book']['spread']:.6f}")
            print(f"  Mid Price: ${data['order_book']['mid_price']:,.2f}")
            print(f"  Walls: {data['walls']['count']} total")
            print(
                f"  Profit Vector: {
        data['profit_vector']['wall_enhanced_profit']:.6f}"
            )

        # Show arbitrage opportunities
        if summary['arbitrage_opportunities']:
            print(f"\nArbitrage Opportunities:")
            for i, opp in enumerate(summary['arbitrage_opportunities'], 1):
                print(f"  {i}. {opp['buy_exchange']} → {opp['sell_exchange']}")
                print(
                    f"     Spread: {
        opp['spread']:.4f}, Profit: {
            opp['profit_potential']:.4f}"
                )

    except Exception as e:
        print(f"Demo failed: {e}")

    finally:
        # Close connections
        await integration.close_connections()

    print("\n✅ CCXT Integration demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_ccxt_integration())
