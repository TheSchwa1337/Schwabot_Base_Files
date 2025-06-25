# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Data Integration Layer for Schwabot

Connects to external APIs (CCXT, Coinbase) to fetch real-time cryptocurrency data
and integrates it with the FaultBus system for unified decision making.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Try to import CCXT for exchange data
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("CCXT not available. Install with: pip install ccxt")

# Try to import Coinbase API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests not available. Install with: pip install requests")

logger = logging.getLogger(__name__)


@dataclass
class CryptoDataPoint:
    """Represents a single cryptocurrency data point."""
    symbol: str
    price: float
    volume: float
    change_24h: float
    timestamp: datetime
    exchange: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None


@dataclass
class MarketState:
    """Represents the current market state across all tracked assets."""
    timestamp: datetime
    assets: Dict[str, CryptoDataPoint]
    overall_volatility: float
    market_sentiment: str  # 'bullish', 'bearish', 'neutral'
    volume_trend: str  # 'increasing', 'decreasing', 'stable'


class DataIntegrationLayer:
    """
    Integrates multiple data sources and provides unified access to market data.
    """

    def __init__(self, update_interval: float = 225.0):  # 3.75 minutes
        """
        Initialize the data integration layer.

        Args:
            update_interval: Time between data updates in seconds
        """
        self.update_interval = update_interval
        self.tracked_symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
        self.exchanges = {}
        self.market_data: Dict[str, CryptoDataPoint] = {}
        self.market_history: List[MarketState] = []
        self.max_history_size = 1000
        self.is_running = False

        # Initialize exchanges
        self._initialize_exchanges()

        logger.info(f"Data Integration Layer initialized with {len(self.tracked_symbols)} symbols")

    def _initialize_exchanges(self) -> None:
        """Initialize exchange connections."""
        if CCXT_AVAILABLE:
            # Initialize major exchanges
            exchanges_to_try = ['binance', 'coinbase', 'kraken']

            for exchange_name in exchanges_to_try:
                try:
                    exchange_class = getattr(ccxt, exchange_name)
                    exchange = exchange_class({
                        'enableRateLimit': True,
                        'timeout': 30000,
                    })

                    # Test connection
                    exchange.load_markets()
                    self.exchanges[exchange_name] = exchange
                    logger.info(f"✅ Connected to {exchange_name}")

                except Exception as e:
                    logger.warning(f"❌ Failed to connect to {exchange_name}: {e}")

        if not self.exchanges:
            logger.warning("⚠️ No exchanges available. Using mock data.")

    async def start_data_feed(self) -> None:
        """Start the continuous data feed."""
        if self.is_running:
            logger.warning("Data feed already running")
            return

        self.is_running = True
        logger.info("🚀 Starting data integration feed...")

        try:
            while self.is_running:
                await self._update_market_data()
                await asyncio.sleep(self.update_interval)

        except Exception as e:
            logger.error(f"❌ Data feed error: {e}")
            self.is_running = False

    async def stop_data_feed(self) -> None:
        """Stop the data feed."""
        self.is_running = False
        logger.info("🛑 Stopping data integration feed...")

    async def _update_market_data(self) -> None:
        """Update market data from all sources."""
        try:
            new_data = {}

            # Fetch data from exchanges
            if self.exchanges:
                for symbol in self.tracked_symbols:
                    data_point = await self._fetch_from_exchanges(symbol)
                    if data_point:
                        new_data[symbol] = data_point

            # Fallback to mock data if no exchanges available
            if not new_data:
                new_data = self._generate_mock_data()

            # Update market data
            self.market_data.update(new_data)

            # Calculate market state
            market_state = self._calculate_market_state()
            self.market_history.append(market_state)

            # Trim history
            if len(self.market_history) > self.max_history_size:
                self.market_history = self.market_history[-self.max_history_size:]

            logger.debug(f"📊 Updated market data for {len(new_data)} symbols")

        except Exception as e:
            logger.error(f"❌ Error updating market data: {e}")

    async def _fetch_from_exchanges(self, symbol: str) -> Optional[CryptoDataPoint]:
        """Fetch data for a symbol from available exchanges."""
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Fetch ticker data
                ticker = await asyncio.get_event_loop().run_in_executor(
                    None, exchange.fetch_ticker, symbol
                )

                if ticker and ticker.get('last'):
                    return CryptoDataPoint(
                        symbol=symbol,
                        price=float(ticker['last']),
                        volume=float(ticker.get('baseVolume', 0)),
                        change_24h=float(ticker.get('percentage', 0)),
                        timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                        exchange=exchange_name,
                        bid=float(ticker.get('bid', 0)),
                        ask=float(ticker.get('ask', 0)),
                        high_24h=float(ticker.get('high', 0)),
                        low_24h=float(ticker.get('low', 0))
                    )

            except Exception as e:
                logger.debug(f"Failed to fetch {symbol} from {exchange_name}: {e}")
                continue

        return None

    def _generate_mock_data(self) -> Dict[str, CryptoDataPoint]:
        """Generate mock data for testing when exchanges are unavailable."""
        mock_data = {}
        base_prices = {
            'BTC/USDT': 45000,
            'ETH/USDT': 3000,
            'XRP/USDT': 0.5
        }

        for symbol in self.tracked_symbols:
            base_price = base_prices.get(symbol, 100)

            # Add some realistic variation
            price_variation = (time.time() % 100) / 100  # Cyclic variation
            price = base_price + (price_variation - 0.5) * base_price * 0.1

            mock_data[symbol] = CryptoDataPoint(
                symbol=symbol,
                price=price,
                volume=1000000 + (price_variation * 500000),
                change_24h=(price_variation - 0.5) * 10,
                timestamp=datetime.now(),
                exchange='mock',
                bid=price * 0.999,
                ask=price * 1.001,
                high_24h=price * 1.05,
                low_24h=price * 0.95
            )

        return mock_data

    def _calculate_market_state(self) -> MarketState:
        """Calculate overall market state from current data."""
        if not self.market_data:
            return MarketState(
                timestamp=datetime.now(),
                assets={},
                overall_volatility=0.0,
                market_sentiment='neutral',
                volume_trend='stable'
            )

        # Calculate volatility
        prices = [data.price for data in self.market_data.values()]
        if len(prices) > 1:
            volatility = (unified_math.max(prices) - unified_math.min(prices)) / (sum(prices) / len(prices))
        else:
            volatility = 0.0

        # Calculate sentiment based on 24h changes
        changes = [data.change_24h for data in self.market_data.values()]
        avg_change = sum(changes) / len(changes) if changes else 0

        if avg_change > 2:
            sentiment = 'bullish'
        elif avg_change < -2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        # Calculate volume trend
        volumes = [data.volume for data in self.market_data.values()]
        if len(self.market_history) > 1:
            prev_volumes = [data.volume for data in self.market_history[-2].assets.values()]
            if len(volumes) == len(prev_volumes):
                current_avg = sum(volumes) / len(volumes)
                prev_avg = sum(prev_volumes) / len(prev_volumes)
                if current_avg > prev_avg * 1.1:
                    volume_trend = 'increasing'
                elif current_avg < prev_avg * 0.9:
                    volume_trend = 'decreasing'
                else:
                    volume_trend = 'stable'
            else:
                volume_trend = 'stable'
        else:
            volume_trend = 'stable'

        return MarketState(
            timestamp=datetime.now(),
            assets=self.market_data.copy(),
            overall_volatility=volatility,
            market_sentiment=sentiment,
            volume_trend=volume_trend
        )

    def get_current_data(self) -> Dict[str, Any]:
        """Get current market data in a format suitable for the FaultBus."""
        if not self.market_data:
            return {}

        # Convert to FaultBus-compatible format
        fault_bus_data = {
            'timestamp': datetime.now().isoformat(),
            'assets': {}
        }

        for symbol, data in self.market_data.items():
            # Extract asset name (e.g., 'BTC/USDT' -> 'BTC')
            asset_name = symbol.split('/')[0]

            fault_bus_data['assets'][asset_name] = {
                'price': data.price,
                'volume': data.volume,
                'change_24h': data.change_24h,
                'bid': data.bid,
                'ask': data.ask,
                'high_24h': data.high_24h,
                'low_24h': data.low_24h,
                'exchange': data.exchange
            }

        # Add market state information
        if self.market_history:
            latest_state = self.market_history[-1]
            fault_bus_data['market_state'] = {
                'volatility': latest_state.overall_volatility,
                'sentiment': latest_state.market_sentiment,
                'volume_trend': latest_state.volume_trend
            }

        return fault_bus_data

    def get_asset_data(self, symbol: str) -> Optional[CryptoDataPoint]:
        """Get data for a specific asset."""
        return self.market_data.get(symbol)

    def get_market_history(self, limit: int = 100) -> List[MarketState]:
        """Get recent market history."""
        return self.market_history[-limit:] if self.market_history else []

    def get_volatility_analysis(self) -> Dict[str, float]:
        """Get volatility analysis for all tracked assets."""
        if not self.market_history or len(self.market_history) < 2:
            return {}

        volatility_data = {}
        for symbol in self.tracked_symbols:
            prices = []
            for state in self.market_history[-20:]:  # Last 20 data points
                if symbol in state.assets:
                    prices.append(state.assets[symbol].price)

            if len(prices) > 1:
                # Calculate price volatility
                price_changes = [unified_math.abs(prices[i] - prices[i-1]) / prices[i-1]
                               for i in range(1, len(prices))]
                volatility_data[symbol] = sum(price_changes) / len(price_changes)
            else:
                volatility_data[symbol] = 0.0

        return volatility_data

    def export_data(self, filename: str) -> None:
        """Export current market data to JSON file."""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'market_data': {},
                'market_history': []
            }

            # Export current market data
            for symbol, data in self.market_data.items():
                export_data['market_data'][symbol] = {
                    'price': data.price,
                    'volume': data.volume,
                    'change_24h': data.change_24h,
                    'timestamp': data.timestamp.isoformat(),
                    'exchange': data.exchange
                }

            # Export recent market history
            for state in self.market_history[-10:]:
                history_entry = {
                    'timestamp': state.timestamp.isoformat(),
                    'volatility': state.overall_volatility,
                    'sentiment': state.market_sentiment,
                    'volume_trend': state.volume_trend
                }
                export_data['market_history'].append(history_entry)

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"📁 Market data exported to {filename}")

        except Exception as e:
            logger.error(f"❌ Error exporting data: {e}")


# WebSocket server for real-time data broadcasting
class DataWebSocketServer:
    """WebSocket server for broadcasting real-time market data."""

    def __init__(self, data_layer: DataIntegrationLayer, host: str = 'localhost', port: int = 8765):
        self.data_layer = data_layer
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None

    async def start_server(self):
        """Start the WebSocket server."""
        try:
            import websockets

            async def handler(websocket, path):
                self.clients.unified_math.add(websocket)
                try:
                    async for message in websocket:
                        # Handle client messages if needed
                        pass
                finally:
                    self.clients.remove(websocket)

            self.server = await websockets.serve(handler, self.host, self.port)
            logger.info(f"🌐 WebSocket server started on ws://{self.host}:{self.port}")

        except ImportError:
            logger.warning("WebSockets not available. Install with: pip install websockets")
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")

    async def broadcast_data(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients."""
        if not self.clients:
            return

        try:
            import websockets
            message = json.dumps(data)
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"❌ Error broadcasting data: {e}")


# Example usage and testing
async def main():
    """Test the data integration layer."""
    logging.basicConfig(level=logging.INFO)

    # Initialize data layer
    data_layer = DataIntegrationLayer(update_interval=30.0)  # 30 seconds for testing

    # Start data feed
    data_task = asyncio.create_task(data_layer.start_data_feed())

    # Wait for some data to accumulate
    await asyncio.sleep(60)

    # Print current data
    current_data = data_layer.get_current_data()
    safe_print("📊 Current Market Data:")
    print(json.dumps(current_data, indent=2))

    # Export data
    data_layer.export_data('market_data_export.json')

    # Stop data feed
    await data_layer.stop_data_feed()
    data_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
