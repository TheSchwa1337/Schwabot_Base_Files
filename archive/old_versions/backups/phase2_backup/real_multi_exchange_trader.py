"""Module for Schwabot trading system."""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

    try:
    import ccxt.async_support as ccxt  # type: ignore
        except ImportError:
        ccxt = None  # type: ignore
        print("❌ CCXT not installed. Run: pip install ccxt")

        from utils.secure_config_manager import SecureConfigManager

        #!/usr/bin/env python3
        """
        Real Multi-Exchange Trading Engine
        Supports multiple exchanges with mathematical arbitrage detection and routing optimization
        Based on Schwabot's mathematical optimization framework'
        """

        # Removed redundant placeholder try/except block

        logger = logging.getLogger(__name__)


            class SupportedExchange(Enum):
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Supported exchanges for real trading."""

            COINBASE_PRO = "coinbasepro"
            BINANCE = "binance"
            BINANCE_US = "binanceus"
            KRAKEN = "kraken"
            KUCOIN = "kucoin"
            BYBIT = "bybit"
            BITFINEX = "bitfinex"
            HUOBI = "huobi"


            @dataclass
                class ExchangeConfig:
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                """Configuration for exchange connection."""

                exchange: SupportedExchange
                api_key: str
                secret: str
            passphrase: Optional[str] = None  # Required for Coinbase Pro
            sandbox: bool = True
            test_connectivity: bool = True


            @dataclass
                class ArbitrageOpportunity:
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                """Represents an arbitrage opportunity between exchanges."""

                buy_exchange: str
                sell_exchange: str
                symbol: str
                buy_price: float
                sell_price: float
                price_diff: float
                profit_percentage: float
                volume_available: float
                estimated_profit: float
                fees_total: float
                net_profit: float
                timestamp: float


                @dataclass
                    class RouteScore:
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """Represents routing score for exchange selection."""

                    exchange: str
                    score: float
                    latency: float
                    fees: float
                    slippage: float
                    liquidity: float


                        class RealMultiExchangeTrader:
    """Class for Schwabot trading functionality."""
                        """Class for Schwabot trading functionality."""
                        """
                        Real trading across multiple exchanges with mathematical optimization.

                            Implements:
                            1. Arbitrage opportunity detection
                            2. Optimal routing based on mathematical scoring
                            3. Multi-exchange order management
                            4. Risk-adjusted execution
                            """

                                def __init__(self) -> None:
                                self.exchanges: Dict[str, Any] = {}
                                self.secure_config = SecureConfigManager()
                                self.active_exchanges = []
                                self.price_cache: Dict[str, Dict[str, float]] = {}
                                self.latency_cache: Dict[str, float] = {}
                                self.fee_cache: Dict[str, Dict[str, float]] = {}

                                # Mathematical parameters
                                self.theta_arb = 0.02  # Arbitrage threshold (0.2%)
                                self.tau_fees = 0.01  # Fee impact threshold (0.1%)
                                self.lambda_latency = 0.1  # Latency weight
                                self.gamma_liquidity = 0.3  # Liquidity weight

                                # Performance tracking
                                self.arbitrage_history: List[ArbitrageOpportunity] = []
                                self.execution_history: List[Dict[str, Any]] = []

                                    async def setup_exchange(self, exchange_name: str, config: Dict[str, Any]) -> bool:
                                    """Setup and test connection to exchange."""
                                        try:
                                            if ccxt is None:
                                            logger.error("CCXT not available")
                                        return False

                                        exchange_class = getattr(ccxt, exchange_name)

                                        # Create exchange instance with proper config
                                        exchange_config = {
                                        "apiKey": config.get("api_key"),
                                        "secret": config.get("secret"),
                                        "sandbox": config.get("sandbox", True),
                                        "enableRateLimit": True,
                                        "timeout": 30000,
                                        }

                                        # Add passphrase for Coinbase Pro
                                            if exchange_name == "coinbasepro":
                                            exchange_config["passphrase"] = config.get("passphrase")

                                            exchange = exchange_class(exchange_config)

                                            # Test connection
                                                if config.get("test_connectivity", True):
                                                await self._test_exchange_connection(exchange, exchange_name)

                                                self.exchanges[exchange_name] = exchange
                                                self.active_exchanges.append(exchange_name)

                                                # Initialize caches
                                                self.price_cache[exchange_name] = {}
                                                self.latency_cache[exchange_name] = 0.0
                                                self.fee_cache[exchange_name] = {}

                                                logger.info("✅ {0} connected successfully".format(exchange_name.upper()))
                                            return True

                                                except Exception as e:
                                                logger.error("❌ Failed to setup {0}: {1}".format(exchange_name, e))
                                            return False

                                                async def _test_exchange_connection(self, exchange, name: str):
                                                """Test exchange connection and API permissions."""
                                                    try:
                                                    start_time = time.time()

                                                    # Test basic connectivity
                                                    await exchange.load_markets()

                                                    # Test balance access (requires read, permission)
                                                    balance = await exchange.fetch_balance()

                                                    # Record latency
                                                    latency = time.time() - start_time
                                                    self.latency_cache[name] = latency

                                                    logger.info()
                                                    f"🔗 {name.upper()} connection test passed (latency: {latency:.3f}s)"
                                                    )

                                                        except Exception as e:
                                                    raise Exception("Connection test failed for {0}: {1}".format(name, e))

                                                        def calculate_arbitrage_delta(self, price_1: float, price_2: float) -> float:
                                                        """
                                                        Calculate arbitrage price differential.

                                                            Mathematical formula:
                                                            Δ_arbitrage = abs(P_exchange1 - P_exchange2)
                                                            """
                                                        return abs(price_1 - price_2)

                                                        def evaluate_arbitrage_opportunity()
                                                        self,
                                                        buy_price: float,
                                                        sell_price: float,
                                                        buy_exchange: str,
                                                        sell_exchange: str,
                                                        symbol: str,
                                                        volume: float,
                                                            ) -> Optional[ArbitrageOpportunity]:
                                                            """
                                                            Evaluate arbitrage opportunity using mathematical criteria.

                                                                Criteria:
                                                                    if Δ_arbitrage > θ_arb and fee_impact < τ_fees:
                                                                    Execute_Arb_BuySell(P₁, P₂)
                                                                    """
                                                                        if buy_price >= sell_price:
                                                                    return None

                                                                    price_diff = sell_price - buy_price
                                                                    profit_percentage = price_diff / buy_price

                                                                    # Check arbitrage threshold
                                                                        if profit_percentage < self.theta_arb:
                                                                    return None

                                                                    # Calculate fees
                                                                    buy_fee = self.fee_cache.get(buy_exchange, {}).get("taker", 0.01)
                                                                    sell_fee = self.fee_cache.get(sell_exchange, {}).get("taker", 0.01)
                                                                    fees_total = (buy_price * buy_fee) + (sell_price * sell_fee)

                                                                    # Calculate net profit
                                                                    gross_profit = price_diff * volume
                                                                    net_profit = gross_profit - fees_total

                                                                    # Check fee impact threshold
                                                                    fee_impact = fees_total / gross_profit
                                                                        if fee_impact > self.tau_fees:
                                                                    return None

                                                                return ArbitrageOpportunity()
                                                                buy_exchange = buy_exchange,
                                                                sell_exchange = sell_exchange,
                                                                symbol = symbol,
                                                                buy_price = buy_price,
                                                                sell_price = sell_price,
                                                                price_diff = price_diff,
                                                                profit_percentage = profit_percentage,
                                                                volume_available = volume,
                                                                estimated_profit = gross_profit,
                                                                fees_total = fees_total,
                                                                net_profit = net_profit,
                                                                timestamp = time.time(),
                                                                )

                                                                def calculate_route_score()
                                                                self, exchange: str, symbol: str, amount: float
                                                                    ) -> RouteScore:
                                                                    """
                                                                    Calculate routing score for exchange selection.

                                                                        Mathematical formula:
                                                                        Route_Score = (Profit_Expected - Slippage - Fees) / Latency
                                                                        """
                                                                        # Get cached metrics
                                                                        latency = self.latency_cache.get(exchange, 1.0)
                                                                        fees = self.fee_cache.get(exchange, {}).get("taker", 0.01)

                                                                        # Estimate slippage (simplified, model)
                                                                        slippage = amount * 0.001  # 0.1% per unit

                                                                        # Estimate liquidity (simplified)
                                                                        liquidity = 1.0  # Would be based on order book depth

                                                                        # Calculate composite score
                                                                        fee_cost = amount * fees
                                                                        total_cost = fee_cost + slippage

                                                                        # Score formula: higher is better
                                                                        score = (liquidity * self.gamma_liquidity - total_cost) / ()
                                                                        latency * self.lambda_latency + 1e-6
                                                                        )

                                                                    return RouteScore()
                                                                    exchange = exchange,
                                                                    score = score,
                                                                    latency = latency,
                                                                    fees = fee_cost,
                                                                    slippage = slippage,
                                                                    liquidity = liquidity,
                                                                    )

                                                                    def select_optimal_exchange()
                                                                    self, symbol: str, amount: float, exclude_exchanges: List[str] = None
                                                                        ) -> Optional[str]:
                                                                        """Select optimal exchange based on routing scores."""
                                                                        exclude_exchanges = exclude_exchanges or []

                                                                        available_exchanges = []
                                                                        ex for ex in self.active_exchanges if ex not in exclude_exchanges
                                                                        ]
                                                                            if not available_exchanges:
                                                                        return None

                                                                        # Calculate scores for all available exchanges
                                                                        scores = []
                                                                            for exchange in available_exchanges:
                                                                            score = self.calculate_route_score(exchange, symbol, amount)
                                                                            scores.append(score)

                                                                            # Select exchange with highest score
                                                                            best_score = max(scores, key=lambda x: x.score)

                                                                            logger.info()
                                                                            f"Selected {best_score.exchange} for {symbol} (score: {best_score.score:.3f})"
                                                                            )
                                                                        return best_score.exchange

                                                                        async def scan_arbitrage_opportunities()
                                                                        self, symbols: List[str]
                                                                            ) -> List[ArbitrageOpportunity]:
                                                                            """Scan for arbitrage opportunities across all exchanges."""
                                                                            opportunities = []

                                                                                if len(self.active_exchanges) < 2:
                                                                            return opportunities

                                                                                for symbol in symbols:
                                                                                    try:
                                                                                    # Fetch prices from all exchanges
                                                                                    prices = {}
                                                                                        for exchange_name in self.active_exchanges:
                                                                                        exchange = self.exchanges[exchange_name]
                                                                                            try:
                                                                                            ticker = await exchange.fetch_ticker(symbol)
                                                                                            prices[exchange_name] = {}
                                                                                            "bid": ticker.get("bid", 0),
                                                                                            "ask": ticker.get("ask", 0),
                                                                                            "volume": ticker.get("baseVolume", 0),
                                                                                            }
                                                                                                except Exception as e:
                                                                                                logger.warning()
                                                                                                "Failed to fetch {0} from {1}: {2}".format()
                                                                                                symbol, exchange_name, e
                                                                                                )
                                                                                                )
                                                                                            continue

                                                                                            # Find arbitrage opportunities
                                                                                                for buy_exchange, buy_data in prices.items():
                                                                                                    for sell_exchange, sell_data in prices.items():
                                                                                                        if buy_exchange == sell_exchange:
                                                                                                    continue

                                                                                                    # Use ask price for buying, bid price for selling
                                                                                                    buy_price = buy_data["ask"]
                                                                                                    sell_price = sell_data["bid"]
                                                                                                    volume = ()
                                                                                                    min(buy_data["volume"], sell_data["volume"]) * 0.1
                                                                                                    )  # 1% of volume

                                                                                                        if buy_price > 0 and sell_price > 0:
                                                                                                        opportunity = self.evaluate_arbitrage_opportunity()
                                                                                                        buy_price,
                                                                                                        sell_price,
                                                                                                        buy_exchange,
                                                                                                        sell_exchange,
                                                                                                        symbol,
                                                                                                        volume,
                                                                                                        )

                                                                                                            if opportunity:
                                                                                                            opportunities.append(opportunity)
                                                                                                            logger.info()
                                                                                                            f"Arbitrage opportunity: {symbol} {buy_exchange}→{sell_exchange} "
                                                                                                            f"profit: {opportunity.profit_percentage:.2%}"
                                                                                                            )

                                                                                                                except Exception as e:
                                                                                                                logger.error("Error scanning arbitrage for {0}: {1}".format(symbol, e))

                                                                                                                # Store in history
                                                                                                                self.arbitrage_history.extend(opportunities)

                                                                                                                # Limit history size
                                                                                                                    if len(self.arbitrage_history) > 1000:
                                                                                                                    self.arbitrage_history = self.arbitrage_history[-1000:]

                                                                                                                return opportunities

                                                                                                                async def execute_arbitrage_trade()
                                                                                                                self, opportunity: ArbitrageOpportunity
                                                                                                                    ) -> Dict[str, Any]:
                                                                                                                    """Execute arbitrage trade across two exchanges."""
                                                                                                                        try:
                                                                                                                        # Execute buy order
                                                                                                                        buy_exchange = self.exchanges[opportunity.buy_exchange]
                                                                                                                        buy_order = await buy_exchange.create_market_buy_order()
                                                                                                                        opportunity.symbol, opportunity.volume_available
                                                                                                                        )

                                                                                                                        # Execute sell order
                                                                                                                        sell_exchange = self.exchanges[opportunity.sell_exchange]
                                                                                                                        sell_order = await sell_exchange.create_market_sell_order()
                                                                                                                        opportunity.symbol, opportunity.volume_available
                                                                                                                        )

                                                                                                                        # Calculate actual profit
                                                                                                                        actual_buy_cost = buy_order.get("cost", 0)
                                                                                                                        actual_sell_revenue = sell_order.get("cost", 0)
                                                                                                                        actual_profit = actual_sell_revenue - actual_buy_cost

                                                                                                                        result = {}
                                                                                                                        "success": True,
                                                                                                                        "opportunity": opportunity,
                                                                                                                        "buy_order": buy_order,
                                                                                                                        "sell_order": sell_order,
                                                                                                                        "actual_profit": actual_profit,
                                                                                                                        "expected_profit": opportunity.net_profit,
                                                                                                                        "profit_variance": actual_profit - opportunity.net_profit,
                                                                                                                        "timestamp": time.time(),
                                                                                                                        }

                                                                                                                        # Store in execution history
                                                                                                                        self.execution_history.append(result)

                                                                                                                        logger.info()
                                                                                                                        f"🚀 Arbitrage executed: {opportunity.symbol} profit: ${actual_profit:.2f}"
                                                                                                                        )

                                                                                                                    return result

                                                                                                                        except Exception as e:
                                                                                                                        logger.error("❌ Arbitrage execution failed: {0}".format(e))
                                                                                                                    return {}
                                                                                                                    "success": False,
                                                                                                                    "error": str(e),
                                                                                                                    "opportunity": opportunity,
                                                                                                                    "timestamp": time.time(),
                                                                                                                    }

                                                                                                                    async def execute_optimal_trade()
                                                                                                                    self, symbol: str, side: str, amount: float, order_type: str = "market"
                                                                                                                        ) -> Dict[str, Any]:
                                                                                                                        """Execute trade on optimal exchange."""

                                                                                                                        # Select optimal exchange
                                                                                                                        optimal_exchange = self.select_optimal_exchange(symbol, amount)

                                                                                                                            if not optimal_exchange:
                                                                                                                        return {"error": "No suitable exchange found"}

                                                                                                                    return await self.execute_real_trade()
                                                                                                                    optimal_exchange, symbol, side, amount, order_type
                                                                                                                    )

                                                                                                                    async def execute_real_trade()
                                                                                                                    self,
                                                                                                                    exchange_name: str,
                                                                                                                    symbol: str,
                                                                                                                    side: str,
                                                                                                                    amount: float,
                                                                                                                    order_type: str = "market",
                                                                                                                        ) -> Dict[str, Any]:
                                                                                                                        """Execute real trade on specified exchange."""
                                                                                                                            if exchange_name not in self.exchanges:
                                                                                                                        return {"error": "Exchange {0} not configured".format(exchange_name)}

                                                                                                                        exchange = self.exchanges[exchange_name]

                                                                                                                            try:
                                                                                                                            start_time = time.time()

                                                                                                                                if order_type == "market":
                                                                                                                                    if side.lower() == "buy":
                                                                                                                                    order = await exchange.create_market_buy_order(symbol, amount)
                                                                                                                                        elif side.lower() == "sell":
                                                                                                                                        order = await exchange.create_market_sell_order(symbol, amount)
                                                                                                                                            else:
                                                                                                                                        return {"error": "Invalid side: {0}".format(side)}
                                                                                                                                            else:
                                                                                                                                        return {"error": "Order type {0} not implemented".format(order_type)}

                                                                                                                                        execution_time = time.time() - start_time

                                                                                                                                        # Update latency cache
                                                                                                                                        self.latency_cache[exchange_name] = execution_time

                                                                                                                                        result = {}
                                                                                                                                        "success": True,
                                                                                                                                        "exchange": exchange_name,
                                                                                                                                        "order": order,
                                                                                                                                        "trade_id": order.get("id"),
                                                                                                                                        "symbol": symbol,
                                                                                                                                        "side": side,
                                                                                                                                        "amount": amount,
                                                                                                                                        "price": order.get("price"),
                                                                                                                                        "cost": order.get("cost"),
                                                                                                                                        "fee": order.get("fee"),
                                                                                                                                        "timestamp": order.get("timestamp"),
                                                                                                                                        "execution_time": execution_time,
                                                                                                                                        }

                                                                                                                                        # Store in execution history
                                                                                                                                        self.execution_history.append(result)

                                                                                                                                        logger.info()
                                                                                                                                        "🚀 REAL TRADE EXECUTED on {}: {} {} {} in {:.3f}s".format()
                                                                                                                                        exchange_name.upper(), side, amount, symbol, execution_time
                                                                                                                                        )
                                                                                                                                        )

                                                                                                                                    return result

                                                                                                                                        except Exception as e:
                                                                                                                                        logger.error()
                                                                                                                                        "❌ Trade execution failed on {0}: {1}".format(exchange_name, e)
                                                                                                                                        )
                                                                                                                                    return {"error": str(e), "exchange": exchange_name}

                                                                                                                                        async def update_fee_cache(self) -> None:
                                                                                                                                        """Update fee information for all exchanges."""
                                                                                                                                            for exchange_name in self.active_exchanges:
                                                                                                                                                try:
                                                                                                                                                exchange = self.exchanges[exchange_name]
                                                                                                                                                markets = await exchange.load_markets()

                                                                                                                                                # Calculate average fees
                                                                                                                                                total_maker_fee = 0
                                                                                                                                                total_taker_fee = 0
                                                                                                                                                count = 0

                                                                                                                                                    for market in markets.values():
                                                                                                                                                    if ()
                                                                                                                                                    market.get("maker") is not None
                                                                                                                                                    and market.get("taker") is not None
                                                                                                                                                        ):
                                                                                                                                                        total_maker_fee += market["maker"]
                                                                                                                                                        total_taker_fee += market["taker"]
                                                                                                                                                        count += 1

                                                                                                                                                            if count > 0:
                                                                                                                                                            self.fee_cache[exchange_name] = {}
                                                                                                                                                            "maker": total_maker_fee / count,
                                                                                                                                                            "taker": total_taker_fee / count,
                                                                                                                                                            }

                                                                                                                                                                except Exception as e:
                                                                                                                                                                logger.warning()
                                                                                                                                                                "Failed to update fees for {0}: {1}".format(exchange_name, e)
                                                                                                                                                                )

                                                                                                                                                                    def unified_exchange_hash_router(self, input_vector: List[float]) -> str:
                                                                                                                                                                    """
                                                                                                                                                                    Generate unified hash for exchange routing.

                                                                                                                                                                        Mathematical formula:
                                                                                                                                                                        hash(str(input_vector)) % 1024
                                                                                                                                                                        """
                                                                                                                                                                        hash_value = hash(str(input_vector)) % 1024
                                                                                                                                                                    return f"route_{hash_value:04d}"

                                                                                                                                                                        def get_performance_metrics(self) -> Dict[str, Any]:
                                                                                                                                                                        """Get comprehensive performance metrics."""
                                                                                                                                                                        total_trades = len(self.execution_history)
                                                                                                                                                                        successful_trades = sum()
                                                                                                                                                                        1 for trade in self.execution_history if trade.get("success", False)
                                                                                                                                                                        )

                                                                                                                                                                        arbitrage_count = len(self.arbitrage_history)
                                                                                                                                                                        total_arbitrage_profit = sum(arb.net_profit for arb in self.arbitrage_history)

                                                                                                                                                                        avg_latency = ()
                                                                                                                                                                        np.mean(list(self.latency_cache.values())) if self.latency_cache else 0.0
                                                                                                                                                                        )

                                                                                                                                                                    return {}
                                                                                                                                                                    "total_trades": total_trades,
                                                                                                                                                                    "successful_trades": successful_trades,
                                                                                                                                                                    "success_rate": successful_trades / max(1, total_trades),
                                                                                                                                                                    "arbitrage_opportunities": arbitrage_count,
                                                                                                                                                                    "total_arbitrage_profit": total_arbitrage_profit,
                                                                                                                                                                    "average_latency": avg_latency,
                                                                                                                                                                    "active_exchanges": len(self.active_exchanges),
                                                                                                                                                                    "exchange_latencies": self.latency_cache.copy(),
                                                                                                                                                                    }

                                                                                                                                                                        async def get_account_balance(self, exchange_name: str) -> Dict[str, Any]:
                                                                                                                                                                        """Get account balance from exchange."""
                                                                                                                                                                            if exchange_name not in self.exchanges:
                                                                                                                                                                        return {"error": "Exchange {0} not configured".format(exchange_name)}

                                                                                                                                                                            try:
                                                                                                                                                                            exchange = self.exchanges[exchange_name]
                                                                                                                                                                            balance = await exchange.fetch_balance()
                                                                                                                                                                        return {}
                                                                                                                                                                        "success": True,
                                                                                                                                                                        "balance": balance,
                                                                                                                                                                        "exchange": exchange_name,
                                                                                                                                                                        }
                                                                                                                                                                            except Exception as e:
                                                                                                                                                                        return {"error": str(e), "exchange": exchange_name}

                                                                                                                                                                            async def get_market_price(self, exchange_name: str, symbol: str) -> Dict[str, Any]:
                                                                                                                                                                            """Get current market price from exchange."""
                                                                                                                                                                                if exchange_name not in self.exchanges:
                                                                                                                                                                            return {"error": "Exchange {0} not configured".format(exchange_name)}

                                                                                                                                                                                try:
                                                                                                                                                                                exchange = self.exchanges[exchange_name]
                                                                                                                                                                                ticker = await exchange.fetch_ticker(symbol)

                                                                                                                                                                                # Update price cache
                                                                                                                                                                                self.price_cache[exchange_name][symbol] = ticker.get("last", 0)

                                                                                                                                                                            return {}
                                                                                                                                                                            "success": True,
                                                                                                                                                                            "symbol": symbol,
                                                                                                                                                                            "price": ticker.get("last"),
                                                                                                                                                                            "bid": ticker.get("bid"),
                                                                                                                                                                            "ask": ticker.get("ask"),
                                                                                                                                                                            "volume": ticker.get("baseVolume"),
                                                                                                                                                                            "exchange": exchange_name,
                                                                                                                                                                            }
                                                                                                                                                                                except Exception as e:
                                                                                                                                                                            return {"error": str(e), "exchange": exchange_name}

                                                                                                                                                                                async def initialize_all_exchanges(self) -> Dict[str, bool]:
                                                                                                                                                                                """Initialize all supported exchanges with stored credentials."""
                                                                                                                                                                                results = {}

                                                                                                                                                                                    for exchange in SupportedExchange:
                                                                                                                                                                                    exchange_name = exchange.value

                                                                                                                                                                                        try:
                                                                                                                                                                                        # Get stored credentials
                                                                                                                                                                                        api_key = self.secure_config.get_secure_api_key()
                                                                                                                                                                                        "{0}_api_key".format(exchange_name)
                                                                                                                                                                                        )
                                                                                                                                                                                        secret = self.secure_config.get_secure_api_key()
                                                                                                                                                                                        "{0}_secret".format(exchange_name)
                                                                                                                                                                                        )

                                                                                                                                                                                            if api_key and secret:
                                                                                                                                                                                            config = {}
                                                                                                                                                                                            "api_key": api_key,
                                                                                                                                                                                            "secret": secret,
                                                                                                                                                                                            "sandbox": True,  # Use sandbox by default
                                                                                                                                                                                            "test_connectivity": True,
                                                                                                                                                                                            }

                                                                                                                                                                                            # Add passphrase for Coinbase Pro
                                                                                                                                                                                                if exchange_name == "coinbasepro":
                                                                                                                                                                                            passphrase = self.secure_config.get_secure_api_key()
                                                                                                                                                                                            "{0}_passphrase".format(exchange_name)
                                                                                                                                                                                            )
                                                                                                                                                                                                if passphrase:
                                                                                                                                                                                                config["passphrase"] = passphrase

                                                                                                                                                                                                success = await self.setup_exchange(exchange_name, config)
                                                                                                                                                                                                results[exchange_name] = success
                                                                                                                                                                                                    else:
                                                                                                                                                                                                    logger.warning("No credentials found for {0}".format(exchange_name))
                                                                                                                                                                                                    results[exchange_name] = False

                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                        logger.error("Failed to initialize {0}: {1}".format(exchange_name, e))
                                                                                                                                                                                                        results[exchange_name] = False

                                                                                                                                                                                                        # Update fee cache for successfully connected exchanges
                                                                                                                                                                                                        await self.update_fee_cache()

                                                                                                                                                                                                    return results

                                                                                                                                                                                                        async def close_all_connections(self) -> None:
                                                                                                                                                                                                        """Close all exchange connections."""
                                                                                                                                                                                                            for exchange_name, exchange in self.exchanges.items():
                                                                                                                                                                                                                try:
                                                                                                                                                                                                                await exchange.close()
                                                                                                                                                                                                                logger.info("Closed connection to {0}".format(exchange_name))
                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                    logger.error("Error closing {0}: {1}".format(exchange_name, e))

                                                                                                                                                                                                                    self.exchanges.clear()
                                                                                                                                                                                                                    self.active_exchanges.clear()


                                                                                                                                                                                                                    # Global instance for easy access
                                                                                                                                                                                                                    _global_trader = None


                                                                                                                                                                                                                        def get_multi_exchange_trader() -> RealMultiExchangeTrader:
                                                                                                                                                                                                                        """Get global multi-exchange trader instance."""
                                                                                                                                                                                                                        global _global_trader
                                                                                                                                                                                                                            if _global_trader is None:
                                                                                                                                                                                                                            _global_trader = RealMultiExchangeTrader()
                                                                                                                                                                                                                        return _global_trader


                                                                                                                                                                                                                            async def main():
                                                                                                                                                                                                                            """Main function for testing."""
                                                                                                                                                                                                                            trader = RealMultiExchangeTrader()

                                                                                                                                                                                                                            # Initialize exchanges
                                                                                                                                                                                                                            results = await trader.initialize_all_exchanges()
                                                                                                                                                                                                                            print("Exchange initialization results:", results)

                                                                                                                                                                                                                            # Scan for arbitrage opportunities
                                                                                                                                                                                                                            opportunities = await trader.scan_arbitrage_opportunities(["BTC/USDT", "ETH/USDT"])
                                                                                                                                                                                                                            print("Found {0} arbitrage opportunities".format(len(opportunities)))

                                                                                                                                                                                                                            # Get performance metrics
                                                                                                                                                                                                                            metrics = trader.get_performance_metrics()
                                                                                                                                                                                                                            print("Performance metrics:", metrics)

                                                                                                                                                                                                                            # Close connections
                                                                                                                                                                                                                            await trader.close_all_connections()


                                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                                                asyncio.run(main())
