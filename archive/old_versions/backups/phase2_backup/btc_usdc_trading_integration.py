"""Module for Schwabot trading system."""

#!/usr/bin/env python3
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .algorithmic_portfolio_balancer import AlgorithmicPortfolioBalancer
from .ccxt_trading_executor import CCXTTradingExecutor
from .clean_trading_pipeline import TradingAction, TradingDecision
from .phantom_registry import PhantomRegistry

"""
BTC/USDC Trading Integration
============================

    Specialized integration for BTC/USDC trading with:
    - Optimized order execution
    - Enhanced risk management
    - Market microstructure analysis
    - Integration with Phantom Math
    - Portfolio balancing coordination
    """

    logger = logging.getLogger(__name__)


    @dataclass
        class BTCUSDCTradingConfig:
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """BTC/USDC trading configuration."""

        symbol: str = "BTC/USDC"
        base_order_size: float = 0.01  # 0.01 BTC
        max_order_size: float = 0.1  # 0.1 BTC max
        min_order_size: float = 0.001  # 0.001 BTC min

        # Risk management
        max_position_size_pct: float = 0.2  # 20% max position
        stop_loss_pct: float = 0.2  # 2% stop loss
        take_profit_pct: float = 0.5  # 5% take profit
        max_daily_trades: int = 50

        # Market analysis
        order_book_depth: int = 100
        spread_threshold: float = 0.01  # 0.1% max spread
        volume_threshold: float = 1000000  # $1M min volume

        # Phantom Math integration
        phantom_entropy_threshold: float = 0.6
        phantom_potential_threshold: float = 0.7

        # Portfolio integration
        enable_portfolio_balancing: bool = True
        rebalance_threshold: float = 0.5  # 5% rebalance threshold


            class BTCUSDCTradingIntegration:
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Specialized BTC/USDC trading integration."""

                def __init__(self, config: Dict[str, Any]) -> None:
                self.config = BTCUSDCTradingConfig(**config.get("btc_usdc_config", {}))

                # Initialize components
                self.trading_executor = CCXTTradingExecutor(config.get("exchange_config", {}))
                self.portfolio_balancer = AlgorithmicPortfolioBalancer(config.get("portfolio_config", {}))
                self.phantom_registry = PhantomRegistry()

                # Trading state
                self.current_position = 0.0
                self.daily_trades = 0
                self.last_trade_time = 0
                self.position_entry_price = 0.0

                # Market data cache
                self.market_data_cache = {}
                self.order_book_cache = {}

                # Performance tracking
                self.trade_history = []
                self.performance_metrics = {}

                logger.info("BTC/USDC Trading Integration initialized")

                    async def initialize(self) -> bool:
                    """Initialize the trading integration."""
                        try:
                        # Initialize trading executor
                            if not await self.trading_executor.initialize():
                            logger.error("Failed to initialize trading executor")
                        return False

                        # Initialize portfolio balancer
                        await self.portfolio_balancer.update_portfolio_state({})

                        # Load initial market data
                        await self._update_market_data()

                        logger.info("BTC/USDC Trading Integration initialized successfully")
                    return True

                        except Exception as e:
                        logger.error(f"Error initializing BTC/USDC trading integration: {e}")
                    return False

                        async def process_market_data(self, market_data: Dict[str, Any]) -> Optional[TradingDecision]:
                        """Process market data and generate trading decisions."""
                            try:
                            # Update market data cache
                            self.market_data_cache.update(market_data)

                            # Check if we should trade
                                if not self._should_trade():
                            return None

                            # Analyze market conditions
                            market_analysis = await self._analyze_market_conditions()

                            # Check Phantom Math signals
                            phantom_signal = await self._check_phantom_signals()

                            # Check portfolio balancing needs
                            portfolio_signal = await self._check_portfolio_balancing()

                            # Generate trading decision
                            decision = await self._generate_trading_decision(market_analysis, phantom_signal, portfolio_signal)

                        return decision

                            except Exception as e:
                            logger.error(f"Error processing market data: {e}")
                        return None

                            async def execute_trade(self, decision: TradingDecision) -> bool:
                            """Execute a trading decision."""
                                try:
                                # Validate decision
                                    if not self._validate_trading_decision(decision):
                                return False

                                # Execute trade
                                success = await self.trading_executor.execute_decision(decision)

                                    if success:
                                    # Update position
                                    await self._update_position(decision)

                                    # Update portfolio state
                                    await self.portfolio_balancer.update_portfolio_state(self.market_data_cache)

                                    # Log trade
                                    self._log_trade(decision)

                                    # Check if rebalancing is needed
                                        if self.config.enable_portfolio_balancing:
                                        await self._check_and_execute_rebalancing()

                                        logger.info(f"Trade executed: {decision.symbol} {decision.action.value} {decision.quantity}")
                                    return True
                                        else:
                                        logger.warning(f"Trade execution failed: {decision.symbol}")
                                    return False

                                        except Exception as e:
                                        logger.error(f"Error executing trade: {e}")
                                    return False

                                        async def validate_trading_decision(self, decision: TradingDecision) -> bool:
                                        """Validate trading decision (async wrapper for _validate_trading_decision)."""
                                            try:
                                        return self._validate_trading_decision(decision)
                                            except Exception as e:
                                            logger.error(f"Error validating trading decision: {e}")
                                        return False

                                            async def calculate_performance_metrics(self) -> Dict[str, Any]:
                                            """Calculate comprehensive performance metrics."""
                                                try:
                                                # Calculate additional metrics
                                                    if self.trade_history:
                                                    # Calculate win rate
                                                    winning_trades = sum(1 for trade in self.trade_history if trade.get("pnl", 0) > 0)
                                                    total_trades = len(self.trade_history)
                                                    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

                                                    # Calculate average profit/loss
                                                    profits = [trade.get("pnl", 0) for trade in self.trade_history if trade.get("pnl", 0) > 0]
                                                    losses = [trade.get("pnl", 0) for trade in self.trade_history if trade.get("pnl", 0) < 0]

                                                    avg_profit = np.mean(profits) if profits else 0.0
                                                    avg_loss = np.mean(losses) if losses else 0.0

                                                    # Calculate profit factor
                                                    total_profit = sum(profits) if profits else 0.0
                                                    total_loss = abs(sum(losses)) if losses else 0.0
                                                    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

                                                    # Calculate max drawdown
                                                    cumulative_pnl = []
                                                    running_total = 0.0
                                                        for trade in self.trade_history:
                                                        running_total += trade.get("pnl", 0)
                                                        cumulative_pnl.append(running_total)

                                                        max_drawdown = 0.0
                                                        peak = 0.0
                                                            for pnl in cumulative_pnl:
                                                                if pnl > peak:
                                                                peak = pnl
                                                                drawdown = (peak - pnl) / peak if peak > 0 else 0.0
                                                                max_drawdown = max(max_drawdown, drawdown)
                                                                    else:
                                                                    win_rate = 0.0
                                                                    avg_profit = 0.0
                                                                    avg_loss = 0.0
                                                                    profit_factor = 0.0
                                                                    max_drawdown = 0.0

                                                                    # Combine metrics
                                                                    performance_metrics = {
                                                                    "win_rate": win_rate,
                                                                    "avg_profit": avg_profit,
                                                                    "avg_loss": avg_loss,
                                                                    "profit_factor": profit_factor,
                                                                    "max_drawdown": max_drawdown,
                                                                    "total_trades": len(self.trade_history),
                                                                    "current_position": self.current_position,
                                                                    "daily_trades": self.daily_trades,
                                                                    }

                                                                return performance_metrics

                                                                    except Exception as e:
                                                                    logger.error(f"Error calculating performance metrics: {e}")
                                                                return {
                                                                "win_rate": 0.0,
                                                                "avg_profit": 0.0,
                                                                "avg_loss": 0.0,
                                                                "profit_factor": 0.0,
                                                                "max_drawdown": 0.0,
                                                                "total_trades": 0,
                                                                "current_position": 0.0,
                                                                "daily_trades": 0,
                                                                }

                                                                    async def get_system_status(self) -> Dict[str, Any]:
                                                                    """Get system status information."""
                                                                        try:
                                                                        # Check if trading executor is available
                                                                        executor_status = "available" if self.trading_executor else "not_available"

                                                                        # Check if portfolio balancer is available
                                                                        balancer_status = "available" if self.portfolio_balancer else "not_available"

                                                                        # Check market data freshness
                                                                        current_time = time.time()
                                                                        btc_data = self.market_data_cache.get("BTC", {})
                                                                        last_update = btc_data.get("timestamp", 0)
                                                                        data_age = current_time - last_update if last_update > 0 else float("inf")

                                                                        data_status = "fresh" if data_age < 60 else "stale" if data_age < 300 else "very_stale"

                                                                        # Overall system status
                                                                            if executor_status == "available" and balancer_status == "available" and data_status == "fresh":
                                                                            overall_status = "operational"
                                                                                elif executor_status == "available" and balancer_status == "available":
                                                                                overall_status = "degraded"
                                                                                    else:
                                                                                    overall_status = "error"

                                                                                return {
                                                                                "overall_status": overall_status,
                                                                                "executor_status": executor_status,
                                                                                "balancer_status": balancer_status,
                                                                                "data_status": data_status,
                                                                                "data_age_seconds": data_age,
                                                                                }

                                                                                    except Exception as e:
                                                                                    logger.error(f"Error getting system status: {e}")
                                                                                return {"overall_status": "error", "reason": str(e)}

                                                                                    async def _analyze_market_conditions(self) -> Dict[str, Any]:
                                                                                    """Analyze current market conditions."""
                                                                                        try:
                                                                                        analysis = {}

                                                                                        # Get current price and volume
                                                                                        btc_data = self.market_data_cache.get("BTC", {})
                                                                                        current_price = btc_data.get("price", 0)
                                                                                        volume_24h = btc_data.get("volume", 0)

                                                                                        # Check volume threshold
                                                                                        volume_sufficient = volume_24h >= self.config.volume_threshold

                                                                                        # Analyze order book
                                                                                        order_book = self.order_book_cache.get("BTC/USDC", {})
                                                                                        spread = self._calculate_spread(order_book)
                                                                                        spread_acceptable = spread <= self.config.spread_threshold

                                                                                        # Calculate volatility
                                                                                        volatility = self._calculate_volatility()

                                                                                        analysis.update(
                                                                                        {
                                                                                        "market_ready": volume_sufficient and spread_acceptable,
                                                                                        "current_price": current_price,
                                                                                        "volume_24h": volume_24h,
                                                                                        "spread": spread,
                                                                                        "volatility": volatility,
                                                                                        }
                                                                                        )

                                                                                    return analysis

                                                                                        except Exception as e:
                                                                                        logger.error(f"Error analyzing market conditions: {e}")
                                                                                    return {"market_ready": False, "reason": str(e)}

                                                                                        async def _check_phantom_signals(self) -> Optional[Dict[str, Any]]:
                                                                                        """Check Phantom Math signals for BTC."""
                                                                                            try:
                                                                                            # Get recent Phantom Zones for BTC
                                                                                            recent_zones = await self.phantom_registry.get_recent_zones("BTC/USDC", hours=1)

                                                                                                if not recent_zones:
                                                                                            return None

                                                                                            # Analyze Phantom Zone characteristics
                                                                                            avg_entropy = np.mean([zone.entropy_score for zone in recent_zones])
                                                                                            avg_potential = np.mean([zone.potential_score for zone in recent_zones])

                                                                                            # Determine signal strength
                                                                                            signal_strength = 0.0
                                                                                            signal_direction = "neutral"

                                                                                                if avg_entropy > self.config.phantom_entropy_threshold:
                                                                                                    if avg_potential > self.config.phantom_potential_threshold:
                                                                                                    signal_strength = min(1.0, avg_potential)
                                                                                                    signal_direction = "buy"
                                                                                                        else:
                                                                                                        signal_strength = min(1.0, 1.0 - avg_potential)
                                                                                                        signal_direction = "sell"

                                                                                                    return {
                                                                                                    "signal_strength": signal_strength,
                                                                                                    "signal_direction": signal_direction,
                                                                                                    "avg_entropy": avg_entropy,
                                                                                                    "avg_potential": avg_potential,
                                                                                                    }

                                                                                                        except Exception as e:
                                                                                                        logger.error(f"Error checking Phantom signals: {e}")
                                                                                                    return None

                                                                                                        async def _check_portfolio_balancing(self) -> Optional[Dict[str, Any]]:
                                                                                                        """Check portfolio balancing needs."""
                                                                                                            try:
                                                                                                                if not self.config.enable_portfolio_balancing:
                                                                                                            return None

                                                                                                            # Check if rebalancing is needed
                                                                                                            needs_rebalancing = await self.portfolio_balancer.check_rebalancing_needs()

                                                                                                                if needs_rebalancing:
                                                                                                                # Generate rebalancing decisions
                                                                                                                decisions = await self.portfolio_balancer.generate_rebalancing_decisions(self.market_data_cache)

                                                                                                                # Find BTC-specific rebalancing
                                                                                                                btc_decision = next((d for d in decisions if d.symbol == "BTC/USDC"), None)

                                                                                                                    if btc_decision:
                                                                                                                return {
                                                                                                                "needs_rebalancing": True,
                                                                                                                "action": btc_decision.action,
                                                                                                                "quantity": btc_decision.quantity,
                                                                                                                "confidence": btc_decision.confidence,
                                                                                                                }

                                                                                                            return {"needs_rebalancing": False}

                                                                                                                except Exception as e:
                                                                                                                logger.error(f"Error checking portfolio balancing: {e}")
                                                                                                            return None

                                                                                                            async def _generate_trading_decision(
                                                                                                            self,
                                                                                                            market_analysis: Dict[str, Any],
                                                                                                            phantom_signal: Optional[Dict[str, Any]],
                                                                                                            portfolio_signal: Optional[Dict[str, Any]],
                                                                                                                ) -> Optional[TradingDecision]:
                                                                                                                """Generate trading decision based on all signals."""
                                                                                                                    try:
                                                                                                                    # Check if market is ready
                                                                                                                        if not market_analysis.get("market_ready", False):
                                                                                                                    return None

                                                                                                                    current_price = market_analysis["current_price"]
                                                                                                                        if current_price <= 0:
                                                                                                                    return None

                                                                                                                    # Determine action and quantity
                                                                                                                    action = TradingAction.HOLD
                                                                                                                    quantity = 0.0
                                                                                                                    confidence = 0.0

                                                                                                                    # Portfolio balancing takes priority
                                                                                                                        if portfolio_signal and portfolio_signal.get("needs_rebalancing"):
                                                                                                                        action = portfolio_signal["action"]
                                                                                                                        quantity = portfolio_signal["quantity"]
                                                                                                                        confidence = portfolio_signal["confidence"]

                                                                                                                        # Phantom Math signal as secondary
                                                                                                                            elif phantom_signal and phantom_signal["signal_strength"] > 0.7:
                                                                                                                                if phantom_signal["signal_direction"] == "buy":
                                                                                                                                action = TradingAction.BUY
                                                                                                                                    elif phantom_signal["signal_direction"] == "sell":
                                                                                                                                    action = TradingAction.SELL

                                                                                                                                    quantity = self._calculate_position_size(phantom_signal["signal_strength"])
                                                                                                                                    confidence = phantom_signal["signal_strength"]

                                                                                                                                    # No action if no clear signal
                                                                                                                                        if action == TradingAction.HOLD:
                                                                                                                                    return None

                                                                                                                                    # Create trading decision
                                                                                                                                    decision = TradingDecision(
                                                                                                                                    timestamp=time.time(),
                                                                                                                                    symbol=self.config.symbol,
                                                                                                                                    action=action,
                                                                                                                                    quantity=quantity,
                                                                                                                                    price=current_price,
                                                                                                                                    confidence=confidence,
                                                                                                                                    strategy_branch="btc_usdc_integration",
                                                                                                                                    profit_potential=self.config.take_profit_pct,
                                                                                                                                    risk_score=self.config.stop_loss_pct,
                                                                                                                                    metadata={
                                                                                                                                    "market_analysis": market_analysis,
                                                                                                                                    "phantom_signal": phantom_signal,
                                                                                                                                    "portfolio_signal": portfolio_signal,
                                                                                                                                    "position_size": self.current_position,
                                                                                                                                    },
                                                                                                                                    )

                                                                                                                                return decision

                                                                                                                                    except Exception as e:
                                                                                                                                    logger.error(f"Error generating trading decision: {e}")
                                                                                                                                return None

                                                                                                                                    def _should_trade(self) -> bool:
                                                                                                                                    """Check if we should trade based on constraints."""
                                                                                                                                    current_time = time.time()

                                                                                                                                    # Check daily trade limit
                                                                                                                                        if self.daily_trades >= self.config.max_daily_trades:
                                                                                                                                    return False

                                                                                                                                    # Check minimum time between trades
                                                                                                                                    if current_time - self.last_trade_time < 60:  # 1 minute minimum
                                                                                                                                return False

                                                                                                                            return True

                                                                                                                                def _validate_trading_decision(self, decision: TradingDecision) -> bool:
                                                                                                                                """Validate trading decision."""
                                                                                                                                    try:
                                                                                                                                    # Check quantity limits
                                                                                                                                        if decision.quantity < self.config.min_order_size:
                                                                                                                                    return False

                                                                                                                                        if decision.quantity > self.config.max_order_size:
                                                                                                                                    return False

                                                                                                                                    # Check position size limits
                                                                                                                                        if decision.action == TradingAction.BUY:
                                                                                                                                        new_position = self.current_position + decision.quantity
                                                                                                                                        max_position = self._calculate_max_position()
                                                                                                                                            if new_position > max_position:
                                                                                                                                        return False

                                                                                                                                    return True

                                                                                                                                        except Exception as e:
                                                                                                                                        logger.error(f"Error validating trading decision: {e}")
                                                                                                                                    return False

                                                                                                                                        def _calculate_position_size(self, signal_strength: float) -> float:
                                                                                                                                        """Calculate position size based on signal strength."""
                                                                                                                                        base_size = self.config.base_order_size
                                                                                                                                        size_multiplier = signal_strength

                                                                                                                                        # Apply position size constraints
                                                                                                                                        size = base_size * size_multiplier
                                                                                                                                        size = max(self.config.min_order_size, min(self.config.max_order_size, size))

                                                                                                                                    return size

                                                                                                                                        def _calculate_max_position(self) -> float:
                                                                                                                                        """Calculate maximum allowed position size."""
                                                                                                                                        portfolio_value = float(self.portfolio_balancer.portfolio_state.total_value)
                                                                                                                                        max_position_value = portfolio_value * self.config.max_position_size_pct

                                                                                                                                        current_price = self.market_data_cache.get("BTC", {}).get("price", 1)
                                                                                                                                    return max_position_value / current_price if current_price > 0 else 0

                                                                                                                                        def _calculate_spread(self, order_book: Dict[str, Any]) -> float:
                                                                                                                                        """Calculate bid-ask spread."""
                                                                                                                                            try:
                                                                                                                                            bids = order_book.get("bids", [])
                                                                                                                                            asks = order_book.get("asks", [])

                                                                                                                                                if not bids or not asks:
                                                                                                                                            return float("inf")

                                                                                                                                            best_bid = bids[0][0] if bids else 0
                                                                                                                                            best_ask = asks[0][0] if asks else 0

                                                                                                                                                if best_bid <= 0 or best_ask <= 0:
                                                                                                                                            return float("inf")

                                                                                                                                            spread = (best_ask - best_bid) / best_bid
                                                                                                                                        return spread

                                                                                                                                            except Exception as e:
                                                                                                                                            logger.error(f"Error calculating spread: {e}")
                                                                                                                                        return float("inf")

                                                                                                                                            def _calculate_volatility(self) -> float:
                                                                                                                                            """Calculate price volatility."""
                                                                                                                                                try:
                                                                                                                                                # Use recent price data to calculate volatility
                                                                                                                                                prices = []
                                                                                                                                                for data in list(self.market_data_cache.values())[-20:]:  # Last 20 data points
                                                                                                                                                    if "price" in data:
                                                                                                                                                    prices.append(data["price"])

                                                                                                                                                        if len(prices) < 2:
                                                                                                                                                    return 0.0

                                                                                                                                                returns = np.diff(prices) / prices[:-1]
                                                                                                                                                volatility = np.std(returns)
                                                                                                                                            return volatility

                                                                                                                                                except Exception as e:
                                                                                                                                                logger.error(f"Error calculating volatility: {e}")
                                                                                                                                            return 0.0

                                                                                                                                                async def _update_position(self, decision: TradingDecision) -> None:
                                                                                                                                                """Update current position based on trade decision."""
                                                                                                                                                    if decision.action == TradingAction.BUY:
                                                                                                                                                    self.current_position += decision.quantity
                                                                                                                                                        elif decision.action == TradingAction.SELL:
                                                                                                                                                        self.current_position -= decision.quantity

                                                                                                                                                            async def _check_and_execute_rebalancing(self) -> None:
                                                                                                                                                            """Check for rebalancing needs and execute if necessary."""
                                                                                                                                                            rebalancing_signal = await self._check_portfolio_balancing()
                                                                                                                                                                if rebalancing_signal and rebalancing_signal.get("needs_rebalancing"):
                                                                                                                                                                decision = TradingDecision(
                                                                                                                                                                timestamp=time.time(),
                                                                                                                                                                symbol="BTC/USDC",
                                                                                                                                                                action=rebalancing_signal["action"],
                                                                                                                                                                quantity=rebalancing_signal["quantity"],
                                                                                                                                                                price=self.market_data_cache.get("BTC", {}).get("price", 0),
                                                                                                                                                                confidence=rebalancing_signal["confidence"],
                                                                                                                                                                strategy_branch="portfolio_rebalancing",
                                                                                                                                                                profit_potential=0,
                                                                                                                                                                risk_score=0,
                                                                                                                                                                metadata={"type": "rebalancing"},
                                                                                                                                                                )
                                                                                                                                                                await self.execute_trade(decision)

                                                                                                                                                                    def _log_trade(self, decision: TradingDecision) -> None:
                                                                                                                                                                    """Log trade for performance tracking."""
                                                                                                                                                                        try:
                                                                                                                                                                        trade_record = {
                                                                                                                                                                        "timestamp": decision.timestamp,
                                                                                                                                                                        "symbol": decision.symbol,
                                                                                                                                                                        "action": decision.action.value,
                                                                                                                                                                        "quantity": decision.quantity,
                                                                                                                                                                        "price": decision.price,
                                                                                                                                                                        "confidence": decision.confidence,
                                                                                                                                                                        "position_size": self.current_position,
                                                                                                                                                                        }
                                                                                                                                                                        self.trade_history.append(trade_record)
                                                                                                                                                                        self.daily_trades += 1
                                                                                                                                                                            except Exception as e:
                                                                                                                                                                            logger.error(f"Error logging trade: {e}")

                                                                                                                                                                                async def _update_market_data(self) -> None:
                                                                                                                                                                                """Fetch and update market data (placeholder)."""
                                                                                                                                                                                    try:
                                                                                                                                                                                    # This would fetch real market data from your exchange
                                                                                                                                                                                    # For now, we will use placeholder data
                                                                                                                                                                                    self.market_data_cache["BTC"] = {
                                                                                                                                                                                    "price": 50000.0,
                                                                                                                                                                                    "volume": 2000000,
                                                                                                                                                                                    "timestamp": time.time(),
                                                                                                                                                                                    }
                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                        logger.error(f"Error updating market data: {e}")

                                                                                                                                                                                            async def get_performance_metrics(self) -> Dict[str, Any]:
                                                                                                                                                                                            """Get trading performance metrics."""
                                                                                                                                                                                                try:
                                                                                                                                                                                                metrics = {
                                                                                                                                                                                                "current_position": self.current_position,
                                                                                                                                                                                                "daily_trades": self.daily_trades,
                                                                                                                                                                                                "trade_history_count": len(self.trade_history),
                                                                                                                                                                                                "portfolio_metrics": await self.portfolio_balancer.get_portfolio_metrics(),
                                                                                                                                                                                                }
                                                                                                                                                                                            return metrics
                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                logger.error(f"Error getting performance metrics: {e}")
                                                                                                                                                                                            return {}


                                                                                                                                                                                            # Factory function for easy integration


                                                                                                                                                                                                def create_btc_usdc_integration(config: Dict[str, Any]) -> BTCUSDCTradingIntegration:
                                                                                                                                                                                                """Create a BTC/USDC trading integration instance."""
                                                                                                                                                                                            return BTCUSDCTradingIntegration(config)
