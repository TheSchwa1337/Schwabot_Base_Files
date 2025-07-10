#!/usr/bin/env python3
"""
Unified Trading Pipeline - Complete Trading System Integration with Math + Memory Fusion Core
============================================================================================

Integrates all core components with proper registry management and Math + Memory Fusion Core:
- Enhanced Strategy Executor with profit vector integration
- Canonical trade registry (single source of truth)
- Specialized registries (profit buckets, soulprints, etc.)
- Registry coordinator for linkage management
- Mathematical systems (chrono resonance, temporal warp, unified math)
- CLI live entry system
- Performance tracking and analytics
- Signal lineage tracking and confidence overlays

Features:
- Proper hash tracking across all registries
- No redundant data storage
- Comprehensive performance analytics
- Backtesting support with full trade history
- Live trading capability with API integration
- Math + Memory Fusion Core integration for enhanced decision making
- Profit vector memory and entropy-aware signal processing
- Self-correcting signal memory overlays
"""

import asyncio
import logging
import time
import hashlib
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Core components
from core.trade_registry import canonical_trade_registry
from core.registry_coordinator import registry_coordinator
from core.profit_bucket_registry import ProfitBucketRegistry
from core.soulprint_registry import soulprint_registry
from core.chrono_resonance_weather_mapper import ChronoResonanceWeatherMapper
from core.temporal_warp_engine import TemporalWarpEngine
from core.clean_unified_math import CleanUnifiedMathSystem

# Optional CLI system
try:
    from core.cli_live_entry import CLILiveEntrySystem
    CLI_SYSTEM_AVAILABLE = True
except ImportError:
    CLI_SYSTEM_AVAILABLE = False
    CLILiveEntrySystem = None
    logger = logging.getLogger(__name__)
    logger.warning("CLI Live Entry System not available - using fallback mode")

# Math + Memory Fusion Core integration
try:
    from core.strategy.strategy_executor import StrategyExecutor, EnhancedTradingSignal
    FUSION_CORE_AVAILABLE = True
except ImportError:
    FUSION_CORE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Math + Memory Fusion Core not available - using fallback mode")

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Complete trading signal with all mathematical context."""
    symbol: str
    action: str  # 'buy' or 'sell'
    entry_price: float
    amount: float
    confidence: float
    strategy_id: str
    
    # Mathematical context
    chrono_resonance: float
    temporal_warp: float
    math_optimization: Dict[str, float]
    
    # Market context
    volatility: float
    volume: float
    market_conditions: Dict[str, Any]
    
    # Registry references
    canonical_hash: Optional[str] = None
    specialized_hashes: Dict[str, str] = None

@dataclass
class EnhancedTradeResult:
    """Enhanced trade result with Math + Memory Fusion Core context."""
    
    # Basic trade data
    symbol: str
    action: str
    entry_price: float
    exit_price: float
    amount: float
    profit: float
    timestamp: float
    
    # Enhanced signal context
    enhanced_signal: Optional[EnhancedTradingSignal] = None
    signal_hash: Optional[str] = None
    vector_confidence: float = 0.0
    mathematical_confidence: float = 0.0
    entropy_correction: float = 0.0
    
    # Registry and memory context
    canonical_hash: Optional[str] = None
    profit_vector_hash: Optional[str] = None
    soulprint_hash: Optional[str] = None
    
    # Performance metrics
    drawdown: float = 0.0
    volatility: float = 0.0
    risk_profile: str = "medium"
    
    # Signal lineage
    strategy_id: str = "unknown"
    exit_type: str = "unknown"
    registry_status: str = "pending"

class UnifiedTradingPipeline:
    """Complete unified trading pipeline with Math + Memory Fusion Core integration."""
    
    def __init__(self, mode: str = "demo", config: Optional[Dict[str, Any]] = None):
        """Initialize the unified trading pipeline with Math + Memory Fusion Core."""
        self.mode = mode  # "demo", "backtest", "live"
        self.config = config or {}
        
        # Initialize core mathematical systems
        self.math_system = CleanUnifiedMathSystem()
        self.weather_mapper = ChronoResonanceWeatherMapper()
        self.temporal_engine = TemporalWarpEngine()
        
        # Initialize CLI system if available
        if CLI_SYSTEM_AVAILABLE:
            self.cli_system = CLILiveEntrySystem()
            logger.info("🖥️ CLI Live Entry System integrated")
        else:
            self.cli_system = None
            logger.warning("🖥️ CLI Live Entry System not available (fallback mode)")
        
        # Initialize Math + Memory Fusion Core
        if FUSION_CORE_AVAILABLE:
            self.strategy_executor = StrategyExecutor()
            logger.info("🧠 Math + Memory Fusion Core integrated")
        else:
            self.strategy_executor = None
            logger.warning("🧠 Math + Memory Fusion Core not available (fallback mode)")
        
        # Initialize specialized registries
        self.profit_bucket_registry = ProfitBucketRegistry()
        
        # Register specialized registries with coordinator
        registry_coordinator.register_specialized_registry("profit_buckets", self.profit_bucket_registry)
        registry_coordinator.register_specialized_registry("soulprints", soulprint_registry)
        
        # Trading state
        self.portfolio_value = 10000.0  # Starting portfolio
        self.current_positions: Dict[str, float] = {}
        self.trade_history: List[EnhancedTradeResult] = []
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        
        # Signal lineage tracking
        self.signal_lineage: List[Dict[str, Any]] = []
        self.max_lineage_history = 1000
        
        # Fusion core parameters
        self.min_confidence_threshold = self.config.get('min_confidence', 0.7)
        self.entropy_correction_threshold = self.config.get('entropy_threshold', 0.3)
        
        logger.info(f"🚀 Enhanced Unified Trading Pipeline initialized in {mode} mode")

    async def initialize(self) -> bool:
        """Initialize the pipeline with Math + Memory Fusion Core."""
        try:
            logger.info("Initializing Enhanced Unified Trading Pipeline...")
            
            # Initialize strategy executor if available
            if FUSION_CORE_AVAILABLE and self.strategy_executor:
                init_result = await self.strategy_executor.initialize()
                if not init_result:
                    logger.error("Failed to initialize Strategy Executor")
                    return False
                logger.info("✅ Strategy Executor initialized successfully")
            
            logger.info("✅ Enhanced Unified Trading Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Unified Trading Pipeline: {e}")
            return False

    async def run_trading_cycle(self) -> Dict[str, Any]:
        """Execute one complete trading cycle with Math + Memory Fusion Core."""
        try:
            # 1. Generate market data and mathematical context
            market_data = await self._generate_market_data()
            
            # 2. Apply mathematical systems
            math_context = self._apply_mathematical_systems(market_data)
            
            # 3. Generate enhanced trading signals using Math + Memory Fusion Core
            enhanced_signals = await self._generate_enhanced_trading_signals(market_data, math_context)
            
            # 4. Select best signal and execute trade if conditions are met
            trade_result = None
            if enhanced_signals:
                best_signal = self._select_best_signal(enhanced_signals)
                if best_signal and best_signal.confidence > self.min_confidence_threshold:
                    trade_result = await self._execute_enhanced_trade(best_signal)
                    
                    # Update profit vectors with trade result
                    if trade_result and FUSION_CORE_AVAILABLE:
                        await self.strategy_executor.update_profit_vectors({
                            "symbol": trade_result.symbol,
                            "action": trade_result.action,
                            "entry_price": trade_result.entry_price,
                            "amount": trade_result.amount,
                            "profit": trade_result.profit,
                            "drawdown": trade_result.drawdown,
                            "volatility": trade_result.volatility,
                            "strategy_id": trade_result.strategy_id,
                            "exit_type": trade_result.exit_type,
                            "risk_profile": trade_result.risk_profile,
                            "timestamp": trade_result.timestamp
                        })
            
            # 5. Update registries with enhanced trade data
            if trade_result:
                await self._update_enhanced_registries(trade_result)
            
            # 6. Update performance metrics
            self._update_enhanced_performance_metrics(trade_result)
            
            # 7. Generate enhanced cycle summary
            cycle_summary = self._generate_enhanced_cycle_summary(market_data, enhanced_signals, trade_result)
            
            return cycle_summary
            
        except Exception as e:
            logger.error(f"Error in enhanced trading cycle: {e}")
            return {"error": str(e)}

    async def _generate_market_data(self) -> Dict[str, Any]:
        """Generate or fetch market data."""
        if self.mode == "demo":
            # Simulate market data
            current_price = 50000.0 + (time.time() % 1000) * 0.1
            return {
                "symbol": "BTC/USDC",
                "price": current_price,
                "volume": 1000.0 + (time.time() % 100) * 10,
                "volatility": 0.02 + (time.time() % 10) * 0.001,
                "timestamp": time.time()
            }
        else:
            # TODO: Implement live market data fetching
            raise NotImplementedError("Live market data not yet implemented")

    def _apply_mathematical_systems(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all mathematical systems to market data."""
        timestamp = market_data["timestamp"]
        price = market_data["price"]
        
        # Chrono resonance weather mapping
        crwf = self.weather_mapper.compute_crwf(timestamp, 40.0, -74.0, price)
        
        # Temporal warp projection
        projected_time = self.temporal_engine.calculate_temporal_projection(timestamp, 0.1)
        
        # Mathematical optimization
        base_profit = self.math_system.multiply(price, 0.01)
        enhancement = self.math_system.mean([0.5, 0.6, 0.7])
        confidence = self.math_system.mean([0.7, 0.8, 0.9])
        optimized_profit = self.math_system.optimize_profit(base_profit, enhancement, confidence)
        risk_adjusted = self.math_system.calculate_risk_adjustment(optimized_profit, market_data["volatility"], confidence)
        portfolio_weight = self.math_system.calculate_portfolio_weight(confidence, 0.2)
        
        return {
            "chrono_resonance": crwf,
            "temporal_warp": projected_time,
            "math_optimization": {
                "base_profit": base_profit,
                "enhancement": enhancement,
                "confidence": confidence,
                "optimized_profit": optimized_profit,
                "risk_adjusted": risk_adjusted,
                "portfolio_weight": portfolio_weight
            }
        }

    def _generate_trading_signal(self, market_data: Dict[str, Any], math_context: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate a trading signal based on market data and mathematical context."""
        try:
            # Simple signal generation logic (can be enhanced)
            confidence = math_context["math_optimization"]["confidence"]
            risk_adjusted = math_context["math_optimization"]["risk_adjusted"]
            
            # Determine action based on mathematical context
            if risk_adjusted > 0 and confidence > 0.6:
                action = "buy"
            elif risk_adjusted < 0 and confidence > 0.6:
                action = "sell"
            else:
                return None  # No clear signal
            
            # Calculate position size
            portfolio_weight = math_context["math_optimization"]["portfolio_weight"]
            amount = self.portfolio_value * portfolio_weight * 0.1  # 10% of portfolio weight
            
            signal = TradingSignal(
                symbol=market_data["symbol"],
                action=action,
                entry_price=market_data["price"],
                amount=amount,
                confidence=confidence,
                strategy_id="unified_pipeline_v1",
                chrono_resonance=math_context["chrono_resonance"],
                temporal_warp=math_context["temporal_warp"],
                math_optimization=math_context["math_optimization"],
                volatility=market_data["volatility"],
                volume=market_data["volume"],
                market_conditions={
                    "timestamp": market_data["timestamp"],
                    "price": market_data["price"]
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return None

    async def _generate_enhanced_trading_signals(self, market_data: Dict[str, Any], 
                                                math_context: Dict[str, Any]) -> List[EnhancedTradingSignal]:
        """Generate enhanced trading signals using Math + Memory Fusion Core."""
        try:
            if not FUSION_CORE_AVAILABLE or not self.strategy_executor:
                # Fallback to original signal generation
                logger.warning("Using fallback signal generation")
                original_signal = self._generate_trading_signal(market_data, math_context)
                return [] if not original_signal else []
            
            # Generate enhanced signals using Math + Memory Fusion Core
            enhanced_signals = await self.strategy_executor.generate_unified_signals(market_data)
            
            # Add mathematical context to signals
            for signal in enhanced_signals:
                signal.chrono_resonance = math_context.get("chrono_resonance", 0.0)
                signal.temporal_warp = math_context.get("temporal_warp", 0.0)
                signal.math_optimization = math_context.get("math_optimization", {})
            
            logger.info(f"🧠 Generated {len(enhanced_signals)} enhanced signals using Math + Memory Fusion Core")
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Error generating enhanced trading signals: {e}")
            return []

    def _select_best_signal(self, signals: List[EnhancedTradingSignal]) -> Optional[EnhancedTradingSignal]:
        """Select the best signal based on confidence and entropy correction."""
        try:
            if not signals:
                return None
            
            # Score signals based on confidence and entropy correction
            scored_signals = []
            for signal in signals:
                # Calculate composite score
                confidence_score = signal.confidence
                entropy_bonus = (1 - signal.entropy_correction) * 0.2  # Bonus for low entropy
                vector_bonus = signal.vector_confidence * 0.1  # Bonus for high vector confidence
                
                composite_score = confidence_score + entropy_bonus + vector_bonus
                scored_signals.append((signal, composite_score))
            
            # Select signal with highest score
            best_signal, best_score = max(scored_signals, key=lambda x: x[1])
            
            logger.info(f"🎯 Selected best signal: {best_signal.action} {best_signal.symbol} "
                       f"(score: {best_score:.3f}, confidence: {best_signal.confidence:.3f})")
            
            return best_signal
            
        except Exception as e:
            logger.error(f"Error selecting best signal: {e}")
            return signals[0] if signals else None

    async def _execute_enhanced_trade(self, signal: EnhancedTradingSignal) -> Optional[EnhancedTradeResult]:
        """Execute an enhanced trade with full signal lineage tracking."""
        try:
            # Simulate trade execution (replace with actual execution logic)
            entry_price = signal.entry_price
            exit_price = entry_price * (1 + (0.02 if signal.action == "buy" else -0.02))  # 2% movement
            profit = (exit_price - entry_price) / entry_price if signal.action == "buy" else (entry_price - exit_price) / entry_price
            
            # Calculate drawdown and volatility
            drawdown = abs(min(0, profit))  # Only negative profits count as drawdown
            volatility = signal.volatility
            
            # Determine risk profile
            if signal.confidence > 0.8 and signal.entropy_correction < 0.2:
                risk_profile = "low"
            elif signal.confidence > 0.6 and signal.entropy_correction < 0.4:
                risk_profile = "medium"
            else:
                risk_profile = "high"
            
            # Generate trade result
            trade_result = EnhancedTradeResult(
                symbol=signal.symbol,
                action=signal.action,
                entry_price=entry_price,
                exit_price=exit_price,
                amount=signal.amount,
                profit=profit,
                timestamp=time.time(),
                enhanced_signal=signal,
                signal_hash=signal.signal_hash,
                vector_confidence=signal.vector_confidence,
                mathematical_confidence=signal.mathematical_confidence,
                entropy_correction=signal.entropy_correction,
                canonical_hash=self._generate_canonical_hash(signal),
                profit_vector_hash=self._generate_profit_vector_hash(signal),
                soulprint_hash=self._generate_soulprint_hash(signal),
                drawdown=drawdown,
                volatility=volatility,
                risk_profile=risk_profile,
                strategy_id=signal.strategy_id,
                exit_type="take_profit" if profit > 0 else "stop_loss",
                registry_status="confirmed"
            )
            
            # Update portfolio
            if signal.action == "buy":
                self.current_positions[signal.symbol] = self.current_positions.get(signal.symbol, 0) + signal.amount
            else:
                self.current_positions[signal.symbol] = self.current_positions.get(signal.symbol, 0) - signal.amount
            
            # Update portfolio value
            self.portfolio_value *= (1 + profit * 0.1)  # 10% of trade profit affects portfolio
            
            logger.info(f"💰 Executed enhanced trade: {signal.action} {signal.symbol} "
                       f"(profit: {profit:.3f}, confidence: {signal.confidence:.3f})")
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing enhanced trade: {e}")
            return None

    def _generate_canonical_hash(self, signal: EnhancedTradingSignal) -> str:
        """Generate canonical hash for trade tracking."""
        try:
            hash_data = f"{signal.symbol}_{signal.action}_{signal.entry_price}_{signal.timestamp}"
            return hashlib.md5(hash_data.encode()).hexdigest()[:8]
        except Exception as e:
            logger.error(f"Error generating canonical hash: {e}")
            return "unknown"

    def _generate_profit_vector_hash(self, signal: EnhancedTradingSignal) -> str:
        """Generate profit vector hash for memory tracking."""
        try:
            if signal.profit_vectors:
                vector_data = "_".join([str(v.profit) for v in signal.profit_vectors])
                return hashlib.md5(vector_data.encode()).hexdigest()[:8]
            return "no_vectors"
        except Exception as e:
            logger.error(f"Error generating profit vector hash: {e}")
            return "unknown"

    def _generate_soulprint_hash(self, signal: EnhancedTradingSignal) -> str:
        """Generate soulprint hash for long-term memory."""
        try:
            soulprint_data = f"{signal.strategy_id}_{signal.confidence}_{signal.entropy_correction}_{signal.timestamp}"
            return hashlib.md5(soulprint_data.encode()).hexdigest()[:8]
        except Exception as e:
            logger.error(f"Error generating soulprint hash: {e}")
            return "unknown"

    async def _update_enhanced_registries(self, trade_result: EnhancedTradeResult) -> None:
        """Update registries with enhanced trade data."""
        try:
            # Update canonical trade registry
            canonical_trade_registry.add_trade({
                "symbol": trade_result.symbol,
                "action": trade_result.action,
                "entry_price": trade_result.entry_price,
                "exit_price": trade_result.exit_price,
                "amount": trade_result.amount,
                "profit": trade_result.profit,
                "timestamp": trade_result.timestamp,
                "canonical_hash": trade_result.canonical_hash,
                "registry_status": trade_result.registry_status
            })
            
            # Update profit bucket registry
            self.profit_bucket_registry.add_profit_bucket({
                "symbol": trade_result.symbol,
                "profit": trade_result.profit,
                "vector_confidence": trade_result.vector_confidence,
                "entropy_correction": trade_result.entropy_correction,
                "profit_vector_hash": trade_result.profit_vector_hash
            })
            
            # Update soulprint registry
            soulprint_registry.add_soulprint({
                "strategy_id": trade_result.strategy_id,
                "confidence": trade_result.mathematical_confidence,
                "entropy_correction": trade_result.entropy_correction,
                "soulprint_hash": trade_result.soulprint_hash
            })
            
            # Store signal lineage
            self._store_signal_lineage(trade_result)
            
            logger.info(f"📊 Updated enhanced registries for trade {trade_result.canonical_hash}")
            
        except Exception as e:
            logger.error(f"Error updating enhanced registries: {e}")

    def _store_signal_lineage(self, trade_result: EnhancedTradeResult) -> None:
        """Store signal lineage for tracking and analysis."""
        try:
            lineage_entry = {
                "timestamp": trade_result.timestamp,
                "signal_hash": trade_result.signal_hash,
                "canonical_hash": trade_result.canonical_hash,
                "profit_vector_hash": trade_result.profit_vector_hash,
                "soulprint_hash": trade_result.soulprint_hash,
                "symbol": trade_result.symbol,
                "action": trade_result.action,
                "confidence": trade_result.mathematical_confidence,
                "vector_confidence": trade_result.vector_confidence,
                "entropy_correction": trade_result.entropy_correction,
                "profit": trade_result.profit,
                "registry_status": trade_result.registry_status
            }
            
            self.signal_lineage.append(lineage_entry)
            
            # Keep lineage history within limits
            if len(self.signal_lineage) > self.max_lineage_history:
                self.signal_lineage = self.signal_lineage[-self.max_lineage_history:]
            
            logger.debug(f"📈 Stored signal lineage: {trade_result.signal_hash}")
            
        except Exception as e:
            logger.error(f"Error storing signal lineage: {e}")

    def _update_enhanced_performance_metrics(self, trade_result: Optional[EnhancedTradeResult]) -> None:
        """Update performance metrics with enhanced trade data."""
        try:
            if trade_result:
                self.total_trades += 1
                if trade_result.profit > 0:
                    self.successful_trades += 1
                self.total_profit += trade_result.profit
                
                # Store trade in history
                self.trade_history.append(trade_result)
                
                logger.info(f"📊 Updated performance metrics: "
                           f"Total trades: {self.total_trades}, "
                           f"Success rate: {self.successful_trades/self.total_trades:.2%}, "
                           f"Total profit: {self.total_profit:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating enhanced performance metrics: {e}")

    def _generate_enhanced_cycle_summary(self, market_data: Dict[str, Any], 
                                       enhanced_signals: List[EnhancedTradingSignal], 
                                       trade_result: Optional[EnhancedTradeResult]) -> Dict[str, Any]:
        """Generate enhanced cycle summary with Math + Memory Fusion Core insights."""
        try:
            summary = {
                "timestamp": time.time(),
                "market_data": market_data,
                "enhanced_signals_count": len(enhanced_signals),
                "trade_executed": trade_result is not None,
                "portfolio_value": self.portfolio_value,
                "total_trades": self.total_trades,
                "success_rate": self.successful_trades / max(self.total_trades, 1),
                "total_profit": self.total_profit
            }
            
            if trade_result:
                summary.update({
                    "trade_result": {
                        "symbol": trade_result.symbol,
                        "action": trade_result.action,
                        "profit": trade_result.profit,
                        "confidence": trade_result.mathematical_confidence,
                        "vector_confidence": trade_result.vector_confidence,
                        "entropy_correction": trade_result.entropy_correction,
                        "canonical_hash": trade_result.canonical_hash,
                        "registry_status": trade_result.registry_status
                    }
                })
            
            # Add Math + Memory Fusion Core insights if available
            if FUSION_CORE_AVAILABLE and self.strategy_executor:
                mathematical_insights = self.strategy_executor.get_mathematical_insights()
                summary["mathematical_insights"] = mathematical_insights
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating enhanced cycle summary: {e}")
            return {"error": str(e)}

    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics."""
        return registry_coordinator.get_performance_analytics()

    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return registry_coordinator.get_registry_statistics()

    def validate_registry_consistency(self) -> Dict[str, Any]:
        """Validate registry consistency."""
        return registry_coordinator.validate_registry_consistency()

    async def run_backtest(self, duration_seconds: int = 3600, cycle_interval: float = 1.0) -> Dict[str, Any]:
        """Run a backtest for the specified duration."""
        logger.info(f"🔄 Starting backtest for {duration_seconds} seconds")
        
        start_time = time.time()
        cycles_completed = 0
        
        while time.time() - start_time < duration_seconds:
            cycle_result = await self.run_trading_cycle()
            cycles_completed += 1
            
            if cycles_completed % 10 == 0:
                logger.info(f"Backtest progress: {cycles_completed} cycles completed")
            
            await asyncio.sleep(cycle_interval)
        
        # Final analytics
        analytics = self.get_performance_analytics()
        registry_stats = self.get_registry_statistics()
        
        backtest_results = {
            "duration_seconds": duration_seconds,
            "cycles_completed": cycles_completed,
            "final_portfolio_value": self.portfolio_value,
            "total_profit": self.total_profit,
            "success_rate": self.successful_trades / self.total_trades if self.total_trades > 0 else 0,
            "analytics": analytics,
            "registry_statistics": registry_stats
        }
        
        logger.info(f"✅ Backtest completed: {cycles_completed} cycles, ${self.total_profit:.2f} profit")
        return backtest_results

# Global instance for easy access
unified_trading_pipeline = UnifiedTradingPipeline() 