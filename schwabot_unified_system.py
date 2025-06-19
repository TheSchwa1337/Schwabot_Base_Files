"""
Schwabot Unified System
=======================

This demonstrates how the Enhanced Fitness Oracle integrates with a simplified
FerrisWheelScheduler to create a complete profit-seeking navigation system.

The architecture hierarchy is:
Market Data -> Enhanced Fitness Oracle -> Simplified Scheduler -> Trade Execution
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

from enhanced_fitness_oracle import EnhancedFitnessOracle, UnifiedFitnessScore, MarketSnapshot
from schwabot_integration import SchwabotOrchestrator
# Specific imports from enhanced fitness config
from config.enhanced_fitness_config import (
    UnifiedMathematicalProcessor,
    AnalysisResult,
    EnhancedFitnessOracle,
    MathematicalCore,
    FitnessMetrics,
    TradingParameters,
    OptimizationConfig
)

logger = logging.getLogger(__name__)

@dataclass
class TradeDecision:
    """Simplified trade decision structure"""
    timestamp: datetime
    action: str  # "BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"
    symbol: str
    position_size: float
    confidence: float
    reasoning: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_time: Optional[timedelta] = None

class SimplifiedFerrisWheelScheduler:
    """
    Simplified scheduler that uses the Enhanced Fitness Oracle
    for ALL decision making - no more complex signal aggregation
    """
    
    def __init__(self, fitness_oracle: EnhancedFitnessOracle, 
                 symbols: List[str] = ["BTC/USDT", "ETH/USDT"]):
        self.fitness_oracle = fitness_oracle
        self.symbols = symbols
        self.tick_count = 0
        self.active_positions = {}
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'fitness_accuracy': 0.0
        }
        
        logger.info(f"Simplified FerrisWheelScheduler initialized for {symbols}")

    async def tick_loop(self, market_data_provider, max_ticks: Optional[int] = None):
        """
        Simplified tick loop that delegates ALL analysis to the Fitness Oracle
        This is much cleaner than the previous complex aggregation logic
        """
        logger.info("Starting simplified tick loop with Fitness Oracle integration")
        
        while max_ticks is None or self.tick_count < max_ticks:
            try:
                # === STEP 1: GET MARKET DATA ===
                market_data = await self._get_market_data(market_data_provider)
                
                # === STEP 2: LET FITNESS ORACLE DO ALL THE ANALYSIS ===
                # This is the key simplification - one call does everything
                fitness_score = await self._analyze_with_fitness_oracle(market_data)
                
                # === STEP 3: MAKE TRADING DECISION BASED ON FITNESS ===
                trade_decision = self._make_trading_decision(fitness_score)
                
                # === STEP 4: EXECUTE TRADE IF NEEDED ===
                if trade_decision.action != "HOLD":
                    await self._execute_trade_decision(trade_decision)
                
                # === STEP 5: LOG AND UPDATE ===
                self._log_tick_summary(fitness_score, trade_decision)
                self._update_performance_metrics(fitness_score, trade_decision)
                
                self.tick_count += 1
                await asyncio.sleep(1)  # 1-second ticks
                
            except Exception as e:
                logger.error(f"Tick {self.tick_count} failed: {e}")
                await asyncio.sleep(1)

    async def _get_market_data(self, provider) -> Dict[str, Any]:
        """Get market data from provider (simplified)"""
        # In real implementation, this would fetch from exchange
        # For demo, we'll simulate realistic market data
        base_price = 100.0
        price_series = [base_price + np.random.normal(0, 2) for _ in range(20)]
        volume_series = [1000 + np.random.normal(0, 200) for _ in range(20)]
        
        return {
            "symbol": self.symbols[0],  # For simplicity, focus on first symbol
            "price": price_series[-1],
            "volume": volume_series[-1],
            "price_series": price_series,
            "volume_series": volume_series,
            "timestamp": datetime.now()
        }

    async def _analyze_with_fitness_oracle(self, market_data: Dict[str, Any]) -> UnifiedFitnessScore:
        """
        THIS IS THE KEY INTEGRATION POINT
        All complex analysis is delegated to the Fitness Oracle
        """
        # Capture market snapshot (this runs all the engines)
        market_snapshot = await self.fitness_oracle.capture_market_snapshot(market_data)
        
        # Calculate unified fitness (this makes the decision)
        fitness_score = self.fitness_oracle.calculate_unified_fitness(market_snapshot)
        
        return fitness_score

    def _make_trading_decision(self, fitness_score: UnifiedFitnessScore) -> TradeDecision:
        """
        Convert fitness score to actionable trade decision
        This is much simpler than before - the Oracle did all the work
        """
        # The fitness score already contains the recommended action
        action = fitness_score.action
        position_size = fitness_score.position_size
        confidence = fitness_score.confidence
        
        # Generate reasoning based on dominant factors
        reasoning_parts = []
        if fitness_score.dominant_factors:
            top_factor = max(fitness_score.dominant_factors.items(), key=lambda x: abs(x[1]))
            reasoning_parts.append(f"Dominant factor: {top_factor[0]} ({top_factor[1]:.3f})")
        
        if fitness_score.profit_tier_detected:
            reasoning_parts.append("PROFIT TIER DETECTED")
        
        if fitness_score.loop_warning:
            reasoning_parts.append("Loop warning - reduced confidence")
            confidence *= 0.5  # Reduce confidence on loop warning
        
        reasoning = f"Regime: {fitness_score.market_regime}, " + ", ".join(reasoning_parts)
        
        return TradeDecision(
            timestamp=fitness_score.timestamp,
            action=action,
            symbol=self.symbols[0],  # Simplified to first symbol
            position_size=position_size,
            confidence=confidence,
            reasoning=reasoning,
            stop_loss=fitness_score.stop_loss,
            take_profit=fitness_score.take_profit,
            max_hold_time=fitness_score.max_hold_time
        )

    async def _execute_trade_decision(self, decision: TradeDecision):
        """Execute the trade decision (simplified)"""
        # In real implementation, this would call exchange API
        
        if decision.action in ["BUY", "STRONG_BUY"]:
            # Enter or increase position
            current_position = self.active_positions.get(decision.symbol, 0.0)
            new_position = current_position + decision.position_size
            self.active_positions[decision.symbol] = new_position
            
            logger.info(
                f"🟢 {decision.action}: {decision.symbol} | "
                f"Size: {decision.position_size:.2f} | "
                f"Total Position: {new_position:.2f} | "
                f"Confidence: {decision.confidence:.3f}"
            )
            
        elif decision.action in ["SELL", "STRONG_SELL"]:
            # Reduce or exit position
            current_position = self.active_positions.get(decision.symbol, 0.0)
            reduction = min(decision.position_size, current_position)
            new_position = current_position - reduction
            
            if new_position <= 0:
                self.active_positions.pop(decision.symbol, None)
            else:
                self.active_positions[decision.symbol] = new_position
            
            logger.info(
                f"🔴 {decision.action}: {decision.symbol} | "
                f"Reduction: {reduction:.2f} | "
                f"Remaining Position: {new_position:.2f} | "
                f"Confidence: {decision.confidence:.3f}"
            )
        
        # Store trade in history
        self.trade_history.append(decision)
        self.performance_metrics['total_trades'] += 1

    def _log_tick_summary(self, fitness_score: UnifiedFitnessScore, decision: TradeDecision):
        """Log summary of tick analysis and decision"""
        logger.info(
            f"Tick {self.tick_count} | "
            f"Fitness: {fitness_score.overall_fitness:.3f} | "
            f"Action: {decision.action} | "
            f"Regime: {fitness_score.market_regime} | "
            f"Confidence: {decision.confidence:.3f}"
        )
        
        # Log special conditions
        if fitness_score.profit_tier_detected:
            logger.warning("🎯 PROFIT TIER DETECTED - High opportunity!")
        
        if fitness_score.loop_warning:
            logger.warning("⚠️ RECURSIVE LOOP WARNING - Pattern repetition detected!")

    def _update_performance_metrics(self, fitness_score: UnifiedFitnessScore, decision: TradeDecision):
        """Update performance tracking"""
        # Simple performance tracking
        # In real implementation, this would be much more sophisticated
        
        if decision.action != "HOLD":
            # Track fitness accuracy (simplified)
            if fitness_score.overall_fitness > 0 and decision.action in ["BUY", "STRONG_BUY"]:
                self.performance_metrics['fitness_accuracy'] += 1
            elif fitness_score.overall_fitness < 0 and decision.action in ["SELL", "STRONG_SELL"]:
                self.performance_metrics['fitness_accuracy'] += 1
        
        # Calculate accuracy percentage
        if self.performance_metrics['total_trades'] > 0:
            accuracy = self.performance_metrics['fitness_accuracy'] / self.performance_metrics['total_trades']
            self.performance_metrics['fitness_accuracy_pct'] = accuracy

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'tick_count': self.tick_count,
            'active_positions': self.active_positions.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'recent_trades': self.trade_history[-10:] if self.trade_history else [],
            'total_position_value': sum(self.active_positions.values())
        }

class UnifiedSchwabotSystem:
    """
    Complete unified system that combines:
    - Enhanced Fitness Oracle (for analysis)
    - Simplified FerrisWheelScheduler (for execution)
    - Performance tracking and monitoring
    """
    
    def __init__(self, config_path: str = "config/enhanced_fitness_config.yaml"):
        # Initialize the Enhanced Fitness Oracle
        self.fitness_oracle = EnhancedFitnessOracle(config_path)
        
        # Initialize the Simplified Scheduler
        self.scheduler = SimplifiedFerrisWheelScheduler(self.fitness_oracle)
        
        # System monitoring
        self.system_start_time = datetime.now()
        self.system_metrics = {
            'uptime': timedelta(0),
            'total_decisions': 0,
            'oracle_calls': 0,
            'errors': 0
        }
        
        logger.info("Unified Schwabot System initialized")

    async def run_system(self, duration_minutes: int = 60):
        """Run the complete system for specified duration"""
        logger.info(f"Starting Unified Schwabot System for {duration_minutes} minutes")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Mock market data provider
        async def market_data_provider() -> Any:
            while datetime.now() < end_time:
                yield {
                    "timestamp": datetime.now(),
                    "price": 100.0 + np.random.normal(0, 2),
                    "volume": 1000 + np.random.normal(0, 200)
                }
                await asyncio.sleep(1)
        
        try:
            # Run the scheduler with time limit
            max_ticks = duration_minutes * 60  # 60 ticks per minute
            await self.scheduler.tick_loop(market_data_provider(), max_ticks)
            
        except Exception as e:
            logger.error(f"System error: {e}")
            self.system_metrics['errors'] += 1
        
        finally:
            # Update system metrics
            self.system_metrics['uptime'] = datetime.now() - self.system_start_time
            self.system_metrics['total_decisions'] = self.scheduler.tick_count
            
            # Log final summary
            self._log_system_summary()

    def _log_system_summary(self):
        """Log comprehensive system summary"""
        scheduler_performance = self.scheduler.get_performance_summary()
        
        logger.info("=" * 50)
        logger.info("UNIFIED SCHWABOT SYSTEM SUMMARY")
        logger.info("=" * 50)
        logger.info(f"System Uptime: {self.system_metrics['uptime']}")
        logger.info(f"Total Ticks: {scheduler_performance['tick_count']}")
        logger.info(f"Total Trades: {scheduler_performance['performance_metrics']['total_trades']}")
        logger.info(f"Active Positions: {scheduler_performance['active_positions']}")
        logger.info(f"Total Position Value: {scheduler_performance['total_position_value']:.2f}")
        
        if 'fitness_accuracy_pct' in scheduler_performance['performance_metrics']:
            accuracy = scheduler_performance['performance_metrics']['fitness_accuracy_pct']
            logger.info(f"Fitness Accuracy: {accuracy:.1%}")
        
        logger.info("=" * 50)

    def export_comprehensive_report(self, filename: str = None) -> Dict[str, Any]:
        """Export comprehensive system report"""
        if filename is None:
            filename = f"schwabot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'system_info': {
                'start_time': self.system_start_time.isoformat(),
                'uptime_seconds': self.system_metrics['uptime'].total_seconds(),
                'version': "Unified System v1.0"
            },
            'fitness_oracle': {
                'config': self.fitness_oracle.config,
                'current_regime': self.fitness_oracle.current_regime,
                'market_history_length': len(self.fitness_oracle.market_history),
                'fitness_history_length': len(self.fitness_oracle.fitness_history)
            },
            'scheduler_performance': self.scheduler.get_performance_summary(),
            'system_metrics': self.system_metrics
        }
        
        # Export to file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report exported to {filename}")
        return report

# Example usage and demonstration
async def demo_unified_system() -> Any:
    """Demonstrate the unified system in action"""
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize the unified system
    system = UnifiedSchwabotSystem()
    
    try:
        # Run for 5 minutes as demo
        await system.run_system(duration_minutes=5)
        
        # Export comprehensive report
        report = system.export_comprehensive_report()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Check the exported report for detailed analysis")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")

# Alternative integration with existing Schwabot Orchestrator
class IntegratedSchwabotOrchestrator(SchwabotOrchestrator):
    """
    Enhanced version of SchwabotOrchestrator that uses the Enhanced Fitness Oracle
    for decision making instead of complex internal logic
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add the Enhanced Fitness Oracle
        self.fitness_oracle = EnhancedFitnessOracle()
        
        logger.info("Integrated Schwabot Orchestrator with Enhanced Fitness Oracle")

    async def _process_trading_pair(self, symbol: str, market_data: Dict, timestamp: datetime):
        """
        Override the original complex processing with Fitness Oracle
        """
        # Use Fitness Oracle for all analysis
        market_snapshot = await self.fitness_oracle.capture_market_snapshot(market_data)
        fitness_score = self.fitness_oracle.calculate_unified_fitness(market_snapshot)
        
        # Convert fitness score to trade signal format expected by parent class
        if fitness_score.action in ["BUY", "STRONG_BUY"]:
            trade_signal = {
                'action': 'ENTER',
                'direction': 'LONG',
                'confidence': fitness_score.confidence,
                'volume_weight': fitness_score.position_size,
                'stop_loss': fitness_score.stop_loss,
                'take_profit': fitness_score.take_profit,
                'sha_signature': f"fitness_{fitness_score.timestamp.timestamp()}"
            }
            await self._execute_trade_signal(symbol, trade_signal, market_data)
            
        elif fitness_score.action in ["SELL", "STRONG_SELL"]:
            trade_signal = {
                'action': 'EXIT',
                'reason': f'Fitness Oracle recommendation: {fitness_score.action}'
            }
            await self._execute_trade_signal(symbol, trade_signal, market_data)
        
        # Log fitness-based decision
        self.logger.info(
            f"Fitness Oracle Decision for {symbol}: "
            f"Action={fitness_score.action}, "
            f"Fitness={fitness_score.overall_fitness:.3f}, "
            f"Confidence={fitness_score.confidence:.3f}"
        )

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_unified_system()) 