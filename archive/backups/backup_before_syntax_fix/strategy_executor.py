#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Executor - Execute trading strategies
=============================================

This module handles the execution of trading strategies, signal generation,
and coordination between different strategies.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..type_defs import TradingStrategy

logger = logging.getLogger(__name__)


class StrategyExecutor:
    """
    Execute trading strategies and generate signals.
    
    This class coordinates the execution of multiple trading strategies,
    combines their signals, and provides a unified interface for signal generation.
    """
    
    def __init__(self):
        """Initialize the strategy executor."""
        self.active_strategies: Dict[str, TradingStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.is_running = False
        self.is_initialized = False
        self.execution_task: Optional[asyncio.Task] = None
        
        # Signal tracking
        self.signal_history: List[Dict[str, Any]] = []
        self.max_signal_history = 1000
        
        logger.info("Strategy Executor initialized")
    
    async def initialize(self) -> bool:
        """Initialize the strategy executor."""
        try:
            logger.info("Initializing Strategy Executor...")
            
            # Set default strategy weights
            self.strategy_weights = {
                "ExampleStrategy": 1.0,
                "VolumeWeightedHashOscillator": 0.8,
                "MultiPhaseStrategyWeightTensor": 0.9,
                "ZygotZalgoEntropyDualKeyGate": 0.7
            }
            
            self.is_initialized = True
            logger.info("Strategy Executor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Strategy Executor: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the strategy executor."""
        if not self.is_initialized:
            logger.error("Strategy Executor not initialized")
            return False
        
        try:
            logger.info("Starting Strategy Executor...")
            
            self.is_running = True
            
            # Start execution task
            self.execution_task = asyncio.create_task(self._execution_loop())
            
            logger.info("Strategy Executor started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Strategy Executor: {e}")
            return False
    
    async def stop(self):
        """Stop the strategy executor."""
        if not self.is_running:
            return
        
        logger.info("Stopping Strategy Executor...")
        
        try:
            self.is_running = False
            
            # Cancel execution task
            if self.execution_task:
                self.execution_task.cancel()
                try:
                    await self.execution_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Strategy Executor stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Strategy Executor: {e}")
    
    async def _execution_loop(self):
        """Main execution loop for strategies."""
        logger.info("Strategy execution loop started")
        
        try:
            while self.is_running:
                # Process active strategies
                await self._process_strategies()
                
                # Sleep for next iteration
                await asyncio.sleep(1)  # 1 second interval
                
        except asyncio.CancelledError:
            logger.info("Strategy execution loop cancelled")
        except Exception as e:
            logger.error(f"Error in strategy execution loop: {e}")
    
    async def _process_strategies(self):
        """Process all active strategies."""
        for strategy_name, strategy in self.active_strategies.items():
            try:
                # Check if strategy is still valid
                if not hasattr(strategy, 'is_initialized') or not strategy.is_initialized:
                    logger.warning(f"Strategy {strategy_name} not initialized, skipping")
                    continue
                
                # Process strategy (this would typically involve getting market data)
                # For now, we'll just log that we're processing
                logger.debug(f"Processing strategy: {strategy_name}")
                
            except Exception as e:
                logger.error(f"Error processing strategy {strategy_name}: {e}")
    
    def add_strategy(self, strategy_name: str, strategy: TradingStrategy, weight: float = 1.0) -> bool:
        """Add a strategy to the executor."""
        try:
            if not hasattr(strategy, 'is_initialized') or not strategy.is_initialized:
                logger.error(f"Strategy {strategy_name} is not initialized")
                return False
            
            self.active_strategies[strategy_name] = strategy
            self.strategy_weights[strategy_name] = weight
            
            logger.info(f"Added strategy: {strategy_name} with weight {weight}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add strategy {strategy_name}: {e}")
            return False
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy from the executor."""
        try:
            if strategy_name in self.active_strategies:
                del self.active_strategies[strategy_name]
                if strategy_name in self.strategy_weights:
                    del self.strategy_weights[strategy_name]
                
                logger.info(f"Removed strategy: {strategy_name}")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove strategy {strategy_name}: {e}")
            return False
    
    def set_strategy_weight(self, strategy_name: str, weight: float) -> bool:
        """Set the weight for a strategy."""
        try:
            if strategy_name in self.active_strategies:
                self.strategy_weights[strategy_name] = weight
                logger.info(f"Set weight for {strategy_name}: {weight}")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to set weight for strategy {strategy_name}: {e}")
            return False
    
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from all active strategies."""
        try:
            all_signals = []
            
            for strategy_name, strategy in self.active_strategies.items():
                try:
                    # Generate signals from this strategy
                    strategy_signals = await strategy.generate_signals(analysis)
                    
                    # Apply strategy weight
                    weight = self.strategy_weights.get(strategy_name, 1.0)
                    for signal in strategy_signals:
                        signal['strategy'] = strategy_name
                        signal['weight'] = weight
                        signal['confidence'] = signal.get('confidence', 0.5) * weight
                        all_signals.append(signal)
                    
                except Exception as e:
                    logger.error(f"Error generating signals from strategy {strategy_name}: {e}")
            
            # Combine and rank signals
            combined_signals = await self._combine_signals(all_signals)
            
            # Store in history
            self._store_signals(combined_signals)
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _combine_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine and rank signals from multiple strategies."""
        try:
            if not signals:
                return []
            
            # Group signals by symbol and type
            signal_groups = {}
            for signal in signals:
                key = (signal.get('symbol', 'UNKNOWN'), signal.get('type', 'UNKNOWN'))
                if key not in signal_groups:
                    signal_groups[key] = []
                signal_groups[key].append(signal)
            
            # Combine signals for each group
            combined_signals = []
            for (symbol, signal_type), group_signals in signal_groups.items():
                if len(group_signals) == 1:
                    # Single signal, use as is
                    combined_signals.append(group_signals[0])
                else:
                    # Multiple signals, combine them
                    combined_signal = await self._combine_signal_group(group_signals)
                    combined_signals.append(combined_signal)
            
            # Sort by confidence
            combined_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return signals
    
    async def _combine_signal_group(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine a group of signals for the same symbol and type."""
        try:
            if not signals:
                return {}
            
            # Weighted average of quantities and confidences
            total_weight = sum(signal.get('weight', 1.0) for signal in signals)
            weighted_quantity = sum(
                signal.get('quantity', 0) * signal.get('weight', 1.0) 
                for signal in signals
            ) / total_weight if total_weight > 0 else 0
            
            weighted_confidence = sum(
                signal.get('confidence', 0) * signal.get('weight', 1.0) 
                for signal in signals
            ) / total_weight if total_weight > 0 else 0
            
            # Use the first signal as base and update with combined values
            combined_signal = signals[0].copy()
            combined_signal['quantity'] = weighted_quantity
            combined_signal['confidence'] = weighted_confidence
            combined_signal['strategies'] = [s.get('strategy', 'unknown') for s in signals]
            combined_signal['combined_from'] = len(signals)
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error combining signal group: {e}")
            return signals[0] if signals else {}
    
    def _store_signals(self, signals: List[Dict[str, Any]]):
        """Store signals in history."""
        try:
            timestamp = datetime.now()
            
            for signal in signals:
                signal_record = {
                    'timestamp': timestamp,
                    'signal': signal.copy()
                }
                
                self.signal_history.append(signal_record)
            
            # Trim history if too long
            if len(self.signal_history) > self.max_signal_history:
                self.signal_history = self.signal_history[-self.max_signal_history:]
                
        except Exception as e:
            logger.error(f"Error storing signals: {e}")
    
    def get_signal_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get signal history."""
        try:
            history = self.signal_history.copy()
            if limit:
                history = history[-limit:]
            return history
            
        except Exception as e:
            logger.error(f"Error getting signal history: {e}")
            return []
    
    def get_active_strategies(self) -> Dict[str, TradingStrategy]:
        """Get all active strategies."""
        return self.active_strategies.copy()
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get strategy weights."""
        return self.strategy_weights.copy()
    
    def get_executor_status(self) -> Dict[str, Any]:
        """Get executor status."""
        return {
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "active_strategies": list(self.active_strategies.keys()),
            "strategy_weights": self.strategy_weights.copy(),
            "signal_history_count": len(self.signal_history),
            "execution_task_running": self.execution_task is not None and not self.execution_task.done()
        }
    
    async def test_strategy(self, strategy_name: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a specific strategy with test data."""
        try:
            if strategy_name not in self.active_strategies:
                return {"error": f"Strategy {strategy_name} not found"}
            
            strategy = self.active_strategies[strategy_name]
            
            # Analyze test data
            analysis = await strategy.analyze(test_data)
            
            # Generate signals
            signals = await strategy.generate_signals(analysis)
            
            return {
                "strategy_name": strategy_name,
                "analysis": analysis,
                "signals": signals,
                "signal_count": len(signals)
            }
            
        except Exception as e:
            logger.error(f"Error testing strategy {strategy_name}: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            logger.info("Cleaning up Strategy Executor...")
            
            # Stop executor
            await self.stop()
            
            # Clean up strategies
            for strategy in self.active_strategies.values():
                if hasattr(strategy, 'cleanup'):
                    await strategy.cleanup()
            
            self.active_strategies.clear()
            self.strategy_weights.clear()
            self.signal_history.clear()
            
            logger.info("Strategy Executor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Strategy Executor cleanup: {e}") 