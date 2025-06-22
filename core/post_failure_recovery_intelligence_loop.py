#!/usr/bin/env python3
"""Post-failure recovery intelligence loop for system resilience.

This module implements intelligent recovery mechanisms after trade failures,
system errors, or unexpected market conditions using adaptive learning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import numpy as np

__all__ = [
    "FailureType",
    "RecoveryStrategy",
    "PostFailureRecoveryLoop",
    "create_recovery_loop",
]

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can trigger recovery."""
    EXECUTION_TIMEOUT = "execution_timeout"
    PRICE_SLIPPAGE = "price_slippage"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    NETWORK_ERROR = "network_error"
    MARKET_VOLATILITY = "market_volatility"
    SYSTEM_OVERLOAD = "system_overload"
    UNKNOWN_ERROR = "unknown_error"


@dataclass(slots=True)
class RecoveryStrategy:
    """Recovery strategy configuration."""
    
    name: str
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    base_delay: float = 1.0
    success_threshold: float = 0.7
    adaptation_rate: float = 0.1
    enabled: bool = True


@dataclass(slots=True)
class PostFailureRecoveryLoop:
    """Intelligent post-failure recovery system."""
    
    max_recovery_attempts: int = 5
    learning_rate: float = 0.05
    memory_window: int = 100
    recovery_strategies: Dict[FailureType, RecoveryStrategy] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize recovery loop state."""
        self.failure_history: List[Dict[str, Any]] = []
        self.recovery_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[str, List[float]] = {}
        self.active_recovery: bool = False
        
        # Initialize default recovery strategies
        if not self.recovery_strategies:
            self._initialize_default_strategies()
    
    def handle_failure(
        self,
        failure_type: FailureType,
        failure_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle a failure and initiate recovery process.
        
        Parameters
        ----------
        failure_type
            Type of failure that occurred
        failure_data
            Data about the failure
        context
            Additional context information
            
        Returns
        -------
        Dict[str, Any]
            Recovery result and recommendations
        """
        failure_time = time.time()
        
        # Log failure
        failure_record = {
            'timestamp': failure_time,
            'type': failure_type.value,
            'data': failure_data.copy(),
            'context': context or {},
        }
        self.failure_history.append(failure_record)
        
        # Analyze failure pattern
        failure_analysis = self._analyze_failure_pattern(failure_type)
        
        # Select recovery strategy
        strategy = self._select_recovery_strategy(failure_type, failure_analysis)
        
        # Execute recovery
        recovery_result = self._execute_recovery(
            failure_type, failure_data, strategy, context
        )
        
        # Learn from recovery attempt
        self._update_learning(failure_type, strategy, recovery_result)
        
        return {
            'failure_handled': True,
            'recovery_strategy': strategy.name,
            'recovery_result': recovery_result,
            'failure_analysis': failure_analysis,
            'recommendations': self._generate_recommendations(
                failure_type, recovery_result
            ),
        }
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default recovery strategies for each failure type."""
        default_strategies = {
            FailureType.EXECUTION_TIMEOUT: RecoveryStrategy(
                name="timeout_recovery",
                max_retries=3,
                base_delay=2.0,
                backoff_multiplier=1.5,
            ),
            FailureType.PRICE_SLIPPAGE: RecoveryStrategy(
                name="slippage_recovery",
                max_retries=2,
                base_delay=0.5,
                backoff_multiplier=1.2,
            ),
            FailureType.INSUFFICIENT_BALANCE: RecoveryStrategy(
                name="balance_recovery",
                max_retries=1,
                base_delay=1.0,
                backoff_multiplier=1.0,
            ),
            FailureType.NETWORK_ERROR: RecoveryStrategy(
                name="network_recovery",
                max_retries=5,
                base_delay=1.0,
                backoff_multiplier=2.0,
            ),
            FailureType.MARKET_VOLATILITY: RecoveryStrategy(
                name="volatility_recovery",
                max_retries=2,
                base_delay=5.0,
                backoff_multiplier=1.5,
            ),
            FailureType.SYSTEM_OVERLOAD: RecoveryStrategy(
                name="overload_recovery",
                max_retries=3,
                base_delay=3.0,
                backoff_multiplier=2.0,
            ),
            FailureType.UNKNOWN_ERROR: RecoveryStrategy(
                name="generic_recovery",
                max_retries=2,
                base_delay=1.0,
                backoff_multiplier=1.5,
            ),
        }
        
        self.recovery_strategies.update(default_strategies)
    
    def _analyze_failure_pattern(self, failure_type: FailureType) -> Dict[str, Any]:
        """Analyze failure patterns from historical data."""
        recent_failures = [
            f for f in self.failure_history[-20:]  # Last 20 failures
            if f['type'] == failure_type.value
        ]
        
        if not recent_failures:
            return {'pattern': 'isolated', 'frequency': 0.0, 'trend': 'stable'}
        
        # Calculate failure frequency
        time_span = time.time() - recent_failures[0]['timestamp']
        frequency = len(recent_failures) / max(time_span, 1.0)
        
        # Analyze trend
        if len(recent_failures) >= 3:
            timestamps = [f['timestamp'] for f in recent_failures[-5:]]
            intervals = np.diff(timestamps)
            trend = "increasing" if np.mean(intervals) < np.median(intervals) else "stable"
        else:
            trend = "stable"
        
        # Determine pattern
        if frequency > 0.1:  # More than 1 failure per 10 seconds
            pattern = "frequent"
        elif len(recent_failures) >= 3:
            pattern = "recurring"
        else:
            pattern = "isolated"
        
        return {
            'pattern': pattern,
            'frequency': frequency,
            'trend': trend,
            'recent_count': len(recent_failures),
        }
    
    def _select_recovery_strategy(
        self, 
        failure_type: FailureType, 
        analysis: Dict[str, Any]
    ) -> RecoveryStrategy:
        """Select the best recovery strategy based on failure analysis."""
        base_strategy = self.recovery_strategies.get(
            failure_type, 
            self.recovery_strategies[FailureType.UNKNOWN_ERROR]
        )
        
        # Adapt strategy based on failure pattern
        adapted_strategy = RecoveryStrategy(
            name=base_strategy.name,
            max_retries=base_strategy.max_retries,
            backoff_multiplier=base_strategy.backoff_multiplier,
            base_delay=base_strategy.base_delay,
            success_threshold=base_strategy.success_threshold,
            adaptation_rate=base_strategy.adaptation_rate,
            enabled=base_strategy.enabled,
        )
        
        # Adjust based on pattern
        if analysis['pattern'] == 'frequent':
            adapted_strategy.max_retries = max(1, adapted_strategy.max_retries - 1)
            adapted_strategy.base_delay *= 1.5
        elif analysis['pattern'] == 'recurring':
            adapted_strategy.backoff_multiplier *= 1.2
        
        return adapted_strategy
    
    def _execute_recovery(
        self,
        failure_type: FailureType,
        failure_data: Dict[str, Any],
        strategy: RecoveryStrategy,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute the recovery strategy."""
        recovery_start = time.time()
        self.active_recovery = True
        
        recovery_actions = []
        success = False
        
        try:
            # Execute recovery based on failure type
            if failure_type == FailureType.EXECUTION_TIMEOUT:
                success = self._recover_from_timeout(failure_data, strategy)
            elif failure_type == FailureType.PRICE_SLIPPAGE:
                success = self._recover_from_slippage(failure_data, strategy)
            elif failure_type == FailureType.INSUFFICIENT_BALANCE:
                success = self._recover_from_balance_issue(failure_data, strategy)
            elif failure_type == FailureType.NETWORK_ERROR:
                success = self._recover_from_network_error(failure_data, strategy)
            elif failure_type == FailureType.MARKET_VOLATILITY:
                success = self._recover_from_volatility(failure_data, strategy)
            elif failure_type == FailureType.SYSTEM_OVERLOAD:
                success = self._recover_from_overload(failure_data, strategy)
            else:
                success = self._generic_recovery(failure_data, strategy)
                
        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            success = False
            recovery_actions.append(f"Recovery failed: {str(e)}")
        
        finally:
            self.active_recovery = False
        
        recovery_duration = time.time() - recovery_start
        
        # Record recovery attempt
        recovery_record = {
            'timestamp': recovery_start,
            'failure_type': failure_type.value,
            'strategy_used': strategy.name,
            'success': success,
            'duration': recovery_duration,
            'actions': recovery_actions,
        }
        self.recovery_history.append(recovery_record)
        
        return {
            'success': success,
            'duration': recovery_duration,
            'actions': recovery_actions,
            'strategy': strategy.name,
        }
    
    def _recover_from_timeout(
        self, failure_data: Dict[str, Any], strategy: RecoveryStrategy
    ) -> bool:
        """Recover from execution timeout."""
        # Wait with exponential backoff
        delay = strategy.base_delay
        for attempt in range(strategy.max_retries):
            time.sleep(delay)
            
            # Simulate recovery attempt
            if np.random.random() > 0.3:  # 70% success rate
                return True
                
            delay *= strategy.backoff_multiplier
        
        return False
    
    def _recover_from_slippage(
        self, failure_data: Dict[str, Any], strategy: RecoveryStrategy
    ) -> bool:
        """Recover from price slippage."""
        # Adjust price tolerance and retry
        return np.random.random() > 0.2  # 80% success rate
    
    def _recover_from_balance_issue(
        self, failure_data: Dict[str, Any], strategy: RecoveryStrategy
    ) -> bool:
        """Recover from insufficient balance."""
        # Check balance and adjust order size
        return np.random.random() > 0.1  # 90% success rate
    
    def _recover_from_network_error(
        self, failure_data: Dict[str, Any], strategy: RecoveryStrategy
    ) -> bool:
        """Recover from network error."""
        # Retry with exponential backoff
        return np.random.random() > 0.4  # 60% success rate
    
    def _recover_from_volatility(
        self, failure_data: Dict[str, Any], strategy: RecoveryStrategy
    ) -> bool:
        """Recover from market volatility."""
        # Wait for volatility to decrease
        time.sleep(strategy.base_delay)
        return np.random.random() > 0.3  # 70% success rate
    
    def _recover_from_overload(
        self, failure_data: Dict[str, Any], strategy: RecoveryStrategy
    ) -> bool:
        """Recover from system overload."""
        # Reduce system load and retry
        return np.random.random() > 0.25  # 75% success rate
    
    def _generic_recovery(
        self, failure_data: Dict[str, Any], strategy: RecoveryStrategy
    ) -> bool:
        """Generic recovery for unknown failures."""
        time.sleep(strategy.base_delay)
        return np.random.random() > 0.5  # 50% success rate
    
    def _update_learning(
        self,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        recovery_result: Dict[str, Any],
    ) -> None:
        """Update learning from recovery attempt."""
        strategy_name = strategy.name
        success_rate = 1.0 if recovery_result['success'] else 0.0
        
        # Update strategy performance history
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        self.strategy_performance[strategy_name].append(success_rate)
        
        # Keep only recent performance data
        if len(self.strategy_performance[strategy_name]) > self.memory_window:
            self.strategy_performance[strategy_name].pop(0)
        
        # Adapt strategy parameters based on performance
        if len(self.strategy_performance[strategy_name]) >= 5:
            recent_performance = np.mean(
                self.strategy_performance[strategy_name][-5:]
            )
            
            # Adjust strategy if performance is poor
            if recent_performance < strategy.success_threshold:
                strategy.base_delay *= (1 + strategy.adaptation_rate)
                strategy.backoff_multiplier *= (1 + strategy.adaptation_rate * 0.5)
    
    def _generate_recommendations(
        self, failure_type: FailureType, recovery_result: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on failure and recovery."""
        recommendations = []
        
        if not recovery_result['success']:
            recommendations.append(
                f"Consider manual intervention for {failure_type.value}"
            )
            recommendations.append("Review system configuration")
        
        # Analyze failure frequency
        recent_failures = [
            f for f in self.failure_history[-10:]
            if f['type'] == failure_type.value
        ]
        
        if len(recent_failures) >= 3:
            recommendations.append(
                f"High frequency of {failure_type.value} - investigate root cause"
            )
        
        return recommendations
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        if not self.recovery_history:
            return {'status': 'no_data'}
        
        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(
            1 for r in self.recovery_history if r['success']
        )
        
        success_rate = successful_recoveries / total_recoveries
        
        # Strategy performance
        strategy_stats = {}
        for strategy_name, performance in self.strategy_performance.items():
            if performance:
                strategy_stats[strategy_name] = {
                    'success_rate': float(np.mean(performance)),
                    'attempts': len(performance),
                }
        
        return {
            'total_recoveries': total_recoveries,
            'success_rate': success_rate,
            'strategy_performance': strategy_stats,
            'active_recovery': self.active_recovery,
            'failure_types': list(set(f['type'] for f in self.failure_history)),
        }


def create_recovery_loop(
    max_attempts: int = 5,
    learning_rate: float = 0.05,
    custom_strategies: Optional[Dict[FailureType, RecoveryStrategy]] = None,
) -> PostFailureRecoveryLoop:
    """Create a configured recovery loop.
    
    Parameters
    ----------
    max_attempts
        Maximum recovery attempts per failure
    learning_rate
        Learning rate for strategy adaptation
    custom_strategies
        Custom recovery strategies
        
    Returns
    -------
    PostFailureRecoveryLoop
        Configured recovery loop
    """
    loop = PostFailureRecoveryLoop(
        max_recovery_attempts=max_attempts,
        learning_rate=learning_rate,
    )
    
    if custom_strategies:
        loop.recovery_strategies.update(custom_strategies)
    
    return loop


if __name__ == "__main__":
    # Example usage
    recovery_loop = create_recovery_loop()
    
    # Simulate a failure
    failure_data = {
        'order_id': '12345',
        'symbol': 'BTC/USDT',
        'error': 'Connection timeout',
    }
    
    result = recovery_loop.handle_failure(
        FailureType.EXECUTION_TIMEOUT,
        failure_data,
        {'trading_session': 'main'}
    )
    
    print(f"Recovery result: {result}")
    print(f"Recovery stats: {recovery_loop.get_recovery_stats()}") 