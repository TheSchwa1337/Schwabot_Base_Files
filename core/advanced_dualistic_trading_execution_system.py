#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Dualistic Trading Execution System Module
==================================================
Provides advanced dualistic trading execution functionality for the Schwabot trading system.

Main Classes:
- ExecutionMode: Core dualistic trading execution functionality
- DecisionCurveCalculator: Decision curve calculations
- TradeSwitchResolver: Trade switch resolution logic

Key Functions:
- sigmoid: Sigmoid activation function for decision making
- calculate_decision_curve: Calculate decision curves for trading signals
- resolve_trade_switch: Resolve trade switching decisions
- execute_dualistic_trade: Execute dualistic trading operations

"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Import dependencies
try:
    from core.math_cache import MathResultCache
    from core.math_config_manager import MathConfigManager
    from core.math_orchestrator import MathOrchestrator
    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Math infrastructure not available")


class Status(Enum):
    """System status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PROCESSING = "processing"


class Mode(Enum):
    """Operation mode enumeration."""
    NORMAL = "normal"
    DEBUG = "debug"
    TEST = "test"
    PRODUCTION = "production"


class TradeDirection(Enum):
    """Trade direction enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class DualisticConfig:
    """Dualistic Trading Configuration data class."""
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False
    sigmoid_steepness: float = 1.0  # Sigmoid steepness parameter
    decision_threshold: float = 0.5  # Decision threshold
    switch_sensitivity: float = 0.1  # Switch sensitivity


@dataclass
class TradingResult:
    """Trading Result data class."""
    success: bool = False
    direction: Optional[TradeDirection] = None
    confidence: Optional[float] = None
    decision_curve: Optional[np.ndarray] = None
    switch_resolved: Optional[bool] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class DecisionCurveCalculator:
    """Decision Curve Calculator for dualistic trading."""
    
    def __init__(self, config: Optional[DualisticConfig] = None):
        self.config = config or DualisticConfig()
        self.logger = logging.getLogger(f"{__name__}.DecisionCurveCalculator")
        
    def sigmoid(self, x: Union[float, np.ndarray], 
                steepness: float = None) -> Union[float, np.ndarray]:
        """
        Sigmoid activation function for decision making.
        
        Args:
            x: Input value(s)
            steepness: Sigmoid steepness parameter
            
        Returns:
            Sigmoid output value(s)
        """
        try:
            if steepness is None:
                steepness = self.config.sigmoid_steepness
            
            # Sigmoid function: 1 / (1 + e^(-steepness * x))
            sigmoid_output = 1.0 / (1.0 + np.exp(-steepness * x))
            
            self.logger.debug(f"Sigmoid calculated with steepness {steepness}")
            return sigmoid_output
            
        except Exception as e:
            self.logger.error(f"Error calculating sigmoid: {e}")
            return 0.5  # Default to neutral
    
    def calculate_decision_curve(self, signal_data: np.ndarray, 
                               time_points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate decision curve for trading signals.
        
        Args:
            signal_data: Input signal data array
            time_points: Time points for the curve (optional)
            
        Returns:
            Decision curve array
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(signal_data, np.ndarray):
                signal_data = np.array(signal_data)
            
            # Normalize signal data to [-1, 1] range
            signal_normalized = 2 * (signal_data - np.min(signal_data)) / \
                              (np.max(signal_data) - np.min(signal_data) + 1e-10) - 1
            
            # Apply sigmoid to create decision curve
            decision_curve = self.sigmoid(signal_normalized)
            
            # Apply smoothing if time points are provided
            if time_points is not None and len(time_points) > 1:
                # Simple moving average smoothing
                window_size = min(5, len(decision_curve) // 10)
                if window_size > 1:
                    decision_curve = np.convolve(decision_curve, 
                                               np.ones(window_size) / window_size, 
                                               mode='same')
            
            self.logger.debug(f"Decision curve calculated: {len(decision_curve)} points")
            return decision_curve
            
        except Exception as e:
            self.logger.error(f"Error calculating decision curve: {e}")
            return np.zeros_like(signal_data)


class TradeSwitchResolver:
    """Trade Switch Resolver for dualistic trading decisions."""
    
    def __init__(self, config: Optional[DualisticConfig] = None):
        self.config = config or DualisticConfig()
        self.logger = logging.getLogger(f"{__name__}.TradeSwitchResolver")
        self.last_direction = TradeDirection.HOLD
        
    def resolve_trade_switch(self, decision_curve: np.ndarray, 
                           threshold: float = None) -> Tuple[TradeDirection, float]:
        """
        Resolve trade switching decisions based on decision curve.
        
        Args:
            decision_curve: Decision curve array
            threshold: Decision threshold
            
        Returns:
            Tuple of (trade_direction, confidence)
        """
        try:
            if threshold is None:
                threshold = self.config.decision_threshold
            
            # Get current decision value (last point in curve)
            current_decision = decision_curve[-1]
            
            # Calculate confidence based on distance from threshold
            confidence = abs(current_decision - threshold) / (1.0 - threshold)
            confidence = min(confidence, 1.0)
            
            # Determine trade direction
            if current_decision > threshold + self.config.switch_sensitivity:
                direction = TradeDirection.BUY
            elif current_decision < threshold - self.config.switch_sensitivity:
                direction = TradeDirection.SELL
            else:
                direction = TradeDirection.HOLD
            
            # Check if direction has changed (switch occurred)
            switch_resolved = direction != self.last_direction
            self.last_direction = direction
            
            self.logger.debug(f"Trade switch resolved: {direction.value}, "
                             f"confidence: {confidence:.3f}")
            
            return direction, confidence
            
        except Exception as e:
            self.logger.error(f"Error resolving trade switch: {e}")
            return TradeDirection.HOLD, 0.0


class ExecutionMode:
    """
    ExecutionMode Implementation
    Provides core advanced dualistic trading execution system functionality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ExecutionMode with configuration."""
        self.config = DualisticConfig(**(config or {}))
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False
        
        # Initialize components
        self.decision_calculator = DecisionCurveCalculator(self.config)
        self.switch_resolver = TradeSwitchResolver(self.config)

        # Initialize math infrastructure if available
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_config = MathConfigManager()
            self.math_cache = MathResultCache()
            self.math_orchestrator = MathOrchestrator()

        self._initialize_system()

    def _initialize_system(self) -> None:
        """Initialize the system."""
        try:
            self.logger.info(f"Initializing {self.__class__.__name__}")
            self.initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False

    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False

        try:
            self.active = True
            self.logger.info(f"✅ {self.__class__.__name__} activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating {self.__class__.__name__}: {e}")
            return False

    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info(f"✅ {self.__class__.__name__} deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
            return False

    def sigmoid(self, x: Union[float, np.ndarray], 
                steepness: float = None) -> Union[float, np.ndarray]:
        """
        Sigmoid activation function for decision making.
        
        Args:
            x: Input value(s)
            steepness: Sigmoid steepness parameter
            
        Returns:
            Sigmoid output value(s)
        """
        if not self.active:
            return 0.5
        
        return self.decision_calculator.sigmoid(x, steepness)

    def calculate_decision_curve(self, signal_data: Union[List, np.ndarray], 
                               time_points: Optional[Union[List, np.ndarray]] = None) -> np.ndarray:
        """
        Calculate decision curve for trading signals.
        
        Args:
            signal_data: Input signal data
            time_points: Time points for the curve (optional)
            
        Returns:
            Decision curve array
        """
        if not self.active:
            return np.zeros_like(signal_data)
        
        # Convert to numpy arrays
        signal_array = np.array(signal_data)
        time_array = np.array(time_points) if time_points is not None else None
        
        return self.decision_calculator.calculate_decision_curve(signal_array, time_array)

    def resolve_trade_switch(self, decision_curve: Union[List, np.ndarray], 
                           threshold: float = None) -> Tuple[TradeDirection, float]:
        """
        Resolve trade switching decisions based on decision curve.
        
        Args:
            decision_curve: Decision curve array
            threshold: Decision threshold
            
        Returns:
            Tuple of (trade_direction, confidence)
        """
        if not self.active:
            return TradeDirection.HOLD, 0.0
        
        curve_array = np.array(decision_curve)
        return self.switch_resolver.resolve_trade_switch(curve_array, threshold)

    def execute_dualistic_trade(self, signal_data: Union[List, np.ndarray], 
                              time_points: Optional[Union[List, np.ndarray]] = None) -> TradingResult:
        """
        Execute dualistic trading operation.
        
        Args:
            signal_data: Input signal data
            time_points: Time points for the curve (optional)
            
        Returns:
            Trading result
        """
        try:
            if not self.active:
                return TradingResult(success=False, error="System not active")
            
            # Convert to numpy arrays
            signal_array = np.array(signal_data)
            time_array = np.array(time_points) if time_points is not None else None
            
            # Calculate decision curve
            decision_curve = self.calculate_decision_curve(signal_array, time_array)
            
            # Resolve trade switch
            direction, confidence = self.resolve_trade_switch(decision_curve)
            
            # Check if switch occurred
            switch_resolved = direction != self.switch_resolver.last_direction
            
            return TradingResult(
                success=True,
                direction=direction,
                confidence=confidence,
                decision_curve=decision_curve,
                switch_resolved=switch_resolved,
                data={
                    "signal_data": signal_data,
                    "time_points": time_points,
                    "threshold": self.config.decision_threshold
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in dualistic trade execution: {e}")
            return TradingResult(success=False, error=str(e))

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config.__dict__,
            'last_direction': self.switch_resolver.last_direction.value,
        }


# Factory function
def create_advanced_dualistic_trading_execution_system(config: Optional[Dict[str, Any]] = None) -> ExecutionMode:
    """Create an advanced dualistic trading execution system instance."""
    return ExecutionMode(config)
