"""
Schwabot Net Stop Loss Pattern Value Book
Implements stop loss pattern detection and management
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

class StopPatternState(Enum):
    """Enumeration of possible stop pattern states"""
    NORMAL = "normal"
    WARNING = "warning"
    ALERT = "alert"
    TRIGGERED = "triggered"
    RECOVERY = "recovery"
    RESET = "reset"

@dataclass
class StopPattern:
    """Data structure for stop pattern information"""
    state: StopPatternState
    timestamp: datetime
    value: float
    threshold: float
    duration: int
    confidence: float
    metadata: Dict

class SchwabotStopBook:
    """
    Implements the Schwabot Net Stop Loss Pattern Value Book
    Manages stop loss patterns and their state transitions
    """
    
    def __init__(self, 
                 warning_threshold: float = 0.02,
                 alert_threshold: float = 0.05,
                 trigger_threshold: float = 0.08,
                 recovery_threshold: float = 0.03,
                 min_duration: int = 3,
                 max_duration: int = 20,
                 confidence_threshold: float = 0.7):
        """
        Initialize the stop loss pattern book
        
        Args:
            warning_threshold: Threshold for warning state
            alert_threshold: Threshold for alert state
            trigger_threshold: Threshold for triggered state
            recovery_threshold: Threshold for recovery state
            min_duration: Minimum duration for pattern recognition
            max_duration: Maximum duration for pattern tracking
            confidence_threshold: Minimum confidence for pattern validation
        """
        self.warning_threshold = warning_threshold
        self.alert_threshold = alert_threshold
        self.trigger_threshold = trigger_threshold
        self.recovery_threshold = recovery_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.confidence_threshold = confidence_threshold
        
        self.current_patterns: Dict[str, StopPattern] = {}
        self.pattern_history: List[StopPattern] = []
        self.state_transitions: List[Tuple[StopPatternState, StopPatternState, datetime]] = []
    
    def update_pattern(self, 
                      pattern_id: str,
                      value: float,
                      timestamp: datetime,
                      metadata: Optional[Dict] = None) -> StopPatternState:
        """
        Update a stop pattern with new value and determine state transition
        
        Args:
            pattern_id: Unique identifier for the pattern
            value: Current pattern value
            timestamp: Current timestamp
            metadata: Additional pattern metadata
            
        Returns:
            New pattern state
        """
        if pattern_id not in self.current_patterns:
            # Initialize new pattern
            pattern = StopPattern(
                state=StopPatternState.NORMAL,
                timestamp=timestamp,
                value=value,
                threshold=self.warning_threshold,
                duration=1,
                confidence=1.0,
                metadata=metadata or {}
            )
            self.current_patterns[pattern_id] = pattern
            return pattern.state
        
        pattern = self.current_patterns[pattern_id]
        old_state = pattern.state
        
        # Update pattern values
        pattern.value = value
        pattern.timestamp = timestamp
        pattern.duration += 1
        if metadata:
            pattern.metadata.update(metadata)
        
        # Calculate confidence based on duration and value stability
        pattern.confidence = self._calculate_confidence(pattern)
        
        # Determine state transition
        new_state = self._determine_state_transition(pattern)
        pattern.state = new_state
        
        # Record state transition
        if new_state != old_state:
            self.state_transitions.append((old_state, new_state, timestamp))
        
        # Handle pattern completion
        if new_state in [StopPatternState.TRIGGERED, StopPatternState.RESET]:
            self.pattern_history.append(pattern)
            del self.current_patterns[pattern_id]
        
        return new_state
    
    def _calculate_confidence(self, pattern: StopPattern) -> float:
        """
        Calculate pattern confidence based on duration and value stability
        
        Args:
            pattern: Stop pattern to evaluate
            
        Returns:
            Confidence score between 0 and 1
        """
        duration_factor = min(pattern.duration / self.min_duration, 1.0)
        stability_factor = 1.0 - min(abs(pattern.value) / self.trigger_threshold, 1.0)
        return (duration_factor + stability_factor) / 2
    
    def _determine_state_transition(self, pattern: StopPattern) -> StopPatternState:
        """
        Determine the new state based on current pattern values
        
        Args:
            pattern: Stop pattern to evaluate
            
        Returns:
            New pattern state
        """
        if pattern.confidence < self.confidence_threshold:
            return StopPatternState.RESET
        
        value = abs(pattern.value)
        
        if pattern.state == StopPatternState.NORMAL:
            if value >= self.warning_threshold:
                return StopPatternState.WARNING
            return StopPatternState.NORMAL
            
        elif pattern.state == StopPatternState.WARNING:
            if value >= self.alert_threshold:
                return StopPatternState.ALERT
            elif value < self.warning_threshold:
                return StopPatternState.NORMAL
            return StopPatternState.WARNING
            
        elif pattern.state == StopPatternState.ALERT:
            if value >= self.trigger_threshold:
                return StopPatternState.TRIGGERED
            elif value < self.warning_threshold:
                return StopPatternState.NORMAL
            return StopPatternState.ALERT
            
        elif pattern.state == StopPatternState.TRIGGERED:
            if value < self.recovery_threshold:
                return StopPatternState.RECOVERY
            return StopPatternState.TRIGGERED
            
        elif pattern.state == StopPatternState.RECOVERY:
            if value < self.warning_threshold:
                return StopPatternState.NORMAL
            elif value >= self.alert_threshold:
                return StopPatternState.ALERT
            return StopPatternState.RECOVERY
            
        return StopPatternState.RESET
    
    def get_active_patterns(self) -> Dict[str, StopPattern]:
        """
        Get all currently active patterns
        
        Returns:
            Dictionary of active patterns
        """
        return self.current_patterns
    
    def get_pattern_history(self) -> List[StopPattern]:
        """
        Get history of completed patterns
        
        Returns:
            List of completed patterns
        """
        return self.pattern_history
    
    def get_state_transitions(self) -> List[Tuple[StopPatternState, StopPatternState, datetime]]:
        """
        Get history of state transitions
        
        Returns:
            List of state transitions with timestamps
        """
        return self.state_transitions
    
    def recalibrate_thresholds(self, 
                             new_warning: Optional[float] = None,
                             new_alert: Optional[float] = None,
                             new_trigger: Optional[float] = None,
                             new_recovery: Optional[float] = None):
        """
        Recalibrate pattern thresholds
        
        Args:
            new_warning: New warning threshold
            new_alert: New alert threshold
            new_trigger: New trigger threshold
            new_recovery: New recovery threshold
        """
        if new_warning is not None:
            self.warning_threshold = new_warning
        if new_alert is not None:
            self.alert_threshold = new_alert
        if new_trigger is not None:
            self.trigger_threshold = new_trigger
        if new_recovery is not None:
            self.recovery_threshold = new_recovery 