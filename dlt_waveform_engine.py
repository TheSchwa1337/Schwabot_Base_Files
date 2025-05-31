"""
Diogenic Logic Trading (DLT) Waveform Engine
Implements recursive pattern recognition and phase validation for trading decisions
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
from datetime import datetime, timedelta

class PhaseDomain(Enum):
    SHORT = "short"    # Seconds to Hours
    MID = "mid"        # Hours to Days  
    LONG = "long"      # Days to Months

@dataclass
class PhaseTrust:
    """Trust metrics for each phase domain"""
    successful_echoes: int
    entropy_consistency: float
    last_validation: datetime
    trust_threshold: float = 0.8

@dataclass 
class BitmapTrigger:
    """Represents a trigger point in the 16-bit trading map"""
    phase: PhaseDomain
    time_window: timedelta
    diogenic_score: float
    frequency: float
    last_trigger: datetime
    success_count: int

class DLTWaveformEngine:
    """
    Core engine for Diogenic Logic Trading pattern recognition
    """
    
    def __init__(self):
        # 16-bit trading map (4-bit, 8-bit, 16-bit allocations)
        self.bitmap: np.ndarray = np.zeros(16, dtype=bool)
        
        # Phase trust tracking
        self.phase_trust: Dict[PhaseDomain, PhaseTrust] = {
            PhaseDomain.SHORT: PhaseTrust(0, 0.0, datetime.now()),
            PhaseDomain.MID: PhaseTrust(0, 0.0, datetime.now()),
            PhaseDomain.LONG: PhaseTrust(0, 0.0, datetime.now())
        }
        
        # Trigger memory
        self.triggers: List[BitmapTrigger] = []
        
        # Phase validation thresholds
        self.phase_thresholds = {
            PhaseDomain.LONG: 3,    # 3+ successful echoes in 90d
            PhaseDomain.MID: 5,     # 5+ echoes with entropy consistency
            PhaseDomain.SHORT: 10   # 10+ phase-aligned echoes
        }
        
    def update_phase_trust(self, phase: PhaseDomain, success: bool, entropy: float):
        """Update trust metrics for a phase domain"""
        trust = self.phase_trust[phase]
        
        if success:
            trust.successful_echoes += 1
            trust.entropy_consistency = (trust.entropy_consistency * 0.9 + entropy * 0.1)
        
        trust.last_validation = datetime.now()
        
    def is_phase_trusted(self, phase: PhaseDomain) -> bool:
        """Check if a phase domain has sufficient trust for trading"""
        trust = self.phase_trust[phase]
        threshold = self.phase_thresholds[phase]
        
        return (
            trust.successful_echoes >= threshold and
            trust.entropy_consistency >= trust.trust_threshold
        )
        
    def compute_trigger_score(self, t: datetime, phase: PhaseDomain) -> float:
        """
        Compute trigger score based on bitmap pattern and phase
        Returns score between 0 and 1
        """
        if not self.is_phase_trusted(phase):
            return 0.0
            
        # Get relevant triggers for this phase
        phase_triggers = [tr for tr in self.triggers if tr.phase == phase]
        
        if not phase_triggers:
            return 0.0
            
        # Compute weighted sum of diogenic scores and frequencies
        total_score = 0.0
        total_weight = 0.0
        
        for trigger in phase_triggers:
            # Weight by recency and success
            time_weight = np.exp(-(t - trigger.last_trigger).total_seconds() / 86400)  # 24h decay
            success_weight = np.log(1 + trigger.success_count)
            
            weight = time_weight * success_weight
            score = trigger.diogenic_score * trigger.frequency
            
            total_score += score * weight
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        return total_score / total_weight
        
    def add_trigger(self, phase: PhaseDomain, window: timedelta, 
                   diogenic_score: float, frequency: float):
        """Add a new trigger point to the memory"""
        trigger = BitmapTrigger(
            phase=phase,
            time_window=window,
            diogenic_score=diogenic_score,
            frequency=frequency,
            last_trigger=datetime.now(),
            success_count=1
        )
        self.triggers.append(trigger)
        
        # Update bitmap
        bit_index = self._get_bit_index(phase, window)
        if bit_index is not None:
            self.bitmap[bit_index] = True
            
    def _get_bit_index(self, phase: PhaseDomain, window: timedelta) -> Optional[int]:
        """Convert phase and window to bitmap index"""
        if phase == PhaseDomain.SHORT:
            # 0-3: seconds to hours
            hours = window.total_seconds() / 3600
            if hours <= 1:
                return 0
            elif hours <= 4:
                return 1
            elif hours <= 12:
                return 2
            elif hours <= 24:
                return 3
        elif phase == PhaseDomain.MID:
            # 4-11: hours to days
            days = window.total_seconds() / 86400
            if days <= 2:
                return 4
            elif days <= 4:
                return 5
            elif days <= 7:
                return 6
            elif days <= 14:
                return 7
            elif days <= 21:
                return 8
            elif days <= 30:
                return 9
            elif days <= 45:
                return 10
            elif days <= 60:
                return 11
        elif phase == PhaseDomain.LONG:
            # 12-15: days to months
            days = window.total_seconds() / 86400
            if days <= 90:
                return 12
            elif days <= 120:
                return 13
            elif days <= 150:
                return 14
            elif days <= 180:
                return 15
        return None
        
    def evaluate_trade_trigger(self, phase: PhaseDomain, 
                             current_time: datetime,
                             entropy: float,
                             volume: float) -> Tuple[bool, float]:
        """
        Evaluate if current conditions match a trusted trigger pattern
        Returns (should_trigger, confidence)
        """
        # Check phase trust
        if not self.is_phase_trusted(phase):
            return False, 0.0
            
        # Compute trigger score
        score = self.compute_trigger_score(current_time, phase)
        
        # Additional validation for short-term trades
        if phase == PhaseDomain.SHORT:
            if volume < 1000000:  # Example minimum volume
                return False, 0.0
                
        # Final decision
        should_trigger = score > 0.7  # Example threshold
        
        return should_trigger, score 