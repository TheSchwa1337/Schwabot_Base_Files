"""
Enhanced Lockout Matrix with Self-Healing
=========================================

Advanced lockout system that prevents repeated failures while allowing
recovery through time-based expiration and pattern evolution detection.
Implements self-healing unlock logic to avoid permanent profit zone blocks.

Key Features:
- Time-based lockout expiration (half-life decay)
- Pattern evolution detection for early unlock
- Forced reevaluation after repeated suppressions
- Adaptive lockout duration based on failure severity
"""

import numpy as np
import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger(__name__)

class LockoutStatus(Enum):
    """Lockout entry status"""
    ACTIVE = "active"
    DECAYING = "decaying"
    EXPIRED = "expired"
    FORCE_REEVALUATE = "force_reevaluate"
    PATTERN_EVOLVED = "pattern_evolved"

class LockoutSeverity(Enum):
    """Severity levels for lockout duration"""
    MINOR = "minor"          # 5-15 minutes
    MODERATE = "moderate"    # 30-60 minutes  
    SEVERE = "severe"        # 2-6 hours
    CRITICAL = "critical"    # 12-24 hours

@dataclass
class LockoutEntry:
    """Individual lockout entry with self-healing properties"""
    signature_hash: str
    creation_time: float
    pattern_data: Dict[str, Any]
    failure_count: int
    severity: LockoutSeverity
    base_duration: float  # Base lockout duration in seconds
    half_life: float      # Half-life for exponential decay
    status: LockoutStatus = LockoutStatus.ACTIVE
    unlock_time: Optional[float] = None
    suppression_count: int = 0
    last_pattern_check: float = 0.0
    evolution_threshold: float = 0.3
    forced_reevaluations: int = 0

@dataclass
class LockoutMetrics:
    """Lockout system performance metrics"""
    total_lockouts_created: int = 0
    active_lockouts: int = 0
    expired_lockouts: int = 0
    self_healed_lockouts: int = 0
    forced_reevaluations: int = 0
    pattern_evolutions: int = 0
    avg_lockout_duration: float = 0.0
    lockout_effectiveness: float = 0.0

class EnhancedLockoutMatrix:
    """
    Self-healing lockout matrix that prevents permanent profit zone blocks.
    
    Implements intelligent lockout management with time-based expiration,
    pattern evolution detection, and forced reevaluation mechanisms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced lockout matrix.
        
        Args:
            config: Configuration parameters for lockout system
        """
        self.config = config or {}
        
        # Lockout duration parameters (in seconds)
        self.duration_ranges = {
            LockoutSeverity.MINOR: (300, 900),      # 5-15 minutes
            LockoutSeverity.MODERATE: (1800, 3600), # 30-60 minutes
            LockoutSeverity.SEVERE: (7200, 21600),  # 2-6 hours
            LockoutSeverity.CRITICAL: (43200, 86400) # 12-24 hours
        }
        
        # Self-healing parameters
        self.max_suppression_count = self.config.get('max_suppressions', 5)
        self.pattern_evolution_threshold = self.config.get('evolution_threshold', 0.3)
        self.forced_reevaluation_interval = self.config.get('reevaluation_interval', 3600)
        
        # Lockout storage
        self.active_lockouts: Dict[str, LockoutEntry] = {}
        self.expired_lockouts: Dict[str, LockoutEntry] = {}
        self.lockout_history: deque = deque(maxlen=1000)
        
        # Pattern tracking for evolution detection
        self.pattern_snapshots: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Metrics
        self.metrics = LockoutMetrics()
        
        logger.info("Enhanced Lockout Matrix initialized with self-healing capabilities")
    
    def create_lockout(self, pattern_data: Dict[str, Any], failure_reason: str,
                      severity: Optional[LockoutSeverity] = None) -> str:
        """
        Create new lockout entry with adaptive duration.
        
        Args:
            pattern_data: Pattern data that caused the failure
            failure_reason: Reason for the lockout
            severity: Lockout severity (auto-determined if None)
            
        Returns:
            Signature hash of the locked pattern
        """
        # Generate pattern signature
        signature_hash = self._generate_pattern_signature(pattern_data)
        
        # Check if lockout already exists
        if signature_hash in self.active_lockouts:
            return self._update_existing_lockout(signature_hash, failure_reason)
        
        # Determine severity if not provided
        if severity is None:
            severity = self._determine_lockout_severity(pattern_data, failure_reason)
        
        # Calculate lockout duration and half-life
        base_duration = self._calculate_lockout_duration(severity)
        half_life = base_duration * 0.5  # Half-life is 50% of base duration
        
        # Create lockout entry
        lockout_entry = LockoutEntry(
            signature_hash=signature_hash,
            creation_time=time.time(),
            pattern_data=pattern_data.copy(),
            failure_count=1,
            severity=severity,
            base_duration=base_duration,
            half_life=half_life,
            unlock_time=time.time() + base_duration
        )
        
        # Store lockout
        self.active_lockouts[signature_hash] = lockout_entry
        self.lockout_history.append(lockout_entry)
        
        # Initialize pattern tracking
        self.pattern_snapshots[signature_hash].append(pattern_data.copy())
        
        # Update metrics
        self.metrics.total_lockouts_created += 1
        self.metrics.active_lockouts += 1
        
        logger.info(f"Lockout created: {signature_hash[:8]} ({severity.value}) "
                   f"duration: {base_duration/60:.1f}min")
        
        return signature_hash
    
    def check_lockout_status(self, pattern_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if pattern is currently locked out.
        
        Args:
            pattern_data: Pattern data to check
            
        Returns:
            Tuple of (is_locked, reason_or_none)
        """
        signature_hash = self._generate_pattern_signature(pattern_data)
        
        # Check if pattern is in active lockouts
        if signature_hash not in self.active_lockouts:
            return False, None
        
        lockout_entry = self.active_lockouts[signature_hash]
        current_time = time.time()
        
        # Check if lockout has naturally expired
        if current_time >= lockout_entry.unlock_time:
            self._expire_lockout(signature_hash)
            return False, "lockout_expired"
        
        # Check for pattern evolution
        if self._check_pattern_evolution(signature_hash, pattern_data):
            self._unlock_pattern_evolution(signature_hash)
            return False, "pattern_evolved"
        
        # Check for forced reevaluation
        if self._should_force_reevaluation(lockout_entry):
            self._trigger_forced_reevaluation(signature_hash)
            return False, "forced_reevaluation"
        
        # Pattern is still locked
        remaining_time = lockout_entry.unlock_time - current_time
        return True, f"locked_for_{remaining_time/60:.1f}min"
    
    def update_lockout_weights(self) -> Dict[str, float]:
        """
        Update lockout weights based on time decay and self-healing.
        
        Returns:
            Dictionary of signature_hash -> current_lockout_weight
        """
        current_time = time.time()
        lockout_weights = {}
        expired_lockouts = []
        
        for signature_hash, lockout_entry in self.active_lockouts.items():
            # Calculate time-based decay weight
            time_elapsed = current_time - lockout_entry.creation_time
            
            # Exponential decay based on half-life
            decay_weight = np.exp(-time_elapsed / lockout_entry.half_life)
            
            # Check if weight has decayed below threshold
            if decay_weight < 0.1 or current_time >= lockout_entry.unlock_time:
                expired_lockouts.append(signature_hash)
            else:
                lockout_weights[signature_hash] = decay_weight
        
        # Process expired lockouts
        for signature_hash in expired_lockouts:
            self._expire_lockout(signature_hash)
        
        return lockout_weights
    
    def suppress_pattern_attempt(self, pattern_data: Dict[str, Any]) -> bool:
        """
        Record pattern suppression attempt and check for forced reevaluation.
        
        Args:
            pattern_data: Pattern that was suppressed
            
        Returns:
            True if suppression was recorded, False if forced reevaluation triggered
        """
        signature_hash = self._generate_pattern_signature(pattern_data)
        
        if signature_hash not in self.active_lockouts:
            return True  # No lockout exists, suppression is valid
        
        lockout_entry = self.active_lockouts[signature_hash]
        lockout_entry.suppression_count += 1
        
        # Check if we should trigger forced reevaluation
        if lockout_entry.suppression_count >= self.max_suppression_count:
            self._trigger_forced_reevaluation(signature_hash)
            return False
        
        return True
    
    def unlock_zone(self, signature_hash: str, reason: str = "manual_unlock") -> bool:
        """
        Manually unlock a pattern zone.
        
        Args:
            signature_hash: Hash of pattern to unlock
            reason: Reason for manual unlock
            
        Returns:
            True if unlock was successful
        """
        if signature_hash in self.active_lockouts:
            lockout_entry = self.active_lockouts[signature_hash]
            lockout_entry.status = LockoutStatus.EXPIRED
            
            # Move to expired lockouts
            self.expired_lockouts[signature_hash] = self.active_lockouts.pop(signature_hash)
            
            # Update metrics
            self.metrics.active_lockouts -= 1
            self.metrics.expired_lockouts += 1
            
            logger.info(f"Pattern {signature_hash[:8]} manually unlocked: {reason}")
            return True
        
        return False
    
    def _generate_pattern_signature(self, pattern_data: Dict[str, Any]) -> str:
        """Generate unique signature hash for pattern data."""
        # Create deterministic string representation
        pattern_str = str(sorted(pattern_data.items()))
        
        # Generate hash
        return hashlib.md5(pattern_str.encode()).hexdigest()
    
    def _update_existing_lockout(self, signature_hash: str, failure_reason: str) -> str:
        """Update existing lockout with new failure."""
        lockout_entry = self.active_lockouts[signature_hash]
        lockout_entry.failure_count += 1
        
        # Escalate severity if multiple failures
        if lockout_entry.failure_count >= 3 and lockout_entry.severity != LockoutSeverity.CRITICAL:
            old_severity = lockout_entry.severity
            lockout_entry.severity = self._escalate_severity(lockout_entry.severity)
            
            # Recalculate duration
            lockout_entry.base_duration = self._calculate_lockout_duration(lockout_entry.severity)
            lockout_entry.half_life = lockout_entry.base_duration * 0.5
            lockout_entry.unlock_time = time.time() + lockout_entry.base_duration
            
            logger.info(f"Lockout {signature_hash[:8]} escalated: {old_severity.value} → {lockout_entry.severity.value}")
        
        return signature_hash
    
    def _determine_lockout_severity(self, pattern_data: Dict[str, Any], 
                                  failure_reason: str) -> LockoutSeverity:
        """Determine appropriate lockout severity based on failure context."""
        # Check for critical failure indicators
        if "collapse" in failure_reason.lower() or "critical" in failure_reason.lower():
            return LockoutSeverity.CRITICAL
        
        # Check pattern volatility
        volatility = pattern_data.get('volatility', 0.0)
        if volatility > 0.8:
            return LockoutSeverity.SEVERE
        elif volatility > 0.5:
            return LockoutSeverity.MODERATE
        
        # Check profit delta magnitude
        profit_delta = abs(pattern_data.get('profit_delta', 0.0))
        if profit_delta > 100:  # Large loss
            return LockoutSeverity.SEVERE
        elif profit_delta > 50:
            return LockoutSeverity.MODERATE
        
        return LockoutSeverity.MINOR
    
    def _escalate_severity(self, current_severity: LockoutSeverity) -> LockoutSeverity:
        """Escalate lockout severity for repeated failures."""
        escalation_map = {
            LockoutSeverity.MINOR: LockoutSeverity.MODERATE,
            LockoutSeverity.MODERATE: LockoutSeverity.SEVERE,
            LockoutSeverity.SEVERE: LockoutSeverity.CRITICAL,
            LockoutSeverity.CRITICAL: LockoutSeverity.CRITICAL  # Max level
        }
        return escalation_map[current_severity]
    
    def _calculate_lockout_duration(self, severity: LockoutSeverity) -> float:
        """Calculate lockout duration based on severity."""
        min_duration, max_duration = self.duration_ranges[severity]
        
        # Add some randomization to prevent predictable patterns
        duration = np.random.uniform(min_duration, max_duration)
        
        return duration
    
    def _check_pattern_evolution(self, signature_hash: str, 
                               current_pattern: Dict[str, Any]) -> bool:
        """Check if pattern has evolved significantly since lockout."""
        if signature_hash not in self.pattern_snapshots:
            return False
        
        lockout_entry = self.active_lockouts[signature_hash]
        current_time = time.time()
        
        # Only check evolution periodically
        if current_time - lockout_entry.last_pattern_check < 300:  # 5 minutes
            return False
        
        lockout_entry.last_pattern_check = current_time
        
        # Compare with original pattern
        original_pattern = lockout_entry.pattern_data
        evolution_score = self._calculate_pattern_evolution(original_pattern, current_pattern)
        
        # Store current pattern snapshot
        self.pattern_snapshots[signature_hash].append(current_pattern.copy())
        
        return evolution_score > lockout_entry.evolution_threshold
    
    def _calculate_pattern_evolution(self, original: Dict[str, Any], 
                                   current: Dict[str, Any]) -> float:
        """Calculate how much a pattern has evolved."""
        common_keys = set(original.keys()) & set(current.keys())
        
        if not common_keys:
            return 1.0  # Complete evolution
        
        differences = []
        
        for key in common_keys:
            orig_val = original[key]
            curr_val = current[key]
            
            if isinstance(orig_val, (int, float)) and isinstance(curr_val, (int, float)):
                if abs(orig_val) > 1e-6:
                    diff = abs(orig_val - curr_val) / abs(orig_val)
                else:
                    diff = abs(curr_val)
                differences.append(diff)
        
        return np.mean(differences) if differences else 0.0
    
    def _should_force_reevaluation(self, lockout_entry: LockoutEntry) -> bool:
        """Check if lockout should be forced for reevaluation."""
        current_time = time.time()
        
        # Check suppression count threshold
        if lockout_entry.suppression_count >= self.max_suppression_count:
            return True
        
        # Check time-based forced reevaluation
        time_since_creation = current_time - lockout_entry.creation_time
        if time_since_creation > self.forced_reevaluation_interval:
            return True
        
        return False
    
    def _trigger_forced_reevaluation(self, signature_hash: str):
        """Trigger forced reevaluation of locked pattern."""
        lockout_entry = self.active_lockouts[signature_hash]
        lockout_entry.status = LockoutStatus.FORCE_REEVALUATE
        lockout_entry.forced_reevaluations += 1
        
        # Move to expired (allowing reevaluation)
        self.expired_lockouts[signature_hash] = self.active_lockouts.pop(signature_hash)
        
        # Update metrics
        self.metrics.active_lockouts -= 1
        self.metrics.forced_reevaluations += 1
        
        logger.info(f"Forced reevaluation triggered for {signature_hash[:8]} "
                   f"(suppressions: {lockout_entry.suppression_count})")
    
    def _unlock_pattern_evolution(self, signature_hash: str):
        """Unlock pattern due to significant evolution."""
        lockout_entry = self.active_lockouts[signature_hash]
        lockout_entry.status = LockoutStatus.PATTERN_EVOLVED
        
        # Move to expired
        self.expired_lockouts[signature_hash] = self.active_lockouts.pop(signature_hash)
        
        # Update metrics
        self.metrics.active_lockouts -= 1
        self.metrics.pattern_evolutions += 1
        
        logger.info(f"Pattern evolution unlock: {signature_hash[:8]}")
    
    def _expire_lockout(self, signature_hash: str):
        """Move lockout from active to expired."""
        if signature_hash in self.active_lockouts:
            lockout_entry = self.active_lockouts[signature_hash]
            lockout_entry.status = LockoutStatus.EXPIRED
            
            # Move to expired
            self.expired_lockouts[signature_hash] = self.active_lockouts.pop(signature_hash)
            
            # Update metrics
            self.metrics.active_lockouts -= 1
            self.metrics.expired_lockouts += 1
            self.metrics.self_healed_lockouts += 1
            
            logger.debug(f"Lockout expired: {signature_hash[:8]}")
    
    def get_lockout_summary(self) -> Dict[str, Any]:
        """Get comprehensive lockout system summary."""
        # Calculate average lockout duration
        if self.lockout_history:
            durations = [entry.base_duration for entry in self.lockout_history]
            avg_duration = np.mean(durations)
        else:
            avg_duration = 0.0
        
        # Calculate effectiveness (expired vs total)
        total_lockouts = self.metrics.total_lockouts_created
        if total_lockouts > 0:
            effectiveness = self.metrics.self_healed_lockouts / total_lockouts
        else:
            effectiveness = 0.0
        
        return {
            "total_lockouts_created": self.metrics.total_lockouts_created,
            "active_lockouts": len(self.active_lockouts),
            "expired_lockouts": len(self.expired_lockouts),
            "self_healed_lockouts": self.metrics.self_healed_lockouts,
            "forced_reevaluations": self.metrics.forced_reevaluations,
            "pattern_evolutions": self.metrics.pattern_evolutions,
            "avg_lockout_duration_minutes": avg_duration / 60,
            "system_effectiveness": effectiveness,
            "active_lockout_weights": self.update_lockout_weights()
        }
    
    def cleanup_old_lockouts(self, max_age_hours: float = 48.0):
        """Clean up very old expired lockouts."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_lockouts = []
        for signature_hash, lockout_entry in self.expired_lockouts.items():
            if current_time - lockout_entry.creation_time > max_age_seconds:
                old_lockouts.append(signature_hash)
        
        for signature_hash in old_lockouts:
            del self.expired_lockouts[signature_hash]
            if signature_hash in self.pattern_snapshots:
                del self.pattern_snapshots[signature_hash]
        
        if old_lockouts:
            logger.info(f"Cleaned up {len(old_lockouts)} old expired lockouts")
    
    def reset_lockout_system(self):
        """Reset entire lockout system."""
        self.active_lockouts.clear()
        self.expired_lockouts.clear()
        self.lockout_history.clear()
        self.pattern_snapshots.clear()
        self.metrics = LockoutMetrics()
        logger.info("Lockout matrix system reset")

# Example usage and testing
if __name__ == "__main__":
    # Test enhanced lockout matrix
    lockout_matrix = EnhancedLockoutMatrix()
    
    # Create test pattern
    test_pattern = {
        "braid_signal": 0.3,
        "paradox_signal": 0.2,
        "profit_delta": -75.0,
        "volatility": 0.6
    }
    
    # Create lockout
    signature = lockout_matrix.create_lockout(test_pattern, "profit_collapse", LockoutSeverity.MODERATE)
    print(f"Created lockout: {signature}")
    
    # Check lockout status
    is_locked, reason = lockout_matrix.check_lockout_status(test_pattern)
    print(f"Lockout status: {is_locked}, reason: {reason}")
    
    # Test pattern evolution
    evolved_pattern = test_pattern.copy()
    evolved_pattern["braid_signal"] = 0.8  # Significant change
    evolved_pattern["profit_delta"] = 50.0
    
    is_locked_evolved, reason_evolved = lockout_matrix.check_lockout_status(evolved_pattern)
    print(f"Evolved pattern locked: {is_locked_evolved}, reason: {reason_evolved}")
    
    # Get system summary
    summary = lockout_matrix.get_lockout_summary()
    print(f"Lockout summary: {summary}") 