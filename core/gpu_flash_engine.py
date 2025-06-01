"""
GPU Flash Engine
===============

Manages GPU flash operations with ZPE risk awareness and phase drift analysis.
Integrates with cursor math for stability calculations and entropy shell classification.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import time
from .cursor_math_integration import CursorMath, PhaseShell

@dataclass
class FlashState:
    """Represents the state of a GPU flash operation"""
    timestamp: float
    z_score: float
    phase_angle: float
    entropy_class: str
    matrix_state: str
    is_safe: bool

class GPUFlashEngine:
    """Manages GPU flash operations with ZPE risk awareness"""
    
    def __init__(self):
        self.math_core = CursorMath()
        self.flash_history: List[FlashState] = []
        self.last_flash_time: Optional[float] = None
        self.cooldown_period: float = 0.1  # seconds
        
    def check_flash_permission(self, z_score: float, phase_angle: float) -> Tuple[bool, str]:
        """
        Check if flash operation is permitted based on ZPE risk and phase drift.
        
        Args:
            z_score: Current Z-score of entropy
            phase_angle: Current phase angle
            
        Returns:
            (is_permitted, reason) tuple
        """
        # Check cooldown period
        current_time = time.time()
        if (self.last_flash_time is not None and 
            current_time - self.last_flash_time < self.cooldown_period):
            return False, "cooldown_period"
            
        # Classify entropy shell
        entropy_class = self.math_core.classify_entropy_shell(z_score)
        if entropy_class == 'critical_bloom':
            return False, "critical_entropy"
            
        # Check phase shell
        phase_shell = self.math_core.classify_phase_shell(phase_angle)
        if phase_shell.shell_type == 'symmetry':
            return False, "symmetry_zone"
            
        # Compute matrix stability
        matrix_state = self.math_core.compute_matrix_stability(
            binding_energy=7.5,  # Default value, should be computed from actual data
            phase_stability=phase_shell.stability,
            entropy_class=entropy_class
        )
        
        is_safe = matrix_state == 'matrix_safe'
        
        # Record flash state
        self.flash_history.append(FlashState(
            timestamp=current_time,
            z_score=z_score,
            phase_angle=phase_angle,
            entropy_class=entropy_class,
            matrix_state=matrix_state,
            is_safe=is_safe
        ))
        
        if is_safe:
            self.last_flash_time = current_time
            
        return is_safe, matrix_state
    
    def get_flash_history(self, limit: Optional[int] = None) -> List[FlashState]:
        """Get recent flash history"""
        if limit is None:
            return self.flash_history
        return self.flash_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear flash history"""
        self.flash_history.clear()
        self.last_flash_time = None
    
    def get_entropy_stats(self) -> Dict[str, float]:
        """Get statistics about entropy classifications"""
        if not self.flash_history:
            return {}
            
        total = len(self.flash_history)
        stats = {
            'critical_bloom': 0.0,
            'unstable': 0.0,
            'stable': 0.0
        }
        
        for state in self.flash_history:
            stats[state.entropy_class] += 1
            
        return {k: v/total for k, v in stats.items()}
    
    def get_phase_stats(self) -> Dict[str, float]:
        """Get statistics about phase shell classifications"""
        if not self.flash_history:
            return {}
            
        total = len(self.flash_history)
        stats = {
            'symmetry': 0.0,
            'drift_positive': 0.0,
            'drift_negative': 0.0
        }
        
        for state in self.flash_history:
            shell = self.math_core.classify_phase_shell(state.phase_angle)
            stats[shell.shell_type] += 1
            
        return {k: v/total for k, v in stats.items()}

# Example usage
if __name__ == "__main__":
    flash_engine = GPUFlashEngine()
    
    # Test flash permission
    is_permitted, reason = flash_engine.check_flash_permission(
        z_score=1.5,
        phase_angle=np.pi + 0.2
    )
    print(f"Flash permitted: {is_permitted}, reason: {reason}")
    
    # Get statistics
    entropy_stats = flash_engine.get_entropy_stats()
    phase_stats = flash_engine.get_phase_stats()
    print(f"Entropy stats: {entropy_stats}")
    print(f"Phase stats: {phase_stats}") 