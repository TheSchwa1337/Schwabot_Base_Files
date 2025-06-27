"""
Tick Resonance Engine - Harmony Score Calculator.

This module computes harmony scores(𝝁) that measure how well tick timing
aligns with expected phase gates (4-bit, 8-bit, 42-bit). The harmony score
feeds into the entropy-weighted entry score calculation.

Mathematical Foundation:
𝝁 = exp(-mean(|tick_i - phi_target|)²)

Where:
- tick_i: Time deltas between consecutive ticks
- phi_target: Target phase timing for current bit depth
- Result in [0, 1] where 1 = perfect harmony

Windows CLI compatible with ASCII fallback for special characters.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import Windows CLI compatibility
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, safe_format_error, log_safe
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
        
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
        
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)

logger = logging.getLogger(__name__)

# Phase target timings (in seconds)
PHASE_TARGETS = {
    4: 0.25,   # 4-bit: 250ms target
    8: 0.125,  # 8-bit: 125ms target
    42: 0.024  # 42-bit: ~24ms target (high frequency)
}

# Harmony calculation parameters
HARMONY_WINDOW_SIZE = 20  # Number of recent ticks to analyze
MIN_TICKS_REQUIRED = 3    # Minimum ticks needed for calculation


def compute_harmony_vector(
    tick_deltas: np.ndarray,
    target_phase: float,
    window_size: int = HARMONY_WINDOW_SIZE
) -> float:
    """
    Compute harmony score for tick timing alignment.
    
    Parameters
    ----------
    tick_deltas : np.ndarray
        Array of time deltas between consecutive ticks (in seconds)
    target_phase : float
        Target timing for current phase (in seconds)
    window_size : int, optional
        Number of recent deltas to analyze
        
    Returns
    -------
    float
        Harmony score in [0, 1] where 1 = perfect alignment
    """
    try:
        if len(tick_deltas) < MIN_TICKS_REQUIRED:
            logger.debug(f"Insufficient ticks for harmony: {len(tick_deltas)}")
            return 0.0
        
        # Use most recent window
        recent_deltas = tick_deltas[-window_size:]
        
        # Calculate absolute deviations from target
        deviations = np.abs(recent_deltas - target_phase)
        
        # Compute mean squared deviation
        mean_sq_deviation = np.mean(deviations ** 2)
        
        # Calculate harmony score using exponential decay
        harmony_score = np.exp(-mean_sq_deviation)
        
        # Ensure result is in [0, 1]
        harmony_score = np.clip(harmony_score, 0.0, 1.0)
        
        return float(harmony_score)
        
    except Exception as e:
        error_msg = safe_format_error(e, 'harmony_calculation')
        logger.error(f"Harmony calculation failed: {error_msg}")
        return 0.0


def calculate_phase_alignment(
    tick_deltas: np.ndarray,
    bit_depth: int
) -> Dict[str, float]:
    """
    Calculate phase alignment metrics for given bit depth.
    
    Parameters
    ----------
    tick_deltas : np.ndarray
        Array of time deltas between consecutive ticks
    bit_depth : int
        Target bit depth (4, 8, or 42)
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing alignment metrics
    """
    try:
        if bit_depth not in PHASE_TARGETS:
            logger.warning(f"Unsupported bit depth: {bit_depth}")
            return {'harmony': 0.0, 'alignment': 0.0, 'stability': 0.0}
        
        target_phase = PHASE_TARGETS[bit_depth]
        
        # Calculate harmony score
        harmony_score = compute_harmony_vector(tick_deltas, target_phase)
        
        # Calculate alignment precision
        if len(tick_deltas) >= MIN_TICKS_REQUIRED:
            recent_deltas = tick_deltas[-HARMONY_WINDOW_SIZE:]
            alignment_precision = 1.0 - np.std(recent_deltas) / target_phase
            alignment_precision = np.clip(alignment_precision, 0.0, 1.0)
        else:
            alignment_precision = 0.0
        
        # Calculate stability (consistency over time)
        if len(tick_deltas) >= 10:
            stability = 1.0 - np.std(tick_deltas[-10:]) / np.mean(tick_deltas[-10:])
            stability = np.clip(stability, 0.0, 1.0)
        else:
            stability = 0.0
        
        return {
            'harmony': harmony_score,
            'alignment': alignment_precision,
            'stability': stability
        }
        
    except Exception as e:
        error_msg = safe_format_error(e, 'phase_alignment')
        logger.error(f"Phase alignment calculation failed: {error_msg}")
        return {'harmony': 0.0, 'alignment': 0.0, 'stability': 0.0}


def get_optimal_phase(tick_deltas: np.ndarray) -> Tuple[int, float]:
    """
    Determine optimal bit depth based on tick timing patterns.
    
    Parameters
    ----------
    tick_deltas : np.ndarray
        Array of time deltas between consecutive ticks
        
    Returns
    -------
    Tuple[int, float]
        Optimal bit depth and corresponding harmony score
    """
    try:
        if len(tick_deltas) < MIN_TICKS_REQUIRED:
            return 8, 0.0  # Default to 8-bit if insufficient data
        
        best_harmony = 0.0
        optimal_depth = 8
        
        # Test each bit depth
        for bit_depth, target_phase in PHASE_TARGETS.items():
            harmony_score = compute_harmony_vector(tick_deltas, target_phase)
            
            if harmony_score > best_harmony:
                best_harmony = harmony_score
                optimal_depth = bit_depth
        
        return optimal_depth, best_harmony
        
    except Exception as e:
        error_msg = safe_format_error(e, 'optimal_phase')
        logger.error(f"Optimal phase calculation failed: {error_msg}")
        return 8, 0.0


class TickResonanceEngine:
    """
    Tick resonance engine for real-time harmony score calculation.
    
    Maintains tick history and computes harmony scores for different
    bit depths in real-time.
    """
    
    def __init__(self, default_bit_depth: int = 8):
        """Initialize tick resonance engine."""
        self.default_bit_depth = default_bit_depth
        self.tick_timestamps: List[float] = []
        self.tick_deltas: List[float] = []
        self.harmony_history: Dict[int, List[float]] = {
            4: [],
            8: [],
            42: []
        }
        self.last_update_time = 0.0
        
        safe_print("🎵 Tick Resonance Engine initialized")
    
    def update_tick(self, timestamp: float) -> None:
        """
        Update engine with new tick timestamp.
        
        Parameters
        ----------
        timestamp : float
            Current tick timestamp
        """
        try:
            current_time = time.time()
            
            # Add timestamp
            self.tick_timestamps.append(timestamp)
            
            # Calculate delta if we have previous ticks
            if len(self.tick_timestamps) > 1:
                delta = timestamp - self.tick_timestamps[-2]
                self.tick_deltas.append(delta)
            
            # Maintain history size
            max_history = HARMONY_WINDOW_SIZE * 2
            if len(self.tick_timestamps) > max_history:
                self.tick_timestamps.pop(0)
            if len(self.tick_deltas) > max_history:
                self.tick_deltas.pop(0)
            
            # Update harmony scores
            self._update_harmony_scores()
            
            self.last_update_time = current_time
            
        except Exception as e:
            error_msg = safe_format_error(e, 'tick_update')
            logger.error(f"Tick update failed: {error_msg}")
    
    def _update_harmony_scores(self) -> None:
        """Update harmony scores for all bit depths."""
        try:
            if len(self.tick_deltas) < MIN_TICKS_REQUIRED:
                return
            
            tick_deltas_array = np.array(self.tick_deltas)
            
            for bit_depth in PHASE_TARGETS.keys():
                harmony_score = compute_harmony_vector(
                    tick_deltas_array, 
                    PHASE_TARGETS[bit_depth]
                )
                
                self.harmony_history[bit_depth].append(harmony_score)
                
                # Maintain history size
                if len(self.harmony_history[bit_depth]) > HARMONY_WINDOW_SIZE:
                    self.harmony_history[bit_depth].pop(0)
                    
        except Exception as e:
            error_msg = safe_format_error(e, 'harmony_update')
            logger.error(f"Harmony score update failed: {error_msg}")
    
    def get_current_harmony(self, bit_depth: Optional[int] = None) -> float:
        """
        Get current harmony score for specified bit depth.
        
        Parameters
        ----------
        bit_depth : int, optional
            Target bit depth. If None, uses default.
            
        Returns
        -------
        float
            Current harmony score
        """
        try:
            if bit_depth is None:
                bit_depth = self.default_bit_depth
            
            if bit_depth not in self.harmony_history:
                return 0.0
            
            history = self.harmony_history[bit_depth]
            if not history:
                return 0.0
            
            return history[-1]
            
        except Exception as e:
            error_msg = safe_format_error(e, 'harmony_retrieval')
            logger.error(f"Harmony score retrieval failed: {error_msg}")
            return 0.0
    
    def get_optimal_bit_depth(self) -> Tuple[int, float]:
        """
        Get optimal bit depth based on current tick patterns.
        
        Returns
        -------
        Tuple[int, float]
            Optimal bit depth and corresponding harmony score
        """
        try:
            if len(self.tick_deltas) < MIN_TICKS_REQUIRED:
                return self.default_bit_depth, 0.0
            
            tick_deltas_array = np.array(self.tick_deltas)
            return get_optimal_phase(tick_deltas_array)
            
        except Exception as e:
            error_msg = safe_format_error(e, 'optimal_depth')
            logger.error(f"Optimal bit depth calculation failed: {error_msg}")
            return self.default_bit_depth, 0.0
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics information.
        
        Returns
        -------
        Dict[str, Any]
            Diagnostic information
        """
        try:
            diagnostics = {
                'tick_count': len(self.tick_timestamps),
                'delta_count': len(self.tick_deltas),
                'last_update': self.last_update_time,
                'harmony_scores': {},
                'optimal_depth': self.get_optimal_bit_depth()[0],
                'phase_alignments': {}
            }
            
            # Add harmony scores for each bit depth
            for bit_depth in PHASE_TARGETS.keys():
                diagnostics['harmony_scores'][f'{bit_depth}_bit'] = self.get_current_harmony(bit_depth)
            
            # Add phase alignments if we have sufficient data
            if len(self.tick_deltas) >= MIN_TICKS_REQUIRED:
                tick_deltas_array = np.array(self.tick_deltas)
                for bit_depth in PHASE_TARGETS.keys():
                    diagnostics['phase_alignments'][f'{bit_depth}_bit'] = calculate_phase_alignment(
                        tick_deltas_array, bit_depth
                    )
            
            return diagnostics
            
        except Exception as e:
            error_msg = safe_format_error(e, 'diagnostics')
            logger.error(f"Diagnostics calculation failed: {error_msg}")
            return {'error': error_msg}
    
    def reset(self) -> None:
        """Reset engine state."""
        try:
            self.tick_timestamps.clear()
            self.tick_deltas.clear()
            for bit_depth in self.harmony_history:
                self.harmony_history[bit_depth].clear()
            self.last_update_time = 0.0
            
            safe_print("🔄 Tick Resonance Engine reset")
            
        except Exception as e:
            error_msg = safe_format_error(e, 'engine_reset')
            logger.error(f"Engine reset failed: {error_msg}")


def validate_tick_deltas(tick_deltas: np.ndarray) -> bool:
    """
    Validate tick delta array for processing.
    
    Parameters
    ----------
    tick_deltas : np.ndarray
        Array of tick deltas to validate
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    try:
        if len(tick_deltas) == 0:
            return False
        
        # Check for negative deltas
        if np.any(tick_deltas < 0):
            return False
        
        # Check for extreme values (likely errors)
        if np.any(tick_deltas > 10.0):  # More than 10 seconds
            return False
        
        # Check for zero deltas (duplicate timestamps)
        if np.any(tick_deltas == 0):
            return False
        
        return True
        
    except Exception as e:
        error_msg = safe_format_error(e, 'delta_validation')
        logger.error(f"Tick delta validation failed: {error_msg}")
        return False


# Global engine instance
_tick_resonance_engine: Optional[TickResonanceEngine] = None


def get_tick_resonance_engine() -> TickResonanceEngine:
    """Get or create the global tick resonance engine instance."""
    global _tick_resonance_engine
    if _tick_resonance_engine is None:
        _tick_resonance_engine = TickResonanceEngine()
    return _tick_resonance_engine


def main():
    """Test the tick resonance engine."""
    try:
        # Initialize engine
        engine = get_tick_resonance_engine()
        
        # Simulate some ticks
        base_time = time.time()
        for i in range(10):
            tick_time = base_time + i * 0.125  # 8-bit timing
            engine.update_tick(tick_time)
            time.sleep(0.01)  # Small delay
        
        # Get diagnostics
        diagnostics = engine.get_diagnostics()
        safe_print(f"📊 Diagnostics: {diagnostics}")
        
        # Get optimal bit depth
        optimal_depth, harmony = engine.get_optimal_bit_depth()
        safe_print(f"🎯 Optimal bit depth: {optimal_depth}, harmony: {harmony:.3f}")
        
        safe_print("🎉 Tick resonance engine test completed successfully")
        
    except Exception as e:
        safe_print(f"❌ Test failed: {safe_format_error(e, 'main_test')}")


if __name__ == "__main__":
    main()


