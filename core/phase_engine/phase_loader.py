"""
Phase Configuration Loader
=======================

Handles loading and validation of phase configurations from YAML files.
Provides type-safe access to phase region definitions and metrics.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PhaseRegion:
    """Represents a phase region with its metric ranges"""
    name: str
    profit_trend_range: List[float]
    stability_range: List[float]
    memory_coherence_range: List[float]
    paradox_pressure_range: List[float]
    entropy_rate_range: List[float]
    thermal_state_range: List[float]
    bit_depth_range: List[int]
    trust_score_range: List[float]
    
    def is_in_range(self, metric_name: str, value: float) -> bool:
        """Check if a metric value falls within this phase's range"""
        range_name = f"{metric_name}_range"
        if not hasattr(self, range_name):
            raise ValueError(f"Unknown metric: {metric_name}")
            
        min_val, max_val = getattr(self, range_name)
        return min_val <= value <= max_val
        
    def get_metric_ranges(self) -> Dict[str, List[float]]:
        """Get all metric ranges for this phase"""
        return {
            "profit_trend": self.profit_trend_range,
            "stability": self.stability_range,
            "memory_coherence": self.memory_coherence_range,
            "paradox_pressure": self.paradox_pressure_range,
            "entropy_rate": self.entropy_rate_range,
            "thermal_state": self.thermal_state_range,
            "bit_depth": self.bit_depth_range,
            "trust_score": self.trust_score_range
        }

class PhaseConfigLoader:
    """Loads and validates phase configurations"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the phase config loader"""
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "schema.yaml"
        self.phase_regions: Dict[str, PhaseRegion] = {}
        self._load_config()
        
    def _load_config(self) -> None:
        """Load phase configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Load phase regions
            for name, params in config.get("phases", {}).items():
                self.phase_regions[name] = PhaseRegion(
                    name=name,
                    **params
                )
                
            logger.info(f"Loaded {len(self.phase_regions)} phase regions")
            
        except Exception as e:
            logger.error(f"Error loading phase config: {e}")
            raise
            
    def get_phase_region(self, name: str) -> Optional[PhaseRegion]:
        """Get a phase region by name"""
        return self.phase_regions.get(name)
        
    def get_all_phase_regions(self) -> Dict[str, PhaseRegion]:
        """Get all phase regions"""
        return self.phase_regions.copy()
        
    def classify_phase(self, metrics: Dict[str, float]) -> str:
        """Classify metrics into a phase"""
        best_match = None
        best_score = -1
        
        for name, region in self.phase_regions.items():
            score = self._calculate_phase_score(region, metrics)
            if score > best_score:
                best_score = score
                best_match = name
                
        return best_match
        
    def _calculate_phase_score(self, region: PhaseRegion, metrics: Dict[str, float]) -> float:
        """Calculate how well metrics match a phase region"""
        score = 0.0
        total_metrics = 0
        
        for metric_name, value in metrics.items():
            if metric_name.endswith("_range"):
                continue
                
            if region.is_in_range(metric_name, value):
                score += 1
            total_metrics += 1
            
        return score / total_metrics if total_metrics > 0 else 0.0
        
    def validate_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Validate metrics against phase definitions"""
        errors = []
        
        # Check required metrics
        required_metrics = {
            "profit_trend", "stability", "memory_coherence",
            "paradox_pressure", "entropy_rate", "thermal_state",
            "bit_depth", "trust_score"
        }
        
        missing_metrics = required_metrics - set(metrics.keys())
        if missing_metrics:
            errors.append(f"Missing required metrics: {missing_metrics}")
            
        # Check metric ranges
        for metric_name, value in metrics.items():
            if metric_name.endswith("_range"):
                continue
                
            valid_ranges = False
            for region in self.phase_regions.values():
                if region.is_in_range(metric_name, value):
                    valid_ranges = True
                    break
                    
            if not valid_ranges:
                errors.append(f"Metric {metric_name}={value} outside all phase ranges")
                
        return errors
        
    def get_metric_ranges(self, phase_name: str) -> Dict[str, List[float]]:
        """Get metric ranges for a specific phase"""
        region = self.get_phase_region(phase_name)
        if not region:
            raise ValueError(f"Unknown phase: {phase_name}")
            
        return region.get_metric_ranges()
        
    def get_phase_transition_rules(self) -> Dict[str, List[str]]:
        """Get allowed phase transitions"""
        return {
            "STABLE": ["SMART_MONEY", "UNSTABLE"],
            "SMART_MONEY": ["STABLE", "OVERLOADED"],
            "UNSTABLE": ["STABLE", "OVERLOADED"],
            "OVERLOADED": ["UNSTABLE", "SMART_MONEY"]
        }
        
    def is_valid_transition(self, from_phase: str, to_phase: str) -> bool:
        """Check if a phase transition is valid"""
        rules = self.get_phase_transition_rules()
        return to_phase in rules.get(from_phase, []) 