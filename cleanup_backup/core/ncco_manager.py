#!/usr/bin/env python3
"""
NCCO Manager - Core Neural Circuit Control Object Management System
==================================================================

This module provides comprehensive NCCO (Neural Circuit Control Object) management
for the Schwabot system. It handles NCCO generation, validation, storage, and
integration with the main trading pipeline.

Core Functionality:
- NCCO generation and validation
- NCCO storage and retrieval
- NCCO integration with main pipeline
- NCCO performance tracking
- NCCO lifecycle management
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class NCCOState:
    """NCCO state information."""
    ncco_id: str
    generation_time: datetime
    state_hash: str
    performance_score: float
    activation_count: int
    last_activation: datetime
    is_active: bool
    metadata: Dict[str, Any]


@dataclass
class NCCOGenerationResult:
    """Result of NCCO generation operation."""
    success: bool
    ncco_id: str
    generation_time: datetime
    confidence_score: float
    state_hash: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class NCCOManager:
    """Core NCCO management system for Schwabot."""
    
    def __init__(self):
        """Initialize the NCCO manager."""
        self.ncco_states: Dict[str, NCCOState] = {}
        self.generation_history: List[NCCOGenerationResult] = []
        self.active_nccos: List[str] = []
        self.performance_cache: Dict[str, float] = {}
        self.generation_count = 0
        
        logger.info("NCCO Manager initialized")
    
    def generate_ncco(self, input_data: Dict[str, Any], ncco_type: str = "standard") -> NCCOGenerationResult:
        """Generate a new NCCO based on input data."""
        try:
            # Generate unique NCCO ID
            ncco_id = f"ncco_{self.generation_count}_{int(time.time())}"
            
            # Create NCCO state
            state_data = {
                "input_data": input_data,
                "ncco_type": ncco_type,
                "generation_parameters": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0",
                    "complexity": self._calculate_complexity(input_data)
                }
            }
            
            # Generate state hash
            state_hash = self._generate_state_hash(state_data)
            
            # Create NCCO state
            ncco_state = NCCOState(
                ncco_id=ncco_id,
                generation_time=datetime.now(),
                state_hash=state_hash,
                performance_score=0.0,
                activation_count=0,
                last_activation=datetime.now(),
                is_active=True,
                metadata=state_data
            )
            
            # Store NCCO state
            self.ncco_states[ncco_id] = ncco_state
            self.active_nccos.append(ncco_id)
            
            result = NCCOGenerationResult(
                success=True,
                ncco_id=ncco_id,
                generation_time=datetime.now(),
                confidence_score=1.0,
                state_hash=state_hash,
                metadata={"ncco_type": ncco_type, "complexity": state_data["generation_parameters"]["complexity"]}
            )
            
            self.generation_history.append(result)
            self.generation_count += 1
            
            logger.info(f"NCCO generated successfully: {ncco_id}")
            return result
            
        except Exception as e:
            logger.error(f"NCCO generation error: {e}")
            return NCCOGenerationResult(
                success=False,
                ncco_id="",
                generation_time=datetime.now(),
                confidence_score=0.0,
                state_hash="",
                error_message=str(e)
            )
    
    def _calculate_complexity(self, input_data: Dict[str, Any]) -> float:
        """Calculate complexity score for input data."""
        try:
            # Simple complexity calculation based on data structure
            data_size = len(str(input_data))
            key_count = len(input_data.keys())
            nested_depth = self._calculate_nested_depth(input_data)
            
            complexity = (data_size * 0.1 + key_count * 0.2 + nested_depth * 0.3) / 100
            return min(complexity, 1.0)
            
        except Exception as e:
            logger.error(f"Complexity calculation error: {e}")
            return 0.5
    
    def _calculate_nested_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate nested depth of data structure."""
        if not isinstance(obj, (dict, list)):
            return current_depth
        
        max_depth = current_depth
        if isinstance(obj, dict):
            for value in obj.values():
                max_depth = max(max_depth, self._calculate_nested_depth(value, current_depth + 1))
        elif isinstance(obj, list):
            for item in obj:
                max_depth = max(max_depth, self._calculate_nested_depth(item, current_depth + 1))
        
        return max_depth
    
    def _generate_state_hash(self, state_data: Dict[str, Any]) -> str:
        """Generate hash for state data."""
        try:
            state_string = json.dumps(state_data, sort_keys=True)
            return hashlib.sha256(state_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"State hash generation error: {e}")
            return ""
    
    def activate_ncco(self, ncco_id: str, activation_data: Dict[str, Any]) -> bool:
        """Activate an NCCO with new data."""
        try:
            if ncco_id not in self.ncco_states:
                logger.warning(f"NCCO not found: {ncco_id}")
                return False
            
            ncco_state = self.ncco_states[ncco_id]
            
            # Update activation count and time
            ncco_state.activation_count += 1
            ncco_state.last_activation = datetime.now()
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(activation_data)
            ncco_state.performance_score = performance_score
            
            # Update performance cache
            self.performance_cache[ncco_id] = performance_score
            
            logger.debug(f"NCCO activated: {ncco_id} (score: {performance_score:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"NCCO activation error: {e}")
            return False
    
    def _calculate_performance_score(self, activation_data: Dict[str, Any]) -> float:
        """Calculate performance score for activation data."""
        try:
            # Simple performance scoring based on data quality
            data_completeness = len(activation_data.keys()) / 10  # Normalize to 0-1
            data_consistency = 0.8  # Placeholder for consistency check
            data_freshness = 0.9  # Placeholder for freshness check
            
            performance_score = (data_completeness + data_consistency + data_freshness) / 3
            return min(performance_score, 1.0)
            
        except Exception as e:
            logger.error(f"Performance score calculation error: {e}")
            return 0.5
    
    def deactivate_ncco(self, ncco_id: str) -> bool:
        """Deactivate an NCCO."""
        try:
            if ncco_id not in self.ncco_states:
                logger.warning(f"NCCO not found for deactivation: {ncco_id}")
                return False
            
            ncco_state = self.ncco_states[ncco_id]
            ncco_state.is_active = False
            
            if ncco_id in self.active_nccos:
                self.active_nccos.remove(ncco_id)
            
            logger.info(f"NCCO deactivated: {ncco_id}")
            return True
            
        except Exception as e:
            logger.error(f"NCCO deactivation error: {e}")
            return False
    
    def get_ncco_state(self, ncco_id: str) -> Optional[NCCOState]:
        """Get NCCO state by ID."""
        return self.ncco_states.get(ncco_id)
    
    def get_active_nccos(self) -> List[NCCOState]:
        """Get all active NCCOs."""
        return [self.ncco_states[ncco_id] for ncco_id in self.active_nccos 
                if ncco_id in self.ncco_states]
    
    def get_top_performing_nccos(self, limit: int = 10) -> List[NCCOState]:
        """Get top performing NCCOs."""
        try:
            # Sort by performance score
            sorted_nccos = sorted(
                self.ncco_states.values(),
                key=lambda x: x.performance_score,
                reverse=True
            )
            
            return sorted_nccos[:limit]
            
        except Exception as e:
            logger.error(f"Error getting top performing NCCOs: {e}")
            return []
    
    def validate_ncco_integrity(self, ncco_id: str) -> bool:
        """Validate NCCO integrity."""
        try:
            if ncco_id not in self.ncco_states:
                return False
            
            ncco_state = self.ncco_states[ncco_id]
            
            # Recalculate state hash
            current_hash = self._generate_state_hash(ncco_state.metadata)
            
            # Compare with stored hash
            integrity_valid = current_hash == ncco_state.state_hash
            
            if not integrity_valid:
                logger.warning(f"NCCO integrity check failed: {ncco_id}")
            
            return integrity_valid
            
        except Exception as e:
            logger.error(f"NCCO integrity validation error: {e}")
            return False
    
    def cleanup_inactive_nccos(self, max_age_hours: int = 24) -> int:
        """Clean up inactive NCCOs older than specified age."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time.replace(hour=current_time.hour - max_age_hours)
            
            nccos_to_remove = []
            
            for ncco_id, ncco_state in self.ncco_states.items():
                if (not ncco_state.is_active and 
                    ncco_state.last_activation < cutoff_time):
                    nccos_to_remove.append(ncco_id)
            
            # Remove inactive NCCOs
            for ncco_id in nccos_to_remove:
                del self.ncco_states[ncco_id]
                if ncco_id in self.performance_cache:
                    del self.performance_cache[ncco_id]
            
            logger.info(f"Cleaned up {len(nccos_to_remove)} inactive NCCOs")
            return len(nccos_to_remove)
            
        except Exception as e:
            logger.error(f"NCCO cleanup error: {e}")
            return 0
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get NCCO manager statistics."""
        total_nccos = len(self.ncco_states)
        active_nccos = len(self.active_nccos)
        total_generations = len(self.generation_history)
        successful_generations = sum(1 for result in self.generation_history if result.success)
        
        avg_performance = 0.0
        if self.performance_cache:
            avg_performance = sum(self.performance_cache.values()) / len(self.performance_cache)
        
        return {
            "total_nccos": total_nccos,
            "active_nccos": active_nccos,
            "inactive_nccos": total_nccos - active_nccos,
            "total_generations": total_generations,
            "successful_generations": successful_generations,
            "generation_success_rate": successful_generations / total_generations if total_generations > 0 else 0.0,
            "average_performance": avg_performance,
            "performance_cache_size": len(self.performance_cache)
        }


def main() -> None:
    """Main function for testing NCCO manager."""
    manager = NCCOManager()
    
    # Test NCCO generation
    test_data = {"market_data": "test", "parameters": {"param1": 1.0}}
    result = manager.generate_ncco(test_data, "test_type")
    print(f"NCCO generation result: {result.success}")
    
    # Test NCCO activation
    if result.success:
        activation_success = manager.activate_ncco(result.ncco_id, {"test": "data"})
        print(f"NCCO activation result: {activation_success}")
    
    # Get statistics
    stats = manager.get_manager_statistics()
    print(f"Manager statistics: {stats}")


if __name__ == "__main__":
    main()
