# -*- coding: utf-8 -*-
"""
Lantern Core - Connective and Holistic System

Implements connective and holistic system that relays input states into 2-bit 
logic gates with connection matrix tracking and state history management.

Mathematical Foundation:
- Bit state extraction: bit_state = (hash_int & 0b11)
- Connection matrix: C[i,j] = connection_strength between bit gates i and j
- State processing: processed_state = bit_gate.process(input_state)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import logging
import time

import numpy as np

from core.unified_math_system import unified_math

# Configure logging
logger = logging.getLogger(__name__)


class BitGateType(Enum):
    """Bit gate types for 2-bit logic processing"""
    NULL_VECTOR = "NULL_VECTOR"
    LOW_TIER = "LOW_TIER"
    MID_TIER = "MID_TIER"
    PEAK_TIER = "PEAK_TIER"


@dataclass
class BitGate:
    """Individual 2-bit logic gate"""
    gate_type: BitGateType
    emoji_symbol: str
    processing_history: List[Dict[str, Any]]
    
    def __post_init__(self):
        """Initialize processing history if not provided"""
        if not self.processing_history:
            self.processing_history = []
    
    def process(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process input state through bit gate"""
        try:
            processed_state = input_state.copy()
            
            # Add bit gate metadata
            processed_state["bit_gate_type"] = self.gate_type.value
            processed_state["bit_gate_emoji"] = self.emoji_symbol
            processed_state["processing_timestamp"] = time.time()
            
            # Apply gate-specific processing
            if self.gate_type == BitGateType.NULL_VECTOR:
                processed_state = self._process_null_vector(processed_state)
            elif self.gate_type == BitGateType.LOW_TIER:
                processed_state = self._process_low_tier(processed_state)
            elif self.gate_type == BitGateType.MID_TIER:
                processed_state = self._process_mid_tier(processed_state)
            elif self.gate_type == BitGateType.PEAK_TIER:
                processed_state = self._process_peak_tier(processed_state)
            
            # Store in processing history
            self.processing_history.append(processed_state)
            
            # Limit history size
            if len(self.processing_history) > 1000:
                self.processing_history = self.processing_history[-1000:]
            
            return processed_state
            
        except Exception as e:
            logger.error(f"Failed to process state through bit gate {self.gate_type.value}: {e}")
            return input_state
    
    def _process_null_vector(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process through NULL_VECTOR gate (reset/idle state)"""
        # Reset or idle processing
        data["null_vector_processed"] = True
        data["processing_intensity"] = 0.0
        data["state_energy"] = 0.1  # Minimal energy for null state
        
        # Clear any active processing flags
        for key in list(data.keys()):
            if key.endswith("_active") and isinstance(data[key], bool):
                data[key] = False
        
        return data
    
    def _process_low_tier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process through LOW_TIER gate (micro-profit flag)"""
        # Low-tier processing for micro-profit signals
        data["low_tier_processed"] = True
        data["processing_intensity"] = 0.3
        data["state_energy"] = 0.5
        data["profit_potential"] = data.get("profit_potential", 0.0) * 0.5
        
        # Add low-tier specific flags
        data["micro_profit_flag"] = True
        data["conservative_mode"] = True
        
        return data
    
    def _process_mid_tier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process through MID_TIER gate (momentum logic)"""
        # Mid-tier processing for momentum and trend analysis
        data["mid_tier_processed"] = True
        data["processing_intensity"] = 0.7
        data["state_energy"] = 0.8
        data["profit_potential"] = data.get("profit_potential", 0.0) * 1.2
        
        # Add mid-tier specific flags
        data["momentum_analysis"] = True
        data["trend_tracking"] = True
        data["balanced_mode"] = True
        
        return data
    
    def _process_peak_tier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process through PEAK_TIER gate (max flip/lantern overlay)"""
        # Peak-tier processing for maximum profit potential
        data["peak_tier_processed"] = True
        data["processing_intensity"] = 1.0
        data["state_energy"] = 1.0
        data["profit_potential"] = data.get("profit_potential", 0.0) * 2.0
        
        # Add peak-tier specific flags
        data["max_profit_mode"] = True
        data["lantern_overlay"] = True
        data["aggressive_mode"] = True
        
        return data
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics for this bit gate"""
        try:
            if not self.processing_history:
                return {"total_processed": 0, "average_energy": 0.0}
            
            total_processed = len(self.processing_history)
            energies = [state.get("state_energy", 0.0) for state in self.processing_history]
            average_energy = sum(energies) / len(energies) if energies else 0.0
            
            return {
                "total_processed": total_processed,
                "average_energy": average_energy,
                "gate_type": self.gate_type.value,
                "emoji_symbol": self.emoji_symbol
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {"total_processed": 0, "average_energy": 0.0}


class LanternCore:
    """Connective and holistic system that relays into 2-bit logic gates"""
    
    def __init__(self):
        # Initialize bit gates
        self.bit_gates = {
            "00": BitGate(BitGateType.NULL_VECTOR, "⚫", []),
            "01": BitGate(BitGateType.LOW_TIER, "🟢", []),
            "10": BitGate(BitGateType.MID_TIER, "🟡", []),
            "11": BitGate(BitGateType.PEAK_TIER, "🔴", [])
        }
        
        # Connection matrix (4x4 for 4 bit gates)
        self.connection_matrix = np.zeros((4, 4))
        self.state_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_states_processed = 0
        self.bit_state_distribution = {"00": 0, "01": 0, "10": 0, "11": 0}
        self.average_processing_time = 0.0
    
    def relay_to_bit_gates(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        """Relay input state to appropriate 2-bit logic gates"""
        start_time = time.time()
        
        try:
            # Extract 2-bit state from input
            bit_state = self._extract_bit_state(input_state)
            
            # Get corresponding bit gate
            bit_gate = self.bit_gates[bit_state]
            
            # Process through bit gate
            processed_state = bit_gate.process(input_state)
            
            # Update connection matrix
            self._update_connection_matrix(bit_state, processed_state)
            
            # Store in state history
            self.state_history.append(processed_state)
            
            # Update performance tracking
            processing_time = time.time() - start_time
            self._update_performance_metrics(bit_state, processing_time)
            
            # Limit state history size
            if len(self.state_history) > 10000:
                self.state_history = self.state_history[-10000:]
            
            logger.debug(f"Processed state through bit gate {bit_state} in {processing_time:.4f}s")
            return processed_state
            
        except Exception as e:
            logger.error(f"Failed to relay state to bit gates: {e}")
            return input_state
    
    def _extract_bit_state(self, state: Dict[str, Any]) -> str:
        """Extract 2-bit state from input state"""
        try:
            # Create a hash of the state to determine bit state
            state_str = str(sorted(state.items()))
            state_hash = hashlib.sha256(state_str.encode()).hexdigest()
            hash_int = int(state_hash[:8], 16)
            
            # Extract 2-bit state
            bit_state = format(hash_int & 0b11, '02b')
            
            return bit_state
            
        except Exception as e:
            logger.error(f"Failed to extract bit state: {e}")
            return "00"  # Default to NULL_VECTOR
    
    def _update_connection_matrix(self, bit_state: str, processed_state: Dict[str, Any]):
        """Update connection matrix based on bit state and processed state"""
        try:
            # Convert bit state to matrix indices
            bit_state_to_index = {"00": 0, "01": 1, "10": 2, "11": 3}
            current_index = bit_state_to_index.get(bit_state, 0)
            
            # Calculate connection strength based on state energy
            state_energy = processed_state.get("state_energy", 0.0)
            processing_intensity = processed_state.get("processing_intensity", 0.0)
            
            # Update connection matrix
            for i in range(4):
                if i == current_index:
                    # Self-connection based on state energy
                    self.connection_matrix[i, i] = state_energy
                else:
                    # Cross-connections based on processing intensity
                    connection_strength = processing_intensity * 0.1
                    self.connection_matrix[current_index, i] += connection_strength
                    self.connection_matrix[i, current_index] += connection_strength
            
            # Normalize connection matrix to prevent overflow
            self.connection_matrix = np.clip(self.connection_matrix, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to update connection matrix: {e}")
    
    def _update_performance_metrics(self, bit_state: str, processing_time: float):
        """Update performance tracking metrics"""
        try:
            self.total_states_processed += 1
            self.bit_state_distribution[bit_state] += 1
            
            # Update average processing time
            if self.total_states_processed == 1:
                self.average_processing_time = processing_time
            else:
                self.average_processing_time = (
                    (self.average_processing_time * (self.total_states_processed - 1) + processing_time) /
                    self.total_states_processed
                )
                
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def get_connection_matrix(self) -> np.ndarray:
        """Get current connection matrix"""
        return self.connection_matrix.copy()
    
    def get_bit_gate_statistics(self) -> Dict[str, Any]:
        """Get statistics for all bit gates"""
        try:
            gate_stats = {}
            for bit_state, gate in self.bit_gates.items():
                gate_stats[bit_state] = gate.get_processing_statistics()
            
            return {
                "gate_statistics": gate_stats,
                "bit_state_distribution": self.bit_state_distribution,
                "total_states_processed": self.total_states_processed,
                "average_processing_time": self.average_processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get bit gate statistics: {e}")
            return {}
    
    def get_holistic_connectivity_score(self) -> float:
        """Calculate holistic connectivity score from connection matrix"""
        try:
            # Calculate connectivity score based on connection matrix
            total_connections = np.sum(self.connection_matrix)
            max_possible_connections = self.connection_matrix.size
            connectivity_score = total_connections / max_possible_connections if max_possible_connections > 0 else 0.0
            
            return connectivity_score
            
        except Exception as e:
            logger.error(f"Failed to calculate connectivity score: {e}")
            return 0.0
    
    def get_state_history_summary(self) -> Dict[str, Any]:
        """Get summary of state history"""
        try:
            if not self.state_history:
                return {"total_states": 0, "average_energy": 0.0}
            
            total_states = len(self.state_history)
            energies = [state.get("state_energy", 0.0) for state in self.state_history]
            average_energy = sum(energies) / len(energies) if energies else 0.0
            
            # Count processing types
            processing_types = {}
            for state in self.state_history:
                for key, value in state.items():
                    if key.endswith("_processed") and isinstance(value, bool) and value:
                        processing_type = key.replace("_processed", "")
                        processing_types[processing_type] = processing_types.get(processing_type, 0) + 1
            
            return {
                "total_states": total_states,
                "average_energy": average_energy,
                "processing_type_distribution": processing_types
            }
            
        except Exception as e:
            logger.error(f"Failed to get state history summary: {e}")
            return {"total_states": 0, "average_energy": 0.0}
    
    def clear_state_history(self) -> int:
        """Clear state history and return count of cleared states"""
        cleared_count = len(self.state_history)
        self.state_history.clear()
        logger.info(f"Cleared {cleared_count} states from history")
        return cleared_count
    
    def reset_connection_matrix(self):
        """Reset connection matrix to zero"""
        self.connection_matrix = np.zeros((4, 4))
        logger.info("Reset connection matrix")


# Global Lantern Core instance
lantern_core = LanternCore()


def get_lantern_core() -> LanternCore:
    """Get global lantern core instance"""
    return lantern_core


def relay_state_to_bit_gates(input_state: Dict[str, Any]) -> Dict[str, Any]:
    """Relay input state to bit gates through lantern core"""
    return lantern_core.relay_to_bit_gates(input_state)


def get_lantern_statistics() -> Dict[str, Any]:
    """Get comprehensive lantern core statistics"""
    return {
        "bit_gates": lantern_core.get_bit_gate_statistics(),
        "connectivity_score": lantern_core.get_holistic_connectivity_score(),
        "state_history": lantern_core.get_state_history_summary(),
        "connection_matrix": lantern_core.get_connection_matrix().tolist()
    } 