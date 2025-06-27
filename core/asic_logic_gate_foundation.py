# -*- coding: utf-8 -*-
"""
ASIC Logic Gate Foundation - Dualistic Emoji Routing System

Implements ASIC-compatible logic gates with dualistic emoji routing,
2-bit state extraction, and SHA-256 hash signatures for profit vectorization.

Mathematical Foundation:
- 2-bit state extraction: bit_state = (ord(emoji) & 0b11)
- Hash signature: H(σ) = SHA256(emoji + bit_state + gate_type)
- Profit vector: P(σ) = base_weight(emoji) × bit_multiplier(bit_state)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import logging
import time

import numpy as np

from core.unified_math_system import unified_math

# Configure logging
logger = logging.getLogger(__name__)


class GateType(Enum):
    """ASIC Logic Gate Types"""
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NAND = "NAND"
    NOR = "NOR"
    XNOR = "XNOR"


class BitState(Enum):
    """2-bit state enumeration"""
    NULL_VECTOR = "00"
    LOW_TIER = "01"
    MID_TIER = "10"
    PEAK_TIER = "11"


@dataclass
class ASICLogicGate:
    """ASIC-compatible logic gate with dualistic emoji routing"""
    
    gate_type: GateType
    emoji_symbol: str
    bit_state: str
    hash_signature: str
    profit_vector: float
    timestamp: float
    
    def __post_init__(self):
        """Validate and initialize gate after creation"""
        if not self.bit_state:
            self.bit_state = self._extract_2bit_state(self.emoji_symbol)
        if not self.hash_signature:
            self.hash_signature = self._generate_hash_signature()
        if self.profit_vector == 0.0:
            self.profit_vector = self._calculate_profit_vector()
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def _extract_2bit_state(self, emoji: str) -> str:
        """Extract 2-bit state from Unicode symbol"""
        try:
            val = ord(emoji)
            return format(val & 0b11, '02b')  # Returns "00", "01", "10", or "11"
        except Exception as e:
            logger.warning(f"Failed to extract 2-bit state from {emoji}: {e}")
            return "00"  # Default to NULL_VECTOR
    
    def _generate_hash_signature(self) -> str:
        """Generate SHA-256 hash for ASIC routing"""
        try:
            input_data = f"{self.emoji_symbol}_{self.bit_state}_{self.gate_type.value}"
            return hashlib.sha256(input_data.encode()).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Failed to generate hash signature: {e}")
            return "0000000000000000"
    
    def _calculate_profit_vector(self) -> float:
        """Calculate profit vector based on emoji and bit state"""
        try:
            # Base weights for different emoji symbols
            base_weights = {
                "💰": 1.5, "🔥": 2.0, "⚡": 1.2, "🎯": 2.5,
                "📈": 1.6, "🧠": 2.2, "🔄": 1.0, "⚠️": 0.8,
                "🟢": 1.1, "🔴": 0.9, "🟡": 1.3, "🟠": 1.4,
                "⚫": 0.5, "⚪": 0.7, "🔵": 1.0, "🟣": 1.2
            }
            
            # Bit state multipliers
            bit_multipliers = {
                "00": 0.5,  # NULL_VECTOR
                "01": 1.0,  # LOW_TIER
                "10": 1.5,  # MID_TIER
                "11": 2.0   # PEAK_TIER
            }
            
            base_weight = base_weights.get(self.emoji_symbol, 1.0)
            bit_multiplier = bit_multipliers.get(self.bit_state, 1.0)
            
            return base_weight * bit_multiplier
            
        except Exception as e:
            logger.error(f"Failed to calculate profit vector: {e}")
            return 1.0
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through ASIC logic gate"""
        try:
            processed_data = input_data.copy()
            
            # Add gate metadata
            processed_data["gate_type"] = self.gate_type.value
            processed_data["emoji_symbol"] = self.emoji_symbol
            processed_data["bit_state"] = self.bit_state
            processed_data["hash_signature"] = self.hash_signature
            processed_data["profit_vector"] = self.profit_vector
            processed_data["processing_timestamp"] = time.time()
            
            # Apply gate-specific logic
            if self.gate_type == GateType.AND:
                processed_data = self._apply_and_logic(processed_data)
            elif self.gate_type == GateType.OR:
                processed_data = self._apply_or_logic(processed_data)
            elif self.gate_type == GateType.XOR:
                processed_data = self._apply_xor_logic(processed_data)
            elif self.gate_type == GateType.NAND:
                processed_data = self._apply_nand_logic(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to process input through gate: {e}")
            return input_data
    
    def _apply_and_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply AND logic to input data"""
        # Extract boolean values from data
        bool_values = []
        for key, value in data.items():
            if isinstance(value, bool):
                bool_values.append(value)
            elif isinstance(value, (int, float)):
                bool_values.append(value > 0)
        
        # Apply AND logic
        result = all(bool_values) if bool_values else True
        data["and_result"] = result
        data["logic_applied"] = "AND"
        
        return data
    
    def _apply_or_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply OR logic to input data"""
        # Extract boolean values from data
        bool_values = []
        for key, value in data.items():
            if isinstance(value, bool):
                bool_values.append(value)
            elif isinstance(value, (int, float)):
                bool_values.append(value > 0)
        
        # Apply OR logic
        result = any(bool_values) if bool_values else False
        data["or_result"] = result
        data["logic_applied"] = "OR"
        
        return data
    
    def _apply_xor_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply XOR logic to input data"""
        # Extract boolean values from data
        bool_values = []
        for key, value in data.items():
            if isinstance(value, bool):
                bool_values.append(value)
            elif isinstance(value, (int, float)):
                bool_values.append(value > 0)
        
        # Apply XOR logic (odd number of True values)
        true_count = sum(bool_values)
        result = (true_count % 2) == 1
        data["xor_result"] = result
        data["logic_applied"] = "XOR"
        
        return data
    
    def _apply_nand_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply NAND logic to input data"""
        # Apply AND logic first
        data = self._apply_and_logic(data)
        
        # Invert the result
        if "and_result" in data:
            data["nand_result"] = not data["and_result"]
            data["logic_applied"] = "NAND"
        
        return data


class ASICLogicGateManager:
    """Manager for ASIC logic gates"""
    
    def __init__(self):
        self.gates: Dict[str, ASICLogicGate] = {}
        self.gate_history: List[ASICLogicGate] = []
        self.active_gates: set = set()
        
        # Initialize default gates
        self._initialize_default_gates()
    
    def _initialize_default_gates(self):
        """Initialize default ASIC logic gates"""
        default_gates = [
            (GateType.AND, "💰"), (GateType.OR, "🔥"), 
            (GateType.XOR, "⚡"), (GateType.NAND, "🎯"),
            (GateType.OR, "📈"), (GateType.AND, "🧠"),
            (GateType.XOR, "🔄"), (GateType.NAND, "⚠️")
        ]
        
        for gate_type, emoji in default_gates:
            gate = ASICLogicGate(
                gate_type=gate_type,
                emoji_symbol=emoji,
                bit_state="",
                hash_signature="",
                profit_vector=0.0,
                timestamp=0.0
            )
            self.register_gate(gate)
    
    def register_gate(self, gate: ASICLogicGate) -> str:
        """Register a new ASIC logic gate"""
        gate_key = f"{gate.emoji_symbol}_{gate.hash_signature[:8]}"
        self.gates[gate_key] = gate
        self.gate_history.append(gate)
        self.active_gates.add(gate_key)
        
        logger.info(f"Registered ASIC gate: {gate_key} -> {gate.gate_type.value}")
        return gate_key
    
    def get_gate(self, gate_key: str) -> Optional[ASICLogicGate]:
        """Get gate by key"""
        return self.gates.get(gate_key)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through all active gates"""
        try:
            processed_data = input_data.copy()
            active_gates = list(self.active_gates)
            
            # Process through each active gate
            for gate_key in active_gates:
                gate = self.gates.get(gate_key)
                if gate:
                    processed_data = gate.process_input(processed_data)
            
            # Add processing metadata
            processed_data["total_gates_processed"] = len(active_gates)
            processed_data["processing_timestamp"] = time.time()
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to process input through gates: {e}")
            return input_data
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered gates"""
        return {
            "total_gates": len(self.gates),
            "active_gates": len(self.active_gates),
            "gate_types": {gate.gate_type.value: sum(1 for g in self.gates.values() if g.gate_type == gate.gate_type) for gate in GateType},
            "bit_states": {state.value: sum(1 for g in self.gates.values() if g.bit_state == state.value) for state in BitState},
            "average_profit_vector": np.mean([gate.profit_vector for gate in self.gates.values()]) if self.gates else 0.0
        }
    
    def clear_inactive_gates(self) -> int:
        """Clear inactive gates and return count of cleared gates"""
        inactive_gates = [key for key in self.gates.keys() if key not in self.active_gates]
        for key in inactive_gates:
            del self.gates[key]
        
        logger.info(f"Cleared {len(inactive_gates)} inactive gates")
        return len(inactive_gates)


# Global ASIC Logic Gate Manager instance
asic_gate_manager = ASICLogicGateManager()


def get_asic_gate_manager() -> ASICLogicGateManager:
    """Get global ASIC logic gate manager instance"""
    return asic_gate_manager


def create_asic_gate(gate_type: GateType, emoji_symbol: str) -> ASICLogicGate:
    """Create a new ASIC logic gate"""
    return ASICLogicGate(
        gate_type=gate_type,
        emoji_symbol=emoji_symbol,
        bit_state="",
        hash_signature="",
        profit_vector=0.0,
        timestamp=0.0
    )


def process_through_asic_gates(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process input data through all ASIC logic gates"""
    return asic_gate_manager.process_input(input_data) 