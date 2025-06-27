# -*- coding: utf-8 -*-
"""
Bit-Phase Sequencer - Multi-Phase Transition Engine
===================================================

This module implements the bit-phase sequencing system for Schwabot, handling
transitions between 2-bit, 4-bit, 8-bit, and 42-phase states for event-driven
profit routing. It provides temporal bit gate compression models and cross-phase
error handling for the symbolic profit navigation system.

Core Phases:
- 2-bit: Fundamental flip states (0, 1, 10, 11)
- 4-bit: Primary logic atomization and opcodes
- 8-bit: Memory register patterns and triggers
- 42-bit: Symbolic recursion depth cycles
- 256: SHA-256 encrypted identity locks
"""

import math
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import struct
import logging

logger = logging.getLogger(__name__)


class BitPhase(Enum):
    """Bit-phase levels for sequencing"""
    TWO_BIT = 2  # Fundamental flip states
    FOUR_BIT = 4  # Primary logic atomization
    EIGHT_BIT = 8  # Memory register patterns
    FORTY_TWO = 42  # Symbolic recursion depth
    TWO_FIFTY_SIX = 256  # SHA-256 identity


class SequenceState(Enum):
    """States of bit-phase sequences"""
    INITIALIZING = "initializing"
    STABLE = "stable"
    TRANSITIONING = "transitioning"
    COMPRESSED = "compressed"
    OVERFLOW = "overflow"
    LOCKED = "locked"


class GateOperation(Enum):
    """Temporal bit gate operations"""
    AND_GATE = "and"
    OR_GATE = "or"
    XOR_GATE = "xor"
    NAND_GATE = "nand"
    NOR_GATE = "nor"
    COMPRESS = "compress"
    EXPAND = "expand"
    PHASE_SHIFT = "phase_shift"


@dataclass
class BitSequence:
    """Container for bit-phase sequence data"""
    sequence_id: str
    phase: BitPhase
    state: SequenceState
    bit_pattern: str
    value: int
    entropy_score: float
    temporal_position: float  # Position in time
    compression_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PhaseTransition:
    """Transition between bit phases"""
    from_phase: BitPhase
    to_phase: BitPhase
    transition_id: str
    gate_operations: List[GateOperation]
    success_probability: float
    energy_cost: float  # Computational cost
    data_preservation: float  # How much data survives transition
    triggers: List[str]


@dataclass
class CompressionModel:
    """Temporal bit gate compression model"""
    model_id: str
    source_phase: BitPhase
    target_phase: BitPhase
    compression_function: Callable
    efficiency_score: float
    stability_rating: float
    thermal_tolerance: float


class BitPhaseSequencer:
    """Main bit-phase sequencing and transition engine"""

    def __init__(self):
        """Initialize the bit-phase sequencer"""
        self.active_sequences: Dict[str, BitSequence] = {}
        self.transition_registry: Dict[str, PhaseTransition] = {}
        self.compression_models: Dict[str, CompressionModel] = {}
        self.phase_statistics: Dict[BitPhase, Dict[str, float]] = {}
        self.temporal_clock = 0.0

        self._initialize_phase_statistics()
        self._initialize_compression_models()
        self._initialize_standard_transitions()
        logger.info("BitPhaseSequencer initialized")

    def _initialize_phase_statistics(self):
        """Initialize statistics tracking for each phase"""
        for phase in BitPhase:
            self.phase_statistics[phase] = {
                'total_sequences': 0,
                'successful_transitions': 0,
                'failed_transitions': 0,
                'average_entropy': 0.0,
                'compression_efficiency': 0.0,
                'thermal_load': 0.0
            }

    def _initialize_compression_models(self):
        """Initialize compression models for phase transitions"""

        # 2-bit to 4-bit compression
        self.compression_models['2to4'] = CompressionModel(
            model_id='2to4',
            source_phase=BitPhase.TWO_BIT,
            target_phase=BitPhase.FOUR_BIT,
            compression_function=self._compress_2_to_4,
            efficiency_score=0.9,
            stability_rating=0.95,
            thermal_tolerance=0.8
        )

        # 4-bit to 8-bit compression
        self.compression_models['4to8'] = CompressionModel(
            model_id='4to8',
            source_phase=BitPhase.FOUR_BIT,
            target_phase=BitPhase.EIGHT_BIT,
            compression_function=self._compress_4_to_8,
            efficiency_score=0.85,
            stability_rating=0.9,
            thermal_tolerance=0.7
        )

        # 8-bit to 42-phase compression
        self.compression_models['8to42'] = CompressionModel(
            model_id='8to42',
            source_phase=BitPhase.EIGHT_BIT,
            target_phase=BitPhase.FORTY_TWO,
            compression_function=self._compress_8_to_42,
            efficiency_score=0.7,
            stability_rating=0.8,
            thermal_tolerance=0.6
        )

        # 42-phase to 256-bit compression
        self.compression_models['42to256'] = CompressionModel(
            model_id='42to256',
            source_phase=BitPhase.FORTY_TWO,
            target_phase=BitPhase.TWO_FIFTY_SIX,
            compression_function=self._compress_42_to_256,
            efficiency_score=0.6,
            stability_rating=0.75,
            thermal_tolerance=0.5
        )

    def _initialize_standard_transitions(self):
        """Initialize standard phase transitions"""

        # Create upward transitions (compression)
        phases = [BitPhase.TWO_BIT, BitPhase.FOUR_BIT, BitPhase.EIGHT_BIT,
                  BitPhase.FORTY_TWO, BitPhase.TWO_FIFTY_SIX]

        for i in range(len(phases) - 1):
            from_phase = phases[i]
            to_phase = phases[i + 1]

            transition = PhaseTransition(
                from_phase=from_phase,
                to_phase=to_phase,
                transition_id=f"up_{from_phase.value}to{to_phase.value}",
                gate_operations=[
                    GateOperation.COMPRESS,
                    GateOperation.PHASE_SHIFT],
                success_probability=0.8 - (i * 0.1),  # Decreasing probability
                energy_cost=2.0 ** i,  # Exponential cost increase
                data_preservation=1.0 - (i * 0.5),  # Slight data loss
                triggers=[f"entropy_threshold_{i}", "manual_trigger"]
            )

            self.transition_registry[transition.transition_id] = transition

        # Create downward transitions (expansion)
        for i in range(len(phases) - 1, 0, -1):
            from_phase = phases[i]
            to_phase = phases[i - 1]

            transition = PhaseTransition(
                from_phase=from_phase,
                to_phase=to_phase,
                transition_id=f"down_{from_phase.value}to{to_phase.value}",
                gate_operations=[
                    GateOperation.EXPAND,
                    GateOperation.PHASE_SHIFT],
                success_probability=0.9,  # Easier to expand
                energy_cost=1.0,  # Lower cost
                data_preservation=0.95,  # Better preservation
                triggers=["overflow_detected", "stability_restore"]
            )

            self.transition_registry[transition.transition_id] = transition

    def create_sequence(self, initial_value: int, phase: BitPhase,
                        context: str = "") -> BitSequence:
        """Create a new bit-phase sequence"""

        # Generate sequence ID
        sequence_id = f"seq_{
            phase.value}_{
            int(
                time.time())}_{
                hash(context) %
            1000}"

        # Convert value to bit pattern based on phase
        bit_pattern = self._value_to_bit_pattern(initial_value, phase)

        # Calculate entropy score
        entropy_score = self._calculate_entropy(bit_pattern)

        # Calculate temporal position
        temporal_position = self.temporal_clock + (time.time() % 1.0)

        # Calculate compression ratio
        compression_ratio = self._calculate_compression_ratio(
            bit_pattern, phase)

        sequence = BitSequence(
            sequence_id=sequence_id,
            phase=phase,
            state=SequenceState.INITIALIZING,
            bit_pattern=bit_pattern,
            value=initial_value,
            entropy_score=entropy_score,
            temporal_position=temporal_position,
            compression_ratio=compression_ratio,
            metadata={
                'context': context,
                'creation_method': 'manual',
                'thermal_signature': time.time() % 1.0
            }
        )

        # Register the sequence
        self.active_sequences[sequence_id] = sequence

        # Update statistics
        stats = self.phase_statistics[phase]
        stats['total_sequences'] += 1
        stats['average_entropy'] = (
            stats['average_entropy'] + entropy_score) / 2.0

        sequence.state = SequenceState.STABLE

        logger.info(f"Created sequence {sequence_id} in phase {phase.value} "
                    f"(entropy: {entropy_score:.3f})")

    def _value_to_bit_pattern(self, value: int, phase: BitPhase) -> str:
        """Convert integer value to bit pattern for specified phase"""

        # Ensure value fits in phase bit width
        max_value = (2 ** phase.value) - 1
        clamped_value = value % (max_value + 1)

        # Format to binary string (unused but kept for potential future use)
        # bit_pattern = format(clamped_value, f'0{phase.value}b')

    def _calculate_entropy(self, bit_pattern: str) -> float:
        """Calculate Shannon entropy of bit pattern"""
        if not bit_pattern:
            return 0.0

        # Count bit frequencies
        ones = bit_pattern.count('1')
        zeros = bit_pattern.count('0')
        total = len(bit_pattern)

        if ones == 0 or zeros == 0:
            return 0.0  # No entropy in uniform patterns

        # Calculate Shannon entropy
        p_one = ones / total
        p_zero = zeros / total

        entropy = -(p_one * math.log2(p_one) + p_zero * math.log2(p_zero))

        return entropy

    def _calculate_compression_ratio(
            self,
            bit_pattern: str,
            phase: BitPhase) -> float:
        """Calculate compression ratio for bit pattern in phase"""

        # Count run lengths (consecutive same bits)
        runs = []
        current_run = 1

        for i in range(1, len(bit_pattern)):
            if bit_pattern[i] == bit_pattern[i - 1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)

        # Calculate theoretical compression based on run lengths
        original_bits = len(bit_pattern)
        compressed_bits = sum(math.ceil(math.log2(run + 1))
                              for run in runs) + len(runs)

        compression_ratio = compressed_bits / original_bits if original_bits > 0 else 1.0

        return compression_ratio

    def transition_phase(self, sequence_id: str, target_phase: BitPhase,
                         force: bool = False) -> bool:
        """Transition a sequence to a different phase"""

        sequence = self.active_sequences.get(sequence_id)
        if not sequence:
            logger.error(f"Sequence {sequence_id} not found")
            return False

        current_phase = sequence.phase

        if current_phase == target_phase:
            logger.warning(
                f"Sequence {sequence_id} already in phase {
                    target_phase.value}")
            return True

        # Find appropriate transition
        transition_id = f"{
            'up' if target_phase.value > current_phase.value else 'down'}_{
            current_phase.value}to{
            target_phase.value}"
        transition = self.transition_registry.get(transition_id)

        if not transition and not force:
            logger.error(
                f"No transition found from {
                    current_phase.value} to {
                    target_phase.value}")
            return False

        # Mark sequence as transitioning
        sequence.state = SequenceState.TRANSITIONING

        try:
            # Apply compression / expansion model if available
            model_key = f"{current_phase.value}to{target_phase.value}"
            model = self.compression_models.get(model_key)

            if model:
                new_bit_pattern = model.compression_function(
                    sequence.bit_pattern)
                success_probability = model.efficiency_score
            else:
                # Direct bit manipulation
                new_bit_pattern = self._direct_phase_conversion(
                    sequence.bit_pattern, current_phase, target_phase)
                success_probability = 0.5  # Lower probability for direct conversion

            # Check if transition should succeed
            if not force and (time.time() % 1.0) > success_probability:
                sequence.state = SequenceState.OVERFLOW
                logger.warning(
                    f"Phase transition failed for sequence {sequence_id}")
                return False

            # Apply transition
            sequence.phase = target_phase
            sequence.bit_pattern = new_bit_pattern
            sequence.value = int(new_bit_pattern, 2) if new_bit_pattern else 0
            sequence.entropy_score = self._calculate_entropy(new_bit_pattern)
            sequence.compression_ratio = self._calculate_compression_ratio(
                new_bit_pattern, target_phase)
            sequence.state = SequenceState.STABLE
            sequence.temporal_position = self.temporal_clock + \
                (time.time() % 1.0)

            # Update statistics
            if transition:
                stats = self.phase_statistics[current_phase]
                stats['successful_transitions'] += 1
                target_stats = self.phase_statistics[target_phase]
                target_stats['total_sequences'] += 1

            logger.info(
                f"Successfully transitioned sequence {sequence_id} from " f"{
                    current_phase.value} to {
                    target_phase.value}")

            return True

        except Exception as e:
            sequence.state = SequenceState.OVERFLOW
            stats = self.phase_statistics[current_phase]
            stats['failed_transitions'] += 1
            logger.error(
                f"Phase transition failed for sequence {sequence_id}: {e}")
            return False

    def _direct_phase_conversion(self, bit_pattern: str, from_phase: BitPhase,
                                 to_phase: BitPhase) -> str:
        """Direct conversion between phases without compression model"""

        if to_phase.value > from_phase.value:
            # Expanding - pad with zeros
            padding = '0' * (to_phase.value - from_phase.value)
            return bit_pattern + padding
        else:
            # Compressing - truncate
            return bit_pattern[:to_phase.value]

    # Compression model functions
    def _compress_2_to_4(self, bit_pattern: str) -> str:
        """Compress 2-bit pattern to 4-bit using duplication"""
        if len(bit_pattern) >= 2:
            return bit_pattern * 2  # Duplicate the pattern
        return bit_pattern.ljust(4, '0')

    def _compress_4_to_8(self, bit_pattern: str) -> str:
        """Compress 4-bit pattern to 8-bit using XOR expansion"""
        if len(bit_pattern) >= 4:
            # XOR the pattern with its reverse
            reversed_pattern = bit_pattern[::-1]
            result = ''
            for i in range(4):
                xor_result = str(
                    int(bit_pattern[i]) ^ int(reversed_pattern[i]))
                result += bit_pattern[i] + xor_result
            return result[:8]
        return bit_pattern.ljust(8, '0')

    def _compress_8_to_42(self, bit_pattern: str) -> str:
        """Compress 8-bit pattern to 42-bit using mathematical expansion"""
        if len(bit_pattern) >= 8:
            # Convert to number and apply mathematical transformation
            value = int(bit_pattern, 2)

            # Apply golden ratio expansion (42 is related to universal
            # constants)
            golden_ratio = 1.618033988749
            expanded_value = int(value * golden_ratio * 255)  # Scale up

            # Convert back to 42-bit pattern
            return format(expanded_value % (2**42), '042b')
        return bit_pattern.ljust(42, '0')

    def _compress_42_to_256(self, bit_pattern: str) -> str:
        """Compress 42-bit pattern to 256-bit using SHA-256 expansion"""
        if len(bit_pattern) >= 42:
            # Use SHA-256 to create cryptographically secure expansion
            hash_input = bit_pattern.encode('utf-8')
            sha_hash = hashlib.sha256(hash_input).digest()

            # Convert hash bytes to bit string
            bit_string = ''.join(format(byte, '08b') for byte in sha_hash)
            return bit_string  # Should be exactly 256 bits
        return bit_pattern.ljust(256, '0')

    def apply_gate_operation(
            self,
            sequence_id1: str,
            sequence_id2: str,
            operation: GateOperation) -> Optional[BitSequence]:
        """Apply temporal bit gate operation between two sequences"""

        seq1 = self.active_sequences.get(sequence_id1)
        seq2 = self.active_sequences.get(sequence_id2)

        if not seq1 or not seq2:
            logger.error("One or both sequences not found for gate operation")
            return None

        # Ensure sequences are in same phase for gate operations
        if seq1.phase != seq2.phase and operation not in [
                GateOperation.COMPRESS, GateOperation.EXPAND]:
            logger.error("Sequences must be in same phase for gate operations")
            return None

        try:
            # Apply the gate operation
            if operation == GateOperation.AND_GATE:
                result_pattern = self._apply_and_gate(
                    seq1.bit_pattern, seq2.bit_pattern)
            elif operation == GateOperation.OR_GATE:
                result_pattern = self._apply_or_gate(
                    seq1.bit_pattern, seq2.bit_pattern)
            elif operation == GateOperation.XOR_GATE:
                result_pattern = self._apply_xor_gate(
                    seq1.bit_pattern, seq2.bit_pattern)
            elif operation == GateOperation.NAND_GATE:
                result_pattern = self._apply_nand_gate(
                    seq1.bit_pattern, seq2.bit_pattern)
            elif operation == GateOperation.NOR_GATE:
                result_pattern = self._apply_nor_gate(
                    seq1.bit_pattern, seq2.bit_pattern)
            else:
                logger.error(f"Unsupported gate operation: {operation}")
                return None

            # Create result sequence
            result_value = int(result_pattern, 2) if result_pattern else 0
            result_sequence = self.create_sequence(
                result_value,
                seq1.phase,
                f"gate_op_{operation.value}_{sequence_id1[:4]}_{sequence_id2[:4]}"
            )

            logger.info(f"Applied {operation.value} gate to sequences {sequence_id1[:8]} "
                        f"and {sequence_id2[:8]}")

            return result_sequence

        except Exception as e:
            logger.error(f"Gate operation {operation.value} failed: {e}")
            return None

    def _apply_and_gate(self, pattern1: str, pattern2: str) -> str:
        """Apply AND gate operation"""
        min_len = min(len(pattern1), len(pattern2))
        result = ''
        for i in range(min_len):
            result += str(int(pattern1[i]) & int(pattern2[i]))
        return result

    def _apply_or_gate(self, pattern1: str, pattern2: str) -> str:
        """Apply OR gate operation"""
        min_len = min(len(pattern1), len(pattern2))
        result = ''
        for i in range(min_len):
            result += str(int(pattern1[i]) | int(pattern2[i]))
        return result

    def _apply_xor_gate(self, pattern1: str, pattern2: str) -> str:
        """Apply XOR gate operation"""
        min_len = min(len(pattern1), len(pattern2))
        result = ''
        for i in range(min_len):
            result += str(int(pattern1[i]) ^ int(pattern2[i]))
        return result

    def _apply_nand_gate(self, pattern1: str, pattern2: str) -> str:
        """Apply NAND gate operation"""
        and_result = self._apply_and_gate(pattern1, pattern2)
        # Invert the result
        return ''.join('1' if bit == '0' else '0' for bit in and_result)

    def _apply_nor_gate(self, pattern1: str, pattern2: str) -> str:
        """Apply NOR gate operation"""
        or_result = self._apply_or_gate(pattern1, pattern2)
        # Invert the result
        return ''.join('1' if bit == '0' else '0' for bit in or_result)

    def advance_temporal_clock(self, delta: float = 0.1):
        """Advance the temporal clock for all sequences"""
        self.temporal_clock += delta

        # Update temporal positions of all sequences
        for sequence in self.active_sequences.values():
            sequence.temporal_position += delta

        # Check for temporal overflow
        if sequence.temporal_position > 100.0:  # Arbitrary overflow threshold
            sequence.state = SequenceState.OVERFLOW
            logger.warning(
                f"Temporal overflow for sequence {
                    sequence.sequence_id}")

    def compress_sequence(self, sequence_id: str) -> bool:
        """Compress a sequence to its optimal phase"""
        sequence = self.active_sequences.get(sequence_id)
        if not sequence:
            return False

        # Find optimal compression based on entropy
        optimal_phase = self._find_optimal_phase(sequence)

        if optimal_phase != sequence.phase:
            return self.transition_phase(sequence_id, optimal_phase)

        # Mark as compressed
        sequence.state = SequenceState.COMPRESSED
        return True

    def _find_optimal_phase(self, sequence: BitSequence) -> BitPhase:
        """Find optimal phase for sequence based on entropy and compression"""
        current_entropy = sequence.entropy_score

        # Higher entropy sequences benefit from higher phases
        if current_entropy > 0.9:
            return BitPhase.TWO_FIFTY_SIX
        elif current_entropy > 0.7:
            return BitPhase.FORTY_TWO
        elif current_entropy > 0.5:
            return BitPhase.EIGHT_BIT
        elif current_entropy > 0.3:
            return BitPhase.FOUR_BIT
        else:
            return BitPhase.TWO_BIT

    def get_sequence_summary(self) -> Dict[str, Any]:
        """Get summary of all sequences and phases"""
        summary = {
            'total_sequences': len(self.active_sequences),
            'temporal_clock': self.temporal_clock,
            'phase_distribution': {},
            'state_distribution': {},
            'average_entropy_by_phase': {},
            'phase_statistics': self.phase_statistics
        }

        # Count by phase and state
        for sequence in self.active_sequences.values():
            phase = sequence.phase.value
            state = sequence.state.value

            summary['phase_distribution'][phase] = summary['phase_distribution'].get(
                phase, 0) + 1
            summary['state_distribution'][state] = summary['state_distribution'].get(
                state, 0) + 1

        # Calculate average entropy by phase
        entropy_sums = {}
        entropy_counts = {}

        for sequence in self.active_sequences.values():
            phase = sequence.phase.value
            if phase not in entropy_sums:
                entropy_sums[phase] = 0.0
                entropy_counts[phase] = 0

            entropy_sums[phase] += sequence.entropy_score
            entropy_counts[phase] += 1

        for phase in entropy_sums:
            summary['average_entropy_by_phase'][phase] = entropy_sums[phase] / \
                entropy_counts[phase]

        return summary


def main():
    """Test the bit-phase sequencer"""
    sequencer = BitPhaseSequencer()

    print("Testing bit-phase sequencing:")

    # Create sequences in different phases
    test_values = [3, 15, 255, 1000, 50000]
    sequences = []

    for i, value in enumerate(test_values):
        phase = list(BitPhase)[i % len(BitPhase)]
        seq = sequencer.create_sequence(value, phase, f"test_{i}")
        sequences.append(seq)
        print(f"Created sequence {seq.sequence_id[:8]}: {seq.bit_pattern} "
              f"(phase: {seq.phase.value}, entropy: {seq.entropy_score:.3f})")

    print("\nTesting phase transitions:")

    # Test transitions
    for seq in sequences[:3]:
        target_phase = BitPhase.FORTY_TWO
        success = sequencer.transition_phase(seq.sequence_id, target_phase)
        print(
            f"Transition {seq.sequence_id[:8]} to phase {target_phase.value}: {'✓' if success else '✗'}")

    print("\nTesting gate operations:")

    # Test gate operations
    if len(sequences) >= 2:
        gate_ops = [
            GateOperation.AND_GATE,
            GateOperation.XOR_GATE,
            GateOperation.OR_GATE]
        for op in gate_ops:
            result = sequencer.apply_gate_operation(
                sequences[0].sequence_id, sequences[1].sequence_id, op)
            if result:
                print(
                    f"{op.value}: {result.bit_pattern} (entropy: {result.entropy_score:.3f})")

    print(f"\nSequencer Summary:")
    summary = sequencer.get_sequence_summary()
    for key, value in summary.items():
        if key != 'phase_statistics':
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
