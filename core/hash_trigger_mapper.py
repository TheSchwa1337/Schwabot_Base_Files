# #!/usr/bin/env python3
"""
Hash Trigger Mapper - Enhanced Hash-to-Strategy Mapping System
============================================================

This module provides sophisticated hash trigger mapping functionality that integrates
with existing systems (HashTriggerEngine, BitResolutionEngine, GhostSignal) to provide
enhanced strategy pathway determination.

Core Functionality:
- Hash trigger to strategy pathway mapping
- Integration with existing hash systems
- Enhanced decision logic for GhostSignal
- Type-safe mathematical operations
- Unicode/emoji-safe CLI output
- Comprehensive error handling

This module enhances rather than replaces existing functionality.
"""

import hashlib
import time
import json
from typing import Dict, List, Optional, Any, Union, Literal, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

# Import our robust systems with Unicode fallback
try:
    # Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}"), safe_math
except ImportError:
    # Fallback for CLI compatibility with proper Unicode handling
    def safe_print(*args, **kwargs):
        """Safe print function with Unicode fallback."""
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe output
safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', 'replace').decode('ascii'))
                else:
safe_args.append(arg)
            print(*safe_args, **kwargs)

    def info(*args, **kwargs):
        """Info logging with Unicode fallback."""
        try:
            print("[INFO]", *args, **kwargs)
        except UnicodeEncodeError:
safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', 'replace').decode('ascii'))
                else:
safe_args.append(arg)
            print("[INFO]", *safe_args, **kwargs)

    def warn(*args, **kwargs):
        """Warning logging with Unicode fallback."""
        try:
            print("[WARN]", *args, **kwargs)
        except UnicodeEncodeError:
safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', 'replace').decode('ascii'))
                else:
safe_args.append(arg)
            print("[WARN]", *safe_args, **kwargs)

    def error(*args, **kwargs):
        """Error logging with Unicode fallback."""
        try:
            print("[ERROR]", *args, **kwargs)
        except UnicodeEncodeError:
safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', 'replace').decode('ascii'))
                else:
safe_args.append(arg)
            print("[ERROR]", *safe_args, **kwargs)

    def success(*args, **kwargs):
        """Success logging with Unicode fallback."""
        try:
            print("[SUCCESS]", *args, **kwargs)
        except UnicodeEncodeError:
safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', 'replace').decode('ascii'))
                else:
safe_args.append(arg)
            print("[SUCCESS]", *safe_args, **kwargs)

    def debug(*args, **kwargs):
        """Debug logging with Unicode fallback."""
        try:
            print("[DEBUG]", *args, **kwargs)
        except UnicodeEncodeError:
safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', 'replace').decode('ascii'))
                else:
safe_args.append(arg)
            print("[DEBUG]", *safe_args, **kwargs)

    def safe_math(*args, **kwargs):
        """Math logging with Unicode fallback."""
        try:
            print("[MATH]", *args, **kwargs)
        except UnicodeEncodeError:
safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', 'replace').decode('ascii'))
                else:
safe_args.append(arg)
            print("[MATH]", *safe_args, **kwargs)

try:
    from core.unified_math_system import unified_math
except ImportError:
    # Fallback math system with proper type annotations
    import numpy as np

    class FallbackMath:
        """Fallback math system for when unified_math_system is unavailable."""

@staticmethod
        def mean(data: List[float]) -> float:
            """Calculate mean of data."""
            return float(np.mean(data))

@staticmethod
        def std(data: List[float]) -> float:
            """Calculate standard deviation of data."""
            return float(np.std(data))

@staticmethod
        def min(data: List[float]) -> float:
            """Calculate minimum of data."""
            return float(np.min(data))

@staticmethod
        def max(data: List[float]) -> float:
            """Calculate maximum of data."""
            return float(np.max(data))

@staticmethod
        def abs(value: float) -> float:
            """Calculate absolute value."""
            return float(np.abs(value))

@staticmethod
        def correlation(data1: List[float], data2: List[float]) -> float:
            """Calculate correlation between two datasets."""
            if len(data1) > 1:
                return float(np.corrcoef(data1, data2)[0, 1])
            return 0.0

@staticmethod
        def sqrt(value: float) -> float:
            """Calculate square root."""
            return float(np.sqrt(value))

@staticmethod
        def log(value: float) -> float:
            """Calculate natural logarithm."""
            return float(np.log(value))

unified_math = FallbackMath()

# Type definitions
HashTriggerLevel = Literal["low", "medium", "high", "critical"]
StrategyPathway = Literal[
"aggressive_ghost", "momentum_ghost", "cautious_ghost",
"adaptive_ghost", "defensive_ghost", "monitor_ghost"
]
MappingConfidence = Literal["low", "medium", "high", "critical"]

class HashPatternType(Enum):
    """Types of hash patterns for mapping."""
SEQUENTIAL = "sequential"
REPEATING = "repeating"
RANDOM = "random"
PATTERNED = "patterned"
CRITICAL = "critical"


@dataclass
class HashTriggerMapping:
    """
Hash trigger mapping configuration.

This dataclass represents a mapping from a hash trigger to a strategy pathway
    with associated confidence and metadata.
"""

    # Core mapping data
hash_trigger: str
strategy_pathway: StrategyPathway
confidence_level: MappingConfidence
pattern_type: HashPatternType

    # Enhanced mapping data
mapping_score: float  # 0.0 to 1.0
volatility_factor: float
entropy_factor: float
momentum_factor: float

    # Timing and frequency data
frequency_count: int
last_seen: datetime
average_interval: float  # Average time between occurrences

    # Integration data
bit_phase_compatibility: List[str]  # Compatible bit phases
trigger_engine_compatible: bool
ghost_signal_compatible: bool

    # Metadata
metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert mapping to dictionary for serialization."""
        return {
"hash_trigger": self.hash_trigger,
"strategy_pathway": self.strategy_pathway,
"confidence_level": self.confidence_level,
"pattern_type": self.pattern_type.value,
"mapping_score": self.mapping_score,
"volatility_factor": self.volatility_factor,
"entropy_factor": self.entropy_factor,
"momentum_factor": self.momentum_factor,
"frequency_count": self.frequency_count,
"last_seen": self.last_seen.isoformat(),
            "average_interval": self.average_interval,
"bit_phase_compatibility": self.bit_phase_compatibility,
"trigger_engine_compatible": self.trigger_engine_compatible,
"ghost_signal_compatible": self.ghost_signal_compatible,
"metadata": self.metadata
}


class HashTriggerMapper:
    """
Enhanced hash trigger mapper for strategy pathway determination.

This class provides sophisticated mapping from hash triggers to strategy pathways,
    integrating with existing systems while providing enhanced decision logic.
"""

    def __init__(self, max_mappings: int = 10000) -> None:
        """Initialize the hash trigger mapper."""
self.mappings: Dict[str, HashTriggerMapping] = {}
self.mapping_history: List[HashTriggerMapping] = []
self.max_mappings = max_mappings

        # Pattern detection
self.pattern_cache: Dict[str, HashPatternType] = {}

        # Integration flags
self.hash_trigger_engine_available = False
self.bit_resolution_engine_available = False
self.ghost_signal_available = False

        # Initialize with default mappings
self._initialize_default_mappings()

info("Hash Trigger Mapper initialized")

    def _initialize_default_mappings(self) -> None:
        """Initialize default hash trigger mappings."""
default_mappings = [
            # Aggressive patterns
("000000", "aggressive_ghost", "high", HashPatternType.CRITICAL),
            ("fff", "aggressive_ghost", "high", HashPatternType.CRITICAL),
            ("123456", "aggressive_ghost", "medium", HashPatternType.SEQUENTIAL),

            # Momentum patterns
("a1b2c3", "momentum_ghost", "medium", HashPatternType.PATTERNED),
            ("d4e5f6", "momentum_ghost", "medium", HashPatternType.PATTERNED),

            # Cautious patterns
("111111", "cautious_ghost", "low", HashPatternType.REPEATING),
            ("222222", "cautious_ghost", "low", HashPatternType.REPEATING),

            # Adaptive patterns
("abcde", "adaptive_ghost", "medium", HashPatternType.SEQUENTIAL),
            ("fedcba", "adaptive_ghost", "medium", HashPatternType.SEQUENTIAL),

            # Defensive patterns
("999999", "defensive_ghost", "high", HashPatternType.REPEATING),
            ("888888", "defensive_ghost", "high", HashPatternType.REPEATING),
        ]

        for hash_trigger, pathway, confidence, pattern_type in default_mappings:
mapping = HashTriggerMapping(
                hash_trigger=hash_trigger,
strategy_pathway=pathway,
confidence_level=confidence,
pattern_type=pattern_type,
mapping_score=0.7,
volatility_factor=0.5,
entropy_factor=0.5,
momentum_factor=0.5,
frequency_count=1,
last_seen=datetime.now(),
                average_interval=3600.0,
bit_phase_compatibility=["4bit", "8bit", "42bit"],
trigger_engine_compatible=True,
ghost_signal_compatible=True

self.mappings[hash_trigger] = mapping

info(f"Initialized {len(default_mappings)} default mappings")

    def map_hash_trigger(
        self,
hash_trigger: str,
market_data: Optional[Dict[str, Any]] = None,
ghost_signal_data: Optional[Dict[str, Any]] = None
) -> HashTriggerMapping:
"""
Map a hash trigger to a strategy pathway.

Args:
hash_trigger: The hash trigger to map
market_data: Optional market data for enhanced mapping
ghost_signal_data: Optional ghost signal data for integration

Returns:
HashTriggerMapping with strategy pathway and confidence
"""
        try:
            # Check if mapping already exists
            if hash_trigger in self.mappings:
mapping = self.mappings[hash_trigger]
self._update_mapping_frequency(mapping)
                return mapping

            # Analyze hash pattern
pattern_type = self._analyze_hash_pattern(hash_trigger)

            # Determine strategy pathway
strategy_pathway = self._determine_strategy_pathway(
                hash_trigger, pattern_type, market_data, ghost_signal_data


            # Calculate mapping confidence
confidence_level = self._calculate_mapping_confidence(
                hash_trigger, pattern_type, market_data


            # Calculate mapping factors
volatility_factor = market_data.get('volatility', 0.5) if market_data else 0.5
            entropy_factor = market_data.get('entropy', 0.5) if market_data else 0.5
            momentum_factor = market_data.get('momentum', 0.5) if market_data else 0.5

            # Calculate mapping score
mapping_score = self._calculate_mapping_score(
                pattern_type, confidence_level, volatility_factor, entropy_factor, momentum_factor


            # Create mapping
mapping = HashTriggerMapping(
                hash_trigger=hash_trigger,
strategy_pathway=strategy_pathway,
confidence_level=confidence_level,
pattern_type=pattern_type,
mapping_score=mapping_score,
volatility_factor=volatility_factor,
entropy_factor=entropy_factor,
momentum_factor=momentum_factor,
frequency_count=1,
last_seen=datetime.now(),
                average_interval=3600.0,
bit_phase_compatibility=self._determine_bit_phase_compatibility(hash_trigger),
                trigger_engine_compatible=self.hash_trigger_engine_available,
ghost_signal_compatible=self.ghost_signal_available


            # Store mapping
self.mappings[hash_trigger] = mapping
self.mapping_history.append(mapping)

            # Maintain mapping size
            if len(self.mappings) > self.max_mappings:
                oldest_key = next(iter(self.mappings))
                del self.mappings[oldest_key]

info(f"Mapped hash trigger {hash_trigger} to {strategy_pathway} (confidence: {confidence_level})")
            return mapping

        except Exception as e:
error(f"Error mapping hash trigger {hash_trigger}: {e}")
            return self._create_fallback_mapping(hash_trigger)

    def _analyze_hash_pattern(self, hash_trigger: str) -> HashPatternType:
        """Analyze the pattern type of a hash trigger."""
        try:
            # Check for sequential patterns
            if self._is_sequential(hash_trigger):
                return HashPatternType.SEQUENTIAL

            # Check for repeating patterns
            if self._is_repeating(hash_trigger):
                return HashPatternType.REPEATING

            # Check for critical patterns
            if self._is_critical(hash_trigger):
                return HashPatternType.CRITICAL

            # Check for patterned sequences
            if self._is_patterned(hash_trigger):
                return HashPatternType.PATTERNED

            # Default to random
            return HashPatternType.RANDOM

        except Exception as e:
error(f"Error analyzing hash pattern: {e}")
            return HashPatternType.RANDOM

    def _is_sequential(self, hash_trigger: str) -> bool:
        """Check if hash trigger has sequential pattern."""
        try:
            # Check for consecutive characters
            for i in range(len(hash_trigger) - 1):
                if ord(hash_trigger[i + 1]) - ord(hash_trigger[i]) == 1:
                    return True

            # Check for numeric sequences
            if hash_trigger.isdigit():
                for i in range(len(hash_trigger) - 1):
                    if int(hash_trigger[i + 1]) - int(hash_trigger[i]) == 1:
                        return True

            return False

        except Exception:
            return False

    def _is_repeating(self, hash_trigger: str) -> bool:
        """Check if hash trigger has repeating pattern."""
        try:
            # Check for all same characters
            if len(set(hash_trigger)) == 1:
                return True

            # Check for repeating pairs
            if len(hash_trigger) >= 4:
                for i in range(0, len(hash_trigger) - 2, 2):
                    if hash_trigger[i:i+2] == hash_trigger[i+2:i+4]:
                        return True

            return False

        except Exception:
            return False

    def _is_critical(self, hash_trigger: str) -> bool:
        """Check if hash trigger is critical pattern."""
        try:
critical_patterns = [
"000000", "fff", "111111", "999999",
"aaaaaa", "bbbbbb", "cccccc", "dddddd"
]
            return hash_trigger.lower() in critical_patterns

        except Exception:
            return False

    def _is_patterned(self, hash_trigger: str) -> bool:
        """Check if hash trigger has patterned sequence."""
        try:
            # Check for alternating patterns
            if len(hash_trigger) >= 4:
                pattern1 = hash_trigger[0]
pattern2 = hash_trigger[1]

                for i in range(2, len(hash_trigger), 2):
                    if i + 1 < len(hash_trigger):
                        if hash_trigger[i] != pattern1 or hash_trigger[i + 1] != pattern2:
                            return False
                    else:
                        if hash_trigger[i] != pattern1:
                            return False
                return True

            return False

        except Exception:
            return False

    def _determine_strategy_pathway(
        self,
hash_trigger: str,
pattern_type: HashPatternType,
market_data: Optional[Dict[str, Any]],
ghost_signal_data: Optional[Dict[str, Any]]
) -> StrategyPathway:
"""Determine strategy pathway based on hash trigger and context."""
        try:
            # Get market conditions
volatility = market_data.get('volatility', 0.5) if market_data else 0.5
            entropy = market_data.get('entropy', 0.5) if market_data else 0.5
            momentum = market_data.get('momentum', 0.5) if market_data else 0.5

            # Get ghost signal context
phase_state = ghost_signal_data.get('phase_state', 'dormant') if ghost_signal_data else 'dormant'
            signal_strength = ghost_signal_data.get('signal_strength', 0.5) if ghost_signal_data else 0.5

            # Pattern-based pathway determination
            if pattern_type == HashPatternType.CRITICAL:
                if volatility > 0.05 and entropy > 0.7:
                    return "defensive_ghost"
                else:
                    return "aggressive_ghost"

            elif pattern_type == HashPatternType.SEQUENTIAL:
                if momentum > 0.005 and signal_strength > 0.6:
                    return "momentum_ghost"
                else:
                    return "adaptive_ghost"

            elif pattern_type == HashPatternType.REPEATING:
                if entropy < 0.3 and volatility < 0.02:
                    return "cautious_ghost"
                else:
                    return "defensive_ghost"

            elif pattern_type == HashPatternType.PATTERNED:
                if phase_state == "resonant" and signal_strength > 0.4:
                    return "momentum_ghost"
                else:
                    return "adaptive_ghost"

            else:  # RANDOM
                if volatility > 0.03 or entropy > 0.6:
                    return "adaptive_ghost"
                else:
                    return "monitor_ghost"

        except Exception as e:
error(f"Error determining strategy pathway: {e}")
            return "monitor_ghost"

    def _calculate_mapping_confidence(
        self,
hash_trigger: str,
pattern_type: HashPatternType,
market_data: Optional[Dict[str, Any]]
) -> MappingConfidence:
"""Calculate mapping confidence level."""
        try:
            # Base confidence from pattern type
pattern_confidence = {
HashPatternType.CRITICAL: 0.9,
HashPatternType.SEQUENTIAL: 0.7,
HashPatternType.PATTERNED: 0.6,
HashPatternType.REPEATING: 0.5,
HashPatternType.RANDOM: 0.3
}

base_confidence = pattern_confidence.get(pattern_type, 0.5)

            # Adjust for hash length
length_factor = min(len(hash_trigger) / 6.0, 1.0)

            # Adjust for market conditions
market_factor = 0.5
            if market_data:
volatility = market_data.get('volatility', 0.5)
                entropy = market_data.get('entropy', 0.5)
                market_factor = (1.0 - volatility) * 0.6 + (1.0 - entropy) * 0.4

            # Calculate final confidence
final_confidence = (base_confidence * 0.5 + length_factor * 0.3 + market_factor * 0.2)

            # Map to confidence level
            if final_confidence >= 0.8:
                return "critical"
            elif final_confidence >= 0.6:
                return "high"
            elif final_confidence >= 0.4:
                return "medium"
            else:
                return "low"

        except Exception as e:
error(f"Error calculating mapping confidence: {e}")
            return "low"

    def _calculate_mapping_score(
        self,
pattern_type: HashPatternType,
confidence_level: MappingConfidence,
volatility_factor: float,
entropy_factor: float,
momentum_factor: float
) -> float:
"""Calculate overall mapping score."""
        try:
            # Pattern type weight
pattern_weights = {
HashPatternType.CRITICAL: 0.9,
HashPatternType.SEQUENTIAL: 0.7,
HashPatternType.PATTERNED: 0.6,
HashPatternType.REPEATING: 0.5,
HashPatternType.RANDOM: 0.3
}

pattern_score = pattern_weights.get(pattern_type, 0.5)

            # Confidence weight
confidence_weights = {
"critical": 0.9,
"high": 0.7,
"medium": 0.5,
"low": 0.3
}

confidence_score = confidence_weights.get(confidence_level, 0.5)

            # Market factors
market_score = (volatility_factor + entropy_factor + momentum_factor) / 3.0

            # Calculate final score
final_score = (
                pattern_score * 0.4 +
confidence_score * 0.3 +
market_score * 0.3


            return min(max(final_score, 0.0), 1.0)

        except Exception as e:
error(f"Error calculating mapping score: {e}")
            return 0.5

    def _determine_bit_phase_compatibility(self, hash_trigger: str) -> List[str]:
        """Determine bit phase compatibility for hash trigger."""
        try:
compatibility = []

            # 4-bit compatibility
            if len(hash_trigger) >= 4:
                compatibility.append("4bit")

            # 8-bit compatibility
            if len(hash_trigger) >= 8:
                compatibility.append("8bit")

            # 42-bit compatibility (always compatible)
            compatibility.append("42bit")

            return compatibility

        except Exception as e:
error(f"Error determining bit phase compatibility: {e}")
            return ["42bit"]

    def _update_mapping_frequency(self, mapping: HashTriggerMapping) -> None:
        """Update frequency count and timing for existing mapping."""
        try:
current_time = datetime.now()
            time_diff = (current_time - mapping.last_seen).total_seconds()

            # Update frequency count
mapping.frequency_count += 1

            # Update average interval
            if mapping.frequency_count > 1:
mapping.average_interval = (
                    (mapping.average_interval * (mapping.frequency_count - 1) + time_diff) /
                    mapping.frequency_count

            else:
mapping.average_interval = time_diff

            # Update last seen
mapping.last_seen = current_time

        except Exception as e:
error(f"Error updating mapping frequency: {e}")

    def _create_fallback_mapping(self, hash_trigger: str) -> HashTriggerMapping:
        """Create fallback mapping when normal mapping fails."""
        try:
            return HashTriggerMapping(
                hash_trigger=hash_trigger,
strategy_pathway="monitor_ghost",
confidence_level="low",
pattern_type=HashPatternType.RANDOM,
mapping_score=0.3,
volatility_factor=0.5,
entropy_factor=0.5,
momentum_factor=0.5,
frequency_count=1,
last_seen=datetime.now(),
                average_interval=3600.0,
bit_phase_compatibility=["42bit"],
trigger_engine_compatible=False,
ghost_signal_compatible=False


        except Exception as e:
error(f"Error creating fallback mapping: {e}")
            # Return minimal fallback
            return HashTriggerMapping(
                hash_trigger=hash_trigger,
strategy_pathway="monitor_ghost",
confidence_level="low",
pattern_type=HashPatternType.RANDOM,
mapping_score=0.1,
volatility_factor=0.5,
entropy_factor=0.5,
momentum_factor=0.5,
frequency_count=1,
last_seen=datetime.now(),
                average_interval=3600.0,
bit_phase_compatibility=["42bit"],
trigger_engine_compatible=False,
ghost_signal_compatible=False


    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get comprehensive mapping statistics."""
        try:
            if not self.mappings:
                return {"total_mappings": 0}

total_mappings = len(self.mappings)
            pathway_counts: Dict[str, int] = {}
confidence_counts: Dict[str, int] = {}
pattern_counts: Dict[str, int] = {}

            for mapping in self.mappings.values():
                # Count pathways
pathway_counts[mapping.strategy_pathway] = pathway_counts.get(mapping.strategy_pathway, 0) + 1

                # Count confidence levels
confidence_counts[mapping.confidence_level] = confidence_counts.get(mapping.confidence_level, 0) + 1

                # Count pattern types
pattern_counts[mapping.pattern_type.value] = pattern_counts.get(mapping.pattern_type.value, 0) + 1

            # Calculate averages
avg_mapping_score = unified_math.mean([m.mapping_score for m in self.mappings.values()])
            avg_frequency = unified_math.mean([m.frequency_count for m in self.mappings.values()])
            avg_interval = unified_math.mean([m.average_interval for m in self.mappings.values()])

            return {
"total_mappings": total_mappings,
"pathway_distribution": pathway_counts,
"confidence_distribution": confidence_counts,
"pattern_distribution": pattern_counts,
"average_mapping_score": avg_mapping_score,
"average_frequency": avg_frequency,
"average_interval": avg_interval
}

        except Exception as e:
error(f"Error getting mapping statistics: {e}")
            return {"error": str(e)}

    def clear_mappings(self) -> None:
        """Clear all mappings."""
        try:
self.mappings.clear()
            self.mapping_history.clear()
            self.pattern_cache.clear()
            info("All hash trigger mappings cleared")

        except Exception as e:
error(f"Error clearing mappings: {e}")


# Test function
def test_hash_trigger_mapper() -> None:
    """Test the hash trigger mapper functionality."""
    print("Testing Hash Trigger Mapper")
    print("=" * 50)

    # Initialize mapper
mapper = HashTriggerMapper()

    # Test hash triggers
test_triggers = [
"000000",  # Critical - should map to aggressive/defensive
"123456",  # Sequential - should map to momentum/adaptive
"a1b2c3",  # Patterned - should map to momentum/adaptive
"111111",  # Repeating - should map to cautious/defensive
"abcde",  # Sequential - should map to momentum/adaptive
"random1",  # Random - should map to adaptive/monitor
]

    for trigger in test_triggers:
        print(f"\nTesting trigger: {trigger}")

        # Create mock market data
market_data = {
"volatility": 0.025,
"entropy": 0.3,
"momentum": 0.003
}

        # Create mock ghost signal data
ghost_data = {
"phase_state": "active",
"signal_strength": 0.6
}

        # Map trigger
mapping = mapper.map_hash_trigger(trigger, market_data, ghost_data)

        print(f"  Strategy Pathway: {mapping.strategy_pathway}")
        print(f"  Confidence Level: {mapping.confidence_level}")
        print(f"  Pattern Type: {mapping.pattern_type.value}")
        print(f"  Mapping Score: {mapping.mapping_score:.4f}")
        print(f"  Frequency Count: {mapping.frequency_count}")

    # Get statistics
stats = mapper.get_mapping_statistics()
    print("\nStatistics:")
    print(f"  Total mappings: {stats['total_mappings']}")
    print(f"  Pathway distribution: {stats['pathway_distribution']}")
    print(f"  Average mapping score: {stats['average_mapping_score']:.4f}")

    print("\nHash Trigger Mapper test completed!")


if __name__ == "__main__":
test_hash_trigger_mapper()
