#!/usr/bin/env python3
"""
Ghost Strategy Integration - Enhanced Strategy Pathway Integration
================================================================

This module integrates HashTriggerMapper with GhostSignal to provide
enhanced strategy pathway determination and decision making.

Core Functionality:
- Integration between HashTriggerMapper and GhostSignal
- Enhanced strategy pathway determination
- Multi-factor decision logic
- Type-safe mathematical operations
- Unicode/emoji-safe CLI output
- Comprehensive error handling

This module provides the bridge between hash trigger mapping and ghost signal processing.
"""

import time
from typing import Dict, List, Optional, Any, Union, Literal, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import our robust systems with Unicode fallback
try:
    # Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}"), safe_math
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

# Import our modules
try:
    from core.ghost_signal import GhostSignal, GhostSignalProcessor
    from core.hash_trigger_mapper import HashTriggerMapper, HashTriggerMapping
except ImportError as e:
    error(f"Import error: {e}")
    # Create mock classes for testing
    class MockGhostSignal:
        """Mock GhostSignal for testing."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class MockGhostSignalProcessor:
        """Mock GhostSignalProcessor for testing."""
        def __init__(self):
            self.signal_history = []
            self.last_signal = None

        def create_signal(self, btc_vector, entropy, timestamp=None):
            signal = MockGhostSignal(
                price=btc_vector.mean_price,
                volatility=btc_vector.volatility,
                momentum=btc_vector.momentum,
                mean_price=btc_vector.mean_price,
                hash_trigger=btc_vector.hash_trigger,
                entropy=entropy,
                timestamp=timestamp or time.time(),
                phase_state="active",
                signal_strength=0.5,
                drift_direction="positive",
                drift_magnitude=0.001,
                resonance_score=0.6,
                hash_confidence=0.7,
                cycle_position=0.5,
                time_delta=1.0,
                frequency_score=0.3,
                suggested_pathway="adaptive_ghost",
                confidence_threshold=0.6,
                risk_level="medium"
            )
            self.signal_history.append(signal)
            self.last_signal = signal
            return signal

    class MockHashTriggerMapping:
        """Mock HashTriggerMapping for testing."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class MockHashTriggerMapper:
        """Mock HashTriggerMapper for testing."""
        def __init__(self):
            self.mappings = {}

        def map_hash_trigger(self, hash_trigger, market_data=None, ghost_signal_data=None):
            return MockHashTriggerMapping(
                hash_trigger=hash_trigger,
                strategy_pathway="adaptive_ghost",
                confidence_level="medium",
                pattern_type="random",
                mapping_score=0.6,
                volatility_factor=0.5,
                entropy_factor=0.5,
                momentum_factor=0.5,
                frequency_count=1,
                last_seen=datetime.now(),
                average_interval=3600.0,
                bit_phase_compatibility=["42bit"],
                trigger_engine_compatible=True,
                ghost_signal_compatible=True
            )

    GhostSignal = MockGhostSignal
    GhostSignalProcessor = MockGhostSignalProcessor
    HashTriggerMapping = MockHashTriggerMapping
    HashTriggerMapper = MockHashTriggerMapper

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
IntegrationMode = Literal["enhanced", "fallback", "hybrid"]
StrategyDecision = Literal["execute", "hold", "modify", "abort"]

class IntegrationStatus(Enum):
    """Integration status for the strategy system."""
    ACTIVE = "active"
    STANDBY = "standby"
    ERROR = "error"
    INITIALIZING = "initializing"


@dataclass
class EnhancedStrategyDecision:
    """
    Enhanced strategy decision with integration data.

    This dataclass represents a comprehensive strategy decision that combines
    hash trigger mapping with ghost signal analysis.
    """

    # Core decision data
    decision: StrategyDecision
    strategy_pathway: str
    confidence_score: float

    # Integration data
    hash_mapping: HashTriggerMapping
    ghost_signal: GhostSignal

    # Enhanced analysis data
    combined_score: float  # Combined score from both systems
    integration_mode: IntegrationMode
    decision_factors: Dict[str, float]

    # Timing and metadata
    timestamp: datetime
    processing_time: float  # Time taken to make decision
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary for serialization."""
        return {
            "decision": self.decision,
            "strategy_pathway": self.strategy_pathway,
            "confidence_score": self.confidence_score,
            "hash_mapping": self.hash_mapping.to_dict() if hasattr(self.hash_mapping, 'to_dict') else {},
            "ghost_signal": self.ghost_signal.to_dict() if hasattr(self.ghost_signal, 'to_dict') else {},
            "combined_score": self.combined_score,
            "integration_mode": self.integration_mode,
            "decision_factors": self.decision_factors,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }


class GhostStrategyIntegrator:
    """
    Enhanced strategy integrator that combines HashTriggerMapper with GhostSignal.

    This class provides sophisticated strategy decision making by integrating
    hash trigger mapping with ghost signal analysis.
    """

    def __init__(self, integration_mode: IntegrationMode = "enhanced") -> None:
        """Initialize the ghost strategy integrator."""
        self.integration_mode = integration_mode
        self.status = IntegrationStatus.INITIALIZING

        # Initialize components
        self.hash_mapper = HashTriggerMapper()
        self.ghost_processor = GhostSignalProcessor()

        # Decision history
        self.decision_history: List[EnhancedStrategyDecision] = []
        self.max_history = 1000

        # Performance tracking
        self.total_decisions = 0
        self.successful_decisions = 0
        self.average_processing_time = 0.0

        # Integration flags
        self.hash_mapper.ghost_signal_available = True

        self.status = IntegrationStatus.ACTIVE
        info("Ghost Strategy Integrator initialized")

    def make_enhanced_decision(
        self,
        btc_vector: Any,  # BTCVector type
        entropy: float,
        timestamp: Optional[float] = None
    ) -> EnhancedStrategyDecision:
        """
        Make an enhanced strategy decision using both hash mapping and ghost signals.

        Args:
            btc_vector: BTCVector instance with market data
            entropy: Current market entropy
            timestamp: Current timestamp (defaults to time.time())

        Returns:
            EnhancedStrategyDecision with comprehensive strategy information
        """
        start_time = time.time()

        try:
            if timestamp is None:
                timestamp = time.time()

            # Step 1: Create ghost signal
            ghost_signal = self.ghost_processor.create_signal(
                btc_vector=btc_vector,
                entropy=entropy,
                timestamp=timestamp
            )

            # Step 2: Create market data for hash mapping
            market_data = {
                "volatility": ghost_signal.volatility,
                "entropy": ghost_signal.entropy,
                "momentum": ghost_signal.momentum,
                "price": ghost_signal.price,
                "mean_price": ghost_signal.mean_price
            }

            # Step 3: Create ghost signal data for hash mapping
            ghost_data = {
                "phase_state": ghost_signal.phase_state,
                "signal_strength": getattr(ghost_signal, 'signal_strength', 0.5),
                "resonance_score": ghost_signal.resonance_score,
                "drift_direction": ghost_signal.drift_direction,
                "drift_magnitude": ghost_signal.drift_magnitude
            }

            # Step 4: Map hash trigger
            hash_mapping = self.hash_mapper.map_hash_trigger(
                hash_trigger=btc_vector.hash_trigger,
                market_data=market_data,
                ghost_signal_data=ghost_data
            )

            # Step 5: Integrate decisions
            decision = self._integrate_decisions(ghost_signal, hash_mapping)

            # Step 6: Calculate combined score
            combined_score = self._calculate_combined_score(ghost_signal, hash_mapping)

            # Step 7: Determine final strategy pathway
            final_pathway = self._determine_final_pathway(ghost_signal, hash_mapping, combined_score)

            # Step 8: Calculate decision factors
            decision_factors = self._calculate_decision_factors(ghost_signal, hash_mapping)

            # Step 9: Determine strategy decision
            strategy_decision = self._determine_strategy_decision(combined_score, decision_factors)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create enhanced decision
            enhanced_decision = EnhancedStrategyDecision(
                decision=strategy_decision,
                strategy_pathway=final_pathway,
                confidence_score=combined_score,
                hash_mapping=hash_mapping,
                ghost_signal=ghost_signal,
                combined_score=combined_score,
                integration_mode=self.integration_mode,
                decision_factors=decision_factors,
                timestamp=datetime.now(),
                processing_time=processing_time
            )

            # Update history and statistics
            self._update_statistics(enhanced_decision)

            info(f"Enhanced decision made: {strategy_decision} -> {final_pathway} (confidence: {combined_score:.4f})")
            return enhanced_decision

        except Exception as e:
            error(f"Error making enhanced decision: {e}")
            return self._create_fallback_decision(btc_vector, entropy, timestamp, start_time)

    def _integrate_decisions(
        self,
        ghost_signal: GhostSignal,
        hash_mapping: HashTriggerMapping
    ) -> Dict[str, Any]:
        """Integrate decisions from both ghost signal and hash mapping."""
        try:
            # Check for conflicts
            ghost_pathway = ghost_signal.suggested_pathway
            hash_pathway = hash_mapping.strategy_pathway

            # Calculate agreement score
            agreement_score = 1.0 if ghost_pathway == hash_pathway else 0.5

            # Determine integration mode
            if agreement_score > 0.8:
                integration_mode = "enhanced"
            elif agreement_score > 0.5:
                integration_mode = "hybrid"
            else:
                integration_mode = "fallback"

            return {
                "ghost_pathway": ghost_pathway,
                "hash_pathway": hash_pathway,
                "agreement_score": agreement_score,
                "integration_mode": integration_mode,
                "conflict_resolved": agreement_score < 1.0
            }

        except Exception as e:
            error(f"Error integrating decisions: {e}")
            return {
                "ghost_pathway": "monitor_ghost",
                "hash_pathway": "monitor_ghost",
                "agreement_score": 1.0,
                "integration_mode": "fallback",
                "conflict_resolved": False
            }

    def _calculate_combined_score(
        self,
        ghost_signal: GhostSignal,
        hash_mapping: HashTriggerMapping
    ) -> float:
        """Calculate combined confidence score from both systems."""
        try:
            # Ghost signal confidence factors
            ghost_confidence = ghost_signal.confidence_threshold
            ghost_resonance = ghost_signal.resonance_score
            ghost_hash_confidence = ghost_signal.hash_confidence

            # Hash mapping confidence factors
            hash_confidence = hash_mapping.mapping_score
            hash_pattern_confidence = self._get_pattern_confidence(hash_mapping.pattern_type)
            hash_frequency_confidence = min(hash_mapping.frequency_count / 10.0, 1.0)

            # Weighted combination
            ghost_score = (ghost_confidence * 0.4 + ghost_resonance * 0.3 + ghost_hash_confidence * 0.3)
            hash_score = (hash_confidence * 0.4 + hash_pattern_confidence * 0.3 + hash_frequency_confidence * 0.3)

            # Combined score with integration mode weighting
            if self.integration_mode == "enhanced":
                combined_score = (ghost_score * 0.6 + hash_score * 0.4)
            elif self.integration_mode == "hybrid":
                combined_score = (ghost_score * 0.5 + hash_score * 0.5)
            else:  # fallback
                combined_score = max(ghost_score, hash_score)

            return min(max(combined_score, 0.0), 1.0)

        except Exception as e:
            error(f"Error calculating combined score: {e}")
            return 0.5

    def _get_pattern_confidence(self, pattern_type: Any) -> float:
        """Get confidence score for pattern type."""
        try:
            pattern_confidences = {
                "critical": 0.9,
                "sequential": 0.7,
                "patterned": 0.6,
                "repeating": 0.5,
                "random": 0.3
            }

            pattern_name = pattern_type.value if hasattr(pattern_type, 'value') else str(pattern_type)
            return pattern_confidences.get(pattern_name, 0.5)

        except Exception as e:
            error(f"Error getting pattern confidence: {e}")
            return 0.5

    def _determine_final_pathway(
        self,
        ghost_signal: GhostSignal,
        hash_mapping: HashTriggerMapping,
        combined_score: float
    ) -> str:
        """Determine final strategy pathway based on integration."""
        try:
            ghost_pathway = ghost_signal.suggested_pathway
            hash_pathway = hash_mapping.strategy_pathway

            # High confidence agreement
            if combined_score > 0.8:
                if ghost_pathway == hash_pathway:
                    return ghost_pathway
                else:
                    # Prefer ghost signal for high confidence
                    return ghost_pathway

            # Medium confidence
            elif combined_score > 0.6:
                # Use hash mapping for medium confidence
                return hash_pathway

            # Low confidence
            else:
                # Use safer pathway
                safe_pathways = ["monitor_ghost", "cautious_ghost", "defensive_ghost"]
                if ghost_pathway in safe_pathways:
                    return ghost_pathway
                elif hash_pathway in safe_pathways:
                    return hash_pathway
                else:
                    return "monitor_ghost"

        except Exception as e:
            error(f"Error determining final pathway: {e}")
            return "monitor_ghost"

    def _calculate_decision_factors(
        self,
        ghost_signal: GhostSignal,
        hash_mapping: HashTriggerMapping
    ) -> Dict[str, float]:
        """Calculate decision factors for strategy determination."""
        try:
            return {
                "volatility_factor": ghost_signal.volatility,
                "entropy_factor": ghost_signal.entropy,
                "momentum_factor": ghost_signal.momentum,
                "resonance_factor": ghost_signal.resonance_score,
                "hash_confidence_factor": ghost_signal.hash_confidence,
                "mapping_score_factor": hash_mapping.mapping_score,
                "frequency_factor": min(hash_mapping.frequency_count / 10.0, 1.0),
                "drift_factor": ghost_signal.drift_magnitude,
                "cycle_position_factor": ghost_signal.cycle_position,
                "signal_strength_factor": getattr(ghost_signal, 'signal_strength', 0.5)
            }

        except Exception as e:
            error(f"Error calculating decision factors: {e}")
            return {
                "volatility_factor": 0.5,
                "entropy_factor": 0.5,
                "momentum_factor": 0.5,
                "resonance_factor": 0.5,
                "hash_confidence_factor": 0.5,
                "mapping_score_factor": 0.5,
                "frequency_factor": 0.5,
                "drift_factor": 0.5,
                "cycle_position_factor": 0.5,
                "signal_strength_factor": 0.5
            }

    def _determine_strategy_decision(
        self,
        combined_score: float,
        decision_factors: Dict[str, float]
    ) -> StrategyDecision:
        """Determine the final strategy decision."""
        try:
            # High confidence execution
            if combined_score > 0.8:
                return "execute"

            # Medium confidence with good factors
            elif combined_score > 0.6:
                # Check for favorable conditions
                favorable_factors = sum([
                    decision_factors.get("resonance_factor", 0.5) > 0.7,
                    decision_factors.get("hash_confidence_factor", 0.5) > 0.7,
                    decision_factors.get("mapping_score_factor", 0.5) > 0.7
                ])

                if favorable_factors >= 2:
                    return "execute"
                else:
                    return "hold"

            # Low confidence or poor conditions
            else:
                # Check for dangerous conditions
                dangerous_factors = sum([
                    decision_factors.get("volatility_factor", 0.5) > 0.8,
                    decision_factors.get("entropy_factor", 0.5) > 0.8,
                    decision_factors.get("drift_factor", 0.5) > 0.01
                ])

                if dangerous_factors >= 2:
                    return "abort"
                else:
                    return "hold"

        except Exception as e:
            error(f"Error determining strategy decision: {e}")
            return "hold"

    def _update_statistics(self, decision: EnhancedStrategyDecision) -> None:
        """Update integrator statistics."""
        try:
            # Update decision history
            self.decision_history.append(decision)
            self.total_decisions += 1

            # Maintain history size
            if len(self.decision_history) > self.max_history:
                self.decision_history = self.decision_history[-self.max_history:]

            # Update successful decisions
            if decision.decision in ["execute", "hold"]:
                self.successful_decisions += 1

            # Update average processing time
            if self.total_decisions == 1:
                self.average_processing_time = decision.processing_time
            else:
                self.average_processing_time = (
                    (self.average_processing_time * (self.total_decisions - 1) + decision.processing_time) /
                    self.total_decisions
                )

        except Exception as e:
            error(f"Error updating statistics: {e}")

    def _create_fallback_decision(
        self,
        btc_vector: Any,
        entropy: float,
        timestamp: float,
        start_time: float
    ) -> EnhancedStrategyDecision:
        """Create fallback decision when normal decision making fails."""
        try:
            # Create minimal ghost signal
            ghost_signal = GhostSignal(
                price=btc_vector.mean_price,
                volatility=btc_vector.volatility,
                momentum=btc_vector.momentum,
                mean_price=btc_vector.mean_price,
                hash_trigger=btc_vector.hash_trigger,
                entropy=entropy,
                timestamp=timestamp,
                phase_state="dormant",
                signal_strength=0.1,
                drift_direction="neutral",
                drift_magnitude=0.0,
                resonance_score=0.1,
                hash_confidence=0.1,
                cycle_position=0.5,
                time_delta=0.0,
                frequency_score=0.0,
                suggested_pathway="monitor_ghost",
                confidence_threshold=0.1,
                risk_level="high"
            )

            # Create minimal hash mapping
            hash_mapping = HashTriggerMapping(
                hash_trigger=btc_vector.hash_trigger,
                strategy_pathway="monitor_ghost",
                confidence_level="low",
                pattern_type="random",
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
            )

            processing_time = time.time() - start_time

            return EnhancedStrategyDecision(
                decision="hold",
                strategy_pathway="monitor_ghost",
                confidence_score=0.1,
                hash_mapping=hash_mapping,
                ghost_signal=ghost_signal,
                combined_score=0.1,
                integration_mode="fallback",
                decision_factors={"fallback": 1.0},
                timestamp=datetime.now(),
                processing_time=processing_time
            )

        except Exception as e:
            error(f"Error creating fallback decision: {e}")
            # Return minimal fallback
            return EnhancedStrategyDecision(
                decision="abort",
                strategy_pathway="monitor_ghost",
                confidence_score=0.0,
                hash_mapping=None,
                ghost_signal=None,
                combined_score=0.0,
                integration_mode="fallback",
                decision_factors={"error": 1.0},
                timestamp=datetime.now(),
                processing_time=time.time() - start_time
            )

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        try:
            if not self.decision_history:
                return {"total_decisions": 0}

            # Basic statistics
            total_decisions = len(self.decision_history)
            success_rate = self.successful_decisions / self.total_decisions if self.total_decisions > 0 else 0.0

            # Decision distribution
            decision_counts: Dict[str, int] = {}
            pathway_counts: Dict[str, int] = {}
            mode_counts: Dict[str, int] = {}

            for decision in self.decision_history:
                # Count decisions
                decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1

                # Count pathways
                pathway_counts[decision.strategy_pathway] = pathway_counts.get(decision.strategy_pathway, 0) + 1

                # Count integration modes
                mode_counts[decision.integration_mode] = mode_counts.get(decision.integration_mode, 0) + 1

            # Calculate averages
            avg_confidence = unified_math.mean([d.confidence_score for d in self.decision_history])
            avg_combined_score = unified_math.mean([d.combined_score for d in self.decision_history])

            return {
                "total_decisions": total_decisions,
                "successful_decisions": self.successful_decisions,
                "success_rate": success_rate,
                "average_processing_time": self.average_processing_time,
                "decision_distribution": decision_counts,
                "pathway_distribution": pathway_counts,
                "integration_mode_distribution": mode_counts,
                "average_confidence": avg_confidence,
                "average_combined_score": avg_combined_score,
                "status": self.status.value
            }

        except Exception as e:
            error(f"Error getting integration statistics: {e}")
            return {"error": str(e)}

    def clear_history(self) -> None:
        """Clear decision history."""
        try:
            self.decision_history.clear()
            self.total_decisions = 0
            self.successful_decisions = 0
            self.average_processing_time = 0.0
            info("Decision history cleared")

        except Exception as e:
            error(f"Error clearing history: {e}")


# Test function
def test_ghost_strategy_integration() -> None:
    """Test the ghost strategy integration functionality."""
    print("Testing Ghost Strategy Integration")
    print("=" * 50)

    # Initialize integrator
    integrator = GhostStrategyIntegrator()

    # Create mock BTCVector
    class MockBTCVector:
        """Mock BTCVector for testing."""

        def __init__(self) -> None:
            self.price = 50000.0
            self.volatility = 0.025
            self.momentum = 0.003
            self.mean_price = 50000.0
            self.hash_trigger = "a1b2c3"

    # Test different scenarios
    test_scenarios = [
        {"name": "Low Risk", "entropy": 0.2, "volatility": 0.01, "momentum": 0.001},
        {"name": "Medium Risk", "entropy": 0.5, "volatility": 0.025, "momentum": 0.003},
        {"name": "High Risk", "entropy": 0.8, "volatility": 0.06, "momentum": 0.01},
    ]

    for scenario in test_scenarios:
        print(f"\nTesting scenario: {scenario['name']}")

        # Create BTCVector with scenario conditions
        btc_vector = MockBTCVector()
        btc_vector.volatility = scenario['volatility']
        btc_vector.momentum = scenario['momentum']

        # Make enhanced decision
        decision = integrator.make_enhanced_decision(
            btc_vector=btc_vector,
            entropy=scenario['entropy'],
            timestamp=time.time()
        )

        print(f"  Decision: {decision.decision}")
        print(f"  Strategy Pathway: {decision.strategy_pathway}")
        print(f"  Confidence Score: {decision.confidence_score:.4f}")
        print(f"  Combined Score: {decision.combined_score:.4f}")
        print(f"  Integration Mode: {decision.integration_mode}")
        print(f"  Processing Time: {decision.processing_time:.4f}s")

    # Get statistics
    stats = integrator.get_integration_statistics()
    print(f"\nIntegration Statistics:")
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Average processing time: {stats['average_processing_time']:.4f}s")
    print(f"  Decision distribution: {stats['decision_distribution']}")
    print(f"  Pathway distribution: {stats['pathway_distribution']}")

    print("\nGhost Strategy Integration test completed!")


if __name__ == "__main__":
    test_ghost_strategy_integration()
