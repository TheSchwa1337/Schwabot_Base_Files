"""
Matrix Fault Resolver
===================

Handles matrix transitions and fault resolution with ZPE risk awareness
and drift loop detection. Integrates with cursor math for stability analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import yaml
from pathlib import Path
from .cursor_math_integration import CursorMath, PhaseShell
import logging
from tools.phase_hash_utils import match_cyclic_number, Δt_prime_drift
from tools.euler_trigger_index import EulerSignatureMap
from .fractal_core import ForeverFractalCore, FractalState
from .triplet_matcher import TripletMatcher, TripletMatch
from .config import load_yaml_config, ConfigError, MATRIX_RESPONSE_SCHEMA
import time
import numpy as np
from collections import deque

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class FaultState:
    """Represents a matrix fault state"""
    fault_type: str
    severity: float
    matrix_state: str
    phase_shell: PhaseShell
    entropy_class: str
    timestamp: float
    fractal_state: Optional[FractalState] = None

@dataclass
class FractalState:
    """Represents a fractal truth-memory node state"""
    node_id: int
    truth_value: float
    phase: float
    entropy: float
    timestamp: float
    recursive_depth: int = 0

@dataclass
class TripletState:
    """Represents a triplet state for recursive matrix logic"""
    i: FractalState
    j: FractalState
    k: FractalState
    coherence_score: float
    timestamp: float

class RecursiveMemoryLayer:
    """Handles persistent storage and retrieval of fractal patterns"""
    
    def __init__(self, max_depth: int = 10, decay_factor: float = 0.95):
        self.max_depth = max_depth
        self.decay_factor = decay_factor
        self.pattern_store: Dict[Tuple[int], List[Tuple[float, float]]] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # 1 hour
        
    def store_pattern(self, pattern_key: Tuple[int], 
                     coherence: float, timestamp: float) -> None:
        """
        Store a fractal pattern with its coherence score
        
        Args:
            pattern_key: Quantized pattern key
            coherence: Pattern coherence score
            timestamp: Pattern timestamp
        """
        if pattern_key not in self.pattern_store:
            self.pattern_store[pattern_key] = []
            
        self.pattern_store[pattern_key].append((coherence, timestamp))
        
        # Cleanup old patterns periodically
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_patterns()
    
    def _cleanup_old_patterns(self) -> None:
        """Remove old patterns and decay coherence scores"""
        current_time = time.time()
        self.last_cleanup = current_time
        
        for pattern_key in list(self.pattern_store.keys()):
            # Filter and decay patterns
            updated_patterns = []
            for coherence, timestamp in self.pattern_store[pattern_key]:
                # Skip if too old
                if current_time - timestamp > 24 * 3600:  # 24 hours
                    continue
                    
                # Apply decay
                age = current_time - timestamp
                decayed_coherence = coherence * (self.decay_factor ** age)
                
                if decayed_coherence > 0.1:  # Keep if still significant
                    updated_patterns.append((decayed_coherence, timestamp))
            
            # Update or remove pattern
            if updated_patterns:
                self.pattern_store[pattern_key] = updated_patterns
            else:
                del self.pattern_store[pattern_key]
    
    def find_similar_pattern(self, pattern_key: Tuple[int], 
                           min_coherence: float = 0.7) -> Optional[Tuple[float, float]]:
        """
        Find a similar pattern in storage
        
        Args:
            pattern_key: Pattern to match
            min_coherence: Minimum coherence threshold
            
        Returns:
            (coherence, timestamp) tuple if found, None otherwise
        """
        if pattern_key not in self.pattern_store:
            return None
            
        # Get most recent pattern with sufficient coherence
        patterns = self.pattern_store[pattern_key]
        if not patterns:
            return None
            
        # Sort by timestamp descending
        patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Find first pattern meeting coherence threshold
        for coherence, timestamp in patterns:
            if coherence >= min_coherence:
                return coherence, timestamp
                
        return None
    
    def get_pattern_stats(self) -> Dict[str, float]:
        """Get statistics about stored patterns"""
        if not self.pattern_store:
            return {}
            
        total_patterns = sum(len(patterns) for patterns in self.pattern_store.values())
        total_unique = len(self.pattern_store)
        
        # Calculate average coherence
        total_coherence = 0
        pattern_count = 0
        for patterns in self.pattern_store.values():
            for coherence, _ in patterns:
                total_coherence += coherence
                pattern_count += 1
                
        avg_coherence = total_coherence / pattern_count if pattern_count > 0 else 0
        
        return {
            'total_patterns': total_patterns,
            'unique_patterns': total_unique,
            'avg_coherence': avg_coherence
        }

class MatrixFaultResolver:
    """Resolves matrix faults with ZPE risk awareness and fractal integration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.math_core = CursorMath()
        self.fault_history: List[FaultState] = []
        self.current_matrix: str = 'matrix_safe'
        
        # Initialize fractal components
        self.fractal_core = ForeverFractalCore(
            decay_power=2.0,
            terms=50,
            dimension=3
        )
        self.triplet_matcher = TripletMatcher(
            fractal_core=self.fractal_core,
            epsilon=0.1,
            min_coherence=0.7
        )
        
        # Load fallback strategies
        self.fallback_strategies = self._load_fallback_strategies(config_path)
    
    def _load_fallback_strategies(self, config_path: Optional[str]) -> Dict:
        """Load fallback strategies from YAML config"""
        try:
            if config_path is None:
                # Use centralized config loader with schema validation
                return load_yaml_config('matrix_response_paths.yaml', schema=MATRIX_RESPONSE_SCHEMA)
            else:
                # Use provided config path
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    # Validate against schema
                    validate_config(config, MATRIX_RESPONSE_SCHEMA)
                    return config
        except (ConfigError, FileNotFoundError) as e:
            logging.warning(f"Error loading config: {e}, using default fallback strategies")
            return MATRIX_RESPONSE_SCHEMA.default_values
    
    def handle_fault(self, fault_type: str, severity: float,
                    phase_angle: float, z_score: float) -> Tuple[str, str]:
        """
        Handle matrix fault with ZPE risk awareness and fractal integration.
        
        Args:
            fault_type: Type of fault
            severity: Fault severity (0-1)
            phase_angle: Current phase angle
            z_score: Current Z-score
            
        Returns:
            (target_matrix, fallback_strategy) tuple
        """
        # Classify phase shell
        phase_shell = self.math_core.classify_phase_shell(phase_angle)
        
        # Classify entropy
        entropy_class = self.math_core.classify_entropy_shell(z_score)
        
        # Generate fractal state
        fractal_vector = self.fractal_core.generate_fractal_vector(
            t=time.time(),
            phase_shift=phase_angle
        )
        
        fractal_state = FractalState(
            vector=fractal_vector,
            timestamp=time.time(),
            phase=phase_angle,
            entropy=z_score
        )
        
        # Store state in fractal core
        self.fractal_core.store_state(
            vector=fractal_vector,
            phase=phase_angle,
            entropy=z_score
        )
        
        # Check for triplet matches
        recent_states = self.fractal_core.get_recent_states(count=3)
        if len(recent_states) == 3:
            match = self.triplet_matcher.find_matching_triplet(recent_states)
            if match:
                # Apply fractal correction
                corrected_state = self._apply_fractal_correction(match)
                
                # Update matrix state
                matrix_state = self.math_core.compute_matrix_stability(
                    phase_shell,
                    entropy_class,
                    corrected_state
                )
                
                # Check for collapse
                if self.triplet_matcher.check_collapse():
                    return matrix_state, 'fractal_collapse'
                    
                return matrix_state, 'fractal_corrected'
        
        # Create fault state
        fault = FaultState(
            fault_type=fault_type,
            severity=severity,
            matrix_state=self.current_matrix,
            phase_shell=phase_shell,
            entropy_class=entropy_class,
            timestamp=time.time(),
            fractal_state=fractal_state
        )
        
        # Add to history
        self.fault_history.append(fault)
        
        # Compute matrix stability
        matrix_state = self.math_core.compute_matrix_stability(phase_shell, entropy_class)
        
        return matrix_state, 'matrix_safe'
    
    def _apply_fractal_correction(self, match: TripletMatch) -> List[float]:
        """
        Apply fractal-based correction to matrix state.
        
        Args:
            match: Matched triplet
            
        Returns:
            Corrected state vector
        """
        # Get correction matrix
        correction_matrix = self.math_core.get_correction_matrix()
        
        # Compute weighted correction
        weights = [1.0, 0.5, 0.25]  # Weight recent states more heavily
        corrected = np.zeros_like(match.states[0].vector)
        
        for state, weight in zip(match.states, weights):
            corrected += weight * np.array(state.vector)
            
        corrected /= sum(weights)
        
        return corrected.tolist()
    
    def get_fractal_stats(self) -> Dict[str, float]:
        """
        Get statistics about fractal patterns and matches.
        
        Returns:
            Dictionary of statistics
        """
        match_stats = self.triplet_matcher.get_match_stats()
        
        return {
            'match_stats': match_stats,
            'fractal_states': len(self.fractal_core.state_history),
            'mirror_patterns': len(self.fractal_core.mirror_memory)
        }
    
    def register_mirror_fractal(self, triplet: TripletState) -> None:
        """
        Store mirrored version of fractal triplet for recursive echo detection.
        
        Args:
            triplet: Triplet state to mirror
        """
        # Create mirror key by reversing the triplet
        mirror_key = self.fractal_core.quantize_fractal_state([
            triplet.k.truth_value,
            triplet.j.truth_value,
            triplet.i.truth_value
        ])
        
        # Store with timestamp
        self.mirror_log[mirror_key] = time.time()
    
    def check_mirror_match(self, current_triplet: TripletState) -> bool:
        """
        Check if current triplet matches a mirrored pattern.
        
        Args:
            current_triplet: Current triplet to check
            
        Returns:
            True if mirror match found
        """
        # Create mirror key
        mirror_key = self.fractal_core.quantize_fractal_state([
            current_triplet.k.truth_value,
            current_triplet.j.truth_value,
            current_triplet.i.truth_value
        ])
        
        # Check if mirror exists and is recent
        if mirror_key in self.mirror_log:
            mirror_time = self.mirror_log[mirror_key]
            if time.time() - mirror_time < 3600:  # 1 hour
                return True
                
        return False
    
    def get_fault_history(self, limit: Optional[int] = None) -> List[FaultState]:
        """Get recent fault history"""
        if limit is None:
            return self.fault_history
        return self.fault_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear fault history"""
        self.fault_history.clear()
        self.current_matrix = 'matrix_safe'
    
    def get_fault_stats(self) -> Dict[str, float]:
        """Get statistics about fault types"""
        if not self.fault_history:
            return {}
            
        total = len(self.fault_history)
        stats = {}
        
        for state in self.fault_history:
            if state.fault_type not in stats:
                stats[state.fault_type] = 0
            stats[state.fault_type] += 1
            
        return {k: v/total for k, v in stats.items()}
    
    def get_matrix_transitions(self) -> List[Tuple[str, str]]:
        """Get history of matrix transitions"""
        transitions = []
        prev_matrix = None
        
        for state in self.fault_history:
            if prev_matrix is not None and prev_matrix != state.matrix_state:
                transitions.append((
                    prev_matrix,
                    state.matrix_state,
                    state.timestamp
                ))
            prev_matrix = state.matrix_state
            
        return transitions

    def inject_mode(self, mode="euler_realign"):
        if mode == "euler_realign":
            # Store the current hash in euler_ring_set[]
            self.push_fault(node_id)
            print(f"Node {node_id} is part of an Euler-coded decay and will be stored in euler_ring_set[].")

class QRingHandler:
    def __init__(self):
        self.ring = []
        self.cursor = 0
        self.euler_ring_set = []  # New list to store Euler-stable hashes
        self.euler_priority = True  # Flag to enable Euler priority

    def push_fault(self, node_id):
        self.ring.append(node_id)
        if self.euler_ring_set and self.euler_ring_set[-1] == node_id:
            self.euler_ring_set.pop()  # Remove duplicates
        self.euler_ring_set.append(node_id)  # Store in euler_ring_set

    def next(self):
        if not self.ring:
            return None
            
        if self.euler_priority and self.euler_ring_set:
            # Prefer Euler-stable hashes
            return self.euler_ring_set[0]
            
        self.cursor = (self.cursor + 1) % len(self.ring)
        return self.ring[self.cursor]

    def prefer_euler(self, enable=True):
        """Enable/disable Euler priority in ring selection"""
        self.euler_priority = enable

    def inject_mode(self, mode="euler_realign"):
        if mode == "euler_realign":
            # Store the current hash in euler_ring_set[]
            node_id = self.next()
            if node_id is not None:
                self.push_fault(node_id)
                print(f"Node {node_id} is part of an Euler-coded decay and will be stored in euler_ring_set[].")

class CyclicHashClassifier:
    def __init__(self, cyclic_numbers_file):
        with open(cyclic_numbers_file, 'r') as file:
            self.cyclic_numbers = set(int(line.strip()) for line in file)

    def classify(self, hash_value):
        return match_cyclic_number(hash_value, self.cyclic_numbers)

class EulerSignatureMap:
    def __init__(self, euler_triggers_file):
        with open(euler_triggers_file, 'r') as file:
            self.triggers = {}
            for line in file:
                node_id, signature = line.strip().split(',')
                self.triggers[int(node_id)] = signature

    def compare_signature(self, profile, node_id):
        if node_id in self.triggers:
            return profile == self.triggers[node_id]
        return False

    def compare_shape_drift(self, profile, collapse_curve):
        """Compare drift shape against Euler-coded profiles"""
        if not collapse_curve:
            return False

        # Calculate drift characteristics
        drift_magnitude = sum(abs(x) for x in collapse_curve)
        drift_direction = sum(x for x in collapse_curve) / len(collapse_curve)
        
        # Compare against profile characteristics
        profile_magnitude = sum(abs(x) for x in profile)
        profile_direction = sum(x for x in profile) / len(profile)
        
        # Check if drift matches profile characteristics
        magnitude_match = abs(drift_magnitude - profile_magnitude) < 0.1
        direction_match = abs(drift_direction - profile_direction) < 0.1
        
        return magnitude_match and direction_match

    def similarity_score(self, current_hash, target_hash):
        """Calculate similarity score between hashes"""
        # XOR the hashes and count set bits
        xor_result = current_hash ^ target_hash
        set_bits = bin(xor_result).count('1')
        
        # Normalize score to [0,1] range
        max_bits = 64  # Assuming 64-bit hashes
        return 1.0 - (set_bits / max_bits)

class DriftTracker:
    def __init__(self, max_drift=0.5):
        self.max_drift = max_drift
        self.drift_vectors = {}

    def update_drift(self, node_id, current_hash, previous_hash):
        if node_id not in self.drift_vectors:
            self.drift_vectors[node_id] = []

        # Calculate the drift vector using Δt_prime_drift
        drift_vector = Δt_prime_drift(current_hash, previous_hash)

        # Normalize the drift vector to a unit vector
        magnitude = abs(drift_vector)
        if magnitude > 0:
            normalized_drift_vector = drift_vector / magnitude
        else:
            normalized_drift_vector = (0, 0)  # Handle zero drift

        self.drift_vectors[node_id].append(normalized_drift_vector)

    def get_drift_vectors(self):
        return self.drift_vectors

    def get_recent_drift(self, node_id, num_samples=5):
        if node_id in self.drift_vectors:
            return self.drift_vectors[node_id][-num_samples:]
        return []

class MatrixGradientAligner:
    def __init__(self, substrate_vector=(1, 0)):  # assume ideal lattice vector
        self.substrate_vector = substrate_vector
        self.braid_vectors = []  # Store braided mechanophore vectors
        self.beta_index = 0  # Braid index (β(B))
        self.closure_threshold = 0.1  # Threshold for braid closure

    def cosine_similarity(self, vec1, vec2):
        dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        norm1 = (vec1[0]**2 + vec1[1]**2)**0.5
        norm2 = (vec2[0]**2 + vec2[1]**2)**0.5
        return dot / (norm1 * norm2 + 1e-6)

    def is_aligned(self, incoming_vector, threshold=0.85):
        sim = self.cosine_similarity(incoming_vector, self.substrate_vector)
        return sim >= threshold

    def add_braid_vector(self, vector):
        """Add a vector to the braid sequence"""
        self.braid_vectors.append(vector)
        self._update_beta_index()

    def _update_beta_index(self):
        """Calculate braid index (β(B)) based on vector crossovers"""
        if len(self.braid_vectors) < 2:
            self.beta_index = 0
            return

        crossovers = 0
        for i in range(len(self.braid_vectors) - 1):
            v1 = self.braid_vectors[i]
            v2 = self.braid_vectors[i + 1]
            # Calculate cross product to detect direction changes
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) > 0.1:  # Significant direction change
                crossovers += 1

        self.beta_index = crossovers

    def check_braid_closure(self):
        """Check if braid sequence forms a closed loop"""
        if len(self.braid_vectors) < 3:
            return False

        # Sum all vectors
        sum_vector = [0, 0]
        for v in self.braid_vectors:
            sum_vector[0] += v[0]
            sum_vector[1] += v[1]

        # Check if sum is close to zero (closed loop)
        magnitude = (sum_vector[0]**2 + sum_vector[1]**2)**0.5
        return magnitude < self.closure_threshold

    def get_braid_metrics(self):
        """Get braid metrics including β index and closure status"""
        return {
            'beta_index': self.beta_index,
            'is_closed': self.check_braid_closure(),
            'vector_count': len(self.braid_vectors)
        }

class EulerRealignHandler:
    def __init__(self, euler_triggers_file):
        self.euler_resolver = EulerSignatureMap(euler_triggers_file)
        self.ring_handler = QRingHandler()
        self.drift_tracker = DriftTracker(max_drift=0.5)
        self.aligner = MatrixGradientAligner()
        self.logger = RecursiveEnvelopeLogger()
        self.cyclic_classifier = CyclicHashClassifier('cyclicNumbers.txt')
        self.zpe_threshold = 0.15  # ZPE risk threshold

    def compute_reentry_vector(self, current_hash, previous_hash):
        # Calculate the drift vector using Δt_prime_drift
        drift_vector = Δt_prime_drift(current_hash, previous_hash)

        # Normalize the drift vector to a unit vector
        magnitude = abs(drift_vector)
        if magnitude > 0:
            normalized_drift_vector = drift_vector / magnitude
        else:
            normalized_drift_vector = (0, 0)  # Handle zero drift

        return normalized_drift_vector

    def perform_xor_drift_test(self, current_hash, previous_known_hash):
        xor_result = current_hash ^ previous_known_hash
        return xor_result

    def inject_into_fault_resolver(self, node_id, current_hash, previous_hash, collapse_curve=None):
        """
        Enhanced fault resolution with recursive harmonic correction
        
        Args:
            node_id: Node identifier
            current_hash: Current hash value
            previous_hash: Previous hash value
            collapse_curve: Optional collapse curve for drift analysis
        """
        # 1. Check Euler-coded trigger match
        if self.euler_resolver.compare_signature('signature_profile', node_id):
            print("Current hash matches an Euler-coded decay.")
            self.ring_handler.push_fault(node_id)
            
            # Compute and analyze reentry vector
            reentry_vector = self.compute_reentry_vector(current_hash, previous_hash)
            self.aligner.add_braid_vector(reentry_vector)
            
            # Log braid metrics
            braid_metrics = self.aligner.get_braid_metrics()
            self.logger.drift_braid_logger(
                self.aligner.braid_vectors,
                braid_metrics['beta_index'],
                braid_metrics['is_closed']
            )
            
            # Check braid closure
            if self.aligner.check_braid_closure():
                print("Braid closure detected - retaining in memory")
                return "retain"
            
            # Check lattice alignment
            if self.aligner.is_aligned(reentry_vector):
                print("Reentry vector is aligned with lattice. Proceeding with reinjection.")
                return "reinject"
            else:
                print("Vector misaligned. Entering spiral holding pattern.")
                self.hold_node_in_spiral_loop(node_id)
                return "spiral_hold"
        
        # 2. Check cyclic number pattern
        if self.cyclic_classifier.classify(current_hash):
            print("Node is part of a cyclic pattern - delaying execution")
            return "delay"
        
        # 3. Check ZPE risk
        drift_vector = self.compute_reentry_vector(current_hash, previous_hash)
        if abs(drift_vector) > self.zpe_threshold:
            print("ZPE risk detected - entering bloom loop")
            return "bloom_hold"
        
        # 4. Check collapse curve if provided
        if collapse_curve:
            if self.euler_resolver.compare_shape_drift('signature_profile', collapse_curve):
                print("Collapse curve matches Euler profile - realigning")
                return "realign"
        
        # 5. Default to fallback if no special conditions met
        print("No special conditions met - triggering fallback")
        return "fallback"

    def hold_node_in_spiral_loop(self, node_id):
        """Enhanced spiral holding pattern with recursive delay"""
        print(f"Node {node_id} is being held in a spiral loop for reentry.")
        
        # Initialize spiral parameters
        delay_base = 0.1  # Base delay in seconds
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            # Calculate recursive delay
            delay = delay_base * (2 ** attempt)  # Exponential backoff
            
            # Check if node can be released
            if self._check_spiral_release(node_id):
                print(f"Node {node_id} released from spiral loop")
                return True
            
            # Apply delay
            time.sleep(delay)
            attempt += 1
        
        print(f"Node {node_id} exceeded maximum spiral attempts")
        return False

    def _check_spiral_release(self, node_id):
        """Check if node can be released from spiral loop"""
        # Get recent drift vectors
        recent_drifts = self.drift_tracker.get_recent_drift(node_id)
        
        if not recent_drifts:
            return False
        
        # Check if drift is stabilizing
        last_drift = recent_drifts[-1]
        if abs(last_drift) < 0.05:  # Drift threshold
            return True
        
        return False

class RecursiveEnvelopeLogger:
    def __init__(self):
        self.drift_data = []
        self.braid_logs = []  # Store braid-specific logs
        self.epsilon_threshold = 0.1  # Example epsilon threshold

    def log_drift(self, origin_hash, collapse_curve, entropy_loss, drift_type):
        entry = {
            "origin": origin_hash,
            "collapse_curve": collapse_curve,
            "entropy_loss": entropy_loss,
            "drift_type": drift_type
        }
        self.drift_data.append(entry)

    def drift_braid_logger(self, braid_vectors, beta_index, closure_status):
        """Log braid-specific drift information"""
        entry = {
            "timestamp": time.time(),
            "braid_vectors": braid_vectors,
            "beta_index": beta_index,
            "closure_status": closure_status
        }
        self.braid_logs.append(entry)

    def closure_checker(self, braid_vectors):
        """Check if a sequence of vectors forms a closed braid"""
        if len(braid_vectors) < 3:
            return False

        # Sum all vectors
        sum_vector = [0, 0]
        for v in braid_vectors:
            sum_vector[0] += v[0]
            sum_vector[1] += v[1]

        # Check if sum is close to zero (closed loop)
        magnitude = (sum_vector[0]**2 + sum_vector[1]**2)**0.5
        return magnitude < self.epsilon_threshold

    def get_braid_logs(self):
        """Get all braid-specific logs"""
        return self.braid_logs

class ForeverFractal:
    """Implements the Forever Fractal mathematical framework"""
    
    def __init__(self, decay_power: float = 2.0, epsilon: float = 0.1):
        self.decay_power = decay_power
        self.epsilon = epsilon
        self.truth_memory: Dict[int, FractalState] = {}
        self.triplet_history: List[TripletState] = []
        self.recursive_depth = 0
        self.max_depth = 10
        
    def compute_truth_function(self, t: float, node_id: int) -> float:
        """
        Compute the fractal truth-function 𝔉(t) for a given node
        
        Args:
            t: Recursive time parameter
            node_id: Node identifier
            
        Returns:
            Truth function value
        """
        if node_id not in self.truth_memory:
            return 0.0
            
        # Get node state
        state = self.truth_memory[node_id]
        
        # Compute recursive sum
        truth_sum = 0.0
        for n in range(self.recursive_depth + 1):
            # Compute Ψ_n(t) component
            psi_n = self._compute_psi_n(t, n, state)
            # Add to sum with decay
            truth_sum += (1.0 / (n + 1) ** self.decay_power) * psi_n
            
        return truth_sum
    
    def _compute_psi_n(self, t: float, n: int, state: FractalState) -> float:
        """Compute the nth recursive truth-memory node state"""
        # Base case
        if n == 0:
            return state.truth_value
            
        # Recursive case - compute phase-shifted state
        phase_shift = 2 * np.pi * n / (self.recursive_depth + 1)
        return state.truth_value * np.cos(phase_shift + state.phase)
    
    def register_triplet(self, i: FractalState, j: FractalState, k: FractalState) -> TripletState:
        """
        Register a new triplet state
        
        Args:
            i, j, k: Fractal states forming the triplet
            
        Returns:
            Registered triplet state
        """
        # Compute coherence score
        coherence = self._compute_triplet_coherence(i, j, k)
        
        # Create triplet state
        triplet = TripletState(
            i=i,
            j=j,
            k=k,
            coherence_score=coherence,
            timestamp=time.time()
        )
        
        # Store in history
        self.triplet_history.append(triplet)
        
        return triplet
    
    def _compute_triplet_coherence(self, i: FractalState, j: FractalState, k: FractalState) -> float:
        """Compute coherence score between triplet states"""
        # Compute phase differences
        phase_diff_ij = abs(i.phase - j.phase)
        phase_diff_jk = abs(j.phase - k.phase)
        
        # Compute entropy differences
        entropy_diff_ij = abs(i.entropy - j.entropy)
        entropy_diff_jk = abs(j.entropy - k.entropy)
        
        # Combine into coherence score
        coherence = 1.0 - (
            0.5 * (phase_diff_ij + phase_diff_jk) / (2 * np.pi) +
            0.5 * (entropy_diff_ij + entropy_diff_jk)
        )
        
        return max(0.0, min(1.0, coherence))
    
    def find_matching_triplet(self, current_triplet: TripletState) -> Optional[TripletState]:
        """
        Find a matching triplet in history
        
        Args:
            current_triplet: Current triplet to match
            
        Returns:
            Matching triplet if found, None otherwise
        """
        for historical_triplet in reversed(self.triplet_history):
            # Skip if too old
            if time.time() - historical_triplet.timestamp > 3600:  # 1 hour
                continue
                
            # Compute difference
            diff = abs(
                current_triplet.coherence_score - 
                historical_triplet.coherence_score
            )
            
            if diff < self.epsilon:
                return historical_triplet
                
        return None
    
    def quantize_fractal_state(self, state_vector: List[float], precision: float = 0.001) -> Tuple[int]:
        """
        Convert fractal state vector to quantized format
        
        Args:
            state_vector: State vector to quantize
            precision: Quantization precision
            
        Returns:
            Quantized state tuple
        """
        return tuple(int(x / precision) for x in state_vector)
    
    def inject_recursive_correction(self, triplet: TripletState, 
                                  fault_log: Dict, 
                                  correction_matrix: List[List[float]]) -> List[float]:
        """
        Apply fault-corrective logic based on triplet match
        
        Args:
            triplet: Current triplet state
            fault_log: Fault history
            correction_matrix: Correction coefficients
            
        Returns:
            Corrected state vector
        """
        # Convert triplet to quantized form
        quantized = self.quantize_fractal_state([
            triplet.i.truth_value,
            triplet.j.truth_value,
            triplet.k.truth_value
        ])
        
        # Check if triplet has fault history
        if quantized in fault_log:
            correction = correction_matrix[fault_log[quantized]]
            return [sum(x * y for x, y in zip(quantized, row)) 
                   for row in correction_matrix]
            
        return list(quantized)  # No correction needed

# Example usage
if __name__ == "__main__":
    # Initialize classifiers and resolver
    cyclic_classifier = CyclicHashClassifier('cyclicNumbers.txt')
    euler_resolver = EulerSignatureMap('EulerCodedTriggers.txt')

    # Simulate fault resolution logic
    current_hash = 1234567890
    previous_hash = 9876543210

    if Δt_prime_drift(current_hash, previous_hash) < 0.12:
        qring_handler = QRingHandler()
        node_id = 123  # Example node ID
        qring_handler.push_fault(node_id)
        next_node = qring_handler.next()
        print(f"Next node in Q-ring: {next_node}")

    if cyclic_classifier.classify(current_hash):
        print("Node is part of a cyclic pattern.")

    if euler_resolver.compare_signature('signature_profile', current_hash):
        print("Current hash matches an Euler-coded decay.")
        # Compute reentry vector
        reentry_vector = drift_tracker.compute_reentry_vector(current_hash, previous_hash)
        print(f"Reentry vector: {reentry_vector}")

        # Perform XOR drift test with previous known Euler-aligned hash
        xor_result = drift_tracker.perform_xor_drift_test(current_hash, previous_known_hash)
        print(f"XOR result: {xor_result}")

    # Initialize drift tracker
    drift_tracker = DriftTracker(max_drift=0.5)

    # Simulate fault resolution logic
    current_hash = 1234567890
    previous_hash = 9876543210

    if Δt_prime_drift(current_hash, previous_hash) < 0.12:
        drift_tracker.update_drift(node_id=123, current_hash=current_hash, previous_hash=previous_hash)

    # Get recent drift vectors
    recent_drifts = drift_tracker.get_recent_drift(node_id=123)
    print(f"Recent drift vectors for node 123: {recent_drifts}")

    # Initialize the logger
    logger = RecursiveEnvelopeLogger()

    # Simulate drifted collapse data
    origin_hash = 1234567890
    collapse_curve = [0.1, -0.2, 0.3, -0.4]  # Example sequence of angular changes
    entropy_loss = sum(abs(change) for change in collapse_curve)
    drift_type = "spiral"  # Example drift type

    # Log the drifted collapse data
    logger.log_drift(origin_hash, collapse_curve, entropy_loss, drift_type)

    # Retrieve and print the drift log
    drift_log = logger.get_drift_log()
    for entry in drift_log:
        print(entry)

    # Check if realignment trigger conditions are met
    if logger.is_realignment_trigger_met(current_hash, previous_hash, collapse_curve):
        # Perform realignment logic here
        print("Realignment triggered.")
    else:
        print("Realignment not triggered.")

    unittest.main() 