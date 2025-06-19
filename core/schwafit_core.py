import numpy as np
import random
from typing import List, Dict, Any, Callable, Tuple, Optional
import yaml
from pathlib import Path

# Import the new Priority 3 systems
from .validation_engine import ValidationEngine, create_validation_engine, ValidationStatus
from .shell_memory import ShellMemory, create_shell_memory, MemoryPatternType, hash_signal_for_memory
from .safe_run_utils import safe_run, FallbackStrategy, ErrorSeverity

def load_yaml_config(config_name: str) -> dict:
    """Load a YAML configuration file relative to the repository."""
    config_path = Path(__file__).resolve().parent.parent / 'config' / config_name
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def validate_config(config: dict) -> bool:
    """Validate the configuration file for required fields."""
    required_fields = ['meta_tag', 'fallback_matrix', 'scoring']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    return True

def generate_default_config(config_name: str) -> dict:
    """Generate a default YAML configuration file if it is absent."""
    default_config = {
        'meta_tag': 'default',
        'fallback_matrix': 'default_fallback',
        'scoring': {
            'hash_weight': 0.3,
            'volume_weight': 0.2,
            'drift_weight': 0.4,
            'error_weight': 0.1
        }
    }
    config_path = Path(__file__).resolve().parent.parent / 'config' / config_name
    if not config_path.exists():
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file)
    return default_config

class SchwafitManager:
    """
    Implements the Schwafitting antifragile validation and partitioning system.
    Now supports meta_tag tracking, validation history, and tag-based filtering.
    Enhanced with Priority 3 validation framework and shell memory evolution.
    """
    def __init__(self, min_ratio: float = 0.01, max_ratio: float = 0.9, cycle_period: int = 1000, noise_scale: float = 0.05):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.cycle_period = cycle_period
        self.noise_scale = noise_scale
        self.t = 0
        self.variance_pool = 0.0
        self.memory_keys = {}  # shell class -> memory vector
        self.profit_tiers = {}  # tier -> calibration
        self.last_ratio = self.dynamic_holdout_ratio()
        self.validation_history = []  # List of dicts: {strategy, meta_tag, score, timestamp, ...}
        self.strategy_tags = {}  # strategy_id -> meta_tag
        self.config = load_yaml_config('strategies.yaml')  # Load strategy configuration
        validate_config(self.config)  # Validate the configuration
        if not self.config:
            self.config = generate_default_config('strategies.yaml')  # Generate default config if absent

        # PRIORITY 3 ENHANCEMENT: Initialize validation engine and shell memory
        self.validation_engine = create_validation_engine({
            'default_tolerance': 0.1,
            'coherence_min': 0.0,
            'coherence_max': 1.0,
            'profit_signal_min': -100.0,
            'profit_signal_max': 100.0
        })
        
        self.shell_memory = create_shell_memory({
            'max_patterns': 1000,
            'min_recurrence_for_routing': 3,
            'recurrence_weight': 0.3,
            'success_weight': 0.4,
            'profit_weight': 0.2,
            'recency_weight': 0.1
        })

    def register_strategy(self, strategy_id: str, meta_tag: str):
        self.strategy_tags[strategy_id] = meta_tag

    def dynamic_holdout_ratio(self) -> float:
        """Dynamic r(t) with sinusoidal and stochastic components."""
        alpha = (self.max_ratio + self.min_ratio) / 2
        beta = (self.max_ratio - self.min_ratio) / 2
        T = self.cycle_period
        xi = np.random.normal(0, 1)
        gamma = self.noise_scale
        r = alpha + beta * np.sin(2 * np.pi * self.t / T) + gamma * xi
        r = max(self.min_ratio, min(self.max_ratio, r))
        self.last_ratio = r
        self.t += 1
        return r

    def split_data(self, data: List[Any], shell_states: List[Any] = None) -> Tuple[List[Any], List[Any]]:
        """Partition data into visible and holdout sets using current r(t)."""
        r = self.dynamic_holdout_ratio()
        n = len(data)
        holdout_size = int(n * r)
        indices = list(range(n))
        random.shuffle(indices)
        holdout_idx = indices[:holdout_size]
        visible_idx = indices[holdout_size:]
        holdout = [data[i] for i in holdout_idx]
        visible = [data[i] for i in visible_idx]
        return visible, holdout

    def compute_shell_weights(self, holdout: List[Any], shell_states: List[Any]) -> np.ndarray:
        """
        Compute shell weights for holdout set with shell-state aware weighting.
        FIXED: Enhanced implementation using shell memory evolution system.
        """
        if not holdout:
            return np.array([])
        
        weights = np.ones(len(holdout))
        
        # Use shell memory to weight based on historical performance
        if shell_states and len(shell_states) == len(holdout):
            for i, (data_point, shell_state) in enumerate(zip(holdout, shell_states)):
                # Create hash for this data point + shell state combination
                combined_data = {'data': data_point, 'shell_state': shell_state}
                pattern_hash = hash_signal_for_memory(combined_data)
                
                # Get evolution score from shell memory
                evolution_score = self.shell_memory.get_score(pattern_hash)
                
                # Weight based on evolution score (higher score = higher weight)
                weights[i] = max(0.1, evolution_score)  # Minimum weight of 0.1
        
        # Normalize weights
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        return weights

    def schwafit_validation_tensor(self, strategies: List[Callable], holdout: List[Any], shell_states: List[Any], meta_tags: Optional[List[str]] = None) -> np.ndarray:
        """
        Binary tensor: [strategy, sample, shell_class].
        FIXED: Filled T with actual validation results using ValidationEngine.
        """
        m = len(strategies)
        k = len(holdout)
        p = 1  # For now, assume 1 shell class
        T = np.zeros((m, k, p))
        
        # PRIORITY 3 FIX: Fill T with actual validation results
        for i, strategy in enumerate(strategies):
            strategy_name = getattr(strategy, '__name__', f'strategy_{i}')
            
            for j, holdout_sample in enumerate(holdout):
                for l in range(p):
                    try:
                        # Generate a prediction using the strategy
                        def run_strategy():
                            if callable(strategy):
                                # Assume strategy takes holdout sample and returns prediction
                                return strategy(holdout_sample)
                            return 0.0
                        
                        # Use safe_run for strategy execution
                        prediction = safe_run(
                            run_strategy,
                            context=f"strategy_validation_{strategy_name}",
                            fallback_strategy=FallbackStrategy.RETURN_ZERO,
                            error_severity=ErrorSeverity.MEDIUM
                        )
                        
                        # Validate the prediction is within reasonable bounds
                        is_valid = self.validation_engine.validate_signal(
                            prediction or 0.0,
                            (-1000.0, 1000.0),  # Reasonable prediction range
                            f"strategy_{i}_sample_{j}_validation"
                        )
                        
                        # Set T value based on validation result
                        T[i, j, l] = 1.0 if is_valid else 0.0
                        
                        # Track pattern in shell memory if shell states available
                        if shell_states and j < len(shell_states):
                            pattern_data = {
                                'strategy': strategy_name,
                                'holdout_sample': holdout_sample,
                                'shell_state': shell_states[j],
                                'prediction': prediction
                            }
                            pattern_hash = hash_signal_for_memory(pattern_data)
                            
                            # Evolve pattern in shell memory
                            self.shell_memory.evolve(
                                pattern_hash,
                                MemoryPatternType.STRATEGY_PATTERN,
                                success=is_valid,
                                metadata={'strategy_name': strategy_name, 'sample_index': j}
                            )
                        
                    except Exception as e:
                        # Log validation failure
                        T[i, j, l] = 0.0  # Mark as failed
                        
                        # Record error in shell memory
                        error_pattern = hash_signal_for_memory(f"error_{strategy_name}_{j}")
                        self.shell_memory.evolve(
                            error_pattern,
                            MemoryPatternType.ERROR_PATTERN,
                            success=False,
                            metadata={'error': str(e), 'strategy': strategy_name}
                        )
        
        return T

    def adaptive_confidence_scores(self, T: np.ndarray, weights: np.ndarray, predictions: np.ndarray, targets: np.ndarray, decay_lambda: float = 1.0) -> np.ndarray:
        """Compute Schwafit scores for each strategy."""
        m, k, p = T.shape
        scores = np.zeros(m)
        for i in range(m):
            score = 0.0
            for j in range(k):
                for l in range(p):
                    error = np.abs(predictions[i, j] - targets[j])
                    score += weights[j] * T[i, j, l] * np.exp(-decay_lambda * error)
            scores[i] = score / max(1, k)
        return scores

    def update_variance_pool(self, holdout: List[Any]):
        """Recursive variance injection."""
        eta = 1.0
        mu = 0.1
        var_ht = np.var(holdout) if len(holdout) > 0 else 0.0
        self.variance_pool = self.variance_pool + eta * var_ht - mu * self.variance_pool

    def evolve_memory_keys(self, holdout: List[Any], shell_states: List[Any]):
        """
        Update memory keys for each shell class.
        FIXED: Implement shell class memory evolution using ShellMemory system.
        """
        if not shell_states:
            return
        
        # Calculate mean shell state as basic memory key
        if shell_states:
            mean_shell_state = np.mean(shell_states, axis=0) if isinstance(shell_states[0], (list, np.ndarray)) else np.mean(shell_states)
            self.memory_keys['default'] = mean_shell_state
        
        # Use shell memory system for more sophisticated evolution
        for i, (holdout_sample, shell_state) in enumerate(zip(holdout, shell_states)):
            # Create pattern for this shell state
            pattern_data = {'holdout': holdout_sample, 'shell_state': shell_state}
            pattern_hash = hash_signal_for_memory(pattern_data)
            
            # Evolve the pattern in shell memory
            self.shell_memory.evolve(
                pattern_hash,
                MemoryPatternType.SIGNAL_HASH,
                metadata={'shell_class': 'default', 'sample_index': i}
            )
        
        # Update memory keys based on best performing patterns
        best_patterns = self.shell_memory.get_best_patterns(n=5, pattern_type=MemoryPatternType.SIGNAL_HASH)
        if best_patterns:
            # Use the evolution scores to weight the memory update
            evolution_scores = [p.evolution_score for p in best_patterns]
            weighted_mean = np.average(evolution_scores) if evolution_scores else 0.5
            self.memory_keys['evolved'] = weighted_mean

    def calibrate_profits(self, scores: np.ndarray, base_tiers: Dict[str, float] = None):
        """Calibrate profit tiers based on Schwafit scores."""
        if base_tiers is None:
            base_tiers = {'tier1': 1.0, 'tier2': 1.0}
        for tier, base in base_tiers.items():
            self.profit_tiers[tier] = base * np.prod(1 + 0.1 * scores)  # Example

    def schwafit_update(self, data: List[Any], shell_states: List[Any], strategies: List[Callable], predictions: np.ndarray, targets: np.ndarray, meta_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main Schwafit update step: partition, score, update variance/memory/profit.
        Enhanced with Priority 3 validation and shell memory evolution.
        """
        visible, holdout = self.split_data(data, shell_states)
        weights = self.compute_shell_weights(holdout, shell_states)
        T = self.schwafit_validation_tensor(strategies, holdout, shell_states, meta_tags)
        scores = self.adaptive_confidence_scores(T, weights, predictions, targets)
        self.update_variance_pool(holdout)
        self.evolve_memory_keys(holdout, shell_states)
        self.calibrate_profits(scores)
        
        # Log validation history with meta_tag
        timestamp = self.t
        for i, strategy in enumerate(strategies):
            tag = meta_tags[i] if meta_tags else self.strategy_tags.get(getattr(strategy, '__name__', str(i)), 'unknown')
            
            # Enhanced logging with validation results
            validation_report = self.validation_engine.get_report(include_recent_only=True)
            
            history_entry = {
                'strategy': getattr(strategy, '__name__', str(i)),
                'meta_tag': tag,
                'score': scores[i],
                'timestamp': timestamp,
                'validation_pass_rate': validation_report.pass_rate if validation_report else 0.0,
                'shell_memory_state': self.shell_memory.get_evolution_state()
            }
            self.validation_history.append(history_entry)
        
        return {
            'scores': scores,
            'memory_keys': self.memory_keys,
            'profit_tiers': self.profit_tiers,
            'variance_pool': self.variance_pool,
            'last_ratio': self.last_ratio,
            'validation_history': self.validation_history[-100:],  # last 100 entries
            'validation_engine_stats': self.validation_engine.get_performance_stats(),
            'shell_memory_stats': {
                'total_patterns': len(self.shell_memory.evolution_map),
                'evolution_state': self.shell_memory.get_evolution_state(),
                'routing_decisions': self.shell_memory.routing_decisions
            }
        }

    def get_top_strategies(self, n: int = 3, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return top-N strategies by recent score, optionally filtered by meta_tag.
        Enhanced with shell memory routing recommendations.
        """
        filtered = [h for h in self.validation_history if (tag is None or h['meta_tag'] == tag)]
        filtered = sorted(filtered, key=lambda x: x['score'], reverse=True)
        
        # Enhance results with shell memory routing recommendations
        enhanced_results = []
        for entry in filtered[:n]:
            strategy_name = entry['strategy']
            strategy_hash = hash_signal_for_memory(strategy_name)
            
            # Get routing recommendation from shell memory
            routing_rec = self.shell_memory.get_routing_recommendation(strategy_hash)
            
            enhanced_entry = entry.copy()
            enhanced_entry['routing_recommendation'] = routing_rec
            enhanced_entry['memory_evolution_score'] = self.shell_memory.get_score(strategy_hash)
            
            enhanced_results.append(enhanced_entry)
        
        return enhanced_results
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive system report including all Priority 3 enhancements"""
        return {
            'schwafit_state': {
                'current_ratio': self.last_ratio,
                'variance_pool': self.variance_pool,
                'total_strategies': len(self.strategy_tags),
                'validation_history_size': len(self.validation_history)
            },
            'validation_engine': {
                'performance_stats': self.validation_engine.get_performance_stats(),
                'recent_report': self.validation_engine.get_report(include_recent_only=True)
            },
            'shell_memory': {
                'evolution_state': self.shell_memory.get_evolution_state(),
                'performance_stats': {
                    'total_evolutions': self.shell_memory.total_evolutions,
                    'routing_decisions': self.shell_memory.routing_decisions,
                    'cleanup_operations': self.shell_memory.cleanup_operations
                },
                'category_performance': dict(self.shell_memory.category_performance)
            },
            'system_health': {
                'total_patterns_tracked': len(self.shell_memory.evolution_map),
                'validation_pass_rate': self.validation_engine.get_performance_stats().get('overall_pass_rate', 0.0),
                'memory_efficiency': self.shell_memory.get_evolution_state().memory_efficiency
            }
        } 