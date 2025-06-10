import numpy as np
import random
from typing import List, Dict, Any, Callable, Tuple, Optional
import yaml
from pathlib import Path

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
        """Compute shell weights for holdout set (stub: uniform)."""
        # TODO: Implement shell-state aware weighting
        return np.ones(len(holdout)) / max(1, len(holdout))

    def schwafit_validation_tensor(self, strategies: List[Callable], holdout: List[Any], shell_states: List[Any], meta_tags: Optional[List[str]] = None) -> np.ndarray:
        """Binary tensor: [strategy, sample, shell_class]."""
        m = len(strategies)
        k = len(holdout)
        p = 1  # For now, assume 1 shell class
        T = np.zeros((m, k, p))
        # TODO: Fill T with actual validation results
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
        """Update memory keys for each shell class (stub: mean of shell_states)."""
        # TODO: Implement shell class memory evolution
        if shell_states:
            self.memory_keys['default'] = np.mean(shell_states, axis=0)

    def calibrate_profits(self, scores: np.ndarray, base_tiers: Dict[str, float] = None):
        """Calibrate profit tiers based on Schwafit scores."""
        if base_tiers is None:
            base_tiers = {'tier1': 1.0, 'tier2': 1.0}
        for tier, base in base_tiers.items():
            self.profit_tiers[tier] = base * np.prod(1 + 0.1 * scores)  # Example

    def schwafit_update(self, data: List[Any], shell_states: List[Any], strategies: List[Callable], predictions: np.ndarray, targets: np.ndarray, meta_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main Schwafit update step: partition, score, update variance/memory/profit.
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
            self.validation_history.append({
                'strategy': getattr(strategy, '__name__', str(i)),
                'meta_tag': tag,
                'score': scores[i],
                'timestamp': timestamp
            })
        return {
            'scores': scores,
            'memory_keys': self.memory_keys,
            'profit_tiers': self.profit_tiers,
            'variance_pool': self.variance_pool,
            'last_ratio': self.last_ratio,
            'validation_history': self.validation_history[-100:]  # last 100 entries
        }

    def get_top_strategies(self, n: int = 3, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return top-N strategies by recent score, optionally filtered by meta_tag.
        """
        filtered = [h for h in self.validation_history if (tag is None or h['meta_tag'] == tag)]
        filtered = sorted(filtered, key=lambda x: x['score'], reverse=True)
        return filtered[:n] 