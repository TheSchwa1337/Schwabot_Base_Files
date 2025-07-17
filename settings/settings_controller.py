#!/usr/bin/env python3
"""
Settings Controller - Manages system configuration and parameter optimization.
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class MathematicalFlowParams:
    """Mathematical flow parameters for trading algorithms."""
    entropy_threshold: float = 0.75
    fractal_dimension: float = 1.5
    quantum_drift_factor: float = 0.25
    vector_confidence_min: float = 0.6
    matrix_basket_size: int = 16
    tick_sync_interval: float = 3.75
    volume_delta_threshold: float = 0.1
    hash_confidence_decay: float = 0.95
    ghost_strategy_weight: float = 0.3
    backlog_retention_cycles: int = 1000


@dataclass
class ReinforcementLearningParams:
    """Reinforcement learning parameters from backtest failures."""
    learning_rate: float = 0.01
    failure_penalty_weight: float = 0.5
    success_reward_weight: float = 1.0
    exploration_rate: float = 0.1
    memory_size: int = 10000
    batch_size: int = 32
    update_interval: int = 100
    convergence_threshold: float = 0.001
    max_iterations: int = 1000
    adaptive_learning: bool = True


@dataclass
class DemoBacktestParams:
    """Demo backtesting parameters."""
    enabled: bool = True
    simulation_duration: int = 3600  # seconds
    tick_interval: float = 3.75
    initial_balance: float = 10000.0
    max_positions: int = 5
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    slippage: float = 0.001
    commission: float = 0.001
    data_source: str = "simulated"
    validation_mode: bool = False
    target_profit: float = 0.1


class SettingsController:
    """Main settings controller for Schwabot."""

    def __init__(self, config_dir: str = "settings"):
        """Initialize the settings controller."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Initialize parameters
        self.math_params = MathematicalFlowParams()
        self.rl_params = ReinforcementLearningParams()
        self.demo_params = DemoBacktestParams()

        # State tracking
        self.last_update = datetime.now()
        self.update_count = 0
        self.failure_history = []
        self.success_history = []
        self.known_bad_vectors = {}

        # Threading
        self.lock = threading.RLock()
        self.background_running = False
        self.background_thread = None

        # Load existing configuration
        self.load_configuration()

        # Start background updates
        self.start_background_updates()

    def load_configuration(self) -> None:
        """Load configuration from YAML and JSON files."""
        try:
            # Load demo backtest configuration
            demo_config_path = self.config_dir / "demo_backtest_mode.yaml"
            if demo_config_path.exists():
                with open(demo_config_path, 'r') as f:
                    demo_config = yaml.safe_load(f)
                    self.demo_params = DemoBacktestParams(**demo_config.get('demo_params', {}))

            # Load vector settings
            vector_config_path = self.config_dir / "vector_settings_experiment.yaml"
            if vector_config_path.exists():
                with open(vector_config_path, 'r') as f:
                    vector_config = yaml.safe_load(f)
                    math_config = vector_config.get('mathematical_flow', {})
                    self.math_params = MathematicalFlowParams(**math_config)
                    rl_config = vector_config.get('reinforcement_learning', {})
                    self.rl_params = ReinforcementLearningParams(**rl_config)

            # Load known bad vectors
            bad_vectors_path = self.config_dir / "known_bad_vector_map.json"
            if bad_vectors_path.exists():
                with open(bad_vectors_path, 'r') as f:
                    self.known_bad_vectors = json.load(f)
            else:
                self.known_bad_vectors = {}

            logger.info("Configuration loaded successfully")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.create_default_configuration()

    def save_configuration(self) -> None:
        """Save current configuration to files."""
        try:
            with self.lock:
                # Save demo backtest configuration
                demo_config = {
                    'demo_params': asdict(self.demo_params),
                    'last_updated': datetime.now().isoformat()
                }
                with open(self.config_dir / "demo_backtest_mode.yaml", 'w') as f:
                    yaml.dump(demo_config, f, default_flow_style=False)

                # Save vector settings
                vector_config = {
                    'mathematical_flow': asdict(self.math_params),
                    'reinforcement_learning': asdict(self.rl_params),
                    'last_updated': datetime.now().isoformat()
                }
                with open(self.config_dir / "vector_settings_experiment.yaml", 'w') as f:
                    yaml.dump(vector_config, f, default_flow_style=False)

                # Save known bad vectors
                with open(self.config_dir / "known_bad_vector_map.json", 'w') as f:
                    json.dump(self.known_bad_vectors, f, indent=2)

            logger.info("Configuration saved successfully")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def create_default_configuration(self) -> None:
        """Create default configuration files."""
        self.save_configuration()

    def update_mathematical_flow(self, **kwargs) -> None:
        """Update mathematical flow parameters."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.math_params, key):
                    setattr(self.math_params, key, value)
                    logger.info(f"Updated mathematical flow parameter: {key} = {value}")

        self.save_configuration()

    def update_reinforcement_learning(self, **kwargs) -> None:
        """Update reinforcement learning parameters."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.rl_params, key):
                    setattr(self.rl_params, key, value)
                    logger.info(f"Updated RL parameter: {key} = {value}")

        self.save_configuration()

    def update_demo_backtest(self, **kwargs) -> None:
        """Update demo backtest parameters."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.demo_params, key):
                    setattr(self.demo_params, key, value)
                    logger.info(f"Updated demo backtest parameter: {key} = {value}")

        self.save_configuration()

    def record_backtest_failure(self, failure_data: Dict[str, Any]) -> None:
        """Record a backtest failure for reinforcement learning."""
        with self.lock:
            failure_data['timestamp'] = datetime.now().isoformat()
            failure_data['update_count'] = self.update_count
            self.failure_history.append(failure_data)

            # Keep only recent failures
            if len(self.failure_history) > self.rl_params.memory_size:
                self.failure_history = self.failure_history[-self.rl_params.memory_size:]

            # Update parameters based on failure
            self._apply_failure_learning(failure_data)

        logger.info(f"Recorded backtest failure: {failure_data.get('reason', 'Unknown')}")

    def record_backtest_success(self, success_data: Dict[str, Any]) -> None:
        """Record a backtest success for reinforcement learning."""
        with self.lock:
            success_data['timestamp'] = datetime.now().isoformat()
            success_data['update_count'] = self.update_count
            self.success_history.append(success_data)

            # Keep only recent successes
            if len(self.success_history) > self.rl_params.memory_size:
                self.success_history = self.success_history[-self.rl_params.memory_size:]

            # Update parameters based on success
            self._apply_success_learning(success_data)

        logger.info(f"Recorded backtest success: {success_data.get('profit', 0):.2f}")

    def _apply_failure_learning(self, failure_data: Dict[str, Any]) -> None:
        """Apply learning from failure to adjust parameters."""
        failure_reason = failure_data.get('reason', '')

        if 'entropy' in failure_reason.lower():
            # Reduce entropy threshold
            new_threshold = self.math_params.entropy_threshold * (1 - self.rl_params.failure_penalty_weight * 0.1)
            self.math_params.entropy_threshold = max(0.1, new_threshold)

        elif 'confidence' in failure_reason.lower():
            # Increase confidence requirements
            new_confidence = self.math_params.vector_confidence_min * (1 + self.rl_params.failure_penalty_weight * 0.1)
            self.math_params.vector_confidence_min = min(0.95, new_confidence)

        elif 'volume' in failure_reason.lower():
            # Adjust volume delta threshold
            new_threshold = self.math_params.volume_delta_threshold * (1 + self.rl_params.failure_penalty_weight * 0.1)
            self.math_params.volume_delta_threshold = min(0.5, new_threshold)

        # Adaptive learning rate adjustment
        if self.rl_params.adaptive_learning:
            self.rl_params.learning_rate *= 0.99  # Gradually reduce learning rate

        self.update_count += 1

    def _apply_success_learning(self, success_data: Dict[str, Any]) -> None:
        """Apply learning from success to adjust parameters."""
        profit = success_data.get('profit', 0.0)

        if profit > 0:
            # Reward successful strategies
            if self.rl_params.adaptive_learning:
                self.rl_params.learning_rate *= 1.01  # Gradually increase learning rate

            # Adjust parameters based on success
            if profit > self.demo_params.target_profit:
                # Exceeded target, can be more aggressive
                self.math_params.entropy_threshold *= 1.05
                self.math_params.vector_confidence_min *= 0.98

        self.update_count += 1

    def get_optimized_parameters(self) -> Dict[str, Any]:
        """Get optimized parameters based on learning history."""
        with self.lock:
            return {
                'mathematical_flow': asdict(self.math_params),
                'reinforcement_learning': asdict(self.rl_params),
                'demo_backtest': asdict(self.demo_params),
                'update_count': self.update_count,
                'failure_count': len(self.failure_history),
                'success_count': len(self.success_history),
            }

    def add_known_bad_vector(self, vector_hash: str, reason: str = "Unknown") -> None:
        """Add a vector to the known bad vectors list."""
        with self.lock:
            self.known_bad_vectors[vector_hash] = {
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'avoidance_count': 0
            }

    def is_known_bad_vector(self, vector_hash: str) -> bool:
        """Check if a vector is in the known bad vectors list."""
        return vector_hash in self.known_bad_vectors

    def get_vector_avoidance_count(self, vector_hash: str) -> int:
        """Get the avoidance count for a known bad vector."""
        return self.known_bad_vectors.get(vector_hash, {}).get('avoidance_count', 0)

    def increment_avoidance_count(self, vector_hash: str) -> None:
        """Increment the avoidance count for a known bad vector."""
        if vector_hash in self.known_bad_vectors:
            self.known_bad_vectors[vector_hash]['avoidance_count'] += 1

    def start_background_updates(self) -> None:
        """Start background parameter optimization."""
        self.background_running = True
        self.background_thread = threading.Thread(target=self._background_update_loop, daemon=True)
        self.background_thread.start()
        logger.info("Background parameter optimization started")

    def stop_background_updates(self) -> None:
        """Stop background parameter optimization."""
        self.background_running = False
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5.0)
        logger.info("Background parameter optimization stopped")

    def _background_update_loop(self) -> None:
        """Background loop for parameter optimization."""
        while self.background_running:
            try:
                time.sleep(self.rl_params.update_interval)
                if self.background_running:
                    self._optimize_parameters()
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
                time.sleep(10)  # Wait before retrying

    def _optimize_parameters(self) -> None:
        """Optimize parameters based on recent performance."""
        try:
            # Analyze recent performance
            recent_failures = len([f for f in self.failure_history 
                                 if (datetime.now() - datetime.fromisoformat(f['timestamp'])).days < 1])
            recent_successes = len([s for s in self.success_history 
                                  if (datetime.now() - datetime.fromisoformat(s['timestamp'])).days < 1])

            # Adjust parameters based on recent performance
            if recent_failures > recent_successes:
                # More failures, be more conservative
                self.math_params.entropy_threshold *= 0.95
                self.math_params.vector_confidence_min *= 1.05
            elif recent_successes > recent_failures:
                # More successes, can be more aggressive
                self.math_params.entropy_threshold *= 1.05
                self.math_params.vector_confidence_min *= 0.95

            # Save updated configuration
            self.save_configuration()

        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics."""
        with self.lock:
            total_tests = len(self.failure_history) + len(self.success_history)
            success_rate = len(self.success_history) / total_tests if total_tests > 0 else 0.0

            return {
                'total_tests': total_tests,
                'success_count': len(self.success_history),
                'failure_count': len(self.failure_history),
                'success_rate': success_rate,
                'update_count': self.update_count,
                'known_bad_vectors': len(self.known_bad_vectors),
                'background_running': self.background_running,
                'last_update': datetime.now().isoformat(),
            }

    def reset_learning(self) -> None:
        """Reset all learning history and parameters."""
        with self.lock:
            self.failure_history.clear()
            self.success_history.clear()
            self.known_bad_vectors.clear()
            self.update_count = 0
            
            # Reset to default parameters
            self.math_params = MathematicalFlowParams()
            self.rl_params = ReinforcementLearningParams()
            self.demo_params = DemoBacktestParams()
            
            self.save_configuration()
            logger.info("Learning history and parameters reset")

    def export_configuration(self, export_path: str) -> None:
        """Export current configuration to a file."""
        try:
            config_data = {
                'mathematical_flow': asdict(self.math_params),
                'reinforcement_learning': asdict(self.rl_params),
                'demo_backtest': asdict(self.demo_params),
                'known_bad_vectors': self.known_bad_vectors,
                'export_timestamp': datetime.now().isoformat(),
            }
            
            with open(export_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")

    def import_configuration(self, import_path: str) -> None:
        """Import configuration from a file."""
        try:
            with open(import_path, 'r') as f:
                config_data = json.load(f)
            
            with self.lock:
                if 'mathematical_flow' in config_data:
                    self.math_params = MathematicalFlowParams(**config_data['mathematical_flow'])
                if 'reinforcement_learning' in config_data:
                    self.rl_params = ReinforcementLearningParams(**config_data['reinforcement_learning'])
                if 'demo_backtest' in config_data:
                    self.demo_params = DemoBacktestParams(**config_data['demo_backtest'])
                if 'known_bad_vectors' in config_data:
                    self.known_bad_vectors = config_data['known_bad_vectors']
            
            self.save_configuration()
            logger.info(f"Configuration imported from {import_path}")
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")

    def get_settings_controller(self) -> 'SettingsController':
        """Get the settings controller instance."""
        return self


def get_settings_controller(config_dir: str = "settings") -> SettingsController:
    """Get a settings controller instance."""
    return SettingsController(config_dir)
