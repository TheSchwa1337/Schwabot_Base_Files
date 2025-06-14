from pathlib import Path
import yaml
import os

# Function to generate default YAML files if they are absent
def generate_default_config():
    config_path = Path(__file__).resolve().parent / 'config/matrix_response_paths.yaml'
    if not config_path.exists():
        print(f"Config file {config_path} does not exist. Generating default configuration...")
        with open(config_path, 'w') as f:
            yaml.dump({
                'default_value': 'example',
                'another_key': 123
            }, f)
        print("Default configuration generated.")

# Helper function to load YAML with error handling
def load_yaml(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Config file {file_path} not found.")
    except Exception as e:
        raise ValueError(f"Error loading config file {file_path}: {e}")

# DataProvider interface
class DataProvider:
    def get_price(self):
        pass

# Example implementation of DataProvider for backtesting
class HistoricalDataProvider(DataProvider):
    def __init__(self, historical_data):
        self.historical_data = historical_data

    def get_price(self):
        return self.historical_data.pop(0)

# Main class with updated config handling
class FerrisWheelScheduler:
    def __init__(self, drift_engine, cluster_mapper, strategy_bundler, echo_logger, vault_router, config):
        self.drift_engine = drift_engine
        self.cluster_mapper = cluster_mapper
        self.strategy_bundler = strategy_bundler
        self.echo_logger = echo_logger
        self.vault_router = vault_router
        self.config = load_yaml(Path(__file__).resolve().parent / 'config/matrix_response_paths.yaml')
        self.tick_count = 0

    def tick_loop(self, debug_clusters=False, debug_drifts=False, simulate_strategy=False):
        generate_default_config()  # Ensure default config is generated if missing
        while True:
            tick_data = self._get_next_tick()
            hash_block = tick_data['hash']
            entropy_window = tick_data['entropy_window']

            # 1. Drift calculation
            drift_score = self.drift_engine.compute_drift_variance(tick_data, hash_block, entropy_window)
            if debug_drifts:
                print(f"[Tick {self.tick_count}] ΔΨᵢ: {drift_score:.4f}")

            # 2. Cluster mapping
            cluster_node = self.cluster_mapper.link_clusters_by_entropy(tick_data)
            if debug_clusters:
                print(f"[Tick {self.tick_count}] Cluster: {cluster_node}")

            # 3. Echo family scoring
            Fk = self.cluster_mapper.get_echo_score(cluster_node)

            # 4. Strategy bundling
            bundle = self.strategy_bundler.bundle_strategies_by_tier(drift_score, Fk, tick_data['strategy_hint'])
            if bundle:
                print(f"[Tick {self.tick_count}] Triggered Strategy: {bundle['strategy']}")
                # 5. Vault execution
                action = self.vault_router.trigger_execution(bundle)
                print(f"[Tick {self.tick_count}] Vault Action: {action}")

            # 6. Memory update
            self.echo_logger.log_cluster_memory(
                cluster_id=cluster_node['id'],
                strategy_id=bundle['strategy'] if bundle else None,
                entropy_signature=drift_score
            )

            # 7. Output summary
            print(f"Tick {self.tick_count} | ΔΨᵢ: {drift_score:.4f} | Strategy: {bundle['strategy'] if bundle else 'None'} | Vault: {action if bundle else 'None'}")
            self.tick_count += 1

            # Simulate tick interval
            import time
            time.sleep(self.config.get('tick_interval', 1.0))

    def _get_next_tick(self):
        # Placeholder: Replace with real tick data ingestion
        import random
        return {
            'hash': hex(random.getrandbits(256)),
            'entropy_window': [random.random() for _ in range(10)],
            'strategy_hint': 'FlipHold'
        }

# Example usage of DataProvider during backtesting
historical_data = ['price1', 'price2', 'price3']
backtest_provider = HistoricalDataProvider(historical_data)
scheduler = FerrisWheelScheduler(
    drift_engine=None,  # Placeholder for actual drift engine
    cluster_mapper=None,  # Placeholder for actual cluster mapper
    strategy_bundler=None,  # Placeholder for actual strategy bundler
    echo_logger=None,  # Placeholder for actual echo logger
    vault_router=None,  # Placeholder for actual vault router
    config=scheduler.config
)
scheduler.tick_loop(debug_clusters=True, debug_drifts=True, simulate_strategy=True)

# tests/test_ferris_wheel_scheduler.py
import unittest
from core.ferris_wheel_scheduler import FerrisWheelScheduler

class TestFerrisWheelScheduler(unittest.TestCase):
    def test_config_loading(self):
        scheduler = FerrisWheelScheduler(
            drift_engine=None,
            cluster_mapper=None,
            strategy_bundler=None,
            echo_logger=None,
            vault_router=None,
            config={}
        )
        with self.assertRaises(ValueError) as context:
            scheduler.tick_loop(debug_clusters=True, debug_drifts=True, simulate_strategy=True)
        self.assertIn("Config file", str(context.exception))

if __name__ == '__main__':
    unittest.main()
