import unittest
from strategy_loader import load_active_strategies
from strategy_config import StrategyConfig

class TestStrategyLoader(unittest.TestCase):
    def test_load_valid_yaml(self):
        configs = load_active_strategies("config/strategies.yaml")
        self.assertIn("BTC_hash_v3", configs)
        cfg = configs["BTC_hash_v3"]
        self.assertIsInstance(cfg, StrategyConfig)
        self.assertEqual(cfg.meta_tag, "echo_branch_A")
        self.assertAlmostEqual(cfg.scoring["hash_weight"], 0.4)

    def test_inactive_strategies_filtered(self):
        configs = load_active_strategies("config/strategies.yaml")
        self.assertNotIn("ETH_voltrap_4h", configs)

if __name__ == '__main__':
    unittest.main() 