import unittest
from core.drift_shell_engine import DriftShellEngine


class TestDriftShellEngine(unittest.TestCase):
    def test_entropy_deviation(self):
        engine = DriftShellEngine()
        tick_data = {'entropy_window': [0.1, 0.2, 0.3, 0.4]}
        hash_block = '0xabc123'
        result = engine.compute_drift_variance(
            tick_data,
            hash_block,
            tick_data['entropy_window']
        )
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)


if __name__ == '__main__':
    unittest.main()