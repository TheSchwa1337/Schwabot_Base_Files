"""
Test suite for the Adaptive Profit Chain Framework (APCF)
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add parent directory to path to import APCF
sys.path.append(str(Path(__file__).parent.parent))
from apcf.adaptive_profit_chain import (
    APCFSystem,
    MarketRegime,
    SpectralBucket,
    ProfitLink,
    FormalityStretcher,
    FourierBucketizer,
    FlipSequencer,
    FerrisRunner
)

class TestAPCF(unittest.TestCase):
    """Test cases for APCF components"""
    
    def setUp(self):
        """Set up test data and environment"""
        # Create sample price and volume data
        np.random.seed(42)
        self.n_points = 1000
        self.timestamps = pd.date_range(start='2024-01-01', periods=self.n_points, freq='1min')
        
        # Generate synthetic price data with known patterns
        t = np.linspace(0, 20, self.n_points)
        self.price_series = (
            100 +  # Base price
            10 * np.sin(2 * np.pi * 0.1 * t) +  # Long-term cycle
            5 * np.sin(2 * np.pi * 0.5 * t) +   # Medium-term cycle
            2 * np.sin(2 * np.pi * 2.0 * t) +   # Short-term cycle
            np.random.normal(0, 1, self.n_points)  # Noise
        )
        
        # Generate synthetic volume data
        self.volume_series = (
            1000 +  # Base volume
            500 * np.sin(2 * np.pi * 0.1 * t) +  # Volume cycles
            np.random.normal(0, 100, self.n_points)  # Volume noise
        )
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'timestamp': self.timestamps,
            'price': self.price_series,
            'volume': self.volume_series
        })
        
        # Initialize APCF system
        self.apcf = APCFSystem()
        
    def test_formality_stretcher(self):
        """Test FormalityStretcher functionality"""
        stretcher = FormalityStretcher()
        
        # Test regime classification
        regime = stretcher.classify_regime(self.price_series, self.volume_series)
        self.assertIsInstance(regime, MarketRegime)
        self.assertIn(regime.regime_type, ['high_vol', 'rangebound', 'trending', 'breakout', 'consolidation'])
        
        # Test stretch parameters
        params = stretcher.get_stretch_params(regime)
        self.assertIn('profit_multiplier', params)
        self.assertIn('stop_multiplier', params)
        self.assertIn('bucket_tolerance', params)
        
    def test_fourier_bucketizer(self):
        """Test FourierBucketizer functionality"""
        bucketizer = FourierBucketizer(self.price_series, self.volume_series)
        
        # Test bucket finding
        buckets = bucketizer.find_spectral_buckets()
        self.assertIsInstance(buckets, list)
        if buckets:
            self.assertIsInstance(buckets[0], SpectralBucket)
            
        # Test bucket properties
        for bucket in buckets:
            self.assertGreater(bucket.confidence, 0)
            self.assertGreater(bucket.period, 0)
            self.assertIsInstance(bucket.price_range, tuple)
            self.assertEqual(len(bucket.price_range), 2)
            
    def test_flip_sequencer(self):
        """Test FlipSequencer functionality"""
        stretcher = FormalityStretcher()
        bucketizer = FourierBucketizer(self.price_series, self.volume_series)
        buckets = bucketizer.find_spectral_buckets()
        
        sequencer = FlipSequencer(buckets, stretcher)
        chains = sequencer.build_profit_chains()
        
        self.assertIsInstance(chains, list)
        if chains:
            self.assertIsInstance(chains[0], list)
            self.assertIsInstance(chains[0][0], ProfitLink)
            
    def test_ferris_runner(self):
        """Test FerrisRunner functionality"""
        stretcher = FormalityStretcher()
        bucketizer = FourierBucketizer(self.price_series, self.volume_series)
        buckets = bucketizer.find_spectral_buckets()
        sequencer = FlipSequencer(buckets, stretcher)
        chains = sequencer.build_profit_chains()
        
        runner = FerrisRunner(chains)
        
        # Test cycle execution
        for i in range(100):
            result = runner.run_cycle(
                current_price=self.price_series[i],
                current_time=i,
                current_volume=self.volume_series[i]
            )
            self.assertIsInstance(result, dict)
            self.assertIn('status', result)
            self.assertIn('action', result)
            
    def test_full_system(self):
        """Test complete APCF system"""
        # Initialize system
        self.apcf.initialize_system(self.price_series, self.volume_series)
        
        # Run backtest
        results = self.apcf.run_backtest(self.data)
        
        # Verify results structure
        self.assertIn('trades', results)
        self.assertIn('performance', results)
        self.assertIn('summary', results)
        
        # Verify summary statistics
        summary = results['summary']
        self.assertIn('total_trades', summary)
        self.assertIn('win_rate', summary)
        self.assertIn('avg_profit', summary)
        self.assertIn('total_return', summary)
        
    def test_market_regime_transitions(self):
        """Test system behavior during regime transitions"""
        # Create data with regime transition
        transition_point = self.n_points // 2
        
        # First half: trending
        t1 = np.linspace(0, 10, transition_point)
        price1 = 100 + 2 * t1 + np.random.normal(0, 1, transition_point)
        volume1 = 1000 + 0.5 * t1 + np.random.normal(0, 100, transition_point)
        
        # Second half: rangebound
        t2 = np.linspace(0, 10, self.n_points - transition_point)
        price2 = 120 + 5 * np.sin(2 * np.pi * 0.5 * t2) + np.random.normal(0, 1, self.n_points - transition_point)
        volume2 = 1500 + 200 * np.sin(2 * np.pi * 0.5 * t2) + np.random.normal(0, 100, self.n_points - transition_point)
        
        # Combine data
        transition_data = pd.DataFrame({
            'timestamp': self.timestamps,
            'price': np.concatenate([price1, price2]),
            'volume': np.concatenate([volume1, volume2])
        })
        
        # Run system
        self.apcf.initialize_system(transition_data['price'].values, transition_data['volume'].values)
        results = self.apcf.run_backtest(transition_data)
        
        # Verify system adapted to regime change
        self.assertGreater(len(results['trades']), 0)
        self.assertIn('summary', results)
        
    def test_error_handling(self):
        """Test system error handling"""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'timestamp': self.timestamps,
            'price': [np.nan] * self.n_points,
            'volume': [np.nan] * self.n_points
        })
        
        # Should handle gracefully
        self.apcf.initialize_system(invalid_data['price'].values, invalid_data['volume'].values)
        results = self.apcf.run_backtest(invalid_data)
        
        # Verify error handling
        self.assertIsInstance(results, dict)
        self.assertIn('trades', results)
        self.assertIn('performance', results)
        
    def test_performance_metrics(self):
        """Test performance metric calculations"""
        # Create sample performance data
        self.apcf.runner = FerrisRunner([])
        self.apcf.runner.performance_log = [
            {'profit': 0.01, 'expected_profit': 0.015},
            {'profit': -0.005, 'expected_profit': 0.01},
            {'profit': 0.02, 'expected_profit': 0.018},
            {'profit': 0.015, 'expected_profit': 0.012},
            {'profit': -0.01, 'expected_profit': 0.008}
        ]
        
        # Calculate metrics
        metrics = self.apcf._calculate_summary_stats()
        
        # Verify calculations
        self.assertEqual(metrics['total_trades'], 5)
        self.assertAlmostEqual(metrics['win_rate'], 0.6)  # 3 wins out of 5
        self.assertAlmostEqual(metrics['avg_profit'], 0.006)  # Average of profits
        self.assertAlmostEqual(metrics['total_return'], 0.03)  # Sum of profits
        
if __name__ == '__main__':
    unittest.main() 