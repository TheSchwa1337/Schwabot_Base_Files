"""
Test Suite for Enhanced Tesseract Processor
=========================================

Comprehensive test coverage for EnhancedTesseractProcessor including:
- Unit tests for core functionality
- Integration tests with risk monitoring
- Synthetic market data generation
- Alert simulation and validation
- Async operation testing
"""

import unittest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import yaml
from pathlib import Path
import json

from core.enhanced_tesseract_processor import EnhancedTesseractProcessor
from core.quantum_cellular_risk_monitor import AdvancedRiskMetrics
from core.risk_indexer import RiskIndexer

class MockQuantumCellularRiskMonitor:
    """Mock implementation of QuantumCellularRiskMonitor for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_regime = "NORMAL"
        self.coherence = 0.8
        self.homeostasis = 0.7
        self.fhs_var = 0.02
        self.hrp_weights = [0.25, 0.25, 0.25, 0.25]
        
    async def update_risk_state(self, market_data: Dict[str, Any]) -> AdvancedRiskMetrics:
        """Mock risk state update"""
        # Simulate some risk metric variations based on market data
        volatility = market_data.get('volatility', 0.0)
        self.fhs_var = min(0.05, max(0.01, volatility * 0.1))
        self.coherence = max(0.3, 1.0 - volatility)
        
        if volatility > 0.03:
            self.current_regime = "HIGH_VOLATILITY"
        elif volatility < 0.01:
            self.current_regime = "LOW_VOLATILITY"
        else:
            self.current_regime = "NORMAL"
            
        return AdvancedRiskMetrics(
            current_regime=self.current_regime,
            coherence=self.coherence,
            homeostasis=self.homeostasis,
            fhs_var=self.fhs_var,
            hrp_weights=self.hrp_weights
        )

class TestEnhancedTesseractProcessor(unittest.TestCase):
    """Core test suite for EnhancedTesseractProcessor"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test configuration and fixtures"""
        # Create test config
        cls.test_config = {
            'processing': {
                'baseline_reset_flip_frequency': 100
            },
            'dimensions': {
                'labels': ['price', 'volume', 'volatility', 'momentum', 
                          'trend', 'liquidity', 'sentiment', 'correlation']
            },
            'monitoring': {
                'alerts': {
                    'var_threshold': 0.05,
                    'var_indexed_threshold': 1.5,
                    'coherence_threshold': 0.5,
                    'coherence_indexed_threshold': 0.8
                }
            },
            'debug': {
                'test_mode': True
            }
        }
        
        # Save test config
        cls.config_path = Path("test_tesseract_config.yaml")
        with open(cls.config_path, 'w') as f:
            yaml.dump(cls.test_config, f)
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if cls.config_path.exists():
            cls.config_path.unlink()
            
    def setUp(self):
        """Initialize test fixtures"""
        # Patch the risk monitor with our mock
        self.risk_monitor_patcher = patch(
            'core.enhanced_tesseract_processor.QuantumCellularRiskMonitor',
            MockQuantumCellularRiskMonitor
        )
        self.risk_monitor_patcher.start()
        
        # Initialize processor with test config
        self.processor = EnhancedTesseractProcessor(str(self.config_path))
        
        # Initialize test market data
        self.market_data = self._generate_synthetic_market_data()
        
    def tearDown(self):
        """Clean up test fixtures"""
        self.risk_monitor_patcher.stop()
        
    def _generate_synthetic_market_data(self) -> Dict[str, Any]:
        """Generate synthetic market data for testing"""
        return {
            'price': 100.0 + np.random.normal(0, 1),
            'volume': 1000000 * (1 + np.random.normal(0, 0.1)),
            'volatility': 0.02 * (1 + np.random.normal(0, 0.2)),
            'momentum': np.random.normal(0, 1),
            'trend': np.random.normal(0, 1),
            'liquidity': 1000000 * (1 + np.random.normal(0, 0.1)),
            'sentiment': np.random.normal(0, 1),
            'correlation': np.random.normal(0, 0.2),
            'timestamp': datetime.now().timestamp()
        }
        
    async def _process_market_tick(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method to process market tick"""
        return await self.processor.process_market_tick(market_data)
        
    def test_initialization(self):
        """Test processor initialization"""
        self.assertIsNotNone(self.processor.risk_monitor)
        self.assertIsNotNone(self.processor.risk_indexer)
        self.assertEqual(self.processor.dimension_labels, self.test_config['dimensions']['labels'])
        self.assertTrue(self.processor.test_mode)
        
    def test_extract_8d_pattern(self):
        """Test 8D pattern extraction"""
        pattern = self.processor._extract_8d_pattern(self.market_data)
        
        self.assertEqual(len(pattern), 8)
        self.assertTrue(all(isinstance(x, float) for x in pattern))
        self.assertEqual(pattern[0], self.market_data['price'])
        self.assertEqual(pattern[1], self.market_data['volume'])
        
    async def test_market_tick_processing(self):
        """Test market tick processing"""
        result = await self._process_market_tick(self.market_data)
        
        self.assertIn('signal_strength', result)
        self.assertIn('position_size', result)
        self.assertIn('risk_metrics_raw', result)
        self.assertIn('risk_metrics_indexed', result)
        self.assertIn('pattern', result)
        self.assertIn('regime', result)
        self.assertIn('coherence', result)
        self.assertIn('homeostasis', result)
        self.assertIn('timestamp', result)
        
    async def test_risk_adjusted_signal(self):
        """Test risk-adjusted signal calculation"""
        # Test with normal market conditions
        normal_data = self._generate_synthetic_market_data()
        normal_data['volatility'] = 0.02
        result = await self._process_market_tick(normal_data)
        self.assertTrue(-1.0 <= result['signal_strength'] <= 1.0)
        
        # Test with high volatility
        high_vol_data = self._generate_synthetic_market_data()
        high_vol_data['volatility'] = 0.05
        result = await self._process_market_tick(high_vol_data)
        self.assertTrue(-1.0 <= result['signal_strength'] <= 1.0)
        self.assertLess(abs(result['signal_strength']), 
                       abs(result['position_size']))
        
    async def test_regime_aware_position_sizing(self):
        """Test regime-aware position sizing"""
        # Test normal regime
        normal_data = self._generate_synthetic_market_data()
        normal_data['volatility'] = 0.02
        result = await self._process_market_tick(normal_data)
        self.assertTrue(0.01 <= result['position_size'] <= 0.5)
        
        # Test high volatility regime
        high_vol_data = self._generate_synthetic_market_data()
        high_vol_data['volatility'] = 0.05
        result = await self._process_market_tick(high_vol_data)
        self.assertLess(result['position_size'], 0.3)  # Should be reduced
        
    async def test_risk_alerts(self):
        """Test risk alert generation"""
        # Test VaR breach
        var_breach_data = self._generate_synthetic_market_data()
        var_breach_data['volatility'] = 0.1  # High volatility to trigger VaR alert
        result = await self._process_market_tick(var_breach_data)
        self.assertEqual(result['regime'], "HIGH_VOLATILITY")
        
        # Test coherence alert
        low_coherence_data = self._generate_synthetic_market_data()
        low_coherence_data['volatility'] = 0.08  # High enough to reduce coherence
        result = await self._process_market_tick(low_coherence_data)
        self.assertLess(result['coherence'], 0.5)
        
    async def test_async_operation(self):
        """Test async operation with multiple ticks"""
        # Process multiple ticks
        results = []
        for _ in range(5):
            market_data = self._generate_synthetic_market_data()
            result = await self._process_market_tick(market_data)
            results.append(result)
            
        # Verify results
        self.assertEqual(len(results), 5)
        self.assertTrue(all(isinstance(r, dict) for r in results))
        self.assertTrue(all('signal_strength' in r for r in results))
        
    def test_export_shell_map(self):
        """Test shell map export functionality"""
        # Generate test data
        market_data = self._generate_synthetic_market_data()
        result = asyncio.run(self._process_market_tick(market_data))
        
        # Export to temporary file
        temp_path = Path("temp_shell_map.json")
        self.processor.export_shell_map(str(temp_path))
        
        # Verify file contents
        self.assertTrue(temp_path.exists())
        with open(temp_path) as f:
            data = json.load(f)
            self.assertIn("timestamp", data)
            self.assertIn("states", data)
            self.assertTrue(len(data["states"]) > 0)
            
        # Cleanup
        temp_path.unlink()
        
    async def test_baseline_reset(self):
        """Test baseline reset functionality"""
        # Process multiple ticks to trigger reset
        for _ in range(self.processor.flip_frequency + 1):
            market_data = self._generate_synthetic_market_data()
            await self._process_market_tick(market_data)
            
        # Verify baseline was reset
        self.assertEqual(self.processor.tick_step_counter, self.processor.flip_frequency + 1)
        
    def test_error_handling(self):
        """Test error handling with invalid data"""
        # Test with missing required fields
        invalid_data = {'price': 100.0}  # Missing other fields
        with self.assertRaises(KeyError):
            asyncio.run(self._process_market_tick(invalid_data))
            
        # Test with invalid data types
        invalid_data = {k: "invalid" for k in self.market_data.keys()}
        with self.assertRaises(TypeError):
            asyncio.run(self._process_market_tick(invalid_data))

if __name__ == "__main__":
    unittest.main() 