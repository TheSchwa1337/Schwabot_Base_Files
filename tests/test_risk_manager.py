"""
Test suite for the Risk Management System
"""

import unittest
from datetime import datetime
import numpy as np
from core.risk_manager import RiskManager, PositionRisk, RiskLevel

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = RiskManager(
            max_portfolio_risk=0.02,
            max_position_size=0.1,
            max_drawdown=0.15,
            var_confidence=0.95,
            update_interval=1.0
        )
        
    def test_kelly_fraction_calculation(self):
        """Test Kelly Criterion fraction calculation"""
        # Test case 1: 60% win rate, 1:1 win/loss ratio
        kelly = self.risk_manager.calculate_kelly_fraction(0.6, 1.0)
        self.assertAlmostEqual(kelly, 0.2, places=2)
        
        # Test case 2: 70% win rate, 2:1 win/loss ratio
        kelly = self.risk_manager.calculate_kelly_fraction(0.7, 2.0)
        self.assertAlmostEqual(kelly, 0.4, places=2)
        
        # Test case 3: Invalid win rate
        kelly = self.risk_manager.calculate_kelly_fraction(1.5, 1.0)
        self.assertEqual(kelly, 0.0)
        
    def test_position_size_calculation(self):
        """Test position size calculation"""
        # Test case 1: Normal conditions
        size = self.risk_manager.calculate_position_size(
            symbol="BTC/USD",
            price=50000.0,
            volatility=0.02,
            win_rate=0.6,
            win_loss_ratio=1.0
        )
        self.assertGreater(size, 0.0)
        self.assertLessEqual(size, self.risk_manager.max_position_size)
        
        # Test case 2: High volatility
        size = self.risk_manager.calculate_position_size(
            symbol="BTC/USD",
            price=50000.0,
            volatility=0.5,
            win_rate=0.6,
            win_loss_ratio=1.0
        )
        self.assertLess(size, 0.1)  # Should be reduced due to high volatility
        
    def test_dynamic_stop_loss(self):
        """Test dynamic stop-loss calculation"""
        stop_loss, take_profit = self.risk_manager.calculate_dynamic_stop_loss(
            symbol="BTC/USD",
            price=50000.0,
            volatility=0.02,
            atr=1000.0
        )
        
        self.assertLess(stop_loss, 50000.0)  # Stop loss should be below entry
        self.assertGreater(take_profit, 50000.0)  # Take profit should be above entry
        
    def test_portfolio_risk_update(self):
        """Test portfolio risk metrics update"""
        # Create test positions
        positions = {
            "BTC/USD": PositionRisk(
                symbol="BTC/USD",
                size=0.1,
                entry_price=50000.0,
                current_price=51000.0,
                stop_loss=49000.0,
                take_profit=53000.0,
                risk_level=RiskLevel.MEDIUM,
                kelly_fraction=0.2,
                max_drawdown=0.05,
                volatility=0.02,
                timestamp=datetime.now()
            ),
            "ETH/USD": PositionRisk(
                symbol="ETH/USD",
                size=0.05,
                entry_price=3000.0,
                current_price=3100.0,
                stop_loss=2900.0,
                take_profit=3200.0,
                risk_level=RiskLevel.LOW,
                kelly_fraction=0.1,
                max_drawdown=0.03,
                volatility=0.015,
                timestamp=datetime.now()
            )
        }
        
        # Update portfolio risk
        portfolio_risk = self.risk_manager.update_portfolio_risk(positions)
        
        # Verify portfolio risk metrics
        self.assertIsNotNone(portfolio_risk)
        self.assertEqual(portfolio_risk.total_exposure, 0.15)  # 0.1 + 0.05
        self.assertGreater(portfolio_risk.portfolio_volatility, 0.0)
        self.assertGreater(portfolio_risk.var_95, 0.0)
        
    def test_risk_limits(self):
        """Test risk limit checks"""
        # Create test position
        position = PositionRisk(
            symbol="BTC/USD",
            size=0.2,  # Exceeds max_position_size
            entry_price=50000.0,
            current_price=51000.0,
            stop_loss=49000.0,
            take_profit=53000.0,
            risk_level=RiskLevel.MEDIUM,
            kelly_fraction=0.2,
            max_drawdown=0.05,
            volatility=0.02,
            timestamp=datetime.now()
        )
        
        # Test position size limit
        self.assertFalse(self.risk_manager.check_risk_limits("BTC/USD", position))
        
        # Test with valid position size
        position.size = 0.05
        self.assertTrue(self.risk_manager.check_risk_limits("BTC/USD", position))
        
    def test_risk_report(self):
        """Test risk report generation"""
        # Create test positions
        self.risk_manager.positions = {
            "BTC/USD": PositionRisk(
                symbol="BTC/USD",
                size=0.1,
                entry_price=50000.0,
                current_price=51000.0,
                stop_loss=49000.0,
                take_profit=53000.0,
                risk_level=RiskLevel.MEDIUM,
                kelly_fraction=0.2,
                max_drawdown=0.05,
                volatility=0.02,
                timestamp=datetime.now()
            )
        }
        
        # Update portfolio risk
        self.risk_manager.update_portfolio_risk(self.risk_manager.positions)
        
        # Get risk report
        report = self.risk_manager.get_risk_report()
        
        # Verify report structure
        self.assertIn("timestamp", report)
        self.assertIn("total_exposure", report)
        self.assertIn("current_drawdown", report)
        self.assertIn("portfolio_volatility", report)
        self.assertIn("risk_level", report)
        self.assertIn("var_95", report)
        self.assertIn("position_risks", report)
        
if __name__ == '__main__':
    unittest.main() 