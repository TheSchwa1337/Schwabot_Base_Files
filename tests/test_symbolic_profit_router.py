#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Symbolic Profit Router - Schwabot UROS v1.0
===============================================

Comprehensive tests for the symbolic profit router and 2-bit mapping system.
"""

import hashlib
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules to test
try:
    from core.symbolic_profit_router import (
        SymbolicProfitRouter, ProfitTier, ProfitVaultAction, 
        ProfitTrigger, TriggerType, BitPhase, route_profit_phase,
        hash_to_strategy, fold_hash_to_2bit
    )
    from core.profit_routing_engine import (
        ProfitRoutingEngine, RoutingDecision, ProfitRoutingConfig,
        route_profit, activate_profit_vault
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    MODULES_AVAILABLE = False


class TestSymbolicProfitRouter(unittest.TestCase):
    """Test cases for the SymbolicProfitRouter class."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        self.router = SymbolicProfitRouter()
        self.test_hash = hashlib.sha256(b"test_vault_trigger:BTC:mid:24hr").hexdigest()

    def test_fold_hash_to_2bit(self):
        """Test hash folding to 2-bit sequences."""
        # Test with known hash
        result = self.router.fold_hash_to_2bit(self.test_hash)
        self.assertIn(result, ["00", "01", "10", "11"])
        
        # Test with different hash
        different_hash = hashlib.sha256(b"different_trigger").hexdigest()
        result2 = self.router.fold_hash_to_2bit(different_hash)
        self.assertIn(result2, ["00", "01", "10", "11"])
        
        # Test error handling
        result3 = self.router.fold_hash_to_2bit("invalid_hash")
        self.assertEqual(result3, "00")  # Should default to IDLE

    def test_hash_to_strategy(self):
        """Test hash to strategy conversion."""
        # Test with symbolic input
        strategy = self.router.hash_to_strategy("vault_trigger::BTC::long::32hr")
        
        self.assertEqual(strategy["asset"], "BTC")
        self.assertEqual(strategy["tier"], "long")
        self.assertEqual(strategy["expected_horizon"], "32hr")
        self.assertGreaterEqual(strategy["confidence"], 0.0)
        self.assertLessEqual(strategy["confidence"], 1.0)
        self.assertIn(strategy["bit_sequence"], ["00", "01", "10", "11"])
        
        # Test with actual hash
        strategy2 = self.router.hash_to_strategy(self.test_hash)
        self.assertIn("asset", strategy2)
        self.assertIn("confidence", strategy2)
        self.assertIn("bit_sequence", strategy2)

    def test_calculate_hash_confidence(self):
        """Test hash confidence calculation."""
        # Test with balanced hash
        balanced_hash = "a" * 64  # All same character
        confidence = self.router._calculate_hash_confidence(balanced_hash)
        self.assertEqual(confidence, 0.5)  # Should be neutral
        
        # Test with varied hash
        varied_hash = "0123456789abcdef" * 4
        confidence2 = self.router._calculate_hash_confidence(varied_hash)
        self.assertGreaterEqual(confidence2, 0.0)
        self.assertLessEqual(confidence2, 1.0)

    def test_route_profit_phase(self):
        """Test profit phase routing."""
        # Test with 2-bit phase and up bias
        vault_action = self.router.route_profit_phase("2-bit", "up", "10", "BTC", 0.15)
        
        self.assertIsInstance(vault_action, ProfitVaultAction)
        self.assertIsInstance(vault_action.tier, ProfitTier)
        self.assertGreaterEqual(vault_action.allocation, 0.0)
        self.assertLessEqual(vault_action.allocation, 1.0)
        self.assertEqual(vault_action.trigger.asset, "BTC")
        self.assertEqual(vault_action.trigger.expected_return, 0.15)
        
        # Test with override trigger
        vault_action2 = self.router.route_profit_phase("2-bit", "up", "11", "ETH", 0.25)
        self.assertIsInstance(vault_action2, ProfitVaultAction)

    def test_determine_trigger_type(self):
        """Test trigger type determination."""
        # Test emoji hash match
        trigger_type = self.router._determine_trigger_type("2-bit", "up", "10")
        self.assertEqual(trigger_type, TriggerType.EMOJI_HASH_MATCH)
        
        # Test symbolic override
        trigger_type2 = self.router._determine_trigger_type("2-bit", "up", "11")
        self.assertEqual(trigger_type2, TriggerType.SYMBOLIC_OVERRIDE)
        
        # Test momentum shift
        trigger_type3 = self.router._determine_trigger_type("momentum", "up", "01")
        self.assertEqual(trigger_type3, TriggerType.MOMENTUM_SHIFT)

    def test_determine_tier(self):
        """Test tier determination."""
        # Test high return and confidence
        tier = self.router._determine_tier(ProfitTier.MID, 0.30, 0.9)
        self.assertEqual(tier, ProfitTier.LONG)
        
        # Test medium return and confidence
        tier2 = self.router._determine_tier(ProfitTier.SHORT, 0.15, 0.7)
        self.assertEqual(tier2, ProfitTier.MID)
        
        # Test low return
        tier3 = self.router._determine_tier(ProfitTier.LONG, 0.03, 0.8)
        self.assertEqual(tier3, ProfitTier.SHORT)

    def test_calculate_allocation(self):
        """Test allocation calculation."""
        # Test with high tier and good metrics
        allocation = self.router._calculate_allocation(ProfitTier.LONG, 0.25, 0.9)
        self.assertGreater(allocation, 0.0)
        self.assertLessEqual(allocation, self.router.config["max_allocation"])
        
        # Test with low tier
        allocation2 = self.router._calculate_allocation(ProfitTier.SHORT, 0.05, 0.5)
        self.assertGreaterEqual(allocation2, 0.0)

    def test_get_routing_stats(self):
        """Test routing statistics."""
        # Route some test actions first
        self.router.route_profit_phase("2-bit", "up", "10", "BTC", 0.15)
        self.router.route_profit_phase("2-bit", "down", "01", "ETH", 0.08)
        
        stats = self.router.get_routing_stats()
        
        self.assertIn("total_actions", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("tier_distribution", stats)
        self.assertIn("hash_registry_size", stats)
        self.assertIn("log_entries", stats)
        
        self.assertGreaterEqual(stats["total_actions"], 2)
        self.assertGreaterEqual(stats["log_entries"], 2)


class TestProfitRoutingEngine(unittest.TestCase):
    """Test cases for the ProfitRoutingEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        self.config = ProfitRoutingConfig(
            enable_2bit_mapping=True,
            enable_hash_triggers=True,
            enable_recursive_learning=True,
            confidence_threshold=0.75,
            max_allocation=0.7,
            min_expected_return=0.042,
            enable_temporal_correction=True,
            enable_failure_recovery=True,
            log_level="INFO"
        )
        self.engine = ProfitRoutingEngine(self.config)

    def test_route_profit(self):
        """Test main profit routing function."""
        payload = {
            "phase": "2-bit",
            "flip_bias": "up",
            "asset": "BTC",
            "expected_return": 0.15,
            "hash_input": "vault_trigger::BTC::mid::24hr"
        }
        
        result = self.engine.route_profit(payload)
        
        self.assertIsInstance(result.decision, RoutingDecision)
        self.assertIsInstance(result.tier, ProfitTier)
        self.assertGreaterEqual(result.allocation, 0.0)
        self.assertLessEqual(result.allocation, 1.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_activate_profit_vault(self):
        """Test profit vault activation."""
        result = self.engine.activate_profit_vault(
            level="mid", 
            trigger="emoji_hash_match", 
            asset="BTC", 
            expected_return=0.12
        )
        
        self.assertIsInstance(result.decision, RoutingDecision)
        self.assertIsInstance(result.tier, ProfitTier)
        self.assertGreaterEqual(result.allocation, 0.0)

    def test_temporal_correction(self):
        """Test temporal correction logic."""
        # Test during normal hours
        with patch('core.profit_routing_engine.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 14, 0)  # 2 PM
            is_correct = self.engine._is_temporally_correct("BTC")
            self.assertTrue(is_correct)
        
        # Test during low liquidity hours
        with patch('core.profit_routing_engine.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 4, 0)  # 4 AM
            is_correct = self.engine._is_temporally_correct("BTC")
            self.assertFalse(is_correct)

    def test_failure_recovery(self):
        """Test failure recovery logic."""
        # Set failure recovery
        self.engine.set_failure_recovery("BTC", 24)
        
        # Check if in recovery mode
        self.assertTrue(self.engine._is_in_recovery_mode("BTC"))
        
        # Test routing during recovery
        payload = {
            "phase": "2-bit",
            "flip_bias": "up",
            "asset": "BTC",
            "expected_return": 0.15
        }
        
        result = self.engine.route_profit(payload)
        self.assertIn("recovery_mode", result.metadata)
        
        # Clear recovery
        self.engine.clear_failure_recovery("BTC")
        self.assertFalse(self.engine._is_in_recovery_mode("BTC"))

    def test_get_routing_stats(self):
        """Test routing statistics."""
        # Route some test actions first
        payload = {
            "phase": "2-bit",
            "flip_bias": "up",
            "asset": "BTC",
            "expected_return": 0.15
        }
        self.engine.route_profit(payload)
        
        stats = self.engine.get_routing_stats()
        
        self.assertIn("total_decisions", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("decision_distribution", stats)
        self.assertIn("tier_distribution", stats)
        self.assertIn("vault_states", stats)
        self.assertIn("failure_recovery_count", stats)
        self.assertIn("symbolic_router_stats", stats)
        
        self.assertGreaterEqual(stats["total_decisions"], 1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_route_profit_phase_convenience(self):
        """Test route_profit_phase convenience function."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        vault_action = route_profit_phase("2-bit", "up", "10", "BTC", 0.15)
        self.assertIsInstance(vault_action, ProfitVaultAction)

    def test_hash_to_strategy_convenience(self):
        """Test hash_to_strategy convenience function."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        strategy = hash_to_strategy("vault_trigger::BTC::long::32hr")
        self.assertIn("asset", strategy)
        self.assertIn("tier", strategy)

    def test_fold_hash_to_2bit_convenience(self):
        """Test fold_hash_to_2bit convenience function."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        test_hash = hashlib.sha256(b"test").hexdigest()
        result = fold_hash_to_2bit(test_hash)
        self.assertIn(result, ["00", "01", "10", "11"])

    def test_route_profit_convenience(self):
        """Test route_profit convenience function."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        payload = {
            "phase": "2-bit",
            "flip_bias": "up",
            "asset": "BTC",
            "expected_return": 0.15
        }
        result = route_profit(payload)
        self.assertIsInstance(result.decision, RoutingDecision)

    def test_activate_profit_vault_convenience(self):
        """Test activate_profit_vault convenience function."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        result = activate_profit_vault("mid", "emoji_hash_match", "BTC")
        self.assertIsInstance(result.decision, RoutingDecision)


class TestBitPhaseMapping(unittest.TestCase):
    """Test 2-bit phase mapping system."""

    def test_bit_phase_enum(self):
        """Test BitPhase enum values."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        self.assertEqual(BitPhase.IDLE.value, "00")
        self.assertEqual(BitPhase.SOFT_TRIGGER.value, "01")
        self.assertEqual(BitPhase.HARD_ENTRY.value, "10")
        self.assertEqual(BitPhase.OVERRIDE.value, "11")

    def test_profit_tier_enum(self):
        """Test ProfitTier enum values."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        self.assertEqual(ProfitTier.SHORT.value, "short")
        self.assertEqual(ProfitTier.MID.value, "mid")
        self.assertEqual(ProfitTier.LONG.value, "long")
        self.assertEqual(ProfitTier.OVERRIDE.value, "override")

    def test_trigger_type_enum(self):
        """Test TriggerType enum values."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        self.assertEqual(TriggerType.EMOJI_HASH_MATCH.value, "emoji_hash_match")
        self.assertEqual(TriggerType.MOMENTUM_SHIFT.value, "momentum_shift")
        self.assertEqual(TriggerType.VOLUME_SPIKE.value, "volume_spike")
        self.assertEqual(TriggerType.PRICE_BREAKOUT.value, "price_breakout")
        self.assertEqual(TriggerType.SYMBOLIC_OVERRIDE.value, "symbolic_override")


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2) 