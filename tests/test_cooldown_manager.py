"""
Test suite for the Cooldown Management System
"""

import unittest
import time
from core.cooldown_manager import (
    CooldownManager, CooldownRule, CooldownScope, CooldownAction
)
from config.cooldown_config import get_cooldown_rules


class TestCooldownManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.cooldown_manager = CooldownManager()
        self.current_tick = 0
        self.current_time = time.time()

        # Add test rules
        for rule in get_cooldown_rules():
            self.cooldown_manager.add_rule(rule)

    def test_rule_activation(self):
        """Test cooldown rule activation"""
        # Test stop-loss rule activation
        event_data = {
            "asset_id": "BTC/USD",
            "volatility_level": "high",
            "loss_value": 150
        }

        self.cooldown_manager.register_event(
            "stop_loss_hit",
            event_data,
            self.current_tick,
            self.current_time
        )

        # Check if rule is active
        self.assertFalse(
            self.cooldown_manager.can_proceed(
                CooldownScope.ASSET_SPECIFIC,
                "BTC/USD",
                self.current_tick,
                self.current_time
            )
        )

    def test_cooldown_expiration(self):
        """Test cooldown rule expiration"""
        # Activate a rule
        event_data = {
            "asset_id": "BTC/USD",
            "volatility_level": "high",
            "loss_value": 150
        }

        self.cooldown_manager.register_event(
            "stop_loss_hit",
            event_data,
            self.current_tick,
            self.current_time
        )

        # Advance time past cooldown duration
        # 301 seconds (past 5-minute cooldown)
        future_time = self.current_time + 301

        # Check if cooldown has expired
        self.assertTrue(
            self.cooldown_manager.can_proceed(
                CooldownScope.ASSET_SPECIFIC,
                "BTC/USD",
                self.current_tick,
                future_time
            )
        )

    def test_multiple_rules(self):
        """Test multiple active cooldown rules"""
        # Activate global rule
        self.cooldown_manager.register_event(
            "basket_swap_completed",
            {},
            self.current_tick,
            self.current_time
        )

        # Activate asset-specific rule
        event_data = {
            "asset_id": "ETH/USD",
            "volatility_level": "high",
            "loss_value": 100
        }

        self.cooldown_manager.register_event(
            "stop_loss_hit",
            event_data,
            self.current_tick,
            self.current_time
        )

        # Check global scope
        self.assertFalse(
            self.cooldown_manager.can_proceed(
                CooldownScope.GLOBAL,
                None,
                self.current_tick,
                self.current_time
            )
        )

        # Check asset-specific scope
        self.assertFalse(
            self.cooldown_manager.can_proceed(
                CooldownScope.ASSET_SPECIFIC,
                "ETH/USD",
                self.current_tick,
                self.current_time
            )
        )

    def test_active_actions(self):
        """Test retrieval of active cooldown actions"""
        # Activate a rule
        event_data = {
            "asset_id": "BTC/USD",
            "volatility_level": "high",
            "loss_value": 150
        }

        self.cooldown_manager.register_event(
            "stop_loss_hit",
            event_data,
            self.current_tick,
            self.current_time
        )

        # Get active actions
        actions = self.cooldown_manager.get_active_actions(
            CooldownScope.ASSET_SPECIFIC,
            "BTC/USD"
        )

        # Verify actions
        self.assertIn(CooldownAction.BLOCK_NEW_ENTRIES, actions)
        self.assertIn(CooldownAction.INCREASE_STOP_DISTANCE, actions)

    def test_rule_priority(self):
        """Test rule priority handling"""
        # Create rules with different priorities
        high_priority_rule = CooldownRule(
            rule_id="HIGH_PRIORITY",
            trigger_events=["test_event"],
            cooldown_duration_seconds=60,
            scope=CooldownScope.GLOBAL,
            actions_during_cooldown=[CooldownAction.BLOCK_NEW_ENTRIES],
            priority=20
        )

        low_priority_rule = CooldownRule(
            rule_id="LOW_PRIORITY",
            trigger_events=["test_event"],
            cooldown_duration_seconds=30,
            scope=CooldownScope.GLOBAL,
            actions_during_cooldown=[CooldownAction.MONITOR_ONLY],
            priority=10
        )

        # Add rules to manager
        self.cooldown_manager.add_rule(high_priority_rule)
        self.cooldown_manager.add_rule(low_priority_rule)

        # Activate both rules
        self.cooldown_manager.register_event(
            "test_event",
            {},
            self.current_tick,
            self.current_time
        )

        # Get active actions
        actions = self.cooldown_manager.get_active_actions(
            CooldownScope.GLOBAL
        )

        # High priority rule should take precedence
        self.assertIn(CooldownAction.BLOCK_NEW_ENTRIES, actions)
        self.assertNotIn(CooldownAction.MONITOR_ONLY, actions)

    def test_active_cooldowns_info(self):
        """Test retrieval of active cooldowns information"""
        # Activate a rule
        event_data = {
            "asset_id": "BTC/USD",
            "volatility_level": "high",
            "loss_value": 150
        }

        self.cooldown_manager.register_event(
            "stop_loss_hit",
            event_data,
            self.current_tick,
            self.current_time
        )

        # Get active cooldowns info
        active_info = self.cooldown_manager.get_active_cooldowns()

        # Verify information
        self.assertGreater(len(active_info), 0)
        for key, info in active_info.items():
            self.assertIn('rule_id', info)
            self.assertIn('scope', info)
            self.assertIn('actions', info)
            self.assertIn('activation_time', info)


if __name__ == '__main__':
    unittest.main()