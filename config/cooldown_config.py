"""
Cooldown Management System Configuration
Contains default settings and rule definitions for the cooldown management system.
"""

from typing import Dict, Any, List
from core.cooldown_manager import CooldownRule, CooldownScope, CooldownAction

# Default cooldown durations
DEFAULT_COOLDOWN_CONFIG = {
    'stop_loss': {
        'high_volatility': 300,  # 5 minutes
        'medium_volatility': 180,  # 3 minutes
        'low_volatility': 60  # 1 minute
    },
    'take_profit': {
        'high_volatility': 180,  # 3 minutes
        'medium_volatility': 120,  # 2 minutes
        'low_volatility': 60  # 1 minute
    },
    'basket_swap': {
        'global': 600,  # 10 minutes
        'partial': 300  # 5 minutes
    },
    'decay_flip': {
        'asset_specific': 120,  # 2 minutes
        'strategy_specific': 180  # 3 minutes
    },
    'timing_based': {
        'paradox_resolution': 240,  # 4 minutes
        'phase_transition': 180,    # 3 minutes
        'recursion_depth': 300,     # 5 minutes
        'echo_memory': 120         # 2 minutes
    }
}

# Default cooldown rules
DEFAULT_COOLDOWN_RULES = [
    # Stop-loss cooldown rules
    CooldownRule(
        rule_id="SL_HIGH_VOL",
        trigger_events=["stop_loss_hit"],
        conditions=lambda data: data.get("volatility_level") == "high",
        cooldown_duration_seconds=DEFAULT_COOLDOWN_CONFIG['stop_loss']['high_volatility'],
        scope=CooldownScope.ASSET_SPECIFIC,
        actions_during_cooldown=[
            CooldownAction.BLOCK_NEW_ENTRIES,
            CooldownAction.INCREASE_STOP_DISTANCE
        ],
        priority=10
    ),
    
    # Take-profit cooldown rules
    CooldownRule(
        rule_id="TP_RAPID_WINS",
        trigger_events=["take_profit_hit"],
        conditions=lambda data: data.get("consecutive_wins", 0) >= 3,
        cooldown_duration_seconds=60,
        scope=CooldownScope.STRATEGY_SPECIFIC,
        actions_during_cooldown=[
            CooldownAction.REDUCE_POSITION_SIZE,
            CooldownAction.MONITOR_ONLY
        ],
        priority=5
    ),
    
    # Basket swap cooldown rules
    CooldownRule(
        rule_id="BASKET_SWAP_GLOBAL",
        trigger_events=["basket_swap_completed"],
        cooldown_duration_seconds=DEFAULT_COOLDOWN_CONFIG['basket_swap']['global'],
        scope=CooldownScope.GLOBAL,
        actions_during_cooldown=[
            CooldownAction.BLOCK_NEW_ENTRIES,
            CooldownAction.REDUCE_LEVERAGE
        ],
        priority=20
    ),
    
    # Decay flip cooldown rules
    CooldownRule(
        rule_id="DECAY_FLIP_ASSET",
        trigger_events=["decay_flip_detected"],
        cooldown_duration_seconds=DEFAULT_COOLDOWN_CONFIG['decay_flip']['asset_specific'],
        scope=CooldownScope.ASSET_SPECIFIC,
        actions_during_cooldown=[
            CooldownAction.BLOCK_NEW_ENTRIES,
            CooldownAction.FLAG_FOR_REVIEW
        ],
        priority=15
    ),
    
    # New timing-based cooldown rules
    CooldownRule(
        rule_id="TIMING_PARADOX",
        trigger_events=["paradox_detected"],
        conditions=lambda data: data.get("paradox_resolution", 0) > 0.8,
        cooldown_duration_seconds=DEFAULT_COOLDOWN_CONFIG['timing_based']['paradox_resolution'],
        scope=CooldownScope.GLOBAL,
        actions_during_cooldown=[
            CooldownAction.BLOCK_NEW_ENTRIES,
            CooldownAction.REDUCE_LEVERAGE,
            CooldownAction.FLAG_FOR_REVIEW
        ],
        priority=25
    ),
    
    CooldownRule(
        rule_id="TIMING_PHASE",
        trigger_events=["phase_transition"],
        conditions=lambda data: data.get("phase_alignment", 0) < 0.3,
        cooldown_duration_seconds=DEFAULT_COOLDOWN_CONFIG['timing_based']['phase_transition'],
        scope=CooldownScope.STRATEGY_SPECIFIC,
        actions_during_cooldown=[
            CooldownAction.MONITOR_ONLY,
            CooldownAction.REDUCE_POSITION_SIZE
        ],
        priority=15
    ),
    
    CooldownRule(
        rule_id="TIMING_RECURSION",
        trigger_events=["recursion_depth_change"],
        conditions=lambda data: data.get("recursion_depth", 0) > 50,
        cooldown_duration_seconds=DEFAULT_COOLDOWN_CONFIG['timing_based']['recursion_depth'],
        scope=CooldownScope.GLOBAL,
        actions_during_cooldown=[
            CooldownAction.BLOCK_NEW_ENTRIES,
            CooldownAction.REDUCE_LEVERAGE,
            CooldownAction.INCREASE_STOP_DISTANCE
        ],
        priority=30
    ),
    
    CooldownRule(
        rule_id="TIMING_ECHO",
        trigger_events=["echo_memory_anomaly"],
        conditions=lambda data: data.get("echo_memory", 0) > 0.9,
        cooldown_duration_seconds=DEFAULT_COOLDOWN_CONFIG['timing_based']['echo_memory'],
        scope=CooldownScope.ASSET_SPECIFIC,
        actions_during_cooldown=[
            CooldownAction.MONITOR_ONLY,
            CooldownAction.FLAG_FOR_REVIEW
        ],
        priority=10
    )
]

# Asset-specific cooldown configurations
ASSET_COOLDOWN_CONFIG = {
    'BTC/USD': {
        'stop_loss_cooldown': 240,  # 4 minutes
        'take_profit_cooldown': 120,  # 2 minutes
        'max_consecutive_trades': 3,
        'timing_based': {
            'paradox_resolution': 300,  # 5 minutes
            'phase_transition': 240,    # 4 minutes
            'recursion_depth': 360,     # 6 minutes
            'echo_memory': 180         # 3 minutes
        }
    },
    'ETH/USD': {
        'stop_loss_cooldown': 180,  # 3 minutes
        'take_profit_cooldown': 90,  # 1.5 minutes
        'max_consecutive_trades': 4,
        'timing_based': {
            'paradox_resolution': 240,  # 4 minutes
            'phase_transition': 180,    # 3 minutes
            'recursion_depth': 300,     # 5 minutes
            'echo_memory': 120         # 2 minutes
        }
    }
}

# Strategy-specific cooldown configurations
STRATEGY_COOLDOWN_CONFIG = {
    'scalping': {
        'min_cooldown_ticks': 5,
        'max_trades_per_minute': 12,
        'cooldown_after_loss': 30,  # seconds
        'timing_based': {
            'paradox_resolution': 180,  # 3 minutes
            'phase_transition': 120,    # 2 minutes
            'recursion_depth': 240,     # 4 minutes
            'echo_memory': 60          # 1 minute
        }
    },
    'swing_trading': {
        'min_cooldown_ticks': 20,
        'max_trades_per_hour': 6,
        'cooldown_after_loss': 300,  # seconds
        'timing_based': {
            'paradox_resolution': 360,  # 6 minutes
            'phase_transition': 300,    # 5 minutes
            'recursion_depth': 420,     # 7 minutes
            'echo_memory': 240         # 4 minutes
        }
    }
}

def get_cooldown_rules(asset: str = None, strategy: str = None) -> List[CooldownRule]:
    """
    Get cooldown rules with asset and strategy specific modifications.
    
    Args:
        asset: Asset symbol (e.g., 'BTC/USD')
        strategy: Strategy identifier
        
    Returns:
        List of CooldownRule objects
    """
    rules = DEFAULT_COOLDOWN_RULES.copy()
    
    # Apply asset-specific modifications
    if asset and asset in ASSET_COOLDOWN_CONFIG:
        for rule in rules:
            if rule.rule_id.startswith("SL_") and rule.scope == CooldownScope.ASSET_SPECIFIC:
                rule.cooldown_duration_seconds = ASSET_COOLDOWN_CONFIG[asset]['stop_loss_cooldown']
            elif rule.rule_id.startswith("TP_") and rule.scope == CooldownScope.ASSET_SPECIFIC:
                rule.cooldown_duration_seconds = ASSET_COOLDOWN_CONFIG[asset]['take_profit_cooldown']
            elif rule.rule_id.startswith("TIMING_") and rule.scope == CooldownScope.ASSET_SPECIFIC:
                timing_config = ASSET_COOLDOWN_CONFIG[asset]['timing_based']
                if rule.rule_id == "TIMING_PARADOX":
                    rule.cooldown_duration_seconds = timing_config['paradox_resolution']
                elif rule.rule_id == "TIMING_PHASE":
                    rule.cooldown_duration_seconds = timing_config['phase_transition']
                elif rule.rule_id == "TIMING_RECURSION":
                    rule.cooldown_duration_seconds = timing_config['recursion_depth']
                elif rule.rule_id == "TIMING_ECHO":
                    rule.cooldown_duration_seconds = timing_config['echo_memory']
    
    # Apply strategy-specific modifications
    if strategy and strategy in STRATEGY_COOLDOWN_CONFIG:
        for rule in rules:
            if rule.scope == CooldownScope.STRATEGY_SPECIFIC:
                if rule.rule_id.startswith("TP_"):
                    rule.cooldown_duration_ticks = STRATEGY_COOLDOWN_CONFIG[strategy]['min_cooldown_ticks']
                elif rule.rule_id.startswith("TIMING_"):
                    timing_config = STRATEGY_COOLDOWN_CONFIG[strategy]['timing_based']
                    if rule.rule_id == "TIMING_PARADOX":
                        rule.cooldown_duration_seconds = timing_config['paradox_resolution']
                    elif rule.rule_id == "TIMING_PHASE":
                        rule.cooldown_duration_seconds = timing_config['phase_transition']
                    elif rule.rule_id == "TIMING_RECURSION":
                        rule.cooldown_duration_seconds = timing_config['recursion_depth']
                    elif rule.rule_id == "TIMING_ECHO":
                        rule.cooldown_duration_seconds = timing_config['echo_memory']
    
    return rules 