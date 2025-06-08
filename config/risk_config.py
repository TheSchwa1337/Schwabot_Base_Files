"""
Risk Management System Configuration
Contains default settings and configuration parameters for the risk management system.
"""

from typing import Dict, Any, Optional
from copy import deepcopy

# ============================================================================
# 1. Portfolio-Level Default Risk Configuration
# ============================================================================

DEFAULT_RISK_CONFIG: Dict[str, Any] = {
    "max_portfolio_risk": 0.02,
    "max_position_size": 0.10,
    "max_drawdown": 0.15,
    "var_confidence": 0.95,

    "kelly_fraction_cap": 0.5,
    "min_position_size": 0.01,
    "volatility_adjustment": 1.0,

    "atr_multiplier": 2.0,
    "profit_target_ratio": 1.5,
    "trailing_stop_activation": 0.02,

    "smart_money": {
        "order_spoof_detection": True,
        "min_liquidity_imbalance_ratio": 1.5,
        "whale_wall_threshold": 250000.0,
        "hidden_order_trailback_window": 12,
        "price_spread_regulation_threshold": 0.003,
        "swing_rings": {
            "small": 0.05,
            "medium": 0.15,
            "large": 0.30
        }
    },

    "price_regulation": {
        "max_spread_ratio": 0.005,
        "tick_velocity_alert_threshold": 2.0,
        "mean_reversion_tolerance": 0.01,
        "price_band_monitoring": True,
        "band_margin_ratio": 0.02,
        "dual_channel_flow": True,
        "flow_patterns": [
            "implicit", "explicit", "implicit", "explicit",
            "explicit", "implicit", "explicit", "explicit"
        ]
    },

    "drawup_scale_start": 0.08,
    "drawup_scale_slope": 0.5,

    "risk_levels": {
        "extreme": {"exposure": 0.9, "volatility": 0.3, "var": 0.1},
        "high":    {"exposure": 0.7, "volatility": 0.2, "var": 0.05},
        "medium":  {"exposure": 0.5, "volatility": 0.1, "var": 0.02}
    },

    "alert_thresholds": {
        "drawdown": 0.10,
        "volatility": 0.20,
        "exposure": 0.80,
        "spoof": 0.50,
        "wall": 0.60,
        "spread": 0.0015
    },

    "drawdown_response": {
        "freeze_allocation_on_max_drawdown": True,
        "gradual_risk_reset_ticks": 48
    },

    "update_interval": 1.0,
    "history_size": 1000
}

# ============================================================================
# 2. Asset-Specific Overrides
# ============================================================================

ASSET_RISK_CONFIG: Dict[str, Dict[str, Any]] = {
    "BTC/USD": {
        "max_position_size": 0.15,
        "min_position_size": 0.02,
        "volatility_adjustment": 1.2,
        "smart_money": {
            "whale_wall_threshold": 300000.0,
            "swing_rings": {
                "small": 0.06,
                "medium": 0.18,
                "large": 0.35
            }
        },
        "price_regulation": {
            "max_spread_ratio": 0.004,
            "dual_channel_flow": True
        }
    },
    "ETH/USD": {
        "max_position_size": 0.10,
        "min_position_size": 0.01,
        "volatility_adjustment": 1.0
    }
}

# ============================================================================
# 3. Correlation-Aware Filtering Parameters
# ============================================================================

CORRELATION_CONFIG: Dict[str, Any] = {
    "high_correlation_threshold": 0.70,
    "max_correlated_exposure": 0.30,
    "min_correlation_threshold": -0.30,
    "correlation_filtering": {
        "dynamic_allocation_adjustment": True,
        "correlation_lookback_window": 72,
        "target_correlation_diversity_score": 0.65
    }
}

# ============================================================================
# 4. Monitoring Configuration
# ============================================================================

MONITORING_CONFIG: Dict[str, Any] = {
    "report_interval": 60,
    "alert_channels": ["log", "email"],
    "metrics_to_track": [
        "total_exposure", "portfolio_volatility", "current_drawdown",
        "var_95", "correlation_matrix", "position_risks",
        "smart_money_score", "spoof_score", "wall_score",
        "bid_ask_spread", "swing_ring_class"
    ]
}

# ============================================================================
# 5. Utility Functions for Risk Fetch + Merging
# ============================================================================

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result

def get_risk_config(asset: Optional[str] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_RISK_CONFIG)
    if asset and asset in ASSET_RISK_CONFIG:
        config = deep_merge(config, ASSET_RISK_CONFIG[asset])
    return config

def get_monitoring_config() -> Dict[str, Any]:
    return deepcopy(MONITORING_CONFIG)

def get_correlation_config() -> Dict[str, Any]:
    return deepcopy(CORRELATION_CONFIG)

# ============================================================================
# 6. Runtime Sanity Check (Optional CLI Test)
# ============================================================================

if __name__ == "__main__":
    import json
    print("=== Default Config ===")
    print(json.dumps(get_risk_config(), indent=2))
    print("\n=== BTC/USD Override ===")
    print(json.dumps(get_risk_config("BTC/USD"), indent=2))

# Add to risk_config.py
BASKET_RISK_CONFIG = {
    "core_stability": {
        "risk_metrics": {
            "cvar_confidence": 0.975,
            "max_drawdown": 0.10,
            "sharpe_target": 1.5,
            "sortino_target": 2.0
        },
        "position_sizing": {
            "kelly_fraction_cap": 0.2,
            "min_position_size": 0.01,
            "max_position_size": 0.15
        }
    },
    "volatility_hedge": {
        "risk_metrics": {
            "iv_rv_spread_threshold": 0.05,
            "skew_threshold": 0.1,
            "gamma_threshold": 0.15
        },
        "position_sizing": {
            "vega_cap": 100000,
            "delta_cap": 0.3,
            "gamma_cap": 0.1
        }
    },
    # ... other baskets
}

# Add to risk_config.py
PORTFOLIO_RISK_CONFIG = {
    "aggregation": {
        "copula_type": "gaussian",
        "correlation_window": 100,
        "var_confidence": 0.99,
        "cvar_confidence": 0.975
    },
    "circuit_breakers": {
        "total_drawdown": 0.15,
        "basket_drawdown": 0.2,
        "volatility_spike": 0.5,
        "liquidity_crisis": 0.3
    },
    "monitoring": {
        "update_interval": 1.0,
        "history_size": 1000,
        "alert_thresholds": {
            "warning": {
                "drawdown": 0.1,
                "volatility": 0.3,
                "exposure": 0.8
            },
            "critical": {
                "drawdown": 0.15,
                "volatility": 0.5,
                "exposure": 0.9
            }
        }
    }
}

# Add to risk_config.py
MEMORY_CONFIG = {
    "profit_archive": {
        "max_size": 1000,
        "decay_rate": 0.1,
        "min_profit": 0.01,
        "min_win_rate": 0.5
    },
    "basket_performance": {
        "window_size": 100,
        "update_interval": 1.0,
        "metrics": [
            "profit_loss",
            "win_rate",
            "drawdown",
            "sharpe_ratio"
        ]
    }
}
