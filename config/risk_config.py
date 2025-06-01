"""
Risk Management System Configuration
Contains default settings and configuration parameters for the risk management system.
"""

from typing import Dict, Any

# Default risk parameters
DEFAULT_RISK_CONFIG = {
    # Portfolio-level risk limits
    'max_portfolio_risk': 0.02,      # 2% maximum portfolio risk
    'max_position_size': 0.1,        # 10% maximum position size
    'max_drawdown': 0.15,            # 15% maximum drawdown
    'var_confidence': 0.95,          # 95% Value at Risk confidence level
    
    # Position sizing parameters
    'kelly_fraction_cap': 0.5,       # Maximum Kelly fraction (50%)
    'min_position_size': 0.01,       # Minimum position size (1%)
    'volatility_adjustment': 1.0,    # Volatility adjustment factor
    
    # Stop-loss parameters
    'atr_multiplier': 2.0,           # ATR multiplier for stop-loss
    'profit_target_ratio': 1.5,      # Risk:reward ratio for take-profit
    'trailing_stop_activation': 0.02, # Activate trailing stop at 2% profit
    
    # Risk level thresholds
    'risk_levels': {
        'extreme': {
            'exposure': 0.9,
            'volatility': 0.3,
            'var': 0.1
        },
        'high': {
            'exposure': 0.7,
            'volatility': 0.2,
            'var': 0.05
        },
        'medium': {
            'exposure': 0.5,
            'volatility': 0.1,
            'var': 0.02
        }
    },
    
    # Monitoring parameters
    'update_interval': 1.0,          # Update interval in seconds
    'history_size': 1000,            # Number of historical risk snapshots to keep
    'alert_thresholds': {
        'drawdown': 0.1,             # Alert at 10% drawdown
        'volatility': 0.2,           # Alert at 20% volatility
        'exposure': 0.8              # Alert at 80% exposure
    }
}

# Asset-specific risk parameters
ASSET_RISK_CONFIG = {
    'BTC/USD': {
        'max_position_size': 0.15,   # Higher limit for BTC
        'volatility_adjustment': 1.2, # Higher volatility adjustment
        'min_position_size': 0.02    # Higher minimum position
    },
    'ETH/USD': {
        'max_position_size': 0.1,    # Standard limit for ETH
        'volatility_adjustment': 1.0, # Standard volatility adjustment
        'min_position_size': 0.01    # Standard minimum position
    }
}

# Correlation thresholds for portfolio diversification
CORRELATION_CONFIG = {
    'high_correlation_threshold': 0.7,  # Assets with correlation > 0.7 are considered highly correlated
    'max_correlated_exposure': 0.3,     # Maximum exposure to highly correlated assets
    'min_correlation_threshold': -0.3    # Assets with correlation < -0.3 are considered negatively correlated
}

# Risk monitoring and reporting
MONITORING_CONFIG = {
    'report_interval': 60,           # Generate risk report every 60 seconds
    'alert_channels': ['log', 'email'],  # Alert channels
    'metrics_to_track': [
        'total_exposure',
        'portfolio_volatility',
        'current_drawdown',
        'var_95',
        'correlation_matrix',
        'position_risks'
    ]
}

def get_risk_config(asset: str = None) -> Dict[str, Any]:
    """
    Get risk configuration for a specific asset or default configuration.
    
    Args:
        asset: Asset symbol (e.g., 'BTC/USD')
        
    Returns:
        Dict containing risk configuration parameters
    """
    config = DEFAULT_RISK_CONFIG.copy()
    
    if asset and asset in ASSET_RISK_CONFIG:
        # Override default config with asset-specific settings
        config.update(ASSET_RISK_CONFIG[asset])
    
    return config 