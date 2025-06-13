"""
Mode Manager
===========

Handles switching between live trading and backtesting modes,
with appropriate data source routing and safety checks.
"""

from enum import Enum
from typing import Optional
import yaml
from pathlib import Path

class TradingMode(Enum):
    LIVE = "live"
    BACKTEST = "backtest"

class ModeManager:
    def __init__(self, config_file: str = "recursive.yaml"):
        self.config_path = Path(__file__).resolve().parent / "config" / config_file
        self._mode: Optional[TradingMode] = None
        self._load_mode()
        
    def _load_mode(self):
        """Load trading mode from config"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                mode_str = config.get('mode', 'backtest')
                self._mode = TradingMode(mode_str)
        except Exception as e:
            print(f"Warning: Could not load mode config: {e}")
            self._mode = TradingMode.BACKTEST
            
    def set_mode(self, mode: TradingMode):
        """Set trading mode"""
        self._mode = mode
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            config['mode'] = mode.value
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f)
        except Exception as e:
            print(f"Warning: Could not save mode config: {e}")
            
    def is_live_mode(self) -> bool:
        """Check if in live trading mode"""
        return self._mode == TradingMode.LIVE
        
    def is_backtest_mode(self) -> bool:
        """Check if in backtest mode"""
        return self._mode == TradingMode.BACKTEST
        
    def get_mode(self) -> TradingMode:
        """Get current trading mode"""
        return self._mode 