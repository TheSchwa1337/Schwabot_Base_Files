from pathlib import Path
from strategy_config import load_strategies_from_yaml, StrategyConfig
from typing import Dict

def load_active_strategies(yaml_path: str = "config/strategies.yaml") -> Dict[str, StrategyConfig]:
    strategy_dict = load_strategies_from_yaml(yaml_path)
    active_only = {sid: cfg for sid, cfg in strategy_dict.items() if cfg.active}
    return active_only 