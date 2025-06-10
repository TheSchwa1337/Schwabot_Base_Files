from dataclasses import dataclass, field
from typing import Optional, Dict
import yaml

@dataclass
class StrategyConfig:
    strategy_id: str
    meta_tag: Optional[str] = None
    fallback_matrix: Optional[str] = None
    active: bool = True
    scoring: Dict[str, float] = field(default_factory=lambda: {
        'hash_weight': 0.3,
        'volume_weight': 0.2,
        'drift_weight': 0.4,
        'error_weight': 0.1
    })

def load_strategies_from_yaml(yaml_path: str) -> Dict[str, StrategyConfig]:
    with open(yaml_path, 'r') as f:
        raw_data = yaml.safe_load(f)

    configs = {}
    for strategy_id, attrs in raw_data.items():
        config = StrategyConfig(
            strategy_id=strategy_id,
            meta_tag=attrs.get('meta_tag'),
            fallback_matrix=attrs.get('fallback_matrix'),
            active=attrs.get('active', True),
            scoring=attrs.get('scoring', {})
        )
        configs[strategy_id] = config

    return configs 