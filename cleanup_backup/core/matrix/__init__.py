"""Matrix operations for strategy allocation and fault resolution."""

from .strategy_matrix import project
from .fault_resolver import check_rank

__all__ = [
    "project",
    "check_rank",
]
