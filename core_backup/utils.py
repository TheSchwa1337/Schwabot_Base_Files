""""""
core/utils.py
Shared utility functions for backend (formatting, math, etc.).
""""""

from datetime import datetime
from typing import Any


def format_number(num: float, decimals: int = 2) -> str:
    """Format a number with a given number of decimals."""
    return f"{num:,.{decimals}f}"


def format_date(dt: Any) -> str:
    """Format a date or datetime as a string."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
