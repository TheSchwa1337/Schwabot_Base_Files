"""TODO: document module."""
from datetime import datetime
import os
import platform
from typing import Any, Dict, List

from pydantic import BaseModel
from pydantic import Field

from models.enums import FillType
from models.enums import Side

# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================


class WindowsCliCompatibilityHandler:
    """Windows CLI compatibility for emoji and Unicode handling."""

    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment."""
        return platform.system() == "Windows" and (
            "cmd" in os.environ.get("COMSPEC", "").lower()
            or "powershell" in os.environ.get("PSModulePath", "").lower()
        )

    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """Print message safely with Windows CLI compatibility."""
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            emoji_mapping = {
                "🚨": "[ALERT]",
                "⚠️": "[WARNING]",
                "✅": "[SUCCESS]",
                "❌": "[ERROR]",
                "🔄": "[PROCESSING]",
                "🎯": "[TARGET]",
            }
            for emoji, marker in emoji_mapping.items():
                message = message.replace(emoji, marker)
        return message

    @staticmethod
    def log_safe(logger: Any, level: str, message: str) -> None:
        """Log message safely with Windows CLI compatibility."""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        try:
            getattr(logger, level.lower())(safe_message)
        except UnicodeEncodeError:
            ascii_message = safe_message.encode(
                "ascii", errors="replace"
            ).decode("ascii")
            getattr(logger, level.lower())(ascii_message)


class OrderBookUpdate(BaseModel):
    """TODO: document OrderBookUpdate."""

    ts: datetime
    price: float
    volume: float
    side: Side


class Fill(BaseModel):
    """TODO: document Fill."""

    ts: datetime
    ftype: FillType
    price: float
    quantity: float
    meta: Dict = Field(default_factory=dict)


class TickPacket(BaseModel):
    """TODO: document TickPacket."""

    tick_id: int
    mid_price: float
    wall_snap: Dict
    tension: float
    zeta: bool
    fills: List[Fill] = Field(default_factory=list)
