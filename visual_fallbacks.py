import platform
import os
import json
from typing import Optional


class VisualFallback:
    """Handles visual-safe symbols for status output, with user override."""

    def __init__(self, use_emoji: Optional[bool] = None):
        # User config override
        config_path = os.path.expanduser("~/.schwabotrc.json")
        user_cfg = None
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_cfg = json.load(f)
            except Exception:
                user_cfg = None

        # OS detection
        system = platform.system().lower()
        self.default_to_unicode = system == "windows"
        # User config takes precedence
        if user_cfg and "visual_mode" in user_cfg:
            self.use_emoji = user_cfg["visual_mode"].lower() == "emoji"
        elif use_emoji is not None:
            self.use_emoji = use_emoji
        else:
            self.use_emoji = not self.default_to_unicode

        self.symbols = {
            "PASS": "✅" if self.use_emoji else "✔️",
            "FAIL": "❌" if self.use_emoji else "✖️",
            "SKIP": "⚠️" if self.use_emoji else "‼️",
            "READY": "🟢" if self.use_emoji else "●",
            "PARTIAL": "🟡" if self.use_emoji else "◐",
            "NOT_READY": "🔴" if self.use_emoji else "■",
            "ERROR": "💥" if self.use_emoji else "!!",
            "INFO": "ℹ️" if self.use_emoji else "i",
            "SAVE": "💾" if self.use_emoji else "[S]",
        }

    def get(self, key: str) -> str:
        """Return the symbol for a given status key."""
        return self.symbols.get(key.upper(), "?")
