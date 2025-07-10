# -*- coding: utf-8 -*-
import json
import os
import platform
from typing import Optional

from dual_unicore_handler import DualUnicoreHandler

# Initialize Unicode handler
unicore = DualUnicoreHandler()


class VisualFallback:
    """Handles visual-safe symbols for status output, with user override."""


def __init__(self, use_emoji: Optional[bool] = None):
        # User config override"""
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

self.symbols = {}
            "PASS": "\\u2705" if self.use_emoji else "\\u2714\\ufe0f","
            "FAIL": "\\u274c" if self.use_emoji else "\\u2716\\ufe0f","
            "SKIP": "\\u26a0\\ufe0f" if self.use_emoji else "\\u203c\\ufe0f","
            "READY": "\\u1f7e2" if self.use_emoji else "\\u25cf","
            "PARTIAL": "\\u1f7e1" if self.use_emoji else "\\u25d0",
            "NOT_READY": "\\u1f534" if self.use_emoji else "\\u25a0",
            "ERROR": "\\u1f4a5" if self.use_emoji else "!!",
            "INFO": "\\u2139\\ufe0f" if self.use_emoji else "i",
            "SAVE": "\\u1f4be" if self.use_emoji else "[S]",

def get(self, key: str) -> str:
        """Return the symbol for a given status key.""""""
    return self.symbols.get(key.upper(), "?")
