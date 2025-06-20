import platform
import os
from typing import Any

# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================


class WindowsCliCompatibilityHandler:
    """Windows CLI compatibility for emoji and Unicode handling"""

    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment"""
        return platform.system() == "Windows" and (
            "cmd" in os.environ.get("COMSPEC", "").lower()
            or "powershell" in os.environ.get("PSModulePath", "").lower()
        )

    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """Print message safely with Windows CLI compatibility"""
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
        """Log message safely with Windows CLI compatibility"""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        try:
            getattr(logger, level.lower())(safe_message)
        except UnicodeEncodeError:
            ascii_message = safe_message.encode(
                "ascii", errors="replace"
            ).decode("ascii")
            getattr(logger, level.lower())(ascii_message)


class SchwabotMetrics:

    def __init__(self: Any) -> None:
        self.zygot_metrics = {
            "drift_resonance": [],
            "alignment_score": [],
            "shell_states": [],
        }
        self.gan_metrics = {"anomaly_scores": [], "filter_confidence": []}
        self.fill_metrics = {"order_latency": [], "fill_rates": []}

    def record_zygot_metric(self: Any, metric_name: str, value: float) -> None:
        """Record ZygotShell metric"""
        if metric_name in self.zygot_metrics:
            self.zygot_metrics[metric_name].append(value)
