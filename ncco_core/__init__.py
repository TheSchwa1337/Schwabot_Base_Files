from .ncco_generator import generate_nccos
from .ncco_scorer import score_nccos

from .control_panel import AdvancedControlPanel
from .fill_conjunction import FillConjunctionEngine
from .harmony_memory import HarmonyMemory
from .ncco import NCCO

# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================

class WindowsCliCompatibilityHandler:
    """Windows CLI compatibility for emoji and Unicode handling"""
    
    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment"""
        return (platform.system() == "Windows" and 
                ("cmd" in os.environ.get("COMSPEC", "").lower() or
                 "powershell" in os.environ.get("PSModulePath", "").lower()))
    
    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """Print message safely with Windows CLI compatibility"""
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            emoji_mapping = {
                '🚨': '[ALERT]', '⚠️': '[WARNING]', '✅': '[SUCCESS]',
                '❌': '[ERROR]', '🔄': '[PROCESSING]', '🎯': '[TARGET]'
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
            ascii_message = safe_message.encode('ascii', errors='replace').decode('ascii')
            getattr(logger, level.lower())(ascii_message)


__all__ = (
    ['NCCO', 'generate_nccos', 'score_nccos', 'FillConjunctionEngine', 'AdvancedControlPanel', 'HarmonyMemory']
)
