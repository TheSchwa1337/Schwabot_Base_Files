
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


class HarmonyMemory:

    def __init__(self: Any) -> None:
        self.patterns = {}

    def add_pattern(self: Any, pattern_id: Dict[str, Any], pattern_data: Dict[str, Any]) -> None:
        self.patterns[pattern_id] = pattern_data

    def get_pattern(self: Any, pattern_id: Dict[str, Any]) -> None:
        return self.patterns.get(pattern_id)
