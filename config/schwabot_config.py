
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


# Constants (Magic Number Replacements)
DEFAULT_PROFIT_MARGIN = 0.1
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_INDENT_SIZE = 4


class SchwabotConfig:

    def __init__(self: Any) -> None:
        self.zygot_config = {
            'drift_threshold': 0.5,
            'alignment_threshold': 0.7,
            'shell_radius': 144.44
        }
        self.gan_config = {
            'input_dim': 32,
            'latent_dim': 16,
            'learning_rate': 0.001
        }
        self.hook_config = {
            'ack_timeout': 1.0,
            'max_retries': 3,
            'backoff': 0.1
        }
