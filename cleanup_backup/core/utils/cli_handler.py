from utils.safe_print import safe_print, info, warn, error, success, debug
#!/usr/bin/env python3
"""CLI compatibility handler for Windows systems.

This module provides safe printing and logging functions that work
across different Windows CLI environments.
"""

import logging

logger = logging.getLogger(__name__)


class CLIHandler:
    """CLI compatibility handler for Windows systems."""

    @staticmethod
    def safe_emoji_print(message: str, force_ascii: bool = False) -> str:
        """Convert emojis to ASCII-safe representations.

        Args:
            message: Message containing potential emojis.
            force_ascii: Whether to force ASCII conversion.

        Returns:
            Message with emojis converted to ASCII representations.
        """
        emoji_mapping = {
            "✅": "[SUCCESS]",
            "❌": "[ERROR]",
            "⚠️": "[WARNING]",
            "🚨": "[ALERT]",
            "🎉": "[COMPLETE]",
            "🔄": "[PROCESSING]",
            "⏳": "[WAITING]",
            "⭐": "[STAR]",
            "🚀": "[LAUNCH]",
            "🔧": "[TOOLS]",
            "🛠️": "[REPAIR]",
            "⚡": "[FAST]",
            "🔍": "[SEARCH]",
            "🎯": "[TARGET]",
            "🔥": "[HOT]",
            "❄️": "[COOL]",
            "📊": "[DATA]",
            "📈": "[PROFIT]",
            "📉": "[LOSS]",
            "💰": "[MONEY]",
            "🧪": "[TEST]",
            "⚖️": "[BALANCE]",
            "🔬": "[ANALYZE]",
            "📱": "[MOBILE]",
            "🌐": "[NETWORK]",
            "🔒": "[SECURE]",
            "🔓": "[UNLOCK]",
            "🔑": "[KEY]",
            "🛡️": "[SHIELD]",
            "🧮": "[CALC]",
            "📐": "[MATH]",
            "🔢": "[NUMBERS]",
            "∞": "[INFINITY]",
            "φ": "[PHI]",
            "π": "[PI]",
            "∑": "[SUM]",
            "∫": "[INTEGRAL]",
        }

        if force_ascii:
            for emoji, replacement in emoji_mapping.items():
                message = message.replace(emoji, replacement)

        return message

    @staticmethod
    def safe_print(message: str, force_ascii: bool = False) -> None:
        """Safe print function with CLI compatibility.

        Args:
            message: Message to print.
            force_ascii: Whether to force ASCII conversion.
        """
        safe_message = CLIHandler.safe_emoji_print(message, force_ascii)
        print(safe_message)


def safe_log(
    logger_instance: logging.Logger,
    level: str,
    message: str,
    context: str = "",
) -> bool:
    """Safe logging function with CLI compatibility.

    Args:
        logger_instance: Logger instance to use.
        level: Log level (debug, info, warning, error).
        message: Log message.
        context: Additional context information.

    Returns:
        True if logging was successful, False otherwise.
    """
    try:
        safe_message = CLIHandler.safe_emoji_print(message, force_ascii=True)

        if context:
            safe_message = f"[{context}] {safe_message}"

        if level.lower() == "debug":
            logger_instance.debug(safe_message)
        elif level.lower() == "info":
            logger_instance.info(safe_message)
        elif level.lower() == "warning":
            logger_instance.warning(safe_message)
        elif level.lower() == "error":
            logger_instance.error(safe_message)
        else:
            logger_instance.info(safe_message)

        return True
    except Exception:
        # Fallback to basic print if logging fails
        safe_print(f"[{level.upper()}] {message}")
        return False
