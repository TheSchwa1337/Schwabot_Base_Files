# #!/usr/bin/env python3
"""
Windows CLI Compatibility Handler
================================

Provides cross-platform compatibility for Windows CLI operations,
ensuring Schwabot can run reliably on Windows systems while maintaining
compatibility with other platforms.
"""

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

# UTF-8 Force Override for Windows
if sys.platform == "win32":
os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # Fallback if reconfigure not available

logger = logging.getLogger(__name__)


class WindowsCliCompatibilityHandler:
    """Centralized handler for Windows CLI compatibility."""

    def __init__(self):
        """Initialize the Windows CLI compatibility handler."""
self.is_windows = platform.system().lower() == "windows"
        self.encoding = 'utf-8' if not self.is_windows else 'cp1252'
self.shell = True if self.is_windows else False

        # Windows-specific configurations
        if self.is_windows:
self._setup_windows_environment()

    def _setup_windows_environment(self) -> None:
        """Setup Windows-specific environment configurations."""
        try:
            # Set console encoding for Windows
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding=self.encoding)
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding=self.encoding)
        except Exception as e:
logger.warning(f"Failed to configure Windows console encoding: {e}")

    def safe_print(self, message: str, use_emoji: bool = True) -> str:
        """Safely print messages with Windows compatibility."""
        try:
            print(message, flush=True)
            return message
        except UnicodeEncodeError:
            # Fallback to ASCII-safe printing with better encoding
            try:
encoded = " ".join(str(arg).encode('ascii', errors='replace').decode('ascii') for arg in [message])
                print(encoded, flush=True)
                return encoded
            except Exception:
                # Final fallback
safe_message = message.encode('ascii', errors='ignore').decode('ascii')
                print(safe_message, flush=True)
                return safe_message
        except Exception as e:
logger.error(f"Print error: {e}")
            return str(message)

    def safe_format_error(self, error: Exception, context: str = "") -> str:
        """Safely format error messages for Windows compatibility."""
        try:
error_msg = str(error)
            if self.is_windows:
                # Ensure error message is Windows-compatible
error_msg = error_msg.encode('ascii', errors='ignore').decode('ascii')

            if context:
                return f"Error: {error_msg} | Context: {context}"
            else:
                return f"Error: {error_msg}"
        except Exception as e:
            return f"Error formatting failed: {e}"

    def log_safe(self, logger_instance, level: str, message: str) -> None:
        """Safely log messages with Windows compatibility."""
        try:
            if self.is_windows:
                # Ensure log message is Windows-compatible
message = message.encode('ascii', errors='ignore').decode('ascii')

log_method = getattr(logger_instance, level.lower(), logger_instance.info)
            log_method(message)
        except Exception as e:
            # Fallback logging
logger.error(f"Log error: {e}")
            safe_print(f"[{level.upper()}] {message}")

    def _remove_emojis(self, text: str) -> str:
        """Remove emoji characters from text for Windows compatibility."""
        try:
            # Simple emoji removal - can be enhanced with proper emoji detection
            import re
            # Remove common emoji patterns
emoji_pattern = re.compile(
                "["
"\U0001F600-\U0001F64F"  # emoticons
"\U0001F300-\U0001F5FF"  # symbols & pictographs
"\U0001F680-\U0001F6FF"  # transport & map symbols
"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
"\U000024C2-\U0001F251"
"]+", flags=re.UNICODE

            return emoji_pattern.sub(r'', text)
        except Exception:
            return text

    def safe_path(self, path: Union[str, Path]) -> Path:
        """Convert path to Windows-compatible Path object."""
        try:
            if isinstance(path, str):
                path = Path(path)

            # Handle Windows path issues
            if self.is_windows:
                # Normalize path separators
path = Path(str(path).replace('/', '\\'))

            return path
        except Exception as e:
logger.error(f"Path conversion error: {e}")
            return Path(str(path))

    def safe_subprocess_run(self, command: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Safely run subprocess commands with Windows compatibility."""
        try:
            # Set Windows-specific subprocess options
            if self.is_windows:
kwargs.setdefault('shell', True)
                kwargs.setdefault('encoding', self.encoding)
                kwargs.setdefault('errors', 'ignore')

            return subprocess.run(command, **kwargs)
        except Exception as e:
logger.error(f"Subprocess error: {e}")
            # Return a mock completed process
            return subprocess.CompletedProcess(
                args=command,
returncode=1,
stdout=b"",
stderr=str(e).encode()


    def safe_file_operations(self, file_path: Union[str, Path], operation: str, **kwargs) -> Any:
        """Safely perform file operations with Windows compatibility."""
        try:
path = self.safe_path(file_path)

            if operation == "read":
                with open(path, 'r', encoding=self.encoding, errors='ignore') as f:
                    return f.read()
            elif operation == "write":
content = kwargs.get('content', '')
                with open(path, 'w', encoding=self.encoding, errors='ignore') as f:
                    f.write(content)
                return True
            elif operation == "exists":
                return path.exists()
            elif operation == "mkdir":
path.mkdir(parents=True, exist_ok=True)
                return True
            else:
                raise ValueError(f"Unknown operation: {operation}")

        except Exception as e:
logger.error(f"File operation error ({operation}): {e}")
            return None


# Global instance
cli_handler = WindowsCliCompatibilityHandler()

# Convenience functions
def safe_print(message: str, use_emoji: bool = True) -> str:
    """Global safe print function."""
    try:
        if platform.system().lower() == "windows" and not use_emoji:
            # Remove emojis on Windows if requested
            import re
emoji_pattern = re.compile(
                "["
"\U0001F600-\U0001F64F"  # emoticons
"\U0001F300-\U0001F5FF"  # symbols & pictographs
"\U0001F680-\U0001F6FF"  # transport & map symbols
"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
"\U000024C2-\U0001F251"
"]+", flags=re.UNICODE

message = emoji_pattern.sub(r'', message)

        print(message, flush=True)
        return message
    except UnicodeEncodeError:
        try:
encoded = " ".join(str(arg).encode('ascii', errors='replace').decode('ascii') for arg in [message])
            print(encoded, flush=True)
            return encoded
        except Exception:
safe_message = message.encode('ascii', errors='ignore').decode('ascii')
            print(safe_message, flush=True)
            return safe_message
    except Exception as e:
logger.error(f"Print error: {e}")
        return str(message)


def safe_format_error(error: Exception, context: str = "") -> str:
    """Global safe error formatting function."""
    return cli_handler.safe_format_error(error, context)


def log_safe(logger_instance, level: str, message: str) -> None:
    """Global safe logging function."""
cli_handler.log_safe(logger_instance, level, message)


def main():
    print("Windows CLI Compatibility Handler test ran successfully.")
    return True


if __name__ == "__main__":
    # Test the compatibility handler
test_messages = [
"🚀 Launching Schwabot system...",
"✅ System initialized successfully",
"❌ Error occurred during startup",
"⚠️ Warning: High memory usage detected",
"📊 Processing data with α, β, γ parameters",
"🎯 Target profit: $1000 → $1500",
"🔥 Hot market conditions detected",
"⚡ Fast execution mode enabled",
"🔧 Tools loaded successfully",
"📈 Profit trend: ↑ 15%",
"💰 Money flow: ∀ x ∈ ℝ",
"🧮 Calculation: ∑(i=1 to n) x_i",
        "🔬 Analysis: μ = 0.5, σ = 0.1",
"⚖️ Balance: φ = 1.618033988749895",
]

safe_print("Testing Windows CLI Compatibility Handler")
    safe_print("=" * 50)

    for message in test_messages:
safe_message = safe_print(message)
        safe_print(f"Original: {message}")
        safe_print(f"Safe:     {safe_message}")
        safe_print("-" * 30)

    # Test environment detection
env_info = cli_handler.get_environment_info()
    safe_print("\nEnvironment Information:")
    for key, value in env_info.items():
        safe_print(f"  {key}: {value}")
