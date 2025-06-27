# -*- coding: utf-8 -*-
"""
Windows CLI Compatibility Layer - Schwabot System

Provides safe printing and color handling for Windows CLI environments
that may not support UTF-8 or ANSI escape codes by default.

Features:
- safe_print: Prints text, gracefully handling UnicodeEncodeError
- Color functions (info, warn, error, success, debug): Wrap text with appropriate colors and prefixes
- Automatic enabling of ANSI escape code processing on Windows
"""

import logging
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# UTF-8 Force Override for Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # Fallback if reconfigure not available

# Enable ANSI escape codes on Windows
if sys.platform == "win32":
    try:
        from ctypes import windll
        kernel32 = windll.kernel32
        # Enables virtual terminal processing for the console
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass  # Fallback for environments where ctypes fails


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


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
    
    def safe_print(self, message: str, **kwargs):
        """
        Safely prints a message to the console, handling potential
        UnicodeEncodeErrors by replacing problematic characters.
        
        Args:
            message (str): The message to print.
            **kwargs: Additional arguments for the built-in print function.
        """
        try:
            print(message, **kwargs)
        except UnicodeEncodeError:
            # Fallback for environments that cannot handle the character set
            cleaned_message = message.encode(
                sys.stdout.encoding, 
                errors='replace'
            ).decode(sys.stdout.encoding)
            print(cleaned_message, **kwargs)
        except Exception as e:
            # Catch other potential printing errors
            print(f"[safe_print error] Could not print message. Error: {e}")
    
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
                message = message.encode('ascii', errors='ignore').decode('ascii')
            
            if hasattr(logger_instance, level.lower()):
                getattr(logger_instance, level.lower())(message)
            else:
                print(f"[{level.upper()}] {message}")
        except Exception as e:
            print(f"[LOG ERROR] Failed to log message: {e}")
    
    def handle_output(self, text: str) -> str:
        """Handle text output with Windows compatibility."""
        try:
            if self.is_windows:
                return text.encode('ascii', errors='ignore').decode('ascii')
            return text
        except Exception as e:
            logger.warning(f"Failed to handle output: {e}")
            return str(text)


# Global handler instance
cli_handler = WindowsCliCompatibilityHandler()


def safe_print(message: str, **kwargs):
    """Global safe print function."""
    cli_handler.safe_print(message, **kwargs)


def safe_format_error(error: Exception, context: str = "") -> str:
    """Global safe error formatting function."""
    return cli_handler.safe_format_error(error, context)


def log_safe(message: str) -> None:
    """Global safe logging function."""
    try:
        logger.info(message)
    except Exception as e:
        print(f"[LOG] {message}")


# Color helper functions
def info(text: str) -> str:
    """Format text as info message."""
    return f"{Colors.BLUE}[INFO]{Colors.RESET} {text}"


def warn(text: str) -> str:
    """Format text as warning message."""
    return f"{Colors.YELLOW}[WARN]{Colors.RESET} {text}"


def error(text: str) -> str:
    """Format text as error message."""
    return f"{Colors.RED}[ERROR]{Colors.RESET} {text}"


def success(text: str) -> str:
    """Format text as success message."""
    return f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {text}"


def debug(text: str) -> str:
    """Format text as debug message."""
    return f"{Colors.MAGENTA}[DEBUG]{Colors.RESET} {text}"
