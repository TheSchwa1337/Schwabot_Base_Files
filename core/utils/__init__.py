# -*- coding: utf-8 -*-
"""
Core Utils Package

This package contains utility modules for Schwabot core functionality.
"""

try:
    from .windows_cli_compatibility import (
        WindowsCliCompatibilityHandler,
        safe_print,
        safe_format_error,
        log_safe,
        cli_handler,
    )
except ImportError:
    # Fallback stubs if windows_cli_compatibility is not available
    class WindowsCliCompatibilityHandler:
        def __init__(self):
            pass
        
        def handle_output(self, text):
            return str(text)
    
    def safe_print(text):
        print(str(text))
    
    def safe_format_error(error):
        return str(error)
    
    def log_safe(message):
        print(f"LOG: {message}")
    
    cli_handler = WindowsCliCompatibilityHandler()

__all__ = [
    "WindowsCliCompatibilityHandler",
    "safe_print",
    "safe_format_error", 
    "log_safe",
    "cli_handler",
]
