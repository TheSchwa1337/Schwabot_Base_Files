#!/usr/bin/env python3
"""
Safe Print Utility - Schwabot UROS v1.0
======================================

Provides safe printing functions that handle Unicode and emoji characters
across different CLI environments, especially Windows.
"""

import sys
import os
from typing import Any, Optional


def safe_print(*args, force_ascii: bool = False, **kwargs) -> None:
    """
    Safely print text with Unicode/emoji support.

    Args:
        *args: Text to print
        force_ascii: Force ASCII output if True
        **kwargs: Additional print arguments
    """
    try:
        if force_ascii:
            # Convert to ASCII-safe string
            safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', errors='replace').decode('ascii'))
                else:
                    safe_args.append(arg)
            print(*safe_args, **kwargs)
        else:
            print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback to ASCII if Unicode fails
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg.encode('ascii', errors='replace').decode('ascii'))
            else:
                safe_args.append(arg)
        print(*safe_args, **kwargs)
    except Exception as e:
        # Ultimate fallback
        print(f"[SAFE_PRINT_ERROR] {e}: {args}")


def info(*args, **kwargs) -> None:
    """Print info message."""
    safe_print("[INFO]", *args, **kwargs)


def warn(*args, **kwargs) -> None:
    """Print warning message."""
    safe_print("[WARN]", *args, **kwargs)


def error(*args, **kwargs) -> None:
    """Print error message."""
    safe_print("[ERROR]", *args, **kwargs)


def success(*args, **kwargs) -> None:
    """Print success message."""
    safe_print("[SUCCESS]", *args, **kwargs)


def debug(*args, **kwargs) -> None:
    """Print debug message."""
    safe_print("[DEBUG]", *args, **kwargs)


def safe_math(*args, **kwargs) -> None:
    """Print mathematical operations safely."""
    safe_print("[MATH]", *args, **kwargs)

# Test function


def test_safe_print():
    """Test safe print functionality."""
    print("Testing safe print functions...")

    # Test basic printing
    safe_print("Hello, World!")
    info("This is an info message")
    warn("This is a warning message")
    error("This is an error message")
    success("This is a success message")
    debug("This is a debug message")
    safe_math("2 + 2 = 4")

    # Test Unicode/emoji
    safe_print("🚀 Rocket emoji test")
    safe_print("Unicode: café, naïve, résumé")

    print("Safe print test completed!")


if __name__ == "__main__":
    test_safe_print()
