#!/usr/bin/env python3


# -*- coding: utf-8 -*-


"""



Safe Print Utilities



====================







Safe printing utilities for Schwabot system with enhanced error handling



and logging integration.



"""


from __future__ import annotations

import logging
import sys
import traceback
from typing import Optional

# Get logger for this module


logger = logging.getLogger(__name__)


def safe_print(*args, **kwargs) -> None:
    """



    Safe print function that handles unicode and encoding errors gracefully.







    Args:



        *args: Arguments to print



        **kwargs: Keyword arguments for print function



    """

    try:

        print(*args, **kwargs)

    except UnicodeEncodeError:

        # Handle unicode encoding errors

        safe_args = []

        for arg in args:

            try:

                safe_args.append(str(arg).encode("ascii", "replace").decode("ascii"))

            except Exception:

                safe_args.append(repr(arg))

        print(*safe_args, **kwargs)

    except Exception as e:

        # Fallback for any other errors

        try:

            print(f"[PRINT ERROR: {e}] {repr(args)}", file=sys.stderr)

        except Exception:

            # Ultimate fallback

            sys.stderr.write("[CRITICAL PRINT ERROR]\n")


def info(*args, **kwargs) -> None:
    """Print info message with INFO prefix."""

    try:

        message = " ".join(str(arg) for arg in args)

        safe_print(f"[INFO] {message}", **kwargs)

        logger.info(message)

    except Exception as e:

        safe_print(f"[INFO ERROR: {e}] {repr(args)}")


def warn(*args, **kwargs) -> None:
    """Print warning message with WARNING prefix."""

    try:

        message = " ".join(str(arg) for arg in args)

        safe_print(f"[WARNING] {message}", **kwargs)

        logger.warning(message)

    except Exception as e:

        safe_print(f"[WARNING ERROR: {e}] {repr(args)}")


def error(*args, **kwargs) -> None:
    """Print error message with ERROR prefix."""

    try:

        message = " ".join(str(arg) for arg in args)

        safe_print(f"[ERROR] {message}", file=sys.stderr, **kwargs)

        logger.error(message)

    except Exception as e:

        safe_print(f"[ERROR ERROR: {e}] {repr(args)}", file=sys.stderr)


def success(*args, **kwargs) -> None:
    """Print success message with SUCCESS prefix."""

    try:

        message = " ".join(str(arg) for arg in args)

        safe_print(f"[SUCCESS] {message}", **kwargs)

        logger.info(f"SUCCESS: {message}")

    except Exception as e:

        safe_print(f"[SUCCESS ERROR: {e}] {repr(args)}")


def debug(*args, **kwargs) -> None:
    """Print debug message with DEBUG prefix."""

    try:

        message = " ".join(str(arg) for arg in args)

        safe_print(f"[DEBUG] {message}", **kwargs)

        logger.debug(message)

    except Exception as e:

        safe_print(f"[DEBUG ERROR: {e}] {repr(args)}")


def critical(*args, **kwargs) -> None:
    """Print critical message with CRITICAL prefix."""

    try:

        message = " ".join(str(arg) for arg in args)

        safe_print(f"[CRITICAL] {message}", file=sys.stderr, **kwargs)

        logger.critical(message)

    except Exception as e:

        safe_print(f"[CRITICAL ERROR: {e}] {repr(args)}", file=sys.stderr)


def print_exception(exc: Exception, context: Optional[str] = None) -> None:
    """Safely print exception information."""

    try:
        context_str = f" in {context}" if context else ""

        safe_print(
            f"[EXCEPTION{context_str}] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        # Print traceback safely
        try:
            tb_str = traceback.format_exc()
            safe_print(tb_str, file=sys.stderr)
        except Exception:
            safe_print("[TRACEBACK ERROR]", file=sys.stderr)

        # Log the exception
        logger.exception(f"Exception{context_str}: {exc}")

    except Exception as print_exc:
        # Ultimate fallback
        sys.stderr.write(f"[CRITICAL: Cannot print exception {exc} due to {print_exc}]\n")


def print_separator(char: str = "=", length: int = 60) -> None:
    """Print a separator line."""

    try:
        safe_print(char * length)
    except Exception:
        safe_print("=" * 60)


def print_header(title: str, char: str = "=", length: int = 60) -> None:
    """Print a formatted header."""

    try:
        print_separator(char, length)
        title_line = f" {title} "
        padding = (length - len(title_line)) // 2
        centered_title = char * padding + title_line + char * padding
        
        if len(centered_title) < length:
            centered_title += char
        
        safe_print(centered_title)
        print_separator(char, length)

    except Exception as e:
        safe_print(f"[HEADER ERROR: {e}] {title}")


def print_dict(data: dict, indent: int = 2, max_depth: int = 3, current_depth: int = 0) -> None:
    """Safely print dictionary with indentation."""

    if current_depth >= max_depth:
        safe_print("  " * indent + "...")
        return

    try:
        for key, value in data.items():
            key_str = str(key)
            
            if isinstance(value, dict):
                safe_print("  " * indent + f"{key_str}:")
                print_dict(value, indent + 1, max_depth, current_depth + 1)
            elif isinstance(value, (list, tuple)):
                safe_print("  " * indent + f"{key_str}: [{len(value)} items]")
                if len(value) > 0 and current_depth < max_depth - 1:
                    for i, item in enumerate(value[:3]):  # Show first 3 items
                        safe_print("  " * (indent + 1) + f"[{i}]: {repr(item)}")
                    if len(value) > 3:
                        safe_print("  " * (indent + 1) + "...")
            else:
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                safe_print("  " * indent + f"{key_str}: {value_str}")

    except Exception as e:
        safe_print(f"[DICT PRINT ERROR: {e}] {repr(data)}")


def print_list(data: list, indent: int = 2, max_items: int = 10) -> None:
    """Safely print list with indentation."""

    try:
        total_items = len(data)
        safe_print("  " * indent + f"List with {total_items} items:")
        
        for i, item in enumerate(data[:max_items]):
            item_str = str(item)
            if len(item_str) > 80:
                item_str = item_str[:77] + "..."
            safe_print("  " * (indent + 1) + f"[{i}]: {item_str}")
        
        if total_items > max_items:
            safe_print("  " * (indent + 1) + f"... and {total_items - max_items} more items")

    except Exception as e:
        safe_print(f"[LIST PRINT ERROR: {e}] {repr(data)}")


def print_status(component: str, status: bool, details: Optional[str] = None) -> None:
    """Print component status with colored indicators."""

    try:
        status_icon = "✓" if status else "✗"
        status_text = "ACTIVE" if status else "INACTIVE"
        message = f"{status_icon} {component}: {status_text}"
        
        if details:
            message += f" - {details}"
        
        safe_print(message)

    except Exception as e:
        safe_print(
            f"[STATUS PRINT ERROR: {e}] {component}: {'OK' if status else 'FAIL'}"
        )


def print_progress(current: int, total: int, description: str = "", bar_length: int = 40) -> None:
    """Print a progress bar."""

    try:
        if total <= 0:
            safe_print(f"{description}: [INVALID TOTAL]")
            return

        percent = min(100.0, (current / total) * 100)
        filled_length = int(bar_length * current // total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        progress_line = f"{description}: |{bar}| {percent:.1f}% ({current}/{total})"
        
        safe_print(f"\r{progress_line}", end="", flush=True)
        
        if current >= total:
            safe_print()  # New line when complete

    except Exception as e:
        safe_print(f"[PROGRESS ERROR: {e}] {current}/{total}")


# Convenience aliases


log_info = info


log_warn = warn


log_error = error


log_debug = debug


log_critical = critical
