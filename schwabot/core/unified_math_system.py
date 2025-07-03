#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Math System (compatibility wrapper)

This thin wrapper keeps the original import path `core.unified_math_system` alive while
handing off all real work to the modern, thoroughly-tested implementation in
`core.clean_unified_math`.
"""

from core.clean_unified_math import *  # noqa: F401,F403

# Re-export public symbols so `from core.unified_math_system import X` still works
__all__ = [name for name in globals() if not name.startswith("_")]
