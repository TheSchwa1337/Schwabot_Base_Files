# -*- coding: utf-8 -*-
# #!/usr/bin/env python3
"""Lantern hash echo – H_L = hash(Ξₜ) · e^−τ."""

from __future__ import annotations

import hashlib
from core.unified_math_system import unified_math
from typing import Final
import math
# from core.unified_math_system import unified_math  # F811: duplicate import

__all__: list[str] = ["lantern_hash_echo"]

_MOD: Final = 1 / (2**32 - 1)  # scale 32-bit int to (0,1)


def lantern_hash_echo(xi_t: str, tau: float) -> float:  # noqa: D401
    """Return decayed numeric echo of glyph hash."""
    digest = hashlib.sha256(xi_t.encode()).digest()[:4]
    val = int.from_bytes(digest, "big") * _MOD
    return val * unified_math.exp(-tau)
