#!/usr/bin/env python3
"""Ghost pipeline – stealth-mode validator & orchestrator.

This module glues together the *stealth* helpers that were recently added to
Schwabot's core:

* :func:`flux_compensator.sync_flux_compensator` – entropy drift guard.
* :func:`thermal_shift.thermal_delta_switch` – temperature drift guard.
* :func:`phase_hash_gate` – deterministic hash-phase gating.

The goal is to expose **one** public convenience wrapper –
:func:`ghost_validator_pipeline` – so that legacy callers can perform
"all-in-one" validation without having to stitch the pieces manually.

References (math primer)
------------------------
See the design doc in chat (Γ_ghost, Φ_ghost, ζ etc.).  The current Python
implementation intentionally keeps the maths simple – mostly logical gating –
so that we avoid heavy computational cost inside tight trading loops.  The API
surface is stable and future-proof: each component can be swapped for a more
sophisticated version without changing the public signature.
"""

from __future__ import annotations

from typing import Final, Tuple

from .flux_compensator import sync_flux_compensator
from .thermal_shift import thermal_delta_switch
from .hash_phase_switch import phase_hash_gate

__all__: list[str] = ["GhostPipeline", "ghost_validator_pipeline"]

# -----------------------------------------------------------------------------
# Internal constants
# -----------------------------------------------------------------------------

_ENTROPY_THRESHOLD: Final = 5.0  # same default as FluxCompensator.threshold
_TEMP_THRESHOLD: Final = 2.5  # °C – mirrors ThermalShift default
_BASE_CYCLE: Final = 42  # phase_hash_gate default


class GhostPipeline:
    """Runtime container that evaluates ghost-mode pre-conditions."""

    entropy_threshold: float
    temp_threshold: float
    base_cycle: int

    def __init__(
        self,
        *,
        entropy_threshold: float = _ENTROPY_THRESHOLD,
        temp_threshold: float = _TEMP_THRESHOLD,
        base_cycle: int = _BASE_CYCLE,
    ) -> None:  # noqa: D401
        self.entropy_threshold = entropy_threshold
        self.temp_threshold = temp_threshold
        self.base_cycle = base_cycle

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(
        self,
        entropy: float,
        temp_current: float,
        temp_previous: float,
        tick: int,
        *,
        salt: str = "",
    ) -> Tuple[bool, dict[str, bool]]:
        """Return overall validity flag and individual component map."""
        entropy_ok = sync_flux_compensator(entropy, self.entropy_threshold)
        temp_ok = thermal_delta_switch(
            temp_current, temp_previous, threshold=self.temp_threshold
        )
        phase_ok = phase_hash_gate(
            tick, base_cycle=self.base_cycle, salt=salt
        )

        all_ok = entropy_ok and temp_ok and phase_ok
        component_map = {
            "entropy_ok": entropy_ok,
            "temp_ok": temp_ok,
            "phase_ok": phase_ok,
        }
        return all_ok, component_map


# -----------------------------------------------------------------------------
# Legacy functional wrapper – mirrors historical stub signature
# -----------------------------------------------------------------------------


def ghost_validator_pipeline(
    entropy: float,
    temp_current: float,
    temp_previous: float,
    tick: int,
    *,
    entropy_threshold: float = _ENTROPY_THRESHOLD,
    temp_threshold: float = _TEMP_THRESHOLD,
    base_cycle: int = _BASE_CYCLE,
    salt: str = "",
) -> bool:
    """One-shot validation wrapper around :class:`GhostPipeline`.

    Returns ``True`` only if *all* component validators pass.
    """
    pipeline = GhostPipeline(
        entropy_threshold=entropy_threshold,
        temp_threshold=temp_threshold,
        base_cycle=base_cycle,
    )
    result, _ = pipeline.validate(
        entropy, temp_current, temp_previous, tick, salt=salt
    )
    return result 