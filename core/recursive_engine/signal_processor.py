from __future__ import annotations
import logging
import time
from typing import Any

from init.event_bus import EventBus, EventState  # type: ignore
from .primary_loop import RecursiveEngine

logger = logging.getLogger(__name__)

# Name of the bus key where we publish the metrics
RECURSIVE_METRICS_KEY = "recursive_metrics"

class RecursiveSignalHandler:
    """Event-bus adapter that feeds market data into RecursiveEngine.

    Usage
    -----
    >>> handler = RecursiveSignalHandler(bus)
    During system bootstrap, instantiate this class once and it will
    automatically begin listening to "market_tick" events published on
    the `EventBus`.
    """

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._engine = RecursiveEngine()
        self._prev_ts = time.time()

        # Subscribe to market ticks
        self._bus.subscribe("market_tick", self._on_tick)
        logger.info("RecursiveSignalHandler subscribed to 'market_tick' events.")

    # ------------------------------------------------------------------
    # EventBus callback
    # ------------------------------------------------------------------

    def _on_tick(self, state: EventState):  # signature defined by EventBus
        """Handle incoming tick event and publish recursive metrics."""
        now = time.time()
        dt = max(now - self._prev_ts, 1e-6)  # avoid division by zero
        self._prev_ts = now

        metadata = state.metadata or {}

        try:
            F = float(metadata.get("fractal_output", 0.0))
            P = float(metadata.get("profit_ratio", 0.0))
            C_placeholder = metadata.get("coherence_score")  # may be None – let engine calc
            Lambda = float(metadata.get("paradox_phase", 0.0))
            phi = float(metadata.get("phase_shift", 0.0))
            R = float(metadata.get("recursive_trigger", 1.0))

            metrics = self._engine.process_tick(
                F=F,
                P=P,
                Lambda=Lambda,
                phi=phi,
                R=R,
                dt=dt,
            )

            # Publish back to bus
            self._bus.update(RECURSIVE_METRICS_KEY, metrics, source="recursive_engine")
        except Exception as exc:
            logger.exception("Error processing recursive tick: %s", exc) 