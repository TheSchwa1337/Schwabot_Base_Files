# -*- coding: utf-8 -*-
"""
APCF Visual Trade Feedback Core
===============================

This module generates the necessary data structures for visualizing the
performance and behavior of the APCF over time. It prepares data for
real-time plotting in a web dashboard, CLI, or OBS stream, providing
immediate feedback on the bot's decision-making process.
"""

import logging
from collections import deque
from typing import Any, Dict, List

# from .adaptive_profit_cycle_function import APCFResult

logger = logging.getLogger(__name__)


class APCFFeedbackPlotter:
    """
    Prepares data for plotting APCF values, trade results, and other
    contextual information.
    """

    def __init__(self, buffer_size: int = 64):
        """
        Initializes the feedback plotter.

        Args:
            buffer_size: The number of recent ticks to keep in memory for plotting.
        """
        self.buffer_size = buffer_size
        self.plot_data_buffer = deque(maxlen=buffer_size)
        logger.info(f"APCF Visual Feedback Plotter initialized with buffer size {buffer_size}.")

    def add_tick_data(self, apcf_result: Any, trade_outcome: Dict[str, Any] = None):
        """
        Adds data from a new tick to the plot buffer.

        Args:
            apcf_result: The APCFResult from the current tick.
            trade_outcome: An optional dictionary describing the result of any
                           trade action taken on this tick (e.g., {'roi': 0.02}).
        """

        # Determine plot color based on outcome
        if trade_outcome:
            roi = trade_outcome.get("roi", 0.0)
            plot_color = "green" if roi > 0 else ("red" if roi < 0 else "grey")
        else:
            plot_color = "blue"  # Default color for non-action ticks

        tick_data = {
            "timestamp": apcf_result.timestamp,
            "apcf_value": apcf_result.apcf_value,
            "apcf_state": apcf_result.state.value,
            "trade_result_roi": trade_outcome.get("roi") if trade_outcome else None,
            "plot_color": plot_color,
            "annotations": {
                "ferris_phase": apcf_result.components.get("ferris_phase"),
                "entropy": apcf_result.components.get("entropy"),
                "hash_similarity": apcf_result.components.get("hash_similarity"),
                "ai_votes": None,  # Placeholder for future AI agent integration
            },
        }
        self.plot_data_buffer.append(tick_data)

    def get_plot_data_structure(self) -> Dict[str, Any]:
        """
        Generates a dictionary formatted for a plotting library (e.g., Plotly, Matplotlib).

        Returns:
            A dictionary containing all data needed to render the APCF feedback chart.
        """

        timestamps = [d["timestamp"] for d in self.plot_data_buffer]
        apcf_values = [d["apcf_value"] for d in self.plot_data_buffer]
        colors = [d["plot_color"] for d in self.plot_data_buffer]
        annotations = [d["annotations"] for d in self.plot_data_buffer]

        plot_structure = {
            "title": "APCF Real-Time Feedback",
            "x_axis": {"label": "Time", "data": timestamps},
            "y_axis": {"label": "APCF Value", "data": apcf_values},
            "series": {"type": "scatter", "colors": colors, "annotations": annotations},
            "threshold_lines": {"execute": 1.0, "overload_2x": 2.0, "overload_3x": 3.0},
        }

        logger.debug("Generated new plot data structure.")
        return plot_structure


# Global instance
apcf_feedback_plotter = APCFFeedbackPlotter()
