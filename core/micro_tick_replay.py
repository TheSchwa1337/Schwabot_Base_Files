# -*- coding: utf-8 -*-
"""
Micro-Tick Replay Buffer
========================

This module implements a "regret engine." When an APCF-triggered action
results in a loss, this system logs the bit-state that led to the
decision. It can then replay this logic during future, similar market
conditions to test alternative outcomes, feeding this "regret" data back
into the meta-predictor.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MicroTickReplayBuffer:
    """
    Logs failed APCF cycles and provides a mechanism to replay them
    under new conditions.
    """

    def __init__(self, log_file_path: str = "core/logs/micro_tick_replay.log"):
        """
        Initializes the replay buffer.

        Args:
            log_file_path: The file to store logs of failed cycles.
        """
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(exist_ok=True)
        logger.info(f"Micro-Tick Replay Buffer initialized. Logging to {self.log_file_path}")

    def log_failed_cycle(self, apcf_result: Dict[str, Any], trade_outcome: Dict[str, Any]):
        """
        Logs the details of an APCF cycle that resulted in a loss.

        Args:
            apcf_result: The APCFResult components that led to the trade.
            trade_outcome: A dictionary containing the 'roi' or 'pnl' of the trade.
        """

        # We only log cycles with negative outcomes.
        roi = trade_outcome.get("roi", 0.0)
        if roi >= 0:
            return

        log_entry = {
            "timestamp": apcf_result.get("timestamp"),
            "apcf_value": apcf_result.get("apcf_value"),
            "apcf_state": apcf_result.get("state"),
            "components": apcf_result.get("components"),
            "outcome_roi": roi,
            "signature": apcf_result.get("mathematical_signature"),
        }

        try:
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info(f"Logged failed APCF cycle (ROI: {roi:.3f}) with signature {log_entry['signature']}.")
        except IOError as e:
            logger.error(f"Failed to write to micro-tick replay log: {e}")

    def get_failed_cycles(self) -> List[Dict[str, Any]]:
        """
        Reads the log file and returns a list of failed cycles for analysis.
        """
        if not self.log_file_path.exists():
            return []

        failed_cycles = []
        try:
            with open(self.log_file_path, "r") as f:
                for line in f:
                    try:
                        failed_cycles.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line in replay log: {line.strip()}")
            return failed_cycles
        except IOError as e:
            logger.error(f"Failed to read micro-tick replay log: {e}")
            return []


# Global instance
micro_tick_replay_buffer = MicroTickReplayBuffer()
