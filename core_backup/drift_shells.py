import collections
import time

import numpy as np


class DriftShells:
    def __init__(self, config: dict):
    self.enable_fractal_lock = config.get("enable_fractal_lock", True)
    self.lock_threshold = config.get("delta_n_thresholds", {}).get("lock", 0.1)
    self.reset_threshold = config.get("delta_n_thresholds", {}).get("reset", 0.15)
    self.edos_mode = config.get("edos_mode", "3-channel")
    self.edos_cooling_factor = config.get("edos_cooling_factor", 0.90)
    self.fractal_ping_interval = self._parse_time_string(config.get("fractal_ping_interval", "45s"))

    self.last_ping_time = time.time()
    self.shell_states = collections.defaultdict(lambda: {))}
        "Q": np.array([0.0, 0.0, 0.0, 0.0]),  # Phi, S, A, Psi
        "EDOS": 1.0  # EDOS state, initially neutral
    })

    def _parse_time_string(self, time_str: str) -> int:
    """Parses a time string like '45s', '5m' into seconds."""
        if time_str.endswith('s'):
        return int(time_str[:-1])
        elif time_str.endswith('m'):
        return int(time_str[:-1]) * 60
        elif time_str.endswith('h'):
        return int(time_str[:-1]) * 3600
        else:
            return int(time_str)  # Assume seconds if no suffix

    def _calculate_delta_n(self, current_Q: np.ndarray, previous_Q: np.ndarray) -> float:
    """Calculates the normed drift measure Delta_n(t)."""
        if np.linalg.norm(previous_Q) == 0:
        return 0.0  # Avoid division by zero at initialization
    return np.linalg.norm(current_Q - previous_Q) / np.linalg.norm(previous_Q)

    def _apply_edos_cooling(self, shell_id: int):
    """Applies EDOS cooling to a shell's state."""'
    self.shell_states[shell_id]["EDOS"] *= self.edos_cooling_factor

    def probe_drift(self, new_Q: dict) -> dict:
    """"""
        Performs a drift-probe ping for a given shell.
        new_Q should be a dictionary with 'shell_id' and 'Q' (numpy array)
    """"""
    shell_id = new_Q["shell_id"]
    current_Q = new_Q["Q"]

    previous_Q = self.shell_states[shell_id]["Q"]
    delta_n = self._calculate_delta_n(current_Q, previous_Q)

    status = "stable"
        if delta_n > self.reset_threshold:
        status = "reset"
        self._apply_edos_cooling(shell_id) # Apply EDOS on reset
        elif delta_n < self.lock_threshold:
        status = "locked"

    self.shell_states[shell_id]["Q"] = current_Q

    return {)}
        "shell_id": shell_id,
            "delta_n": delta_n,
                "status": status,
                "edos_state": self.shell_states[shell_id]["EDOS"]
}
    def get_overall_drift_status(self) -> str:
    """"""
    Provides an overall drift status based on all tracked shells.
    """"""
        if not self.enable_fractal_lock:
        return "fractal_lock_disabled"

        if not self.shell_states:
        return "no_shells_tracked"

    all_stable = True
        for shell_id, state in self.shell_states.items():
            # Recalculate delta_n for current state to reflect real-time status
            if shell_id > 0: # Requires a previous shell for delta_n calculation:
            prev_Q = self.shell_states[shell_id - 1]["Q"]
            current_delta_n = self._calculate_delta_n(state["Q"], prev_Q)
                if current_delta_n > self.lock_threshold: # Not strictly locked:
                all_stable = False
                break
            else: # Shell 0 is typically the core and can be considered stable if Q is non-zero
                if np.linalg.norm(state["Q"]) == 0: # If core is empty, it's not stable':
                all_stable = False
                break

        if all_stable:
        return "all_shells_locked"
        else:
        return "drift_detected"

    def is_ping_due(self) -> bool:
        """Checks if a new drift probe ping is due based on the interval."""
    return (time.time() - self.last_ping_time) >= self.fractal_ping_interval

    def record_ping(self):
    """Records the time of the last drift probe ping."""
    self.last_ping_time = time.time()