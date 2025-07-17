import logging
from typing import Any, Dict, Optional

import numpy as np

# smart_money/zygot_shell.py


logger = logging.getLogger(__name__)


def compute_stability_index(Z: float, N: float, params: Optional[Dict[str, float]] = None) -> float:
    """
    Compute nuclear-inspired stability index for trade logic.
    Mathematical Logic: PnL / A = StabilityIndex (derived from semi-empirical mass formula)

    Args:
        Z (float): Conceptual 'Proton Z' - represents volume trend strength.
        N (float): Conceptual 'Neutron N' - represents volatility buffer/hedge capacity.
        params (Optional[Dict[str, float]]): Coefficient dictionary for analog constants.

    Returns:
        float: The calculated stability index.
    """
    if params is None:
        params = {
            "alpha_v": 15.8,
            "alpha_s": 18.3,
            "alpha_c": 0.714,
            "alpha_a": 23.2,
            "alpha_p": 12.0,
        }
    A = Z + N
    if A <= 0:  # Avoid division by zero or negative mass
        logger.warning("Attempted to compute stability with A <= 0. Returning -np.inf.")
        return -np.inf

    # Binding Energy (B) analog
    B = (
        params["alpha_v"] * A
        - params["alpha_s"] * (A ** (2 / 3))
        - params["alpha_c"] * (Z**2) / (A ** (1 / 3))
        - params["alpha_a"] * ((N - Z) ** 2) / A
        + params["alpha_p"] / (A**0.5)
    )

    stability_index = B / A
    logger.debug(
        f"Computed ZygotShell Stability Index: {stability_index:.4f} (Z={Z}, N={N})"
    )
    return stability_index


def check_shell_trade_signal(volume_signal: float, volatility_map: float, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Decides whether to enter a trade using post-Euler stability logic.

    Args:
        volume_signal (float): The current volume trend strength (analogous to Z).
        volatility_map (float): The current volatility buffer (analogous to N).
        config (Optional[Dict[str, Any]]): Configuration dictionary for thresholds.

    Returns:
        str: Trade signal ('ENTER_AGGRESSIVE', 'ENTER_SCALED', 'HOLD').
    """
    if config is None:
        config = {"stability_threshold_high": 7.5, "stability_threshold_low": 6.0}
    stability_index = compute_stability_index(
        volume_signal, volatility_map, config.get("params")
    )

    if stability_index > config["stability_threshold_high"]:
        logger.info(
            f"ZygotShell: ENTER_AGGRESSIVE (Stability Index: {stability_index:.2f})"
        )
        return "ENTER_AGGRESSIVE"
    elif stability_index > config["stability_threshold_low"]:
        logger.info(
            f"ZygotShell: ENTER_SCALED (Stability Index: {stability_index:.2f})"
        )
        return "ENTER_SCALED"
    else:
        logger.info(f"ZygotShell: HOLD (Stability Index: {stability_index:.2f})")
        return "HOLD"


if __name__ == "__main__":
    # Example Usage
    print("\n--- Testing compute_stability_index ---")
    # Example 1: Stable zone (like Carbon or Oxygen)
    z_carbon = 6.0
    n_carbon = 6.0
    stability_c = compute_stability_index(z_carbon, n_carbon)
    print(f"Stability for Z={z_carbon}, N={n_carbon}: {stability_c:.4f}")

    # Example 2: Highly stable (like Iron)
    z_iron = 26.0
    n_iron = 30.0
    stability_fe = compute_stability_index(z_iron, n_iron)
    print(f"Stability for Z={z_iron}, N={n_iron}: {stability_fe:.4f}")

    # Example 3: Unstable (high Z, low N)
    z_unstable = 50.0
    n_unstable = 40.0
    stability_unstable = compute_stability_index(z_unstable, n_unstable)
    print(f"Stability for Z={z_unstable}, N={n_unstable}: {stability_unstable:.4f}")

    print("\n--- Testing check_shell_trade_signal ---")
    # Scenario 1: Aggressive entry
    signal1 = check_shell_trade_signal(25.0, 30.0)  # High Z, N similar to Fe
    print(f"Signal for Z=25, N=30: {signal1}")

    # Scenario 2: Scaled entry
    signal2 = check_shell_trade_signal(6.0, 6.0)  # Like Carbon
    print(f"Signal for Z=6, N=6: {signal2}")

    # Scenario 3: Hold
    signal3 = check_shell_trade_signal(50.0, 40.0)  # Unstable
    print(f"Signal for Z=50, N=40: {signal3}")

    # Scenario 4: Custom parameters
    custom_params = {
        "alpha_v": 16.0,
        "alpha_s": 18.0,
        "alpha_c": 0.7,
        "alpha_a": 23.0,
        "alpha_p": 12.5,
    }
    custom_config = {
        "stability_threshold_high": 7.6,
        "stability_threshold_low": 6.1,
        "params": custom_params,
    }
    signal4 = check_shell_trade_signal(26.0, 30.0, custom_config)
    print(f"Signal for Z=26, N=30 (custom config): {signal4}")
