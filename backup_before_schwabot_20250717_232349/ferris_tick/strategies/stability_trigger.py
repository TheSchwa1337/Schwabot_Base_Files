# ferris_tick/strategies/stability_trigger.py

import logging
from typing import Any, Dict, Optional

# Assuming zygot_shell is in a reachable path like `smart_money`
# You might need to adjust this import based on your project structure or add a __init__.py for relative imports
try:
    from smart_money.zygot_shell import compute_stability_index
except ImportError:
    # Fallback for direct testing or different project setups
    logging.warning(
        "Could not import compute_stability_index from smart_money.zygot_shell. Please ensure the path is correct."
    )

    def compute_stability_index(Z: float, N: float, params: Optional[Dict[str, float]] = None) -> float:
        logging.error(
            "Placeholder compute_stability_index used. Please resolve import issue for smart_money.zygot_shell."
        )
        return 0.0


logger = logging.getLogger(__name__)


def check_shell_trade_signal(
    volume_signal: float, volatility_map: float, config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Decides whether to enter a trade using post-Euler stability logic.
    This function acts as a hook within the Ferris Tick processing loop.

    Args:
        volume_signal (float): The current volume trend strength (analogous to Z).
        volatility_map (float): The current volatility buffer (analogous to N).
        config (Optional[Dict[str, Any]]): Configuration dictionary for thresholds and ZygotShell parameters.

    Returns:
        str: Trade signal ('ENTER_AGGRESSIVE', 'ENTER_SCALED', 'HOLD').
    """
    if config is None:
        config = {
            "stability_threshold_high": 7.5,
            "stability_threshold_low": 6.0,
            "params": {  # Default ZygotShell parameters if not provided
                "alpha_v": 15.8,
                "alpha_s": 18.3,
                "alpha_c": 0.714,
                "alpha_a": 23.2,
                "alpha_p": 12.0,
            },
        }
    # Pass the ZygotShell parameters from the config to compute_stability_index
    stability_index = compute_stability_index(volume_signal, volatility_map, config.get("params"))

    if stability_index > config["stability_threshold_high"]:
        logger.info(f"StabilityTrigger: ENTER_AGGRESSIVE (Stability Index: {stability_index:.2f})")
        return "ENTER_AGGRESSIVE"
    elif stability_index > config["stability_threshold_low"]:
        logger.info(f"StabilityTrigger: ENTER_SCALED (Stability Index: {stability_index:.2f})")
        return "ENTER_SCALED"
    else:
        logger.info(f"StabilityTrigger: HOLD (Stability Index: {stability_index:.2f})")
        return "HOLD"


if __name__ == "__main__":
    # Example Usage
    # Setup basic logging for example output
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n--- Testing check_shell_trade_signal ---")
    # Scenario 1: Aggressive entry (simulated high stability)
    signal1 = check_shell_trade_signal(26.0, 30.0)  # Analogous to Fe-56
    print(f"Signal for Fe-like state (Z=26, N=30): {signal1}")

    # Scenario 2: Scaled entry (simulated moderate stability)
    signal2 = check_shell_trade_signal(6.0, 6.0)  # Analogous to C-12
    print(f"Signal for C-like state (Z=6, N=6): {signal2}")

    # Scenario 3: Hold (simulated low stability)
    signal3 = check_shell_trade_signal(50.0, 40.0)  # Analogous to unstable heavy nucleus
    print(f"Signal for unstable state (Z=50, N=40): {signal3}")

    # Scenario 4: Custom configuration
    custom_config = {
        "stability_threshold_high": 7.6,
        "stability_threshold_low": 6.1,
        "params": {
            "alpha_v": 16.0,
            "alpha_s": 18.0,
            "alpha_c": 0.7,
            "alpha_a": 23.0,
            "alpha_p": 12.5,
        },
    }
    signal4 = check_shell_trade_signal(26.0, 30.0, custom_config)
    print(f"Signal for Fe-like state (custom config): {signal4}")
