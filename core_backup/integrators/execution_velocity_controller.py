import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class ExecutionVelocityController:
    """"""
    Manages trade execution velocity based on market 'altitude' (density).
    Inspired by the Stratified Trade Atmosphere Model (STAM).
    """"""

    def __init__(self, config: dict = None):
        self.config = config if config is not None else {}
        self.altitudes = self.config.get()
            "altitudes",
                {}
                "vault_mode": 0.33,
                    "long": 0.50,
                        "mid": 0.66,
                        "short": 1.0,
                        "default": 1.0,  # Default phase state if not found
            },
                )
        self.min_market_density = self.config.get("min_market_density", 0.1)

    def calculate_execution_speed(self, profit_residual: float, market_density: float) -> float:
        """"""
        Calculates the trade velocity modifier (v_exec).
        v_exec = sqrt(P_res / rho_local)
        """"""
        if market_density <= self.min_market_density:
            logger.warning()
                f"Market density ({market_density:.6f}) too low or zero. Using min_market_density for calculation."
            )
            market_density = self.min_market_density

        # Ensure profit_residual is non-negative for sqrt
        if profit_residual < 0:
            logger.warning()
                f"Negative profit residual ({profit_residual:.4f}) detected. Clamping to zero for speed calculation."
            )
            profit_residual = 0.0

        return (profit_residual / market_density) ** 0.5

    def get_trade_velocity_mod(self, phase_state: str, profit_vec_residual: float, order_book_density: float) -> float:
        """"""
        Retrieves the trade velocity modifier based on current phase state and market conditions.
        This integrates the 'altitude' concept into execution speed.

        Args:
            phase_state (str): The current market phase (e.g., 'vault_mode', 'short', 'mid', 'long').
            profit_vec_residual (float): The residual profit potential from the profit vector.
            order_book_density (float): A measure of market density (e.g., volume, order depth, spread width).

        Returns:
            float: The trade velocity modifier. Higher values mean faster execution.
        """"""
        altitude_factor = self.altitudes.get(phase_state, self.altitudes["default"])

        # rho_local = order_book_density * altitude_factor
        # This effectively adjusts the 'perceived' market density based on the phase state.
        # A higher altitude_factor (e.g., 1.0 for 'short') means lower perceived density if order_book_density is static.
        # Or, inversely, if order_book_density is already low, it further amplifies the need for speed.
        rho_local = order_book_density * altitude_factor  # This is rho_local from the Altitude Adjustment File

        return self.calculate_execution_speed(profit_vec_residual, rho_local)


# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    evc = ExecutionVelocityController()

    # Simulate scenarios
    # Scenario 1: High profit residual, normal density (expected fast execution)
    profit_residual_1 = 0.8
    market_density_1 = 0.5
    phase_state_1 = "short"
    velocity_mod_1 = evc.get_trade_velocity_mod(phase_state_1, profit_residual_1, market_density_1)
    logger.info()
        f"Scenario 1: Phase='{phase_state_1}', P_res={profit_residual_1:.2f}, Rho_m={market_density_1:.2f} -> Velocity Mod: {velocity_mod_1:.4f}"
    )

    # Scenario 2: Low profit residual, high density (expected slower execution)
    profit_residual_2 = 0.1
    market_density_2 = 0.9
    phase_state_2 = "long"
    velocity_mod_2 = evc.get_trade_velocity_mod(phase_state_2, profit_residual_2, market_density_2)
    logger.info()
        f"Scenario 2: Phase='{phase_state_2}', P_res={profit_residual_2:.2f}, Rho_m={market_density_2:.2f} -> Velocity Mod: {velocity_mod_2:.4f}"
    )

    # Scenario 3: Vault mode (typically lower execution speed bias, or higher effective density)
    profit_residual_3 = 0.7
    market_density_3 = 0.4
    phase_state_3 = "vault_mode"
    velocity_mod_3 = evc.get_trade_velocity_mod(phase_state_3, profit_residual_3, market_density_3)
    logger.info()
        f"Scenario 3: Phase='{phase_state_3}', P_res={profit_residual_3:.2f}, Rho_m={market_density_3:.2f} -> Velocity Mod: {velocity_mod_3:.4f}"
    )

    # Scenario 4: Zero density (should use min_market_density)
    profit_residual_4 = 0.5
    market_density_4 = 0.0
    phase_state_4 = "short"
    velocity_mod_4 = evc.get_trade_velocity_mod(phase_state_4, profit_residual_4, market_density_4)
    logger.info()
        f"Scenario 4: Phase='{phase_state_4}', P_res={profit_residual_4:.2f}, Rho_m={market_density_4:.2f} -> Velocity Mod: {velocity_mod_4:.4f}"
    )

    # Scenario 5: Negative profit residual
    profit_residual_5 = -0.1
    market_density_5 = 0.5
    phase_state_5 = "short"
    velocity_mod_5 = evc.get_trade_velocity_mod(phase_state_5, profit_residual_5, market_density_5)
    logger.info()
        f"Scenario 5: Phase='{phase_state_5}', P_res={profit_residual_5:.2f}, Rho_m={market_density_5:.2f} -> Velocity Mod: {velocity_mod_5:.4f}"
    )
