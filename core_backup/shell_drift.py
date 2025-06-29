# core/shell_drift.py

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def post_euler_phase_drift(theta: float) -> Tuple[float, float]:
    """"""
    Models phase drift away from the Euler collapse point at pi (pi).
    This function quantifies a 'loss shell' (near symmetry) and a 'profit shell'
    (regions where asymmetry supports rotational reentry).

    Mathematical Logic:
    Loss Shell = |sin(theta)| * exp(-((theta - pi)^2) * K)
    Profit Shell = 1 - Loss Shell
    Where K is a sensitivity constant.

    Args:
        theta (float): The phase angle in radians (e.g., from 0 to 2*pi).

    Returns:
        Tuple[float, float]: A tuple containing (loss_shell_value, profit_shell_value).
                              Both values are normalized between 0 and 1.
    """"""
    # Sensitivity constant for the exponential decay, making the loss shell sharper
    sensitivity_constant = 10.0

    # Calculate the loss shell: high near pi, low elsewhere
    # np.abs(np.sin(theta)) ensures it's always positive and peaks at pi/2, 3pi/2'
    # The exponential term makes it collapse sharply around pi
    loss_shell = np.abs(np.sin(theta)) * np.exp(-((theta - np.pi) ** 2) * sensitivity_constant)

    # Clamp loss_shell to be between 0 and 1 (though it should naturally be within this range)
    loss_shell = max(0.0, min(1.0, loss_shell))

    # Profit shell is the inverse of the loss shell
    profit_shell = 1.0 - loss_shell

    logger.debug(f"Phase Drift (Theta={theta:.2f}): Loss Shell={loss_shell:.4f}, Profit Shell={profit_shell:.4f}")
    return loss_shell, profit_shell


if __name__ == "__main__":
    # Example Usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n--- Testing post_euler_phase_drift ---")

    # Test around key points of the circle (0 to 2*pi)
    test_thetas = []
        0.0,
            np.pi / 4,
                np.pi / 2,
                3 * np.pi / 4,
                np.pi,
                5 * np.pi / 4,
                3 * np.pi / 2,
                7 * np.pi / 4,
                2 * np.pi,
]
    for t in test_thetas:
        loss, profit = post_euler_phase_drift(t)
        print(f"Theta={t:.2f} (rad): Loss Shell={loss:.4f}, Profit Shell={profit:.4f}")

    print("\n--- Detailed Test around Pi (Euler Collapse Point) ---")
    # Closer look around pi
    detailed_thetas = np.linspace(np.pi - 0.5, np.pi + 0.5, 11)  # 11 points around pi
    for t in detailed_thetas:
        loss, profit = post_euler_phase_drift(t)
        print(f"Theta={t:.4f} (rad): Loss Shell={loss:.4f}, Profit Shell={profit:.4f}")
