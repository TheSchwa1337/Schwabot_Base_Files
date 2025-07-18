import csv
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cupy as cp
import numba
import numpy as np
from numba import cuda

#!/usr/bin/env python3
"""
Trading Matrix Visualizer (Enhanced)
===================================

- Bar index for time/row tracking
- Modular strategy_fn for custom signals
- CSV loader for real market data
- Backtesting loop for historical data
- Live visual feedback
- GPU/CPU optimization with proper fallbacks
- Flake8/mypy compliant
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GPU/CPU Detection and Optimization
    class HardwareOptimizer:
    """Detects and optimizes for available hardware."""

    def __init__(self):
        self.gpu_available = False
        self.cuda_available = False
        self.numba_available = False
        self.optimization_mode = "cpu"
        self._detect_hardware()

    def _detect_hardware(self):
        """Detect available hardware for optimization."""
        try:

            self.gpu_available = True
            self.cuda_available = True
            logger.info("CuPy GPU acceleration detected")
        except ImportError:
            logger.info("CuPy not available, using CPU fallback")

        try:

            self.numba_available = True
            logger.info("Numba CUDA acceleration detected")
        except ImportError:
            logger.info("Numba not available, using CPU fallback")

        if self.gpu_available or self.numba_available:
            self.optimization_mode = "gpu"
        else:
            self.optimization_mode = "cpu"

        logger.info(f"Hardware optimization mode: {self.optimization_mode}")

    def get_optimization_info():-> Dict[str, Any]:
        """Get current optimization configuration."""
        return {}
            "gpu_available": self.gpu_available,
            "cuda_available": self.cuda_available,
            "numba_available": self.numba_available,
            "optimization_mode": self.optimization_mode,
            "timestamp": datetime.now().isoformat(),
        }


# --- Trading Matrix Core ---
    class TradingMatrix:
    """High-speed trading matrix for prices, signals, and positions with GPU/CPU optimization."""

    def __init__():-> None:
        self.n_assets = n_assets
        self.window = window
        self.optimizer = optimizer or HardwareOptimizer()

        # Initialize matrices with proper data types
        self.price_matrix = np.zeros((window, n_assets), dtype=np.float64)
        self.signal_matrix = np.zeros((window, n_assets), dtype=np.float64)
        self.position_matrix = np.zeros((window, n_assets), dtype=np.int8)
        self.ptr = 0
        self.bar_index = 0

        # Performance tracking
        self.performance_metrics = {}
            "updates": 0,
            "total_time": 0.0,
            "avg_update_time": 0.0,
        }
        logger.info()
            f"TradingMatrix initialized: {n_assets} assets, {window} window, {self.optimizer.optimization_mode} mode"
        )

    def update():-> None:
        """Update matrices with new prices and signals using optimized operations."""
        start_time = time.time()

        # Validate inputs
        if len(prices) != self.n_assets or len(signals) != self.n_assets:
            raise ValueError(f"Input arrays must have length {self.n_assets}")

        # Update matrices
        self.price_matrix[self.ptr % self.window] = prices
        self.signal_matrix[self.ptr % self.window] = signals
        self.position_matrix[self.ptr % self.window] = self.generate_positions(signals)

        self.ptr += 1
        self.bar_index += 1

        # Update performance metrics
        update_time = time.time() - start_time
        self.performance_metrics["updates"] += 1
        self.performance_metrics["total_time"] += update_time
        self.performance_metrics["avg_update_time"] = ()
            self.performance_metrics["total_time"] / self.performance_metrics["updates"]
        )

    def generate_positions():-> np.ndarray:
        """Fast bitwise signal-to-position logic with optimization."""
        if self.optimizer.optimization_mode == "gpu":
            return self._generate_positions_gpu(signals)
        else:
            return self._generate_positions_cpu(signals)

    def _generate_positions_cpu():-> np.ndarray:
        """CPU-optimized position generation."""
        return np.where(signals > 0.5, 1, np.where(signals < -0.5, -1, 0))

    def _generate_positions_gpu():-> np.ndarray:
        """GPU-optimized position generation with fallback."""
        try:
            if self.optimizer.gpu_available:

                signals_gpu = cp.array(signals)
                positions_gpu = cp.where()
                    signals_gpu > 0.5, 1, cp.where(signals_gpu < -0.5, -1, 0)
                )
                return cp.asnumpy(positions_gpu)
            else:
                return self._generate_positions_cpu(signals)
        except Exception as e:
            logger.warning(f"GPU position generation failed, falling back to CPU: {e}")
            return self._generate_positions_cpu(signals)

    def get_recent():-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get recent data with proper indexing."""
        if self.ptr == 0:
            return self.price_matrix, self.signal_matrix, self.position_matrix

        start_idx = max(0, self.ptr - self.window)
        end_idx = self.ptr
        indices = np.arange(start_idx, end_idx) % self.window

        return ()
            self.price_matrix[indices],
            self.signal_matrix[indices],
            self.position_matrix[indices],
        )

    def drift_vector():-> np.ndarray:
        """Calculate drift vector with optimization."""
        prices, _, _ = self.get_recent()
        if len(prices) == 0:
            return np.zeros(self.n_assets)

        anchor = np.mean(prices, axis=0)
        return prices[-1] - anchor

    def entropy():-> np.ndarray:
        """Calculate entropy with optimization."""
        prices, _, _ = self.get_recent()
        if len(prices) == 0:
            return np.zeros(self.n_assets)

        return np.std(prices, axis=0)

    def consensus():-> np.ndarray:
        """Calculate consensus with optimization."""
        _, signals, _ = self.get_recent()
        if len(signals) == 0:
            return np.zeros(self.n_assets)

        return np.where(np.mean(signals > 0, axis=0) > 0.5, 1, -1)

    def get_performance_metrics():-> Dict[str, Any]:
        """Get performance metrics."""
        return {}
            **self.performance_metrics,
            "hardware_info": self.optimizer.get_optimization_info(),
            "matrix_size": f"{self.window}x{self.n_assets}",
            "current_bar": self.bar_index,
        }


# --- Glyph Visualizer ---
GLYPHS = ["1", "i", "·", " ", "⊥"]


def trading_state_to_glyphs():-> List[List[str]]:
    """Map trading state to a glyph matrix for visualization."""
    n = len(drift)
    matrix: List[List[str]] = []

    for row in range(4):
        glyph_row: List[str] = []
        for col in range(n):
            if consensus[col] > 0:
                glyph = "1" if abs(drift[col]) < 0.5 else "i"
            elif consensus[col] < 0:
                glyph = "i" if abs(drift[col]) < 0.5 else "·"
            else:
                glyph = " " if entropy[col] < 0.5 else "·"
            glyph_row.append(glyph)
        matrix.append(glyph_row)

    return matrix


def print_glyph_matrix():-> None:
    """Print glyph matrix with proper formatting."""
    for row in matrix:
        print("".join(row))


# --- Market Data Loader ---
    def load_market_data_csv():-> np.ndarray:
    """Load market data from CSV with error handling."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CSV file not found: {filename}")

    data: List[List[float]] = []
    try:
        with open(filename, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for row_num, row in enumerate(reader, start=2):
                try:
                    prices = [float(row[price_col_start + i]) for i in range(n_assets)]
                    data.append(prices)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Invalid data in row {row_num}: {e}")
                    continue

        if not data:
            raise ValueError("No valid data found in CSV file")

        return np.array(data, dtype=np.float64)

    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise


# --- Strategy Function Example ---
    def example_strategy_fn():-> np.ndarray:
    """Simple momentum strategy: signal = price change over last bar."""
    if bar_index == 0:
        return np.zeros_like(prices)

    # Calculate price changes
    if bar_index == 1:
        return np.sign(prices)
    else:
        return np.sign(prices - np.roll(prices, 1))


# --- Backtesting Loop ---
    def run_backtest():-> None:
    """Run backtest with hardware optimization."""
    n_bars, n_assets = price_data.shape
    optimizer = optimizer or HardwareOptimizer()

    tm = TradingMatrix(n_assets=n_assets, window=min(32, n_bars), optimizer=optimizer)

    logger.info(f"Starting backtest: {n_bars} bars, {n_assets} assets")
    logger.info(f"Hardware mode: {optimizer.optimization_mode}")

    try:
        for t in range(n_bars):
            prices = price_data[t]
            signals = strategy_fn(prices, t)
            tm.update(prices, signals)

            drift = tm.drift_vector()
            entropy = tm.entropy()
            consensus = tm.consensus()
            glyph_matrix = trading_state_to_glyphs(drift, entropy, consensus)

            # Clear screen
            os.system("cls" if os.name == "nt" else "clear")

            # Display information
            print(f"Trading Matrix Visualizer | Bar {t + 1}/{n_bars}")
            print(f"Hardware: {optimizer.optimization_mode.upper()}")
            print()
                f"Performance: {tm.performance_metrics['avg_update_time']:.4f}s avg\n"
            )

            print_glyph_matrix(glyph_matrix)
            print(f"\nDrift:     {np.round(drift, 2)}")
            print(f"Entropy:   {np.round(entropy, 2)}")
            print(f"Consensus: {consensus}")

            time.sleep(delay)

    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise
    finally:
        # Display final performance metrics
        metrics = tm.get_performance_metrics()
        logger.info(f"Backtest completed. Performance: {metrics}")


# --- Main Entrypoint ---
    def main():-> None:
    """Main entry point with comprehensive error handling."""
    try:
        # Initialize hardware optimizer
        optimizer = HardwareOptimizer()

        # Configuration
        use_csv = False
        csv_file = "market_data.csv"  # Replace with your file
        n_assets = 12
        steps = 100

        logger.info("Trading Matrix Visualizer starting...")
        logger.info(f"Hardware configuration: {optimizer.get_optimization_info()}")

        if use_csv:
            if not os.path.exists(csv_file):
                logger.warning(f"CSV file not found: {csv_file}")
                logger.info("Falling back to simulated data")
                use_csv = False

            if use_csv:
                price_data = load_market_data_csv(csv_file, n_assets)
                run_backtest(price_data, example_strategy_fn, optimizer=optimizer)

        if not use_csv:
            # Generate simulated data
            np.random.seed(42)  # For reproducible results
            price_data = ()
                np.cumsum(np.random.normal(0, 1, (steps, n_assets)), axis=0) + 100
            )
            run_backtest(price_data, example_strategy_fn, optimizer=optimizer)

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
