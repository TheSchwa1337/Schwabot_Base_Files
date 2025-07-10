from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

# -*- coding: utf-8 -*-
"""
Mathematical Library V2 - Enhanced Mathematical Functions.

Enhanced mathematical library with improved algorithms and additional
functionality that bridges V1 foundational and V3 AI-infused capabilities.

V2 Focus: Advanced indicators, statistical analysis, pattern recognition
Integrates with: mathlib.py(V1), mathlib_v3.py, advanced_mathematical_core.py
"""



if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Type aliases for better readability
Vector = np.ndarray
Matrix = np.ndarray


@dataclass
class AdvancedIndicators:
    """Container for advanced trading indicators."""

    atr: float
    williams_r: float
    stochastic_k: float
    stochastic_d: float
    commodity_channel_index: float


@dataclass
class HashMemoryBlock:
    """Memory block for hash-based fractal pattern storage."""

    hash_signature: str
    profit: float
    strategy_id: str
    entry_vector: List[float]
    exit_vector: List[float]
    timestamp: float
    similarity_score: float = 0.0


class CoreMathLibV2:
    """Enhanced mathematical library V2 with hash-based fractal memory."""

    def __init__():-> None:
        """Initialize the enhanced mathematical library."""
        self.version = "2.0_0"
        self.initialized = True
        self.hash_memory = {}  # Fractal hash memory storage
        self.memory_blocks = []  # 256-block memory system
        self.max_memory_blocks = 256

        logger.info(f"CoreMathLibV2 v{self.version} initialized")

    def calculate_vwap():-> Vector:
        """Calculate Volume Weighted Average Price."""
        if len(prices) != len(volumes) or len(prices) == 0:
            return np.zeros_like(prices)

        cumulative_volume = np.cumsum(volumes)
        cumulative_pv = np.cumsum(prices * volumes)

        # Avoid division by zero
        vwap = np.divide(
            cumulative_pv,
            cumulative_volume,
            out=np.zeros_like(cumulative_pv),
            where=cumulative_volume != 0,
        )
        return vwap

    def calculate_true_range():-> Vector:
        """Calculate True Range for ATR."""
        if len(high) != len(low) or len(low) != len(close) or len(high) < 2:
            return np.zeros_like(high)

        # Previous close
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first element

        # True Range components
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        # Maximum of the three
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range

    def calculate_atr():-> Vector:
        """Calculate Average True Range."""
        true_range = self.calculate_true_range(high, low, close)

        if len(true_range) < period:
            return np.full_like(true_range, np.mean(true_range))

        # Calculate ATR using exponential moving average
        atr = np.zeros_like(true_range)
        atr[:period] = np.mean(true_range[:period])  # Initial ATR

        # Smoothing factor
        alpha = 1.0 / period

        for i in range(period, len(true_range)):
            atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i - 1]

        return atr

    def calculate_rsi():-> Vector:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return np.full_like(prices, 50.0)

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        rsi = np.zeros(len(prices))
        rsi[:period] = 50.0  # Neutral RSI for initial values

        # Calculate initial average gain and loss
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # Smoothing factor
        alpha = 1.0 / period

        for i in range(period, len(prices) - 1):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))

        return np.clip(rsi, 0, 100)

    def calculate_williams_r():-> Vector:
        """Calculate Williams % R."""
        if len(high) < period:
            return np.zeros_like(high)

        williams_r = np.zeros_like(high)

        for i in range(period - 1, len(high)):
            highest_high = np.max(high[i - period + 1 : i + 1])
            lowest_low = np.min(low[i - period + 1 : i + 1])

            if highest_high - lowest_low == 0:
                williams_r[i] = -50.0
            else:
                williams_r[i] = (
                    -100 * (highest_high - close[i]) / (highest_high - lowest_low)
                )

        return williams_r

    def calculate_stochastic():-> Dict[str, Vector]:
        """Calculate Stochastic Oscillator."""
        if len(high) < k_period:
            return {
                "k_percent": np.zeros_like(high),
                "d_percent": np.zeros_like(high),
            }
        k_percent = np.zeros_like(high)

        for i in range(k_period - 1, len(high)):
            highest_high = np.max(high[i - k_period + 1 : i + 1])
            lowest_low = np.min(low[i - k_period + 1 : i + 1])

            if highest_high - lowest_low == 0:
                k_percent[i] = 50.0
            else:
                k_percent[i] = (
                    100 * (close[i] - lowest_low) / (highest_high - lowest_low)
                )

        # Calculate %D as moving average of %K
        d_percent = np.zeros_like(k_percent)
        for i in range(d_period - 1, len(k_percent)):
            d_percent[i] = np.mean(k_percent[i - d_period + 1 : i + 1])

        return {"k_percent": k_percent, "d_percent": d_percent}

    def calculate_cci():-> Vector:
        """Calculate Commodity Channel Index."""
        if len(high) < period:
            return np.zeros_like(high)

        # Typical Price
        typical_price = (high + low + close) / 3

        cci = np.zeros_like(typical_price)

        for i in range(period - 1, len(typical_price)):
            tp_period = typical_price[i - period + 1 : i + 1]
            sma_tp = np.mean(tp_period)
            mean_deviation = np.mean(np.abs(tp_period - sma_tp))

            if mean_deviation == 0:
                cci[i] = 0
            else:
                cci[i] = (typical_price[i] - sma_tp) / (0.015 * mean_deviation)

        return cci

    def advanced_statistical_analysis():-> Dict[str, float]:
        """Perform advanced statistical analysis of data."""
        if len(data) == 0:
            return {"error": "Empty data"}

        # Basic statistics
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)

        # Skewness and Kurtosis
        n = len(data)
        if std_val == 0 or n < 3:
            skewness = 0.0
            kurtosis = 0.0
        else:
            # Skewness calculation
            skewness = (n / ((n - 1) * (n - 2))) * np.sum(
                ((data - mean_val) / std_val) ** 3
            )

            # Kurtosis calculation (excess kurtosis)
            if n < 4:
                kurtosis = 0.0
            else:
                kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(
                    ((data - mean_val) / std_val) ** 4
                ) - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

        # Jarque-Bera test statistic for normality
        jb_statistic = (n / 6) * (skewness**2 + (kurtosis**2) / 4) if n > 6 else 0.0

        return {
            "mean": float(mean_val),
            "std": float(std_val),
            "variance": float(std_val**2),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "jarque_bera": float(jb_statistic),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
        }

    def entropy_analysis():-> Dict[str, float]:
        """Perform entropy analysis of data distribution."""
        if len(data) == 0:
            return {"shannon_entropy": 0.0, "normalized_entropy": 0.0}

        # Create histogram
        hist, _ = np.histogram(data, bins=bins, density=True)

        # Normalize to probabilities
        hist = hist / np.sum(hist)

        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]

        # Shannon entropy
        shannon_entropy = -np.sum(hist * np.log2(hist))

        # Normalized entropy (0 to 1)
        max_entropy = np.log2(len(hist)) if len(hist) > 1 else 1.0
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "shannon_entropy": float(shannon_entropy),
            "normalized_entropy": float(normalized_entropy),
            "max_entropy": float(max_entropy),
        }

    def moving_average_variants():-> Dict[str, float]:
        """Calculate various moving average types."""
        if len(data) < period:
            period = len(data)

        if period == 0:
            return {"sma": 0.0, "ema": 0.0, "wma": 0.0, "hull_ma": 0.0}

        recent_data = data[-period:]

        # Simple Moving Average
        sma = np.mean(recent_data)

        # Exponential Moving Average
        alpha = 2.0 / (period + 1)
        ema = recent_data[0]
        for price in recent_data[1:]:
            ema = alpha * price + (1 - alpha) * ema

        # Weighted Moving Average
        weights = np.arange(1, period + 1)
        wma = np.sum(recent_data * weights) / np.sum(weights)

        # Hull Moving Average (simplified)
        half_period = period // 2
        if len(data) >= period and half_period > 0:
            wma_half = np.sum(
                data[-half_period:] * np.arange(1, half_period + 1)
            ) / np.sum(np.arange(1, half_period + 1))
            wma_full = wma
            hull_ma = 2 * wma_half - wma_full
        else:
            hull_ma = sma

        return {
            "sma": float(sma),
            "ema": float(ema),
            "wma": float(wma),
            "hull_ma": float(hull_ma),
        }

    def generate_hash_signature():-> str:
        """Generate SHA-256 hash signature for data and strategy."""
        data_str = f"{strategy_id}_{np.array2string(data, precision=6)}"
        return hashlib.sha256(data_str.encode()).hexdigest()

    def hash_similarity_score():-> float:
        """Calculate hash similarity score using Hamming distance."""
        if not memory_hashes:
            return 0.0

        # Convert hex hashes to binary for Hamming distance
        new_binary = bin(int(new_hash, 16))[2:].zfill(256)

        similarities = []
        for old_hash in memory_hashes:
            old_binary = bin(int(old_hash, 16))[2:].zfill(256)

            # Calculate Hamming distance
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(new_binary, old_binary))

            # Convert to similarity score (0-1)
            similarity = 1.0 - (hamming_dist / 256)
            similarities.append(similarity)

        return np.mean(similarities)

    def store_hash_memory():-> None:
        """Store hash memory block in fractal memory system."""
        # Add to memory blocks
        self.memory_blocks.append(hash_block)

        # Keep only max_memory_blocks
        if len(self.memory_blocks) > self.max_memory_blocks:
            self.memory_blocks = self.memory_blocks[-self.max_memory_blocks :]

        # Store in hash memory dictionary
        self.hash_memory[hash_block.hash_signature] = hash_block

        logger.info(f"Stored hash memory: {hash_block.hash_signature[:16]}...")

    def recall_profitable_patterns():-> List[HashMemoryBlock]:
        """Recall profitable patterns based on hash similarity."""
        profitable_patterns = []

        for block in self.memory_blocks:
            similarity = self.hash_similarity_score(
                current_hash, [block.hash_signature]
            )
            if similarity > threshold and block.profit > 0:
                block.similarity_score = similarity
                profitable_patterns.append(block)

        # Sort by profit and similarity
        profitable_patterns.sort(
            key=lambda x: (x.profit, x.similarity_score), reverse=True
        )

        return profitable_patterns

    def export_hash_memory():-> str:
        """Export hash memory to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hash_memory_{timestamp}.json"

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "version": self.version,
            "memory_blocks": [
                {
                    "hash_signature": block.hash_signature,
                    "profit": block.profit,
                    "strategy_id": block.strategy_id,
                    "entry_vector": block.entry_vector,
                    "exit_vector": block.exit_vector,
                    "timestamp": block.timestamp,
                    "similarity_score": block.similarity_score,
                }
                for block in self.memory_blocks
            ],
        }
        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Hash memory exported to: {filename}")
        return filename


def process_waveform():-> Dict[str, Any]:
    """
    Process waveform data with various analysis types.

    Args:
        signal: Input signal data
        sample_rate: Sampling rate of the signal
        analysis_type: Type of analysis ("basic", "advanced", "spectral")

    Returns:
        Dictionary with analysis results
    """
    try:
        mathlib = CoreMathLibV2()

        if len(signal) == 0:
            return {"status": "error", "error": "Empty signal"}

        result = {
            "status": "success",
            "signal_length": len(signal),
            "sample_rate": sample_rate,
        }
        if analysis_type == "basic":
            # Basic statistical analysis
            stats = mathlib.advanced_statistical_analysis(signal)
            result.update(stats)

        elif analysis_type == "advanced":
            # Advanced statistical + entropy analysis
            stats = mathlib.advanced_statistical_analysis(signal)
            entropy = mathlib.entropy_analysis(signal)
            moving_avgs = mathlib.moving_average_variants(signal)

            result.update(
                {
                    "statistics": stats,
                    "entropy": entropy,
                    "moving_averages": moving_avgs,
                }
            )

        elif analysis_type == "spectral":
            # Basic spectral analysis (simplified)
            fft_result = np.fft.fft(signal)
            power_spectrum = np.abs(fft_result) ** 2
            dominant_freq_idx = np.argmax(power_spectrum[: len(power_spectrum) // 2])
            dominant_frequency = dominant_freq_idx * sample_rate / len(signal)

            result.update(
                {
                    "dominant_frequency": float(dominant_frequency),
                    "spectral_power": float(np.sum(power_spectrum)),
                    "spectral_centroid": float(
                        np.sum(np.arange(len(power_spectrum)) * power_spectrum)
                        / np.sum(power_spectrum)
                    ),
                }
            )

        return result

    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():-> None:
    """Demo of CoreMathLibV2 capabilities."""
    try:
        mathlib = CoreMathLibV2()
        print(f"✅ CoreMathLibV2 v{mathlib.version} initialized")

        # Demo data
        prices = np.array([100, 102, 98, 105, 103, 107, 104, 108, 106, 110])
        volumes = np.array([1000, 1200, 800, 1500, 1100, 1300, 900, 1400, 1000, 1600])

        # Test VWAP
        vwap = mathlib.calculate_vwap(prices, volumes)
        print(f"📊 VWAP: {vwap[-1]:.2f}")

        # Test RSI
        rsi = mathlib.calculate_rsi(prices)
        print(f"📈 RSI: {rsi[-1]:.2f}")

        # Test statistical analysis
        stats = mathlib.advanced_statistical_analysis(prices)
        print(f"📊 Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")

        # Test hash memory system
        hash_sig = mathlib.generate_hash_signature(prices, "RSI_STRATEGY")
        print(f"🔐 Hash Signature: {hash_sig[:16]}...")

        # Create and store hash memory block
        hash_block = HashMemoryBlock(
            hash_signature=hash_sig,
            profit=0.082,
            strategy_id="RSI_ECHO_T5",
            entry_vector=prices.tolist(),
            exit_vector=prices.tolist(),
            timestamp=datetime.now().timestamp(),
        )
        mathlib.store_hash_memory(hash_block)

        print("🎉 CoreMathLibV2 demo completed!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
