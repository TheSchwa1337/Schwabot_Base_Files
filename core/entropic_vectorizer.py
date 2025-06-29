import hashlib

import numpy as np


class EntropicVectorizer:
    def __init__(self, config: dict = None):
        # Configuration for entropic vectorization can be added here if needed
        pass

    def fold_256(self, src: bytes, out_bits: int = 16) -> int:
        """
        Folds a SHA256 hash (32 bytes) into a smaller integer representation
        of specified output_bits length using an XOR cascade.
        """
        h = hashlib.sha256(src).digest()

        # Ensure out_bits is a multiple of 8 (byte-aligned) and within valid range
        if out_bits % 8 != 0 or out_bits <= 0 or out_bits > 256:
            raise ValueError("out_bits must be a positive multiple of 8, up to 256.")

        step = out_bits // 8
        val = 0
        # XOR cascade folding
        for i in range(0, 32, step):
            # Ensure we don't go out of bounds for the last chunk
            chunk = int.from_bytes(h[i : i + step], "big")
            val ^= chunk

        # Mask to ensure the result fits within out_bits
        return val & ((1 << out_bits) - 1)

    def build_strategy_vec(self, block_hash: str, price_hash: str, extra_seed: bytes = b"") -> tuple[int, float, int]:
        """
        Builds a strategy vector from block hash, price hash, and an optional extra seed.
        Returns (class_id, risk_scalar, xor_drift_value).

        Args:
            block_hash (str): Hex string of the BTC block hash.
            price_hash (str): Hex string of the synthetic/real BTC price hash.
            extra_seed (bytes): Optional additional seed for entropy (e.g., from an XOR asset).

        Returns:
            tuple[int, float, int]:
                - class_id (int): 4-bit strategy bucket ID (0-15).
                - risk_scalar (float): 8-bit risk value normalized to 0-1.
                - xor_drift_value (int): 4-bit XOR drift value for further modulation.
        """
        # Combine sources into a single byte string
        try:
            src = bytes.fromhex(block_hash) + bytes.fromhex(price_hash) + extra_seed
        except ValueError as e:
            raise ValueError(f"Invalid hex string provided for hashing: {e}")

        # Generate a 16-bit folded value for primary strategy and risk
        vec16 = self.fold_256(src, out_bits=16)

        # Extract 4-bit class_id (high nibble of vec16)
        class_id = (vec16 >> 12) & 0xF  # Shift right by 12 bits, take last 4 bits

        # Extract 8-bit risk_nibble (middle 8 bits of vec16)
        risk_nibble = (vec16 >> 4) & 0xFF  # Shift right by 4 bits, take last 8 bits
        risk_scalar = risk_nibble / 255.0  # Normalize to 0-1

        # Extract 4-bit xor_drift (low nibble of vec16)
        xor_drift_value = vec16 & 0xF  # Take the last 4 bits

        return class_id, risk_scalar, xor_drift_value

    def calculate_entropy_slope(self, historical_entropy_values: collections.deque) -> float:
        """
        Calculates the rate of change of entropy over a series of historical values.
        Args:
            historical_entropy_values (collections.deque): A deque containing historical entropy values (floats).
        Returns:
            float: The entropy slope.
        """
        if len(historical_entropy_values) < 2:
            return 0.0

        # Simple linear slope for now, can be replaced with more complex regression
        s_t = historical_entropy_values[-1]
        s_t_minus_1 = historical_entropy_values[-2]

        return s_t - s_t_minus_1
