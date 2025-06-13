"""
ColdBase Bridge
===============

Provides memory echo matching:
- similarity(A_current, A_memory) = exp(-Δ_sym/σ)

Invariants:
- Memory hit rate: successful match if similarity > ε

See docs/math/coldbase.md for details.
"""
import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ColdBaseBridge:
    def __init__(self, sigma: float = 0.25, epsilon: float = 0.85):
        """
        Initialize the ColdBase memory matcher.

        Parameters:
            sigma (float): Controls the sharpness of the similarity decay curve.
            epsilon (float): Threshold for a match to be considered valid.
        """
        self.sigma = sigma
        self.epsilon = epsilon
        self.memory_bank: List[Dict[str, Any]] = []

    def add_memory(self, anchor: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store an anchor in memory.

        Parameters:
            anchor (str): The memory anchor string.
            metadata (Optional[Dict[str, Any]]): Optional metadata attached to the anchor.
        """
        self.memory_bank.append({
            "anchor": anchor,
            "metadata": metadata or {}
        })

    def _similarity(self, a: str, b: str) -> float:
        """
        Compute exponential similarity between anchors.

        Parameters:
            a (str): First anchor.
            b (str): Second anchor.

        Returns:
            float: Similarity score between 0 and 1.
        """
        delta = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        similarity = math.exp(-delta / self.sigma)
        return similarity

    def match(self, anchor: str) -> Optional[Dict[str, Any]]:
        """
        Find best memory match for anchor.

        Parameters:
            anchor (str): Anchor string to match against the memory bank.

        Returns:
            Optional[Dict[str, Any]]: Closest match with metadata, or None if no match found.
        """
        logger.info(f"[ColdBase] Matching anchor: '{anchor}'")
        best_match = None
        best_score = 0.0

        for entry in self.memory_bank:
            score = self._similarity(anchor, entry["anchor"])
            logger.debug(f"Compared to '{entry['anchor']}', score={score:.4f}")
            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= self.epsilon:
            logger.info(f"Memory match found with similarity {best_score:.4f}")
            return best_match
        else:
            logger.info(f"No match found above epsilon={self.epsilon}")
            return None

    def clear_memory(self):
        """Clears the memory bank."""
        self.memory_bank.clear()

    @staticmethod
    def load_default_config() -> Dict[str, Any]:
        """
        Load default configuration for ColdBaseBridge.

        Returns:
            Dict[str, Any]: Default configuration.
        """
        return {
            "sigma": 0.25,
            "epsilon": 0.85
        }

    @staticmethod
    def save_config(config: Dict[str, Any], file_path: str):
        """
        Save configuration to a file.

        Parameters:
            config (Dict[str, Any]): Configuration dictionary.
            file_path (str): Path to the configuration file.
        """
        with open(file_path, 'w') as f:
            f.write(f"sigma={config['sigma']}\n")
            f.write(f"epsilon={config['epsilon']}\n")

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.

        Parameters:
            file_path (str): Path to the configuration file.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        config = {}
        with open(file_path, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                config[key] = float(value)
        return config

# Example usage
if __name__ == "__main__":
    # Load default configuration
    config = ColdBaseBridge.load_default_config()
    logger.info(f"Loaded default configuration: {config}")

    # Create an instance of ColdBaseBridge
    bridge = ColdBaseBridge(**config)

    # Add some memory entries
    bridge.add_memory("BTC", {"price": 10000.0})
    bridge.add_memory("ETH", {"price": 2000.0})

    # Match an anchor
    match = bridge.match("BTC")
    if match:
        logger.info(f"Match found: {match}")
    else:
        logger.info("No match found.")

    # Clear the memory bank
    bridge.clear_memory()

    # Save and load configuration
    ColdBaseBridge.save_config(config, "config.txt")
    loaded_config = ColdBaseBridge.load_config("config.txt")
    logger.info(f"Loaded saved configuration: {loaded_config}") 