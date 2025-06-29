#!/usr/bin/env python3
""""""
Strategy Loader - Core component for loading and managing trading strategies.
""""""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


class StrategyLoader:
    """Loads and manages trading strategies from various sources."""

    def __init__(self, strategy_dir: str = "strategies"):
        """Initialize strategy loader."""

        Args:
            strategy_dir: Directory containing strategy files
        """"""
        self.strategy_dir = Path(strategy_dir)
        self.strategies = {}
        self.loaded_strategies = {}

    def load_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Load a specific strategy by name."""

        Args:
            strategy_name: Name of the strategy to load

        Returns:
            Strategy configuration dictionary
        """"""
        try:
            strategy_file = self.strategy_dir / f"{strategy_name}.json"
            if strategy_file.exists():
                with open(strategy_file, "r") as f:
                    strategy = json.load(f)
                self.loaded_strategies[strategy_name] = strategy
                return strategy
            else:
                raise FileNotFoundError(f"Strategy file not found: {strategy_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load strategy {strategy_name}: {e}")

    def load_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load all available strategies."""

        Returns:
            Dictionary of all loaded strategies
        """"""
        try:
            for strategy_file in self.strategy_dir.glob("*.json"):
                strategy_name = strategy_file.stem
                self.load_strategy(strategy_name)
            return self.loaded_strategies
        except Exception as e:
            raise RuntimeError(f"Failed to load all strategies: {e}")

    def validate_strategy(self, strategy: Dict[str, Any]) -> bool:
        """Validate strategy configuration."""

        Args:
            strategy: Strategy configuration to validate

        Returns:
            True if valid, False otherwise
        """"""
        required_fields = ["name", "type", "parameters"]
        return all(field in strategy for field in required_fields)

    def get_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Get parameters for a specific strategy."""

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy parameters dictionary
        """"""
        if strategy_name not in self.loaded_strategies:
            self.load_strategy(strategy_name)
        return self.loaded_strategies[strategy_name].get("parameters", {})

    def list_available_strategies(self) -> List[str]:
        """List all available strategy files."""

        Returns:
            List of strategy names
        """"""
        return [f.stem for f in self.strategy_dir.glob("*.json")]


def main():
    """Main function for testing."""
    loader = StrategyLoader()
    print("Strategy Loader initialized successfully!")

    # List available strategies
    strategies = loader.list_available_strategies()
    print(f"Available strategies: {strategies}")


if __name__ == "__main__":
    main()
