"""Module for Schwabot trading system."""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
    class SoulprintEntry:
    """Class for Schwabot trading functionality."""
    """Class for Schwabot trading functionality."""
    soulprint: str
    timestamp: str
    vector: Dict[str, float]  # DriftVector serialized
    strategy_id: str
    confidence: float
    is_executed: bool = False
    profit_result: Optional[float] = None
    replayable: bool = True


        class SoulprintRegistry:
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """
        Registry for logging
        querying Schwafit triggers, profit vectors, phase/drift optimization, and cross-asset analytics.
        """

            def __init__(self, registry_file: Optional[str] = None) -> None:
            self.triggers: List[Dict[str, Any]] = []
            self.registry_file = registry_file
                if registry_file:
                self._load()

                def log_trigger()
                self,
                asset: str,
                phase: float,
                drift: float,
                schwafit_info: dict,
                trade_result: dict,
                timestamp: Optional[float] = None,
                    ):
                    """Log a trigger event (entry/exit) with all math, phase, drift, and trade outcome."""
                    entry = {}
                    "asset": asset,
                    "phase": phase,
                    "drift": drift,
                    "schwafit_info": schwafit_info,
                    "trade_result": trade_result,
                    "timestamp": timestamp or time.time(),
                    }
                    self.triggers.append(entry)
                        if self.registry_file:
                        self._save()

                            def log_backtest_signal(self, signal_data: Dict[str, Any]) -> None:
                            """Logs a signal from a backtest run, capturing key performance and context indicators."""
                            entry = {}
                            "type": "backtest_signal",
                            "timestamp": signal_data.get("timestamp", time.time()),
                            "asset": signal_data.get("asset"),
                            "mode": signal_data.get("mode"),
                            "hash_id": signal_data.get("hash_id"),
                            "signal_vector": signal_data.get("signal_vector"),
                            "projected_gain": signal_data.get("projected_gain"),
                            # Store the full signal for deep analysis
                            "trade_details": signal_data.get("trade_details"),
                            }
                            self.triggers.append(entry)
                                if self.registry_file:
                                self._save()

                                    def get_best_phase(self, asset: str, window: int = 1000) -> dict:
                                    """Return the phase/drift/tensor config with the highest profit in the last N triggers."""
                                    filtered = [t for t in self.triggers if t["asset"] == asset][-window:]
                                        if not filtered:
                                    return {}
                                    best = max(filtered, key=lambda t: t["trade_result"].get("profit", 0))
                                return best

                                def get_profit_vector()
                                self, asset: str, phase: float = None, drift: float = None, window: int = 1000
                                    ) -> List[float]:
                                    """Return the rolling profit vector for a given asset/phase/drift."""
                                    filtered = [t for t in self.triggers if t["asset"] == asset][-window:]
                                        if phase is not None:
                                        filtered = [t for t in filtered if abs(t["phase"] - phase) < 1e-4]
                                            if drift is not None:
                                            filtered = [t for t in filtered if abs(t["drift"] - drift) < 1e-4]
                                        return [t["trade_result"].get("profit", 0) for t in filtered]

                                            def get_cross_asset_best(self, window: int = 1000) -> dict:
                                            """Return the best asset/phase/drift combo for profit in the last N triggers."""
                                            filtered = self.triggers[-window:]
                                                if not filtered:
                                            return {}
                                            best = max(filtered, key=lambda t: t["trade_result"].get("profit", 0))
                                        return best

                                            def get_last_triggers(self, asset: str, n: int = 10) -> List[Dict[str, Any]]:
                                            """Return the last N triggers for an asset, with all math and trade info."""
                                            filtered = [t for t in self.triggers if t["asset"] == asset]
                                        return filtered[-n:]

                                            def _save(self) -> None:
                                                if self.registry_file:
                                                    with open(self.registry_file, "w") as f:
                                                    json.dump(self.triggers, f, indent=2)

                                                        def _load(self) -> None:
                                                            try:
                                                                with open(self.registry_file, "r") as f:
                                                                self.triggers = json.load(f)
                                                                    except Exception:
                                                                    self.triggers = []


                                                                    # Example usage and testing
                                                                        def main():
                                                                        # You can set the registry path via env var, class method, or constructor
                                                                        # os.environ['SOULPRINT_REGISTRY_PATH'] = 'custom/path/registry.json'
                                                                        # SoulprintRegistry.set_default_registry_path('custom/path/registry.json')
                                                                        # registry = SoulprintRegistry('custom/path/registry.json')
                                                                        registry = SoulprintRegistry()

                                                                        # Example: Register a soulprint from a drift vector
                                                                        test_vector = {}
                                                                        "pair": "BTC/USDC",
                                                                        "entropy": 0.88,
                                                                        "momentum": 0.4,
                                                                        "volatility": 0.19,
                                                                        "temporal_variance": 0.92,
                                                                        }

                                                                        soulprint = registry.register_soulprint()
                                                                        vector =test_vector, strategy_id="momentum_breakout", confidence=0.85
                                                                        )

                                                                        print("🌀 Registered Soulprint: {0}".format(soulprint))

                                                                        # Mark as executed with profit
                                                                        registry.mark_executed(soulprint, profit_result=0.23)

                                                                        # Get registry statistics
                                                                        stats = registry.get_registry_stats()
                                                                        print("📊 Registry Stats: {0}".format(stats))

                                                                        # Find similar patterns
                                                                        similar = registry.get_similar_soulprints(test_vector)
                                                                        print("🔍 Found {0} similar soulprints".format(len(similar)))


                                                                            if __name__ == "__main__":
                                                                            main()
