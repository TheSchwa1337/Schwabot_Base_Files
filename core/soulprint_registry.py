from typing import Dict, Optional, List


@dataclass
class SoulprintEntry:
    soulprint: str
    timestamp: str
    vector: Dict[str, float]  # DriftVector serialized
    strategy_id: str
    confidence: float
    is_executed: bool = False
    profit_result: Optional[float] = None
    replayable: bool = True


class SoulprintRegistry:
    _default_registry_path
    = os.environ.get("SOULPRINT_REGISTRY_PATH", "data/soulprint_registry.json")

    @classmethod
    def set_default_registry_path(cls, path: str):
        """Set a new default path for the soulprint registry."""
        cls._default_registry_path = path

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the SoulprintRegistry.
        Args:
registry_path: Optional custom path for the registry file. If not provided, uses env var or default.
        """
        self.registry_path = registry_path or self._default_registry_path
        self.registry: Dict[str, SoulprintEntry] = {}
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                self.registry = {
                    k: SoulprintEntry(**v) for k, v in data.items()
                }
        else:
            self.registry = {}

    def _save_registry(self):
        parent_dir = os.path.dirname(self.registry_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.registry.items()}, f, indent=2)

def register_soulprint(self, vector: Dict[str, float], strategy_id: str, confidence: float) -> str:
        raw_components = [
            vector.get('pair', ''),
            str(vector.get('entropy', 0)),
            str(vector.get('momentum', 0)),
            str(vector.get('volatility', 0)),
            str(vector.get('temporal_variance', 0))
        ]
        hash_input = "|".join(raw_components)
        soulprint = hashlib.sha256(hash_input.encode()).hexdigest()

        entry = SoulprintEntry(
            soulprint=soulprint,
            timestamp=datetime.now(timezone.utc).isoformat(),
            vector=vector,
            strategy_id=strategy_id,
            confidence=confidence
        )

        self.registry[soulprint] = entry
        self._save_registry()
        return soulprint

    def mark_executed(self, soulprint: str, profit_result: Optional[float] = None):
        if soulprint in self.registry:
            self.registry[soulprint].is_executed = True
            self.registry[soulprint].profit_result = profit_result
            self._save_registry()

    def get_entry(self, soulprint: str) -> Optional[SoulprintEntry]:
        return self.registry.get(soulprint)

    def find_replayable(self, min_confidence: float = 0.8) -> List[SoulprintEntry]:
        return [
            e for e in self.registry.values()
            if e.replayable and e.confidence >= min_confidence and not e.is_executed
        ]

    def all_entries(self) -> List[SoulprintEntry]:
        return list(self.registry.values())

    def get_similar_soulprints(self, target_vector: Dict[str, float], threshold: float = 0.85)
    -> List[SoulprintEntry]:
        """
        Find soulprints with similar vector characteristics
        Uses simple Euclidean distance for similarity
        """
        similar_entries = []

        for entry in self.registry.values():
            # Calculate similarity based on key vector components
            target_components = [
                target_vector.get('entropy', 0),
                target_vector.get('momentum', 0),
                target_vector.get('volatility', 0)
            ]

            entry_components = [
                entry.vector.get('entropy', 0),
                entry.vector.get('momentum', 0),
                entry.vector.get('volatility', 0)
            ]

            # Simple Euclidean distance
            distance = sum((a - b) ** 2 for a, b in zip(target_components, entry_components)) ** 0.5
            similarity = 1.0 / (1.0 + distance)  # Convert to similarity score

            if similarity >= threshold:
                similar_entries.append(entry)

        return similar_entries

    def get_profitable_patterns(self, min_profit: float = 0.01) -> List[SoulprintEntry]:
        """
        Find soulprints that resulted in profitable trades
        """
        return [
            e for e in self.registry.values()
            if e.is_executed and e.profit_result and e.profit_result >= min_profit
        ]

    def get_registry_stats(self) -> Dict[str, any]:
        """
        Get statistics about the soulprint registry
        """
        total_entries = len(self.registry)
        executed_entries = sum(1 for e in self.registry.values() if e.is_executed)
        replayable_entries = sum(1 for e in self.registry.values() if e.replayable)

        profitable_entries = [
            e for e in self.registry.values()
            if e.is_executed and e.profit_result and e.profit_result > 0
        ]

        avg_confidence = sum(e.confidence for e in self.registry.values())
    / total_entries if total_entries > 0 else 0

        return {
            'total_entries': total_entries,
            'executed_entries': executed_entries,
            'replayable_entries': replayable_entries,
            'profitable_entries': len(profitable_entries),
            'avg_confidence': avg_confidence,
            'execution_rate': executed_entries / total_entries if total_entries > 0 else 0,
            'profit_rate': len(profitable_entries) / executed_entries if executed_entries > 0 else 0
        }


# Example usage and testing
def main():
    # You can set the registry path via env var, class method, or constructor
    # os.environ['SOULPRINT_REGISTRY_PATH'] = 'custom/path/registry.json'
    # SoulprintRegistry.set_default_registry_path('custom/path/registry.json')
    # registry = SoulprintRegistry('custom/path/registry.json')
    registry = SoulprintRegistry()

    # Example: Register a soulprint from a drift vector
    test_vector = {
        'pair': 'BTC/USDC',
        'entropy': 0.88,
        'momentum': 0.04,
        'volatility': 0.19,
        'temporal_variance': 0.92
    }

    soulprint = registry.register_soulprint(
        vector=test_vector,
        strategy_id='momentum_breakout',
        confidence=0.85
    )

    print(f"🌀 Registered Soulprint: {soulprint}")

    # Mark as executed with profit
    registry.mark_executed(soulprint, profit_result=0.023)

    # Get registry statistics
    stats = registry.get_registry_stats()
    print(f"📊 Registry Stats: {stats}")

    # Find similar patterns
    similar = registry.get_similar_soulprints(test_vector)
    print(f"🔍 Found {len(similar)} similar soulprints")


if __name__ == "__main__":
    main()