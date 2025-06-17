import pytest
np = pytest.importorskip("numpy")
import hashlib
from datetime import datetime

from core.future_corridor_engine import (
    FutureCorridorEngine,
    CorridorState,
    ExecutionPath,
    ProfitTier,
)


@pytest.fixture
def engine():
    """Return a new engine instance with deterministic RNG."""
    np.random.seed(0)
    return FutureCorridorEngine()


def _make_state(price: float, vol: float) -> tuple[CorridorState, str]:
    """Create a corridor state and corresponding hash."""
    market_state = f"{price:.2f}_{vol:.4f}"
    market_hash = hashlib.sha256(market_state.encode()).hexdigest()
    state = CorridorState(
        price=price,
        duration=1.0,
        volatility=vol,
        timestamp=datetime.now(),
        hash_signature=market_hash,
    )
    return state, market_hash


def test_profit_tier_classification(engine):
    """Tiers should reflect volatility and profit context."""
    high_state, _ = _make_state(100.0, 0.9)
    low_state, _ = _make_state(100.0, 0.1)

    assert engine.calculate_profit_tier(high_state, 80.0) is ProfitTier.HIGHENTRY
    assert engine.calculate_profit_tier(low_state, 10.0) is ProfitTier.DISCARD


def test_probabilistic_dispatch_vector(engine):
    """Dispatch probabilities should be valid and ordered by score."""
    probs = engine.probabilistic_dispatch_vector(
        execution_time=0.1,
        entropy=0.1,
        profit_tier=ProfitTier.SCOUT,
        t=0.0,
    )

    total = probs.cpu_sync + probs.cpu_async + probs.gpu_async
    assert pytest.approx(total, rel=1e-6) == 1.0
    assert probs.cpu_sync >= probs.cpu_async
    assert probs.cpu_sync >= probs.gpu_async
    assert probs.selected_path in ExecutionPath


def test_recursive_intent_loop_simulation(engine):
    """Simulate several corridor states and ensure valid outputs."""
    prices = [100.0, 101.5, 103.0]
    volumes = [1000.0, 1050.0, 1100.0]
    vols = [0.8, 0.85, 0.9]

    results = []
    for i in range(3):
        engine.update_corridor_memory(prices[i], volumes[i], vols[i])
        state, market_hash = _make_state(prices[i], vols[i])
        market_data = {
            "price_series": prices[: i + 1],
            "volume_series": volumes[: i + 1],
            "volatility_series": vols[: i + 1],
        }
        res = engine.recursive_intent_loop(
            t=i * 0.1,
            market_hash=market_hash,
            corridor_state=state,
            profit_context=80.0,
            execution_time=0.5,
            entropy=0.2,
            market_data=market_data,
        )
        results.append(res)

    assert len(engine.dispatch_history) == 3
    for res in results:
        assert res["dispatch_path"] in [e.value for e in ExecutionPath]
        assert 0.0 <= res["dispatch_confidence"] <= 1.0
        assert res["profit_tier"] == ProfitTier.HIGHENTRY.name 