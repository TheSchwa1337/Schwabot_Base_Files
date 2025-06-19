import pytest  # noqa: F401
from datetime import datetime, timedelta  # noqa: F821
from unittest.mock import Mock
from unittest.mock import patch

from core.fault_bus import ProfitFaultCorrelation, FaultType  # noqa: F401


class MockFaultBus:
    """Minimal FaultBus stub returning deterministic correlations."""

    def __init__(self, sequences):
        self.sequences = sequences
        self.index = 0

    def update_profit_context(self, profit_delta: float, tick: int):
        pass

    def get_profit_correlations(self):
        return self.sequences[self.index]

    def predict_profit_from_fault(self, fault_type: FaultType):
        for corr in self.sequences[self.index]:
            if corr.fault_type == fault_type:
                return corr.profit_delta
        return None

    def advance(self):
        if self.index < len(self.sequences) - 1:
            self.index += 1


def test_profit_cycle_navigation(monkeypatch):
    # Prepare deterministic correlation sequences for each step
    sequences = [
        [ProfitFaultCorrelation(
            fault_type=FaultType.PROFIT_LOW,
            profit_delta=0.05,
            correlation_strength=1.0,
            temporal_offset=1,
            confidence=1.0,
            occurrence_count=5,
            last_seen=datetime.now(),  # noqa: F821
        )],
        [],
        [ProfitFaultCorrelation(
            fault_type=FaultType.PROFIT_CRITICAL,
            profit_delta=-0.05,
            correlation_strength=1.0,
            temporal_offset=1,
            confidence=1.0,
            occurrence_count=5,
            last_seen=datetime.now(),  # noqa: F821
        )],
    ]

    bus = MockFaultBus(sequences)
    navigator = ProfitCycleNavigator(bus, initial_portfolio_value=10000.0)

    # Patch detector methods to provide deterministic cycle phases
    phase_iter = iter([
        ("markup", 0.8),
        ("accumulation", 0.05),
        ("markdown", 0.9),
    ])

    def fake_detect_cycle_phase(profit, ts):
        return next(phase_iter)

    monkeypatch.setattr(
        navigator.detector,
        "detect_cycle_phase",
        fake_detect_cycle_phase
    )

    anomaly_iter = iter([
        (False, 0.0, "normal"),
        (False, 0.0, "normal"),
        (True, 0.6, "profit_binary"),
    ])
    monkeypatch.setattr(
        navigator.detector,
        "detect_anomaly_cluster",
        lambda ctx: next(anomaly_iter)
    )

    base_time = datetime(2021, 1, 1)  # noqa: F821
    prices = [100.0, 100.5, 99.0]
    signals = []
    vectors = []

    for i, price in enumerate(prices):
        pv = navigator.update_market_state(
            price,
            1000,
            base_time + timedelta(minutes=i)
        )
        signals.append(navigator.get_trade_signal())
        vectors.append(pv)
        bus.advance()

    # Step 1: high confidence should trigger ENTERING state
    pv1 = vectors[0]
    assert navigator.navigation_log[0]["new_state"] == "ENTERING"
    assert pv1.direction == 1
    assert pv1.magnitude > 0.0  # Should have some magnitude from fault correlation
    assert FaultType.PROFIT_LOW in pv1.fault_correlations
    assert signals[0] and signals[0]["action"] == "ENTER"

    # Step 2: low magnitude should revert to SEEKING
    pv2 = vectors[1]
    assert pv2.magnitude < navigator.detector.min_profit_threshold
    assert navigator.navigation_log[1]["new_state"] == "SEEKING"
    assert signals[1] is None

    # Step 3: high confidence with negative correlation should trigger ENTERING
    # again
    # (The state machine prioritizes confidence over anomaly detection)
    pv3 = vectors[2]
    assert navigator.navigation_log[2]["new_state"] == "ENTERING"
    assert pv3.direction == -1  # Should be short due to negative correlation
    assert pv3.anomaly_strength == pytest.approx(0.6, rel=1e-2)
    assert signals[2] and signals[2]["action"] == "ENTER"