import json
import sys
import types
import math
from dataclasses import dataclass
import pytest

# Provide minimal numpy and corridor engine stubs for testing environments
numpy_stub = types.SimpleNamespace(
    mean=lambda x: sum(x)/len(x) if x else 0.0,
    std=lambda x: math.sqrt(sum((i - sum(x)/len(x)) ** 2 for i in x) / len(x)) if x else 0.0,
    zeros=lambda shape: [0.0] * shape if isinstance(shape, int) else [[0.0]*shape[1] for _ in range(shape[0])],
    exp=math.exp,
    sin=math.sin,
    array=lambda x: x,
    linalg=types.SimpleNamespace(norm=lambda v: math.sqrt(sum(i*i for i in v))),
    random=types.SimpleNamespace(
        choice=lambda seq, p=None: seq[0],
        normal=lambda loc=0.0, scale=1.0, size=None: 0.0 if size is None else [0.0]*size,
        uniform=lambda low=0.0, high=1.0, size=None: 0.0 if size is None else [0.0]*size,
        seed=lambda x: None,
    ),
)
sys.modules.setdefault("numpy", numpy_stub)
yaml_stub = types.SimpleNamespace(safe_load=lambda *a, **k: {})
sys.modules.setdefault("yaml", yaml_stub)

psutil_stub = types.SimpleNamespace(cpu_percent=lambda interval=0.1: 0.0)
sys.modules.setdefault("psutil", psutil_stub)

for mod in [
    "core.hash_recollection",
    "core.entropy_tracker",
    "core.bit_operations",
    "core.pattern_utils",
    "core.strange_loop_detector",
    "core.risk_engine",
]:
    m = types.ModuleType(mod)
    # Add dummy classes/vars required by core.__init__
    attrs = {
        "HashRecollectionSystem": type("HashRecollectionSystem", (), {}),
        "HashEntry": type("HashEntry", (), {}),
        "EntropyTracker": type("EntropyTracker", (), {}),
        "EntropyState": type("EntropyState", (), {}),
        "BitOperations": type("BitOperations", (), {}),
        "PhaseState": type("PhaseState", (), {}),
        "PatternUtils": type("PatternUtils", (), {}),
        "PatternMatch": type("PatternMatch", (), {}),
        "ENTRY_KEYS": [],
        "EXIT_KEYS": [],
        "StrangeLoopDetector": type("StrangeLoopDetector", (), {}),
        "EchoPattern": type("EchoPattern", (), {}),
        "RiskEngine": type("RiskEngine", (), {}),
        "PositionSignal": type("PositionSignal", (), {}),
        "RiskMetrics": type("RiskMetrics", (), {}),
    }
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(mod, m)

corridor_stub = types.ModuleType("future_corridor_engine")

class ExecutionPath(types.SimpleNamespace):
    CPU_SYNC = "cpu_sync"
    CPU_ASYNC = "cpu_async"
    GPU_ASYNC = "gpu_async"


class ProfitTier(types.SimpleNamespace):
    DISCARD = 0


@dataclass
class CorridorState:
    price: float
    duration: float
    volatility: float
    timestamp: object
    hash_signature: str


class FutureCorridorEngine:
    def __init__(self, *args, **kwargs):
        pass

    def recursive_intent_loop(self, *args, **kwargs):
        return {
            "dispatch_path": ExecutionPath.CPU_SYNC,
            "dispatch_confidence": 1.0,
            "profit_tier": "DISCARD",
            "ecmp_magnitude": 0.0,
            "ecmp_direction": [0, 0, 0],
            "next_target_price": 0.0,
            "next_target_volatility": 0.0,
            "activation_mode": "STANDBY",
            "anomaly_level": 0.0,
            "feedback_strength": {"jumbo": 0, "ghost": 0, "thermal": 0},
        }

    def update_corridor_memory(self, *args, **kwargs):
        pass

    def get_performance_metrics(self):
        return {}


corridor_stub.FutureCorridorEngine = FutureCorridorEngine
corridor_stub.CorridorState = CorridorState
corridor_stub.ExecutionPath = ExecutionPath
corridor_stub.ProfitTier = ProfitTier

sys.modules.setdefault("core.future_corridor_engine", corridor_stub)

from core.fault_bus import FaultBus, FaultBusEvent, FaultType
import psutil


def test_fault_bus_event_creation():
    event = FaultBusEvent(
        tick=1,
        module="unit_test",
        type=FaultType.PROFIT_LOW,
        severity=0.8,
        metadata={"price": 100},
        profit_context=50.0,
        sha_signature="abc"
    )
    assert event.type is FaultType.PROFIT_LOW
    assert event.age >= 0


def test_profit_context_and_path_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=0.1: 0.0)
    bus = FaultBus(log_path=str(tmp_path))

    event = FaultBusEvent(
        tick=0,
        module="tester",
        type=FaultType.PROFIT_LOW,
        severity=0.8,
        profit_context=50.0,
    )

    bus.push(event)
    metrics = bus._calculate_path_selection_score(event)
    assert 0.0 <= metrics.final_score <= 1.0
    # expected score computed from weights when cpu load is zero
    assert abs(metrics.final_score - 0.704) < 1e-3
    assert metrics.selected_path == "async"

    bus.memory_log.append(event)
    bus.update_profit_context(5.0, tick=1)
    corr_json = bus.export_correlation_matrix()
    data = json.loads(corr_json)
    assert data
    assert data[0]["fault_type"] == FaultType.PROFIT_LOW.value
    assert bus.correlation_matrix.correlations[FaultType.PROFIT_LOW].occurrence_count == 2 