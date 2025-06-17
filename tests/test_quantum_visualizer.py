import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time

from quantum_visualizer import PanicDriftVisualizer, plot_entropy_waveform


def test_add_data_point_limit():
    viz = PanicDriftVisualizer()
    for i in range(1005):
        viz.add_data_point(i, float(i), float(i))
    assert len(viz.time_points) == 1000
    assert len(viz.entropy_values) == 1000
    assert len(viz.coherence_values) == 1000
    assert viz.time_points[0] == 5


def test_render_calls_show(monkeypatch):
    viz = PanicDriftVisualizer()
    viz.add_data_point(time.time(), 1.0, 0.5)
    called = {}

    def fake_show():
        called["show"] = True
    monkeypatch.setattr(plt, "show", fake_show)

    viz.render()
    assert called.get("show") is True


def test_plot_entropy_waveform_executes(monkeypatch):
    data = np.random.normal(0, 1, 50).tolist()
    called = {}

    def fake_show():
        called["show"] = True
    monkeypatch.setattr(plt, "show", fake_show)

    plot_entropy_waveform(data)
    assert called.get("show") is True 