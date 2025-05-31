"""
RDE_core - Radial Dynamic Engine
===============================

A profit-psychometric mapper for Schwabot: converts live market telemetry
into neuro-biotype-style radar charts and deterministic SHA hash that
feeds the Ferris-Wheel spin selector.

Features:
- Market-aware axis templates with {asset} tokens
- Dynamic asset list support
- Canonical tag enrichment with asset prefix
- 4/8/42 bit map integration
- Real-time price analysis and visualization
"""

from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import yaml

class RDEEngine:
    """RDE_core - generate market-aware radar imprints each Ferris tick."""
    
    def __init__(self, cfg_path: str | Path, *, assets: List[str] | None = None):
        self.cfg_path = Path(cfg_path).expanduser()
        self.cfg = self._load_cfg()
        self.assets = assets or self.cfg.get("assets", ["BTC"])
        self.axis_tpl = self.cfg["axes"]
        self.axes = self._expand_axes_for_assets()
        self.decay = float(self.cfg["scales"].get("decay", 0.8))
        self.lookback = int(self.cfg["scales"].get("lookback_hours", 24))
        self.hist_dir = Path(self.cfg["paths"]["history_dir"]).expanduser()
        self.hist_dir.mkdir(parents=True, exist_ok=True)
        
        # Rolling buffers {set_name: [(ts, vec), ...]}
        self._buffers: Dict[str, List[Tuple[float, np.ndarray]]] = {
            k: [] for k in self.axes
        }
        
        # Enhanced bit mode integration
        self.bit_modes = self.cfg["scales"].get("bit_modes", [4, 8, 42])
        self.current_bit_mode = max(self.bit_modes)  # Default to highest precision
        self._bit_mode_history: List[Tuple[float, int]] = []  # [(timestamp, mode), ...]
        self._mode_switch_threshold = 0.75  # Threshold for mode switching
        
        # Ferris Wheel integration state
        self._last_spin_tag: Optional[str] = None
        self._spin_history: List[Dict] = []
        self._mode_performance: Dict[int, float] = {mode: 0.0 for mode in self.bit_modes}

    def _load_cfg(self) -> Dict:
        """Load and validate YAML configuration."""
        with open(self.cfg_path) as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _expand_axes_for_assets(self) -> Dict[str, List[str]]:
        """Expand {asset} templates for each configured asset."""
        expanded: Dict[str, List[str]] = {}
        for set_name, names in self.axis_tpl.items():
            bucket: List[str] = []
            for name in names:
                if "{asset}" in name:
                    bucket.extend([name.format(asset=a) for a in self.assets])
                else:
                    bucket.append(name)
            expanded[set_name] = bucket
        return expanded

    def update_signals(self, signals: Dict[str, float]) -> None:
        """Inject per-tick telemetry (missing axes default to 0)."""
        ts = datetime.now(timezone.utc).timestamp()
        for set_name, axis_list in self.axes.items():
            vec = np.array([float(signals.get(a, 0.0)) for a in axis_list])
            self._push(set_name, ts, vec)
        
        # Check if bit mode switch is needed
        self._evaluate_bit_mode()

    def _push(self, set_name: str, ts: float, vec: np.ndarray) -> None:
        """Add new signal vector to rolling buffer with decay."""
        buf = self._buffers[set_name]
        buf.append((ts, vec))
        cutoff = ts - self.lookback * 3600
        while buf and buf[0][0] < cutoff:
            buf.pop(0)

    def compute_biotype(self) -> str:
        """Generate canonical spin-ID with asset prefix using current bit mode."""
        concat = b"".join([
            buf[-1][1].tobytes() if buf else b"" 
            for buf in self._buffers.values()
        ])
        
        # Apply bit mode to hash computation
        if self.current_bit_mode == 4:
            # Use first 4 bits of each byte
            masked = bytes(b & 0xF0 for b in concat)
            digest = hashlib.sha1(masked).hexdigest()[:4]
        elif self.current_bit_mode == 8:
            # Use full byte
            digest = hashlib.sha1(concat).hexdigest()[:8]
        else:  # 42-bit mode
            # Use 42 bits (5.25 bytes) of the hash
            digest = hashlib.sha1(concat).hexdigest()[:11]  # 11 hex chars = 44 bits
            
        return f"{self.assets[0]}_{digest}"

    def _evaluate_bit_mode(self) -> None:
        """Evaluate if bit mode switch is needed based on signal stability."""
        if not all(self._buffers.values()):
            return
            
        # Calculate signal stability across all buffers
        stabilities = []
        for buf in self._buffers.values():
            if len(buf) < 2:
                continue
            recent = np.stack([v for _, v in buf[-10:]])
            std = np.std(recent, axis=0).mean()
            stabilities.append(1.0 / (1.0 + std))
            
        if not stabilities:
            return
            
        stability = np.mean(stabilities)
        
        # Determine appropriate bit mode based on stability
        if stability < 0.3:
            new_mode = min(self.bit_modes)  # Use lowest precision for high volatility
        elif stability > 0.7:
            new_mode = max(self.bit_modes)  # Use highest precision for stable signals
        else:
            new_mode = self.bit_modes[len(self.bit_modes)//2]  # Use middle precision
            
        if new_mode != self.current_bit_mode:
            self.set_bit_mode(new_mode)

    def set_bit_mode(self, mode: int) -> None:
        """Set current bit mode and log the switch."""
        if mode not in self.bit_modes:
            raise ValueError(f"Invalid bit mode. Must be one of {self.bit_modes}")
            
        old_mode = self.current_bit_mode
        self.current_bit_mode = mode
        self._bit_mode_history.append((datetime.now(timezone.utc).timestamp(), mode))
        
        # Log mode switch
        if self._last_spin_tag:
            self.log_spin(self._last_spin_tag, mode_switch=(old_mode, mode))

    def _latest_norm(self, set_name: str) -> np.ndarray:
        """Compute normalized EMA of latest signals."""
        buf = self._buffers[set_name]
        if not buf:
            return np.zeros(len(self.axes[set_name]))
            
        weights = np.array([self.decay ** i for i in range(len(buf))[::-1]])
        weights /= weights.sum()
        stacked = np.stack([v for _, v in buf])
        ema = (stacked.T @ weights).T
        
        # Normalize to [0,1]
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        denom = np.where(maxs - mins == 0, 1, maxs - mins)
        return (ema - mins) / denom

    def render_plots(self, tag: str) -> List[Path]:
        """Generate radar plots for each axis group."""
        paths: List[Path] = []
        for set_name, axis_list in self.axes.items():
            values = self._latest_norm(set_name)
            fig = self._radar_figure(values, axis_list, f"{tag} – {set_name}")
            out = self.hist_dir / f"{tag}_{set_name}.png"
            fig.savefig(out, dpi=self.cfg["paths"].get("fig_dpi", 150))
            plt.close(fig)
            paths.append(out)
        return paths

    @staticmethod
    def _radar_figure(values: np.ndarray, labels: List[str], title: str) -> plt.Figure:
        """Create radar plot with given values and labels."""
        N = len(values)
        angles = np.linspace(0, 2 * math.pi, N, endpoint=False).tolist()
        values = values.tolist() + values[:1].tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels([])
        ax.set_title(title, size=13, y=1.1)
        return fig

    def log_spin(self, tag: str, mode_switch: Optional[Tuple[int, int]] = None) -> Path:
        """Log spin metadata and vectors to JSON."""
        meta = {
            "tag": tag,
            "utc": datetime.utcnow().isoformat(),
            "assets": self.assets,
            "bit_mode": self.current_bit_mode,
            "buffers": {k: self._serialise_buffer(v) for k, v in self._buffers.items()},
        }
        
        if mode_switch:
            meta["mode_switch"] = {
                "from": mode_switch[0],
                "to": mode_switch[1],
                "reason": "stability_threshold"
            }
            
        out = self.hist_dir / f"{tag}.json"
        out.write_text(json.dumps(meta, indent=2))
        self._last_spin_tag = tag
        self._spin_history.append(meta)
        return out

    @staticmethod
    def _serialise_buffer(buf: List[Tuple[float, np.ndarray]]) -> List[List]:
        """Convert buffer to JSON-serializable format."""
        return [[ts, v.tolist()] for ts, v in buf]

    def get_mode_performance(self) -> Dict[int, float]:
        """Get performance metrics for each bit mode."""
        return self._mode_performance.copy()

    def update_mode_performance(self, mode: int, performance: float) -> None:
        """Update performance metrics for a bit mode."""
        if mode not in self.bit_modes:
            raise ValueError(f"Invalid bit mode: {mode}")
        self._mode_performance[mode] = performance

    def get_spin_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get recent spin history."""
        if limit is None:
            return self._spin_history
        return self._spin_history[-limit:]

    def get_bit_mode_history(self) -> List[Tuple[float, int]]:
        """Get history of bit mode switches."""
        return self._bit_mode_history.copy()

# Example YAML configuration:
"""
axes:
  price_metrics:
    - "{asset}_price_delta"
    - "{asset}_volatility"
    - "{asset}_volume_ema"
    - "{asset}_spread_bp"
  market_state:
    - "{asset}_sentiment_bull"
    - "{asset}_sentiment_bear"
    - "{asset}_exec_latency"
    - "{asset}_cognitive_score"
  profit_metrics:
    - "{asset}_hold_roi"
    - "{asset}_swap_roi"
    - "{asset}_hedge_roi"
assets: [BTC, ETH, XRP]
scales:
  lookback_hours: 24
  decay: 0.85
paths:
  history_dir: "~/Schwabot/init/spin_history"
  fig_dpi: 150
""" 