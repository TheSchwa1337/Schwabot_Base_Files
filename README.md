````markdown
# Core Mathematical Library for Quantitative Trading  
**_Schwabot Base Stack Update — Snapshot Sync: 6/1/2025_**  
**Version: `v0.38 → v0.5 (pre-integration)`**  
Path: `~/Schwabot/init/lib/math/`

---

## 📌 Overview

This mathematical framework underpins the **Schwabot recursive trading system**, integrating real-time market signals, AI-verified strategy logic, and memory-safe profit pipelines. The library is modular, hash-aware, and built for advanced recursive systems such as those running on Raspberry Pi clusters or GPU-accelerated nodes.  

> _This core library feeds directly into the Ferris Wheel cycle, orchestrating profit logic across short/mid/long-term hash-reactive trade loops._

---

## 🚀 Feature Roadmap (As of 6/1/2025)

### ✅ Core Features (Stable: v0.1)
- 📈 Price/Volume Delta Analysis
- 🔁 Recursive Hash-Driven Strategy Engine
- 📊 Technical Indicators (Z-score, EMA, VPT, BB)
- 🛡️ Risk Controls: Stop-Loss, Sharpe Ratio
- 🧮 Drift/Entropy Tracking
- ⚖️ Adaptive Profit Coefficients + Tick-Based Filters

### 🔁 Active Integration (Pre-Stable: v0.2x → v0.5)
- 📍 VWAP, ATR, RSI, Keltner Channels
- 🧠 Kelly Criterion for Adaptive Sizing
- 🔗 Pairwise Correlation / Z-Score Divergence
- ⛓️ Ornstein-Uhlenbeck Mean Reversion
- 🧬 Memory Kernel w/ Decay Vector Tracking
- 💹 Smart Money Anchor Zones + Order Book Footprint Integration

### 🔬 Experimental Additions (Quantum Mode)
- 🔺 Klein Bottle Fractal Anchors for recursive drift zones
- 📉 Velocity-aware Liquidity Vacuum Detection
- 🔄 Cycle-phase Detection for Trade Re-entry
- 🧠 AI-Inferred Recursive Zone Multipliers

---

## 🧠 Live Class Access

```python
from schwabot_math.core_v1 import CoreMathLib          # Stable v0.1 Logic
from schwabot_math.core_v2x import CoreMathLibV2       # Extended Metrics (RSI/VWAP)
from schwabot_math.klein_logic import KleinDecayField  # Experimental (v0.5 quantum kernel)
````

---

## 🧪 Test Entry Points

```bash
python tests/test_mathlib.py        # 🧪 Legacy Core: Z-Score, Drift, BB
python tests/test_mathlib_v2.py     # 🧪 Extended Indicators: VWAP, RSI, ATR
python tests/test_klein.py          # 🧪 Drift Decay Fractal Band Mapping
```

---

## 🧰 Library Features

### 📚 Core Mathematical Components (v0.1)

* `calc_price_returns(prices)`
* `hash_sequence_entropy(data_stream)`
* `volume_strategy_weights(volumes)`
* `apply_profit_threshold(signal, coef=0.8)`
* `drift_vector(trailing_window)`

### 📈 Technical Indicators

* `bollinger_bands(prices, window=20)`
* `z_score(series)`
* `momentum_metric(data)`
* `ema_crossover_detector(prices)`
* `volume_price_trend(close, volume)`

### 🧮 Risk and Allocation

* `kelly_fractional_size(win_rate, payoff_ratio)`
* `risk_parity_weights(cov_matrix)`
* `adaptive_stop_loss(current_price, volatility)`
* `sharpe_ratio(returns, risk_free_rate=0.01)`

### 🌀 Advanced Simulation Tools

* `simulate_ou_process(mu, sigma, theta, x0, n)`
* `correlation_matrix(assets_matrix)`
* `pair_trade_spread_zscore(asset1, asset2)`
* `memory_kernel(decay_rate, input_series)`
* `keltner_channels(high, low, close, atr_len=10)`

---

## 💻 CLI Entry (Schwabot Integration)

```bash
python cli/generate_strategy.py --strategy drift_zone --profile BTCUSDC
python cli/hash_profiler.py --assets ETH BTC XRP --window 64
python cli/recursive_entry.py --cycle_type ferris --trigger_hash auto
```

These tools invoke live asset data (via CCXT, Coinbase), simulate multi-layer entry/exit logic, and bind hash triggers to trading engines inside the main `Schwabot` recursive loop.

---

## 📦 Dependencies

* `numpy` ≥ 1.24
* `pandas` ≥ 1.5
* `matplotlib` ≥ 3.8
* `scipy` ≥ 1.11
* `pywt` for memory compression and signal wavelets
* `ccxt` for live trading (optional)

---

## 🔐 Memory Structure & File Tree (Snapshot: `~/Schwabot/init/lib/math/`)

```
math/
├── __init__.py
├── core_v1.py                    # ✅ Base functions
├── core_v2x.py                   # 🔁 Advanced logic
├── klein_logic.py                # 🧬 Recursive decay kernel
├── profit_entropy_map.py         # 🧠 Entropy bound logics
├── volume_alloc.py               # 💹 Weighting by flow
├── correlation_matrix_tools.py   # 🔗 Correlation-based filters
├── strategy_selectors.py         # 🧭 Strategy activation by hash logic
├── memory_decay_utils.py         # ⏳ Exponential decay & memory kernel
```

---

## 📈 Visual Tools (Dev Tools for Cursor or Dashboard UI)

* `plot_band_decay()`
* `render_drift_map()`
* `overlay_liquidity_vacuum(prices)`
* `zone_profits_overlay()`

---

## ⚖️ License

Liscense.txt © Schwabot_base_files (Schwabot @ Open Source MIT) v0.5 — Schwa/Nexus Recursive Stack

---

## 🛠️ Contributing

Pull requests welcome. Focus areas:

* Latency-optimized indicators
* Multi-threaded batch simulation for hash re-entry triggers
* Lightweight memory-safe vectorization for ARM

---

## 🧠 Recommended Next Targets (Schwabot v0.5+)

* 🔄 Integrate this library with `profit_cycle_allocator.py`
* 📡 Link into `matrix_fault_resolver.py` for zone-level corrections
* 🧬 Quantize drift entropy into Zygot memory core
* ⏳ Run memory-weighted backtest overlay on 16 dip + 3 peak long-curve patterns
* 🔃 Build `fractal_memory_binder.py` to match entropy shifts with recursive tick shifts

---

*This README now reflects the live core state of Schwabot’s quant engine as of June 1, 2025.*

For future reference, this update is signed:
`Update ID: 6-1-25_SCHWABOT_MATHLIB_v0.5-SNAPSHOT`

```
```
