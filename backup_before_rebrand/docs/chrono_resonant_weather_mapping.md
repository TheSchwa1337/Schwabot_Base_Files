# Chrono Resonant Weather Mapping (CRWM)

CRWM is a core component of Schwabot's advanced temporal analysis, focusing on **field-level time-resonance, macro-patterns, harmonics, gradients, and phase shifts** in market "weather."

## Role and Data Grain

- **Macro time-phase, gradient, resonance mapping:** Utilizes operations like Fourier, Wavelet, Nabla (∇), and Laplacian (Δ²) on price "weather."
- **Operates over windows:** Analyzes data across various timescales such as 1-hour, 4-hour, 1-day, 1-week, etc.
- **Goal:** To provide **Field Awareness** – understanding the overall market flow, wave patterns, and current regime.
- **Memory:** Windowed and phase-anchored.

## Mathematical Basis

For every time window \(\tau\), CRWM computes vector signatures \(W_{\tau}\) that represent trend, volatility, and harmonics:

- **Price Gradient (Nabla):** \(\nabla p = \frac{dp}{dt}\)
- **Laplacian of Price:** \(\Delta^2 p = \frac{d^2p}{dt^2}\)
- **Fourier Transform:** \(F(p) = \text{FFT}(p)\)
- **Wavelet Transform:** \(W(p) = \text{Wavelet}(p)\)

These operations yield a vector signature \(W_{\tau}\) (e.g., [trend, volatility, harmonics]) for each rolling macro window.

## Integration into Unified Chrono-Causal Layer

CRWM works in conjunction with CRTPM (Causal Retentive Tick Pathway Memory) to form a **Unified Chrono-Causal Layer**. This layer uses a Dual-Index Recall Matrix to cross-index macro-level market weather with micro-level causal event pathways.

The hash of each CRWM signature, \(h_{\tau} = \text{SHA256}(W_{\tau})\), serves as a "weather key" for looking up relevant CRTPM pathways. This allows Schwabot to recall, for example, "paths that succeeded in a bullish-to-sideways gradient," providing comprehensive context for decision-making. 