# Unified Chrono-Causal Layer

Schwabot's intelligence is powered by a **Unified Chrono-Causal Layer** that seamlessly integrates macro-level market "weather" (from CRWM) with micro-level causal event pathways (from CRTPM). This layer is crucial for advanced recall, decision-making, and the bot's self-improvement capabilities.

## Dual-Index Recall Matrix

This matrix provides a powerful mechanism for cross-indexing and retrieving historical market states and trading outcomes based on both temporal and causal characteristics.

### Time-Phase Index (from CRWM)

Represents the resonance/gradient fingerprint for each rolling macro window. It captures the overall "weather" of the market over various timescales.

\(\text{Weather}_{\tau} = \text{CRWM}(p_{t-\tau:t}) \quad \forall \tau \in \{1h, 4h, 1d, 1w, \dots \}\)

### Pathway Event Index (from CRTPM)

Represents a detailed tick-sequence of events, ending in a trade or profit/loss inflection. Each path is a causal chain of actions and market responses.

\(\text{Path}_k = \{ \text{tick}_j, \text{event}_j, w_j, \psi_j, h_j \}_{j=1}^{n_k}\)

### Cross-Indexing

Every `Path_k` is tagged with the `Weather` snapshot at its start and end, forming a composite cross-key:

\((\text{Weather}_{\tau, \text{start}}, \text{Weather}_{\tau, \text{end}}, \text{Path}_k)\)

This allows for highly contextualized recall, enabling Schwabot to instantly retrieve, for example, "paths that succeeded in a bullish-to-sideways gradient" or "what micro-paths failed during high-entropy turbulence."

## Mathematical Unification

The CRWM and CRTPM components are mathematically unified through hashing and a comprehensive scoring mechanism:

### CRWM Vector Signatures

For every window \(\tau\), CRWM computes vector signatures \(W_{\tau}\) (representing trend, volatility, harmonics) using operations like \(\nabla p\), \(\Delta^2 p\), \(F(p)\), and \(W(p)\).

Hash of CRWM signature: \(h_{\tau} = \text{SHA256}(W_{\tau})\)

### CRTPM Pathway Scores

For every path \(k\), CRTPM computes a \(\psi_k\) (Psi-score) which aggregates weighted price changes and impact indicators:

\(\psi_k = \sum_j w_j \cdot \Delta p_j \cdot I_{\text{impact}}\)

Hash of CRTPM pathway: \(h_k = \text{SHA256}(\text{Path}_k)\)

This dual hashing ensures that every path can be looked up by its time-window "weather" or by its specific event hash, creating a rich, interconnected memory for the bot.

## Recall Logic: Multi-Timescale Retrieval

When making a trade decision or a SERC/SECR₂ patch proposal, the system performs the following:

1.  **Determine Current CRWM Signature:** Calculates \(W_{\tau}\) for all active windows.
2.  **Lookup CRTPM Paths:** Retrieves all CRTPM paths whose weather keys match a neighborhood (\(\pm\epsilon\)) of the current \(W_{\tau}\).
3.  **Aggregate and Score:** The retrieved paths are aggregated and scored based on profit/loss, event-chain similarity, and anomaly incidence.
4.  **Decision-Making:** The trade or patch decision is made based on a weighted blend of the current CRWM vector and the causal-path outcomes from CRTPM. 

## Organizational Strategy for Data and Recall

- **Partition Memory:** Data is organized by both time-phase (CRWM) and event-path (CRTPM).
- **Comprehensive Tagging:** Each SERC/SECR₂ decision, trade, patch, and recall event is tagged with both a `weather_hash` and a `path_hash`, ensuring perfect replayability and comprehensive post-mortem analysis.
- **Self-Optimizing Memory:** The system self-tunes memory by decaying/pruning the oldest windows and event-paths that do not contribute to profit navigation or repeat in significance (via usage frequency and impact weighting).
- **Auditable Proposals:** Patch proposals reference both hashes, allowing the UI and audit systems to answer "Why did you patch now?" with full chrono-causal context. 