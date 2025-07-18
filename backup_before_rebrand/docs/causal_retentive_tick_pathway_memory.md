# Causal Retentive Tick Pathway Memory (CRTPM)

CRTPM is a critical component of Schwabot's intelligence, focusing on **event-wise, hash-to-pathway microstructure** of market data. It delves into the actual tick-by-tick memory, actions, and cause-effect chains to build the backbone of Schwabot's adaptive memory.

## Role and Data Grain

- **Micro tick-sequence, causal-chain, memory mapping:** Records pathways of events, hashed and weighted by their impact.
- **Operates over paths:** Focuses on individual ticks and trades, recursively tracking events leading to profit or loss.
- **Goal:** To understand **Sequence Causality** – what exact steps led to a win or loss, and why.
- **Memory:** Path-stacked and causally-pruned, ensuring only relevant information is retained.

## Mathematical Structure

Given ticks \(t_0, t_1, ..., t_n\), each tick carries a pathway-tag:

\(T_i = \{event_j, w_j, p_j\}_{j=0}^i\)

Where:

- \(event_j\) = order fill, patch trigger, regime switch, etc.
- \(w_j\) = weight (derived from \(\Delta p\)-importance, anomaly, entropy)
- \(p_j\) = mid-price at event

The retention function, \(\text{Retain}(T_i)\), determines which pathways are kept:

\(\text{Retain}(T_i) = \begin{cases} 1 & \sum_j w_j > \kappa \land \text{causal chain triggers profit anomaly} \\ 0 & \text{otherwise} \end{cases}\)

Where \(\kappa\) is an adaptively-tuned threshold. This system prioritizes retaining sequences that significantly alter the profit pathway.

Schwabot also uses a new retention math: "Causal Impact-Weighted Memory," where retention is approximately equal to functional information gain:

\(\text{Retention}(t_{0:n}) = \sum_{k \in \text{Seq}(t_{0:n})} I[\Delta\psi_k > \lambda] \cdot \log \frac{P(\text{profit} \mid k)}{P(\text{profit} \mid \neg k)}\)

Where:
- \(k\) = unique event-path sequence
- \(I\) = indicator for significant impact
- \(\lambda\) = self-tuning threshold
- \(P(\text{profit} \mid k)\) = empirical conditional profit probability for path \(k\)

## Integration into Unified Chrono-Causal Layer

CRTPM works in conjunction with CRWM (Chrono Resonant Weather Mapping) to form a **Unified Chrono-Causal Layer**. The hash of each CRTPM pathway, \(h_k = \text{SHA256}(\text{Path}_k)\), is cross-indexed with CRWM's "weather keys." This allows Schwabot to recall specific micro-paths based on macro market conditions, providing a comprehensive historical context for decision-making and self-improvement. 