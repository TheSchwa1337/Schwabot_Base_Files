# Mathematical Content Archive

## core/auto_scaler.py

**Mathematical Foundation:**

```
scale_factor = base_scale * (1 + confidence_multiplier + profit_multiplier)

Where:
- confidence_multiplier = max(0, (Ξ - threshold) * confidence_weight)
- profit_multiplier = projected_profit * profit_weight
- Result is clamped to [min_scale, max_scale] range
```

**Risk Management:**
- MAX_POSITION_RISK = 0.02  # 2% of portfolio per position
- MIN_POSITION_SIZE = 0.001  # Minimum position size

**Function:**
- `scale_position(confidence, projected_profit, ...)` calculates position scale factor based on confidence and profit.
- `calculate_position_size(base_position, confidence, projected_profit, account_balance, ...)` applies risk management constraints.

---

## core/demo_entry_simulator.py

**Mathematical/Simulation Concepts:**
- Integrates with vector validator and matrix allocator
- Entry simulation uses confidence, ghost_signal_strength, entropy_level, volume_ratio, and matrix_performance
- Entry strategies include: ghost_signal, volume_spike, entropy_low, fractal_pattern, hash_confidence, tick_delta, matrix_weight, combined_strategy
- Market condition generators: bull_market, bear_market, sideways, high_volatility, low_volume

**Key Data Fields:**
- confidence: float
- ghost_signal_strength: float
- entropy_level: float
- volume_ratio: float
- matrix_performance: Dict[str, float]

**Analysis:**
- EntryAnalysis includes: success_rate, average_confidence, average_ghost_signal, average_entropy, strategy_performance, matrix_performance, market_condition_analysis

**Simulation Logic:**
- Simulates trade entries with various strategies and market conditions
- Calculates success probability based on entry data, validation, allocation, and market conditions
- Analyzes results for success rate and performance metrics

---

## core/cursor_math_integration.py

*No mathematical content found (stub only).* 