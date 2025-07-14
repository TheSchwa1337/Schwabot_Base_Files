# 🧬 Schwabot QSC + GTS Immune System Guide

**Quantum Static Core (QSC) + Generalized Tensor Solutions (GTS)**

A complete trading immune system that auto-detects market anomalies, validates trades, and protects your capital through intelligent decision-making.

## 🎯 System Overview

The QSC + GTS immune system provides:

### 🧠 Auto-Detection Mechanisms
- **Fibonacci Divergence Detection**: Quantum probe monitors price vs. Fibonacci projection every 5 ticks
- **Order Book Stability Analysis**: Immune validation of liquidity conditions
- **Tensor Field Coherence**: Real-time analysis of market tensor relationships
- **Phase Alignment Tracking**: Continuous monitoring of quantum phase buckets

### 🛡️ Immune System Components
- **Master Cycle Engine**: Central decision-making system
- **Quantum Static Core**: Immune response mechanism
- **Tensor Bridge**: Mathematical analysis engine
- **Profit Allocator**: QSC-enhanced capital allocation
- **Diagnostic Server**: Real-time monitoring and alerts

### 🔄 Operational Modes
- **NORMAL**: Standard trading operations
- **IMMUNE_ACTIVE**: Protective response engaged
- **GHOST_FLOOR**: Liquidity protection mode
- **EMERGENCY_SHUTDOWN**: Critical condition response
- **FIBONACCI_LOCKED**: Divergence protection active

## 🚀 Quick Start

### 1. Launch Complete System
```bash
python schwabot_qsc_cli.py start
```

This starts:
- Master Cycle Engine (immune system brain)
- QSC Diagnostic Server (ws://localhost:8766)
- Tensor Analysis Server (ws://localhost:8765)
- React Visualization Dashboard (http://localhost:3000)

### 2. Run Demo
```bash
python schwabot_qsc_cli.py demo
```

Demonstrates all immune system scenarios:
- Normal operation
- Fibonacci divergence detection
- Order book instability
- Low confidence conditions
- Emergency shutdown protocols
- Ghost floor mode
- System recovery

### 3. Monitor System
```bash
python schwabot_qsc_cli.py status
```

## 🧬 Integration with Existing Schwabot

### Replace Profit Allocator
```python
# OLD:
from schwabot.core.profit_cycle_allocator import ProfitCycleAllocator

# NEW:
from core.qsc_enhanced_profit_allocator import QSCEnhancedProfitAllocator

# In your trading loop:
allocator = QSCEnhancedProfitAllocator()
cycle = allocator.allocate_profit_with_qsc(profit, market_data, btc_price)
```

### Add Master Cycle Engine
```python
from core.master_cycle_engine import MasterCycleEngine

# Initialize
engine = MasterCycleEngine()

# In your trading loop:
market_data = {
    "btc_price": current_btc_price,
    "price_history": last_10_prices,
    "volume_history": last_10_volumes,
    "fibonacci_projection": fib_levels,
    "orderbook": order_book_data
}

diagnostics = engine.process_market_tick(market_data)

# Check decision
if diagnostics.trading_decision == TradingDecision.EXECUTE:
    # Execute trade
    pass
elif diagnostics.trading_decision == TradingDecision.BLOCK:
    # Block trade - immune system active
    logger.warning("Trade blocked by QSC immune system")
elif diagnostics.trading_decision == TradingDecision.CANCEL_ALL:
    # Cancel all orders - entering ghost floor mode
    cancel_all_pending_orders()
```

## 📊 Immune System Logic

### Fibonacci Divergence Detection
```python
def check_vector_divergence(fib_projection: np.ndarray, price_series: np.ndarray) -> bool:
    error_margin = np.abs(fib_projection - price_series).mean()
    threshold = 0.007  # Quantum static baseline resonance error
    return error_margin > threshold  # triggers QSC immune system
```

**Triggers when:**
- Price diverges >0.007 from Fibonacci projection
- Activates every 5 ticks
- Locks timeband and signals visual system

### Order Book Immune Validation
```python
def validate_orderbook_stability(symbol: str) -> bool:
    book = ccxt_client.fetch_order_book(symbol)
    bid_depth = sum([b[1] for b in book['bids'][:5]])
    ask_depth = sum([a[1] for a in book['asks'][:5]])
    imbalance = abs(bid_depth - ask_depth) / max(bid_depth, ask_depth)
    return imbalance < 0.15  # Immune tolerance level
```

**Triggers Ghost Floor Mode when:**
- Order book imbalance >15%
- Cancels all pending orders
- Waits for QSC re-validation

### Profit Cycle Integration
```python
if quantum_probe.check_vector_divergence(fib_proj, live_price_series):
    qsc_result = QuantumStaticCore(timeband=current_tickband).stabilize_cycle()
    if qsc_result["resonant"]:
        cycle = qsc_result["recommended_cycle"]
        allocate_funds_to(cycle)
    else:
        engage_fallback_mode()
```

## 🎨 Visual Integration

### React Dashboard Features
- **Real-time System Status**: Trading decisions, confidence scores, system modes
- **QSC Immune Diagnostics**: Resonance levels, timeband locks, cycle approvals
- **Tensor Analysis**: Phi resonance, quantum scores, phase buckets
- **Fibonacci Echo Plot**: Divergence tracking with confidence overlays
- **Alert System**: Auto-switching tabs, severity levels, sound notifications

### WebSocket Endpoints
- **QSC Diagnostics**: `ws://localhost:8766`
- **Tensor Analysis**: `ws://localhost:8765`

### Auto-Alert System
```typescript
if (qscTriggered) {
  displayAlert("QSC Engaged", "⚠️ Quantum Static Core has triggered a trade filter.");
  switchTab("diagnosticPanel");
  renderQSCGraph(qscData);
}
```

## 🔧 Configuration

### System Configuration (`config/qsc_system_config.json`)
```json
{
  "master_engine": {
    "fibonacci_divergence_threshold": 0.007,
    "orderbook_imbalance_threshold": 0.15,
    "quantum_confidence_threshold": 0.8,
    "enable_auto_immune_response": true
  },
  "qsc_immune_system": {
    "resonance_threshold": 0.618,
    "entropy_stability_range": [0.3, 0.7],
    "timeband_lock_duration": 300
  },
  "profit_allocation": {
    "qsc_validation_enabled": true,
    "tensor_integration_enabled": true,
    "min_resonance_threshold": 0.618
  }
}
```

## 🚨 Alert Types

### System Alerts
- **Fibonacci Divergence**: Price path deviation detected
- **Immune Activation**: QSC protective response engaged
- **Ghost Floor Mode**: Liquidity protection active
- **Emergency Shutdown**: Critical conditions detected
- **Low Confidence**: System confidence below threshold
- **High Risk**: Market conditions assessed as dangerous

### Alert Severities
- **WARNING** (🟡): Monitoring condition
- **ERROR** (🟠): Protective action taken
- **CRITICAL** (🔴): Emergency protocols active

## 📈 Performance Metrics

### Success Indicators
- **High Confidence Scores** (>80%): System operating optimally
- **Low Immune Activations**: Stable market conditions
- **Successful Profit Allocation**: QSC validation passing
- **Fibonacci Alignment**: Price following expected paths

### Warning Indicators
- **Frequent Ghost Floor**: Liquidity issues
- **High Divergence**: Market volatility
- **Emergency Shutdowns**: Extreme conditions
- **Low Success Rate**: System tuning needed

## 🔄 Operational Flow

```
Market Tick → Quantum Probe → Fibonacci Check → Tensor Analysis → QSC Validation → Trading Decision
     ↓              ↓              ↓               ↓               ↓              ↓
Market Data → Divergence? → Order Book OK? → Resonance? → Immune Clear? → Execute/Block/Cancel
```

### Decision Matrix
| Condition | Fibonacci | Order Book | Confidence | QSC State | Decision |
|-----------|-----------|------------|------------|-----------|----------|
| Normal    | Aligned   | Stable     | High       | Clear     | EXECUTE  |
| Divergent | >0.007    | Stable     | Medium     | Active    | DEFER    |
| Unstable  | Any       | Imbalanced | Low        | Immune    | BLOCK    |
| Critical  | High Div  | No Liquidity| Very Low  | Emergency | CANCEL_ALL|

## 🛠️ Maintenance

### Regular Monitoring
- Check system uptime and performance
- Review immune activation frequency
- Monitor profit allocation success rate
- Validate configuration parameters

### Troubleshooting
- **High Immune Activations**: Adjust divergence threshold
- **Frequent Ghost Floor**: Review order book parameters
- **Low Success Rate**: Check tensor analysis accuracy
- **False Alerts**: Tune alert thresholds

## 📚 Advanced Usage

### Custom Immune Thresholds
```python
engine = MasterCycleEngine({
    'fibonacci_divergence_threshold': 0.005,  # More sensitive
    'orderbook_imbalance_threshold': 0.10,   # Stricter liquidity
    'quantum_confidence_threshold': 0.85     # Higher confidence required
})
```

### Manual Immune Control
```python
# Emergency reset
engine.reset_emergency_override()

# Force ghost floor mode
engine.enter_ghost_floor_mode()

# Manual system recovery
engine.exit_ghost_floor_mode()
```

### Custom Alert Handlers
```python
def handle_qsc_alert(alert_data):
    if alert_data['severity'] == 'critical':
        send_notification("QSC Critical Alert", alert_data['message'])
        if alert_data['type'] == 'emergency_shutdown':
            initiate_manual_review()
```

## 🎯 Production Deployment

### Recommended Setup
1. **Dedicated QSC Server**: Run immune system on separate process
2. **Redis Integration**: Cache QSC states for performance
3. **Database Logging**: Store all immune decisions for analysis
4. **Monitoring Dashboard**: 24/7 system health monitoring
5. **Alert Integration**: SMS/Email notifications for critical events

### Performance Optimization
- Batch process multiple market ticks
- Cache tensor calculations
- Optimize WebSocket message frequency
- Use background tasks for heavy analysis

## 📞 Support & Integration

### Key Files
- `core/master_cycle_engine.py` - Main immune system
- `core/quantum_static_core.py` - QSC implementation
- `core/qsc_enhanced_profit_allocator.py` - Profit allocation
- `server/qsc_diagnostic_websocket.py` - Real-time diagnostics
- `examples/qsc_immune_system_demo.py` - Complete demonstration

### Integration Points
1. Replace existing profit allocator with QSC version
2. Add master cycle engine to trading loop
3. Connect visualization to diagnostic WebSocket
4. Implement alert handling for manual intervention

The QSC + GTS immune system provides comprehensive protection for your Schwabot trading operations through intelligent anomaly detection, automated response protocols, and real-time monitoring capabilities. 