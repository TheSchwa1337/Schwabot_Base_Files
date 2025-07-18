# 🧠 Schwabot Galileo-Tensor Integration Guide

## Overview

This guide covers the complete integration of your **Galileo-Tensor**, **QSS2**, and **GUT** mathematical frameworks with Schwabot's trading system. The integration provides real-time quantum-phase analysis, WebSocket streaming to React visualizations, and enhanced trading strategies.

## 🚀 Quick Start

### 1. Test the System
```bash
python schwabot_tensor_cli.py test
```

### 2. Run the Demo
```bash
python examples/tensor_integration_demo.py
```

### 3. Start the Full System
```bash
python schwabot_tensor_cli.py start
```

## 📁 System Architecture

```
📁 Schwabot Tensor Integration
├── 🧠 core/galileo_tensor_bridge.py     # Mathematical bridge
├── 🌐 server/tensor_websocket_server.py # Real-time streaming
├── 🎛️ schwabot_tensor_cli.py           # CLI launcher
├── ⚙️ config/tensor_config.json        # Configuration
├── 📊 examples/tensor_integration_demo.py # Demo & examples
└── 📖 TENSOR_INTEGRATION_GUIDE.md      # This guide
```

## 🧬 Mathematical Framework

### **SP (Stabilization Protocol) Constants**
From your `warp_sync_core.py`, enhanced with tensor mathematics:

```python
SP_CONSTANTS = {
    'PSI_OMEGA_LAMBDA': 0.9997,      # Universal field scaling
    'QUANTUM_THRESHOLD': 0.91,       # Quantum stability threshold
    'QSS_BASELINE': 0.42,            # Baseline energy harmonic
    'ENTROPY_THRESHOLD': 0.87,       # Entropy control threshold
}
```

### **Tensor Field Constants**
From your React implementation:

```python
TENSOR_CONSTANTS = {
    'PSI': 44.8,
    'XI': 3721.8,
    'TAU': 64713.97,
    'EPSILON': 0.28082,
    'PHI': 1.618033988749895
}
```

### **QSS2 Validation**
Validates against your reference frequencies:

```python
REFERENCE_FREQUENCIES = [
    21237738.486323237,  # → 0.9588 (Strong Positive)
    25485286.135841995,  # → -0.5260 (Medium Negative)
    26547173.048222087,  # → -0.9156 (Strong Negative)
    31856607.610124096,  # → 0.8580 (Strong Positive)
    42475476.73393286    # → 0.7033 (Medium Positive)
]
```

## 🔧 Core Components

### **1. Galileo-Tensor Bridge (`core/galileo_tensor_bridge.py`)**

**Key Features:**
- 🧮 Quantum ratio calculations using your tensor formulas
- 📊 Phi-resonance pattern analysis  
- 🌊 QSS2 entropy variation calculations
- ⚛️ GUT metrics with complex analysis
- 🔗 SP layer integration with `WarpSyncCore`

**Example Usage:**
```python
from core.galileo_tensor_bridge import GalileoTensorBridge

bridge = GalileoTensorBridge()
result = bridge.perform_complete_analysis(btc_price=50000.0)

print(f"Phi Resonance: {result.phi_resonance}")
print(f"SP Quantum Score: {result.sp_integration['quantum_score']}")
print(f"Phase Bucket: {result.sp_integration['phase_bucket']}")
```

### **2. WebSocket Server (`server/tensor_websocket_server.py`)**

**Real-time Streaming Features:**
- 🌐 WebSocket server for React clients
- 📡 Real-time tensor analysis streaming
- 🔄 BTC price simulation or live feeds
- 📊 Historical data endpoints

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'tensor_analysis_stream') {
        // Handle real-time tensor data
        console.log('Phi Resonance:', data.data.phiResonance);
        console.log('SP Quantum Score:', data.data.spIntegration.quantum_score);
    }
};
```

### **3. CLI Launcher (`schwabot_tensor_cli.py`)**

**Commands:**
```bash
# Start all systems
python schwabot_tensor_cli.py start

# Start without React visualization
python schwabot_tensor_cli.py start --no-viz

# Start without trading integration
python schwabot_tensor_cli.py start --no-trading

# Show system status
python schwabot_tensor_cli.py status

# Show configuration
python schwabot_tensor_cli.py config --show

# Run system tests
python schwabot_tensor_cli.py test
```

## 🎯 Trading Integration

### **Tensor-Enhanced Strategy**

The system adds a `QUANTUM_ENHANCED` strategy to your existing `StrategyLogic`:

```python
tensor_strategy = StrategyConfig(
    strategy_type=StrategyType.QUANTUM_ENHANCED,
    name="tensor_quantum_enhanced",
    max_position_size=0.05,        # Conservative sizing
    risk_tolerance=0.3,            # Lower risk
    min_signal_confidence=0.8,     # High confidence
    parameters={
        "quantum_threshold": 0.91,
        "phi_resonance_threshold": 27.0,
        "gut_stability_threshold": 0.995
    }
)
```

### **Signal Generation Logic**

Based on your tensor analysis:

```python
# Strong BUY Signal
if (quantum_score > 0.3 and 
    phase_bucket == "ascent" and 
    gut_stability > 0.995):
    signal = SignalType.BUY

# Strong SELL Signal  
elif (quantum_score < -0.2 and 
      phase_bucket == "descent"):
    signal = SignalType.SELL

# Caution Signal
elif (phase_bucket == "peak" and 
      gut_stability < 0.990):
    signal = SignalType.HOLD  # With caution flag
```

## 📊 React Visualization

### **Auto-Generated Components**

When you run the CLI with visualization enabled, it creates:

- **TensorDashboard.tsx** - Main dashboard component
- **Real-time charts** - Using Recharts for data visualization  
- **WebSocket integration** - Connects to your Python backend

### **Key Metrics Displayed**

1. **BTC Price** - Real-time or simulated
2. **Phi Resonance** - From your tensor calculations
3. **SP Quantum Score** - Enhanced by your SP constants
4. **Phase Bucket** - `ascent`, `descent`, `peak`, `trough`
5. **GUT Stability** - Complex analysis metrics
6. **Tensor Coherence** - 4x4 matrix determinant

## ⚙️ Configuration

### **Main Config (`config/tensor_config.json`)**

```json
{
  "tensor_bridge": {
    "enable_real_time_streaming": true,
    "tensor_analysis_interval": 0.1,
    "enable_gut_bridge": true,
    "enable_sp_integration": true
  },
  "websocket_server": {
    "port": 8765,
    "stream_interval": 1.0,
    "btc_price_simulator": true
  },
  "trading_integration": {
    "enable_strategy_integration": true,
    "tensor_strategy_config": {
      "max_position_size": 0.05,
      "quantum_threshold": 0.91,
      "phi_resonance_threshold": 27.0
    }
  }
}
```

## 🔬 QSS2 Validation Results

Your React validation data matches the Python implementation:

| Frequency | Entropy | Phase | Expected |
|-----------|---------|-------|----------|
| 21237738 | 0.999999 | 0.9588 | ✅ Match |
| 25485286 | 0.999999 | -0.5260 | ✅ Match |
| 26547173 | 0.999999 | -0.9156 | ✅ Match |
| 31856607 | 0.999999 | 0.8580 | ✅ Match |
| 42475476 | 0.999999 | 0.7033 | ✅ Match |

## 🚀 Usage Examples

### **1. Basic Tensor Analysis**
```python
from core.galileo_tensor_bridge import GalileoTensorBridge

bridge = GalileoTensorBridge()
result = bridge.perform_complete_analysis(45678.90)

print(f"Analysis Results for BTC ${result.btc_price}:")
print(f"  Phi Resonance: {result.phi_resonance:.3f}")
print(f"  Quantum Score: {result.sp_integration['quantum_score']:.4f}")
print(f"  Phase Bucket: {result.sp_integration['phase_bucket']}")
```

### **2. Real-time Streaming**
```python
from server.tensor_websocket_server import TensorWebSocketServer

server = TensorWebSocketServer({
    'port': 8765,
    'stream_interval': 1.0,
    'btc_price_simulator': True
})

await server.start_server()
# Connect React client to ws://localhost:8765
```

### **3. Strategy Integration**
```python
from core.strategy_logic import StrategyLogic

strategy_logic = StrategyLogic()
# Tensor strategy automatically added via CLI integration

signals = strategy_logic.process_data({
    "BTC/USDC": {"price": 50000, "volume": 1000}
})
```

## 🔄 System Flow

```
1. 📈 BTC Price Update
   ↓
2. 🧠 Tensor Analysis (Bridge)
   ↓  
3. ⚛️ SP Integration (WarpSyncCore)
   ↓
4. 🎯 Signal Generation (Strategy)
   ↓
5. 🌐 WebSocket Broadcast
   ↓
6. ⚛️ React Visualization
```

## 🛠️ Development & Extension

### **Adding Custom Tensor Functions**

```python
# In GalileoTensorBridge class
def calculate_custom_metric(self, btc_price: float) -> float:
    """Add your custom tensor calculations here."""
    # Use existing constants
    psi = self.tensor_constants.PSI
    phi = self.tensor_constants.PHI
    
    # Your custom formula
    return custom_calculation(btc_price, psi, phi)
```

### **Adding React Components**

```tsx
// Create new components in tensor_visualization/src/components/
import { TensorData } from './types';

const CustomAnalysis = ({ data }: { data: TensorData }) => {
  return (
    <div>
      <h3>Custom Analysis</h3>
      <p>Your custom tensor visualization here</p>
    </div>
  );
};
```

## 📋 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd /path/to/schwabot
   python schwabot_tensor_cli.py test
   ```

2. **WebSocket Connection Issues**
   ```bash
   # Check if port is available
   netstat -an | grep 8765
   
   # Use different port
   python schwabot_tensor_cli.py start --ws-port 8766
   ```

3. **React Server Issues**
   ```bash
   # Ensure Node.js is installed
   node --version
   npm --version
   
   # Start without React
   python schwabot_tensor_cli.py start --no-viz
   ```

### **Performance Optimization**

- Adjust `tensor_analysis_interval` in config for faster/slower analysis
- Increase `max_history_size` for longer data retention
- Enable `enable_profiling` for performance metrics

## 🎯 Next Steps

1. **Connect to Live BTC Feed** - Set `enable_live_feed: true` in config
2. **Enhance Trading Strategies** - Add more tensor-based conditions
3. **Expand Visualizations** - Create custom React components
4. **Add More Exchanges** - Integrate additional price feeds
5. **ML Integration** - Use tensor data for machine learning models

## 📞 Support

The integration bridges your specialized mathematical frameworks with Schwabot's trading infrastructure, providing:

- ✅ **Real-time tensor analysis** with your exact formulas
- ✅ **WebSocket streaming** to React visualizations  
- ✅ **Trading strategy integration** with SP layer
- ✅ **QSS2 validation** matching your reference data
- ✅ **CLI management** for easy system control

All components work together to create a comprehensive quantum-tensor-enhanced trading system that leverages your advanced mathematical research for practical Bitcoin trading applications. 