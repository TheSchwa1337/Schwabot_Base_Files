# 🧠 Unified Schwabot Integration System

## Overview

The Unified Schwabot Integration System brings together all of Schwabot's mathematical frameworks into a cohesive, AI-enhanced trading system. This system respects your **16-bit positioning system**, **10,000-tick map**, and all core logic (**CCO**, **UFS**, **SFS**, **SFSS**) while providing entropy-driven API triggers and multi-AI model consensus.

## 🌟 Key Features

### 🔄 Entropy-Driven Architecture
- **Hash-based triggers** that respond to market entropy changes
- **16-bit positioning system** for precise market positioning
- **10,000-tick map** for historical pattern recognition
- **Real-time entropy calculations** based on price, volume, and system state

### 🤖 Multi-AI Integration
- **ChatGPT (GPT-4)** integration for trading analysis
- **Anthropic Claude** integration for risk assessment
- **Google Gemini** integration for market analysis
- **Consensus-based decision making** from multiple AI models

### 🧮 Mathematical Framework Respect
- **CCO (Core Control Orchestrator)** - Centralized control logic
- **UFS (Unified Fault System)** - Integrated fault handling
- **SFS (Sequential Fractal Stack)** - Fractal pattern recognition
- **SFSS (Sequential Fractal Strategy Signal Stack)** - Strategy coordination

### 📊 Real-Time Data Integration
- **CCXT** for multi-exchange data
- **Coinbase API** for primary trading data
- **WebSocket broadcasting** for real-time updates
- **Flask API endpoints** for external access

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED SCHWABOT INTEGRATION             │
├─────────────────────────────────────────────────────────────┤
│  🧠 FaultBus (Core Engine)                                  │
│  ├── DLT Waveform Engine                                    │
│  ├── Multi-Bit BTC Processor                                │
│  ├── Riddle GEMM Engine                                     │
│  └── Temporal Execution Correction Layer                    │
├─────────────────────────────────────────────────────────────┤
│  📊 Data Integration Layer                                  │
│  ├── CCXT Exchange Connectors                               │
│  ├── Coinbase API Integration                               │
│  └── WebSocket Broadcasting                                 │
├─────────────────────────────────────────────────────────────┤
│  🔄 Entropy API Layer                                       │
│  ├── 16-Bit Positioning System                              │
│  ├── Hash-Based Command Functions                           │
│  ├── Entropy Calculation Engine                             │
│  └── Flask API Endpoints                                    │
├─────────────────────────────────────────────────────────────┤
│  🤖 AI Integration Bridge                                   │
│  ├── ChatGPT Integration                                    │
│  ├── Claude Integration                                     │
│  ├── Gemini Integration                                     │
│  └── Consensus Engine                                       │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install required dependencies
pip install flask flask-cors websockets ccxt openai anthropic google-generativeai

# Clone the repository
git clone <your-repo-url>
cd schwabot-unified
```

### 2. Configuration

Create a configuration file `config.json`:

```json
{
  "ai_models": {
    "gpt": {
      "api_key": "your-openai-api-key",
      "model_id": "gpt-4",
      "enabled": true
    },
    "claude": {
      "api_key": "your-anthropic-api-key",
      "model_id": "claude-3-sonnet-20240229",
      "enabled": true
    },
    "gemini": {
      "api_key": "your-google-api-key",
      "model_id": "gemini-pro",
      "enabled": true
    }
  },
  "exchanges": {
    "binance": {
      "enabled": true,
      "sandbox": true
    },
    "coinbase": {
      "enabled": true,
      "sandbox": true
    }
  },
  "entropy": {
    "threshold": 0.5,
    "update_interval": 225.0
  }
}
```

### 3. Run the System

```python
import asyncio
from core.unified_schwabot_integration import create_unified_schwabot_integration

async def main():
    # Create unified integration
    integration = create_unified_schwabot_integration()
    
    # Start the system
    await integration.start()

if __name__ == "__main__":
    asyncio.run(main())
```

## 📡 API Endpoints

### Entropy API (Flask - Port 5000)

#### Get Current Entropy
```http
GET /api/entropy/current
```
Returns current entropy value and threshold.

#### Get Entropy History
```http
GET /api/entropy/history?limit=100
```
Returns entropy history with specified limit.

#### Get 16-Bit Positions
```http
GET /api/bit-positions
```
Returns current 16-bit positioning system state.

#### Get Hash Commands
```http
GET /api/hash-commands
```
Returns registered hash-based command functions.

#### Register Hash Command
```http
POST /api/hash-commands
Content-Type: application/json

{
  "command_id": "my_command",
  "hash_pattern": "a",
  "execution_function": "trigger_ai_analysis",
  "parameters": {"analysis_type": "market_analysis"},
  "priority": 5
}
```

#### Get AI Responses
```http
GET /api/ai/responses?limit=50
```
Returns recent AI model responses.

#### Get AI Consensus
```http
GET /api/ai/consensus
```
Returns AI consensus on recent decisions.

#### Get Market State
```http
GET /api/market/state
```
Returns current market state and system metrics.

#### Get System Status
```http
GET /api/system/status
```
Returns system health and performance metrics.

### WebSocket Server (Port 8765)

Connect to `ws://localhost:8765` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

// Subscribe to updates
ws.send(JSON.stringify({
    type: 'subscribe',
    client: 'my_client'
}));
```

## 🧮 Mathematical Framework Details

### 16-Bit Positioning System

The system maintains 16 individual bit positions (0-15), each representing a specific market condition:

```python
# Example bit positions
bit_positions = {
    0: {'active': True, 'hash': 'a1b2c3d4', 'data': {...}},
    1: {'active': False, 'hash': 'e5f6g7h8', 'data': {...}},
    # ... up to bit 15
}
```

Each bit position:
- Has a unique hash signature
- Can be active or inactive
- Contains market data and timing information
- Updates every 3.75 minutes (225 seconds)

### 10,000-Tick Map

The system maintains a 10,000-tick historical map for pattern recognition:

```python
position_history = deque(maxlen=10000)  # 10,000 tick map
```

Each tick contains:
- Timestamp
- All 16-bit positions
- Entropy value
- Market state snapshot

### Hash-Based Command Functions

Commands are triggered based on hash patterns:

```python
# Example hash commands
hash_commands = {
    'high_entropy_alert': {
        'hash_pattern': 'f',
        'execution_function': 'trigger_ai_analysis',
        'priority': 10
    },
    'bit_position_update': {
        'hash_pattern': '0',
        'execution_function': 'update_bit_positions',
        'priority': 5
    }
}
```

### Entropy Calculation

Entropy is calculated from multiple factors:

```python
entropy = (
    normalized_volatility * 0.3 +
    normalized_volume * 0.25 +
    normalized_hash * 0.25 +
    normalized_faults * 0.2
)
```

## 🤖 AI Integration Details

### Multi-Model Consensus

The system queries multiple AI models and generates consensus:

```python
# Example AI consensus
consensus = {
    'consensus_action': 'buy',
    'confidence': 0.85,
    'agreement_level': 0.8,
    'model_responses': [
        {'model': 'gpt', 'action': 'buy', 'confidence': 0.9},
        {'model': 'claude', 'action': 'buy', 'confidence': 0.8},
        {'model': 'gemini', 'action': 'hold', 'confidence': 0.7}
    ]
}
```

### AI Prompt Structure

AI models receive structured prompts including:
- Current entropy value
- 16-bit position status
- Market state information
- Decision context

### Response Format

AI models respond in JSON format:

```json
{
    "action": "buy|sell|hold",
    "confidence": 0.85,
    "reasoning": "Detailed reasoning...",
    "risk": "low|medium|high",
    "analysis": "Market analysis..."
}
```

## 🔧 Advanced Configuration

### Custom Hash Commands

Register custom hash commands for specific patterns:

```python
entropy_api.register_hash_command(
    command_id='custom_analysis',
    hash_pattern='b',
    execution_function='trigger_ai_analysis',
    parameters={'analysis_type': 'custom'},
    priority=8
)
```

### AI Model Configuration

Configure individual AI models:

```python
from core.ai_integration_bridge import AIModelConfig

configs = {
    'gpt': AIModelConfig(
        model_name='gpt',
        api_key='your-key',
        model_id='gpt-4',
        max_tokens=1000,
        temperature=0.7,
        enabled=True,
        priority=1
    )
}
```

### FaultBus Event Handlers

Register custom event handlers:

```python
@fault_bus.register_handler("profit_anomaly")
def handle_profit_anomaly(event):
    # Custom handling logic
    pass
```

## 📊 Monitoring and Analytics

### System Health

Get comprehensive system health information:

```python
health = integration.get_system_health()
print(health)
```

### Entropy Analytics

Get detailed entropy analytics:

```python
analytics = integration.get_entropy_analytics()
print(analytics)
```

### AI Consensus Summary

Get AI consensus summary:

```python
consensus = integration.get_ai_consensus_summary()
print(consensus)
```

## 🔒 Security Considerations

1. **API Keys**: Store API keys securely, never commit them to version control
2. **Rate Limiting**: Respect exchange and AI API rate limits
3. **Sandbox Mode**: Use sandbox mode for testing
4. **Error Handling**: Implement proper error handling for all external APIs
5. **Logging**: Monitor system logs for unusual activity

## 🚨 Emergency Procedures

The system includes emergency response procedures:

1. **Thermal Critical**: Automatic system throttling
2. **Recursive Loop**: Entropy threshold adjustment
3. **Profit Anomaly**: AI analysis trigger
4. **System Failure**: Graceful shutdown procedures

## 📈 Performance Optimization

### Memory Management
- 10,000-tick map with automatic cleanup
- Entropy history with configurable limits
- AI response caching with TTL

### Processing Optimization
- Async/await for all I/O operations
- Threading for CPU-intensive tasks
- WebSocket for real-time updates

### Scalability
- Modular component architecture
- Configurable update intervals
- Horizontal scaling support

## 🔮 Future Enhancements

1. **Additional AI Models**: Integration with more AI providers
2. **Advanced Analytics**: Machine learning for pattern recognition
3. **Multi-Asset Support**: Extension beyond BTC/USDC
4. **Cloud Deployment**: Kubernetes and Docker support
5. **Advanced Visualization**: Real-time charts and dashboards

## 📞 Support

For questions and support:
- Check the logs in `logs/` directory
- Monitor system health via API endpoints
- Review error messages in console output
- Check WebSocket connection status

## 📄 License

This system is part of the Schwabot project. Please refer to the main project license for details.

---

**Note**: This system is designed for educational and research purposes. Always test thoroughly in sandbox environments before using with real funds. 