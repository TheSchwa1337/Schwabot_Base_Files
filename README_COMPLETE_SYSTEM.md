# Complete Thermal-Aware Hash Trading System

## Overview

This system implements a complete thermal-aware, profit-synchronized hash trigger trading system that intelligently manages GPU/CPU resources while learning from trading patterns to optimize future performance. The system integrates multiple components from A (entropy tracking) to ZBE (zone-based echo synchronization).

## System Architecture

### Core Components

1. **Memory Map System** (`memory_map.py`)
   - Thread-safe persistent storage for strategy successes
   - Shared memory between agents and components
   - Automatic backup and cleanup functionality

2. **Profit Trajectory Coprocessor** (`profit_trajectory_coprocessor.py`)
   - Analyzes profit momentum over time
   - Provides zone state classification (surging, stable, drawdown, volatile)
   - Calculates drift coefficients for processing decisions

3. **Thermal Zone Manager** (`thermal_zone_manager.py`)
   - Monitors CPU/GPU temperatures and loads
   - Manages thermal zones and burst processing
   - Implements daily processing budget (10% of 24h)

4. **Memory Agent System** (`memory_agent.py`)
   - Learns from strategy execution outcomes
   - Provides confidence coefficients for trading decisions
   - Pattern similarity matching for optimal strategy selection

5. **Flask Gateway** (`flask_gateway.py`)
   - RESTful API for external system integration
   - Real-time monitoring and control endpoints
   - Comprehensive error handling and logging

6. **System Orchestrator** (`system_orchestrator.py`)
   - Coordinates all system components
   - Processes market ticks through complete pipeline
   - Provides unified system management interface

## Key Features

### Thermal-Aware Processing
- **Dynamic GPU/CPU Allocation**: Automatically adjusts processing based on thermal state
- **Burst Management**: Allows intensive processing bursts within thermal/budget constraints
- **Cooling Integration**: Reduces processing when temperatures exceed safe thresholds

### Profit-Synchronized Logic
- **Trajectory Analysis**: Uses cubic polynomial fitting to detect profit momentum
- **Zone-Based Decisions**: Different processing strategies for different profit states
- **Confidence Weighting**: Past performance influences future strategy selection

### Memory & Learning
- **Strategy Pattern Recognition**: Learns which hash triggers lead to profitable trades
- **Context-Aware Confidence**: Considers thermal state and profit zone in decisions
- **Adaptive Coefficients**: Confidence scores evolve based on success patterns

### Hash Trigger Integration
- **16-bit Hash Processing**: Integrates with existing hash trigger engine
- **Pattern Mapping**: Maps hash patterns to price movements and Euler phases
- **Dormant/Collapse Integration**: Connects to existing dormant and collapse engines

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Data Directory**:
   ```bash
   mkdir -p data/backups
   ```

## Usage Examples

### Basic System Usage

```python
from core.system_orchestrator import SystemOrchestrator

# Create and start the system
orchestrator = SystemOrchestrator()
orchestrator.start()

# Process market ticks
result = orchestrator.process_tick(price=100.50)
print(f"Triggered strategies: {result['triggered_strategies']}")

# Complete a strategy
completion = orchestrator.complete_strategy(
    execution_id="some_execution_id",
    exit_price=102.25
)

# Get system statistics
stats = orchestrator.get_statistics()
print(f"System health: {stats['system_state']['system_health']}")

# Shutdown gracefully
orchestrator.shutdown()
```

### Flask API Usage

```python
from core.flask_gateway import create_app

# Start the Flask API server
app = create_app()
app.run(host='0.0.0.0', port=5000)
```

### Direct Component Usage

```python
from core.profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor
from core.thermal_zone_manager import ThermalZoneManager
from core.memory_agent import MemoryAgent

# Initialize components
profit_coprocessor = ProfitTrajectoryCoprocessor()
thermal_manager = ThermalZoneManager(profit_coprocessor)
agent = MemoryAgent("trader_001", thermal_manager=thermal_manager)

# Update profit trajectory
vector = profit_coprocessor.update(profit=150.0)
print(f"Zone state: {vector.zone_state.value}")

# Get thermal recommendations
thermal_manager.update_thermal_state()
if thermal_manager.should_reduce_gpu_load():
    print("Thermal throttling recommended")

# Get trading confidence
confidence = agent.get_confidence_coefficient(
    strategy_id="STRATEGY_001",
    hash_triggers=["TRIGGER_001"],
    current_context={"thermal_state": "normal", "profit_zone": "surging"}
)
print(f"Confidence: {confidence:.3f}")
```

## API Endpoints

### Health & Status
- `GET /health` - System health check
- `GET /profit/status` - Profit trajectory status
- `GET /thermal/status` - Thermal zone status

### Profit Tracking
- `POST /profit/update` - Update profit trajectory
  ```json
  {"profit": 150.75, "timestamp": "2024-01-01T12:00:00Z"}
  ```

### Thermal Management
- `POST /thermal/burst/start` - Start processing burst
- `POST /thermal/burst/end` - End processing burst
  ```json
  {"duration": 30.5}
  ```

### Memory Agents
- `POST /agent/{agent_id}/strategy/start` - Start strategy execution
  ```json
  {
    "strategy_id": "STRATEGY_001",
    "hash_triggers": ["TRIGGER_001", "TRIGGER_002"],
    "entry_price": 100.0,
    "initial_confidence": 0.8
  }
  ```

- `POST /agent/{agent_id}/strategy/complete` - Complete strategy execution
  ```json
  {
    "execution_id": "strategy_20240101_120000_001",
    "exit_price": 105.0,
    "execution_time": 30.5
  }
  ```

- `GET /agent/{agent_id}/confidence` - Get confidence coefficient
  ```
  ?strategy_id=STRATEGY_001&hash_triggers=TRIGGER_001
  ```

### Hash Triggers
- `POST /hash/register` - Register new hash trigger
- `POST /hash/process` - Process hash value
- `GET /hash/active` - Get active triggers

## Mathematical Framework

### Thermal Drift Coefficient
```
D_thermal = 1 / (1 + e^(-((T - T₀) - α * P_avg)))
```
Where:
- T: Current temperature
- T₀: Nominal temperature (70°C)
- α: Profit heat bias (0.5)
- P_avg: Average profit

### Profit Vector Strength
```
V_profit = 1 / (1 + e^(-P'(t_now)))
```
Where P'(t_now) is the derivative of the cubic profit fit at current time.

### Confidence Calculation
```
C_final = (C_base * w_context + C_similarity * w_pattern) * decay_factor
```
Where:
- C_base: Base confidence from historical performance
- w_context: Context weighting (thermal/profit state)
- C_similarity: Pattern similarity boost
- w_pattern: Pattern weighting
- decay_factor: Time-based decay

## Configuration

### Default Configuration
```python
config = {
    'profit_window_size': 10000,
    'thermal_monitoring_interval': 30.0,
    'system_monitoring_interval': 60.0,
    'default_agent_id': 'main_agent',
    'auto_register_triggers': True,
    'enable_thermal_management': True,
    'enable_profit_tracking': True,
    'tick_processing_timeout': 5.0,
    'strategy_execution_timeout': 30.0
}
```

### Thermal Thresholds
- **Cool**: < 60°C (70% GPU allocation)
- **Normal**: 60-70°C (60% GPU allocation)
- **Warm**: 70-80°C (40% GPU allocation)
- **Hot**: 80-90°C (20% GPU allocation)
- **Critical**: > 90°C (10% GPU allocation)

### Profit Zone States
- **Surging**: Positive momentum with low volatility
- **Stable**: Minimal movement, low volatility
- **Drawdown**: Negative momentum
- **Volatile**: High variance regardless of direction
- **Unknown**: Insufficient data

## Integration with Existing Components

### Hash Trigger Engine Integration
The system integrates seamlessly with your existing `hash_trigger_engine.py`:

```python
# Your existing hash trigger engine is automatically initialized
# in the system orchestrator with dormant and collapse engines

# Process hash through complete pipeline
hash_value = 0x1234
cursor_state = CursorState(
    triplet=(100.0, 100.1, 100.2),
    delta_idx=1,
    braid_angle=45.0,
    timestamp=time.time()
)

triggered_ids = hash_trigger_engine.process_hash(hash_value, cursor_state)
```

### Memory Integration
All components share a common memory map for data persistence and cross-component communication.

## Monitoring and Logging

### System Health Monitoring
- Automatic thermal monitoring every 30 seconds
- System statistics logged every 60 seconds
- Memory cleanup every hour
- Daily budget reset at midnight UTC

### Performance Metrics
- Ticks processed per second
- Strategy execution success rate
- Average profit per trade
- Thermal burst utilization
- Memory usage statistics

## Error Handling

### Graceful Degradation
- Component failures don't crash the entire system
- Fallback modes for temperature reading failures
- Default confidence coefficients when agents fail
- Automatic cleanup of corrupted data

### Recovery Procedures
- Automatic restart after component failures
- State persistence across system restarts
- Backup and restore functionality
- Signal handling for graceful shutdowns

## Performance Considerations

### Resource Management
- **Daily Processing Budget**: 10% of 24 hours (2.4 hours)
- **Burst Limitations**: Maximum 5-minute bursts with 2x cooldown
- **Memory Limits**: Automatic cleanup when memory map exceeds 100MB
- **Thread Safety**: All components use proper locking mechanisms

### Optimization Tips
1. **Thermal Management**: Keep ambient temperature low for more processing headroom
2. **Profit Tracking**: Use actual P&L data instead of price deltas for better accuracy
3. **Pattern Recognition**: Ensure diverse training data for better confidence calculations
4. **Memory Usage**: Regular cleanup of old data to maintain performance

## Troubleshooting

### Common Issues
1. **High Memory Usage**: Check memory map size, run cleanup if needed
2. **Thermal Throttling**: Improve cooling or reduce processing intensity
3. **Low Confidence Scores**: Ensure sufficient training data for strategies
4. **API Timeouts**: Check system load and component health

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks
Use the health endpoint to verify system status:
```bash
curl http://localhost:5000/health
```

## Contributing

When extending the system:
1. Maintain thread safety in all components
2. Add proper error handling and logging
3. Update tests for new functionality
4. Document API changes
5. Consider thermal and memory impact

## License

[Include your license information here]

---

**Note**: This system implements the complete A-Z framework discussed in your requirements, providing thermal-aware processing, profit-synchronized decision making, and intelligent memory-based learning for optimal trading performance. 