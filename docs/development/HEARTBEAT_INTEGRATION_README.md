# Schwabot Heartbeat Integration System

## Overview

The **Heartbeat Integration Manager** is the central coordination system for Schwabot's advanced trading modules. It orchestrates all components with a 5-minute heartbeat cycle, ensuring organic, self-regulated decision-making based on system state and learned profit patterns.

## 🎯 Key Features

- **5-Minute Heartbeat Cycle**: Coordinated execution of all modules
- **Thermal-Aware Routing**: Strategy selection based on system temperature and entropy
- **Autonomic Limits**: Self-governing execution constraints
- **GPU Acceleration**: CuPy-based tensor analysis for strategy matrices
- **Profit Projection**: Multi-method profit forecasting
- **Memory Management**: Intelligent cache and memory optimization
- **Health Monitoring**: Real-time system health tracking

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Heartbeat Integration Manager               │
├─────────────────────────────────────────────────────────────┤
│  💓 5-Minute Heartbeat Cycle                               │
│  🔄 Module Coordination                                    │
│  📊 Performance Monitoring                                 │
│  🏥 Health Management                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Modules                             │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│🌡️ Thermal   │🛡️ Autonomic │🎮 GPU Logic │💰 Profit         │
│  Router     │  Limit      │  Mapper     │  Projection      │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│📦 API Tick  │📈 Drift     │📊 Cache     │🔍 Analysis       │
│  Cache      │  Profiler   │  Management │  Engine          │
└─────────────┴─────────────┴─────────────┴───────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install cupy-cuda11x  # Replace with your CUDA version
```

### 2. Basic Usage

```python
import asyncio
from core.heartbeat_integration_manager import HeartbeatIntegrationManager

async def main():
    # Initialize the heartbeat integration manager
    heartbeat_manager = HeartbeatIntegrationManager()
    
    # Initialize all modules
    await heartbeat_manager.initialize()
    
    # Start the heartbeat system
    await heartbeat_manager.start()
    
    # Run a single heartbeat cycle
    result = await heartbeat_manager.run_heartbeat_cycle()
    print(f"Cycle completed: {result['status']}")
    
    # Get system statistics
    stats = heartbeat_manager.get_integration_stats()
    print(f"Total heartbeats: {stats['heartbeat_manager']['heartbeat_count']}")
    
    # Stop the system
    await heartbeat_manager.stop()

# Run the example
asyncio.run(main())
```

### 3. Continuous Operation

```python
async def run_continuous_heartbeat():
    heartbeat_manager = HeartbeatIntegrationManager()
    await heartbeat_manager.initialize()
    await heartbeat_manager.start()
    
    # Run continuous heartbeat cycles
    await heartbeat_manager.run_continuous_heartbeat()

# Run continuous operation
asyncio.run(run_continuous_heartbeat())
```

## 📋 Module Details

### 🌡️ Thermal Strategy Router

Routes strategies based on system thermal state and entropy:

```python
from core.thermal_strategy_router import ThermalStrategyRouter

router = ThermalStrategyRouter()
mode = router.determine_mode()  # Returns: "short-term-sniper", "mid-cycle-batcher", or "macroframe-planner"
result = router.engage_strategy()
```

**Modes:**
- **Short-term Sniper**: High-frequency, low-latency strategies (cool system)
- **Mid-cycle Batcher**: Balanced processing strategies (normal system)
- **Macroframe Planner**: Long-term planning strategies (warm system)

### 🛡️ Autonomic Limit Layer

Enforces self-governing execution constraints:

```python
from core.autonomic_limit_layer import AutonomicLimitLayer

layer = AutonomicLimitLayer()
is_valid, reason, data = layer.validate_strategy_execution("strategy_tag", strategy_data)
```

**Limits:**
- Maximum drawdown: 4%
- Cycle repetition limits
- Thermal pressure thresholds
- Memory usage limits
- Strategy confidence requirements

### 🎮 GPU Logic Mapper

Transfers strategy matrices to GPU for tensor analysis:

```python
from core.gpu_logic_mapper import GPULogicMapper

mapper = GPULogicMapper()
result = mapper.map_strategy_to_gpu("strategy_hash", strategy_matrix)
```

**Features:**
- CuPy-based GPU acceleration
- Tensor analysis (eigenvalues, SVD, entropy)
- Memory management
- CPU fallback support

### 💰 Profit Projection Engine

Estimates profit projections using multiple methods:

```python
from core.profit_projection_engine import ProfitProjectionEngine

engine = ProfitProjectionEngine()
projection = engine.project_profit(strategy_data)
```

**Methods:**
- Tensor analysis
- Historical regression
- Market condition analysis
- Strategy confidence scoring
- Risk-adjusted projection

## ⚙️ Configuration

The system is configured via `config/heartbeat_integration_config.yaml`:

```yaml
# Core heartbeat settings
heartbeat:
  interval: 300  # 5 minutes
  enable_thermal_routing: true
  enable_autonomic_limits: true
  enable_gpu_mapping: true
  enable_profit_projection: true

# Performance thresholds
  max_concurrent_strategies: 10
  strategy_confidence_threshold: 0.6
  thermal_pressure_threshold: 0.8
  memory_usage_threshold: 0.85
```

## 📊 Monitoring and Health

### Health Status

```python
health = heartbeat_manager.get_health_status()
print(f"Overall Health: {health['overall_health']}")
print(f"Success Rate: {health['success_rate']:.3f}")
print(f"Last Heartbeat Age: {health['last_heartbeat_age']:.1f}s")
```

### Performance Metrics

```python
stats = heartbeat_manager.get_integration_stats()

# Heartbeat metrics
hb_stats = stats['heartbeat_manager']
print(f"Total Heartbeats: {hb_stats['heartbeat_count']}")
print(f"Average Cycle Time: {hb_stats['performance_metrics']['average_cycle_time']:.3f}s")

# Module status
for module, available in stats['modules'].items():
    status = "✅ Available" if available else "❌ Not Available"
    print(f"{module}: {status}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_heartbeat_integration.py
```

The test suite includes:
- Individual module testing
- Integration testing
- Performance benchmarking
- Health monitoring validation

## 🔧 Advanced Usage

### Custom Configuration

```python
config = {
    "heartbeat_interval": 180,  # 3 minutes
    "enable_gpu_mapping": True,
    "max_concurrent_strategies": 20,
    "strategy_confidence_threshold": 0.7
}

heartbeat_manager = HeartbeatIntegrationManager(config)
```

### Module-Specific Configuration

```python
# GPU Mapper configuration
gpu_config = {
    "gpu_memory_limit": 0.9,
    "tensor_analysis": {
        "methods": ["eigenvalue_decomposition", "entropy_calculation"],
        "weights": {"eigenvalue": 0.5, "entropy": 0.5}
    }
}

# Profit Engine configuration
profit_config = {
    "projection_horizon": 48,  # 48 hours
    "methods": ["tensor_analysis", "historical_regression"],
    "weights": {"tensor": 0.4, "historical": 0.6}
}
```

### Event Handling

```python
async def on_heartbeat_cycle(cycle_result):
    print(f"Heartbeat cycle {cycle_result['cycle_number']} completed")
    print(f"Strategies processed: {cycle_result['strategies_processed']}")
    print(f"Thermal state: {cycle_result['thermal_state']}")

# Register event handler (if supported)
heartbeat_manager.register_cycle_callback(on_heartbeat_cycle)
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```
   ⚠️ CuPy not available, using CPU fallback
   ```
   - Install CuPy: `pip install cupy-cuda11x`
   - Or disable GPU: `enable_gpu_mapping: false`

2. **Memory Issues**
   ```
   ❌ Memory usage too high
   ```
   - Reduce `max_concurrent_strategies`
   - Lower `memory_usage_threshold`
   - Enable memory cleanup

3. **Thermal Issues**
   ```
   ❌ Thermal pressure too high
   ```
   - Check system cooling
   - Reduce processing load
   - Adjust thermal thresholds

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode in configuration
config = {
    "development": {
        "debug": {
            "enable_debug_mode": True,
            "enable_trace_logging": True
        }
    }
}
```

## 📈 Performance Optimization

### GPU Optimization

1. **Memory Management**
   ```yaml
   gpu_mapper:
     memory:
       cleanup_threshold: 0.8  # Cleanup at 80% usage
       sync_interval: 50       # Sync every 50 operations
   ```

2. **Batch Processing**
   ```yaml
   gpu_mapper:
     matrix:
       batch_processing_enabled: true
       batch_size: 20
   ```

### Cache Optimization

1. **API Cache**
   ```yaml
   api_cache:
     ttl_seconds: 180          # 3 minutes TTL
     max_cache_size: 5000      # Reduce cache size
     enable_compression: true
   ```

2. **Profit Cache**
   ```yaml
   profit_cache:
     max_profit_history: 500   # Reduce history size
     decay_factor: 0.9         # Faster decay
   ```

## 🔒 Security Considerations

1. **Data Protection**
   ```yaml
   security:
     data:
       enable_encryption: true
       encryption_algorithm: "AES-256"
   ```

2. **Access Control**
   ```yaml
   security:
     access:
       enable_authentication: true
       max_connections: 50
   ```

3. **Audit Logging**
   ```yaml
   security:
     audit:
       enable_audit_logging: true
       log_sensitive_operations: true
   ```

## 📚 API Reference

### HeartbeatIntegrationManager

#### Methods

- `initialize()` → `bool`: Initialize all modules
- `start()` → `bool`: Start the heartbeat system
- `stop()` → `None`: Stop the heartbeat system
- `run_heartbeat_cycle()` → `Dict`: Run a single heartbeat cycle
- `run_continuous_heartbeat()` → `None`: Run continuous cycles
- `get_integration_stats()` → `Dict`: Get comprehensive statistics
- `get_health_status()` → `Dict`: Get system health status

#### Properties

- `is_running`: System running status
- `is_initialized`: Initialization status
- `heartbeat_count`: Number of completed cycles
- `last_heartbeat`: Timestamp of last heartbeat

### Global Functions

- `start_heartbeat_integration()` → `bool`: Start global instance
- `stop_heartbeat_integration()` → `None`: Stop global instance
- `run_heartbeat_cycle()` → `Dict`: Run cycle on global instance
- `get_integration_stats()` → `Dict`: Get global stats
- `get_health_status()` → `Dict`: Get global health

## 🤝 Contributing

1. **Code Style**: Follow PEP 8 guidelines
2. **Testing**: Add tests for new features
3. **Documentation**: Update this README for changes
4. **Performance**: Monitor impact on system performance

## 📄 License

This project is part of the Schwabot trading system. See the main project license for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test suite for examples
3. Enable debug logging for detailed information
4. Check system resources (CPU, memory, GPU)

---

**Note**: This system is designed for production trading environments. Always test thoroughly in a safe environment before deploying to live trading. 