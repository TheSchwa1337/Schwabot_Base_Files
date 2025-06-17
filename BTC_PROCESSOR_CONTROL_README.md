# BTC Processor Control System

A comprehensive control system for managing BTC data processor features to prevent system overload during live testing and hash processing.

## 🎯 Overview

The BTC Processor Control System provides multiple interfaces to manage processor features and system resources:

- **Web UI**: Real-time dashboard with toggles and monitoring
- **CLI**: Command-line interface for quick control
- **API**: Programmatic control via Python classes
- **Automatic Monitoring**: Background resource monitoring with auto-cleanup

## 🚀 Quick Start

### 1. Web Interface (Recommended)

Start the web dashboard for real-time control:

```bash
python core/btc_processor_ui.py
```

Then open http://localhost:5000 in your browser.

### 2. Command Line Interface

Quick status check:
```bash
python tools/btc_processor_cli.py status
```

Disable all analysis for live testing:
```bash
python tools/btc_processor_cli.py disable-all
```

### 3. Programmatic Control

```python
from core.btc_processor_controller import BTCProcessorController

controller = BTCProcessorController()

# Disable mining analysis to reduce CPU load
await controller.disable_feature('mining_analysis')

# Enable emergency monitoring
await controller.start_monitoring()
```

## 📋 Feature Categories

### Analysis Features (Can be disabled for live testing)
- `mining_analysis` - Bitcoin mining pattern analysis
- `block_timing` - Block timing pattern analysis  
- `nonce_sequences` - Nonce sequence pattern analysis
- `difficulty_tracking` - Difficulty adjustment tracking

### Core Features (Keep enabled for processing)
- `hash_generation` - Core hash generation
- `load_balancing` - CPU/GPU load balancing
- `memory_management` - Memory allocation and cleanup

### Storage Features (Optional)
- `storage` - Historical data storage
- `monitoring` - Performance monitoring

## 🌐 Web Interface Features

### System Metrics Panel
- Real-time CPU, Memory, GPU usage
- Progress bars with visual indicators
- Monitoring status indicator

### Feature Controls Panel
- Toggle switches for all features
- Bulk enable/disable buttons
- Individual feature control

### Emergency Controls Panel
- Emergency memory cleanup
- System monitoring controls
- Critical resource warnings

### Configuration Panel
- Resource limit settings
- Threshold configuration
- Save/load configurations

## 🖥️ CLI Commands

### Status and Monitoring
```bash
# Check current status
python tools/btc_processor_cli.py status

# Monitor system for 30 seconds
python tools/btc_processor_cli.py monitor --duration 30

# Start/stop monitoring
python tools/btc_processor_cli.py start-monitoring
python tools/btc_processor_cli.py stop-monitoring
```

### Feature Control
```bash
# List available features
python tools/btc_processor_cli.py list-features

# Enable specific feature
python tools/btc_processor_cli.py enable mining_analysis

# Disable specific feature
python tools/btc_processor_cli.py disable nonce_sequences

# Bulk controls
python tools/btc_processor_cli.py enable-all
python tools/btc_processor_cli.py disable-all
```

### Configuration Management
```bash
# Update thresholds
python tools/btc_processor_cli.py set-thresholds \
  --memory-warning 6 \
  --memory-critical 8 \
  --cpu-warning 60

# Update processor limits
python tools/btc_processor_cli.py set-config \
  --max-memory 8 \
  --max-cpu 70 \
  --max-gpu 80

# Save/load configurations
python tools/btc_processor_cli.py save-config --file my_config.json
python tools/btc_processor_cli.py load-config --file my_config.json
```

### Emergency Procedures
```bash
# Emergency memory cleanup
python tools/btc_processor_cli.py emergency-cleanup
```

## ⚙️ Configuration Options

### System Thresholds
- `memory_warning`: Memory usage warning level (GB)
- `memory_critical`: Memory usage critical level (GB) 
- `cpu_warning`: CPU usage warning level (%)
- `cpu_critical`: CPU usage critical level (%)
- `gpu_warning`: GPU usage warning level (%)
- `gpu_critical`: GPU usage critical level (%)

### Resource Limits
- `max_memory_usage_gb`: Maximum memory usage (default: 10 GB)
- `max_cpu_usage_percent`: Maximum CPU usage (default: 80%)
- `max_gpu_usage_percent`: Maximum GPU usage (default: 85%)

### Backlog Management
- `max_backlog_size`: Maximum queue size (default: 10,000)
- `auto_cleanup_enabled`: Automatic cleanup (default: true)

## 🔧 Usage Scenarios

### Scenario 1: Live Testing Preparation

```bash
# 1. Disable analysis features to free resources
python tools/btc_processor_cli.py disable-all

# 2. Set conservative thresholds
python tools/btc_processor_cli.py set-thresholds \
  --memory-warning 6 --cpu-warning 50

# 3. Start monitoring
python tools/btc_processor_cli.py start-monitoring

# 4. Run your live testing...

# 5. Restore normal operations
python tools/btc_processor_cli.py enable-all
```

### Scenario 2: Emergency Resource Management

```bash
# Quick emergency cleanup
python tools/btc_processor_cli.py emergency-cleanup

# Check if system recovered
python tools/btc_processor_cli.py status

# Gradually re-enable features
python tools/btc_processor_cli.py enable hash_generation
python tools/btc_processor_cli.py enable load_balancing
```

### Scenario 3: Memory Optimization

```python
from core.btc_processor_controller import BTCProcessorController

controller = BTCProcessorController()

# Set aggressive memory limits
await controller.update_configuration({
    'max_memory_usage_gb': 8.0
})

# Disable memory-intensive features
await controller.disable_feature('mining_analysis')
await controller.disable_feature('storage')

# Enable automatic cleanup
controller.config.auto_cleanup_enabled = True
```

## 📊 Monitoring and Alerts

### Automatic Actions

The system automatically takes action when thresholds are exceeded:

**Memory Warning** (8 GB default):
- Reduces buffer sizes
- Disables pattern storage
- Increases cleanup frequency

**Memory Critical** (12 GB default):
- Emergency memory cleanup
- Disables analysis features
- Clears all buffers
- Forces garbage collection

**CPU Warning** (70% default):
- Reduces processing workers
- Increases sleep intervals
- Optimizes task scheduling

**CPU Critical** (90% default):
- Disables CPU-intensive analysis
- Switches to minimal processing mode

**GPU Warning/Critical** (80%/95% default):
- Switches processing to CPU
- Clears GPU memory cache

### Emergency Shutdown

If resources exceed critical levels for extended periods:
- All analysis features disabled
- Memory buffers cleared
- Manual intervention required

## 🛡️ Safety Features

### Resource Protection
- Automatic threshold monitoring
- Progressive feature disabling
- Emergency cleanup procedures
- Memory pressure detection

### Data Protection
- Configuration backup/restore
- Graceful feature shutdown
- Buffer preservation during cleanup
- Recovery procedures

### System Stability
- Load balancing between CPU/GPU
- Queue size management
- Error tracking and recovery
- Thermal monitoring integration

## 📈 Performance Optimization

### Memory Management
- Three-tier memory allocation (short/mid/long-term)
- TTL-based cleanup
- Mathematical property-based categorization
- Thermal-aware allocation

### Processing Optimization
- Intelligent CPU/GPU load balancing
- Phase-aware processing mode selection
- Mining pattern-based optimization
- Backlog pressure monitoring

### Resource Efficiency
- Feature-based resource allocation
- Dynamic threshold adjustment
- Predictive cleanup scheduling
- Minimal overhead monitoring

## 🔍 Troubleshooting

### Common Issues

**High Memory Usage**:
```bash
# Check current usage
python tools/btc_processor_cli.py status

# Emergency cleanup
python tools/btc_processor_cli.py emergency-cleanup

# Disable memory-intensive features
python tools/btc_processor_cli.py disable mining_analysis
python tools/btc_processor_cli.py disable storage
```

**High CPU Usage**:
```bash
# Disable CPU-intensive analysis
python tools/btc_processor_cli.py disable mining_analysis
python tools/btc_processor_cli.py disable difficulty_tracking

# Set lower CPU limits
python tools/btc_processor_cli.py set-config --max-cpu 60
```

**System Unresponsive**:
```bash
# Force emergency procedures
python tools/btc_processor_cli.py emergency-cleanup

# Check if monitoring is active
python tools/btc_processor_cli.py status

# Restart monitoring if needed
python tools/btc_processor_cli.py stop-monitoring
python tools/btc_processor_cli.py start-monitoring
```

### Logs and Debugging

System logs provide detailed information:
- Feature enable/disable events
- Resource threshold violations
- Emergency procedures triggered
- Configuration changes

## 🎮 Demo and Examples

### Run Complete Demo
```bash
python examples/btc_processor_control_demo.py
```

This demonstrates:
- Basic feature control
- Bulk operations
- Threshold monitoring
- Emergency procedures
- Configuration management
- Live testing scenarios

### Mining Analysis Demo
```bash
python examples/btc_mining_analysis_demo.py
```

Shows the full mining analysis capabilities and how to control them.

## 🔧 Integration

### With Existing Systems

The control system integrates with:
- `BTCDataProcessor` - Main processing engine
- `BitcoinMiningAnalyzer` - Mining analysis engine
- `MemoryManager` - Memory allocation system
- `LoadBalancer` - CPU/GPU load balancing

### Custom Integration

```python
from core.btc_processor_controller import BTCProcessorController
from core.btc_data_processor import BTCDataProcessor

# Initialize with custom processor
processor = BTCDataProcessor('my_config.yaml')
controller = BTCProcessorController(processor)

# Custom threshold handling
def my_threshold_handler(metrics):
    if metrics.memory_usage_gb > 5.0:
        asyncio.run(controller.disable_feature('mining_analysis'))

# Custom monitoring
controller.system_thresholds['memory_warning'] = 5.0
```

## 📋 Best Practices

### Live Testing
1. **Pre-test preparation**: Disable analysis features
2. **Set conservative limits**: Lower thresholds during testing
3. **Monitor continuously**: Keep monitoring active
4. **Plan recovery**: Have restoration commands ready

### Production Use
1. **Gradual feature enabling**: Start with core features only
2. **Monitor trends**: Track resource usage patterns
3. **Regular cleanup**: Schedule periodic cleanups
4. **Configuration backup**: Save working configurations

### Resource Management
1. **Understand feature costs**: Know which features use most resources
2. **Use bulk controls**: Disable multiple features efficiently
3. **Monitor trends**: Watch for gradual resource increases
4. **Plan capacity**: Set limits based on system capabilities

## ⚡ Performance Impact

### Feature Resource Usage

**High Resource Features** (disable for live testing):
- `mining_analysis`: High CPU, moderate memory
- `block_timing`: Moderate CPU, low memory
- `difficulty_tracking`: Low CPU, moderate memory
- `nonce_sequences`: Moderate CPU, high memory

**Core Features** (keep enabled):
- `hash_generation`: Moderate CPU/GPU, low memory
- `load_balancing`: Low CPU, minimal memory
- `memory_management`: Minimal CPU, manages memory

**Optional Features**:
- `storage`: Low CPU, high disk I/O
- `monitoring`: Minimal CPU, low memory

### System Overhead

The control system itself uses minimal resources:
- ~50MB memory overhead
- ~1-2% CPU for monitoring
- Configurable monitoring intervals
- Optional web UI (~10MB additional)

## 🎯 Summary

The BTC Processor Control System ensures your system stays within the 10-50 GB information synthesis limit while maintaining maximum processing efficiency. Use it to:

- **Prevent overload** during live testing
- **Optimize resources** for specific tasks  
- **Manage complexity** of mining analysis
- **Ensure stability** during intensive processing
- **Maintain performance** within system limits

Choose the interface that works best for your workflow:
- **Web UI** for interactive control and monitoring
- **CLI** for scripting and automation
- **API** for programmatic integration
- **Automatic monitoring** for hands-off operation 