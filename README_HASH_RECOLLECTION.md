# Hash Recollection System v0.045

Advanced pattern recognition and memory optimization system for BTC trading, utilizing entropy-based hash algorithms for real-time market analysis.

## 🎯 Overview

The Hash Recollection System implements sophisticated pattern recognition through:
- **Shannon Entropy Analysis** across multiple time windows
- **42-bit Pattern Encoding** for high-precision pattern matching  
- **GPU-Accelerated Hash Computation** for real-time processing
- **Multi-scale Variance Tracking** for pattern stability analysis
- **Tetragram Matrix** for 3D entropy state tracking

## 🏗️ Architecture

### Core Components

```
Hash Recollection System
├── EntropyTracker      - Shannon entropy calculation & normalization
├── BitOperations       - 42-bit pattern encoding & phase extraction
├── PatternUtils        - Entry/exit rules & confidence scoring
├── GPU Worker          - Parallel hash computation
└── CPU Worker          - Pattern matching & signal generation
```

### Mathematical Framework

The system implements the following mathematical rules:

1. **Log Return Normalization**: `z_t = (r_t - μ_r) / σ_r`
2. **Shannon Entropy**: `H = -Σ p_i * log2(p_i)`
3. **42-bit Encoding**: `b42 = (entropy * 1e12) & ((1 << 42) - 1)`
4. **Phase Extraction**: Extract 4-bit, 8-bit, and 42-bit components
5. **Entry/Exit Rules**: Based on density thresholds and pattern strength
6. **Hash Similarity**: `S = 0.5 * [1 - Ham(h1,h2)/64] + 0.5 * [1 - ||E1-E2||/max||E||]`

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd hash-recollection-system

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU support
pip install cupy-cuda11x  # For CUDA 11.x
```

### Basic Usage

```python
from core import HashRecollectionSystem

# Initialize system
hrs = HashRecollectionSystem()

# Register signal callback
def handle_signal(signal):
    print(f"Signal: {signal['action']} @ ${signal['price']:.2f}")

hrs.register_signal_callback(handle_signal)

# Start processing
hrs.start()

# Process market data
hrs.process_tick(price=35000.0, volume=1.5)

# Get system metrics
metrics = hrs.get_pattern_metrics()
print(f"Patterns detected: {metrics['patterns_detected']}")

# Stop system
hrs.stop()
```

### Run Demo

```bash
python examples/hash_recollection_demo.py
```

## 📊 Configuration

Create a `config.yaml` file:

```yaml
system:
  gpu_enabled: true
  gpu_batch_size: 1000
  sync_interval: 100

entropy:
  history_length: 1000
  window_sizes:
    short: 5
    mid: 16
    long: 64

patterns:
  density_entry: 0.57
  density_exit: 0.42
  variance_entry: 0.002
  variance_exit: 0.007
  confidence_min: 0.7
```

## 🔧 Components Deep Dive

### EntropyTracker

Calculates Shannon entropy across multiple time windows:

```python
from core import EntropyTracker

tracker = EntropyTracker(maxlen=1000)
state = tracker.update(price=35000, volume=1.5, timestamp=time.time())

print(f"Price entropy: {state.price_entropy:.3f}")
print(f"Volume entropy: {state.volume_entropy:.3f}")

# Multi-window analysis
entropies = tracker.get_multi_window_entropies()
```

### BitOperations

Handles 42-bit pattern encoding and analysis:

```python
from core import BitOperations

bit_ops = BitOperations()

# Encode entropy to 42-bit pattern
pattern = bit_ops.calculate_42bit_float(entropy=3.14159)

# Extract phase components
b4, b8, b42 = bit_ops.extract_phase_bits(pattern)

# Analyze pattern strength
analysis = bit_ops.analyze_bit_pattern(pattern)
print(f"Pattern strength: {analysis['pattern_strength']:.3f}")
```

### PatternUtils

Implements trading logic and signal generation:

```python
from core import PatternUtils, PhaseState

pattern_utils = PatternUtils()

# Create phase state
phase_state = PhaseState(
    b4=0x6, b8=0xFF, b42=pattern,
    tier=3, density=0.65, timestamp=time.time()
)

# Check entry conditions
is_entry, reasons = pattern_utils.is_entry_phase(phase_state, analysis)

# Generate trading signal
pattern_match = pattern_utils.check_pattern_match(
    hash_value, phase_state, analysis, hash_database
)
```

## 📈 Trading Signals

The system generates signals based on pattern analysis:

### Entry Signals
- **Conditions**: 4-bit pattern ∈ {0x3, 0x5, 0x6, 0x9, 0xA, 0xC}
- **Density**: > 0.57
- **Variance**: < 0.002
- **Confidence**: > 0.7

### Exit Signals  
- **Conditions**: Pattern not in entry keys OR density < 0.42 OR variance > 0.007
- **Tier**: ≤ 1 (exit zone)

### Signal Structure
```json
{
  "action": "entry|exit|hold|wait",
  "confidence": 0.85,
  "price": 35000.0,
  "tier": 3,
  "density": 0.65,
  "size": 0.1,
  "stop_loss": 34300.0,
  "take_profit": 36050.0,
  "reasons": ["Entry signal: b4=6, density=0.650..."]
}
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python tests/test_hash_recollection_system.py

# Run with coverage
python -m pytest tests/ --cov=core --cov-report=html
```

## ⚡ Performance

### Metrics
- **Hash Rate**: ~10,000 hashes/second (GPU)
- **Latency**: <1ms pattern matching
- **Memory**: <100MB for 1M pattern database
- **CPU Usage**: <5% (with GPU acceleration)

### Optimization Tips
1. **Enable GPU**: Set `gpu_enabled: true` in config
2. **Batch Size**: Increase `gpu_batch_size` for throughput
3. **Memory**: Limit `history_length` for memory efficiency
4. **Threads**: Adjust worker thread count based on CPU cores

## 🔮 Advanced Features

### Pattern Similarity Scoring
Combines hash distance and entropy similarity:
```python
similarity = pattern_utils.compare_hashes(h1, h2, entropy1, entropy2)
```

### Confidence Calculation
Uses frequency-based cluster analysis:
```python
confidence = pattern_utils.calculate_confidence(
    hash_value, similar_hashes, hash_database, pattern_analysis
)
```

### Latency Compensation
Adjusts thresholds based on measured latency:
```python
adjusted = pattern_utils.adjust_for_latency(threshold, 'exit')
```

## 📋 API Reference

### HashRecollectionSystem

Main system class for pattern recognition:

- `start()` - Start worker threads
- `stop()` - Stop system gracefully  
- `process_tick(price, volume, timestamp)` - Process market data
- `register_signal_callback(callback)` - Register signal handler
- `get_pattern_metrics()` - Get system metrics
- `get_system_report()` - Generate comprehensive report

### EntropyTracker

- `update(price, volume, timestamp) -> EntropyState`
- `get_multi_window_entropies() -> Dict`
- `get_latest_state() -> EntropyState`
- `get_entropy_vector() -> np.ndarray`

### BitOperations

- `calculate_42bit_float(entropy) -> int`
- `extract_phase_bits(pattern) -> Tuple[int, int, int]`
- `calculate_bit_density(pattern) -> float`
- `analyze_bit_pattern(pattern) -> Dict`

### PatternUtils

- `is_entry_phase(phase_state, analysis) -> Tuple[bool, List[str]]`
- `is_exit_phase(phase_state, analysis) -> Tuple[bool, List[str]]`
- `check_pattern_match(...) -> PatternMatch`
- `calculate_confidence(...) -> float`

## 🛠️ Development

### Project Structure
```
hash-recollection-system/
├── core/                 # Core system modules
│   ├── __init__.py
│   ├── hash_recollection.py
│   ├── entropy_tracker.py
│   ├── bit_operations.py
│   └── pattern_utils.py
├── examples/             # Usage examples
├── tests/               # Test suite
├── config/              # Configuration files
└── requirements.txt     # Dependencies
```

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Create Pull Request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings for all public methods
- Maintain >90% test coverage

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

- **Documentation**: See inline docstrings and type hints
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join community discussions

## 🔗 Related Projects

- **Schwabot**: Main trading system integration
- **DLT Waveform Engine**: Advanced signal processing
- **Profit Cycle Navigator**: Portfolio management

---

*Built with ❤️ for algorithmic trading* 