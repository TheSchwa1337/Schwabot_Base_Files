# Schwabot Architectural Standards

## Core Principles

**Add, Don't Replace**: All mathematical and architectural components should be extended, never removed or overwritten.

**Dual-Mode Compatibility**: Systems must work with both emoji and ASCII text for Windows CLI compatibility.

**Descriptive Naming**: All files, classes, and functions must use descriptive names that explain their purpose.

## Mathematical Framework Standards

### Core Components (Never Remove)
- `RecursiveIdentityFunction`: Ψₙ(x) = f(Ψₙ₋₁(x), Δ(t), T(Φₚ))
- `EntropyStabilizedFeedback`: Eₙ = min(∂Ψₙ/∂x + ∂Ψₙ/∂t, S_threshold)
- `InformationDensityMap`: Iₙ = ∫(Ψₙ(x)·Φ(x,t))dx
- `BTC256SHAPipeline`: Price processing and hash generation
- `FerrisWheelVisualizer`: Recursive state visualization

### Extension Pattern
```python
# ✅ Correct: Extend existing functionality
class NewDriftLayer(DriftShellEngine):
    def compute_new_drift(self, x: float) -> float:
        # New logic here
        return super().compute_drift(x) + new_component

# ❌ Wrong: Replace existing functionality
def compute_drift(x: float) -> float:
    # This overwrites existing logic
    return new_logic_only
```

## Emoji and Windows CLI Handling

### Emoji Mapping (Extensible)
```python
emoji_to_asic_mapping = {
    '🚨': '[ALERT]',
    '⚠️': '[WARNING]', 
    '✅': '[SUCCESS]',
    '❌': '[ERROR]',
    '🔄': '[PROCESSING]',
    '💰': '[PROFIT]',
    '📊': '[DATA]',
    '🔧': '[CONFIG]',
    '🎯': '[TARGET]',
    '⚡': '[FAST]',
    '🔍': '[SEARCH]',
    '📈': '[METRICS]',
    '🧠': '[INTELLIGENCE]',
    '🛡️': '[PROTECTION]',
    '🔥': '[HOT]',
    '❄️': '[COOL]',
    '⭐': '[STAR]',
    '🚀': '[LAUNCH]',
    '🎉': '[COMPLETE]',
    '💥': '[CRITICAL]',
    '🧪': '[TEST]',
    '🛠️': '[TOOLS]',
    '⚖️': '[BALANCE]',
    '🎨': '[VISUAL]',
    '🌟': '[EXCELLENT]'
}
```

### Adding New Emojis
When using new emojis, add them to the mapping with clear English equivalents:
```python
# Add to emoji_to_asic_mapping
'🦴': '[BONE]',  # Example for bone emoji
'🔄': '[SYNC]',  # Example for sync emoji
```

## Naming Schema Standards

### Test Files
**Pattern**: `test_[system]_[functionality].py`

**Examples**:
- `test_drift_shell_integration.py`
- `test_quantum_entropy_stabilization.py`
- `test_btc_pipeline_processing.py`
- `test_ferris_wheel_visualization.py`

**Avoid**:
- `test1.py`
- `simple_test.py`
- `quick_diagnostic.py`
- `run_tests_fixed.py`

### Component Classes
**Pattern**: `[Component][Type]`

**Types**:
- `Engine`: Core processing engines
- `Manager`: Resource and state management
- `Handler`: Event and error handling
- `Processor`: Data transformation
- `Controller`: System coordination

**Examples**:
- `DriftShellEngine`
- `QuantumStateManager`
- `WindowsCliCompatibilityHandler`
- `BTC256SHAProcessor`
- `FerrisWheelController`

### Functions
**Pattern**: `[action]_[target]_[context]`

**Examples**:
- `compute_unified_drift_field`
- `allocate_ring_drift`
- `generate_quantum_hash`
- `validate_mathematical_operation`

## Configuration Standards

### Mathematical Framework Config
All parameters must be configurable through `config/mathematical_framework_config.py`:

```python
@dataclass
class RecursionConfig:
    max_depth: int = 50
    convergence_threshold: float = 1e-6
    memoization_cache_size: int = 128
    enable_depth_guards: bool = True
    enable_convergence_checking: bool = True
    enable_memoization: bool = True
```

### No Hardcoded Values
```python
# ❌ Wrong: Hardcoded values
def compute_drift(x: float) -> float:
    return x * 1.618033988749  # Magic number

# ✅ Correct: Config-driven
def compute_drift(x: float) -> float:
    return x * self.config.drift_shell.psi_infinity
```

## Error Handling Standards

### Windows CLI Compatibility
```python
@staticmethod
def safe_print(message: str, use_emoji: bool = True) -> str:
    if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
        for emoji, asic_replacement in emoji_to_asic_mapping.items():
            message = message.replace(emoji, asic_replacement)
    return message
```

### Structured Exception Handling
```python
# ❌ Wrong: Bare exceptions
try:
    result = risky_operation()
except:
    pass

# ✅ Correct: Structured handling
try:
    result = risky_operation()
except Exception as e:
    error_message = self.cli_handler.safe_format_error(e, "operation_name")
    self.cli_handler.log_safe(self.logger, 'error', error_message)
    raise
```

## Type Annotation Standards

### Required Annotations
All functions must have return type annotations:
```python
def compute_drift_field(x: float, y: float, z: float, time: float) -> float:
    return drift_value

def process_price_data(price: float, timestamp: float) -> Dict[str, Any]:
    return processing_result
```

### Mathematical Types
Use types from `core.type_defs`:
```python
from core.type_defs import (
    DriftCoefficient, Entropy, Tensor, Vector, Matrix,
    QuantumState, EnergyLevel, Price, Volume
)
```

## Integration Standards

### BTC256SH-A Pipeline
- Price history management
- Hash generation with unified framework
- Mathematical analysis integration
- Drift field computation

### Ferris Wheel Visualizer
- Recursive state visualization
- Entropy stabilization display
- Data export capabilities
- Real-time monitoring

### Memory and Cataloging
- Short-term: Recent trade data
- Mid-term: Strategy performance
- Long-term: Historical patterns
- Dynamic bucket allocation

## Compliance Checklist

- [ ] All core math components preserved
- [ ] Emoji mapping updated for new symbols
- [ ] Descriptive naming used throughout
- [ ] Configuration-driven parameters
- [ ] Windows CLI compatibility implemented
- [ ] Type annotations complete
- [ ] Structured error handling
- [ ] No hardcoded magic numbers
- [ ] Extensible architecture maintained

## Maintenance Guidelines

1. **When adding new math**: Extend existing classes, don't replace
2. **When using new emojis**: Add to mapping immediately
3. **When creating files**: Follow naming schema
4. **When encountering errors**: Update handler mappings
5. **When optimizing**: Preserve core functionality
6. **When documenting**: Use Markdown, not deletion

This document serves as the single source of truth for maintaining Schwabot's architectural integrity and mathematical framework standards. 