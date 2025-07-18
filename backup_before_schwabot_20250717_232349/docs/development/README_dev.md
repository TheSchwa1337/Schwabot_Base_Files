# Schwabot 0.046 - Development Guidelines

## 🛠️ Development Environment Setup

### Required Tools

```bash
# Core development tools
pip install black flake8 isort mypy

# Optional but recommended
pip install pre-commit pytest coverage
```

### Pre-Commit Configuration

Create `pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        args: [--line-length=88]
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --select=E,W,F,C,B]
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile, black]
```

Install pre-commit hooks:
```bash
pre-commit install
```

## 🔧 Code Quality Standards

### Flake8 Configuration

**Why Flake8 Matters for Schwabot:**
- **Mathematical Precision**: Indentation errors can misalign tick deltas
- **Runtime Determinism**: Whitespace affects hash alignment calculations
- **Pipeline Integrity**: Formatting errors break recursive execution paths
- **GPU/CPU Coordination**: Timing-sensitive operations require exact formatting

### Critical Error Codes

| Code | Description | Schwabot Impact |
|------|-------------|-----------------|
| **F401** | Unused import | Hidden signal variables disappear |
| **F841** | Unused variable | Mathematical components not used |
| **E501** | Line too long | Formula operator precedence issues |
| **E722** | Bare except | Silent failures in GAN filter |
| **C901** | Too complex | Untestable 40-line trade evaluation |
| **B006** | Mutable default | State corruption across backtests |

### Formatting Rules

```python
# ✅ CORRECT: Mathematical formula formatting
execution_confidence = (
    (triplet_entropy * theta_drift) +
    (coherence * loop_volatility) +
    profit_decay
)

# ❌ WRONG: Long line breaks formula readability
execution_confidence = (triplet_entropy * theta_drift) + (coherence * loop_volatility) + profit_decay
```

## 🧮 Mathematical Function Standards

### Function Signature Requirements

```python
def execution_confidence(
    triplet_entropy: float,      # T - Information rate from cursor patterns
    theta_drift: float,          # Δθ - Braid angle drift
    coherence: float,            # ε - Fractal coherence score
    loop_volatility: float,      # σ_f - Loop sum volatility
    profit_decay: float,         # τ_p - Time-weighted profit modifier
) -> float:
    """Calculate execution confidence scalar Ξ.
    
    Formula: Ξ = (T · Δθ) + (ε × σ_f) + τ_p
    
    Parameters
    ----------
    triplet_entropy : float
        Information rate from cursor patterns (T)
    theta_drift : float  
        Braid angle drift from geometric analysis (Δθ)
    coherence : float
        Fractal coherence score (ε) 
    loop_volatility : float
        Loop sum volatility from collapse engine (σ_f)
    profit_decay : float
        Time-weighted profit modifier (τ_p)
        
    Returns
    -------
    float
        Execution confidence scalar Ξ
    """
    return (triplet_entropy * theta_drift) + (coherence * loop_volatility) + profit_decay
```

### Required Documentation Elements

1. **Mathematical Formula**: LaTeX or ASCII representation
2. **Symbol Mapping**: Each parameter mapped to mathematical symbol
3. **Parameter Descriptions**: Clear explanation of each input
4. **Return Type**: Explicit return type annotation
5. **Error Handling**: Specific exception types, no bare except

## 🔄 Development Workflow

### Before Each Commit

```bash
# 1. Format code
black core/ --line-length=88

# 2. Sort imports
isort core/ --profile black

# 3. Check code quality
flake8 core/ --max-line-length=88 --select=E,W,F,C,B

# 4. Type checking
mypy core/

# 5. Run tests
python -m pytest tests/ -v

# 6. Validate runtime integrity
python runtime/validator.py
```

### Automated Quality Check Script

Create `scripts/check_quality.sh`:

```bash
#!/bin/bash
set -e

echo "🔍 Running code quality checks..."

echo "1. Formatting with Black..."
black core/ --line-length=88 --check --diff

echo "2. Sorting imports..."
isort core/ --profile black --check-only --diff

echo "3. Flake8 compliance..."
flake8 core/ --max-line-length=88 --select=E,W,F,C,B --statistics

echo "4. Type checking..."
mypy core/ --ignore-missing-imports

echo "5. Runtime validation..."
python runtime/validator.py

echo "✅ All quality checks passed!"
```

Make executable:
```bash
chmod +x scripts/check_quality.sh
```

## 🚨 Critical Formatting Rules

### Why Whitespace Matters in Schwabot

**Real-time recursive execution** leaves no room for indentation errors:

1. **Tick Path Recursion**: Misaligned indentation shifts phase triggers
2. **Hash Alignment**: Timing stalls from spacing errors affect GPU/CPU coordination
3. **Volume-Accelerated Loops**: Execution pathing reroutes mid-profit cycle
4. **Mathematical Precision**: Formula operator precedence depends on exact spacing

### Whitespace Violations That Break Trading Logic

```python
# ❌ DANGEROUS: Misaligned indentation in tick processing
def process_tick(tick_data):
    if tick_data.volume > threshold:
        calculate_execution_pressure()
     execute_trade()  # Wrong indentation = always executes!

# ✅ SAFE: Proper indentation maintains logic flow
def process_tick(tick_data):
    if tick_data.volume > threshold:
        calculate_execution_pressure()
        execute_trade()  # Correct indentation = conditional execution
```

### Line Length Impact on Mathematical Formulas

```python
# ❌ DANGEROUS: Long line breaks operator precedence
result = triplet_entropy * theta_drift + coherence * loop_volatility + profit_decay * time_factor + risk_adjustment

# ✅ SAFE: Proper line breaks preserve mathematical meaning
result = (
    triplet_entropy * theta_drift +
    coherence * loop_volatility +
    profit_decay * time_factor +
    risk_adjustment
)
```

## 🧪 Testing Standards

### Mathematical Function Testing

```python
def test_execution_confidence():
    """Test execution confidence calculation with known values."""
    # Known input values
    triplet_entropy = 0.75
    theta_drift = 0.12
    coherence = 0.68
    loop_volatility = 0.25
    profit_decay = 0.05
    
    # Expected output (calculated manually)
    expected = (0.75 * 0.12) + (0.68 * 0.25) + 0.05
    
    # Test function
    result = execution_confidence(
        triplet_entropy=triplet_entropy,
        theta_drift=theta_drift,
        coherence=coherence,
        loop_volatility=loop_volatility,
        profit_decay=profit_decay
    )
    
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
```

### Error Handling Testing

```python
def test_execution_confidence_error_handling():
    """Test execution confidence with invalid inputs."""
    # Test NaN handling
    result = execution_confidence(
        triplet_entropy=float('nan'),
        theta_drift=0.12,
        coherence=0.68,
        loop_volatility=0.25,
        profit_decay=0.05
    )
    
    # Should return safe fallback value
    assert not math.isnan(result), "Function should handle NaN inputs gracefully"
    assert 0.0 <= result <= 2.0, "Result should be in reasonable range"
```

## 🔐 Security & Reliability

### Input Validation Standards

```python
def validate_trading_inputs(
    price: float,
    volume: float,
    confidence: float,
) -> bool:
    """Validate trading inputs for safety."""
    # Price validation
    if not (0 < price < 1e6):  # Reasonable BTC price range
        logger.error(f"Invalid price: {price}")
        return False
    
    # Volume validation
    if not (0 <= volume < 1e6):  # Reasonable volume range
        logger.error(f"Invalid volume: {volume}")
        return False
    
    # Confidence validation
    if not (0.0 <= confidence <= 2.0):  # Confidence range
        logger.error(f"Invalid confidence: {confidence}")
        return False
    
    return True
```

### Error Logging Standards

```python
# ✅ CORRECT: Structured error logging
try:
    result = risky_calculation(market_data)
except ValueError as e:
    logger.warning(f"Calculation failed with invalid input: {e}")
    result = safe_fallback_value()
except ZeroDivisionError as e:
    logger.error(f"Division by zero in calculation: {e}")
    result = 0.0
except Exception as e:
    logger.critical(f"Unexpected error in calculation: {e}")
    result = emergency_fallback_value()

# ❌ WRONG: Silent failures
try:
    result = risky_calculation(market_data)
except:
    result = 0.0  # Silent failure - could hide critical bugs
```

## 📊 Performance Standards

### Latency Requirements

- **Signal Processing**: < 2ms per tick
- **Mathematical Core**: < 1ms per calculation
- **Decision Logic**: < 3ms per decision
- **Total Pipeline**: < 10ms end-to-end

### Memory Management

```python
# ✅ CORRECT: Limited history with auto-pruning
class SignalProcessor:
    def __init__(self):
        self.history = []
        self.max_history = 1000
    
    def add_signal(self, signal):
        self.history.append(signal)
        if len(self.history) > self.max_history:
            self.history = self.history[-500:]  # Keep recent half

# ❌ WRONG: Unbounded growth
class BadSignalProcessor:
    def __init__(self):
        self.history = []
    
    def add_signal(self, signal):
        self.history.append(signal)  # Memory leak!
```

## 🚀 Deployment Checklist

### Pre-Deployment Validation

```bash
# 1. Full validation suite
python runtime/validator.py

# 2. Performance benchmarks
python scripts/benchmark_pipeline.py

# 3. Backtesting validation
python scripts/run_backtest.py --validate

# 4. Integration tests
python -m pytest tests/integration/ -v

# 5. Security audit
python scripts/security_audit.py
```

### Production Monitoring

```python
# Health check endpoint
def health_check():
    """System health check for production monitoring."""
    checks = {
        "flake8_compliance": run_flake8_check(),
        "mathematical_integrity": validate_math_functions(),
        "memory_usage": check_memory_usage(),
        "latency_performance": measure_pipeline_latency(),
        "error_rate": calculate_error_rate(),
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": time.time(),
    }
```

## 📚 Documentation Standards

### Module Documentation

```python
#!/usr/bin/env python3
"""Module Name - Brief Description.

Detailed description of module purpose, mathematical foundations,
and integration with the broader Schwabot system.

Mathematical Foundation:
- Key formula 1: LaTeX or ASCII representation
- Key formula 2: Symbol definitions
- Key formula 3: Parameter ranges

Windows CLI compatible with comprehensive error handling.
"""
```