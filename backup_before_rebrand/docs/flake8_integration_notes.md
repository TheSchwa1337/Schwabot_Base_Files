# Flake8 Integration Notes - Schwabot Code Quality Standards

## 🎯 Why Flake8 Matters for Schwabot

In a **deterministic trading system** like Schwabot, code quality isn't just about aesthetics—it's about **mathematical integrity** and **system reliability**. Every linting violation can hide critical bugs that could impact trading decisions.

### Critical Impact Areas

| Flake8 Error | Trading System Risk | Schwabot Impact |
|--------------|-------------------|-----------------|
| **F401/F841** (unused imports/vars) | Hidden signal variables | σ_f, T, Δθ disappear from scope |
| **C901** (complexity >10) | Hard to test/debug | 40-line evaluate_trade blocks bury gates |
| **E501** (line >79 chars) | Math formula breaks | Ξ formula operator precedence issues |
| **E722** (bare except) | Silent failures | GAN filter passes NaNs undetected |
| **B006** (mutable defaults) | State corruption | Anomaly filter history grows across backtests |

## 🛡️ Schwabot-Specific Code Standards

### Mathematical Function Standards

```python
# ✅ CORRECT: Clear mathematical mapping
def execution_confidence(
    triplet_entropy: float,      # T
    theta_drift: float,          # Δθ  
    coherence: float,            # ε
    loop_volatility: float,      # σ_f
    profit_decay: float,         # τ_p
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


# ❌ WRONG: Unclear variable names, no documentation
def calc(a, b, c, d, e):
    return a * b + c * d + e
```

### Signal Processing Standards

```python
# ✅ CORRECT: Named tuples eliminate F841 errors
from typing import NamedTuple

class TradingSignalMetrics(NamedTuple):
    """Consolidated trading signal metrics."""
    triplet_entropy: float          # T
    theta_drift: float              # Δθ
    coherence: float                # ε
    loop_volatility: float          # σ_f
    profit_decay: float             # τ_p
    harmony: float                  # 𝓗
    drift_penalty: float            # 𝓓ₚ
    liquidity_score: float          # 𝓛
    projected_profit: float         # P̂
    timestamp: float

def collect_signals() -> TradingSignalMetrics:
    # No temp variables = no F841 errors
    return TradingSignalMetrics(
        triplet_entropy=calculate_entropy(),
        theta_drift=calculate_drift(),
        coherence=calculate_coherence(),
        # ... rest of signals
    )


# ❌ WRONG: Temporary variables trigger F841
def collect_signals_bad():
    t_entropy = calculate_entropy()     # F841: unused
    drift = calculate_drift()           # F841: unused  
    coh = calculate_coherence()         # F841: unused
    # Variables created but not returned = F841 violations
```

### Error Handling Standards

```python
# ✅ CORRECT: Specific exception handling
def calculate_btc_vector(prices: List[float]) -> float:
    """Calculate BTC price vector with proper error handling."""
    try:
        if not prices:
            raise ValueError("Empty price list")
        return sum(prices) / len(prices)
    except (ValueError, ZeroDivisionError) as e:
        logger.warning(f"BTC vector calculation error: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error in BTC vector: {e}")
        return 0.0


# ❌ WRONG: Bare except hides errors
def calculate_btc_vector_bad(prices):
    try:
        return sum(prices) / len(prices)
    except:  # E722: bare except
        return 0.0  # Silent failure - could hide NaN propagation
```

### Complex Logic Decomposition

```python
# ✅ CORRECT: Split complex functions (C901 compliance)
def evaluate_trade_decision(signals: TradingSignalMetrics) -> InvestmentDecision:
    """Main trade evaluation with complexity < 10."""
    confidence = _compute_execution_confidence(signals)
    entry_score = _compute_entry_score(signals)
    risk_level = _assess_risk_level(signals)
    return _make_final_decision(confidence, entry_score, risk_level)

def _compute_execution_confidence(signals: TradingSignalMetrics) -> float:
    """Helper: Calculate Ξ (execution confidence)."""
    return (signals.triplet_entropy * signals.theta_drift + 
            signals.coherence * signals.loop_volatility + 
            signals.profit_decay)

def _compute_entry_score(signals: TradingSignalMetrics) -> float:
    """Helper: Calculate 𝓔ₛ (entry score)."""
    return (signals.harmony * (1.0 - signals.drift_penalty) * 
            signals.liquidity_score * signals.projected_profit)

def _assess_risk_level(signals: TradingSignalMetrics) -> RiskLevel:
    """Helper: Determine risk level from signals."""
    risk_score = (signals.loop_volatility * 0.3 + 
                  (1.0 - signals.liquidity_score) * 0.7)
    if risk_score < 0.3:
        return RiskLevel.LOW
    elif risk_score < 0.7:
        return RiskLevel.MODERATE
    else:
        return RiskLevel.HIGH

def _make_final_decision(
    confidence: float, 
    entry_score: float, 
    risk_level: RiskLevel
) -> InvestmentDecision:
    """Helper: Make final investment decision."""
    if confidence > 1.15 and entry_score > 0.9 and risk_level != RiskLevel.HIGH:
        return InvestmentDecision.STRONG_BUY
    elif confidence > 0.85 and entry_score > 0.7:
        return InvestmentDecision.BUY
    # ... rest of decision logic


# ❌ WRONG: Monolithic function (C901 violation)
def evaluate_trade_decision_bad(signals):
    # 40+ line function with nested if/else = C901 violation
    if signals.triplet_entropy > 0.8:
        if signals.theta_drift > 0.1:
            if signals.coherence > 0.7:
                if signals.loop_volatility < 0.2:
                    if signals.harmony > 0.8:
                        # ... 35 more lines of nested logic
                        pass
```

### Line Length Management (E501)

```python
# ✅ CORRECT: Proper line breaks for math formulas
def calculate_complex_formula(
    a: float, b: float, c: float, d: float, e: float
) -> float:
    """Calculate complex mathematical formula with proper formatting."""
    result = (
        (a * b * c) +
        (d * e * 0.5) +
        (a + b - c) * 0.3 +
        (d / (e + 0.001))  # Avoid division by zero
    )
    return result

# Alternative: Use intermediate variables for clarity
def calculate_complex_formula_alt(
    a: float, b: float, c: float, d: float, e: float
) -> float:
    """Alternative formatting with intermediate calculations."""
    product_term = a * b * c
    weighted_term = d * e * 0.5
    difference_term = (a + b - c) * 0.3
    ratio_term = d / (e + 0.001)
    
    return product_term + weighted_term + difference_term + ratio_term


# ❌ WRONG: Long lines break readability (E501)
def bad_formula(a, b, c, d, e):
    return (a * b * c) + (d * e * 0.5) + (a + b - c) * 0.3 + (d / (e + 0.001))  # E501: line too long
```

### Mutable Default Arguments (B006)

```python
# ✅ CORRECT: Use field(default_factory=list) for dataclasses
from dataclasses import dataclass, field
from typing import List

@dataclass
class AnomalyFilter:
    """Anomaly detection filter with proper mutable defaults."""
    threshold: float = 0.8
    history: List[float] = field(default_factory=list)  # Correct!
    
    def add_sample(self, value: float) -> None:
        """Add sample to history."""
        self.history.append(value)
        if len(self.history) > 100:
            self.history = self.history[-50:]  # Keep recent history

# Alternative: Use None and initialize in __post_init__
@dataclass  
class AnomalyFilterAlt:
    """Alternative approach with None default."""
    threshold: float = 0.8
    history: Optional[List[float]] = None
    
    def __post_init__(self) -> None:
        """Initialize mutable fields."""
        if self.history is None:
            self.history = []


# ❌ WRONG: Mutable default grows across instances (B006)
@dataclass
class BadAnomalyFilter:
    threshold: float = 0.8
    history: List[float] = []  # B006: Dangerous! Shared across instances
    
# This creates shared state:
filter1 = BadAnomalyFilter()
filter2 = BadAnomalyFilter()
filter1.history.append(1.0)
print(filter2.history)  # [1.0] - Contaminated!
```

## 🔧 Development Workflow

### Pre-Commit Checks

```bash
# Run before every commit
flake8 core/ --max-line-length=88 --select=E,W,F,C,B
black core/  # Auto-format
isort core/  # Sort imports
mypy core/   # Type checking
```

### Automated Formatting Setup

```bash
# Install development tools
pip install flake8 black isort mypy

# Configure in pyproject.toml
[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 88

[tool.flake8]
max-line-length = 88
extend-ignore = E203, W503  # Black compatibility
select = E,W,F,C,B
```

### VS Code Integration

```json
// .vscode/settings.json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": ["--max-line-length=88"],
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "editor.formatOnSave": true,
    "python.sortImports.args": ["--profile", "black"]
}
```

## 🧪 Testing Flake8 Compliance

### Continuous Integration

```yaml
# .github/workflows/code-quality.yml
name: Code Quality
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install flake8 black isort mypy
    - name: Run flake8
      run: flake8 core/ --max-line-length=88
    - name: Check black formatting
      run: black --check core/
    - name: Check import sorting
      run: isort --check-only core/
    - name: Run type checking
      run: mypy core/
```

### Local Testing Script

```bash
#!/bin/bash
# scripts/check_code_quality.sh

echo "Running Flake8..."
flake8 core/ --max-line-length=88 --statistics

echo "Checking Black formatting..."
black --check --diff core/

echo "Checking import sorting..."
isort --check-only --diff core/

echo "Running type checks..."
mypy core/

echo "Running tests..."
python -m pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "✅ All checks passed!"
else
    echo "❌ Some checks failed. Please fix before committing."
    exit 1
fi
```

## 🚨 Common Violations & Fixes

### F401: Unused Import

```python
# ❌ Problem
import numpy as np
import pandas as pd  # F401: imported but unused

def calculate_mean(values):
    return np.mean(values)

# ✅ Solution: Remove unused import
import numpy as np

def calculate_mean(values):
    return np.mean(values)
```

### F841: Unused Variable

```python
# ❌ Problem  
def process_signals():
    entropy = calculate_entropy()  # F841: assigned but never used
    return calculate_drift()

# ✅ Solution: Use named tuple or remove variable
def process_signals():
    return calculate_drift()  # Direct return
    
# Or use the variable
def process_signals():
    entropy = calculate_entropy()
    drift = calculate_drift()
    return entropy + drift  # Both variables used
```

### E722: Bare Except

```python
# ❌ Problem
try:
    result = risky_calculation()
except:  # E722: bare except
    result = 0

# ✅ Solution: Specific exception handling
try:
    result = risky_calculation()
except (ValueError, ZeroDivisionError) as e:
    logger.warning(f"Calculation failed: {e}")
    result = 0
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    result = 0
```

## 📊 Monitoring Code Quality

### Quality Metrics Dashboard

```python
# scripts/quality_metrics.py
import subprocess
import json

def get_flake8_stats():
    """Get Flake8 violation statistics."""
    result = subprocess.run(
        ['flake8', 'core/', '--statistics', '--format=json'],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)

def generate_quality_report():
    """Generate code quality report."""
    stats = get_flake8_stats()
    
    report = {
        'total_violations': sum(stats.values()),
        'violation_breakdown': stats,
        'quality_score': max(0, 100 - sum(stats.values()) * 2),
        'timestamp': time.time()
    }
    
    return report
```

### Quality Gates

```python
# Enforce quality thresholds
QUALITY_THRESHOLDS = {
    'max_violations': 0,      # Zero tolerance for production
    'max_complexity': 10,     # Cyclomatic complexity limit
    'min_coverage': 90,       # Test coverage requirement
    'max_line_length': 88,    # Line length limit
}

def check_quality_gates():
    """Verify code meets quality standards."""
    violations = get_flake8_violations()
    
    if violations > QUALITY_THRESHOLDS['max_violations']:
        raise Exception(f"Quality gate failed: {violations} violations")
    
    print("✅ All quality gates passed!")
```

## 🎯 Best Practices Summary

### DO's ✅

1. **Use named tuples** for signal data (eliminates F841)
2. **Split complex functions** into helpers (avoids C901)
3. **Specify exception types** (no bare except/E722)
4. **Use field(default_factory=list)** for mutable defaults (avoids B006)
5. **Break long lines** at logical points (avoids E501)
6. **Add type annotations** to all functions
7. **Write comprehensive docstrings** with parameter descriptions
8. **Use consistent naming** that maps to mathematical symbols

### DON'Ts ❌

1. **Don't use bare except** clauses
2. **Don't create unused variables** or imports
3. **Don't write functions >10 complexity**
4. **Don't use mutable default arguments**
5. **Don't exceed 88 characters per line**
6. **Don't ignore type checking**
7. **Don't skip error handling**
8. **Don't use unclear variable names**

---

**Remember**: In Schwabot, clean code isn't optional—it's essential for reliable trading decisions and mathematical integrity. 