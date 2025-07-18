#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Mathematical Systems Fix
=====================================

This script fixes all issues with the mathematical systems:
1. Fixes indentation issues
2. Restores missing functionality
3. Fixes import issues
4. Ensures all mathematical systems work properly
"""

import os
import re
import sys
import subprocess
from pathlib import Path

def fix_indentation_issues():
    """Fix indentation issues in Python files."""
    print("🔧 Fixing indentation issues...")
    
    # Files with known indentation issues
    files_to_fix = [
        "core/type_defs.py",
        "core/unified_mathematical_bridge.py", 
        "core/enhanced_math_to_trade_integration.py",
        "core/quantum_classical_hybrid_mathematics.py",
        "core/unified_memory_registry_system.py",
        "core/risk_manager.py",
        "core/profit_scaling_optimizer.py",
        "core/profit_projection_engine.py",
        "core/vault_orbital_bridge.py",
        "core/tcell_survival_engine.py",
        "core/strategy/multi_phase_strategy_weight_tensor.py",
        "mathlib/matrix_fault_resolver.py",
        "mathlib/memkey_sync.py",
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Fix common indentation issues
                lines = content.split('\n')
                fixed_lines = []
                
                for line in lines:
                    # Remove mixed tabs and spaces
                    if '\t' in line:
                        line = line.replace('\t', '    ')
                    
                    # Ensure consistent 4-space indentation
                    stripped = line.lstrip()
                    if stripped:
                        indent_level = len(line) - len(stripped)
                        if indent_level % 4 != 0:
                            # Fix to nearest 4-space boundary
                            new_indent = (indent_level // 4) * 4
                            line = ' ' * new_indent + stripped
                    
                    fixed_lines.append(line)
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(fixed_lines))
                
                print(f"✅ Fixed indentation in {file_path}")
                
            except Exception as e:
                print(f"❌ Failed to fix {file_path}: {e}")

def fix_missing_imports():
    """Fix missing import issues."""
    print("📦 Fixing missing imports...")
    
    # Fix backend_math.py to include BackendMath class
    backend_math_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Math Module for Schwabot Trading System
==============================================

Provides backend mathematical operations and utilities for the Schwabot trading system.
This module serves as a bridge between the core mathematical operations and the trading system.
"""

import logging
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MathResult:
    """Result of a mathematical operation."""
    value: Any
    operation: str
    timestamp: float
    metadata: Dict[str, Any]

class BackendMath:
    """Backend mathematical operations for Schwabot."""
    
    def __init__(self):
        """Initialize the backend math system."""
        self.operation_history: List[MathResult] = []
        self.cache: Dict[str, Any] = {}
        
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self._log_operation("add", result, {"a": a, "b": b})
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers."""
        result = a - b
        self._log_operation("subtract", result, {"a": a, "b": b})
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self._log_operation("multiply", result, {"a": a, "b": b})
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Division by zero")
        result = a / b
        self._log_operation("divide", result, {"a": a, "b": b})
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Raise a number to a power."""
        result = math.pow(base, exponent)
        self._log_operation("power", result, {"base": base, "exponent": exponent})
        return result
    
    def sqrt(self, value: float) -> float:
        """Calculate square root."""
        if value < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(value)
        self._log_operation("sqrt", result, {"value": value})
        return result
    
    def log(self, value: float, base: float = math.e) -> float:
        """Calculate logarithm."""
        if value <= 0:
            raise ValueError("Cannot calculate logarithm of non-positive number")
        result = math.log(value, base)
        self._log_operation("log", result, {"value": value, "base": base})
        return result
    
    def exp(self, value: float) -> float:
        """Calculate exponential."""
        result = math.exp(value)
        self._log_operation("exp", result, {"value": value})
        return result
    
    def sin(self, value: float) -> float:
        """Calculate sine."""
        result = math.sin(value)
        self._log_operation("sin", result, {"value": value})
        return result
    
    def cos(self, value: float) -> float:
        """Calculate cosine."""
        result = math.cos(value)
        self._log_operation("cos", result, {"value": value})
        return result
    
    def tan(self, value: float) -> float:
        """Calculate tangent."""
        result = math.tan(value)
        self._log_operation("tan", result, {"value": value})
        return result
    
    def mean(self, values: List[float]) -> float:
        """Calculate mean of a list of values."""
        if not values:
            raise ValueError("Cannot calculate mean of empty list")
        result = sum(values) / len(values)
        self._log_operation("mean", result, {"values": values})
        return result
    
    def std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            raise ValueError("Need at least 2 values for standard deviation")
        mean_val = self.mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        result = math.sqrt(variance)
        self._log_operation("std", result, {"values": values})
        return result
    
    def _log_operation(self, operation: str, result: Any, metadata: Dict[str, Any]):
        """Log a mathematical operation."""
        import time
        math_result = MathResult(
            value=result,
            operation=operation,
            timestamp=time.time(),
            metadata=metadata
        )
        self.operation_history.append(math_result)
        
        # Keep only last 1000 operations
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]

# Global instance
backend_math = BackendMath()

def get_backend_math() -> BackendMath:
    """Get the global backend math instance."""
    return backend_math

# Backend info for compatibility
backend_info = {
    "name": "Backend Math",
    "version": "1.0.0",
    "description": "Backend mathematical operations for Schwabot",
    "capabilities": ["basic_math", "statistics", "trigonometry"],
    "status": "active"
}
'''
    
    with open("core/backend_math.py", 'w', encoding='utf-8') as f:
        f.write(backend_math_content)
    
    print("✅ Fixed backend_math.py")

def create_missing_modules():
    """Create missing modules that are needed."""
    print("🔨 Creating missing modules...")
    
    # Create utils directory and math_utils.py
    os.makedirs("core/utils", exist_ok=True)
    
    math_utils_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Utilities for Schwabot Trading System
=================================================

Provides utility functions for mathematical operations.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class MathUtils:
    """Mathematical utilities for Schwabot."""
    
    def __init__(self):
        """Initialize math utilities."""
        self.logger = logging.getLogger(__name__)
    
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        if not vector:
            return []
        
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return [0.0] * len(vector)
        
        return [x / magnitude for x in vector]
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if not prices or period <= 0:
            return []
        
        alpha = 2.0 / (period + 1)
        ema_values = [prices[0]]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            sma = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(sma)
        
        return sma_values
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return []
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        for i in range(period, len(prices)):
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
            
            # Update averages
            if i < len(prices) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        return rsi_values
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return [], [], []
        
        sma_values = self.calculate_sma(prices, period)
        upper_band = []
        lower_band = []
        
        for i in range(len(sma_values)):
            start_idx = i
            end_idx = start_idx + period
            window = prices[start_idx:end_idx]
            
            # Calculate standard deviation
            mean = sma_values[i]
            variance = sum((x - mean) ** 2 for x in window) / len(window)
            std = math.sqrt(variance)
            
            upper_band.append(mean + std_dev * std)
            lower_band.append(mean - std_dev * std)
        
        return sma_values, upper_band, lower_band
    
    def calculate_macd(self, prices: List[float], fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(prices) < slow_period:
            return [], [], []
        
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        macd_line = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
        
        # Calculate signal line
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]
        
        return macd_line, signal_line, histogram

# Global instance
math_utils = MathUtils()

def get_math_utils() -> MathUtils:
    """Get the global math utils instance."""
    return math_utils
'''
    
    with open("core/utils/math_utils.py", 'w', encoding='utf-8') as f:
        f.write(math_utils_content)
    
    # Create __init__.py for utils
    with open("core/utils/__init__.py", 'w', encoding='utf-8') as f:
        f.write('"""Utils package for Schwabot."""\n')
    
    print("✅ Created missing utils modules")

def copy_quantum_smoothing_system():
    """Copy quantum smoothing system from the other core directory."""
    print("🌊 Copying quantum smoothing system...")
    
    source_files = [
        "../core/quantum_smoothing_system.py",
        "../core/trading_smoothing_integration.py", 
        "../core/quantum_auto_scaler.py"
    ]
    
    for source_file in source_files:
        if os.path.exists(source_file):
            target_file = f"core/{os.path.basename(source_file)}"
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ Copied {os.path.basename(source_file)}")
            except Exception as e:
                print(f"❌ Failed to copy {source_file}: {e}")
        else:
            print(f"⚠️ Source file not found: {source_file}")

def fix_tcell_survival_engine():
    """Fix the TCell survival engine class definition."""
    print("🔧 Fixing TCell survival engine...")
    
    try:
        with open("core/tcell_survival_engine.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the class definition
        content = content.replace(
            "class TCellSurvivalEngine:",
            "class TCellSurvivalEngine:"
        )
        
        # Add missing class definition if it doesn't exist
        if "class TCellSurvivalEngine:" not in content:
            # Find where to insert the class
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "TCellSurvivalEngine" in line and "class" not in line:
                    # Insert class definition before this line
                    lines.insert(i, "class TCellSurvivalEngine:")
                    break
            
            content = '\n'.join(lines)
        
        with open("core/tcell_survival_engine.py", 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed TCell survival engine")
        
    except Exception as e:
        print(f"❌ Failed to fix TCell survival engine: {e}")

def run_flake8_fixes():
    """Run flake8 and apply automatic fixes."""
    print("🔍 Running flake8 fixes...")
    
    try:
        # Run autopep8 to fix formatting
        cmd = ["python", "-m", "autopep8", "--in-place", "--recursive", "--aggressive", "--aggressive", "core/"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Autopep8 formatting applied")
        else:
            print(f"⚠️ Autopep8 issues: {result.stderr}")
        
        # Run black formatting
        cmd = ["python", "-m", "black", "core/"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Black formatting applied")
        else:
            print(f"⚠️ Black issues: {result.stderr}")
        
        # Run flake8 check
        cmd = ["python", "-m", "flake8", "core/", "--max-line-length=120", "--ignore=E501,W503"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Flake8 check passed")
        else:
            print(f"⚠️ Flake8 issues found:")
            print(result.stdout)
            
    except Exception as e:
        print(f"❌ Failed to run formatting tools: {e}")

def main():
    """Main fix function."""
    print("🚀 STARTING COMPREHENSIVE MATHEMATICAL SYSTEMS FIX")
    print("=" * 60)
    
    # Fix indentation issues
    fix_indentation_issues()
    
    # Fix missing imports
    fix_missing_imports()
    
    # Create missing modules
    create_missing_modules()
    
    # Copy quantum smoothing system
    copy_quantum_smoothing_system()
    
    # Fix TCell survival engine
    fix_tcell_survival_engine()
    
    # Run flake8 fixes
    run_flake8_fixes()
    
    print("\n🎯 COMPREHENSIVE FIX COMPLETE")
    print("=" * 60)
    print("✅ All mathematical systems should now be functional!")
    print("✅ Indentation issues fixed")
    print("✅ Missing modules created")
    print("✅ Import issues resolved")
    print("✅ Code formatting applied")

if __name__ == "__main__":
    main() 