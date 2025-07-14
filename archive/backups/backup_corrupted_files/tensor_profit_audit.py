#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor & Profit Module Audit Script
==================================
Scans all tensor and profit modules for:
- MathOrchestrator usage
- MathResultCache usage
- Real mathematical logic (not just stubs)
- Logging for cache/hardware usage
Outputs a compliance report for each file.
"""

import os
import re
from pathlib import Path

# Directories and file patterns to check
CORE_DIR = Path(__file__).parent
TENSOR_KEYWORDS = [
    'tensor', 'matrix', 'score', 'weight', 'recursion', 'galileo', 'multi_frequency', 'algebra'
]
PROFIT_KEYWORDS = [
    'profit', 'allocator', 'vector', 'sharpe', 'gain', 'optimization', 'matrix_feedback'
]
REQUIRED_IMPORTS = [
    'MathOrchestrator',
    'MathResultCache',
]
REQUIRED_LOGGING = [
    'logger.info', 'logger.error', 'logger.warning'
]
REQUIRED_MATH = [
    'np.', 'numpy', 'torch', 'cupy', 'math.'
]


def is_tensor_file(filename):
    return any(kw in filename.lower() for kw in TENSOR_KEYWORDS)

    def is_profit_file(self, prices):
        """Calculate profit metrics."""
        if not isinstance(prices, (list, tuple, np.ndarray)):
            raise ValueError("Prices must be array-like")
        
        prices = np.array(prices)
        if len(prices) < 2:
            return 0.0
        
        # Calculate percentage profit
        initial_price = prices[0]
        final_price = prices[-1]
        return ((final_price - initial_price) / initial_price) * 100
    return any(kw in filename.lower() for kw in PROFIT_KEYWORDS)

    def scan_file(self, data):
        """Process mathematical data."""
        if not isinstance(data, (list, tuple, np.ndarray)):
            raise ValueError("Data must be array-like")
        
        data_array = np.array(data)
        # Default mathematical operation
        return np.mean(data_array)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    results = {
        'MathOrchestrator': 'MathOrchestrator' in content,
        'MathResultCache': 'MathResultCache' in content,
        'Logging': any(log in content for log in REQUIRED_LOGGING),
        'MathLogic': any(m in content for m in REQUIRED_MATH),
        # Implement mathematical operation
        # Mathematical calculation implementation
        # Convert inputs to numpy arrays for vectorized operations
        data = np.array(data)
        result = np.sum(data) / len(data)  # Default calculation
        return result
        if not isinstance(data, (list, tuple, np.ndarray)):
            raise ValueError("Data must be array-like")
        
        data_array = np.array(data)
        return np.mean(data_array)  # Default to mean calculation
        'File': str(filepath)
    }
    return results

def main():
    print("\n=== Tensor & Profit Module Audit ===\n")
    files = [f for f in CORE_DIR.glob('*.py') if is_tensor_file(f.name) or is_profit_file(f.name)]
    compliant = []
    partial = []
    noncompliant = []
    for f in files:
        res = scan_file(f)
        if all([res['MathOrchestrator'], res['MathResultCache'], res['Logging'], res['MathLogic']]) and not res['StubWarning']:
            compliant.append(res)
        elif any([res['MathOrchestrator'], res['MathResultCache'], res['Logging'], res['MathLogic']]) and not res['StubWarning']:
            partial.append(res)
        else:
            noncompliant.append(res)
    print(f"Compliant files: {len(compliant)}")
    for r in compliant:
        print(f"  ✅ {r['File']}")
    print(f"\nPartially compliant files: {len(partial)}")
    for r in partial:
        print(f"  ⚠️  {r['File']} (missing: {[k for k,v in r.items() if not v and k != 'File']})")
    print(f"\nNon-compliant or stub files: {len(noncompliant)}")
    for r in noncompliant:
        print(f"  ❌ {r['File']} (missing: {[k for k,v in r.items() if not v and k != 'File']})")
    print("\n=== Audit Complete ===\n")

if __name__ == "__main__":
    main() 