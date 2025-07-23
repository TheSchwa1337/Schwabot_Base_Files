#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 KAPREKAR ENGINE - VOLATILITY CONVERGENCE INDEX
==================================================

Core Kaprekar engine for calculating iterations to reach Kaprekar's constant (6174).
This forms the foundation of our volatility convergence index for tick analysis.

Features:
- Kaprekar iteration calculation for 4-digit numbers
- Volatility convergence detection
- Cycle detection and non-convergence handling
- Integration with tick processing pipeline
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def kaprekar_iterations(n: int, max_iter: int = 10) -> int:
    """
    Calculate the number of iterations to reach Kaprekar's constant (6174)
    for a 4-digit number n. This forms our volatility convergence index.
    
    Args:
        n: 4-digit number to process
        max_iter: Maximum iterations before giving up
        
    Returns:
        Number of iterations to reach 6174, or -1 if non-convergent
    """
    try:
        # Ensure 4-digit format
        if n < 1000:
            n = n * 1000  # Pad with zeros
        elif n > 9999:
            n = n % 10000  # Take last 4 digits
            
        seen = set()
        iter_count = 0
        current = n

        while current != 6174 and iter_count < max_iter:
            # Convert to 4-digit string
            digits = f"{current:04d}"
            
            # Sort ascending and descending
            asc = int("".join(sorted(digits)))
            desc = int("".join(sorted(digits, reverse=True)))
            
            # Calculate difference
            current = desc - asc
            
            # Check for cycles (non-convergence)
            if current in seen:
                logger.debug(f"Cycle detected at iteration {iter_count} for input {n}")
                break
                
            seen.add(current)
            iter_count += 1

        # Return iterations if converged, -1 otherwise
        return iter_count if current == 6174 else -1
        
    except Exception as e:
        logger.error(f"Error in kaprekar_iterations for {n}: {e}")
        return -1


def analyze_kaprekar_convergence(numbers: list) -> dict:
    """
    Analyze Kaprekar convergence patterns across a list of numbers.
    
    Args:
        numbers: List of numbers to analyze
        
    Returns:
        Dictionary with convergence statistics
    """
    try:
        results = []
        convergent_count = 0
        total_iterations = 0
        
        for num in numbers:
            iterations = kaprekar_iterations(num)
            results.append({
                'number': num,
                'iterations': iterations,
                'convergent': iterations != -1
            })
            
            if iterations != -1:
                convergent_count += 1
                total_iterations += iterations
        
        convergence_rate = convergent_count / len(numbers) if numbers else 0
        avg_iterations = total_iterations / convergent_count if convergent_count > 0 else 0
        
        return {
            'total_numbers': len(numbers),
            'convergent_count': convergent_count,
            'convergence_rate': convergence_rate,
            'average_iterations': avg_iterations,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_kaprekar_convergence: {e}")
        return {
            'total_numbers': 0,
            'convergent_count': 0,
            'convergence_rate': 0.0,
            'average_iterations': 0.0,
            'results': []
        }


def get_volatility_classification(iterations: int) -> str:
    """
    Classify volatility based on Kaprekar iterations.
    
    Args:
        iterations: Number of iterations to reach 6174
        
    Returns:
        Volatility classification string
    """
    try:
        if iterations == -1:
            return "NON_CONVERGENT"
        elif iterations <= 3:
            return "LOW_VOLATILITY"
        elif iterations <= 5:
            return "MEDIUM_VOLATILITY"
        elif iterations <= 7:
            return "HIGH_VOLATILITY"
        else:
            return "EXTREME_VOLATILITY"
            
    except Exception as e:
        logger.error(f"Error in get_volatility_classification: {e}")
        return "UNKNOWN"


def validate_kaprekar_input(n: int) -> bool:
    """
    Validate input for Kaprekar processing.
    
    Args:
        n: Number to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if number has at least 2 different digits
        digits = set(str(n))
        return len(digits) >= 2
        
    except Exception as e:
        logger.error(f"Error in validate_kaprekar_input: {e}")
        return False


def debug_kaprekar_iterations(n: int, max_iter: int = 10) -> None:
    """
    Debug Kaprekar iterations step by step.
    
    Args:
        n: 4-digit number to process
        max_iter: Maximum iterations
    """
    try:
        print(f"Debugging Kaprekar iterations for {n}")
        
        if n < 1000:
            n = n * 1000
        elif n > 9999:
            n = n % 10000
            
        seen = set()
        iter_count = 0
        current = n

        while current != 6174 and iter_count < max_iter:
            digits = f"{current:04d}"
            asc = int("".join(sorted(digits)))
            desc = int("".join(sorted(digits, reverse=True)))
            diff = desc - asc
            
            print(f"Iteration {iter_count + 1}: {current} -> {desc} - {asc} = {diff}")
            
            if current in seen:
                print(f"Cycle detected at iteration {iter_count + 1}")
                break
                
            seen.add(current)
            current = diff
            iter_count += 1

        if current == 6174:
            print(f"Converged to 6174 in {iter_count} iterations")
        else:
            print(f"Did not converge within {max_iter} iterations")
            
    except Exception as e:
        logger.error(f"Error in debug_kaprekar_iterations: {e}")


def test_kaprekar_engine():
    """Test the Kaprekar engine functionality."""
    try:
        print("🧮 Testing Kaprekar Engine...")
        
        # Test basic functionality
        test_numbers = [1234, 5678, 9999, 1000, 6174]
        
        for num in test_numbers:
            iterations = kaprekar_iterations(num)
            classification = get_volatility_classification(iterations)
            print(f"Number {num}: {iterations} iterations -> {classification}")
        
        # Test convergence analysis
        analysis = analyze_kaprekar_convergence(test_numbers)
        print(f"Convergence Analysis: {analysis['convergence_rate']:.2%} convergence rate")
        
        print("✅ Kaprekar Engine tests completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in test_kaprekar_engine: {e}")
        return False


if __name__ == "__main__":
    test_kaprekar_engine() 