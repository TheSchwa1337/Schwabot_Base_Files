#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌉 TICK-KAPREKAR BRIDGE - PRICE TO VOLATILITY INDEX
===================================================

Bridge module that connects price ticks to Kaprekar analysis by normalizing
float prices into 4-digit integers for volatility convergence indexing.

Features:
- Price normalization for Kaprekar processing
- Multi-decimal precision handling
- Integration with existing tick processing
- Volatility index generation
"""

import logging
import re
from typing import Dict, Any, Optional
from .kaprekar_engine import kaprekar_iterations, get_volatility_classification

logger = logging.getLogger(__name__)

def price_to_kaprekar_index(price: float) -> int:
    """
    Normalize float price into a 4-digit integer, feed into kaprekar.
    
    Args:
        price: Float price value (e.g., 2045.29)
        
    Returns:
        Kaprekar iteration count for normalized price
    """
    try:
        # Convert price to string and remove decimal point
        price_str = str(price)
        
        # Extract digits only (remove decimal point)
        digits_only = re.sub(r'[^\d]', '', price_str)
        
        # Take first 4 digits
        if len(digits_only) >= 4:
            normalized = int(digits_only[:4])
        else:
            # Pad with zeros if less than 4 digits
            normalized = int(digits_only.ljust(4, '0'))
        
        # Calculate Kaprekar iterations
        return kaprekar_iterations(normalized)
        
    except Exception as e:
        logger.error(f"Error in price_to_kaprekar_index for {price}: {e}")
        return -1


def analyze_price_volatility(price: float) -> Dict[str, Any]:
    """
    Analyze price volatility using Kaprekar convergence.
    
    Args:
        price: Float price value
        
    Returns:
        Dictionary with volatility analysis
    """
    try:
        k_index = price_to_kaprekar_index(price)
        volatility_class = get_volatility_classification(k_index)
        
        return {
            'price': price,
            'kaprekar_index': k_index,
            'volatility_classification': volatility_class,
            'convergent': k_index != -1,
            'normalized_price': _extract_normalized_price(price)
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_price_volatility for {price}: {e}")
        return {
            'price': price,
            'kaprekar_index': -1,
            'volatility_classification': 'error',
            'convergent': False,
            'error': str(e)
        }


def _extract_normalized_price(price: float) -> int:
    """
    Extract normalized 4-digit price from float.
    
    Args:
        price: Float price value
        
    Returns:
        Normalized 4-digit integer
    """
    try:
        price_str = str(price)
        digits_only = re.sub(r'[^\d]', '', price_str)
        
        if len(digits_only) >= 4:
            return int(digits_only[:4])
        else:
            return int(digits_only.ljust(4, '0'))
            
    except Exception as e:
        logger.error(f"Error in _extract_normalized_price for {price}: {e}")
        return 1000


def batch_analyze_prices(prices: list) -> Dict[str, Any]:
    """
    Analyze multiple prices for volatility patterns.
    
    Args:
        prices: List of float prices
        
    Returns:
        Dictionary with batch analysis results
    """
    try:
        results = []
        convergent_count = 0
        total_iterations = 0
        
        for price in prices:
            analysis = analyze_price_volatility(price)
            results.append(analysis)
            
            if analysis['convergent']:
                convergent_count += 1
                total_iterations += analysis['kaprekar_index']
        
        convergence_rate = convergent_count / len(prices) if prices else 0
        avg_iterations = total_iterations / convergent_count if convergent_count > 0 else 0
        
        return {
            'total_prices': len(prices),
            'convergent_count': convergent_count,
            'convergence_rate': convergence_rate,
            'average_iterations': avg_iterations,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in batch_analyze_prices: {e}")
        return {
            'total_prices': 0,
            'convergent_count': 0,
            'convergence_rate': 0.0,
            'average_iterations': 0.0,
            'results': []
        }


def get_volatility_signal(price: float) -> str:
    """
    Get volatility signal based on price analysis.
    
    Args:
        price: Float price value
        
    Returns:
        Volatility signal string
    """
    try:
        analysis = analyze_price_volatility(price)
        classification = analysis['volatility_classification']
        
        if classification == 'LOW_VOLATILITY':
            return 'STABLE'
        elif classification == 'MEDIUM_VOLATILITY':
            return 'MODERATE'
        elif classification == 'HIGH_VOLATILITY':
            return 'VOLATILE'
        elif classification == 'EXTREME_VOLATILITY':
            return 'EXTREME'
        else:
            return 'UNKNOWN'
            
    except Exception as e:
        logger.error(f"Error in get_volatility_signal for {price}: {e}")
        return 'ERROR'


def validate_price_input(price: float) -> bool:
    """
    Validate price input for processing.
    
    Args:
        price: Float price value
        
    Returns:
        True if valid, False otherwise
    """
    try:
        return isinstance(price, (int, float)) and price > 0
        
    except Exception as e:
        logger.error(f"Error in validate_price_input for {price}: {e}")
        return False


def test_tick_kaprekar_bridge():
    """Test the tick-kaprekar bridge functionality."""
    try:
        print("🌉 Testing Tick-Kaprekar Bridge...")
        
        # Test price analysis
        test_prices = [2045.29, 1234.56, 9999.99, 100.0, 6174.0]
        
        for price in test_prices:
            analysis = analyze_price_volatility(price)
            signal = get_volatility_signal(price)
            print(f"Price {price}: {analysis['kaprekar_index']} iterations -> {signal}")
        
        # Test batch analysis
        batch_result = batch_analyze_prices(test_prices)
        print(f"Batch Analysis: {batch_result['convergence_rate']:.2%} convergence rate")
        
        print("✅ Tick-Kaprekar Bridge tests completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in test_tick_kaprekar_bridge: {e}")
        return False


if __name__ == "__main__":
    test_tick_kaprekar_bridge() 