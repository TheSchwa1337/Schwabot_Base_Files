#!/usr/bin/env python3
"""
Core Utilities Module
Consolidated utilities from various small files.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np

# ============================================================================
# Backend Math Utilities (from backend_math.py)
# ============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if abs(denominator) < 1e-15:
        return default
    return numerator / denominator

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] range."""
    if arr.size == 0:
        return arr
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

# ============================================================================
# Glyph Router Utilities (from glyph_router.py)
# ============================================================================

class GlyphRouter:
    """Routes glyph-based operations."""
    
    def __init__(self):
        self.glyph_map = {}
    
    def register_glyph(self, glyph: str, handler: callable):
        """Register a glyph handler."""
        self.glyph_map[glyph] = handler
    
    def route(self, glyph: str, *args, **kwargs) -> Any:
        """Route a glyph to its handler."""
        if glyph in self.glyph_map:
            return self.glyph_map[glyph](*args, **kwargs)
        raise ValueError(f"Unknown glyph: {glyph}")

# ============================================================================
# Integration Utilities (from integration_orchestrator.py, integration_test.py)
# ============================================================================

class IntegrationOrchestrator:
    """Orchestrates system integrations."""
    
    def __init__(self):
        self.integrations = {}
        self.test_results = {}
    
    def register_integration(self, name: str, integration_func: callable):
        """Register an integration function."""
        self.integrations[name] = integration_func
    
    def run_integration(self, name: str, *args, **kwargs) -> Any:
        """Run a registered integration."""
        if name in self.integrations:
            return self.integrations[name](*args, **kwargs)
        raise ValueError(f"Unknown integration: {name}")
    
    def test_integration(self, name: str) -> Dict[str, Any]:
        """Test an integration."""
        try:
            result = self.run_integration(name)
            self.test_results[name] = {"status": "success", "result": result}
            return self.test_results[name]
        except Exception as e:
            self.test_results[name] = {"status": "error", "error": str(e)}
            return self.test_results[name]

# ============================================================================
# Order Wall Analyzer (from order_wall_analyzer.py)
# ============================================================================

def analyze_order_wall(order_book: Dict[str, List], threshold: float = 0.1) -> Dict[str, Any]:
    """Analyze order book for significant order walls."""
    bids = order_book.get('bids', [])
    asks = order_book.get('asks', [])
    
    bid_walls = []
    ask_walls = []
    
    # Analyze bid walls
    for i, (price, volume) in enumerate(bids):
        if volume > threshold:
            bid_walls.append({
                'price': price,
                'volume': volume,
                'position': i
            })
    
    # Analyze ask walls
    for i, (price, volume) in enumerate(asks):
        if volume > threshold:
            ask_walls.append({
                'price': price,
                'volume': volume,
                'position': i
            })
    
    return {
        'bid_walls': bid_walls,
        'ask_walls': ask_walls,
        'total_bid_volume': sum(bid['volume'] for bid in bid_walls),
        'total_ask_volume': sum(ask['volume'] for ask in ask_walls)
    }

# ============================================================================
# Profit Tier Adjuster (from profit_tier_adjuster.py)
# ============================================================================

def adjust_profit_tier(current_tier: int, performance: float, 
                      thresholds: List[float]) -> int:
    """Adjust profit tier based on performance."""
    for i, threshold in enumerate(thresholds):
        if performance >= threshold:
            return i
    return len(thresholds) - 1

# ============================================================================
# Swing Pattern Recognition (from swing_pattern_recognition.py)
# ============================================================================

def detect_swing_pattern(prices: np.ndarray, window: int = 20) -> Dict[str, Any]:
    """Detect swing high/low patterns in price data."""
    if len(prices) < window * 2:
        return {"pattern": "insufficient_data"}
    
    highs = []
    lows = []
    
    for i in range(window, len(prices) - window):
        # Check for swing high
        if all(prices[i] >= prices[j] for j in range(i - window, i + window + 1)):
            highs.append(i)
        
        # Check for swing low
        if all(prices[i] <= prices[j] for j in range(i - window, i + window + 1)):
            lows.append(i)
    
    return {
        "swing_highs": highs,
        "swing_lows": lows,
        "pattern": "swing_detected" if (highs or lows) else "no_pattern"
    }

# ============================================================================
# Unified API Coordinator (from unified_api_coordinator.py)
# ============================================================================

class UnifiedAPICoordinator:
    """Coordinates API calls across different services."""
    
    def __init__(self):
        self.api_handlers = {}
        self.request_count = 0
    
    def register_api_handler(self, service: str, handler: callable):
        """Register an API handler for a service."""
        self.api_handlers[service] = handler
    
    def call_api(self, service: str, method: str, *args, **kwargs) -> Any:
        """Make an API call to a registered service."""
        self.request_count += 1
        
        if service in self.api_handlers:
            return self.api_handlers[service](method, *args, **kwargs)
        raise ValueError(f"Unknown API service: {service}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API coordination statistics."""
        return {
            "total_requests": self.request_count,
            "registered_services": list(self.api_handlers.keys())
        }

# Global instances
glyph_router = GlyphRouter()
integration_orchestrator = IntegrationOrchestrator()
api_coordinator = UnifiedAPICoordinator()
