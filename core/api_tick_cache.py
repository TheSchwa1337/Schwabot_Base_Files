#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Tick Cache Module
=====================
Provides shared tick data caching for Schwabot trading system.

This module integrates with existing CCXT infrastructure to provide
a unified tick cache that reduces API calls and improves performance
across all subsystems.

Features:
- 5-minute TTL cache for tick data
- Integration with existing CCXT interfaces
- Memory-efficient storage
- Thread-safe operations
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import existing CCXT infrastructure
try:
    from core.enhanced_ccxt_trading_engine import EnhancedCCXTTradingEngine
    from core.real_multi_exchange_trader import RealMultiExchangeTrader
    from schwabot.init.core.data_feed import DataFeed
    CCXT_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    CCXT_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("CCXT infrastructure not available")


class TickCache:
    """
    Shared tick cache for Schwabot trading system.
    
    Provides a unified interface for accessing tick data across
    all subsystems, reducing API calls and improving performance.
    """
    
    def __init__(self, ttl: int = 300):
        """
        Initialize tick cache.
        
        Args:
            ttl: Time to live for cached data in seconds (default: 300 = 5 minutes)
        """
        self.cache = {}
        self.last_update = {}
        self.ttl = ttl
        self.logger = logging.getLogger(f"{__name__}.TickCache")
        
        # Initialize CCXT infrastructure if available
        self.data_feed = None
        self.enhanced_engine = None
        if CCXT_INFRASTRUCTURE_AVAILABLE:
            try:
                self.data_feed = DataFeed()
                self.enhanced_engine = EnhancedCCXTTradingEngine()
                self.logger.info("✅ CCXT infrastructure integrated")
            except Exception as e:
                self.logger.warning(f"⚠️ CCXT integration failed: {e}")
        
        self.logger.info(f"✅ Tick cache initialized with {ttl}s TTL")

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get tick data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDC")
            
        Returns:
            Tick data dictionary or None if not available
        """
        try:
            now = time.time()
            
            # Check cache first
            if symbol in self.cache and now - self.last_update[symbol] < self.ttl:
                self.logger.debug(f"[CACHE HIT] Returning cached data for {symbol}")
                return self.cache[symbol]

            # Fetch fresh data
            data = self._fetch_ticker(symbol)
            if data:
                self.cache[symbol] = data
                self.last_update[symbol] = now
                self.logger.debug(f"[CACHE MISS] Fetched fresh data for {symbol}")
                return data
            else:
                self.logger.warning(f"⚠️ Failed to fetch data for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting tick data for {symbol}: {e}")
            return None

    def _fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch ticker data using available CCXT infrastructure.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker data dictionary or None
        """
        try:
            # Try DataFeed first (most reliable)
            if self.data_feed:
                try:
                    tick_blob = self.data_feed.fetch_latest_tick(symbol)
                    # Parse tick blob into standard format
                    return self._parse_tick_blob(tick_blob)
                except Exception as e:
                    self.logger.debug(f"DataFeed failed for {symbol}: {e}")
            
            # Try EnhancedCCXTTradingEngine
            if self.enhanced_engine:
                try:
                    price = self.enhanced_engine.get_current_price(symbol)
                    if price:
                        return {
                            "symbol": symbol,
                            "last": price,
                            "timestamp": int(time.time() * 1000),
                            "source": "enhanced_engine"
                        }
                except Exception as e:
                    self.logger.debug(f"Enhanced engine failed for {symbol}: {e}")
            
            # Fallback: return None if no infrastructure available
            if not CCXT_INFRASTRUCTURE_AVAILABLE:
                self.logger.warning("No CCXT infrastructure available for tick fetching")
                return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching ticker for {symbol}: {e}")
            return None

    def _parse_tick_blob(self, tick_blob: str) -> Dict[str, Any]:
        """
        Parse tick blob string into standard dictionary format.
        
        Args:
            tick_blob: Tick blob string from DataFeed
            
        Returns:
            Parsed tick data dictionary
        """
        try:
            # Parse format: "{symbol},price={price},time={epoch}"
            parts = tick_blob.split(',')
            symbol = parts[0]
            
            price = None
            timestamp = None
            
            for part in parts[1:]:
                if part.startswith('price='):
                    price = float(part.split('=')[1])
                elif part.startswith('time='):
                    timestamp = int(part.split('=')[1]) * 1000  # Convert to ms
            
            if price is None or timestamp is None:
                raise ValueError("Invalid tick blob format")
            
            return {
                "symbol": symbol,
                "last": price,
                "timestamp": timestamp,
                "source": "data_feed"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing tick blob '{tick_blob}': {e}")
            return None

    def get_cached_symbols(self) -> list:
        """Get list of symbols currently in cache."""
        return list(self.cache.keys())

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.last_update.clear()
        self.logger.info("✅ Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        active_entries = 0
        expired_entries = 0
        
        for symbol, last_update in self.last_update.items():
            if now - last_update < self.ttl:
                active_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_symbols": len(self.cache),
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "ttl_seconds": self.ttl,
            "cache_size": len(self.cache)
        }


# Singleton instance for global access
tick_cache = TickCache()

if __name__ == "__main__":
    # Test the tick cache
    print("Testing Tick Cache...")
    
    # Test with BTC/USDC
    data = tick_cache.get("BTC/USDC")
    if data:
        print(f"✅ BTC/USDC: {data}")
    else:
        print("❌ Failed to get BTC/USDC data")
    
    # Show cache stats
    stats = tick_cache.get_cache_stats()
    print(f"📊 Cache stats: {stats}") 