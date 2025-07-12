#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profit Echo Cache Module
========================
Provides temporal profit memory for Schwabot trading system.

This module tracks profit history for strategies and provides
temporal memory for Schwabot's decision making, enabling it to
learn from past performance and optimize future trades.

Features:
- Persistent profit history storage
- Strategy tag-based organization
- Temporal analysis capabilities
- Integration with existing registry systems
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProfitEchoCache:
    """
    Profit echo cache for Schwabot trading system.
    
    Tracks profit history for strategies and provides temporal memory
    for Schwabot's decision making, enabling it to learn from past
    performance and optimize future trades.
    """
    
    def __init__(self, path: str = "data/profit_echo.json"):
        """
        Initialize profit echo cache.
        
        Args:
            path: Path to the profit echo cache file
        """
        self.path = Path(path)
        self.logger = logging.getLogger(f"{__name__}.ProfitEchoCache")
        
        # Ensure data directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing echo data
        self.echo = self._load()
        
        self.logger.info(f"✅ Profit echo cache initialized at {self.path}")
        self.logger.info(f"📊 Loaded {len(self.echo)} strategy tags")

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load profit echo data from file.
        
        Returns:
            Dictionary of strategy tags to profit history
        """
        try:
            if self.path.exists():
                with open(self.path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    self.logger.info(f"✅ Loaded profit echo data from {self.path}")
                    return data
            else:
                self.logger.info(f"📝 Creating new profit echo file at {self.path}")
                return {}
        except Exception as e:
            self.logger.error(f"❌ Error loading profit echo data: {e}")
            return {}

    def _save(self) -> None:
        """Save profit echo data to file."""
        try:
            with open(self.path, "w", encoding='utf-8') as f:
                json.dump(self.echo, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"💾 Saved profit echo data to {self.path}")
        except Exception as e:
            self.logger.error(f"❌ Error saving profit echo data: {e}")

    def record(self, tag: str, profit: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a profit entry for a strategy tag.
        
        Args:
            tag: Strategy tag identifier
            profit: Profit value (can be negative for losses)
            metadata: Optional metadata about the trade
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # Initialize tag if it doesn't exist
            if tag not in self.echo:
                self.echo[tag] = []
            
            # Create entry
            entry = {
                "time": timestamp,
                "profit": profit,
                "metadata": metadata or {}
            }
            
            # Add to history
            self.echo[tag].append(entry)
            
            # Retain only latest 10 entries per tag
            self.echo[tag] = self.echo[tag][-10:]
            
            # Save to file
            self._save()
            
            self.logger.debug(f"📝 Recorded profit {profit:.6f} for tag '{tag}'")
            
        except Exception as e:
            self.logger.error(f"❌ Error recording profit for tag '{tag}': {e}")

    def get_recent_profits(self, tag: str) -> List[float]:
        """
        Get recent profit values for a strategy tag.
        
        Args:
            tag: Strategy tag identifier
            
        Returns:
            List of recent profit values
        """
        try:
            if tag not in self.echo:
                return []
            
            return [entry["profit"] for entry in self.echo[tag]]
            
        except Exception as e:
            self.logger.error(f"❌ Error getting recent profits for tag '{tag}': {e}")
            return []

    def average_profit(self, tag: str) -> float:
        """
        Calculate average profit for a strategy tag.
        
        Args:
            tag: Strategy tag identifier
            
        Returns:
            Average profit value
        """
        try:
            profits = self.get_recent_profits(tag)
            if not profits:
                return 0.0
            
            return sum(profits) / len(profits)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating average profit for tag '{tag}': {e}")
            return 0.0

    def get_profit_trend(self, tag: str) -> Dict[str, Any]:
        """
        Get profit trend analysis for a strategy tag.
        
        Args:
            tag: Strategy tag identifier
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            profits = self.get_recent_profits(tag)
            if len(profits) < 2:
                return {
                    "trend": "insufficient_data",
                    "direction": "neutral",
                    "slope": 0.0,
                    "volatility": 0.0
                }
            
            # Calculate trend
            recent = profits[-3:] if len(profits) >= 3 else profits
            older = profits[:-3] if len(profits) >= 3 else profits[:1]
            
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older)
            
            # Determine trend direction
            if recent_avg > older_avg * 1.05:  # 5% improvement
                direction = "improving"
            elif recent_avg < older_avg * 0.95:  # 5% decline
                direction = "declining"
            else:
                direction = "stable"
            
            # Calculate slope (simple linear trend)
            if len(profits) >= 2:
                slope = (profits[-1] - profits[0]) / len(profits)
            else:
                slope = 0.0
            
            # Calculate volatility
            if len(profits) >= 2:
                import numpy as np
                volatility = np.std(profits)
            else:
                volatility = 0.0
            
            return {
                "trend": "calculated",
                "direction": direction,
                "slope": slope,
                "volatility": volatility,
                "recent_avg": recent_avg,
                "older_avg": older_avg,
                "data_points": len(profits)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating profit trend for tag '{tag}': {e}")
            return {
                "trend": "error",
                "direction": "unknown",
                "slope": 0.0,
                "volatility": 0.0
            }

    def get_top_performers(self, min_entries: int = 3) -> List[Dict[str, Any]]:
        """
        Get top performing strategy tags.
        
        Args:
            min_entries: Minimum number of entries required
            
        Returns:
            List of top performing tags with their stats
        """
        try:
            performers = []
            
            for tag, entries in self.echo.items():
                if len(entries) >= min_entries:
                    avg_profit = self.average_profit(tag)
                    trend = self.get_profit_trend(tag)
                    
                    performers.append({
                        "tag": tag,
                        "average_profit": avg_profit,
                        "entries": len(entries),
                        "trend": trend,
                        "latest_profit": entries[-1]["profit"] if entries else 0.0
                    })
            
            # Sort by average profit (descending)
            performers.sort(key=lambda x: x["average_profit"], reverse=True)
            
            return performers
            
        except Exception as e:
            self.logger.error(f"❌ Error getting top performers: {e}")
            return []

    def get_strategy_confidence(self, tag: str) -> float:
        """
        Calculate confidence score for a strategy tag.
        
        Args:
            tag: Strategy tag identifier
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            profits = self.get_recent_profits(tag)
            if not profits:
                return 0.0
            
            # Base confidence on number of entries
            entry_confidence = min(len(profits) / 10.0, 1.0)
            
            # Profit consistency confidence
            if len(profits) >= 2:
                import numpy as np
                consistency = 1.0 - min(np.std(profits) / abs(np.mean(profits) + 1e-6), 1.0)
            else:
                consistency = 0.5
            
            # Trend confidence
            trend = self.get_profit_trend(tag)
            if trend["direction"] == "improving":
                trend_confidence = 1.0
            elif trend["direction"] == "stable":
                trend_confidence = 0.7
            else:
                trend_confidence = 0.3
            
            # Weighted average
            confidence = (0.3 * entry_confidence + 
                         0.4 * consistency + 
                         0.3 * trend_confidence)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating confidence for tag '{tag}': {e}")
            return 0.0

    def get_all_tags(self) -> List[str]:
        """Get all strategy tags in the cache."""
        return list(self.echo.keys())

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = sum(len(entries) for entries in self.echo.values())
        total_tags = len(self.echo)
        
        return {
            "total_tags": total_tags,
            "total_entries": total_entries,
            "average_entries_per_tag": total_entries / max(total_tags, 1),
            "file_path": str(self.path),
            "file_size_bytes": self.path.stat().st_size if self.path.exists() else 0
        }

    def clear_tag(self, tag: str) -> None:
        """
        Clear all entries for a specific tag.
        
        Args:
            tag: Strategy tag to clear
        """
        try:
            if tag in self.echo:
                del self.echo[tag]
                self._save()
                self.logger.info(f"🗑️ Cleared tag '{tag}'")
            else:
                self.logger.warning(f"⚠️ Tag '{tag}' not found in cache")
                
        except Exception as e:
            self.logger.error(f"❌ Error clearing tag '{tag}': {e}")

    def clear_all(self) -> None:
        """Clear all profit echo data."""
        try:
            self.echo.clear()
            self._save()
            self.logger.info("🗑️ Cleared all profit echo data")
            
        except Exception as e:
            self.logger.error(f"❌ Error clearing all data: {e}")


# Singleton instance for global access
profit_echo_cache = ProfitEchoCache()

if __name__ == "__main__":
    # Test the profit echo cache
    print("Testing Profit Echo Cache...")
    
    # Record some test profits
    profit_echo_cache.record("btc_usdc_snipe", 0.023, {"strategy": "snipe", "asset": "BTC"})
    profit_echo_cache.record("btc_usdc_snipe", 0.018, {"strategy": "snipe", "asset": "BTC"})
    profit_echo_cache.record("btc_usdc_snipe", 0.031, {"strategy": "snipe", "asset": "BTC"})
    
    profit_echo_cache.record("eth_btc_rotation", 0.045, {"strategy": "rotation", "asset": "ETH"})
    profit_echo_cache.record("eth_btc_rotation", 0.038, {"strategy": "rotation", "asset": "ETH"})
    
    # Test retrieval
    print(f"Recent profits for btc_usdc_snipe: {profit_echo_cache.get_recent_profits('btc_usdc_snipe')}")
    print(f"Average profit for btc_usdc_snipe: {profit_echo_cache.average_profit('btc_usdc_snipe'):.4f}")
    
    # Test trend analysis
    trend = profit_echo_cache.get_profit_trend("btc_usdc_snipe")
    print(f"Trend for btc_usdc_snipe: {trend}")
    
    # Test confidence
    confidence = profit_echo_cache.get_strategy_confidence("btc_usdc_snipe")
    print(f"Confidence for btc_usdc_snipe: {confidence:.3f}")
    
    # Show cache stats
    stats = profit_echo_cache.get_cache_stats()
    print(f"📊 Cache stats: {stats}") 