"""
Historical Ledger Manager
========================

Manages integration of historical ledger data from major cryptocurrencies:
- Bitcoin (BTC) blockchain data
- Ethereum (ETH) blockchain data  
- Ripple (XRP) ledger data

Features:
- RAM → mid-term → long-term storage pipeline
- Chart information integration for analytical spine
- Volume and trade data processing
- Ghost file architecture integration
- High-volume trading data preparation
"""

import asyncio
import logging
import json
import time
import gzip
import pickle
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
import hashlib

# Data processing imports
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class LedgerDataType(Enum):
    """Types of ledger data"""
    PRICE_DATA = "price_data"
    VOLUME_DATA = "volume_data"
    TRANSACTION_DATA = "transaction_data"
    BLOCK_DATA = "block_data"
    ORDERBOOK_DATA = "orderbook_data"

class StorageTier(Enum):
    """Storage tiers for ledger data"""
    RAM_CACHE = "ram_cache"           # Active trading data
    MID_TERM = "mid_term"             # Recent historical data
    LONG_TERM = "long_term"           # Extended historical data
    ARCHIVE = "archive"               # Permanent storage

@dataclass
class LedgerRecord:
    """Individual ledger record"""
    timestamp: datetime
    symbol: str
    data_type: LedgerDataType
    data: Dict[str, Any]
    importance_score: float = 0.5
    storage_tier: StorageTier = StorageTier.RAM_CACHE
    
    def __post_init__(self):
        self.record_id = self._generate_record_id()
    
    def _generate_record_id(self) -> str:
        """Generate unique record ID"""
        content = f"{self.timestamp.isoformat()}_{self.symbol}_{self.data_type.value}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

class HistoricalLedgerManager:
    """
    Manages historical ledger data with intelligent storage tiering
    and integration with the pipeline management system.
    """
    
    def __init__(self, 
                 pipeline_manager=None,
                 storage_base_path: str = "data/ledgers"):
        """
        Initialize the historical ledger manager
        
        Args:
            pipeline_manager: Advanced pipeline manager for memory coordination
            storage_base_path: Base path for ledger data storage
        """
        self.pipeline_manager = pipeline_manager
        self.storage_base_path = Path(storage_base_path)
        self.storage_base_path.mkdir(parents=True, exist_ok=True)
        
        # Storage tiers
        self.ram_cache: Dict[str, LedgerRecord] = {}
        self.mid_term_storage: Dict[str, str] = {}    # record_id -> file_path
        self.long_term_storage: Dict[str, str] = {}   # record_id -> file_path
        self.archive_storage: Dict[str, str] = {}     # record_id -> file_path
        
        # Storage limits (in number of records)
        self.storage_limits = {
            StorageTier.RAM_CACHE: 10000,       # 10k records in RAM
            StorageTier.MID_TERM: 100000,       # 100k records mid-term
            StorageTier.LONG_TERM: 1000000,     # 1M records long-term
            StorageTier.ARCHIVE: -1             # Unlimited archive
        }
        
        # Analytical spine configuration
        self.analytical_spine = {
            "primary_symbol": "BTC/USDT",
            "importance_weights": {
                "BTC": 1.0,     # Primary importance
                "ETH": 0.8,     # High importance
                "XRP": 0.6,     # Moderate importance
                "others": 0.4   # Lower importance
            }
        }
        
        # Chart integration settings
        self.chart_config = {
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "indicators": ["volume", "volatility", "momentum"],
            "depth_days": 30,
            "real_time_updates": True
        }
        
        # Performance tracking
        self.stats = {
            "records_processed": 0,
            "storage_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_saved_mb": 0
        }
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        logger.info("HistoricalLedgerManager initialized")
    
    async def start_manager(self) -> bool:
        """Start the historical ledger manager"""
        try:
            logger.info("Starting Historical Ledger Manager...")
            
            # Initialize storage directories
            self._initialize_storage_structure()
            
            # Load existing data indices
            await self._load_storage_indices()
            
            # Start background maintenance tasks
            await self._start_background_tasks()
            
            # Initialize real-time data feeds
            await self._initialize_data_feeds()
            
            self.is_running = True
            logger.info("✅ Historical Ledger Manager started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting ledger manager: {e}")
            return False
    
    async def stop_manager(self) -> bool:
        """Stop the historical ledger manager"""
        try:
            logger.info("Stopping Historical Ledger Manager...")
            
            # Save current state
            await self._save_storage_indices()
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Flush RAM cache to storage
            await self._flush_ram_cache()
            
            self.is_running = False
            logger.info("✅ Historical Ledger Manager stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping ledger manager: {e}")
            return False
    
    async def ingest_ledger_data(self, 
                               symbol: str,
                               data_type: LedgerDataType,
                               data: Dict[str, Any],
                               timestamp: Optional[datetime] = None) -> str:
        """
        Ingest new ledger data with intelligent storage allocation
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            data_type: Type of ledger data
            data: Actual data content
            timestamp: Data timestamp (defaults to now)
            
        Returns:
            Record ID for tracking
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Calculate importance score based on analytical spine
        importance_score = self._calculate_importance_score(symbol, data_type, data)
        
        # Create ledger record
        record = LedgerRecord(
            timestamp=timestamp,
            symbol=symbol,
            data_type=data_type,
            data=data,
            importance_score=importance_score
        )
        
        # Determine initial storage tier
        storage_tier = self._determine_storage_tier(record)
        record.storage_tier = storage_tier
        
        # Store in appropriate tier
        await self._store_record(record)
        
        # Update statistics
        self.stats["records_processed"] += 1
        
        # Trigger analytical processing if this is important data
        if importance_score > 0.8:
            await self._trigger_analytical_processing(record)
        
        logger.debug(f"Ingested {symbol} {data_type.value} data to {storage_tier.value}")
        return record.record_id
    
    async def query_historical_data(self,
                                  symbol: str,
                                  data_type: LedgerDataType,
                                  start_time: datetime,
                                  end_time: datetime,
                                  limit: Optional[int] = None) -> List[LedgerRecord]:
        """
        Query historical data across all storage tiers
        
        Args:
            symbol: Trading symbol
            data_type: Type of data to query
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of records to return
            
        Returns:
            List of matching ledger records
        """
        results = []
        
        # Search RAM cache first (fastest)
        cache_results = await self._search_ram_cache(symbol, data_type, start_time, end_time)
        results.extend(cache_results)
        
        # Search mid-term storage if needed
        if not limit or len(results) < limit:
            mid_term_results = await self._search_mid_term_storage(symbol, data_type, start_time, end_time)
            results.extend(mid_term_results)
        
        # Search long-term storage if needed
        if not limit or len(results) < limit:
            long_term_results = await self._search_long_term_storage(symbol, data_type, start_time, end_time)
            results.extend(long_term_results)
        
        # Search archive if needed
        if not limit or len(results) < limit:
            archive_results = await self._search_archive_storage(symbol, data_type, start_time, end_time)
            results.extend(archive_results)
        
        # Sort by timestamp and apply limit
        results.sort(key=lambda r: r.timestamp)
        if limit:
            results = results[:limit]
        
        # Update cache statistics
        if cache_results:
            self.stats["cache_hits"] += len(cache_results)
        else:
            self.stats["cache_misses"] += 1
        
        return results
    
    async def get_chart_data_for_analysis(self,
                                        symbol: str,
                                        timeframe: str = "1h",
                                        depth_days: int = 30) -> Dict[str, Any]:
        """
        Get chart data optimized for analytical spine processing
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            depth_days: Number of days of historical data
            
        Returns:
            Structured chart data for analysis
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=depth_days)
        
        # Get price data
        price_records = await self.query_historical_data(
            symbol=symbol,
            data_type=LedgerDataType.PRICE_DATA,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get volume data
        volume_records = await self.query_historical_data(
            symbol=symbol,
            data_type=LedgerDataType.VOLUME_DATA,
            start_time=start_time,
            end_time=end_time
        )
        
        # Process into chart format
        chart_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "period": f"{depth_days}d",
            "data_points": self._process_records_to_chart_format(price_records, volume_records, timeframe),
            "analytics": self._calculate_chart_analytics(price_records, volume_records),
            "importance_score": self.analytical_spine["importance_weights"].get(
                symbol.split('/')[0], 
                self.analytical_spine["importance_weights"]["others"]
            )
        }
        
        return chart_data
    
    async def integrate_with_ghost_architecture(self,
                                              profit_data: Dict[str, Any],
                                              target_storage_tier: StorageTier = StorageTier.LONG_TERM) -> bool:
        """
        Integrate profit data with ghost file architecture
        
        Args:
            profit_data: Profit information from trading operations
            target_storage_tier: Target storage tier for the data
            
        Returns:
            Success status
        """
        try:
            # Create ghost architecture record
            ghost_record = LedgerRecord(
                timestamp=datetime.now(timezone.utc),
                symbol=profit_data.get("symbol", "BTC/USDT"),
                data_type=LedgerDataType.TRANSACTION_DATA,
                data={
                    "type": "ghost_profit_handoff",
                    "profit_amount": profit_data.get("profit", 0.0),
                    "entry_price": profit_data.get("entry_price", 0.0),
                    "exit_price": profit_data.get("exit_price", 0.0),
                    "volume": profit_data.get("volume", 0.0),
                    "confidence": profit_data.get("confidence", 0.5),
                    "hash_triggers": profit_data.get("hash_triggers", []),
                    "execution_time": profit_data.get("execution_time", 0.0),
                    "metadata": profit_data.get("metadata", {})
                },
                importance_score=0.9,  # High importance for profit data
                storage_tier=target_storage_tier
            )
            
            # Store the record
            await self._store_record(ghost_record)
            
            # If pipeline manager is available, also store there
            if self.pipeline_manager:
                await self.pipeline_manager.allocate_memory_dynamically(
                    data=profit_data,
                    importance_level=0.9,
                    expected_lifetime_hours=168  # 1 week
                )
            
            logger.info(f"Ghost architecture integration completed: {ghost_record.record_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in ghost architecture integration: {e}")
            return False
    
    def _calculate_importance_score(self,
                                  symbol: str,
                                  data_type: LedgerDataType,
                                  data: Dict[str, Any]) -> float:
        """Calculate importance score for ledger data"""
        base_score = 0.5
        
        # Symbol importance from analytical spine
        symbol_weight = self.analytical_spine["importance_weights"].get(
            symbol.split('/')[0], 
            self.analytical_spine["importance_weights"]["others"]
        )
        
        # Data type importance
        type_weights = {
            LedgerDataType.PRICE_DATA: 1.0,
            LedgerDataType.VOLUME_DATA: 0.9,
            LedgerDataType.TRANSACTION_DATA: 0.8,
            LedgerDataType.ORDERBOOK_DATA: 0.7,
            LedgerDataType.BLOCK_DATA: 0.6
        }
        type_weight = type_weights.get(data_type, 0.5)
        
        # Recency boost (more recent data is more important)
        if "timestamp" in data:
            try:
                data_time = datetime.fromisoformat(str(data["timestamp"]))
                age_hours = (datetime.now(timezone.utc) - data_time).total_seconds() / 3600
                recency_weight = max(0.1, 1.0 - (age_hours / (24 * 7)))  # Decay over 1 week
            except:
                recency_weight = 0.5
        else:
            recency_weight = 1.0  # Assume recent if no timestamp
        
        # Volume/significance boost
        volume_weight = 1.0
        if "volume" in data and isinstance(data["volume"], (int, float)):
            # Higher volume = higher importance
            volume_weight = min(1.5, 1.0 + (data["volume"] / 1000000))  # Boost for high volume
        
        # Calculate final score
        final_score = base_score * symbol_weight * type_weight * recency_weight * volume_weight
        return min(1.0, max(0.0, final_score))
    
    def _determine_storage_tier(self, record: LedgerRecord) -> StorageTier:
        """Determine appropriate storage tier for a record"""
        # High importance or recent data goes to RAM
        if record.importance_score > 0.8:
            return StorageTier.RAM_CACHE
        
        # Medium importance goes to mid-term
        if record.importance_score > 0.6:
            return StorageTier.MID_TERM
        
        # Lower importance but still recent goes to long-term
        age_hours = (datetime.now(timezone.utc) - record.timestamp).total_seconds() / 3600
        if age_hours < 24:  # Less than 1 day old
            return StorageTier.LONG_TERM
        
        # Everything else goes to archive
        return StorageTier.ARCHIVE
    
    async def _store_record(self, record: LedgerRecord) -> None:
        """Store record in appropriate storage tier"""
        if record.storage_tier == StorageTier.RAM_CACHE:
            await self._store_in_ram(record)
        elif record.storage_tier == StorageTier.MID_TERM:
            await self._store_in_mid_term(record)
        elif record.storage_tier == StorageTier.LONG_TERM:
            await self._store_in_long_term(record)
        elif record.storage_tier == StorageTier.ARCHIVE:
            await self._store_in_archive(record)
        
        self.stats["storage_operations"] += 1
    
    async def _store_in_ram(self, record: LedgerRecord) -> None:
        """Store record in RAM cache"""
        # Check if we need to evict old records
        if len(self.ram_cache) >= self.storage_limits[StorageTier.RAM_CACHE]:
            await self._evict_from_ram_cache()
        
        self.ram_cache[record.record_id] = record
    
    async def _evict_from_ram_cache(self) -> None:
        """Evict least important records from RAM cache"""
        if not self.ram_cache:
            return
        
        # Sort by importance score (lowest first)
        sorted_records = sorted(
            self.ram_cache.items(),
            key=lambda x: x[1].importance_score
        )
        
        # Evict bottom 10% of records
        evict_count = max(1, len(sorted_records) // 10)
        
        for i in range(evict_count):
            record_id, record = sorted_records[i]
            
            # Move to appropriate storage tier
            if record.importance_score > 0.5:
                record.storage_tier = StorageTier.MID_TERM
                await self._store_in_mid_term(record)
            else:
                record.storage_tier = StorageTier.LONG_TERM
                await self._store_in_long_term(record)
            
            # Remove from RAM cache
            del self.ram_cache[record_id]
    
    async def _store_in_mid_term(self, record: LedgerRecord) -> None:
        """Store record in mid-term storage"""
        file_path = self._get_mid_term_file_path(record)
        await self._save_record_to_file(record, file_path)
        self.mid_term_storage[record.record_id] = str(file_path)
    
    async def _store_in_long_term(self, record: LedgerRecord) -> None:
        """Store record in long-term storage with compression"""
        file_path = self._get_long_term_file_path(record)
        await self._save_record_to_file(record, file_path, compress=True)
        self.long_term_storage[record.record_id] = str(file_path)
    
    async def _store_in_archive(self, record: LedgerRecord) -> None:
        """Store record in archive storage with maximum compression"""
        file_path = self._get_archive_file_path(record)
        await self._save_record_to_file(record, file_path, compress=True)
        self.archive_storage[record.record_id] = str(file_path)
    
    def _get_mid_term_file_path(self, record: LedgerRecord) -> Path:
        """Get file path for mid-term storage"""
        date_str = record.timestamp.strftime("%Y/%m/%d")
        return self.storage_base_path / "mid_term" / record.symbol.replace('/', '_') / date_str / f"{record.record_id}.json"
    
    def _get_long_term_file_path(self, record: LedgerRecord) -> Path:
        """Get file path for long-term storage"""
        date_str = record.timestamp.strftime("%Y/%m")
        return self.storage_base_path / "long_term" / record.symbol.replace('/', '_') / date_str / f"{record.record_id}.gz"
    
    def _get_archive_file_path(self, record: LedgerRecord) -> Path:
        """Get file path for archive storage"""
        year_str = record.timestamp.strftime("%Y")
        return self.storage_base_path / "archive" / record.symbol.replace('/', '_') / year_str / f"{record.record_id}.gz"
    
    async def _save_record_to_file(self, 
                                 record: LedgerRecord, 
                                 file_path: Path, 
                                 compress: bool = False) -> None:
        """Save record to file with optional compression"""
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize record
        record_data = asdict(record)
        record_data["timestamp"] = record.timestamp.isoformat()
        
        if compress:
            # Save as compressed pickle
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(record_data, f)
        else:
            # Save as JSON
            with open(file_path, 'w') as f:
                json.dump(record_data, f, indent=2)
    
    def _initialize_storage_structure(self) -> None:
        """Initialize storage directory structure"""
        storage_dirs = [
            self.storage_base_path / "mid_term",
            self.storage_base_path / "long_term", 
            self.storage_base_path / "archive",
            self.storage_base_path / "indices"
        ]
        
        for dir_path in storage_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        # Storage optimization task
        optimize_task = asyncio.create_task(self._storage_optimization_loop())
        self.background_tasks.append(optimize_task)
        
        # Data aging task
        aging_task = asyncio.create_task(self._data_aging_loop())
        self.background_tasks.append(aging_task)
        
        # Analytics processing task
        analytics_task = asyncio.create_task(self._analytics_processing_loop())
        self.background_tasks.append(analytics_task)
        
        logger.info("Background tasks started")
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks"""
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        logger.info("Background tasks stopped")
    
    async def _storage_optimization_loop(self) -> None:
        """Background task for storage optimization"""
        while self.is_running:
            try:
                # Optimize RAM cache
                if len(self.ram_cache) > self.storage_limits[StorageTier.RAM_CACHE] * 0.8:
                    await self._evict_from_ram_cache()
                
                # Compress old files
                await self._compress_old_files()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in storage optimization: {e}")
                await asyncio.sleep(600)
    
    async def _data_aging_loop(self) -> None:
        """Background task for data aging and migration"""
        while self.is_running:
            try:
                # Age data from mid-term to long-term
                await self._age_mid_term_data()
                
                # Age data from long-term to archive
                await self._age_long_term_data()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in data aging: {e}")
                await asyncio.sleep(7200)
    
    async def _analytics_processing_loop(self) -> None:
        """Background task for analytical processing"""
        while self.is_running:
            try:
                # Process high-importance data for analytics
                await self._process_analytical_data()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in analytics processing: {e}")
                await asyncio.sleep(120)
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        return {
            "ram_cache_size": len(self.ram_cache),
            "mid_term_size": len(self.mid_term_storage),
            "long_term_size": len(self.long_term_storage),
            "archive_size": len(self.archive_storage),
            "total_records": self.stats["records_processed"],
            "storage_operations": self.stats["storage_operations"],
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"]),
            "compression_saved_mb": self.stats["compression_saved_mb"],
            "storage_limits": {tier.value: limit for tier, limit in self.storage_limits.items()},
            "analytical_spine": self.analytical_spine
        } 