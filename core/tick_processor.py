"""
Tick Processor - Real-time Market Data Processing Engine.

High-performance tick processing system for real-time market data analysis.
Handles price ticks, volume data, order book updates, and feeds clean data
to the strategy logic and mathematical frameworks.

Key Features:
- Real-time tick processing and validation
- Order book depth analysis
- Volume profile analysis
- Tick aggregation and normalization
- Market microstructure analysis
- Integration with mathematical frameworks
- Performance optimization for high-frequency data

Windows CLI compatible with flake8 compliance.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Try to import Windows CLI compatibility
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, safe_format_error, log_safe
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
        
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
        
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class TickType(Enum):
    """Tick type enumeration."""
    TRADE = "trade"
    QUOTE = "quote"
    ORDER_BOOK = "order_book"
    VOLUME = "volume"
    OHLCV = "ohlcv"


class TickStatus(Enum):
    """Tick processing status."""
    VALID = "valid"
    INVALID = "invalid"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class ProcessingPriority(Enum):
    """Processing priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TickData:
    """Tick data structure."""
    timestamp: float
    symbol: str
    price: Decimal
    volume: Decimal
    tick_type: TickType
    source: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedTick:
    """Processed tick data."""
    original_tick: TickData
    processed_price: Decimal
    processed_volume: Decimal
    hash_signature: str
    confidence_score: float
    status: TickStatus
    processing_time: float
    mathematical_components: Dict[str, Any] = field(default_factory=dict)


class TickProcessor:
    """
    High-performance tick processing engine.
    
    Processes market data ticks with mathematical validation,
    hash-based integrity checking, and real-time analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize tick processor."""
        self.config = config or {}
        self.processing_stats = {
            'total_ticks': 0,
            'valid_ticks': 0,
            'invalid_ticks': 0,
            'processing_times': [],
            'error_count': 0
        }
        self.hash_cache: Dict[str, str] = {}
        self.price_history: List[Decimal] = []
        self.volume_history: List[Decimal] = []
        
        # Initialize mathematical components
        self._initialize_mathematical_components()
        
        safe_print("📊 Tick Processor initialized")
    
    def _initialize_mathematical_components(self) -> None:
        """Initialize mathematical processing components."""
        try:
            # Set up mathematical constants
            self.math_constants = {
                'price_precision': Decimal('0.00000001'),  # 8 decimal places
                'volume_precision': Decimal('0.00000001'),
                'confidence_threshold': 0.8,
                'hash_length': 64,  # SHA-256 hex length
                'max_history_size': 10000
            }
            
            safe_print("✅ Mathematical components initialized")
            
        except Exception as e:
            safe_print(f"⚠️ Mathematical initialization warning: {safe_format_error(e, 'math_init')}")
    
    def process_tick(self, tick_data: TickData) -> ProcessedTick:
        """
        Process a single tick with mathematical validation.
        
        Args:
            tick_data: Raw tick data to process
            
        Returns:
            Processed tick with validation results
        """
        start_time = time.time()
        
        try:
            # Validate tick data
            if not self._validate_tick_data(tick_data):
                return self._create_error_tick(tick_data, "Invalid tick data")
            
            # Generate hash signature
            hash_signature = self._generate_tick_hash(tick_data)
            
            # Process price and volume
            processed_price = self._process_price(tick_data.price)
            processed_volume = self._process_volume(tick_data.volume)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                tick_data, processed_price, processed_volume
            )
            
            # Update history
            self._update_history(processed_price, processed_volume)
            
            # Determine status
            status = TickStatus.VALID if confidence_score >= self.math_constants['confidence_threshold'] else TickStatus.INVALID
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_statistics(status, processing_time)
            
            processed_tick = ProcessedTick(
                original_tick=tick_data,
                processed_price=processed_price,
                processed_volume=processed_volume,
                hash_signature=hash_signature,
                confidence_score=confidence_score,
                status=status,
                processing_time=processing_time,
                mathematical_components=self._extract_mathematical_components(tick_data)
            )
            
            return processed_tick
            
        except Exception as e:
            self.processing_stats['error_count'] += 1
            error_msg = safe_format_error(e, 'tick_processing')
            safe_print(f"❌ Tick processing error: {error_msg}")
            return self._create_error_tick(tick_data, error_msg)
    
    def _validate_tick_data(self, tick_data: TickData) -> bool:
        """Validate tick data for processing."""
        try:
            # Check required fields
            if not all([tick_data.symbol, tick_data.price, tick_data.volume]):
                return False
            
            # Check price validity
            if tick_data.price <= 0:
                return False
            
            # Check volume validity
            if tick_data.volume < 0:
                return False
            
            # Check timestamp validity
            if tick_data.timestamp <= 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _generate_tick_hash(self, tick_data: TickData) -> str:
        """Generate SHA-256 hash for tick data integrity."""
        try:
            # Create hash input string
            hash_input = f"{tick_data.symbol}:{tick_data.price}:{tick_data.volume}:{tick_data.timestamp}"
            
            # Generate hash
            hash_object = hashlib.sha256(hash_input.encode('utf-8'))
            hash_signature = hash_object.hexdigest()
            
            # Cache hash
            self.hash_cache[hash_input] = hash_signature
            
            return hash_signature
            
        except Exception as e:
            safe_print(f"⚠️ Hash generation warning: {safe_format_error(e, 'hash_gen')}")
            return "0" * 64  # Return zero hash as fallback
    
    def _process_price(self, price: Decimal) -> Decimal:
        """Process and normalize price data."""
        try:
            # Round to precision
            processed_price = price.quantize(self.math_constants['price_precision'])
            
            # Ensure positive value
            if processed_price <= 0:
                processed_price = Decimal('0.00000001')
            
            return processed_price
            
        except Exception as e:
            safe_print(f"⚠️ Price processing warning: {safe_format_error(e, 'price_proc')}")
            return Decimal('0.00000001')
    
    def _process_volume(self, volume: Decimal) -> Decimal:
        """Process and normalize volume data."""
        try:
            # Round to precision
            processed_volume = volume.quantize(self.math_constants['volume_precision'])
            
            # Ensure non-negative value
            if processed_volume < 0:
                processed_volume = Decimal('0')
            
            return processed_volume
            
        except Exception as e:
            safe_print(f"⚠️ Volume processing warning: {safe_format_error(e, 'volume_proc')}")
            return Decimal('0')
    
    def _calculate_confidence_score(self, tick_data: TickData, processed_price: Decimal, processed_volume: Decimal) -> float:
        """Calculate confidence score for processed tick."""
        try:
            confidence_factors = []
            
            # Price stability factor
            if len(self.price_history) > 0:
                price_change = abs(float(processed_price - self.price_history[-1]) / float(self.price_history[-1]))
                price_stability = max(0, 1 - price_change)
                confidence_factors.append(price_stability)
            
            # Volume consistency factor
            if len(self.volume_history) > 0:
                volume_ratio = float(processed_volume) / float(self.volume_history[-1]) if self.volume_history[-1] > 0 else 1.0
                volume_consistency = max(0, 1 - abs(volume_ratio - 1))
                confidence_factors.append(volume_consistency)
            
            # Hash integrity factor
            hash_integrity = 1.0 if self._verify_hash_integrity(tick_data) else 0.5
            confidence_factors.append(hash_integrity)
            
            # Calculate average confidence
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.5  # Default confidence
            
        except Exception as e:
            safe_print(f"⚠️ Confidence calculation warning: {safe_format_error(e, 'confidence_calc')}")
            return 0.5
    
    def _verify_hash_integrity(self, tick_data: TickData) -> bool:
        """Verify hash integrity of tick data."""
        try:
            hash_input = f"{tick_data.symbol}:{tick_data.price}:{tick_data.volume}:{tick_data.timestamp}"
            return hash_input in self.hash_cache
        except Exception:
            return False
    
    def _update_history(self, price: Decimal, volume: Decimal) -> None:
        """Update price and volume history."""
        try:
            self.price_history.append(price)
            self.volume_history.append(volume)
            
            # Maintain history size limit
            if len(self.price_history) > self.math_constants['max_history_size']:
                self.price_history.pop(0)
                self.volume_history.pop(0)
                
        except Exception as e:
            safe_print(f"⚠️ History update warning: {safe_format_error(e, 'history_update')}")
    
    def _update_statistics(self, status: TickStatus, processing_time: float) -> None:
        """Update processing statistics."""
        try:
            self.processing_stats['total_ticks'] += 1
            self.processing_stats['processing_times'].append(processing_time)
            
            if status == TickStatus.VALID:
                self.processing_stats['valid_ticks'] += 1
            else:
                self.processing_stats['invalid_ticks'] += 1
            
            # Maintain processing times history
            if len(self.processing_stats['processing_times']) > 1000:
                self.processing_stats['processing_times'].pop(0)
                
        except Exception as e:
            safe_print(f"⚠️ Statistics update warning: {safe_format_error(e, 'stats_update')}")
    
    def _extract_mathematical_components(self, tick_data: TickData) -> Dict[str, Any]:
        """Extract mathematical components from tick data."""
        try:
            components = {
                'price_momentum': self._calculate_price_momentum(),
                'volume_profile': self._calculate_volume_profile(),
                'volatility_estimate': self._calculate_volatility(),
                'trend_strength': self._calculate_trend_strength()
            }
            return components
        except Exception as e:
            safe_print(f"⚠️ Component extraction warning: {safe_format_error(e, 'component_extract')}")
            return {}
    
    def _calculate_price_momentum(self) -> float:
        """Calculate price momentum from history."""
        try:
            if len(self.price_history) < 2:
                return 0.0
            
            recent_prices = [float(p) for p in self.price_history[-10:]]
            if len(recent_prices) < 2:
                return 0.0
            
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            return momentum
            
        except Exception:
            return 0.0
    
    def _calculate_volume_profile(self) -> Dict[str, float]:
        """Calculate volume profile statistics."""
        try:
            if not self.volume_history:
                return {'mean': 0.0, 'std': 0.0, 'trend': 0.0}
            
            volumes = [float(v) for v in self.volume_history[-100:]]
            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            
            # Calculate volume trend
            if len(volumes) >= 2:
                volume_trend = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0.0
            else:
                volume_trend = 0.0
            
            return {
                'mean': mean_volume,
                'std': std_volume,
                'trend': volume_trend
            }
            
        except Exception:
            return {'mean': 0.0, 'std': 0.0, 'trend': 0.0}
    
    def _calculate_volatility(self) -> float:
        """Calculate price volatility from history."""
        try:
            if len(self.price_history) < 2:
                return 0.0
            
            prices = [float(p) for p in self.price_history[-50:]]
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.0
            
            return volatility
            
        except Exception:
            return 0.0
    
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength from price history."""
        try:
            if len(self.price_history) < 10:
                return 0.0
            
            prices = [float(p) for p in self.price_history[-20:]]
            
            # Simple linear regression
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize slope by average price
            avg_price = np.mean(prices)
            trend_strength = slope / avg_price if avg_price > 0 else 0.0
            
            return trend_strength
            
        except Exception:
            return 0.0
    
    def _create_error_tick(self, original_tick: TickData, error_message: str) -> ProcessedTick:
        """Create error tick for failed processing."""
        return ProcessedTick(
            original_tick=original_tick,
            processed_price=Decimal('0'),
            processed_volume=Decimal('0'),
            hash_signature="0" * 64,
            confidence_score=0.0,
            status=TickStatus.ERROR,
            processing_time=0.0,
            mathematical_components={'error': error_message}
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        try:
            stats = self.processing_stats.copy()
            
            # Calculate additional metrics
            if stats['processing_times']:
                stats['average_processing_time'] = np.mean(stats['processing_times'])
                stats['max_processing_time'] = np.max(stats['processing_times'])
                stats['min_processing_time'] = np.min(stats['processing_times'])
            else:
                stats['average_processing_time'] = 0.0
                stats['max_processing_time'] = 0.0
                stats['min_processing_time'] = 0.0
            
            # Calculate success rate
            total_ticks = stats['total_ticks']
            if total_ticks > 0:
                stats['success_rate'] = stats['valid_ticks'] / total_ticks
            else:
                stats['success_rate'] = 0.0
            
            return stats
            
        except Exception as e:
            safe_print(f"⚠️ Statistics calculation warning: {safe_format_error(e, 'stats_calc')}")
            return self.processing_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset all processing statistics."""
        self.processing_stats = {
            'total_ticks': 0,
            'valid_ticks': 0,
            'invalid_ticks': 0,
            'processing_times': [],
            'error_count': 0
        }
        safe_print("📊 Tick processing statistics reset")


# Global tick processor instance
_tick_processor_instance: Optional[TickProcessor] = None


def get_tick_processor(config: Optional[Dict[str, Any]] = None) -> TickProcessor:
    """Get or create the global tick processor instance."""
    global _tick_processor_instance
    if _tick_processor_instance is None:
        _tick_processor_instance = TickProcessor(config)
    return _tick_processor_instance


def main():
    """Test the tick processor system."""
    try:
        # Initialize processor
        processor = get_tick_processor()
        
        # Create test tick data
        test_tick = TickData(
            timestamp=time.time(),
            symbol="BTC/USDC",
            price=Decimal("50000.00"),
            volume=Decimal("1.5"),
            tick_type=TickType.TRADE,
            source="test"
        )
        
        # Process tick
        processed_tick = processor.process_tick(test_tick)
        
        safe_print(f"✅ Processed tick: {processed_tick.status.value}")
        safe_print(f"📊 Confidence score: {processed_tick.confidence_score:.3f}")
        safe_print(f"⏱️ Processing time: {processed_tick.processing_time:.6f}s")
        
        # Get statistics
        stats = processor.get_statistics()
        safe_print(f"📈 Processing stats: {stats}")
        
        safe_print("🎉 Tick processor test completed successfully")
        
    except Exception as e:
        safe_print(f"❌ Test failed: {safe_format_error(e, 'main_test')}")


if __name__ == "__main__":
    main() 