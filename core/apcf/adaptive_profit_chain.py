"""
Adaptive Profit Chain Framework (APCF)
Core implementation of the fractal-hive profit chain system.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch
import logging
from pathlib import Path
import json
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # 'high_vol', 'rangebound', 'trending', 'breakout', 'consolidation'
    volatility: float
    trend_strength: float
    volume_profile: float
    confidence: float
    timestamp: datetime = datetime.now()

@dataclass
class SpectralBucket:
    """Spectral analysis bucket for profit opportunities"""
    center_time: float
    period: float
    price_range: Tuple[float, float]
    confidence: float
    frequency: float
    regime: MarketRegime
    volume_profile: float
    correlation: float

@dataclass
class ProfitLink:
    """Link in the profit chain"""
    entry_bucket: SpectralBucket
    exit_bucket: SpectralBucket
    direction: str  # 'long' or 'short'
    expected_profit: float
    risk_ratio: float
    formality_stretch: float
    confidence: float
    volume_requirement: float

class FormalityStretcher:
    """Dynamically adjusts trading strictness based on market regime"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.regime_params = self.config.get('formality_stretching', {})
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tesseract" / "fractal_hive_config.yaml"
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return {}
            
    def get_stretch_params(self, regime: MarketRegime) -> Dict[str, float]:
        """Returns adjusted parameters based on market regime"""
        params = self.regime_params.get(regime.regime_type, self.regime_params.get('rangebound', {}))
        
        # Adjust based on regime confidence
        confidence_multiplier = 0.5 + (regime.confidence * 0.5)  # 0.5 to 1.0 range
        
        return {
            'profit_multiplier': params.get('profit_multiplier', 1.0) * confidence_multiplier,
            'stop_multiplier': params.get('stop_multiplier', 1.0) * confidence_multiplier,
            'bucket_tolerance': params.get('bucket_tolerance', 0.2) * confidence_multiplier
        }
        
    def classify_regime(self, price_series: np.ndarray, volume_series: np.ndarray) -> MarketRegime:
        """Classify current market regime based on price/volume analysis"""
        # Calculate volatility metrics
        returns = np.diff(price_series) / price_series[:-1]
        volatility = np.std(returns) * np.sqrt(len(returns))
        
        # Calculate trend strength
        sma_20 = np.convolve(price_series, np.ones(20)/20, mode='valid')
        trend_strength = abs(price_series[-1] - sma_20[-1]) / sma_20[-1]
        
        # Volume analysis
        avg_volume = np.mean(volume_series)
        recent_volume = np.mean(volume_series[-10:])
        volume_profile = recent_volume / avg_volume
        
        # Classification logic
        if volatility > 0.05:
            regime_type = 'high_vol'
            confidence = min(1.0, volatility * 10)
        elif trend_strength > 0.03 and volume_profile > 1.2:
            regime_type = 'breakout'
            confidence = min(1.0, trend_strength * 20)
        elif trend_strength > 0.02:
            regime_type = 'trending'
            confidence = min(1.0, trend_strength * 15)
        elif volatility < 0.01:
            regime_type = 'consolidation'
            confidence = min(1.0, (1 - volatility) * 50)
        else:
            regime_type = 'rangebound'
            confidence = 0.7
            
        return MarketRegime(
            regime_type=regime_type,
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            confidence=confidence
        )

class FourierBucketizer:
    """Spectral analysis to identify cyclic profit opportunities"""
    
    def __init__(self, price_series: np.ndarray, volume_series: np.ndarray):
        self.price_series = price_series
        self.volume_series = volume_series
        self.sample_rate = 1.0  # Assuming 1 sample per time unit
        
    def find_spectral_buckets(self, min_confidence: float = 0.6) -> List[SpectralBucket]:
        """Find high-probability flip windows using FFT analysis"""
        # Perform FFT on price data
        price_fft = fft(self.price_series)
        frequencies = fftfreq(len(self.price_series), self.sample_rate)
        
        # Get power spectrum
        power_spectrum = np.abs(price_fft)**2
        
        # Find dominant frequencies
        peaks, properties = find_peaks(
            power_spectrum[:len(power_spectrum)//2],
            height=np.max(power_spectrum) * 0.1
        )
        
        buckets = []
        for peak in peaks:
            freq = frequencies[peak]
            if freq > 0:  # Only positive frequencies
                period = 1.0 / freq
                confidence = power_spectrum[peak] / np.max(power_spectrum)
                
                if confidence >= min_confidence and period > 2:  # Filter noise
                    # Calculate price range for this frequency component
                    reconstructed = self._reconstruct_component(freq, len(self.price_series))
                    price_range = (np.min(reconstructed), np.max(reconstructed))
                    
                    # Calculate volume profile and correlation
                    volume_profile = self._calculate_volume_profile(period)
                    correlation = self._calculate_correlation(period)
                    
                    # Create bucket centered at period intervals
                    for i in range(0, len(self.price_series), int(period)):
                        if i + period/2 < len(self.price_series):
                            bucket = SpectralBucket(
                                center_time=i + period/2,
                                period=period,
                                price_range=price_range,
                                confidence=confidence,
                                frequency=freq,
                                regime=MarketRegime('rangebound', 0, 0, 0, 0),  # Will be updated
                                volume_profile=volume_profile,
                                correlation=correlation
                            )
                            buckets.append(bucket)
        
        return self._filter_overlapping_buckets(buckets)
        
    def _reconstruct_component(self, frequency: float, length: int) -> np.ndarray:
        """Reconstruct the signal component for a specific frequency"""
        t = np.arange(length)
        return np.sin(2 * np.pi * frequency * t)
        
    def _calculate_volume_profile(self, period: float) -> float:
        """Calculate volume profile for a given period"""
        # Implementation depends on volume data structure
        return 1.0
        
    def _calculate_correlation(self, period: float) -> float:
        """Calculate correlation between price and volume for a given period"""
        # Implementation depends on data structure
        return 0.0
        
    def _filter_overlapping_buckets(self, buckets: List[SpectralBucket]) -> List[SpectralBucket]:
        """Remove overlapping buckets, keeping highest confidence ones"""
        filtered = []
        buckets.sort(key=lambda x: x.confidence, reverse=True)
        
        for bucket in buckets:
            overlap = False
            for existing in filtered:
                if abs(bucket.center_time - existing.center_time) < min(bucket.period, existing.period) * 0.5:
                    overlap = True
                    break
            if not overlap:
                filtered.append(bucket)
        
        return filtered

class FlipSequencer:
    """Builds optimal profit chains from spectral buckets"""
    
    def __init__(self, buckets: List[SpectralBucket], stretcher: FormalityStretcher):
        self.buckets = buckets
        self.stretcher = stretcher
        self.profit_chains = []
        
    def build_profit_chains(self) -> List[List[ProfitLink]]:
        """Create sequences of profit links from bucket analysis"""
        chains = []
        
        # Sort buckets by time
        sorted_buckets = sorted(self.buckets, key=lambda x: x.center_time)
        
        for i, entry_bucket in enumerate(sorted_buckets[:-1]):
            for j, exit_bucket in enumerate(sorted_buckets[i+1:], i+1):
                # Check if buckets form a valid profit opportunity
                if self._validate_bucket_pair(entry_bucket, exit_bucket):
                    # Determine direction and expected profit
                    direction = self._determine_direction(entry_bucket, exit_bucket)
                    expected_profit = self._calculate_expected_profit(entry_bucket, exit_bucket, direction)
                    
                    # Get formality stretch parameters
                    stretch_params = self.stretcher.get_stretch_params(entry_bucket.regime)
                    
                    profit_link = ProfitLink(
                        entry_bucket=entry_bucket,
                        exit_bucket=exit_bucket,
                        direction=direction,
                        expected_profit=expected_profit * stretch_params['profit_multiplier'],
                        risk_ratio=self._calculate_risk_ratio(entry_bucket, exit_bucket),
                        formality_stretch=stretch_params['bucket_tolerance'],
                        confidence=min(entry_bucket.confidence, exit_bucket.confidence),
                        volume_requirement=self._calculate_volume_requirement(entry_bucket, exit_bucket)
                    )
                    
                    # Build chain
                    chain = [profit_link]
                    chains.append(chain)
        
        return self._optimize_chains(chains)
        
    def _validate_bucket_pair(self, entry: SpectralBucket, exit: SpectralBucket) -> bool:
        """Check if bucket pair forms valid profit opportunity"""
        time_diff = exit.center_time - entry.center_time
        return (time_diff > entry.period * 0.5 and 
                time_diff < entry.period * 3 and
                entry.confidence > 0.6 and exit.confidence > 0.6)
        
    def _determine_direction(self, entry: SpectralBucket, exit: SpectralBucket) -> str:
        """Determine trade direction based on bucket price ranges"""
        entry_mid = sum(entry.price_range) / 2
        exit_mid = sum(exit.price_range) / 2
        return 'long' if exit_mid > entry_mid else 'short'
        
    def _calculate_expected_profit(self, entry: SpectralBucket, exit: SpectralBucket, direction: str) -> float:
        """Calculate expected profit percentage"""
        entry_price = sum(entry.price_range) / 2
        exit_price = sum(exit.price_range) / 2
        
        if direction == 'long':
            return (exit_price - entry_price) / entry_price
        else:
            return (entry_price - exit_price) / entry_price
            
    def _calculate_risk_ratio(self, entry: SpectralBucket, exit: SpectralBucket) -> float:
        """Calculate risk-to-reward ratio"""
        price_range_entry = entry.price_range[1] - entry.price_range[0]
        expected_move = abs(sum(exit.price_range)/2 - sum(entry.price_range)/2)
        return expected_move / price_range_entry if price_range_entry > 0 else 0
        
    def _calculate_volume_requirement(self, entry: SpectralBucket, exit: SpectralBucket) -> float:
        """Calculate required volume for trade execution"""
        return max(entry.volume_profile, exit.volume_profile)
        
    def _optimize_chains(self, chains: List[List[ProfitLink]]) -> List[List[ProfitLink]]:
        """Optimize chains for maximum risk-adjusted return"""
        # Sort by expected profit / risk ratio
        optimized = sorted(
            chains,
            key=lambda chain: sum(link.expected_profit * link.risk_ratio for link in chain),
            reverse=True
        )
        return optimized[:10]  # Return top 10 chains

class FerrisRunner:
    """Executes profit chains in cyclic fashion"""
    
    def __init__(self, profit_chains: List[List[ProfitLink]]):
        self.profit_chains = profit_chains
        self.current_chain_index = 0
        self.current_link_index = 0
        self.active_positions = []
        self.performance_log = []
        
    def run_cycle(self, current_price: float, current_time: float, current_volume: float) -> Dict[str, Any]:
        """Execute one cycle of the Ferris wheel runner"""
        if not self.profit_chains:
            return {"status": "no_chains", "action": "wait"}
            
        current_chain = self.profit_chains[self.current_chain_index]
        
        if self.current_link_index >= len(current_chain):
            # Move to next chain
            self.current_chain_index = (self.current_chain_index + 1) % len(self.profit_chains)
            self.current_link_index = 0
            return {"status": "chain_complete", "action": "cycle_next"}
            
        current_link = current_chain[self.current_link_index]
        
        # Check if we're in entry bucket window
        if self._in_bucket_window(current_time, current_price, current_volume, current_link.entry_bucket):
            # Execute entry
            action = self._execute_entry(current_link, current_price, current_time)
            return action
            
        # Check if we're in exit bucket window for any active positions
        for position in self.active_positions:
            if self._in_bucket_window(current_time, current_price, current_volume, position['exit_bucket']):
                action = self._execute_exit(position, current_price, current_time)
                return action
                
        return {"status": "waiting", "action": "monitor"}
        
    def _in_bucket_window(self, current_time: float, current_price: float, current_volume: float, bucket: SpectralBucket) -> bool:
        """Check if current conditions are within bucket parameters"""
        time_window = bucket.period * bucket.formality_stretch  # Use formality stretch for window size
        time_in_window = abs(current_time - bucket.center_time) <= time_window
        price_in_range = bucket.price_range[0] <= current_price <= bucket.price_range[1]
        volume_sufficient = current_volume >= bucket.volume_requirement
        return time_in_window and price_in_range and volume_sufficient
        
    def _execute_entry(self, link: ProfitLink, price: float, time: float) -> Dict[str, Any]:
        """Execute entry into position"""
        position = {
            'entry_time': time,
            'entry_price': price,
            'direction': link.direction,
            'exit_bucket': link.exit_bucket,
            'expected_profit': link.expected_profit,
            'link': link
        }
        self.active_positions.append(position)
        self.current_link_index += 1
        
        return {
            "status": "entry_executed",
            "action": "open_position",
            "direction": link.direction,
            "price": price,
            "time": time,
            "confidence": link.confidence
        }
        
    def _execute_exit(self, position: Dict[str, Any], price: float, time: float) -> Dict[str, Any]:
        """Execute exit from position"""
        entry_price = position['entry_price']
        direction = position['direction']
        
        if direction == 'long':
            profit = (price - entry_price) / entry_price
        else:
            profit = (entry_price - price) / entry_price
            
        # Log performance
        self.performance_log.append({
            'entry_time': position['entry_time'],
            'exit_time': time,
            'entry_price': entry_price,
            'exit_price': price,
            'direction': direction,
            'profit': profit,
            'expected_profit': position['expected_profit']
        })
        
        # Remove from active positions
        self.active_positions.remove(position)
        
        return {
            "status": "exit_executed",
            "action": "close_position",
            "profit": profit,
            "price": price,
            "time": time
        }

class APCFSystem:
    """Main Adaptive Profit Chain Framework system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.stretcher = FormalityStretcher(config_path)
        self.bucketizer = None
        self.sequencer = None
        self.runner = None
        self.historical_data = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tesseract" / "fractal_hive_config.yaml"
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return {}
            
    def load_backtest_data(self, file_path: str) -> pd.DataFrame:
        """Load historical data for backtesting"""
        # Assuming CSV format with columns: timestamp, price, volume
        data = pd.read_csv(file_path)
        self.historical_data = data
        return data
        
    def initialize_system(self, price_series: np.ndarray, volume_series: np.ndarray):
        """Initialize all system components"""
        # Initialize bucketizer
        self.bucketizer = FourierBucketizer(price_series, volume_series)
        
        # Find spectral buckets
        buckets = self.bucketizer.find_spectral_buckets()
        
        # Classify regime and update buckets
        regime = self.stretcher.classify_regime(price_series, volume_series)
        for bucket in buckets:
            bucket.regime = regime
            
        # Initialize sequencer and build chains
        self.sequencer = FlipSequencer(buckets, self.stretcher)
        profit_chains = self.sequencer.build_profit_chains()
        
        # Initialize runner
        self.runner = FerrisRunner(profit_chains)
        
        logger.info(f"System initialized with {len(buckets)} buckets and {len(profit_chains)} profit chains")
        
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete backtest on historical data"""
        results = []
        
        for i, row in data.iterrows():
            if i < 50:  # Need minimum data for analysis
                continue
                
            # Get recent data window
            window = data.iloc[max(0, i-50):i]
            price_series = window['price'].values
            volume_series = window['volume'].values
            
            # Reinitialize system with updated data every 20 periods
            if i % 20 == 0:
                self.initialize_system(price_series, volume_series)
                
            # Run cycle
            if self.runner:
                result = self.runner.run_cycle(
                    current_price=row['price'],
                    current_time=i,
                    current_volume=row['volume']
                )
                results.append({
                    'timestamp': row['timestamp'],
                    'price': row['price'],
                    'volume': row['volume'],
                    'action': result
                })
                
        return {
            'trades': results,
            'performance': self.runner.performance_log if self.runner else [],
            'summary': self._calculate_summary_stats()
        }
        
    def _calculate_summary_stats(self) -> Dict[str, float]:
        """Calculate performance summary statistics"""
        if not self.runner or not self.runner.performance_log:
            return {}
            
        profits = [trade['profit'] for trade in self.runner.performance_log]
        
        return {
            'total_trades': len(profits),
            'win_rate': len([p for p in profits if p > 0]) / len(profits) if profits else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'total_return': sum(profits) if profits else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0,
            'sharpe_ratio': np.mean(profits) / np.std(profits) if profits and np.std(profits) > 0 else 0
        } 