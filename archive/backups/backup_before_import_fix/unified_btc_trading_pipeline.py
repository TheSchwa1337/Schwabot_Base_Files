#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified BTC Trading Pipeline 🚀

Complete BTC/USDC trading pipeline integrating all mathematical components:
• BTC Trading Engine + Mathematical Framework Integrator
• Strategy matrices → profit matrices → tensor calculations
• Ghost basket internal state management
• Real mathematical implementations from YAML configs
• Thermal-aware and multi-bit processing
• Entry/exit functions for BTC/USDC trading

Features:
- Complete integration of all mathematical components
- Real BTC/USDC trading logic (not generic arbitrage)
- Strategy matrix to profit matrix pipeline
- Tensor calculations for entry/exit decisions
- Internal state management (ghost baskets)
- Thermal and multi-bit processing integration
"""

import logging


import logging


import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import cupy as cp
    import numpy as np
    USING_CUDA = True
    xp = cp
    _backend = 'cupy (GPU)'
except ImportError:
    try:
        import numpy as np
        USING_CUDA = False
        xp = np
        _backend = 'numpy (CPU)'
    except ImportError:
        xp = None
        _backend = 'none'

logger = logging.getLogger(__name__)
if xp is None:
    logger.warning("❌ NumPy not available for tensor operations")
else:
    logger.info(f"⚡ UnifiedBTCTradingPipeline using {_backend} for tensor operations")


@dataclass
class BTCTradingPipelineConfig:
    """Configuration for BTC trading pipeline."""
    # Trading parameters
    symbol: str = "BTC/USDC"
    base_position_size: float = 0.01  # BTC
    max_positions: int = 10
    profit_target_bp: int = 10  # 0.1%
    stop_loss_bp: int = 5       # 0.05%
    
    # Mathematical parameters
    entropy_threshold: float = 2.5
    fit_threshold: float = 0.85
    confidence_threshold: float = 0.75
    
    # Thermal parameters
    thermal_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'optimal_performance': 65.0,
        'balanced_processing': 75.0,
        'thermal_efficient': 85.0,
        'emergency_throttle': 90.0,
        'critical_protection': 95.0
    })
    
    # Bit level parameters
    bit_level_configs: Dict[int, Dict[str, Any]] = field(default_factory=lambda: {
        4: {'signal_strength': 'noise', 'confidence_threshold': 0.9, 'position_multiplier': 0.3},
        8: {'signal_strength': 'low', 'confidence_threshold': 0.8, 'position_multiplier': 0.5},
        16: {'signal_strength': 'medium', 'confidence_threshold': 0.75, 'position_multiplier': 1.0},
        32: {'signal_strength': 'high', 'confidence_threshold': 0.7, 'position_multiplier': 1.2},
        42: {'signal_strength': 'critical', 'confidence_threshold': 0.65, 'position_multiplier': 1.5},
        64: {'signal_strength': 'critical', 'confidence_threshold': 0.6, 'position_multiplier': 1.8}
    })


@dataclass
class BTCTradingSignal:
    """BTC trading signal with complete mathematical analysis."""
    signal_type: str  # 'buy', 'sell', 'hold'
    price: float
    amount: float
    confidence: float
    tensor_score: float
    bit_phase: int
    thermal_state: float
    basket_id: str
    hash_value: str
    mathematical_analysis: Dict[str, Any]
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class BTCTradingResult:
    """Result from BTC trading pipeline processing."""
    success: bool
    signal: Optional[BTCTradingSignal]
    mathematical_summary: Dict[str, Any]
    ghost_basket_update: Dict[str, Any]
    execution_recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedBTCTradingPipeline:
    """
    Unified BTC Trading Pipeline integrating all mathematical components.
    Handles complete BTC/USDC trading from price data to execution signals.
    """
    
    def __init__(self, config: Optional[BTCTradingPipelineConfig] = None):
        self.config = config or BTCTradingPipelineConfig()
        
        # Import core components
        try:
            from core.btc_usdc_trading_engine import BTCTradingEngine
            from core.mathematical_framework_integrator import MathematicalFrameworkIntegrator
            from core.profit_optimization_engine import profit_optimization_engine
            from core.risk_manager import risk_manager
            from core.secure_exchange_manager import exchange_manager
            
            self.btc_engine = BTCTradingEngine()
            self.math_integrator = MathematicalFrameworkIntegrator()
            self.exchange_manager = exchange_manager
            self.risk_manager = risk_manager
            self.profit_optimizer = profit_optimization_engine
            
            self.components_available = True
            logger.info("✅ All core components imported successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ Some core components not available: {e}")
            self.components_available = False
        
        # Initialize pipeline state
        self.price_history: List[Dict[str, Any]] = []
        self.trading_signals: List[BTCTradingSignal] = []
        self.ghost_baskets: Dict[str, Dict[str, Any]] = {}
        self.tick_counter = 0
        
        logger.info("✅ Unified BTC Trading Pipeline initialized")
    
    def process_btc_price(self, price: float, volume: float, 
                         thermal_state: float = 65.0) -> BTCTradingResult:
        """Process BTC price data through complete trading pipeline."""
        try:
            self.tick_counter += 1
            
            # Generate hash from price data
            price_str = f"{price:.2f}_{volume:.2f}_{self.tick_counter}"
            hash_value = hashlib.sha256(price_str.encode()).hexdigest()
            
            # Store price data
            price_data = {
                'timestamp': int(time.time() * 1000),
                'price': price,
                'volume': volume,
                'hash_value': hash_value,
                'tick': self.tick_counter,
                'thermal_state': thermal_state
            }
            self.price_history.append(price_data)
            
            # Keep only recent history
            if len(self.price_history) > 1000:
                self.price_history.pop(0)
            
            # Process through mathematical framework
            if self.components_available:
                mathematical_result = self._process_mathematical_framework(
                    price, volume, hash_value, self.tick_counter
                )
            else:
                mathematical_result = self._process_mathematical_fallback(
                    price, volume, hash_value, self.tick_counter
                )
            
            # Generate trading signal
            signal = self._generate_trading_signal(
                price, volume, thermal_state, hash_value, mathematical_result
            )
            
            # Update ghost basket
            ghost_basket_update = self._update_ghost_basket(signal, mathematical_result)
            
            # Determine execution recommendation
            execution_recommendation = self._determine_execution_recommendation(
                signal, mathematical_result, thermal_state
            )
            
            result = BTCTradingResult(
                success=True,
                signal=signal,
                mathematical_summary=mathematical_result,
                ghost_basket_update=ghost_basket_update,
                execution_recommendation=execution_recommendation,
                metadata={
                    'tick': self.tick_counter,
                    'thermal_state': thermal_state,
                    'hash_value': hash_value
                }
            )
            
            if signal and signal.signal_type != 'hold':
                logger.info(f"📊 BTC Signal: {signal.signal_type.upper()} @ {price:.2f}, "
                           f"confidence={signal.confidence:.3f}, basket={signal.basket_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process BTC price: {e}")
            return BTCTradingResult(
                success=False,
                signal=None,
                mathematical_summary={},
                ghost_basket_update={},
                execution_recommendation="error"
            )
    
    def _process_mathematical_framework(self, price: float, volume: float,
                                      hash_value: str, tick: int) -> Dict[str, Any]:
        """Process through complete mathematical framework."""
        try:
            # Get entry price from recent history
            entry_price = self._get_entry_price()
            
            # Process through mathematical framework integrator
            result = self.math_integrator.process_btc_trading_complete(
                price=price,
                volume=volume,
                entry_price=entry_price,
                hash_value=hash_value,
                tick=tick
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process mathematical framework: {e}")
            return self._process_mathematical_fallback(price, volume, hash_value, tick)
    
    def _process_mathematical_fallback(self, price: float, volume: float,
                                     hash_value: str, tick: int) -> Dict[str, Any]:
        """Fallback mathematical processing when components not available."""
        try:
            # Simple mathematical calculations
            entry_price = self._get_entry_price()
            
            # Calculate basic metrics
            price_change = (price - entry_price) / entry_price if entry_price > 0 else 0.0
            
            # Simple bit phase calculation
            bit_phase = int(hash_value[:4], 16) % 65536
            
            # Simple tensor score
            tensor_score = price_change * (bit_phase + 1)
            
            # Simple entropy calculation
            if len(self.price_history) >= 10:
                recent_prices = [p['price'] for p in self.price_history[-10:]]
                price_changes = xp.diff(recent_prices)
                entropy = float(xp.std(price_changes)) if len(price_changes) > 0 else 0.0
            else:
                entropy = 0.0
            
            return {
                'waveform': {
                    'waveform_value': 0.0,
                    'entropy': entropy,
                    'bit_phase': bit_phase,
                    'tensor_score': tensor_score,
                    'confidence': min(1.0, abs(tensor_score) / 5.0)
                },
                'matrix': {
                    'basket_id': f"basket_{bit_phase % 1024:04d}",
                    'tensor_score': tensor_score,
                    'confidence': min(1.0, abs(tensor_score) / 5.0),
                    'bit_phase': bit_phase
                },
                'profit': {
                    'allocation_success': tensor_score > 0,
                    'allocated_amount': volume * abs(tensor_score) if tensor_score > 0 else 0.0,
                    'profit_score': tensor_score,
                    'confidence': min(1.0, abs(tensor_score) / 5.0),
                    'basket_id': f"basket_{bit_phase % 1024:04d}"
                },
                'btc_encoding': {
                    'original_price': price,
                    'decoded_price': price,
                    'bit_depth': 16,
                    'encoding_accuracy': 1.0
                },
                'summary': {
                    'overall_confidence': min(1.0, abs(tensor_score) / 5.0),
                    'trading_signal': 'buy' if tensor_score > 0.5 else 'sell' if tensor_score < -0.5 else 'hold',
                    'backend': _backend
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process mathematical fallback: {e}")
            return {
                'summary': {
                    'overall_confidence': 0.0,
                    'trading_signal': 'hold',
                    'backend': _backend
                }
            }
    
    def _get_entry_price(self) -> float:
        """Get entry price from recent history."""
        try:
            if len(self.price_history) >= 10:
                # Use average of last 10 prices as entry
                recent_prices = [p['price'] for p in self.price_history[-10:]]
                return float(xp.mean(recent_prices))
            elif len(self.price_history) > 0:
                return self.price_history[0]['price']
            else:
                return 50000.0  # Default BTC price
        except Exception:
            return 50000.0
    
    def _generate_trading_signal(self, price: float, volume: float, thermal_state: float,
                               hash_value: str, mathematical_result: Dict[str, Any]) -> Optional[BTCTradingSignal]:
        """Generate trading signal based on mathematical analysis."""
        try:
            summary = mathematical_result.get('summary', {})
            overall_confidence = summary.get('overall_confidence', 0.0)
            trading_signal = summary.get('trading_signal', 'hold')
            
            # Check confidence threshold
            if overall_confidence < self.config.confidence_threshold:
                return None
            
            # Check if signal is hold
            if trading_signal == 'hold':
                return None
            
            # Determine thermal mode
            thermal_mode = self._determine_thermal_mode(thermal_state)
            
            # Get bit phase
            bit_phase = mathematical_result.get('waveform', {}).get('bit_phase', 0)
            
            # Calculate position size
            amount = self._calculate_position_size(price, thermal_mode, bit_phase)
            
            # Get basket ID
            basket_id = mathematical_result.get('matrix', {}).get('basket_id', 'basket_0000')
            
            # Get tensor score
            tensor_score = mathematical_result.get('waveform', {}).get('tensor_score', 0.0)
            
            signal = BTCTradingSignal(
                signal_type=trading_signal,
                price=price,
                amount=amount,
                confidence=overall_confidence,
                tensor_score=tensor_score,
                bit_phase=bit_phase,
                thermal_state=thermal_state,
                basket_id=basket_id,
                hash_value=hash_value,
                mathematical_analysis=mathematical_result
            )
            
            self.trading_signals.append(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Failed to generate trading signal: {e}")
            return None
    
    def _determine_thermal_mode(self, thermal_state: float) -> str:
        """Determine trading mode based on thermal state."""
        thresholds = self.config.thermal_thresholds
        
        if thermal_state <= thresholds.get('optimal_performance', 65.0):
            return "optimal_aggressive"
        elif thermal_state <= thresholds.get('balanced_processing', 75.0):
            return "balanced_consistent"
        elif thermal_state <= thresholds.get('thermal_efficient', 85.0):
            return "efficient_conservative"
        elif thermal_state <= thresholds.get('emergency_throttle', 90.0):
            return "throttle_safety"
        else:
            return "critical_halt"
    
    def _calculate_position_size(self, price: float, thermal_mode: str, bit_phase: int) -> float:
        """Calculate position size based on thermal mode and bit phase."""
        try:
            base_position = self.config.base_position_size
            
            # Get bit level
            if bit_phase < 16:
                bit_level = 4
            elif bit_phase < 256:
                bit_level = 8
            elif bit_phase < 65536:
                bit_level = 16
            elif bit_phase < 4294967296:
                bit_level = 32
            elif bit_phase < 2**42:
                bit_level = 42
            else:
                bit_level = 64
            
            # Get bit level multiplier
            bit_config = self.config.bit_level_configs.get(bit_level, {})
            position_multiplier = bit_config.get('position_multiplier', 1.0)
            
            # Thermal mode multiplier
            thermal_multipliers = {
                "optimal_aggressive": 1.5,
                "balanced_consistent": 1.0,
                "efficient_conservative": 0.7,
                "throttle_safety": 0.3,
                "critical_halt": 0.0
            }
            thermal_multiplier = thermal_multipliers.get(thermal_mode, 1.0)
            
            position_size = base_position * position_multiplier * thermal_multiplier
            
            return float(xp.clip(position_size, 0.0, base_position))
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate position size: {e}")
            return 0.0
    
    def _update_ghost_basket(self, signal: Optional[BTCTradingSignal], 
                           mathematical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update ghost basket with new trading signal."""
        try:
            if not signal or signal.signal_type == 'hold':
                return {'updated': False}
            
            basket_id = signal.basket_id
            
            if basket_id not in self.ghost_baskets:
                self.ghost_baskets[basket_id] = {
                    'total_value': 0.0,
                    'total_pnl': 0.0,
                    'positions': [],
                    'last_update': int(time.time() * 1000)
                }
            
            basket = self.ghost_baskets[basket_id]
            
            # Add position
            position = {
                'signal_type': signal.signal_type,
                'price': signal.price,
                'amount': signal.amount,
                'confidence': signal.confidence,
                'tensor_score': signal.tensor_score,
                'bit_phase': signal.bit_phase,
                'timestamp': signal.timestamp
            }
            
            basket['positions'].append(position)
            basket['total_value'] += signal.amount * signal.price
            basket['last_update'] = signal.timestamp
            
            # Calculate PnL
            if signal.signal_type == 'buy':
                basket['total_pnl'] += signal.amount * signal.price * 0.001  # Assume 0.1% profit
            elif signal.signal_type == 'sell':
                basket['total_pnl'] -= signal.amount * signal.price * 0.001  # Assume 0.1% loss
            
            return {
                'updated': True,
                'basket_id': basket_id,
                'total_value': basket['total_value'],
                'total_pnl': basket['total_pnl'],
                'position_count': len(basket['positions'])
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to update ghost basket: {e}")
            return {'updated': False, 'error': str(e)}
    
    def _determine_execution_recommendation(self, signal: Optional[BTCTradingSignal],
                                          mathematical_result: Dict[str, Any],
                                          thermal_state: float) -> str:
        """Determine execution recommendation based on analysis."""
        try:
            if not signal:
                return "hold"
            
            # Check thermal state
            if thermal_state > 90.0:
                return "critical_halt"
            elif thermal_state > 85.0:
                return "throttle"
            
            # Check confidence
            if signal.confidence < 0.8:
                return "review"
            
            # Check tensor score
            if abs(signal.tensor_score) < 0.5:
                return "weak_signal"
            
            # Check position limits
            if len(self.trading_signals) >= self.config.max_positions:
                return "position_limit"
            
            # All checks passed
            return "execute"
            
        except Exception as e:
            logger.error(f"❌ Failed to determine execution recommendation: {e}")
            return "error"
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        try:
            return {
                'trading_signals': len(self.trading_signals),
                'recent_signals': len([s for s in self.trading_signals if s.timestamp > time.time() * 1000 - 3600000]),  # Last hour
                'ghost_baskets': len(self.ghost_baskets),
                'price_history_size': len(self.price_history),
                'tick_counter': self.tick_counter,
                'config': {
                    'symbol': self.config.symbol,
                    'base_position_size': self.config.base_position_size,
                    'max_positions': self.config.max_positions,
                    'profit_target_bp': self.config.profit_target_bp,
                    'stop_loss_bp': self.config.stop_loss_bp
                },
                'components_available': self.components_available,
                'backend': _backend
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get pipeline summary: {e}")
            return {"error": str(e)}
    
    def get_ghost_basket_summary(self) -> Dict[str, Any]:
        """Get summary of all ghost baskets."""
        try:
            summary = {}
            for basket_id, basket in self.ghost_baskets.items():
                summary[basket_id] = {
                    'total_value': basket['total_value'],
                    'total_pnl': basket['total_pnl'],
                    'position_count': len(basket['positions']),
                    'last_update': basket['last_update']
                }
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get ghost basket summary: {e}")
            return {"error": str(e)}


# Singleton instance for global use
unified_btc_trading_pipeline = UnifiedBTCTradingPipeline() 