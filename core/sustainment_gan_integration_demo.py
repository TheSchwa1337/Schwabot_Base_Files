"""
Sustainment-GAN Integration Demo
===============================

Demonstrates the integration between:
- GAN Filter with sustainment awareness
- 8-principle sustainment framework
- UFS/NCCO file systems
- BTC trading cycle optimization

This shows how all components work together to ensure sustainable
profit extraction from recurring BTC price cycles.
"""

import numpy as np
import time
import logging
from datetime import datetime
from typing import Dict, List, Any

# Import the integrated components
try:
    from .gan_filter import SustainmentAwareGANFilter
    from .mathlib_v3 import SustainmentMathLib, MathematicalContext
    from .sustainment_integration_hooks import EnhancedSustainmentIntegrationHooks
    from .ufs_registry import UFSRegistry
    from .ufs_echo_logger import UFSEchoLogger
    FULL_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Integration components not available: {e}")
    FULL_INTEGRATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BTCTradingCycleManager:
    """
    Manages BTC trading cycles using sustainment-aware GAN filter
    for anomaly detection and profit optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        if not FULL_INTEGRATION_AVAILABLE:
            logger.error("Required components not available")
            return
        
        # Initialize core components
        self.gan_filter = SustainmentAwareGANFilter(
            config=config.get('gan_config', {}),
            input_dim=config.get('input_dim', 32)
        )
        
        self.sustainment_lib = SustainmentMathLib(
            sustainment_threshold=config.get('sustainment_threshold', 0.65)
        )
        
        self.integration_hooks = EnhancedSustainmentIntegrationHooks(
            config=config.get('integration_config', {})
        )
        
        self.ufs_registry = UFSRegistry()
        self.ufs_logger = UFSEchoLogger(
            log_path=config.get('ufs_log_path', 'logs/btc_cycle_ufs.jsonl')
        )
        
        # Register GAN filter with integration hooks
        self.integration_hooks.register_controller('gan_filter', self.gan_filter)
        
        # Trading state
        self.btc_price_history = []
        self.profit_cycles = []
        self.current_cycle = None
        
        logger.info("BTC Trading Cycle Manager initialized with full sustainment integration")
    
    def process_btc_tick(self, price: float, volume: float, timestamp: float = None) -> Dict[str, Any]:
        """
        Process a BTC price tick through the full sustainment pipeline.
        
        Args:
            price: Current BTC price
            volume: Current trading volume
            timestamp: Optional timestamp
            
        Returns:
            Complete analysis including anomaly detection and sustainment metrics
        """
        if not FULL_INTEGRATION_AVAILABLE:
            return {'error': 'Integration not available'}
        
        if timestamp is None:
            timestamp = time.time()
        
        # Store price history
        self.btc_price_history.append({
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })
        
        # Keep last 1000 ticks
        if len(self.btc_price_history) > 1000:
            self.btc_price_history.pop(0)
        
        # Create feature vector for GAN analysis
        feature_vector = self._create_feature_vector(price, volume)
        
        # Run GAN anomaly detection with sustainment
        trading_context = {
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        }
        
        anomaly_metrics, sustainment_vector = self.gan_filter.detect_with_sustainment(
            feature_vector, trading_context
        )
        
        # Calculate cycle opportunity
        cycle_opportunity = self._analyze_cycle_opportunity(
            anomaly_metrics, sustainment_vector, price, volume
        )
        
        # Update UFS registry with current state
        self._update_ufs_state(price, volume, anomaly_metrics, sustainment_vector)
        
        # Get global sustainment state
        global_sustainment = self.integration_hooks.get_global_sustainment_state()
        
        return {
            'timestamp': timestamp,
            'price': price,
            'volume': volume,
            'anomaly_metrics': {
                'anomaly_score': anomaly_metrics.anomaly_score,
                'reconstruction_error': anomaly_metrics.reconstruction_error,
                'is_anomaly': anomaly_metrics.is_anomaly,
                'confidence': anomaly_metrics.confidence
            },
            'sustainment_metrics': {
                'sustainment_index': sustainment_vector.sustainment_index() if sustainment_vector else None,
                'is_sustainable': sustainment_vector.is_sustainable() if sustainment_vector else None,
                'principle_values': sustainment_vector.principles.tolist() if sustainment_vector else None
            },
            'cycle_opportunity': cycle_opportunity,
            'global_sustainment': global_sustainment,
            'profit_signals': self._extract_profit_signals(anomaly_metrics, sustainment_vector),
            'trading_recommendation': self._generate_trading_recommendation(
                anomaly_metrics, sustainment_vector, cycle_opportunity
            )
        }
    
    def _create_feature_vector(self, price: float, volume: float) -> np.ndarray:
        """Create feature vector for GAN analysis"""
        # Technical indicators
        features = []
        
        if len(self.btc_price_history) >= 20:
            recent_prices = [h['price'] for h in self.btc_price_history[-20:]]
            recent_volumes = [h['volume'] for h in self.btc_price_history[-20:]]
            
            # Price-based features
            features.extend([
                price / np.mean(recent_prices),  # Price relative to MA
                (price - recent_prices[-2]) / recent_prices[-2] if len(recent_prices) > 1 else 0,  # Price change
                np.std(recent_prices) / np.mean(recent_prices),  # Price volatility
            ])
            
            # Volume-based features
            features.extend([
                volume / np.mean(recent_volumes),  # Volume relative to MA
                np.std(recent_volumes) / np.mean(recent_volumes),  # Volume volatility
            ])
            
            # Technical momentum
            if len(recent_prices) >= 10:
                short_ma = np.mean(recent_prices[-5:])
                long_ma = np.mean(recent_prices[-10:])
                features.append((short_ma - long_ma) / long_ma)  # Momentum
        else:
            # Default features when insufficient history
            features = [1.0, 0.0, 0.1, 1.0, 0.1, 0.0]
        
        # Pad or truncate to required size
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32], dtype=np.float32)
    
    def _analyze_cycle_opportunity(self, anomaly_metrics, sustainment_vector, 
                                 price: float, volume: float) -> Dict[str, Any]:
        """Analyze current cycle opportunity for profit extraction"""
        opportunity = {
            'cycle_phase': 'unknown',
            'opportunity_score': 0.0,
            'profit_potential': 0.0,
            'risk_level': 'medium',
            'recommended_action': 'hold'
        }
        
        # Determine cycle phase based on anomaly and sustainment
        if anomaly_metrics.is_anomaly and anomaly_metrics.anomaly_score > 0.8:
            if sustainment_vector and sustainment_vector.is_sustainable():
                opportunity['cycle_phase'] = 'profitable_anomaly'
                opportunity['opportunity_score'] = 0.8
                opportunity['profit_potential'] = anomaly_metrics.confidence * 0.1
                opportunity['recommended_action'] = 'buy'
            else:
                opportunity['cycle_phase'] = 'risky_anomaly'
                opportunity['opportunity_score'] = 0.3
                opportunity['risk_level'] = 'high'
                opportunity['recommended_action'] = 'avoid'
        
        elif not anomaly_metrics.is_anomaly and sustainment_vector and sustainment_vector.is_sustainable():
            opportunity['cycle_phase'] = 'stable_profitable'
            opportunity['opportunity_score'] = 0.6
            opportunity['profit_potential'] = 0.05
            opportunity['recommended_action'] = 'hold_long'
        
        elif anomaly_metrics.anomaly_score < 0.3:
            opportunity['cycle_phase'] = 'low_volatility'
            opportunity['opportunity_score'] = 0.4
            opportunity['recommended_action'] = 'accumulate'
        
        return opportunity
    
    def _update_ufs_state(self, price: float, volume: float, 
                         anomaly_metrics, sustainment_vector):
        """Update UFS registry and logging"""
        # Register current state in UFS
        state_path = f"btc/tick_{int(time.time())}"
        state_size = 1024  # Estimated state size
        
        metadata = {
            'price': price,
            'volume': volume,
            'anomaly_score': anomaly_metrics.anomaly_score,
            'is_anomaly': anomaly_metrics.is_anomaly,
            'sustainment_index': sustainment_vector.sustainment_index() if sustainment_vector else None
        }
        
        self.ufs_registry.register(state_path, state_size, metadata)
        
        # Log to UFS echo system
        cluster_id = f"btc_cycle_{datetime.now().strftime('%Y%m%d_%H%M')}"
        strategy_id = "cycle_profit_extraction"
        entropy_signature = anomaly_metrics.anomaly_score
        
        self.ufs_logger.log_cluster_memory(cluster_id, strategy_id, entropy_signature)
    
    def _extract_profit_signals(self, anomaly_metrics, sustainment_vector) -> Dict[str, float]:
        """Extract profit signals from analysis"""
        signals = {
            'anomaly_profit': 1.0 - anomaly_metrics.anomaly_score,
            'sustainment_profit': sustainment_vector.sustainment_index() if sustainment_vector else 0.5,
            'confidence_multiplier': anomaly_metrics.confidence,
            'combined_signal': 0.0
        }
        
        # Calculate combined signal
        base_signal = (signals['anomaly_profit'] + signals['sustainment_profit']) / 2
        signals['combined_signal'] = base_signal * signals['confidence_multiplier']
        
        return signals
    
    def _generate_trading_recommendation(self, anomaly_metrics, sustainment_vector, 
                                       cycle_opportunity) -> Dict[str, Any]:
        """Generate specific trading recommendation"""
        recommendation = {
            'action': 'hold',
            'confidence': 0.5,
            'position_size': 0.0,
            'stop_loss': None,
            'take_profit': None,
            'reasoning': []
        }
        
        # Base recommendation on cycle opportunity
        if cycle_opportunity['recommended_action'] == 'buy':
            recommendation['action'] = 'buy'
            recommendation['confidence'] = cycle_opportunity['opportunity_score']
            recommendation['position_size'] = min(0.1, cycle_opportunity['profit_potential'] * 10)
            recommendation['reasoning'].append("Profitable anomaly detected with sustainable conditions")
        
        elif cycle_opportunity['recommended_action'] == 'hold_long':
            recommendation['action'] = 'hold'
            recommendation['confidence'] = 0.7
            recommendation['reasoning'].append("Stable sustainable conditions favor holding")
        
        elif cycle_opportunity['recommended_action'] == 'accumulate':
            recommendation['action'] = 'buy'
            recommendation['confidence'] = 0.6
            recommendation['position_size'] = 0.05
            recommendation['reasoning'].append("Low volatility presents accumulation opportunity")
        
        # Adjust based on sustainment
        if sustainment_vector and not sustainment_vector.is_sustainable():
            recommendation['confidence'] *= 0.5
            recommendation['position_size'] *= 0.5
            recommendation['reasoning'].append("Reduced confidence due to sustainment concerns")
        
        return recommendation
    
    def start_integration_monitoring(self):
        """Start continuous integration monitoring"""
        if FULL_INTEGRATION_AVAILABLE:
            self.integration_hooks.start_continuous_integration()
            logger.info("Integration monitoring started")
    
    def stop_integration_monitoring(self):
        """Stop continuous integration monitoring"""
        if FULL_INTEGRATION_AVAILABLE:
            self.integration_hooks.stop_continuous_integration()
            logger.info("Integration monitoring stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        if not FULL_INTEGRATION_AVAILABLE:
            return {'status': 'unavailable'}
        
        return {
            'integration_available': FULL_INTEGRATION_AVAILABLE,
            'gan_filter_status': {
                'clusters': len(self.gan_filter.cluster_db),
                'cache_size': len(self.gan_filter.tick_cache),
                'model_accuracy': self.gan_filter.performance_metrics['model_accuracy']
            },
            'ufs_status': self.gan_filter.get_ufs_status(),
            'sustainment_status': self.integration_hooks.get_global_sustainment_state(),
            'integration_metrics': self.integration_hooks.get_integration_metrics(),
            'price_history_size': len(self.btc_price_history),
            'profit_cycles': len(self.profit_cycles)
        }

def demo_btc_cycle_integration():
    """Demonstrate the full BTC cycle integration"""
    print("=== Sustainment-GAN Integration Demo ===")
    
    if not FULL_INTEGRATION_AVAILABLE:
        print("❌ Integration components not available")
        return
    
    # Configuration
    config = {
        'gan_config': {
            'latent_dim': 32,
            'hidden_dim': 64,
            'anomaly_threshold': 0.7,
            'use_gpu': False  # Use CPU for demo
        },
        'sustainment_threshold': 0.65,
        'integration_config': {
            'synthesis_interval': 2.0,
            'correction_interval': 1.0
        },
        'input_dim': 32,
        'ufs_log_path': 'logs/demo_btc_ufs.jsonl'
    }
    
    # Initialize system
    print("🔧 Initializing BTC Trading Cycle Manager...")
    cycle_manager = BTCTradingCycleManager(config)
    
    # Start monitoring
    cycle_manager.start_integration_monitoring()
    
    try:
        # Simulate BTC price data
        print("📈 Simulating BTC price ticks...")
        base_price = 45000.0
        
        for i in range(50):
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            price = base_price * (1 + price_change)
            volume = np.random.uniform(1000, 5000)
            
            # Add some anomalies
            if i in [15, 30, 42]:
                price *= (1.1 if np.random.random() > 0.5 else 0.9)  # 10% spike/drop
                volume *= 3  # Volume surge
            
            # Process tick
            result = cycle_manager.process_btc_tick(price, volume)
            
            if result.get('error'):
                print(f"❌ Error processing tick {i}: {result['error']}")
                continue
            
            # Print significant events
            if result['anomaly_metrics']['is_anomaly'] or (
                result['sustainment_metrics']['sustainment_index'] and 
                result['sustainment_metrics']['sustainment_index'] < 0.5
            ):
                print(f"🚨 Tick {i}: Price=${price:.0f}, "
                      f"Anomaly={result['anomaly_metrics']['is_anomaly']}, "
                      f"SI={result['sustainment_metrics']['sustainment_index']:.3f if result['sustainment_metrics']['sustainment_index'] else 'N/A'}, "
                      f"Action={result['trading_recommendation']['action']}")
            
            base_price = price
            time.sleep(0.1)  # Small delay for demo
        
        # Show final status
        print("\n📊 Final System Status:")
        status = cycle_manager.get_system_status()
        
        print(f"GAN Clusters: {status['gan_filter_status']['clusters']}")
        print(f"Cache Size: {status['gan_filter_status']['cache_size']}")
        print(f"Model Accuracy: {status['gan_filter_status']['model_accuracy']:.3f}")
        
        if status['sustainment_status'].get('sustainment_index'):
            print(f"Global Sustainment Index: {status['sustainment_status']['sustainment_index']:.3f}")
            print(f"Is Sustainable: {status['sustainment_status']['is_sustainable']}")
        
        print(f"Integration Success Rate: {status['integration_metrics']['success_rate']:.2%}")
        
        print("\n✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    
    finally:
        # Cleanup
        cycle_manager.stop_integration_monitoring()
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    demo_btc_cycle_integration() 