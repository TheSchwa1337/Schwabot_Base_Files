#!/usr/bin/env python3
"""
Phantom Math Integration Test Harness
====================================

Comprehensive test suite for the complete Phantom Math system.
Tests all components together with live simulation capabilities.

Features:
- Full Phantom Math system integration testing
- Live simulation with realistic market data
- Performance analysis and validation
- Strategy backtesting capabilities
- Cross-component communication testing
"""

import sys
import os
import time
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.phantom_detector import PhantomDetector, PhantomZone
from core.phantom_logger import PhantomLogger
from core.phantom_registry import PhantomRegistry
from strategies.phantom_band_navigator import PhantomBandNavigator, PhantomSignal

class PhantomMathTestHarness:
    """Comprehensive test harness for Phantom Math system."""
    
    def __init__(self):
        """Initialize the test harness."""
        self.detector = PhantomDetector()
        self.logger = PhantomLogger()
        self.registry = PhantomRegistry()
        self.navigator = PhantomBandNavigator()
        
        # Test data storage
        self.test_results = {}
        self.performance_metrics = {}
        self.simulation_data = {}
        
        print("🔮 Phantom Math Test Harness initialized")
    
    def generate_realistic_market_data(self, symbol: str, duration_hours: int = 24, 
                                     base_price: float = 50000.0) -> List[float]:
        """Generate realistic market data with Phantom-like patterns."""
        np.random.seed(42)  # For reproducible results
        
        # Calculate number of ticks (1 tick per second)
        total_ticks = duration_hours * 3600
        
        prices = [base_price]
        
        for i in range(total_ticks):
            # Base volatility
            volatility = 0.0001  # 0.01% per tick
            
            # Add market cycles
            cycle_factor = np.sin(i / 1000) * 0.5 + 1.0
            
            # Add Phantom-like patterns every 30 minutes
            if i % 1800 == 0:  # Every 30 minutes
                # Create flatness period
                for j in range(300):  # 5 minutes of flatness
                    if i + j < total_ticks:
                        change = np.random.normal(0, volatility * 0.1)
                        new_price = prices[-1] * (1 + change)
                        prices.append(new_price)
                i += 300
                continue
            
            # Normal price movement
            change = np.random.normal(0, volatility * cycle_factor)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices[:total_ticks]
    
    def test_phantom_detection(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Test Phantom Zone detection capabilities."""
        print(f"\n🔍 Testing Phantom Detection for {symbol}")
        print("=" * 50)
        
        # Generate test data
        test_prices = self.generate_realistic_market_data(symbol, duration_hours=2)
        
        # Test detection
        detections = []
        window_size = 20
        
        for i in range(window_size, len(test_prices)):
            window = test_prices[i-window_size:i]
            
            if self.detector.detect(window, symbol):
                phantom_zone = self.detector.detect_phantom_zone(window, symbol)
                if phantom_zone:
                    detections.append({
                        'index': i,
                        'price': test_prices[i],
                        'phantom_zone': phantom_zone
                    })
        
        # Analyze results
        results = {
            'total_ticks': len(test_prices),
            'detections': len(detections),
            'detection_rate': len(detections) / len(test_prices),
            'avg_confidence': np.mean([d['phantom_zone'].confidence_score for d in detections]) if detections else 0.0,
            'detections': detections
        }
        
        print(f"Total ticks: {results['total_ticks']}")
        print(f"Phantom detections: {results['detections']}")
        print(f"Detection rate: {results['detection_rate']:.4f}")
        print(f"Average confidence: {results['avg_confidence']:.4f}")
        
        self.test_results['phantom_detection'] = results
        return results
    
    def test_phantom_logging(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Test Phantom Zone logging system."""
        print(f"\n📝 Testing Phantom Logging for {symbol}")
        print("=" * 50)
        
        # Create test Phantom Zones
        test_zones = []
        base_price = 50000.0
        
        for i in range(5):
            entry_tick = base_price + i * 10
            exit_tick = entry_tick + np.random.uniform(-50, 100)
            duration = np.random.uniform(60, 300)
            confidence = np.random.uniform(0.6, 0.9)
            profit = exit_tick - entry_tick
            
            phantom_zone = PhantomZone(
                symbol=symbol,
                entry_tick=entry_tick,
                exit_tick=exit_tick,
                entry_time=time.time() - duration,
                exit_time=time.time(),
                duration=duration,
                entropy_delta=np.random.uniform(0.001, 0.005),
                flatness_score=np.random.uniform(0.02, 0.08),
                similarity_score=np.random.uniform(0.6, 0.8),
                phantom_potential=np.random.uniform(0.4, 0.7),
                confidence_score=confidence,
                hash_signature=f"test_hash_{i}",
                profit_actual=profit
            )
            
            test_zones.append(phantom_zone)
        
        # Test logging
        for phantom_zone in test_zones:
            self.logger.log_zone(
                phantom_zone,
                profit_actual=phantom_zone.profit_actual,
                market_condition="bull" if phantom_zone.profit_actual > 0 else "bear",
                strategy_used="phantom_band_navigator"
            )
        
        # Get statistics
        stats = self.logger.get_phantom_statistics(symbol)
        
        print(f"Logged zones: {len(test_zones)}")
        print(f"Total phantoms in log: {stats['total_phantoms']}")
        print(f"Success rate: {stats['success_rate']:.4f}")
        print(f"Average profit: {stats['avg_profit']:.4f}")
        
        self.test_results['phantom_logging'] = {
            'logged_zones': len(test_zones),
            'statistics': stats
        }
        return self.test_results['phantom_logging']
    
    def test_phantom_registry(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Test Phantom Registry system."""
        print(f"\n🗄️ Testing Phantom Registry for {symbol}")
        print("=" * 50)
        
        # Store test entries
        test_entries = []
        base_price = 50000.0
        
        for i in range(10):
            entry_tick = base_price + i * 20
            exit_tick = entry_tick + np.random.uniform(-100, 200)
            duration = np.random.uniform(120, 600)
            confidence = np.random.uniform(0.5, 0.95)
            profit = exit_tick - entry_tick
            
            hash_sig = self.registry.store_zone(
                symbol=symbol,
                entry_tick=entry_tick,
                exit_tick=exit_tick,
                duration=duration,
                confidence=confidence,
                profit_actual=profit,
                market_condition="bull" if profit > 0 else "bear",
                strategy_used="phantom_band_navigator",
                entropy_delta=np.random.uniform(0.001, 0.006),
                flatness_score=np.random.uniform(0.01, 0.1),
                similarity_score=np.random.uniform(0.5, 0.9),
                phantom_potential=np.random.uniform(0.3, 0.8)
            )
            
            test_entries.append(hash_sig)
        
        # Test pattern matching
        target_features = {
            'price_change': 0.02,
            'duration_normalized': 0.1,
            'confidence': 0.8,
            'entropy_delta': 0.002,
            'flatness_score': 0.05,
            'similarity_score': 0.7,
            'phantom_potential': 0.6,
            'volatility': 0.001
        }
        
        similar_patterns = self.registry.find_similar_patterns(target_features)
        
        # Get statistics
        stats = self.registry.get_registry_statistics()
        
        print(f"Stored entries: {len(test_entries)}")
        print(f"Total registry entries: {stats['total_entries']}")
        print(f"Similar patterns found: {len(similar_patterns)}")
        print(f"Average profit: {stats['avg_profit']:.4f}")
        
        self.test_results['phantom_registry'] = {
            'stored_entries': len(test_entries),
            'similar_patterns': len(similar_patterns),
            'statistics': stats
        }
        return self.test_results['phantom_registry']
    
    def test_phantom_navigation(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Test Phantom Band Navigator strategy."""
        print(f"\n🧭 Testing Phantom Band Navigator for {symbol}")
        print("=" * 50)
        
        # Generate test data
        test_prices = self.generate_realistic_market_data(symbol, duration_hours=4)
        
        # Test strategy
        signals = []
        trades = []
        available_balance = 10000.0
        
        window_size = 20
        
        for i in range(window_size, len(test_prices)):
            window = test_prices[i-window_size:i]
            current_price = test_prices[i]
            
            # Generate signal
            signal = self.navigator.phantom_band_navigator(symbol, window, available_balance)
            
            if signal:
                signals.append(signal)
                
                # Execute signal
                result = self.navigator.execute_signal(signal, current_price)
                if result['action'] in ['enter', 'exit']:
                    trades.append(result)
        
        # Get strategy statistics
        stats = self.navigator.get_strategy_statistics()
        
        print(f"Generated signals: {len(signals)}")
        print(f"Executed trades: {len(trades)}")
        print(f"Total profit: {stats['total_profit']:.4f}")
        print(f"Success rate: {stats['success_rate']:.4f}")
        print(f"Active positions: {stats['active_positions']}")
        
        self.test_results['phantom_navigation'] = {
            'signals': len(signals),
            'trades': len(trades),
            'statistics': stats
        }
        return self.test_results['phantom_navigation']
    
    def test_full_integration(self, symbol: str = "BTC", duration_hours: int = 6) -> Dict[str, Any]:
        """Test full Phantom Math system integration."""
        print(f"\n🔄 Testing Full Phantom Math Integration for {symbol}")
        print("=" * 60)
        
        # Generate comprehensive test data
        test_prices = self.generate_realistic_market_data(symbol, duration_hours)
        
        # Initialize tracking
        detections = []
        signals = []
        trades = []
        registry_entries = []
        
        window_size = 20
        available_balance = 10000.0
        
        # Run full simulation
        for i in range(window_size, len(test_prices)):
            window = test_prices[i-window_size:i]
            current_price = test_prices[i]
            
            # 1. Phantom Detection
            if self.detector.detect(window, symbol):
                phantom_zone = self.detector.detect_phantom_zone(window, symbol)
                if phantom_zone:
                    detections.append(phantom_zone)
                    
                    # 2. Strategy Navigation
                    signal = self.navigator.phantom_band_navigator(symbol, window, available_balance)
                    if signal:
                        signals.append(signal)
                        
                        # 3. Signal Execution
                        result = self.navigator.execute_signal(signal, current_price)
                        if result['action'] == 'enter':
                            trades.append(result)
                        elif result['action'] == 'exit':
                            # 4. Registry Storage
                            if 'phantom_zone' in result:
                                hash_sig = self.registry.store_zone(
                                    symbol=symbol,
                                    entry_tick=result['entry_price'],
                                    exit_tick=result['price'],
                                    duration=result.get('duration', 300),
                                    confidence=result.get('confidence', 0.7),
                                    profit_actual=result['profit'],
                                    market_condition=result.get('market_condition', 'unknown'),
                                    strategy_used='phantom_band_navigator',
                                    entropy_delta=phantom_zone.entropy_delta,
                                    flatness_score=phantom_zone.flatness_score,
                                    similarity_score=phantom_zone.similarity_score,
                                    phantom_potential=phantom_zone.phantom_potential
                                )
                                registry_entries.append(hash_sig)
        
        # Calculate performance metrics
        total_profit = sum(trade['profit'] for trade in trades if 'profit' in trade)
        profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        success_rate = profitable_trades / len(trades) if trades else 0.0
        
        # Get final statistics
        detector_stats = self.detector.get_phantom_statistics()
        logger_stats = self.logger.get_phantom_statistics(symbol)
        registry_stats = self.registry.get_registry_statistics()
        navigator_stats = self.navigator.get_strategy_statistics()
        
        results = {
            'simulation_duration': duration_hours,
            'total_ticks': len(test_prices),
            'phantom_detections': len(detections),
            'signals_generated': len(signals),
            'trades_executed': len(trades),
            'registry_entries': len(registry_entries),
            'total_profit': total_profit,
            'success_rate': success_rate,
            'detector_statistics': detector_stats,
            'logger_statistics': logger_stats,
            'registry_statistics': registry_stats,
            'navigator_statistics': navigator_stats
        }
        
        print(f"Simulation duration: {duration_hours} hours")
        print(f"Total ticks: {len(test_prices)}")
        print(f"Phantom detections: {len(detections)}")
        print(f"Signals generated: {len(signals)}")
        print(f"Trades executed: {len(trades)}")
        print(f"Registry entries: {len(registry_entries)}")
        print(f"Total profit: ${total_profit:.4f}")
        print(f"Success rate: {success_rate:.4f}")
        
        self.test_results['full_integration'] = results
        return results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print(f"\n📊 Generating Performance Report")
        print("=" * 50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'system_performance': {
                'detection_efficiency': self._calculate_detection_efficiency(),
                'strategy_performance': self._calculate_strategy_performance(),
                'registry_efficiency': self._calculate_registry_efficiency(),
                'overall_system_health': self._calculate_system_health()
            },
            'recommendations': self._generate_recommendations()
        }
        
        # Print summary
        print("📈 Performance Summary:")
        print(f"  Detection Efficiency: {report['system_performance']['detection_efficiency']:.2f}")
        print(f"  Strategy Performance: {report['system_performance']['strategy_performance']:.2f}")
        print(f"  Registry Efficiency: {report['system_performance']['registry_efficiency']:.2f}")
        print(f"  Overall System Health: {report['system_performance']['overall_system_health']:.2f}")
        
        return report
    
    def _calculate_detection_efficiency(self) -> float:
        """Calculate Phantom detection efficiency."""
        if 'phantom_detection' not in self.test_results:
            return 0.0
        
        results = self.test_results['phantom_detection']
        detection_rate = results['detection_rate']
        avg_confidence = results['avg_confidence']
        
        # Efficiency based on detection rate and confidence
        efficiency = (detection_rate * 0.6 + avg_confidence * 0.4)
        return min(efficiency, 1.0)
    
    def _calculate_strategy_performance(self) -> float:
        """Calculate strategy performance score."""
        if 'phantom_navigation' not in self.test_results:
            return 0.0
        
        stats = self.test_results['phantom_navigation']['statistics']
        success_rate = stats['success_rate']
        total_profit = stats['total_profit']
        
        # Performance based on success rate and profit
        performance = (success_rate * 0.7 + min(total_profit / 100, 1.0) * 0.3)
        return max(performance, 0.0)
    
    def _calculate_registry_efficiency(self) -> float:
        """Calculate registry efficiency score."""
        if 'phantom_registry' not in self.test_results:
            return 0.0
        
        stats = self.test_results['phantom_registry']['statistics']
        total_entries = stats['total_entries']
        avg_profit = stats['avg_profit']
        
        # Efficiency based on entry count and average profit
        efficiency = (min(total_entries / 100, 1.0) * 0.5 + min(avg_profit / 10, 1.0) * 0.5)
        return min(efficiency, 1.0)
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        detection_efficiency = self._calculate_detection_efficiency()
        strategy_performance = self._calculate_strategy_performance()
        registry_efficiency = self._calculate_registry_efficiency()
        
        # Weighted average
        health = (detection_efficiency * 0.4 + 
                 strategy_performance * 0.4 + 
                 registry_efficiency * 0.2)
        
        return health
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations."""
        recommendations = []
        
        detection_efficiency = self._calculate_detection_efficiency()
        strategy_performance = self._calculate_strategy_performance()
        registry_efficiency = self._calculate_registry_efficiency()
        
        if detection_efficiency < 0.5:
            recommendations.append("Low detection efficiency - consider adjusting Phantom thresholds")
        
        if strategy_performance < 0.5:
            recommendations.append("Low strategy performance - review risk management parameters")
        
        if registry_efficiency < 0.5:
            recommendations.append("Low registry efficiency - check pattern storage and retrieval")
        
        if detection_efficiency > 0.8 and strategy_performance > 0.8:
            recommendations.append("Excellent system performance - consider increasing position sizes")
        
        return recommendations
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite."""
        print("🚀 Starting Comprehensive Phantom Math Test Suite")
        print("=" * 70)
        
        # Run individual component tests
        self.test_phantom_detection("BTC")
        self.test_phantom_logging("BTC")
        self.test_phantom_registry("BTC")
        self.test_phantom_navigation("BTC")
        
        # Run full integration test
        self.test_full_integration("BTC", duration_hours=2)
        
        # Generate performance report
        report = self.generate_performance_report()
        
        print(f"\n✅ Comprehensive test suite completed!")
        print(f"📄 Performance report generated with {len(report['recommendations'])} recommendations")
        
        return report

def main():
    """Main test execution."""
    # Initialize test harness
    harness = PhantomMathTestHarness()
    
    # Run comprehensive test
    report = harness.run_comprehensive_test()
    
    # Save report
    import json
    with open("phantom_math_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Test report saved to phantom_math_test_report.json")

if __name__ == "__main__":
    main() 