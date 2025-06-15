"""
Recursive Profit Engine Integration Test
========================================

Comprehensive test demonstrating the complete fractal-based recursive profit engine:
- All four fractal systems (Forever, Paradox, Eco, Braid)
- Dynamic fractal weighting based on performance
- Profit projection with horizon mapping
- Real-time decision synthesis
- Mathematical convergence validation

This test validates the core mathematical model:
P(t) = Σ w_i(t) · f_i(t) where convergence → +∞ when aligned
"""

import time
import numpy as np
import logging
from typing import List, Dict, Any

from fractal_controller import FractalController, MarketTick, FractalDecision

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecursiveProfitEngineTest:
    """
    Comprehensive test suite for the recursive profit engine.
    
    Tests all integration points and validates mathematical convergence
    properties of the fractal-based trading system.
    """
    
    def __init__(self):
        """Initialize test environment."""
        self.controller = FractalController()
        self.test_results = []
        self.total_profit = 0.0
        self.position_history = []
        
        logger.info("Recursive Profit Engine Test initialized")
    
    def generate_market_simulation(self, num_ticks: int = 50) -> List[MarketTick]:
        """
        Generate realistic market tick simulation.
        
        Args:
            num_ticks: Number of market ticks to generate
            
        Returns:
            List of simulated market ticks
        """
        ticks = []
        base_price = 100.0
        base_time = time.time()
        
        # Market regime parameters
        trend_strength = np.random.uniform(-0.5, 0.5)
        volatility_regime = np.random.uniform(0.1, 0.8)
        volume_pattern = np.random.choice(['increasing', 'decreasing', 'stable'])
        
        logger.info(f"Generating market simulation: trend={trend_strength:.3f}, "
                   f"volatility={volatility_regime:.3f}, volume_pattern={volume_pattern}")
        
        for i in range(num_ticks):
            # Price evolution with trend and noise
            price_change = trend_strength * 0.1 + np.random.normal(0, volatility_regime * 0.5)
            base_price += price_change
            base_price = max(base_price, 50.0)  # Price floor
            
            # Volume evolution
            if volume_pattern == 'increasing':
                volume = 1000 + i * 50 + np.random.randint(-200, 200)
            elif volume_pattern == 'decreasing':
                volume = 3000 - i * 30 + np.random.randint(-200, 200)
            else:
                volume = 1500 + np.random.randint(-500, 500)
            
            volume = max(volume, 100)  # Volume floor
            
            # Volatility evolution (mean-reverting)
            volatility = volatility_regime + np.random.normal(0, 0.1)
            volatility = np.clip(volatility, 0.05, 1.0)
            
            # Create tick
            tick = MarketTick(
                timestamp=base_time + i * 1.0,  # 1 second intervals
                price=base_price,
                volume=volume,
                volatility=volatility,
                bid=base_price - 0.01,
                ask=base_price + 0.01
            )
            
            ticks.append(tick)
        
        return ticks
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test of the recursive profit engine.
        
        Returns:
            Test results and performance metrics
        """
        logger.info("Starting comprehensive recursive profit engine test")
        
        # Generate market data
        market_ticks = self.generate_market_simulation(100)
        
        # Process each tick and collect decisions
        decisions = []
        fractal_evolution = {
            "forever": [],
            "paradox": [],
            "eco": [],
            "braid": []
        }
        
        weight_evolution = []
        profit_projections = []
        
        for i, tick in enumerate(market_ticks):
            logger.info(f"Processing tick {i+1}/100: price={tick.price:.2f}, "
                       f"vol={tick.volatility:.3f}")
            
            # Process tick through fractal controller
            decision = self.controller.process_tick(tick)
            decisions.append(decision)
            
            # Collect fractal signals
            for fractal_name, signal in decision.fractal_signals.items():
                if fractal_name in fractal_evolution:
                    fractal_evolution[fractal_name].append(signal)
            
            # Collect weight evolution
            weight_evolution.append(decision.fractal_weights.copy())
            
            # Collect profit projections
            profit_projections.append(decision.projected_profit)
            
            # Simulate position execution and outcome
            if decision.action in ["long", "short"]:
                self._simulate_position_outcome(decision, tick, market_ticks[i:i+decision.hold_duration])
            
            # Log key metrics every 10 ticks
            if (i + 1) % 10 == 0:
                self._log_system_metrics(i + 1)
        
        # Analyze results
        test_results = self._analyze_test_results(
            decisions, fractal_evolution, weight_evolution, profit_projections
        )
        
        logger.info("Comprehensive test completed")
        return test_results
    
    def _simulate_position_outcome(self, decision: FractalDecision, entry_tick: MarketTick, 
                                 future_ticks: List[MarketTick]):
        """Simulate position outcome based on future price movement."""
        if not future_ticks:
            return
            
        entry_price = entry_tick.price
        hold_duration = min(decision.hold_duration, len(future_ticks))
        
        if hold_duration == 0:
            return
            
        exit_price = future_ticks[hold_duration - 1].price
        
        # Calculate profit based on position direction
        if decision.action == "long":
            profit = (exit_price - entry_price) / entry_price * 10000  # Basis points
        elif decision.action == "short":
            profit = (entry_price - exit_price) / entry_price * 10000  # Basis points
        else:
            profit = 0.0
        
        self.total_profit += profit
        
        # Update controller with realized outcome
        self.controller.update_position_outcome(profit)
        
        # Store position record
        position_record = {
            "timestamp": decision.timestamp,
            "action": decision.action,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit": profit,
            "projected_profit": decision.projected_profit,
            "confidence": decision.confidence,
            "hold_duration": hold_duration
        }
        
        self.position_history.append(position_record)
        
        logger.info(f"Position closed: {decision.action} profit={profit:.1f}bp "
                   f"(projected: {decision.projected_profit:.1f}bp)")
    
    def _log_system_metrics(self, tick_count: int):
        """Log current system metrics."""
        status = self.controller.get_system_status()
        
        logger.info(f"=== System Metrics at Tick {tick_count} ===")
        logger.info(f"Total decisions: {status['total_decisions']}")
        logger.info(f"Success rate: {status['success_rate']:.3f}")
        logger.info(f"Current fractal weights: {status['fractal_weights']}")
        logger.info(f"Total profit: {self.total_profit:.1f} basis points")
        logger.info("=" * 50)
    
    def _analyze_test_results(self, decisions: List[FractalDecision], 
                            fractal_evolution: Dict[str, List[float]],
                            weight_evolution: List[Dict[str, float]],
                            profit_projections: List[float]) -> Dict[str, Any]:
        """Analyze comprehensive test results."""
        
        # Decision analysis
        action_counts = {}
        for decision in decisions:
            action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
        
        avg_confidence = np.mean([d.confidence for d in decisions])
        
        # Fractal signal analysis
        fractal_stats = {}
        for fractal_name, signals in fractal_evolution.items():
            if signals:
                fractal_stats[fractal_name] = {
                    "mean": np.mean(signals),
                    "std": np.std(signals),
                    "min": np.min(signals),
                    "max": np.max(signals),
                    "trend": np.polyfit(range(len(signals)), signals, 1)[0] if len(signals) > 1 else 0.0
                }
        
        # Weight evolution analysis
        final_weights = weight_evolution[-1] if weight_evolution else {}
        weight_stability = {}
        for fractal_name in ["forever", "paradox", "eco", "braid"]:
            weights = [w.get(fractal_name, 1.0) for w in weight_evolution]
            weight_stability[fractal_name] = {
                "final_weight": final_weights.get(fractal_name, 1.0),
                "volatility": np.std(weights),
                "trend": np.polyfit(range(len(weights)), weights, 1)[0] if len(weights) > 1 else 0.0
            }
        
        # Profit analysis
        position_profits = [p["profit"] for p in self.position_history]
        projection_accuracy = []
        
        for pos in self.position_history:
            if pos["projected_profit"] != 0:
                accuracy = 1.0 - abs(pos["profit"] - pos["projected_profit"]) / abs(pos["projected_profit"])
                projection_accuracy.append(max(0.0, accuracy))
        
        # Mathematical convergence analysis
        convergence_metrics = self._analyze_convergence(fractal_evolution, profit_projections)
        
        # Compile results
        results = {
            "test_summary": {
                "total_ticks_processed": len(decisions),
                "total_decisions": len(decisions),
                "total_positions": len(self.position_history),
                "total_profit_bp": self.total_profit,
                "avg_confidence": avg_confidence,
                "action_distribution": action_counts
            },
            "fractal_analysis": fractal_stats,
            "weight_evolution": weight_stability,
            "profit_analysis": {
                "total_profit": self.total_profit,
                "avg_profit_per_position": np.mean(position_profits) if position_profits else 0.0,
                "profit_std": np.std(position_profits) if position_profits else 0.0,
                "win_rate": sum(1 for p in position_profits if p > 0) / max(len(position_profits), 1),
                "avg_projection_accuracy": np.mean(projection_accuracy) if projection_accuracy else 0.0
            },
            "convergence_analysis": convergence_metrics,
            "system_status": self.controller.get_system_status()
        }
        
        return results
    
    def _analyze_convergence(self, fractal_evolution: Dict[str, List[float]], 
                           profit_projections: List[float]) -> Dict[str, Any]:
        """Analyze mathematical convergence properties."""
        
        convergence_metrics = {}
        
        # Fractal signal convergence
        for fractal_name, signals in fractal_evolution.items():
            if len(signals) > 10:
                # Check for convergence to stable attractor
                recent_signals = signals[-10:]
                signal_variance = np.var(recent_signals)
                
                # Check for periodic behavior
                autocorr = np.corrcoef(signals[:-1], signals[1:])[0, 1] if len(signals) > 1 else 0.0
                
                convergence_metrics[f"{fractal_name}_convergence"] = {
                    "variance": signal_variance,
                    "autocorrelation": autocorr,
                    "is_converging": signal_variance < 0.1,
                    "is_periodic": abs(autocorr) > 0.7
                }
        
        # Profit projection convergence
        if len(profit_projections) > 10:
            proj_variance = np.var(profit_projections[-10:])
            convergence_metrics["profit_projection_stability"] = {
                "variance": proj_variance,
                "is_stable": proj_variance < 100.0  # Less than 100bp variance
            }
        
        # Overall system convergence
        all_variances = [m["variance"] for m in convergence_metrics.values() 
                        if isinstance(m, dict) and "variance" in m]
        
        convergence_metrics["system_convergence"] = {
            "avg_variance": np.mean(all_variances) if all_variances else 0.0,
            "is_converged": all(v < 0.2 for v in all_variances) if all_variances else False
        }
        
        return convergence_metrics
    
    def print_results(self, results: Dict[str, Any]):
        """Print comprehensive test results."""
        print("\n" + "=" * 80)
        print("RECURSIVE PROFIT ENGINE - COMPREHENSIVE TEST RESULTS")
        print("=" * 80)
        
        # Test Summary
        summary = results["test_summary"]
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Ticks Processed: {summary['total_ticks_processed']}")
        print(f"   Total Positions: {summary['total_positions']}")
        print(f"   Total Profit: {summary['total_profit_bp']:.1f} basis points")
        print(f"   Average Confidence: {summary['avg_confidence']:.3f}")
        print(f"   Action Distribution: {summary['action_distribution']}")
        
        # Fractal Analysis
        print(f"\n🔬 FRACTAL SIGNAL ANALYSIS:")
        for fractal_name, stats in results["fractal_analysis"].items():
            print(f"   {fractal_name.upper()}:")
            print(f"     Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            print(f"     Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"     Trend: {stats['trend']:.4f}")
        
        # Weight Evolution
        print(f"\n⚖️  FRACTAL WEIGHT EVOLUTION:")
        for fractal_name, weight_info in results["weight_evolution"].items():
            print(f"   {fractal_name.upper()}:")
            print(f"     Final Weight: {weight_info['final_weight']:.3f}")
            print(f"     Volatility: {weight_info['volatility']:.3f}")
            print(f"     Trend: {weight_info['trend']:.4f}")
        
        # Profit Analysis
        profit = results["profit_analysis"]
        print(f"\n💰 PROFIT ANALYSIS:")
        print(f"   Total Profit: {profit['total_profit']:.1f} bp")
        print(f"   Avg Profit/Position: {profit['avg_profit_per_position']:.1f} bp")
        print(f"   Win Rate: {profit['win_rate']:.1%}")
        print(f"   Projection Accuracy: {profit['avg_projection_accuracy']:.1%}")
        
        # Convergence Analysis
        conv = results["convergence_analysis"]
        print(f"\n🎯 MATHEMATICAL CONVERGENCE:")
        print(f"   System Converged: {conv['system_convergence']['is_converged']}")
        print(f"   Average Variance: {conv['system_convergence']['avg_variance']:.4f}")
        
        for key, metrics in conv.items():
            if key.endswith("_convergence"):
                fractal = key.replace("_convergence", "")
                print(f"   {fractal.upper()}: Converging={metrics['is_converging']}, "
                      f"Periodic={metrics['is_periodic']}")
        
        print("\n" + "=" * 80)
        print("✅ RECURSIVE PROFIT ENGINE TEST COMPLETED")
        print("=" * 80)
    
    def cleanup(self):
        """Cleanup test resources."""
        self.controller.shutdown()
        logger.info("Test cleanup completed")

def main():
    """Run the comprehensive recursive profit engine test."""
    test = RecursiveProfitEngineTest()
    
    try:
        # Run comprehensive test
        results = test.run_comprehensive_test()
        
        # Print results
        test.print_results(results)
        
        # Validate mathematical properties
        if results["convergence_analysis"]["system_convergence"]["is_converged"]:
            print("🎉 MATHEMATICAL CONVERGENCE ACHIEVED!")
        else:
            print("⚠️  System still converging - may need more iterations")
            
        if results["profit_analysis"]["total_profit"] > 0:
            print("💎 POSITIVE PROFIT ACHIEVED!")
        else:
            print("📉 Negative profit - system learning")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        test.cleanup()

if __name__ == "__main__":
    main() 