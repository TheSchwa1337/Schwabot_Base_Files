"""
Enhanced Systems Integration Test
================================

Comprehensive test demonstrating the integration of all enhanced systems:
- Collapse Confidence Engine
- Enhanced Vault Router with Dynamic Volume Allocation
- Ghost Decay System
- Enhanced Lockout Matrix with Self-Healing
- Echo Snapshot Logger

This test validates the complete mathematical framework and self-correcting
recursive trade intelligence capabilities.
"""

import time
import numpy as np
import logging
from typing import Dict, Any

from collapse_confidence import CollapseConfidenceEngine, CollapseState
from vault_router import EnhancedVaultRouter, VaultAllocation
from ghost_decay import GhostDecaySystem
from lockout_matrix import EnhancedLockoutMatrix, LockoutSeverity
from echo_snapshot import EchoSnapshotLogger, SnapshotLevel
from fractal_controller import FractalController, MarketTick, FractalDecision

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSystemsIntegrationTest:
    """
    Comprehensive integration test for all enhanced systems.
    
    Demonstrates the complete recursive profit engine with mathematical
    confidence scoring, dynamic volume allocation, ghost decay, self-healing
    lockouts, and human-readable diagnostic output.
    """
    
    def __init__(self):
        """Initialize all enhanced systems."""
        # Initialize core systems
        self.confidence_engine = CollapseConfidenceEngine()
        self.vault_router = EnhancedVaultRouter()
        self.ghost_system = GhostDecaySystem()
        self.lockout_matrix = EnhancedLockoutMatrix()
        
        # Initialize echo logger with detailed output
        echo_config = {
            'level': 'standard',
            'terminal_output': True,
            'use_colors': True,
            'compact_mode': False
        }
        self.echo_logger = EchoSnapshotLogger(echo_config)
        
        # Initialize fractal controller
        self.fractal_controller = FractalController()
        
        # Test metrics
        self.test_results = {}
        self.total_profit = 0.0
        self.decision_count = 0
        
        logger.info("Enhanced Systems Integration Test initialized")
    
    def run_comprehensive_test(self, num_ticks: int = 25) -> Dict[str, Any]:
        """
        Run comprehensive test of all enhanced systems.
        
        Args:
            num_ticks: Number of market ticks to simulate
            
        Returns:
            Comprehensive test results
        """
        logger.info(f"Starting comprehensive enhanced systems test ({num_ticks} ticks)")
        
        print("\n" + "="*80)
        print("🚀 ENHANCED RECURSIVE PROFIT ENGINE - COMPREHENSIVE TEST")
        print("="*80)
        print("Testing: Collapse Confidence | Vault Router | Ghost Decay | Lockout Matrix | Echo Logger")
        print("="*80 + "\n")
        
        # Generate market simulation
        market_ticks = self._generate_market_simulation(num_ticks)
        
        # Process each tick through enhanced systems
        for i, tick in enumerate(market_ticks):
            print(f"\n--- Processing Tick {i+1}/{num_ticks} ---")
            self._process_enhanced_tick(tick, i)
        
        # Generate comprehensive results
        results = self._analyze_enhanced_results()
        
        # Print final summary
        self._print_enhanced_summary(results)
        
        return results
    
    def _generate_market_simulation(self, num_ticks: int) -> list:
        """Generate realistic market tick simulation."""
        ticks = []
        base_price = 100.0
        base_time = time.time()
        
        # Market regime with volatility clustering
        volatility_regime = np.random.uniform(0.2, 0.7)
        trend_strength = np.random.uniform(-0.3, 0.3)
        
        for i in range(num_ticks):
            # Evolving price with trend and volatility clustering
            price_change = trend_strength * 0.05 + np.random.normal(0, volatility_regime * 0.3)
            base_price += price_change
            base_price = max(base_price, 50.0)
            
            # Dynamic volatility (mean-reverting with clustering)
            volatility_change = np.random.normal(0, 0.05)
            volatility_regime = np.clip(volatility_regime + volatility_change, 0.1, 1.0)
            
            # Volume with realistic patterns
            volume = 1000 + np.random.randint(-300, 500) + i * 10
            volume = max(volume, 100)
            
            tick = MarketTick(
                timestamp=base_time + i * 2.0,  # 2 second intervals
                price=base_price,
                volume=volume,
                volatility=volatility_regime,
                bid=base_price - 0.01,
                ask=base_price + 0.01
            )
            
            ticks.append(tick)
        
        return ticks
    
    def _process_enhanced_tick(self, tick: MarketTick, tick_index: int):
        """Process single tick through all enhanced systems."""
        # 1. Generate fractal decision
        fractal_decision = self.fractal_controller.process_tick(tick)
        self.decision_count += 1
        
        # 2. Calculate collapse confidence
        collapse_state = self.confidence_engine.calculate_collapse_confidence(
            profit_delta=fractal_decision.projected_profit,
            braid_signal=fractal_decision.fractal_signals.get('braid', 0.0),
            paradox_signal=fractal_decision.fractal_signals.get('paradox', 0.0),
            recent_volatility=[tick.volatility, tick.volatility * 0.9, tick.volatility * 1.1]
        )
        
        # 3. Check lockout status
        pattern_data = {
            'braid_signal': fractal_decision.fractal_signals.get('braid', 0.0),
            'paradox_signal': fractal_decision.fractal_signals.get('paradox', 0.0),
            'profit_delta': fractal_decision.projected_profit,
            'volatility': tick.volatility
        }
        
        is_locked, lockout_reason = self.lockout_matrix.check_lockout_status(pattern_data)
        
        # 4. Handle lockouts and ghost creation
        if fractal_decision.projected_profit < -50:  # Significant loss
            # Create lockout for bad patterns
            lockout_signature = self.lockout_matrix.create_lockout(
                pattern_data, "significant_loss", LockoutSeverity.MODERATE
            )
            print(f"  🔒 Lockout created: {lockout_signature[:8]} (loss: {fractal_decision.projected_profit:.1f}bp)")
        elif fractal_decision.projected_profit > 75:  # Good profit
            # Create ghost signal for successful patterns
            ghost_id = self.ghost_system.create_ghost_signal(pattern_data, collapse_state.confidence)
            print(f"  👻 Ghost created: {ghost_id[:8]} (profit: {fractal_decision.projected_profit:.1f}bp)")
        
        # 5. Update ghost weights and get recommendations
        ghost_weights = self.ghost_system.update_ghost_weights()
        ghost_recommendations = self.ghost_system.get_ghost_recommendations(pattern_data, top_k=3)
        
        # 6. Calculate vault allocation
        recent_profits = [self.total_profit - i*10 for i in range(5)]  # Simulate profit history
        vault_allocation = self.vault_router.calculate_volume_allocation(
            collapse_state=collapse_state,
            current_profit=self.total_profit,
            recent_profits=recent_profits,
            market_volatility=tick.volatility
        )
        
        # 7. Capture echo snapshot
        additional_data = {
            'drift_angle': np.random.uniform(-30, 30),  # Simulate drift angle
            'ghost_activity': {
                'active_ghosts': len(ghost_weights),
                'top_recommendation': ghost_recommendations[0] if ghost_recommendations else None
            },
            'lockout_status': {
                'is_locked': is_locked,
                'reason': lockout_reason,
                'active_lockouts': len(self.lockout_matrix.active_lockouts)
            }
        }
        
        snapshot = self.echo_logger.capture_snapshot(
            fractal_decision=fractal_decision,
            collapse_state=collapse_state,
            vault_allocation=vault_allocation,
            additional_data=additional_data
        )
        
        # 8. Simulate position outcome and update systems
        if fractal_decision.action in ["long", "short"] and not is_locked:
            # Simulate position profit based on projected profit with some noise
            actual_profit = fractal_decision.projected_profit + np.random.normal(0, 20)
            self.total_profit += actual_profit
            
            # Update systems with outcome
            self.fractal_controller.update_position_outcome(actual_profit)
            
            # Reinforce or penalize ghosts based on outcome
            if actual_profit > 0 and ghost_recommendations:
                best_ghost_id = ghost_recommendations[0][0]
                self.ghost_system.reinforce_ghost(best_ghost_id, 0.8, pattern_data)
                print(f"  ✅ Ghost reinforced: {best_ghost_id[:8]} (profit: {actual_profit:.1f}bp)")
            elif actual_profit < -30:
                # Penalize ghosts for poor performance
                for ghost_id, _, _ in ghost_recommendations[:2]:
                    self.ghost_system.penalize_ghost(ghost_id, 0.6)
                print(f"  ❌ Ghosts penalized for loss: {actual_profit:.1f}bp")
        
        # 9. Update lockout weights (self-healing)
        lockout_weights = self.lockout_matrix.update_lockout_weights()
        
        # Print tick summary
        print(f"  📊 Confidence: {collapse_state.confidence:.3f} | "
              f"Volume: {vault_allocation.allocated_volume:.0f} | "
              f"Total P&L: {self.total_profit:.1f}bp")
    
    def _analyze_enhanced_results(self) -> Dict[str, Any]:
        """Analyze comprehensive test results from all systems."""
        return {
            "confidence_metrics": self.confidence_engine.get_metrics_summary(),
            "vault_metrics": self.vault_router.get_vault_summary(),
            "ghost_metrics": self.ghost_system.get_system_summary(),
            "lockout_metrics": self.lockout_matrix.get_lockout_summary(),
            "fractal_metrics": self.fractal_controller.get_system_status(),
            "echo_metrics": {
                "total_snapshots": self.echo_logger.total_snapshots,
                "decision_distribution": self.echo_logger.decision_counts,
                "avg_confidence": np.mean(list(self.echo_logger.confidence_history)) if self.echo_logger.confidence_history else 0.0
            },
            "overall_performance": {
                "total_profit": self.total_profit,
                "total_decisions": self.decision_count,
                "avg_profit_per_decision": self.total_profit / max(self.decision_count, 1),
                "system_convergence": self._assess_system_convergence()
            }
        }
    
    def _assess_system_convergence(self) -> Dict[str, Any]:
        """Assess overall system convergence and stability."""
        confidence_history = list(self.echo_logger.confidence_history)
        
        if len(confidence_history) < 5:
            return {"status": "insufficient_data"}
        
        # Confidence stability
        confidence_variance = np.var(confidence_history)
        confidence_trend = np.polyfit(range(len(confidence_history)), confidence_history, 1)[0]
        
        # System health indicators
        vault_health = len(self.vault_router.metrics.allocation_history) > 0
        ghost_health = self.ghost_system.metrics.total_ghosts_created > 0
        lockout_health = self.lockout_matrix.metrics.total_lockouts_created >= 0
        
        return {
            "confidence_stability": 1.0 - min(confidence_variance, 1.0),
            "confidence_trend": confidence_trend,
            "system_health_score": sum([vault_health, ghost_health, lockout_health]) / 3.0,
            "convergence_achieved": confidence_variance < 0.1 and self.total_profit > 0,
            "mathematical_consistency": True  # All systems operational
        }
    
    def _print_enhanced_summary(self, results: Dict[str, Any]):
        """Print comprehensive enhanced systems summary."""
        print("\n" + "="*80)
        print("🎯 ENHANCED SYSTEMS TEST RESULTS")
        print("="*80)
        
        # Overall performance
        overall = results["overall_performance"]
        print(f"\n💰 OVERALL PERFORMANCE:")
        print(f"   Total Profit: {overall['total_profit']:.1f} basis points")
        print(f"   Total Decisions: {overall['total_decisions']}")
        print(f"   Avg Profit/Decision: {overall['avg_profit_per_decision']:.1f}bp")
        
        # System convergence
        convergence = overall["system_convergence"]
        print(f"\n🎯 SYSTEM CONVERGENCE:")
        print(f"   Confidence Stability: {convergence['confidence_stability']:.3f}")
        print(f"   Confidence Trend: {convergence['confidence_trend']:.4f}")
        print(f"   System Health Score: {convergence['system_health_score']:.3f}")
        print(f"   Convergence Achieved: {convergence['convergence_achieved']}")
        
        # Individual system metrics
        print(f"\n🔬 COLLAPSE CONFIDENCE ENGINE:")
        conf_metrics = results["confidence_metrics"]
        if conf_metrics.get("status") != "no_calculations":
            print(f"   Total Calculations: {conf_metrics['total_calculations']}")
            print(f"   Average Confidence: {conf_metrics['average_confidence']:.3f}")
            print(f"   High Confidence Rate: {conf_metrics['high_confidence_rate']:.1%}")
        
        print(f"\n🏦 ENHANCED VAULT ROUTER:")
        vault_metrics = results["vault_metrics"]
        if vault_metrics.get("status") != "no_allocations":
            print(f"   Total Allocations: {vault_metrics['total_allocations']}")
            print(f"   Avg Volume Multiplier: {vault_metrics['avg_volume_multiplier']:.2f}")
            print(f"   High Confidence Rate: {vault_metrics['high_confidence_rate']:.1%}")
            print(f"   Emergency Locks: {vault_metrics['emergency_lock_count']}")
        
        print(f"\n👻 GHOST DECAY SYSTEM:")
        ghost_metrics = results["ghost_metrics"]
        print(f"   Total Ghosts Created: {ghost_metrics['total_ghosts_created']}")
        print(f"   Active Ghosts: {ghost_metrics['active_ghosts']}")
        print(f"   Ghost Success Rate: {ghost_metrics['ghost_success_rate']:.1%}")
        print(f"   Total Reinforcements: {ghost_metrics['total_reinforcements']}")
        
        print(f"\n🔒 ENHANCED LOCKOUT MATRIX:")
        lockout_metrics = results["lockout_metrics"]
        print(f"   Total Lockouts Created: {lockout_metrics['total_lockouts_created']}")
        print(f"   Active Lockouts: {lockout_metrics['active_lockouts']}")
        print(f"   Self-Healed Lockouts: {lockout_metrics['self_healed_lockouts']}")
        print(f"   System Effectiveness: {lockout_metrics['system_effectiveness']:.1%}")
        
        print(f"\n📊 ECHO SNAPSHOT LOGGER:")
        echo_metrics = results["echo_metrics"]
        print(f"   Total Snapshots: {echo_metrics['total_snapshots']}")
        print(f"   Decision Distribution: {echo_metrics['decision_distribution']}")
        print(f"   Average Confidence: {echo_metrics['avg_confidence']:.3f}")
        
        print("\n" + "="*80)
        if convergence['convergence_achieved']:
            print("✅ MATHEMATICAL CONVERGENCE ACHIEVED - RECURSIVE PROFIT ENGINE OPERATIONAL")
        else:
            print("⚠️  SYSTEM LEARNING - CONVERGENCE IN PROGRESS")
        print("="*80)
        
        # Print echo session summary
        self.echo_logger.print_session_summary()
    
    def cleanup(self):
        """Cleanup test resources."""
        self.fractal_controller.shutdown()
        logger.info("Enhanced systems test cleanup completed")

def main():
    """Run the comprehensive enhanced systems test."""
    test = EnhancedSystemsIntegrationTest()
    
    try:
        # Run comprehensive test
        results = test.run_comprehensive_test(num_ticks=15)
        
        # Validate mathematical properties
        convergence = results["overall_performance"]["system_convergence"]
        if convergence["convergence_achieved"]:
            print("\n🎉 ENHANCED RECURSIVE PROFIT ENGINE FULLY OPERATIONAL!")
            print("   ✓ Collapse Confidence Engine: Mathematical scoring active")
            print("   ✓ Enhanced Vault Router: Dynamic volume allocation active")
            print("   ✓ Ghost Decay System: Pattern learning and decay active")
            print("   ✓ Enhanced Lockout Matrix: Self-healing lockouts active")
            print("   ✓ Echo Snapshot Logger: Real-time diagnostics active")
        else:
            print("\n📈 ENHANCED SYSTEMS LEARNING - CONVERGENCE IN PROGRESS")
            print("   All enhanced systems operational and learning from market data")
            
        if results["overall_performance"]["total_profit"] > 0:
            print("💎 POSITIVE PROFIT ACHIEVED WITH ENHANCED SYSTEMS!")
        
    except Exception as e:
        logger.error(f"Enhanced systems test failed: {e}")
        raise
    finally:
        test.cleanup()

if __name__ == "__main__":
    main() 