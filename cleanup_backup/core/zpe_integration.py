from utils.safe_print import safe_print, info, warn, error, success, debug
#!/usr/bin/env python3
"""
ZPE Integration Layer
====================

Connects the ZPE core to existing Schwabot systems like strategy_mapper, profit_cycle_allocator, and fractal_core.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .zpe_core import ZPECore

logger = logging.getLogger(__name__)


class ZPEIntegration:
    """
    Integration layer that connects ZPE mathematical framework to Schwabot's existing systems.

    This is where Schwabot becomes the wheel - spinning into profit, not pinging against it.
    """

    def __init__(self):
        """Initialize ZPE Integration."""
        self.zpe_core = ZPECore()
        self.integration_status = {
            'strategy_mapper': False,
            'profit_cycle_allocator': False,
            'fractal_core': False,
            'lantern_memory': False,
            'fault_bus': False,
            'hash_registry': False
        }

        logger.info("ZPE Integration Layer initialized")

    def integrate_with_strategy_mapper(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate ZPE core with strategy_mapper.py

        Applies ZPE mathematical framework to strategy vector calculations.
        """
        try:
            # Extract strategy vectors for multi-asset alignment
            strategy_vectors = strategy_data.get('vectors', {})
            weights = strategy_data.get('weights', {})

            # Apply ZPE multi-vector alignment
            aligned_vector = self.zpe_core.calculate_multi_vector_alignment(strategy_vectors, weights)

            # Update strategy data with ZPE calculations
            strategy_data['zpe_alignment'] = aligned_vector
            strategy_data['zpe_work'] = self.zpe_core.calculate_zpe_work(
                strategy_data.get('trend_strength', 0.0),
                strategy_data.get('entry_exit_range', 0.0)
            )

            self.integration_status['strategy_mapper'] = True
            logger.info("✅ ZPE integrated with strategy_mapper")

            return strategy_data

        except Exception as e:
            logger.error(f"❌ ZPE strategy_mapper integration failed: {e}")
            return strategy_data

    def integrate_with_profit_cycle_allocator(self, profit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate ZPE core with profit_cycle_allocator.py

        Applies ZPE profit reinjection and thermal efficiency calculations.
        """
        try:
            # Calculate thermal efficiency
            profit_generated = profit_data.get('profit_generated', 0.0)
            capital_exposure = profit_data.get('capital_exposure', 1.0)
            efficiency = self.zpe_core.calculate_thermal_efficiency(profit_generated, capital_exposure)

            # Calculate profit reinjection
            profit_delta = profit_data.get('profit_delta', 0.0)
            market_heat = profit_data.get('market_heat', 0.5)
            reinjected_profit = self.zpe_core.calculate_profit_reinjection(profit_delta, market_heat)

            # Update profit data with ZPE calculations
            profit_data['zpe_efficiency'] = efficiency
            profit_data['zpe_reinjection'] = reinjected_profit
            profit_data['total_profit'] = profit_generated + reinjected_profit

            self.integration_status['profit_cycle_allocator'] = True
            logger.info("✅ ZPE integrated with profit_cycle_allocator")

            return profit_data

        except Exception as e:
            logger.error(f"❌ ZPE profit_cycle_allocator integration failed: {e}")
            return profit_data

    def integrate_with_fractal_core(self, fractal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate ZPE core with fractal_core.py

        Applies ZPE recursive cycle depth and rotational torque calculations.
        """
        try:
            # Update recursive cycle depth
            tick_interval = fractal_data.get('tick_interval', 1.0)
            price_trigger = fractal_data.get('price_trigger', 0.0)
            recursion_depth = self.zpe_core.update_recursive_cycle_depth(tick_interval, price_trigger)

            # Calculate rotational torque
            liquidity_depth = fractal_data.get('liquidity_depth', 1.0)
            trend_change_rate = fractal_data.get('trend_change_rate', 0.0)
            torque = self.zpe_core.calculate_rotational_torque(liquidity_depth, trend_change_rate)

            # Update fractal data with ZPE calculations
            fractal_data['zpe_recursion_depth'] = recursion_depth
            fractal_data['zpe_torque'] = torque
            fractal_data['zpe_angular_velocity'] = torque / fractal_data.get('inertia', 1.0)

            self.integration_status['fractal_core'] = True
            logger.info("✅ ZPE integrated with fractal_core")

            return fractal_data

        except Exception as e:
            logger.error(f"❌ ZPE fractal_core integration failed: {e}")
            return fractal_data

    def integrate_with_lantern_memory(self, lantern_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate ZPE core with lantern_vector_memory.py

        Applies ZPE news/lantern signal mapping and elastic resonance calculations.
        """
        try:
            # Map news/lantern signals
            news_density = lantern_data.get('news_density', 0.0)
            sentiment_delta = lantern_data.get('sentiment_delta', 0.0)
            lantern_signal = self.zpe_core.map_news_lantern_signals(news_density, sentiment_delta)

            # Calculate elastic resonance
            price_derivative = lantern_data.get('price_derivative', 0.0)
            frequency = lantern_data.get('frequency', 1.0)
            phase_offset = lantern_data.get('phase_offset', 0.0)
            time_window = lantern_data.get('time_window', 1.0)
            resonance = self.zpe_core.calculate_elastic_resonance(
                price_derivative, frequency, phase_offset, time_window)

            # Update lantern data with ZPE calculations
            lantern_data['zpe_lantern_signal'] = lantern_signal
            lantern_data['zpe_resonance'] = resonance
            lantern_data['zpe_signal_strength'] = (lantern_signal + resonance) / 2.0

            self.integration_status['lantern_memory'] = True
            logger.info("✅ ZPE integrated with lantern_memory")

            return lantern_data

        except Exception as e:
            logger.error(f"❌ ZPE lantern_memory integration failed: {e}")
            return lantern_data

    def integrate_with_fault_bus(self, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate ZPE core with fault_bus.py

        Applies ZPE temporal fault correction and agent consensus calculations.
        """
        try:
            # Calculate temporal fault correction
            expected_phase = fault_data.get('expected_phase', 0.0)
            actual_phase = fault_data.get('actual_phase', 0.0)
            fault_correction = self.zpe_core.calculate_temporal_fault_correction(expected_phase, actual_phase)

            # Update agent consensus
            agent_name = fault_data.get('agent_name', 'Claude')
            confidence = fault_data.get('confidence', 0.5)
            consensus = self.zpe_core.update_agent_consensus(agent_name, confidence)

            # Update fault data with ZPE calculations
            fault_data['zpe_fault_correction'] = fault_correction
            fault_data['zpe_consensus'] = consensus
            fault_data['zpe_agent_consensus'] = self.zpe_core.agent_consensus.copy()

            self.integration_status['fault_bus'] = True
            logger.info("✅ ZPE integrated with fault_bus")

            return fault_data

        except Exception as e:
            logger.error(f"❌ ZPE fault_bus integration failed: {e}")
            return fault_data

    def integrate_with_hash_registry(self, hash_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate ZPE core with hash_registry.py

        Applies ZPE mathematical framework to hash-based memory and strategy tracking.
        """
        try:
            # Apply ZPE calculations to hash data
            hash_data['zpe_timestamp'] = datetime.now()
            hash_data['zpe_recursion_depth'] = self.zpe_core.recursion_depth
            hash_data['zpe_thermal_efficiency'] = self.zpe_core.thermal_history[-1]['efficiency'] if self.zpe_core.thermal_history else 0.0

            self.integration_status['hash_registry'] = True
            logger.info("✅ ZPE integrated with hash_registry")

            return hash_data

        except Exception as e:
            logger.error(f"❌ ZPE hash_registry integration failed: {e}")
            return hash_data

    def spin_complete_system(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Spin the complete ZPE profit wheel across all integrated systems.

        This is where Schwabot becomes the wheel - spinning into profit, not pinging against it.
        """
        logger.info("🔄 Spinning Complete ZPE System...")

        # Spin the ZPE profit wheel
        zpe_result = self.zpe_core.spin_profit_wheel(market_data)

        # Integrate with all systems
        integrated_data = {
            'zpe_core': zpe_result,
            'strategy_mapper': self.integrate_with_strategy_mapper(market_data.get('strategy', {})),
            'profit_cycle_allocator': self.integrate_with_profit_cycle_allocator(market_data.get('profit', {})),
            'fractal_core': self.integration_status,
            'lantern_memory': self.integration_status,
            'fault_bus': self.integration_status,
            'hash_registry': self.integration_status
        }

        # Calculate overall system spin decision
        spin_score = zpe_result.get('spin_score', 0.0)
        should_spin = zpe_result.get('should_spin', False)

        integrated_data['system_spin_decision'] = {
            'spin_score': spin_score,
            'should_spin': should_spin,
            'integration_status': self.integration_status,
            'timestamp': datetime.now()
        }

        logger.info(f"🎯 Complete System Decision: {'SPIN' if should_spin else 'HOLD'} (score: {spin_score:.6f})")
        return integrated_data

    def get_integration_status(self) -> Dict[str, bool]:
        """Get the status of all ZPE integrations."""
        return self.integration_status.copy()

    def reset_integration_status(self):
        """Reset all integration status flags."""
        for key in self.integration_status:
            self.integration_status[key] = False
        logger.info("🔄 ZPE integration status reset")


def main():
    """Test the ZPE Integration Layer."""
    safe_print("🧠 Testing Schwabot ZPE Integration Layer")
    safe_print("=" * 50)

    integration = ZPEIntegration()

    # Test market data
    market_data = {
        'trend_strength': 0.8,
        'entry_exit_range': 0.05,
        'liquidity_depth': 0.7,
        'trend_change_rate': 0.3,
        'price_derivative': 0.02,
        'news_density': 0.6,
        'sentiment_delta': 0.2,
        'strategy': {
            'vectors': {
                'BTC': {'magnitude': 0.8, 'resonance': 0.7},
                'ETH': {'magnitude': 0.6, 'resonance': 0.5},
                'XRP': {'magnitude': 0.4, 'resonance': 0.3}
            },
            'weights': {'BTC': 0.5, 'ETH': 0.3, 'XRP': 0.2}
        },
        'profit': {
            'profit_generated': 100.0,
            'capital_exposure': 1000.0,
            'profit_delta': 50.0,
            'market_heat': 0.7
        }
    }

    # Spin the complete system
    result = integration.spin_complete_system(market_data)

    safe_print(f"ZPE Core Spin Score: {result['zpe_core']['spin_score']:.6f}")
    safe_print(f"System Should Spin: {result['system_spin_decision']['should_spin']}")
    safe_print(f"Integration Status: {result['system_spin_decision']['integration_status']}")

    safe_print("\n🎉 ZPE Integration Layer test complete!")


if __name__ == "__main__":
    main()
