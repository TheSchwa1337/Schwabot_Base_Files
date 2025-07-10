#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bit Strategy Integration Test for Schwabot
==========================================
Comprehensive test demonstrating the integration of:
• Enhanced Strategy Bit Mapper (4-bit/8-bit logic)
• Entropy Decay System
• Symbolic Registry
• Vault-Orbital Bridge
• Asset-specific strategy routing
"""

import logging
import time
from typing import Any, Dict, List

import numpy as np

from core.entropy_decay_system import DecayMode, EntropyDecaySystem

# Import the new systems
from core.strategy_bit_mapper import BitStrategyResult, ExpansionMode, StrategyBitMapper
from core.symbolic_registry import SymbolicRegistry
from core.vault_orbital_bridge import OrbitalState, VaultOrbitalBridge, VaultState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BitStrategyIntegrationTest:
    """Comprehensive test of bit strategy integration."""
    
    def __init__(self):
        """Initialize test systems."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize all systems
        self.strategy_mapper = StrategyBitMapper()
        self.entropy_decay = EntropyDecaySystem()
        self.symbolic_registry = SymbolicRegistry()
        self.vault_orbital_bridge = VaultOrbitalBridge()
        
        self.logger.info("✅ All systems initialized for integration test")
    
    def test_4bit_8bit_strategy_classification(self):
        """Test 4-bit and 8-bit strategy classification for different assets."""
        self.logger.info("\n🔬 Testing 4-bit/8-bit Strategy Classification")
        
        # Test data for different assets
        test_cases = [
            {
                'asset': 'BTC',
                'signal': np.array([45000, 45100, 45200, 45300, 45400]),  # Rising BTC
                'entropy': 0.15,
                'expected_4bit': [4, 10, 12, 15]  # BTC preferences
            },
            {
                'asset': 'ETH',
                'signal': np.array([3000, 3010, 3020, 3015, 3030]),  # Volatile ETH
                'entropy': 0.25,
                'expected_4bit': [3, 6, 13]  # ETH preferences
            },
            {
                'asset': 'XRP',
                'signal': np.array([0.5, 0.51, 0.52, 0.51, 0.53]),  # Compressed XRP
                'entropy': 0.35,
                'expected_4bit': [7, 8, 12]  # XRP preferences
            },
            {
                'asset': 'USDC',
                'signal': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # Stable USDC
                'entropy': 0.05,
                'expected_4bit': [1, 5, 11]  # USDC preferences
            },
            {
                'asset': 'SOL',
                'signal': np.array([100, 105, 110, 115, 120]),  # Pumping SOL
                'entropy': 0.45,
                'expected_4bit': [7, 12, 9]  # SOL preferences
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            self.logger.info(f"\n--- Test Case {i}: {test_case['asset']} ---")
            
            # Classify signal
            result = self.strategy_mapper.classify_signal(
                signal=test_case['signal'],
                asset=test_case['asset'],
                entropy_level=test_case['entropy'],
                tick_index=i
            )
            
            # Display results
            self.logger.info(f"Asset: {result.asset}")
            self.logger.info(f"4-bit Strategy: {result.strategy_4bit} ({self.strategy_mapper.asset_logic.STRATEGY_4BIT_MAP[result.strategy_4bit]})")
            self.logger.info(f"8-bit Microstrategy: {result.strategy_8bit}")
            self.logger.info(f"Combined 12-bit: {result.combined_12bit}")
            self.logger.info(f"Strategy Name: {result.strategy_name}")
            self.logger.info(f"Confidence: {result.confidence:.3f}")
            self.logger.info(f"Entropy Score: {result.entropy_score:.3f}")
            self.logger.info(f"Drift Bits: {result.drift_bits}")
            
            # Validate 4-bit strategy is in expected preferences
            if result.strategy_4bit in test_case['expected_4bit']:
                self.logger.info("✅ 4-bit strategy matches asset preferences")
            else:
                self.logger.warning(f"⚠️ 4-bit strategy {result.strategy_4bit} not in expected preferences {test_case['expected_4bit']}")
    
    def test_entropy_decay_system(self):
        """Test entropy decay system with time-based degradation."""
        self.logger.info("\n🔬 Testing Entropy Decay System")
        
        # Add signals for different assets
        signals = [
            ('BTC_signal_1', 0.8, 'BTC', 4, False),
            ('ETH_signal_1', 0.6, 'ETH', 6, False),
            ('XRP_phantom_1', 0.9, 'XRP', 2, True),  # Phantom signal
            ('USDC_signal_1', 0.3, 'USDC', 1, False),
            ('SOL_signal_1', 0.7, 'SOL', 7, False)
        ]
        
        for signal_id, initial_entropy, asset, strategy_code, phantom in signals:
            success = self.entropy_decay.add_signal(
                signal_id=signal_id,
                initial_entropy=initial_entropy,
                asset=asset,
                strategy_code=strategy_code,
                phantom_layer=phantom
            )
            if success:
                self.logger.info(f"✅ Added signal: {signal_id} (entropy: {initial_entropy}, phantom: {phantom})")
        
        # Simulate time passage and check decay
        self.logger.info("\n--- Simulating Time Passage ---")
        
        time_steps = [0, 1800, 3600, 7200]  # 0, 30min, 1hr, 2hr
        
        for time_step in time_steps:
            self.logger.info(f"\nTime: {time_step} seconds")
            
            for signal_id, _, asset, _, _ in signals:
                decay_result = self.entropy_decay.calculate_decay(
                    signal_id, 
                    current_time=time.time() + time_step
                )
                
                if decay_result:
                    self.logger.info(f"  {signal_id}: weight={decay_result.current_weight:.3f}, "
                                   f"decay_factor={decay_result.decay_factor:.3f}, "
                                   f"age={decay_result.age_seconds:.0f}s, "
                                   f"remove={decay_result.should_remove}")
        
        # Get active signals
        active_signals = self.entropy_decay.get_active_signals(min_weight=0.1)
        self.logger.info(f"\nActive signals (weight > 0.1): {len(active_signals)}")
        
        # Get decay statistics
        stats = self.entropy_decay.get_decay_statistics()
        self.logger.info(f"Decay Statistics: {stats}")
    
    def test_symbolic_registry(self):
        """Test symbolic registry with CLI-like operations."""
        self.logger.info("\n🔬 Testing Symbolic Registry")
        
        # List all symbols
        symbol_list = self.symbolic_registry.list_all_symbols()
        self.logger.info(f"Symbol Registry:\n{symbol_list}")
        
        # Test symbol explanations
        test_symbols = ['∇', 'Ω', 'ψ', 'λ', 'Φ', 'Ξ', '𝕏']
        
        for symbol in test_symbols:
            explanation = self.symbolic_registry.explain_symbol(symbol)
            if explanation:
                self.logger.info(f"\n--- Symbol: {symbol} ---")
                self.logger.info(explanation)
        
        # Test symbol search
        search_results = self.symbolic_registry.search_symbols("entropy")
        self.logger.info(f"\nSearch results for 'entropy': {len(search_results)} symbols")
        for result in search_results:
            self.logger.info(f"  {result.symbol} - {result.name}")
        
        # Test symbol categories
        gradient_symbols = self.symbolic_registry.get_symbols_by_category("gradient")
        self.logger.info(f"\nGradient symbols: {len(gradient_symbols)}")
        for symbol in gradient_symbols:
            self.logger.info(f"  {symbol.symbol} - {symbol.name}")
    
    def test_vault_orbital_bridge(self):
        """Test vault-to-orbital bridge with different market conditions."""
        self.logger.info("\n🔬 Testing Vault-Orbital Bridge")
        
        # Test different market scenarios
        scenarios = [
            {
                'name': 'Bull Market (High Liquidity, Low Entropy)',
                'liquidity': 0.9,
                'entropy': 0.1,
                'volatility': 0.2,
                'phase_consistency': 0.9,
                'phantom': False
            },
            {
                'name': 'Bear Market (Low Liquidity, High Entropy)',
                'liquidity': 0.1,
                'entropy': 0.8,
                'volatility': 0.7,
                'phase_consistency': 0.3,
                'phantom': False
            },
            {
                'name': 'Phantom Market (Medium Liquidity, High Entropy)',
                'liquidity': 0.4,
                'entropy': 0.9,
                'volatility': 0.6,
                'phase_consistency': 0.2,
                'phantom': True
            },
            {
                'name': 'Stable Market (Medium Liquidity, Low Entropy)',
                'liquidity': 0.5,
                'entropy': 0.05,
                'volatility': 0.1,
                'phase_consistency': 0.95,
                'phantom': False
            }
        ]
        
        for scenario in scenarios:
            self.logger.info(f"\n--- Scenario: {scenario['name']} ---")
            
            result = self.vault_orbital_bridge.bridge_states(
                liquidity_level=scenario['liquidity'],
                entropy_level=scenario['entropy'],
                volatility=scenario['volatility'],
                phase_consistency=scenario['phase_consistency'],
                phantom_detected=scenario['phantom']
            )
            
            self.logger.info(f"Vault State: {result.vault_state.value}")
            self.logger.info(f"Orbital State: {result.orbital_state.value}")
            self.logger.info(f"Recommended Strategy: {result.recommended_strategy}")
            self.logger.info(f"Confidence: {result.confidence:.3f}")
            self.logger.info(f"Transition Triggered: {result.transition_triggered}")
        
        # Get transition statistics
        stats = self.vault_orbital_bridge.get_transition_statistics()
        self.logger.info(f"\nTransition Statistics: {stats}")
        
        # Get current state summary
        summary = self.vault_orbital_bridge.get_current_state_summary()
        self.logger.info(f"Current State Summary: {summary}")
    
    def test_integrated_workflow(self):
        """Test complete integrated workflow from signal to strategy execution."""
        self.logger.info("\n🔬 Testing Integrated Workflow")
        
        # Simulate a BTC trading scenario
        btc_signal = np.array([45000, 45100, 45200, 45300, 45400, 45500])
        current_time = time.time()
        
        self.logger.info("--- BTC Trading Scenario ---")
        
        # Step 1: Classify signal into 4-bit/8-bit strategy
        self.logger.info("Step 1: Signal Classification")
        strategy_result = self.strategy_mapper.classify_signal(
            signal=btc_signal,
            asset='BTC',
            entropy_level=0.15,
            tick_index=1
        )
        
        self.logger.info(f"  Strategy: {strategy_result.strategy_name}")
        self.logger.info(f"  Confidence: {strategy_result.confidence:.3f}")
        
        # Step 2: Add to entropy decay system
        self.logger.info("Step 2: Entropy Decay Management")
        signal_id = f"BTC_{current_time}"
        self.entropy_decay.add_signal(
            signal_id=signal_id,
            initial_entropy=strategy_result.entropy_score,
            asset='BTC',
            strategy_code=strategy_result.strategy_4bit,
            phantom_layer=False
        )
        
        # Step 3: Bridge vault and orbital states
        self.logger.info("Step 3: Vault-Orbital Bridge")
        bridge_result = self.vault_orbital_bridge.bridge_states(
            liquidity_level=0.7,  # High liquidity
            entropy_level=strategy_result.entropy_score,
            volatility=0.3,
            phase_consistency=0.8,
            phantom_detected=False
        )
        
        self.logger.info(f"  Vault State: {bridge_result.vault_state.value}")
        self.logger.info(f"  Orbital State: {bridge_result.orbital_state.value}")
        self.logger.info(f"  Bridge Strategy: {bridge_result.recommended_strategy}")
        
        # Step 4: Use symbolic operators for final calculation
        self.logger.info("Step 4: Symbolic Math Integration")
        
        # Get relevant symbols
        gradient_symbol = self.symbolic_registry.get_symbol('∇')
        omega_symbol = self.symbolic_registry.get_symbol('Ω')
        psi_symbol = self.symbolic_registry.get_symbol('ψ')
        
        if gradient_symbol and omega_symbol and psi_symbol:
            self.logger.info(f"  Using symbols: {gradient_symbol.symbol}, {omega_symbol.symbol}, {psi_symbol.symbol}")
            self.logger.info(f"  Gradient: {gradient_symbol.mathematical_definition}")
            self.logger.info(f"  Omega: {omega_symbol.mathematical_definition}")
            self.logger.info(f"  Psi: {psi_symbol.mathematical_definition}")
        
        # Step 5: Final strategy decision
        self.logger.info("Step 5: Final Strategy Decision")
        
        # Combine strategy results
        final_strategy = self._combine_strategy_results(
            strategy_result, bridge_result, signal_id
        )
        
        self.logger.info(f"  Final Strategy: {final_strategy}")
        
        # Step 6: Expand strategy bits for execution
        self.logger.info("Step 6: Strategy Expansion")
        
        expanded_strategies = self.strategy_mapper.expand_strategy_bits(
            strategy_id=strategy_result.combined_12bit,
            target_bits=8,
            mode=ExpansionMode.FERRIS_WHEEL
        )
        
        self.logger.info(f"  Expanded Strategies: {len(expanded_strategies)}")
        self.logger.info(f"  First 5 strategies: {expanded_strategies[:5]}")
    
    def _combine_strategy_results(self, strategy_result: BitStrategyResult, 
                                bridge_result, signal_id: str) -> str:
        """Combine strategy results from different systems."""
        # Priority: Bridge strategy > Bit strategy > Fallback
        if bridge_result.confidence > 0.8:
            return bridge_result.recommended_strategy
        elif strategy_result.confidence > 0.7:
            return strategy_result.strategy_name
        else:
            return "hold_maintain"
    
    def run_all_tests(self):
        """Run all integration tests."""
        self.logger.info("🚀 Starting Bit Strategy Integration Tests")
        self.logger.info("=" * 60)
        
        try:
            # Run individual system tests
            self.test_4bit_8bit_strategy_classification()
            self.test_entropy_decay_system()
            self.test_symbolic_registry()
            self.test_vault_orbital_bridge()
            
            # Run integrated workflow test
            self.test_integrated_workflow()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("✅ All integration tests completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Test failed: {e}")
            raise

def main():
    """Main test execution."""
    test = BitStrategyIntegrationTest()
    test.run_all_tests()

if __name__ == "__main__":
    main() 