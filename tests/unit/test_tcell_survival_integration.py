#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Cell Survival Integration Test for Schwabot
=============================================
Demonstrates biological immune system logic for front-running strategies:
• T-cell strategies as immune receptors
• Signal patterns as antigens
• Profit/Loss driving clonal expansion/apoptosis
• DNA-like strategy memory encoding
• Integration with critical math components
"""

import logging
import time
from typing import Any, Dict, List

import numpy as np

from core.bitmap_hash_folding import BitmapHashFolding, FoldingMode
from core.entropy_drift_engine import DriftMode, EntropyDriftEngine
from core.orbital_energy_quantizer import OrbitalEnergyQuantizer, OrbitalState
from core.strategy_bit_mapper import StrategyBitMapper
from core.symbolic_registry import SymbolicRegistry

# Import the T-cell survival engine and critical math systems
from core.tcell_survival_engine import TCellState, TCellStrategy, TCellSurvivalEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TCellSurvivalIntegrationTest:
    """Test of T-cell survival engine with critical math integration."""
    
    def __init__(self):
        """Initialize test systems."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize T-cell survival engine
        self.tcell_engine = TCellSurvivalEngine()
        
        # Initialize critical math systems
        self.entropy_drift = EntropyDriftEngine()
        self.orbital_quantizer = OrbitalEnergyQuantizer()
        self.bitmap_folding = BitmapHashFolding()
        self.strategy_mapper = StrategyBitMapper()
        self.symbolic_registry = SymbolicRegistry()
        
        self.logger.info("✅ T-Cell Survival Integration Test initialized")
    
    def test_tcell_strategy_creation(self):
        """Test creation of T-cell strategies (naive T-cells)."""
        self.logger.info("\n🧬 Testing T-Cell Strategy Creation")
        self.logger.info("Creating naive T-cells for front-running strategies")
        
        # Create various T-cell strategies
        strategies = [
            ("BTC", "ETH", 2, 0.4, 0.3, 0.3),   # BTC → ETH, 2 tick delay
            ("BTC", "XRP", -1, 0.4, 0.3, 0.3),  # BTC → XRP, -1 tick (XRP leads)
            ("ETH", "SOL", 1, 0.4, 0.3, 0.3),   # ETH → SOL, 1 tick delay
            ("XRP", "USDC", 3, 0.4, 0.3, 0.3),  # XRP → USDC, 3 tick delay
            ("SOL", "BTC", -1, 0.4, 0.3, 0.3),  # SOL → BTC, -1 tick
        ]
        
        created_hashes = []
        for source, target, delta_t, omega_w, xi_w, phi_w in strategies:
            strategy_hash = self.tcell_engine.create_tcell_strategy(
                source_asset=source,
                target_asset=target,
                delta_t=delta_t,
                omega_weight=omega_w,
                xi_weight=xi_w,
                phi_weight=phi_w
            )
            
            if strategy_hash:
                created_hashes.append(strategy_hash)
                tcell = self.tcell_engine.tcell_population[strategy_hash]
                
                self.logger.info(f"Created T-cell: {strategy_hash[:16]}...")
                self.logger.info(f"  Source: {tcell.source_asset} → Target: {tcell.target_asset}")
                self.logger.info(f"  Δt: {tcell.delta_t}, State: {tcell.state.value}")
                self.logger.info(f"  DNA: {tcell.dna_sequence[:20]}...")
                self.logger.info(f"  Survival Score: {tcell.survival_score:.3f}")
        
        self.logger.info(f"\nCreated {len(created_hashes)} T-cell strategies")
        return created_hashes
    
    def test_tcell_activation(self):
        """Test T-cell activation (TCR binding to antigen)."""
        self.logger.info("\n🧬 Testing T-Cell Activation")
        self.logger.info("𝓣_f = sigmoid[(Ω_A × Ξ_A × Φ_B) / (𝓥_B + ε)]")
        
        # Generate test signals
        btc_signal = np.array([45000, 45100, 45200, 45300, 45400, 45500, 45600, 45700])
        omega_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        phi_values = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        xi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        # Calculate entropy drift for BTC
        drift_result = self.entropy_drift.calculate_entropy_drift(
            signal=btc_signal,
            asset='BTC',
            omega_values=omega_values,
            phi_values=phi_values,
            xi_values=xi_values
        )
        
        self.logger.info(f"BTC Entropy Drift: {drift_result.drift_value:.6f}")
        self.logger.info(f"Orbital Energy: {drift_result.orbital_energy:.6f}")
        
        # Test T-cell activation for BTC-based strategies
        vault_pressures = {
            'ETH': 0.2,
            'XRP': 0.1,
            'SOL': 0.3,
            'USDC': 0.05
        }
        
        # Get front-run candidates
        candidates = self.tcell_engine.get_front_run_candidates(
            source_asset='BTC',
            omega_a=drift_result.orbital_energy,
            xi_a=drift_result.drift_value,
            vault_pressures=vault_pressures
        )
        
        self.logger.info(f"\nFound {len(candidates)} activated T-cell candidates:")
        
        for strategy_hash, activation_prob in candidates[:3]:  # Show top 3
            tcell = self.tcell_engine.tcell_population[strategy_hash]
            
            self.logger.info(f"  {strategy_hash[:16]}...")
            self.logger.info(f"    {tcell.source_asset} → {tcell.target_asset} (Δt={tcell.delta_t})")
            self.logger.info(f"    Activation Probability: {activation_prob:.3f}")
            self.logger.info(f"    Survival Score: {tcell.survival_score:.3f}")
            self.logger.info(f"    State: {tcell.state.value}")
    
    def test_clonal_expansion_and_apoptosis(self):
        """Test clonal expansion and apoptosis based on strategy performance."""
        self.logger.info("\n🧬 Testing Clonal Expansion and Apoptosis")
        self.logger.info("𝓒 = 𝓣_f × ROI_gain × recurrence_score")
        
        # Get some T-cell strategies to test
        tcell_hashes = list(self.tcell_engine.tcell_population.keys())[:3]
        
        # Simulate different performance outcomes
        test_scenarios = [
            (tcell_hashes[0], 0.15, True),   # High ROI, success → clonal expansion
            (tcell_hashes[1], -0.05, False), # Low ROI, failure → potential apoptosis
            (tcell_hashes[2], 0.02, True),   # Moderate ROI, success → maintain
        ]
        
        for strategy_hash, roi, success in test_scenarios:
            if strategy_hash in self.tcell_engine.tcell_population:
                tcell_before = self.tcell_engine.tcell_population[strategy_hash]
                
                self.logger.info(f"\nTesting strategy: {strategy_hash[:16]}...")
                self.logger.info(f"  Before: Survival={tcell_before.survival_score:.3f}, "
                               f"State={tcell_before.state.value}, Clones={tcell_before.clonal_count}")
                
                # Update survival based on performance
                self.tcell_engine.update_tcell_survival(strategy_hash, roi, success)
                
                tcell_after = self.tcell_engine.tcell_population[strategy_hash]
                
                self.logger.info(f"  ROI: {roi:.3f}, Success: {success}")
                self.logger.info(f"  After: Survival={tcell_after.survival_score:.3f}, "
                               f"State={tcell_after.state.value}, Clones={tcell_after.clonal_count}")
                
                # Check if mutation was triggered
                if tcell_after.mutation_count > tcell_before.mutation_count:
                    self.logger.info(f"  → Mutation triggered! New mutation count: {tcell_after.mutation_count}")
    
    def test_tcell_mutation(self):
        """Test T-cell mutation (somatic hypermutation)."""
        self.logger.info("\n🧬 Testing T-Cell Mutation")
        self.logger.info("Somatic hypermutation for failed strategies")
        
        # Find a T-cell with low survival score
        low_survival_tcells = [
            (hash, tcell) for hash, tcell in self.tcell_engine.tcell_population.items()
            if tcell.survival_score < 0.3
        ]
        
        if low_survival_tcells:
            strategy_hash, original_tcell = low_survival_tcells[0]
            
            self.logger.info(f"Original T-cell: {strategy_hash[:16]}...")
            self.logger.info(f"  Source: {original_tcell.source_asset} → Target: {original_tcell.target_asset}")
            self.logger.info(f"  Δt: {original_tcell.delta_t}, Survival: {original_tcell.survival_score:.3f}")
            self.logger.info(f"  DNA: {original_tcell.dna_sequence[:20]}...")
            
            # Trigger mutation
            new_strategy_hash = self.tcell_engine.mutate_tcell_strategy(strategy_hash)
            
            if new_strategy_hash:
                mutated_tcell = self.tcell_engine.tcell_population[new_strategy_hash]
                
                self.logger.info(f"Mutated T-cell: {new_strategy_hash[:16]}...")
                self.logger.info(f"  Source: {mutated_tcell.source_asset} → Target: {mutated_tcell.target_asset}")
                self.logger.info(f"  Δt: {mutated_tcell.delta_t} (changed by {mutated_tcell.delta_t - original_tcell.delta_t})")
                self.logger.info(f"  Survival: {mutated_tcell.survival_score:.3f}")
                self.logger.info(f"  DNA: {mutated_tcell.dna_sequence[:20]}...")
                self.logger.info(f"  Mutation Count: {mutated_tcell.mutation_count}")
                
                # Check DNA differences
                dna_differences = sum(1 for a, b in zip(original_tcell.dna_sequence, mutated_tcell.dna_sequence) if a != b)
                self.logger.info(f"  DNA Differences: {dna_differences} bases")
        else:
            self.logger.info("No low-survival T-cells found for mutation testing")
    
    def test_integrated_front_running_workflow(self):
        """Test complete front-running workflow with T-cell survival."""
        self.logger.info("\n🧬 Testing Integrated Front-Running Workflow")
        self.logger.info("Complete biological front-running strategy execution")
        
        # 1. Market signal detection (antigen presentation)
        self.logger.info("Step 1: Market Signal Detection (Antigen Presentation)")
        
        # Generate BTC spike signal
        btc_signal = np.array([45000, 45100, 45200, 45300, 45400, 45500, 45600, 45700])
        omega_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        phi_values = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        xi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        # Calculate entropy drift
        drift_result = self.entropy_drift.calculate_entropy_drift(
            signal=btc_signal,
            asset='BTC',
            omega_values=omega_values,
            phi_values=phi_values,
            xi_values=xi_values
        )
        
        self.logger.info(f"  BTC Signal Detected: Drift={drift_result.drift_value:.6f}, "
                        f"Energy={drift_result.orbital_energy:.6f}")
        
        # 2. T-cell activation (TCR binding)
        self.logger.info("Step 2: T-Cell Activation (TCR Binding)")
        
        vault_pressures = {'ETH': 0.2, 'XRP': 0.1, 'SOL': 0.3, 'USDC': 0.05}
        
        candidates = self.tcell_engine.get_front_run_candidates(
            source_asset='BTC',
            omega_a=drift_result.orbital_energy,
            xi_a=drift_result.drift_value,
            vault_pressures=vault_pressures
        )
        
        self.logger.info(f"  Activated T-cells: {len(candidates)}")
        
        # 3. Front-running execution
        self.logger.info("Step 3: Front-Running Execution")
        
        executed_trades = []
        for strategy_hash, activation_prob in candidates[:2]:  # Execute top 2
            tcell = self.tcell_engine.tcell_population[strategy_hash]
            
            self.logger.info(f"  Executing: {tcell.source_asset} → {tcell.target_asset}")
            self.logger.info(f"    Δt: {tcell.delta_t} ticks")
            self.logger.info(f"    Activation Probability: {activation_prob:.3f}")
            self.logger.info(f"    Strategy Hash: {strategy_hash[:16]}...")
            
            # Simulate trade execution
            simulated_roi = np.random.normal(0.05, 0.02)  # Random ROI
            success = simulated_roi > 0
            
            executed_trades.append((strategy_hash, simulated_roi, success))
            
            self.logger.info(f"    Simulated ROI: {simulated_roi:.3f}")
            self.logger.info(f"    Success: {success}")
        
        # 4. Survival update (clonal expansion/apoptosis)
        self.logger.info("Step 4: Survival Update (Clonal Expansion/Apoptosis)")
        
        for strategy_hash, roi, success in executed_trades:
            tcell_before = self.tcell_engine.tcell_population[strategy_hash]
            
            self.logger.info(f"  Updating: {strategy_hash[:16]}...")
            self.logger.info(f"    Before: Survival={tcell_before.survival_score:.3f}, "
                           f"State={tcell_before.state.value}")
            
            # Update survival
            self.tcell_engine.update_tcell_survival(strategy_hash, roi, success)
            
            tcell_after = self.tcell_engine.tcell_population[strategy_hash]
            
            self.logger.info(f"    After: Survival={tcell_after.survival_score:.3f}, "
                           f"State={tcell_after.state.value}, Clones={tcell_after.clonal_count}")
            
            # Check for state changes
            if tcell_after.state != tcell_before.state:
                self.logger.info(f"    → State change: {tcell_before.state.value} → {tcell_after.state.value}")
        
        # 5. Population cleanup
        self.logger.info("Step 5: Population Cleanup (Apoptosis)")
        
        initial_count = len(self.tcell_engine.tcell_population)
        self.tcell_engine.cleanup_exhausted_tcells()
        final_count = len(self.tcell_engine.tcell_population)
        
        self.logger.info(f"  Population: {initial_count} → {final_count} T-cells")
        self.logger.info(f"  Removed: {initial_count - final_count} exhausted T-cells")
    
    def test_biological_statistics(self):
        """Test biological statistics and population health."""
        self.logger.info("\n🧬 Testing Biological Statistics")
        
        # Get survival statistics
        stats = self.tcell_engine.get_survival_statistics()
        
        self.logger.info("T-Cell Population Statistics:")
        self.logger.info(f"  Total T-cells: {stats['total_tcells']}")
        self.logger.info(f"  Average Survival Score: {stats['average_survival_score']:.3f}")
        
        self.logger.info("State Distribution:")
        for state, count in stats['state_distribution'].items():
            self.logger.info(f"  {state}: {count}")
        
        self.logger.info("Survival Statistics:")
        for metric, value in stats['survival_stats'].items():
            self.logger.info(f"  {metric}: {value}")
        
        # Calculate population health metrics
        memory_cells = stats['state_distribution'].get('memory', 0)
        exhausted_cells = stats['state_distribution'].get('exhausted', 0)
        total_cells = stats['total_tcells']
        
        if total_cells > 0:
            memory_ratio = memory_cells / total_cells
            exhausted_ratio = exhausted_cells / total_cells
            health_score = memory_ratio - exhausted_ratio
            
            self.logger.info(f"Population Health Metrics:")
            self.logger.info(f"  Memory Cell Ratio: {memory_ratio:.3f}")
            self.logger.info(f"  Exhausted Cell Ratio: {exhausted_ratio:.3f}")
            self.logger.info(f"  Health Score: {health_score:.3f}")
            
            if health_score > 0.3:
                self.logger.info("  → Population: HEALTHY")
            elif health_score > 0.0:
                self.logger.info("  → Population: STABLE")
            else:
                self.logger.info("  → Population: NEEDS ATTENTION")
    
    def test_dna_evolution_tracking(self):
        """Test DNA sequence evolution tracking."""
        self.logger.info("\n🧬 Testing DNA Evolution Tracking")
        
        # Analyze DNA sequences across T-cell population
        dna_sequences = []
        for tcell in self.tcell_engine.tcell_population.values():
            dna_sequences.append(tcell.dna_sequence)
        
        if dna_sequences:
            # Calculate DNA diversity
            unique_sequences = set(dna_sequences)
            diversity_ratio = len(unique_sequences) / len(dna_sequences)
            
            self.logger.info(f"DNA Diversity Analysis:")
            self.logger.info(f"  Total Sequences: {len(dna_sequences)}")
            self.logger.info(f"  Unique Sequences: {len(unique_sequences)}")
            self.logger.info(f"  Diversity Ratio: {diversity_ratio:.3f}")
            
            # Analyze mutation patterns
            mutation_counts = [tcell.mutation_count for tcell in self.tcell_engine.tcell_population.values()]
            avg_mutations = np.mean(mutation_counts) if mutation_counts else 0
            
            self.logger.info(f"Mutation Analysis:")
            self.logger.info(f"  Average Mutations per T-cell: {avg_mutations:.2f}")
            self.logger.info(f"  Max Mutations: {max(mutation_counts) if mutation_counts else 0}")
            
            # Show some example DNA sequences
            self.logger.info("Example DNA Sequences:")
            for i, tcell in enumerate(list(self.tcell_engine.tcell_population.values())[:3]):
                self.logger.info(f"  T-cell {i+1}: {tcell.dna_sequence[:32]}... "
                               f"(mutations: {tcell.mutation_count})")
    
    def run_all_tests(self):
        """Run all T-cell survival integration tests."""
        self.logger.info("🧬 Starting T-Cell Survival Integration Tests")
        self.logger.info("=" * 80)
        
        try:
            # Run individual system tests
            self.test_tcell_strategy_creation()
            self.test_tcell_activation()
            self.test_clonal_expansion_and_apoptosis()
            self.test_tcell_mutation()
            
            # Run integrated workflow test
            self.test_integrated_front_running_workflow()
            
            # Run biological analysis tests
            self.test_biological_statistics()
            self.test_dna_evolution_tracking()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✅ All T-cell survival tests completed successfully!")
            self.logger.info("🧬 Biological Front-Running Strategy Implemented:")
            self.logger.info("  • T-cell strategies as immune receptors")
            self.logger.info("  • Signal patterns as antigens")
            self.logger.info("  • Profit/Loss driving clonal expansion/apoptosis")
            self.logger.info("  • DNA-like strategy memory encoding")
            self.logger.info("  • Adaptive mutation and survival selection")
            self.logger.info("  • Integration with critical math components")
            
        except Exception as e:
            self.logger.error(f"❌ Test failed: {e}")
            raise

def main():
    """Main test execution."""
    test = TCellSurvivalIntegrationTest()
    test.run_all_tests()

if __name__ == "__main__":
    main() 