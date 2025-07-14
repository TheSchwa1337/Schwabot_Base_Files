#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Cell Survival Engine for Schwabot
===================================
Implements biological immune system logic for front-running strategies:
• Strategy hashes = T-cell receptors (TCR)
• Signal patterns = Antigens
• Profit/Loss = Clonal expansion/Apoptosis
• DNA-like strategy memory encoding
• Adaptive mutation and survival selection
"""

import logging


import logging


import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class TCellState(Enum):
    """T-cell states in immune system."""
    NAIVE = "naive"           # New strategy, untested
    ACTIVATED = "activated"   # Strategy triggered
    MEMORY = "memory"         # Successful strategy retained
    EXHAUSTED = "exhausted"   # Failed strategy, marked for apoptosis
    MUTATED = "mutated"       # Strategy modified due to failure

@dataclass
class TCellStrategy:
    """T-cell strategy with biological properties."""
    strategy_hash: str
    source_asset: str
    target_asset: str
    delta_t: int
    omega_weight: float
    xi_weight: float
    phi_weight: float
    activation_threshold: float
    survival_score: float
    clonal_count: int
    mutation_count: int
    last_activation: float
    total_roi: float
    success_rate: float
    state: TCellState
    dna_sequence: str = ""  # DNA-like encoding of strategy

@dataclass
class TCellActivationResult:
    """Result of T-cell activation."""
    activated: bool
    activation_probability: float
    strategy_hash: str
    predicted_roi: float
    confidence: float
    mutation_triggered: bool

class TCellSurvivalEngine:
    """T-cell survival engine for biological strategy management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize T-cell survival engine."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # T-cell population (strategies)
        self.tcell_population: Dict[str, TCellStrategy] = {}
        
        # Antigen patterns (signal patterns)
        self.antigen_patterns: Dict[str, np.ndarray] = {}
        
        # MHC correlation matrix (cross-asset delays)
        self.mhc_correlation_matrix = self._build_mhc_matrix()
        
        # Survival statistics
        self.survival_stats = {
            'total_activations': 0,
            'successful_activations': 0,
            'mutations': 0,
            'apoptosis': 0,
            'clonal_expansions': 0
        }
        
        # Load existing T-cell population
        self._load_tcell_population()
        
        self.logger.info("✅ T-Cell Survival Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'activation_threshold': 0.76,
            'survival_threshold': 0.3,
            'clonal_expansion_threshold': 0.7,
            'mutation_rate': 0.1,
            'apoptosis_threshold': 0.1,
            'max_population_size': 1000,
            'memory_decay_rate': 0.95,
            'enable_mutations': True,
            'enable_clonal_expansion': True,
            'dna_sequence_length': 64,
            'tcell_registry_file': 'tcell_strategy_registry.json'
        }
    
    def _build_mhc_matrix(self) -> Dict[str, Dict[str, int]]:
        """Build MHC correlation matrix (cross-asset delays)."""
        return {
            "BTC": {
                "ETH": 2,    # ETH follows BTC by 2 ticks
                "XRP": -1,   # XRP leads BTC by 1 tick
                "SOL": 1,    # SOL follows BTC by 1 tick
                "USDC": 0    # USDC is stable reference
            },
            "ETH": {
                "BTC": -2,   # BTC leads ETH by 2 ticks
                "XRP": -3,   # XRP leads ETH by 3 ticks
                "SOL": -1,   # SOL leads ETH by 1 tick
                "USDC": -2   # USDC reference
            },
            "XRP": {
                "BTC": 1,    # XRP leads BTC by 1 tick
                "ETH": 3,    # XRP leads ETH by 3 ticks
                "SOL": 2,    # XRP leads SOL by 2 ticks
                "USDC": 1    # USDC reference
            },
            "SOL": {
                "BTC": -1,   # SOL follows BTC by 1 tick
                "ETH": 1,    # SOL follows ETH by 1 tick
                "XRP": -2,   # SOL follows XRP by 2 ticks
                "USDC": -1   # USDC reference
            }
        }
    
    def create_tcell_strategy(self, source_asset: str, target_asset: str,
                            delta_t: int, omega_weight: float = 0.4,
                            xi_weight: float = 0.3, phi_weight: float = 0.3) -> str:
        """
        Create a new T-cell strategy (naive T-cell).
        Returns the strategy hash (TCR).
        """
        try:
            # Generate strategy parameters
            strategy_params = {
                'source': source_asset,
                'target': target_asset,
                'delta_t': delta_t,
                'omega_weight': omega_weight,
                'xi_weight': xi_weight,
                'phi_weight': phi_weight,
                'timestamp': time.time()
            }
            
            # Generate strategy hash (TCR)
            strategy_hash = self._generate_strategy_hash(strategy_params)
            
            # Generate DNA sequence
            dna_sequence = self._generate_dna_sequence(strategy_params)
            
            # Create T-cell strategy
            tcell_strategy = TCellStrategy(
                strategy_hash=strategy_hash,
                source_asset=source_asset,
                target_asset=target_asset,
                delta_t=delta_t,
                omega_weight=omega_weight,
                xi_weight=xi_weight,
                phi_weight=phi_weight,
                activation_threshold=self.config['activation_threshold'],
                survival_score=0.5,  # Neutral starting score
                clonal_count=1,
                mutation_count=0,
                last_activation=0.0,
                total_roi=0.0,
                success_rate=0.5,
                state=TCellState.NAIVE,
                dna_sequence=dna_sequence
            )
            
            # Add to population
            self.tcell_population[strategy_hash] = tcell_strategy
            
            self.logger.info(f"Created T-cell strategy: {strategy_hash[:16]}... "
                           f"({source_asset} → {target_asset}, Δt={delta_t})")
            
            return strategy_hash
            
        except Exception as e:
            self.logger.error(f"Error creating T-cell strategy: {e}")
            return ""
    
    def activate_tcell(self, strategy_hash: str, omega_a: float, xi_a: float,
                      phi_b: float, vault_pressure: float = 0.1) -> TCellActivationResult:
        """
        Activate T-cell strategy (TCR binding to antigen).
        𝓣_f = sigmoid[(Ω_A × Ξ_A × Φ_B) / (𝓥_B + ε)]
        """
        try:
            if strategy_hash not in self.tcell_population:
                return TCellActivationResult(
                    activated=False, activation_probability=0.0,
                    strategy_hash=strategy_hash, predicted_roi=0.0,
                    confidence=0.0, mutation_triggered=False
                )
            
            tcell = self.tcell_population[strategy_hash]
            
            # Calculate activation probability (T-cell activation function)
            # 𝓣_f = sigmoid[(Ω_A × Ξ_A × Φ_B) / (𝓥_B + ε)]
            numerator = (omega_a * tcell.omega_weight + 
                        xi_a * tcell.xi_weight + 
                        phi_b * tcell.phi_weight)
            denominator = vault_pressure + 1e-6  # ε stabilizer
            
            activation_probability = self._sigmoid(numerator / denominator)
            
            # Determine if activated
            activated = activation_probability > tcell.activation_threshold
            
            # Update T-cell state
            if activated:
                tcell.state = TCellState.ACTIVATED
                tcell.last_activation = time.time()
                self.survival_stats['total_activations'] += 1
            
            # Calculate predicted ROI based on historical performance
            predicted_roi = tcell.total_roi * tcell.success_rate
            
            # Calculate confidence based on survival score
            confidence = tcell.survival_score
            
            # Check if mutation should be triggered
            mutation_triggered = self._should_mutate(tcell)
            
            return TCellActivationResult(
                activated=activated,
                activation_probability=activation_probability,
                strategy_hash=strategy_hash,
                predicted_roi=predicted_roi,
                confidence=confidence,
                mutation_triggered=mutation_triggered
            )
            
        except Exception as e:
            self.logger.error(f"Error activating T-cell: {e}")
            return TCellActivationResult(
                activated=False, activation_probability=0.0,
                strategy_hash=strategy_hash, predicted_roi=0.0,
                confidence=0.0, mutation_triggered=False
            )
    
    def update_tcell_survival(self, strategy_hash: str, roi: float, success: bool):
        """
        Update T-cell survival based on strategy outcome.
        Implements clonal expansion or apoptosis logic.
        """
        try:
            if strategy_hash not in self.tcell_population:
                return
            
            tcell = self.tcell_population[strategy_hash]
            
            # Update ROI and success rate
            tcell.total_roi = (tcell.total_roi + roi) / 2  # Rolling average
            tcell.success_rate = (tcell.success_rate * 0.9 + (1.0 if success else 0.0) * 0.1)
            
            # Calculate clonal expansion factor
            # 𝓒 = 𝓣_f × ROI_gain × recurrence_score
            clonal_factor = (tcell.survival_score * 
                           max(0, tcell.total_roi) * 
                           tcell.success_rate)
            
            # Determine survival outcome
            if clonal_factor > self.config['clonal_expansion_threshold']:
                # Clonal expansion - strengthen strategy
                tcell.survival_score = min(1.0, tcell.survival_score + 0.1)
                tcell.clonal_count += 1
                tcell.state = TCellState.MEMORY
                self.survival_stats['clonal_expansions'] += 1
                
                self.logger.info(f"T-cell clonal expansion: {strategy_hash[:16]}... "
                               f"(survival_score: {tcell.survival_score:.3f})")
                
            elif clonal_factor < self.config['apoptosis_threshold']:
                # Apoptosis - mark for removal
                tcell.survival_score = max(0.0, tcell.survival_score - 0.2)
                tcell.state = TCellState.EXHAUSTED
                self.survival_stats['apoptosis'] += 1
                
                self.logger.info(f"T-cell apoptosis: {strategy_hash[:16]}... "
                               f"(survival_score: {tcell.survival_score:.3f})")
                
            else:
                # Maintain current state
                tcell.survival_score = max(0.0, min(1.0, 
                    tcell.survival_score + (0.05 if success else -0.05)))
            
            # Update success count
            if success:
                self.survival_stats['successful_activations'] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating T-cell survival: {e}")
    
    def mutate_tcell_strategy(self, strategy_hash: str) -> Optional[str]:
        """
        Mutate T-cell strategy (somatic hypermutation).
        Creates new strategy with modified parameters.
        """
        try:
            if strategy_hash not in self.tcell_population:
                return None
            
            original_tcell = self.tcell_population[strategy_hash]
            
            # Apply mutations to parameters
            mutated_params = {
                'source': original_tcell.source_asset,
                'target': original_tcell.target_asset,
                'delta_t': original_tcell.delta_t + np.random.randint(-2, 3),  # ±2 ticks
                'omega_weight': max(0.1, min(0.9, original_tcell.omega_weight + np.random.normal(0, 0.1))),
                'xi_weight': max(0.1, min(0.9, original_tcell.xi_weight + np.random.normal(0, 0.1))),
                'phi_weight': max(0.1, min(0.9, original_tcell.phi_weight + np.random.normal(0, 0.1))),
                'timestamp': time.time()
            }
            
            # Generate new strategy hash
            new_strategy_hash = self._generate_strategy_hash(mutated_params)
            
            # Generate new DNA sequence
            new_dna_sequence = self._generate_dna_sequence(mutated_params)
            
            # Create mutated T-cell
            mutated_tcell = TCellStrategy(
                strategy_hash=new_strategy_hash,
                source_asset=mutated_params['source'],
                target_asset=mutated_params['target'],
                delta_t=mutated_params['delta_t'],
                omega_weight=mutated_params['omega_weight'],
                xi_weight=mutated_params['xi_weight'],
                phi_weight=mutated_params['phi_weight'],
                activation_threshold=original_tcell.activation_threshold,
                survival_score=original_tcell.survival_score * 0.8,  # Slightly reduced
                clonal_count=1,
                mutation_count=original_tcell.mutation_count + 1,
                last_activation=0.0,
                total_roi=0.0,
                success_rate=0.5,
                state=TCellState.MUTATED,
                dna_sequence=new_dna_sequence
            )
            
            # Add to population
            self.tcell_population[new_strategy_hash] = mutated_tcell
            
            # Mark original as exhausted
            original_tcell.state = TCellState.EXHAUSTED
            original_tcell.survival_score *= 0.5
            
            self.survival_stats['mutations'] += 1
            
            self.logger.info(f"T-cell mutation: {strategy_hash[:16]}... → {new_strategy_hash[:16]}...")
            
            return new_strategy_hash
            
        except Exception as e:
            self.logger.error(f"Error mutating T-cell strategy: {e}")
            return None
    
    def get_front_run_candidates(self, source_asset: str, omega_a: float, 
                               xi_a: float, vault_pressures: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Get front-run candidates based on T-cell population.
        Returns list of (strategy_hash, activation_probability) tuples.
        """
        try:
            candidates = []
            
            for strategy_hash, tcell in self.tcell_population.items():
                # Only consider strategies for this source asset
                if tcell.source_asset != source_asset:
                    continue
                
                # Skip exhausted T-cells
                if tcell.state == TCellState.EXHAUSTED:
                    continue
                
                # Get vault pressure for target asset
                vault_pressure = vault_pressures.get(tcell.target_asset, 0.1)
                
                # Estimate phi_b based on historical patterns
                phi_b = 0.5  # Default estimate, could be improved with historical data
                
                # Calculate activation probability
                activation_result = self.activate_tcell(
                    strategy_hash, omega_a, xi_a, phi_b, vault_pressure
                )
                
                if activation_result.activated:
                    candidates.append((strategy_hash, activation_result.activation_probability))
            
            # Sort by activation probability (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error getting front-run candidates: {e}")
            return []
    
    def cleanup_exhausted_tcells(self):
        """Remove exhausted T-cells from population (apoptosis)."""
        try:
            initial_count = len(self.tcell_population)
            
            # Find exhausted T-cells
            exhausted_hashes = []
            for strategy_hash, tcell in self.tcell_population.items():
                if (tcell.state == TCellState.EXHAUSTED and 
                    tcell.survival_score < self.config['apoptosis_threshold']):
                    exhausted_hashes.append(strategy_hash)
            
            # Remove exhausted T-cells
            for strategy_hash in exhausted_hashes:
                del self.tcell_population[strategy_hash]
            
            removed_count = initial_count - len(self.tcell_population)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} exhausted T-cells")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up exhausted T-cells: {e}")
    
    def _generate_strategy_hash(self, params: Dict[str, Any]) -> str:
        """Generate strategy hash (TCR) from parameters."""
        try:
            # Create deterministic string from parameters
            param_string = f"{params['source']}_{params['target']}_{params['delta_t']}_" \
                          f"{params['omega_weight']:.3f}_{params['xi_weight']:.3f}_{params['phi_weight']:.3f}_" \
                          f"{params['timestamp']}"
            
            # Generate SHA-256 hash
            return hashlib.sha256(param_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error generating strategy hash: {e}")
            return ""
    
    def _generate_dna_sequence(self, params: Dict[str, Any]) -> str:
        """Generate DNA-like sequence for strategy encoding."""
        try:
            # Convert parameters to binary-like sequence
            sequence_length = self.config['dna_sequence_length']
            
            # Create sequence from parameters
            param_bytes = str(params).encode()
            hash_bytes = hashlib.sha256(param_bytes).digest()
            
            # Convert to DNA-like sequence (A, T, G, C)
            dna_bases = ['A', 'T', 'G', 'C']
            dna_sequence = ""
            
            for byte in hash_bytes:
                for bit in range(4):
                    base_index = (byte >> (bit * 2)) & 3
                    dna_sequence += dna_bases[base_index]
            
            # Truncate or pad to desired length
            if len(dna_sequence) > sequence_length:
                dna_sequence = dna_sequence[:sequence_length]
            else:
                dna_sequence = dna_sequence.ljust(sequence_length, 'A')
            
            return dna_sequence
            
        except Exception as e:
            self.logger.error(f"Error generating DNA sequence: {e}")
            return "A" * self.config['dna_sequence_length']
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-x))
    
    def _should_mutate(self, tcell: TCellStrategy) -> bool:
        """Determine if T-cell should mutate."""
        if not self.config['enable_mutations']:
            return False
        
        # Higher mutation rate for low-performing T-cells
        mutation_probability = self.config['mutation_rate'] * (1.0 - tcell.survival_score)
        return np.random.random() < mutation_probability
    
    def _load_tcell_population(self):
        """Load T-cell population from file."""
        try:
            registry_file = self.config['tcell_registry_file']
            
            if not registry_file:
                return
            
            # Implementation would load from JSON file
            # For now, create some default T-cells
            self._create_default_tcells()
            
        except Exception as e:
            self.logger.error(f"Error loading T-cell population: {e}")
    
    def _create_default_tcells(self):
        """Create default T-cell strategies."""
        try:
            # Create some default front-running strategies
            default_strategies = [
                ("BTC", "ETH", 2),
                ("BTC", "XRP", -1),
                ("ETH", "SOL", 1),
                ("XRP", "USDC", 3)
            ]
            
            for source, target, delta_t in default_strategies:
                self.create_tcell_strategy(source, target, delta_t)
            
            self.logger.info(f"Created {len(default_strategies)} default T-cell strategies")
            
        except Exception as e:
            self.logger.error(f"Error creating default T-cells: {e}")
    
    def get_survival_statistics(self) -> Dict[str, Any]:
        """Get T-cell survival statistics."""
        try:
            total_tcells = len(self.tcell_population)
            
            state_counts = {}
            for state in TCellState:
                state_counts[state.value] = sum(
                    1 for tcell in self.tcell_population.values() 
                    if tcell.state == state
                )
            
            avg_survival_score = np.mean([tcell.survival_score for tcell in self.tcell_population.values()]) if total_tcells > 0 else 0.0
            
            return {
                'total_tcells': total_tcells,
                'state_distribution': state_counts,
                'average_survival_score': avg_survival_score,
                'survival_stats': self.survival_stats,
                'mutation_rate': self.config['mutation_rate'],
                'activation_threshold': self.config['activation_threshold']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting survival statistics: {e}")
            return {}

# Factory function
def create_tcell_survival_engine(config: Optional[Dict[str, Any]] = None) -> TCellSurvivalEngine:
    """Create a T-cell survival engine instance."""
    return TCellSurvivalEngine(config) 