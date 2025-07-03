import numpy as np
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .clean_math_foundation import CleanMathFoundation


class QuantumSyncStatus(Enum):
    """Quantum synchronization status levels."""
    UNSYNCED = "unsynced"
    PARTIAL_SYNC = "partial_sync"
    FULL_SYNC = "full_sync"
    RESONANCE = "resonance"
    QUANTUM_HOLD = "quantum_hold"


@dataclass
class ZPEVector:
    """Zero Point Energy Vector for quantum state representation."""
    energy: float
    frequency: float
    mass_coefficient: float
    sync_status: QuantumSyncStatus
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZBEBalance:
    """Zero-Based Equilibrium Balance representation."""
    status: float  # [-1, 0, 1] representing market equilibrium
    entry_price: float
    current_price: float
    lower_bound: float
    upper_bound: float
    stability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZPEZBECore:
    """
    Comprehensive Zero Point Energy and Zero-Based Equilibrium Core.
    
    Integrates quantum-inspired mathematical principles for 
    strategy decision making and environmental synchronization.
    """

    # Fundamental constants
    PLANCK_CONSTANT = 6.62607015e-34  # Joule-seconds
    EARTH_SCHUMANN_FREQUENCY = 7.83  # Hz, Earth's fundamental resonance frequency
    QUANTUM_SYNC_THRESHOLD = 2.72e-33  # Quantum synchronization energy threshold

    def __init__(self, math_foundation: Optional[CleanMathFoundation] = None) -> None:
        """
        Initialize ZPE-ZBE Core with optional mathematical foundation.
        
        Args:
            math_foundation: Optional mathematical foundation for advanced calculations
        """
        self.math_foundation: CleanMathFoundation = math_foundation or CleanMathFoundation()
        
    def calculate_zero_point_energy(
        self, 
        frequency: float = EARTH_SCHUMANN_FREQUENCY, 
        mass_coefficient: float = 1e-6
    ) -> ZPEVector:
        """
        Calculate Zero Point Energy with quantum synchronization assessment.
        
        Args:
            frequency: Oscillation frequency (default: Earth's Schumann resonance)
            mass_coefficient: Mass scaling factor
        
        Returns:
            ZPE Vector with quantum sync status
        """
        # Basic ZPE calculation: E = (1/2) * h * f
        zpe_energy = 0.5 * self.PLANCK_CONSTANT * frequency * mass_coefficient
        
        # Determine quantum sync status based on energy levels
        sync_status = self._assess_quantum_sync(zpe_energy)
        
        return ZPEVector(
            energy=zpe_energy,
            frequency=frequency,
            mass_coefficient=mass_coefficient,
            sync_status=sync_status,
            timestamp=time.time(),
            metadata={
                "quantum_potential": zpe_energy / self.QUANTUM_SYNC_THRESHOLD,
                "resonance_factor": frequency / self.EARTH_SCHUMANN_FREQUENCY
            }
        )
    
    def calculate_zbe_balance(
        self, 
        entry_price: float, 
        current_price: float, 
        lower_bound: float, 
        upper_bound: float
    ) -> ZBEBalance:
        """
        Calculate Zero-Based Equilibrium Balance.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            lower_bound: Lower equilibrium boundary
            upper_bound: Upper equilibrium boundary
        
        Returns:
            ZBE Balance representation
        """
        # Calculate equilibrium status
        if current_price < lower_bound:
            status = -1.0  # Deep under equilibrium
        elif lower_bound <= current_price <= upper_bound:
            status = 0.0   # Inside stable ZBE range
        else:
            status = 1.0   # Over equilibrium break band
        
        # Calculate stability score
        price_range = upper_bound - lower_bound
        normalized_position = (current_price - lower_bound) / price_range if price_range > 0 else 0.5
        stability_score = 1.0 - abs(normalized_position - 0.5) * 2
        
        return ZBEBalance(
            status=status,
            entry_price=entry_price,
            current_price=current_price,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            stability_score=stability_score,
            metadata={
                "price_deviation": abs(current_price - entry_price) / entry_price,
                "equilibrium_range": price_range
            }
        )
    
    def _assess_quantum_sync(self, zpe_energy: float) -> QuantumSyncStatus:
        """
        Assess quantum synchronization status based on ZPE energy.
        
        Args:
            zpe_energy: Calculated Zero Point Energy
        
        Returns:
            Quantum synchronization status
        """
        if zpe_energy > self.QUANTUM_SYNC_THRESHOLD * 2:
            return QuantumSyncStatus.RESONANCE
        elif zpe_energy > self.QUANTUM_SYNC_THRESHOLD:
            return QuantumSyncStatus.FULL_SYNC
        elif zpe_energy > self.QUANTUM_SYNC_THRESHOLD * 0.5:
            return QuantumSyncStatus.PARTIAL_SYNC
        else:
            return QuantumSyncStatus.UNSYNCED
    
    def dual_matrix_sync_trigger(
        self, 
        zpe_vector: ZPEVector, 
        zbe_balance: ZBEBalance
    ) -> Dict[str, Any]:
        """
        Determine dual matrix synchronization trigger conditions.
        
        Args:
            zpe_vector: Zero Point Energy vector
            zbe_balance: Zero-Based Equilibrium balance
        
        Returns:
            Synchronization trigger metadata
        """
        # Quantum sync conditions
        is_quantum_synced = (
            zpe_vector.sync_status in [
                QuantumSyncStatus.FULL_SYNC, 
                QuantumSyncStatus.RESONANCE
            ] and 
            zbe_balance.status == 0
        )
        
        return {
            "is_synced": is_quantum_synced,
            "sync_strategy": "LotusHold_Ω33" if is_quantum_synced else "NeutralHold",
            "zpe_energy": zpe_vector.energy,
            "zbe_status": zbe_balance.status,
            "recommended_action": "hold" if is_quantum_synced else "monitor"
        }

    def generate_quantum_soulprint_vector(
        self, 
        zpe_vector: ZPEVector, 
        zbe_balance: ZBEBalance
    ) -> Dict[str, float]:
        """
        Generate a soulprint-compatible vector from ZPE and ZBE calculations.
        
        Args:
            zpe_vector: Zero Point Energy vector
            zbe_balance: Zero-Based Equilibrium balance
        
        Returns:
            Dictionary compatible with SoulprintRegistry vector format
        """
        return {
            'entropy': zbe_balance.stability_score,
            'momentum': zpe_vector.metadata.get('quantum_potential', 0.0),
            'volatility': abs(zbe_balance.status),
            'temporal_variance': zpe_vector.energy / self.QUANTUM_SYNC_THRESHOLD,
            'quantum_sync_status': zpe_vector.sync_status.value,
            'zbe_status': zbe_balance.status
        }

    def assess_quantum_strategy_confidence(
        self, 
        zpe_vector: ZPEVector, 
        zbe_balance: ZBEBalance
    ) -> float:
        """
        Calculate confidence score based on quantum synchronization metrics.
        
        Args:
            zpe_vector: Zero Point Energy vector
            zbe_balance: Zero-Based Equilibrium balance
        
        Returns:
            Confidence score between 0 and 1
        """
        # Quantum sync status confidence
        sync_confidence_map = {
            QuantumSyncStatus.RESONANCE: 0.95,
            QuantumSyncStatus.FULL_SYNC: 0.85,
            QuantumSyncStatus.PARTIAL_SYNC: 0.65,
            QuantumSyncStatus.UNSYNCED: 0.35
        }
        
        sync_confidence = sync_confidence_map.get(zpe_vector.sync_status, 0.5)
        
        # Stability and equilibrium factors
        stability_factor = zbe_balance.stability_score
        equilibrium_factor = 1.0 - abs(zbe_balance.status)
        
        # Quantum potential contribution
        quantum_potential = zpe_vector.metadata.get('quantum_potential', 0.0)
        
        # Composite confidence calculation
        confidence = (
            sync_confidence * 0.4 +
            stability_factor * 0.3 +
            equilibrium_factor * 0.2 +
            quantum_potential * 0.1
        )
        
        return max(0.0, min(1.0, confidence))


@dataclass
class QuantumPerformanceEntry:
    """
    Detailed performance tracking for quantum-synchronized strategies.
    Extends traditional performance metrics with quantum-specific insights.
    """
    strategy_id: str
    quantum_sync_status: str
    zpe_energy: float
    zbe_status: float
    entry_timestamp: float
    exit_timestamp: Optional[float] = None
    profit: float = 0.0
    risk_score: float = 0.0
    thermal_state: str = "neutral"
    bit_phase: int = 16
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumPerformanceRegistry:
    """
    Performance tracking and adaptive learning registry for quantum-synchronized strategies.
    """
    def __init__(self, max_entries: int = 1000) -> None:
        """
        Initialize the quantum performance registry.
        
        Args:
            max_entries: Maximum number of performance entries to maintain
        """
        self.performance_entries: List[QuantumPerformanceEntry] = []
        self.max_entries: int = max_entries
        
    def add_performance_entry(self, entry: QuantumPerformanceEntry) -> None:
        """
        Add a new performance entry, maintaining max size.
        
        Args:
            entry: Quantum performance entry to add
        """
        # Maintain max entries by removing oldest if necessary
        if len(self.performance_entries) >= self.max_entries:
            self.performance_entries.pop(0)
        
        self.performance_entries.append(entry)
    
    def analyze_quantum_performance(self) -> Dict[str, Any]:
        """
        Analyze overall quantum strategy performance.
        
        Returns:
            Comprehensive performance analysis dictionary
        """
        if not self.performance_entries:
            return {
                "total_strategies": 0,
                "average_profit": 0.0,
                "performance_by_sync_status": {},
                "optimal_thermal_state": "neutral"
            }
        
        # Performance by quantum sync status
        sync_performance = {}
        total_profit = 0.0
        profitable_entries = 0
        
        for entry in self.performance_entries:
            total_profit += entry.profit
            if entry.profit > 0:
                profitable_entries += 1
            
            sync_status = entry.quantum_sync_status
            if sync_status not in sync_performance:
                sync_performance[sync_status] = {
                    "count": 0,
                    "total_profit": 0.0,
                    "avg_risk_score": 0.0
                }
            
            sync_performance[sync_status]["count"] += 1
            sync_performance[sync_status]["total_profit"] += entry.profit
            sync_performance[sync_status]["avg_risk_score"] += entry.risk_score
        
        # Normalize performance metrics
        for status, metrics in sync_performance.items():
            metrics["avg_profit"] = (
                metrics["total_profit"] / metrics["count"] 
                if metrics["count"] > 0 else 0.0
            )
            metrics["avg_risk_score"] /= metrics["count"] if metrics["count"] > 0 else 1.0
        
        # Determine optimal thermal state
        thermal_performance = {}
        for entry in self.performance_entries:
            if entry.thermal_state not in thermal_performance:
                thermal_performance[entry.thermal_state] = {
                    "total_profit": 0.0,
                    "count": 0
                }
            
            thermal_performance[entry.thermal_state]["total_profit"] += entry.profit
            thermal_performance[entry.thermal_state]["count"] += 1
        
        optimal_thermal_state = max(
            thermal_performance.items(), 
            key=lambda x: x[1]["total_profit"] / (x[1]["count"] or 1)
        )[0] if thermal_performance else "neutral"
        
        return {
            "total_strategies": len(self.performance_entries),
            "average_profit": total_profit / len(self.performance_entries) if self.performance_entries else 0.0,
            "profit_rate": profitable_entries / len(self.performance_entries) if self.performance_entries else 0.0,
            "performance_by_sync_status": sync_performance,
            "optimal_thermal_state": optimal_thermal_state
        }
    
    def recommend_quantum_strategy_params(self) -> Dict[str, Any]:
        """
        Recommend optimal strategy parameters based on historical performance.
        
        Returns:
            Recommended quantum strategy parameters
        """
        performance_analysis = self.analyze_quantum_performance()
        
        # Prioritize sync statuses with best performance
        best_sync_status = max(
            performance_analysis['performance_by_sync_status'].items(), 
            key=lambda x: x[1]['avg_profit']
        )[0] if performance_analysis['performance_by_sync_status'] else None
        
        return {
            "recommended_thermal_state": performance_analysis['optimal_thermal_state'],
            "best_sync_status": best_sync_status,
            "recommended_bit_phase": {
                "resonance": 32,
                "full_sync": 24,
                "partial_sync": 16,
                "unsynced": 8
            }.get(best_sync_status, 16),
            "risk_tolerance": 1.0 - performance_analysis.get('average_profit', 0.5)
        }


class ZPEZBEPerformanceTracker:
    """
    Performance tracking and adaptive learning for ZPE-ZBE quantum strategies.
    """
    def __init__(self) -> None:
        """
        Initialize performance tracking for ZPE-ZBE quantum strategies.
        """
        self.performance_registry: QuantumPerformanceRegistry = QuantumPerformanceRegistry()
    
    def log_strategy_performance(
        self, 
        zpe_vector: ZPEVector, 
        zbe_balance: ZBEBalance,
        strategy_metadata: Dict[str, Any]
    ) -> None:
        """
        Log performance of a quantum-synchronized strategy.
        
        Args:
            zpe_vector: Zero Point Energy vector
            zbe_balance: Zero-Based Equilibrium balance
            strategy_metadata: Additional strategy performance metadata
        """
        performance_entry = QuantumPerformanceEntry(
            strategy_id=strategy_metadata.get('strategy_id', 'unknown'),
            quantum_sync_status=zpe_vector.sync_status.value,
            zpe_energy=zpe_vector.energy,
            zbe_status=zbe_balance.status,
            entry_timestamp=zpe_vector.timestamp,
            profit=strategy_metadata.get('profit', 0.0),
            risk_score=strategy_metadata.get('risk_score', 0.5),
            thermal_state=strategy_metadata.get('thermal_state', 'neutral'),
            bit_phase=strategy_metadata.get('bit_phase', 16),
            metadata=strategy_metadata
        )
        
        self.performance_registry.add_performance_entry(performance_entry)
    
    def get_quantum_strategy_recommendations(self) -> Dict[str, Any]:
        """
        Get quantum strategy recommendations based on historical performance.
        
        Returns:
            Recommended strategy parameters
        """
        return self.performance_registry.recommend_quantum_strategy_params()
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive performance analysis.
        
        Returns:
            Detailed performance analysis
        """
        return self.performance_registry.analyze_quantum_performance()


def create_zpe_zbe_core() -> ZPEZBECore:
    """Factory function for creating ZPE-ZBE Core instance."""
    return ZPEZBECore() 