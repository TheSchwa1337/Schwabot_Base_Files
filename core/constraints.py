from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_FRACTAL_PATH = CONFIG_DIR / "fractal_core.yaml"

class ConstraintType(Enum):
    """Types of system constraints"""
    MATHEMATICAL = "mathematical"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    PATHWAY = "pathway"
    BUSINESS = "business"
    SAFETY = "safety"

class ConstraintSeverity(Enum):
    """Constraint violation severity levels"""
    CRITICAL = "critical"      # System must halt
    HIGH = "high"             # Immediate attention required
    MEDIUM = "medium"         # Monitor closely
    LOW = "low"              # Advisory only

@dataclass
class SystemConstraint:
    """Individual system constraint definition"""
    constraint_id: str
    constraint_type: ConstraintType
    description: str
    validation_function: callable
    violation_message: str
    severity: ConstraintSeverity
    enabled: bool = True
    violation_count: int = 0
    last_violation: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class IntegrationBounds:
    """Bounds for system integration parameters"""
    # Trading bounds
    max_position_size: float = 1.0
    min_position_size: float = 0.001
    max_leverage: float = 10.0
    min_leverage: float = 0.1
    
    # Processing bounds
    max_tick_processing_rate: int = 10000  # per hour
    max_memory_usage_gb: float = 8.0
    max_cpu_usage_percent: float = 80.0
    max_gpu_usage_percent: float = 90.0
    
    # Mathematical bounds
    min_sustainment_index: float = 0.65
    max_entropy_threshold: float = 0.95
    min_correlation_threshold: float = 0.25
    max_drift_coefficient: float = 2.0
    
    # Integration bounds
    max_component_integration_level: float = 1.0
    min_component_integration_level: float = 0.0
    max_pathway_routing_depth: int = 10
    min_vector_routing_strength: float = 0.1

@dataclass
class PathwayConstraints:
    """Constraints for pathway routing and integration"""
    # NCCO constraints
    ncco_max_volume_control: float = 10.0
    ncco_min_volume_control: float = 0.1
    ncco_integration_bounds: Tuple[float, float] = (0.0, 1.0)
    
    # SFS constraints  
    sfs_max_speed_multiplier: float = 5.0
    sfs_min_speed_multiplier: float = 0.2
    sfs_integration_bounds: Tuple[float, float] = (0.0, 1.0)
    
    # ALIF pathway constraints
    alif_max_pathway_depth: int = 20
    alif_min_pathway_strength: float = 0.1
    alif_pathway_integration_bounds: Tuple[float, float] = (0.0, 1.0)
    
    # GAN constraints
    gan_max_generation_rate: int = 1000  # per minute
    gan_min_discriminator_accuracy: float = 0.7
    gan_integration_bounds: Tuple[float, float] = (0.0, 1.0)
    
    # UFS constraints
    ufs_max_fractal_depth: int = 50
    ufs_min_fractal_coherence: float = 0.3
    ufs_integration_bounds: Tuple[float, float] = (0.0, 1.0)

class SystemConstraintsManager:
    """
    Central constraints manager for the entire Schwabot system.
    
    This ensures all components operate within defined bounds and maintains
    system stability across all integration points including NCCO, SFS, 
    ALIF pathways, GAN, UFS, and tesseract visualizers.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.integration_bounds = IntegrationBounds()
        self.pathway_constraints = PathwayConstraints()
        self.constraints: Dict[str, SystemConstraint] = {}
        self.violation_history: List[Dict] = []
        
        # Initialize core constraints
        self._initialize_mathematical_constraints()
        self._initialize_integration_constraints()
        self._initialize_pathway_constraints()
        self._initialize_performance_constraints()
        self._initialize_safety_constraints()
        
        logger.info(f"SystemConstraintsManager initialized with {len(self.constraints)} constraints")
    
    def _initialize_mathematical_constraints(self):
        """Initialize mathematical system constraints"""
        
        # Sustainment index bounds
        self.add_constraint(SystemConstraint(
            constraint_id="sustainment_index_bounds",
            constraint_type=ConstraintType.MATHEMATICAL,
            description="Sustainment index must remain above critical threshold",
            validation_function=lambda v: v.get('sustainment_index', 1.0) >= self.integration_bounds.min_sustainment_index,
            violation_message="Sustainment index below critical threshold",
            severity=ConstraintSeverity.CRITICAL,
            tags=["sustainment", "mathematical", "critical"]
        ))
        
        # Entropy bounds
        self.add_constraint(SystemConstraint(
            constraint_id="entropy_bounds",
            constraint_type=ConstraintType.MATHEMATICAL,
            description="System entropy must not exceed maximum threshold",
            validation_function=lambda v: v.get('entropy', 0.0) <= self.integration_bounds.max_entropy_threshold,
            violation_message="System entropy exceeds maximum threshold",
            severity=ConstraintSeverity.HIGH,
            tags=["entropy", "mathematical", "stability"]
        ))
        
        # Correlation bounds
        self.add_constraint(SystemConstraint(
            constraint_id="correlation_bounds",
            constraint_type=ConstraintType.MATHEMATICAL,
            description="Hash correlations must meet minimum threshold",
            validation_function=lambda v: v.get('correlation_strength', 0.0) >= self.integration_bounds.min_correlation_threshold,
            violation_message="Hash correlation below minimum threshold",
            severity=ConstraintSeverity.MEDIUM,
            tags=["correlation", "mathematical", "hash"]
        ))
        
        # Drift coefficient bounds
        self.add_constraint(SystemConstraint(
            constraint_id="drift_coefficient_bounds",
            constraint_type=ConstraintType.MATHEMATICAL,
            description="Drift coefficient must remain within bounds",
            validation_function=lambda v: abs(v.get('drift_coefficient', 0.0)) <= self.integration_bounds.max_drift_coefficient,
            violation_message="Drift coefficient exceeds maximum bounds",
            severity=ConstraintSeverity.HIGH,
            tags=["drift", "mathematical", "stability"]
        ))
    
    def _initialize_integration_constraints(self):
        """Initialize component integration constraints"""
        
        # Component integration level bounds
        self.add_constraint(SystemConstraint(
            constraint_id="component_integration_bounds",
            constraint_type=ConstraintType.INTEGRATION,
            description="Component integration levels must be within valid range",
            validation_function=lambda v: self.integration_bounds.min_component_integration_level <= v.get('integration_level', 0.5) <= self.integration_bounds.max_component_integration_level,
            violation_message="Component integration level out of bounds",
            severity=ConstraintSeverity.MEDIUM,
            tags=["integration", "component", "bounds"]
        ))
        
        # Pathway routing depth
        self.add_constraint(SystemConstraint(
            constraint_id="pathway_routing_depth",
            constraint_type=ConstraintType.INTEGRATION,
            description="Pathway routing depth must not exceed maximum",
            validation_function=lambda v: v.get('routing_depth', 0) <= self.integration_bounds.max_pathway_routing_depth,
            violation_message="Pathway routing depth exceeds maximum",
            severity=ConstraintSeverity.HIGH,
            tags=["pathway", "routing", "depth"]
        ))
        
        # Vector routing strength
        self.add_constraint(SystemConstraint(
            constraint_id="vector_routing_strength",
            constraint_type=ConstraintType.INTEGRATION,
            description="Vector routing strength must meet minimum requirements",
            validation_function=lambda v: v.get('vector_strength', 0.0) >= self.integration_bounds.min_vector_routing_strength,
            violation_message="Vector routing strength below minimum",
            severity=ConstraintSeverity.MEDIUM,
            tags=["vector", "routing", "strength"]
        ))
    
    def _initialize_pathway_constraints(self):
        """Initialize pathway-specific constraints"""
        
        # NCCO volume control bounds
        self.add_constraint(SystemConstraint(
            constraint_id="ncco_volume_bounds",
            constraint_type=ConstraintType.PATHWAY,
            description="NCCO volume control must be within bounds",
            validation_function=lambda v: self.pathway_constraints.ncco_min_volume_control <= v.get('ncco_volume', 1.0) <= self.pathway_constraints.ncco_max_volume_control,
            violation_message="NCCO volume control out of bounds",
            severity=ConstraintSeverity.HIGH,
            tags=["ncco", "volume", "control"]
        ))
        
        # SFS speed control bounds
        self.add_constraint(SystemConstraint(
            constraint_id="sfs_speed_bounds",
            constraint_type=ConstraintType.PATHWAY,
            description="SFS speed control must be within bounds",
            validation_function=lambda v: self.pathway_constraints.sfs_min_speed_multiplier <= v.get('sfs_speed', 1.0) <= self.pathway_constraints.sfs_max_speed_multiplier,
            violation_message="SFS speed control out of bounds",
            severity=ConstraintSeverity.HIGH,
            tags=["sfs", "speed", "control"]
        ))
        
        # ALIF pathway depth
        self.add_constraint(SystemConstraint(
            constraint_id="alif_pathway_depth",
            constraint_type=ConstraintType.PATHWAY,
            description="ALIF pathway depth must not exceed maximum",
            validation_function=lambda v: v.get('alif_depth', 0) <= self.pathway_constraints.alif_max_pathway_depth,
            violation_message="ALIF pathway depth exceeds maximum",
            severity=ConstraintSeverity.MEDIUM,
            tags=["alif", "pathway", "depth"]
        ))
        
        # GAN generation rate
        self.add_constraint(SystemConstraint(
            constraint_id="gan_generation_rate",
            constraint_type=ConstraintType.PATHWAY,
            description="GAN generation rate must not exceed maximum",
            validation_function=lambda v: v.get('gan_rate', 0) <= self.pathway_constraints.gan_max_generation_rate,
            violation_message="GAN generation rate exceeds maximum",
            severity=ConstraintSeverity.MEDIUM,
            tags=["gan", "generation", "rate"]
        ))
        
        # UFS fractal depth
        self.add_constraint(SystemConstraint(
            constraint_id="ufs_fractal_depth",
            constraint_type=ConstraintType.PATHWAY,
            description="UFS fractal depth must not exceed maximum",
            validation_function=lambda v: v.get('ufs_depth', 0) <= self.pathway_constraints.ufs_max_fractal_depth,
            violation_message="UFS fractal depth exceeds maximum",
            severity=ConstraintSeverity.MEDIUM,
            tags=["ufs", "fractal", "depth"]
        ))
    
    def _initialize_performance_constraints(self):
        """Initialize performance-related constraints"""
        
        # Memory usage bounds
        self.add_constraint(SystemConstraint(
            constraint_id="memory_usage_bounds",
            constraint_type=ConstraintType.PERFORMANCE,
            description="Memory usage must not exceed maximum",
            validation_function=lambda v: v.get('memory_gb', 0.0) <= self.integration_bounds.max_memory_usage_gb,
            violation_message="Memory usage exceeds maximum",
            severity=ConstraintSeverity.HIGH,
            tags=["memory", "performance", "resource"]
        ))
        
        # CPU usage bounds
        self.add_constraint(SystemConstraint(
            constraint_id="cpu_usage_bounds",
            constraint_type=ConstraintType.PERFORMANCE,
            description="CPU usage must not exceed maximum",
            validation_function=lambda v: v.get('cpu_percent', 0.0) <= self.integration_bounds.max_cpu_usage_percent,
            violation_message="CPU usage exceeds maximum",
            severity=ConstraintSeverity.HIGH,
            tags=["cpu", "performance", "resource"]
        ))
        
        # Tick processing rate
        self.add_constraint(SystemConstraint(
            constraint_id="tick_processing_rate",
            constraint_type=ConstraintType.PERFORMANCE,
            description="Tick processing rate must not exceed maximum",
            validation_function=lambda v: v.get('ticks_per_hour', 0) <= self.integration_bounds.max_tick_processing_rate,
            violation_message="Tick processing rate exceeds maximum",
            severity=ConstraintSeverity.MEDIUM,
            tags=["tick", "processing", "rate"]
        ))
    
    def _initialize_safety_constraints(self):
        """Initialize safety and risk constraints"""
        
        # Position size bounds
        self.add_constraint(SystemConstraint(
            constraint_id="position_size_bounds",
            constraint_type=ConstraintType.SAFETY,
            description="Position size must be within safe bounds",
            validation_function=lambda v: self.integration_bounds.min_position_size <= v.get('position_size', 0.1) <= self.integration_bounds.max_position_size,
            violation_message="Position size out of safe bounds",
            severity=ConstraintSeverity.CRITICAL,
            tags=["position", "safety", "risk"]
        ))
        
        # Leverage bounds
        self.add_constraint(SystemConstraint(
            constraint_id="leverage_bounds",
            constraint_type=ConstraintType.SAFETY,
            description="Leverage must be within safe bounds",
            validation_function=lambda v: self.integration_bounds.min_leverage <= v.get('leverage', 1.0) <= self.integration_bounds.max_leverage,
            violation_message="Leverage out of safe bounds",
            severity=ConstraintSeverity.CRITICAL,
            tags=["leverage", "safety", "risk"]
        ))
    
    def add_constraint(self, constraint: SystemConstraint):
        """Add a new constraint to the system"""
        self.constraints[constraint.constraint_id] = constraint
        logger.debug(f"Added constraint: {constraint.constraint_id}")
    
    def validate_system_state(self, system_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate complete system state against all constraints
        
        Args:
            system_state: Current system state to validate
            
        Returns:
            Tuple of (is_valid, violation_messages)
        """
        violations = []
        critical_violations = []
        
        for constraint_id, constraint in self.constraints.items():
            if not constraint.enabled:
                continue
            
            try:
                if not constraint.validation_function(system_state):
                    violation_msg = f"{constraint.constraint_id}: {constraint.violation_message}"
                    violations.append(violation_msg)
                    
                    # Update violation tracking
                    constraint.violation_count += 1
                    constraint.last_violation = datetime.now()
                    
                    # Track critical violations separately
                    if constraint.severity == ConstraintSeverity.CRITICAL:
                        critical_violations.append(violation_msg)
                    
                    # Log violation
                    self.violation_history.append({
                        'constraint_id': constraint_id,
                        'violation_message': constraint.violation_message,
                        'severity': constraint.severity.value,
                        'timestamp': datetime.now().isoformat(),
                        'system_state_excerpt': self._extract_relevant_state(system_state, constraint.tags)
                    })
                    
            except Exception as e:
                logger.error(f"Error validating constraint {constraint_id}: {e}")
                violations.append(f"{constraint_id}: Validation error - {e}")
        
        # Critical violations mean system is not valid
        is_valid = len(critical_violations) == 0
        
        if violations:
            logger.warning(f"System constraint violations: {len(violations)} total, {len(critical_violations)} critical")
        
        return is_valid, violations
    
    def _extract_relevant_state(self, system_state: Dict[str, Any], tags: List[str]) -> Dict[str, Any]:
        """Extract relevant state information based on constraint tags"""
        relevant_state = {}
        
        for tag in tags:
            for key, value in system_state.items():
                if tag.lower() in key.lower():
                    relevant_state[key] = value
        
        return relevant_state
    
    def get_constraint_violations_summary(self) -> Dict[str, Any]:
        """Get summary of constraint violations"""
        violation_counts = {}
        recent_violations = []
        
        for constraint in self.constraints.values():
            if constraint.violation_count > 0:
                violation_counts[constraint.constraint_id] = {
                    'count': constraint.violation_count,
                    'severity': constraint.severity.value,
                    'last_violation': constraint.last_violation.isoformat() if constraint.last_violation else None
                }
        
        # Get recent violations (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        for violation in self.violation_history:
            violation_time = datetime.fromisoformat(violation['timestamp'])
            if violation_time > cutoff_time:
                recent_violations.append(violation)
        
        return {
            'total_constraints': len(self.constraints),
            'enabled_constraints': len([c for c in self.constraints.values() if c.enabled]),
            'constraints_with_violations': len(violation_counts),
            'violation_counts': violation_counts,
            'recent_violations': recent_violations,
            'violation_history_size': len(self.violation_history)
        }
    
    def validate_pathway_integration(self, component_id: str, integration_data: Dict[str, Any]) -> bool:
        """
        Validate specific pathway integration parameters
        
        Args:
            component_id: ID of the component being integrated
            integration_data: Integration parameters to validate
            
        Returns:
            True if integration parameters are valid
        """
        # Validate based on component type
        if 'ncco' in component_id.lower():
            return self._validate_ncco_integration(integration_data)
        elif 'sfs' in component_id.lower():
            return self._validate_sfs_integration(integration_data)
        elif 'alif' in component_id.lower():
            return self._validate_alif_integration(integration_data)
        elif 'gan' in component_id.lower():
            return self._validate_gan_integration(integration_data)
        elif 'ufs' in component_id.lower():
            return self._validate_ufs_integration(integration_data)
        else:
            return self._validate_generic_integration(integration_data)
    
    def _validate_ncco_integration(self, data: Dict[str, Any]) -> bool:
        """Validate NCCO-specific integration parameters"""
        volume_control = data.get('volume_control', 1.0)
        integration_level = data.get('integration_level', 0.5)
        
        return (self.pathway_constraints.ncco_min_volume_control <= volume_control <= self.pathway_constraints.ncco_max_volume_control and
                self.pathway_constraints.ncco_integration_bounds[0] <= integration_level <= self.pathway_constraints.ncco_integration_bounds[1])
    
    def _validate_sfs_integration(self, data: Dict[str, Any]) -> bool:
        """Validate SFS-specific integration parameters"""
        speed_multiplier = data.get('speed_multiplier', 1.0)
        integration_level = data.get('integration_level', 0.5)
        
        return (self.pathway_constraints.sfs_min_speed_multiplier <= speed_multiplier <= self.pathway_constraints.sfs_max_speed_multiplier and
                self.pathway_constraints.sfs_integration_bounds[0] <= integration_level <= self.pathway_constraints.sfs_integration_bounds[1])
    
    def _validate_alif_integration(self, data: Dict[str, Any]) -> bool:
        """Validate ALIF pathway integration parameters"""
        pathway_depth = data.get('pathway_depth', 1)
        pathway_strength = data.get('pathway_strength', 0.5)
        integration_level = data.get('integration_level', 0.5)
        
        return (pathway_depth <= self.pathway_constraints.alif_max_pathway_depth and
                pathway_strength >= self.pathway_constraints.alif_min_pathway_strength and
                self.pathway_constraints.alif_pathway_integration_bounds[0] <= integration_level <= self.pathway_constraints.alif_pathway_integration_bounds[1])
    
    def _validate_gan_integration(self, data: Dict[str, Any]) -> bool:
        """Validate GAN integration parameters"""
        generation_rate = data.get('generation_rate', 100)
        discriminator_accuracy = data.get('discriminator_accuracy', 0.8)
        integration_level = data.get('integration_level', 0.5)
        
        return (generation_rate <= self.pathway_constraints.gan_max_generation_rate and
                discriminator_accuracy >= self.pathway_constraints.gan_min_discriminator_accuracy and
                self.pathway_constraints.gan_integration_bounds[0] <= integration_level <= self.pathway_constraints.gan_integration_bounds[1])
    
    def _validate_ufs_integration(self, data: Dict[str, Any]) -> bool:
        """Validate UFS integration parameters"""
        fractal_depth = data.get('fractal_depth', 10)
        fractal_coherence = data.get('fractal_coherence', 0.5)
        integration_level = data.get('integration_level', 0.5)
        
        return (fractal_depth <= self.pathway_constraints.ufs_max_fractal_depth and
                fractal_coherence >= self.pathway_constraints.ufs_min_fractal_coherence and
                self.pathway_constraints.ufs_integration_bounds[0] <= integration_level <= self.pathway_constraints.ufs_integration_bounds[1])
    
    def _validate_generic_integration(self, data: Dict[str, Any]) -> bool:
        """Validate generic integration parameters"""
        integration_level = data.get('integration_level', 0.5)
        return (self.integration_bounds.min_component_integration_level <= integration_level <= self.integration_bounds.max_component_integration_level)
    
    def update_bounds(self, new_bounds: Dict[str, Any]):
        """Update system bounds dynamically"""
        for key, value in new_bounds.items():
            if hasattr(self.integration_bounds, key):
                setattr(self.integration_bounds, key, value)
                logger.info(f"Updated integration bound {key} to {value}")
            elif hasattr(self.pathway_constraints, key):
                setattr(self.pathway_constraints, key, value)
                logger.info(f"Updated pathway constraint {key} to {value}")
    
    def get_system_bounds_summary(self) -> Dict[str, Any]:
        """Get complete summary of system bounds and constraints"""
        return {
            'integration_bounds': {
                'trading': {
                    'max_position_size': self.integration_bounds.max_position_size,
                    'min_position_size': self.integration_bounds.min_position_size,
                    'max_leverage': self.integration_bounds.max_leverage,
                    'min_leverage': self.integration_bounds.min_leverage,
                },
                'processing': {
                    'max_tick_processing_rate': self.integration_bounds.max_tick_processing_rate,
                    'max_memory_usage_gb': self.integration_bounds.max_memory_usage_gb,
                    'max_cpu_usage_percent': self.integration_bounds.max_cpu_usage_percent,
                    'max_gpu_usage_percent': self.integration_bounds.max_gpu_usage_percent,
                },
                'mathematical': {
                    'min_sustainment_index': self.integration_bounds.min_sustainment_index,
                    'max_entropy_threshold': self.integration_bounds.max_entropy_threshold,
                    'min_correlation_threshold': self.integration_bounds.min_correlation_threshold,
                    'max_drift_coefficient': self.integration_bounds.max_drift_coefficient,
                }
            },
            'pathway_constraints': {
                'ncco': {
                    'volume_bounds': (self.pathway_constraints.ncco_min_volume_control, self.pathway_constraints.ncco_max_volume_control),
                    'integration_bounds': self.pathway_constraints.ncco_integration_bounds,
                },
                'sfs': {
                    'speed_bounds': (self.pathway_constraints.sfs_min_speed_multiplier, self.pathway_constraints.sfs_max_speed_multiplier),
                    'integration_bounds': self.pathway_constraints.sfs_integration_bounds,
                },
                'alif': {
                    'pathway_depth_max': self.pathway_constraints.alif_max_pathway_depth,
                    'pathway_strength_min': self.pathway_constraints.alif_min_pathway_strength,
                    'integration_bounds': self.pathway_constraints.alif_pathway_integration_bounds,
                },
                'gan': {
                    'generation_rate_max': self.pathway_constraints.gan_max_generation_rate,
                    'discriminator_accuracy_min': self.pathway_constraints.gan_min_discriminator_accuracy,
                    'integration_bounds': self.pathway_constraints.gan_integration_bounds,
                },
                'ufs': {
                    'fractal_depth_max': self.pathway_constraints.ufs_max_fractal_depth,
                    'fractal_coherence_min': self.pathway_constraints.ufs_min_fractal_coherence,
                    'integration_bounds': self.pathway_constraints.ufs_integration_bounds,
                }
            },
            'constraint_summary': self.get_constraint_violations_summary()
        }

# Global constraints manager instance
constraints_manager = SystemConstraintsManager()

# Convenience functions for system-wide access
def validate_system_state(system_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Global system state validation"""
    return constraints_manager.validate_system_state(system_state)

def validate_pathway_integration(component_id: str, integration_data: Dict[str, Any]) -> bool:
    """Global pathway integration validation"""
    return constraints_manager.validate_pathway_integration(component_id, integration_data)

def get_system_bounds() -> Dict[str, Any]:
    """Get current system bounds"""
    return constraints_manager.get_system_bounds_summary()

def update_system_bounds(new_bounds: Dict[str, Any]):
    """Update system bounds globally"""
    constraints_manager.update_bounds(new_bounds)
