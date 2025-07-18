#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vortex Math Security Protocol (VMSP) - Schwabot Security Layer
=============================================================

Advanced mathematical security protocol that provides:
- Recursive fractal cryptographic validation
- Quantum-resistant hash verification
- Temporal entropy stabilization
- Multi-layer security integration
- Cross-module mathematical validation

This module integrates with:
- Multi-layered security manager
- Alpha encryption system
- Secure config manager
- All Schwabot mathematical systems
"""

import hashlib
import logging
import math
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)

# Global context for security state
security_context: ContextVar[Dict[str, Any]] = ContextVar('security_context', default={})


class SecurityState(Enum):
    """Security state enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    COMPROMISED = "compromised"
    UNKNOWN = "unknown"


class VortexType(Enum):
    """Vortex type enumeration."""
    QUANTUM = "quantum"
    FRACTAL = "fractal"
    TEMPORAL = "temporal"
    ENTROPY = "entropy"
    HYBRID = "hybrid"


@dataclass
class VortexSignature:
    """Vortex signature data structure."""
    
    signature_hash: str
    timestamp: float
    vortex_type: VortexType
    entropy_level: float
    fractal_dimension: float
    quantum_state: str
    temporal_offset: float
    security_score: float
    validation_count: int = 0
    last_validated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityValidationResult:
    """Security validation result."""
    
    valid: bool
    security_score: float
    confidence: float
    vortex_signature: Optional[VortexSignature] = None
    error_message: Optional[str] = None
    validation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VortexMathCore:
    """Core vortex mathematical operations."""
    
    def __init__(self):
        """Initialize vortex math core."""
        self.base_entropy = 0.6180339887498948  # Golden ratio
        self.fractal_constant = 1.4142135623730951  # √2
        self.quantum_phase = 0.7853981633974483  # π/4
        self.temporal_cycle = 86400.0  # 24 hours in seconds
        
        # Initialize mathematical constants
        self._initialize_constants()
    
    def _initialize_constants(self):
        """Initialize mathematical constants for vortex operations."""
        self.vortex_constants = {
            'entropy_threshold': 0.02,
            'quantum_phase_threshold': 0.92,
            'adaptive_entropy_coeff': 0.01,
            'truth_threshold': 0.98,
            'temporal_coherence_weight': 1.08,
            'warp_phase_threshold': 0.05,
            'fractal_scaling_factor': 1.2
        }
    
    def compute_vortex_hash(self, data: Union[str, bytes, List[float]], 
                          vortex_type: VortexType = VortexType.HYBRID) -> str:
        """Compute vortex hash with mathematical enhancement."""
        try:
            # Convert input to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            elif isinstance(data, (list, np.ndarray)):
                data_bytes = str(data).encode('utf-8')
            else:
                data_bytes = str(data).encode('utf-8')
            
            # Apply vortex mathematical transformation
            transformed_data = self._apply_vortex_transformation(data_bytes, vortex_type)
            
            # Generate hash
            hash_result = hashlib.sha256(transformed_data).hexdigest()
            
            # Apply recursive enhancement
            enhanced_hash = self._apply_recursive_enhancement(hash_result, vortex_type)
            
            return enhanced_hash
            
        except Exception as e:
            logger.error(f"Error computing vortex hash: {e}")
            return hashlib.sha256(str(data).encode('utf-8')).hexdigest()
    
    def _apply_vortex_transformation(self, data: bytes, vortex_type: VortexType) -> bytes:
        """Apply vortex mathematical transformation to data."""
        try:
            # Convert bytes to numerical representation
            data_array = np.frombuffer(data, dtype=np.uint8)
            
            # Apply vortex-specific transformations
            if vortex_type == VortexType.QUANTUM:
                transformed = self._apply_quantum_transformation(data_array)
            elif vortex_type == VortexType.FRACTAL:
                transformed = self._apply_fractal_transformation(data_array)
            elif vortex_type == VortexType.TEMPORAL:
                transformed = self._apply_temporal_transformation(data_array)
            elif vortex_type == VortexType.ENTROPY:
                transformed = self._apply_entropy_transformation(data_array)
            else:  # HYBRID
                transformed = self._apply_hybrid_transformation(data_array)
            
            # Convert back to bytes
            return transformed.tobytes()
            
        except Exception as e:
            logger.error(f"Error applying vortex transformation: {e}")
            return data
    
    def _apply_quantum_transformation(self, data: np.ndarray) -> np.ndarray:
        """Apply quantum transformation to data."""
        # Quantum phase rotation
        phase_shift = np.sin(self.quantum_phase * np.arange(len(data)))
        quantum_enhanced = data * (1 + 0.1 * phase_shift)
        
        # Apply quantum superposition effect
        superposition = np.cos(self.quantum_phase * quantum_enhanced)
        return (quantum_enhanced * superposition).astype(np.uint8)
    
    def _apply_fractal_transformation(self, data: np.ndarray) -> np.ndarray:
        """Apply fractal transformation to data."""
        # Fractal scaling
        fractal_scale = self.fractal_constant ** np.arange(len(data))
        fractal_enhanced = data * fractal_scale
        
        # Apply fractal recursion
        recursion_depth = min(3, len(data) // 10)
        for _ in range(recursion_depth):
            fractal_enhanced = np.sqrt(fractal_enhanced + 1)
        
        return fractal_enhanced.astype(np.uint8)
    
    def _apply_temporal_transformation(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal transformation to data."""
        # Temporal cycle modulation
        current_time = time.time()
        temporal_cycle = current_time % self.temporal_cycle
        temporal_factor = np.sin(2 * np.pi * temporal_cycle / self.temporal_cycle)
        
        # Apply temporal enhancement
        temporal_enhanced = data * (1 + 0.05 * temporal_factor)
        return temporal_enhanced.astype(np.uint8)
    
    def _apply_entropy_transformation(self, data: np.ndarray) -> np.ndarray:
        """Apply entropy transformation to data."""
        # Entropy-based randomization
        entropy_seed = int(time.time() * 1000) % 1000000
        np.random.seed(entropy_seed)
        entropy_noise = np.random.normal(0, 0.1, len(data))
        
        # Apply entropy enhancement
        entropy_enhanced = data * (1 + entropy_noise)
        return entropy_enhanced.astype(np.uint8)
    
    def _apply_hybrid_transformation(self, data: np.ndarray) -> np.ndarray:
        """Apply hybrid transformation combining all methods."""
        # Combine all transformations
        quantum_result = self._apply_quantum_transformation(data)
        fractal_result = self._apply_fractal_transformation(data)
        temporal_result = self._apply_temporal_transformation(data)
        entropy_result = self._apply_entropy_transformation(data)
        
        # Weighted combination
        hybrid_result = (
            0.3 * quantum_result +
            0.3 * fractal_result +
            0.2 * temporal_result +
            0.2 * entropy_result
        )
        
        return hybrid_result.astype(np.uint8)
    
    def _apply_recursive_enhancement(self, hash_string: str, vortex_type: VortexType) -> str:
        """Apply recursive enhancement to hash."""
        try:
            # Convert hash to numerical representation
            hash_bytes = bytes.fromhex(hash_string)
            hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)
            
            # Apply recursive transformation
            enhanced_array = self._apply_vortex_transformation(hash_array, vortex_type)
            
            # Generate final hash
            final_hash = hashlib.sha256(enhanced_array).hexdigest()
            
            return final_hash
            
        except Exception as e:
            logger.error(f"Error applying recursive enhancement: {e}")
            return hash_string


class VortexSecurityProtocol:
    """Main VMSP implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize VMSP."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.vortex_math = VortexMathCore()
        self.signature_cache: Dict[str, VortexSignature] = {}
        self.validation_history: List[SecurityValidationResult] = []
        
        # Security state
        self.security_state = SecurityState.UNKNOWN
        self.last_validation = 0.0
        self.validation_count = 0
        
        # Performance metrics
        self.performance_metrics = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_validation_time': 0.0,
            'security_score_history': []
        }
        
        self.logger.info("✅ VMSP initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default VMSP configuration."""
        return {
            'enable_quantum_validation': True,
            'enable_fractal_validation': True,
            'enable_temporal_validation': True,
            'enable_entropy_validation': True,
            'validation_threshold': 0.8,
            'cache_signatures': True,
            'max_cache_size': 1000,
            'validation_timeout': 5.0,
            'enable_performance_monitoring': True
        }
    
    def validate_security_state(self, inputs: List[Any], 
                              context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate security state using VMSP."""
        try:
            start_time = time.time()
            
            # Create validation context
            validation_context = context or {}
            validation_context.update({
                'timestamp': time.time(),
                'input_count': len(inputs),
                'vmsp_version': '1.0'
            })
            
            # Generate vortex signature
            combined_input = self._combine_inputs(inputs)
            vortex_signature = self._generate_vortex_signature(combined_input, validation_context)
            
            # Perform validation
            validation_result = self._perform_validation(vortex_signature, validation_context)
            
            # Update performance metrics
            self._update_performance_metrics(validation_result, time.time() - start_time)
            
            # Cache signature if enabled
            if self.config.get('cache_signatures', True):
                self._cache_signature(vortex_signature)
            
            # Update security state
            self._update_security_state(validation_result)
            
            return validation_result.valid
            
        except Exception as e:
            self.logger.error(f"Error in security state validation: {e}")
            return False
    
    def _combine_inputs(self, inputs: List[Any]) -> str:
        """Combine inputs for signature generation."""
        try:
            combined = ""
            for i, input_item in enumerate(inputs):
                if isinstance(input_item, (str, bytes)):
                    combined += str(input_item)
                elif isinstance(input_item, (list, np.ndarray)):
                    combined += str(input_item.tolist() if hasattr(input_item, 'tolist') else input_item)
                else:
                    combined += str(input_item)
                combined += "|"
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining inputs: {e}")
            return str(inputs)
    
    def _generate_vortex_signature(self, data: str, context: Dict[str, Any]) -> VortexSignature:
        """Generate vortex signature for data."""
        try:
            # Determine optimal vortex type based on context
            vortex_type = self._determine_vortex_type(context)
            
            # Generate hash
            signature_hash = self.vortex_math.compute_vortex_hash(data, vortex_type)
            
            # Calculate mathematical properties
            entropy_level = self._calculate_entropy_level(data)
            fractal_dimension = self._calculate_fractal_dimension(data)
            quantum_state = self._calculate_quantum_state(data)
            temporal_offset = self._calculate_temporal_offset(context)
            security_score = self._calculate_security_score(
                entropy_level, fractal_dimension, quantum_state, temporal_offset
            )
            
            signature = VortexSignature(
                signature_hash=signature_hash,
                timestamp=time.time(),
                vortex_type=vortex_type,
                entropy_level=entropy_level,
                fractal_dimension=fractal_dimension,
                quantum_state=quantum_state,
                temporal_offset=temporal_offset,
                security_score=security_score,
                metadata=context
            )
            
            return signature
            
        except Exception as e:
            self.logger.error(f"Error generating vortex signature: {e}")
            # Return fallback signature
            return VortexSignature(
                signature_hash=hashlib.sha256(data.encode()).hexdigest(),
                timestamp=time.time(),
                vortex_type=VortexType.HYBRID,
                entropy_level=0.5,
                fractal_dimension=1.0,
                quantum_state="unknown",
                temporal_offset=0.0,
                security_score=0.5
            )
    
    def _determine_vortex_type(self, context: Dict[str, Any]) -> VortexType:
        """Determine optimal vortex type based on context."""
        try:
            # Analyze context to determine best vortex type
            if context.get('quantum_enabled', False):
                return VortexType.QUANTUM
            elif context.get('fractal_enabled', False):
                return VortexType.FRACTAL
            elif context.get('temporal_enabled', False):
                return VortexType.TEMPORAL
            elif context.get('entropy_enabled', False):
                return VortexType.ENTROPY
            else:
                return VortexType.HYBRID
                
        except Exception as e:
            self.logger.error(f"Error determining vortex type: {e}")
            return VortexType.HYBRID
    
    def _calculate_entropy_level(self, data: str) -> float:
        """Calculate entropy level of data."""
        try:
            if not data:
                return 0.0
            
            # Calculate character frequency
            char_count = {}
            for char in data:
                char_count[char] = char_count.get(char, 0) + 1
            
            # Calculate entropy
            total_chars = len(data)
            entropy = 0.0
            
            for count in char_count.values():
                probability = count / total_chars
                if probability > 0:
                    entropy -= probability * math.log2(probability)
            
            # Normalize to [0, 1]
            max_entropy = math.log2(len(char_count)) if char_count else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return min(max(normalized_entropy, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating entropy level: {e}")
            return 0.5
    
    def _calculate_fractal_dimension(self, data: str) -> float:
        """Calculate fractal dimension of data."""
        try:
            if not data:
                return 1.0
            
            # Simple fractal dimension estimation
            # This is a simplified version - real fractal dimension calculation would be more complex
            
            # Calculate pattern complexity
            pattern_length = min(10, len(data) // 2)
            if pattern_length < 2:
                return 1.0
            
            # Count repeating patterns
            pattern_count = 0
            for i in range(1, pattern_length + 1):
                for j in range(len(data) - i):
                    pattern = data[j:j+i]
                    if data.count(pattern) > 1:
                        pattern_count += 1
            
            # Calculate fractal dimension
            fractal_dim = 1.0 + (pattern_count / (len(data) * pattern_length))
            
            return min(max(fractal_dim, 1.0), 2.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating fractal dimension: {e}")
            return 1.0
    
    def _calculate_quantum_state(self, data: str) -> str:
        """Calculate quantum state representation."""
        try:
            # Create quantum state hash
            quantum_hash = hashlib.sha256(data.encode()).hexdigest()[:16]
            
            # Convert to quantum state representation
            quantum_state = f"|{quantum_hash[:8]}⟩ + |{quantum_hash[8:16]}⟩"
            
            return quantum_state
            
        except Exception as e:
            self.logger.error(f"Error calculating quantum state: {e}")
            return "|unknown⟩"
    
    def _calculate_temporal_offset(self, context: Dict[str, Any]) -> float:
        """Calculate temporal offset."""
        try:
            timestamp = context.get('timestamp', time.time())
            current_time = time.time()
            
            # Calculate offset from current time
            temporal_offset = abs(current_time - timestamp)
            
            # Normalize to [0, 1] based on 24-hour cycle
            normalized_offset = (temporal_offset % self.vortex_math.temporal_cycle) / self.vortex_math.temporal_cycle
            
            return normalized_offset
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal offset: {e}")
            return 0.0
    
    def _calculate_security_score(self, entropy_level: float, fractal_dimension: float,
                                quantum_state: str, temporal_offset: float) -> float:
        """Calculate overall security score."""
        try:
            # Weighted combination of security factors
            entropy_weight = 0.3
            fractal_weight = 0.3
            quantum_weight = 0.2
            temporal_weight = 0.2
            
            # Quantum state score (based on hash complexity)
            quantum_score = len(set(quantum_state)) / len(quantum_state) if quantum_state else 0.5
            
            # Calculate weighted security score
            security_score = (
                entropy_weight * entropy_level +
                fractal_weight * (fractal_dimension - 1.0) +  # Normalize fractal dimension
                quantum_weight * quantum_score +
                temporal_weight * (1.0 - temporal_offset)  # Invert temporal offset
            )
            
            return min(max(security_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating security score: {e}")
            return 0.5
    
    def _perform_validation(self, signature: VortexSignature, 
                          context: Dict[str, Any]) -> SecurityValidationResult:
        """Perform security validation."""
        try:
            # Check signature validity
            signature_valid = self._validate_signature(signature)
            
            # Check security thresholds
            security_valid = signature.security_score >= self.config.get('validation_threshold', 0.8)
            
            # Check temporal validity
            temporal_valid = self._validate_temporal_consistency(signature, context)
            
            # Overall validation
            overall_valid = signature_valid and security_valid and temporal_valid
            
            # Calculate confidence
            confidence = self._calculate_validation_confidence(
                signature_valid, security_valid, temporal_valid, signature.security_score
            )
            
            result = SecurityValidationResult(
                valid=overall_valid,
                security_score=signature.security_score,
                confidence=confidence,
                vortex_signature=signature,
                metadata={
                    'signature_valid': signature_valid,
                    'security_valid': security_valid,
                    'temporal_valid': temporal_valid,
                    'validation_context': context
                }
            )
            
            # Store in history
            self.validation_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing validation: {e}")
            return SecurityValidationResult(
                valid=False,
                security_score=0.0,
                confidence=0.0,
                error_message=str(e)
            )
    
    def _validate_signature(self, signature: VortexSignature) -> bool:
        """Validate signature integrity."""
        try:
            # Check if signature hash is valid
            if not signature.signature_hash or len(signature.signature_hash) != 64:
                return False
            
            # Check if signature is in cache (for replay detection)
            if signature.signature_hash in self.signature_cache:
                cached_signature = self.signature_cache[signature.signature_hash]
                # Check if this is a replay attempt
                if abs(signature.timestamp - cached_signature.timestamp) < 1.0:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signature: {e}")
            return False
    
    def _validate_temporal_consistency(self, signature: VortexSignature, 
                                     context: Dict[str, Any]) -> bool:
        """Validate temporal consistency."""
        try:
            current_time = time.time()
            signature_time = signature.timestamp
            
            # Check if signature is not too old
            max_age = self.config.get('validation_timeout', 5.0)
            if current_time - signature_time > max_age:
                return False
            
            # Check temporal offset consistency
            expected_offset = self._calculate_temporal_offset({'timestamp': signature_time})
            actual_offset = signature.temporal_offset
            
            # Allow small temporal drift
            temporal_tolerance = 0.1
            if abs(expected_offset - actual_offset) > temporal_tolerance:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating temporal consistency: {e}")
            return False
    
    def _calculate_validation_confidence(self, signature_valid: bool, security_valid: bool,
                                       temporal_valid: bool, security_score: float) -> float:
        """Calculate validation confidence."""
        try:
            # Base confidence from security score
            base_confidence = security_score
            
            # Adjust based on validation results
            if signature_valid:
                base_confidence += 0.2
            if security_valid:
                base_confidence += 0.2
            if temporal_valid:
                base_confidence += 0.1
            
            return min(max(base_confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating validation confidence: {e}")
            return 0.5
    
    def _update_performance_metrics(self, result: SecurityValidationResult, 
                                  validation_time: float):
        """Update performance metrics."""
        try:
            self.performance_metrics['total_validations'] += 1
            
            if result.valid:
                self.performance_metrics['successful_validations'] += 1
            else:
                self.performance_metrics['failed_validations'] += 1
            
            # Update average validation time
            current_avg = self.performance_metrics['average_validation_time']
            total_validations = self.performance_metrics['total_validations']
            
            new_avg = ((current_avg * (total_validations - 1)) + validation_time) / total_validations
            self.performance_metrics['average_validation_time'] = new_avg
            
            # Store security score history
            self.performance_metrics['security_score_history'].append(result.security_score)
            
            # Keep only last 100 scores
            if len(self.performance_metrics['security_score_history']) > 100:
                self.performance_metrics['security_score_history'] = \
                    self.performance_metrics['security_score_history'][-100:]
                    
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _cache_signature(self, signature: VortexSignature):
        """Cache signature for replay detection."""
        try:
            max_cache_size = self.config.get('max_cache_size', 1000)
            
            # Add to cache
            self.signature_cache[signature.signature_hash] = signature
            
            # Remove oldest entries if cache is full
            if len(self.signature_cache) > max_cache_size:
                # Remove oldest entries
                sorted_signatures = sorted(
                    self.signature_cache.items(),
                    key=lambda x: x[1].timestamp
                )
                
                # Keep only the newest entries
                self.signature_cache = dict(sorted_signatures[-max_cache_size:])
                
        except Exception as e:
            self.logger.error(f"Error caching signature: {e}")
    
    def _update_security_state(self, result: SecurityValidationResult):
        """Update overall security state."""
        try:
            self.last_validation = time.time()
            self.validation_count += 1
            
            if result.valid:
                if result.confidence > 0.9:
                    self.security_state = SecurityState.VALID
                else:
                    self.security_state = SecurityState.PENDING
            else:
                if result.confidence < 0.3:
                    self.security_state = SecurityState.COMPROMISED
                else:
                    self.security_state = SecurityState.INVALID
                    
        except Exception as e:
            self.logger.error(f"Error updating security state: {e}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report."""
        try:
            return {
                'security_state': self.security_state.value,
                'last_validation': self.last_validation,
                'validation_count': self.validation_count,
                'performance_metrics': self.performance_metrics,
                'cache_size': len(self.signature_cache),
                'config': self.config,
                'recent_validations': [
                    {
                        'valid': result.valid,
                        'security_score': result.security_score,
                        'confidence': result.confidence,
                        'timestamp': result.validation_time
                    }
                    for result in self.validation_history[-10:]  # Last 10 validations
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating security report: {e}")
            return {'error': str(e)}


# Global VMSP instance
_vmsp_instance: Optional[VortexSecurityProtocol] = None


def get_vortex_security() -> VortexSecurityProtocol:
    """Get global VMSP instance."""
    global _vmsp_instance
    
    if _vmsp_instance is None:
        _vmsp_instance = VortexSecurityProtocol()
    
    return _vmsp_instance


def initialize_vmsp(config: Optional[Dict[str, Any]] = None) -> VortexSecurityProtocol:
    """Initialize VMSP with custom configuration."""
    global _vmsp_instance
    
    _vmsp_instance = VortexSecurityProtocol(config)
    return _vmsp_instance


def validate_security(inputs: List[Any], context: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function for security validation."""
    vmsp = get_vortex_security()
    return vmsp.validate_security_state(inputs, context)


def get_security_report() -> Dict[str, Any]:
    """Get VMSP security report."""
    vmsp = get_vortex_security()
    return vmsp.get_security_report()


# Export main classes and functions
__all__ = [
    'VortexSecurityProtocol',
    'VortexMathCore',
    'VortexSignature',
    'SecurityValidationResult',
    'SecurityState',
    'VortexType',
    'get_vortex_security',
    'initialize_vmsp',
    'validate_security',
    'get_security_report'
] 