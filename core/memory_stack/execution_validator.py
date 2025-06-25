# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Execution Validator - Ghost-Based Logic Path Validation
======================================================

Checks execution conforms to ghost-based logic paths.
Provides validation for trade execution against expected ghost patterns.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from core.unified_math_system import unified_math
import hashlib
import numpy as np
from numpy.typing import NDArray

# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        WindowsCliCompatibilityHandler,
        safe_print,
        safe_format_error,
        log_safe,
        cli_handler,
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)
    cli_handler = None

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Enumeration of validation statuses."""
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    REJECTED = "rejected"
    PENDING = "pending"
    FAILED = "failed"


class DriftLevel(Enum):
    """Enumeration of drift levels."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class CostType(Enum):
    """Enumeration of cost types."""
    BASE = "base"
    COMPLEXITY = "complexity"
    MARKET_IMPACT = "market_impact"
    NETWORK = "network"
    COMPUTATIONAL = "computational"


@dataclass
class ExecutionCost:
    """Execution cost structure."""
    cost_id: str
    command_id: str
    base_cost: float
    complexity_cost: float
    market_impact_cost: float
    network_cost: float
    computational_cost: float
    total_cost: float
    cost_efficiency: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.metadata:
            self.metadata = {}


@dataclass
class DriftValidation:
    """Drift validation structure."""
    validation_id: str
    command_id: str
    expected_time: datetime
    actual_time: datetime
    drift_magnitude: float
    drift_level: DriftLevel
    drift_factor: float
    validation_score: float
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.metadata:
            self.metadata = {}


@dataclass
class ExecutionValidation:
    """Represents execution validation result."""
    validation_id: str
    trade_data: Dict[str, Any]
    expected_hash: str
    actual_hash: str
    validation_score: float
    is_valid: bool
    drift_magnitude: float
    confidence_level: float
    timestamp: datetime
    validation_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GhostLogicPath:
    """Represents a ghost logic path for validation."""
    path_id: str
    expected_sequence: List[str]
    hash_signature: str
    confidence_threshold: float
    drift_tolerance: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionValidator:
    """
    Execution Validator for Ghost-Based Logic Paths.
    
    This validator ensures that trade executions conform to expected
    ghost-based logic paths and detects deviations from expected patterns.
    """
    
    def __init__(self, validation_file: str = "memory_stack/execution_validations.json"):
        """Initialize the execution validator."""
        self.validation_file = validation_file
        self.logger = logging.getLogger("execution_validator")
        self.logger.setLevel(logging.INFO)
        
        # Validation storage
        self.execution_costs: Dict[str, ExecutionCost] = {}
        self.drift_validations: Dict[str, DriftValidation] = {}
        self.execution_validations: Dict[str, ExecutionValidation] = {}
        
        # Configuration parameters
        self.base_cost_threshold = 10.0
        self.complexity_factor = 0.1
        self.market_impact_factor = 0.05
        self.network_cost_factor = 0.02
        self.computational_cost_factor = 0.03
        
        # Drift thresholds
        self.drift_thresholds = {
            DriftLevel.NONE: 0.0,
            DriftLevel.MINOR: 1.0,
            DriftLevel.MODERATE: 3.0,
            DriftLevel.MAJOR: 5.0,
            DriftLevel.CRITICAL: 10.0
        }
        
        # Validation thresholds
        self.approval_threshold = 0.7
        self.conditional_threshold = 0.5
        self.rejection_threshold = 0.3
        
        # Performance tracking
        self.total_validations = 0
        self.successful_validations = 0
        self.validation_success_rate = 0.0
        
        # Validation parameters
        self.default_confidence_threshold = 0.7
        self.default_drift_tolerance = 0.3
        self.hash_similarity_threshold = 0.8
        self.sequence_match_threshold = 0.6
        
        # CLI compatibility
        self.cli_handler = WindowsCliCompatibilityHandler()
        
        # Load existing validations
        self._load_validations()
        
        safe_safe_print("✅ Execution Validator initialized - Cost simulation active")
    
    def _load_validations(self) -> None:
        """Load existing validations from file."""
        try:
            if os.path.exists(self.validation_file):
                with open(self.validation_file, 'r') as f:
                    validation_data = json.load(f)
                
                # Load execution costs
                for cost_data in validation_data.get('execution_costs', []):
                    execution_cost = ExecutionCost(
                        cost_id=cost_data['cost_id'],
                        command_id=cost_data['command_id'],
                        base_cost=cost_data['base_cost'],
                        complexity_cost=cost_data['complexity_cost'],
                        market_impact_cost=cost_data['market_impact_cost'],
                        network_cost=cost_data['network_cost'],
                        computational_cost=cost_data['computational_cost'],
                        total_cost=cost_data['total_cost'],
                        cost_efficiency=cost_data['cost_efficiency'],
                        timestamp=datetime.fromisoformat(cost_data['timestamp']),
                        metadata=cost_data.get('metadata', {})
                    )
                    self.execution_costs[execution_cost.cost_id] = execution_cost
                
                # Load drift validations
                for drift_data in validation_data.get('drift_validations', []):
                    drift_validation = DriftValidation(
                        validation_id=drift_data['validation_id'],
                        command_id=drift_data['command_id'],
                        expected_time=datetime.fromisoformat(drift_data['expected_time']),
                        actual_time=datetime.fromisoformat(drift_data['actual_time']),
                        drift_magnitude=drift_data['drift_magnitude'],
                        drift_level=DriftLevel(drift_data['drift_level']),
                        drift_factor=drift_data['drift_factor'],
                        validation_score=drift_data['validation_score'],
                        recommendations=drift_data.get('recommendations', []),
                        metadata=drift_data.get('metadata', {})
                    )
                    self.drift_validations[drift_validation.validation_id] = drift_validation
                
                # Load execution validations
                for exec_data in validation_data.get('execution_validations', []):
                    execution_validation = ExecutionValidation(
                        validation_id=exec_data['validation_id'],
                        trade_data=exec_data['trade_data'],
                        expected_hash=exec_data['expected_hash'],
                        actual_hash=exec_data['actual_hash'],
                        validation_score=exec_data['validation_score'],
                        is_valid=exec_data['is_valid'],
                        drift_magnitude=exec_data['drift_magnitude'],
                        confidence_level=exec_data['confidence_level'],
                        timestamp=datetime.fromisoformat(exec_data['timestamp']),
                        validation_details=exec_data.get('validation_details', {}),
                        metadata=exec_data.get('metadata', {})
                    )
                    self.execution_validations[execution_validation.validation_id] = execution_validation
                
                safe_safe_print(f"✅ Loaded {len(self.execution_costs)} costs, {len(self.drift_validations)} drift validations, {len(self.execution_validations)} execution validations")
                
        except Exception as e:
            error_msg = safe_format_error(e, "load_validations")
            safe_safe_print(f"⚠️ Failed to load validations: {error_msg}")
    
    def _save_validations(self) -> None:
        """Save validations to file."""
        try:
            os.makedirs(os.path.dirname(self.validation_file), exist_ok=True)
            
            validation_data = {
                'execution_costs': [],
                'drift_validations': [],
                'execution_validations': [],
                'last_updated': datetime.now().isoformat(),
                'total_costs': len(self.execution_costs),
                'total_drift_validations': len(self.drift_validations),
                'total_execution_validations': len(self.execution_validations)
            }
            
            # Save execution costs
            for cost in self.execution_costs.values():
                cost_data = asdict(cost)
                cost_data['timestamp'] = cost.timestamp.isoformat()
                validation_data['execution_costs'].append(cost_data)
            
            # Save drift validations
            for drift in self.drift_validations.values():
                drift_data = asdict(drift)
                drift_data['expected_time'] = drift.expected_time.isoformat()
                drift_data['actual_time'] = drift.actual_time.isoformat()
                drift_data['drift_level'] = drift.drift_level.value
                validation_data['drift_validations'].append(drift_data)
            
            # Save execution validations
            for validation in self.execution_validations.values():
                validation_data = asdict(validation)
                validation_data['timestamp'] = validation.timestamp.isoformat()
                validation_data['validation_status'] = ValidationStatus.APPROVED.value if validation.is_valid else ValidationStatus.REJECTED.value
                validation_data['execution_cost_id'] = validation.execution_cost.cost_id if validation.execution_cost else None
                validation_data['drift_validation_id'] = validation.drift_validation.validation_id if validation.drift_validation else None
                validation_data['execution_validations'].append(validation_data)
            
            with open(self.validation_file, 'w') as f:
                json.dump(validation_data, f, indent=2)
                
        except Exception as e:
            error_msg = safe_format_error(e, "save_validations")
            safe_safe_print(f"⚠️ Failed to save validations: {error_msg}")
    
    def simulate_execution_cost(self, trade: Dict[str, Any]) -> float:
        """
        Simulate execution cost for trade validation.
        
        Args:
            trade: Trade execution data
            
        Returns:
            Simulated execution cost
        """
        try:
            # Base cost
            base_cost = 0.001  # 0.1% base cost
            
            # Volume-based cost adjustment
            quantity = trade.get('quantity', 0.0)
            if quantity > 1000:
                volume_factor = 1.2  # Higher cost for large volumes
            elif quantity > 100:
                volume_factor = 1.0  # Standard cost
            else:
                volume_factor = 0.8  # Lower cost for small volumes
            
            # Market condition adjustment
            market_data = trade.get('market_data', {})
            volatility = market_data.get('volatility', 0.0)
            
            if volatility > 0.3:
                volatility_factor = 1.3  # Higher cost in volatile markets
            elif volatility > 0.1:
                volatility_factor = 1.0  # Standard cost
            else:
                volatility_factor = 0.9  # Lower cost in stable markets
            
            # Calculate total cost
            total_cost = base_cost * volume_factor * volatility_factor
            
            return float(total_cost)
            
        except Exception:
            return 0.001
    
    def validate(self, trade: Dict[str, Any], expected_hash: str) -> bool:
        """
        Validate trade execution against expected hash.
        
        Args:
            trade: Trade execution data
            expected_hash: Expected hash for validation
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            start_time = time.time()
            
            # Generate actual hash from trade data
            actual_hash = self._generate_trade_hash(trade)
            
            # Calculate validation metrics
            validation_score = self._calculate_validation_score(trade, expected_hash, actual_hash)
            drift_magnitude = self._calculate_drift_magnitude(expected_hash, actual_hash)
            confidence_level = self._calculate_confidence_level(trade, validation_score)
            
            # Determine if validation passes
            is_valid = (
                validation_score >= self.default_confidence_threshold and
                drift_magnitude <= self.default_drift_tolerance
            )
            
            # Create validation record
            validation = ExecutionValidation(
                validation_id=self._generate_validation_id(trade),
                trade_data=trade,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
                validation_score=validation_score,
                is_valid=is_valid,
                drift_magnitude=drift_magnitude,
                confidence_level=confidence_level,
                timestamp=datetime.now(),
                validation_details={
                    'score_components': self._get_score_components(trade, expected_hash, actual_hash),
                    'drift_analysis': self._analyze_drift(expected_hash, actual_hash),
                    'confidence_factors': self._get_confidence_factors(trade, validation_score)
                }
            )
            
            self.execution_validations[validation.validation_id] = validation
            
            # Update performance metrics
            self.total_validations += 1
            if is_valid:
                self.successful_validations += 1
            self.validation_success_rate = self.successful_validations / self.total_validations
            
            execution_time = time.time() - start_time
            logger.info(f"Validation completed in {execution_time:.3f}s - Valid: {is_valid}, Score: {validation_score:.3f}")
            
            return is_valid
            
        except Exception as e:
            error_msg = safe_format_error(e, "ExecutionValidator.validate")
            logger.error(error_msg)
            return False
    
    def _generate_trade_hash(self, trade: Dict[str, Any]) -> str:
        """
        Generate hash from trade data.
        
        Args:
            trade: Trade execution data
            
        Returns:
            Hash string
        """
        try:
            # Extract key trade parameters
            trade_params = {
                'price': trade.get('price', 0.0),
                'quantity': trade.get('quantity', 0.0),
                'side': trade.get('side', 'unknown'),
                'timestamp': trade.get('timestamp', ''),
                'symbol': trade.get('symbol', ''),
                'order_type': trade.get('order_type', 'market')
            }
            
            # Create hash input string
            hash_input = f"{trade_params['price']}_{trade_params['quantity']}_{trade_params['side']}_{trade_params['symbol']}_{trade_params['order_type']}"
            
            # Generate hash
            hash_result = hashlib.sha256(hash_input.encode()).hexdigest()
            return hash_result[:16]  # Return first 16 characters
            
        except Exception as e:
            logger.error(f"Trade hash generation failed: {e}")
            return "0000000000000000"
    
    def _calculate_validation_score(self, trade: Dict[str, Any], expected_hash: str, actual_hash: str) -> float:
        """
        Calculate validation score based on multiple factors.
        
        Args:
            trade: Trade execution data
            expected_hash: Expected hash
            actual_hash: Actual hash
            
        Returns:
            Validation score (0.0 to 1.0)
        """
        try:
            scores = []
            
            # Hash similarity score
            hash_similarity = self._calculate_hash_similarity(expected_hash, actual_hash)
            scores.append(hash_similarity * 0.4)  # 40% weight
            
            # Trade parameter consistency score
            param_consistency = self._calculate_parameter_consistency(trade)
            scores.append(param_consistency * 0.3)  # 30% weight
            
            # Timing consistency score
            timing_consistency = self._calculate_timing_consistency(trade)
            scores.append(timing_consistency * 0.2)  # 20% weight
            
            # Market condition alignment score
            market_alignment = self._calculate_market_alignment(trade)
            scores.append(market_alignment * 0.1)  # 10% weight
            
            # Calculate weighted average
            total_score = sum(scores)
            return float(total_score)
            
        except Exception as e:
            logger.error(f"Validation score calculation failed: {e}")
            return 0.5
    
    def _calculate_hash_similarity(self, expected_hash: str, actual_hash: str) -> float:
        """Calculate similarity between expected and actual hashes."""
        try:
            if len(expected_hash) != len(actual_hash):
                return 0.0
            
            # Calculate Hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(expected_hash, actual_hash))
            max_distance = len(expected_hash)
            
            # Convert to similarity score
            similarity = 1.0 - (distance / max_distance)
            return float(similarity)
            
        except Exception:
            return 0.0
    
    def _calculate_parameter_consistency(self, trade: Dict[str, Any]) -> float:
        """Calculate consistency of trade parameters."""
        try:
            consistency_scores = []
            
            # Price consistency
            price = trade.get('price', 0.0)
            if price > 0:
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.0)
            
            # Quantity consistency
            quantity = trade.get('quantity', 0.0)
            if quantity > 0:
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.0)
            
            # Side consistency
            side = trade.get('side', '').lower()
            if side in ['buy', 'sell']:
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.0)
            
            # Symbol consistency
            symbol = trade.get('symbol', '')
            if symbol and len(symbol) > 0:
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.0)
            
            return float(np.mean(consistency_scores)) if consistency_scores else 0.0
            
        except Exception:
            return 0.5
    
    def _calculate_timing_consistency(self, trade: Dict[str, Any]) -> float:
        """Calculate timing consistency of trade execution."""
        try:
            # Extract timing information
            timestamp = trade.get('timestamp', '')
            execution_time = trade.get('execution_time', 0.0)
            
            # Basic timing validation
            if timestamp and execution_time > 0:
                # Check if execution time is reasonable (less than 5 seconds)
                if execution_time < 5.0:
                    return 1.0
                elif execution_time < 10.0:
                    return 0.8
                elif execution_time < 30.0:
                    return 0.6
                else:
                    return 0.3
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_market_alignment(self, trade: Dict[str, Any]) -> float:
        """Calculate alignment with current market conditions."""
        try:
            # This would typically compare against current market data
            # For now, return a default score
            market_data = trade.get('market_data', {})
            
            if market_data:
                # Check if trade aligns with market volatility
                volatility = market_data.get('volatility', 0.0)
                price = trade.get('price', 0.0)
                
                if volatility > 0 and price > 0:
                    # Simple alignment check
                    return 0.8
                else:
                    return 0.6
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_drift_magnitude(self, expected_hash: str, actual_hash: str) -> float:
        """
        Calculate drift magnitude between expected and actual hashes.
        
        Args:
            expected_hash: Expected hash
            actual_hash: Actual hash
            
        Returns:
            Drift magnitude (0.0 to 1.0)
        """
        try:
            # Use hash similarity to calculate drift
            similarity = self._calculate_hash_similarity(expected_hash, actual_hash)
            drift = 1.0 - similarity
            
            return float(drift)
            
        except Exception:
            return 0.5
    
    def _calculate_confidence_level(self, trade: Dict[str, Any], validation_score: float) -> float:
        """
        Calculate confidence level for validation result.
        
        Args:
            trade: Trade execution data
            validation_score: Validation score
            
        Returns:
            Confidence level (0.0 to 1.0)
        """
        try:
            confidence_factors = []
            
            # Base confidence from validation score
            confidence_factors.append(validation_score)
            
            # Additional confidence from trade quality
            trade_quality = self._assess_trade_quality(trade)
            confidence_factors.append(trade_quality)
            
            # Market condition confidence
            market_confidence = self._assess_market_confidence(trade)
            confidence_factors.append(market_confidence)
            
            # Calculate weighted confidence
            weights = [0.5, 0.3, 0.2]  # Validation score, trade quality, market confidence
            confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
            
            return float(confidence)
            
        except Exception:
            return validation_score
    
    def _assess_trade_quality(self, trade: Dict[str, Any]) -> float:
        """Assess overall trade quality."""
        try:
            quality_scores = []
            
            # Price quality
            price = trade.get('price', 0.0)
            if price > 0:
                quality_scores.append(1.0)
            else:
                quality_scores.append(0.0)
            
            # Quantity quality
            quantity = trade.get('quantity', 0.0)
            if quantity > 0:
                quality_scores.append(1.0)
            else:
                quality_scores.append(0.0)
            
            # Execution quality
            execution_time = trade.get('execution_time', 0.0)
            if execution_time > 0 and execution_time < 5.0:
                quality_scores.append(1.0)
            elif execution_time > 0 and execution_time < 10.0:
                quality_scores.append(0.8)
            else:
                quality_scores.append(0.5)
            
            return float(np.mean(quality_scores)) if quality_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _assess_market_confidence(self, trade: Dict[str, Any]) -> float:
        """Assess confidence based on market conditions."""
        try:
            market_data = trade.get('market_data', {})
            
            if not market_data:
                return 0.5
            
            # Check market volatility
            volatility = market_data.get('volatility', 0.0)
            if volatility < 0.1:
                return 0.9  # Low volatility - high confidence
            elif volatility < 0.3:
                return 0.7  # Medium volatility - medium confidence
            else:
                return 0.5  # High volatility - lower confidence
                
        except Exception:
            return 0.5
    
    def _get_score_components(self, trade: Dict[str, Any], expected_hash: str, actual_hash: str) -> Dict[str, float]:
        """Get individual score components for detailed analysis."""
        try:
            return {
                'hash_similarity': self._calculate_hash_similarity(expected_hash, actual_hash),
                'parameter_consistency': self._calculate_parameter_consistency(trade),
                'timing_consistency': self._calculate_timing_consistency(trade),
                'market_alignment': self._calculate_market_alignment(trade)
            }
        except Exception:
            return {
                'hash_similarity': 0.0,
                'parameter_consistency': 0.0,
                'timing_consistency': 0.0,
                'market_alignment': 0.0
            }
    
    def _analyze_drift(self, expected_hash: str, actual_hash: str) -> Dict[str, Any]:
        """Analyze drift between expected and actual hashes."""
        try:
            drift_magnitude = self._calculate_drift_magnitude(expected_hash, actual_hash)
            
            return {
                'magnitude': drift_magnitude,
                'severity': 'low' if drift_magnitude < 0.2 else 'medium' if drift_magnitude < 0.5 else 'high',
                'tolerance_exceeded': drift_magnitude > self.default_drift_tolerance
            }
        except Exception:
            return {
                'magnitude': 0.5,
                'severity': 'medium',
                'tolerance_exceeded': True
            }
    
    def _get_confidence_factors(self, trade: Dict[str, Any], validation_score: float) -> Dict[str, float]:
        """Get confidence factors for validation result."""
        try:
            return {
                'validation_score': validation_score,
                'trade_quality': self._assess_trade_quality(trade),
                'market_confidence': self._assess_market_confidence(trade)
            }
        except Exception:
            return {
                'validation_score': validation_score,
                'trade_quality': 0.5,
                'market_confidence': 0.5
            }
    
    def _generate_validation_id(self, trade: Dict[str, Any]) -> str:
        """Generate unique validation ID."""
        try:
            timestamp = datetime.now().isoformat()
            trade_id = trade.get('trade_id', 'unknown')
            return f"val_{timestamp}_{trade_id}"
        except Exception:
            return f"val_{int(time.time())}"
    
    def validate_drift(self, expected_sequence: List[str], actual_sequence: List[str]) -> bool:
        """
        Validate drift in execution sequence.
        
        Args:
            expected_sequence: Expected execution sequence
            actual_sequence: Actual execution sequence
            
        Returns:
            True if drift is within tolerance
        """
        try:
            if len(expected_sequence) != len(actual_sequence):
                return False
            
            # Calculate sequence similarity
            matches = sum(1 for exp, act in zip(expected_sequence, actual_sequence) if exp == act)
            similarity = matches / len(expected_sequence)
            
            return similarity >= self.sequence_match_threshold
            
        except Exception:
            return False
    
    def validate_execution(self, trade: Dict[str, Any], expected_pattern: Dict[str, Any]) -> bool:
        """
        Validate execution against expected pattern.
        
        Args:
            trade: Trade execution data
            expected_pattern: Expected execution pattern
            
        Returns:
            True if execution matches pattern
        """
        try:
            # Check price pattern
            expected_price = expected_pattern.get('price', 0.0)
            actual_price = trade.get('price', 0.0)
            price_match = abs(actual_price - expected_price) / max(expected_price, 1e-8) < 0.05  # 5% tolerance
            
            # Check timing pattern
            expected_time = expected_pattern.get('execution_time', 0.0)
            actual_time = trade.get('execution_time', 0.0)
            time_match = abs(actual_time - expected_time) < 2.0  # 2 second tolerance
            
            # Check quantity pattern
            expected_quantity = expected_pattern.get('quantity', 0.0)
            actual_quantity = trade.get('quantity', 0.0)
            quantity_match = abs(actual_quantity - expected_quantity) / max(expected_quantity, 1e-8) < 0.1  # 10% tolerance
            
            return price_match and time_match and quantity_match
            
        except Exception:
            return False
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation execution statistics."""
        try:
            return {
                "total_validations": self.total_validations,
                "successful_validations": self.successful_validations,
                "success_rate": self.validation_success_rate,
                "average_validation_score": np.mean([v.validation_score for v in self.execution_validations.values()]) if self.execution_validations else 0.0,
                "average_drift_magnitude": np.mean([v.drift_magnitude for v in self.execution_validations.values()]) if self.execution_validations else 0.0,
                "average_confidence_level": np.mean([v.confidence_level for v in self.execution_validations.values()]) if self.execution_validations else 0.0
            }
        except Exception:
            return {
                "total_validations": 0,
                "successful_validations": 0,
                "success_rate": 0.0,
                "average_validation_score": 0.0,
                "average_drift_magnitude": 0.0,
                "average_confidence_level": 0.0
            }


# Global instance for easy access
execution_validator = ExecutionValidator()


# Convenience functions for external access
def validate_execution(trade: Dict[str, Any], expected_hash: str) -> bool:
    """Convenience function to validate trade execution."""
    return execution_validator.validate(trade, expected_hash)


def simulate_execution_cost(trade: Dict[str, Any]) -> float:
    """Convenience function to simulate execution cost."""
    return execution_validator.simulate_execution_cost(trade)


if __name__ == "__main__":
    # Test the execution validator
    test_trades = [
        {
            'trade_id': 'test_001',
            'price': 50000.0,
            'quantity': 0.1,
            'side': 'buy',
            'symbol': 'BTC/USD',
            'order_type': 'market',
            'execution_time': 0.5,
            'timestamp': '2024-01-01T12:00:00Z',
            'market_data': {'volatility': 0.15}
        },
        {
            'trade_id': 'test_002',
            'price': 51000.0,
            'quantity': 0.05,
            'side': 'sell',
            'symbol': 'BTC/USD',
            'order_type': 'limit',
            'execution_time': 1.2,
            'timestamp': '2024-01-01T12:05:00Z',
            'market_data': {'volatility': 0.25}
        }
    ]
    
    validator = ExecutionValidator()
    
    for trade in test_trades:
        safe_print(f"\nTesting trade: {trade['trade_id']}")
        
        # Generate expected hash
        expected_hash = "a1b2c3d4e5f67890"
        
        # Validate execution
        is_valid = validator.validate(trade, expected_hash)
        safe_print(f"Validation result: {is_valid}")
        
        # Simulate execution cost
        cost = validator.simulate_execution_cost(trade)
        safe_print(f"Execution cost: {cost:.4f}")
    
    # Print statistics
    stats = validator.get_validation_statistics()
    safe_print(f"\nValidator Statistics: {stats}") 