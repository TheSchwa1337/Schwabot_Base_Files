#!/usr/bin/env python3
"""
Route Verification Classifier - Schwabot Mathematical Framework
==============================================================

Implements AI-powered route verification and classification system that validates
and potentially overrides allocator decisions based on probabilistic analysis,
pattern recognition, and learned trading behaviors.

Key capabilities:
- Route classification (optimal, volatile, decaying, trap)
- Override authority for allocator decisions  
- Probabilistic route validation
- Pattern-based risk assessment
- Self-learning feedback integration

Based on SxN-Math specifications and hybrid allocator-classifier architecture.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from decimal import Decimal, getcontext
from dataclasses import dataclass
from enum import Enum
import logging
import hashlib
from datetime import datetime
import json

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class RouteClassification(Enum):
    """Classification types for trading routes"""
    OPTIMAL = "optimal"
    VOLATILE = "volatile"  
    DECAYING = "decaying"
    TRAP = "trap"
    UNKNOWN = "unknown"


@dataclass
class RouteVector:
    """Comprehensive route vector for classification"""
    route_id: str
    asset_pair: str
    entry_price: Decimal
    exit_price: Decimal
    volume: Decimal
    thermal_index: Decimal
    timestamp: datetime
    efficiency_ratio: float
    profit: Decimal
    
    # Additional features for classification
    volatility: float = 0.0
    trend_strength: float = 0.0
    volume_profile: float = 0.0
    market_momentum: float = 0.0
    liquidity_depth: float = 0.0


@dataclass
class ClassificationResult:
    """Result of route classification"""
    route_id: str
    classification: RouteClassification
    confidence: float
    override_decision: bool
    reason: str
    alternative_route: Optional[str] = None
    risk_score: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RouteFeatureExtractor:
    """Extracts features from route vectors for classification"""
    
    def __init__(self):
        self.feature_weights = {
            'efficiency_ratio': 0.25,
            'profit_magnitude': 0.20,
            'volatility_risk': 0.15,
            'thermal_cost': 0.15,
            'trend_alignment': 0.15,
            'liquidity_quality': 0.10
        }
        logger.info("Route feature extractor initialized")
    
    def extract_features(self, route: RouteVector) -> Vector:
        """
        Extract numerical features from route vector
        
        Args:
            route: Route vector to analyze
            
        Returns:
            Feature vector for classification
        """
        try:
            # Normalize profit by volume
            profit_per_unit = float(route.profit / (route.volume + Decimal('1e-10')))
            
            # Price movement magnitude
            price_change = float(abs(route.exit_price - route.entry_price) / route.entry_price)
            
            # Thermal efficiency (inverse of thermal cost)
            thermal_efficiency = 1.0 / (float(route.thermal_index) + 1e-6)
            
            # Risk-adjusted return approximation
            risk_adjusted_return = profit_per_unit / (route.volatility + 1e-6)
            
            # Volume quality indicator
            volume_quality = min(1.0, float(route.volume) / 10.0)  # Normalize volume
            
            # Trend alignment score
            trend_score = route.trend_strength * np.sign(profit_per_unit)
            
            features = np.array([
                route.efficiency_ratio,          # Direct efficiency metric
                profit_per_unit,                 # Profit magnitude
                price_change,                    # Price volatility proxy
                thermal_efficiency,              # Cost efficiency
                risk_adjusted_return,            # Risk-adjusted performance
                volume_quality,                  # Volume liquidity
                trend_score,                     # Trend alignment
                route.liquidity_depth,          # Market depth
                route.market_momentum,           # Market momentum
                float(route.thermal_index)      # Raw thermal cost
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(10)  # Return default feature vector
    
    def compute_risk_score(self, route: RouteVector) -> float:
        """
        Compute risk score for route
        
        Args:
            route: Route to analyze
            
        Returns:
            Risk score (0 = low risk, 1 = high risk)
        """
        try:
            # Volatility component
            vol_risk = min(1.0, route.volatility / 0.5)  # Normalize to 50% volatility
            
            # Thermal cost component  
            thermal_risk = min(1.0, float(route.thermal_index) / 5.0)  # Normalize to thermal=5
            
            # Liquidity risk (inverse of liquidity depth)
            liquidity_risk = max(0.0, 1.0 - route.liquidity_depth)
            
            # Concentration risk (for large single trades)
            volume_risk = min(1.0, float(route.volume) / 100.0)  # Risk increases with volume
            
            # Composite risk score
            risk_score = 0.3 * vol_risk + 0.25 * thermal_risk + 0.25 * liquidity_risk + 0.2 * volume_risk
            
            return np.clip(risk_score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Risk score computation failed: {e}")
            return 0.5  # Default medium risk


class RouteClassifier:
    """
    AI-powered route classifier using pattern recognition and probabilistic analysis
    """
    
    def __init__(self):
        self.feature_extractor = RouteFeatureExtractor()
        self.classification_history: List[ClassificationResult] = []
        self.route_memory: Dict[str, List[RouteVector]] = {}
        self.classification_thresholds = {
            'optimal_efficiency': 0.7,
            'trap_risk': 0.8,
            'volatility_limit': 0.4,
            'minimum_confidence': 0.6
        }
        
        # Simple learned weights (would be ML model in production)
        self.learned_weights = np.array([
            0.3,   # efficiency_ratio weight
            0.25,  # profit_per_unit weight  
            -0.2,  # price_change weight (negative = penalize volatility)
            0.15,  # thermal_efficiency weight
            0.2,   # risk_adjusted_return weight
            0.1,   # volume_quality weight
            0.15,  # trend_score weight
            0.1,   # liquidity_depth weight
            0.05,  # market_momentum weight
            -0.1   # thermal_index weight (negative = penalize cost)
        ])
        
        logger.info("Route classifier initialized")
    
    def classify_route(self, route: RouteVector) -> ClassificationResult:
        """
        Classify a trading route and determine if override is needed
        
        Args:
            route: Route vector to classify
            
        Returns:
            Classification result with override decision
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(route)
            
            # Compute classification scores
            classification_scores = self._compute_classification_scores(features, route)
            
            # Determine primary classification
            primary_class = max(classification_scores.keys(), 
                              key=lambda k: classification_scores[k])
            
            # Calculate confidence
            confidence = classification_scores[primary_class]
            
            # Compute risk score
            risk_score = self.feature_extractor.compute_risk_score(route)
            
            # Determine if override is needed
            override_decision, reason, alternative = self._should_override(
                primary_class, confidence, risk_score, route
            )
            
            # Create result
            result = ClassificationResult(
                route_id=route.route_id,
                classification=RouteClassification(primary_class),
                confidence=confidence,
                override_decision=override_decision,
                reason=reason,
                alternative_route=alternative,
                risk_score=risk_score
            )
            
            # Store in history for learning
            self.classification_history.append(result)
            self._update_route_memory(route)
            
            logger.debug(f"Route {route.route_id} classified as {primary_class} "
                        f"(confidence: {confidence:.3f}, override: {override_decision})")
            
            return result
            
        except Exception as e:
            logger.error(f"Route classification failed: {e}")
            return ClassificationResult(
                route_id=route.route_id,
                classification=RouteClassification.UNKNOWN,
                confidence=0.0,
                override_decision=True,
                reason=f"Classification error: {str(e)}",
                risk_score=1.0
            )
    
    def _compute_classification_scores(self, features: Vector, route: RouteVector) -> Dict[str, float]:
        """Compute classification scores for each route type"""
        try:
            # Simple linear classifier (would be replaced with trained ML model)
            base_score = np.dot(features, self.learned_weights)
            
            # Normalize to probability-like scores
            base_prob = 1.0 / (1.0 + np.exp(-base_score))  # Sigmoid
            
            scores = {}
            
            # OPTIMAL: High efficiency, good profit, low risk
            optimal_score = base_prob
            if route.efficiency_ratio > self.classification_thresholds['optimal_efficiency']:
                optimal_score *= 1.2
            if route.volatility < self.classification_thresholds['volatility_limit']:
                optimal_score *= 1.1
            scores['optimal'] = min(1.0, optimal_score)
            
            # VOLATILE: High volatility, unpredictable patterns
            volatile_score = route.volatility * 2.0
            if route.trend_strength < 0.3:  # Weak trend = more volatile
                volatile_score *= 1.3
            scores['volatile'] = min(1.0, volatile_score)
            
            # DECAYING: Decreasing efficiency over time
            decay_score = 0.5  # Default
            if len(self.route_memory.get(route.asset_pair, [])) > 3:
                recent_routes = self.route_memory[route.asset_pair][-3:]
                if all(r.efficiency_ratio < route.efficiency_ratio for r in recent_routes):
                    decay_score = 0.8
            scores['decaying'] = decay_score
            
            # TRAP: High risk indicators, potential for loss
            trap_score = 0.0
            if route.efficiency_ratio < 0:  # Negative efficiency
                trap_score += 0.4
            if route.volatility > self.classification_thresholds['volatility_limit']:
                trap_score += 0.3
            if float(route.thermal_index) > 3.0:  # High cost
                trap_score += 0.2
            if route.liquidity_depth < 0.3:  # Low liquidity
                trap_score += 0.1
            scores['trap'] = min(1.0, trap_score)
            
            # Normalize scores to sum to 1
            total_score = sum(scores.values())
            if total_score > 0:
                scores = {k: v / total_score for k, v in scores.items()}
            
            return scores
            
        except Exception as e:
            logger.error(f"Classification score computation failed: {e}")
            return {'unknown': 1.0}
    
    def _should_override(self, classification: str, confidence: float, 
                        risk_score: float, route: RouteVector) -> Tuple[bool, str, Optional[str]]:
        """
        Determine if allocator decision should be overridden
        
        Returns:
            (should_override, reason, alternative_route_id)
        """
        try:
            # Override conditions
            
            # 1. Trap classification with high confidence
            if classification == 'trap' and confidence > 0.7:
                return True, "Route classified as trap with high confidence", None
            
            # 2. High risk score regardless of classification
            if risk_score > self.classification_thresholds['trap_risk']:
                return True, f"Risk score too high: {risk_score:.3f}", None
            
            # 3. Low confidence in any classification
            if confidence < self.classification_thresholds['minimum_confidence']:
                return True, f"Classification confidence too low: {confidence:.3f}", None
            
            # 4. Volatile classification with poor market conditions
            if (classification == 'volatile' and 
                route.market_momentum < -0.3 and 
                route.liquidity_depth < 0.4):
                return True, "Volatile route in poor market conditions", None
                
            # 5. Decaying route with recent poor performance
            if classification == 'decaying' and route.efficiency_ratio < 0.2:
                return True, "Decaying route with poor efficiency", None
            
            # No override needed
            return False, f"Route approved: {classification} classification", None
            
        except Exception as e:
            logger.error(f"Override decision failed: {e}")
            return True, f"Override due to error: {str(e)}", None
    
    def _update_route_memory(self, route: RouteVector) -> None:
        """Update route memory for pattern learning"""
        try:
            if route.asset_pair not in self.route_memory:
                self.route_memory[route.asset_pair] = []
            
            self.route_memory[route.asset_pair].append(route)
            
            # Keep only recent history (last 50 routes per pair)
            if len(self.route_memory[route.asset_pair]) > 50:
                self.route_memory[route.asset_pair] = self.route_memory[route.asset_pair][-50:]
                
        except Exception as e:
            logger.error(f"Route memory update failed: {e}")
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about recent classifications"""
        try:
            if not self.classification_history:
                return {"message": "No classification history available"}
            
            recent_history = self.classification_history[-100:]  # Last 100 classifications
            
            # Count classifications
            class_counts = {}
            override_count = 0
            total_confidence = 0.0
            
            for result in recent_history:
                class_name = result.classification.value
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                if result.override_decision:
                    override_count += 1
                total_confidence += result.confidence
            
            # Calculate stats
            override_rate = override_count / len(recent_history)
            avg_confidence = total_confidence / len(recent_history)
            
            return {
                "total_classifications": len(recent_history),
                "classification_distribution": class_counts,
                "override_rate": override_rate,
                "average_confidence": avg_confidence,
                "most_common_class": max(class_counts.keys(), key=lambda k: class_counts[k]),
                "memory_size": sum(len(routes) for routes in self.route_memory.values())
            }
            
        except Exception as e:
            logger.error(f"Stats computation failed: {e}")
            return {"error": str(e)}
    
    def update_learning_weights(self, feedback: Dict[str, float]) -> None:
        """
        Update learned weights based on feedback
        
        Args:
            feedback: Dictionary with performance feedback
        """
        try:
            # Simple learning rate
            learning_rate = 0.01
            
            # Update weights based on feedback (simplified)
            if 'efficiency_importance' in feedback:
                self.learned_weights[0] += learning_rate * feedback['efficiency_importance']
            
            if 'risk_sensitivity' in feedback:
                self.learned_weights[2] -= learning_rate * feedback['risk_sensitivity']  # Increase risk penalty
            
            # Normalize weights to prevent unbounded growth
            self.learned_weights = np.clip(self.learned_weights, -1.0, 1.0)
            
            logger.info("Learning weights updated based on feedback")
            
        except Exception as e:
            logger.error(f"Weight update failed: {e}")


class IntegratedRouteManager:
    """
    Integrated manager that combines allocator and classifier decisions
    """
    
    def __init__(self):
        self.classifier = RouteClassifier()
        self.approved_routes: Dict[str, RouteVector] = {}
        self.rejected_routes: Dict[str, Tuple[RouteVector, ClassificationResult]] = {}
        
        logger.info("Integrated route manager initialized")
    
    def validate_route(self, route: RouteVector) -> Tuple[bool, ClassificationResult]:
        """
        Validate route through classifier and return decision
        
        Args:
            route: Route to validate
            
        Returns:
            (approved, classification_result)
        """
        try:
            # Get classification
            result = self.classifier.classify_route(route)
            
            # Make decision
            if result.override_decision:
                # Route rejected
                self.rejected_routes[route.route_id] = (route, result)
                logger.warning(f"Route {route.route_id} rejected: {result.reason}")
                return False, result
            else:
                # Route approved
                self.approved_routes[route.route_id] = route
                logger.info(f"Route {route.route_id} approved: {result.classification.value}")
                return True, result
                
        except Exception as e:
            logger.error(f"Route validation failed: {e}")
            # Reject on error for safety
            error_result = ClassificationResult(
                route_id=route.route_id,
                classification=RouteClassification.UNKNOWN,
                confidence=0.0,
                override_decision=True,
                reason=f"Validation error: {str(e)}",
                risk_score=1.0
            )
            return False, error_result
    
    def get_route_summary(self) -> Dict[str, Any]:
        """Get summary of route validation activity"""
        try:
            total_routes = len(self.approved_routes) + len(self.rejected_routes)
            approval_rate = len(self.approved_routes) / total_routes if total_routes > 0 else 0
            
            # Get classifier stats
            classifier_stats = self.classifier.get_classification_stats()
            
            return {
                "total_routes_processed": total_routes,
                "approved_routes": len(self.approved_routes),
                "rejected_routes": len(self.rejected_routes),
                "approval_rate": approval_rate,
                "classifier_stats": classifier_stats
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {"error": str(e)}


def main() -> None:
    """Test and demonstration function"""
    print("Testing Route Verification Classifier...")
    
    # Create test route
    test_route = RouteVector(
        route_id="test_route_001",
        asset_pair="BTC/USDC", 
        entry_price=Decimal("26000"),
        exit_price=Decimal("27200"),
        volume=Decimal("0.5"),
        thermal_index=Decimal("1.2"),
        timestamp=datetime.now(),
        efficiency_ratio=0.8,
        profit=Decimal("600"),
        volatility=0.15,
        trend_strength=0.7,
        volume_profile=0.6,
        market_momentum=0.3,
        liquidity_depth=0.8
    )
    
    # Test classification
    manager = IntegratedRouteManager()
    approved, result = manager.validate_route(test_route)
    
    print(f"Route validation result:")
    print(f"  Approved: {approved}")
    print(f"  Classification: {result.classification.value}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Risk Score: {result.risk_score:.3f}")
    print(f"  Reason: {result.reason}")
    
    # Get summary
    summary = manager.get_route_summary()
    print(f"\nRoute Manager Summary: {summary}")
    
    print("Route Verification Classifier test completed successfully")


if __name__ == "__main__":
    main() 