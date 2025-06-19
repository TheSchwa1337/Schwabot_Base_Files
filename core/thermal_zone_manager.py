#!/usr/bin/env python3
"""
Thermal Zone Manager - Schwabot Mathematical Framework.

Advanced thermal zone management system for monitoring and controlling
thermal conditions across different trading operations and mathematical
computations with adaptive thresholds and real-time optimization.

Key Features:
- Dynamic thermal zone creation and management
- Adaptive thermal threshold monitoring
- Real-time thermal signature tracking
- Zone-based performance optimization
- Integration with mathematical trading systems
- Windows CLI compatibility
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing_extensions import Self

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


@dataclass
class ThermalZone:
    """Container for thermal zone configuration and state."""
    
    zone_id: str
    zone_name: str
    base_temperature: Decimal
    current_temperature: Decimal
    thermal_threshold: Decimal
    adaptive_factor: Decimal
    zone_type: str
    created_at: float
    last_updated: float
    performance_metrics: Dict[str, float]
    active: bool = True


@dataclass
class ThermalAlert:
    """Container for thermal alert information."""
    
    alert_id: str
    zone_id: str
    alert_type: str  # 'threshold_exceeded', 'rapid_change', 'zone_unstable'
    severity: str    # 'low', 'medium', 'high', 'critical'
    temperature: Decimal
    threshold: Decimal
    timestamp: float
    message: str
    resolved: bool = False


@dataclass
class ThermalSnapshot:
    """Container for thermal zone snapshot data."""
    
    zone_id: str
    timestamp: float
    temperature: Decimal
    thermal_drift: Decimal
    stability_score: float
    performance_impact: float
    zone_efficiency: float


class ThermalThresholdController:
    """Controls adaptive thermal thresholds based on zone performance."""
    
    def __init__(self) -> None:
        """Initialize thermal threshold controller."""
        self.adaptation_rate = Decimal("0.05")
        self.min_threshold = Decimal("0.1")
        self.max_threshold = Decimal("5.0")
        self.stability_weight = Decimal("0.7")
        self.performance_weight = Decimal("0.3")
    
    def calculate_adaptive_threshold(self, zone: ThermalZone, 
                                   performance_history: List[float]) -> Decimal:
        """Calculate adaptive threshold based on zone performance."""
        if not performance_history:
            return zone.thermal_threshold
        
        # Calculate performance trend
        recent_performance = performance_history[-5:] if len(performance_history) >= 5 else performance_history
        avg_performance = sum(recent_performance) / len(recent_performance)
        
        # Calculate thermal stability
        temp_variance = zone.performance_metrics.get('temperature_variance', 0.0)
        stability_factor = 1.0 / (1.0 + temp_variance)
        
        # Adaptive adjustment
        performance_factor = Decimal(str(avg_performance))
        stability_factor_decimal = Decimal(str(stability_factor))
        
        # Weighted adjustment
        adjustment = (self.performance_weight * performance_factor + 
                     self.stability_weight * stability_factor_decimal)
        
        # Apply adaptation
        new_threshold = zone.thermal_threshold * (Decimal("1.0") + self.adaptation_rate * adjustment)
        
        # Apply bounds
        return max(self.min_threshold, min(self.max_threshold, new_threshold))
    
    def detect_thermal_anomalies(self, zone: ThermalZone, 
                                temperature_history: List[Decimal]) -> List[str]:
        """Detect thermal anomalies in zone."""
        anomalies = []
        
        if len(temperature_history) < 3:
            return anomalies
        
        # Check for rapid temperature changes
        recent_temps = temperature_history[-3:]
        temp_deltas = [recent_temps[i+1] - recent_temps[i] for i in range(len(recent_temps)-1)]
        
        max_delta = max(abs(d) for d in temp_deltas)
        if max_delta > zone.thermal_threshold * Decimal("0.5"):
            anomalies.append("rapid_temperature_change")
        
        # Check for sustained high temperature
        if all(t > zone.thermal_threshold for t in recent_temps):
            anomalies.append("sustained_high_temperature")
        
        # Check for temperature oscillation
        if len(temp_deltas) >= 2:
            if (temp_deltas[0] > 0 and temp_deltas[1] < 0) or (temp_deltas[0] < 0 and temp_deltas[1] > 0):
                if abs(temp_deltas[0]) > zone.thermal_threshold * Decimal("0.3"):
                    anomalies.append("temperature_oscillation")
        
        return anomalies


class ThermalPerformanceAnalyzer:
    """Analyzes thermal performance and optimization opportunities."""
    
    def __init__(self) -> None:
        """Initialize thermal performance analyzer."""
        self.efficiency_threshold = 0.8
        self.stability_threshold = 0.7
    
    def calculate_zone_efficiency(self, zone: ThermalZone, 
                                processing_time: float,
                                thermal_cost: Decimal) -> float:
        """Calculate thermal efficiency of a zone."""
        if processing_time <= 0 or thermal_cost <= 0:
            return 0.0
        
        # Base efficiency calculation
        base_efficiency = 1.0 / processing_time
        
        # Thermal cost adjustment
        thermal_penalty = float(thermal_cost / zone.thermal_threshold)
        adjusted_efficiency = base_efficiency / (1.0 + thermal_penalty)
        
        # Temperature stability bonus
        temp_variance = zone.performance_metrics.get('temperature_variance', 1.0)
        stability_bonus = 1.0 / (1.0 + temp_variance)
        
        final_efficiency = adjusted_efficiency * stability_bonus
        
        return min(1.0, max(0.0, final_efficiency))
    
    def analyze_thermal_patterns(self, snapshots: List[ThermalSnapshot]) -> Dict[str, Any]:
        """Analyze thermal patterns across snapshots."""
        if not snapshots:
            return {'status': 'no_data'}
        
        # Temperature trend analysis
        temperatures = [float(s.temperature) for s in snapshots]
        times = [s.timestamp for s in snapshots]
        
        # Calculate temperature slope (trend)
        if len(temperatures) >= 2:
            time_deltas = [times[i+1] - times[i] for i in range(len(times)-1)]
            temp_deltas = [temperatures[i+1] - temperatures[i] for i in range(len(temperatures)-1)]
            
            avg_slope = sum(td / max(td_time, 1e-6) for td, td_time in zip(temp_deltas, time_deltas)) / len(temp_deltas)
        else:
            avg_slope = 0.0
        
        # Stability analysis
        temp_mean = sum(temperatures) / len(temperatures)
        temp_variance = sum((t - temp_mean) ** 2 for t in temperatures) / len(temperatures)
        stability_score = 1.0 / (1.0 + temp_variance)
        
        # Efficiency analysis
        efficiency_scores = [s.zone_efficiency for s in snapshots]
        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)
        
        # Performance impact analysis
        performance_impacts = [s.performance_impact for s in snapshots]
        avg_performance_impact = sum(performance_impacts) / len(performance_impacts)
        
        return {
            'status': 'analyzed',
            'sample_count': len(snapshots),
            'temperature_trend': {
                'slope': avg_slope,
                'direction': 'increasing' if avg_slope > 0.01 else 'decreasing' if avg_slope < -0.01 else 'stable',
                'mean_temperature': temp_mean,
                'temperature_variance': temp_variance
            },
            'stability_metrics': {
                'stability_score': stability_score,
                'is_stable': stability_score >= self.stability_threshold
            },
            'efficiency_metrics': {
                'average_efficiency': avg_efficiency,
                'is_efficient': avg_efficiency >= self.efficiency_threshold,
                'performance_impact': avg_performance_impact
            },
            'recommendations': self._generate_optimization_recommendations(
                avg_slope, stability_score, avg_efficiency
            )
        }
    
    def _generate_optimization_recommendations(self, slope: float, 
                                             stability: float, 
                                             efficiency: float) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        if slope > 0.05:
            recommendations.append("Consider cooling strategies due to increasing temperature trend")
        elif slope < -0.05:
            recommendations.append("Monitor for potential underutilization")
        
        if stability < self.stability_threshold:
            recommendations.append("Implement thermal stabilization measures")
        
        if efficiency < self.efficiency_threshold:
            recommendations.append("Optimize thermal management for better efficiency")
        
        if not recommendations:
            recommendations.append("Thermal zone operating within optimal parameters")
        
        return recommendations


class ThermalZoneManager:
    """Main thermal zone management system."""
    
    def __init__(self) -> None:
        """Initialize thermal zone manager."""
        self.version = "1.0.0"
        self.zones: Dict[str, ThermalZone] = {}
        self.thermal_history: Dict[str, List[ThermalSnapshot]] = {}
        self.alerts: List[ThermalAlert] = []
        self.threshold_controller = ThermalThresholdController()
        self.performance_analyzer = ThermalPerformanceAnalyzer()
        
        # Configuration
        self.max_history_length = 1000
        self.alert_cooldown = 300.0  # 5 minutes
        self.last_alert_times: Dict[str, float] = {}
        
        logger.info(f"ThermalZoneManager v{self.version} initialized")
    
    def create_thermal_zone(self, zone_name: str, base_temperature: float,
                          thermal_threshold: float, zone_type: str = "default") -> str:
        """Create a new thermal zone."""
        import uuid
        
        zone_id = str(uuid.uuid4())[:8]
        current_time = time.time()
        
        zone = ThermalZone(
            zone_id=zone_id,
            zone_name=zone_name,
            base_temperature=Decimal(str(base_temperature)),
            current_temperature=Decimal(str(base_temperature)),
            thermal_threshold=Decimal(str(thermal_threshold)),
            adaptive_factor=Decimal("1.0"),
            zone_type=zone_type,
            created_at=current_time,
            last_updated=current_time,
            performance_metrics={
                'temperature_variance': 0.0,
                'efficiency_score': 1.0,
                'processing_time': 0.0,
                'thermal_cost': 0.0
            }
        )
        
        self.zones[zone_id] = zone
        self.thermal_history[zone_id] = []
        
        logger.info(f"Created thermal zone '{zone_name}' with ID {zone_id}")
        return zone_id
    
    def update_zone_temperature(self, zone_id: str, new_temperature: float,
                              processing_time: float = 0.0, 
                              thermal_cost: float = 0.0) -> Dict[str, Any]:
        """Update zone temperature and analyze thermal conditions."""
        if zone_id not in self.zones:
            return {'status': 'error', 'error': 'Zone not found'}
        
        zone = self.zones[zone_id]
        current_time = time.time()
        
        # Update zone temperature
        old_temperature = zone.current_temperature
        zone.current_temperature = Decimal(str(new_temperature))
        zone.last_updated = current_time
        
        # Calculate thermal drift
        thermal_drift = zone.current_temperature - old_temperature
        
        # Update performance metrics
        zone.performance_metrics['processing_time'] = processing_time
        zone.performance_metrics['thermal_cost'] = thermal_cost
        
        # Calculate efficiency
        efficiency = self.performance_analyzer.calculate_zone_efficiency(
            zone, processing_time, Decimal(str(thermal_cost))
        )
        zone.performance_metrics['efficiency_score'] = efficiency
        
        # Calculate stability score
        zone_history = self.thermal_history[zone_id]
        if len(zone_history) >= 2:
            recent_temps = [float(s.temperature) for s in zone_history[-5:]]
            recent_temps.append(new_temperature)
            temp_mean = sum(recent_temps) / len(recent_temps)
            temp_variance = sum((t - temp_mean) ** 2 for t in recent_temps) / len(recent_temps)
            zone.performance_metrics['temperature_variance'] = temp_variance
            stability_score = 1.0 / (1.0 + temp_variance)
        else:
            stability_score = 1.0
        
        # Create thermal snapshot
        snapshot = ThermalSnapshot(
            zone_id=zone_id,
            timestamp=current_time,
            temperature=zone.current_temperature,
            thermal_drift=thermal_drift,
            stability_score=stability_score,
            performance_impact=1.0 - efficiency,
            zone_efficiency=efficiency
        )
        
        # Store snapshot
        self.thermal_history[zone_id].append(snapshot)
        
        # Trim history if too long
        if len(self.thermal_history[zone_id]) > self.max_history_length:
            self.thermal_history[zone_id] = self.thermal_history[zone_id][-self.max_history_length:]
        
        # Check for thermal alerts
        alerts = self._check_thermal_alerts(zone_id)
        
        # Adaptive threshold adjustment
        performance_history = [s.zone_efficiency for s in self.thermal_history[zone_id][-10:]]
        new_threshold = self.threshold_controller.calculate_adaptive_threshold(
            zone, performance_history
        )
        zone.thermal_threshold = new_threshold
        
        return {
            'status': 'success',
            'zone_id': zone_id,
            'old_temperature': float(old_temperature),
            'new_temperature': float(zone.current_temperature),
            'thermal_drift': float(thermal_drift),
            'efficiency': efficiency,
            'stability_score': stability_score,
            'adaptive_threshold': float(new_threshold),
            'alerts_generated': len(alerts),
            'zone_active': zone.active
        }
    
    def _check_thermal_alerts(self, zone_id: str) -> List[ThermalAlert]:
        """Check for thermal alerts in a zone."""
        zone = self.zones[zone_id]
        alerts = []
        current_time = time.time()
        
        # Check cooldown
        last_alert_time = self.last_alert_times.get(zone_id, 0.0)
        if current_time - last_alert_time < self.alert_cooldown:
            return alerts
        
        # Threshold exceeded alert
        if zone.current_temperature > zone.thermal_threshold:
            alert = self._create_alert(
                zone_id, "threshold_exceeded", "high",
                f"Temperature {zone.current_temperature:.3f} exceeds threshold {zone.thermal_threshold:.3f}"
            )
            alerts.append(alert)
        
        # Rapid change alert
        zone_history = self.thermal_history[zone_id]
        if len(zone_history) >= 2:
            recent_drift = abs(zone_history[-1].thermal_drift)
            if recent_drift > zone.thermal_threshold * Decimal("0.5"):
                alert = self._create_alert(
                    zone_id, "rapid_change", "medium",
                    f"Rapid temperature change detected: {recent_drift:.3f}"
                )
                alerts.append(alert)
        
        # Zone instability alert
        temp_variance = zone.performance_metrics.get('temperature_variance', 0.0)
        if temp_variance > 1.0:
            alert = self._create_alert(
                zone_id, "zone_unstable", "medium",
                f"Zone thermal instability detected (variance: {temp_variance:.3f})"
            )
            alerts.append(alert)
        
        # Update last alert time if alerts were generated
        if alerts:
            self.last_alert_times[zone_id] = current_time
            self.alerts.extend(alerts)
        
        return alerts
    
    def _create_alert(self, zone_id: str, alert_type: str, severity: str, message: str) -> ThermalAlert:
        """Create a thermal alert."""
        import uuid
        
        zone = self.zones[zone_id]
        alert_id = str(uuid.uuid4())[:8]
        
        return ThermalAlert(
            alert_id=alert_id,
            zone_id=zone_id,
            alert_type=alert_type,
            severity=severity,
            temperature=zone.current_temperature,
            threshold=zone.thermal_threshold,
            timestamp=time.time(),
            message=message
        )
    
    def get_zone_status(self, zone_id: str) -> Dict[str, Any]:
        """Get comprehensive status of a thermal zone."""
        if zone_id not in self.zones:
            return {'status': 'error', 'error': 'Zone not found'}
        
        zone = self.zones[zone_id]
        zone_history = self.thermal_history[zone_id]
        
        # Analyze thermal patterns
        pattern_analysis = self.performance_analyzer.analyze_thermal_patterns(zone_history[-50:])
        
        # Get recent alerts
        recent_alerts = [a for a in self.alerts 
                        if a.zone_id == zone_id and not a.resolved 
                        and time.time() - a.timestamp < 3600]  # Last hour
        
        return {
            'status': 'success',
            'zone_info': {
                'zone_id': zone_id,
                'zone_name': zone.zone_name,
                'zone_type': zone.zone_type,
                'active': zone.active,
                'created_at': zone.created_at,
                'last_updated': zone.last_updated
            },
            'thermal_status': {
                'current_temperature': float(zone.current_temperature),
                'base_temperature': float(zone.base_temperature),
                'thermal_threshold': float(zone.thermal_threshold),
                'adaptive_factor': float(zone.adaptive_factor)
            },
            'performance_metrics': zone.performance_metrics,
            'pattern_analysis': pattern_analysis,
            'recent_alerts': [
                {
                    'alert_id': a.alert_id,
                    'alert_type': a.alert_type,
                    'severity': a.severity,
                    'message': a.message,
                    'timestamp': a.timestamp
                }
                for a in recent_alerts
            ],
            'history_length': len(zone_history)
        }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get overview of all thermal zones."""
        total_zones = len(self.zones)
        active_zones = sum(1 for z in self.zones.values() if z.active)
        
        # Calculate system-wide metrics
        if self.zones:
            avg_temperature = sum(float(z.current_temperature) for z in self.zones.values()) / total_zones
            avg_efficiency = sum(z.performance_metrics.get('efficiency_score', 0.0) 
                               for z in self.zones.values()) / total_zones
            
            # Hot zones (temperature > threshold)
            hot_zones = sum(1 for z in self.zones.values() 
                           if z.current_temperature > z.thermal_threshold)
            
            # Critical alerts
            critical_alerts = sum(1 for a in self.alerts 
                                if a.severity == 'critical' and not a.resolved)
        else:
            avg_temperature = 0.0
            avg_efficiency = 0.0
            hot_zones = 0
            critical_alerts = 0
        
        return {
            'system_status': {
                'total_zones': total_zones,
                'active_zones': active_zones,
                'hot_zones': hot_zones,
                'system_efficiency': avg_efficiency,
                'average_temperature': avg_temperature
            },
            'alert_summary': {
                'total_alerts': len(self.alerts),
                'unresolved_alerts': sum(1 for a in self.alerts if not a.resolved),
                'critical_alerts': critical_alerts
            },
            'zones': [
                {
                    'zone_id': zid,
                    'zone_name': zone.zone_name,
                    'current_temperature': float(zone.current_temperature),
                    'efficiency': zone.performance_metrics.get('efficiency_score', 0.0),
                    'active': zone.active
                }
                for zid, zone in self.zones.items()
            ],
            'version': self.version
        }


def main() -> None:
    """Demo of thermal zone manager system."""
    try:
        manager = ThermalZoneManager()
        print(f"✅ ThermalZoneManager v{manager.version} initialized")
        
        # Create test thermal zones
        btc_zone = manager.create_thermal_zone("BTC_Trading", 1.0, 2.5, "trading")
        eth_zone = manager.create_thermal_zone("ETH_Trading", 0.8, 2.0, "trading")
        math_zone = manager.create_thermal_zone("Mathematical_Core", 0.5, 1.5, "computation")
        
        print(f"🌡️  Created thermal zones:")
        print(f"   BTC Zone: {btc_zone}")
        print(f"   ETH Zone: {eth_zone}")
        print(f"   Math Zone: {math_zone}")
        
        # Simulate thermal updates
        print(f"\n📊 Simulating thermal updates:")
        
        # Normal operation
        result1 = manager.update_zone_temperature(btc_zone, 1.2, 0.5, 0.3)
        print(f"   BTC update 1: Temp {result1['new_temperature']:.2f}, "
              f"Efficiency {result1['efficiency']:.3f}")
        
        # Rising temperature
        result2 = manager.update_zone_temperature(btc_zone, 2.8, 0.8, 0.6)
        print(f"   BTC update 2: Temp {result2['new_temperature']:.2f}, "
              f"Alerts {result2['alerts_generated']}")
        
        # ETH zone update
        result3 = manager.update_zone_temperature(eth_zone, 1.5, 0.4, 0.2)
        print(f"   ETH update: Temp {result3['new_temperature']:.2f}, "
              f"Efficiency {result3['efficiency']:.3f}")
        
        # Get zone status
        btc_status = manager.get_zone_status(btc_zone)
        if btc_status['status'] == 'success':
            thermal_status = btc_status['thermal_status']
            print(f"\n🎯 BTC Zone Status:")
            print(f"   Current temp: {thermal_status['current_temperature']:.3f}")
            print(f"   Threshold: {thermal_status['thermal_threshold']:.3f}")
            print(f"   Recent alerts: {len(btc_status['recent_alerts'])}")
        
        # System overview
        overview = manager.get_system_overview()
        system_status = overview['system_status']
        print(f"\n📈 System Overview:")
        print(f"   Total zones: {system_status['total_zones']}")
        print(f"   Hot zones: {system_status['hot_zones']}")
        print(f"   System efficiency: {system_status['system_efficiency']:.3f}")
        print(f"   Avg temperature: {system_status['average_temperature']:.3f}")
        
        print("🎉 Thermal zone manager demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main() 