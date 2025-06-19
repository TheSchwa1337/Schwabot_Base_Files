#!/usr/bin/env python3
"""
Risk Monitor - Real-time Risk Management System
==============================================

Comprehensive real-time risk monitoring and alerting system for the
Schwabot mathematical trading framework.

Key Features:
- Real-time portfolio risk monitoring
- VaR and CVaR calculations
- Drawdown tracking and alerts
- Position concentration monitoring
- Correlation risk analysis
- Thermal risk integration
- Emergency stop mechanisms
- Risk reporting and analytics

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from collections import deque

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


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Alert type enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskAlert:
    """Risk alert container"""
    
    alert_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    timestamp: float
    component: str
    metric_value: float
    threshold_value: float
    action_required: str = ""
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics container"""
    
    timestamp: float
    total_value: float
    total_pnl: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float
    correlation_exposure: float
    concentration_risk: float
    thermal_risk_index: float
    overall_risk_score: float


@dataclass
class PositionRiskData:
    """Individual position risk data"""
    
    asset: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    var_contribution: float
    correlation_risk: float
    liquidity_risk: float
    thermal_risk: float
    total_risk_score: float


class RiskMonitor:
    """Real-time risk monitoring system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize risk monitor"""
        self.version = "1.0.0"
        self.config = config or self._default_config()
        
        # Risk thresholds
        self.var_threshold = self.config.get('var_threshold', 0.05)  # 5% daily VaR
        self.cvar_threshold = self.config.get('cvar_threshold', 0.08)  # 8% daily CVaR
        self.max_drawdown_threshold = self.config.get('max_drawdown_threshold', 0.15)  # 15% max drawdown
        self.concentration_threshold = self.config.get('concentration_threshold', 0.20)  # 20% max concentration
        self.correlation_threshold = self.config.get('correlation_threshold', 0.75)  # 75% max correlation
        self.thermal_risk_threshold = self.config.get('thermal_risk_threshold', 0.80)  # 80% thermal risk
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = self.config.get('monitoring_interval', 1.0)  # 1 second
        
        # Data storage
        self.portfolio_history: deque = deque(maxlen=1000)
        self.position_history: Dict[str, deque] = {}
        self.risk_alerts: List[RiskAlert] = []
        self.emergency_stop_triggered = False
        
        # Risk calculation windows
        self.var_window = self.config.get('var_window', 100)
        self.correlation_window = self.config.get('correlation_window', 50)
        
        # Performance tracking
        self.last_calculation_time = 0.0
        self.calculation_count = 0
        
        logger.info(f"RiskMonitor v{self.version} initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'var_threshold': 0.05,
            'cvar_threshold': 0.08,
            'max_drawdown_threshold': 0.15,
            'concentration_threshold': 0.20,
            'correlation_threshold': 0.75,
            'thermal_risk_threshold': 0.80,
            'monitoring_interval': 1.0,
            'var_window': 100,
            'correlation_window': 50,
            'enable_emergency_stop': True,
            'enable_real_time_alerts': True,
            'alert_retention_days': 30
        }
    
    def start_monitoring(self) -> bool:
        """Start real-time risk monitoring"""
        if self.is_monitoring:
            logger.warning("Risk monitoring already active")
            return True
        
        try:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="RiskMonitor"
            )
            self.monitoring_thread.start()
            
            logger.info("Risk monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start risk monitoring: {e}")
            self.is_monitoring = False
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop real-time risk monitoring"""
        if not self.is_monitoring:
            return True
        
        try:
            self.is_monitoring = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            logger.info("Risk monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop risk monitoring: {e}")
            return False
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # Calculate current risk metrics
                self._calculate_portfolio_risk()
                
                # Check for risk violations
                self._check_risk_violations()
                
                # Update performance metrics
                self.calculation_count += 1
                self.last_calculation_time = time.time() - start_time
                
                # Sleep until next monitoring cycle
                time.sleep(max(0, self.monitoring_interval - (time.time() - start_time)))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def update_portfolio_data(self, portfolio_data: Dict[str, Any]) -> None:
        """Update portfolio data for risk calculations"""
        try:
            # Extract portfolio metrics
            total_value = portfolio_data.get('total_value', 0.0)
            total_pnl = portfolio_data.get('total_pnl', 0.0)
            positions = portfolio_data.get('positions', {})
            
            # Calculate portfolio risk metrics
            risk_metrics = self._calculate_portfolio_risk_metrics(
                total_value, total_pnl, positions
            )
            
            # Store in history
            self.portfolio_history.append(risk_metrics)
            
            # Update position history
            for asset, position_data in positions.items():
                if asset not in self.position_history:
                    self.position_history[asset] = deque(maxlen=100)
                
                position_risk = self._calculate_position_risk(asset, position_data)
                self.position_history[asset].append(position_risk)
            
        except Exception as e:
            logger.error(f"Failed to update portfolio data: {e}")
    
    def _calculate_portfolio_risk_metrics(self, total_value: float, total_pnl: float,
                                        positions: Dict[str, Any]) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Calculate returns for VaR/CVaR
            if len(self.portfolio_history) > 1:
                prev_value = self.portfolio_history[-1].total_value
                returns = (total_value - prev_value) / prev_value if prev_value > 0 else 0.0
            else:
                returns = 0.0
            
            # Calculate VaR and CVaR
            var_95, cvar_95 = self._calculate_var_cvar(returns)
            
            # Calculate drawdown
            max_drawdown, current_drawdown = self._calculate_drawdown(total_value)
            
            # Calculate volatility
            volatility = self._calculate_volatility()
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(returns, volatility)
            
            # Calculate correlation exposure
            correlation_exposure = self._calculate_correlation_exposure(positions)
            
            # Calculate concentration risk
            concentration_risk = self._calculate_concentration_risk(positions)
            
            # Calculate thermal risk
            thermal_risk = self._calculate_thermal_risk(positions)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                var_95, cvar_95, max_drawdown, correlation_exposure,
                concentration_risk, thermal_risk
            )
            
            return PortfolioRiskMetrics(
                timestamp=time.time(),
                total_value=total_value,
                total_pnl=total_pnl,
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                beta=1.0,  # Placeholder - would calculate from market data
                correlation_exposure=correlation_exposure,
                concentration_risk=concentration_risk,
                thermal_risk_index=thermal_risk,
                overall_risk_score=overall_risk_score
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio risk metrics: {e}")
            # Return default metrics
            return PortfolioRiskMetrics(
                timestamp=time.time(),
                total_value=total_value,
                total_pnl=total_pnl,
                var_95=0.0,
                cvar_95=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                sharpe_ratio=0.0,
                volatility=0.0,
                beta=1.0,
                correlation_exposure=0.0,
                concentration_risk=0.0,
                thermal_risk_index=0.0,
                overall_risk_score=0.0
            )
    
    def _calculate_var_cvar(self, current_return: float) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        try:
            if len(self.portfolio_history) < self.var_window:
                return 0.0, 0.0
            
            # Get historical returns
            returns = []
            for i in range(1, min(len(self.portfolio_history), self.var_window)):
                prev = self.portfolio_history[-(i+1)]
                curr = self.portfolio_history[-i]
                if prev.total_value > 0:
                    ret = (curr.total_value - prev.total_value) / prev.total_value
                    returns.append(ret)
            
            if not returns:
                return 0.0, 0.0
            
            # Add current return
            returns.append(current_return)
            returns = np.array(returns)
            
            # Calculate VaR (95th percentile)
            var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
            
            # Calculate CVaR (expected loss beyond VaR)
            cvar_95 = np.mean(returns[returns <= var_95])
            
            return abs(var_95), abs(cvar_95)
            
        except Exception as e:
            logger.error(f"VaR/CVaR calculation failed: {e}")
            return 0.0, 0.0
    
    def _calculate_drawdown(self, current_value: float) -> Tuple[float, float]:
        """Calculate maximum and current drawdown"""
        try:
            if not self.portfolio_history:
                return 0.0, 0.0
            
            # Find peak value
            peak_value = max(h.total_value for h in self.portfolio_history)
            peak_value = max(peak_value, current_value)
            
            # Calculate current drawdown
            current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0.0
            
            # Calculate maximum drawdown
            max_drawdown = 0.0
            for history in self.portfolio_history:
                if history.total_value > 0:
                    drawdown = (peak_value - history.total_value) / peak_value
                    max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown, current_drawdown
            
        except Exception as e:
            logger.error(f"Drawdown calculation failed: {e}")
            return 0.0, 0.0
    
    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        try:
            if len(self.portfolio_history) < 2:
                return 0.0
            
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev = self.portfolio_history[i-1]
                curr = self.portfolio_history[i]
                if prev.total_value > 0:
                    ret = (curr.total_value - prev.total_value) / prev.total_value
                    returns.append(ret)
            
            if not returns:
                return 0.0
            
            return np.std(returns)
            
        except Exception as e:
            logger.error(f"Volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, current_return: float, volatility: float) -> float:
        """Calculate Sharpe ratio"""
        try:
            if volatility <= 0:
                return 0.0
            
            # Assume risk-free rate of 0 for simplicity
            risk_free_rate = 0.0
            excess_return = current_return - risk_free_rate
            
            return excess_return / volatility
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_correlation_exposure(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio correlation exposure"""
        try:
            if len(positions) < 2:
                return 0.0
            
            # Simplified correlation calculation
            # In a real implementation, this would use actual correlation data
            position_sizes = [abs(pos.get('size', 0)) for pos in positions.values()]
            total_size = sum(position_sizes)
            
            if total_size <= 0:
                return 0.0
            
            # Calculate concentration-based correlation proxy
            weights = [size / total_size for size in position_sizes]
            concentration = sum(w * w for w in weights)
            
            # Convert to correlation exposure (0 = diversified, 1 = concentrated)
            correlation_exposure = 1.0 - (1.0 / len(positions))  # Base diversification
            correlation_exposure += concentration * 0.5  # Concentration penalty
            
            return min(correlation_exposure, 1.0)
            
        except Exception as e:
            logger.error(f"Correlation exposure calculation failed: {e}")
            return 0.0
    
    def _calculate_concentration_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio concentration risk"""
        try:
            if not positions:
                return 0.0
            
            total_value = sum(abs(pos.get('value', 0)) for pos in positions.values())
            if total_value <= 0:
                return 0.0
            
            # Calculate Herfindahl index
            weights = [abs(pos.get('value', 0)) / total_value for pos in positions.values()]
            concentration = sum(w * w for w in weights)
            
            return concentration
            
        except Exception as e:
            logger.error(f"Concentration risk calculation failed: {e}")
            return 0.0
    
    def _calculate_thermal_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate thermal risk index"""
        try:
            if not positions:
                return 0.0
            
            # Calculate weighted thermal risk
            total_value = sum(abs(pos.get('value', 0)) for pos in positions.values())
            if total_value <= 0:
                return 0.0
            
            thermal_risks = []
            for pos in positions.values():
                thermal_index = pos.get('thermal_index', 1.0)
                position_value = abs(pos.get('value', 0))
                weight = position_value / total_value
                thermal_risks.append(thermal_index * weight)
            
            return sum(thermal_risks)
            
        except Exception as e:
            logger.error(f"Thermal risk calculation failed: {e}")
            return 0.0
    
    def _calculate_overall_risk_score(self, var_95: float, cvar_95: float,
                                    max_drawdown: float, correlation_exposure: float,
                                    concentration_risk: float, thermal_risk: float) -> float:
        """Calculate overall risk score (0-1, where 1 is highest risk)"""
        try:
            # Normalize each risk component
            var_score = min(var_95 / self.var_threshold, 1.0)
            cvar_score = min(cvar_95 / self.cvar_threshold, 1.0)
            drawdown_score = min(max_drawdown / self.max_drawdown_threshold, 1.0)
            correlation_score = min(correlation_exposure / self.correlation_threshold, 1.0)
            concentration_score = min(concentration_risk / self.concentration_threshold, 1.0)
            thermal_score = min(thermal_risk / self.thermal_risk_threshold, 1.0)
            
            # Weighted average (VaR and CVaR get higher weights)
            weights = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]  # Sum to 1.0
            scores = [var_score, cvar_score, drawdown_score, correlation_score, concentration_score, thermal_score]
            
            overall_score = sum(w * s for w, s in zip(weights, scores))
            
            return min(overall_score, 1.0)
            
        except Exception as e:
            logger.error(f"Overall risk score calculation failed: {e}")
            return 0.5  # Default medium risk
    
    def _calculate_position_risk(self, asset: str, position_data: Dict[str, Any]) -> PositionRiskData:
        """Calculate individual position risk metrics"""
        try:
            position_size = position_data.get('size', 0.0)
            entry_price = position_data.get('entry_price', 0.0)
            current_price = position_data.get('current_price', entry_price)
            position_value = position_data.get('value', 0.0)
            
            # Calculate PnL
            unrealized_pnl = position_value - (position_size * entry_price)
            unrealized_pnl_percent = (unrealized_pnl / (position_size * entry_price)) if position_size * entry_price > 0 else 0.0
            
            # Risk metrics (simplified)
            var_contribution = abs(position_value) * 0.02  # 2% VaR contribution
            correlation_risk = position_data.get('correlation_risk', 0.0)
            liquidity_risk = position_data.get('liquidity_risk', 0.0)
            thermal_risk = position_data.get('thermal_risk', 1.0)
            
            # Total risk score
            total_risk_score = (var_contribution + correlation_risk + liquidity_risk + thermal_risk) / 4.0
            
            return PositionRiskData(
                asset=asset,
                position_size=position_size,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_percent=unrealized_pnl_percent,
                var_contribution=var_contribution,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                thermal_risk=thermal_risk,
                total_risk_score=total_risk_score
            )
            
        except Exception as e:
            logger.error(f"Position risk calculation failed for {asset}: {e}")
            return PositionRiskData(
                asset=asset,
                position_size=0.0,
                entry_price=0.0,
                current_price=0.0,
                unrealized_pnl=0.0,
                unrealized_pnl_percent=0.0,
                var_contribution=0.0,
                correlation_risk=0.0,
                liquidity_risk=0.0,
                thermal_risk=0.0,
                total_risk_score=0.0
            )
    
    def _check_risk_violations(self) -> None:
        """Check for risk violations and generate alerts"""
        try:
            if not self.portfolio_history:
                return
            
            current_metrics = self.portfolio_history[-1]
            
            # Check VaR violation
            if current_metrics.var_95 > self.var_threshold:
                self._create_alert(
                    "var_violation",
                    AlertType.WARNING,
                    RiskLevel.HIGH,
                    f"VaR {current_metrics.var_95:.2%} exceeds threshold {self.var_threshold:.2%}",
                    "portfolio_risk",
                    current_metrics.var_95,
                    self.var_threshold,
                    "Consider reducing position sizes or improving diversification"
                )
            
            # Check CVaR violation
            if current_metrics.cvar_95 > self.cvar_threshold:
                self._create_alert(
                    "cvar_violation",
                    AlertType.ERROR,
                    RiskLevel.CRITICAL,
                    f"CVaR {current_metrics.cvar_95:.2%} exceeds threshold {self.cvar_threshold:.2%}",
                    "portfolio_risk",
                    current_metrics.cvar_95,
                    self.cvar_threshold,
                    "Immediate action required: reduce risk exposure"
                )
            
            # Check drawdown violation
            if current_metrics.max_drawdown > self.max_drawdown_threshold:
                self._create_alert(
                    "drawdown_violation",
                    AlertType.CRITICAL,
                    RiskLevel.EMERGENCY,
                    f"Maximum drawdown {current_metrics.max_drawdown:.2%} exceeds threshold {self.max_drawdown_threshold:.2%}",
                    "portfolio_risk",
                    current_metrics.max_drawdown,
                    self.max_drawdown_threshold,
                    "EMERGENCY: Consider stopping all trading activities"
                )
                
                # Trigger emergency stop if enabled
                if self.config.get('enable_emergency_stop', True):
                    self._trigger_emergency_stop()
            
            # Check concentration violation
            if current_metrics.concentration_risk > self.concentration_threshold:
                self._create_alert(
                    "concentration_violation",
                    AlertType.WARNING,
                    RiskLevel.MEDIUM,
                    f"Concentration risk {current_metrics.concentration_risk:.2%} exceeds threshold {self.concentration_threshold:.2%}",
                    "portfolio_risk",
                    current_metrics.concentration_risk,
                    self.concentration_threshold,
                    "Consider diversifying portfolio positions"
                )
            
            # Check thermal risk violation
            if current_metrics.thermal_risk_index > self.thermal_risk_threshold:
                self._create_alert(
                    "thermal_risk_violation",
                    AlertType.ERROR,
                    RiskLevel.HIGH,
                    f"Thermal risk {current_metrics.thermal_risk_index:.2%} exceeds threshold {self.thermal_risk_threshold:.2%}",
                    "thermal_system",
                    current_metrics.thermal_risk_index,
                    self.thermal_risk_threshold,
                    "Reduce computational load or thermal exposure"
                )
            
        except Exception as e:
            logger.error(f"Risk violation check failed: {e}")
    
    def _create_alert(self, alert_id: str, alert_type: AlertType, risk_level: RiskLevel,
                     message: str, component: str, metric_value: float,
                     threshold_value: float, action_required: str) -> None:
        """Create and store a risk alert"""
        try:
            alert = RiskAlert(
                alert_id=alert_id,
                alert_type=alert_type,
                risk_level=risk_level,
                message=message,
                timestamp=time.time(),
                component=component,
                metric_value=metric_value,
                threshold_value=threshold_value,
                action_required=action_required
            )
            
            self.risk_alerts.append(alert)
            
            # Log alert
            log_level = {
                AlertType.INFO: logging.INFO,
                AlertType.WARNING: logging.WARNING,
                AlertType.ERROR: logging.ERROR,
                AlertType.CRITICAL: logging.CRITICAL,
                AlertType.EMERGENCY: logging.CRITICAL
            }.get(alert_type, logging.WARNING)
            
            logger.log(log_level, f"RISK ALERT [{risk_level.value.upper()}]: {message}")
            
            # Clean old alerts
            self._cleanup_old_alerts()
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    def _trigger_emergency_stop(self) -> None:
        """Trigger emergency stop mechanism"""
        try:
            if self.emergency_stop_triggered:
                return
            
            self.emergency_stop_triggered = True
            
            # Create emergency alert
            self._create_alert(
                "emergency_stop",
                AlertType.EMERGENCY,
                RiskLevel.EMERGENCY,
                "EMERGENCY STOP TRIGGERED - All trading activities suspended",
                "risk_monitor",
                1.0,
                0.0,
                "IMMEDIATE: Review risk parameters and system status"
            )
            
            logger.critical("🚨 EMERGENCY STOP TRIGGERED - Trading suspended")
            
            # Here you would integrate with the trading system to stop all activities
            # self.trading_system.emergency_stop()
            
        except Exception as e:
            logger.error(f"Failed to trigger emergency stop: {e}")
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts based on retention policy"""
        try:
            retention_days = self.config.get('alert_retention_days', 30)
            cutoff_time = time.time() - (retention_days * 24 * 3600)
            
            # Remove old alerts
            self.risk_alerts = [
                alert for alert in self.risk_alerts
                if alert.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Failed to cleanup old alerts: {e}")
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary"""
        try:
            if not self.portfolio_history:
                return {
                    'status': 'no_data',
                    'monitoring_active': self.is_monitoring,
                    'emergency_stop': self.emergency_stop_triggered
                }
            
            current_metrics = self.portfolio_history[-1]
            
            return {
                'status': 'active',
                'monitoring_active': self.is_monitoring,
                'emergency_stop': self.emergency_stop_triggered,
                'timestamp': current_metrics.timestamp,
                'portfolio_value': current_metrics.total_value,
                'total_pnl': current_metrics.total_pnl,
                'risk_metrics': {
                    'var_95': current_metrics.var_95,
                    'cvar_95': current_metrics.cvar_95,
                    'max_drawdown': current_metrics.max_drawdown,
                    'current_drawdown': current_metrics.current_drawdown,
                    'sharpe_ratio': current_metrics.sharpe_ratio,
                    'volatility': current_metrics.volatility,
                    'correlation_exposure': current_metrics.correlation_exposure,
                    'concentration_risk': current_metrics.concentration_risk,
                    'thermal_risk': current_metrics.thermal_risk_index,
                    'overall_risk_score': current_metrics.overall_risk_score
                },
                'risk_thresholds': {
                    'var_threshold': self.var_threshold,
                    'cvar_threshold': self.cvar_threshold,
                    'max_drawdown_threshold': self.max_drawdown_threshold,
                    'concentration_threshold': self.concentration_threshold,
                    'correlation_threshold': self.correlation_threshold,
                    'thermal_risk_threshold': self.thermal_risk_threshold
                },
                'alerts': {
                    'total_alerts': len(self.risk_alerts),
                    'unacknowledged_alerts': len([a for a in self.risk_alerts if not a.acknowledged]),
                    'critical_alerts': len([a for a in self.risk_alerts if a.risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]])
                },
                'performance': {
                    'calculation_count': self.calculation_count,
                    'last_calculation_time': self.last_calculation_time,
                    'monitoring_interval': self.monitoring_interval
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get risk status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'monitoring_active': self.is_monitoring,
                'emergency_stop': self.emergency_stop_triggered
            }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a specific alert"""
        try:
            for alert in self.risk_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert {alert_id} acknowledged")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        try:
            for alert in self.risk_alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info(f"Alert {alert_id} resolved")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop state"""
        try:
            self.emergency_stop_triggered = False
            logger.info("Emergency stop reset")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset emergency stop: {e}")
            return False


def main() -> None:
    """Main function for testing risk monitor"""
    try:
        print("🔍 Risk Monitor Test")
        print("=" * 40)
        
        # Initialize risk monitor
        config = {
            'var_threshold': 0.05,
            'cvar_threshold': 0.08,
            'max_drawdown_threshold': 0.15,
            'monitoring_interval': 0.1  # Fast for testing
        }
        
        risk_monitor = RiskMonitor(config)
        
        # Test portfolio data
        portfolio_data = {
            'total_value': 100000.0,
            'total_pnl': 5000.0,
            'positions': {
                'BTC': {
                    'size': 1.0,
                    'entry_price': 25000.0,
                    'current_price': 26000.0,
                    'value': 26000.0,
                    'thermal_index': 1.2
                },
                'ETH': {
                    'size': 10.0,
                    'entry_price': 2000.0,
                    'current_price': 2100.0,
                    'value': 21000.0,
                    'thermal_index': 1.1
                }
            }
        }
        
        # Update portfolio data
        risk_monitor.update_portfolio_data(portfolio_data)
        
        # Get risk status
        status = risk_monitor.get_current_risk_status()
        print(f"✅ Risk Monitor initialized: {status['status']}")
        print(f"✅ Portfolio value: ${status['portfolio_value']:,.2f}")
        print(f"✅ Overall risk score: {status['risk_metrics']['overall_risk_score']:.3f}")
        
        # Start monitoring
        risk_monitor.start_monitoring()
        print("✅ Risk monitoring started")
        
        # Simulate some time
        time.sleep(0.5)
        
        # Stop monitoring
        risk_monitor.stop_monitoring()
        print("✅ Risk monitoring stopped")
        
        print("\n🎉 Risk Monitor test completed successfully!")
        
    except Exception as e:
        print(f"❌ Risk Monitor test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
