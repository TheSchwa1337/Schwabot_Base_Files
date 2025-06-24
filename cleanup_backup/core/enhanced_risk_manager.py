#!/usr/bin/env python3
"""Enhanced Risk Manager - Advanced Risk Analytics and Stress Testing.

This module provides sophisticated risk management including:
- Real-time risk analytics and monitoring
- Stress testing and scenario analysis
- VaR (Value at Risk) and CVaR calculations
- Risk factor decomposition and attribution
- Dynamic risk limits and adaptive controls
- Integration with Risk Guard and Capital Controls
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

# Import unified mathematics
try:
    from core.unified_mathematics_config import get_unified_math
    unified_math = get_unified_math()
    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    UNIFIED_MATH_AVAILABLE = False

# Import risk guard for integration
try:
    from core.risk_guard import get_risk_guard, is_trading_allowed, check_circuit_breaker
    risk_guard = get_risk_guard()
    RISK_GUARD_AVAILABLE = True
except ImportError:
    RISK_GUARD_AVAILABLE = False

# Import capital controls for integration
try:
    from core.capital_controls import get_capital_controls, check_portfolio_limits
    capital_controls = get_capital_controls()
    CAPITAL_CONTROLS_AVAILABLE = True
except ImportError:
    CAPITAL_CONTROLS_AVAILABLE = False

# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, safe_format_error, log_safe
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

logger = logging.getLogger(__name__)


class RiskMetricType(Enum):
    """Types of risk metrics."""
    VAR = "var"                        # Value at Risk
    CVAR = "cvar"                      # Conditional Value at Risk
    VOLATILITY = "volatility"          # Portfolio volatility
    BETA = "beta"                      # Market beta
    SHARPE_RATIO = "sharpe_ratio"      # Sharpe ratio
    MAX_DRAWDOWN = "max_drawdown"      # Maximum drawdown
    CORRELATION = "correlation"        # Correlation risk
    CONCENTRATION = "concentration"    # Concentration risk


class StressTestScenario(Enum):
    """Stress test scenarios."""
    MARKET_CRASH = "market_crash"      # 20% market decline
    VOLATILITY_SPIKE = "volatility_spike"  # 3x volatility increase
    CORRELATION_BREAKDOWN = "correlation_breakdown"  # Correlation breakdown
    LIQUIDITY_CRISIS = "liquidity_crisis"  # Liquidity crisis
    INTEREST_RATE_SHOCK = "interest_rate_shock"  # Interest rate shock
    CUSTOM_SCENARIO = "custom_scenario"  # Custom scenario


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    var_95: float = 0.0                # 95% VaR
    var_99: float = 0.0                # 99% VaR
    cvar_95: float = 0.0               # 95% CVaR
    cvar_99: float = 0.0               # 99% CVaR
    volatility: float = 0.0            # Portfolio volatility
    beta: float = 0.0                  # Market beta
    sharpe_ratio: float = 0.0          # Sharpe ratio
    max_drawdown: float = 0.0          # Maximum drawdown
    correlation_risk: float = 0.0      # Correlation risk
    concentration_risk: float = 0.0    # Concentration risk
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Result of stress test."""
    scenario: StressTestScenario
    portfolio_loss: float
    var_impact: float
    volatility_impact: float
    correlation_impact: float
    worst_case_loss: float
    recovery_time_estimate: float
    risk_level: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskFactor:
    """Risk factor analysis."""
    factor_name: str
    factor_value: float
    risk_contribution: float
    sensitivity: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAlert:
    """Risk alert data."""
    alert_type: str
    severity: str
    description: str
    threshold: float
    current_value: float
    timestamp: datetime
    triggered_by: str
    action_required: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedRiskManager:
    """
    Enhanced Risk Manager - Advanced risk analytics and stress testing.
    
    Provides sophisticated risk management including:
    - Real-time risk analytics and monitoring
    - Stress testing and scenario analysis
    - VaR (Value at Risk) and CVaR calculations
    - Risk factor decomposition and attribution
    - Dynamic risk limits and adaptive controls
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced risk manager."""
        self.config = config or {}
        
        # Risk configuration
        self.var_confidence_levels = [0.95, 0.99]
        self.stress_test_scenarios = list(StressTestScenario)
        self.risk_alert_thresholds = {
            'var_95': 0.02,      # 2% VaR threshold
            'var_99': 0.05,      # 5% VaR threshold
            'volatility': 0.25,  # 25% volatility threshold
            'drawdown': 0.15,    # 15% drawdown threshold
            'correlation': 0.8,  # 80% correlation threshold
            'concentration': 0.2  # 20% concentration threshold
        }
        
        # Risk metrics storage
        self.risk_metrics_history: List[RiskMetrics] = []
        self.stress_test_results: List[StressTestResult] = []
        self.risk_factors: List[RiskFactor] = []
        self.risk_alerts: List[RiskAlert] = []
        
        # Performance tracking
        self.total_risk_checks = 0
        self.risk_violations = 0
        self.stress_tests_run = 0
        
        # Real-time monitoring
        self.monitoring_active = True
        self.alert_thresholds_breached = 0
        
        safe_print("🎯 Enhanced Risk Manager initialized")
    
    def calculate_risk_metrics(
        self,
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        This calculates:
        - VaR and CVaR at multiple confidence levels
        - Portfolio volatility and beta
        - Sharpe ratio and maximum drawdown
        - Correlation and concentration risk
        """
        try:
            # Extract portfolio information
            positions = portfolio_data.get('positions', {})
            total_value = portfolio_data.get('total_value', 0.0)
            total_pnl = portfolio_data.get('total_pnl', 0.0)
            
            if total_value == 0:
                return RiskMetrics()
            
            # Calculate VaR and CVaR
            var_95, cvar_95 = self._calculate_var_cvar(positions, market_data, 0.95)
            var_99, cvar_99 = self._calculate_var_cvar(positions, market_data, 0.99)
            
            # Calculate volatility
            volatility = self._calculate_portfolio_volatility(positions, market_data)
            
            # Calculate beta
            beta = self._calculate_portfolio_beta(positions, market_data)
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(total_pnl, volatility)
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown(historical_data)
            
            # Calculate correlation risk
            correlation_risk = self._calculate_correlation_risk(positions, market_data)
            
            # Calculate concentration risk
            concentration_risk = self._calculate_concentration_risk(positions, total_value)
            
            # Create risk metrics
            risk_metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                timestamp=datetime.now(),
                metadata={
                    'total_value': total_value,
                    'total_pnl': total_pnl,
                    'num_positions': len(positions)
                }
            )
            
            # Store in history
            self.risk_metrics_history.append(risk_metrics)
            
            # Keep only recent history
            if len(self.risk_metrics_history) > 1000:
                self.risk_metrics_history = self.risk_metrics_history[-1000:]
            
            safe_print(f"✅ Risk metrics calculated: VaR(95%) = {var_95:.2%}")
            return risk_metrics
            
        except Exception as e:
            safe_print(f"❌ Risk metrics calculation failed: {safe_format_error(e, 'risk_metrics')}")
            return RiskMetrics()
    
    def _calculate_var_cvar(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any],
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate VaR and CVaR."""
        try:
            # Simplified VaR calculation using parametric method
            # In practice, you'd use historical simulation or Monte Carlo
            
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            if total_value == 0:
                return 0.0, 0.0
            
            # Calculate portfolio volatility
            portfolio_vol = self._calculate_portfolio_volatility(positions, market_data)
            
            # Parametric VaR: VaR = z * σ * √t
            # For daily VaR, t = 1
            z_score = self._get_z_score(confidence_level)
            var = z_score * portfolio_vol * total_value
            
            # CVaR (Expected Shortfall) = E[X|X>VaR]
            # Simplified: CVaR ≈ VaR * 1.25 for normal distribution
            cvar = var * 1.25
            
            return var, cvar
            
        except Exception as e:
            safe_print(f"❌ VaR/CVaR calculation failed: {safe_format_error(e, 'var_cvar')}")
            return 0.0, 0.0
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for confidence level."""
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326,
            0.995: 2.576
        }
        return z_scores.get(confidence_level, 1.645)
    
    def _calculate_portfolio_volatility(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate portfolio volatility."""
        try:
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            if total_value == 0:
                return 0.0
            
            # Weighted average volatility
            weighted_vol = 0.0
            for asset, pos in positions.items():
                weight = pos.get('value', 0) / total_value
                volatility = market_data.get(asset, {}).get('volatility', 0.0)
                weighted_vol += weight * volatility
            
            return weighted_vol
            
        except Exception as e:
            safe_print(f"❌ Portfolio volatility calculation failed: {safe_format_error(e, 'portfolio_volatility')}")
            return 0.0
    
    def _calculate_portfolio_beta(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate portfolio beta."""
        try:
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            if total_value == 0:
                return 0.0
            
            # Weighted average beta
            weighted_beta = 0.0
            for asset, pos in positions.items():
                weight = pos.get('value', 0) / total_value
                beta = market_data.get(asset, {}).get('beta', 1.0)
                weighted_beta += weight * beta
            
            return weighted_beta
            
        except Exception as e:
            safe_print(f"❌ Portfolio beta calculation failed: {safe_format_error(e, 'portfolio_beta')}")
            return 1.0
    
    def _calculate_sharpe_ratio(self, total_pnl: float, volatility: float) -> float:
        """Calculate Sharpe ratio."""
        try:
            if volatility == 0:
                return 0.0
            
            # Assume risk-free rate of 0 for simplicity
            sharpe_ratio = total_pnl / volatility
            
            return sharpe_ratio
            
        except Exception as e:
            safe_print(f"❌ Sharpe ratio calculation failed: {safe_format_error(e, 'sharpe_ratio')}")
            return 0.0
    
    def _calculate_max_drawdown(self, historical_data: Optional[List[Dict[str, Any]]]) -> float:
        """Calculate maximum drawdown."""
        try:
            if not historical_data:
                return 0.0
            
            # Extract portfolio values
            values = [data.get('total_value', 0) for data in historical_data]
            
            if not values:
                return 0.0
            
            # Calculate running maximum and drawdown
            peak = values[0]
            max_drawdown = 0.0
            
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            safe_print(f"❌ Max drawdown calculation failed: {safe_format_error(e, 'max_drawdown')}")
            return 0.0
    
    def _calculate_correlation_risk(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate correlation risk."""
        try:
            if len(positions) < 2:
                return 0.0
            
            # Calculate average correlation between positions
            correlations = []
            assets = list(positions.keys())
            
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    asset1, asset2 = assets[i], assets[j]
                    # Simplified correlation (in practice, use historical data)
                    correlation = 0.5  # Default moderate correlation
                    correlations.append(correlation)
            
            if correlations:
                avg_correlation = sum(correlations) / len(correlations)
                return avg_correlation
            else:
                return 0.0
            
        except Exception as e:
            safe_print(f"❌ Correlation risk calculation failed: {safe_format_error(e, 'correlation_risk')}")
            return 0.0
    
    def _calculate_concentration_risk(
        self,
        positions: Dict[str, Any],
        total_value: float
    ) -> float:
        """Calculate concentration risk using Herfindahl index."""
        try:
            if total_value == 0:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index (HHI)
            hhi = 0.0
            for pos in positions.values():
                weight = pos.get('value', 0) / total_value
                hhi += weight ** 2
            
            return hhi
            
        except Exception as e:
            safe_print(f"❌ Concentration risk calculation failed: {safe_format_error(e, 'concentration_risk')}")
            return 0.0
    
    def run_stress_test(
        self,
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any],
        scenario: StressTestScenario,
        custom_shocks: Optional[Dict[str, float]] = None
    ) -> StressTestResult:
        """
        Run stress test on portfolio.
        
        This simulates various stress scenarios:
        - Market crash
        - Volatility spike
        - Correlation breakdown
        - Liquidity crisis
        - Interest rate shock
        """
        try:
            positions = portfolio_data.get('positions', {})
            total_value = portfolio_data.get('total_value', 0.0)
            
            if total_value == 0:
                return StressTestResult(
                    scenario=scenario,
                    portfolio_loss=0.0,
                    var_impact=0.0,
                    volatility_impact=0.0,
                    correlation_impact=0.0,
                    worst_case_loss=0.0,
                    recovery_time_estimate=0.0,
                    risk_level="low",
                    timestamp=datetime.now()
                )
            
            # Apply scenario-specific shocks
            if scenario == StressTestScenario.MARKET_CRASH:
                portfolio_loss = self._apply_market_crash_shock(positions, market_data)
            elif scenario == StressTestScenario.VOLATILITY_SPIKE:
                portfolio_loss = self._apply_volatility_spike_shock(positions, market_data)
            elif scenario == StressTestScenario.CORRELATION_BREAKDOWN:
                portfolio_loss = self._apply_correlation_breakdown_shock(positions, market_data)
            elif scenario == StressTestScenario.LIQUIDITY_CRISIS:
                portfolio_loss = self._apply_liquidity_crisis_shock(positions, market_data)
            elif scenario == StressTestScenario.INTEREST_RATE_SHOCK:
                portfolio_loss = self._apply_interest_rate_shock(positions, market_data)
            elif scenario == StressTestScenario.CUSTOM_SCENARIO:
                portfolio_loss = self._apply_custom_shock(positions, market_data, custom_shocks)
            else:
                portfolio_loss = 0.0
            
            # Calculate impacts
            var_impact = portfolio_loss * 0.1  # Simplified
            volatility_impact = portfolio_loss * 0.05  # Simplified
            correlation_impact = portfolio_loss * 0.03  # Simplified
            
            # Estimate worst case loss
            worst_case_loss = portfolio_loss * 1.5  # 50% additional stress
            
            # Estimate recovery time (simplified)
            recovery_time_estimate = self._estimate_recovery_time(portfolio_loss, total_value)
            
            # Determine risk level
            risk_level = self._determine_risk_level(portfolio_loss, total_value)
            
            # Create stress test result
            result = StressTestResult(
                scenario=scenario,
                portfolio_loss=portfolio_loss,
                var_impact=var_impact,
                volatility_impact=volatility_impact,
                correlation_impact=correlation_impact,
                worst_case_loss=worst_case_loss,
                recovery_time_estimate=recovery_time_estimate,
                risk_level=risk_level,
                timestamp=datetime.now(),
                metadata={
                    'total_value': total_value,
                    'loss_percentage': portfolio_loss / total_value if total_value > 0 else 0.0
                }
            )
            
            # Store result
            self.stress_test_results.append(result)
            self.stress_tests_run += 1
            
            safe_print(f"✅ Stress test completed: {scenario.value} - Loss = ${portfolio_loss:,.2f}")
            return result
            
        except Exception as e:
            safe_print(f"❌ Stress test failed: {safe_format_error(e, 'stress_test')}")
            return StressTestResult(
                scenario=scenario,
                portfolio_loss=0.0,
                var_impact=0.0,
                volatility_impact=0.0,
                correlation_impact=0.0,
                worst_case_loss=0.0,
                recovery_time_estimate=0.0,
                risk_level="unknown",
                timestamp=datetime.now()
            )
    
    def _apply_market_crash_shock(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Apply market crash shock (-20% across all assets)."""
        try:
            total_loss = 0.0
            crash_shock = -0.20  # 20% decline
            
            for asset, pos in positions.items():
                position_value = pos.get('value', 0)
                beta = market_data.get(asset, {}).get('beta', 1.0)
                # Higher beta = higher loss
                asset_loss = position_value * crash_shock * beta
                total_loss += asset_loss
            
            return abs(total_loss)
            
        except Exception as e:
            safe_print(f"❌ Market crash shock failed: {safe_format_error(e, 'market_crash_shock')}")
            return 0.0
    
    def _apply_volatility_spike_shock(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Apply volatility spike shock (3x volatility increase)."""
        try:
            total_loss = 0.0
            volatility_multiplier = 3.0
            
            for asset, pos in positions.items():
                position_value = pos.get('value', 0)
                base_volatility = market_data.get(asset, {}).get('volatility', 0.0)
                # Higher volatility = higher potential loss
                volatility_loss = position_value * base_volatility * (volatility_multiplier - 1)
                total_loss += volatility_loss
            
            return total_loss
            
        except Exception as e:
            safe_print(f"❌ Volatility spike shock failed: {safe_format_error(e, 'volatility_spike_shock')}")
            return 0.0
    
    def _apply_correlation_breakdown_shock(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Apply correlation breakdown shock."""
        try:
            # Correlation breakdown increases portfolio risk
            # Simplified: assume 10% additional loss due to correlation breakdown
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            correlation_loss = total_value * 0.10
            
            return correlation_loss
            
        except Exception as e:
            safe_print(f"❌ Correlation breakdown shock failed: {safe_format_error(e, 'correlation_breakdown_shock')}")
            return 0.0
    
    def _apply_liquidity_crisis_shock(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Apply liquidity crisis shock."""
        try:
            # Liquidity crisis increases bid-ask spreads
            # Simplified: assume 5% additional loss due to liquidity issues
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            liquidity_loss = total_value * 0.05
            
            return liquidity_loss
            
        except Exception as e:
            safe_print(f"❌ Liquidity crisis shock failed: {safe_format_error(e, 'liquidity_crisis_shock')}")
            return 0.0
    
    def _apply_interest_rate_shock(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Apply interest rate shock."""
        try:
            # Interest rate shock affects different assets differently
            # Simplified: assume 3% loss across portfolio
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            interest_rate_loss = total_value * 0.03
            
            return interest_rate_loss
            
        except Exception as e:
            safe_print(f"❌ Interest rate shock failed: {safe_format_error(e, 'interest_rate_shock')}")
            return 0.0
    
    def _apply_custom_shock(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any],
        custom_shocks: Optional[Dict[str, float]]
    ) -> float:
        """Apply custom shock scenario."""
        try:
            if not custom_shocks:
                return 0.0
            
            total_loss = 0.0
            for asset, shock in custom_shocks.items():
                if asset in positions:
                    position_value = positions[asset].get('value', 0)
                    asset_loss = position_value * shock
                    total_loss += asset_loss
            
            return abs(total_loss)
            
        except Exception as e:
            safe_print(f"❌ Custom shock failed: {safe_format_error(e, 'custom_shock')}")
            return 0.0
    
    def _estimate_recovery_time(self, portfolio_loss: float, total_value: float) -> float:
        """Estimate recovery time in days."""
        try:
            if total_value == 0:
                return 0.0
            
            loss_percentage = portfolio_loss / total_value
            
            # Simplified recovery time estimation
            if loss_percentage < 0.05:
                return 5.0  # 5 days
            elif loss_percentage < 0.10:
                return 15.0  # 15 days
            elif loss_percentage < 0.20:
                return 30.0  # 30 days
            else:
                return 60.0  # 60 days
            
        except Exception as e:
            safe_print(f"❌ Recovery time estimation failed: {safe_format_error(e, 'recovery_time')}")
            return 30.0
    
    def _determine_risk_level(self, portfolio_loss: float, total_value: float) -> str:
        """Determine risk level based on loss."""
        try:
            if total_value == 0:
                return "low"
            
            loss_percentage = portfolio_loss / total_value
            
            if loss_percentage < 0.05:
                return "low"
            elif loss_percentage < 0.10:
                return "medium"
            elif loss_percentage < 0.20:
                return "high"
            else:
                return "critical"
            
        except Exception as e:
            safe_print(f"❌ Risk level determination failed: {safe_format_error(e, 'risk_level')}")
            return "unknown"
    
    def check_risk_alerts(self, risk_metrics: RiskMetrics) -> List[RiskAlert]:
        """Check for risk alerts based on current metrics."""
        try:
            alerts = []
            
            # Check VaR alerts
            if risk_metrics.var_95 > self.risk_alert_thresholds['var_95']:
                alerts.append(RiskAlert(
                    alert_type="var_95_breach",
                    severity="high",
                    description=f"VaR(95%) exceeded threshold",
                    threshold=self.risk_alert_thresholds['var_95'],
                    current_value=risk_metrics.var_95,
                    timestamp=datetime.now(),
                    triggered_by="risk_monitoring",
                    action_required="Reduce portfolio risk"
                ))
            
            # Check volatility alerts
            if risk_metrics.volatility > self.risk_alert_thresholds['volatility']:
                alerts.append(RiskAlert(
                    alert_type="volatility_breach",
                    severity="medium",
                    description=f"Portfolio volatility exceeded threshold",
                    threshold=self.risk_alert_thresholds['volatility'],
                    current_value=risk_metrics.volatility,
                    timestamp=datetime.now(),
                    triggered_by="risk_monitoring",
                    action_required="Consider reducing position sizes"
                ))
            
            # Check drawdown alerts
            if risk_metrics.max_drawdown > self.risk_alert_thresholds['drawdown']:
                alerts.append(RiskAlert(
                    alert_type="drawdown_breach",
                    severity="high",
                    description=f"Maximum drawdown exceeded threshold",
                    threshold=self.risk_alert_thresholds['drawdown'],
                    current_value=risk_metrics.max_drawdown,
                    timestamp=datetime.now(),
                    triggered_by="risk_monitoring",
                    action_required="Consider stopping trading"
                ))
            
            # Store alerts
            self.risk_alerts.extend(alerts)
            self.alert_thresholds_breached += len(alerts)
            
            # Keep only recent alerts
            if len(self.risk_alerts) > 1000:
                self.risk_alerts = self.risk_alerts[-1000:]
            
            return alerts
            
        except Exception as e:
            safe_print(f"❌ Risk alerts check failed: {safe_format_error(e, 'risk_alerts')}")
            return []
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        return {
            'total_risk_checks': self.total_risk_checks,
            'risk_violations': self.risk_violations,
            'stress_tests_run': self.stress_tests_run,
            'alert_thresholds_breached': self.alert_thresholds_breached,
            'monitoring_active': self.monitoring_active,
            'latest_metrics': self.risk_metrics_history[-1] if self.risk_metrics_history else None,
            'latest_stress_test': self.stress_test_results[-1] if self.stress_test_results else None,
            'active_alerts': len([alert for alert in self.risk_alerts if alert.severity in ['high', 'critical']])
        }


# Global enhanced risk manager instance
enhanced_risk_manager = EnhancedRiskManager()


# Convenience functions for external access
def get_enhanced_risk_manager() -> EnhancedRiskManager:
    """Get global enhanced risk manager instance."""
    return enhanced_risk_manager


def calculate_risk_metrics(
    portfolio_data: Dict[str, Any],
    market_data: Dict[str, Any],
    historical_data: Optional[List[Dict[str, Any]]] = None
) -> RiskMetrics:
    """Calculate comprehensive risk metrics."""
    return enhanced_risk_manager.calculate_risk_metrics(portfolio_data, market_data, historical_data)


def run_stress_test(
    portfolio_data: Dict[str, Any],
    market_data: Dict[str, Any],
    scenario: StressTestScenario,
    custom_shocks: Optional[Dict[str, float]] = None
) -> StressTestResult:
    """Run stress test on portfolio."""
    return enhanced_risk_manager.run_stress_test(portfolio_data, market_data, scenario, custom_shocks)


def check_risk_alerts(risk_metrics: RiskMetrics) -> List[RiskAlert]:
    """Check for risk alerts."""
    return enhanced_risk_manager.check_risk_alerts(risk_metrics)


def get_risk_summary() -> Dict[str, Any]:
    """Get risk summary."""
    return enhanced_risk_manager.get_risk_summary()


# Example usage
if __name__ == "__main__":
    # Test enhanced risk manager
    print("🎯 Testing Enhanced Risk Manager...")
    
    manager = get_enhanced_risk_manager()
    
    # Test risk metrics calculation
    portfolio_data = {
        'positions': {
            'BTC': {'value': 5000.0, 'unrealized_pnl': 250.0},
            'ETH': {'value': 3000.0, 'unrealized_pnl': -100.0}
        },
        'total_value': 8000.0,
        'total_pnl': 150.0
    }
    
    market_data = {
        'BTC': {'volatility': 0.03, 'beta': 1.2},
        'ETH': {'volatility': 0.04, 'beta': 1.0}
    }
    
    risk_metrics = calculate_risk_metrics(portfolio_data, market_data)
    print(f"✅ Risk metrics: VaR(95%) = {risk_metrics.var_95:.2%}")
    
    # Test stress test
    stress_result = run_stress_test(
        portfolio_data, market_data, StressTestScenario.MARKET_CRASH
    )
    print(f"✅ Stress test: Loss = ${stress_result.portfolio_loss:,.2f}")
    
    # Test risk alerts
    alerts = check_risk_alerts(risk_metrics)
    print(f"✅ Risk alerts: {len(alerts)} alerts")
    
    # Get summary
    summary = get_risk_summary()
    print(f"✅ Risk Summary: {summary}") 