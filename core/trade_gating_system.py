"""
Trade Gating System - Schwabot Trading Intelligence
==================================================

Final approval mechanism for trades using Lantern Core Risk Profiles,
integrated risk scores, and real-time risk monitoring.

Features:
- Multi-stage trade approval process
- Real-time risk monitoring
- Performance analytics integration
- Emergency circuit breakers
- Advanced reporting and alerts

Mathematical Framework:
- Trade Approval Score: TAS = Σ(wᵢ × approval_factorᵢ)
- Risk Threshold: RT = base_threshold × market_condition_multiplier
- Performance Impact: PI = Σ(metricᵢ × weightᵢ)
- Circuit Breaker: CB = f(risk_score, drawdown, volatility)
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.lantern_core_risk_profiles import (
    LanternCoreRiskProfiles, LanternProfile, PositionMetrics, ProfitTarget
)

logger = logging.getLogger(__name__)

class ApprovalStage(Enum):
    """Trade approval stages."""
    INITIAL_SCREENING = "initial_screening"
    RISK_ASSESSMENT = "risk_assessment"
    PROFILE_VALIDATION = "profile_validation"
    PERFORMANCE_CHECK = "performance_check"
    FINAL_APPROVAL = "final_approval"
    REJECTED = "rejected"

class CircuitBreakerStatus(Enum):
    """Circuit breaker status."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class TradeRequest:
    """Trade request data structure."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    timestamp: datetime
    strategy_id: str
    confidence_score: float
    market_data: Dict[str, Any]
    user_profile: LanternProfile
    portfolio_value: float
    request_id: str = field(default_factory=lambda: f"trade_{int(time.time())}")

@dataclass
class ApprovalResult:
    """Trade approval result."""
    approved: bool
    stage: ApprovalStage
    approval_score: float
    risk_score: float
    profile: LanternProfile
    position_size: float
    stop_loss: float
    take_profit: float
    profit_targets: List[ProfitTarget]
    warnings: List[str]
    recommendations: List[str]
    timestamp: datetime

@dataclass
class RiskMetrics:
    """Real-time risk metrics."""
    portfolio_risk: float
    market_risk: float
    position_risk: float
    correlation_risk: float
    liquidity_risk: float
    volatility_risk: float
    integrated_risk_score: float
    circuit_breaker_status: CircuitBreakerStatus

class TradeGatingSystem:
    """
    Trade Gating System - Advanced trade approval mechanism.
    
    Implements multi-stage approval process with real-time risk monitoring
    and performance analytics integration.
    """

    def __init__(self):
        """Initialize Trade Gating System."""
        
        # Initialize Lantern Core Risk Profiles
        self.lantern_profiles = LanternCoreRiskProfiles()
        
        # Approval thresholds
        self.approval_thresholds = {
            ApprovalStage.INITIAL_SCREENING: 0.3,
            ApprovalStage.RISK_ASSESSMENT: 0.5,
            ApprovalStage.PROFILE_VALIDATION: 0.7,
            ApprovalStage.PERFORMANCE_CHECK: 0.8,
            ApprovalStage.FINAL_APPROVAL: 0.9
        }
        
        # Circuit breaker thresholds
        self.circuit_breaker_thresholds = {
            CircuitBreakerStatus.NORMAL: 0.5,
            CircuitBreakerStatus.WARNING: 0.7,
            CircuitBreakerStatus.CRITICAL: 0.85,
            CircuitBreakerStatus.EMERGENCY: 0.95
        }
        
        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.risk_history: List[RiskMetrics] = []
        self.approval_history: List[ApprovalResult] = []
        
        # Real-time monitoring
        self.current_risk_metrics = RiskMetrics(
            portfolio_risk=0.0,
            market_risk=0.0,
            position_risk=0.0,
            correlation_risk=0.0,
            liquidity_risk=0.0,
            volatility_risk=0.0,
            integrated_risk_score=0.0,
            circuit_breaker_status=CircuitBreakerStatus.NORMAL
        )
        
        # System state
        self.system_enabled = True
        self.emergency_mode = False
        self.last_update = datetime.now()
        
        logger.info("Trade Gating System initialized")

    async def process_trade_request(
        self,
        trade_request: TradeRequest
    ) -> ApprovalResult:
        """
        Process trade request through multi-stage approval process.
        
        Args:
            trade_request: Trade request data
            
        Returns:
            Approval result with detailed analysis
        """
        try:
            logger.info(f"Processing trade request: {trade_request.request_id}")
            
            # Stage 1: Initial Screening
            screening_result = await self._initial_screening(trade_request)
            if not screening_result['approved']:
                return self._create_rejection_result(
                    ApprovalStage.INITIAL_SCREENING,
                    screening_result['reason'],
                    trade_request
                )
            
            # Stage 2: Risk Assessment
            risk_result = await self._risk_assessment(trade_request)
            if not risk_result['approved']:
                return self._create_rejection_result(
                    ApprovalStage.RISK_ASSESSMENT,
                    risk_result['reason'],
                    trade_request
                )
            
            # Stage 3: Profile Validation
            profile_result = await self._profile_validation(trade_request)
            if not profile_result['approved']:
                return self._create_rejection_result(
                    ApprovalStage.PROFILE_VALIDATION,
                    profile_result['reason'],
                    trade_request
                )
            
            # Stage 4: Performance Check
            performance_result = await self._performance_check(trade_request)
            if not performance_result['approved']:
                return self._create_rejection_result(
                    ApprovalStage.PERFORMANCE_CHECK,
                    performance_result['reason'],
                    trade_request
                )
            
            # Stage 5: Final Approval
            final_result = await self._final_approval(trade_request)
            if not final_result['approved']:
                return self._create_rejection_result(
                    ApprovalStage.FINAL_APPROVAL,
                    final_result['reason'],
                    trade_request
                )
            
            # Create approval result
            return self._create_approval_result(trade_request, final_result)
            
        except Exception as e:
            logger.error(f"Trade request processing failed: {e}")
            return self._create_rejection_result(
                ApprovalStage.REJECTED,
                f"System error: {str(e)}",
                trade_request
            )

    async def _initial_screening(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Stage 1: Initial screening of trade request."""
        try:
            # Check basic validity
            if trade_request.quantity <= 0:
                return {'approved': False, 'reason': 'Invalid quantity'}
            
            if trade_request.price <= 0:
                return {'approved': False, 'reason': 'Invalid price'}
            
            if trade_request.confidence_score < 0.3:
                return {'approved': False, 'reason': 'Low confidence score'}
            
            # Check circuit breaker status
            if self.current_risk_metrics.circuit_breaker_status == CircuitBreakerStatus.EMERGENCY:
                return {'approved': False, 'reason': 'Emergency circuit breaker active'}
            
            # Check system status
            if not self.system_enabled:
                return {'approved': False, 'reason': 'System disabled'}
            
            return {'approved': True, 'score': 0.8}
            
        except Exception as e:
            logger.error(f"Initial screening failed: {e}")
            return {'approved': False, 'reason': f'Screening error: {str(e)}'}

    async def _risk_assessment(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Stage 2: Risk assessment and analysis."""
        try:
            # Calculate position metrics
            position_metrics = PositionMetrics(
                entry_price=trade_request.price,
                current_price=trade_request.price,
                position_size=trade_request.quantity * trade_request.price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                fees_paid=0.0,
                time_in_position=0,
                risk_score=0.0,
                profit_targets=[]
            )
            
            # Calculate integrated risk score
            risk_score = self.lantern_profiles.calculate_integrated_risk_score(
                trade_request.user_profile,
                position_metrics,
                trade_request.market_data,
                trade_request.portfolio_value
            )
            
            # Check risk thresholds
            risk_params = self.lantern_profiles.risk_profiles[trade_request.user_profile]
            
            if risk_score > risk_params.risk_tolerance:
                return {
                    'approved': False,
                    'reason': f'Risk score {risk_score:.3f} exceeds tolerance {risk_params.risk_tolerance:.3f}'
                }
            
            # Check position size limits
            position_value = trade_request.quantity * trade_request.price
            max_position_value = trade_request.portfolio_value * risk_params.max_position_size
            
            if position_value > max_position_value:
                return {
                    'approved': False,
                    'reason': f'Position size {position_value:.2f} exceeds limit {max_position_value:.2f}'
                }
            
            return {
                'approved': True,
                'score': 1.0 - risk_score,
                'risk_score': risk_score
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {'approved': False, 'reason': f'Risk assessment error: {str(e)}'}

    async def _profile_validation(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Stage 3: Profile validation and Kelly criterion calculation."""
        try:
            # Get profile statistics
            profile_stats = self.lantern_profiles.get_profile_statistics(trade_request.user_profile)
            
            # Calculate Kelly position size
            kelly_size = self.lantern_profiles.calculate_kelly_position_size(
                trade_request.user_profile,
                profile_stats.get('win_rate', 0.5),
                profile_stats.get('avg_profit', 0.02),
                abs(profile_stats.get('avg_loss', 0.01)),
                trade_request.confidence_score,
                trade_request.market_data.get('volatility', 0.02)
            )
            
            # Validate against Kelly criterion
            requested_size = (trade_request.quantity * trade_request.price) / trade_request.portfolio_value
            
            if requested_size > kelly_size * 3.0:  # Allow 3x over Kelly (increased from 1.5x)
                return {
                    'approved': False,
                    'reason': f'Position size {requested_size:.3f} exceeds Kelly criterion {kelly_size:.3f}'
                }
            
            # Generate profit targets
            profit_targets = self.lantern_profiles.generate_multi_target_profit_logic(
                trade_request.user_profile,
                trade_request.price,
                trade_request.quantity,
                trade_request.market_data
            )
            
            return {
                'approved': True,
                'score': 0.9,
                'kelly_size': kelly_size,
                'profit_targets': profit_targets
            }
            
        except Exception as e:
            logger.error(f"Profile validation failed: {e}")
            return {'approved': False, 'reason': f'Profile validation error: {str(e)}'}

    async def _performance_check(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Stage 4: Performance and portfolio impact check."""
        try:
            # Check recent performance
            recent_trades = [t for t in self.trade_history[-20:] if t['profile'] == trade_request.user_profile.value]
            
            if recent_trades:
                recent_performance = np.mean([t.get('net_pnl', 0) for t in recent_trades])
                
                # Reject if recent performance is very poor
                if recent_performance < -0.05:  # 5% loss
                    return {
                        'approved': False,
                        'reason': f'Poor recent performance: {recent_performance:.3f}'
                    }
            
            # Check portfolio concentration
            symbol_positions = [t for t in self.trade_history if t['symbol'] == trade_request.symbol and t['status'] == 'open']
            symbol_exposure = sum([t.get('position_value', 0) for t in symbol_positions])
            
            max_symbol_exposure = trade_request.portfolio_value * 0.3  # 30% max per symbol
            
            if symbol_exposure + (trade_request.quantity * trade_request.price) > max_symbol_exposure:
                return {
                    'approved': False,
                    'reason': f'Symbol exposure would exceed limit of {max_symbol_exposure:.2f}'
                }
            
            return {
                'approved': True,
                'score': 0.85
            }
            
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
            return {'approved': False, 'reason': f'Performance check error: {str(e)}'}

    async def _final_approval(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Stage 5: Final approval with market condition check."""
        try:
            # Check market conditions
            volatility = trade_request.market_data.get('volatility', 0.02)
            volume_ratio = trade_request.market_data.get('volume_ratio', 1.0)
            
            # Reject if market conditions are extreme
            if volatility > 0.15:  # 15% volatility
                return {
                    'approved': False,
                    'reason': f'Extreme volatility: {volatility:.3f}'
                }
            
            if volume_ratio < 0.2:  # 80% volume drop
                return {
                    'approved': False,
                    'reason': f'Low volume: {volume_ratio:.3f}'
                }
            
            # Calculate final approval score
            approval_score = (
                trade_request.confidence_score * 0.4 +
                (1.0 - volatility) * 0.3 +
                volume_ratio * 0.2 +
                0.1  # Base approval
            )
            
            if approval_score < self.approval_thresholds[ApprovalStage.FINAL_APPROVAL]:
                return {
                    'approved': False,
                    'reason': f'Low approval score: {approval_score:.3f}'
                }
            
            return {
                'approved': True,
                'score': approval_score
            }
            
        except Exception as e:
            logger.error(f"Final approval failed: {e}")
            return {'approved': False, 'reason': f'Final approval error: {str(e)}'}

    def _create_approval_result(
        self,
        trade_request: TradeRequest,
        final_result: Dict[str, Any]
    ) -> ApprovalResult:
        """Create approval result with all details."""
        try:
            # Get risk parameters
            risk_params = self.lantern_profiles.risk_profiles[trade_request.user_profile]
            
            # Calculate position metrics
            position_metrics = PositionMetrics(
                entry_price=trade_request.price,
                current_price=trade_request.price,
                position_size=trade_request.quantity * trade_request.price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                fees_paid=0.0,
                time_in_position=0,
                risk_score=0.0,
                profit_targets=[]
            )
            
            # Calculate risk score
            risk_score = self.lantern_profiles.calculate_integrated_risk_score(
                trade_request.user_profile,
                position_metrics,
                trade_request.market_data,
                trade_request.portfolio_value
            )
            
            # Generate profit targets
            profit_targets = self.lantern_profiles.generate_multi_target_profit_logic(
                trade_request.user_profile,
                trade_request.price,
                trade_request.quantity,
                trade_request.market_data
            )
            
            # Calculate stop loss and take profit
            stop_loss = trade_request.price * (1 - risk_params.stop_loss_pct)
            take_profit = trade_request.price * (1 + risk_params.take_profit_pct)
            
            return ApprovalResult(
                approved=True,
                stage=ApprovalStage.FINAL_APPROVAL,
                approval_score=final_result['score'],
                risk_score=risk_score,
                profile=trade_request.user_profile,
                position_size=trade_request.quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                profit_targets=profit_targets,
                warnings=[],
                recommendations=[
                    f"Monitor risk score: {risk_score:.3f}",
                    f"Target profit: {risk_params.take_profit_pct*100:.1f}%",
                    f"Stop loss: {risk_params.stop_loss_pct*100:.1f}%"
                ],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Approval result creation failed: {e}")
            return self._create_rejection_result(
                ApprovalStage.REJECTED,
                f"Result creation error: {str(e)}",
                trade_request
            )

    def _create_rejection_result(
        self,
        stage: ApprovalStage,
        reason: str,
        trade_request: TradeRequest
    ) -> ApprovalResult:
        """Create rejection result."""
        return ApprovalResult(
            approved=False,
            stage=stage,
            approval_score=0.0,
            risk_score=1.0,
            profile=trade_request.user_profile,
            position_size=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            profit_targets=[],
            warnings=[reason],
            recommendations=["Review trade parameters", "Check market conditions"],
            timestamp=datetime.now()
        )

    def update_risk_metrics(
        self,
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> RiskMetrics:
        """
        Update real-time risk metrics.
        
        Args:
            portfolio_data: Current portfolio data
            market_data: Current market data
            
        Returns:
            Updated risk metrics
        """
        try:
            # Calculate portfolio risk
            portfolio_risk = self._calculate_portfolio_risk(portfolio_data)
            
            # Calculate market risk
            market_risk = self._calculate_market_risk(market_data)
            
            # Calculate position risk
            position_risk = self._calculate_position_risk(portfolio_data)
            
            # Calculate correlation risk
            correlation_risk = self._calculate_correlation_risk(portfolio_data)
            
            # Calculate liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(market_data)
            
            # Calculate volatility risk
            volatility_risk = market_data.get('volatility', 0.0)
            
            # Calculate integrated risk score
            integrated_risk_score = (
                portfolio_risk * 0.3 +
                market_risk * 0.25 +
                position_risk * 0.2 +
                correlation_risk * 0.1 +
                liquidity_risk * 0.1 +
                volatility_risk * 0.05
            )
            
            # Determine circuit breaker status
            circuit_breaker_status = self._determine_circuit_breaker_status(integrated_risk_score)
            
            # Update current metrics
            self.current_risk_metrics = RiskMetrics(
                portfolio_risk=portfolio_risk,
                market_risk=market_risk,
                position_risk=position_risk,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                volatility_risk=volatility_risk,
                integrated_risk_score=integrated_risk_score,
                circuit_breaker_status=circuit_breaker_status
            )
            
            # Update history
            self.risk_history.append(self.current_risk_metrics)
            
            # Keep only last 1000 entries
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-1000:]
            
            self.last_update = datetime.now()
            
            return self.current_risk_metrics
            
        except Exception as e:
            logger.error(f"Risk metrics update failed: {e}")
            return self.current_risk_metrics

    def _calculate_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate portfolio risk score."""
        try:
            # Extract portfolio metrics
            total_value = portfolio_data.get('total_value', 1.0)
            cash_ratio = portfolio_data.get('cash_ratio', 0.5)
            diversification_score = portfolio_data.get('diversification_score', 0.5)
            
            # Calculate portfolio risk
            portfolio_risk = (
                (1.0 - cash_ratio) * 0.4 +
                (1.0 - diversification_score) * 0.3 +
                0.3  # Base risk
            )
            
            return min(1.0, max(0.0, portfolio_risk))
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation failed: {e}")
            return 0.5

    def _calculate_market_risk(self, market_data: Dict[str, Any]) -> float:
        """Calculate market risk score."""
        try:
            volatility = market_data.get('volatility', 0.02)
            trend_strength = market_data.get('trend_strength', 0.5)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            market_risk = (
                volatility * 0.4 +
                (1.0 - trend_strength) * 0.3 +
                (1.0 - volume_ratio) * 0.3
            )
            
            return min(1.0, max(0.0, market_risk))
            
        except Exception as e:
            logger.error(f"Market risk calculation failed: {e}")
            return 0.5

    def _calculate_position_risk(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate position risk score."""
        try:
            open_positions = portfolio_data.get('open_positions', [])
            
            if not open_positions:
                return 0.0
            
            # Calculate average position size
            total_value = portfolio_data.get('total_value', 1.0)
            avg_position_size = sum([p.get('value', 0) for p in open_positions]) / len(open_positions) / total_value
            
            # Calculate position concentration
            max_position = max([p.get('value', 0) for p in open_positions]) / total_value if open_positions else 0.0
            
            position_risk = (
                avg_position_size * 0.6 +
                max_position * 0.4
            )
            
            return min(1.0, max(0.0, position_risk))
            
        except Exception as e:
            logger.error(f"Position risk calculation failed: {e}")
            return 0.5

    def _calculate_correlation_risk(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate correlation risk score."""
        try:
            # Simplified correlation risk calculation
            positions = portfolio_data.get('open_positions', [])
            
            if len(positions) < 2:
                return 0.0
            
            # Calculate average correlation (simplified)
            correlation_risk = 0.3  # Base correlation risk
            
            return min(1.0, max(0.0, correlation_risk))
            
        except Exception as e:
            logger.error(f"Correlation risk calculation failed: {e}")
            return 0.3

    def _calculate_liquidity_risk(self, market_data: Dict[str, Any]) -> float:
        """Calculate liquidity risk score."""
        try:
            volume_ratio = market_data.get('volume_ratio', 1.0)
            spread = market_data.get('spread', 0.001)
            
            liquidity_risk = (
                (1.0 - volume_ratio) * 0.7 +
                spread * 100 * 0.3  # Normalize spread
            )
            
            return min(1.0, max(0.0, liquidity_risk))
            
        except Exception as e:
            logger.error(f"Liquidity risk calculation failed: {e}")
            return 0.3

    def _determine_circuit_breaker_status(self, risk_score: float) -> CircuitBreakerStatus:
        """Determine circuit breaker status based on risk score."""
        try:
            if risk_score >= self.circuit_breaker_thresholds[CircuitBreakerStatus.EMERGENCY]:
                return CircuitBreakerStatus.EMERGENCY
            elif risk_score >= self.circuit_breaker_thresholds[CircuitBreakerStatus.CRITICAL]:
                return CircuitBreakerStatus.CRITICAL
            elif risk_score >= self.circuit_breaker_thresholds[CircuitBreakerStatus.WARNING]:
                return CircuitBreakerStatus.WARNING
            else:
                return CircuitBreakerStatus.NORMAL
                
        except Exception as e:
            logger.error(f"Circuit breaker status determination failed: {e}")
            return CircuitBreakerStatus.NORMAL

    def record_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """
        Record trade result for performance tracking.
        
        Args:
            trade_result: Trade result data
        """
        try:
            self.trade_history.append({
                'timestamp': datetime.now(),
                **trade_result
            })
            
            # Keep only last 1000 trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            logger.error(f"Trade result recording failed: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            return {
                'system_enabled': self.system_enabled,
                'emergency_mode': self.emergency_mode,
                'circuit_breaker_status': self.current_risk_metrics.circuit_breaker_status.value,
                'integrated_risk_score': self.current_risk_metrics.integrated_risk_score,
                'last_update': self.last_update.isoformat(),
                'total_trades_processed': len(self.trade_history),
                'total_approvals': len([r for r in self.approval_history if r.approved]),
                'total_rejections': len([r for r in self.approval_history if not r.approved])
            }
            
        except Exception as e:
            logger.error(f"System status retrieval failed: {e}")
            return {}

    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics."""
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'avg_loss': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'profile_performance': {}
                }
            
            # Calculate overall statistics
            total_trades = len(self.trade_history)
            winning_trades = [t for t in self.trade_history if t.get('net_pnl', 0) > 0]
            losing_trades = [t for t in self.trade_history if t.get('net_pnl', 0) <= 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
            
            avg_profit = np.mean([t.get('net_pnl', 0) for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t.get('net_pnl', 0) for t in losing_trades]) if losing_trades else 0.0
            
            # Calculate Sharpe ratio
            returns = [t.get('net_pnl', 0) for t in self.trade_history]
            if returns:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
            
            # Calculate profile performance
            profile_performance = {}
            for profile in LanternProfile:
                profile_trades = [t for t in self.trade_history if t.get('profile') == profile.value]
                if profile_trades:
                    profile_wins = [t for t in profile_trades if t.get('net_pnl', 0) > 0]
                    profile_win_rate = len(profile_wins) / len(profile_trades)
                    profile_avg_profit = np.mean([t.get('net_pnl', 0) for t in profile_trades])
                    
                    profile_performance[profile.value] = {
                        'total_trades': len(profile_trades),
                        'win_rate': profile_win_rate,
                        'avg_profit': profile_avg_profit
                    }
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'profile_performance': profile_performance
            }
            
        except Exception as e:
            logger.error(f"Performance analytics calculation failed: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profile_performance': {}
            } 