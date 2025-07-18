"""
Lantern Core Risk Profiles - Schwabot Trading Intelligence
========================================================

Implements the four Lantern risk profiles (Green, Blue, Red, Orange) with
Kelly criterion position sizing, integrated risk scores, and multi-target profit logic.

Features:
- Green Lantern: Conservative profile (low risk, steady gains)
- Blue Lantern: Balanced profile (moderate risk/reward)
- Red Lantern: Aggressive profile (high risk, high reward)
- Orange Lantern: Emergency recovery profile

Mathematical Framework:
- Kelly Criterion: f* = (bp - q) / b
- Integrated Risk Score: IRS = Σ(wᵢ × risk_factorᵢ)
- Multi-Target Profit: MTP = Σ(targetᵢ × probabilityᵢ)
- Fee-Aware P&L: Net_P&L = Gross_P&L - Σ(feesᵢ)
"""

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class LanternProfile(Enum):
    """Lantern risk profiles for different trading strategies."""
    GREEN = "green"      # Conservative - Low risk, steady gains
    BLUE = "blue"        # Balanced - Moderate risk/reward
    RED = "red"          # Aggressive - High risk, high reward
    ORANGE = "orange"    # Emergency recovery - Crisis management

@dataclass
class RiskParameters:
    """Risk parameters for each Lantern profile."""
    max_position_size: float
    stop_loss_pct: float
    take_profit_pct: float
    max_drawdown: float
    kelly_factor: float
    risk_tolerance: float
    liquidity_cap: float
    emergency_exit_threshold: float

@dataclass
class ProfitTarget:
    """Multi-target profit structure."""
    target_pct: float
    probability: float
    exit_strategy: str
    time_horizon: int  # seconds

@dataclass
class FeeStructure:
    """Fee structure for fee-aware P&L calculations."""
    maker_fee: float
    taker_fee: float
    withdrawal_fee: float
    network_fee: float

@dataclass
class PositionMetrics:
    """Position metrics for risk assessment."""
    entry_price: float
    current_price: float
    position_size: float
    unrealized_pnl: float
    realized_pnl: float
    fees_paid: float
    time_in_position: int
    risk_score: float
    profit_targets: List[ProfitTarget]

class LanternCoreRiskProfiles:
    """
    Lantern Core Risk Profiles - Advanced risk management system.
    
    Implements four distinct risk profiles with Kelly criterion position sizing,
    integrated risk scores, and multi-target profit logic.
    """

    def __init__(self):
        """Initialize Lantern Core Risk Profiles."""
        
        # Define risk parameters for each profile
        self.risk_profiles = {
            LanternProfile.GREEN: RiskParameters(
                max_position_size=0.05,      # 5% max position
                stop_loss_pct=0.015,         # 1.5% stop loss
                take_profit_pct=0.025,       # 2.5% take profit
                max_drawdown=0.10,           # 10% max drawdown
                kelly_factor=0.5,            # Conservative Kelly
                risk_tolerance=0.2,          # Low risk tolerance
                liquidity_cap=0.20,          # 20% liquidity cap
                emergency_exit_threshold=0.05  # 5% emergency exit
            ),
            LanternProfile.BLUE: RiskParameters(
                max_position_size=0.10,      # 10% max position
                stop_loss_pct=0.025,         # 2.5% stop loss
                take_profit_pct=0.040,       # 4.0% take profit
                max_drawdown=0.15,           # 15% max drawdown
                kelly_factor=0.75,           # Balanced Kelly
                risk_tolerance=0.4,          # Moderate risk tolerance
                liquidity_cap=0.35,          # 35% liquidity cap
                emergency_exit_threshold=0.08  # 8% emergency exit
            ),
            LanternProfile.RED: RiskParameters(
                max_position_size=0.20,      # 20% max position
                stop_loss_pct=0.040,         # 4.0% stop loss
                take_profit_pct=0.060,       # 6.0% take profit
                max_drawdown=0.25,           # 25% max drawdown
                kelly_factor=1.0,            # Full Kelly
                risk_tolerance=0.7,          # High risk tolerance
                liquidity_cap=0.50,          # 50% liquidity cap
                emergency_exit_threshold=0.12  # 12% emergency exit
            ),
            LanternProfile.ORANGE: RiskParameters(
                max_position_size=0.30,      # 30% max position (recovery)
                stop_loss_pct=0.060,         # 6.0% stop loss
                take_profit_pct=0.100,       # 10.0% take profit
                max_drawdown=0.40,           # 40% max drawdown
                kelly_factor=1.5,            # Aggressive Kelly
                risk_tolerance=0.9,          # Very high risk tolerance
                liquidity_cap=0.70,          # 70% liquidity cap
                emergency_exit_threshold=0.20  # 20% emergency exit
            )
        }

        # Default fee structure (Coinbase Pro)
        self.default_fees = FeeStructure(
            maker_fee=0.004,    # 0.4% maker fee
            taker_fee=0.006,    # 0.6% taker fee
            withdrawal_fee=0.0005,  # 0.05% withdrawal fee
            network_fee=0.0001   # 0.01% network fee
        )

        # Performance tracking
        self.performance_history: Dict[LanternProfile, List[Dict[str, Any]]] = {
            profile: [] for profile in LanternProfile
        }

        logger.info("Lantern Core Risk Profiles initialized")

    def calculate_kelly_position_size(
        self,
        profile: LanternProfile,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float,
        volatility: float
    ) -> float:
        """
        Calculate Kelly criterion position size for given profile.
        
        Args:
            profile: Lantern risk profile
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade percentage
            avg_loss: Average losing trade percentage
            confidence: Current signal confidence (0-1)
            volatility: Current market volatility
            
        Returns:
            Kelly position size as fraction of portfolio
        """
        try:
            risk_params = self.risk_profiles[profile]
            
            # Kelly Criterion: f* = (bp - q) / b
            if avg_loss <= 0:
                # If no historical losses, use a conservative approach
                kelly_fraction = min(0.1, win_rate * 0.2)  # Conservative Kelly
            else:
                b = avg_win / avg_loss  # Win/loss ratio
                p = win_rate           # Win probability
                q = 1 - win_rate       # Loss probability
                
                # Calculate Kelly fraction
                kelly_fraction = (b * p - q) / b
                kelly_fraction = max(0.0, kelly_fraction)  # No negative sizing
            
            # Apply profile-specific Kelly factor
            kelly_fraction *= risk_params.kelly_factor
            
            # Apply confidence adjustment
            kelly_fraction *= confidence
            
            # Apply volatility adjustment (reduce size in high volatility)
            volatility_adjustment = 1.0 / (1.0 + volatility * 10)  # More aggressive adjustment
            kelly_fraction *= volatility_adjustment
            
            # Apply maximum position size constraint
            kelly_fraction = min(kelly_fraction, risk_params.max_position_size)
            
            # Ensure minimum reasonable size
            kelly_fraction = max(kelly_fraction, 0.001)  # At least 0.1%
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Kelly calculation failed: {e}")
            # Return a conservative default
            return 0.01  # 1% default

    def calculate_integrated_risk_score(
        self,
        profile: LanternProfile,
        position_metrics: PositionMetrics,
        market_data: Dict[str, Any],
        portfolio_value: float
    ) -> float:
        """
        Calculate integrated risk score combining multiple risk factors.
        
        Args:
            profile: Lantern risk profile
            position_metrics: Current position metrics
            market_data: Market data including volatility, volume, etc.
            portfolio_value: Total portfolio value
            
        Returns:
            Integrated risk score (0-1, higher = more risky)
        """
        try:
            risk_params = self.risk_profiles[profile]
            
            # Calculate individual risk factors
            price_risk = abs(position_metrics.current_price - position_metrics.entry_price) / position_metrics.entry_price
            position_risk = position_metrics.position_size / portfolio_value
            time_risk = min(position_metrics.time_in_position / 3600, 1.0)  # Normalize to 1 hour
            volatility_risk = market_data.get('volatility', 0.0)
            volume_risk = 1.0 - min(market_data.get('volume_ratio', 1.0), 1.0)
            
            # Weight factors based on profile
            weights = {
                LanternProfile.GREEN: [0.3, 0.3, 0.2, 0.1, 0.1],    # Conservative weights
                LanternProfile.BLUE: [0.25, 0.25, 0.2, 0.15, 0.15], # Balanced weights
                LanternProfile.RED: [0.2, 0.2, 0.2, 0.2, 0.2],      # Aggressive weights
                LanternProfile.ORANGE: [0.15, 0.15, 0.2, 0.25, 0.25] # Recovery weights
            }
            
            w = weights[profile]
            
            # Calculate weighted risk score
            risk_score = (
                w[0] * price_risk +
                w[1] * position_risk +
                w[2] * time_risk +
                w[3] * volatility_risk +
                w[4] * volume_risk
            )
            
            # Normalize to 0-1 range
            risk_score = min(1.0, max(0.0, risk_score))
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Risk score calculation failed: {e}")
            return 0.5

    def generate_multi_target_profit_logic(
        self,
        profile: LanternProfile,
        entry_price: float,
        position_size: float,
        market_data: Dict[str, Any]
    ) -> List[ProfitTarget]:
        """
        Generate multi-target profit logic for given profile.
        
        Args:
            profile: Lantern risk profile
            entry_price: Entry price
            position_size: Position size
            market_data: Market data
            
        Returns:
            List of profit targets with probabilities
        """
        try:
            risk_params = self.risk_profiles[profile]
            volatility = market_data.get('volatility', 0.02)
            
            # Define target structure based on profile
            if profile == LanternProfile.GREEN:
                targets = [
                    ProfitTarget(0.015, 0.6, "conservative", 1800),   # 1.5% in 30 min
                    ProfitTarget(0.025, 0.3, "balanced", 3600),       # 2.5% in 1 hour
                    ProfitTarget(0.035, 0.1, "aggressive", 7200)      # 3.5% in 2 hours
                ]
            elif profile == LanternProfile.BLUE:
                targets = [
                    ProfitTarget(0.025, 0.5, "conservative", 1800),   # 2.5% in 30 min
                    ProfitTarget(0.040, 0.3, "balanced", 3600),       # 4.0% in 1 hour
                    ProfitTarget(0.060, 0.2, "aggressive", 7200)      # 6.0% in 2 hours
                ]
            elif profile == LanternProfile.RED:
                targets = [
                    ProfitTarget(0.040, 0.4, "conservative", 1800),   # 4.0% in 30 min
                    ProfitTarget(0.060, 0.4, "balanced", 3600),       # 6.0% in 1 hour
                    ProfitTarget(0.100, 0.2, "aggressive", 7200)      # 10.0% in 2 hours
                ]
            else:  # ORANGE
                targets = [
                    ProfitTarget(0.060, 0.3, "conservative", 1800),   # 6.0% in 30 min
                    ProfitTarget(0.100, 0.4, "balanced", 3600),       # 10.0% in 1 hour
                    ProfitTarget(0.150, 0.3, "aggressive", 7200)      # 15.0% in 2 hours
                ]
            
            # Adjust probabilities based on volatility
            for target in targets:
                volatility_adjustment = 1.0 / (1.0 + volatility * 10)
                target.probability *= volatility_adjustment
                target.probability = min(1.0, max(0.0, target.probability))
            
            return targets
            
        except Exception as e:
            logger.error(f"Multi-target generation failed: {e}")
            return []

    def calculate_fee_aware_pnl(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float,
        fees: Optional[FeeStructure] = None
    ) -> Dict[str, float]:
        """
        Calculate fee-aware P&L including all transaction costs.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size (in base currency, e.g., BTC amount)
            fees: Fee structure (uses default if None)
            
        Returns:
            Dictionary with gross_pnl, fees_paid, net_pnl
        """
        try:
            if fees is None:
                fees = self.default_fees
            
            # Calculate gross P&L (in quote currency, e.g., USD)
            price_change = (exit_price - entry_price) / entry_price
            position_value = position_size * entry_price  # Value in quote currency
            gross_pnl = position_value * price_change
            
            # Calculate fees (in quote currency)
            entry_fees = position_value * fees.taker_fee
            exit_fees = (position_size * exit_price) * fees.taker_fee
            total_fees = entry_fees + exit_fees
            
            # Calculate net P&L
            net_pnl = gross_pnl - total_fees
            
            return {
                'gross_pnl': gross_pnl,
                'fees_paid': total_fees,
                'net_pnl': net_pnl,
                'fee_percentage': total_fees / position_value
            }
            
        except Exception as e:
            logger.error(f"Fee-aware P&L calculation failed: {e}")
            return {'gross_pnl': 0.0, 'fees_paid': 0.0, 'net_pnl': 0.0, 'fee_percentage': 0.0}

    def should_trigger_emergency_exit(
        self,
        profile: LanternProfile,
        position_metrics: PositionMetrics,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Determine if emergency exit should be triggered.
        
        Args:
            profile: Lantern risk profile
            position_metrics: Current position metrics
            market_data: Market data
            
        Returns:
            True if emergency exit should be triggered
        """
        try:
            risk_params = self.risk_profiles[profile]
            
            # Calculate current loss percentage
            current_loss = (position_metrics.entry_price - position_metrics.current_price) / position_metrics.entry_price
            
            # Check emergency exit threshold
            if current_loss >= risk_params.emergency_exit_threshold:
                return True
            
            # Check for extreme volatility
            volatility = market_data.get('volatility', 0.0)
            if volatility > 0.1:  # 10% volatility threshold
                return True
            
            # Check for volume collapse
            volume_ratio = market_data.get('volume_ratio', 1.0)
            if volume_ratio < 0.3:  # 70% volume drop
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Emergency exit check failed: {e}")
            return False

    def get_profile_recommendation(
        self,
        market_conditions: Dict[str, Any],
        portfolio_performance: Dict[str, float],
        risk_preference: str = "balanced"
    ) -> LanternProfile:
        """
        Get recommended Lantern profile based on market conditions and performance.
        
        Args:
            market_conditions: Current market conditions
            portfolio_performance: Recent portfolio performance
            risk_preference: User risk preference
            
        Returns:
            Recommended Lantern profile
        """
        try:
            volatility = market_conditions.get('volatility', 0.02)
            trend_strength = market_conditions.get('trend_strength', 0.5)
            recent_performance = portfolio_performance.get('recent_return', 0.0)
            drawdown = portfolio_performance.get('current_drawdown', 0.0)
            
            # Base recommendation on risk preference
            if risk_preference == "conservative":
                base_profile = LanternProfile.GREEN
            elif risk_preference == "aggressive":
                base_profile = LanternProfile.RED
            else:
                base_profile = LanternProfile.BLUE
            
            # Adjust based on market conditions
            if volatility > 0.05:  # High volatility
                if base_profile == LanternProfile.RED:
                    base_profile = LanternProfile.BLUE
                elif base_profile == LanternProfile.BLUE:
                    base_profile = LanternProfile.GREEN
            
            # Adjust based on performance
            if drawdown > 0.15:  # High drawdown
                base_profile = LanternProfile.ORANGE
            elif recent_performance < -0.05:  # Poor recent performance
                if base_profile == LanternProfile.RED:
                    base_profile = LanternProfile.BLUE
                elif base_profile == LanternProfile.BLUE:
                    base_profile = LanternProfile.GREEN
            
            return base_profile
            
        except Exception as e:
            logger.error(f"Profile recommendation failed: {e}")
            return LanternProfile.BLUE

    def update_performance_history(
        self,
        profile: LanternProfile,
        trade_result: Dict[str, Any]
    ) -> None:
        """
        Update performance history for given profile.
        
        Args:
            profile: Lantern profile used
            trade_result: Trade result data
        """
        try:
            self.performance_history[profile].append({
                'timestamp': datetime.now(),
                'result': trade_result,
                'profile': profile.value
            })
            
            # Keep only last 1000 trades per profile
            if len(self.performance_history[profile]) > 1000:
                self.performance_history[profile] = self.performance_history[profile][-1000:]
                
        except Exception as e:
            logger.error(f"Performance history update failed: {e}")

    def get_profile_statistics(self, profile: LanternProfile) -> Dict[str, Any]:
        """
        Get performance statistics for given profile.
        
        Args:
            profile: Lantern profile
            
        Returns:
            Performance statistics
        """
        try:
            history = self.performance_history[profile]
            
            if not history:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'avg_loss': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Calculate statistics
            total_trades = len(history)
            winning_trades = [t for t in history if t['result'].get('net_pnl', 0) > 0]
            losing_trades = [t for t in history if t['result'].get('net_pnl', 0) <= 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
            
            avg_profit = np.mean([t['result'].get('net_pnl', 0) for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t['result'].get('net_pnl', 0) for t in losing_trades]) if losing_trades else 0.0
            
            # Calculate Sharpe ratio (simplified)
            returns = [t['result'].get('net_pnl', 0) for t in history]
            if returns:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            } 