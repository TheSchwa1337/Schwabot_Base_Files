"""
Schwabot Anti-Pole Profit Navigator
===================================

Main integration module that combines Anti-Pole Theory calculations
with practical trading decisions and portfolio management.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json

from core.antipole import AntiPoleVector, AntiPoleConfig, AntiPoleState
from core.antipole import ZBEThermalCooldown, ThermalMetrics, ThermalState
from core.antipole import TesseractVisualizer, GlyphPacket, TesseractFrame

logger = logging.getLogger(__name__)

@dataclass
class ProfitOpportunity:
    """Identified profit opportunity from Anti-Pole analysis"""
    id: str
    timestamp: datetime
    opportunity_type: str  # "ENTRY", "EXIT", "HOLD", "SCALE_IN", "SCALE_OUT"
    confidence: float
    profit_tier: str
    expected_return: float
    risk_score: float
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_horizon: Optional[timedelta] = None
    metadata: Dict[str, Any] = None

@dataclass
class PortfolioState:
    """Current portfolio state and metrics"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    btc_position: float
    btc_value: float
    unrealized_pnl: float
    realized_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    active_opportunities: int
    thermal_safety: bool

class AntiPoleProfitNavigator:
    """
    Main Anti-Pole Profit Navigation System
    
    Integrates mathematical Anti-Pole calculations with practical
    trading decisions and comprehensive portfolio management.
    """
    
    def __init__(self, 
                 initial_balance: float = 100000.0,
                 max_position_size: float = 0.25,
                 config: Optional[AntiPoleConfig] = None):
        
        # Core components
        self.antipole_vector = AntiPoleVector(config)
        self.thermal_controller = ZBEThermalCooldown()
        self.tesseract_viz = TesseractVisualizer()
        
        # Portfolio management
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_position_size = max_position_size
        self.btc_position = 0.0
        self.btc_avg_price = 0.0
        
        # Trading history
        self.opportunities = []
        self.executed_trades = []
        self.portfolio_history = []
        
        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.max_drawdown_seen = 0.0
        self.last_portfolio_update = None
        
        # Navigation state
        self.active_opportunities = {}
        self.profit_zones = {}
        self.risk_warnings = []
        
        # Event handlers
        self.opportunity_handlers = []
        self.risk_handlers = []
        
        logger.info(f"AntiPoleProfitNavigator initialized with ${initial_balance:,.2f}")

    async def process_market_tick(self, btc_price: float, volume: float, 
                                timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Process a market tick through the complete Anti-Pole navigation system
        
        Returns comprehensive analysis and any trading recommendations
        """
        timestamp = timestamp or datetime.now()
        
        # Step 1: Get thermal metrics first (safety check)
        thermal_metrics = self.thermal_controller.update()
        
        # Step 2: Process through Anti-Pole calculations
        antipole_state = self.antipole_vector.process_tick(
            btc_price=btc_price,
            volume=volume,
            lambda_i=0.1 if thermal_metrics.cooldown_active else 0.0,  # Reduce activity during cooldown
            f_k=0.3  # Base UFS family echo score - could be dynamic
        )
        
        # Step 3: Update visualization
        frame = self.tesseract_viz.update_frame(
            state=antipole_state,
            btc_price=btc_price,
            volume=volume,
            thermal_metrics=thermal_metrics
        )
        
        # Step 4: Analyze for profit opportunities
        opportunities = self._analyze_profit_opportunities(
            antipole_state, btc_price, volume, thermal_metrics
        )
        
        # Step 5: Update portfolio state
        portfolio_state = self._update_portfolio_state(
            btc_price, thermal_metrics, opportunities
        )
        
        # Step 6: Generate trading recommendations
        recommendations = self._generate_trading_recommendations(
            opportunities, portfolio_state, thermal_metrics
        )
        
        # Step 7: Execute automatic trades if enabled and safe
        executed_trades = []
        if thermal_metrics.is_safe_to_trade():
            executed_trades = await self._execute_automatic_trades(
                recommendations, btc_price
            )
        
        # Step 8: Broadcast updates to visualization
        await self.tesseract_viz.broadcast_frame(frame)
        
        # Step 9: Compile comprehensive tick report
        tick_report = {
            'timestamp': timestamp.isoformat(),
            'market_data': {
                'btc_price': btc_price,
                'volume': volume
            },
            'antipole_state': {
                'delta_psi_bar': antipole_state.delta_psi_bar,
                'icap_probability': antipole_state.icap_probability,
                'hash_entropy': antipole_state.hash_entropy,
                'is_ready': antipole_state.is_ready,
                'profit_tier': antipole_state.profit_tier,
                'phase_lock': antipole_state.phase_lock
            },
            'thermal_state': {
                'thermal_load': thermal_metrics.thermal_load,
                'state': thermal_metrics.state.value,
                'cooldown_active': thermal_metrics.cooldown_active,
                'safe_to_trade': thermal_metrics.is_safe_to_trade()
            },
            'opportunities': [asdict(opp) for opp in opportunities],
            'portfolio_state': asdict(portfolio_state),
            'recommendations': recommendations,
            'executed_trades': executed_trades,
            'visualization': {
                'frame_id': frame.frame_id,
                'active_glyphs': len(frame.glyphs),
                'profit_tier': frame.profit_tier
            }
        }
        
        # Step 10: Log significant events
        self._log_significant_events(tick_report)
        
        return tick_report

    def _analyze_profit_opportunities(self, state: AntiPoleState, btc_price: float, 
                                    volume: float, thermal_metrics: ThermalMetrics) -> List[ProfitOpportunity]:
        """Analyze Anti-Pole state for profit opportunities"""
        opportunities = []
        
        # High confidence entry opportunity
        if state.is_ready and state.profit_tier and not thermal_metrics.cooldown_active:
            # Calculate expected return based on tier
            tier_multipliers = {
                'PLATINUM': 0.15,  # 15% expected return
                'GOLD': 0.10,      # 10% expected return
                'SILVER': 0.06,    # 6% expected return
                'BRONZE': 0.03     # 3% expected return
            }
            
            expected_return = tier_multipliers.get(state.profit_tier, 0.02)
            
            # Calculate position size based on confidence and tier
            base_position_size = self.max_position_size
            confidence_multiplier = state.icap_probability
            tier_multiplier = tier_multipliers.get(state.profit_tier, 0.5) / 0.10  # Normalize to GOLD
            
            position_size = base_position_size * confidence_multiplier * tier_multiplier
            position_size = min(position_size, self.max_position_size)
            
            # Create opportunity
            opportunity = ProfitOpportunity(
                id=f"entry_{state.timestamp.timestamp()}",
                timestamp=state.timestamp,
                opportunity_type="ENTRY",
                confidence=state.icap_probability,
                profit_tier=state.profit_tier,
                expected_return=expected_return,
                risk_score=1.0 - state.icap_probability,
                entry_price=btc_price,
                position_size=position_size,
                stop_loss=btc_price * (1.0 - (expected_return * 0.5)),  # 50% of expected return as stop
                take_profit=btc_price * (1.0 + expected_return),
                time_horizon=timedelta(hours=24),  # Base 24-hour horizon
                metadata={
                    'delta_psi_bar': state.delta_psi_bar,
                    'hash_entropy': state.hash_entropy,
                    'volume': volume,
                    'thermal_safe': thermal_metrics.is_safe_to_trade()
                }
            )
            opportunities.append(opportunity)
        
        # Exit opportunity for existing positions
        if self.btc_position > 0:
            # Check if we should exit based on Anti-Pole degradation
            if state.icap_probability < 0.3 or thermal_metrics.cooldown_active:
                opportunity = ProfitOpportunity(
                    id=f"exit_{state.timestamp.timestamp()}",
                    timestamp=state.timestamp,
                    opportunity_type="EXIT",
                    confidence=1.0 - state.icap_probability,
                    profit_tier="RISK_MANAGEMENT",
                    expected_return=self._calculate_current_return(btc_price),
                    risk_score=1.0 - state.icap_probability,
                    exit_price=btc_price,
                    position_size=self.btc_position,
                    metadata={
                        'reason': 'antipole_degradation' if state.icap_probability < 0.3 else 'thermal_risk',
                        'current_pnl': self._calculate_current_return(btc_price)
                    }
                )
                opportunities.append(opportunity)
        
        # Scale in opportunity during phase lock
        if state.phase_lock and state.profit_tier in ['PLATINUM', 'GOLD'] and self.btc_position > 0:
            # Phase lock suggests stability - good time to scale in
            scale_size = min(self.max_position_size * 0.1, 
                           (self.current_balance * 0.05) / btc_price)
            
            if scale_size > 0:
                opportunity = ProfitOpportunity(
                    id=f"scale_in_{state.timestamp.timestamp()}",
                    timestamp=state.timestamp,
                    opportunity_type="SCALE_IN",
                    confidence=state.icap_probability * 0.8,  # Slightly lower confidence for scaling
                    profit_tier=state.profit_tier,
                    expected_return=0.05,  # Conservative 5% for scaling
                    risk_score=0.3,  # Lower risk due to phase lock
                    entry_price=btc_price,
                    position_size=scale_size,
                    metadata={'phase_lock': True, 'stability_factor': state.recursion_stability}
                )
                opportunities.append(opportunity)
        
        return opportunities

    def _update_portfolio_state(self, btc_price: float, thermal_metrics: ThermalMetrics,
                               opportunities: List[ProfitOpportunity]) -> PortfolioState:
        """Update and return current portfolio state"""
        timestamp = datetime.now()
        
        # Calculate current portfolio value
        btc_value = self.btc_position * btc_price
        total_value = self.current_balance + btc_value
        
        # Calculate P&L
        unrealized_pnl = 0.0
        if self.btc_position > 0 and self.btc_avg_price > 0:
            unrealized_pnl = (btc_price - self.btc_avg_price) * self.btc_position
        
        # Calculate performance metrics
        total_return = (total_value - self.initial_balance) / self.initial_balance
        win_rate = self.winning_trades / max(self.trade_count, 1) if self.trade_count > 0 else 0.0
        
        # Simple Sharpe ratio approximation (would need more history for accurate calculation)
        sharpe_ratio = total_return / max(0.1, abs(total_return)) if total_return != 0 else 0.0
        
        # Update max drawdown
        if total_value < (self.initial_balance - self.max_drawdown_seen):
            self.max_drawdown_seen = self.initial_balance - total_value
        
        max_drawdown = self.max_drawdown_seen / self.initial_balance
        
        portfolio_state = PortfolioState(
            timestamp=timestamp,
            total_value=total_value,
            cash_balance=self.current_balance,
            btc_position=self.btc_position,
            btc_value=btc_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.total_profit,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            active_opportunities=len(opportunities),
            thermal_safety=thermal_metrics.is_safe_to_trade()
        )
        
        self.portfolio_history.append(portfolio_state)
        self.last_portfolio_update = timestamp
        
        return portfolio_state

    def _generate_trading_recommendations(self, opportunities: List[ProfitOpportunity],
                                        portfolio_state: PortfolioState,
                                        thermal_metrics: ThermalMetrics) -> List[Dict[str, Any]]:
        """Generate specific trading recommendations"""
        recommendations = []
        
        if not thermal_metrics.is_safe_to_trade():
            recommendations.append({
                'type': 'THERMAL_WARNING',
                'message': f'System thermal state: {thermal_metrics.state.value}. Trading suspended.',
                'priority': 'HIGH',
                'action': 'WAIT'
            })
            return recommendations
        
        for opportunity in opportunities:
            if opportunity.opportunity_type == "ENTRY" and opportunity.confidence > 0.65:
                recommendations.append({
                    'type': 'ENTRY_SIGNAL',
                    'message': f'{opportunity.profit_tier} entry opportunity detected',
                    'price': opportunity.entry_price,
                    'size': opportunity.position_size,
                    'confidence': opportunity.confidence,
                    'expected_return': f"{opportunity.expected_return*100:.1f}%",
                    'stop_loss': opportunity.stop_loss,
                    'take_profit': opportunity.take_profit,
                    'priority': 'HIGH' if opportunity.profit_tier in ['PLATINUM', 'GOLD'] else 'MEDIUM',
                    'action': 'BUY'
                })
            
            elif opportunity.opportunity_type == "EXIT" and opportunity.confidence > 0.7:
                current_return = self._calculate_current_return(opportunity.exit_price or 0)
                recommendations.append({
                    'type': 'EXIT_SIGNAL',
                    'message': f'Exit signal - Current return: {current_return*100:.1f}%',
                    'price': opportunity.exit_price,
                    'size': opportunity.position_size,
                    'confidence': opportunity.confidence,
                    'current_pnl': current_return,
                    'priority': 'HIGH',
                    'action': 'SELL'
                })
        
        # Portfolio health recommendations
        if portfolio_state.max_drawdown > 0.15:  # 15% drawdown warning
            recommendations.append({
                'type': 'RISK_WARNING',
                'message': f'Maximum drawdown at {portfolio_state.max_drawdown*100:.1f}%',
                'priority': 'HIGH',
                'action': 'REDUCE_RISK'
            })
        
        if portfolio_state.win_rate < 0.4 and self.trade_count > 10:  # Low win rate warning
            recommendations.append({
                'type': 'PERFORMANCE_WARNING',
                'message': f'Win rate below 40% ({portfolio_state.win_rate*100:.1f}%)',
                'priority': 'MEDIUM',
                'action': 'REVIEW_STRATEGY'
            })
        
        return recommendations

    async def _execute_automatic_trades(self, recommendations: List[Dict[str, Any]], 
                                       btc_price: float) -> List[Dict[str, Any]]:
        """Execute automatic trades based on recommendations (simulation)"""
        executed = []
        
        for rec in recommendations:
            if rec['type'] == 'ENTRY_SIGNAL' and rec['priority'] == 'HIGH':
                # Simulate buy execution
                size = rec['size']
                cost = size * btc_price
                
                if cost <= self.current_balance:
                    # Execute buy
                    old_position = self.btc_position
                    old_avg_price = self.btc_avg_price
                    
                    # Update position and average price
                    total_btc = old_position + size
                    total_cost = (old_position * old_avg_price) + cost
                    new_avg_price = total_cost / total_btc if total_btc > 0 else btc_price
                    
                    self.btc_position = total_btc
                    self.btc_avg_price = new_avg_price
                    self.current_balance -= cost
                    self.trade_count += 1
                    
                    executed.append({
                        'type': 'BUY_EXECUTED',
                        'price': btc_price,
                        'size': size,
                        'cost': cost,
                        'new_position': self.btc_position,
                        'new_avg_price': self.btc_avg_price,
                        'confidence': rec['confidence']
                    })
                    
                    logger.info(f"🟢 AUTO-BUY: {size:.6f} BTC at ${btc_price:,.2f} "
                               f"(Total: {self.btc_position:.6f} BTC)")
            
            elif rec['type'] == 'EXIT_SIGNAL' and rec['priority'] == 'HIGH':
                # Simulate sell execution
                if self.btc_position > 0:
                    size = min(rec['size'], self.btc_position)
                    proceeds = size * btc_price
                    
                    # Calculate profit/loss for this trade
                    cost_basis = size * self.btc_avg_price
                    trade_pnl = proceeds - cost_basis
                    
                    # Update position
                    self.btc_position -= size
                    self.current_balance += proceeds
                    self.total_profit += trade_pnl
                    self.trade_count += 1
                    
                    if trade_pnl > 0:
                        self.winning_trades += 1
                    
                    executed.append({
                        'type': 'SELL_EXECUTED',
                        'price': btc_price,
                        'size': size,
                        'proceeds': proceeds,
                        'trade_pnl': trade_pnl,
                        'remaining_position': self.btc_position,
                        'confidence': rec['confidence']
                    })
                    
                    logger.info(f"🔴 AUTO-SELL: {size:.6f} BTC at ${btc_price:,.2f} "
                               f"P&L: ${trade_pnl:,.2f}")
        
        return executed

    def _calculate_current_return(self, current_price: float) -> float:
        """Calculate current return on BTC position"""
        if self.btc_position <= 0 or self.btc_avg_price <= 0:
            return 0.0
        
        return (current_price - self.btc_avg_price) / self.btc_avg_price

    def _log_significant_events(self, tick_report: Dict[str, Any]):
        """Log significant trading events"""
        antipole = tick_report['antipole_state']
        thermal = tick_report['thermal_state']
        
        # Log Anti-Pole ready states
        if antipole['is_ready']:
            logger.info(f"🔥 ANTI-POLE READY: {antipole['profit_tier']} | "
                       f"ICAP: {antipole['icap_probability']:.3f} | "
                       f"Drift: {antipole['delta_psi_bar']:.6f}")
        
        # Log thermal warnings
        if thermal['thermal_load'] > 0.8:
            logger.warning(f"🌡️ HIGH THERMAL LOAD: {thermal['thermal_load']:.3f} | "
                          f"State: {thermal['state']}")
        
        # Log executed trades
        for trade in tick_report['executed_trades']:
            if trade['type'] == 'BUY_EXECUTED':
                logger.info(f"💰 BUY: {trade['size']:.6f} BTC @ ${trade['price']:,.2f}")
            elif trade['type'] == 'SELL_EXECUTED':
                pnl_emoji = "📈" if trade['trade_pnl'] > 0 else "📉"
                logger.info(f"{pnl_emoji} SELL: {trade['size']:.6f} BTC @ ${trade['price']:,.2f} "
                           f"P&L: ${trade['trade_pnl']:,.2f}")

    async def start_visualization_server(self):
        """Start the Tesseract visualization WebSocket server"""
        await self.tesseract_viz.start_websocket_server()

    async def stop_visualization_server(self):
        """Stop the Tesseract visualization WebSocket server"""
        await self.tesseract_viz.stop_websocket_server()

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring"""
        return {
            'portfolio': {
                'total_value': self.current_balance + (self.btc_position * 0),  # Would need current price
                'cash_balance': self.current_balance,
                'btc_position': self.btc_position,
                'btc_avg_price': self.btc_avg_price,
                'total_trades': self.trade_count,
                'winning_trades': self.winning_trades,
                'win_rate': self.winning_trades / max(self.trade_count, 1),
                'total_profit': self.total_profit
            },
            'antipole': self.antipole_vector.get_statistics(),
            'thermal': self.thermal_controller.get_thermal_statistics(),
            'visualization': self.tesseract_viz.get_visualization_statistics(),
            'opportunities': len(self.active_opportunities),
            'last_update': self.last_portfolio_update.isoformat() if self.last_portfolio_update else None
        } 