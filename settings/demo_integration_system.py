"""
Schwabot Demo Integration System
Comprehensive demo mode management with full integration support
"""

import json
import yaml
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque

# Import our components
from settings_controller import SettingsController, get_settings_controller
from vector_validator import VectorValidator, get_vector_validator
from matrix_allocator import MatrixAllocator, get_matrix_allocator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DemoConfiguration:
    """Demo configuration parameters"""
    enabled: bool = True
    mode: str = "full_integration"
    simulation_duration: int = 7200
    tick_interval: float = 3.75
    initial_balance: float = 10000.0
    max_positions: int = 5
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    slippage: float = 0.001
    commission: float = 0.001
    data_source: str = "simulated"
    validation_mode: bool = True


@dataclass
class DemoResult:
    """Result of a demo session"""
    success: bool
    session_id: str
    duration: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    sharpe_ratio: float
    total_return: float
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    timestamp: str


class DemoIntegrationSystem:
    """Comprehensive demo integration system"""
    
    def __init__(self, config_dir: str = "settings"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.settings_controller = get_settings_controller()
        self.vector_validator = get_vector_validator()
        self.matrix_allocator = get_matrix_allocator()
        
        # Load configuration
        self.demo_config = self._load_demo_configuration()
        
        # State tracking
        self.active_sessions = {}
        self.session_history = []
        self.performance_metrics = defaultdict(list)
        
        # Threading
        self.lock = threading.RLock()
        self.running = False
        self.demo_thread = None
        
        # Statistics
        self.total_sessions = 0
        self.successful_sessions = 0
        self.failed_sessions = 0
        
        # Start background monitoring
        self.start_background_monitoring()
    
    def _load_demo_configuration(self) -> DemoConfiguration:
        """Load demo configuration from YAML file"""
        try:
            config_path = self.config_dir / "demo_backtest_mode.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    demo_params = config_data.get('demo_params', {})
                    return DemoConfiguration(**demo_params)
            else:
                logger.warning("Demo configuration file not found, using defaults")
                return DemoConfiguration()
                
        except Exception as e:
            logger.error(f"Error loading demo configuration: {e}")
            return DemoConfiguration()
    
    def start_demo_session(self, session_id: str, scenario: str = "moderate") -> bool:
        """Start a new demo session"""
        try:
            with self.lock:
                if session_id in self.active_sessions:
                    logger.warning(f"Session {session_id} already exists")
                    return False
                
                # Load scenario configuration
                scenario_config = self._load_scenario_config(scenario)
                
                # Create session
                session = {
                    'session_id': session_id,
                    'scenario': scenario,
                    'start_time': datetime.now(),
                    'balance': self.demo_config.initial_balance,
                    'positions': [],
                    'trades': [],
                    'performance_metrics': defaultdict(list),
                    'status': 'running',
                    'config': scenario_config
                }
                
                self.active_sessions[session_id] = session
                self.total_sessions += 1
                
                logger.info(f"Started demo session {session_id} with scenario {scenario}")
                return True
                
        except Exception as e:
            logger.error(f"Error starting demo session: {e}")
            return False
    
    def _load_scenario_config(self, scenario: str) -> Dict[str, Any]:
        """Load scenario configuration"""
        try:
            config_path = self.config_dir / "demo_backtest_mode.yaml"
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                scenarios = config_data.get('backtest_scenarios', {})
                return scenarios.get(scenario, {})
                
        except Exception as e:
            logger.error(f"Error loading scenario config: {e}")
            return {}
    
    def stop_demo_session(self, session_id: str) -> Optional[DemoResult]:
        """Stop a demo session and return results"""
        try:
            with self.lock:
                if session_id not in self.active_sessions:
                    logger.warning(f"Session {session_id} not found")
                    return None
                
                session = self.active_sessions[session_id]
                session['status'] = 'stopped'
                session['end_time'] = datetime.now()
                
                # Calculate session results
                result = self._calculate_session_result(session)
                
                # Record session
                self.session_history.append({
                    'session': session,
                    'result': asdict(result),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update statistics
                if result.success:
                    self.successful_sessions += 1
                else:
                    self.failed_sessions += 1
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                
                # Apply learning
                self._apply_session_learning(result)
                
                logger.info(f"Stopped demo session {session_id}")
                return result
                
        except Exception as e:
            logger.error(f"Error stopping demo session: {e}")
            return None
    
    def _calculate_session_result(self, session: Dict[str, Any]) -> DemoResult:
        """Calculate session results"""
        duration = (session['end_time'] - session['start_time']).total_seconds()
        final_balance = session['balance']
        total_trades = len(session['trades'])
        winning_trades = len([t for t in session['trades'] if t.get('profit', 0) > 0])
        losing_trades = total_trades - winning_trades
        
        # Calculate performance metrics
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            total_return = (final_balance - self.demo_config.initial_balance) / self.demo_config.initial_balance
            
            # Calculate max drawdown
            balances = [self.demo_config.initial_balance]
            for trade in session['trades']:
                balances.append(balances[-1] + trade.get('profit', 0))
            
            max_drawdown = self._calculate_max_drawdown(balances)
            
            # Calculate Sharpe ratio (simplified)
            returns = [t.get('profit', 0) / self.demo_config.initial_balance for t in session['trades']]
            if returns:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            win_rate = 0.0
            total_return = 0.0
            max_drawdown = 0.0
            sharpe_ratio = 0.0
        
        # Determine success
        success = (total_return > 0 and sharpe_ratio > 0.5 and max_drawdown < 0.2)
        
        # Generate recommendations
        recommendations = self._generate_session_recommendations(
            session, total_return, sharpe_ratio, max_drawdown, win_rate
        )
        
        return DemoResult(
            success=success,
            session_id=session['session_id'],
            duration=duration,
            final_balance=final_balance,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            performance_metrics={
                'win_rate': win_rate,
                'avg_trade_profit': np.mean([t.get('profit', 0) for t in session['trades']]) if session['trades'] else 0.0,
                'max_consecutive_losses': self._calculate_max_consecutive_losses(session['trades']),
                'avg_trade_duration': np.mean([t.get('duration', 0) for t in session['trades']]) if session['trades'] else 0.0
            },
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_max_drawdown(self, balances: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not balances:
            return 0.0
        
        peak = balances[0]
        max_dd = 0.0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_max_consecutive_losses(self, trades: List[Dict[str, Any]]) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.get('profit', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _generate_session_recommendations(self, session: Dict[str, Any], 
                                        total_return: float, sharpe_ratio: float, 
                                        max_drawdown: float, win_rate: float) -> List[str]:
        """Generate recommendations based on session performance"""
        recommendations = []
        
        # Return recommendations
        if total_return < 0:
            recommendations.append("Session resulted in losses - review strategy parameters")
        elif total_return < 0.05:
            recommendations.append("Low returns - consider increasing risk or improving strategy")
        
        # Sharpe ratio recommendations
        if sharpe_ratio < 0.5:
            recommendations.append("Low Sharpe ratio - optimize risk-adjusted returns")
        elif sharpe_ratio > 2.0:
            recommendations.append("Very high Sharpe ratio - verify risk assumptions")
        
        # Drawdown recommendations
        if max_drawdown > 0.15:
            recommendations.append("High drawdown - implement better risk management")
        
        # Win rate recommendations
        if win_rate < 0.4:
            recommendations.append("Low win rate - improve entry/exit criteria")
        elif win_rate > 0.8:
            recommendations.append("Very high win rate - consider increasing position sizes")
        
        # Trade frequency recommendations
        if len(session['trades']) < 10:
            recommendations.append("Low trade frequency - review entry conditions")
        elif len(session['trades']) > 100:
            recommendations.append("High trade frequency - consider reducing sensitivity")
        
        return recommendations
    
    def _apply_session_learning(self, result: DemoResult) -> None:
        """Apply learning from session results"""
        try:
            if result.success:
                # Record success
                self.settings_controller.record_backtest_success({
                    'profit': result.final_balance - self.demo_config.initial_balance,
                    'strategy': 'demo_session',
                    'duration': result.duration,
                    'session_id': result.session_id,
                    'performance_metrics': result.performance_metrics
                })
            else:
                # Record failure
                self.settings_controller.record_backtest_failure({
                    'reason': 'demo_session_failure',
                    'loss': self.demo_config.initial_balance - result.final_balance,
                    'strategy': 'demo_session',
                    'duration': result.duration,
                    'session_id': result.session_id,
                    'performance_metrics': result.performance_metrics
                })
                
        except Exception as e:
            logger.error(f"Error applying session learning: {e}")
    
    def run_backtest(self, scenario: str = "moderate", duration: int = None) -> DemoResult:
        """Run a complete backtest"""
        session_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Start session
            if not self.start_demo_session(session_id, scenario):
                raise Exception("Failed to start demo session")
            
            # Simulate trading
            if duration is None:
                duration = self.demo_config.simulation_duration
            
            self._simulate_trading(session_id, duration)
            
            # Stop session and get results
            result = self.stop_demo_session(session_id)
            if result is None:
                raise Exception("Failed to get session results")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            # Clean up session if it exists
            if session_id in self.active_sessions:
                self.stop_demo_session(session_id)
            
            return DemoResult(
                success=False,
                session_id=session_id,
                duration=0.0,
                final_balance=self.demo_config.initial_balance,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                total_return=0.0,
                performance_metrics={},
                recommendations=[f"Backtest failed: {str(e)}"],
                timestamp=datetime.now().isoformat()
            )
    
    def _simulate_trading(self, session_id: str, duration: int) -> None:
        """Simulate trading for the session"""
        try:
            session = self.active_sessions[session_id]
            start_time = time.time()
            
            while time.time() - start_time < duration and session['status'] == 'running':
                # Generate market data
                market_data = self._generate_market_data(session)
                
                # Validate vectors
                if self.demo_config.validation_mode:
                    validation_result = self.vector_validator.validate_vector(
                        market_data, f"session_{session_id}"
                    )
                    if not validation_result.is_valid:
                        logger.debug(f"Invalid vector in session {session_id}")
                        time.sleep(self.demo_config.tick_interval)
                        continue
                
                # Make trading decisions
                trade_decision = self._make_trade_decision(session, market_data)
                
                if trade_decision:
                    # Execute trade
                    trade_result = self._execute_trade(session, trade_decision)
                    session['trades'].append(trade_result)
                    
                    # Update balance
                    session['balance'] += trade_result['profit']
                
                # Update matrix allocation if needed
                if len(session['trades']) % 50 == 0:
                    self._update_matrix_allocation(session)
                
                time.sleep(self.demo_config.tick_interval)
                
        except Exception as e:
            logger.error(f"Error in trading simulation: {e}")
    
    def _generate_market_data(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simulated market data"""
        # Simple market data simulation
        base_price = 50000.0
        volatility = 0.02
        trend = 0.001
        
        # Add some randomness
        price_change = np.random.normal(trend, volatility)
        current_price = base_price * (1 + price_change)
        
        # Generate vector components
        components = [
            current_price / base_price - 1,  # Price change
            np.random.random(),  # Volume factor
            np.random.random(),  # Momentum
            np.random.random(),  # Volatility
            np.random.random()   # Market sentiment
        ]
        
        return {
            'price': current_price,
            'volume': np.random.uniform(0.5, 2.0),
            'timestamp': datetime.now().isoformat(),
            'components': components,
            'market_condition': np.random.choice(['bull', 'bear', 'sideways']),
            'volatility': volatility
        }
    
    def _make_trade_decision(self, session: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make trading decision based on market data"""
        try:
            # Simple trading logic based on price movement
            price_change = market_data['components'][0]
            volume_factor = market_data['components'][1]
            
            # Decision thresholds
            buy_threshold = 0.01
            sell_threshold = -0.01
            
            if price_change > buy_threshold and volume_factor > 1.2:
                return {
                    'action': 'buy',
                    'price': market_data['price'],
                    'size': self.demo_config.risk_per_trade * session['balance'] / market_data['price'],
                    'timestamp': market_data['timestamp'],
                    'reason': 'price_momentum_buy'
                }
            elif price_change < sell_threshold and volume_factor > 1.2:
                return {
                    'action': 'sell',
                    'price': market_data['price'],
                    'size': self.demo_config.risk_per_trade * session['balance'] / market_data['price'],
                    'timestamp': market_data['timestamp'],
                    'reason': 'price_momentum_sell'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error making trade decision: {e}")
            return None
    
    def _execute_trade(self, session: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade"""
        try:
            # Simulate trade execution
            entry_price = decision['price']
            size = decision['size']
            
            # Simulate price movement after trade
            price_change = np.random.normal(0, 0.005)
            exit_price = entry_price * (1 + price_change)
            
            # Calculate profit/loss
            if decision['action'] == 'buy':
                profit = (exit_price - entry_price) * size
            else:
                profit = (entry_price - exit_price) * size
            
            # Apply slippage and commission
            profit -= abs(profit) * (self.demo_config.slippage + self.demo_config.commission)
            
            return {
                'action': decision['action'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'profit': profit,
                'timestamp': decision['timestamp'],
                'duration': np.random.uniform(60, 300),  # 1-5 minutes
                'reason': decision['reason']
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                'action': decision['action'],
                'entry_price': decision['price'],
                'exit_price': decision['price'],
                'size': decision['size'],
                'profit': 0.0,
                'timestamp': decision['timestamp'],
                'duration': 0.0,
                'reason': 'execution_error'
            }
    
    def _update_matrix_allocation(self, session: Dict[str, Any]) -> None:
        """Update matrix allocation for the session"""
        try:
            # Create or update basket for session
            basket_id = f"session_{session['session_id']}"
            
            if basket_id not in self.matrix_allocator.matrix_baskets:
                self.matrix_allocator.create_matrix_basket(
                    basket_id, 
                    session['balance'], 
                    0.2,  # risk budget
                    0.1   # return target
                )
            
            # Optimize allocation
            result = self.matrix_allocator.optimize_allocation(basket_id, "risk_parity")
            
            if result.success:
                logger.debug(f"Updated matrix allocation for session {session['session_id']}")
            
        except Exception as e:
            logger.error(f"Error updating matrix allocation: {e}")
    
    def get_demo_statistics(self) -> Dict[str, Any]:
        """Get demo system statistics"""
        with self.lock:
            if self.total_sessions == 0:
                return {
                    'total_sessions': 0,
                    'success_rate': 0.0,
                    'average_return': 0.0,
                    'average_sharpe': 0.0,
                    'active_sessions': 0
                }
            
            success_rate = self.successful_sessions / self.total_sessions
            
            # Calculate averages from recent sessions
            recent_results = [s['result'] for s in self.session_history[-100:]]
            if recent_results:
                avg_return = np.mean([r['total_return'] for r in recent_results])
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in recent_results])
            else:
                avg_return = avg_sharpe = 0.0
            
            return {
                'total_sessions': self.total_sessions,
                'successful_sessions': self.successful_sessions,
                'failed_sessions': self.failed_sessions,
                'success_rate': success_rate,
                'average_return': avg_return,
                'average_sharpe': avg_sharpe,
                'active_sessions': len(self.active_sessions),
                'recent_sessions': len(self.session_history)
            }
    
    def start_background_monitoring(self) -> None:
        """Start background monitoring"""
        if not self.running:
            self.running = True
            self.demo_thread = threading.Thread(target=self._background_monitoring_loop, daemon=True)
            self.demo_thread.start()
            logger.info("Background demo monitoring started")
    
    def stop_background_monitoring(self) -> None:
        """Stop background monitoring"""
        self.running = False
        if self.demo_thread:
            self.demo_thread.join(timeout=5)
        logger.info("Background demo monitoring stopped")
    
    def _background_monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self.running:
            try:
                # Update performance metrics
                self.performance_metrics['demo_stats'].append(self.get_demo_statistics())
                
                # Auto-stop sessions that exceed duration
                current_time = datetime.now()
                for session_id, session in list(self.active_sessions.items()):
                    duration = (current_time - session['start_time']).total_seconds()
                    if duration > self.demo_config.simulation_duration:
                        logger.info(f"Auto-stopping session {session_id} due to duration limit")
                        self.stop_demo_session(session_id)
                
                # Keep only recent metrics
                if len(self.performance_metrics['demo_stats']) > 100:
                    self.performance_metrics['demo_stats'] = self.performance_metrics['demo_stats'][-100:]
                
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in background monitoring loop: {e}")
                time.sleep(30)
    
    def export_demo_data(self, filepath: str) -> None:
        """Export demo data to a file"""
        with self.lock:
            export_data = {
                'active_sessions': self.active_sessions,
                'session_history': self.session_history,
                'statistics': self.get_demo_statistics(),
                'performance_metrics': dict(self.performance_metrics),
                'configuration': asdict(self.demo_config),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Demo data exported to {filepath}")
    
    def clear_demo_history(self) -> None:
        """Clear demo history"""
        with self.lock:
            self.session_history.clear()
            self.performance_metrics.clear()
            self.total_sessions = 0
            self.successful_sessions = 0
            self.failed_sessions = 0
            logger.info("Demo history cleared")


# Global demo integration system instance
demo_integration_system = DemoIntegrationSystem()


def get_demo_integration_system() -> DemoIntegrationSystem:
    """Get the global demo integration system instance"""
    return demo_integration_system


if __name__ == "__main__":
    # Test the demo integration system
    demo_system = DemoIntegrationSystem()
    
    # Run a test backtest
    print("Running test backtest...")
    result = demo_system.run_backtest("moderate", duration=300)  # 5 minutes
    
    print("Backtest Result:")
    print(json.dumps(asdict(result), indent=2))
    
    print("\nDemo Statistics:")
    print(json.dumps(demo_system.get_demo_statistics(), indent=2)) 