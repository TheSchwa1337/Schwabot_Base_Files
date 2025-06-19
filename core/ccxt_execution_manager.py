"""
CCXT Execution Manager
=====================

Integrates your existing mathematical systems (UnifiedMathematicalProcessor, 
EnhancedFitnessOracle) with real CCXT trading execution.

This bridges the gap between mathematical analysis and actual trade execution.
Enhanced with Windows CLI compatibility for cross-platform reliability.

WINDOWS CLI COMPATIBILITY:
This file implements Windows CLI compatibility handling as documented in WINDOWS_CLI_COMPATIBILITY.md
All emoji usage is handled through the WindowsCliCompatibilityHandler class to ensure
cross-platform compatibility and prevent CLI rendering issues on Windows systems.

NAMING CONVENTIONS:
All components follow descriptive naming conventions as outlined in WINDOWS_CLI_COMPATIBILITY.md
- CCXTExecutionManager (describes actual function)
- MathematicalTradeSignal (describes mathematical trade signal)
- ExecutionResult (describes execution result)
"""

import asyncio
import logging
import time
import platform
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

try:
    import ccxt
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None
    ccxt_async = None

# Import your existing mathematical systems
from .math_core import UnifiedMathematicalProcessor, AnalysisResult
from .unified_api_coordinator import UnifiedAPICoordinator, TradingRequest, APIConfiguration
from ..enhanced_fitness_oracle import EnhancedFitnessOracle, UnifiedFitnessScore
from .mathlib_v3 import SustainmentMathLib, GradedProfitVector

logger = logging.getLogger(__name__)

# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================

class WindowsCliCompatibilityHandler:
    """
    Handles Windows CLI compatibility issues including emoji rendering
    and ASIC implementation for plain text output explanations
    
    Addresses the CLI error issues mentioned in the comprehensive testing:
    - Emoji characters causing encoding errors on Windows
    - Need for ASIC plain text output
    - Cross-platform compatibility for error messages
    """
    
    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment"""
        return (platform.system() == "Windows" and 
                ("cmd" in os.environ.get("COMSPEC", "").lower() or
                 "powershell" in os.environ.get("PSModulePath", "").lower()))
    
    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """
        Print message safely with Windows CLI compatibility
        Implements ASIC plain text output for Windows environments
        
        ASIC Implementation: Application-Specific Integrated Circuit approach
        provides specialized text rendering for Windows CLI environments
        """
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            # ASIC plain text markers for Windows CLI compatibility
            emoji_to_asic_mapping = {
                '✅': '[SUCCESS]',    # Success indicator
                '❌': '[ERROR]',      # Error indicator  
                '🔧': '[PROCESSING]', # Processing indicator
                '🚀': '[LAUNCH]',     # Launch/start indicator
                '🎉': '[COMPLETE]',   # Completion indicator
                '💥': '[CRITICAL]',   # Critical alert
                '⚡': '[FAST]',       # Fast execution
                '🔍': '[SEARCH]',     # Search/analysis
                '📊': '[DATA]',       # Data processing
                '🧪': '[TEST]',       # Testing indicator
                '️': '[TOOLS]',      # Tools/utilities
                '⚖️': '[BALANCE]',    # Balance/measurement
                '🔄': '[CYCLE]',      # Cycle/loop
                '🎯': '[TARGET]',     # Target/goal
                '📈': '[PROFIT]',     # Profit indicator
                '🔥': '[HOT]',        # High activity
                '❄️': '[COOL]',       # Cool/low activity
                '⭐': '[STAR]',       # Important/featured
                '🧠': '[INTELLIGENCE]', # Intelligence/analysis
            }
            
            safe_message = message
            for emoji, asic_replacement in emoji_to_asic_mapping.items():
                safe_message = safe_message.replace(emoji, asic_replacement)
            
            return safe_message
        
        return message
    
    @staticmethod
    def log_safe(logger, level: str, message: str):
        """Log message safely with Windows CLI compatibility"""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        try:
            getattr(logger, level.lower())(safe_message)
        except UnicodeEncodeError:
            # Emergency ASCII fallback for Windows CLI
            ascii_message = safe_message.encode('ascii', errors='replace').decode('ascii')
            getattr(logger, level.lower())(ascii_message)
    
    @staticmethod
    def safe_format_error(error: Exception, context: str = "") -> str:
        """Format error messages safely for Windows CLI"""
        error_message = f"Error: {str(error)}"
        if context:
            error_message += f" | Context: {context}"
        
        return WindowsCliCompatibilityHandler.safe_print(error_message)

class ExecutionStatus(Enum):
    """Trade execution status"""
    PENDING = "pending"
    EXECUTED = "executed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MathematicalTradeSignal:
    """Trade signal backed by mathematical analysis"""
    signal_id: str
    timestamp: datetime
    
    # Mathematical Foundation
    unified_analysis: AnalysisResult
    fitness_score: UnifiedFitnessScore
    graded_vector: GradedProfitVector
    
    # Trading Parameters
    symbol: str
    side: str  # 'buy' or 'sell'
    position_size: float
    confidence: float
    
    # Risk Management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_time: Optional[timedelta] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Execution Context
    phase_gate: str = "8b"  # 4b, 8b, 42b
    entropy_score: float = 0.0
    coherence_score: float = 0.0
    
    # Mathematical Validation
    mathematical_validity: bool = False
    klein_bottle_consistent: bool = False
    fractal_convergent: bool = False

@dataclass 
class ExecutionResult:
    """Result of trade execution"""
    signal_id: str
    status: ExecutionStatus
    timestamp: datetime
    
    # Execution Details
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    slippage: float = 0.0
    fees: float = 0.0
    
    # Mathematical Tracking
    original_analysis: Optional[AnalysisResult] = None
    execution_confidence: float = 0.0
    
    # Error Handling
    error_message: Optional[str] = None
    retry_count: int = 0

class CCXTExecutionManager:
    """
    Bridges your mathematical systems with CCXT trading execution.
    Ensures all trades are mathematically validated before execution.
    Enhanced with Windows CLI compatibility for cross-platform reliability.
    """
    
    def __init__(self, 
                 api_coordinator: Optional[UnifiedAPICoordinator] = None,
                 math_processor: Optional[UnifiedMathematicalProcessor] = None,
                 fitness_oracle: Optional[EnhancedFitnessOracle] = None,
                 sustainment_lib: Optional[SustainmentMathLib] = None):
        """
        Initialize CCXT execution manager with mathematical foundations
        
        Args:
            api_coordinator: Your existing UnifiedAPICoordinator
            math_processor: Your existing UnifiedMathematicalProcessor
            fitness_oracle: Your existing EnhancedFitnessOracle
            sustainment_lib: Your existing SustainmentMathLib
        """
        # Your existing mathematical systems
        self.math_processor = math_processor or UnifiedMathematicalProcessor()
        self.fitness_oracle = fitness_oracle or EnhancedFitnessOracle()
        self.sustainment_lib = sustainment_lib or SustainmentMathLib()
        
        # API coordination
        self.api_coordinator = api_coordinator or UnifiedAPICoordinator()
        
        # Windows CLI compatibility handler
        self.cli_handler = WindowsCliCompatibilityHandler()
        
        # Execution tracking
        self.active_signals: Dict[str, MathematicalTradeSignal] = {}
        self.execution_history: List[ExecutionResult] = []
        self.risk_limits = {
            'max_position_size': 0.1,  # 10% max position
            'max_daily_trades': 50,
            'max_loss_per_trade': 0.02,  # 2% max loss
            'min_confidence': 0.7  # 70% minimum confidence
        }
        
        # Performance tracking
        self.stats = {
            'total_signals': 0,
            'executed_trades': 0,
            'successful_trades': 0,
            'mathematical_validations': 0,
            'risk_rejections': 0,
            'profit_total': 0.0
        }
        
        self.cli_handler.log_safe(logger, 'info', "CCXTExecutionManager initialized with mathematical foundation")
    
    async def evaluate_trade_opportunity(self, market_data: Dict[str, Any]) -> Optional[MathematicalTradeSignal]:
        """
        Evaluate trade opportunity using your unified mathematical system
        
        Args:
            market_data: Current market data
            
        Returns:
            MathematicalTradeSignal if opportunity is mathematically validated
        """
        try:
            # STEP 1: Run your unified mathematical analysis
            self.cli_handler.log_safe(logger, 'info', "🧠 Running unified mathematical analysis...")
            math_results = self.math_processor.run_complete_analysis()
            
            # STEP 2: Capture market snapshot with fitness oracle
            self.cli_handler.log_safe(logger, 'info', "📊 Capturing market snapshot...")
            snapshot = await self.fitness_oracle.capture_market_snapshot(market_data)
            
            # STEP 3: Calculate unified fitness score
            self.cli_handler.log_safe(logger, 'info', "🎯 Calculating unified fitness...")
            fitness_score = self.fitness_oracle.calculate_unified_fitness(snapshot)
            
            # STEP 4: Create graded profit vector
            self.cli_handler.log_safe(logger, 'info', "📈 Creating profit vector...")
            trade_dict = {
                'profit': fitness_score.profit_fitness,
                'volume_allocated': fitness_score.position_size,
                'time_held': 3600.0,  # 1 hour default
                'signal_strength': fitness_score.confidence,
                'smart_money_score': fitness_score.overall_fitness
            }
            graded_vector = self.sustainment_lib.grading_vector(trade_dict)
            
            # STEP 5: Mathematical validation checks
            mathematical_validity = self._validate_mathematical_consistency(
                math_results, fitness_score, graded_vector
            )
            
            if not mathematical_validity['is_valid']:
                self.cli_handler.log_safe(logger, 'warning', f"Mathematical validation failed: {mathematical_validity['reason']}")
                return None
            
            # STEP 6: Create mathematical trade signal
            signal = MathematicalTradeSignal(
                signal_id=f"math_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                unified_analysis=AnalysisResult(
                    name="unified_trade_analysis",
                    data=math_results,
                    confidence=mathematical_validity['confidence'],
                    timestamp=time.time()
                ),
                fitness_score=fitness_score,
                graded_vector=graded_vector,
                symbol=market_data.get('symbol', 'BTC/USDT'),
                side=fitness_score.action.lower() if fitness_score.action in ['BUY', 'SELL'] else 'hold',
                position_size=fitness_score.position_size,
                confidence=fitness_score.confidence,
                stop_loss=fitness_score.stop_loss,
                take_profit=fitness_score.take_profit,
                max_hold_time=fitness_score.max_hold_time,
                risk_level=self._assess_risk_level(fitness_score),
                phase_gate=self._determine_phase_gate(math_results),
                entropy_score=math_results.get('entropy', 0.0),
                coherence_score=mathematical_validity['coherence'],
                mathematical_validity=True,
                klein_bottle_consistent=math_results.get('mathematical_validity', {}).get('topology_consistent', False),
                fractal_convergent=math_results.get('mathematical_validity', {}).get('fractal_convergent', False)
            )
            
            self.stats['total_signals'] += 1
            self.stats['mathematical_validations'] += 1
            
            self.cli_handler.log_safe(logger, 'info', f"✅ Mathematical trade signal generated: {signal.signal_id}")
            return signal
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "evaluate_trade_opportunity")
            self.cli_handler.log_safe(logger, 'error', error_message)
            return None
    
    async def execute_signal(self, signal: MathematicalTradeSignal) -> ExecutionResult:
        """
        Execute mathematically validated trade signal
        
        Args:
            signal: MathematicalTradeSignal to execute
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Pre-execution validation
            if not await self._pre_execution_validation(signal):
                return ExecutionResult(
                    signal_id=signal.signal_id,
                    status=ExecutionStatus.FAILED,
                    timestamp=start_time,
                    error_message="Pre-execution validation failed"
                )
            
            # Store active signal
            self.active_signals[signal.signal_id] = signal
            
            # Create trading request for your API coordinator
            trading_request = TradingRequest(
                request_id=signal.signal_id,
                symbol=signal.symbol,
                side=signal.side,
                amount=signal.position_size,
                price=signal.take_profit,  # Use take profit as limit price
                order_type="limit" if signal.take_profit else "market",
                exchange="binance",  # Default exchange
                priority=signal.confidence,
                metadata={
                    'mathematical_analysis': True,
                    'confidence': signal.confidence,
                    'risk_level': signal.risk_level.value,
                    'phase_gate': signal.phase_gate,
                    'entropy_score': signal.entropy_score,
                    'coherence_score': signal.coherence_score,
                    'klein_bottle_consistent': signal.klein_bottle_consistent,
                    'fractal_convergent': signal.fractal_convergent
                }
            )
            
            # Execute through your existing API coordinator
            self.cli_handler.log_safe(logger, 'info', f"🔄 Executing signal {signal.signal_id} via API coordinator...")
            api_result = await self.api_coordinator.request_trading_operation(
                symbol=trading_request.symbol,
                side=trading_request.side,
                amount=trading_request.amount,
                price=trading_request.price,
                order_type=trading_request.order_type,
                exchange=trading_request.exchange,
                priority=trading_request.priority,
                metadata=trading_request.metadata
            )
            
            # Process execution result
            execution_result = self._process_api_result(signal, api_result, start_time)
            
            # Update statistics
            self._update_execution_stats(execution_result)
            
            # Store in history
            self.execution_history.append(execution_result)
            
            # Clean up active signal
            if signal.signal_id in self.active_signals:
                del self.active_signals[signal.signal_id]
            
            self.cli_handler.log_safe(logger, 'info', f"✅ Signal execution completed: {signal.signal_id}")
            return execution_result
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, f"execute_signal {signal.signal_id}")
            self.cli_handler.log_safe(logger, 'error', error_message)
            return ExecutionResult(
                signal_id=signal.signal_id,
                status=ExecutionStatus.FAILED,
                timestamp=start_time,
                error_message=str(e)
            )
    
    def _validate_mathematical_consistency(self, 
                                         math_results: Dict, 
                                         fitness_score: UnifiedFitnessScore,
                                         graded_vector: GradedProfitVector) -> Dict[str, Any]:
        """Validate mathematical consistency across all systems"""
        
        # Check Klein Bottle consistency
        klein_bottle_valid = math_results.get('mathematical_validity', {}).get('topology_consistent', False)
        
        # Check fractal convergence
        fractal_convergent = math_results.get('mathematical_validity', {}).get('fractal_convergent', False)
        
        # Check fitness score alignment
        fitness_aligned = (
            fitness_score.confidence > 0.5 and
            fitness_score.overall_fitness > 0.3 and
            fitness_score.action in ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL']
        )
        
        # Check graded vector validity
        vector_valid = (
            graded_vector.profit != 0 and
            graded_vector.signal_strength > 0.3 and
            graded_vector.smart_money_score > 0.2
        )
        
        # Calculate overall coherence
        coherence_factors = [
            1.0 if klein_bottle_valid else 0.5,
            1.0 if fractal_convergent else 0.6,
            fitness_score.confidence if fitness_aligned else 0.3,
            graded_vector.signal_strength if vector_valid else 0.2
        ]
        
        overall_coherence = float(np.mean(coherence_factors))
        
        is_valid = (
            overall_coherence > 0.6 and
            fitness_aligned and
            vector_valid
        )
        
        return {
            'is_valid': is_valid,
            'coherence': overall_coherence,
            'confidence': overall_coherence,
            'klein_bottle_valid': klein_bottle_valid,
            'fractal_convergent': fractal_convergent,
            'fitness_aligned': fitness_aligned,
            'vector_valid': vector_valid,
            'reason': 'Mathematical validation passed' if is_valid else 'Low coherence or invalid components'
        }
    
    def _assess_risk_level(self, fitness_score: UnifiedFitnessScore) -> RiskLevel:
        """Assess risk level based on fitness score"""
        if fitness_score.risk_fitness < -0.5:
            return RiskLevel.CRITICAL
        elif fitness_score.risk_fitness < -0.2:
            return RiskLevel.HIGH
        elif fitness_score.risk_fitness < 0.2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _determine_phase_gate(self, math_results: Dict) -> str:
        """Determine appropriate phase gate based on mathematical analysis"""
        entropy = math_results.get('entropy', 0.5)
        convergence = math_results.get('fractal_convergence', {}).get('tff_convergence', 1.0)
        
        if entropy < 0.3 and convergence > 0.8:
            return "4b"  # Fast micro trades
        elif entropy < 0.7 and convergence > 0.5:
            return "8b"  # Harmonic trades
        else:
            return "42b"  # Strategic long-term
    
    async def _pre_execution_validation(self, signal: MathematicalTradeSignal) -> bool:
        """Pre-execution validation checks"""
        
        # Check mathematical validity
        if not signal.mathematical_validity:
            logger.warning(f"Signal {signal.signal_id} failed mathematical validation")
            return False
        
        # Check confidence threshold
        if signal.confidence < self.risk_limits['min_confidence']:
            logger.warning(f"Signal {signal.signal_id} below confidence threshold")
            self.stats['risk_rejections'] += 1
            return False
        
        # Check position size limits
        if signal.position_size > self.risk_limits['max_position_size']:
            logger.warning(f"Signal {signal.signal_id} exceeds position size limit")
            self.stats['risk_rejections'] += 1
            return False
        
        # Check daily trade limits
        today_trades = len([
            result for result in self.execution_history
            if result.timestamp.date() == datetime.now().date() and
               result.status == ExecutionStatus.EXECUTED
        ])
        
        if today_trades >= self.risk_limits['max_daily_trades']:
            logger.warning(f"Daily trade limit reached")
            self.stats['risk_rejections'] += 1
            return False
        
        # Check critical risk level
        if signal.risk_level == RiskLevel.CRITICAL:
            logger.warning(f"Signal {signal.signal_id} has critical risk level")
            self.stats['risk_rejections'] += 1
            return False
        
        return True
    
    def _process_api_result(self, 
                           signal: MathematicalTradeSignal, 
                           api_result: Dict[str, Any],
                           start_time: datetime) -> ExecutionResult:
        """Process API result into ExecutionResult"""
        
        if 'error' in api_result:
            return ExecutionResult(
                signal_id=signal.signal_id,
                status=ExecutionStatus.FAILED,
                timestamp=start_time,
                original_analysis=signal.unified_analysis,
                error_message=api_result['error']
            )
        
        # Determine execution status
        if api_result.get('status') == 'simulated':
            status = ExecutionStatus.EXECUTED  # Treat simulation as executed for now
            executed_price = api_result.get('price', 0.0)
            executed_quantity = api_result.get('amount', 0.0)
            fees = api_result.get('fee', 0.0)
        else:
            status = ExecutionStatus.PARTIAL  # API coordinator queued it
            executed_price = None
            executed_quantity = None
            fees = 0.0
        
        return ExecutionResult(
            signal_id=signal.signal_id,
            status=status,
            timestamp=start_time,
            executed_price=executed_price,
            executed_quantity=executed_quantity,
            fees=fees,
            original_analysis=signal.unified_analysis,
            execution_confidence=signal.confidence
        )
    
    def _update_execution_stats(self, result: ExecutionResult) -> None:
        """Update execution statistics"""
        if result.status == ExecutionStatus.EXECUTED:
            self.stats['executed_trades'] += 1
            
            # Estimate profit (simplified)
            if result.executed_price and result.executed_quantity:
                estimated_profit = result.executed_quantity * 0.02  # 2% profit assumption
                self.stats['profit_total'] += estimated_profit
                self.stats['successful_trades'] += 1
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary"""
        return {
            'statistics': self.stats.copy(),
            'active_signals': len(self.active_signals),
            'execution_history_count': len(self.execution_history),
            'risk_limits': self.risk_limits.copy(),
            'recent_executions': [
                {
                    'signal_id': result.signal_id,
                    'status': result.status.value,
                    'timestamp': result.timestamp.isoformat(),
                    'executed_price': result.executed_price,
                    'confidence': result.execution_confidence
                }
                for result in self.execution_history[-5:]  # Last 5 executions
            ]
        }

def create_mathematical_execution_system() -> CCXTExecutionManager:
    """
    Factory function to create a fully configured mathematical execution system
    Enhanced with Windows CLI compatibility for cross-platform reliability
    """
    try:
        # Create your existing mathematical systems
        math_processor = UnifiedMathematicalProcessor()
        fitness_oracle = EnhancedFitnessOracle()
        sustainment_lib = SustainmentMathLib()
        
        # Create API coordinator
        api_coordinator = UnifiedAPICoordinator()
        
        # Create execution manager with all systems
        execution_manager = CCXTExecutionManager(
            api_coordinator=api_coordinator,
            math_processor=math_processor,
            fitness_oracle=fitness_oracle,
            sustainment_lib=sustainment_lib
        )
        
        # Initialize Windows CLI compatibility handler
        cli_handler = WindowsCliCompatibilityHandler()
        cli_handler.log_safe(logger, 'info', "🚀 Mathematical execution system created and ready")
        
        return execution_manager
        
    except Exception as e:
        cli_handler = WindowsCliCompatibilityHandler()
        error_message = cli_handler.safe_format_error(e, "create_mathematical_execution_system")
        cli_handler.log_safe(logger, 'error', error_message)
        raise 