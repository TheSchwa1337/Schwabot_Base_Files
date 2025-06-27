# -*- coding: utf-8 -*-
""""""
Unified Mathematical System

This module provides a centralized mathematical system that consolidates all mathematical
operations used throughout the trading system, ensuring consistency and proper integration
with the unified mathematical framework.

Supports:
- Two-bit logic pathways for fast computation
- Profit tier navigation vectorization
- BTC hashing calculations
- Traditional mathematical operations
- Financial calculations
- Linear algebra operations
""""""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Optional, Dict, Any

import numpy as np

# Import core components (with fallbacks to avoid circular imports)
try:
    from core.bit_phase_sequencer import BitPhase, BitSequence
except Exception as e:
    pass

except ImportError:
    BitPhase = None
    BitSequence = None

try:
    from core.dual_error_handler import PhaseState, SickType, SickState
except Exception as e:
    pass

except ImportError:
    PhaseState = None
    SickType = None
    SickState = None

try:
    from core.symbolic_profit_router import ProfitTier, FlipBias, SymbolicState
except Exception as e:
    pass

except ImportError:
    ProfitTier = None
    FlipBias = None
    SymbolicState = None

try:
    from dual_unicore_handler import DualUnicoreHandler
    unicore = DualUnicoreHandler()
except Exception as e:
    pass

except ImportError:
    unicore = None

# Safe print functions with fallbacks
try:
    from utils.safe_print import safe_print, safe_math, info, warn, error
except Exception as e:
    pass

except ImportError:
    def safe_print(*args, **kwargs):
        print(*args, **kwargs)

    def safe_math(*args, **kwargs):
        print(*args, **kwargs)

    def info(*args, **kwargs):
        print("[INFO]", *args, **kwargs)

    def warn(*args, **kwargs):
        print("[WARN]", *args, **kwargs)

    def error(*args, **kwargs):
        print("[ERROR]", *args, **kwargs)


class MathOperation(Enum):
    """Enumeration of mathematical operations."""

    # Basic arithmetic
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    SQRT = "sqrt"
    LOG = "log"
    EXP = "exp"
    
    # Trigonometric
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    ASIN = "asin"
    ACOS = "acos"
    ATAN = "atan"
    
    # Utility
    ABS = "abs"
    MAX = "max"
    MIN = "min"
    ROUND = "round"
    FLOOR = "floor"
    CEIL = "ceil"
    
    # Statistical
    MEAN = "mean"
    STD = "std"
    VAR = "var"
    CORRELATION = "correlation"
    COVARIANCE = "covariance"
    
    # Linear algebra
    DOT_PRODUCT = "dot_product"
    CROSS_PRODUCT = "cross_product"
    MATRIX_MULTIPLY = "matrix_multiply"
    INVERSE = "inverse"
    DETERMINANT = "determinant"
    EIGENVALUES = "eigenvalues"
    EIGENVECTORS = "eigenvectors"
    SVD = "svd"
    QR = "qr"
    LU = "lu"
    CHOLESKY = "cholesky"

    # BTC/Crypto operations
    HASH_RATE = "hash_rate"
    DIFFICULTY_ADJUST = "difficulty_adjust"
    BLOCK_REWARD = "block_reward"
    
    # Profit operations
    PROFIT_VECTOR = "profit_vector"
    TIER_NAVIGATION = "tier_navigation"
    ENTRY_EXIT_OPTIMIZATION = "entry_exit_optimization"


@dataclass
class MathResult:
    """Container for mathematical operation results."""
    value: Any
    operation: MathOperation
    inputs: List[Any]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class UnifiedMathSystem:
    """"""
    Unified mathematical system for the trading platform.

    This class provides a centralized interface for all mathematical operations,
    ensuring consistency and proper error handling across the system.
    
    Features:
    - Traditional mathematical operations
    - BTC hashing and mining calculations
    - Profit tier navigation vectorization
    - Two-bit logic pathway optimization
    - Financial analysis tools
    """"""

    def __init__(self, precision: int = 8, use_safe_print: bool = True):
        """"""
        Initialize the unified math system.

        Args:
            precision: Number of decimal places for floating point operations
            use_safe_print: Whether to use safe print for CLI compatibility
        """"""
        self.precision = precision
        self.use_safe_print = use_safe_print
        self.operation_history = []
        self.error_count = 0
        self.success_count = 0

        # Set numpy precision
        np.set_printoptions(precision=precision)

        # Initialize logging
        self.logger = logging.getLogger(__name__)

        # BTC/Crypto constants
        self.btc_constants = {
            'max_supply': 21_000_000,
            'halving_interval': 210_000,
            'initial_reward': 50.0,
            'target_block_time': 600,  # 10 minutes in seconds
            'difficulty_adjustment_interval': 2016,  # blocks
        }
        
        # Profit tier thresholds
        self.profit_tiers = {
            'micro': 0.1,
            'small': 0.1,
            'medium': 0.1,
            'large': 1.0,
            'whale': 10.0
        }
        
        info("Unified Math System initialized with precision:", precision)
    
    def _log_operation(self, operation: MathOperation, inputs: List[Any], 
                    result: Any, success: bool, error_msg: Optional[str] = None) -> None:
        """Log mathematical operations for debugging and auditing."""
        log_entry = {
            'operation': operation.value,
            'inputs': inputs,
            'result': result,
            'success': success,
            'error_message': error_msg,
            'timestamp': np.datetime64('now')
        }

        self.operation_history.append(log_entry)

        if success:
            self.success_count += 1
            if self.use_safe_print:
                safe_math(f"{operation.value}: {result}")
        else:
            self.error_count += 1
            if self.use_safe_print:
                error(f"Math error in {operation.value}: {error_msg}")

    def _validate_inputs(self, inputs: List[Any], expected_types: List[type]) -> bool:
        """Validate input types for mathematical operations."""
        if len(inputs) != len(expected_types):
            return False

        for input_val, expected_type in zip(inputs, expected_types):
            if not isinstance(input_val, expected_type):
                return False

        return True

    def _safe_operation(self, operation_func, inputs: List[Any], 
                        operation: MathOperation) -> MathResult:
        """Safely execute a mathematical operation with error handling."""
        try:
            result = operation_func(*inputs)

            # Round floating point results to specified precision
            if isinstance(result, (float, np.floating)):
                result = round(result, self.precision)
            elif isinstance(result, np.ndarray) and result.dtype.kind in 'fc':
                result = np.round(result, self.precision)

            math_result = MathResult(
                value=result,
                operation=operation,
                inputs=inputs,
                metadata={'precision': self.precision},
                success=True
            )

            self._log_operation(operation, inputs, result, True)
            return math_result

        except Exception as e:
            error_msg = f"Operation {operation.value} failed: {str(e)}"
            math_result = MathResult(
                value=None,
                operation=operation,
                inputs=inputs,
                metadata={'error': str(e)},
                success=False,
                error_message=error_msg
            )

            self._log_operation(operation, inputs, None, False, error_msg)
            return math_result

    # Basic arithmetic operations
    def add(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> MathResult:
        """Add two numbers or arrays."""
        return self._safe_operation(lambda x, y: x + y, [a, b], MathOperation.ADD)
    
    def subtract(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> MathResult:
        """Subtract two numbers or arrays."""
        return self._safe_operation(lambda x, y: x - y, [a, b], MathOperation.SUBTRACT)
    
    def multiply(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> MathResult:
        """Multiply two numbers or arrays."""
        return self._safe_operation(lambda x, y: x * y, [a, b], MathOperation.MULTIPLY)
    
    def divide(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> MathResult:
        """Divide two numbers or arrays."""
        if isinstance(b, (int, float)) and b == 0:
            return MathResult(
                None, MathOperation.DIVIDE, [a, b], {}, False, "Division by zero"
            )
        return self._safe_operation(lambda x, y: x / y, [a, b], MathOperation.DIVIDE)
    
    def power(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> MathResult:
        """Raise a to the power of b."""
        return self._safe_operation(lambda x, y: x ** y, [a, b], MathOperation.POWER)

    def sqrt(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate square root."""
        if isinstance(a, (int, float)) and a < 0:
            return MathResult(
                None, MathOperation.SQRT, [a], {}, False, "Negative number under square root"
            )
        return self._safe_operation(lambda x: np.sqrt(x), [a], MathOperation.SQRT)

    def log(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate natural logarithm."""
        if isinstance(a, (int, float)) and a <= 0:
            return MathResult(
                None, MathOperation.LOG, [a], {}, False, "Non-positive number for logarithm"
            )
        return self._safe_operation(lambda x: np.log(x), [a], MathOperation.LOG)

    def exp(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate exponential."""
        return self._safe_operation(lambda x: np.exp(x), [a], MathOperation.EXP)

    # Trigonometric operations
    def sin(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate sine."""
        return self._safe_operation(lambda x: np.sin(x), [a], MathOperation.SIN)

    def cos(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate cosine."""
        return self._safe_operation(lambda x: np.cos(x), [a], MathOperation.COS)

    def tan(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate tangent."""
        return self._safe_operation(lambda x: np.tan(x), [a], MathOperation.TAN)

    # Statistical operations
    def mean(self, a: Union[List, np.ndarray]) -> MathResult:
        """Calculate mean."""
        return self._safe_operation(lambda x: np.mean(x), [a], MathOperation.MEAN)

    def std(self, a: Union[List, np.ndarray], ddof: int = 1) -> MathResult:
        """Calculate standard deviation."""
        return self._safe_operation(lambda x: np.std(x, ddof=ddof), [a], MathOperation.STD)

    def var(self, a: Union[List, np.ndarray], ddof: int = 1) -> MathResult:
        """Calculate variance."""
        return self._safe_operation(lambda x: np.var(x, ddof=ddof), [a], MathOperation.VAR)
    
    def correlation(self, a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> MathResult:
        """Calculate correlation coefficient."""
        return self._safe_operation(
            lambda x, y: np.corrcoef(x, y)[0, 1], [a, b], MathOperation.CORRELATION
        )
    
    def covariance(self, a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> MathResult:
        """Calculate covariance."""
        return self._safe_operation(
            lambda x, y: np.cov(x, y)[0, 1], [a, b], MathOperation.COVARIANCE
        )

    # Linear algebra operations
    def dot_product(self, a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> MathResult:
        """Calculate dot product."""
        return self._safe_operation(lambda x, y: np.dot(x, y), [a, b], MathOperation.DOT_PRODUCT)

    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> MathResult:
        """Multiply matrices."""
        return self._safe_operation(lambda x, y: np.matmul(x, y), [a, b], MathOperation.MATRIX_MULTIPLY)

    def inverse(self, a: np.ndarray) -> MathResult:
        """Calculate matrix inverse."""
        return self._safe_operation(lambda x: np.linalg.inv(x), [a], MathOperation.INVERSE)

    def determinant(self, a: np.ndarray) -> MathResult:
        """Calculate matrix determinant."""
        return self._safe_operation(lambda x: np.linalg.det(x), [a], MathOperation.DETERMINANT)

    def eigenvalues(self, a: np.ndarray) -> MathResult:
        """Calculate eigenvalues."""
        return self._safe_operation(lambda x: np.linalg.eigvals(x), [a], MathOperation.EIGENVALUES)

    def eigenvectors(self, a: np.ndarray) -> MathResult:
        """Calculate eigenvalues and eigenvectors."""
        return self._safe_operation(lambda x: np.linalg.eig(x), [a], MathOperation.EIGENVECTORS)

    def svd(self, a: np.ndarray) -> MathResult:
        """Calculate singular value decomposition."""
        return self._safe_operation(lambda x: np.linalg.svd(x), [a], MathOperation.SVD)

    # Utility operations
    def abs(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate absolute value."""
        return self._safe_operation(lambda x: np.abs(x), [a], MathOperation.ABS)
    
    def max(self, a: Union[List, np.ndarray]) -> MathResult:
        """Find maximum value."""
        return self._safe_operation(lambda x: np.max(x), [a], MathOperation.MAX)
    
    def min(self, a: Union[List, np.ndarray]) -> MathResult:
        """Find minimum value."""
        return self._safe_operation(lambda x: np.min(x), [a], MathOperation.MIN)
    
    # BTC/Crypto operations
    def calculate_hash_rate(self, difficulty: float, block_time: float = None) -> MathResult:
        """"""
        Calculate estimated network hash rate from difficulty.
        
        Args:
            difficulty: Current network difficulty
            block_time: Actual block time (default: target block time)
        """"""
        if block_time is None:
            block_time = self.btc_constants['target_block_time']
        
        def hash_rate_calc(diff, time):
            # Hash rate = difficulty * 2^32 / time
            return diff * (2 ** 32) / time
        
#         return self._safe_operation(hash_rate_calc, [difficulty, block_time], MathOperation.HASH_RATE)
    
    def calculate_difficulty_adjustment(self, actual_time: float, target_time: float = None) -> MathResult:
        """"""
        Calculate difficulty adjustment based on actual vs target block times.
        
        Args:
            actual_time: Actual time for difficulty period
            target_time: Target time for difficulty period
        """"""
        if target_time is None:
            target_time = self.btc_constants['target_block_time'] * self.btc_constants['difficulty_adjustment_interval']
        
        def diff_adjust(actual, target):
            # New difficulty = old difficulty * (actual time / target time)
            adjustment_factor = target / actual
            # Clamp between 0.25x and 4x
            return max(0.25, min(4.0, adjustment_factor))
        
#         return self._safe_operation(diff_adjust, [actual_time, target_time], MathOperation.DIFFICULTY_ADJUST)
    
    def calculate_block_reward(self, block_height: int) -> MathResult:
        """"""
        Calculate block reward based on height (including halvings).
        
        Args:
            block_height: Current block height
        """"""
        def reward_calc(height):
            halvings = height // self.btc_constants['halving_interval']
            reward = self.btc_constants['initial_reward'] / (2 ** halvings)
            return max(0, reward)  # Ensure non-negative
        
#         return self._safe_operation(reward_calc, [block_height], MathOperation.BLOCK_REWARD)
    
    # Profit tier navigation operations
    def calculate_profit_vector(self, entry_price: float, current_price: float, 
                                volume: float = 1.0) -> MathResult:
        """"""
        Calculate profit vector for tier navigation.
        
        Args:
            entry_price: Entry price for position
            current_price: Current market price
            volume: Position volume
        """"""
        def profit_vector_calc(entry, current, vol):
            pnl = (current - entry) / entry
            profit_vector = {
                'pnl_percent': pnl * 100,
                'pnl_absolute': (current - entry) * vol,
                'tier': self._classify_profit_tier(abs(pnl)),
                'vector_magnitude': abs(pnl),
                'vector_direction': 1 if pnl >= 0 else -1
            }
            return profit_vector
        
#         return self._safe_operation(
            profit_vector_calc, [entry_price, current_price, volume], MathOperation.PROFIT_VECTOR
        )
    
    def optimize_entry_exit(self, price_series: List[float], lookback: int = 20) -> MathResult:
        """"""
        Optimize entry/exit points using mathematical analysis.
        
        Args:
            price_series: Historical price data
            lookback: Lookback period for analysis
        """"""
        def entry_exit_optimization(prices, window):
            if len(prices) < window:
                return {'error': 'Insufficient data'}
            
            prices_array = np.array(prices)
            
            # Calculate moving averages
            ma_short = np.convolve(prices_array, np.ones(window // 2) / (window // 2), mode='valid')
            ma_long = np.convolve(prices_array, np.ones(window) / window, mode='valid')
            
            # Find crossover points
            if len(ma_short) > len(ma_long):
                ma_short = ma_short[:len(ma_long)]
            elif len(ma_long) > len(ma_short):
                ma_long = ma_long[:len(ma_short)]
            
            crossovers = np.where(np.diff(np.sign(ma_short - ma_long)))[0]
            
            return {
                'entry_points': crossovers[::2].tolist() if len(crossovers) > 0 else [],
                'exit_points': crossovers[1::2].tolist() if len(crossovers) > 1 else [],
                'ma_short': ma_short.tolist(),
                'ma_long': ma_long.tolist()
            }
        
#         return self._safe_operation(
            entry_exit_optimization, [price_series, lookback], MathOperation.ENTRY_EXIT_OPTIMIZATION
        )
    
    def _classify_profit_tier(self, profit_magnitude: float) -> str:
        """Classify profit into tiers based on magnitude."""
        for tier, threshold in sorted(self.profit_tiers.items(), key=lambda x: x[1], reverse=True):
            if profit_magnitude >= threshold:
                return tier
        return 'micro'
    
    # Two-bit logic pathway operations
    def two_bit_encode(self, value: float) -> MathResult:
        """"""
        Encode a value using two-bit logic for fast computation pathways.
        
        Args:
            value: Input value to encode
        """"""
        def encode_two_bit(val):
            # Normalize to 0-3 range and convert to 2-bit representation
            normalized = max(0, min(3, int(abs(val) * 4)))
            binary = format(normalized, '02b')
            return {
                'decimal': normalized,
                'binary': binary,
                'bits': [int(b) for b in binary]
            }
        
#         return self._safe_operation(encode_two_bit, [value], MathOperation.ADD)  # Using ADD as placeholder
    
    def two_bit_pathway_select(self, pathway_a: Any, pathway_b: Any, selector: float) -> MathResult:
        """"""
        Select between two computation pathways based on selector value.
        
        Args:
            pathway_a: First computation pathway
            pathway_b: Second computation pathway  
            selector: Selection value (0-1 range)
        """"""
        def pathway_selection(a, b, sel):
            # Use selector to choose pathway
            if sel < 0.5:
                return {'selected_pathway': 'A', 'result': a, 'selector': sel}
            else:
                return {'selected_pathway': 'B', 'result': b, 'selector': sel}
        
#         return self._safe_operation(
            pathway_selection, [pathway_a, pathway_b, selector], MathOperation.ADD
        )
    
    # Financial calculations
    def calculate_returns(self, prices: List[float]) -> MathResult:
        """Calculate percentage returns from price series."""
        if len(prices) < 2:
            return MathResult(
                None, MathOperation.DIVIDE, [prices], {}, False, "Need at least 2 prices"
            )
        
        try:
            returns = []
            for i in range(1, len(prices)):
                if prices[i - 1] != 0:
                    ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                    returns.append(ret)
                else:
                    returns.append(0.0)

            return MathResult(
                value=returns,
                operation=MathOperation.DIVIDE,
                inputs=[prices],
                metadata={'type': 'returns'},
                success=True
            )

        except Exception as e:
            return MathResult(None, MathOperation.DIVIDE, [prices], {}, False, str(e))
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> MathResult:
        """Calculate Sharpe ratio."""
        if not returns:
            return MathResult(
                None, MathOperation.DIVIDE, [returns], {}, False, "Empty returns list"
            )
        
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return == 0:
                return MathResult(
                    None, MathOperation.DIVIDE, [returns], {}, False, "Zero standard deviation"
                )

            sharpe = (mean_return - risk_free_rate) / std_return

            return MathResult(
                value=sharpe,
                operation=MathOperation.DIVIDE,
                inputs=[returns, risk_free_rate],
                metadata={'type': 'sharpe_ratio'},
                success=True
            )

        except Exception as e:
            return MathResult(None, MathOperation.DIVIDE, [returns], {}, False, str(e))

    def calculate_max_drawdown(self, prices: List[float]) -> MathResult:
        """Calculate maximum drawdown."""
        if not prices:
            return MathResult(
                None, MathOperation.MIN, [prices], {}, False, "Empty prices list"
            )
        
        try:
            peak = prices[0]
            max_dd = 0.0

            for price in prices:
                if price > peak:
                    peak = price
                dd = (peak - price) / peak
                max_dd = max(max_dd, dd)

            return MathResult(
                value=max_dd,
                operation=MathOperation.MIN,
                inputs=[prices],
                metadata={'type': 'max_drawdown'},
                success=True
            )

        except Exception as e:
            return MathResult(
                None, MathOperation.MIN, [prices], {}, False, str(e)
            )
    
    def calculate_volatility(self, returns: List[float], window: int = 252) -> MathResult:
        """Calculate rolling volatility."""
        if len(returns) < window:
            return MathResult(
                None, MathOperation.STD, [returns], {}, False, f"Need at least {window} returns"
            )
        
        try:
            volatilities = []
            for i in range(window, len(returns) + 1):
                window_returns = returns[i - window:i]
                vol = np.std(window_returns) * np.sqrt(252)  # Annualized
                volatilities.append(vol)

            return MathResult(
                value=volatilities,
                operation=MathOperation.STD,
                inputs=[returns, window],
                metadata={'type': 'rolling_volatility', 'window': window},
                success=True
            )

        except Exception as e:
            return MathResult(
                None, MathOperation.STD, [returns], {}, False, str(e)
            )
    
    # System utilities
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'total_operations': len(self.operation_history),
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / max(len(self.operation_history), 1),
            'precision': self.precision,
            'btc_constants': self.btc_constants,
            'profit_tiers': self.profit_tiers
        }

    def clear_history(self) -> None:
        """Clear operation history."""
        self.operation_history.clear()
        self.error_count = 0
        self.success_count = 0
    
    def export_pathway_config(self) -> Dict[str, Any]:
        """Export configuration for pathway optimization."""
        return {
            'two_bit_pathways': {
                'enabled': True,
                'encoding_bits': 2,
                'pathway_threshold': 0.5
            },
            'profit_navigation': {
                'tiers': self.profit_tiers,
                'optimization_enabled': True,
                'vector_tracking': True
            },
            'btc_operations': {
                'constants': self.btc_constants,
                'hash_calculations': True,
                'difficulty_tracking': True
            }
        }


# Create global instance
unified_math = UnifiedMathSystem()


# Export key functions for direct access
def add(a, b):
    """Direct access to addition operation."""
    return unified_math.add(a, b).value


def multiply(a, b):
    """Direct access to multiplication operation."""
    return unified_math.multiply(a, b).value


def exp(a):
    """Direct access to exponential operation."""
    return unified_math.exp(a).value


def log(a):
    """Direct access to logarithm operation."""
    return unified_math.log(a).value


def sqrt(a):
    """Direct access to square root operation."""
    return unified_math.sqrt(a).value


# BTC specific exports
def calculate_hash_rate(difficulty, block_time=None):
    """Direct access to hash rate calculation."""
    return unified_math.calculate_hash_rate(difficulty, block_time).value


def calculate_profit_vector(entry_price, current_price, volume=1.0):
    """Direct access to profit vector calculation."""
    return unified_math.calculate_profit_vector(entry_price, current_price, volume).value


# Export the main class and instance
__all__ = [
    'UnifiedMathSystem', 
    'unified_math', 
    'MathOperation', 
    'MathResult',
    'add', 'multiply', 'exp', 'log', 'sqrt',
    'calculate_hash_rate', 'calculate_profit_vector'
]
