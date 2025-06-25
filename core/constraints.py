# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import math
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""Mathematical Constraints System - Schwabot Framework.

==================================================



Comprehensive mathematical constraints validation system for trading

parameters, risk management, and mathematical operation bounds.



Key Features:

- Trading parameter validation (position size, leverage, risk limits)

- Mathematical bounds checking (matrix conditions, iteration limits)

- Risk management constraints (Sharpe ratio, drawdown limits)

- Portfolio constraint enforcement with diversification requirements

- Windows CLI compatibility with flake8 compliance



This replaces the empty constraints.py file with a complete implementation.

"""

from dataclasses import dataclass
from decimal import Decimal
from decimal import getcontext
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

# from core.unified_math_system import unified_math  # F811: duplicate import
import numpy.typing as npt

if TYPE_CHECKING:
    pass

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


@dataclass
class ConstraintViolation:
    """Container for constraint violation information."""

constraint_name: str
violation_type: str
current_value: Union[float, Decimal]
expected_range: Tuple[Union[float, Decimal], Union[float, Decimal]]
severity: str  # 'warning', 'error', 'critical'
message: str
remediation_suggestion: str = ""


@dataclass
class ValidationResult:
    """Result of constraint validation."""

valid: bool
violations: List[ConstraintViolation]
warnings: List[str]
risk_score: float  # 0.0 to 1.0, where 1.0 is highest risk
execution_time: float = 0.0


class TradingConstraints:
    """Trading-specific constraint validation."""

    def __init__(self) -> None:
        """Initialize trading constraints."""
self.max_position_size = Decimal("1.0")  # 100% of portfolio
        self.max_leverage = Decimal("2.0")
        self.min_liquidity_ratio = Decimal("0.05")  # 5% cash minimum
        self.max_sector_concentration = Decimal("0.30")  # 30% max in any sector
        self.max_single_asset_weight = Decimal("0.20")  # 20% max in single asset
        self.min_diversification_count = 3
self.max_correlation_threshold = 0.85

    def validate_position_size(
        self, position_size: Union[float, Decimal]
) -> Optional[ConstraintViolation]:
"""Validate position size constraints."""
pos_size = (
            Decimal(str(position_size))
            if not isinstance(position_size, Decimal)
            else position_size


        if pos_size < 0:
            return ConstraintViolation(
                constraint_name="position_size_non_negative",
violation_type="invalid_value",
current_value=pos_size,
expected_range=(Decimal("0.0"), self.max_position_size),
                severity="error",
message="Position size cannot be negative",
remediation_suggestion=(
                    "Use positive position size or short position flag"
),


        if pos_size > self.max_position_size:
            return ConstraintViolation(
                constraint_name="position_size_limit",
violation_type="limit_exceeded",
current_value=pos_size,
expected_range=(Decimal("0.0"), self.max_position_size),
                severity="error",
message=(
                    f"Position size {pos_size} exceeds maximum "
f"{self.max_position_size}"
),
remediation_suggestion=(
                    "Reduce position size or increase available capital"
),


        return None

    def validate_leverage(
        self, leverage: Union[float, Decimal]
) -> Optional[ConstraintViolation]:
"""Validate leverage constraints."""
lev = Decimal(str(leverage)) if not isinstance(leverage, Decimal) else leverage

        if lev < Decimal("1.0"):
            return ConstraintViolation(
                constraint_name="leverage_minimum",
violation_type="below_minimum",
current_value=lev,
expected_range=(Decimal("1.0"), self.max_leverage),
                severity="warning",
message=(f"Leverage {lev} is below 1.0 (no leverage)"),
                remediation_suggestion=(
                    "Consider using at least 1.0x leverage for normal trading"
),


        if lev > self.max_leverage:
            return ConstraintViolation(
                constraint_name="leverage_limit",
violation_type="limit_exceeded",
current_value=lev,
expected_range=(Decimal("1.0"), self.max_leverage),
                severity="critical",
message=(f"Leverage {lev} exceeds maximum {self.max_leverage}"),
                remediation_suggestion=("Reduce leverage to acceptable levels"),


        return None

    def validate_portfolio_diversification(
        self, asset_weights: Dict[str, float]
) -> List[ConstraintViolation]:
"""Validate portfolio diversification constraints."""
violations = []

        # Check number of assets
        if len(asset_weights) < self.min_diversification_count:
            violations.append(
                ConstraintViolation(
                    constraint_name="diversification_count",
violation_type="insufficient_diversification",
current_value=len(asset_weights),
                    expected_range=(
                        self.min_diversification_count,
float("in"),
                    ),
severity="warning",
message=(
                        f"Portfolio has only {len(asset_weights)} assets, "
                        f"minimum {self.min_diversification_count}"
),
remediation_suggestion=(
                        "Add more assets to improve diversification"
),



        # Check individual asset concentration
        for asset, weight in asset_weights.items():
            if weight > float(self.max_single_asset_weight):
                violations.append(
                    ConstraintViolation(
                        constraint_name="asset_concentration",
violation_type="concentration_risk",
current_value=weight,
expected_range=(
                            0.0,
float(self.max_single_asset_weight),
                        ),
severity="error",
message=(
                            f"Asset {asset} weight {weight:.1%} exceeds "
f"maximum {self.max_single_asset_weight:.1%}"
),
remediation_suggestion=(
                            f"Reduce {asset} allocation to below "
f"{self.max_single_asset_weight:.1%}"
),



        # Check total weight
total_weight = sum(asset_weights.values())
        if unified_math.abs(total_weight - 1.0) > 0.01:  # 1% tolerance
            violations.append(
                ConstraintViolation(
                    constraint_name="portfolio_weight_sum",
violation_type="weight_mismatch",
current_value=total_weight,
expected_range=(0.99, 1.01),
                    severity="error",
message=(
                        f"Portfolio weights sum to {total_weight:.3f}, "
"should be 1.0"
),
remediation_suggestion=(
                        "Normalize portfolio weights to sum to 1.0"
),



        return violations


class MathematicalConstraints:
    """Mathematical operation constraint validation."""

    def __init__(self) -> None:
        """Initialize mathematical constraints."""
self.max_matrix_size = 10000
self.min_matrix_condition_number = 1e-12
self.max_iterations = 10000
self.numerical_tolerance = 1e-10
self.max_gradient_norm = 1e6

    def validate_matrix_properties(self, matrix: Matrix) -> List[ConstraintViolation]:
        """Validate matrix mathematical properties."""
violations = []

        # Check matrix size
        if matrix.size > self.max_matrix_size:
violations.append(
                ConstraintViolation(
                    constraint_name="matrix_size",
violation_type="size_exceeded",
current_value=matrix.size,
expected_range=(1, self.max_matrix_size),
                    severity="error",
message=(
                        f"Matrix size {matrix.size} exceeds maximum "
f"{self.max_matrix_size}"
),
remediation_suggestion=(
                        "Use smaller matrix or increase size limit"
),



        # Check for NaN or infinite values
        if np.any(np.isnan(matrix)):
            violations.append(
                ConstraintViolation(
                    constraint_name="matrix_nan_values",
violation_type="invalid_values",
current_value=np.sum(np.isnan(matrix)),
                    expected_range=(0, 0),
                    severity="critical",
message="Matrix contains NaN values",
remediation_suggestion=(
                        "Remove or replace NaN values before computation"
),



        if np.any(np.isinf(matrix)):
            violations.append(
                ConstraintViolation(
                    constraint_name="matrix_infinite_values",
violation_type="invalid_values",
current_value=np.sum(np.isinf(matrix)),
                    expected_range=(0, 0),
                    severity="critical",
message="Matrix contains infinite values",
remediation_suggestion=(
                        "Check computation for overflow or division by zero"
),



        # Check condition number for square matrices
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            try:
condition_number = np.linalg.cond(matrix)
                if condition_number > 1.0 / self.min_matrix_condition_number:
violations.append(
                        ConstraintViolation(
                            constraint_name="matrix_condition_number",
violation_type="numerical_instability",
current_value=condition_number,
expected_range=(
                                1.0,
1.0 / self.min_matrix_condition_number,
),
severity="warning",
message=(
                                "Matrix condition number "
f"{condition_number:.2e} indicates "
"potential instability"
),
remediation_suggestion=(
                                "Use regularization or alternative numerical " "methods"
),


            except np.linalg.LinAlgError:
violations.append(
                    ConstraintViolation(
                        constraint_name="matrix_singularity",
violation_type="singular_matrix",
current_value=0.0,
expected_range=(
                            self.min_matrix_condition_number,
float("in"),
                        ),
severity="error",
message="Matrix is singular and cannot be inverted",
remediation_suggestion=(
                            "Add regularization or use pseudo-inverse"
),



        return violations

    def validate_optimization_parameters(
        self,
iterations: int,
tolerance: float,
gradient_norm: Optional[float] = None,
) -> List[ConstraintViolation]:
"""Validate optimization algorithm parameters."""
violations = []

        # Check iteration count
        if iterations > self.max_iterations:
violations.append(
                ConstraintViolation(
                    constraint_name="max_iterations",
violation_type="limit_exceeded",
current_value=iterations,
expected_range=(1, self.max_iterations),
                    severity="warning",
message=(
                        f"Iteration count {iterations} exceeds recommended "
f"maximum {self.max_iterations}"
),
remediation_suggestion=(
                        "Consider using better initial guess or different " "algorithm"
),



        # Check tolerance
        if tolerance < self.numerical_tolerance:
violations.append(
                ConstraintViolation(
                    constraint_name="numerical_tolerance",
violation_type="too_strict",
current_value=tolerance,
expected_range=(self.numerical_tolerance, 1.0),
                    severity="warning",
message=(
                        f"Tolerance {tolerance:.2e} may be too strict for "
"numerical precision"
),
remediation_suggestion=(
                        "Consider using tolerance >= "
f"{self.numerical_tolerance:.2e}"
),



        # Check gradient norm if provided
        if gradient_norm is not None and gradient_norm > self.max_gradient_norm:
violations.append(
                ConstraintViolation(
                    constraint_name="gradient_explosion",
violation_type="numerical_instability",
current_value=gradient_norm,
expected_range=(0.0, self.max_gradient_norm),
                    severity="critical",
message=(
                        f"Gradient norm {gradient_norm:.2e} indicates "
"potential explosion"
),
remediation_suggestion=(
                        "Use gradient clipping or reduce learning rate"
),



        return violations


class RiskConstraints:
    """Risk management constraint validation."""

    def __init__(self) -> None:
        """Initialize risk constraints."""
self.max_var_95 = 0.05  # 5% daily VaR
self.max_drawdown = 0.20  # 20% maximum drawdown
self.min_sharpe_ratio = 0.5
self.max_correlation_exposure = 0.75
self.min_liquidity_score = 0.3

    def validate_risk_metrics(
        self, var_95: float, max_drawdown: float, sharpe_ratio: float
) -> List[ConstraintViolation]:
"""Validate portfolio risk metrics."""
violations = []

        # Check VaR
        if var_95 > self.max_var_95:
violations.append(
                ConstraintViolation(
                    constraint_name="value_at_risk",
violation_type="risk_exceeded",
current_value=var_95,
expected_range=(0.0, self.max_var_95),
                    severity="error",
message=(
                        f"95% VaR {var_95:.1%} exceeds maximum "
f"{self.max_var_95:.1%}"
),
remediation_suggestion=(
                        "Reduce position sizes or improve diversification"
),



        # Check drawdown
        if max_drawdown > self.max_drawdown:
violations.append(
                ConstraintViolation(
                    constraint_name="maximum_drawdown",
violation_type="risk_exceeded",
current_value=max_drawdown,
expected_range=(0.0, self.max_drawdown),
                    severity="critical",
message=(
                        f"Maximum drawdown {max_drawdown:.1%} exceeds "
f"limit {self.max_drawdown:.1%}"
),
remediation_suggestion=(
                        "Implement stop-loss mechanisms or reduce risk " "exposure"
),



        # Check Sharpe ratio
        if sharpe_ratio < self.min_sharpe_ratio:
violations.append(
                ConstraintViolation(
                    constraint_name="sharpe_ratio",
violation_type="performance_below_threshold",
current_value=sharpe_ratio,
expected_range=(self.min_sharpe_ratio, float("in")),
                    severity="warning",
message=(
                        f"Sharpe ratio {sharpe_ratio:.2f} below minimum "
f"{self.min_sharpe_ratio:.2f}"
),
remediation_suggestion=(
                        "Improve risk-adjusted returns or reduce volatility"
),



        return violations


class ConstraintValidator:
    """Main constraint validation system."""

    def __init__(self) -> None:
        """Initialize constraint validator."""
self.version = "1.0.0"
self.trading_constraints = TradingConstraints()
        self.mathematical_constraints = MathematicalConstraints()
        self.risk_constraints = RiskConstraints()

logger.info(f"ConstraintValidator v{self.version} initialized")

    def validate_trading_operation(
        self, operation_params: Dict[str, Any]
) -> ValidationResult:
"""Validate a complete trading operation."""
        import time

start_time = time.time()

violations = []
warnings = []

        # Validate position size
        if "position_size" in operation_params:
pos_violation = self.trading_constraints.validate_position_size(
                operation_params["position_size"]

            if pos_violation:
violations.append(pos_violation)

        # Validate leverage
        if "leverage" in operation_params:
lev_violation = self.trading_constraints.validate_leverage(
                operation_params["leverage"]

            if lev_violation:
violations.append(lev_violation)

        # Validate portfolio weights
        if "asset_weights" in operation_params:
div_violations = (
                self.trading_constraints.validate_portfolio_diversification(
                    operation_params["asset_weights"]


violations.extend(div_violations)

        # Validate risk metrics
        if all(
            key in operation_params
            for key in ["var_95", "max_drawdown", "sharpe_ratio"]
):
risk_violations = self.risk_constraints.validate_risk_metrics(
                operation_params["var_95"],
operation_params["max_drawdown"],
operation_params["sharpe_ratio"],

violations.extend(risk_violations)

        # Calculate risk score
risk_score = self._calculate_risk_score(violations)

        # Determine if valid
critical_violations = [v for v in violations if v.severity == "critical"]
error_violations = [v for v in violations if v.severity == "error"]

valid = len(critical_violations) == 0 and len(error_violations) == 0

execution_time = time.time() - start_time

        return ValidationResult(
            valid=valid,
violations=violations,
warnings=warnings,
risk_score=risk_score,
execution_time=execution_time,


    def validate_mathematical_operation(
        self, math_params: Dict[str, Any]
) -> ValidationResult:
"""Validate a mathematical operation."""
        import time

start_time = time.time()

violations = []
warnings = []

        # Validate matrix properties
        if "matrix" in math_params:
matrix_violations = (
                self.mathematical_constraints.validate_matrix_properties(
                    math_params["matrix"]


violations.extend(matrix_violations)

        # Validate optimization parameters
        if "iterations" in math_params and "tolerance" in math_params:
opt_violations = (
                self.mathematical_constraints.validate_optimization_parameters(
                    math_params["iterations"],
math_params["tolerance"],
math_params.get("gradient_norm"),


violations.extend(opt_violations)

        # Calculate risk score
risk_score = self._calculate_risk_score(violations)

        # Determine if valid
critical_violations = [v for v in violations if v.severity == "critical"]
error_violations = [v for v in violations if v.severity == "error"]

valid = len(critical_violations) == 0 and len(error_violations) == 0

execution_time = time.time() - start_time

        return ValidationResult(
            valid=valid,
violations=violations,
warnings=warnings,
risk_score=risk_score,
execution_time=execution_time,


    def _calculate_risk_score(self, violations: List[ConstraintViolation]) -> float:
        """Calculate overall risk score from violations."""
        if not violations:
            return 0.0

severity_weights = {"warning": 0.1, "error": 0.5, "critical": 1.0}

total_score = sum(severity_weights.get(v.severity, 0.5) for v in violations)
        normalized_score = unified_math.min(total_score / len(violations), 1.0)

        return normalized_score

    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of all constraints."""
        return {
"version": self.version,
"trading_constraints": {
"max_position_size": float(self.trading_constraints.max_position_size),
                "max_leverage": float(self.trading_constraints.max_leverage),
                "min_liquidity_ratio": float(
                    self.trading_constraints.min_liquidity_ratio
),
"max_sector_concentration": float(
                    self.trading_constraints.max_sector_concentration
),
"max_single_asset_weight": float(
                    self.trading_constraints.max_single_asset_weight
),
"min_diversification_count": (
                    self.trading_constraints.min_diversification_count
),
},
"mathematical_constraints": {
"max_matrix_size": (self.mathematical_constraints.max_matrix_size),
                "min_matrix_condition_number": (
                    self.mathematical_constraints.min_matrix_condition_number
),
"max_iterations": (self.mathematical_constraints.max_iterations),
                "numerical_tolerance": (
                    self.mathematical_constraints.numerical_tolerance
),
},
"risk_constraints": {
"max_var_95": self.risk_constraints.max_var_95,
"max_drawdown": self.risk_constraints.max_drawdown,
"min_sharpe_ratio": self.risk_constraints.min_sharpe_ratio,
"max_correlation_exposure": (
                    self.risk_constraints.max_correlation_exposure
),
},
}


def main() -> None:
    """Demo of constraint validation system."""
    try:
validator = ConstraintValidator()
        safe_print(f"[OK] ConstraintValidator v{validator.version} initialized")

        # Test trading operation validation
trading_params = {
"position_size": 0.8,
"leverage": 1.5,
"asset_weights": {"BTC": 0.4, "ETH": 0.3, "USDC": 0.3},
"var_95": 0.03,
"max_drawdown": 0.15,
"sharpe_ratio": 0.75,
}

trading_result = validator.validate_trading_operation(trading_params)
        safe_print(
            "[TRADING] Trading validation: "
f"{'PASS' if trading_result.valid else 'FAIL'}"

safe_print(f"   Risk score: {trading_result.risk_score:.3f}")
        safe_print(f"   Violations: {len(trading_result.violations)}")

        # Test mathematical operation validation
#         from core.unified_math_system import unified_math  # F811: duplicate import

test_matrix = np.random.randn(5, 5)

math_params = {
"matrix": test_matrix,
"iterations": 500,
"tolerance": 1e-8,
"gradient_norm": 10.5,
}

math_result = validator.validate_mathematical_operation(math_params)
        safe_print(
            "[MATH] Mathematical validation: "
f"{'PASS' if math_result.valid else 'FAIL'}"

safe_print(f"   Risk score: {math_result.risk_score:.3f}")
        safe_print(f"   Violations: {len(math_result.violations)}")

        # Display constraint summary
summary = validator.get_constraint_summary()
        safe_print(f"[SUMMARY] Constraint summary available with {len(summary)} categories")

safe_print("[SUCCESS] Constraint validation demo completed successfully!")

    except Exception as e:
safe_print(f"[ERROR] Demo failed: {e}")


if __name__ == "__main__":
main()

# Backward compatibility alias
Constraints = ConstraintValidator
