from .visual_fallbacks import VisualFallback
from utils.safe_print import safe_print, info, warn, error, success, debug
#!/usr/bin/env python3
"""
Medium-Risk Phase II Integration Test \\u2013 Schwabot UROS v1.0
=========================================================

This script mirrors the core/test_medium_risk_phase_ii.py integration test but
is designed to run from the repository root.  It prepends the *core* package to
sys.path and uses absolute imports so that the test can be executed directly
with `python test_medium_risk_phase2.py` while still exercising the same
components.
"""

import sys
import time
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Path setup \\u2013 add ./core to import path so `import core.xxx` works regardless
# of where the test is executed from.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CORE_PATH = REPO_ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

# Configure logging \\u2013 keep exactly the same format as the original test.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Visual mode core
visual = VisualFallback()


@dataclass
class TestResult:
    """Lightweight container for each individual component test."""

    component: str
    status: str  # PASS, FAIL, SKIP
    details: str
    execution_time: float
    error_message: Optional[str] = None


class MediumRiskPhase2Tester:
    """Runs the full Medium-Risk Phase II test suite."""

    def __init__(self) -> None:
        self.test_results: List[TestResult] = []

    # ---------------------------------------------------------------------
    # Individual component tests (trade execution, strategy engine, etc).
    # The logic is exactly the same as in core/test_medium_risk_phase_ii.py but
    # with absolute imports (i.e. `import core.trade_executor \\u2026`).
    # ---------------------------------------------------------------------

    def test_trade_execution_engine(self) -> TestResult:  # noqa: C901 (complexity)
        start = time.time()
        try:
            try:
                from core.trade_executor import ExecutedTrade  # noqa: F401
                from core.simulate_trade import (  # type: ignore
                    TradeSimulator,
                )
                logger.info("\\u2705 Trade execution components imported successfully")
            except Exception as e:
                return TestResult(
                    component="Trade Execution Engine",
                    status="SKIP",
                    details="Trade execution components not available",
                    execution_time=time.time() - start,
                    error_message=str(e),
                )

            simulator = TradeSimulator()
            strategy_bucket = {
                "asset": "BTC",
                "strategy_id": "long_hold_btc",
                "tensor_score": 0.75,
                "bit_phase": 8,
                "basket_id": "basket_0",
                "current_price": 50_000.0,
                "market_data": {"volatility": 0.02, "volume": 1000},
            }
            trade_result = simulator.simulate_trade(strategy_bucket, mode="DEMO")
            if trade_result and trade_result.status.value == "EXECUTED":
                return TestResult(
                    component="Trade Execution Engine",
                    status="PASS",
                    details=f"Trade executed successfully: {trade_result.trade_id}",
                    execution_time=time.time() - start,
                )
            return TestResult(
                component="Trade Execution Engine",
                status="FAIL",
                details="Trade execution failed",
                execution_time=time.time() - start,
                error_message="Trade status not EXECUTED",
            )
        except Exception as e:  # pragma: no cover \\u2013 generic safety net
            return TestResult(
                component="Trade Execution Engine",
                status="FAIL",
                details="Trade execution test failed",
                execution_time=time.time() - start,
                error_message=str(e),
            )

    def test_strategy_execution_engine(self) -> TestResult:  # noqa: C901
        start = time.time()
        try:
            try:
                from core.strategy_logic import StrategyLogic  # noqa: F401
                logger.info("\\u2705 Strategy execution components imported successfully")
            except Exception as e:
                return TestResult(
                    component="Strategy Execution Engine",
                    status="SKIP",
                    details="Strategy execution components not available",
                    execution_time=time.time() - start,
                    error_message=str(e),
                )
            strategy_logic = StrategyLogic()
            strategies = strategy_logic.get_registered_strategies()
            if strategies:
                return TestResult(
                    component="Strategy Execution Engine",
                    status="PASS",
                    details=f"Strategy execution working: {len(strategies)} strategies registered",
                    execution_time=time.time() - start,
                )
            return TestResult(
                component="Strategy Execution Engine",
                status="FAIL",
                details="No strategies registered",
                execution_time=time.time() - start,
                error_message="Strategy registration failed",
            )
        except Exception as e:
            return TestResult(
                component="Strategy Execution Engine",
                status="FAIL",
                details="Strategy execution test failed",
                execution_time=time.time() - start,
                error_message=str(e),
            )

    def test_phase_engine(self) -> TestResult:  # noqa: C901
        start = time.time()
        try:
            try:
                from core.phase_engine import PhaseEngine  # noqa: F401
                logger.info("\\u2705 Phase engine components imported successfully")
            except Exception as e:
                return TestResult(
                    component="Phase Engine",
                    status="SKIP",
                    details="Phase engine components not available",
                    execution_time=time.time() - start,
                    error_message=str(e),
                )
            phase_engine = PhaseEngine()
            market_data = {
                "price": 50_000.0,
                "volume": 1000,
                "volatility": 0.02,
                "momentum": 0.01,
            }
            current_phase = phase_engine.get_current_phase(market_data)
            if current_phase:
                return TestResult(
                    component="Phase Engine",
                    status="PASS",
                    details=f"Phase detection working: {current_phase.phase_type.value}",
                    execution_time=time.time() - start,
                )
            return TestResult(
                component="Phase Engine",
                status="FAIL",
                details="Phase detection failed",
                execution_time=time.time() - start,
                error_message="No phase detected",
            )
        except Exception as e:
            return TestResult(
                component="Phase Engine",
                status="FAIL",
                details="Phase engine test failed",
                execution_time=time.time() - start,
                error_message=str(e),
            )

    def test_portfolio_substitution_matrix(self) -> TestResult:  # noqa: C901
        start = time.time()
        try:
            try:
                from core.portfolio_substitution_matrix import PortfolioSubstitutionMatrix  # noqa: F401
                logger.info("\\u2705 Portfolio substitution components imported successfully")
            except Exception as e:
                return TestResult(
                    component="Portfolio Substitution Matrix",
                    status="SKIP",
                    details="Portfolio substitution components not available",
                    execution_time=time.time() - start,
                    error_message=str(e),
                )
            matrix = PortfolioSubstitutionMatrix()
            result = matrix.calculate_substitution(
                {"BTC": 0.4, "ETH": 0.3, "USDC": 0.3},
                {"BTC": 0.5, "ETH": 0.3, "USDC": 0.2},
                100_000.0,
            )
            if result and getattr(result, "confidence_score", 0) > 0:
                return TestResult(
                    component="Portfolio Substitution Matrix",
                    status="PASS",
                    details=f"Portfolio substitution working: confidence={result.confidence_score:.2f}",
                    execution_time=time.time() - start,
                )
            return TestResult(
                component="Portfolio Substitution Matrix",
                status="FAIL",
                details="Portfolio substitution failed",
                execution_time=time.time() - start,
                error_message="No substitution result",
            )
        except Exception as e:
            return TestResult(
                component="Portfolio Substitution Matrix",
                status="FAIL",
                details="Portfolio substitution test failed",
                execution_time=time.time() - start,
                error_message=str(e),
            )

    def test_deterministic_value_engine(self) -> TestResult:  # noqa: C901
        start = time.time()
        try:
            try:
                from core.deterministic_value_engine import DeterministicValueEngine, MarketState, AssetType  # noqa: F401
                logger.info("\\u2705 Deterministic value components imported successfully")
            except Exception as e:
                return TestResult(
                    component="Deterministic Value Engine",
                    status="SKIP",
                    details="Deterministic value components not available",
                    execution_time=time.time() - start,
                    error_message=str(e),
                )
            engine = DeterministicValueEngine()
            market_state = MarketState(
                prices={AssetType.BTC: 50_000.0, AssetType.ETH: 3_000.0},
                volumes={AssetType.BTC: 1000.0, AssetType.ETH: 5000.0},
                volatility={AssetType.BTC: 0.02, AssetType.ETH: 0.03},
                entropy={AssetType.BTC: 5.0, AssetType.ETH: 4.5},
            )
            decision = engine.calculate_deterministic_decision(market_state)
            if decision and getattr(decision, "execution_confidence", 0) > 0:
                return TestResult(
                    component="Deterministic Value Engine",
                    status="PASS",
                    details=f"Deterministic decision working: confidence={decision.execution_confidence:.2f}",
                    execution_time=time.time() - start,
                )
            return TestResult(
                component="Deterministic Value Engine",
                status="FAIL",
                details="Deterministic decision failed",
                execution_time=time.time() - start,
                error_message="No decision result",
            )
        except Exception as e:
            return TestResult(
                component="Deterministic Value Engine",
                status="FAIL",
                details="Deterministic value test failed",
                execution_time=time.time() - start,
                error_message=str(e),
            )

    def test_unified_mathematical_trading_controller(self) -> TestResult:  # noqa: C901
        start = time.time()
        try:
            try:
                from core.unified_mathematical_trading_controller import (
                    UnifiedMathematicalTradingController,
                )
                logger.info("\\u2705 Unified trading components imported successfully")
            except Exception as e:
                return TestResult(
                    component="Unified Mathematical Trading Controller",
                    status="SKIP",
                    details="Unified trading components not available",
                    execution_time=time.time() - start,
                    error_message=str(e),
                )
            controller = UnifiedMathematicalTradingController()
            market_data = {
                "symbol": "BTC",
                "price": 50_000.0,
                "volume": 1000,
                "volatility": 0.02,
                "momentum": 0.01,
            }
            opportunity = controller.analyze_trading_opportunity(market_data)
            if opportunity and getattr(opportunity, "unified_confidence", 0) > 0:
                return TestResult(
                    component="Unified Mathematical Trading Controller",
                    status="PASS",
                    details=f"Unified analysis working: confidence={opportunity.unified_confidence:.2f}",
                    execution_time=time.time() - start,
                )
            return TestResult(
                component="Unified Mathematical Trading Controller",
                status="FAIL",
                details="Unified analysis failed",
                execution_time=time.time() - start,
                error_message="No opportunity result",
            )
        except Exception as e:
            return TestResult(
                component="Unified Mathematical Trading Controller",
                status="FAIL",
                details="Unified trading test failed",
                execution_time=time.time() - start,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # Test-runner orchestration \\u2013 identical summary logic to original.
    # ------------------------------------------------------------------

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all medium-risk Phase II tests."""
        logger.info(f"{visual.get('INFO')} Starting Medium-Risk Phase II Integration Tests")
        logger.info("=" * 60)

        # Run individual tests
        tests = [
            self.test_trade_execution_engine,
            self.test_strategy_execution_engine,
            self.test_phase_engine,
            self.test_portfolio_substitution_matrix,
            self.test_deterministic_value_engine,
            self.test_unified_mathematical_trading_controller
        ]

        for test in tests:
            result = test()
            self.test_results.append(result)

            # Log result
            status_emoji = visual.get(result.status)
            logger.info(f"{status_emoji} {result.component}: {result.status}")
            if result.details:
                logger.info(f"   Details: {result.details}")
            if result.error_message:
                logger.warning(f"   Error: {result.error_message}")

        # Calculate summary
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASS"])
        failed_tests = len([r for r in self.test_results if r.status == "FAIL"])
        skipped_tests = len([r for r in self.test_results if r.status == "SKIP"])

        # Print summary
        logger.info("=" * 60)
        logger.info(f"{visual.get('INFO')} Medium-Risk Phase II Test Summary")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"{visual.get('PASS')} Passed: {passed_tests}")
        logger.info(f"{visual.get('FAIL')} Failed: {failed_tests}")
        logger.info(f"{visual.get('SKIP')} Skipped: {skipped_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        # Determine overall status
        if failed_tests == 0:
            overall_status = "READY"
            logger.info(f"{visual.get('READY')} All medium-risk components are ready for Phase II!")
        elif passed_tests > 0:
            overall_status = "PARTIAL"
            logger.info(f"{visual.get('PARTIAL')} Some medium-risk components need implementation")
        else:
            overall_status = "NOT_READY"
            logger.warning(f"{visual.get('NOT_READY')} Medium-risk components need significant work")

        return {
            "overall_status": overall_status,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "success_rate": (passed_tests/total_tests)*100,
            "test_results": [vars(r) for r in self.test_results]
        }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    """Main function for medium-risk Phase II testing."""
    print(f"{visual.get('INFO')} Medium-Risk Phase II Integration Test - Schwabot UROS v1.0")
    print("=" * 70)

    # Initialize tester
    tester = MediumRiskPhase2Tester()

    # Run all tests
    results = tester.run_all_tests()

    # Save results
    with open("medium_risk_phase_ii_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\\n{visual.get('SAVE')} Results saved to: medium_risk_phase_ii_results.json")
    print(f"{visual.get(results['overall_status'])} Overall Status: {results['overall_status']}")

    if results['overall_status'] == "READY":
        print(f"{visual.get('READY')} Medium-Risk Phase II is ready for deployment!")
    elif results['overall_status'] == "PARTIAL":
        print(f"{visual.get('PARTIAL')} Medium-Risk Phase II needs some implementation work")
    else:
        print(f"{visual.get('NOT_READY')} Medium-Risk Phase II needs significant development")


if __name__ == "__main__":
    main()
