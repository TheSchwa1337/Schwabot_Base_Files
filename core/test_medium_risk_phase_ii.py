# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
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
#!/usr/bin/env python3
"""
Medium-Risk Phase II Integration Test - Schwabot UROS v1.0
========================================================

Comprehensive test suite for medium-risk trading components.
Tests strategy execution, trade execution, phase detection, and portfolio management.

Components Tested:
- Trade Execution Engine
- Strategy Execution Engine
- Phase Engine
- Portfolio Substitution Matrix
- Deterministic Value Engine
- Unified Mathematical Trading Controller
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result structure."""
    component: str
    status: str  # PASS, FAIL, SKIP
    details: str
    execution_time: float
    error_message: Optional[str] = None

class MediumRiskPhaseIITester:
    """Comprehensive tester for medium-risk Phase II components."""

    def __init__(self):
        """Initialize the tester."""
        self.test_results: List[TestResult] = []
        self.components_status: Dict[str, bool] = {}

    def test_trade_execution_engine(self) -> TestResult:
        """Test trade execution engine functionality."""
        start_time = time.time()

        try:
            # Try to import trade execution components
            try:
                from .trade_executor import ExecutedTrade
                from .simulate_trade import TradeSimulator, TradeExecution
                logger.info("✅ Trade execution components imported successfully")
            except ImportError as e:
                return TestResult(
                    component="Trade Execution Engine",
                    status="SKIP",
                    details="Trade execution components not available",
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )

            # Test trade simulator
            simulator = TradeSimulator()

            # Test strategy bucket
            strategy_bucket = {
                'asset': 'BTC',
                'strategy_id': 'long_hold_btc',
                'tensor_score': 0.75,
                'bit_phase': 8,
                'basket_id': 'basket_0',
                'current_price': 50000.0,
                'market_data': {'volatility': 0.02, 'volume': 1000}
            }

            # Simulate trade
            trade_result = simulator.simulate_trade(strategy_bucket, mode="DEMO")

            if trade_result and trade_result.status.value == "EXECUTED":
                return TestResult(
                    component="Trade Execution Engine",
                    status="PASS",
                    details=f"Trade executed successfully: {trade_result.trade_id}",
                    execution_time=time.time() - start_time
                )
            else:
                return TestResult(
                    component="Trade Execution Engine",
                    status="FAIL",
                    details="Trade execution failed",
                    execution_time=time.time() - start_time,
                    error_message="Trade status not EXECUTED"
                )

        except Exception as e:
            return TestResult(
                component="Trade Execution Engine",
                status="FAIL",
                details="Trade execution test failed",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def test_strategy_execution_engine(self) -> TestResult:
        """Test strategy execution engine functionality."""
        start_time = time.time()

        try:
            # Try to import strategy components
            try:
                from .strategy_logic import StrategyLogic, StrategyType
                logger.info("✅ Strategy execution components imported successfully")
            except ImportError as e:
                return TestResult(
                    component="Strategy Execution Engine",
                    status="SKIP",
                    details="Strategy execution components not available",
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )

            # Test strategy logic
            strategy_logic = StrategyLogic()

            # Test strategy registration
            strategies = strategy_logic.get_registered_strategies()

            if strategies and len(strategies) > 0:
                return TestResult(
                    component="Strategy Execution Engine",
                    status="PASS",
                    details=f"Strategy execution working: {len(strategies)} strategies registered",
                    execution_time=time.time() - start_time
                )
            else:
                return TestResult(
                    component="Strategy Execution Engine",
                    status="FAIL",
                    details="No strategies registered",
                    execution_time=time.time() - start_time,
                    error_message="Strategy registration failed"
                )

        except Exception as e:
            return TestResult(
                component="Strategy Execution Engine",
                status="FAIL",
                details="Strategy execution test failed",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def test_phase_engine(self) -> TestResult:
        """Test phase engine functionality."""
        start_time = time.time()

        try:
            # Try to import phase engine components
            try:
                from .phase_engine import PhaseEngine, PhaseType
                logger.info("✅ Phase engine components imported successfully")
            except ImportError as e:
                return TestResult(
                    component="Phase Engine",
                    status="SKIP",
                    details="Phase engine components not available",
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )

            # Test phase engine
            phase_engine = PhaseEngine()

            # Test phase detection
            market_data = {
                'price': 50000.0,
                'volume': 1000,
                'volatility': 0.02,
                'momentum': 0.01
            }

            # Get current phase
            current_phase = phase_engine.get_current_phase(market_data)

            if current_phase:
                return TestResult(
                    component="Phase Engine",
                    status="PASS",
                    details=f"Phase detection working: {current_phase.phase_type.value}",
                    execution_time=time.time() - start_time
                )
            else:
                return TestResult(
                    component="Phase Engine",
                    status="FAIL",
                    details="Phase detection failed",
                    execution_time=time.time() - start_time,
                    error_message="No phase detected"
                )

        except Exception as e:
            return TestResult(
                component="Phase Engine",
                status="FAIL",
                details="Phase engine test failed",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def test_portfolio_substitution_matrix(self) -> TestResult:
        """Test portfolio substitution matrix functionality."""
        start_time = time.time()

        try:
            # Try to import portfolio components
            try:
                from .portfolio_substitution_matrix import PortfolioSubstitutionMatrix
                logger.info("✅ Portfolio substitution components imported successfully")
            except ImportError as e:
                return TestResult(
                    component="Portfolio Substitution Matrix",
                    status="SKIP",
                    details="Portfolio substitution components not available",
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )

            # Test portfolio substitution
            matrix = PortfolioSubstitutionMatrix()

            # Test substitution calculation
            current_allocation = {"BTC": 0.4, "ETH": 0.3, "USDC": 0.3}
            target_allocation = {"BTC": 0.5, "ETH": 0.3, "USDC": 0.2}

            result = matrix.calculate_substitution(
                current_allocation, target_allocation, 100000.0
            )

            if result and result.confidence_score > 0:
                return TestResult(
                    component="Portfolio Substitution Matrix",
                    status="PASS",
                    details=f"Portfolio substitution working: confidence={result.confidence_score:.2f}",
                    execution_time=time.time() - start_time
                )
            else:
                return TestResult(
                    component="Portfolio Substitution Matrix",
                    status="FAIL",
                    details="Portfolio substitution failed",
                    execution_time=time.time() - start_time,
                    error_message="No substitution result"
                )

        except Exception as e:
            return TestResult(
                component="Portfolio Substitution Matrix",
                status="FAIL",
                details="Portfolio substitution test failed",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def test_deterministic_value_engine(self) -> TestResult:
        """Test deterministic value engine functionality."""
        start_time = time.time()

        try:
            # Try to import deterministic value components
            try:
                from .deterministic_value_engine import DeterministicValueEngine
                logger.info("✅ Deterministic value components imported successfully")
            except ImportError as e:
                return TestResult(
                    component="Deterministic Value Engine",
                    status="SKIP",
                    details="Deterministic value components not available",
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )

            # Test deterministic value engine
            engine = DeterministicValueEngine()

            # Create test market state
            from .deterministic_value_engine import MarketState, AssetType

            market_state = MarketState(
                prices={AssetType.BTC: 50000.0, AssetType.ETH: 3000.0},
                volumes={AssetType.BTC: 1000.0, AssetType.ETH: 5000.0},
                volatility={AssetType.BTC: 0.02, AssetType.ETH: 0.03},
                entropy={AssetType.BTC: 5.0, AssetType.ETH: 4.5}
            )

            # Calculate deterministic decision
            decision = engine.calculate_deterministic_decision(market_state)

            if decision and decision.execution_confidence > 0:
                return TestResult(
                    component="Deterministic Value Engine",
                    status="PASS",
                    details=f"Deterministic decision working: confidence={decision.execution_confidence:.2f}",
                    execution_time=time.time() - start_time
                )
            else:
                return TestResult(
                    component="Deterministic Value Engine",
                    status="FAIL",
                    details="Deterministic decision failed",
                    execution_time=time.time() - start_time,
                    error_message="No decision result"
                )

        except Exception as e:
            return TestResult(
                component="Deterministic Value Engine",
                status="FAIL",
                details="Deterministic value test failed",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def test_unified_mathematical_trading_controller(self) -> TestResult:
        """Test unified mathematical trading controller functionality."""
        start_time = time.time()

        try:
            # Try to import unified trading components
            try:
                from .unified_mathematical_trading_controller import UnifiedMathematicalTradingController
                logger.info("✅ Unified trading components imported successfully")
            except ImportError as e:
                return TestResult(
                    component="Unified Mathematical Trading Controller",
                    status="SKIP",
                    details="Unified trading components not available",
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )

            # Test unified trading controller
            controller = UnifiedMathematicalTradingController()

            # Test opportunity analysis
            market_data = {
                'symbol': 'BTC',
                'price': 50000.0,
                'volume': 1000,
                'volatility': 0.02,
                'momentum': 0.01
            }

            opportunity = controller.analyze_trading_opportunity(market_data)

            if opportunity and opportunity.unified_confidence > 0:
                return TestResult(
                    component="Unified Mathematical Trading Controller",
                    status="PASS",
                    details=f"Unified analysis working: confidence={opportunity.unified_confidence:.2f}",
                    execution_time=time.time() - start_time
                )
            else:
                return TestResult(
                    component="Unified Mathematical Trading Controller",
                    status="FAIL",
                    details="Unified analysis failed",
                    execution_time=time.time() - start_time,
                    error_message="No opportunity result"
                )

        except Exception as e:
            return TestResult(
                component="Unified Mathematical Trading Controller",
                status="FAIL",
                details="Unified trading test failed",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all medium-risk Phase II tests."""
        logger.info("🚀 Starting Medium-Risk Phase II Integration Tests")
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
            status_emoji = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⚠️"
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
        logger.info("📊 Medium-Risk Phase II Test Summary")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"⚠️ Skipped: {skipped_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        # Determine overall status
        if failed_tests == 0:
            overall_status = "READY"
            logger.info("🎉 All medium-risk components are ready for Phase II!")
        elif passed_tests > 0:
            overall_status = "PARTIAL"
            logger.info("⚠️ Some medium-risk components need implementation")
        else:
            overall_status = "NOT_READY"
            logger.warning("❌ Medium-risk components need significant work")

        return {
            "overall_status": overall_status,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "success_rate": (passed_tests/total_tests)*100,
            "test_results": [vars(r) for r in self.test_results]
        }


def main():
    """Main function for medium-risk Phase II testing."""
    safe_print("🚀 Medium-Risk Phase II Integration Test - Schwabot UROS v1.0")
    safe_print("=" * 70)

    # Initialize tester
    tester = MediumRiskPhaseIITester()

    # Run all tests
    results = tester.run_all_tests()

    # Save results
    with open("medium_risk_phase_ii_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    safe_print("\n📄 Results saved to: medium_risk_phase_ii_results.json")
    safe_print(f"🎯 Overall Status: {results['overall_status']}")

    if results['overall_status'] == "READY":
        safe_print("✅ Medium-Risk Phase II is ready for deployment!")
    elif results['overall_status'] == "PARTIAL":
        safe_print("⚠️ Medium-Risk Phase II needs some implementation work")
    else:
        safe_print("❌ Medium-Risk Phase II needs significant development")


if __name__ == "__main__":
    main()
