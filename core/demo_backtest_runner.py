# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import numpy as np
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
"""
Schwabot Demo Backtest Runner
============================

Comprehensive backtest runner that orchestrates all demo testing,
provides detailed analysis, and generates comprehensive reports.

This system:
- Runs comprehensive backtests across all strategies
- Integrates with all demo components
- Provides detailed performance analysis
- Generates comprehensive reports
- Enables reinforcement learning from backtest results
"""

import json
import yaml
# from core.unified_math_system import unified_math  # F811: duplicate import
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import threading
import time

from .settings_controller import get_settings_controller
from .vector_validator import get_vector_validator
from .matrix_allocator import get_matrix_allocator
from .demo_integration_system import get_demo_integration_system
from .demo_entry_simulator import get_demo_entry_simulator


@dataclass
class BacktestConfig:
    """Configuration for backtest runs"""
backtest_id: str
strategy_types: List[str]
market_conditions: List[str]
num_trades_per_strategy: int
base_price: float
price_volatility: float
volume_multiplier: float
enable_reinforcement_learning: bool
enable_performance_tracking: bool
save_detailed_results: bool
timestamp: datetime


@dataclass
class BacktestResult:
    """Result of a backtest run"""
backtest_id: str
total_trades: int
successful_trades: int
success_rate: float
total_profit: float
average_profit: float
max_drawdown: float
sharpe_ratio: float
strategy_performance: Dict[str, Dict[str, float]]
matrix_performance: Dict[str, Dict[str, float]]
market_condition_performance: Dict[str, Dict[str, float]]
reinforcement_learning_updates: Dict[str, Any]
execution_time: float
timestamp: datetime


class DemoBacktestRunner:
    """Comprehensive backtest runner for Schwabot demo system"""

    def __init__(self):
        self.settings_controller = get_settings_controller()
        self.vector_validator = get_vector_validator()
        self.matrix_allocator = get_matrix_allocator()
        self.demo_system = get_demo_integration_system()
        self.entry_simulator = get_demo_entry_simulator()

        # Backtest data
self.backtest_configs: List[BacktestConfig] = []
self.backtest_results: List[BacktestResult] = []
self.backtest_history: Dict[str, List[BacktestResult]] = {}

        # Performance tracking
self.performance_metrics = {
"total_backtests": 0,
"total_trades": 0,
"overall_success_rate": 0.0,
"overall_profit": 0.0,
"best_backtest": None,
"worst_backtest": None
}

        # Initialize backtest directories
self._initialize_backtest_directories()

        # Load existing backtest data
self._load_backtest_data()

    def _initialize_backtest_directories(self):
        """Initialize backtest-related directories"""
backtest_dirs = [
"tests/demo_backlog/",
"tests/demo_results/",
"tests/demo_configs/",
"tests/demo_analysis/",
"tests/demo_reports/"
]

        for dir_path in backtest_dirs:
Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _load_backtest_data(self):
        """Load existing backtest data from files"""
        try:
            # Load backtest results
results_file = Path("tests/demo_results/backtest_results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                    self.backtest_results = [BacktestResult(**result) for result in results_data]

            # Update performance metrics
self._update_performance_metrics()

        except Exception as e:
safe_print(f"Warning: Could not load backtest data: {e}")

    def _update_performance_metrics(self):
        """Update performance metrics from backtest results"""
        if not self.backtest_results:
return

self.performance_metrics["total_backtests"] = len(self.backtest_results)
        self.performance_metrics["total_trades"] = sum(r.total_trades for r in self.backtest_results)

        # Calculate overall success rate
total_successful = sum(r.successful_trades for r in self.backtest_results)
        self.performance_metrics["overall_success_rate"] = total_successful / self.performance_metrics["total_trades"]

        # Calculate overall profit
self.performance_metrics["overall_profit"] = sum(r.total_profit for r in self.backtest_results)

        # Find best and worst backtests
        if self.backtest_results:
best_backtest = unified_math.max(self.backtest_results, key=lambda x: x.success_rate)
            worst_backtest = unified_math.min(self.backtest_results, key=lambda x: x.success_rate)

self.performance_metrics["best_backtest"] = best_backtest.backtest_id
self.performance_metrics["worst_backtest"] = worst_backtest.backtest_id

    def create_backtest_config(self, strategy_types: List[str] = None,
                             market_conditions: List[str] = None,
num_trades_per_strategy: int = 100,
base_price: float = 50000.0,
price_volatility: float = 0.02,
volume_multiplier: float = 1.0,
enable_reinforcement_learning: bool = True,
enable_performance_tracking: bool = True,
save_detailed_results: bool = True) -> BacktestConfig:
"""Create a new backtest configuration"""

        # Default strategies if none provided
        if strategy_types is None:
strategy_types = list(self.entry_simulator.entry_strategies.keys())

        # Default market conditions if none provided
        if market_conditions is None:
market_conditions = list(self.entry_simulator.market_conditions.keys())

        # Generate unique backtest ID
backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(strategy_types) % 1000}"

config = BacktestConfig(
            backtest_id=backtest_id,
strategy_types=strategy_types,
market_conditions=market_conditions,
num_trades_per_strategy=num_trades_per_strategy,
base_price=base_price,
price_volatility=price_volatility,
volume_multiplier=volume_multiplier,
enable_reinforcement_learning=enable_reinforcement_learning,
enable_performance_tracking=enable_performance_tracking,
save_detailed_results=save_detailed_results,
timestamp=datetime.now()


self.backtest_configs.append(config)

        return config

    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Run a comprehensive backtest based on configuration"""
safe_print(f"🚀 Starting backtest: {config.backtest_id}")
        safe_print(f"Strategies: {config.strategy_types}")
        safe_print(f"Market Conditions: {config.market_conditions}")
        safe_print(f"Trades per strategy: {config.num_trades_per_strategy}")

start_time = time.time()

        # Start demo mode
self.demo_system.start_demo_mode("backtest")

        # Initialize result tracking
total_trades = 0
successful_trades = 0
total_profit = 0.0
profits = []

strategy_performance = {}
matrix_performance = {}
market_condition_performance = {}
reinforcement_updates = {}

        # Run backtests for each strategy and market condition combination
        for strategy_type in config.strategy_types:
strategy_performance[strategy_type] = {}

            for market_condition in config.market_conditions:
safe_print(f"Testing {strategy_type} in {market_condition} market...")

                # Create strategy config
strategy_config = {
"base_price": config.base_price,
"price_volatility": config.price_volatility,
"matrix_id": "SFS8-A5",  # Will be overridden by strategy
"strategy_type": strategy_type
}

                # Run backtest for this combination
backtest_analysis = self.demo_system.run_backtest(
                    strategy_config, config.num_trades_per_strategy


                # Update tracking
total_trades += backtest_analysis["total_trades"]
successful_trades += backtest_analysis["successful_trades"]
total_profit += backtest_analysis["total_profit"]

                # Store profits for drawdown calculation
profits.extend([backtest_analysis["average_profit"]] * backtest_analysis["total_trades"])

                # Store strategy performance
strategy_performance[strategy_type][market_condition] = {
"success_rate": backtest_analysis["success_rate"],
"total_profit": backtest_analysis["total_profit"],
"average_profit": backtest_analysis["average_profit"],
"total_trades": backtest_analysis["total_trades"]
}

                # Update matrix performance
                for matrix_id, perf in backtest_analysis["matrix_performance"].items():
                    if matrix_id not in matrix_performance:
matrix_performance[matrix_id] = {"trades": 0, "successes": 0, "profit": 0.0}

matrix_performance[matrix_id]["trades"] += perf["trades"]
matrix_performance[matrix_id]["successes"] += perf["successes"]
matrix_performance[matrix_id]["profit"] += perf["profit"]

                # Update market condition performance
                if market_condition not in market_condition_performance:
market_condition_performance[market_condition] = {"trades": 0, "successes": 0, "profit": 0.0}

market_condition_performance[market_condition]["trades"] += backtest_analysis["total_trades"]
market_condition_performance[market_condition]["successes"] += backtest_analysis["successful_trades"]
market_condition_performance[market_condition]["profit"] += backtest_analysis["total_profit"]

        # Stop demo mode
self.demo_system.stop_demo_mode()

        # Calculate final metrics
success_rate = successful_trades / total_trades if total_trades > 0 else 0.0
average_profit = total_profit / total_trades if total_trades > 0 else 0.0

        # Calculate max drawdown
max_drawdown = self._calculate_max_drawdown(profits)

        # Calculate Sharpe ratio
sharpe_ratio = self._calculate_sharpe_ratio(profits)

        # Calculate success rates for matrices and market conditions
        for matrix_id, perf in matrix_performance.items():
            perf["success_rate"] = perf["successes"] / perf["trades"]
perf["average_profit"] = perf["profit"] / perf["trades"]

        for market_condition, perf in market_condition_performance.items():
            perf["success_rate"] = perf["successes"] / perf["trades"]
perf["average_profit"] = perf["profit"] / perf["trades"]

        # Get reinforcement learning updates
        if config.enable_reinforcement_learning:
reinforcement_updates = {
"matrix_weights": self.settings_controller.matrix_path_weights,
"known_bad_vectors": len(self.settings_controller.known_bad_vectors),
                "vector_validator_performance": self.vector_validator.get_performance_summary()
            }

        # Create backtest result
result = BacktestResult(
            backtest_id=config.backtest_id,
total_trades=total_trades,
successful_trades=successful_trades,
success_rate=success_rate,
total_profit=total_profit,
average_profit=average_profit,
max_drawdown=max_drawdown,
sharpe_ratio=sharpe_ratio,
strategy_performance=strategy_performance,
matrix_performance=matrix_performance,
market_condition_performance=market_condition_performance,
reinforcement_learning_updates=reinforcement_updates,
execution_time=time.time() - start_time,
            timestamp=datetime.now()


        # Store result
self.backtest_results.append(result)

        # Update performance metrics
self._update_performance_metrics()

        # Save results if requested
        if config.save_detailed_results:
self._save_backtest_results()

safe_print("✅ Backtest completed!")
        safe_print(f"Success Rate: {success_rate:.2%}")
        safe_print(f"Total Profit: {total_profit:.2f}")
        safe_print(f"Execution Time: {result.execution_time:.2f}s")

        return result

    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """Calculate maximum drawdown from profit series"""
        if not profits:
            return 0.0

cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max

        return unified_math.abs(unified_math.min(drawdown)) if len(drawdown) > 0 else 0.0

    def _calculate_sharpe_ratio(self, profits: List[float]) -> float:
        """Calculate Sharpe ratio from profit series"""
        if not profits or len(profits) < 2:
            return 0.0

returns = np.array(profits)
        mean_return = unified_math.unified_math.mean(returns)
        std_return = unified_math.unified_math.std(returns)

        if std_return == 0:
            return 0.0

        # Assuming risk-free rate of 0 for simplicity
sharpe_ratio = mean_return / std_return

        return sharpe_ratio

    def run_comprehensive_backtest(self, num_trades_per_strategy: int = 50) -> Dict[str, Any]:
        """Run comprehensive backtest across all strategies and market conditions"""
safe_print("🚀 Starting comprehensive backtest...")

        # Create comprehensive config
config = self.create_backtest_config(
            strategy_types=list(self.entry_simulator.entry_strategies.keys()),
            market_conditions=list(self.entry_simulator.market_conditions.keys()),
            num_trades_per_strategy=num_trades_per_strategy,
enable_reinforcement_learning=True,
save_detailed_results=True


        # Run backtest
result = self.run_backtest(config)

        # Generate comprehensive analysis
analysis = self._generate_comprehensive_analysis(result)

        return analysis

    def _generate_comprehensive_analysis(self, result: BacktestResult) -> Dict[str, Any]:
        """Generate comprehensive analysis of backtest results"""
analysis = {
"backtest_id": result.backtest_id,
"summary": {
"total_trades": result.total_trades,
"success_rate": result.success_rate,
"total_profit": result.total_profit,
"average_profit": result.average_profit,
"max_drawdown": result.max_drawdown,
"sharpe_ratio": result.sharpe_ratio,
"execution_time": result.execution_time
},
"strategy_analysis": {},
"matrix_analysis": {},
"market_condition_analysis": {},
"reinforcement_learning_analysis": {},
"recommendations": []
}

        # Strategy analysis
strategy_performance = {}
        for strategy_type, market_results in result.strategy_performance.items():
            avg_success_rate = unified_math.mean([r["success_rate"] for r in market_results.values()])
            avg_profit = unified_math.mean([r["total_profit"] for r in market_results.values()])
            strategy_performance[strategy_type] = {
"avg_success_rate": avg_success_rate,
"avg_profit": avg_profit,
"market_performance": market_results
}

analysis["strategy_analysis"] = strategy_performance

        # Matrix analysis
analysis["matrix_analysis"] = result.matrix_performance

        # Market condition analysis
analysis["market_condition_analysis"] = result.market_condition_performance

        # Reinforcement learning analysis
        if result.reinforcement_learning_updates:
analysis["reinforcement_learning_analysis"] = {
"matrix_weight_changes": result.reinforcement_learning_updates.get("matrix_weights", {}),
                "bad_vectors_count": result.reinforcement_learning_updates.get("known_bad_vectors", 0),
                "vector_validator_summary": result.reinforcement_learning_updates.get("vector_validator_performance", {})
            }

        # Generate recommendations
recommendations = []

        # Best strategy recommendation
best_strategy = unified_math.max(strategy_performance.items(), key=lambda x: x[1]["avg_success_rate"])
        recommendations.append(f"Best performing strategy: {best_strategy[0]} (Success rate: {best_strategy[1]['avg_success_rate']:.2%})")

        # Best matrix recommendation
best_matrix = unified_math.max(result.matrix_performance.items(), key=lambda x: x[1]["success_rate"])
        recommendations.append(f"Best performing matrix: {best_matrix[0]} (Success rate: {best_matrix[1]['success_rate']:.2%})")

        # Best market condition recommendation
best_market = unified_math.max(result.market_condition_performance.items(), key=lambda x: x[1]["success_rate"])
        recommendations.append(f"Best market condition: {best_market[0]} (Success rate: {best_market[1]['success_rate']:.2%})")

        # Risk management recommendations
        if result.max_drawdown > 0.1:
recommendations.append("High drawdown detected - consider implementing stricter risk management")

        if result.sharpe_ratio < 1.0:
recommendations.append("Low Sharpe ratio - consider optimizing risk-adjusted returns")

analysis["recommendations"] = recommendations

        return analysis

    def generate_backtest_report(self, result: BacktestResult,
                               filepath: str = None) -> str:
"""Generate a comprehensive backtest report"""
        if filepath is None:
filepath = f"tests/demo_reports/backtest_report_{result.backtest_id}.md"

Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Generate comprehensive analysis
analysis = self._generate_comprehensive_analysis(result)

        # Create markdown report
report = """# Schwabot Backtest Report

## Backtest Summary
- **Backtest ID**: {result.backtest_id}
- **Timestamp**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Trades**: {result.total_trades:,}
- **Success Rate**: {result.success_rate:.2%}
- **Total Profit**: ${result.total_profit:,.2f}
- **Average Profit**: ${result.average_profit:.2f}
- **Max Drawdown**: {result.max_drawdown:.2%}
- **Sharpe Ratio**: {result.sharpe_ratio:.3f}
- **Execution Time**: {result.execution_time:.2f}s

## Strategy Performance

"""

        # Add strategy performance
        for strategy_type, perf in analysis["strategy_analysis"].items():
            report += f"### {strategy_type.replace('_', ' ').title()}\n"
            report += f"- Average Success Rate: {perf['avg_success_rate']:.2%}\n"
report += f"- Average Profit: ${perf['avg_profit']:.2f}\n\n"

        # Add matrix performance
report += "## Matrix Performance\n\n"
        for matrix_id, perf in result.matrix_performance.items():
            report += f"### {matrix_id}\n"
report += f"- Success Rate: {perf['success_rate']:.2%}\n"
report += f"- Total Trades: {perf['trades']}\n"
report += f"- Total Profit: ${perf['profit']:.2f}\n\n"

        # Add market condition performance
report += "## Market Condition Performance\n\n"
        for market_condition, perf in result.market_condition_performance.items():
            report += f"### {market_condition.replace('_', ' ').title()}\n"
            report += f"- Success Rate: {perf['success_rate']:.2%}\n"
report += f"- Total Trades: {perf['trades']}\n"
report += f"- Total Profit: ${perf['profit']:.2f}\n\n"

        # Add recommendations
report += "## Recommendations\n\n"
        for recommendation in analysis["recommendations"]:
report += f"- {recommendation}\n"

        # Add reinforcement learning analysis
        if result.reinforcement_learning_updates:
report += "\n## Reinforcement Learning Analysis\n\n"
report += f"- Known Bad Vectors: {result.reinforcement_learning_updates.get('known_bad_vectors', 0)}\n"
            report += f"- Matrix Weights Updated: {len(result.reinforcement_learning_updates.get('matrix_weights', {}))}\n"

        # Save report
        with open(filepath, 'w') as f:
            f.write(report)

safe_print(f"📊 Backtest report saved to {filepath}")

        return filepath

    def _save_backtest_results(self):
        """Save backtest results to file"""
        try:
results_file = Path("tests/demo_results/backtest_results.json")

data = {
"backtest_results": [asdict(r) for r in self.backtest_results],
                "performance_metrics": self.performance_metrics,
"timestamp": datetime.now().isoformat()
            }

            with open(results_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

safe_print("💾 Backtest results saved successfully")

        except Exception as e:
safe_print(f"Error saving backtest results: {e}")

    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get comprehensive backtest summary"""
        return {
"performance_metrics": self.performance_metrics,
"total_backtest_configs": len(self.backtest_configs),
            "total_backtest_results": len(self.backtest_results),
            "recent_backtests": [
{
"backtest_id": r.backtest_id,
"success_rate": r.success_rate,
"total_profit": r.total_profit,
"timestamp": r.timestamp.isoformat()
                }
                for r in sorted(self.backtest_results, key=lambda x: x.timestamp, reverse=True)[:5]
            ],
"best_performing_strategies": self._get_best_performing_strategies(),
            "best_performing_matrices": self._get_best_performing_matrices()
        }

    def _get_best_performing_strategies(self) -> Dict[str, float]:
        """Get best performing strategies across all backtests"""
strategy_performance = {}

        for result in self.backtest_results:
            for strategy_type, perf in result.strategy_performance.items():
                if strategy_type not in strategy_performance:
strategy_performance[strategy_type] = []

strategy_performance[strategy_type].append(perf.get("success_rate", 0.0))

        # Calculate average performance for each strategy
avg_performance = {}
        for strategy_type, rates in strategy_performance.items():
            avg_performance[strategy_type] = unified_math.unified_math.mean(rates)

        # Return top 3 strategies
sorted_strategies = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_strategies[:3])

    def _get_best_performing_matrices(self) -> Dict[str, float]:
        """Get best performing matrices across all backtests"""
matrix_performance = {}

        for result in self.backtest_results:
            for matrix_id, perf in result.matrix_performance.items():
                if matrix_id not in matrix_performance:
matrix_performance[matrix_id] = []

matrix_performance[matrix_id].append(perf.get("success_rate", 0.0))

        # Calculate average performance for each matrix
avg_performance = {}
        for matrix_id, rates in matrix_performance.items():
            avg_performance[matrix_id] = unified_math.unified_math.mean(rates)

        # Return top 3 matrices
sorted_matrices = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_matrices[:3])


# Global demo backtest runner instance
demo_backtest_runner = DemoBacktestRunner()


def get_demo_backtest_runner() -> DemoBacktestRunner:
    """Get the global demo backtest runner instance"""
    return demo_backtest_runner


if __name__ == "__main__":
    # Test the demo backtest runner
runner = DemoBacktestRunner()

safe_print("=== Schwabot Demo Backtest Runner Test ===")

    # Create backtest config
config = runner.create_backtest_config(
        strategy_types=["ghost_signal", "volume_spike"],
market_conditions=["bull_market", "sideways"],
num_trades_per_strategy=20


    # Run backtest
result = runner.run_backtest(config)

safe_print(f"Backtest ID: {result.backtest_id}")
    safe_print(f"Success Rate: {result.success_rate:.2%}")
    safe_print(f"Total Profit: {result.total_profit:.2f}")
    safe_print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")

    # Generate report
report_path = runner.generate_backtest_report(result)
    safe_print(f"Report generated: {report_path}")

    # Get summary
summary = runner.get_backtest_summary()
    safe_print(f"Best Strategies: {summary['best_performing_strategies']}")
    safe_print(f"Best Matrices: {summary['best_performing_matrices']}")

safe_print("Demo backtest runner test completed!")
