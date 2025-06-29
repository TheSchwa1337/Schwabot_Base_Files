import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

# Assuming FaultBus is available for error reporting
from core.fault_bus import FaultBus, FaultType

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - %(levelname)s - %(message)s")


class MetricTensorRegistry:
    """
    Manages the registration and retrieval of various performance metrics as tensors.
    Ensures data consistency and provides a unified interface for metric access.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, np.ndarray] = {}
        self.fault_bus = FaultBus()

    def register_metric(self, name: str, metric_data: Union[List[float], np.ndarray]) -> None:
        """
        Registers a new metric or updates an existing one.
        Converts list data to numpy arrays for consistency.
        """
        if isinstance(metric_data, list):
            metric_array = np.array(metric_data, dtype=np.float64)
        elif isinstance(metric_data, np.ndarray):
            metric_array = metric_data.astype(np.float64)
        else:
            self.fault_bus.push(
                FaultType.DATA_INTEGRITY_VIOLATION,
                "Invalid metric data type provided.",
                metric_name=name,
                provided_type=type(metric_data).__name__,
            )
            logging.error(f"Invalid metric data type for {name}: {type(metric_data).__name__}")
            return

        self._metrics[name] = metric_array
        logging.info(f"Metric '{name}' registered/updated with shape {metric_array.shape}.")

    def get_metric(self, name: str) -> Optional[np.ndarray]:
        """
        Retrieves a registered metric by name.
        """
        metric = self._metrics.get(name)
        if metric is None:
            self.fault_bus.push(
                FaultType.DATA_INTEGRITY_VIOLATION, "Attempted to retrieve non-existent metric.", metric_name=name
            )
            logging.warning(f"Attempted to retrieve non-existent metric: {name}")
        return metric

    def list_metrics(self) -> List[str]:
        """
        Lists all registered metric names.
        """
        return list(self._metrics.keys())

    def clear_metrics(self) -> None:
        """
        Clears all registered metrics.
        """
        self._metrics.clear()
        logging.info("All metrics cleared from registry.")


class AdaptiveThreshold:
    """
    Dynamically adjusts performance thresholds based on historical data and volatility.
    Uses exponential moving average for smoothing and adaptability.

    Threshold = Base_Threshold * (1 + Sensitivity * Volatility_Factor)
    Volatility_Factor = EMA(abs(metric - EMA(metric)))
    """

    def __init__(self, base_threshold: float = 0.05, sensitivity: float = 1.5, ema_alpha: float = 0.1) -> None:
        self.base_threshold = base_threshold
        self.sensitivity = sensitivity
        self.ema_alpha = ema_alpha
        self._ema_metric: Optional[float] = None
        self._ema_volatility: Optional[float] = None
        self.fault_bus = FaultBus()

    def update(self, current_metric_value: float) -> float:
        """
        Updates the EMA for the metric and volatility, then calculates the adaptive threshold.
        """
        if self._ema_metric is None:
            self._ema_metric = current_metric_value
            self._ema_volatility = 0.0  # Initial volatility is zero
        else:
            self._ema_metric = self.ema_alpha * current_metric_value + (1 - self.ema_alpha) * self._ema_metric
            absolute_deviation = abs(current_metric_value - self._ema_metric)
            if self._ema_volatility is None:
                self._ema_volatility = absolute_deviation
            else:
                self._ema_volatility = self.ema_alpha * absolute_deviation + (1 - self.ema_alpha) * self._ema_volatility

        if self._ema_volatility is None:
            # This case should ideally not be hit after the first update, but for type safety
            self.fault_bus.push(FaultType.SYSTEM_ERROR, "EMA Volatility is None after update.")
            logging.error("EMA Volatility is None after update. This indicates a logic error.")
            volatility_factor = 0.0
        else:
            volatility_factor = self._ema_volatility

        adaptive_threshold = self.base_threshold * (1 + self.sensitivity * volatility_factor)
        logging.debug(
            f"Adaptive Threshold: {
                adaptive_threshold:.6f} (Metric EMA: {
                self._ema_metric:.6f}, Volatility EMA: {
                volatility_factor:.6f})"
        )
        return adaptive_threshold

    def get_current_threshold(self) -> float:
        """
        Returns the last calculated adaptive threshold. If no update has occurred,
        returns the base threshold.
        """
        if self._ema_metric is None or self._ema_volatility is None:
            return self.base_threshold
        return self.base_threshold * (1 + self.sensitivity * self._ema_volatility)


class FeedbackLoop:
    """
    Manages the feedback mechanism from performance metrics to strategy adjustment.
    Incorporates a proportional-integral-derivative (PID)-like control for smooth adaptation.

    Adjustment = Kp * Error + Ki * Integral(Error) + Kd * Derivative(Error)
    Error = Desired_Performance - Actual_Performance
    """

    def __init__(
        self, kp: float = 0.1, ki: float = 0.01, kd: float = 0.05, integral_limit: float = 10.0, dt: float = 1.0
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.dt = dt  # Time step between updates

        self._previous_error: Optional[float] = None
        self._integral_error: float = 0.0
        self.fault_bus = FaultBus()

    def calculate_adjustment(self, actual_performance: float, desired_performance: float) -> float:
        """
        Calculates the strategic adjustment based on the deviation from desired performance.
        """
        error = desired_performance - actual_performance

        # Proportional term
        proportional = self.kp * error

        # Integral term with anti-windup
        self._integral_error += error * self.dt
        self._integral_error = np.clip(self._integral_error, -self.integral_limit, self.integral_limit)
        integral = self.ki * self._integral_error

        # Derivative term
        derivative = 0.0
        if self._previous_error is not None:
            derivative = self.kd * (error - self._previous_error) / self.dt
        self._previous_error = error

        adjustment = proportional + integral + derivative
        logging.debug(
            f"Feedback Adjustment: {
                adjustment:.6f} (P: {
                proportional:.6f}, I: {
                integral:.6f}, D: {
                    derivative:.6f})"
        )
        return adjustment

    def reset(self) -> None:
        """
        Resets the internal state of the feedback loop (integral and previous error).
        """
        self._previous_error = None
        self._integral_error = 0.0
        logging.info("FeedbackLoop state reset.")


class FitnessOracle:
    """
    The central oracle for assessing Schwabot's overall fitness and guiding adaptive behaviors.
    Integrates metric monitoring, adaptive thresholds, and feedback loops.
    """

    def __init__(self) -> None:
        self.metric_registry = MetricTensorRegistry()
        self.adaptive_thresholds: Dict[str, AdaptiveThreshold] = {}
        self.feedback_loops: Dict[str, FeedbackLoop] = {}
        self.fault_bus = FaultBus()

    def add_metric_for_monitoring(
        self, metric_name: str, base_threshold: float = 0.05, sensitivity: float = 1.5, ema_alpha: float = 0.1
    ) -> None:
        """
        Adds a metric to be monitored with its own adaptive threshold.
        """
        if metric_name in self.adaptive_thresholds:
            logging.warning(f"Metric '{metric_name}' already being monitored. Overwriting adaptive threshold.")
        self.adaptive_thresholds[metric_name] = AdaptiveThreshold(base_threshold, sensitivity, ema_alpha)
        logging.info(f"Metric '{metric_name}' added for adaptive monitoring.")

    def add_feedback_loop(
        self,
        loop_name: str,
        kp: float = 0.1,
        ki: float = 0.01,
        kd: float = 0.05,
        integral_limit: float = 10.0,
        dt: float = 1.0,
    ) -> None:
        """
        Adds a feedback loop for a specific adjustment mechanism.
        """
        if loop_name in self.feedback_loops:
            logging.warning(f"Feedback loop '{loop_name}' already exists. Overwriting.")
        self.feedback_loops[loop_name] = FeedbackLoop(kp, ki, kd, integral_limit, dt)
        logging.info(f"Feedback loop '{loop_name}' added.")

    def update_and_evaluate_metric(self, metric_name: str, metric_value: float) -> bool:
        """
        Updates a specific metric's adaptive threshold and evaluates if it's within bounds.
        Returns True if within acceptable bounds, False otherwise.
        """
        threshold_obj = self.adaptive_thresholds.get(metric_name)
        if threshold_obj is None:
            self.fault_bus.push(
                FaultType.CONFIG_CHANGE, "Attempted to update unmonitored metric.", metric_name=metric_name
            )
            logging.warning(f"Metric '{metric_name}' is not set up for adaptive monitoring.")
            return False  # Cannot evaluate if not monitored

        current_threshold = threshold_obj.update(metric_value)
        is_within_bounds = abs(metric_value) <= current_threshold  # Assuming metric should be close to zero
        if not is_within_bounds:
            self.fault_bus.push(
                FaultType.METRIC_ANOMALY,
                f"Metric '{metric_name}' ({
                    metric_value:.6f}) exceeded adaptive threshold ({
                    current_threshold:.6f}).",
                metric=metric_name,
                value=metric_value,
                threshold=current_threshold,
            )
            logging.warning(
                f"Metric '{metric_name}' ({
                    metric_value:.6f}) exceeded adaptive threshold ({
                    current_threshold:.6f})."
            )
        else:
            logging.info(
                f"Metric '{metric_name}' ({
                    metric_value:.6f}) is within adaptive threshold ({
                    current_threshold:.6f})."
            )
        return is_within_bounds

    def get_strategic_adjustment(self, loop_name: str, actual_performance: float, desired_performance: float) -> float:
        """
        Calculates a strategic adjustment using a specified feedback loop.
        """
        feedback_loop = self.feedback_loops.get(loop_name)
        if feedback_loop is None:
            self.fault_bus.push(
                FaultType.CONFIG_CHANGE,
                "Attempted to get adjustment from non-existent feedback loop.",
                loop_name=loop_name,
            )
            logging.error(f"Feedback loop '{loop_name}' not found.")
            return 0.0
        return feedback_loop.calculate_adjustment(actual_performance, desired_performance)

    def overall_fitness_score(self) -> float:
        """
        Calculates an aggregated fitness score based on all monitored metrics.
        This is a placeholder and should be expanded with more sophisticated aggregation logic.
        For now, it returns an average of inverse normalized thresholds.
        """
        scores: List[float] = []
        for metric_name, threshold_obj in self.adaptive_thresholds.items():
            metric_value = self.metric_registry.get_metric(metric_name)
            if metric_value is not None and metric_value.size > 0:
                current_threshold = threshold_obj.get_current_threshold()
                # A simple scoring: 1 - (absolute_value / threshold) clipped at 0
                # More advanced scoring would involve fuzzy logic, weighted sums, etc.
                score = 1.0 - (np.mean(np.abs(metric_value)) / current_threshold)
                scores.append(max(0.0, score))  # Score cannot be negative
            else:
                scores.append(0.0)  # If metric not available, score it as 0 for this part

        if not scores:
            return 0.0  # No metrics, no fitness

        # Aggregate scores (e.g., simple average, weighted average, minimum score)
        aggregated_score = np.mean(scores)
        logging.info(f"Overall Fitness Score: {aggregated_score:.6f}")
        return aggregated_score

    def export_fitness_report(self, file_path: str = "fitness_report.json") -> None:
        """
        Exports a detailed fitness report including current metric states and thresholds.
        """
        import json

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_fitness_score": self.overall_fitness_score(),
            "metrics_summary": {
                name: {
                    "current_value": (
                        self.metric_registry.get_metric(name).tolist()
                        if self.metric_registry.get_metric(name) is not None
                        else None
                    ),
                    "adaptive_threshold": self.adaptive_thresholds[name].get_current_threshold(),
                }
                for name in self.adaptive_thresholds.keys()
            },
            "fault_bus_memory_log_sample": self.fault_bus.memory_log[-5:],  # Sample last 5 fault entries
        }
        try:
            with open(file_path, "w") as f:
                json.dump(report, f, indent=4)
            logging.info(f"Fitness report exported to '{file_path}'.")
        except IOError as e:
            self.fault_bus.push(
                FaultType.SYSTEM_ERROR, "Failed to export fitness report.", file=file_path, error=str(e)
            )
            logging.error(f"Failed to export fitness report to '{file_path}': {e}", exc_info=True)


# Example Usage (for testing and demonstration purposes, can be removed in final deployment)
if __name__ == "__main__":
    # This part needs an asyncio event loop if using async FaultBus dispatches
    async def main_fitness_oracle_test():
        oracle = FitnessOracle()

        # Add metrics for monitoring
        oracle.add_metric_for_monitoring("profit_loss_deviation", base_threshold=0.01, sensitivity=2.0)
        oracle.add_metric_for_monitoring("slippage_rate", base_threshold=0.005, sensitivity=3.0)
        oracle.add_feedback_loop("strategy_adjuster", kp=0.2, ki=0.02, kd=0.01)

        # Simulate data over time
        profit_deviations = [0.008, 0.012, 0.009, 0.015, 0.007, 0.020, 0.011, 0.006, 0.018, 0.010]
        slippage_rates = [0.003, 0.006, 0.004, 0.008, 0.003, 0.010, 0.005, 0.002, 0.007, 0.004]

        for i in range(len(profit_deviations)):
            print(f"\n--- Simulation Step {i + 1} ---")

            # Register/Update metrics
            oracle.metric_registry.register_metric("profit_loss_deviation", [profit_deviations[i]])
            oracle.metric_registry.register_metric("slippage_rate", [slippage_rates[i]])

            # Evaluate metrics
            is_profit_ok = oracle.update_and_evaluate_metric("profit_loss_deviation", profit_deviations[i])
            is_slippage_ok = oracle.update_and_evaluate_metric("slippage_rate", slippage_rates[i])

            print(f"Profit Deviation within bounds: {is_profit_ok}")
            print(f"Slippage Rate within bounds: {is_slippage_ok}")

            # Get adjustment based on profit deviation
            # Let's say desired profit deviation is 0.005
            adjustment = oracle.get_strategic_adjustment("strategy_adjuster", profit_deviations[i], 0.005)
            print(f"Calculated Strategy Adjustment: {adjustment:.6f}")

            oracle.overall_fitness_score()

        print("\n--- Final Report ---")
        oracle.export_fitness_report("simulation_fitness_report.json")

    # Run the async test main function
    asyncio.run(main_fitness_oracle_test())
