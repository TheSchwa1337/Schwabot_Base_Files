# -*- coding: utf-8 -*-
""""""
Simplified BTC Integration Module.

This module provides a simplified interface for BTC trading operations
with integrated CLI compatibility and performance tracking.
""""""

import threading
import time
from typing import Any, Dict, Optional

from .exchange_apis.coinbase_api import CoinbaseAPI, ExchangeConfig, ExchangeType
from .utils.cli_handler import CLIHandler
from .utils.logger import logger, safe_log


class PerformanceMetrics:
    """Performance tracking metrics for the integration."""

    def __init__(self):
        """Initialize performance metrics."""
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.average_execution_time = 0.0
        self.total_execution_time = 0.0
        self.average_slippage = 0.0
        self.total_volume = 0.0
        self.api_calls = 0
        self.api_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0


class SimplifiedBTCIntegration:
    """Simplified BTC integration with CLI compatibility."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize simplified BTC integration."""

        Args:
            config: Integration configuration.
        """"""
        self.version = "1.0.0"
        self.config = config or self._default_config()

        # Initialize CLI compatibility handler
        self.cli_handler = CLIHandler()

        # Exchange APIs
        self.exchanges: Dict[str, Any] = {}
        self.active_exchange: Optional[Any] = None

        # Performance tracking
        self.performance_metrics = PerformanceMetrics()

        # Threading and synchronization
        self.integration_lock = threading.Lock()
        self.order_lock = threading.Lock()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

        # Initialize exchanges
        self._initialize_exchanges()

        # Start monitoring if enabled
        if self.config.get("enable_monitoring", True):
            self._start_monitoring()

        # Log initialization
        init_message = f"SimplifiedBTCIntegration v{self.version} initialized"
        safe_log(logger, "info", init_message)

    def _default_config(self) -> Dict[str, Any]:
        """Get default integration configuration."""

        Returns:
            Default configuration dictionary.
        """"""
        return {}
            "enable_monitoring": True,
                "enable_cache": True,
                    "cache_timeout": 5.0,  # 5 seconds
            "max_retries": 3,
                "retry_delay": 1.0,
                    "enable_cli_compatibility": True,
                    "force_ascii_output": False,
                    "enable_performance_tracking": True,
                    "default_exchange": "coinbase",
                    "sandbox_mode": True,
                    "rate_limit": 100,  # requests per minute
            "timeout": 30,
}
    def safe_print(self, message: str, force_ascii: Optional[bool] = None) -> None:
        """Safe print function with CLI compatibility."""

        Args:
            message: Message to print.
            force_ascii: Whether to force ASCII conversion.
        """"""
        if force_ascii is None:
            force_ascii = self.config.get("force_ascii_output", False)

        self.cli_handler.safe_print(message, force_ascii)

    def safe_log(self, level: str, message: str, context: str = "") -> bool:
        """Safe logging with CLI compatibility."""

        Args:
            level: Log level.
            message: Log message.
            context: Additional context.

        Returns:
            True if logging was successful.
        """"""
        return safe_log(logger, level, message, context)

    def _initialize_exchanges(self) -> None:
        """Initialize exchange connections."""
        try:
            # Add default exchanges based on configuration
            default_exchange = self.config.get("default_exchange", "coinbase")

            if default_exchange == "coinbase":
                # Create Coinbase configuration
                coinbase_config = ExchangeConfig()
                    exchange_type=ExchangeType.COINBASE,
                        api_key=self.config.get("coinbase_api_key", ""),
                            api_secret=self.config.get("coinbase_api_secret", ""),
                            sandbox=self.config.get("sandbox_mode", True),
                            )

                self.add_exchange(ExchangeType.COINBASE, coinbase_config)
                self.set_active_exchange("coinbase")

            self.safe_log("info", f"Initialized {len(self.exchanges)} exchanges")

        except Exception as e:
            error_msg = f"Error initializing exchanges: {e}"
            self.safe_log("error", error_msg)

    def add_exchange(self, exchange_type: ExchangeType, config: ExchangeConfig) -> bool:
        """Add exchange to the integration."""

        Args:
            exchange_type: Type of exchange to add.
            config: Exchange configuration.

        Returns:
            True if exchange was added successfully.
        """"""
        try:
            with self.integration_lock:
                if exchange_type == ExchangeType.COINBASE:
                    exchange = CoinbaseAPI(config)
                    self.exchanges["coinbase"] = exchange
                    self.safe_log("info", "Added Coinbase exchange")
                    return True
                else:
                    self.safe_log()
                        "warning",
                            f"Unsupported exchange type: {exchange_type}",
                                )
                    return False

        except Exception as e:
            error_msg = f"Error adding exchange {exchange_type}: {e}"
            self.safe_log("error", error_msg)
            return False

    def set_active_exchange(self, exchange_name: str) -> bool:
        """Set the active exchange for operations."""

        Args:
            exchange_name: Name of exchange to set as active.

        Returns:
            True if exchange was set successfully.
        """"""
        try:
            with self.integration_lock:
                if exchange_name in self.exchanges:
                    self.active_exchange = self.exchanges[exchange_name]
                    self.safe_log("info", f"Set active exchange: {exchange_name}")
                    return True
                else:
                    self.safe_log("error", f"Exchange not found: {exchange_name}")
                    return False

        except Exception as e:
            error_msg = f"Error setting active exchange: {e}"
            self.safe_log("error", error_msg)
            return False

    def _start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.safe_log("info", "Started monitoring thread")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Monitor exchange health
                if self.active_exchange:
                    health_status = self.active_exchange.get_health_status()
                    if not health_status.get("healthy", True):
                        self.safe_log("warning", "Exchange health check failed")

                # Log performance metrics periodically
                if self.config.get("enable_performance_tracking", True):
                    self._log_performance_metrics()

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                error_msg = f"Error in monitoring loop: {e}"
                self.safe_log("error", error_msg)
                time.sleep(60)  # Wait longer on error

    def _log_performance_metrics(self) -> None:
        """Log current performance metrics."""
        metrics = self.performance_metrics
        if metrics.total_orders > 0:
            success_rate = (metrics.successful_orders / metrics.total_orders) * 100
            self.safe_log()
                "info", f"Performance: {success_rate:.1f}% success rate, " f"{metrics.total_orders} total orders"
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""

        Returns:
            Dictionary containing performance metrics.
        """"""
        metrics = self.performance_metrics
        return {}
            "total_orders": metrics.total_orders,
                "successful_orders": metrics.successful_orders,
                    "failed_orders": metrics.failed_orders,
                    "success_rate": ()
                (metrics.successful_orders / metrics.total_orders * 100) if metrics.total_orders > 0 else 0.0
            ),
                "average_execution_time": metrics.average_execution_time,
                    "total_volume": metrics.total_volume,
                    "api_calls": metrics.api_calls,
                    "api_errors": metrics.api_errors,
                    "cache_hits": metrics.cache_hits,
                    "cache_misses": metrics.cache_misses,
}
    def shutdown(self) -> None:
        """Shutdown the integration gracefully."""
        self.safe_log("info", "Shutting down SimplifiedBTCIntegration")
        self.monitoring_active = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        self.safe_log("info", "SimplifiedBTCIntegration shutdown complete")


# Global instance for easy access
simplified_btc_integration = SimplifiedBTCIntegration()


def get_integration() -> SimplifiedBTCIntegration:
    """Get the global integration instance."""

    Returns:
        The global SimplifiedBTCIntegration instance.
    """"""
    return simplified_btc_integration


if __name__ == "__main__":
    # Demo the integration
    integration = SimplifiedBTCIntegration()

    print("Simplified BTC Integration Demo")
    print("=" * 40)

    # Show configuration
    print(f"Version: {integration.version}")
    print(f"Exchanges: {list(integration.exchanges.keys())}")
    print(f"Active Exchange: {integration.active_exchange}")

    # Show performance metrics
    metrics = integration.get_performance_metrics()
    print(f"Performance Metrics: {metrics}")

    # Cleanup
    integration.shutdown()
