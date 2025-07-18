# -*- coding: utf-8 -*-
"""
Historical Data Manager for Backtesting.

This module is responsible for providing structured historical market data
for backtesting purposes. It can generate mock data or be extended to load
from external sources like CSV files or databases.
"""

import logging
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, Optional
import asyncio

logger = logging.getLogger(__name__)


class HistoricalDataManager:
    """Manages and provides historical market data."""

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        interval_minutes: int = 1440,  # Default to daily data
        initial_price: Decimal = Decimal("40000.0"),
        volatility_factor: float = 0.005,  # Daily price change up to +/- 0.5%
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.interval_minutes = interval_minutes
        self.initial_price = initial_price
        self.volatility_factor = volatility_factor
        logger.info(
            f"HistoricalDataManager initialized for {start_date.date()} to {end_date.date()} at {interval_minutes} min interval."
        )

    async def generate_mock_data(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generates mock OHLCV historical price data points.
        Each data point represents a specific time interval.
        In a real system, this would load from a data source.
        """
        current_date = self.start_date
        current_price = self.initial_price

        while current_date <= self.end_date:
            open_price = current_price

            # Simulate price fluctuations within the interval
            # For simplicity, we'll use daily OHLCV for now, where interval_minutes is large
            price_change_pct = Decimal(str(random.uniform(-self.volatility_factor, self.volatility_factor)))
            close_price = current_price * (Decimal("1") + price_change_pct)
            close_price = round(close_price, 8)  # Round to 8 decimal places for crypto prices

            # Simple high/low calculation
            high_price = max(open_price, close_price) * Decimal(
                str(1 + random.uniform(0, self.volatility_factor * 0.5))
            )
            low_price = min(open_price, close_price) * Decimal(str(1 - random.uniform(0, self.volatility_factor * 0.5)))
            high_price = round(high_price, 8)
            low_price = round(low_price, 8)

            # Simulate volume
            volume = random.uniform(1000.0, 5000.0)  # Example volume

            yield {
                "timestamp": int(current_date.timestamp() * 1000),  # Milliseconds timestamp
                "datetime": current_date.isoformat(),
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": volume,
            }
            current_price = close_price  # Next interval starts from this close price
            current_date += timedelta(minutes=self.interval_minutes)  # Move to next interval

    async def load_data_from_csv(self, file_path: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Placeholder for loading historical data from a CSV file.
        Expected CSV format: timestamp,open,high,low,close,volume
        """
        logger.warning(f"CSV data loading not yet implemented. Using mock data instead for {file_path}.")
        async for data_point in self.generate_mock_data():
            yield data_point

    async def get_historical_data(
        self, source: str = "mock", file_path: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Provides historical data based on the specified source.

        Args:
            source (str): 'mock' for generated data, 'csv' for CSV file.
            file_path (Optional[str]): Path to the CSV file if source is 'csv'.

        Returns:
            AsyncGenerator: An asynchronous generator yielding historical data points.
        """
        if source == "mock":
            async for data_point in self.generate_mock_data():
                yield data_point
        elif source == "csv":
            if not file_path:
                raise ValueError("file_path must be provided for 'csv' data source.")
            async for data_point in self.load_data_from_csv(file_path):
                yield data_point
        else:
            raise ValueError(f"Unsupported data source: {source}")


# Example Usage (for testing purposes, not part of the main backtester run flow)
async def _main_data_manager_test():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 5)
    data_manager = HistoricalDataManager(start_date=start, end_date=end, interval_minutes=60)  # Hourly data

    logger.info("Testing mock data generation (hourly)...")
    count = 0
    async for data_point in data_manager.get_historical_data(source="mock"):
        logger.info(f"Data Point: {data_point}")
        count += 1
    logger.info(f"Generated {count} data points.")


if __name__ == "__main__":
    asyncio.run(_main_data_manager_test())
