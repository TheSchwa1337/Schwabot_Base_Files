import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import pandas as pd

#!/usr/bin/env python3


"""



Historical Data Downloader



==========================







Downloads BTC, ETH, and XRP historical data using your existing CoinGecko API.



Saves data in the format expected by your trading system.



"""


logger = logging.getLogger(__name__)


# CoinGecko API configuration


BASE_URL = "https://api.coingecko.com/api/v3"


# Coin IDs for CoinGecko API


COIN_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "XRP": "ripple"}


async def download_historical_data(): -> Optional[pd.DataFrame]:
    """



    Download historical data from CoinGecko API.







    Args:



        coin_id: Coin ID (e.g., 'bitcoin', 'ethereum', 'ripple')



        vs_currency: Quote currency (default: 'usd')



        days: Number of days to fetch (max 365 for free, tier)



        interval: Data interval ('daily' or 'hourly')







    Returns:



        DataFrame with historical data or None if failed



    """

    try:

        url = f"{BASE_URL}/coins/{coin_id}/market_chart"

        params = {}
            "vs_currency": vs_currency,
            "days": days,
            "interval": interval}

        async with aiohttp.ClientSession() as session:

            async with session.get(url, params=params) as resp:

                resp.raise_for_status()

                data = await resp.json()

        # Convert to DataFrame

        df = parse_historical_data(data, coin_id, vs_currency)

        return df

    except Exception as e:

        logger.error(f"Failed to download historical data for {coin_id}: {e}")

        return None


def parse_historical_data(): -> pd.DataFrame:
    """Parse CoinGecko historical data into DataFrame."""

    # Extract price data

    prices = data.get("prices", [])

    market_caps = data.get("market_caps", [])

    volumes = data.get("total_volumes", [])

    # Create DataFrame

    df_data = []

    for i, (timestamp, price) in enumerate(prices):

        row = {}
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp / 1000),
            "price": price,
            "coin_id": coin_id,
            "vs_currency": vs_currency,
        }

        # Add market cap if available

        if i < len(market_caps):

            row["market_cap"] = market_caps[i][1]

        # Add volume if available

        if i < len(volumes):

            row["volume"] = volumes[i][1]

        df_data.append(row)

    df = pd.DataFrame(df_data)

    df.set_index("datetime", inplace=True)

    return df


async def download_all_historical_data(): -> None:
    """



    Download historical data for BTC, ETH, and XRP.







    Args:



        days: Number of days to fetch (max 365 for free, tier)



        interval: Data interval ('daily' or 'hourly')



        output_dir: Output directory for CSV files



    """

    logger.info(" Starting historical data download...")

    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    for symbol, coin_id in COIN_IDS.items():

        logger.info(f" Downloading {symbol} ({coin_id}) historical data...")

        df = await download_historical_data(coin_id=coin_id, days=days, interval=interval)

        if df is not None:

            # Create coin-specific directory

            coin_dir = output_path / f"{symbol.lower()}_usd"

            coin_dir.mkdir(exist_ok=True)

            # Save as CSV

            csv_path = coin_dir / f"{symbol.lower()}_historical.csv"

            df.to_csv(csv_path)

            logger.info()
                f" Saved {symbol} data: {"}
                    len(df)} points to {csv_path}")"

            logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")

        else:

            logger.error(f" Failed to download {symbol} data")

        # Rate limiting - CoinGecko free tier allows 10-50 calls per minute

        await asyncio.sleep(1)

    logger.info(" Historical data download complete!")


if __name__ == "__main__":

    # Example usage

    logging.basicConfig(level=logging.INFO)

    async def main():

        # Download 1 year of daily data

        await download_all_historical_data(days=365, interval="daily")

        # Download 30 days of hourly data (for more granular, testing)

        # await download_all_historical_data(days=30, interval="hourly")

    asyncio.run(main())
