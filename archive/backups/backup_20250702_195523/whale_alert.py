# !/usr/bin/env python3
# -*- coding: utf-8 -*-
WhaleAlert API Handler =====================

Fetches whale transaction data from WhaleAlert API.
Tracks large crypto transactions and provides insights into whale movements.

import asyncio
import logging
from typing import Any, Dict, List

import aiohttp
import requests

from .base_handler import BaseAPIHandler

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)

# WhaleAlert API configuration
BASE_URL = https://api.whale-alert.io/v1


class WhaleAlertHandler(BaseAPIHandler):
    WhaleAlert API handler for tracking large cryptocurrency transactions.NAME =  whale_alertCACHE_SUBDIR =  whale_dataREFRESH_INTERVAL = 180  # 3-minute updates for whale tracking

    def __init__(self, api_key: str = None, cache_root: str = flask/feeds) -> None:Initialize WhaleAlert handler with API key and cache configuration.super().__init__(cache_root)
        self.api_key = api_key or demo-key# Use demo key if none provided
        self.min_value = 500000  # Minimum transaction value USD

    async def _fetch_raw(self) -> Any:
        Fetch raw whale transaction data from WhaleAlert API.params = {api_key: self.api_key,min_value: self.min_value,limit: 100,
        }

        if aiohttp: session = await self._get_session()
            async with session.get(f{BASE_URL}/transactions, params = params) as resp:
                if resp.status == 401:
                    logger.warning(WhaleAlert API key invalid, returning empty data)
                    return {result:success,transactions: []}
                resp.raise_for_status()
                return await resp.json()
        elif requests: loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(f{BASE_URL}/transactions, params = params, timeout=15)
            )
            if response.status_code == 401:
                logger.warning(WhaleAlert API key invalid, returning empty data)
                return {result:success,transactions: []}
            response.raise_for_status()
            return response.json()
        else:
            raise RuntimeError(Neither aiohttp nor requests is available for HTTP calls)

    async def _parse_raw(self, raw: Any) -> Dict[str, Any]:Parse whale transaction data into normalized format.try:
            if raw.get(result) !=success":
                logger.warning(WhaleAlert API returned non-success result)
                return {transactions: [],summary: {}}

            transactions = raw.get(transactions, [])

            # Process transactions
            processed_transactions = []
            btc_volume = 0.0
            eth_volume = 0.0
            total_volume_usd = 0.0

            for tx in transactions: processed_tx = {id: tx.get(id),blockchain: tx.get(blockchain),symbol: tx.get(symbol),amount": float(tx.get(amount", 0)),amount_usd": float(tx.get(amount_usd", 0)),from_address": tx.get(from, {}).get(address,unknown),to_address": tx.get(to", {}).get(address,unknown),from_owner": tx.get(from, {}).get(owner,unknown),to_owner": tx.get(to", {}).get(owner,unknown),timestamp": tx.get(timestamp", 0),transaction_type: tx.get(transaction_type,transfer),hash: tx.get(hash,),
                }
                processed_transactions.append(processed_tx)

                # Accumulate volumes
                symbol = tx.get(symbol,).upper()
                amount_usd = float(tx.get(amount_usd, 0))
                total_volume_usd += amount_usd

                if symbol == BTC:
                    btc_volume += amount_usd
                elif symbol == ETH:
                    eth_volume += amount_usd

            # Create summary statistics
            summary = {total_transactions: len(processed_transactions),
                total_volume_usd: total_volume_usd,btc_volume_usd: btc_volume,eth_volume_usd: eth_volume,average_transaction_size": total_volume_usd / max(len(processed_transactions), 1),
                # Scale to 0-100
                whale_activity_score: min(100, (total_volume_usd / 1000000) * 10),dominant_blockchain": self._get_dominant_blockchain(processed_transactions),
            }

            return {transactions: processed_transactions,summary: summary,last_updated: raw.get(cursor, {}).get(last, 0),
            }

        except Exception as exc:
            logger.error(%s: failed to parse whale data – %s", self.NAME, exc)
            return {transactions: [],summary: {}}

    def _get_dominant_blockchain(self, transactions: List[Dict]) -> str:Determine the most active blockchain by transaction count.blockchain_counts = {}
        for tx in transactions: blockchain = tx.get(blockchain, unknown)
            blockchain_counts[blockchain] = blockchain_counts.get(blockchain, 0) + 1

        if not blockchain_counts:
            returnunknownreturn max(blockchain_counts, key=blockchain_counts.get)
