"""
schwa_engine.py
---------------
Core runtime orchestrator for Schwabot.

High-level flow:
1. Pull latest market tick (placeholder for now)
2. Collect AI proposals & votes (placeholder demo values)
3. Run VoteRegistry to evaluate consensus
4. Validate hash similarity and band gatekeeper alignment
5. Execute trade if *all* gates pass (placeholder execution)
"""
from __future__ import annotations

from typing import Dict, Tuple

from .registry_vote_matrix import VoteRegistry
from .agent_memory import AgentMemory
from .hash_drift_sync import market_hash, hash_similarity
from .data_feed import fetch_latest_tick
from .strategy_layered_gatekeeper import StrategyLayeredGatekeeper
from .trade_executor import TradeExecutor
from .risk_manager import RiskManager
from core.secure_exchange_manager import get_exchange_manager, ExchangeType
import time


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def launch_schwabot(api_keys: Dict[str, str], mode: str = "test") -> None:
    print(f"[Schwabot] Launching in {mode.upper()} mode…")
    # Multi-asset symbol rotation
    symbols = api_keys.get("symbol_list", ["BTC/USDC", "ETH/USDC", "XRP/USDC", "SOL/USDC", "USDT/USDC"])
    interval = int(api_keys.get("rotation_interval", 225))  # seconds; default 3.75min
    router = SymbolRouter(symbols, interval)

    # ------------------------------------------------------------------
    # 1. Fetch live market tick data (via CCXT)
    # ------------------------------------------------------------------
    # Rotate symbol dynamically
    symbol = router.get_symbol()
    try:
        tick_blob = fetch_latest_tick(symbol=symbol, exchange_id=api_keys.get("exchange", "binance"))
    except Exception as exc:
        print(f"[Schwabot] ⚠️  Failed to fetch live tick: {exc}. Falling back to simulation.")
        tick_blob = f"{symbol},price=63000,time=16912000"  # fallback placeholder
    current_hash = market_hash(tick_blob)

    # ------------------------------------------------------------------
    # 2. Collect AI agent votes (True = approve trade proposal)
    # ------------------------------------------------------------------
    votes = {
        "r1": True,
        "gpt4o": True,
        "claude": False,
    }

    # ------------------------------------------------------------------
    # Load persistent agent scores and build VoteRegistry
    # ------------------------------------------------------------------
    memory = AgentMemory()
    registry = VoteRegistry(memory.get_performance_db())
    consensus_ok = registry.evaluate(votes)

    # ------------------------------------------------------------------
    # 3. Hash match & vector band alignment checks
    # ------------------------------------------------------------------
    pattern_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"  # placeholder
    hash_ok = hash_similarity(current_hash, pattern_hash)

    # ------------------------------------------------------------------
    # 3. Evaluate layered gates (TA + profit bucket overrides)
    # ------------------------------------------------------------------
    gatekeeper = StrategyLayeredGatekeeper()
    vector_ok, gate_reason, gate_confidence = gatekeeper.evaluate_all_gates(tick_blob)
    
    # Get exit strategy if available
    exit_strategy = gatekeeper.get_exit_strategy(tick_blob)

    # ------------------------------------------------------------------
    # 4. Final execution gate -> integrate RiskManager for exit logic
    # ------------------------------------------------------------------
    if all([consensus_ok, hash_ok, vector_ok]):
        print(f"[Schwabot] Gate reason: {gate_reason} (confidence: {gate_confidence:.2f})")
        # Setup risk manager with optional per-symbol profiles
        risk_profiles = api_keys.get('risk_profiles')
        profile_path = api_keys.get('risk_profile_path')
        risk = RiskManager(profiles=risk_profiles, profile_path=profile_path)
        # Determine symbol from tick_blob and register trade
        symbol = tick_blob.split(',',1)[0]
        # Prepare executors: dry vs secure live
        manager = get_exchange_manager() if mode == "live" else None
        executor = TradeExecutor(api_keys, mode="dry") if mode != "live" else None
        # Entry price from live fetch for both modes
        if mode == "live":
            entry_resp = manager.execute_trade(
                ExchangeType(api_keys.get('exchange', 'binance')), symbol, 'buy', api_keys.get('amount', 0.001)
            )
            entry_price = entry_resp.get('price') if entry_resp.get('price') else manager.get_balance(
                ExchangeType(api_keys.get('exchange', 'binance')), symbol.split('/')[0]
            ).get('currency')
        else:
            # dry-run
            entry_price = executor._fetch_price()
            executor.symbol = symbol
            executor.execute(side="buy")
        # Register trade in risk manager
        trade_id = f"{symbol}_{int(time.time())}"
        risk.register_trade(trade_id, entry_price, timestamp=time.time(), symbol=symbol)
        # Monitor for exit triggers
        opp_side = "sell"
        while True:
            time.sleep(1)
            current_price = manager.fetch_ticker(symbol)['last'] if mode == "live" else executor._fetch_price()
            decision = risk.update_price(trade_id, current_price)
            if decision in ['STOP', 'LOCK', 'TTL_EXIT']:
                print(f"[Schwabot] RiskManager decision: {decision}")
                if mode == "live":
                    manager.execute_trade(
                        ExchangeType(api_keys.get('exchange', 'binance')),
                        symbol, opp_side, api_keys.get('amount', 0.001)
                    )
                else:
                    executor.execute(side=opp_side)
                risk.cancel_trade(trade_id)
                break
        # Reward agents after exit
        for agent_id, approve in votes.items():
            reward = 0.05 if approve else -0.05
            memory.update_score(agent_id, reward)
    else:
        print(f"[Schwabot] Gate failed: {gate_reason}")
        # Penalise those who wanted to trade when gates failed.
        for agent_id, approve in votes.items():
            reward = -0.02 if approve else 0.02
            memory.update_score(agent_id, reward)


# -----------------------------------------------------------------------------
# Internal Helpers
# -----------------------------------------------------------------------------

def _execute_trade(api_keys: Dict[str, str], tick_blob: str, mode: str, exit_strategy: Tuple[float, int] | None = None) -> None:
    """Placeholder trade execution hook.

    Replace this with CCXT order placement once strategy logic is proven.
    """
    print("[Schwabot] ✅ All gates passed — EXECUTING TRADE!")
    print(f"[Schwabot] Tick context: {tick_blob}")

    executor = TradeExecutor(api_keys, mode="live" if mode == "live" else "dry")
    # Override executor symbol to current asset
    executor.symbol = symbol
    executor.exit_strategy = exit_strategy
    executor.execute(side="buy") 