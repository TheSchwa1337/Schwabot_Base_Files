"""
Schwabot Engine
==============

Core runtime orchestrator for Schwabot trading system.

High-level flow:
1. Pull latest market tick (via CCXT)
2. Collect AI proposals & votes
3. Run VoteRegistry to evaluate consensus
4. Validate hash similarity and gatekeeper alignment
5. Execute trade if all gates pass

Mathematical Components:
- Symbol rotation logic with configurable intervals
- Consensus evaluation from multiple AI agents
- Hash similarity validation
- Risk management integration
- Trade execution with entry/exit logic
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .agent_memory import AgentMemory
from .registry_vote_matrix import VoteRegistry
from .risk_manager import RiskManager
from .strategy_layered_gatekeeper import StrategyLayeredGatekeeper


@dataclass
class SymbolRouter:
    """Manages symbol rotation for multi-asset trading."""

    symbols: list[str]
    rotation_interval: int  # seconds
    current_index: int = 0
    last_rotation: float = 0.0

    def get_symbol(self) -> str:
        """Get current symbol, rotating if interval has passed."""
        current_time = time.time()

        if current_time - self.last_rotation >= self.rotation_interval:
            self.current_index = (self.current_index + 1) % len(self.symbols)
            self.last_rotation = current_time

        return self.symbols[self.current_index]

    def get_all_symbols(self) -> list[str]:
        """Get all available symbols."""
        return self.symbols.copy()


class SchwabotEngine:
    """Main Schwabot trading engine orchestrator."""

    def __init__(self,  api_keys: Dict[str, str], mode: str = "test") -> None:
        """Initialize Schwabot engine."""
        self.api_keys = api_keys
        self.mode = mode
        self.memory = AgentMemory()
        self.data_feed = DataFeed(api_keys.get("exchange", "binance"))

        # Initialize symbol rotation
        symbols = api_keys.get("symbol_list", ["BTC/USDC", "ETH/USDC", "XRP/USDC", "SOL/USDC"])
        interval = int(api_keys.get("rotation_interval", 225))  # 3.75min default
        self.router = SymbolRouter(symbols, interval)

        # Initialize components
        self.gatekeeper = StrategyLayeredGatekeeper()
        self.risk_manager = RiskManager()

        print(f"[Schwabot] Engine initialized in {mode.upper()} mode")

    def run_trading_cycle(self) -> Dict[str, any]:
        """Execute one complete trading cycle."""
        try:
            # 1. Fetch live market data
            symbol = self.router.get_symbol()
            tick_blob = self._fetch_market_data(symbol)
            current_hash = self._generate_market_hash(tick_blob)

            # 2. Collect AI agent votes
            votes = self._collect_agent_votes()

            # 3. Evaluate consensus
            registry = VoteRegistry(self.memory.get_performance_db())
            consensus_ok = registry.evaluate(votes)

            # 4. Validate hash similarity
            hash_ok = self._validate_hash_similarity(current_hash)

            # 5. Evaluate strategy gates
            vector_ok, gate_reason, gate_confidence = self.gatekeeper.evaluate_all_gates(tick_blob)
            exit_strategy = self.gatekeeper.get_exit_strategy(tick_blob)

            # 6. Execute trade if all gates pass
            trade_executed = False
            if all([consensus_ok, hash_ok, vector_ok]):
                trade_executed = self._execute_trade(symbol, tick_blob, exit_strategy)

            # 7. Update agent scores
            self._update_agent_scores(votes, trade_executed)

            return {
                "symbol": symbol,
                "tick_blob": tick_blob,
                "consensus_ok": consensus_ok,
                "hash_ok": hash_ok,
                "vector_ok": vector_ok,
                "trade_executed": trade_executed,
                "gate_reason": gate_reason,
                "gate_confidence": gate_confidence,
            }

        except Exception as e:
            print(f"[Schwabot] Error in trading cycle: {e}")
            return {"error": str(e)}

    def _fetch_market_data(self,  symbol: str) -> str:
        """Fetch latest market data for symbol."""
        try:
            return self.data_feed.fetch_latest_tick(symbol)
        except Exception as e:
            print(f"[Schwabot] Failed to fetch market data: {e}")
            # Fallback to simulation data
            return f"{symbol},price=63000,time={int(time.time())}"

    def _generate_market_hash(self,  tick_blob: str) -> str:
        """Generate hash from market tick data."""
        import hashlib

        return hashlib.sha256(tick_blob.encode()).hexdigest()

    def _collect_agent_votes(self) -> Dict[str, bool]:
        """Collect votes from AI agents (placeholder implementation)."""
        # This would integrate with actual AI agents
        return {
            "r1": True,
            "gpt4o": True,
            "claude": False,
        }

    def _validate_hash_similarity(self,  current_hash: str) -> bool:
        """Validate hash similarity with historical patterns."""
        # Placeholder implementation - would compare with historical patterns
        pattern_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

        # Simple prefix matching
        match_length = 0
        for _i, (a, b) in enumerate(zip(current_hash, pattern_hash)):
            if a == b:
                match_length += 1
            else:
                break

        similarity = match_length / len(current_hash)
        return similarity > 0.6

    def _execute_trade(self,  symbol: str, tick_blob: str, exit_strategy: Optional[Tuple[float, int]]) -> bool:
        """Execute trade with risk management."""
        try:
            print(f"[Schwabot] Executing trade for {symbol}")

            # Extract entry price from tick blob
            entry_price = float(tick_blob.split("price=")[1].split(",")[0])

            # Register trade in risk manager
            trade_id = f"{symbol}_{int(time.time())}"
            self.risk_manager.register_trade(trade_id, entry_price, time.time(), symbol)

            # Execute trade (placeholder - would use actual exchange API)
            if self.mode == "live":
                print(f"[Schwabot] LIVE TRADE: Buying {symbol} at {entry_price}")
                # Here you would call actual exchange API
            else:
                print(f"[Schwabot] DRY RUN: Would buy {symbol} at {entry_price}")

            return True

        except Exception as e:
            print(f"[Schwabot] Trade execution failed: {e}")
            return False

    def _update_agent_scores(self,  votes: Dict[str, bool], trade_executed: bool) -> None:
        """Update agent performance scores based on trade outcome."""
        for agent_id, approve in votes.items():
            if trade_executed:
                # Reward agents who approved successful trade
                reward = 0.05 if approve else -0.05
            else:
                # Penalize agents who approved failed trade
                reward = -0.02 if approve else 0.02

            self.memory.update_score(agent_id, reward)

    def get_engine_stats(self) -> Dict[str, any]:
        """Get engine statistics."""
        return {
            "mode": self.mode,
            "symbols": self.router.get_all_symbols(),
            "current_symbol": self.router.get_symbol(),
            "agent_stats": self.memory.get_agent_stats(),
            "exchange_info": self.data_feed.get_exchange_info(),
        }


def launch_schwabot(api_keys: Dict[str, str], mode: str = "test") -> None:
    """Launch Schwabot trading system."""
    print(f"[Schwabot] Launching in {mode.upper()} mode...")

    engine = SchwabotEngine(api_keys, mode)

    try:
        while True:
            result = engine.run_trading_cycle()
            if "error" in result:
                print(f"[Schwabot] Cycle error: {result['error']}")

            # Wait before next cycle
            time.sleep(60)  # 1 minute between cycles

    except KeyboardInterrupt:
        print("[Schwabot] Shutting down...")
    except Exception as e:
        print(f"[Schwabot] Fatal error: {e}")
