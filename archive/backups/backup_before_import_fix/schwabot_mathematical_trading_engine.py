#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Schwabot Mathematical Trading Engine

This is the main orchestrator that properly connects all mathematical components:
- Hash generation from market data
- Strategy bit mapping with expansion modes
- Qutrit gate application
- Entropy signal integration
- Matrix mapping and fallback logic
- Trading signal generation and execution

Mathematical Chain:
Market Data → Hash → Bit Expansion → Qutrit Gate → Entropy Adjustment → Trading Decision
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.matrix_mapper import EnhancedMatrixMapper
from core.real_time_market_data import RealTimeMarketData
from core.strategy_bit_mapper import ExpansionMode, StrategyBitMapper
from core.trading_strategy_executor import MathematicalStrategyResult, TradingStrategyExecutor

logger = logging.getLogger(__name__)


@dataclass
class MathematicalTradingContext:
    """Context for mathematical trading decision."""

    symbol: str
    market_data: Dict[str, Any]
    tick_blob: str
    market_hash: str
    strategy_id: str
    timestamp: float


class SchwabotMathematicalTradingEngine:
    """
    Mathematical Trading Engine that orchestrates the complete trading chain.

    This engine properly implements your mathematical framework:
    1. Market data → Hash generation
    2. Hash → Strategy bit mapping with expansion
    3. Strategy → Qutrit gate application
    4. Qutrit result → Entropy signal integration
    5. Mathematical result → Trading signal conversion
    6. Trading signal → Execution with proper risk management
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the mathematical trading engine."""
        self.config = config

        # Core mathematical components
        self.strategy_bit_mapper: Optional[StrategyBitMapper] = None
        self.matrix_mapper: Optional[EnhancedMatrixMapper] = None
        self.trading_executor: Optional[TradingStrategyExecutor] = None
        self.market_data_feed: Optional[RealTimeMarketData] = None

        # Engine settings
        self.matrix_dir = config.get("matrix_dir", "./matrices")
        self.weather_api_key = config.get("weather_api_key")
        self.trading_symbols = config.get("trading_symbols", ["BTC/USDC"])
        self.hash_bits = config.get("hash_bits", 8)
        self.expansion_mode = config.get("expansion_mode", ExpansionMode.ORBITAL_ADAPTIVE)

        # Mathematical thresholds
        self.confidence_threshold = config.get("math_confidence_threshold", 0.7)
        self.entropy_threshold = config.get("entropy_threshold", 0.5)
        self.hash_similarity_threshold = config.get("hash_similarity_threshold", 0.8)

        # Performance tracking
        self.executed_trades = []
        self.mathematical_decisions = []
        self.hash_performance = {}

        logger.info("🧠 Mathematical Trading Engine initialized")

    async def initialize(self):
        """Initialize all mathematical components."""
        try:
            # Initialize strategy bit mapper
            self.strategy_bit_mapper = StrategyBitMapper(
                matrix_dir=self.matrix_dir, weather_api_key=self.weather_api_key
            )

            # Initialize matrix mapper
            self.matrix_mapper = EnhancedMatrixMapper(matrix_dir=self.matrix_dir, weather_api_key=self.weather_api_key)

            # Initialize trading executor
            self.trading_executor = TradingStrategyExecutor(self.config)

            # Initialize market data feed
            self.market_data_feed = RealTimeMarketData(self.config)

            logger.info("✅ All mathematical components initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize mathematical components: {e}")
            raise

    async def process_market_tick(self, symbol: str, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a market tick through the complete mathematical chain.

        This is the main method that implements your mathematical trading logic:
        1. Generate hash from tick data
        2. Apply bit expansion strategies
        3. Use qutrit gates for decision making
        4. Integrate entropy signals
        5. Execute trades based on mathematical confidence
        """
        try:
            # Step 1: Create trading context
            context = self._create_trading_context(symbol, tick_data)

            # Step 2: Generate strategy from hash
            strategy_id = self._hash_to_strategy_id(context.market_hash)

            # Step 3: Apply bit expansion
            expanded_strategy_id = self.strategy_bit_mapper.expand_strategy_bits(
                strategy_id=strategy_id,
                target_bits=self.hash_bits,
                mode=self.expansion_mode,
                market_data=context.market_data,
            )

            # Step 4: Apply qutrit gate with entropy integration
            qutrit_result = self.strategy_bit_mapper.apply_qutrit_gate(
                strategy_id=str(expanded_strategy_id),
                seed=context.market_hash[:16],  # Use hash segment as seed
                market_data=context.market_data,
            )

            # Step 5: Evaluate mathematical decision
            decision_result = await self._evaluate_mathematical_decision(context, qutrit_result)

            # Step 6: Execute trade if decision is positive
            execution_result = None
            if decision_result["should_execute"]:
                execution_result = await self.trading_executor.process_mathematical_strategy(
                    strategy_id=str(expanded_strategy_id),
                    market_data=context.market_data,
                    hash_seed=context.market_hash[:16],
                )

            # Step 7: Track performance and update mathematical models
            await self._track_mathematical_performance(context, qutrit_result, decision_result, execution_result)

            return {
                "context": context,
                "strategy_id": expanded_strategy_id,
                "qutrit_result": qutrit_result,
                "decision_result": decision_result,
                "execution_result": execution_result,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error processing market tick: {e}")
            return None

    def _create_trading_context(self, symbol: str, tick_data: Dict[str, Any]) -> MathematicalTradingContext:
        """Create trading context from market tick."""
        # Generate tick blob for hashing
        tick_blob = self._generate_tick_blob(symbol, tick_data)

        # Generate market hash
        market_hash = hashlib.sha256(tick_blob.encode()).hexdigest()

        # Create strategy ID from hash
        strategy_id = self._hash_to_strategy_id(market_hash)

        return MathematicalTradingContext(
            symbol=symbol,
            market_data=tick_data,
            tick_blob=tick_blob,
            market_hash=market_hash,
            strategy_id=strategy_id,
            timestamp=time.time(),
        )

    def _generate_tick_blob(self, symbol: str, tick_data: Dict[str, Any]) -> str:
        """Generate standardized tick blob for hashing."""
        price = tick_data.get("price", 0.0)
        volume = tick_data.get("volume", 0.0)
        timestamp = tick_data.get("timestamp", time.time())

        # Create standardized blob format
        tick_blob = f"{symbol},price={price:.8f},volume={volume:.8f},time={int(timestamp)}"

        # Add additional market data if available
        if "bid" in tick_data and "ask" in tick_data:
            tick_blob += f",bid={tick_data['bid']:.8f},ask={tick_data['ask']:.8f}"

        if "volatility" in tick_data:
            tick_blob += f",vol={tick_data['volatility']:.6f}"

        return tick_blob

    def _hash_to_strategy_id(self, market_hash: str) -> str:
        """Convert hash to strategy ID."""
        # Use first 8 characters of hash as strategy ID
        return market_hash[:8]

    async def _evaluate_mathematical_decision(
        self, context: MathematicalTradingContext, qutrit_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate whether to execute trade based on mathematical criteria.

        This implements your multi-layered decision logic:
        - Qutrit state analysis
        - Confidence thresholds
        - Entropy adjustments
        - Hash similarity validation
        """
        # Extract key metrics
        action = qutrit_result.get("action", "defer")
        confidence = qutrit_result.get("confidence", 0.0)
        entropy_adjustment = qutrit_result.get("entropy_adjustment", 1.0)
        qutrit_state = qutrit_result.get("qutrit_state", "unknown")

        # Calculate adjusted confidence
        adjusted_confidence = confidence * entropy_adjustment

        # Decision criteria
        criteria = {
            "action_execute": action == "execute",
            "confidence_threshold": adjusted_confidence >= self.confidence_threshold,
            "entropy_threshold": entropy_adjustment >= self.entropy_threshold,
            "qutrit_state_valid": qutrit_state in ["EXECUTE", "execute"],
            "hash_similarity": await self._validate_hash_similarity(context.market_hash),
        }

        # Overall decision
        should_execute = all(criteria.values())

        # Calculate decision score
        decision_score = sum(criteria.values()) / len(criteria)

        return {
            "should_execute": should_execute,
            "decision_score": decision_score,
            "adjusted_confidence": adjusted_confidence,
            "criteria": criteria,
            "reasoning": self._generate_decision_reasoning(criteria, qutrit_result),
        }

    async def _validate_hash_similarity(self, market_hash: str) -> bool:
        """Validate hash similarity with historical patterns."""
        # This is where you would implement your hash similarity validation
        # For now, implementing a basic validation

        if not hasattr(self, '_historical_hashes'):
            self._historical_hashes = []

        # Add current hash to history
        self._historical_hashes.append(market_hash)

        # Keep only recent hashes
        if len(self._historical_hashes) > 100:
            self._historical_hashes = self._historical_hashes[-100:]

        # Simple similarity check (you can enhance this)
        if len(self._historical_hashes) < 2:
            return True

        # Check similarity with recent hashes
        recent_hash = self._historical_hashes[-2]
        similarity = self._calculate_hash_similarity(market_hash, recent_hash)

        return similarity >= self.hash_similarity_threshold

    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes."""
        # Count matching characters in same positions
        matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
        return matches / max(len(hash1), len(hash2))

    def _generate_decision_reasoning(self, criteria: Dict[str, bool], qutrit_result: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the decision."""
        passed = [k for k, v in criteria.items() if v]
        failed = [k for k, v in criteria.items() if not v]

        if len(passed) == len(criteria):
            return f"All criteria passed: {', '.join(passed)}"
        elif len(failed) == len(criteria):
            return f"All criteria failed: {', '.join(failed)}"
        else:
            return f"Passed: {', '.join(passed)}. Failed: {', '.join(failed)}"

    async def _track_mathematical_performance(
        self,
        context: MathematicalTradingContext,
        qutrit_result: Dict[str, Any],
        decision_result: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]],
    ) -> None:
        """Track performance of mathematical decisions."""
        # Record mathematical decision
        decision_record = {
            "timestamp": context.timestamp,
            "symbol": context.symbol,
            "market_hash": context.market_hash,
            "strategy_id": context.strategy_id,
            "qutrit_state": qutrit_result.get("qutrit_state"),
            "confidence": qutrit_result.get("confidence"),
            "entropy_adjustment": qutrit_result.get("entropy_adjustment"),
            "decision_score": decision_result.get("decision_score"),
            "should_execute": decision_result.get("should_execute"),
            "executed": execution_result is not None and execution_result.get("executed", False),
        }

        self.mathematical_decisions.append(decision_record)

        # Track hash performance
        hash_key = context.market_hash[:8]
        if hash_key not in self.hash_performance:
            self.hash_performance[hash_key] = {
                "count": 0,
                "executed": 0,
                "successful": 0,
                "total_confidence": 0.0,
            }

        perf = self.hash_performance[hash_key]
        perf["count"] += 1
        perf["total_confidence"] += qutrit_result.get("confidence", 0.0)

        if execution_result and execution_result.get("executed"):
            perf["executed"] += 1
            # You would track success based on actual trade outcomes
            # For now, assuming high confidence trades are more likely to succeed
            if qutrit_result.get("confidence", 0.0) > 0.8:
                perf["successful"] += 1

        # Keep only recent decisions
        if len(self.mathematical_decisions) > 1000:
            self.mathematical_decisions = self.mathematical_decisions[-1000:]

    def get_mathematical_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for mathematical trading."""
        if not self.mathematical_decisions:
            return {"status": "No decisions recorded"}

        total_decisions = len(self.mathematical_decisions)
        executed_decisions = sum(1 for d in self.mathematical_decisions if d["executed"])
        avg_confidence = sum(d["confidence"] for d in self.mathematical_decisions) / total_decisions
        avg_decision_score = sum(d["decision_score"] for d in self.mathematical_decisions) / total_decisions

        return {
            "total_decisions": total_decisions,
            "executed_decisions": executed_decisions,
            "execution_rate": executed_decisions / total_decisions if total_decisions > 0 else 0,
            "average_confidence": avg_confidence,
            "average_decision_score": avg_decision_score,
            "hash_performance": dict(list(self.hash_performance.items())[:10]),  # Top 10
        }


async def create_mathematical_trading_engine(
    config: Dict[str, Any],
) -> SchwabotMathematicalTradingEngine:
    """Create and initialize the mathematical trading engine."""
    engine = SchwabotMathematicalTradingEngine(config)
    await engine.initialize()
    return engine
