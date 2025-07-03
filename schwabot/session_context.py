from __future__ import annotations

"""
Session Context Manager for SchwaBot
====================================

Provides AsyncLocalStorage-like functionality for Python, integrated with
the Vortex Math Security Protocol (VMSP) for secure session management.

This module enables:
- Persistent context across async tick cycles
- Secure session handoff between AI agents
- Pattern-based authentication using recursive mathematics
- Integration with trading strategy memory
"""

from contextvars import ContextVar, copy_context
import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from .vortex_security import SecurityState, get_vortex_security


class SessionType(Enum):
    """Types of trading sessions"""

    TRADING = "trading"
    ANALYSIS = "analysis"
    BACKTEST = "backtest"
    DIAGNOSTIC = "diagnostic"


@dataclass
class TradeContext:
    """Complete trading context state"""

    session_id: str
    session_type: SessionType
    ai_agent: str
    strategy_hash: str
    market_pair: str
    entry_price: Optional[float] = None
    entry_time: Optional[float] = None
    decision_vector: str = ""
    market_state: str = ""
    security_state: Optional[SecurityState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "session_type": self.session_type.value,
            "ai_agent": self.ai_agent,
            "strategy_hash": self.strategy_hash,
            "market_pair": self.market_pair,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "decision_vector": self.decision_vector,
            "market_state": self.market_state,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class SecureSessionManager:
    """
    Manages secure trading sessions with VMSP integration
    """

    def __init__(self):
        # Core context variable
        self._context: ContextVar[Optional[TradeContext]] = ContextVar(
            "trade_session_context", default=None
        )

        # Security integration
        self.security = get_vortex_security()

        # Session tracking
        self.active_sessions: Dict[str, TradeContext] = {}
        self.session_history: List[TradeContext] = []

        # Activity logging for pattern analysis
        self.activity_log: List[Dict[str, Any]] = []

    def create_session(
        self,
        ai_agent: str,
        strategy_hash: str,
        market_pair: str,
        session_type: SessionType = SessionType.TRADING,
        decision_vector: str = "",
        market_state: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TradeContext:
        """
        Create new secure trading session with VMSP protection
        """
        session_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}

        # Create security state
        threat_inputs = [
            0.3,  # Base threat level
            len(self.active_sessions) * 0.1,  # Load factor
            hash(ai_agent) % 100 / 100.0,  # Agent entropy
        ]

        security_state = self.security.create_secure_session(
            decision_vector=decision_vector or f"{ai_agent}_{strategy_hash}",
            market_state=market_state or f"{market_pair}_neutral",
            threat_inputs=threat_inputs,
        )

        # Create trade context
        context = TradeContext(
            session_id=session_id,
            session_type=session_type,
            ai_agent=ai_agent,
            strategy_hash=strategy_hash,
            market_pair=market_pair,
            decision_vector=decision_vector,
            market_state=market_state,
            security_state=security_state,
            metadata=metadata,
        )

        # Store session
        self.active_sessions[session_id] = context
        self.session_history.append(context)

        # Set as current context
        self._context.set(context)

        # Log activity
        self.log_activity(
            "session_created",
            {
                "session_id": session_id,
                "ai_agent": ai_agent,
                "strategy_hash": strategy_hash,
                "market_pair": market_pair,
                "security_hash": security_state.hash,
            },
        )

        return context

    def get_current_context(self) -> Optional[TradeContext]:
        """Get current trading context"""
        return self._context.get()

    def set_context(self, context: TradeContext) -> None:
        """Set trading context"""
        self._context.set(context)

    def update_context(self, **updates) -> TradeContext:
        """Update current context with new values"""
        current = self.get_current_context()
        if not current:
            raise ValueError("No active context to update")

        # Apply updates
        for key, value in updates.items():
            if hasattr(current, key):
                setattr(current, key, value)
            else:
                current.metadata[key] = value

        # Validate security
        if not self.validate_context_security(current):
            self.security.enforce_security_lockdown(
                "Context update failed security validation", current.security_state
            )

        return current

    def validate_context_security(self, context: TradeContext) -> bool:
        """Validate context using VMSP"""
        if not context.security_state:
            return False

        # Create validation inputs
        validation_inputs = [
            hash(context.ai_agent) % 100 / 100.0,
            hash(context.strategy_hash) % 100 / 100.0,
            time.time() - context.created_at,  # Age factor
        ]

        return self.security.validate_security_state(validation_inputs)

    def log_activity(self, action: str, data: Dict[str, Any]) -> None:
        """
        Log trading activity for pattern analysis
        Similar to trackActivity in Node.js example
        """
        current_context = self.get_current_context()

        activity_entry = {
            "action": action,
            "data": data,
            "timestamp": time.time(),
            "session_id": current_context.session_id if current_context else None,
            "security_hash": (
                current_context.security_state.hash
                if current_context and current_context.security_state
                else None
            ),
        }

        self.activity_log.append(activity_entry)

        # Maintain log size
        if len(self.activity_log) > 1000:
            self.activity_log = self.activity_log[-800:]  # Keep last 800 entries

    def close_session(
        self, session_id: str, exit_price: Optional[float] = None
    ) -> None:
        """Close trading session and calculate results"""
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]

        # Update with exit information
        if exit_price is not None:
            session.metadata["exit_price"] = exit_price
            session.metadata["exit_time"] = time.time()

            # Calculate profit if we have entry price
            if session.entry_price:
                profit = (exit_price - session.entry_price) / session.entry_price
                session.metadata["profit_percentage"] = profit

        # Log closure
        self.log_activity(
            "session_closed",
            {
                "session_id": session_id,
                "exit_price": exit_price,
                "duration": time.time() - session.created_at,
                "metadata": session.metadata,
            },
        )

        # Remove from active sessions
        del self.active_sessions[session_id]

        # Clear context if this was current
        current = self.get_current_context()
        if current and current.session_id == session_id:
            self._context.set(None)

    async def run_with_context(
        self, context: TradeContext, coro: Callable[[], Any]
    ) -> Any:
        """
        Run coroutine with specific context (like AsyncLocalStorage.run)
        """
        # Copy current context
        ctx = copy_context()

        # Set the trade context in the copy
        ctx[self._context] = context

        # Run the coroutine in the context
        return await asyncio.create_task(coro(), context=ctx)

    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics about session patterns"""
        if not self.session_history:
            return {"status": "no_data"}

        # Analyze patterns
        agent_counts = {}
        strategy_counts = {}
        pair_counts = {}

        total_profit = 0.0
        profitable_sessions = 0

        for session in self.session_history:
            # Count usage
            agent_counts[session.ai_agent] = agent_counts.get(session.ai_agent, 0) + 1
            strategy_counts[session.strategy_hash] = (
                strategy_counts.get(session.strategy_hash, 0) + 1
            )
            pair_counts[session.market_pair] = (
                pair_counts.get(session.market_pair, 0) + 1
            )

            # Calculate profits
            profit = session.metadata.get("profit_percentage")
            if profit is not None:
                total_profit += profit
                if profit > 0:
                    profitable_sessions += 1

        # Activity pattern analysis
        recent_activities = [
            a for a in self.activity_log if time.time() - a["timestamp"] < 3600
        ]  # Last hour

        activity_counts = {}
        for activity in recent_activities:
            action = activity["action"]
            activity_counts[action] = activity_counts.get(action, 0) + 1

        return {
            "status": "active",
            "total_sessions": len(self.session_history),
            "active_sessions": len(self.active_sessions),
            "agent_usage": agent_counts,
            "strategy_usage": strategy_counts,
            "pair_usage": pair_counts,
            "profit_metrics": {
                "total_profit": total_profit,
                "profitable_sessions": profitable_sessions,
                "success_rate": (
                    profitable_sessions / len(self.session_history)
                    if self.session_history
                    else 0
                ),
            },
            "recent_activity": activity_counts,
            "security_analytics": self.security.get_security_analytics(),
        }


# Global session manager instance
_session_manager: Optional[SecureSessionManager] = None


def get_session_manager() -> SecureSessionManager:
    """Get or create global session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SecureSessionManager()
    return _session_manager


# Convenience functions for easy integration


def create_trading_session(
    ai_agent: str, strategy_hash: str, market_pair: str, **kwargs
) -> TradeContext:
    """Create new trading session"""
    return get_session_manager().create_session(
        ai_agent=ai_agent,
        strategy_hash=strategy_hash,
        market_pair=market_pair,
        session_type=SessionType.TRADING,
        **kwargs,
    )


def get_current_session() -> Optional[TradeContext]:
    """Get current trading session context"""
    return get_session_manager().get_current_context()


def log_trading_activity(action: str, **data) -> None:
    """Log trading activity with current context"""
    get_session_manager().log_activity(action, data)


def update_session(**updates) -> TradeContext:
    """Update current session context"""
    return get_session_manager().update_context(**updates)


async def execute_with_session(context: TradeContext, coro: Callable[[], Any]) -> Any:
    """Execute coroutine with specific session context"""
    return await get_session_manager().run_with_context(context, coro)
