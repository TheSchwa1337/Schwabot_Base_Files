from __future__ import annotations
import math

# Import safe print for Windows compatibility
try:
    pass
    pass
from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    pass
    pass
    try:
    pass
    pass
#         from core.utils.windows_cli_compatibility import safe_print, safe_format_error, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
    pass
    pass
def safe_print(message):


    pass
    pass
    print(message)
def info(message):


    pass
    pass
    print(f"[INFO] {message}")
def warn(message):


    pass
    pass
    print(f"[WARN] {message}")
def error(message):


    pass
    pass
    print(f"[ERROR] {message}")
def success(message):


    pass
    pass
    print(f"[SUCCESS] {message}")
def debug(message):


    pass
    pass
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""Risk Guard - Safety and Capital Controls for Schwabot.

This module provides comprehensive risk management including:
- Global daily-loss, single-trade, and exposure caps
- Circuit-breaker tied to abnormal entropy/volatility spikes
- Position reconciliation against exchange balances
- Manual panic button CLI
- Integration with Fault Bus for automated safety
"""


import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

# Import unified mathematics
try:
    pass
    pass
from core.unified_mathematics_config import get_unified_math
unified_math = get_unified_math()
    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    pass
    pass
UNIFIED_MATH_AVAILABLE = False

# Import fault bus for integration
try:
    pass
    pass
from core.fault_bus import get_fault_bus
fault_bus = get_fault_bus()
    FAULT_BUS_AVAILABLE = True
except ImportError:
    pass
    pass
FAULT_BUS_AVAILABLE = False

# Import centralized CLI handler
try:
    pass
    pass
from core.utils.windows_cli_compatibility import (, safe_format_error
        safe_print, safe_format_error, log_safe

CLI_HANDLER_AVAILABLE = True
except ImportError:
    pass
    pass
CLI_HANDLER_AVAILABLE = False
def safe_print(message: str, use_emoji: bool = True) -> str:


    pass
    pass
        return message
def safe_format_error(error: Exception, context: str = "") -> str:


    pass
    pass
        return f"Error: {str(error)} | Context: {context}"
def log_safe(logger, level: str, message: str) -> None:


    pass
    pass
        getattr(logger, level.lower())(message)

logger = logging.getLogger(__name__)


class RiskLevel(Enum):


    """Risk levels for different market conditions."""
LOW = "low"          # Normal market conditions
MEDIUM = "medium"    # Elevated volatility
HIGH = "high"        # High risk conditions
CRITICAL = "critical"  # Emergency conditions


class CircuitBreakerState(Enum):


    """Circuit breaker states."""
NORMAL = "normal"        # Normal operation
WARNING = "warning"      # Warning threshold reached
TRIPPED = "tripped"      # Circuit breaker activated
RESET = "reset"          # Circuit breaker reset


@dataclass
class RiskLimits:


    """Risk limits configuration."""
daily_loss_limit: float = 1000.0      # Maximum daily loss in USD
single_trade_limit: float = 100.0     # Maximum single trade size in USD
exposure_limit: float = 5000.0        # Maximum total exposure in USD
volatility_threshold: float = 0.05    # Volatility threshold for circuit breaker
entropy_threshold: float = 0.8        # Entropy threshold for circuit breaker
position_reconciliation_interval: int = 300  # Reconciliation interval in seconds


@dataclass
class PositionData:


    """Position data for reconciliation."""
asset: str
quantity: float
entry_price: float
current_price: float
unrealized_pnl: float
timestamp: datetime
exchange_balance: Optional[float] = None
reconciled: bool = False


@dataclass
class RiskEvent:


    """Risk event data."""
event_type: str
severity: RiskLevel
description: str
timestamp: datetime
triggered_by: str
action_taken: str
metadata: Dict[str, Any] = field(default_factory=dict)


class RiskGuard:


    """
Risk Guard - Safety and capital controls for Schwabot.

Provides comprehensive risk management including:
- Global daily-loss, single-trade, and exposure caps
- Circuit-breaker tied to abnormal entropy/volatility spikes
- Position reconciliation against exchange balances
- Manual panic button CLI
"""

def __init__(self, config: Optional[Dict[str, Any]] = None):


    pass
    pass
        """Initialize risk guard."""
self.config = config or {}

        # Risk limits
self.risk_limits = RiskLimits()
        self.current_risk_level = RiskLevel.LOW
self.circuit_breaker_state = CircuitBreakerState.NORMAL

        # Daily tracking
self.daily_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.daily_pnl = 0.0
self.daily_trades = 0
self.daily_volume = 0.0

        # Position tracking
self.positions: Dict[str, PositionData] = {}
self.total_exposure = 0.0
self.last_reconciliation = datetime.now()

        # Circuit breaker tracking
self.volatility_history: List[float] = []
self.entropy_history: List[float] = []
self.circuit_breaker_events: List[RiskEvent] = []

        # Risk events
self.risk_events: List[RiskEvent] = []
self.panic_mode = False
self.panic_triggered_at: Optional[datetime] = None

        # Performance tracking
self.total_risk_checks = 0
self.risk_violations = 0
self.circuit_breaker_trips = 0

safe_safe_print("🛡️ Risk Guard initialized")

def set_risk_limits(self, limits: RiskLimits) -> None:


    pass
    pass
        """Set risk limits."""
self.risk_limits = limits
safe_safe_print(f"✅ Risk limits updated: Daily loss = ${limits.daily_loss_limit}")

def check_daily_loss_limit(self, trade_pnl: float) -> bool:


    pass
    pass
        """Check if trade would exceed daily loss limit."""
        try:
    pass
    pass
            # Check if we need to reset daily tracking
now = datetime.now()
            if now.date() > self.daily_start_time.date():
                self._reset_daily_tracking()

            # Calculate new daily PnL
new_daily_pnl = self.daily_pnl + trade_pnl

            # Check limit
            if new_daily_pnl < -self.risk_limits.daily_loss_limit:
self._record_risk_event(
                    "daily_loss_limit",
RiskLevel.HIGH,
f"Daily loss limit would be exceeded: ${new_daily_pnl:.2f}",
"daily_loss_check"

                return False

            return True

        except Exception as e:
safe_safe_print(f"❌ Daily loss check failed: {safe_format_error(e, 'daily_loss_check')}")
            return False

def check_single_trade_limit(self, trade_size: float) -> bool:


    pass
    pass
        """Check if trade size exceeds single trade limit."""
        try:
    pass
    pass
            if trade_size > self.risk_limits.single_trade_limit:
self._record_risk_event(
                    "single_trade_limit",
RiskLevel.MEDIUM,
f"Single trade limit exceeded: ${trade_size:.2f}",
"single_trade_check"

                return False

            return True

        except Exception as e:
safe_safe_print(f"❌ Single trade check failed: {safe_format_error(e, 'single_trade_check')}")
            return False

def check_exposure_limit(self, new_exposure: float) -> bool:


    pass
    pass
        """Check if new exposure would exceed total exposure limit."""
        try:
    pass
    pass
total_exposure = self.total_exposure + new_exposure

            if total_exposure > self.risk_limits.exposure_limit:
self._record_risk_event(
                    "exposure_limit",
RiskLevel.HIGH,
f"Exposure limit would be exceeded: ${total_exposure:.2f}",
"exposure_check"

                return False

            return True

        except Exception as e:
safe_safe_print(f"❌ Exposure check failed: {safe_format_error(e, 'exposure_check')}")
            return False

def check_circuit_breaker(


        self,
volatility: float,
entropy: float,
market_data: Optional[Dict[str, Any]] = None
) -> bool:
"""
Check circuit breaker conditions.

Circuit breaker is triggered by:
- High volatility spikes
- Abnormal entropy levels
- Market anomalies
"""
        try:
    pass
    pass
            # Update history
self.volatility_history.append(volatility)
            self.entropy_history.append(entropy)

            # Keep only recent history
            if len(self.volatility_history) > 100:
                self.volatility_history = self.volatility_history[-100:]
            if len(self.entropy_history) > 100:
                self.entropy_history = self.entropy_history[-100:]

            # Check volatility threshold
volatility_triggered = volatility > self.risk_limits.volatility_threshold

            # Check entropy threshold
entropy_triggered = entropy > self.risk_limits.entropy_threshold

            # Check for volatility spikes (sudden large increases)
            volatility_spike = False
            if len(self.volatility_history) >= 2:
                volatility_change = unified_math.abs(volatility - self.volatility_history[-2])
                volatility_spike = volatility_change > (self.risk_limits.volatility_threshold * 0.5)

            # Determine circuit breaker state
            if volatility_triggered or entropy_triggered or volatility_spike:
                if self.circuit_breaker_state == CircuitBreakerState.NORMAL:
self.circuit_breaker_state = CircuitBreakerState.WARNING
self._record_circuit_breaker_event("warning", volatility, entropy)

                if self.circuit_breaker_state == CircuitBreakerState.WARNING:
self.circuit_breaker_state = CircuitBreakerState.TRIPPED
self.circuit_breaker_trips += 1
self._record_circuit_breaker_event("tripped", volatility, entropy)
                    return False

            elif self.circuit_breaker_state != CircuitBreakerState.NORMAL:
                # Reset circuit breaker if conditions normalize
self.circuit_breaker_state = CircuitBreakerState.NORMAL
self._record_circuit_breaker_event("reset", volatility, entropy)

            return True

        except Exception as e:
safe_safe_print(f"❌ Circuit breaker check failed: {safe_format_error(e, 'circuit_breaker')}")
            return False

def update_position(


        self,
asset: str,
quantity: float,
entry_price: float,
current_price: float
) -> None:
"""Update position data."""
        try:
    pass
    pass
unrealized_pnl = (current_price - entry_price) * quantity

position = PositionData(
                asset=asset,
quantity=quantity,
entry_price=entry_price,
current_price=current_price,
unrealized_pnl=unrealized_pnl,
timestamp=datetime.now()


self.positions[asset] = position

            # Update total exposure
self.total_exposure = sum(unified_math.abs(pos.quantity * pos.current_price) for pos in self.positions.values())

safe_safe_print(f"✅ Position updated: {asset} = ${unrealized_pnl:.2f}")

        except Exception as e:
safe_safe_print(f"❌ Position update failed: {safe_format_error(e, 'update_position')}")

async def reconcile_positions(self, exchange_balances: Dict[str, float]) -> Dict[str, Any]:
        """
Reconcile positions against exchange balances.

This ensures our internal position tracking matches
the actual exchange balances.
"""
        try:
    pass
    pass
reconciliation_results = {
'reconciled': True,
'discrepancies': [],
'total_discrepancy': 0.0
}

            for asset, position in self.positions.items():
                exchange_balance = exchange_balances.get(asset, 0.0)
                internal_balance = position.quantity

discrepancy = unified_math.abs(exchange_balance - internal_balance)

                if discrepancy > 0.001:  # Allow for small rounding differences
reconciliation_results['discrepancies'].append({]]
                        'asset': asset,
'internal': internal_balance,
'exchange': exchange_balance,
'discrepancy': discrepancy
})
reconciliation_results['total_discrepancy'] += discrepancy

                    # Mark position as unreconciled
position.exchange_balance = exchange_balance
position.reconciled = False

self._record_risk_event(
                        "position_discrepancy",
RiskLevel.MEDIUM,
f"Position discrepancy for {asset}: {discrepancy:.6f}",
"position_reconciliation"

                else:
position.exchange_balance = exchange_balance
position.reconciled = True

self.last_reconciliation = datetime.now()

            if reconciliation_results['discrepancies']:
reconciliation_results['reconciled'] = False
safe_safe_print(f"⚠️ Position reconciliation found {len(reconciliation_results['discrepancies'])} discrepancies")
            else:
safe_safe_print("✅ Position reconciliation successful")

            return reconciliation_results

        except Exception as e:
safe_safe_print(f"❌ Position reconciliation failed: {safe_format_error(e, 'position_reconciliation')}")
            return {'reconciled': False, 'error': str(e)}

def trigger_panic_mode(self, reason: str = "Manual trigger") -> None:


    pass
    pass
        """
Trigger panic mode - emergency stop for all trading.

This is the manual panic button that immediately stops
all trading activity.
"""
        try:
    pass
    pass
self.panic_mode = True
self.panic_triggered_at = datetime.now()

self._record_risk_event(
                "panic_mode",
RiskLevel.CRITICAL,
f"Panic mode triggered: {reason}",
"manual_trigger"


            # Notify fault bus if available
            if FAULT_BUS_AVAILABLE:
fault_bus.record_fault(
                    fault_type="risk_guard_panic",
severity="critical",
description=f"Panic mode triggered: {reason}",
context="risk_guard"


safe_safe_print(f"🚨 PANIC MODE TRIGGERED: {reason}")
            safe_safe_print("🛑 All trading activity stopped")

        except Exception as e:
safe_safe_print(f"❌ Panic mode trigger failed: {safe_format_error(e, 'panic_mode')}")

def reset_panic_mode(self) -> None:


    pass
    pass
        """Reset panic mode."""
        try:
    pass
    pass
self.panic_mode = False
self.panic_triggered_at = None

self._record_risk_event(
                "panic_mode_reset",
RiskLevel.LOW,
"Panic mode reset",
"manual_reset"


safe_safe_print("✅ Panic mode reset")

        except Exception as e:
safe_safe_print(f"❌ Panic mode reset failed: {safe_format_error(e, 'panic_reset')}")

def is_trading_allowed(self) -> bool:


    pass
    pass
        """Check if trading is currently allowed."""
        return (
            not self.panic_mode and
self.circuit_breaker_state != CircuitBreakerState.TRIPPED and
self.current_risk_level != RiskLevel.CRITICAL


def get_risk_status(self) -> Dict[str, Any]:


    pass
    pass
        """Get current risk status."""
        return {
'panic_mode': self.panic_mode,
'panic_triggered_at': self.panic_triggered_at.isoformat() if self.panic_triggered_at else None,
            'circuit_breaker_state': self.circuit_breaker_state.value,
'current_risk_level': self.current_risk_level.value,
'daily_pnl': self.daily_pnl,
'daily_trades': self.daily_trades,
'total_exposure': self.total_exposure,
'total_positions': len(self.positions),
            'last_reconciliation': self.last_reconciliation.isoformat(),
            'total_risk_checks': self.total_risk_checks,
'risk_violations': self.risk_violations,
'circuit_breaker_trips': self.circuit_breaker_trips
}

def _reset_daily_tracking(self) -> None:


    pass
    pass
        """Reset daily tracking counters."""
self.daily_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.daily_pnl = 0.0
self.daily_trades = 0
self.daily_volume = 0.0
safe_safe_print("🔄 Daily tracking reset")

def _record_risk_event(


        self,
event_type: str,
severity: RiskLevel,
description: str,
triggered_by: str,
metadata: Optional[Dict[str, Any]] = None
) -> None:
"""Record a risk event."""
        try:
    pass
    pass
event = RiskEvent(
                event_type=event_type,
severity=severity,
description=description,
timestamp=datetime.now(),
                triggered_by=triggered_by,
action_taken="logged",
metadata=metadata or {}


self.risk_events.append(event)
            self.risk_violations += 1

            # Keep only recent events
            if len(self.risk_events) > 1000:
                self.risk_events = self.risk_events[-1000:]

safe_safe_print(f"⚠️ Risk event: {event_type} - {description}")

        except Exception as e:
safe_safe_print(f"❌ Risk event recording failed: {safe_format_error(e, 'record_risk_event')}")

def _record_circuit_breaker_event(


        self,
event_type: str,
volatility: float,
entropy: float
) -> None:
"""Record circuit breaker event."""
        try:
    pass
    pass
event = RiskEvent(
                event_type=f"circuit_breaker_{event_type}",
severity=RiskLevel.HIGH if event_type == "tripped" else RiskLevel.MEDIUM,
description=f"Circuit breaker {event_type}: volatility={volatility:.4f}, entropy={entropy:.4f}",
timestamp=datetime.now(),
                triggered_by="circuit_breaker",
action_taken="circuit_breaker_activation" if event_type == "tripped" else "monitoring",
metadata={
'volatility': volatility,
'entropy': entropy,
'state': self.circuit_breaker_state.value
}


self.circuit_breaker_events.append(event)

            # Keep only recent events
            if len(self.circuit_breaker_events) > 100:
                self.circuit_breaker_events = self.circuit_breaker_events[-100:]

safe_safe_print(f"⚡ Circuit breaker {event_type}: volatility={volatility:.4f}, entropy={entropy:.4f}")

        except Exception as e:
safe_safe_print(f"❌ Circuit breaker event recording failed: {safe_format_error(e, 'record_circuit_breaker')}")


# Global risk guard instance
risk_guard = RiskGuard()


# Convenience functions for external access
def get_risk_guard() -> RiskGuard:


    pass
    pass
    """Get global risk guard instance."""
    return risk_guard


def check_risk_limits(trade_pnl: float, trade_size: float, new_exposure: float) -> bool:


    pass
    pass
    """Check all risk limits for a trade."""
guard = get_risk_guard()

    # Update tracking
guard.total_risk_checks += 1

    # Check all limits
daily_ok = guard.check_daily_loss_limit(trade_pnl)
    trade_ok = guard.check_single_trade_limit(trade_size)
    exposure_ok = guard.check_exposure_limit(new_exposure)

    return daily_ok and trade_ok and exposure_ok


def check_circuit_breaker(volatility: float, entropy: float) -> bool:


    pass
    pass
    """Check circuit breaker conditions."""
guard = get_risk_guard()
    return guard.check_circuit_breaker(volatility, entropy)


def trigger_panic_mode(reason: str = "Manual trigger") -> None:


    pass
    pass
    """Trigger panic mode."""
guard = get_risk_guard()
    guard.trigger_panic_mode(reason)


def reset_panic_mode() -> None:


    pass
    pass
    """Reset panic mode."""
guard = get_risk_guard()
    guard.reset_panic_mode()


def is_trading_allowed() -> bool:


    pass
    pass
    """Check if trading is currently allowed."""
guard = get_risk_guard()
    return guard.is_trading_allowed()


def get_risk_status() -> Dict[str, Any]:


    pass
    pass
    """Get current risk status."""
guard = get_risk_guard()
    return guard.get_risk_status()


# Example usage

if __name__ == "__main__":
    pass
    pass
    # Test risk guard
safe_print("🧪 Testing Risk Guard...")

guard = get_risk_guard()

    # Test risk limits
trade_ok = check_risk_limits(trade_pnl=-50.0, trade_size=75.0, new_exposure=1000.0)
    safe_print(f"✅ Risk limit check: {trade_ok}")

    # Test circuit breaker
circuit_ok = check_circuit_breaker(volatility=0.03, entropy=0.6)
    safe_print(f"✅ Circuit breaker check: {circuit_ok}")

    # Test panic mode
trigger_panic_mode("Test trigger")
    safe_print(f"✅ Panic mode: {guard.panic_mode}")

    # Reset panic mode
reset_panic_mode()
    safe_print(f"✅ Panic mode reset: {not guard.panic_mode}")

    # Get status
status = get_risk_status()
    safe_print(f"✅ Risk Status: {status}")
