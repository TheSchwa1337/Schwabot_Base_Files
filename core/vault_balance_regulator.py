from __future__ import annotations

# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Vault Balance Regulator - Asset Allocation & Risk Management.

This module handles vault balance regulation, asset conversion logic,
and fallback mechanisms for overflow and density rebalancing.

Mathematical Foundation:
- Vault imbalance delta: Δ_vault = |B_target/B_actual - 1|
- Mean reversion trigger: φ(t) = λ * (B_actual - B_mean)
- Threshold ping logic: ζ(t) = ReLU(Δ_vault - δ)
- Rebalance vector: R_vec = D_p * vault_ratio(profit, σ_x)

Windows CLI compatible with comprehensive error handling.
"""


import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)


class Asset(Enum):
    """Supported assets for vault management."""

    BTC = "BTC"
    USDC = "USDC"
    XRP = "XRP"
    ETH = "ETH"


class RebalanceAction(Enum):
    """Rebalance action types."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EMERGENCY_CONVERT = "emergency_convert"


@dataclass
class VaultBalance:
    """Vault balance information."""

    asset: Asset
    balance: float                     # Current balance
    target_allocation: float           # Target allocation percentage [0, 1]
    actual_allocation: float           # Actual allocation percentage [0, 1]
    imbalance_delta: float             # Imbalance from target
    last_rebalance_time: float         # Timestamp of last rebalance
    rebalance_urgency: float           # Urgency of rebalance [0, 1]


@dataclass
class RebalanceSignal:
    """Rebalance signal information."""

    asset: Asset
    action: RebalanceAction
    amount: float                      # Amount to buy/sell
    confidence: float                  # Confidence in rebalance [0, 1]
    urgency: float                     # Urgency level [0, 1]
    reason: str                        # Reason for rebalance
    threshold_triggered: bool          # Whether threshold was triggered


@dataclass
class VaultState:
    """Overall vault state."""

    total_value_usd: float             # Total vault value in USD
    balance_entropy: float             # Balance distribution entropy
    risk_level: float                  # Current risk level [0, 1]
    stability_score: float             # Vault stability score
    last_rebalance_time: float         # Last rebalance timestamp
    rebalance_frequency: float         # Rebalances per hour


class VaultBalanceRegulator:
    """Regulates vault balances and asset allocation."""

    def __init__(self):
        """Initialize vault balance regulator."""
        self.vault_balances: Dict[Asset, VaultBalance] = {}
        self.rebalance_history: List[RebalanceSignal] = []
        self.balance_history: Dict[Asset, List[float]] = {
            asset: [] for asset in Asset
        }

        self.max_history = 200
        self.rebalance_cooldown = 300  # 5 minutes between rebalances

        # Target allocations (can be dynamically adjusted)
        self.target_allocations = {
            Asset.BTC: 0.6,     # 60% BTC
            Asset.USDC: 0.25,   # 25% USDC (stable)
            Asset.XRP: 0.10,    # 10% XRP
            Asset.ETH: 0.05,    # 5% ETH
        }

        # Rebalance parameters
        self.imbalance_threshold = 0.15  # 15% deviation triggers rebalance
        self.emergency_threshold = 0.35  # 35% deviation triggers emergency
        self.mean_reversion_lambda = 0.1
        self.ping_threshold_delta = 0.05

        # Risk management
        self.max_single_asset_allocation = 0.8  # 80% max in any asset
        self.min_stable_allocation = 0.15       # 15% minimum in USDC

        # Initialize vault balances
        self._initialize_vault_balances()

    def _initialize_vault_balances(self) -> None:
        """Initialize vault balance tracking."""
        for asset in Asset:
            self.vault_balances[asset] = VaultBalance(
                asset=asset,
                balance=0.0,
                target_allocation=self.target_allocations[asset],
                actual_allocation=0.0,
                imbalance_delta=0.0,
                last_rebalance_time=0.0,
                rebalance_urgency=0.0,
            )

    def update_balance(self, asset: Asset, new_balance: float) -> None:
        """Update asset balance and recalculate allocations.

        Parameters
        ----------
        asset : Asset
            Asset to update
        new_balance : float
            New balance amount
        """
        try:
            if asset not in self.vault_balances:
                logger.warning(f"Unknown asset: {asset}")
                return

            # Update balance
            self.vault_balances[asset].balance = new_balance

            # Store in history
            self.balance_history[asset].append(new_balance)
            if len(self.balance_history[asset]) > self.max_history:
                self.balance_history[asset] = self.balance_history[asset][-100:]

            # Recalculate allocations
            self._recalculate_allocations()

        except Exception as e:
            logger.error(f"Error updating balance for {asset}: {e}")

    def _recalculate_allocations(self) -> None:
        """Recalculate actual allocations and imbalance deltas."""
        try:
            # Calculate total vault value (assuming USD values)
            total_value = sum(vault.balance for vault in self.vault_balances.values())

            if total_value == 0:
                return

            # Update actual allocations and imbalance deltas
            for asset, vault in self.vault_balances.items():
                vault.actual_allocation = vault.balance / total_value
                vault.imbalance_delta = self.calculate_imbalance_delta(
                    vault.target_allocation, vault.actual_allocation
                )

                # Calculate rebalance urgency
                vault.rebalance_urgency = unified_math.min(1.0, vault.imbalance_delta / self.imbalance_threshold)

        except Exception as e:
            logger.error(f"Error recalculating allocations: {e}")

    def calculate_imbalance_delta(self, target: float, actual: float) -> float:
        """Calculate vault imbalance delta.

        Mathematical Formula:
        Δ_vault = |B_target/B_actual - 1|

        Parameters
        ----------
        target : float
            Target allocation
        actual : float
            Actual allocation

        Returns
        -------
        float
            Imbalance delta
        """
        try:
            if actual == 0:
                return 1.0 if target > 0 else 0.0

            ratio = target / actual
            imbalance_delta = unified_math.abs(ratio - 1.0)

            return imbalance_delta

        except Exception as e:
            logger.error(f"Error calculating imbalance delta: {e}")
            return 0.0

    def calculate_mean_reversion_trigger(
        self,
        asset: Asset,
        current_balance: float,
    ) -> float:
        """Calculate mean reversion trigger.

        Mathematical Formula:
        φ(t) = λ * (B_actual - B_mean)

        Parameters
        ----------
        asset : Asset
            Asset to analyze
        current_balance : float
            Current balance

        Returns
        -------
        float
            Mean reversion trigger value
        """
        try:
            balance_history = self.balance_history[asset]

            if len(balance_history) < 5:
                return 0.0

            # Calculate mean of recent balances
            recent_balances = balance_history[-20:]  # Last 20 measurements
            mean_balance = unified_math.unified_math.mean(recent_balances)

            # Calculate mean reversion trigger
            trigger = self.mean_reversion_lambda * (current_balance - mean_balance)

            return trigger

        except Exception as e:
            logger.error(f"Error calculating mean reversion trigger: {e}")
            return 0.0

    def calculate_threshold_ping(self, imbalance_delta: float) -> float:
        """Calculate threshold ping logic.

        Mathematical Formula:
        ζ(t) = ReLU(Δ_vault - δ)

        Parameters
        ----------
        imbalance_delta : float
            Imbalance delta value

        Returns
        -------
        float
            Threshold ping value
        """
        try:
            # ReLU function: unified_math.max(0, x)
            ping_value = unified_math.max(0.0, imbalance_delta - self.ping_threshold_delta)

            return ping_value

        except Exception as e:
            logger.error(f"Error calculating threshold ping: {e}")
            return 0.0

    def generate_rebalance_signals(
        self,
        profit_factor: float = 1.0,
        volatility_sigma: float = 0.1,
    ) -> List[RebalanceSignal]:
        """Generate rebalance signals for all assets.

        Parameters
        ----------
        profit_factor : float
            Profit factor for rebalance weighting
        volatility_sigma : float
            Volatility factor

        Returns
        -------
        List[RebalanceSignal]
            List of rebalance signals
        """
        try:
            signals = []
            current_time = time.time()

            for asset, vault in self.vault_balances.items():
                # Check cooldown period
                time_since_last = current_time - vault.last_rebalance_time
                if time_since_last < self.rebalance_cooldown:
                    continue

                # Calculate rebalance metrics
                imbalance_delta = vault.imbalance_delta
                threshold_ping = self.calculate_threshold_ping(imbalance_delta)
                mean_reversion = self.calculate_mean_reversion_trigger(asset, vault.balance)

                # Determine if rebalance is needed
                should_rebalance = False
                action = RebalanceAction.HOLD
                amount = 0.0
                confidence = 0.0
                urgency = vault.rebalance_urgency
                reason = "No action needed"

                # Emergency rebalance
                if imbalance_delta > self.emergency_threshold:
                    should_rebalance = True
                    action = RebalanceAction.EMERGENCY_CONVERT
                    confidence = 0.9
                    urgency = 1.0
                    reason = "Emergency rebalance - severe imbalance"

                # Normal rebalance
                elif imbalance_delta > self.imbalance_threshold:
                    should_rebalance = True
                    confidence = unified_math.min(0.8, imbalance_delta * 2)
                    reason = "Standard rebalance - allocation drift"

                    # Determine buy/sell action
                    if vault.actual_allocation < vault.target_allocation:
                        action = RebalanceAction.BUY
                        # Calculate amount to buy
                        total_value = sum(v.balance for v in self.vault_balances.values())
                        target_balance = vault.target_allocation * total_value
                        amount = target_balance - vault.balance
                    else:
                        action = RebalanceAction.SELL
                        # Calculate amount to sell
                        total_value = sum(v.balance for v in self.vault_balances.values())
                        target_balance = vault.target_allocation * total_value
                        amount = vault.balance - target_balance

                # Mean reversion trigger
                elif unified_math.abs(mean_reversion) > 0.1:
                    should_rebalance = True
                    confidence = unified_math.min(0.6, unified_math.abs(mean_reversion) * 5)
                    urgency = unified_math.min(0.7, unified_math.abs(mean_reversion) * 3)
                    reason = "Mean reversion trigger"

                    if mean_reversion > 0:
                        action = RebalanceAction.SELL
                        amount = unified_math.abs(mean_reversion) * vault.balance * 0.1
                    else:
                        action = RebalanceAction.BUY
                        amount = unified_math.abs(mean_reversion) * vault.balance * 0.1

                # Apply profit and volatility adjustments
                if should_rebalance:
                    # Adjust confidence based on profit factor
                    confidence *= profit_factor

                    # Adjust amount based on volatility
                    volatility_adjustment = 1.0 / (1.0 + volatility_sigma)
                    amount *= volatility_adjustment

                    # Create rebalance signal
                    signal = RebalanceSignal(
                        asset=asset,
                        action=action,
                        amount=amount,
                        confidence=confidence,
                        urgency=urgency,
                        reason=reason,
                        threshold_triggered=threshold_ping > 0,
                    )

                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Error generating rebalance signals: {e}")
            return []

    def execute_rebalance(self, signal: RebalanceSignal) -> bool:
        """Execute a rebalance signal.

        Parameters
        ----------
        signal : RebalanceSignal
            Rebalance signal to execute

        Returns
        -------
        bool
            True if rebalance was executed successfully
        """
        try:
            if signal.action == RebalanceAction.HOLD:
                return True

            vault = self.vault_balances[signal.asset]
            current_time = time.time()

            # Validate rebalance
            if signal.confidence < 0.3:
                logger.warning(f"Rebalance confidence too low: {signal.confidence}")
                return False

            # Execute rebalance (simulation - in real implementation would call exchange API)
            logger.info(f"Executing rebalance: {signal.action.value} {signal.amount:.4f} {signal.asset.value}")
            logger.info(f"Reason: {signal.reason}")

            # Update vault state
            vault.last_rebalance_time = current_time

            # Store in history
            self.rebalance_history.append(signal)
            if len(self.rebalance_history) > self.max_history:
                self.rebalance_history = self.rebalance_history[-100:]

            return True

        except Exception as e:
            logger.error(f"Error executing rebalance: {e}")
            return False

    def calculate_vault_state(self) -> VaultState:
        """Calculate overall vault state metrics.

        Returns
        -------
        VaultState
            Current vault state
        """
        try:
            # Calculate total value
            total_value = sum(vault.balance for vault in self.vault_balances.values())

            # Calculate balance entropy
            if total_value > 0:
                allocations = [vault.actual_allocation for vault in self.vault_balances.values()]
                allocations = [a for a in allocations if a > 0]  # Remove zero allocations

                if allocations:
                    entropy = -sum(a * unified_math.unified_math.log(a) for a in allocations)
                    # Normalize by max possible entropy
                    max_entropy = unified_math.unified_math.log(len(allocations))
                    balance_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                else:
                    balance_entropy = 0.0
            else:
                balance_entropy = 0.0

            # Calculate risk level (based on imbalances)
            max_imbalance = unified_math.max(vault.imbalance_delta for vault in self.vault_balances.values())
            risk_level = unified_math.min(1.0, max_imbalance / self.emergency_threshold)

            # Calculate stability score (inverse of risk)
            stability_score = 1.0 - risk_level

            # Calculate rebalance frequency
            recent_rebalances = [
                r for r in self.rebalance_history
                if time.time() - r.confidence < 3600  # Last hour
            ]
            rebalance_frequency = len(recent_rebalances)

            # Last rebalance time
            last_rebalance_time = (
                unified_math.max(vault.last_rebalance_time for vault in self.vault_balances.values())
                if any(vault.last_rebalance_time > 0 for vault in self.vault_balances.values())
                else 0.0
            )

            return VaultState(
                total_value_usd=total_value,
                balance_entropy=balance_entropy,
                risk_level=risk_level,
                stability_score=stability_score,
                last_rebalance_time=last_rebalance_time,
                rebalance_frequency=rebalance_frequency,
            )

        except Exception as e:
            logger.error(f"Error calculating vault state: {e}")
            return VaultState(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    def update_target_allocations(self, new_targets: Dict[Asset, float]) -> bool:
        """Update target allocations.

        Parameters
        ----------
        new_targets : Dict[Asset, float]
            New target allocations

        Returns
        -------
        bool
            True if update was successful
        """
        try:
            # Validate allocations sum to 1.0
            total_allocation = sum(new_targets.values())
            if unified_math.abs(total_allocation - 1.0) > 0.01:
                logger.error(f"Target allocations must sum to 1.0, got {total_allocation}")
                return False

            # Validate individual constraints
            for asset, allocation in new_targets.items():
                if allocation > self.max_single_asset_allocation:
                    logger.error(f"Allocation for {asset.value} exceeds maximum: {allocation}")
                    return False

            # Ensure minimum stable allocation
            if new_targets.get(Asset.USDC, 0) < self.min_stable_allocation:
                logger.error(f"USDC allocation below minimum: {new_targets.get(Asset.USDC, 0)}")
                return False

            # Update targets
            self.target_allocations = new_targets.copy()

            # Update vault balance targets
            for asset, allocation in new_targets.items():
                if asset in self.vault_balances:
                    self.vault_balances[asset].target_allocation = allocation

            # Recalculate imbalances
            self._recalculate_allocations()

            logger.info("Target allocations updated successfully")
            return True

        except Exception as e:
            logger.error(f"Error updating target allocations: {e}")
            return False

    def get_regulator_summary(self) -> Dict:
        """Get vault balance regulator summary."""
        vault_state = self.calculate_vault_state()

        return {
            "total_value_usd": vault_state.total_value_usd,
            "balance_entropy": vault_state.balance_entropy,
            "risk_level": vault_state.risk_level,
            "stability_score": vault_state.stability_score,
            "rebalance_frequency": vault_state.rebalance_frequency,
            "asset_balances": {
                asset.value: vault.balance
                for asset, vault in self.vault_balances.items()
            },
            "asset_allocations": {
                asset.value: {
                    "target": vault.target_allocation,
                    "actual": vault.actual_allocation,
                    "imbalance": vault.imbalance_delta,
                }
                for asset, vault in self.vault_balances.items()
            },
            "recent_rebalances": len(self.rebalance_history),
            "imbalance_threshold": self.imbalance_threshold,
            "emergency_threshold": self.emergency_threshold,
        }


def main() -> None:
    """Demo function for testing vault balance regulator."""
    safe_print("Vault Balance Regulator Demo")
    safe_print("=" * 35)

    regulator = VaultBalanceRegulator()

    # Simulate vault balances
    test_balances = {
        Asset.BTC: 30000.0,   # $30k BTC (should be 60% = $36k)
        Asset.USDC: 20000.0,  # $20k USDC (should be 25% = $15k)
        Asset.XRP: 5000.0,    # $5k XRP (should be 10% = $6k)
        Asset.ETH: 5000.0,    # $5k ETH (should be 5% = $3k)
    }

    safe_print("Setting initial balances:")
    for asset, balance in test_balances.items():
        regulator.update_balance(asset, balance)
        vault = regulator.vault_balances[asset]
        safe_print(f"  {asset.value}: ${balance:,.0f} (Target: {vault.target_allocation:.1%}, Actual: {vault.actual_allocation:.1%})")

    # Generate rebalance signals
    safe_print(f"\nGenerating rebalance signals:")
    signals = regulator.generate_rebalance_signals(profit_factor=1.2, volatility_sigma=0.15)

    for signal in signals:
        safe_print(f"  {signal.asset.value}: {signal.action.value}")
        safe_print(f"    Amount: ${signal.amount:,.2f}")
        safe_print(f"    Confidence: {signal.confidence:.3f}")
        safe_print(f"    Urgency: {signal.urgency:.3f}")
        safe_print(f"    Reason: {signal.reason}")
        safe_print(f"    Threshold Triggered: {signal.threshold_triggered}")

        # Execute rebalance
        executed = regulator.execute_rebalance(signal)
        safe_print(f"    Executed: {executed}")
        print()

    # Calculate vault state
    safe_print("Vault State:")
    vault_state = regulator.calculate_vault_state()
    safe_print(f"  Total Value: ${vault_state.total_value_usd:,.0f}")
    safe_print(f"  Balance Entropy: {vault_state.balance_entropy:.3f}")
    safe_print(f"  Risk Level: {vault_state.risk_level:.3f}")
    safe_print(f"  Stability Score: {vault_state.stability_score:.3f}")
    safe_print(f"  Rebalance Frequency: {vault_state.rebalance_frequency:.1f}/hour")

    # Test target allocation update
    safe_print(f"\nTesting target allocation update:")
    new_targets = {
        Asset.BTC: 0.7,     # Increase BTC to 70%
        Asset.USDC: 0.2,    # Decrease USDC to 20%
        Asset.XRP: 0.05,    # Decrease XRP to 5%
        Asset.ETH: 0.05,    # Keep ETH at 5%
    }

    updated = regulator.update_target_allocations(new_targets)
    safe_print(f"  Target update successful: {updated}")

    # Regulator summary
    safe_print(f"\nRegulator Summary:")
    summary = regulator.get_regulator_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            safe_print(f"  {key}:")
            for subkey, subvalue in value.items():
                safe_print(f"    {subkey}: {subvalue}")
        else:
            safe_print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
