# -*- coding: utf-8 -*-
""""""
Whale Tracker Integration System
================================

Advanced whale tracking system that integrates with multiple whale tracker APIs
and blockchain analysis services to provide real-time whale movement detection
for enhanced trading decisions with 32-bit thermal state integration.:

Mathematical Foundation:
- Whale Impact Vector: W(t) = sumᵢ whale_sizeᵢ * velocity_impactᵢ * thermal_multiplierᵢ
- Thermal Whale Scaling: TWS = whale_movement * thermal_state_multiplier
- Profit Flip Logic: PF = (whale_direction ⊕ thermal_state) -> {enter, exit, hold}
- BTC Correlation: BC = corr(whale_movements, btc_price_changes)
""""""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiohttp
import numpy as np

# Import core systems for thermal integration
        try:
    from core.dual_unicore_handler import DualUnicoreHandler
    from core.phase_bit_integration import BitPhase, PhaseBitIntegration
    from core.unified_math_system import UnifiedMathSystem
    CORE_SYSTEMS_AVAILABLE = True
        except ImportError as e:
        logging.warning(f"Core systems not available: {e}")
        CORE_SYSTEMS_AVAILABLE = False

        logger = logging.getLogger(__name__)

# Thermal state constants for whale tracking
        COOL = "cool"        # Low thermal state (4-bit operations)
        WARM = "warm"        # Mid thermal state (8-bit operations)
        HOT = "hot"          # High thermal state (32-bit operations)
        CRITICAL = "critical"  # Extreme thermal state (42-bit operations)


class WhaleSize(Enum):
"""Whale classification by transaction size."""

SHRIMP = "shrimp"      # < 1 BTC
CRAB = "crab"          # 1-10 BTC
OCTOPUS = "octopus"    # 10-50 BTC
FISH = "fish"          # 50-100 BTC
DOLPHIN = "dolphin"    # 100-500 BTC
SHARK = "shark"        # 500-1000 BTC
WHALE = "whale"        # 1000-5000 BTC
HUMPBACK = "humpback"  # > 5000 BTC


class WhaleMovementType(Enum):
"""Types of whale movements detected."""

ACCUMULATION = "accumulation"      # Large inflows to wallets
DISTRIBUTION = "distribution"      # Large outflows from wallets
EXCHANGE_INFLOW = "exchange_inflow"    # Moving to exchanges (sell pressure)
EXCHANGE_OUTFLOW = "exchange_outflow"  # Moving from exchanges (hodl)
INTER_EXCHANGE = "inter_exchange"      # Between exchanges
UNKNOWN_WALLET = "unknown_wallet"     # Unknown wallet activity


class ThermalWhaleAction(Enum):
"""Trading actions based on thermal whale analysis."""

AGGRESSIVE_BUY = "aggressive_buy"      # Strong buy signal
MODERATE_BUY = "moderate_buy"          # Moderate buy signal
HOLD = "hold"                          # Hold current position
MODERATE_SELL = "moderate_sell"        # Moderate sell signal
AGGRESSIVE_SELL = "aggressive_sell"    # Strong sell signal
    WAIT_CONFIRMATION = "wait_confirmation"  # Wait for confirmation


@dataclass
class WhaleTransaction:
"""Represents a detected whale transaction."""

transaction_hash: str
from_address: str
to_address: str
amount_btc: float
amount_usd: float
whale_size: WhaleSize
movement_type: WhaleMovementType
timestamp: datetime
block_height: int
confirmation_count: int
thermal_state: str = WARM
impact_score: float = 0.0

    def __post_init__(self):
    """Calculate impact score based on amount and thermal state."""
    self.impact_score = self._calculate_impact_score()

    def _calculate_impact_score(self) -> float:
        """Calculate whale transaction impact score with thermal weighting."""
    base_impact = np.log1p(self.amount_btc) * 0.1

    # Thermal multiplier
    thermal_multipliers = {)}
        COOL: 0.8,
            WARM: 1.0,
                HOT: 1.3,      # 32-bit enhanced impact
        CRITICAL: 1.6
}
    thermal_mult = thermal_multipliers.get(self.thermal_state, 1.0)
    return min(base_impact * thermal_mult, 1.0)


@dataclass
class WhaleAlert:
    """Whale movement alert with thermal analysis."""

alert_id: str
whale_transactions: List[WhaleTransaction]
total_volume_btc: float
net_flow_direction: str  # "inflow", "outflow", "neutral"
alert_level: str  # "low", "medium", "high", "critical"
thermal_recommendation: ThermalWhaleAction
confidence_score: float
timestamp: datetime
metadata: Dict[str, Any] = field(default_factory = dict)


class WhaleTrackerIntegration:
""""""
Comprehensive whale tracking integration system.

Connects to multiple whale tracking APIs and provides thermal-enhanced
    analysis for trading decision making with 32-bit thermal state integration.
""""""

    def __init__(self):
    """Initialize whale tracker integration."""
        self.math_system = UnifiedMathSystem() if CORE_SYSTEMS_AVAILABLE else None
        self.phase_integration = PhaseBitIntegration() if CORE_SYSTEMS_AVAILABLE else None
        self.unicore = DualUnicoreHandler() if CORE_SYSTEMS_AVAILABLE else None

    # Whale tracking configuration
    self.api_endpoints = {)}
        "whale_alert": "https://api.whale-alert.io/v1",
            "bitinfocharts": "https://bitinfocharts.com/api",
                "blockchain_info": "https://blockchain.info/api",
                "glassnode": "https://api.glassnode.com/v1/metrics",
                "santiment": "https://api.santiment.net/graphql"
}
    # API keys (should be in environment variables)
    self.api_keys = {)}
            "whale_alert": "demo-api-key",  # Replace with real key
            "glassnode": "demo-api-key",    # Replace with real key
            "santiment": "demo-api-key"     # Replace with real key
}
    # Thermal state management
    self.current_thermal_state = WARM
    self.thermal_history = []

    # Whale tracking state
    self.whale_transactions = []
    self.whale_alerts = []
    self.whale_statistics = {)}
        "total_transactions_24h": 0,
            "total_volume_24h": 0.0,
                "largest_transaction": 0.0,
                "whale_accumulation_score": 0.0
}
        # Session for HTTP requests
    self.session = None

    logger.info("🐋 Whale Tracker Integration initialized")

    async def start_whale_monitoring(self) -> None:
    """Start continuous whale monitoring."""
    self.session = aiohttp.ClientSession()

        try:
        logger.info("🐋 Starting whale monitoring...")

        # Start monitoring tasks
        tasks = [)]
            self._monitor_whale_alert_api(),
                self._monitor_blockchain_movements(),
                    self._analyze_thermal_whale_correlation(),
                    self._update_whale_statistics()
]
        await asyncio.gather(*tasks)

        except Exception as e:
        logger.error(f"Whale monitoring error: {e}")
                finally:
                    if self.session:
                await self.session.close()

                        async def _monitor_whale_alert_api(self) -> None:
                        """Monitor Whale Alert API for large transactions."""
                            while True:
                                try:
                            url = f"{self.api_endpoints['whale_alert']}/transactions"
                            params = {)}
                            "api_key": self.api_keys["whale_alert"],
                                "currency": "btc",
                                    "limit": 100
}
                                    async with self.session.get(url, params = params) as response:
                                        if response.status == 200:
                                    data = await response.json()
                                    await self._process_whale_alert_data(data)
                                            else:
                                        logger.warning(f"Whale Alert API error: {response.status}")

                                        await asyncio.sleep(60)  # Check every minute

                                            except Exception as e:
                                            logger.error(f"Whale Alert monitoring error: {e}")
                                            await asyncio.sleep(300)  # Wait 5 minutes on error

                                                    async def _process_whale_alert_data(self, data: Dict[str, Any]) -> None:
                                                """Process whale alert data and create whale transactions."""
                                                        try:
                                                            if "result" not in data:
                                                    return

                                                            for transaction_data in data["result"]:
                                                        whale_tx = self._create_whale_transaction(transaction_data)
                                                                if whale_tx:
                                                            # Apply thermal analysis
                                                            whale_tx.thermal_state = self.current_thermal_state

                                                            self.whale_transactions.append(whale_tx)

                                                                # Check if this creates an alert
                                                            alert = await self._evaluate_whale_alert([whale_tx])
                                                                    if alert:
                                                                self.whale_alerts.append(alert)
                                                                logger.info(f"🐋 Whale Alert: {alert.alert_level} - {alert.thermal_recommendation.value}")

                                                                # Keep only recent transactions (last 24 hours)
                                                                cutoff_time = datetime.now() - timedelta(hours=24)
                                                                self.whale_transactions = [)]
                                                                    tx for tx in self.whale_transactions
                                                                    if tx.timestamp > cutoff_time:
]
                                                                    except Exception as e:
                                                                    logger.error(f"Processing whale data error: {e}")

    def _create_whale_transaction(self, data: Dict[str, Any]) -> Optional[WhaleTransaction]:
    """Create whale transaction from API data."""
        try:
        # Extract transaction details
        amount_btc = float(data.get("amount", 0))
        amount_usd = float(data.get("amount_usd", 0))

        # Skip small transactions
            if amount_btc < 10:  # Less than 10 BTC:
            return None

        # Determine whale size
        whale_size = self._classify_whale_size(amount_btc)

        # Determine movement type
        movement_type = self._classify_movement_type(data)

        return WhaleTransaction()
            transaction_hash = data.get("hash", ""),
                from_address = data.get("from", {}).get("address", ""),
                    to_address = data.get("to", {}).get("address", ""),
                    amount_btc = amount_btc,
                    amount_usd = amount_usd,
                    whale_size = whale_size,
                    movement_type = movement_type,
                    timestamp = datetime.fromtimestamp(data.get("timestamp", time.time())),
                    block_height = data.get("block_height", 0),
                    confirmation_count = data.get("confirmations", 0)
        )

    except Exception as e:
        logger.error(f"Creating whale transaction error: {e}")
        return None

    def _classify_whale_size(self, amount_btc: float) -> WhaleSize:
    """Classify whale size based on BTC amount."""
        if amount_btc < 1:
        return WhaleSize.SHRIMP
        elif amount_btc < 10:
        return WhaleSize.CRAB
        elif amount_btc < 50:
        return WhaleSize.OCTOPUS
        elif amount_btc < 100:
        return WhaleSize.FISH
        elif amount_btc < 500:
        return WhaleSize.DOLPHIN
        elif amount_btc < 1000:
        return WhaleSize.SHARK
        elif amount_btc < 5000:
        return WhaleSize.WHALE
        else:
        return WhaleSize.HUMPBACK

    def _classify_movement_type(self, data: Dict[str, Any]) -> WhaleMovementType:
    """Classify the type of whale movement."""
    from_owner = data.get("from", {}).get("owner_type", "")
    to_owner = data.get("to", {}).get("owner_type", "")

    # Exchange to wallet (potential hodling)
        if from_owner == "exchange" and to_owner == "wallet":
        return WhaleMovementType.EXCHANGE_OUTFLOW

    # Wallet to exchange (potential selling)
        elif from_owner == "wallet" and to_owner == "exchange":
        return WhaleMovementType.EXCHANGE_INFLOW

    # Between exchanges
        elif from_owner == "exchange" and to_owner == "exchange":
        return WhaleMovementType.INTER_EXCHANGE

    # Wallet accumulation
        elif from_owner == "wallet" and to_owner == "wallet":
        return WhaleMovementType.ACCUMULATION

        else:
        return WhaleMovementType.UNKNOWN_WALLET

    async def _evaluate_whale_alert(self, transactions: List[WhaleTransaction]) -> Optional[WhaleAlert]:
        """Evaluate if whale transactions warrant an alert."""
        try:
            if not transactions:
            return None

        # Calculate total volume
            total_volume = sum(tx.amount_btc for tx in transactions)

        # Determine net flow direction
        exchange_inflow = sum()
                tx.amount_btc for tx in transactions
                if tx.movement_type == WhaleMovementType.EXCHANGE_INFLOW:
        )
        exchange_outflow = sum()
                tx.amount_btc for tx in transactions
                if tx.movement_type == WhaleMovementType.EXCHANGE_OUTFLOW:
        )

        net_flow = exchange_outflow - exchange_inflow
            if net_flow > 100:
            flow_direction = "outflow"
                elif net_flow < -100:
            flow_direction = "inflow"
                    else:
                flow_direction = "neutral"

                # Determine alert level
                alert_level = self._calculate_alert_level(total_volume, transactions)

                # Generate thermal recommendation
                thermal_recommendation = self._generate_thermal_recommendation()
                flow_direction, alert_level, transactions
                )

                # Calculate confidence
                confidence = self._calculate_whale_confidence(transactions)

                        if alert_level in ["medium", "high", "critical"]:
                    alert = WhaleAlert()
                    alert_id = hashlib.sha256(f"{time.time()}_{total_volume}".encode()).hexdigest()[:12],
                        whale_transactions = transactions,
                            total_volume_btc = total_volume,
                            net_flow_direction = flow_direction,
                            alert_level = alert_level,
                            thermal_recommendation = thermal_recommendation,
                            confidence_score = confidence,
                            timestamp = datetime.now())

                return alert

            return None

                except Exception as e:
                logger.error(f"Whale alert evaluation error: {e}")
            return None

    def _calculate_alert_level(self, total_volume: float, transactions: List[WhaleTransaction]) -> str:
    """Calculate whale alert level based on volume and thermal state."""
    base_threshold = 500  # Base BTC threshold

        # Thermal multipliers for sensitivity
    thermal_multipliers = {)}
        COOL: 1.2,    # More conservative in cool state
        WARM: 1.0,    # Standard sensitivity
        HOT: 0.8,     # More sensitive in hot state (32-bit)
        CRITICAL: 0.6  # Very sensitive in critical state
}
    threshold_multiplier = thermal_multipliers.get(self.current_thermal_state, 1.0)
    adjusted_threshold = base_threshold * threshold_multiplier

        # Check for large whale presence
        has_humpback = any(tx.whale_size == WhaleSize.HUMPBACK for tx in transactions)
        has_multiple_whales = len([tx for tx in transactions if tx.whale_size in [WhaleSize.WHALE, WhaleSize.HUMPBACK]]) > 1

        if total_volume > adjusted_threshold * 10 or has_humpback:
        return "critical"
        elif total_volume > adjusted_threshold * 4 or has_multiple_whales:
        return "high"
        elif total_volume > adjusted_threshold:
        return "medium"
        else:
        return "low"

    def _generate_thermal_recommendation(self, flow_direction: str, alert_level: str, ):
                                transactions: List[WhaleTransaction]) -> ThermalWhaleAction:
    """Generate trading recommendation based on thermal whale analysis."""
        try:
        # Base recommendation logic
            if flow_direction == "outflow" and alert_level in ["high", "critical"]:
            # Whales moving to cold storage - bullish
            base_action = ThermalWhaleAction.MODERATE_BUY
                elif flow_direction == "inflow" and alert_level in ["high", "critical"]:
            # Whales moving to exchanges - bearish
            base_action = ThermalWhaleAction.MODERATE_SELL
                    else:
                base_action = ThermalWhaleAction.HOLD

                # Thermal state adjustment
                    if self.current_thermal_state == HOT:  # 32-bit enhanced decision making:
                        if base_action == ThermalWhaleAction.MODERATE_BUY:
                return ThermalWhaleAction.AGGRESSIVE_BUY
                        elif base_action == ThermalWhaleAction.MODERATE_SELL:
                return ThermalWhaleAction.AGGRESSIVE_SELL
                        else:
                return base_action

                        elif self.current_thermal_state == CRITICAL:
                    # More aggressive in critical state
                            if base_action == ThermalWhaleAction.MODERATE_BUY:
                    return ThermalWhaleAction.AGGRESSIVE_BUY
                            elif base_action == ThermalWhaleAction.MODERATE_SELL:
                    return ThermalWhaleAction.AGGRESSIVE_SELL
                            else:
                    return ThermalWhaleAction.WAIT_CONFIRMATION

                            elif self.current_thermal_state == COOL:
                        # More conservative in cool state
                                if base_action in [ThermalWhaleAction.MODERATE_BUY, ThermalWhaleAction.MODERATE_SELL]:
                        return ThermalWhaleAction.WAIT_CONFIRMATION
                                else:
                        return base_action

                    return base_action

                        except Exception as e:
                        logger.error(f"Thermal recommendation error: {e}")
                    return ThermalWhaleAction.HOLD

    def _calculate_whale_confidence(self, transactions: List[WhaleTransaction]) -> float:
        """Calculate confidence score for whale analysis."""
        try:
            if not transactions:
            return 0.0

        # Base confidence from transaction count and volume
        tx_count_factor = min(len(transactions) / 10.0, 1.0)
            volume_factor = min(sum(tx.amount_btc for tx in transactions) / 1000.0, 1.0)

        # Whale size diversity factor
            unique_sizes = len(set(tx.whale_size for tx in transactions))
        diversity_factor = min(unique_sizes / 3.0, 1.0)

        # Thermal state confidence adjustment
        thermal_confidence = {)}
            COOL: 0.7,
                WARM: 0.8,
                    HOT: 0.9,     # Higher confidence in 32-bit state
            CRITICAL: 0.95
        }.get(self.current_thermal_state, 0.8)

        confidence = (tx_count_factor + volume_factor + diversity_factor) / 3.0
        return min(confidence * thermal_confidence, 1.0)

    except Exception as e:
        logger.error(f"Confidence calculation error: {e}")
        return 0.5

    async def _monitor_blockchain_movements(self) -> None:
        """Monitor direct blockchain for large movements."""
        while True:
            try:
            # This would connect to blockchain.info or similar API
                # for direct blockchain monitoring
            await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
            logger.error(f"Blockchain monitoring error: {e}")
            await asyncio.sleep(600)  # Wait 10 minutes on error

                    async def _analyze_thermal_whale_correlation(self) -> None:
                """Analyze correlation between thermal states and whale activity."""
                        while True:
                            try:
                        # Update thermal state based on whale activity
                        recent_alerts = [)]
                            alert for alert in self.whale_alerts
                            if alert.timestamp > datetime.now() - timedelta(hours=1):
]
                                if recent_alerts:
                            # High whale activity increases thermal state
                                critical_alerts = [a for a in recent_alerts if a.alert_level == "critical"]
                                high_alerts = [a for a in recent_alerts if a.alert_level == "high"]

                                    if critical_alerts:
                                self.current_thermal_state = CRITICAL
                                        elif len(high_alerts) > 2:
                                    self.current_thermal_state = HOT  # 32-bit enhanced state
                                            elif recent_alerts:
                                        self.current_thermal_state = WARM
                                                else:
                                            self.current_thermal_state = COOL

                                            # Store thermal history
                                            self.thermal_history.append({))}
                                            "timestamp": datetime.now(),
                                                "thermal_state": self.current_thermal_state,
                                                    "whale_activity_level": len(recent_alerts)
                                            })

                                            # Keep only last 24 hours
                                            cutoff = datetime.now() - timedelta(hours=24)
                                            self.thermal_history = [)]
                                                h for h in self.thermal_history
                                                if h["timestamp"] > cutoff:
]
                                            await asyncio.sleep(600)  # Update every 10 minutes

                                                except Exception as e:
                                                logger.error(f"Thermal correlation analysis error: {e}")
                                                await asyncio.sleep(1200)  # Wait 20 minutes on error

                                                        async def _update_whale_statistics(self) -> None:
                                                    """Update whale tracking statistics."""
                                                            while True:
                                                                try:
                                                            # Calculate 24h statistics
                                                            cutoff_time = datetime.now() - timedelta(hours=24)
                                                            recent_transactions = [)]
                                                                tx for tx in self.whale_transactions
                                                                if tx.timestamp > cutoff_time:
]
                                                            self.whale_statistics = {)}
                                                            "total_transactions_24h": len(recent_transactions),
                                                                "total_volume_24h": sum(tx.amount_btc for tx in recent_transactions),
                                                                    "largest_transaction": max()
                                                                (tx.amount_btc for tx in recent_transactions), default=0.0
                                                            ),
                                                                "whale_accumulation_score": self._calculate_accumulation_score(recent_transactions),
                                                                    "thermal_state": self.current_thermal_state,
                                                                    "current_alert_level": self._get_current_alert_level()
}
                                                            await asyncio.sleep(1800)  # Update every 30 minutes

                                                                except Exception as e:
                                                                logger.error(f"Statistics update error: {e}")
                                                                await asyncio.sleep(3600)  # Wait 1 hour on error

    def _calculate_accumulation_score(self, transactions: List[WhaleTransaction]) -> float:
    """Calculate whale accumulation score."""
        try:
            if not transactions:
            return 0.0

        # Score based on outflow vs inflow
        outflow_volume = sum()
                tx.amount_btc for tx in transactions
                if tx.movement_type == WhaleMovementType.EXCHANGE_OUTFLOW:
        )
        inflow_volume = sum()
                tx.amount_btc for tx in transactions
                if tx.movement_type == WhaleMovementType.EXCHANGE_INFLOW:
        )

        total_volume = outflow_volume + inflow_volume
            if total_volume == 0:
            return 0.0

        # Positive score = accumulation, negative = distribution
        net_accumulation = (outflow_volume - inflow_volume) / total_volume
        return max(-1.0, min(1.0, net_accumulation))

    except Exception as e:
        logger.error(f"Accumulation score error: {e}")
        return 0.0

    def _get_current_alert_level(self) -> str:
    """Get current whale alert level."""
    recent_alerts = [)]
            alert for alert in self.whale_alerts
            if alert.timestamp > datetime.now() - timedelta(hours=1):
]
        if not recent_alerts:
        return "none"

    # Return highest alert level from recent alerts
    levels = ["low", "medium", "high", "critical"]
        max_level = max(alert.alert_level for alert in recent_alerts)
    return max_level

    def get_whale_summary(self) -> Dict[str, Any]:
    """Get comprehensive whale tracking summary."""
    return {)}
        "statistics": self.whale_statistics,
            "recent_alerts": [)]
            {)}
                "alert_id": alert.alert_id,
                    "alert_level": alert.alert_level,
                        "volume_btc": alert.total_volume_btc,
                        "flow_direction": alert.net_flow_direction,
                        "thermal_recommendation": alert.thermal_recommendation.value,
                        "confidence": alert.confidence_score,
                        "timestamp": alert.timestamp.isoformat()
}
                for alert in self.whale_alerts[-10:]  # Last 10 alerts:
        ],
            "thermal_state": self.current_thermal_state,
                "whale_count_by_size": self._get_whale_count_by_size(),
                "thermal_history": self.thermal_history[-24:]  # Last 24 data points
}
    def _get_whale_count_by_size(self) -> Dict[str, int]:
    """Get count of whales by size in last 24 hours."""
    cutoff_time = datetime.now() - timedelta(hours=24)
    recent_transactions = [)]
            tx for tx in self.whale_transactions
            if tx.timestamp > cutoff_time:
]
    counts = {}
        for size in WhaleSize:
        counts[size.value] = len([))]
                tx for tx in recent_transactions
                if tx.whale_size == size:
        ])

    return counts


# Global instance for easy access
whale_tracker = WhaleTrackerIntegration()


# Example usage and testing
    if __name__ == "__main__":
print("🐋 Whale Tracker Integration System")
print("=" * 50)

        async def demo_whale_tracking():
    """Demonstrate whale tracking functionality."""
    tracker = WhaleTrackerIntegration()

        # Simulate whale data for testing
    test_whale_data = {)}
        "hash": "test_hash_12345",
            "from": {"address": "test_from_address", "owner_type": "wallet"},
                "to": {"address": "test_to_address", "owner_type": "exchange"},
                "amount": 1500.0,  # 1500 BTC - large whale
        "amount_usd": 75000000.0,  # 75M USD
        "timestamp": time.time(),
            "block_height": 800000,
                "confirmations": 6
}
    # Process test data
    whale_tx = tracker._create_whale_transaction(test_whale_data)
            if whale_tx:
        print(f"🐋 Created whale transaction:")
        print(f"  Amount: {whale_tx.amount_btc} BTC (${whale_tx.amount_usd:,.0f})")
        print(f"  Whale Size: {whale_tx.whale_size.value}")
        print(f"  Movement: {whale_tx.movement_type.value}")
        print(f"  Impact Score: {whale_tx.impact_score:.3f}")
        print(f"  Thermal State: {whale_tx.thermal_state}")

        # Test alert evaluation
        tracker.whale_transactions.append(whale_tx)
        alert = await tracker._evaluate_whale_alert([whale_tx])

                if alert:
            print(f"\n🚨 Whale Alert Generated:")
            print(f"  Alert Level: {alert.alert_level}")
            print(f"  Flow Direction: {alert.net_flow_direction}")
            print(f"  Thermal Recommendation: {alert.thermal_recommendation.value}")
            print(f"  Confidence: {alert.confidence_score:.3f}")

            # Get summary
            summary = tracker.get_whale_summary()
            print(f"\n📊 Whale Summary:")
            print(f"  Current Thermal State: {summary['thermal_state']}")
            print(f"  Whale Statistics: {summary['statistics']}")

            # Run demo
            import asyncio
            asyncio.run(demo_whale_tracking())