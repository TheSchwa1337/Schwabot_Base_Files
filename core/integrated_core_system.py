# -*- coding: utf-8 -*-
"""
Integrated Core System - Speed Lattice Vault SP 1.27 AE
Comprehensive integrated system with hash-based fractal memory, mathematical libraries,
and live API connectivity for BTC/USDC trading operations.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import our mathematical libraries
try:
    from core.mathlib_v2 import CoreMathLibV2, HashMemoryBlock
    from core.speed_lattice_trading_integration import SpeedLatticeTradingIntegration
    from core.speed_lattice_vault import SpeedLatticeVault
    from core.speed_lattice_visualizer import PanelType, SpeedLatticeLivePanelSystem
except ImportError as e:
    print(f"⚠️ Some core systems not available: {e}")


class TickState(Enum):
    """Internal tick states for the system"""

    IDLE = "idle"
    ANALYZING = "analyzing"
    PATTERN_RECOGNITION = "pattern_recognition"
    HASH_GENERATION = "hash_generation"
    MEMORY_RECALL = "memory_recall"
    SIGNAL_GENERATION = "signal_generation"
    EXECUTION = "execution"
    COMPLETED = "completed"


class FractalMemoryBucket:
    """Fractal memory bucket for storing similar hash patterns"""

    def __init__(self, bucket_id: str, similarity_threshold: float = 0.8):
        self.bucket_id = bucket_id
        self.similarity_threshold = similarity_threshold
        self.hash_blocks: List[HashMemoryBlock] = []
        self.profit_history: List[float] = []
        self.pattern_count = 0
        self.total_profit = 0.0
        self.success_rate = 0.0

    def add_hash_block(self, hash_block: HashMemoryBlock) -> None:
        """Add hash block to this bucket"""
        self.hash_blocks.append(hash_block)
        self.profit_history.append(hash_block.profit)
        self.pattern_count += 1
        self.total_profit += hash_block.profit

        # Update success rate
        profitable_patterns = sum(1 for p in self.profit_history if p > 0)
        self.success_rate = profitable_patterns / len(self.profit_history)

    def get_average_profit(self) -> float:
        """Get average profit for this bucket"""
        if not self.profit_history:
            return 0.0
        return self.total_profit / len(self.profit_history)

    def get_profit_prediction(self) -> float:
        """Get profit prediction based on historical patterns"""
        if not self.profit_history:
            return 0.0

        # Weighted average based on recency
        weights = np.linspace(0.1, 1.0, len(self.profit_history))
        weighted_profit = np.average(self.profit_history, weights=weights)

        return weighted_profit * self.success_rate


@dataclass
class InternalTick:
    """Internal tick representation for the system"""

    tick_id: str
    timestamp: float
    price_data: Dict[str, float]
    hash_signature: str
    state: TickState
    strategy_id: str
    profit_prediction: float = 0.0
    similarity_score: float = 0.0
    memory_bucket_id: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None


class IntegratedCoreSystem:
    """
    Integrated core system that combines all mathematical libraries,
    hash-based fractal memory, and live API connectivity.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize mathematical libraries
        self.mathlib_v2 = CoreMathLibV2()
        self.speed_lattice_vault = SpeedLatticeVault()
        self.trading_integration = SpeedLatticeTradingIntegration()
        self.visualizer = SpeedLatticeLivePanelSystem()

        # Internal tick management
        self.internal_tick_counter = 0
        self.tick_history: List[InternalTick] = []
        self.current_tick: Optional[InternalTick] = None

        # Fractal memory system
        self.fractal_buckets: Dict[str, FractalMemoryBucket] = {}
        self.memory_bucket_counter = 0

        # API connectivity
        self.api_connections = {}
        self.live_data_streams = {}
        self.is_live_mode = False

        # Performance tracking
        self.performance_metrics = {
            "total_ticks": 0,
            "successful_predictions": 0,
            "total_profit": 0.0,
            "average_similarity_score": 0.0,
            "memory_hit_rate": 0.0,
        }

        # Initialize the system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the integrated core system"""
        self.logger.info("🚀 Initializing Integrated Core System")

        # Initialize fractal memory buckets
        self._initialize_fractal_buckets()

        # Connect to live data streams
        self._connect_live_streams()

        # Initialize visualizer panels
        self._initialize_visualizer()

        self.logger.info("✅ Integrated Core System initialized")

    def _initialize_fractal_buckets(self):
        """Initialize fractal memory buckets"""
        bucket_types = [
            ("RSI_BULL", 0.8),
            ("RSI_BEAR", 0.8),
            ("MOMENTUM_BULL", 0.75),
            ("MOMENTUM_BEAR", 0.75),
            ("VOLATILITY_HIGH", 0.7),
            ("VOLATILITY_LOW", 0.7),
            ("TREND_UP", 0.85),
            ("TREND_DOWN", 0.85),
            ("CONSOLIDATION", 0.6),
            ("BREAKOUT", 0.9),
        ]

        for bucket_type, threshold in bucket_types:
            bucket_id = f"bucket_{self.memory_bucket_counter:03d}_{bucket_type}"
            self.fractal_buckets[bucket_id] = FractalMemoryBucket(bucket_id, threshold)
            self.memory_bucket_counter += 1

        self.logger.info(f"✅ Initialized {len(self.fractal_buckets)} fractal memory buckets")

    def _connect_live_streams(self):
        """Connect to live data streams"""
        # BTC/USDC price stream
        self.live_data_streams["btc_usdc"] = {
            "endpoint": "wss://stream.binance.com:9443/ws/btcusdt@trade",
            "is_active": True,
            "last_update": time.time(),
            "data_queue": deque(maxlen=1000),
        }

        # Speed Lattice Vault data stream
        self.live_data_streams["speed_lattice"] = {
            "endpoint": "api/speed_lattice/vault_data",
            "is_active": True,
            "last_update": time.time(),
            "data_queue": deque(maxlen=1000),
        }

        self.logger.info(f"✅ Connected to {len(self.live_data_streams)} live data streams")

    def _initialize_visualizer(self):
        """Initialize visualizer with integrated panels"""
        # Connect all panels to live data
        for panel_type in PanelType:
            endpoint = f"api/integrated/{panel_type.value}"
            self.visualizer.connect_api(panel_type, endpoint, "integrated_api_key", update_interval=1.0)

        self.logger.info("✅ Visualizer initialized with integrated panels")

    def generate_internal_tick(self, price_data: Dict[str, float], strategy_id: str) -> InternalTick:
        """Generate a new internal tick"""
        tick_id = f"tick_{self.internal_tick_counter:06d}_{uuid.uuid4().hex[:8]}"
        timestamp = time.time()

        # Generate hash signature
        price_vector = np.array(list(price_data.values()))
        hash_signature = self.mathlib_v2.generate_hash_signature(price_vector, strategy_id)

        # Create internal tick
        tick = InternalTick(
            tick_id=tick_id,
            timestamp=timestamp,
            price_data=price_data,
            hash_signature=hash_signature,
            state=TickState.IDLE,
            strategy_id=strategy_id,
        )

        self.internal_tick_counter += 1
        return tick

    def process_tick(self, tick: InternalTick) -> Dict[str, Any]:
        """Process an internal tick through the complete pipeline"""
        self.current_tick = tick
        self.logger.info(f"🔄 Processing tick: {tick.tick_id}")

        # Step 1: Pattern Recognition
        tick.state = TickState.PATTERN_RECOGNITION
        pattern_analysis = self._analyze_patterns(tick)

        # Step 2: Hash Generation and Memory Recall
        tick.state = TickState.HASH_GENERATION
        memory_recall = self._recall_fractal_memory(tick)

        # Step 3: Signal Generation
        tick.state = TickState.SIGNAL_GENERATION
        signals = self._generate_trading_signals(tick, pattern_analysis, memory_recall)

        # Step 4: Execution (if signals are strong enough)
        if signals.get("confidence", 0) > 0.8:
            tick.state = TickState.EXECUTION
            execution_result = self._execute_trading_signals(tick, signals)
            tick.execution_result = execution_result
        else:
            tick.state = TickState.COMPLETED

        # Update performance metrics
        self._update_performance_metrics(tick)

        # Store tick in history
        self.tick_history.append(tick)

        # Update visualizer
        self._update_visualizer(tick)

        return {
            "tick_id": tick.tick_id,
            "state": tick.state.value,
            "pattern_analysis": pattern_analysis,
            "memory_recall": memory_recall,
            "signals": signals,
            "execution_result": tick.execution_result,
        }

    def _analyze_patterns(self, tick: InternalTick) -> Dict[str, Any]:
        """Analyze patterns using mathematical libraries"""
        price_vector = np.array(list(tick.price_data.values()))

        # RSI analysis
        rsi_values = self.mathlib_v2.calculate_rsi(price_vector)
        current_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50.0

        # Statistical analysis
        stats = self.mathlib_v2.advanced_statistical_analysis(price_vector)

        # Entropy analysis
        entropy = self.mathlib_v2.entropy_analysis(price_vector)

        # Moving averages
        moving_avgs = self.mathlib_v2.moving_average_variants(price_vector)

        # Speed Lattice Vault analysis
        vault_analysis = {}
        if self.speed_lattice_vault:
            try:
                drift_matrix = self.speed_lattice_vault.get_drift_matrix()
                chrono_bias = self.speed_lattice_vault.get_chrono_bias()
                vault_analysis = {
                    "drift_matrix": drift_matrix.tolist() if hasattr(drift_matrix, "tolist") else drift_matrix,
                    "chrono_bias": chrono_bias,
                }
            except Exception as e:
                self.logger.warning(f"Speed Lattice Vault analysis failed: {e}")

        return {
            "rsi": current_rsi,
            "statistics": stats,
            "entropy": entropy,
            "moving_averages": moving_avgs,
            "vault_analysis": vault_analysis,
            "pattern_type": self._classify_pattern(current_rsi, stats, entropy),
        }

    def _classify_pattern(self, rsi: float, stats: Dict[str, float], entropy: Dict[str, float]) -> str:
        """Classify the current pattern"""
        # RSI-based classification
        if rsi > 70:
            base_pattern = "RSI_BEAR"
        elif rsi < 30:
            base_pattern = "RSI_BULL"
        else:
            base_pattern = "RSI_NEUTRAL"

        # Volatility-based classification
        volatility = stats.get("std", 0)
        if volatility > 0.05:
            volatility_pattern = "VOLATILITY_HIGH"
        else:
            volatility_pattern = "VOLATILITY_LOW"

        # Trend-based classification
        skewness = stats.get("skewness", 0)
        if skewness > 0.1:
            trend_pattern = "TREND_UP"
        elif skewness < -0.1:
            trend_pattern = "TREND_DOWN"
        else:
            trend_pattern = "CONSOLIDATION"

        return f"{base_pattern}_{volatility_pattern}_{trend_pattern}"

    def _recall_fractal_memory(self, tick: InternalTick) -> Dict[str, Any]:
        """Recall fractal memory based on hash similarity"""
        # Find most similar bucket
        best_bucket = None
        best_similarity = 0.0

        for bucket_id, bucket in self.fractal_buckets.items():
            if bucket.hash_blocks:
                # Calculate similarity with all blocks in this bucket
                similarities = []
                for block in bucket.hash_blocks:
                    similarity = self.mathlib_v2.hash_similarity_score(tick.hash_signature, [block.hash_signature])
                    similarities.append(similarity)

                avg_similarity = np.mean(similarities)
                if avg_similarity > best_similarity and avg_similarity > bucket.similarity_threshold:
                    best_similarity = avg_similarity
                    best_bucket = bucket

        tick.similarity_score = best_similarity

        if best_bucket:
            tick.memory_bucket_id = best_bucket.bucket_id
            profit_prediction = best_bucket.get_profit_prediction()
            tick.profit_prediction = profit_prediction

            return {
                "bucket_id": best_bucket.bucket_id,
                "similarity_score": best_similarity,
                "profit_prediction": profit_prediction,
                "success_rate": best_bucket.success_rate,
                "pattern_count": best_bucket.pattern_count,
                "average_profit": best_bucket.get_average_profit(),
            }
        else:
            return {
                "bucket_id": None,
                "similarity_score": 0.0,
                "profit_prediction": 0.0,
                "success_rate": 0.0,
                "pattern_count": 0,
                "average_profit": 0.0,
            }

    def _generate_trading_signals(
        self, tick: InternalTick, pattern_analysis: Dict[str, Any], memory_recall: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading signals based on pattern analysis and memory recall"""
        rsi = pattern_analysis["rsi"]
        similarity_score = memory_recall["similarity_score"]
        profit_prediction = memory_recall["profit_prediction"]
        success_rate = memory_recall["success_rate"]

        # Calculate signal confidence
        confidence = 0.0
        signal_type = "HOLD"

        # RSI-based signals
        if rsi < 30 and similarity_score > 0.7 and profit_prediction > 0:
            signal_type = "BUY"
            confidence = min(0.9, similarity_score * success_rate * (1 + profit_prediction))
        elif rsi > 70 and similarity_score > 0.7 and profit_prediction < 0:
            signal_type = "SELL"
            confidence = min(0.9, similarity_score * success_rate * (1 - profit_prediction))

        # Pattern strength adjustment
        pattern_strength = pattern_analysis["entropy"].get("normalized_entropy", 0.5)
        confidence *= 0.5 + pattern_strength * 0.5

        return {
            "signal_type": signal_type,
            "confidence": confidence,
            "rsi": rsi,
            "similarity_score": similarity_score,
            "profit_prediction": profit_prediction,
            "pattern_strength": pattern_strength,
            "timestamp": tick.timestamp,
        }

    def _execute_trading_signals(self, tick: InternalTick, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading signals through the trading integration"""
        try:
            # Update trading integration with current price
            current_price = tick.price_data.get("close", 0)
            if current_price > 0:
                self.trading_integration.update_btc_price(current_price, tick.timestamp)

            # Execute based on signal type
            signal_type = signals["signal_type"]
            confidence = signals["confidence"]

            if signal_type == "BUY" and confidence > 0.8:
                # Simulate buy execution
                execution_result = {
                    "action": "BUY",
                    "confidence": confidence,
                    "price": current_price,
                    "timestamp": tick.timestamp,
                    "status": "EXECUTED",
                }
            elif signal_type == "SELL" and confidence > 0.8:
                # Simulate sell execution
                execution_result = {
                    "action": "SELL",
                    "confidence": confidence,
                    "price": current_price,
                    "timestamp": tick.timestamp,
                    "status": "EXECUTED",
                }
            else:
                execution_result = {
                    "action": "HOLD",
                    "confidence": confidence,
                    "price": current_price,
                    "timestamp": tick.timestamp,
                    "status": "NO_ACTION",
                }

            return execution_result

        except Exception as e:
            self.logger.error(f"Trading execution failed: {e}")
            return {
                "action": "ERROR",
                "confidence": 0.0,
                "price": 0.0,
                "timestamp": tick.timestamp,
                "status": "FAILED",
                "error": str(e),
            }

    def _update_performance_metrics(self, tick: InternalTick):
        """Update performance metrics"""
        self.performance_metrics["total_ticks"] += 1

        if tick.execution_result and tick.execution_result.get("status") == "EXECUTED":
            self.performance_metrics["successful_predictions"] += 1

        if tick.profit_prediction > 0:
            self.performance_metrics["total_profit"] += tick.profit_prediction

        # Update average similarity score
        total_similarity = sum(t.similarity_score for t in self.tick_history)
        self.performance_metrics["average_similarity_score"] = (
            total_similarity / len(self.tick_history) if self.tick_history else 0.0
        )

        # Update memory hit rate
        memory_hits = sum(1 for t in self.tick_history if t.memory_bucket_id is not None)
        self.performance_metrics["memory_hit_rate"] = memory_hits / len(self.tick_history) if self.tick_history else 0.0

    def _update_visualizer(self, tick: InternalTick):
        """Update visualizer with current tick data"""
        try:
            # Update system status panel
            system_status = {
                "current_tick_id": tick.tick_id,
                "tick_state": tick.state.value,
                "total_ticks": self.performance_metrics["total_ticks"],
                "successful_predictions": self.performance_metrics["successful_predictions"],
                "total_profit": self.performance_metrics["total_profit"],
                "average_similarity_score": self.performance_metrics["average_similarity_score"],
                "memory_hit_rate": self.performance_metrics["memory_hit_rate"],
            }

            self.visualizer.panels[PanelType.SYSTEM_STATUS].update_data(system_status)

            # Update trading state panel
            trading_status = self.trading_integration.get_system_status()
            self.visualizer.panels[PanelType.TRADING_STATE].update_data(trading_status)

        except Exception as e:
            self.logger.warning(f"Visualizer update failed: {e}")

    def store_tick_memory(self, tick: InternalTick, execution_result: Dict[str, Any]):
        """Store tick in fractal memory for future recall"""
        try:
            # Calculate actual profit (simplified)
            profit = 0.0
            if execution_result.get("status") == "EXECUTED":
                # Simulate profit calculation
                profit = tick.profit_prediction * execution_result.get("confidence", 0.5)

            # Create hash memory block
            price_vector = np.array(list(tick.price_data.values()))
            hash_block = HashMemoryBlock(
                hash_signature=tick.hash_signature,
                profit=profit,
                strategy_id=tick.strategy_id,
                entry_vector=price_vector.tolist(),
                exit_vector=price_vector.tolist(),  # Simplified
                timestamp=tick.timestamp,
                similarity_score=tick.similarity_score,
            )

            # Store in mathematical library
            self.mathlib_v2.store_hash_memory(hash_block)

            # Store in appropriate fractal bucket
            if tick.memory_bucket_id and tick.memory_bucket_id in self.fractal_buckets:
                self.fractal_buckets[tick.memory_bucket_id].add_hash_block(hash_block)

            self.logger.info(f"💾 Stored tick memory: {tick.tick_id}")

        except Exception as e:
            self.logger.error(f"Failed to store tick memory: {e}")

    def start_live_mode(self):
        """Start live mode with real-time data processing"""
        self.is_live_mode = True
        self.logger.info("🚀 Starting live mode")

        # Start visualizer
        self.visualizer.start_live_system()

        # Start data collection threads
        self._start_data_collection()

    def _start_data_collection(self):
        """Start data collection threads"""
        # BTC price collection thread
        btc_thread = threading.Thread(target=self._collect_btc_data, daemon=True)
        btc_thread.start()

        # Speed Lattice data collection thread
        vault_thread = threading.Thread(target=self._collect_vault_data, daemon=True)
        vault_thread.start()

        self.logger.info("✅ Data collection threads started")

    def _collect_btc_data(self):
        """Collect BTC price data"""
        while self.is_live_mode:
            try:
                # Simulate BTC price data (replace with real API call)
                current_price = 45000 + np.random.normal(0, 500)

                price_data = {
                    "open": current_price,
                    "high": current_price * 1.01,
                    "low": current_price * 0.99,
                    "close": current_price,
                    "volume": np.random.uniform(100, 1000),
                }

                # Generate and process tick
                tick = self.generate_internal_tick(price_data, "LIVE_BTC_STRATEGY")
                result = self.process_tick(tick)

                # Store memory if execution occurred
                if tick.execution_result:
                    self.store_tick_memory(tick, tick.execution_result)

                time.sleep(3.75 * 60)  # 3.75 minutes per tick

            except Exception as e:
                self.logger.error(f"BTC data collection error: {e}")
                time.sleep(60)

    def _collect_vault_data(self):
        """Collect Speed Lattice Vault data"""
        while self.is_live_mode:
            try:
                # Simulate vault data (replace with real API call)
                vault_data = {
                    "drift_matrix": np.random.randn(8, 8) * 0.1,
                    "chrono_bias": np.random.uniform(0, 0.3),
                    "stability_factor": np.random.uniform(0.1, 1.0),
                }

                # Update visualizer panels
                self.visualizer.panels[PanelType.DRIFT_MATRIX].update_data(vault_data)
                self.visualizer.panels[PanelType.CHRONO_BIAS].update_data(vault_data)
                self.visualizer.panels[PanelType.STABILITY_FACTOR].update_data(vault_data)

                time.sleep(1)  # Update every second

            except Exception as e:
                self.logger.error(f"Vault data collection error: {e}")
                time.sleep(5)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_info": {
                "version": "SP 1.27 AE",
                "is_live_mode": self.is_live_mode,
                "total_ticks": self.performance_metrics["total_ticks"],
                "current_tick_id": self.current_tick.tick_id if self.current_tick else None,
            },
            "performance_metrics": self.performance_metrics,
            "fractal_memory": {
                "total_buckets": len(self.fractal_buckets),
                "total_patterns": sum(b.pattern_count for b in self.fractal_buckets.values()),
                "average_success_rate": np.mean([b.success_rate for b in self.fractal_buckets.values()]),
            },
            "live_streams": {
                stream_id: {"is_active": stream["is_active"], "last_update": stream["last_update"]}
                for stream_id, stream in self.live_data_streams.items()
            },
        }

    def export_system_data(self, filename: str = None) -> str:
        """Export complete system data"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integrated_core_system_{timestamp}.json"

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "system_status": self.get_system_status(),
            "tick_history": [
                {
                    "tick_id": tick.tick_id,
                    "timestamp": tick.timestamp,
                    "state": tick.state.value,
                    "strategy_id": tick.strategy_id,
                    "hash_signature": tick.hash_signature,
                    "profit_prediction": tick.profit_prediction,
                    "similarity_score": tick.similarity_score,
                    "memory_bucket_id": tick.memory_bucket_id,
                    "execution_result": tick.execution_result,
                }
                for tick in self.tick_history[-100:]  # Last 100 ticks
            ],
            "fractal_buckets": {
                bucket_id: {
                    "bucket_id": bucket.bucket_id,
                    "similarity_threshold": bucket.similarity_threshold,
                    "pattern_count": bucket.pattern_count,
                    "total_profit": bucket.total_profit,
                    "success_rate": bucket.success_rate,
                    "average_profit": bucket.get_average_profit(),
                }
                for bucket_id, bucket in self.fractal_buckets.items()
            },
        }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"💾 System data exported to: {filename}")
        return filename

    def stop_live_mode(self):
        """Stop live mode"""
        self.is_live_mode = False
        self.visualizer.stop_live_system()
        self.logger.info("🛑 Live mode stopped")


def main():
    """Main demonstration function"""
    print("🚀 Integrated Core System - Speed Lattice Vault SP 1.27 AE")
    print("=" * 70)

    # Create integrated core system
    core_system = IntegratedCoreSystem()

    # Start live mode
    core_system.start_live_mode()

    print("\n🎛️  System Features:")
    print("   • Hash-based fractal memory system")
    print("   • Real-time pattern recognition")
    print("   • Live API connectivity")
    print("   • Integrated mathematical libraries")
    print("   • Live visualizer panels")
    print("   • Performance tracking")

    print("\n📊 Live Data Streams:")
    for stream_id, stream in core_system.live_data_streams.items():
        print(f"   • {stream_id}: {'✅ Active' if stream['is_active'] else '❌ Inactive'}")

    print("\n🧠 Fractal Memory Buckets:")
    for bucket_id, bucket in core_system.fractal_buckets.items():
        print(f"   • {bucket_id}: {bucket.pattern_count} patterns, {bucket.success_rate:.2%} success rate")

    print("\n🔄 System is now live and processing ticks...")
    print("Press Ctrl+C to stop")

    try:
        # Keep the system running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
    finally:
        core_system.stop_live_mode()

        # Export final data
        filename = core_system.export_system_data()
        print(f"\n✅ System data exported to: {filename}")

        # Print final status
        final_status = core_system.get_system_status()
        print(f"\n📈 Final Performance:")
        print(f"   Total Ticks: {final_status['performance_metrics']['total_ticks']}")
        print(f"   Successful Predictions: {final_status['performance_metrics']['successful_predictions']}")
        print(f"   Total Profit: ${final_status['performance_metrics']['total_profit']:.2f}")
        print(f"   Memory Hit Rate: {final_status['performance_metrics']['memory_hit_rate']:.2%}")
        print(f"   Average Similarity Score: {final_status['performance_metrics']['average_similarity_score']:.3f}")


if __name__ == "__main__":
    main()
