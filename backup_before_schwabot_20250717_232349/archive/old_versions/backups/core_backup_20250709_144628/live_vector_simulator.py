"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 LIVE VECTOR SIMULATOR - SCHWABOT PRICE/ENTROPY STREAM GENERATOR
==================================================================

Advanced live vector simulator that generates realistic price/entropy streams
for testing the hash-based trading system with mathematical accuracy.

    Mathematical Foundation:
    - Price Evolution: P(t+1) = P(t) * (1 + μΔt + σ√Δt * ε + entropy_factor)
    - Entropy Generation: E(t) = σ² * |ΔP/P| + volume_irregularity + spread_factor
    - Volume Dynamics: V(t) = V_base * (1 + volatility_factor * sin(ωt) + noise)
    - Hash Trigger Simulation: H_trigger = f(entropy_threshold, pattern_similarity, time_factor)

    This simulator creates the living, breathing market data that feeds Schwabot's hash engine.
    """

    import asyncio
    import json
    import logging
    import random
    import time
    from dataclasses import dataclass, field
    from enum import Enum
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Tuple, Union

    import numpy as np
    from scipy import stats

    from .hash_match_command_injector import HashMatchCommandInjector, create_hash_match_injector

    logger = logging.getLogger(__name__)


        class MarketRegime(Enum):
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """Market regime types for simulation."""

        BULL_TRENDING = "bull_trending"
        BEAR_TRENDING = "bear_trending"
        SIDEWAYS = "sideways"
        VOLATILE = "volatile"
        CALM = "calm"
        CRASH = "crash"
        PUMP = "pump"


            class EntropyType(Enum):
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Types of entropy for simulation."""

            PRICE_ENTROPY = "price_entropy"
            VOLUME_ENTROPY = "volume_entropy"
            SPREAD_ENTROPY = "spread_entropy"
            ORDER_BOOK_ENTROPY = "order_book_entropy"
            COMPOSITE_ENTROPY = "composite_entropy"


            @dataclass
                class MarketSnapshot:
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                """Complete market snapshot with all components."""

                symbol: str
                price: float
                volume: float
                timestamp: float
                entropy: float
                volatility: float
                spread: float
                bid: float
                ask: float
                market_regime: MarketRegime
                entropy_type: EntropyType
                metadata: Dict[str, Any] = field(default_factory=dict)


                @dataclass
                    class SimulationConfig:
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """Configuration for live vector simulation."""

                    # Price parameters
                    initial_price: float = 50000.0
                    base_volatility: float = 0.02
                    trend_strength: float = 0.001
                    mean_reversion: float = 0.1

                    # Volume parameters
                    base_volume: float = 1000.0
                    volume_volatility: float = 0.3
                    volume_trend: float = 0.0

                    # Entropy parameters
                    base_entropy: float = 0.01
                    entropy_volatility: float = 0.5
                    entropy_spikes: float = 0.1

                    # Market regime parameters
                    regime_duration: float = 3600.0  # 1 hour
                    regime_transition_prob: float = 0.01

                    # Hash trigger parameters
                    hash_trigger_threshold: float = 0.7
                    pattern_similarity_threshold: float = 0.8
                    time_factor_decay: float = 0.95

                    # Simulation parameters
                    tick_interval: float = 1.0  # 1 second
                    simulation_duration: float = 3600.0  # 1 hour
                    random_seed: Optional[int] = None


                        class LiveVectorSimulator:
    """Class for Schwabot trading functionality."""
                        """Class for Schwabot trading functionality."""
                        """
                        🌊 Live Vector Simulator - Schwabot's Market Data Generator

                            Advanced simulator that generates realistic price/entropy streams with:
                            - Mathematical price evolution with trend, volatility, and mean reversion
                            - Dynamic entropy generation based on market conditions
                            - Volume dynamics with seasonal patterns and noise
                            - Market regime transitions with realistic probabilities
                            - Hash trigger simulation for pattern recognition testing

                                Mathematical Foundation:
                                - Price Evolution: P(t+1) = P(t) * (1 + μΔt + σ√Δt * ε + entropy_factor)
                                - Entropy Generation: E(t) = σ² * |ΔP/P| + volume_irregularity + spread_factor
                                - Volume Dynamics: V(t) = V_base * (1 + volatility_factor * sin(ωt) + noise)
                                - Hash Trigger Simulation: H_trigger = f(entropy_threshold, pattern_similarity, time_factor)
                                """

                                    def __init__(self, config: SimulationConfig) -> None:
                                    """
                                    Initialize Live Vector Simulator.

                                        Args:
                                        config: Simulation configuration
                                        """
                                        self.config = config

                                        # Set random seed for reproducibility
                                            if config.random_seed is not None:
                                            np.random.seed(config.random_seed)
                                            random.seed(config.random_seed)

                                            # Market state
                                            self.current_price = config.initial_price
                                            self.current_volume = config.base_volume
                                            self.current_entropy = config.base_entropy
                                            self.current_regime = MarketRegime.SIDEWAYS
                                            self.current_time = time.time()

                                            # Historical data
                                            self.price_history: List[float] = [self.current_price]
                                            self.volume_history: List[float] = [self.current_volume]
                                            self.entropy_history: List[float] = [self.current_entropy]
                                            self.regime_history: List[MarketRegime] = [self.current_regime]
                                            self.timestamp_history: List[float] = [self.current_time]

                                            # Regime parameters
                                            self.regime_start_time = self.current_time
                                            self.regime_parameters = self._initialize_regime_parameters()

                                            # Hash injector for testing
                                            self.hash_injector = create_hash_match_injector()

                                            # Performance tracking
                                            self.total_ticks = 0
                                            self.hash_triggers = 0
                                            self.command_injections = 0

                                            logger.info("🌊 Live Vector Simulator initialized")
                                            logger.info(f"   Initial Price: ${self.current_price:,.2f}")
                                            logger.info(f"   Base Volume: {self.current_volume:,.0f}")
                                            logger.info(f"   Base Entropy: {self.current_entropy:.4f}")
                                            logger.info(f"   Market Regime: {self.current_regime.value}")

                                                def _initialize_regime_parameters(self) -> Dict[MarketRegime, Dict[str, float]]:
                                                """Initialize parameters for each market regime."""
                                            return {
                                            MarketRegime.BULL_TRENDING: {
                                            "trend": 0.002,  # 0.2% per tick
                                            "volatility": 0.015,
                                            "volume_multiplier": 1.2,
                                            "entropy_multiplier": 0.8,
                                            },
                                            MarketRegime.BEAR_TRENDING: {
                                            "trend": -0.002,  # -0.2% per tick
                                            "volatility": 0.025,
                                            "volume_multiplier": 1.5,
                                            "entropy_multiplier": 1.3,
                                            },
                                            MarketRegime.SIDEWAYS: {
                                            "trend": 0.0,
                                            "volatility": 0.02,
                                            "volume_multiplier": 1.0,
                                            "entropy_multiplier": 1.0,
                                            },
                                            MarketRegime.VOLATILE: {
                                            "trend": 0.0,
                                            "volatility": 0.04,
                                            "volume_multiplier": 2.0,
                                            "entropy_multiplier": 2.0,
                                            },
                                            MarketRegime.CALM: {
                                            "trend": 0.0,
                                            "volatility": 0.01,
                                            "volume_multiplier": 0.7,
                                            "entropy_multiplier": 0.6,
                                            },
                                            MarketRegime.CRASH: {
                                            "trend": -0.01,  # -1% per tick
                                            "volatility": 0.06,
                                            "volume_multiplier": 3.0,
                                            "entropy_multiplier": 3.0,
                                            },
                                            MarketRegime.PUMP: {
                                            "trend": 0.01,  # 1% per tick
                                            "volatility": 0.05,
                                            "volume_multiplier": 2.5,
                                            "entropy_multiplier": 2.5,
                                            },
                                            }

                                                def _evolve_price(self) -> float:
                                                """
                                                Evolve price using mathematical model.

                                                Mathematical: P(t+1) = P(t) * (1 + μΔt + σ√Δt * ε + entropy_factor)

                                                    Returns:
                                                    New price
                                                    """
                                                        try:
                                                        # Get regime parameters
                                                        regime_params = self.regime_parameters[self.current_regime]

                                                        # Calculate components
                                                        trend_component = regime_params["trend"]
                                                        volatility_component = regime_params["volatility"] * np.sqrt(1.0) * np.random.normal(0, 1)

                                                        # Mean reversion component
                                                        mean_reversion_component = (
                                                        self.config.mean_reversion * (self.config.initial_price - self.current_price) / self.current_price
                                                        )

                                                        # Entropy factor
                                                        entropy_factor = self.current_entropy * np.random.normal(0, 0.1)

                                                        # Calculate price change
                                                        price_change = trend_component + volatility_component + mean_reversion_component + entropy_factor

                                                        # Apply price change
                                                        new_price = self.current_price * (1 + price_change)

                                                        # Ensure positive price
                                                        new_price = max(new_price, self.current_price * 0.5)

                                                    return new_price

                                                        except Exception as e:
                                                        logger.error(f"Error evolving price: {e}")
                                                    return self.current_price

                                                        def _evolve_volume(self) -> float:
                                                        """
                                                        Evolve volume using mathematical model.

                                                        Mathematical: V(t) = V_base * (1 + volatility_factor * sin(ωt) + noise)

                                                            Returns:
                                                            New volume
                                                            """
                                                                try:
                                                                # Get regime parameters
                                                                regime_params = self.regime_parameters[self.current_regime]

                                                                # Calculate components
                                                                time_factor = np.sin(2 * np.pi * self.total_ticks / 100)  # Seasonal pattern
                                                                volatility_factor = self.config.volume_volatility * np.random.normal(0, 1)
                                                                regime_multiplier = regime_params["volume_multiplier"]

                                                                # Calculate volume change
                                                                volume_change = (
                                                                self.config.volume_trend + volatility_factor * time_factor + np.random.normal(0, 0.1)  # Noise
                                                                )

                                                                # Apply volume change
                                                                new_volume = self.current_volume * (1 + volume_change) * regime_multiplier

                                                                # Ensure positive volume
                                                                new_volume = max(new_volume, self.config.base_volume * 0.1)

                                                            return new_volume

                                                                except Exception as e:
                                                                logger.error(f"Error evolving volume: {e}")
                                                            return self.current_volume

                                                                def _calculate_entropy(self, price_change: float, volume_change: float) -> float:
                                                                """
                                                                Calculate entropy based on market dynamics.

                                                                Mathematical: E(t) = σ² * |ΔP/P| + volume_irregularity + spread_factor

                                                                    Args:
                                                                    price_change: Price change percentage
                                                                    volume_change: Volume change percentage

                                                                        Returns:
                                                                        Entropy value
                                                                        """
                                                                            try:
                                                                            # Get regime parameters
                                                                            regime_params = self.regime_parameters[self.current_regime]

                                                                            # Price entropy component
                                                                            price_entropy = abs(price_change) * self.config.base_volatility

                                                                            # Volume irregularity component
                                                                            volume_irregularity = abs(volume_change) * 0.1

                                                                            # Spread factor (simulated)
                                                                            spread_factor = np.random.exponential(0.01)

                                                                            # Composite entropy
                                                                            composite_entropy = (price_entropy + volume_irregularity + spread_factor) * regime_params[
                                                                            "entropy_multiplier"
                                                                            ]

                                                                            # Add random spikes
                                                                                if np.random.random() < self.config.entropy_spikes:
                                                                                composite_entropy *= np.random.uniform(2.0, 5.0)

                                                                                # Ensure reasonable bounds
                                                                                composite_entropy = max(0.001, min(composite_entropy, 0.1))

                                                                            return composite_entropy

                                                                                except Exception as e:
                                                                                logger.error(f"Error calculating entropy: {e}")
                                                                            return self.config.base_entropy

                                                                                def _check_regime_transition(self) -> bool:
                                                                                """
                                                                                Check if market regime should transition.

                                                                                    Returns:
                                                                                    True if regime should transition, False otherwise
                                                                                    """
                                                                                        try:
                                                                                        # Check time-based transition
                                                                                        time_in_regime = time.time() - self.regime_start_time
                                                                                            if time_in_regime > self.config.regime_duration:
                                                                                        return True

                                                                                        # Check probability-based transition
                                                                                            if np.random.random() < self.config.regime_transition_prob:
                                                                                        return True

                                                                                    return False

                                                                                        except Exception as e:
                                                                                        logger.error(f"Error checking regime transition: {e}")
                                                                                    return False

                                                                                        def _transition_regime(self) -> None:
                                                                                        """Transition to a new market regime."""
                                                                                            try:
                                                                                            # Define transition probabilities
                                                                                            transition_matrix = {
                                                                                            MarketRegime.BULL_TRENDING: {
                                                                                            MarketRegime.BEAR_TRENDING: 0.1,
                                                                                            MarketRegime.SIDEWAYS: 0.3,
                                                                                            MarketRegime.VOLATILE: 0.2,
                                                                                            MarketRegime.CRASH: 0.05,
                                                                                            },
                                                                                            MarketRegime.BEAR_TRENDING: {
                                                                                            MarketRegime.BULL_TRENDING: 0.1,
                                                                                            MarketRegime.SIDEWAYS: 0.3,
                                                                                            MarketRegime.VOLATILE: 0.2,
                                                                                            MarketRegime.PUMP: 0.05,
                                                                                            },
                                                                                            MarketRegime.SIDEWAYS: {
                                                                                            MarketRegime.BULL_TRENDING: 0.2,
                                                                                            MarketRegime.BEAR_TRENDING: 0.2,
                                                                                            MarketRegime.VOLATILE: 0.1,
                                                                                            MarketRegime.CALM: 0.1,
                                                                                            },
                                                                                            MarketRegime.VOLATILE: {
                                                                                            MarketRegime.SIDEWAYS: 0.4,
                                                                                            MarketRegime.CALM: 0.2,
                                                                                            MarketRegime.BULL_TRENDING: 0.1,
                                                                                            MarketRegime.BEAR_TRENDING: 0.1,
                                                                                            },
                                                                                            MarketRegime.CALM: {
                                                                                            MarketRegime.SIDEWAYS: 0.3,
                                                                                            MarketRegime.VOLATILE: 0.2,
                                                                                            MarketRegime.BULL_TRENDING: 0.1,
                                                                                            MarketRegime.BEAR_TRENDING: 0.1,
                                                                                            },
                                                                                            MarketRegime.CRASH: {
                                                                                            MarketRegime.BEAR_TRENDING: 0.3,
                                                                                            MarketRegime.VOLATILE: 0.3,
                                                                                            MarketRegime.SIDEWAYS: 0.2,
                                                                                            },
                                                                                            MarketRegime.PUMP: {
                                                                                            MarketRegime.BULL_TRENDING: 0.3,
                                                                                            MarketRegime.VOLATILE: 0.3,
                                                                                            MarketRegime.SIDEWAYS: 0.2,
                                                                                            },
                                                                                            }

                                                                                            # Get possible transitions
                                                                                            possible_transitions = transition_matrix.get(self.current_regime, {})
                                                                                                if not possible_transitions:
                                                                                            return

                                                                                            # Select new regime based on probabilities
                                                                                            regimes = list(possible_transitions.keys())
                                                                                            probabilities = list(possible_transitions.values())

                                                                                            # Normalize probabilities
                                                                                            total_prob = sum(probabilities)
                                                                                                if total_prob > 0:
                                                                                                normalized_probs = [p / total_prob for p in probabilities]
                                                                                                new_regime = np.random.choice(regimes, p=normalized_probs)

                                                                                                # Update regime
                                                                                                old_regime = self.current_regime
                                                                                                self.current_regime = new_regime
                                                                                                self.regime_start_time = time.time()

                                                                                                logger.info(f"🔄 Market regime transition: {old_regime.value} → {new_regime.value}")

                                                                                                    except Exception as e:
                                                                                                    logger.error(f"Error transitioning regime: {e}")

                                                                                                        def _simulate_hash_trigger(self, market_snapshot: MarketSnapshot) -> bool:
                                                                                                        """
                                                                                                        Simulate hash trigger based on market conditions.

                                                                                                        Mathematical: H_trigger = f(entropy_threshold, pattern_similarity, time_factor)

                                                                                                            Args:
                                                                                                            market_snapshot: Current market snapshot

                                                                                                                Returns:
                                                                                                                True if hash should trigger, False otherwise
                                                                                                                """
                                                                                                                    try:
                                                                                                                    # Calculate trigger probability based on entropy
                                                                                                                    entropy_factor = min(1.0, market_snapshot.entropy / self.config.hash_trigger_threshold)

                                                                                                                    # Time factor (decay over time)
                                                                                                                    time_factor = self.config.time_factor_decay ** (self.total_ticks % 100)

                                                                                                                    # Pattern similarity (simulated)
                                                                                                                    pattern_similarity = np.random.beta(2, 5)  # Skewed towards lower values

                                                                                                                    # Combined trigger probability
                                                                                                                    trigger_probability = entropy_factor * time_factor * pattern_similarity

                                                                                                                    # Check if trigger occurs
                                                                                                                        if np.random.random() < trigger_probability:
                                                                                                                        self.hash_triggers += 1
                                                                                                                    return True

                                                                                                                return False

                                                                                                                    except Exception as e:
                                                                                                                    logger.error(f"Error simulating hash trigger: {e}")
                                                                                                                return False

                                                                                                                    def generate_tick(self) -> MarketSnapshot:
                                                                                                                    """
                                                                                                                    Generate a single market tick.

                                                                                                                        Returns:
                                                                                                                        MarketSnapshot with current market data
                                                                                                                        """
                                                                                                                            try:
                                                                                                                            # Check for regime transition
                                                                                                                                if self._check_regime_transition():
                                                                                                                                self._transition_regime()

                                                                                                                                # Evolve market components
                                                                                                                                old_price = self.current_price
                                                                                                                                old_volume = self.current_volume

                                                                                                                                self.current_price = self._evolve_price()
                                                                                                                                self.current_volume = self._evolve_volume()

                                                                                                                                # Calculate changes
                                                                                                                                price_change = (self.current_price - old_price) / old_price
                                                                                                                                volume_change = (self.current_volume - old_volume) / old_volume

                                                                                                                                # Calculate entropy
                                                                                                                                self.current_entropy = self._calculate_entropy(price_change, volume_change)

                                                                                                                                # Update time
                                                                                                                                self.current_time = time.time()

                                                                                                                                # Calculate spread
                                                                                                                                spread = self.current_price * 0.001  # 0.1% spread
                                                                                                                                bid = self.current_price - spread / 2
                                                                                                                                ask = self.current_price + spread / 2

                                                                                                                                # Create market snapshot
                                                                                                                                snapshot = MarketSnapshot(
                                                                                                                                symbol="BTCUSDT",
                                                                                                                                price=self.current_price,
                                                                                                                                volume=self.current_volume,
                                                                                                                                timestamp=self.current_time,
                                                                                                                                entropy=self.current_entropy,
                                                                                                                                volatility=abs(price_change),
                                                                                                                                spread=spread,
                                                                                                                                bid=bid,
                                                                                                                                ask=ask,
                                                                                                                                market_regime=self.current_regime,
                                                                                                                                entropy_type=EntropyType.COMPOSITE_ENTROPY,
                                                                                                                                metadata={
                                                                                                                                "price_change": price_change,
                                                                                                                                "volume_change": volume_change,
                                                                                                                                "tick_number": self.total_ticks,
                                                                                                                                "regime_duration": self.current_time - self.regime_start_time,
                                                                                                                                },
                                                                                                                                )

                                                                                                                                # Update history
                                                                                                                                self.price_history.append(self.current_price)
                                                                                                                                self.volume_history.append(self.current_volume)
                                                                                                                                self.entropy_history.append(self.current_entropy)
                                                                                                                                self.regime_history.append(self.current_regime)
                                                                                                                                self.timestamp_history.append(self.current_time)

                                                                                                                                # Limit history size
                                                                                                                                max_history = 10000
                                                                                                                                    if len(self.price_history) > max_history:
                                                                                                                                    self.price_history = self.price_history[-max_history:]
                                                                                                                                    self.volume_history = self.volume_history[-max_history:]
                                                                                                                                    self.entropy_history = self.entropy_history[-max_history:]
                                                                                                                                    self.regime_history = self.regime_history[-max_history:]
                                                                                                                                    self.timestamp_history = self.timestamp_history[-max_history:]

                                                                                                                                    self.total_ticks += 1

                                                                                                                                return snapshot

                                                                                                                                    except Exception as e:
                                                                                                                                    logger.error(f"Error generating tick: {e}")
                                                                                                                                    # Return safe default snapshot
                                                                                                                                return MarketSnapshot(
                                                                                                                                symbol="BTCUSDT",
                                                                                                                                price=self.current_price,
                                                                                                                                volume=self.current_volume,
                                                                                                                                timestamp=time.time(),
                                                                                                                                entropy=self.current_entropy,
                                                                                                                                volatility=0.0,
                                                                                                                                spread=0.0,
                                                                                                                                bid=self.current_price,
                                                                                                                                ask=self.current_price,
                                                                                                                                market_regime=self.current_regime,
                                                                                                                                entropy_type=EntropyType.COMPOSITE_ENTROPY,
                                                                                                                                metadata={"error": str(e)},
                                                                                                                                )

                                                                                                                                    async def run_simulation(self, callback: Optional[callable] = None):
                                                                                                                                    """
                                                                                                                                    Run the live vector simulation.

                                                                                                                                        Args:
                                                                                                                                        callback: Optional callback function to process each tick
                                                                                                                                        """
                                                                                                                                            try:
                                                                                                                                            logger.info("🚀 Starting Live Vector Simulation")
                                                                                                                                            logger.info(f"   Duration: {self.config.simulation_duration:.0f} seconds")
                                                                                                                                            logger.info(f"   Tick Interval: {self.config.tick_interval:.1f} seconds")

                                                                                                                                            start_time = time.time()
                                                                                                                                            end_time = start_time + self.config.simulation_duration

                                                                                                                                                while time.time() < end_time:
                                                                                                                                                # Generate tick
                                                                                                                                                snapshot = self.generate_tick()

                                                                                                                                                # Simulate hash trigger
                                                                                                                                                hash_triggered = self._simulate_hash_trigger(snapshot)

                                                                                                                                                    if hash_triggered:
                                                                                                                                                    logger.debug(f"🔗 Hash trigger at tick {self.total_ticks}")

                                                                                                                                                    # Process with hash injector
                                                                                                                                                    tick_data = {
                                                                                                                                                    "symbol": snapshot.symbol,
                                                                                                                                                    "price": snapshot.price,
                                                                                                                                                    "volume": snapshot.volume,
                                                                                                                                                    "timestamp": snapshot.timestamp,
                                                                                                                                                    "entropy": snapshot.entropy,
                                                                                                                                                    "volatility": snapshot.volatility,
                                                                                                                                                    }

                                                                                                                                                    injection_result = await self.hash_injector.process_tick(tick_data)
                                                                                                                                                        if injection_result:
                                                                                                                                                        self.command_injections += 1
                                                                                                                                                        logger.info(f"💉 Command injected: {injection_result.command.command_type.value}")

                                                                                                                                                        # Call callback if provided
                                                                                                                                                            if callback:
                                                                                                                                                            await callback(snapshot, hash_triggered)

                                                                                                                                                            # Wait for next tick
                                                                                                                                                            await asyncio.sleep(self.config.tick_interval)

                                                                                                                                                            # Simulation complete
                                                                                                                                                            duration = time.time() - start_time
                                                                                                                                                            logger.info("✅ Live Vector Simulation completed")
                                                                                                                                                            logger.info(f"   Total Ticks: {self.total_ticks}")
                                                                                                                                                            logger.info(f"   Hash Triggers: {self.hash_triggers}")
                                                                                                                                                            logger.info(f"   Command Injections: {self.command_injections}")
                                                                                                                                                            logger.info(f"   Duration: {duration:.1f} seconds")

                                                                                                                                                                except Exception as e:
                                                                                                                                                                logger.error(f"Error running simulation: {e}")

                                                                                                                                                                    def get_simulation_summary(self) -> Dict[str, Any]:
                                                                                                                                                                    """
                                                                                                                                                                    Get summary of simulation results.

                                                                                                                                                                        Returns:
                                                                                                                                                                        Dictionary with simulation statistics
                                                                                                                                                                        """
                                                                                                                                                                            try:
                                                                                                                                                                            # Calculate statistics
                                                                                                                                                                            price_stats = {
                                                                                                                                                                            "min": min(self.price_history),
                                                                                                                                                                            "max": max(self.price_history),
                                                                                                                                                                            "mean": np.mean(self.price_history),
                                                                                                                                                                            "std": np.std(self.price_history),
                                                                                                                                                                            "final": self.price_history[-1] if self.price_history else 0.0,
                                                                                                                                                                            }

                                                                                                                                                                            volume_stats = {
                                                                                                                                                                            "min": min(self.volume_history),
                                                                                                                                                                            "max": max(self.volume_history),
                                                                                                                                                                            "mean": np.mean(self.volume_history),
                                                                                                                                                                            "std": np.std(self.volume_history),
                                                                                                                                                                            }

                                                                                                                                                                            entropy_stats = {
                                                                                                                                                                            "min": min(self.entropy_history),
                                                                                                                                                                            "max": max(self.entropy_history),
                                                                                                                                                                            "mean": np.mean(self.entropy_history),
                                                                                                                                                                            "std": np.std(self.entropy_history),
                                                                                                                                                                            }

                                                                                                                                                                            # Regime distribution
                                                                                                                                                                            regime_counts = {}
                                                                                                                                                                                for regime in self.regime_history:
                                                                                                                                                                                regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1

                                                                                                                                                                                summary = {
                                                                                                                                                                                "total_ticks": self.total_ticks,
                                                                                                                                                                                "hash_triggers": self.hash_triggers,
                                                                                                                                                                                "command_injections": self.command_injections,
                                                                                                                                                                                "trigger_rate": self.hash_triggers / max(1, self.total_ticks),
                                                                                                                                                                                "injection_rate": self.command_injections / max(1, self.hash_triggers),
                                                                                                                                                                                "price_statistics": price_stats,
                                                                                                                                                                                "volume_statistics": volume_stats,
                                                                                                                                                                                "entropy_statistics": entropy_stats,
                                                                                                                                                                                "regime_distribution": regime_counts,
                                                                                                                                                                                "config": {
                                                                                                                                                                                "initial_price": self.config.initial_price,
                                                                                                                                                                                "base_volatility": self.config.base_volatility,
                                                                                                                                                                                "tick_interval": self.config.tick_interval,
                                                                                                                                                                                "simulation_duration": self.config.simulation_duration,
                                                                                                                                                                                },
                                                                                                                                                                                }

                                                                                                                                                                            return summary

                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                logger.error(f"Error getting simulation summary: {e}")
                                                                                                                                                                            return {"error": str(e)}

                                                                                                                                                                                def export_data(self, filename: str) -> None:
                                                                                                                                                                                """
                                                                                                                                                                                Export simulation data to JSON file.

                                                                                                                                                                                    Args:
                                                                                                                                                                                    filename: Output filename
                                                                                                                                                                                    """
                                                                                                                                                                                        try:
                                                                                                                                                                                        data = {
                                                                                                                                                                                        "price_history": self.price_history,
                                                                                                                                                                                        "volume_history": self.volume_history,
                                                                                                                                                                                        "entropy_history": self.entropy_history,
                                                                                                                                                                                        "regime_history": [r.value for r in self.regime_history],
                                                                                                                                                                                        "timestamp_history": self.timestamp_history,
                                                                                                                                                                                        "summary": self.get_simulation_summary(),
                                                                                                                                                                                        }

                                                                                                                                                                                            with open(filename, 'w') as f:
                                                                                                                                                                                            json.dump(data, f, indent=2)

                                                                                                                                                                                            logger.info(f"📁 Simulation data exported to {filename}")

                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                logger.error(f"Error exporting data: {e}")


                                                                                                                                                                                                    async def test_live_vector_simulator():
                                                                                                                                                                                                    """Test the Live Vector Simulator functionality."""
                                                                                                                                                                                                    logger.info("🧪 Testing Live Vector Simulator")

                                                                                                                                                                                                    # Create simulation config
                                                                                                                                                                                                    config = SimulationConfig(
                                                                                                                                                                                                    initial_price=50000.0,
                                                                                                                                                                                                    base_volatility=0.02,
                                                                                                                                                                                                    tick_interval=0.1,  # 100ms ticks
                                                                                                                                                                                                    simulation_duration=60.0,  # 1 minute
                                                                                                                                                                                                    random_seed=42,
                                                                                                                                                                                                    )

                                                                                                                                                                                                    # Create simulator
                                                                                                                                                                                                    simulator = LiveVectorSimulator(config)

                                                                                                                                                                                                    # Define callback function
                                                                                                                                                                                                        async def tick_callback(snapshot: MarketSnapshot, hash_triggered: bool):
                                                                                                                                                                                                            if hash_triggered:
                                                                                                                                                                                                            logger.info(
                                                                                                                                                                                                            f"🔗 Tick {simulator.total_ticks}: Price=${snapshot.price:,.2f}, Entropy={snapshot.entropy:.4f}, Hash Triggered"
                                                                                                                                                                                                            )
                                                                                                                                                                                                            elif simulator.total_ticks % 100 == 0:  # Log every 100th tick
                                                                                                                                                                                                            logger.info(
                                                                                                                                                                                                            f"📊 Tick {simulator.total_ticks}: Price=${snapshot.price:,.2f}, Entropy={snapshot.entropy:.4f}"
                                                                                                                                                                                                            )

                                                                                                                                                                                                            # Run simulation
                                                                                                                                                                                                            await simulator.run_simulation(callback=tick_callback)

                                                                                                                                                                                                            # Get summary
                                                                                                                                                                                                            summary = simulator.get_simulation_summary()
                                                                                                                                                                                                            logger.info(f"📊 Simulation Summary: {summary}")

                                                                                                                                                                                                            # Export data
                                                                                                                                                                                                            simulator.export_data("simulation_data.json")

                                                                                                                                                                                                            logger.info("🧪 Live Vector Simulator test completed")


                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                                asyncio.run(test_live_vector_simulator())
