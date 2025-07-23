#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠⚛️ ORBITAL SHELL BRAIN SYSTEM - ENHANCED WITH 268 DECIMAL HASHING
===================================================================

Advanced orbital shell brain system implementing quantum-inspired trading logic
with 268 decimal hashing, real portfolio holdings, and BIT strategy integration.

Mathematical Architecture:
- ψₙ(t,r) = Rₙ(r) · Yₙ(θ,φ) · e^(-iEₙt/ħ)
- Eₙ = -(k²/2n²) + λ·σₙ² - μ·∂Rₙ/∂t
- ℵₐ(t) = ∇ψₜ + ρ(t)·εₜ - ∂Φ/∂t
- 𝒞ₛ = Σ(Ψₛ · Θₛ · ωₛ) for s=1 to 8
- H₂₆₈ = SHA256(price_268_decimal + portfolio_holdings + orbital_state)
- BIT_Strategy = f(H₂₆₈, orbital_shell, randomized_holdings)
"""

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set precision for 268 decimal calculations
getcontext().prec = 268

# Import existing Schwabot components
try:
    from .distributed_mathematical_processor import DistributedMathematicalProcessor
    from .enhanced_error_recovery_system import EnhancedErrorRecoverySystem
    from .ghost_core import GhostCore
    from .neural_processing_engine import NeuralProcessingEngine
    from .quantum_mathematical_bridge import QuantumMathematicalBridge
    from .unified_profit_vectorization_system import UnifiedProfitVectorizationSystem
    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some Schwabot components not available: {e}")
    SCHWABOT_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# 🧠 ORBITAL SHELL DEFINITIONS
class OrbitalShell(Enum):
    """8 Orbital Shells based on Electron Model"""
    NUCLEUS = 0  # ColdBase/Reserve pool (USDC, BTC, vault)
    CORE = 1  # High-certainty, long-hold buys
    HOLD = 2  # Mid-conviction, medium horizon trades
    SCOUT = 3  # Short-term entry testing buys
    FEEDER = 4  # Entry dip tracker + trade initiator
    RELAY = 5  # Active trading shell (most frequent, trades)
    FLICKER = 6  # Volatility scalp zone
    GHOST = 7  # Speculative/high-risk AI-only zone


@dataclass
class PortfolioHoldings:
    """Real portfolio holdings with 268 decimal precision"""
    BTC: Decimal = Decimal('0')
    USDC: Decimal = Decimal('0')
    ETH: Decimal = Decimal('0')
    XRP: Decimal = Decimal('0')
    SOL: Decimal = Decimal('0')
    last_update: float = field(default_factory=time.time)
    
    def get_total_value_usdc(self, prices: Dict[str, float]) -> Decimal:
        """Calculate total portfolio value in USDC"""
        total = self.USDC
        for asset, amount in self.__dict__.items():
            if asset in prices and asset != 'USDC' and asset != 'last_update':
                total += amount * Decimal(str(prices[asset]))
        return total
    
    def get_asset_allocation(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Get current asset allocation percentages"""
        total_value = self.get_total_value_usdc(prices)
        if total_value == 0:
            return {}
        
        allocation = {}
        for asset, amount in self.__dict__.items():
            if asset != 'last_update':
                if asset == 'USDC':
                    asset_value = amount
                else:
                    asset_value = amount * Decimal(str(prices.get(asset, 0)))
                allocation[asset] = float(asset_value / total_value)
        return allocation


@dataclass
class BITStrategy:
    """BIT Strategy with randomized holdings"""
    strategy_hash: str
    orbital_shell: OrbitalShell
    randomized_holdings: Dict[str, Decimal]
    confidence_score: float
    profit_potential: float
    risk_level: float
    execution_priority: int
    timestamp: float


@dataclass
class TradingPair:
    """Trading pair with bidirectional support"""
    base: str  # e.g., "BTC"
    quote: str  # e.g., "USDC"
    symbol: str  # e.g., "BTC/USDC"
    reverse_symbol: str  # e.g., "USDC/BTC"
    min_order_size: Decimal
    max_order_size: Decimal
    price_precision: int
    amount_precision: int


@dataclass
class OrbitalState:
    """Quantum state for orbital shell ψₙ(t,r)"""
    shell: OrbitalShell
    radial_probability: float  # Rₙ(r)
    angular_momentum: Tuple[float, float]  # Yₙ(θ,φ)
    energy_level: float  # Eₙ
    time_evolution: complex  # e^(-iEₙt/ħ)
    confidence: float
    asset_allocation: Dict[str, float] = field(default_factory=dict)
    current_holdings: PortfolioHoldings = field(default_factory=PortfolioHoldings)


@dataclass
class ShellMemoryTensor:
    """Orbital Memory Tensor ℳₛ for shell s"""
    shell: OrbitalShell
    memory_vector: np.ndarray  # [t₀, t₁, ..., tₙ]
    entry_history: List[float]
    exit_history: List[float]
    pnl_history: List[float]
    volatility_history: List[float]
    fractal_match_history: List[float]
    last_update: float


@dataclass
class AltitudeVector:
    """Mathematical Altitude Vector ℵₐ(t)"""
    momentum_curvature: float  # ∇ψₜ
    rolling_return: float  # ρ(t)
    entropy_shift: float  # εₜ
    alpha_decay: float  # ∂Φ/∂t
    altitude_value: float  # ℵₐ(t)
    confidence_level: float


@dataclass
class ShellConsensus:
    """Shell Consensus State 𝒞ₛ"""
    consensus_score: float  # 𝒞ₛ = Σ(Ψₛ · Θₛ · ωₛ)
    active_shells: List[OrbitalShell]
    shell_activations: Dict[OrbitalShell, float]  # Ψₛ
    shell_confidences: Dict[OrbitalShell, float]  # Θₛ
    shell_weights: Dict[OrbitalShell, float]  # ωₛ
    threshold_met: bool


@dataclass
class ProfitTierBucket:
    """Profit-Tier Vector Bucket 𝒱ₚ"""
    bucket_id: int
    profit_range: Tuple[float, float]
    stop_loss: float
    take_profit: Optional[float]
    position_size_multiplier: float
    risk_level: float
    reentry_allowed: bool
    dynamic_sl_enabled: bool


class OrbitalBRAINSystem:
    """
    🧠⚛️ Complete Orbital Shell + BRAIN Neural Pathway System
    ENHANCED WITH 268 DECIMAL HASHING AND REAL PORTFOLIO INTEGRATION

    Implements the revolutionary combination of:
    - Electron Orbital Shell Model (8 shells)
    - BRAIN Neural Shell Pathway System
    - 268 Decimal Hashing Logic
    - Real Portfolio Holdings Integration
    - BIT Strategy with Randomized Holdings
    - Bidirectional Trading Support
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        self.config = config or self._default_config()

        # Initialize orbital shells
        self.orbital_states: Dict[OrbitalShell, OrbitalState] = {}
        self.shell_memory_tensors: Dict[OrbitalShell, ShellMemoryTensor] = {}
        self.initialize_orbital_shells()

        # BRAIN Neural Components
        self.neural_shell_weights = np.random.rand(8, 64)  # W₁, W₂ weights
        self.shell_dna_database: Dict[str, Dict[str, Any]] = {}

        # Altitude and Consensus Systems
        self.current_altitude_vector: Optional[AltitudeVector] = None
        self.current_shell_consensus: Optional[ShellConsensus] = None

        # Profit Tier Buckets
        self.profit_buckets = self._initialize_profit_buckets()

        # 268 Decimal Hashing System
        self.hash_268_cache: Dict[str, str] = {}
        self.hash_268_history: List[Tuple[str, float]] = []

        # Real Portfolio Integration
        self.current_portfolio: PortfolioHoldings = PortfolioHoldings()
        self.portfolio_history: List[Tuple[PortfolioHoldings, float]] = []

        # Trading Pairs Configuration
        self.trading_pairs = self._initialize_trading_pairs()

        # BIT Strategy System
        self.bit_strategies: List[BITStrategy] = []
        self.current_bit_strategy: Optional[BITStrategy] = None

        # Initialize Schwabot components if available
        if SCHWABOT_COMPONENTS_AVAILABLE:
            self.quantum_bridge = QuantumMathematicalBridge(quantum_dimension=16)
            self.neural_engine = NeuralProcessingEngine()
            self.distributed_processor = DistributedMathematicalProcessor()
            self.error_recovery = EnhancedErrorRecoverySystem()
            self.profit_vectorizer = UnifiedProfitVectorizationSystem()
            self.ghost_core = GhostCore()

        # System state
        self.active = False
        self.rotation_thread = None
        self.system_lock = threading.Lock()

        logger.info("🧠⚛️ Enhanced Orbital BRAIN System initialized with 268 decimal hashing")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "h_bar": 1.0,  # Normalization constant
            "k_constant": 1.5,  # Historical BTC range entropy
            "lambda_volatility": 0.2,  # Volatility penalty
            "mu_reaction": 0.1,  # Reaction delay coefficient
            "rotation_interval": 300.0,  # 5 minutes
            "consensus_threshold": 0.75,
            "altitude_threshold": 0.6,
            "max_shells_active": 4,
            "assets": ["BTC", "ETH", "XRP", "SOL", "USDC"],
            "hash_268_precision": 268,
            "bit_strategy_randomization": True,
            "bidirectional_trading": True,
        }

    def _initialize_trading_pairs(self) -> Dict[str, TradingPair]:
        """Initialize trading pairs with bidirectional support"""
        pairs = {}
        
        # BTC/USDC and USDC/BTC
        pairs["BTC/USDC"] = TradingPair(
            base="BTC", quote="USDC", symbol="BTC/USDC", reverse_symbol="USDC/BTC",
            min_order_size=Decimal('0.001'), max_order_size=Decimal('10.0'),
            price_precision=2, amount_precision=6
        )
        pairs["USDC/BTC"] = TradingPair(
            base="USDC", quote="BTC", symbol="USDC/BTC", reverse_symbol="BTC/USDC",
            min_order_size=Decimal('10.0'), max_order_size=Decimal('1000000.0'),
            price_precision=8, amount_precision=2
        )
        
        # ETH/USDC and USDC/ETH
        pairs["ETH/USDC"] = TradingPair(
            base="ETH", quote="USDC", symbol="ETH/USDC", reverse_symbol="USDC/ETH",
            min_order_size=Decimal('0.01'), max_order_size=Decimal('100.0'),
            price_precision=2, amount_precision=6
        )
        pairs["USDC/ETH"] = TradingPair(
            base="USDC", quote="ETH", symbol="USDC/ETH", reverse_symbol="ETH/USDC",
            min_order_size=Decimal('10.0'), max_order_size=Decimal('1000000.0'),
            price_precision=8, amount_precision=2
        )
        
        # XRP/USDC and USDC/XRP
        pairs["XRP/USDC"] = TradingPair(
            base="XRP", quote="USDC", symbol="XRP/USDC", reverse_symbol="USDC/XRP",
            min_order_size=Decimal('1.0'), max_order_size=Decimal('100000.0'),
            price_precision=4, amount_precision=2
        )
        pairs["USDC/XRP"] = TradingPair(
            base="USDC", quote="XRP", symbol="USDC/XRP", reverse_symbol="XRP/USDC",
            min_order_size=Decimal('10.0'), max_order_size=Decimal('1000000.0'),
            price_precision=6, amount_precision=2
        )
        
        # SOL/USDC and USDC/SOL
        pairs["SOL/USDC"] = TradingPair(
            base="SOL", quote="USDC", symbol="SOL/USDC", reverse_symbol="USDC/SOL",
            min_order_size=Decimal('0.1'), max_order_size=Decimal('1000.0'),
            price_precision=2, amount_precision=6
        )
        pairs["USDC/SOL"] = TradingPair(
            base="USDC", quote="SOL", symbol="USDC/SOL", reverse_symbol="SOL/USDC",
            min_order_size=Decimal('10.0'), max_order_size=Decimal('1000000.0'),
            price_precision=8, amount_precision=2
        )
        
        return pairs

    def generate_268_decimal_hash(self, price: float, portfolio: PortfolioHoldings, 
                                 orbital_state: OrbitalState) -> str:
        """
        Generate 268 decimal precision hash for trading decisions
        
        H₂₆₈ = SHA256(price_268_decimal + portfolio_holdings + orbital_state)
        """
        try:
            # Convert price to 268 decimal precision
            price_decimal = Decimal(str(price))
            price_268_str = f"{price_decimal:.268f}"
            
            # Create portfolio holdings string
            portfolio_str = f"BTC:{portfolio.BTC}USDC:{portfolio.USDC}ETH:{portfolio.ETH}XRP:{portfolio.XRP}SOL:{portfolio.SOL}"
            
            # Create orbital state string
            orbital_str = f"shell:{orbital_state.shell.value}energy:{orbital_state.energy_level}confidence:{orbital_state.confidence}"
            
            # Combine all components
            combined_data = f"{price_268_str}_{portfolio_str}_{orbital_str}_{time.time():.6f}"
            
            # Generate SHA256 hash
            hash_268 = hashlib.sha256(combined_data.encode()).hexdigest()
            
            # Cache the hash
            self.hash_268_cache[hash_268] = combined_data
            self.hash_268_history.append((hash_268, time.time()))
            
            # Keep only last 1000 hashes
            if len(self.hash_268_history) > 1000:
                self.hash_268_history.pop(0)
            
            return hash_268
            
        except Exception as e:
            logger.error(f"Error generating 268 decimal hash: {e}")
            return hashlib.sha256("fallback".encode()).hexdigest()

    def create_bit_strategy(self, hash_268: str, orbital_shell: OrbitalShell, 
                           portfolio: PortfolioHoldings, prices: Dict[str, float]) -> BITStrategy:
        """
        Create BIT Strategy with randomized holdings based on 268 decimal hash
        
        BIT_Strategy = f(H₂₆₈, orbital_shell, randomized_holdings)
        """
        try:
            # Use hash to seed random number generator for deterministic randomization
            hash_int = int(hash_268[:16], 16)
            random.seed(hash_int)
            
            # Create randomized holdings based on current portfolio
            randomized_holdings = {}
            for asset in ['BTC', 'USDC', 'ETH', 'XRP', 'SOL']:
                current_amount = getattr(portfolio, asset, Decimal('0'))
                
                # Add randomization factor based on hash
                randomization_factor = (hash_int % 1000) / 1000.0  # 0.0 to 1.0
                variation = Decimal(str(randomization_factor * 0.1))  # ±10% variation
                
                # Apply randomization
                if random.choice([True, False]):
                    randomized_amount = current_amount * (Decimal('1') + variation)
                else:
                    randomized_amount = current_amount * (Decimal('1') - variation)
                
                randomized_holdings[asset] = max(Decimal('0'), randomized_amount)
            
            # Calculate confidence score based on orbital shell and hash
            shell_confidence = 1.0 - (orbital_shell.value / 7.0)  # Higher shells = lower confidence
            hash_confidence = (hash_int % 100) / 100.0
            confidence_score = (shell_confidence + hash_confidence) / 2.0
            
            # Calculate profit potential based on orbital shell
            profit_potential = 0.05 - (orbital_shell.value * 0.005)  # 5% to 1.5%
            
            # Calculate risk level based on orbital shell
            risk_level = orbital_shell.value / 7.0  # 0.0 to 1.0
            
            # Calculate execution priority
            execution_priority = int(confidence_score * 100)
            
            return BITStrategy(
                strategy_hash=hash_268,
                orbital_shell=orbital_shell,
                randomized_holdings=randomized_holdings,
                confidence_score=confidence_score,
                profit_potential=profit_potential,
                risk_level=risk_level,
                execution_priority=execution_priority,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error creating BIT strategy: {e}")
            return None

    def update_portfolio_holdings(self, new_holdings: PortfolioHoldings) -> None:
        """Update current portfolio holdings"""
        try:
            self.current_portfolio = new_holdings
            self.portfolio_history.append((new_holdings, time.time()))
            
            # Keep only last 1000 portfolio updates
            if len(self.portfolio_history) > 1000:
                self.portfolio_history.pop(0)
            
            logger.info(f"Portfolio updated: {new_holdings}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio holdings: {e}")

    def get_trading_decision(self, symbol: str, price: float, 
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get trading decision using orbital math, 268 decimal hashing, and BIT strategy
        
        Returns decision for both directions (e.g., BTC/USDC and USDC/BTC)
        """
        try:
            # Get current orbital state
            active_shell = self._get_most_active_shell(market_data)
            orbital_state = self.orbital_states[active_shell]
            
            # Generate 268 decimal hash
            hash_268 = self.generate_268_decimal_hash(price, self.current_portfolio, orbital_state)
            
            # Create BIT strategy
            prices = market_data.get('prices', {})
            bit_strategy = self.create_bit_strategy(hash_268, active_shell, self.current_portfolio, prices)
            
            if not bit_strategy:
                return {"action": "HOLD", "confidence": 0.0, "reason": "BIT strategy creation failed"}
            
            # Get trading pair info
            trading_pair = self.trading_pairs.get(symbol)
            if not trading_pair:
                return {"action": "HOLD", "confidence": 0.0, "reason": "Unknown trading pair"}
            
            # Calculate decision based on BIT strategy and orbital shell
            decision = self._calculate_orbital_decision(bit_strategy, trading_pair, price, market_data)
            
            # Add BIT strategy info to decision
            decision.update({
                "bit_strategy_hash": bit_strategy.strategy_hash,
                "orbital_shell": bit_strategy.orbital_shell.name,
                "randomized_holdings": {k: float(v) for k, v in bit_strategy.randomized_holdings.items()},
                "profit_potential": bit_strategy.profit_potential,
                "risk_level": bit_strategy.risk_level,
                "execution_priority": bit_strategy.execution_priority
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"Error getting trading decision: {e}")
            return {"action": "HOLD", "confidence": 0.0, "reason": f"Error: {str(e)}"}

    def _get_most_active_shell(self, market_data: Dict[str, Any]) -> OrbitalShell:
        """Get the most active orbital shell based on market conditions"""
        try:
            # Calculate shell activations
            shell_activations = {}
            for shell in OrbitalShell:
                activation = self._calculate_shell_activation(shell, market_data)
                shell_activations[shell] = activation
            
            # Return shell with highest activation
            return max(shell_activations.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            logger.error(f"Error getting most active shell: {e}")
            return OrbitalShell.CORE  # Default to CORE shell

    def _calculate_orbital_decision(self, bit_strategy: BITStrategy, trading_pair: TradingPair,
                                  price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate trading decision based on orbital math, BIT strategy, and TECHNICAL INDICATORS
        
        CRITICAL: This method now integrates ALL sophisticated mathematical indicators:
        - RSI (Relative Strength Index) for overbought/oversold detection
        - MACD (Moving Average Convergence Divergence) for momentum
        - Bollinger Bands for volatility and price extremes
        - Moving Averages for trend analysis
        - Volume analysis for confirmation
        - All advanced mathematical systems
        """
        try:
            # Get current holdings for both base and quote assets
            base_holdings = bit_strategy.randomized_holdings.get(trading_pair.base, Decimal('0'))
            quote_holdings = bit_strategy.randomized_holdings.get(trading_pair.quote, Decimal('0'))
            
            # Calculate orbital energy influence
            orbital_energy = bit_strategy.orbital_shell.value / 7.0  # 0.0 to 1.0
            
            # Extract technical indicators from market data
            price_change = market_data.get('price_change', 0.0)
            volatility = market_data.get('volatility', 0.5)
            
            # CRITICAL: Extract technical indicators
            rsi = market_data.get('rsi', 50.0)  # RSI (30 = oversold, 70 = overbought)
            macd_line = market_data.get('macd', 0.0)  # MACD line
            macd_signal = market_data.get('macd_signal', 0.0)  # MACD signal line
            macd_histogram = market_data.get('macd_histogram', 0.0)  # MACD histogram
            bb_position = market_data.get('bb_position', 0.5)  # Bollinger Band position (0-1)
            sma_20 = market_data.get('sma_20', price)  # 20-period SMA
            ema_12 = market_data.get('ema_12', price)  # 12-period EMA
            ema_26 = market_data.get('ema_26', price)  # 26-period EMA
            volume_ratio = market_data.get('volume_ratio', 1.0)  # Volume ratio
            atr = market_data.get('atr', 0.0)  # Average True Range
            
            # CRITICAL: Implement proper "BUY LOW, SELL HIGH" logic with technical indicators
            action = "HOLD"
            confidence = 0.5
            
            # Calculate minimum holdings thresholds
            min_base_holdings = Decimal('0.001') if trading_pair.base == 'BTC' else Decimal('0.01')
            min_quote_holdings = Decimal('10.0')  # Minimum USDC
            
            # Determine trading direction based on current holdings and TECHNICAL INDICATORS
            if bit_strategy.orbital_shell == OrbitalShell.NUCLEUS:
                # Conservative - maintain balance with technical confirmation
                if base_holdings < min_base_holdings and quote_holdings > min_quote_holdings:
                    # Need to buy base asset - check for BUY signals
                    buy_signals = 0
                    total_signals = 0
                    
                    # RSI oversold signal
                    if rsi < 30:
                        buy_signals += 1
                    total_signals += 1
                    
                    # MACD bullish signal
                    if macd_histogram > 0 and macd_line > macd_signal:
                        buy_signals += 1
                    total_signals += 1
                    
                    # Bollinger Band oversold signal
                    if bb_position < 0.2:
                        buy_signals += 1
                    total_signals += 1
                    
                    # Price below moving averages (potential reversal)
                    if price < sma_20 and price < ema_12:
                        buy_signals += 1
                    total_signals += 1
                    
                    # Volume confirmation
                    if volume_ratio > 1.2:
                        buy_signals += 1
                    total_signals += 1
                    
                    # Calculate buy signal strength
                    buy_signal_strength = buy_signals / total_signals if total_signals > 0 else 0
                    
                    if buy_signal_strength > 0.6 and bit_strategy.confidence_score > 0.6:
                        action = "BUY"  # ✅ BUYING LOW with technical confirmation
                        confidence = bit_strategy.confidence_score * buy_signal_strength
                        
                elif base_holdings > min_base_holdings * 2 and quote_holdings < min_quote_holdings:
                    # Need to sell base asset - check for SELL signals
                    sell_signals = 0
                    total_signals = 0
                    
                    # RSI overbought signal
                    if rsi > 70:
                        sell_signals += 1
                    total_signals += 1
                    
                    # MACD bearish signal
                    if macd_histogram < 0 and macd_line < macd_signal:
                        sell_signals += 1
                    total_signals += 1
                    
                    # Bollinger Band overbought signal
                    if bb_position > 0.8:
                        sell_signals += 1
                    total_signals += 1
                    
                    # Price above moving averages (potential reversal)
                    if price > sma_20 and price > ema_12:
                        sell_signals += 1
                    total_signals += 1
                    
                    # Volume confirmation
                    if volume_ratio > 1.2:
                        sell_signals += 1
                    total_signals += 1
                    
                    # Calculate sell signal strength
                    sell_signal_strength = sell_signals / total_signals if total_signals > 0 else 0
                    
                    if sell_signal_strength > 0.6 and bit_strategy.confidence_score > 0.7:
                        action = "SELL"  # ✅ SELLING HIGH with technical confirmation
                        confidence = bit_strategy.confidence_score * sell_signal_strength
                else:
                    action = "HOLD"
                    confidence = 0.7
                    
            elif bit_strategy.orbital_shell == OrbitalShell.CORE:
                # Long-term holds - strategic accumulation with strong technical signals
                if base_holdings < min_base_holdings * 3:  # Need to accumulate base
                    # Strong BUY signals required
                    strong_buy_signals = 0
                    
                    # RSI strongly oversold
                    if rsi < 25:
                        strong_buy_signals += 1
                    
                    # MACD bullish crossover
                    if macd_histogram > 0 and macd_line > macd_signal and macd_histogram > abs(macd_histogram * 0.5):
                        strong_buy_signals += 1
                    
                    # Price significantly below moving averages
                    if price < sma_20 * 0.98 and price < ema_12 * 0.98:
                        strong_buy_signals += 1
                    
                    # Bollinger Band extreme oversold
                    if bb_position < 0.1:
                        strong_buy_signals += 1
                    
                    if strong_buy_signals >= 2 and bit_strategy.confidence_score > 0.8:
                        action = "BUY"  # ✅ BUYING LOW with strong technical signals
                        confidence = bit_strategy.confidence_score
                        
                elif base_holdings > min_base_holdings * 5:  # Have excess base
                    # Strong SELL signals required
                    strong_sell_signals = 0
                    
                    # RSI strongly overbought
                    if rsi > 75:
                        strong_sell_signals += 1
                    
                    # MACD bearish crossover
                    if macd_histogram < 0 and macd_line < macd_signal and abs(macd_histogram) > abs(macd_histogram * 0.5):
                        strong_sell_signals += 1
                    
                    # Price significantly above moving averages
                    if price > sma_20 * 1.02 and price > ema_12 * 1.02:
                        strong_sell_signals += 1
                    
                    # Bollinger Band extreme overbought
                    if bb_position > 0.9:
                        strong_sell_signals += 1
                    
                    if strong_sell_signals >= 2 and bit_strategy.confidence_score > 0.9:
                        action = "SELL"  # ✅ SELLING HIGH with strong technical signals
                        confidence = bit_strategy.confidence_score * 0.9
                else:
                    action = "HOLD"
                    confidence = 0.6
                    
            elif bit_strategy.orbital_shell == OrbitalShell.RELAY:
                # Active trading - frequent rebalancing with technical momentum
                if bit_strategy.confidence_score > 0.7:
                    # Check if we need to rebalance
                    total_value = base_holdings * Decimal(str(price)) + quote_holdings
                    base_value_pct = float((base_holdings * Decimal(str(price)) / total_value) if total_value > 0 else 0)
                    
                    if base_value_pct < 0.3:  # Too much quote, need base
                        # Look for BUY momentum signals
                        momentum_buy_signals = 0
                        
                        # RSI momentum (rising from oversold)
                        if 30 < rsi < 50 and price_change > 0:
                            momentum_buy_signals += 1
                        
                        # MACD momentum (histogram increasing)
                        if macd_histogram > 0 and macd_histogram > market_data.get('prev_macd_histogram', 0):
                            momentum_buy_signals += 1
                        
                        # Price momentum above moving averages
                        if price > ema_12 and ema_12 > ema_26:
                            momentum_buy_signals += 1
                        
                        # Volume momentum
                        if volume_ratio > 1.1:
                            momentum_buy_signals += 1
                        
                        if momentum_buy_signals >= 2:
                            action = "BUY"  # ✅ BUYING LOW with momentum confirmation
                            confidence = bit_strategy.confidence_score
                            
                    elif base_value_pct > 0.7:  # Too much base, need quote
                        # Look for SELL momentum signals
                        momentum_sell_signals = 0
                        
                        # RSI momentum (falling from overbought)
                        if 50 < rsi < 70 and price_change < 0:
                            momentum_sell_signals += 1
                        
                        # MACD momentum (histogram decreasing)
                        if macd_histogram < 0 and macd_histogram < market_data.get('prev_macd_histogram', 0):
                            momentum_sell_signals += 1
                        
                        # Price momentum below moving averages
                        if price < ema_12 and ema_12 < ema_26:
                            momentum_sell_signals += 1
                        
                        # Volume momentum
                        if volume_ratio > 1.1:
                            momentum_sell_signals += 1
                        
                        if momentum_sell_signals >= 2:
                            action = "SELL"  # ✅ SELLING HIGH with momentum confirmation
                            confidence = bit_strategy.confidence_score
                    else:
                        # Balanced - trade based on technical momentum
                        if (rsi < 40 and macd_histogram > 0 and price > ema_12):
                            action = "BUY"  # ✅ BUYING LOW with technical momentum
                            confidence = bit_strategy.confidence_score
                        elif (rsi > 60 and macd_histogram < 0 and price < ema_12):
                            action = "SELL"  # ✅ SELLING HIGH with technical momentum
                            confidence = bit_strategy.confidence_score
                else:
                    action = "HOLD"
                    confidence = 0.4
                    
            elif bit_strategy.orbital_shell == OrbitalShell.FLICKER:
                # Scalping - quick profit taking with precise technical signals
                if bit_strategy.confidence_score > 0.6:
                    # Quick BUY signals
                    if (rsi < 35 and macd_histogram > 0 and bb_position < 0.3):
                        action = "BUY"  # ✅ BUYING LOW for quick scalp
                        confidence = bit_strategy.confidence_score * 0.8
                    # Quick SELL signals
                    elif (rsi > 65 and macd_histogram < 0 and bb_position > 0.7):
                        action = "SELL"  # ✅ SELLING HIGH for quick scalp
                        confidence = bit_strategy.confidence_score * 0.8
                    else:
                        action = "HOLD"
                        confidence = 0.3
                else:
                    action = "HOLD"
                    confidence = 0.3
                    
            else:
                # Other shells - moderate trading with technical confirmation
                if bit_strategy.confidence_score > 0.75:
                    # Check holdings balance with technical signals
                    if base_holdings < min_base_holdings * 2:  # Need more base
                        # Technical BUY signals
                        if (rsi < 45 and macd_histogram > 0 and price < sma_20):
                            action = "BUY"  # ✅ BUYING LOW with technical confirmation
                            confidence = bit_strategy.confidence_score
                    elif base_holdings > min_base_holdings * 4:  # Have excess base
                        # Technical SELL signals
                        if (rsi > 55 and macd_histogram < 0 and price > sma_20):
                            action = "SELL"  # ✅ SELLING HIGH with technical confirmation
                            confidence = bit_strategy.confidence_score
                    else:
                        # Balanced - trade on technical signals
                        if (rsi < 40 and macd_histogram > 0 and bb_position < 0.4):
                            action = "BUY"  # ✅ BUYING LOW with technical signals
                            confidence = bit_strategy.confidence_score
                        elif (rsi > 60 and macd_histogram < 0 and bb_position > 0.6):
                            action = "SELL"  # ✅ SELLING HIGH with technical signals
                            confidence = bit_strategy.confidence_score
                else:
                    action = "HOLD"
                    confidence = 0.4
            
            # CRITICAL: Validate that we have sufficient holdings for the action
            if action == "BUY":
                # For BUY: need sufficient quote currency
                required_quote = Decimal(str(price)) * Decimal('0.01')  # Minimum buy amount
                if quote_holdings < required_quote:
                    action = "HOLD"
                    confidence = 0.3
                    logger.warning(f"Insufficient {trading_pair.quote} for BUY {trading_pair.symbol}")
                    
            elif action == "SELL":
                # For SELL: need sufficient base currency
                if base_holdings < min_base_holdings:
                    action = "HOLD"
                    confidence = 0.3
                    logger.warning(f"Insufficient {trading_pair.base} for SELL {trading_pair.symbol}")
            
            # Calculate position size based on orbital shell and confidence
            position_size = self._calculate_position_size(bit_strategy, action, confidence)
            
            # Add detailed reasoning for debugging with technical indicators
            reasoning = (f"Orbital {bit_strategy.orbital_shell.name} shell: "
                        f"{action} {trading_pair.symbol} "
                        f"(Base: {float(base_holdings):.6f}, Quote: {float(quote_holdings):.2f}, "
                        f"RSI: {rsi:.1f}, MACD: {macd_histogram:.4f}, BB: {bb_position:.2f}, "
                        f"Price change: {price_change:.4f}, Confidence: {bit_strategy.confidence_score:.2f})")
            
            return {
                "action": action,
                "confidence": confidence,
                "position_size": position_size,
                "price": price,
                "symbol": trading_pair.symbol,
                "reverse_symbol": trading_pair.reverse_symbol,
                "orbital_energy": orbital_energy,
                "base_holdings": float(base_holdings),
                "quote_holdings": float(quote_holdings),
                "technical_indicators": {
                    "rsi": rsi,
                    "macd_line": macd_line,
                    "macd_signal": macd_signal,
                    "macd_histogram": macd_histogram,
                    "bb_position": bb_position,
                    "sma_20": sma_20,
                    "ema_12": ema_12,
                    "ema_26": ema_26,
                    "volume_ratio": volume_ratio,
                    "atr": atr
                },
                "reason": reasoning
            }
            
        except Exception as e:
            logger.error(f"Error calculating orbital decision: {e}")
            return {"action": "HOLD", "confidence": 0.0, "reason": f"Calculation error: {str(e)}"}

    def _calculate_position_size(self, bit_strategy: BITStrategy, action: str, confidence: float) -> float:
        """Calculate position size based on orbital shell and confidence"""
        try:
            # Base position size from orbital shell
            base_size = 0.1 - (bit_strategy.orbital_shell.value * 0.01)  # 10% to 3%
            
            # Adjust for confidence
            confidence_multiplier = confidence
            
            # Adjust for risk level
            risk_multiplier = 1.0 - bit_strategy.risk_level
            
            # Calculate final position size
            position_size = base_size * confidence_multiplier * risk_multiplier
            
            # Ensure within bounds
            position_size = max(0.01, min(0.2, position_size))  # 1% to 20%
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.05  # Default 5%

    def initialize_orbital_shells(self) -> None:
        """Initialize all 8 orbital shells with quantum states"""
        shell_configs = {
            OrbitalShell.NUCLEUS: {
                "energy_base": -13.6,
                "risk_tolerance": 0.5,
                "allocation_limit": 0.3,
                "assets": {"USDC": 0.7, "BTC": 0.3},
            },
            OrbitalShell.CORE: {
                "energy_base": -3.4,
                "risk_tolerance": 0.1,
                "allocation_limit": 0.2,
                "assets": {"BTC": 0.8, "ETH": 0.2},
            },
            OrbitalShell.HOLD: {
                "energy_base": -1.5,
                "risk_tolerance": 0.3,
                "allocation_limit": 0.25,
                "assets": {"BTC": 0.6, "ETH": 0.3, "XRP": 0.1},
            },
            OrbitalShell.SCOUT: {
                "energy_base": -0.85,
                "risk_tolerance": 0.6,
                "allocation_limit": 0.15,
                "assets": {"BTC": 0.4, "ETH": 0.4, "SOL": 0.2},
            },
            OrbitalShell.FEEDER: {
                "energy_base": -0.54,
                "risk_tolerance": 0.7,
                "allocation_limit": 0.1,
                "assets": {"BTC": 0.3, "ETH": 0.3, "XRP": 0.2, "SOL": 0.2},
            },
            OrbitalShell.RELAY: {
                "energy_base": -0.38,
                "risk_tolerance": 0.8,
                "allocation_limit": 0.05,
                "assets": {"BTC": 0.2, "ETH": 0.2, "XRP": 0.3, "SOL": 0.3},
            },
            OrbitalShell.FLICKER: {
                "energy_base": -0.28,
                "risk_tolerance": 0.9,
                "allocation_limit": 0.03,
                "assets": {"XRP": 0.4, "SOL": 0.4, "BTC": 0.1, "ETH": 0.1},
            },
            OrbitalShell.GHOST: {
                "energy_base": -0.21,
                "risk_tolerance": 1.0,
                "allocation_limit": 0.02,
                "assets": {"SOL": 0.5, "XRP": 0.3, "ETH": 0.1, "BTC": 0.1},
            },
        }

        for shell, config in shell_configs.items():
            # Initialize quantum state
            orbital_state = OrbitalState(
                shell=shell,
                radial_probability=1.0,
                angular_momentum=(0.0, 0.0),
                energy_level=config["energy_base"],
                time_evolution=1.0 + 0j,
                confidence=0.5,
                asset_allocation=config["assets"].copy(),
                current_holdings=PortfolioHoldings(), # Initialize with empty holdings
            )
            self.orbital_states[shell] = orbital_state

            # Initialize memory tensor
            memory_tensor = ShellMemoryTensor(
                shell=shell,
                memory_vector=np.zeros(100),
                entry_history=[],
                exit_history=[],
                pnl_history=[],
                volatility_history=[],
                fractal_match_history=[],
                last_update=time.time(),
            )
            self.shell_memory_tensors[shell] = memory_tensor

    def _initialize_profit_buckets(self) -> List[ProfitTierBucket]:
        """Initialize profit tier buckets"""
        return [
            ProfitTierBucket(0, (-float('inf'), -0.1), 0.05, None, 0.5, 0.8, False, True),
            ProfitTierBucket(1, (-0.1, 0.0), 0.03, 0.02, 0.7, 0.6, True, True),
            ProfitTierBucket(2, (0.0, 0.05), 0.02, 0.03, 1.0, 0.4, True, False),
            ProfitTierBucket(3, (0.05, 0.15), 0.015, 0.05, 1.2, 0.3, True, False),
            ProfitTierBucket(4, (0.15, float('inf')), 0.01, 0.1, 1.5, 0.2, True, False),
        ]

    def calculate_orbital_wavefunction(self, shell: OrbitalShell, t: float, r: float) -> complex:
        """Calculate orbital wavefunction ψₙ(t,r)"""
        state = self.orbital_states[shell]
        energy = state.energy_level
        h_bar = self.config["h_bar"]
        
        # ψₙ(t,r) = Rₙ(r) · Yₙ(θ,φ) · e^(-iEₙt/ħ)
        radial_part = state.radial_probability
        angular_part = complex(state.angular_momentum[0], state.angular_momentum[1])
        time_part = np.exp(-1j * energy * t / h_bar)
        
        return radial_part * angular_part * time_part

    def calculate_shell_energy(self, shell: OrbitalShell, volatility: float, drift_rate: float) -> float:
        """Calculate shell energy Eₙ = -(k²/2n²) + λ·σₙ² - μ·∂Rₙ/∂t"""
        n = shell.value + 1  # Shell quantum number
        k = self.config["k_constant"]
        lambda_vol = self.config["lambda_volatility"]
        mu = self.config["mu_reaction"]
        
        energy = -(k**2 / (2 * n**2)) + lambda_vol * volatility**2 - mu * drift_rate
        return energy

    def calculate_altitude_vector(self, market_data: Dict[str, Any]) -> AltitudeVector:
        """Calculate altitude vector ℵₐ(t) = ∇ψₜ + ρ(t)·εₜ - ∂Φ/∂t"""
        try:
            # Extract market data
            price_change = market_data.get("price_change", 0.0)
            volume_change = market_data.get("volume_change", 0.0)
            volatility = market_data.get("volatility", 0.5)
            
            # Calculate components
            momentum_curvature = price_change * volume_change  # ∇ψₜ
            rolling_return = np.mean([price_change, volume_change])  # ρ(t)
            entropy_shift = volatility * (1 - abs(price_change))  # εₜ
            alpha_decay = -volatility * abs(price_change)  # ∂Φ/∂t
            
            # Calculate altitude value
            altitude_value = momentum_curvature + rolling_return * entropy_shift + alpha_decay
            confidence_level = min(1.0, max(0.0, 1.0 - abs(altitude_value)))
            
            return AltitudeVector(
                momentum_curvature=momentum_curvature,
                rolling_return=rolling_return,
                entropy_shift=entropy_shift,
                alpha_decay=alpha_decay,
                altitude_value=altitude_value,
                confidence_level=confidence_level
            )
        except Exception as e:
            logger.error(f"Error calculating altitude vector: {e}")
            return AltitudeVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.5)

    def _calculate_shell_activation(self, shell: OrbitalShell, market_data: Dict[str, Any]) -> float:
        """Calculate shell activation Ψₛ based on market conditions"""
        try:
            # Get shell state
            state = self.orbital_states[shell]
            
            # Extract market conditions
            price_change = market_data.get("price_change", 0.0)
            volatility = market_data.get("volatility", 0.5)
            volume_change = market_data.get("volume_change", 0.0)
            
            # Calculate activation based on shell characteristics
            shell_risk = shell.value / 7.0  # Normalized risk (0-1)
            
            # Activation formula: Ψₛ = f(price_change, volatility, shell_risk)
            price_activation = np.tanh(price_change * 10)  # Price momentum
            volatility_activation = np.tanh(volatility * 2)  # Volatility response
            volume_activation = np.tanh(volume_change * 5)  # Volume response
            
            # Combine activations with shell-specific weighting
            activation = (
                0.4 * price_activation +
                0.3 * volatility_activation +
                0.3 * volume_activation
            ) * (1.0 + shell_risk * 0.5)  # Higher risk shells get boost
            
            # Normalize to [0, 1]
            activation = min(1.0, max(0.0, activation))
            
            return float(activation)
            
        except Exception as e:
            logger.error(f"Error calculating shell activation: {e}")
            return 0.5

    def calculate_neural_shell_confidence(self, shell: OrbitalShell, memory_tensor: ShellMemoryTensor) -> float:
        """Calculate neural confidence for shell using BRAIN weights"""
        try:
            # Get shell index
            shell_idx = shell.value
            
            # Extract features from memory tensor
            recent_pnl = memory_tensor.pnl_history[-10:] if memory_tensor.pnl_history else [0.0]
            recent_vol = memory_tensor.volatility_history[-10:] if memory_tensor.volatility_history else [0.5]
            
            # Create feature vector
            features = np.array([
                np.mean(recent_pnl),
                np.std(recent_pnl),
                np.mean(recent_vol),
                np.std(recent_vol),
                memory_tensor.last_update,
                shell_idx,
            ])
            
            # Pad to 64 dimensions
            features_padded = np.pad(features, (0, 64 - len(features)), 'constant')
            
            # Apply neural weights
            confidence = np.dot(self.neural_shell_weights[shell_idx], features_padded)
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating neural shell confidence: {e}")
            return 0.5

    def calculate_shell_consensus(self, market_data: Dict[str, Any]) -> ShellConsensus:
        """Calculate shell consensus 𝒞ₛ = Σ(Ψₛ · Θₛ · ωₛ)"""
        try:
            shell_activations = {}
            shell_confidences = {}
            shell_weights = {}
            active_shells = []
            
            # Calculate for each shell
            for shell in OrbitalShell:
                # Activation Ψₛ
                activation = self._calculate_shell_activation(shell, market_data)
                shell_activations[shell] = activation
                
                # Confidence Θₛ
                memory_tensor = self.shell_memory_tensors[shell]
                confidence = self.calculate_neural_shell_confidence(shell, memory_tensor)
                shell_confidences[shell] = confidence
                
                # Weight ωₛ (based on shell level)
                weight = 1.0 / (shell.value + 1)  # Lower shells have higher weight
                shell_weights[shell] = weight
                
                # Check if shell is active
                if activation > 0.1:
                    active_shells.append(shell)
            
            # Calculate consensus score
            consensus_score = sum(
                shell_activations[shell] * shell_confidences[shell] * shell_weights[shell]
                for shell in OrbitalShell
            )
            
            # Check threshold
            threshold_met = consensus_score >= self.config["consensus_threshold"]
            
            return ShellConsensus(
                consensus_score=consensus_score,
                active_shells=active_shells,
                shell_activations=shell_activations,
                shell_confidences=shell_confidences,
                shell_weights=shell_weights,
                threshold_met=threshold_met
            )
            
        except Exception as e:
            logger.error(f"Error calculating shell consensus: {e}")
            return ShellConsensus(0.0, [], {}, {}, {}, False)

    def calculate_profit_tier_bucket(
        self, pnl: float, altitude: AltitudeVector, consensus: ShellConsensus
    ) -> ProfitTierBucket:
        """Calculate appropriate profit tier bucket"""
        for bucket in self.profit_buckets:
            if bucket.profit_range[0] <= pnl < bucket.profit_range[1]:
                return bucket
        return self.profit_buckets[0]  # Default to lowest tier

    def ferris_rotation_cycle(self, market_data: Dict[str, Any]) -> None:
        """Execute Ferris rotation cycle for orbital shells"""
        try:
            with self.system_lock:
                # Calculate altitude vector
                altitude = self.calculate_altitude_vector(market_data)
                self.current_altitude_vector = altitude
                
                # Calculate shell consensus
                consensus = self.calculate_shell_consensus(market_data)
                self.current_shell_consensus = consensus
                
                # Update shell energies
                for shell in OrbitalShell:
                    volatility = market_data.get("volatility", 0.5)
                    drift_rate = market_data.get("price_change", 0.0)
                    energy = self.calculate_shell_energy(shell, volatility, drift_rate)
                    self.orbital_states[shell].energy_level = energy
                
                # Update memory tensors
                self._update_shell_memory_tensors(market_data)
                
        except Exception as e:
            logger.error(f"Error in Ferris rotation cycle: {e}")

    def _move_asset_to_shell_inward(self, shell: OrbitalShell) -> None:
        """Move assets inward to more conservative shell"""
        pass  # Implementation for asset movement

    def _move_asset_to_shell_outward(self, shell: OrbitalShell) -> None:
        """Move assets outward to more aggressive shell"""
        pass  # Implementation for asset movement

    def _transfer_shell_allocation(self, from_shell: OrbitalShell, to_shell: OrbitalShell, ratio: float) -> None:
        """Transfer allocation between shells"""
        try:
            from_state = self.orbital_states[from_shell]
            to_state = self.orbital_states[to_shell]
            
            # Transfer assets
            for asset, amount in from_state.asset_allocation.items():
                transfer_amount = amount * ratio
                from_state.asset_allocation[asset] -= transfer_amount
                to_state.asset_allocation[asset] = to_state.asset_allocation.get(asset, 0) + transfer_amount
                
        except Exception as e:
            logger.error(f"Error transferring shell allocation: {e}")

    def encode_shell_dna(self, shell: OrbitalShell) -> str:
        """Encode shell DNA for persistence"""
        try:
            state = self.orbital_states[shell]
            memory = self.shell_memory_tensors[shell]
            
            dna_data = {
                "shell": shell.value,
                "energy": state.energy_level,
                "confidence": state.confidence,
                "allocation": state.asset_allocation,
                "memory_size": len(memory.memory_vector),
                "last_update": memory.last_update,
            }
            
            dna_string = json.dumps(dna_data, sort_keys=True)
            return hashlib.sha256(dna_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error encoding shell DNA: {e}")
            return ""

    def start_orbital_brain_system(self) -> None:
        """Start the orbital brain system"""
        if not self.active:
            self.active = True
            self.rotation_thread = threading.Thread(target=self._orbital_brain_loop, daemon=True)
            self.rotation_thread.start()
            logger.info("🧠⚛️ Orbital BRAIN System started")

    def stop_orbital_brain_system(self) -> None:
        """Stop the orbital brain system"""
        self.active = False
        if self.rotation_thread:
            self.rotation_thread.join(timeout=5.0)
        logger.info("🧠⚛️ Orbital BRAIN System stopped")

    def _orbital_brain_loop(self) -> None:
        """Main orbital brain loop"""
        while self.active:
            try:
                # Get market data
                market_data = self._get_simulated_market_data()
                
                # Execute Ferris rotation cycle
                self.ferris_rotation_cycle(market_data)
                
                # Sleep for rotation interval
                time.sleep(self.config["rotation_interval"])
                
            except Exception as e:
                logger.error(f"Error in orbital brain loop: {e}")
                time.sleep(10.0)  # Brief pause on error

    def _get_simulated_market_data(self) -> Dict[str, Any]:
        """Get simulated market data for testing"""
        return {
            "price_change": np.random.normal(0.0, 0.02),
            "volume_change": np.random.normal(0.0, 0.1),
            "volatility": np.random.uniform(0.1, 0.5),
            "timestamp": time.time(),
        }

    def _update_shell_memory_tensors(self, market_data: Dict[str, Any]) -> None:
        """Update shell memory tensors with new market data"""
        try:
            for shell in OrbitalShell:
                memory = self.shell_memory_tensors[shell]
                
                # Update memory vector (rolling window)
                memory.memory_vector = np.roll(memory.memory_vector, -1)
                memory.memory_vector[-1] = market_data.get("price_change", 0.0)
                
                # Update last update time
                memory.last_update = time.time()
                
        except Exception as e:
            logger.error(f"Error updating shell memory tensors: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "active": self.active,
            "orbital_states": {shell.name: {
                "energy": state.energy_level,
                "confidence": state.confidence,
                "allocation": state.asset_allocation
            } for shell, state in self.orbital_states.items()},
            "altitude_vector": {
                "value": self.current_altitude_vector.altitude_value if self.current_altitude_vector else 0.0,
                "confidence": self.current_altitude_vector.confidence_level if self.current_altitude_vector else 0.0,
            } if self.current_altitude_vector else None,
            "shell_consensus": {
                "score": self.current_shell_consensus.consensus_score if self.current_shell_consensus else 0.0,
                "threshold_met": self.current_shell_consensus.threshold_met if self.current_shell_consensus else False,
                "active_shells": [shell.name for shell in (self.current_shell_consensus.active_shells if self.current_shell_consensus else [])],
            } if self.current_shell_consensus else None,
        }


# Global instance for easy access
orbital_brain_system = OrbitalBRAINSystem()
