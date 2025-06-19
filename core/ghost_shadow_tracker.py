"""
Ghost Shadow Tracker - Comprehensive Ghost Signal Analytics
==========================================================

Tracks all ghost signals (executed and unexecuted) to enable post-match analysis,
opportunity discovery, and performance attribution. Implements comprehensive
mathematical models for shadow profit analysis and missed opportunity scoring.

Mathematical Framework:
- Shadow profit calculation using counterfactual analysis
- Opportunity cost modeling with risk-adjusted returns
- Pattern clustering for missed signal identification
- Performance attribution across signal layers and market conditions
"""

import numpy as np
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import sqlite3
import pandas as pd
from pathlib import Path

class ShadowSignalType(Enum):
    """Types of shadow signals tracked"""
    UNEXECUTED_GHOST = "unexecuted_ghost"      # Ghost signal that wasn't triggered
    EXECUTED_GHOST = "executed_ghost"          # Ghost signal that was executed
    FILTERED_SIGNAL = "filtered_signal"        # Signal filtered by risk management
    TIMEOUT_SIGNAL = "timeout_signal"          # Signal that timed out
    CONFIDENCE_REJECT = "confidence_reject"    # Signal rejected due to low confidence

class OutcomeClassification(Enum):
    """Classification of signal outcomes"""
    PROFITABLE_MISS = "profitable_miss"        # Would have been profitable
    LOSS_AVOIDANCE = "loss_avoidance"         # Correctly avoided a loss
    NEUTRAL_MISS = "neutral_miss"             # Minimal impact
    UNCERTAIN = "uncertain"                    # Insufficient data to classify

@dataclass
class ShadowGhostSignal:
    """Complete ghost signal record for shadow analysis"""
    # Basic signal information (required)
    signal_id: str
    ghost_hash: str
    timestamp: float
    signal_type: ShadowSignalType
    confidence_score: float
    layer_contributions: Dict[str, float]
    market_conditions: Dict[str, Any]
    was_triggered: bool
    price_at_signal: float
    volume_at_signal: float
    spread_at_signal: float
    
    # Optional fields with defaults
    trigger_reason: Optional[str] = None
    rejection_reason: Optional[str] = None
    order_book_snapshot: Optional[Dict[str, Any]] = None
    
    # Post-signal tracking
    price_1h_later: Optional[float] = None
    price_4h_later: Optional[float] = None
    price_24h_later: Optional[float] = None
    realized_volatility: Optional[float] = None
    
    # Performance attribution
    shadow_profit_1h: Optional[float] = None
    shadow_profit_4h: Optional[float] = None
    shadow_profit_24h: Optional[float] = None
    opportunity_cost: Optional[float] = None
    outcome_classification: Optional[OutcomeClassification] = None
    
    # Execution details (if triggered)
    execution_price: Optional[float] = None
    execution_quantity: Optional[float] = None
    actual_profit_loss: Optional[float] = None
    execution_latency_ms: Optional[float] = None

@dataclass
class ShadowAnalysisResult:
    """Result of shadow signal analysis"""
    analysis_period: Tuple[float, float]  # (start_timestamp, end_timestamp)
    total_signals: int
    executed_signals: int
    missed_opportunities: int
    correctly_avoided_losses: int
    
    # Performance metrics
    shadow_profit_total: float
    actual_profit_total: float
    opportunity_cost_total: float
    hit_rate: float                    # % of profitable executions
    shadow_hit_rate: float             # % of missed profitable opportunities
    
    # Layer attribution
    layer_performance: Dict[str, Dict[str, float]]
    market_condition_performance: Dict[str, Dict[str, float]]
    
    # Patterns and insights
    missed_patterns: List[Dict[str, Any]]
    optimization_recommendations: List[str]

class GhostShadowTracker:
    """Comprehensive ghost signal tracking and analysis system"""
    
    def __init__(self, db_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.db_path = db_path or "ghost_shadow_tracker.db"
        self.config = config or self._get_default_config()
        self.current_signals: Dict[str, ShadowGhostSignal] = {}
        self.price_tracker: List[Tuple[float, float]] = []  # (timestamp, price)
        
        # Initialize database
        self._initialize_database()
        
        # Analysis cache
        self.analysis_cache: Dict[str, ShadowAnalysisResult] = {}
        self.pattern_clusters: List[Dict[str, Any]] = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for shadow tracking"""
        return {
            'tracking_horizon_hours': 24,
            'min_confidence_for_analysis': 0.3,
            'opportunity_cost_threshold': 0.01,  # 1% minimum opportunity
            'volatility_adjustment_factor': 1.5,
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'pattern_similarity_threshold': 0.75,
            'analysis_batch_size': 1000,
            'max_db_records': 100000
        }
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for shadow signal storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create shadow signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shadow_signals (
                signal_id TEXT PRIMARY KEY,
                ghost_hash TEXT,
                timestamp REAL,
                signal_type TEXT,
                confidence_score REAL,
                layer_contributions TEXT,  -- JSON
                market_conditions TEXT,    -- JSON
                was_triggered BOOLEAN,
                trigger_reason TEXT,
                rejection_reason TEXT,
                price_at_signal REAL,
                volume_at_signal REAL,
                spread_at_signal REAL,
                order_book_snapshot TEXT,  -- JSON
                price_1h_later REAL,
                price_4h_later REAL,
                price_24h_later REAL,
                realized_volatility REAL,
                shadow_profit_1h REAL,
                shadow_profit_4h REAL,
                shadow_profit_24h REAL,
                opportunity_cost REAL,
                outcome_classification TEXT,
                execution_price REAL,
                execution_quantity REAL,
                actual_profit_loss REAL,
                execution_latency_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indices for efficient querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON shadow_signals(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ghost_hash ON shadow_signals(ghost_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_type ON shadow_signals(signal_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcome ON shadow_signals(outcome_classification)')
        
        conn.commit()
        conn.close()
    
    def log_ghost_signal(self, signal: ShadowGhostSignal) -> None:
        """Log a ghost signal for shadow tracking"""
        
        # Store in current signals for tracking
        self.current_signals[signal.signal_id] = signal
        
        # Insert into database
        self._insert_signal_to_db(signal)
        
        # Clean up old current signals
        current_time = time.time()
        tracking_horizon = self.config['tracking_horizon_hours'] * 3600
        expired_signals = [
            sid for sid, sig in self.current_signals.items()
            if current_time - sig.timestamp > tracking_horizon
        ]
        
        for sid in expired_signals:
            del self.current_signals[sid]
    
    def update_price_data(self, timestamp: float, price: float) -> None:
        """Update price data for shadow profit calculations"""
        self.price_tracker.append((timestamp, price))
        
        # Maintain reasonable history size
        max_history = 10000
        if len(self.price_tracker) > max_history:
            self.price_tracker = self.price_tracker[-max_history//2:]
        
        # Update shadow profits for active signals
        self._update_shadow_profits()
    
    def _update_shadow_profits(self) -> None:
        """Update shadow profit calculations for active signals"""
        current_time = time.time()
        
        for signal_id, signal in self.current_signals.items():
            # Skip if already fully analyzed
            if signal.shadow_profit_24h is not None:
                continue
            
            time_since_signal = current_time - signal.timestamp
            
            # Update profits at different time horizons
            if time_since_signal >= 3600 and signal.shadow_profit_1h is None:  # 1 hour
                price_1h = self._get_price_at_time(signal.timestamp + 3600)
                if price_1h:
                    signal.price_1h_later = price_1h
                    signal.shadow_profit_1h = self._calculate_shadow_profit(
                        signal, price_1h, signal.price_at_signal
                    )
            
            if time_since_signal >= 14400 and signal.shadow_profit_4h is None:  # 4 hours
                price_4h = self._get_price_at_time(signal.timestamp + 14400)
                if price_4h:
                    signal.price_4h_later = price_4h
                    signal.shadow_profit_4h = self._calculate_shadow_profit(
                        signal, price_4h, signal.price_at_signal
                    )
            
            if time_since_signal >= 86400 and signal.shadow_profit_24h is None:  # 24 hours
                price_24h = self._get_price_at_time(signal.timestamp + 86400)
                if price_24h:
                    signal.price_24h_later = price_24h
                    signal.shadow_profit_24h = self._calculate_shadow_profit(
                        signal, price_24h, signal.price_at_signal
                    )
                    
                    # Calculate realized volatility
                    signal.realized_volatility = self._calculate_realized_volatility(
                        signal.timestamp, signal.timestamp + 86400
                    )
                    
                    # Calculate opportunity cost
                    signal.opportunity_cost = self._calculate_opportunity_cost(signal)
                    
                    # Classify outcome
                    signal.outcome_classification = self._classify_outcome(signal)
                    
                    # Update database
                    self._update_signal_in_db(signal)
    
    def _get_price_at_time(self, target_timestamp: float) -> Optional[float]:
        """Get price closest to target timestamp"""
        if not self.price_tracker:
            return None
        
        # Find closest price data point
        closest_idx = min(
            range(len(self.price_tracker)),
            key=lambda i: abs(self.price_tracker[i][0] - target_timestamp)
        )
        
        timestamp, price = self.price_tracker[closest_idx]
        
        # Only return if within reasonable time window (5 minutes)
        if abs(timestamp - target_timestamp) <= 300:
            return price
        
        return None
    
    def _calculate_shadow_profit(self, signal: ShadowGhostSignal, 
                               exit_price: float, entry_price: float) -> float:
        """Calculate shadow profit for a ghost signal"""
        
        # Determine trade direction from signal characteristics
        # This is simplified - in practice, would use signal's intended direction
        price_change = (exit_price - entry_price) / entry_price
        
        # Assume ghost signals are momentum-based (profit from price moves)
        # Adjust for confidence and layer contributions
        confidence_multiplier = signal.confidence_score
        
        # Layer-based direction hints (simplified logic)
        geometric_weight = signal.layer_contributions.get('geometric', 0.25)
        smart_money_weight = signal.layer_contributions.get('smart_money', 0.25)
        
        # Geometric patterns typically profit from continuations
        # Smart money patterns profit from reversals
        if geometric_weight > smart_money_weight:
            direction_multiplier = 1.0  # Follow momentum
        else:
            direction_multiplier = -1.0  # Counter-momentum
        
        # Calculate base profit
        base_profit = price_change * direction_multiplier * confidence_multiplier
        
        # Adjust for transaction costs (simplified)
        transaction_cost = 0.001  # 0.1% total costs
        net_profit = base_profit - transaction_cost
        
        return net_profit
    
    def _calculate_realized_volatility(self, start_time: float, end_time: float) -> float:
        """Calculate realized volatility over time period"""
        
        # Get price data within time range
        period_prices = [
            price for timestamp, price in self.price_tracker
            if start_time <= timestamp <= end_time
        ]
        
        if len(period_prices) < 2:
            return 0.0
        
        # Calculate returns
        returns = [
            period_prices[i] / period_prices[i-1] - 1
            for i in range(1, len(period_prices))
        ]
        
        # Calculate realized volatility (annualized)
        if returns:
            daily_vol = np.std(returns)
            # Approximate annualization (assuming hourly data)
            hours_per_year = 365 * 24
            periods_per_hour = len(returns) / ((end_time - start_time) / 3600)
            annualized_vol = daily_vol * np.sqrt(hours_per_year / periods_per_hour)
            return annualized_vol
        
        return 0.0
    
    def _calculate_opportunity_cost(self, signal: ShadowGhostSignal) -> float:
        """Calculate opportunity cost of not executing signal"""
        
        if signal.was_triggered and signal.actual_profit_loss is not None:
            # Opportunity cost is difference between potential and actual
            potential_profit = signal.shadow_profit_24h or 0.0
            actual_profit = signal.actual_profit_loss
            return potential_profit - actual_profit
        
        elif not signal.was_triggered:
            # Opportunity cost is the foregone profit
            potential_profit = signal.shadow_profit_24h or 0.0
            
            # Adjust for risk (higher volatility = higher risk discount)
            risk_adjustment = 1.0
            if signal.realized_volatility:
                vol_factor = self.config['volatility_adjustment_factor']
                risk_adjustment = 1.0 / (1.0 + signal.realized_volatility * vol_factor)
            
            return potential_profit * risk_adjustment
        
        return 0.0
    
    def _classify_outcome(self, signal: ShadowGhostSignal) -> OutcomeClassification:
        """Classify the outcome of a ghost signal"""
        
        threshold = self.config['opportunity_cost_threshold']
        
        if signal.was_triggered:
            # For executed signals, compare actual vs potential
            if signal.actual_profit_loss and signal.shadow_profit_24h:
                if signal.shadow_profit_24h > signal.actual_profit_loss + threshold:
                    return OutcomeClassification.PROFITABLE_MISS  # Could have done better
                else:
                    return OutcomeClassification.NEUTRAL_MISS
        
        else:
            # For unexecuted signals, check if we missed profit
            if signal.shadow_profit_24h and signal.shadow_profit_24h > threshold:
                return OutcomeClassification.PROFITABLE_MISS
            elif signal.shadow_profit_24h and signal.shadow_profit_24h < -threshold:
                return OutcomeClassification.LOSS_AVOIDANCE
            else:
                return OutcomeClassification.NEUTRAL_MISS
        
        return OutcomeClassification.UNCERTAIN
    
    def analyze_shadow_performance(self, start_time: Optional[float] = None,
                                 end_time: Optional[float] = None,
                                 signal_types: Optional[List[ShadowSignalType]] = None) -> ShadowAnalysisResult:
        """Perform comprehensive shadow performance analysis"""
        
        # Set default time range (last 7 days)
        if end_time is None:
            end_time = time.time()
        if start_time is None:
            start_time = end_time - (7 * 24 * 3600)
        
        # Query database for signals in time range
        signals = self._query_signals_from_db(start_time, end_time, signal_types)
        
        if not signals:
            return ShadowAnalysisResult(
                analysis_period=(start_time, end_time),
                total_signals=0, executed_signals=0, missed_opportunities=0,
                correctly_avoided_losses=0, shadow_profit_total=0.0,
                actual_profit_total=0.0, opportunity_cost_total=0.0,
                hit_rate=0.0, shadow_hit_rate=0.0,
                layer_performance={}, market_condition_performance={},
                missed_patterns=[], optimization_recommendations=[]
            )
        
        # Calculate aggregate metrics
        total_signals = len(signals)
        executed_signals = sum(1 for s in signals if s.was_triggered)
        
        # Count outcome classifications
        profitable_misses = sum(1 for s in signals 
                              if s.outcome_classification == OutcomeClassification.PROFITABLE_MISS)
        loss_avoidances = sum(1 for s in signals 
                            if s.outcome_classification == OutcomeClassification.LOSS_AVOIDANCE)
        
        # Calculate profit totals
        shadow_profit_total = sum(s.shadow_profit_24h or 0.0 for s in signals)
        actual_profit_total = sum(s.actual_profit_loss or 0.0 for s in signals if s.was_triggered)
        opportunity_cost_total = sum(s.opportunity_cost or 0.0 for s in signals)
        
        # Calculate hit rates
        executed_profitable = sum(1 for s in signals 
                                if s.was_triggered and (s.actual_profit_loss or 0) > 0)
        hit_rate = executed_profitable / executed_signals if executed_signals > 0 else 0.0
        
        missed_profitable = sum(1 for s in signals 
                              if not s.was_triggered and (s.shadow_profit_24h or 0) > 0)
        unexecuted_signals = total_signals - executed_signals
        shadow_hit_rate = missed_profitable / unexecuted_signals if unexecuted_signals > 0 else 0.0
        
        # Analyze layer performance
        layer_performance = self._analyze_layer_performance(signals)
        
        # Analyze market condition performance
        market_condition_performance = self._analyze_market_condition_performance(signals)
        
        # Identify missed patterns
        missed_patterns = self._identify_missed_patterns(signals)
        
        # Generate optimization recommendations
        optimization_recommendations = self._generate_optimization_recommendations(
            signals, layer_performance, market_condition_performance
        )
        
        return ShadowAnalysisResult(
            analysis_period=(start_time, end_time),
            total_signals=total_signals,
            executed_signals=executed_signals,
            missed_opportunities=profitable_misses,
            correctly_avoided_losses=loss_avoidances,
            shadow_profit_total=shadow_profit_total,
            actual_profit_total=actual_profit_total,
            opportunity_cost_total=opportunity_cost_total,
            hit_rate=hit_rate,
            shadow_hit_rate=shadow_hit_rate,
            layer_performance=layer_performance,
            market_condition_performance=market_condition_performance,
            missed_patterns=missed_patterns,
            optimization_recommendations=optimization_recommendations
        )
    
    def _analyze_layer_performance(self, signals: List[ShadowGhostSignal]) -> Dict[str, Dict[str, float]]:
        """Analyze performance attribution by signal layer"""
        
        layers = ['geometric', 'smart_money', 'depth', 'timeband']
        layer_performance = {}
        
        for layer in layers:
            # Group signals by layer dominance
            layer_dominant_signals = [
                s for s in signals
                if max(s.layer_contributions.items(), key=lambda x: x[1])[0] == layer
            ]
            
            if layer_dominant_signals:
                executed_count = sum(1 for s in layer_dominant_signals if s.was_triggered)
                profitable_count = sum(1 for s in layer_dominant_signals 
                                     if (s.actual_profit_loss or 0) > 0 or (s.shadow_profit_24h or 0) > 0)
                
                avg_confidence = np.mean([s.confidence_score for s in layer_dominant_signals])
                avg_shadow_profit = np.mean([s.shadow_profit_24h or 0 for s in layer_dominant_signals])
                avg_opportunity_cost = np.mean([s.opportunity_cost or 0 for s in layer_dominant_signals])
                
                layer_performance[layer] = {
                    'signal_count': len(layer_dominant_signals),
                    'execution_rate': executed_count / len(layer_dominant_signals),
                    'profitability_rate': profitable_count / len(layer_dominant_signals),
                    'avg_confidence': avg_confidence,
                    'avg_shadow_profit': avg_shadow_profit,
                    'avg_opportunity_cost': avg_opportunity_cost
                }
            else:
                layer_performance[layer] = {
                    'signal_count': 0, 'execution_rate': 0.0, 'profitability_rate': 0.0,
                    'avg_confidence': 0.0, 'avg_shadow_profit': 0.0, 'avg_opportunity_cost': 0.0
                }
        
        return layer_performance
    
    def _analyze_market_condition_performance(self, signals: List[ShadowGhostSignal]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by market conditions"""
        
        # Group signals by volatility regime
        low_vol_signals = [s for s in signals 
                          if s.market_conditions.get('volatility_percentile', 50) < 33]
        med_vol_signals = [s for s in signals 
                          if 33 <= s.market_conditions.get('volatility_percentile', 50) < 67]
        high_vol_signals = [s for s in signals 
                           if s.market_conditions.get('volatility_percentile', 50) >= 67]
        
        condition_performance = {}
        
        for regime, regime_signals in [('low_volatility', low_vol_signals),
                                     ('medium_volatility', med_vol_signals),
                                     ('high_volatility', high_vol_signals)]:
            if regime_signals:
                profitable_count = sum(1 for s in regime_signals 
                                     if (s.shadow_profit_24h or 0) > 0)
                avg_profit = np.mean([s.shadow_profit_24h or 0 for s in regime_signals])
                avg_opportunity_cost = np.mean([s.opportunity_cost or 0 for s in regime_signals])
                
                condition_performance[regime] = {
                    'signal_count': len(regime_signals),
                    'profitability_rate': profitable_count / len(regime_signals),
                    'avg_profit': avg_profit,
                    'avg_opportunity_cost': avg_opportunity_cost
                }
            else:
                condition_performance[regime] = {
                    'signal_count': 0, 'profitability_rate': 0.0,
                    'avg_profit': 0.0, 'avg_opportunity_cost': 0.0
                }
        
        return condition_performance
    
    def _identify_missed_patterns(self, signals: List[ShadowGhostSignal]) -> List[Dict[str, Any]]:
        """Identify patterns in missed profitable opportunities"""
        
        # Focus on profitable misses
        profitable_misses = [
            s for s in signals
            if s.outcome_classification == OutcomeClassification.PROFITABLE_MISS
        ]
        
        if not profitable_misses:
            return []
        
        # Cluster by similar characteristics
        patterns = []
        
        # Pattern 1: High confidence but rejected
        high_confidence_rejects = [
            s for s in profitable_misses
            if s.confidence_score > 0.8 and not s.was_triggered
        ]
        
        if high_confidence_rejects:
            avg_missed_profit = np.mean([s.shadow_profit_24h or 0 for s in high_confidence_rejects])
            patterns.append({
                'pattern_type': 'high_confidence_rejects',
                'count': len(high_confidence_rejects),
                'avg_missed_profit': avg_missed_profit,
                'description': 'High confidence signals rejected by filters',
                'impact_score': len(high_confidence_rejects) * avg_missed_profit
            })
        
        # Pattern 2: Specific layer dominance in misses
        for layer in ['geometric', 'smart_money', 'depth', 'timeband']:
            layer_misses = [
                s for s in profitable_misses
                if max(s.layer_contributions.items(), key=lambda x: x[1])[0] == layer
            ]
            
            if len(layer_misses) > 3:  # Significant pattern
                avg_missed_profit = np.mean([s.shadow_profit_24h or 0 for s in layer_misses])
                patterns.append({
                    'pattern_type': f'{layer}_dominated_misses',
                    'count': len(layer_misses),
                    'avg_missed_profit': avg_missed_profit,
                    'description': f'Missed opportunities dominated by {layer} signals',
                    'impact_score': len(layer_misses) * avg_missed_profit
                })
        
        # Sort patterns by impact score
        patterns.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return patterns[:5]  # Return top 5 patterns
    
    def _generate_optimization_recommendations(self, signals: List[ShadowGhostSignal],
                                             layer_performance: Dict[str, Dict[str, float]],
                                             market_condition_performance: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate actionable optimization recommendations"""
        
        recommendations = []
        
        # Analyze execution rates vs profitability
        for layer, perf in layer_performance.items():
            if perf['signal_count'] > 10:  # Sufficient data
                execution_rate = perf['execution_rate']
                profitability_rate = perf['profitability_rate']
                avg_opportunity_cost = perf['avg_opportunity_cost']
                
                if execution_rate < 0.3 and profitability_rate > 0.6:
                    recommendations.append(
                        f"INCREASE {layer.upper()} EXECUTION: Low execution rate ({execution_rate:.1%}) "
                        f"but high profitability ({profitability_rate:.1%}). "
                        f"Avg opportunity cost: {avg_opportunity_cost:.3f}"
                    )
                
                elif execution_rate > 0.7 and profitability_rate < 0.4:
                    recommendations.append(
                        f"REDUCE {layer.upper()} EXECUTION: High execution rate ({execution_rate:.1%}) "
                        f"but low profitability ({profitability_rate:.1%}). "
                        f"Consider stricter filters."
                    )
        
        # Market condition recommendations
        best_regime = max(market_condition_performance.items(),
                         key=lambda x: x[1]['profitability_rate'])
        worst_regime = min(market_condition_performance.items(),
                          key=lambda x: x[1]['profitability_rate'])
        
        if best_regime[1]['profitability_rate'] - worst_regime[1]['profitability_rate'] > 0.3:
            recommendations.append(
                f"REGIME ADAPTATION: {best_regime[0]} shows {best_regime[1]['profitability_rate']:.1%} "
                f"profitability vs {worst_regime[1]['profitability_rate']:.1%} for {worst_regime[0]}. "
                f"Consider regime-specific thresholds."
            )
        
        # Confidence threshold recommendations
        profitable_misses = [s for s in signals 
                           if s.outcome_classification == OutcomeClassification.PROFITABLE_MISS]
        
        if profitable_misses:
            avg_missed_confidence = np.mean([s.confidence_score for s in profitable_misses])
            current_threshold = self.config.get('min_confidence_threshold', 0.7)
            
            if avg_missed_confidence > current_threshold:
                recommendations.append(
                    f"LOWER CONFIDENCE THRESHOLD: Missed profitable signals have avg confidence "
                    f"{avg_missed_confidence:.2f} vs current threshold {current_threshold:.2f}. "
                    f"Consider lowering to {avg_missed_confidence - 0.1:.2f}"
                )
        
        return recommendations
    
    def _insert_signal_to_db(self, signal: ShadowGhostSignal) -> None:
        """Insert signal record to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO shadow_signals (
                signal_id, ghost_hash, timestamp, signal_type, confidence_score,
                layer_contributions, market_conditions, was_triggered, trigger_reason,
                rejection_reason, price_at_signal, volume_at_signal, spread_at_signal,
                order_book_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.signal_id, signal.ghost_hash, signal.timestamp, signal.signal_type.value,
            signal.confidence_score, json.dumps(signal.layer_contributions),
            json.dumps(signal.market_conditions), signal.was_triggered,
            signal.trigger_reason, signal.rejection_reason, signal.price_at_signal,
            signal.volume_at_signal, signal.spread_at_signal,
            json.dumps(signal.order_book_snapshot) if signal.order_book_snapshot else None
        ))
        
        conn.commit()
        conn.close()
    
    def _update_signal_in_db(self, signal: ShadowGhostSignal) -> None:
        """Update signal record with post-analysis data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE shadow_signals SET
                price_1h_later = ?, price_4h_later = ?, price_24h_later = ?,
                realized_volatility = ?, shadow_profit_1h = ?, shadow_profit_4h = ?,
                shadow_profit_24h = ?, opportunity_cost = ?, outcome_classification = ?,
                execution_price = ?, execution_quantity = ?, actual_profit_loss = ?,
                execution_latency_ms = ?
            WHERE signal_id = ?
        ''', (
            signal.price_1h_later, signal.price_4h_later, signal.price_24h_later,
            signal.realized_volatility, signal.shadow_profit_1h, signal.shadow_profit_4h,
            signal.shadow_profit_24h, signal.opportunity_cost,
            signal.outcome_classification.value if signal.outcome_classification else None,
            signal.execution_price, signal.execution_quantity, signal.actual_profit_loss,
            signal.execution_latency_ms, signal.signal_id
        ))
        
        conn.commit()
        conn.close()
    
    def _query_signals_from_db(self, start_time: float, end_time: float,
                              signal_types: Optional[List[ShadowSignalType]] = None) -> List[ShadowGhostSignal]:
        """Query signals from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM shadow_signals 
            WHERE timestamp BETWEEN ? AND ?
        '''
        params = [start_time, end_time]
        
        if signal_types:
            placeholders = ','.join(['?' for _ in signal_types])
            query += f' AND signal_type IN ({placeholders})'
            params.extend([st.value for st in signal_types])
        
        query += ' ORDER BY timestamp DESC'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert DataFrame rows to ShadowGhostSignal objects
        signals = []
        for _, row in df.iterrows():
            signal = ShadowGhostSignal(
                signal_id=row['signal_id'],
                ghost_hash=row['ghost_hash'],
                timestamp=row['timestamp'],
                signal_type=ShadowSignalType(row['signal_type']),
                confidence_score=row['confidence_score'],
                layer_contributions=json.loads(row['layer_contributions']),
                market_conditions=json.loads(row['market_conditions']),
                was_triggered=bool(row['was_triggered']),
                trigger_reason=row['trigger_reason'],
                rejection_reason=row['rejection_reason'],
                price_at_signal=row['price_at_signal'],
                volume_at_signal=row['volume_at_signal'],
                spread_at_signal=row['spread_at_signal'],
                order_book_snapshot=json.loads(row['order_book_snapshot']) if row['order_book_snapshot'] else None,
                price_1h_later=row['price_1h_later'],
                price_4h_later=row['price_4h_later'],
                price_24h_later=row['price_24h_later'],
                realized_volatility=row['realized_volatility'],
                shadow_profit_1h=row['shadow_profit_1h'],
                shadow_profit_4h=row['shadow_profit_4h'],
                shadow_profit_24h=row['shadow_profit_24h'],
                opportunity_cost=row['opportunity_cost'],
                outcome_classification=OutcomeClassification(row['outcome_classification']) if row['outcome_classification'] else None,
                execution_price=row['execution_price'],
                execution_quantity=row['execution_quantity'],
                actual_profit_loss=row['actual_profit_loss'],
                execution_latency_ms=row['execution_latency_ms']
            )
            signals.append(signal)
        
        return signals
    
    def export_analysis_report(self, analysis: ShadowAnalysisResult, filepath: str) -> None:
        """Export comprehensive analysis report"""
        report = {
            'analysis_summary': {
                'period': {
                    'start': datetime.fromtimestamp(analysis.analysis_period[0]).isoformat(),
                    'end': datetime.fromtimestamp(analysis.analysis_period[1]).isoformat()
                },
                'signal_counts': {
                    'total': analysis.total_signals,
                    'executed': analysis.executed_signals,
                    'missed_opportunities': analysis.missed_opportunities,
                    'avoided_losses': analysis.correctly_avoided_losses
                },
                'performance_metrics': {
                    'shadow_profit_total': analysis.shadow_profit_total,
                    'actual_profit_total': analysis.actual_profit_total,
                    'opportunity_cost_total': analysis.opportunity_cost_total,
                    'hit_rate': analysis.hit_rate,
                    'shadow_hit_rate': analysis.shadow_hit_rate
                }
            },
            'layer_performance': analysis.layer_performance,
            'market_condition_performance': analysis.market_condition_performance,
            'missed_patterns': analysis.missed_patterns,
            'optimization_recommendations': analysis.optimization_recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

# Utility functions for integration
def create_shadow_signal_from_ghost(ghost_hash: str, confidence: float, 
                                   layer_contributions: Dict[str, float],
                                   market_data: Dict[str, Any],
                                   was_triggered: bool = False) -> ShadowGhostSignal:
    """Create shadow signal from ghost signal data"""
    
    signal_id = f"shadow_{ghost_hash[:8]}_{int(time.time())}"
    
    return ShadowGhostSignal(
        signal_id=signal_id,
        ghost_hash=ghost_hash,
        timestamp=time.time(),
        signal_type=ShadowSignalType.EXECUTED_GHOST if was_triggered else ShadowSignalType.UNEXECUTED_GHOST,
        confidence_score=confidence,
        layer_contributions=layer_contributions,
        market_conditions=market_data,
        was_triggered=was_triggered,
        price_at_signal=market_data.get('price', 0.0),
        volume_at_signal=market_data.get('volume', 0.0),
        spread_at_signal=market_data.get('spread', 0.0)
    ) 