from dataclasses import asdict, dataclass
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import numpy as np
import yaml
from pathlib import Path
import uuid
from enum import Enum

from .risk_indexer import RiskIndexer
from .quantum_cellular_risk_monitor import QuantumCellularRiskMonitor, AdvancedRiskMetrics
from .zygot_shell import ZygotShellState, ZygotControlHooks, ZygotShell

logger = logging.getLogger(__name__)

class ProfitBand(Enum):
    """Profit band classification"""
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    EXPANSION = "EXPANSION"
    RETRACTION = "RETRACTION"

@dataclass
class VaultLockState:
    """Vault lock state for strategic position protection"""
    is_locked: bool = False
    lock_timestamp: Optional[float] = None
    lock_reason: str = ""
    signal_quality: float = 0.0
    profit_memory: float = 0.0

class EnhancedTesseractProcessor:
    """
    Enhanced Tesseract Pattern Processor with advanced risk management integration
    and Zygot shell alignment system.
    """
    
    def __init__(self, config_path: str = "config/tesseract_enhanced.yaml"):
        self.config = self._load_config(config_path)
        self.risk_monitor = QuantumCellularRiskMonitor(self.config)
        
        # Initialize RiskIndexer
        self.risk_indexer = RiskIndexer()
        self.tick_step_counter = 0
        self.flip_frequency = self.config.get('processing', {}).get('baseline_reset_flip_frequency', 100)
        
        # Pattern processing
        self.dimension_labels = self.config.get('dimensions', {}).get('labels', [])
        self.pattern_history: List[Dict] = []
        
        # Enhanced monitoring
        self.alert_thresholds = self.config.get('monitoring', {}).get('alerts', {})
        
        # Zygot shell integration
        self.shell_generator = ZygotShell()
        self.shell_history: List[ZygotShellState] = []
        self.previous_peak_vectors: List[np.ndarray] = []
        
        # Vault state
        self.vault = VaultLockState()
        
        # Strategy state
        self.active_strategy = "default"
        self.strategy_state = set()
        self.re_entry_trigger = False
        
        # Test mode
        self.test_mode = self.config.get('debug', {}).get('test_mode', False)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load enhanced configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _compute_fractal_volatility(self, returns: np.ndarray) -> float:
        """Compute fractal volatility score using FFT compression"""
        fft = np.abs(np.fft.fft(returns))
        envelope = np.mean(fft[:8])  # Use first 8 harmonics
        return np.clip(envelope / np.max(fft), 0.0, 1.0)
        
    def _calculate_drift_shell_alignment(self, vector: np.ndarray, 
                                       peak_vectors: List[np.ndarray]) -> float:
        """Calculate alignment score between vector and peak vectors"""
        if not peak_vectors:
            return 1.0
            
        alignments = []
        for peak in peak_vectors:
            dot_product = np.dot(vector, peak)
            norm_product = np.linalg.norm(vector) * np.linalg.norm(peak)
            if norm_product > 0:
                alignments.append(dot_product / norm_product)
                
        return np.mean(alignments) if alignments else 0.0
        
    def _should_activate_vault_lock(self, signal_strength: float, 
                                  metrics: AdvancedRiskMetrics) -> bool:
        """Determine if vault lock should be activated"""
        return (
            metrics.fvs_score > 0.8 and
            metrics.homeostasis > 0.9 and
            signal_strength > 0.9 and
            metrics.profit_memory > metrics.median_trajectory
        )
        
    def _re_entry_protocol(self):
        """Execute re-entry protocol for profitable exit zones"""
        if not self.shell_history:
            return
            
        # Get recent profitable exits
        recent_exits = [
            shell for shell in self.shell_history[-100:]
            if shell.alignment_score > 0.8
        ]
        
        if recent_exits:
            self.re_entry_trigger = True
            logger.info(f"Re-entry protocol activated with {len(recent_exits)} zones")
            
    def _extract_8d_pattern(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract 8D pattern from market data"""
        return [market_data.get(k, 0.0) for k in self.dimension_labels[:8]]
        
    async def process_market_tick(self, market_data: Dict[str, Any], 
                                basket_id: str = "default_asset") -> Dict[str, Any]:
        """
        Process market tick with enhanced risk assessment and indexed metrics
        """
        self.tick_step_counter += 1
        current_time = datetime.now().timestamp()
        
        # Update risk state
        risk_metrics = await self.risk_monitor.update_risk_state(market_data)
        
        # Add FVS to risk metrics
        returns = np.diff([d.get('price', 0.0) for d in self.pattern_history[-100:]])
        risk_metrics.fvs_score = self._compute_fractal_volatility(returns)
        
        # Convert AdvancedRiskMetrics to dict and filter numeric metrics
        raw_metrics_for_indexing = {
            k: v for k, v in asdict(risk_metrics).items() 
            if isinstance(v, (int, float)) and not k.endswith('_weights')
        }
        
        # Update and get indexed risk metrics
        indexed_risk_metrics = self.risk_indexer.update(basket_id, raw_metrics_for_indexing)
        
        # Generate Zygot shell
        shell = self.shell_generator.process_shell_state(
            vector=np.array(self._extract_8d_pattern(market_data)),
            phase_angle=risk_metrics.phase_angle,
            entropy=risk_metrics.entropy
        )
        
        # Inject ghost hash
        shell = ZygotControlHooks.inject_ghost_hash(
            shell, 
            ghost_id=f"tick_{self.tick_step_counter}"
        )
        self.shell_history.append(shell)
        
        # Calculate drift shell alignment
        alignment_score = self._calculate_drift_shell_alignment(
            shell.vector,
            self.previous_peak_vectors
        )
        
        # Pattern extraction
        pattern = self._extract_8d_pattern(market_data)
        
        # Enhanced risk-adjusted signal generation
        signal_strength = self._calculate_risk_adjusted_signal(
            pattern, 
            risk_metrics, 
            indexed_risk_metrics
        )
        
        # Regime-aware position sizing
        position_size = self._calculate_regime_aware_position_size(
            risk_metrics, 
            indexed_risk_metrics
        )
        
        # Calculate profit vector
        profit_vector = np.array([
            signal_strength * position_size,
            risk_metrics.coherence * indexed_risk_metrics.get('fhs_var', 1.0),
            risk_metrics.homeostasis * alignment_score
        ])
        
        # Final profit signal
        final_profit_signal = np.mean(profit_vector) * alignment_score
        
        # Check vault lock conditions
        if self._should_activate_vault_lock(final_profit_signal, risk_metrics):
            self.vault = VaultLockState(
                is_locked=True,
                lock_timestamp=current_time,
                lock_reason="High quality signal with peak metrics",
                signal_quality=final_profit_signal,
                profit_memory=risk_metrics.profit_memory
            )
            
        # Execute re-entry protocol
        if self.tick_step_counter % 288 == 0:
            self._re_entry_protocol()
            
        # Generate symbolic hash
        symbolic_hash = ZygotControlHooks.symbolic_encode(shell)
        
        # Update strategy based on symbolic hash
        if symbolic_hash.startswith("e1a7"):
            self.active_strategy = "inversion_burst_rebound"
            
        # Generate trading signals
        trading_signals = {
            'final_signal_strength': final_profit_signal,
            'symbolic_hash': symbolic_hash,
            'vault_state': asdict(self.vault),
            'strategy': self.active_strategy,
            'profit_band': risk_metrics.profit_band.value,
            'timestamp': current_time,
            'alignment': alignment_score,
            'shell_uuid': str(shell.uuid),
            're_entry_triggered': self.re_entry_trigger,
            'risk_metrics_raw': risk_metrics,
            'risk_metrics_indexed': indexed_risk_metrics,
            'pattern': pattern,
            'regime': risk_metrics.current_regime,
            'coherence': risk_metrics.coherence,
            'homeostasis': risk_metrics.homeostasis
        }
        
        # Check for alerts
        await self._check_risk_alerts(risk_metrics, indexed_risk_metrics)
        
        return trading_signals
        
    def _calculate_risk_adjusted_signal(self, pattern: List[float], 
                                      risk_metrics: AdvancedRiskMetrics,
                                      indexed_metrics: Dict[str, float]) -> float:
        """Calculate risk-adjusted signal strength using both raw and indexed metrics"""
        base_signal = np.mean(pattern)
        
        # Risk adjustments using indexed metrics
        var_adjustment = 1.0 / indexed_metrics.get('fhs_var', 1.0)
        coherence_boost = indexed_metrics.get('coherence', 1.0)
        
        # Profit band multiplier
        band_multiplier = {
            ProfitBand.ACCUMULATION: 1.2,
            ProfitBand.EXPANSION: 1.0,
            ProfitBand.RETRACTION: 0.6,
            ProfitBand.DISTRIBUTION: 0.8
        }.get(risk_metrics.profit_band, 1.0)
        
        adjusted_signal = (
            base_signal * 
            var_adjustment * 
            coherence_boost * 
            band_multiplier
        )
        
        return np.clip(adjusted_signal, -1.0, 1.0)
        
    def _calculate_regime_aware_position_size(self, risk_metrics: AdvancedRiskMetrics,
                                            indexed_metrics: Dict[str, float]) -> float:
        """Calculate position size based on current market regime and indexed metrics"""
        base_size = 0.1  # 10% base position
        
        # Regime adjustments
        if risk_metrics.current_regime == "HIGH_VOLATILITY":
            regime_multiplier = 0.5
        elif risk_metrics.current_regime == "LOW_VOLATILITY":
            regime_multiplier = 1.5
        else:
            regime_multiplier = 1.0
            
        # Risk metric adjustments
        var_multiplier = 1.0 / indexed_metrics.get('fhs_var', 1.0)
        coherence_multiplier = indexed_metrics.get('coherence', 1.0)
        
        # Vault lock adjustment
        vault_multiplier = 0.5 if self.vault.is_locked else 1.0
        
        position_size = (
            base_size * 
            regime_multiplier * 
            var_multiplier * 
            coherence_multiplier *
            vault_multiplier
        )
        
        return np.clip(position_size, 0.01, 0.5)
        
    async def _check_risk_alerts(self, risk_metrics: AdvancedRiskMetrics, 
                                indexed_metrics: Dict[str, float]):
        """Check for risk alert conditions using both raw and indexed metrics"""
        alerts = []
        now = datetime.now()
        
        # VaR breach alert (raw)
        var_threshold_raw = self.alert_thresholds.get('var_threshold', 0.05)
        if abs(risk_metrics.fhs_var) > var_threshold_raw:
            alerts.append({
                'type': 'VAR_BREACH_RAW',
                'severity': 'HIGH',
                'message': f"FHS VaR (raw) exceeded threshold: {risk_metrics.fhs_var:.4f} > {var_threshold_raw}",
                'timestamp': now
            })
            
        # VaR breach alert (indexed)
        var_indexed_threshold = self.alert_thresholds.get('var_indexed_threshold', 1.5)
        if 'fhs_var' in indexed_metrics and indexed_metrics['fhs_var'] > var_indexed_threshold:
            alerts.append({
                'type': 'VAR_BREACH_INDEXED',
                'severity': 'MEDIUM',
                'message': f"FHS VaR (indexed) is {indexed_metrics['fhs_var']:.2f}x baseline, exceeding {var_indexed_threshold}x",
                'timestamp': now
            })
            
        # Coherence alert (raw)
        coherence_threshold_raw = self.alert_thresholds.get('coherence_threshold', 0.5)
        if risk_metrics.coherence < coherence_threshold_raw:
            alerts.append({
                'type': 'COHERENCE_LOW_RAW',
                'severity': 'MEDIUM',
                'message': f"Quantum coherence (raw) below threshold: {risk_metrics.coherence:.4f} < {coherence_threshold_raw}",
                'timestamp': now
            })
            
        # Coherence alert (indexed)
        coherence_indexed_threshold = self.alert_thresholds.get('coherence_indexed_threshold', 0.8)
        if 'coherence' in indexed_metrics and indexed_metrics['coherence'] < coherence_indexed_threshold:
            alerts.append({
                'type': 'COHERENCE_LOW_INDEXED',
                'severity': 'MEDIUM',
                'message': f"Quantum coherence (indexed) is {indexed_metrics['coherence']:.2f}x baseline, below {coherence_indexed_threshold}x",
                'timestamp': now
            })
            
        # Vault lock alert
        if self.vault.is_locked:
            alerts.append({
                'type': 'VAULT_LOCK_ACTIVE',
                'severity': 'INFO',
                'message': f"Vault lock active: {self.vault.lock_reason}",
                'timestamp': now
            })
            
        if alerts:
            await self._send_alerts(alerts)
            
    async def _send_alerts(self, alerts: List[Dict]):
        """Send alerts through configured channels"""
        for alert in alerts:
            logger.warning(
                f"RISK ALERT: {alert['message']} "
                f"(Type: {alert['type']}, Severity: {alert['severity']})"
            )
            # Here you would integrate with your notification system
            # self.alert_bus.publish(alert)  # Placeholder for external system 