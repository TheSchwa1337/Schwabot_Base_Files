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

@dataclass
class AlertBusConfig:
    """Configuration for alert bus integration"""
    enabled: bool = True
    channels: List[str] = None
    severity_levels: Dict[str, int] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = ["log", "console"]
        if self.severity_levels is None:
            self.severity_levels = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}

class EnhancedTesseractProcessor:
    """
    Enhanced Tesseract Pattern Processor with advanced risk management integration
    and Zygot shell alignment system.
    
    Features:
    - Robust YAML configuration loading with path standardization
    - Pattern history management with overflow protection
    - Enhanced error handling and validation
    - Configurable symbolic strategy switching
    - Weighted profit vector blending
    - Test mode with comprehensive logging
    - Alert system integration
    """
    
    def __init__(self, config_path: str = "config/tesseract_enhanced.yaml"):
        # Initialize logging first
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load configuration with robust error handling
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.risk_monitor = QuantumCellularRiskMonitor(self.config)
        self.risk_indexer = RiskIndexer()
        
        # Processing state
        self.tick_step_counter = 0
        self.flip_frequency = self.config.get('processing', {}).get('baseline_reset_flip_frequency', 100)
        
        # Pattern processing with size management
        self.dimension_labels = self.config.get('dimensions', {}).get('labels', [])
        self.pattern_history: List[Dict] = []
        self.max_pattern_history = self.config.get('processing', {}).get('max_pattern_history', 1000)
        
        # Enhanced monitoring
        self.alert_thresholds = self.config.get('monitoring', {}).get('alerts', {})
        self.alert_bus_config = AlertBusConfig(**self.config.get('alert_bus', {}))
        
        # Zygot shell integration
        self.shell_generator = ZygotShell()
        self.shell_history: List[ZygotShellState] = []
        self.previous_peak_vectors: List[np.ndarray] = []
        self.max_shell_history = self.config.get('processing', {}).get('max_shell_history', 500)
        
        # Vault state
        self.vault = VaultLockState()
        
        # Strategy state with configurable triggers
        self.active_strategy = "default"
        self.strategy_state = set()
        self.re_entry_trigger = False
        self.strategy_triggers = self.config.get('strategies', {})
        
        # Test and debug modes
        self.test_mode = self.config.get('debug', {}).get('test_mode', False)
        self.verbose_logging = self.config.get('debug', {}).get('verbose_logging', False)
        
        # Profit vector blending configuration
        self.profit_blend_alpha = self.config.get('processing', {}).get('profit_blend_alpha', 0.7)
        
        # Initialize alert bus placeholder
        self.alert_bus = None  # Will be injected by external system
        
        self.logger.info(f"Enhanced Tesseract Processor initialized with config: {config_path}")
        if self.test_mode:
            self.logger.info("TEST MODE ENABLED - Enhanced debugging active")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load enhanced configuration with robust error handling and path standardization
        """
        # Standardize path relative to repository root
        if not Path(config_path).is_absolute():
            config_path = Path(__file__).resolve().parent.parent / config_path
        else:
            config_path = Path(config_path)
            
        if not config_path.exists():
            self.logger.error(f"Enhanced Tesseract config not found at: {config_path}")
            # Try to create default config
            self._create_default_config(config_path)
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Successfully loaded config from: {config_path}")
                return config
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML config {config_path}: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading config {config_path}: {e}")
            raise
            
    def _create_default_config(self, config_path: Path) -> None:
        """Create default configuration if missing"""
        default_config = {
            'processing': {
                'baseline_reset_flip_frequency': 100,
                'max_pattern_history': 1000,
                'max_shell_history': 500,
                'profit_blend_alpha': 0.7
            },
            'dimensions': {
                'labels': ['price', 'volume', 'volatility', 'momentum', 'rsi', 'macd', 'bb_upper', 'bb_lower']
            },
            'monitoring': {
                'alerts': {
                    'var_threshold': 0.05,
                    'var_indexed_threshold': 1.5,
                    'coherence_threshold': 0.5,
                    'coherence_indexed_threshold': 0.8
                }
            },
            'strategies': {
                'inversion_burst_rebound': {
                    'trigger_prefix': 'e1a7'
                },
                'momentum_cascade': {
                    'trigger_prefix': 'f2b8'
                },
                'volatility_breakout': {
                    'trigger_prefix': 'a3c9'
                }
            },
            'debug': {
                'test_mode': False,
                'verbose_logging': False
            },
            'alert_bus': {
                'enabled': True,
                'channels': ['log', 'console'],
                'severity_levels': {
                    'HIGH': 3,
                    'MEDIUM': 2,
                    'LOW': 1,
                    'INFO': 0
                }
            }
        }
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(default_config, f, default_flow_style=False)
            
        self.logger.info(f"Created default config at: {config_path}")
            
    def _compute_fractal_volatility(self, returns: np.ndarray) -> float:
        """
        Compute fractal volatility score using FFT compression with enhanced error handling
        """
        if len(returns) < 4:
            return 0.0
            
        # Handle flat or zero returns
        if np.all(returns == 0) or np.std(returns) < 1e-10:
            return 0.0
            
        try:
            fft = np.abs(np.fft.fft(returns))
            if len(fft) == 0 or np.max(fft) == 0:
                return 0.0
                
            envelope = np.mean(fft[:min(8, len(fft))])  # Use first 8 harmonics or available
            return np.clip(envelope / np.max(fft), 0.0, 1.0)
        except Exception as e:
            self.logger.warning(f"Error computing fractal volatility: {e}")
            return 0.0
        
    def _calculate_drift_shell_alignment(self, vector: np.ndarray, 
                                       peak_vectors: List[np.ndarray]) -> float:
        """Calculate alignment score between vector and peak vectors with error handling"""
        if not peak_vectors or len(vector) == 0:
            return 1.0
            
        alignments = []
        try:
            for peak in peak_vectors:
                if len(peak) != len(vector):
                    continue
                    
                dot_product = np.dot(vector, peak)
                norm_product = np.linalg.norm(vector) * np.linalg.norm(peak)
                
                if norm_product > 1e-10:  # Avoid division by zero
                    alignments.append(dot_product / norm_product)
                    
            return np.mean(alignments) if alignments else 0.0
        except Exception as e:
            self.logger.warning(f"Error calculating drift shell alignment: {e}")
            return 0.0
        
    def _should_activate_vault_lock(self, signal_strength: float, 
                                  metrics: AdvancedRiskMetrics) -> bool:
        """Determine if vault lock should be activated with safe attribute access"""
        try:
            fvs_score = getattr(metrics, 'fvs_score', 0.0)
            homeostasis = getattr(metrics, 'homeostasis', 0.0)
            profit_memory = getattr(metrics, 'profit_memory', 0.0)
            median_trajectory = getattr(metrics, 'median_trajectory', 0.0)
            
            return (
                fvs_score > 0.8 and
                homeostasis > 0.9 and
                signal_strength > 0.9 and
                profit_memory > median_trajectory
            )
        except Exception as e:
            self.logger.warning(f"Error in vault lock evaluation: {e}")
            return False
        
    def _re_entry_protocol(self):
        """Execute re-entry protocol for profitable exit zones"""
        if not self.shell_history:
            return
            
        try:
            # Get recent profitable exits
            recent_exits = [
                shell for shell in self.shell_history[-100:]
                if getattr(shell, 'alignment_score', 0.0) > 0.8
            ]
            
            if recent_exits:
                self.re_entry_trigger = True
                self.logger.info(f"Re-entry protocol activated with {len(recent_exits)} zones")
        except Exception as e:
            self.logger.warning(f"Error in re-entry protocol: {e}")
            
    def _extract_8d_pattern(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract 8D pattern from market data with validation"""
        try:
            pattern = []
            for label in self.dimension_labels[:8]:
                value = market_data.get(label, 0.0)
                # Validate numeric value
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    pattern.append(float(value))
                else:
                    pattern.append(0.0)
                    
            # Ensure we have exactly 8 dimensions
            while len(pattern) < 8:
                pattern.append(0.0)
                
            return pattern[:8]
        except Exception as e:
            self.logger.warning(f"Error extracting 8D pattern: {e}")
            return [0.0] * 8
            
    def _blend_profit_vector(self, vec: np.ndarray, alignment_score: float) -> float:
        """
        Blend profit vector using weighted geometric mean approach
        """
        try:
            if len(vec) == 0:
                return 0.0
                
            # Handle NaN or infinite values
            vec_clean = vec[np.isfinite(vec)]
            if len(vec_clean) == 0:
                return 0.0
                
            mean_component = np.mean(vec_clean)
            magnitude = np.linalg.norm(vec_clean)
            
            # Weighted blend
            blended = (self.profit_blend_alpha * mean_component + 
                      (1 - self.profit_blend_alpha) * magnitude)
            
            return blended * alignment_score
        except Exception as e:
            self.logger.warning(f"Error blending profit vector: {e}")
            return 0.0
            
    def _update_strategy_from_hash(self, symbolic_hash: str) -> None:
        """Update active strategy based on symbolic hash with configurable triggers"""
        try:
            for strategy_name, config in self.strategy_triggers.items():
                trigger_prefix = config.get('trigger_prefix', '')
                if trigger_prefix and symbolic_hash.startswith(trigger_prefix):
                    if self.active_strategy != strategy_name:
                        self.logger.info(f"Strategy switch: {self.active_strategy} -> {strategy_name}")
                        self.active_strategy = strategy_name
                    break
        except Exception as e:
            self.logger.warning(f"Error updating strategy from hash: {e}")
            
    def _manage_history_size(self) -> None:
        """Manage history sizes to prevent memory overflow"""
        # Trim pattern history
        if len(self.pattern_history) > self.max_pattern_history:
            trim_size = len(self.pattern_history) - self.max_pattern_history
            self.pattern_history = self.pattern_history[trim_size:]
            if self.verbose_logging:
                self.logger.debug(f"Trimmed pattern history by {trim_size} entries")
                
        # Trim shell history
        if len(self.shell_history) > self.max_shell_history:
            trim_size = len(self.shell_history) - self.max_shell_history
            self.shell_history = self.shell_history[trim_size:]
            if self.verbose_logging:
                self.logger.debug(f"Trimmed shell history by {trim_size} entries")
        
    async def process_market_tick(self, market_data: Dict[str, Any], 
                                basket_id: str = "default_asset") -> Dict[str, Any]:
        """
        Process market tick with enhanced risk assessment and indexed metrics
        """
        try:
            self.tick_step_counter += 1
            current_time = datetime.now().timestamp()
            
            # Add market data to pattern history
            self.pattern_history.append(market_data)
            
            # Manage history sizes
            self._manage_history_size()
            
            if self.test_mode:
                self.logger.debug(f"Processing tick #{self.tick_step_counter} for basket: {basket_id}")
            
            # Update risk state
            risk_metrics = await self.risk_monitor.update_risk_state(market_data)
            
            # Calculate FVS with enhanced error handling
            if len(self.pattern_history) >= 4:
                price_data = [
                    d.get('price', 0.0) for d in self.pattern_history[-100:]
                    if d.get('price') is not None and isinstance(d.get('price'), (int, float))
                ]
                
                if len(price_data) >= 2:
                    returns = np.diff(price_data)
                    risk_metrics.fvs_score = self._compute_fractal_volatility(returns)
                else:
                    risk_metrics.fvs_score = 0.0
            else:
                risk_metrics.fvs_score = 0.0
            
            # Convert AdvancedRiskMetrics to dict and filter numeric metrics
            raw_metrics_for_indexing = {
                k: v for k, v in asdict(risk_metrics).items() 
                if isinstance(v, (int, float)) and not k.endswith('_weights')
            }
            
            # Update and get indexed risk metrics
            indexed_risk_metrics = self.risk_indexer.update(basket_id, raw_metrics_for_indexing)
            
            # Generate Zygot shell
            pattern = self._extract_8d_pattern(market_data)
            shell = self.shell_generator.process_shell_state(
                vector=np.array(pattern),
                phase_angle=getattr(risk_metrics, 'phase_angle', 0.0),
                entropy=getattr(risk_metrics, 'entropy', 0.0)
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
            
            # Calculate profit vector with enhanced blending
            profit_vector = np.array([
                signal_strength * position_size,
                getattr(risk_metrics, 'coherence', 0.0) * indexed_risk_metrics.get('fhs_var', 1.0),
                getattr(risk_metrics, 'homeostasis', 0.0) * alignment_score
            ])
            
            # Final profit signal using enhanced blending
            final_profit_signal = self._blend_profit_vector(profit_vector, alignment_score)
            
            # Check vault lock conditions
            if self._should_activate_vault_lock(final_profit_signal, risk_metrics):
                self.vault = VaultLockState(
                    is_locked=True,
                    lock_timestamp=current_time,
                    lock_reason="High quality signal with peak metrics",
                    signal_quality=final_profit_signal,
                    profit_memory=getattr(risk_metrics, 'profit_memory', 0.0)
                )
                
            # Execute re-entry protocol
            if self.tick_step_counter % 288 == 0:
                self._re_entry_protocol()
                
            # Generate symbolic hash
            symbolic_hash = ZygotControlHooks.symbolic_encode(shell)
            
            # Update strategy based on symbolic hash
            self._update_strategy_from_hash(symbolic_hash)
            
            # Generate trading signals
            trading_signals = {
                'final_signal_strength': final_profit_signal,
                'symbolic_hash': symbolic_hash,
                'vault_state': asdict(self.vault),
                'strategy': self.active_strategy,
                'profit_band': getattr(getattr(risk_metrics, 'profit_band', None), 'value', 'UNKNOWN'),
                'timestamp': current_time,
                'alignment': alignment_score,
                'shell_uuid': str(getattr(shell, 'uuid', uuid.uuid4())),
                're_entry_triggered': self.re_entry_trigger,
                'risk_metrics_raw': risk_metrics,
                'risk_metrics_indexed': indexed_risk_metrics,
                'pattern': pattern,
                'regime': getattr(risk_metrics, 'current_regime', 'UNKNOWN'),
                'coherence': getattr(risk_metrics, 'coherence', 0.0),
                'homeostasis': getattr(risk_metrics, 'homeostasis', 0.0),
                'tick_counter': self.tick_step_counter,
                'basket_id': basket_id
            }
            
            # Enhanced logging
            if self.verbose_logging:
                self.logger.debug(f"[{basket_id}] Signal: {final_profit_signal:.4f} | Strategy: {self.active_strategy}")
                self.logger.debug(f"[{basket_id}] Shell UUID: {getattr(shell, 'uuid', 'N/A')} | Symbolic: {symbolic_hash}")
            
            # Check for alerts
            await self._check_risk_alerts(risk_metrics, indexed_risk_metrics, basket_id)
            
            return trading_signals
            
        except Exception as e:
            self.logger.error(f"Error processing market tick for {basket_id}: {e}")
            # Return safe default response
            return {
                'final_signal_strength': 0.0,
                'symbolic_hash': 'error',
                'vault_state': asdict(self.vault),
                'strategy': 'error_fallback',
                'profit_band': 'UNKNOWN',
                'timestamp': datetime.now().timestamp(),
                'alignment': 0.0,
                'shell_uuid': str(uuid.uuid4()),
                're_entry_triggered': False,
                'error': str(e),
                'tick_counter': self.tick_step_counter,
                'basket_id': basket_id
            }
        
    def _calculate_risk_adjusted_signal(self, pattern: List[float], 
                                      risk_metrics: AdvancedRiskMetrics,
                                      indexed_metrics: Dict[str, float]) -> float:
        """Calculate risk-adjusted signal strength using both raw and indexed metrics"""
        try:
            base_signal = np.mean(pattern) if pattern else 0.0
            
            # Risk adjustments using indexed metrics
            var_adjustment = 1.0 / max(indexed_metrics.get('fhs_var', 1.0), 0.01)
            coherence_boost = indexed_metrics.get('coherence', 1.0)
            
            # Profit band multiplier with safe access
            profit_band = getattr(risk_metrics, 'profit_band', None)
            band_multiplier = {
                ProfitBand.ACCUMULATION: 1.2,
                ProfitBand.EXPANSION: 1.0,
                ProfitBand.RETRACTION: 0.6,
                ProfitBand.DISTRIBUTION: 0.8
            }.get(profit_band, 1.0)
            
            adjusted_signal = (
                base_signal * 
                var_adjustment * 
                coherence_boost * 
                band_multiplier
            )
            
            return np.clip(adjusted_signal, -1.0, 1.0)
        except Exception as e:
            self.logger.warning(f"Error calculating risk-adjusted signal: {e}")
            return 0.0
        
    def _calculate_regime_aware_position_size(self, risk_metrics: AdvancedRiskMetrics,
                                            indexed_metrics: Dict[str, float]) -> float:
        """Calculate position size based on current market regime and indexed metrics"""
        try:
            base_size = 0.1  # 10% base position
            
            # Regime adjustments with safe access
            current_regime = getattr(risk_metrics, 'current_regime', 'NORMAL')
            if current_regime == "HIGH_VOLATILITY":
                regime_multiplier = 0.5
            elif current_regime == "LOW_VOLATILITY":
                regime_multiplier = 1.5
            else:
                regime_multiplier = 1.0
                
            # Risk metric adjustments
            var_multiplier = 1.0 / max(indexed_metrics.get('fhs_var', 1.0), 0.01)
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
        except Exception as e:
            self.logger.warning(f"Error calculating position size: {e}")
            return 0.1  # Safe default
        
    async def _check_risk_alerts(self, risk_metrics: AdvancedRiskMetrics, 
                                indexed_metrics: Dict[str, float],
                                basket_id: str = "default"):
        """Check for risk alert conditions using both raw and indexed metrics"""
        if not self.alert_bus_config.enabled:
            return
            
        alerts = []
        now = datetime.now()
        
        try:
            # VaR breach alert (raw)
            var_threshold_raw = self.alert_thresholds.get('var_threshold', 0.05)
            fhs_var = getattr(risk_metrics, 'fhs_var', 0.0)
            if abs(fhs_var) > var_threshold_raw:
                alerts.append({
                    'type': 'VAR_BREACH_RAW',
                    'severity': 'HIGH',
                    'message': f"FHS VaR (raw) exceeded threshold: {fhs_var:.4f} > {var_threshold_raw}",
                    'timestamp': now,
                    'basket_id': basket_id
                })
                
            # VaR breach alert (indexed)
            var_indexed_threshold = self.alert_thresholds.get('var_indexed_threshold', 1.5)
            if 'fhs_var' in indexed_metrics and indexed_metrics['fhs_var'] > var_indexed_threshold:
                alerts.append({
                    'type': 'VAR_BREACH_INDEXED',
                    'severity': 'MEDIUM',
                    'message': f"FHS VaR (indexed) is {indexed_metrics['fhs_var']:.2f}x baseline, exceeding {var_indexed_threshold}x",
                    'timestamp': now,
                    'basket_id': basket_id
                })
                
            # Coherence alert (raw)
            coherence_threshold_raw = self.alert_thresholds.get('coherence_threshold', 0.5)
            coherence = getattr(risk_metrics, 'coherence', 1.0)
            if coherence < coherence_threshold_raw:
                alerts.append({
                    'type': 'COHERENCE_LOW_RAW',
                    'severity': 'MEDIUM',
                    'message': f"Quantum coherence (raw) below threshold: {coherence:.4f} < {coherence_threshold_raw}",
                    'timestamp': now,
                    'basket_id': basket_id
                })
                
            # Coherence alert (indexed)
            coherence_indexed_threshold = self.alert_thresholds.get('coherence_indexed_threshold', 0.8)
            if 'coherence' in indexed_metrics and indexed_metrics['coherence'] < coherence_indexed_threshold:
                alerts.append({
                    'type': 'COHERENCE_LOW_INDEXED',
                    'severity': 'MEDIUM',
                    'message': f"Quantum coherence (indexed) is {indexed_metrics['coherence']:.2f}x baseline, below {coherence_indexed_threshold}x",
                    'timestamp': now,
                    'basket_id': basket_id
                })
                
            # Vault lock alert
            if self.vault.is_locked:
                alerts.append({
                    'type': 'VAULT_LOCK_ACTIVE',
                    'severity': 'INFO',
                    'message': f"Vault lock active: {self.vault.lock_reason}",
                    'timestamp': now,
                    'basket_id': basket_id
                })
                
            if alerts:
                await self._send_alerts(alerts)
                
        except Exception as e:
            self.logger.error(f"Error checking risk alerts: {e}")
            
    async def _send_alerts(self, alerts: List[Dict]):
        """Send alerts through configured channels"""
        for alert in alerts:
            try:
                # Log alert
                if 'log' in self.alert_bus_config.channels:
                    severity = alert.get('severity', 'INFO')
                    if severity == 'HIGH':
                        self.logger.error(f"RISK ALERT: {alert['message']} (Type: {alert['type']})")
                    elif severity == 'MEDIUM':
                        self.logger.warning(f"RISK ALERT: {alert['message']} (Type: {alert['type']})")
                    else:
                        self.logger.info(f"RISK ALERT: {alert['message']} (Type: {alert['type']})")
                
                # Console output
                if 'console' in self.alert_bus_config.channels:
                    print(f"🚨 {alert['severity']}: {alert['message']}")
                
                # External alert bus integration
                if self.alert_bus is not None:
                    try:
                        await self.alert_bus.publish(alert)
                    except Exception as e:
                        self.logger.error(f"Failed to publish alert to external bus: {e}")
                        
            except Exception as e:
                self.logger.error(f"Error sending alert: {e}")
    
    def enable_test_mode(self, verbose: bool = True):
        """Enable test mode with optional verbose logging"""
        self.test_mode = True
        self.verbose_logging = verbose
        self.logger.info("Test mode enabled" + (" with verbose logging" if verbose else ""))
    
    def disable_test_mode(self):
        """Disable test mode"""
        self.test_mode = False
        self.verbose_logging = False
        self.logger.info("Test mode disabled")
    
    def set_alert_bus(self, alert_bus):
        """Set external alert bus for notification integration"""
        self.alert_bus = alert_bus
        self.logger.info("External alert bus configured")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processor status"""
        return {
            'tick_counter': self.tick_step_counter,
            'pattern_history_size': len(self.pattern_history),
            'shell_history_size': len(self.shell_history),
            'active_strategy': self.active_strategy,
            'vault_locked': self.vault.is_locked,
            'test_mode': self.test_mode,
            'verbose_logging': self.verbose_logging,
            're_entry_trigger': self.re_entry_trigger
        }
    
    def reset_state(self):
        """Reset processor state (useful for testing)"""
        self.tick_step_counter = 0
        self.pattern_history.clear()
        self.shell_history.clear()
        self.previous_peak_vectors.clear()
        self.vault = VaultLockState()
        self.active_strategy = "default"
        self.strategy_state.clear()
        self.re_entry_trigger = False
        self.logger.info("Processor state reset") 