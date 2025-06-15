"""
Echo Snapshot Logger
===================

Human-readable diagnostic logging system for fractal signal interactions,
state confidence tracking, and decision flow analysis. Provides real-time
insights into the recursive profit engine's decision-making process.

Key Features:
- Real-time signal state snapshots
- Confidence and decision flow tracking
- Human-readable terminal output
- Performance attribution analysis
- Debug trace generation
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
from enum import Enum
import json

from collapse_confidence import CollapseState
from vault_router import VaultAllocation
from fractal_controller import FractalDecision

logger = logging.getLogger(__name__)

class SnapshotLevel(Enum):
    """Snapshot detail levels"""
    MINIMAL = "minimal"      # Basic tick info only
    STANDARD = "standard"    # Standard signal summary
    DETAILED = "detailed"    # Full signal breakdown
    DEBUG = "debug"          # Complete debug trace

@dataclass
class EchoSnapshot:
    """Single echo snapshot with all system state"""
    tick_id: int
    timestamp: float
    confidence: float
    drift_angle: float
    vault_status: str
    braid_signal: float
    paradox_signal: float
    forever_signal: float
    eco_signal: float
    decision_action: str
    projected_profit: float
    volume_allocation: float
    reasoning: str
    fractal_weights: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    ghost_activity: Dict[str, Any] = field(default_factory=dict)
    lockout_status: Dict[str, Any] = field(default_factory=dict)

class EchoSnapshotLogger:
    """
    Advanced diagnostic logging system for recursive profit engine.
    
    Provides human-readable insights into system state, decision flow,
    and performance attribution for debugging and optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize echo snapshot logger.
        
        Args:
            config: Configuration parameters for logging system
        """
        self.config = config or {}
        
        # Logging configuration
        self.snapshot_level = SnapshotLevel(self.config.get('level', 'standard'))
        self.enable_terminal_output = self.config.get('terminal_output', True)
        self.enable_file_logging = self.config.get('file_logging', False)
        self.log_file_path = self.config.get('log_file', 'echo_snapshots.log')
        
        # Snapshot storage
        self.snapshots: deque = deque(maxlen=1000)
        self.tick_counter = 0
        
        # Performance tracking
        self.session_start_time = time.time()
        self.total_snapshots = 0
        self.decision_counts = {"long": 0, "short": 0, "hold": 0, "exit": 0}
        self.confidence_history: deque = deque(maxlen=100)
        
        # Terminal formatting
        self.use_colors = self.config.get('use_colors', True)
        self.compact_mode = self.config.get('compact_mode', False)
        
        logger.info(f"Echo Snapshot Logger initialized (level: {self.snapshot_level.value})")
    
    def capture_snapshot(self, fractal_decision: FractalDecision, 
                        collapse_state: Optional[CollapseState] = None,
                        vault_allocation: Optional[VaultAllocation] = None,
                        additional_data: Optional[Dict[str, Any]] = None) -> EchoSnapshot:
        """
        Capture comprehensive system snapshot.
        
        Args:
            fractal_decision: Current fractal decision
            collapse_state: Current collapse confidence state
            vault_allocation: Current vault allocation
            additional_data: Additional system data
            
        Returns:
            EchoSnapshot with captured system state
        """
        self.tick_counter += 1
        additional_data = additional_data or {}
        
        # Extract core metrics
        confidence = collapse_state.confidence if collapse_state else 0.0
        drift_angle = additional_data.get('drift_angle', 0.0)
        vault_status = vault_allocation.vault_status.value if vault_allocation else "unknown"
        
        # Create snapshot
        snapshot = EchoSnapshot(
            tick_id=self.tick_counter,
            timestamp=fractal_decision.timestamp,
            confidence=confidence,
            drift_angle=drift_angle,
            vault_status=vault_status,
            braid_signal=fractal_decision.fractal_signals.get('braid', 0.0),
            paradox_signal=fractal_decision.fractal_signals.get('paradox', 0.0),
            forever_signal=fractal_decision.fractal_signals.get('forever', 0.0),
            eco_signal=fractal_decision.fractal_signals.get('eco', 0.0),
            decision_action=fractal_decision.action,
            projected_profit=fractal_decision.projected_profit,
            volume_allocation=vault_allocation.allocated_volume if vault_allocation else 0.0,
            reasoning=fractal_decision.reasoning,
            fractal_weights=fractal_decision.fractal_weights.copy(),
            risk_metrics=fractal_decision.risk_assessment.copy(),
            ghost_activity=additional_data.get('ghost_activity', {}),
            lockout_status=additional_data.get('lockout_status', {})
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        self.total_snapshots += 1
        
        # Update tracking
        self.decision_counts[fractal_decision.action] += 1
        self.confidence_history.append(confidence)
        
        # Generate output
        if self.enable_terminal_output:
            self._print_terminal_snapshot(snapshot)
        
        if self.enable_file_logging:
            self._log_snapshot_to_file(snapshot)
        
        return snapshot
    
    def _print_terminal_snapshot(self, snapshot: EchoSnapshot):
        """Print formatted snapshot to terminal."""
        if self.snapshot_level == SnapshotLevel.MINIMAL:
            self._print_minimal_snapshot(snapshot)
        elif self.snapshot_level == SnapshotLevel.STANDARD:
            self._print_standard_snapshot(snapshot)
        elif self.snapshot_level == SnapshotLevel.DETAILED:
            self._print_detailed_snapshot(snapshot)
        else:  # DEBUG
            self._print_debug_snapshot(snapshot)
    
    def _print_minimal_snapshot(self, snapshot: EchoSnapshot):
        """Print minimal snapshot format."""
        confidence_color = self._get_confidence_color(snapshot.confidence)
        action_color = self._get_action_color(snapshot.decision_action)
        
        output = (f"[EchoTick-{snapshot.tick_id}] "
                 f"Conf={confidence_color}{snapshot.confidence:.2f}{self._reset_color()} | "
                 f"Action={action_color}{snapshot.decision_action.upper()}{self._reset_color()} | "
                 f"Profit={snapshot.projected_profit:.0f}bp")
        
        print(output)
    
    def _print_standard_snapshot(self, snapshot: EchoSnapshot):
        """Print standard snapshot format."""
        confidence_color = self._get_confidence_color(snapshot.confidence)
        action_color = self._get_action_color(snapshot.decision_action)
        vault_color = self._get_vault_color(snapshot.vault_status)
        
        # Main status line
        main_line = (f"[EchoTick-{snapshot.tick_id}] "
                    f"Conf={confidence_color}{snapshot.confidence:.2f}{self._reset_color()} | "
                    f"Drift={snapshot.drift_angle:.1f}° | "
                    f"Vault={vault_color}{snapshot.vault_status.upper()}{self._reset_color()} | "
                    f"Action={action_color}{snapshot.decision_action.upper()}{self._reset_color()}")
        
        # Signal line
        signal_line = (f"  Signals: B={snapshot.braid_signal:.2f} "
                      f"P={snapshot.paradox_signal:.2f} "
                      f"F={snapshot.forever_signal:.2f} "
                      f"E={snapshot.eco_signal:.2f}")
        
        # Profit line
        profit_line = (f"  Profit: {snapshot.projected_profit:.0f}bp | "
                      f"Volume: {snapshot.volume_allocation:.0f}")
        
        print(main_line)
        print(signal_line)
        print(profit_line)
        
        if not self.compact_mode:
            print()  # Empty line for readability
    
    def _print_detailed_snapshot(self, snapshot: EchoSnapshot):
        """Print detailed snapshot format."""
        print(f"{'='*80}")
        print(f"ECHO SNAPSHOT #{snapshot.tick_id} - {time.strftime('%H:%M:%S', time.localtime(snapshot.timestamp))}")
        print(f"{'='*80}")
        
        # Core metrics
        confidence_grade = self._get_confidence_grade(snapshot.confidence)
        print(f"🎯 CONFIDENCE: {snapshot.confidence:.3f} ({confidence_grade})")
        print(f"🌀 DRIFT ANGLE: {snapshot.drift_angle:.2f}°")
        print(f"🏦 VAULT STATUS: {snapshot.vault_status.upper()}")
        print(f"⚡ DECISION: {snapshot.decision_action.upper()}")
        
        # Fractal signals
        print(f"\n🔬 FRACTAL SIGNALS:")
        print(f"   Braid:   {snapshot.braid_signal:.3f}")
        print(f"   Paradox: {snapshot.paradox_signal:.3f}")
        print(f"   Forever: {snapshot.forever_signal:.3f}")
        print(f"   Eco:     {snapshot.eco_signal:.3f}")
        
        # Fractal weights
        print(f"\n⚖️  FRACTAL WEIGHTS:")
        for fractal, weight in snapshot.fractal_weights.items():
            print(f"   {fractal.capitalize()}: {weight:.3f}")
        
        # Profit metrics
        print(f"\n💰 PROFIT METRICS:")
        print(f"   Projected: {snapshot.projected_profit:.1f} basis points")
        print(f"   Volume:    {snapshot.volume_allocation:.0f}")
        
        # Risk assessment
        if snapshot.risk_metrics:
            print(f"\n⚠️  RISK ASSESSMENT:")
            for metric, value in snapshot.risk_metrics.items():
                print(f"   {metric}: {value}")
        
        # Reasoning
        print(f"\n🧠 REASONING: {snapshot.reasoning}")
        
        print(f"{'='*80}\n")
    
    def _print_debug_snapshot(self, snapshot: EchoSnapshot):
        """Print complete debug snapshot."""
        self._print_detailed_snapshot(snapshot)
        
        # Additional debug information
        if snapshot.ghost_activity:
            print(f"👻 GHOST ACTIVITY:")
            for key, value in snapshot.ghost_activity.items():
                print(f"   {key}: {value}")
            print()
        
        if snapshot.lockout_status:
            print(f"🔒 LOCKOUT STATUS:")
            for key, value in snapshot.lockout_status.items():
                print(f"   {key}: {value}")
            print()
    
    def _log_snapshot_to_file(self, snapshot: EchoSnapshot):
        """Log snapshot to file in JSON format."""
        try:
            snapshot_dict = {
                "tick_id": snapshot.tick_id,
                "timestamp": snapshot.timestamp,
                "confidence": snapshot.confidence,
                "drift_angle": snapshot.drift_angle,
                "vault_status": snapshot.vault_status,
                "signals": {
                    "braid": snapshot.braid_signal,
                    "paradox": snapshot.paradox_signal,
                    "forever": snapshot.forever_signal,
                    "eco": snapshot.eco_signal
                },
                "decision": {
                    "action": snapshot.decision_action,
                    "projected_profit": snapshot.projected_profit,
                    "volume_allocation": snapshot.volume_allocation,
                    "reasoning": snapshot.reasoning
                },
                "fractal_weights": snapshot.fractal_weights,
                "risk_metrics": snapshot.risk_metrics,
                "ghost_activity": snapshot.ghost_activity,
                "lockout_status": snapshot.lockout_status
            }
            
            with open(self.log_file_path, 'a') as f:
                f.write(json.dumps(snapshot_dict) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to log snapshot to file: {e}")
    
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color code for confidence level."""
        if not self.use_colors:
            return ""
        
        if confidence >= 0.8:
            return "\033[92m"  # Green
        elif confidence >= 0.6:
            return "\033[93m"  # Yellow
        elif confidence >= 0.4:
            return "\033[96m"  # Cyan
        else:
            return "\033[91m"  # Red
    
    def _get_action_color(self, action: str) -> str:
        """Get color code for decision action."""
        if not self.use_colors:
            return ""
        
        color_map = {
            "long": "\033[92m",   # Green
            "short": "\033[91m",  # Red
            "hold": "\033[93m",   # Yellow
            "exit": "\033[95m"    # Magenta
        }
        return color_map.get(action.lower(), "")
    
    def _get_vault_color(self, vault_status: str) -> str:
        """Get color code for vault status."""
        if not self.use_colors:
            return ""
        
        color_map = {
            "open": "\033[92m",           # Green
            "restricted": "\033[93m",     # Yellow
            "emergency_lock": "\033[91m", # Red
            "closed": "\033[90m"          # Gray
        }
        return color_map.get(vault_status.lower(), "")
    
    def _reset_color(self) -> str:
        """Get color reset code."""
        return "\033[0m" if self.use_colors else ""
    
    def _get_confidence_grade(self, confidence: float) -> str:
        """Get human-readable confidence grade."""
        if confidence >= 0.9:
            return "EXCELLENT"
        elif confidence >= 0.7:
            return "GOOD"
        elif confidence >= 0.5:
            return "MODERATE"
        elif confidence >= 0.3:
            return "POOR"
        else:
            return "VERY POOR"
    
    def print_session_summary(self):
        """Print comprehensive session summary."""
        session_duration = time.time() - self.session_start_time
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0.0
        
        print(f"\n{'='*60}")
        print(f"ECHO SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Session Duration: {session_duration/60:.1f} minutes")
        print(f"Total Snapshots: {self.total_snapshots}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"\nDecision Distribution:")
        for action, count in self.decision_counts.items():
            percentage = (count / max(self.total_snapshots, 1)) * 100
            print(f"  {action.capitalize()}: {count} ({percentage:.1f}%)")
        
        if self.confidence_history:
            print(f"\nConfidence Statistics:")
            print(f"  Min: {min(self.confidence_history):.3f}")
            print(f"  Max: {max(self.confidence_history):.3f}")
            print(f"  Std: {np.std(list(self.confidence_history)):.3f}")
        
        print(f"{'='*60}\n")
    
    def export_snapshots(self, filename: str, format: str = "json") -> bool:
        """
        Export snapshots to file.
        
        Args:
            filename: Output filename
            format: Export format ("json" or "csv")
            
        Returns:
            True if export successful
        """
        try:
            if format.lower() == "json":
                self._export_json(filename)
            elif format.lower() == "csv":
                self._export_csv(filename)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Snapshots exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export snapshots: {e}")
            return False
    
    def _export_json(self, filename: str):
        """Export snapshots as JSON."""
        snapshot_data = []
        
        for snapshot in self.snapshots:
            snapshot_data.append({
                "tick_id": snapshot.tick_id,
                "timestamp": snapshot.timestamp,
                "confidence": snapshot.confidence,
                "drift_angle": snapshot.drift_angle,
                "vault_status": snapshot.vault_status,
                "braid_signal": snapshot.braid_signal,
                "paradox_signal": snapshot.paradox_signal,
                "forever_signal": snapshot.forever_signal,
                "eco_signal": snapshot.eco_signal,
                "decision_action": snapshot.decision_action,
                "projected_profit": snapshot.projected_profit,
                "volume_allocation": snapshot.volume_allocation,
                "reasoning": snapshot.reasoning,
                "fractal_weights": snapshot.fractal_weights,
                "risk_metrics": snapshot.risk_metrics
            })
        
        with open(filename, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
    
    def _export_csv(self, filename: str):
        """Export snapshots as CSV."""
        import csv
        
        fieldnames = [
            "tick_id", "timestamp", "confidence", "drift_angle", "vault_status",
            "braid_signal", "paradox_signal", "forever_signal", "eco_signal",
            "decision_action", "projected_profit", "volume_allocation", "reasoning"
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for snapshot in self.snapshots:
                writer.writerow({
                    "tick_id": snapshot.tick_id,
                    "timestamp": snapshot.timestamp,
                    "confidence": snapshot.confidence,
                    "drift_angle": snapshot.drift_angle,
                    "vault_status": snapshot.vault_status,
                    "braid_signal": snapshot.braid_signal,
                    "paradox_signal": snapshot.paradox_signal,
                    "forever_signal": snapshot.forever_signal,
                    "eco_signal": snapshot.eco_signal,
                    "decision_action": snapshot.decision_action,
                    "projected_profit": snapshot.projected_profit,
                    "volume_allocation": snapshot.volume_allocation,
                    "reasoning": snapshot.reasoning
                })
    
    def reset_session(self):
        """Reset session tracking."""
        self.snapshots.clear()
        self.tick_counter = 0
        self.session_start_time = time.time()
        self.total_snapshots = 0
        self.decision_counts = {"long": 0, "short": 0, "hold": 0, "exit": 0}
        self.confidence_history.clear()
        logger.info("Echo snapshot session reset")

# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Test echo snapshot logger
    logger_config = {
        'level': 'standard',
        'terminal_output': True,
        'use_colors': True,
        'compact_mode': False
    }
    
    echo_logger = EchoSnapshotLogger(logger_config)
    
    # Create mock fractal decision
    from fractal_controller import FractalDecision
    
    mock_decision = FractalDecision(
        timestamp=time.time(),
        action="long",
        confidence=0.75,
        projected_profit=85.0,
        hold_duration=12,
        fractal_signals={
            "braid": 0.8,
            "paradox": 0.6,
            "forever": 0.7,
            "eco": 0.5
        },
        fractal_weights={
            "braid": 1.2,
            "paradox": 0.9,
            "forever": 1.1,
            "eco": 0.8
        },
        risk_assessment={"volatility_risk": "moderate"},
        reasoning="High confidence braid signal with positive momentum"
    )
    
    # Create mock collapse state
    from collapse_confidence import CollapseState
    
    mock_collapse = CollapseState(
        timestamp=time.time(),
        collapse_id="test_collapse",
        confidence=0.75,
        profit_delta=85.0,
        braid_volatility=0.3,
        coherence=0.8,
        coherence_amplification=1.1,
        stability_epsilon=0.01,
        raw_confidence=2.5
    )
    
    # Capture snapshot
    snapshot = echo_logger.capture_snapshot(
        mock_decision, 
        mock_collapse,
        additional_data={"drift_angle": 15.5}
    )
    
    print(f"Captured snapshot: {snapshot.tick_id}")
    
    # Test multiple snapshots
    for i in range(5):
        mock_decision.action = np.random.choice(["long", "short", "hold"])
        mock_decision.confidence = np.random.uniform(0.3, 0.9)
        mock_decision.projected_profit = np.random.uniform(-50, 150)
        
        echo_logger.capture_snapshot(mock_decision, mock_collapse)
    
    # Print session summary
    echo_logger.print_session_summary() 