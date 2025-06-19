"""
Integrated ALIF/ALEPH System
============================
Complete integration of tick management, ghost recovery, ALEPH core modules,
and NCCO systems for real-time trading intelligence.
"""

import asyncio
import threading
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Import our custom modules
from .tick_management_system import (
    TickManager, RuntimeCounters, CompressionMode, 
    ALIFCore, ALEPHCore, TickContext, create_tick_manager
)
from .ghost_data_recovery import (
    GhostDataRecoveryManager, create_ghost_recovery_manager
)

# Import ALEPH and NCCO cores
try:
    from aleph_core import (
        DetonationSequencer, EntropyAnalyzer, PatternMatcher,
        SmartMoneyAnalyzer, ParadoxVisualizer
    )
    from ncco_core import NCCO, generate_nccos, score_nccos
    CORES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core modules not available: {e}")
    CORES_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SystemHealthMetrics:
    """Comprehensive system health metrics"""
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    total_ticks_processed: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    ghost_recoveries: int = 0
    system_errors: int = 0
    average_tick_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.successful_trades + self.failed_trades
        return self.successful_trades / total if total > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        return self.system_errors / self.total_ticks_processed if self.total_ticks_processed > 0 else 0.0

@dataclass
class TradingDecision:
    """Complete trading decision with ALIF/ALEPH consensus"""
    tick_id: int
    timestamp: datetime
    action: str  # "EXECUTE", "HOLD", "VAULT"
    confidence: float
    alif_contribution: Dict[str, Any]
    aleph_contribution: Dict[str, Any]
    smart_money_score: float = 0.0
    pattern_confidence: float = 0.0
    detonation_approved: bool = False
    risk_assessment: str = "UNKNOWN"
    execution_priority: str = "LOW"

class IntegratedALIFALEPHSystem:
    """
    Complete integrated system combining:
    - ALIF (Adaptive Logic Integration Framework)
    - ALEPH (Autonomous Logic Execution Path Hierarchy)  
    - Tick Management with drift correction
    - Ghost data recovery
    - Smart Money analysis
    - Pattern matching and detonation sequencing
    """
    
    def __init__(self, 
                 tick_interval: float = 1.0,
                 log_directory: str = "logs",
                 enable_recovery: bool = True):
        
        self.start_time = datetime.now()
        self.enabled = False
        
        # Initialize core components
        self.tick_manager = create_tick_manager(tick_interval)
        
        if enable_recovery:
            self.recovery_manager = create_ghost_recovery_manager(log_directory)
        else:
            self.recovery_manager = None
        
        # Initialize ALEPH modules if available
        if CORES_AVAILABLE:
            self.detonation_sequencer = DetonationSequencer()
            self.entropy_analyzer = EntropyAnalyzer()
            self.pattern_matcher = PatternMatcher()
            self.smart_money_analyzer = SmartMoneyAnalyzer()
            logger.info("ALEPH modules initialized successfully")
        else:
            logger.warning("ALEPH modules not available - running in simulation mode")
        
        # System metrics
        self.health_metrics = SystemHealthMetrics()
        self.trading_decisions = []
        self.active_threads = []
        
        # Decision callbacks
        self.decision_callbacks: List[Callable] = []
        
        # Setup tick callback
        self.tick_manager.register_callback(self._process_tick_with_intelligence)
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        logger.info("Integrated ALIF/ALEPH System initialized")
    
    def register_decision_callback(self, callback: Callable[[TradingDecision], None]):
        """Register callback for trading decisions"""
        self.decision_callbacks.append(callback)
    
    def start_system(self):
        """Start the complete integrated system"""
        if self.enabled:
            logger.warning("System already running")
            return
        
        self.enabled = True
        self._stop_event.clear()
        
        logger.info("🚀 Starting Integrated ALIF/ALEPH System...")
        
        # Start tick processing thread
        tick_thread = threading.Thread(target=self._tick_processing_loop, daemon=True)
        tick_thread.start()
        self.active_threads.append(tick_thread)
        
        # Start health monitoring thread
        health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        health_thread.start()
        self.active_threads.append(health_thread)
        
        # Start recovery monitoring if enabled
        if self.recovery_manager:
            recovery_thread = threading.Thread(target=self._recovery_monitoring_loop, daemon=True)
            recovery_thread.start()
            self.active_threads.append(recovery_thread)
        
        logger.info(f"✅ System started with {len(self.active_threads)} active threads")
    
    def stop_system(self):
        """Stop the system gracefully"""
        if not self.enabled:
            return
        
        logger.info("🛑 Stopping Integrated ALIF/ALEPH System...")
        
        self.enabled = False
        self._stop_event.set()
        
        # Wait for threads to complete
        for thread in self.active_threads:
            thread.join(timeout=5.0)
        
        # Export final diagnostics
        self._export_final_diagnostics()
        
        logger.info("✅ System stopped successfully")
    
    def _tick_processing_loop(self):
        """Main tick processing loop"""
        logger.info("Tick processing loop started")
        
        while self.enabled and not self._stop_event.is_set():
            try:
                # Process tick cycle
                tick_context = self.tick_manager.run_tick_cycle()
                
                if tick_context:
                    with self._lock:
                        self.health_metrics.total_ticks_processed += 1
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Tick processing error: {e}")
                with self._lock:
                    self.health_metrics.system_errors += 1
        
        logger.info("Tick processing loop stopped")
    
    def _health_monitoring_loop(self):
        """Health monitoring and metrics updates"""
        logger.info("Health monitoring loop started")
        
        while self.enabled and not self._stop_event.is_set():
            try:
                # Update health metrics
                self._update_health_metrics()
                
                # Check for critical errors
                if self.health_metrics.error_rate > 0.1:  # 10% error rate
                    logger.warning(f"High error rate detected: {self.health_metrics.error_rate:.1%}")
                
                # Sleep for 10 seconds between health checks
                time.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
        
        logger.info("Health monitoring loop stopped")
    
    def _recovery_monitoring_loop(self):
        """Ghost data recovery monitoring"""
        if not self.recovery_manager:
            return
        
        logger.info("Recovery monitoring loop started")
        
        while self.enabled and not self._stop_event.is_set():
            try:
                # Perform recovery scan every 5 minutes
                recovery_results = self.recovery_manager.full_system_recovery_scan()
                
                recovered_count = recovery_results["recovery_stats"]["total_recoveries"]
                if recovered_count > 0:
                    with self._lock:
                        self.health_metrics.ghost_recoveries += recovered_count
                    logger.info(f"Recovery scan completed: {recovered_count} items recovered")
                
                # Sleep for 5 minutes between recovery scans
                time.sleep(300.0)
                
            except Exception as e:
                logger.error(f"Recovery monitoring error: {e}")
        
        logger.info("Recovery monitoring loop stopped")
    
    def _process_tick_with_intelligence(self, tick_context: TickContext, 
                                       alif_result: Dict, aleph_result: Dict):
        """Process tick with full intelligence integration"""
        try:
            # Create trading decision
            decision = self._create_trading_decision(tick_context, alif_result, aleph_result)
            
            # Store decision
            with self._lock:
                self.trading_decisions.append(decision)
                
                # Keep only last 1000 decisions
                if len(self.trading_decisions) > 1000:
                    self.trading_decisions = self.trading_decisions[-1000:]
            
            # Execute decision callbacks
            for callback in self.decision_callbacks:
                try:
                    callback(decision)
                except Exception as e:
                    logger.error(f"Decision callback error: {e}")
            
            # Log decision
            self._log_trading_decision(decision)
            
        except Exception as e:
            logger.error(f"Intelligence processing error: {e}")
            with self._lock:
                self.health_metrics.system_errors += 1
    
    def _create_trading_decision(self, tick_context: TickContext, 
                               alif_result: Dict, aleph_result: Dict) -> TradingDecision:
        """Create comprehensive trading decision"""
        
        # Determine base action from ALIF/ALEPH
        alif_action = alif_result.get("action", "unknown")
        aleph_action = aleph_result.get("action", "unknown")
        
        # Calculate consensus
        action, confidence = self._calculate_consensus(alif_action, aleph_action, tick_context)
        
        # Get smart money analysis if available
        smart_money_score = 0.0
        pattern_confidence = 0.0
        detonation_approved = False
        
        if CORES_AVAILABLE and hasattr(self, 'smart_money_analyzer'):
            try:
                # Simulate market data for smart money analysis
                market_data = self._generate_market_simulation(tick_context)
                
                smart_money_metrics = self.smart_money_analyzer.analyze_tick(
                    price=market_data["price"],
                    volume=market_data["volume"],
                    order_book=market_data["order_book"],
                    trades=market_data["trades"]
                )
                
                smart_money_score = smart_money_metrics.smart_money_score
                
                # Check detonation sequencer
                detonation_result = self.detonation_sequencer.initiate_detonation(
                    payload={"pattern": alif_action, "confidence": confidence},
                    price=market_data["price"],
                    volume=market_data["volume"],
                    order_book=market_data["order_book"],
                    trades=market_data["trades"]
                )
                
                detonation_approved = detonation_result["detonation_activated"]
                pattern_confidence = detonation_result["confidence"]
                
            except Exception as e:
                logger.error(f"Smart money analysis error: {e}")
        
        # Determine risk assessment and priority
        risk_assessment = self._assess_risk(tick_context, confidence, smart_money_score)
        execution_priority = self._determine_priority(action, confidence, smart_money_score)
        
        return TradingDecision(
            tick_id=tick_context.tick_id,
            timestamp=tick_context.timestamp,
            action=action,
            confidence=confidence,
            alif_contribution=alif_result,
            aleph_contribution=aleph_result,
            smart_money_score=smart_money_score,
            pattern_confidence=pattern_confidence,
            detonation_approved=detonation_approved,
            risk_assessment=risk_assessment,
            execution_priority=execution_priority
        )
    
    def _calculate_consensus(self, alif_action: str, aleph_action: str, 
                           tick_context: TickContext) -> Tuple[str, float]:
        """Calculate consensus action and confidence from ALIF/ALEPH"""
        
        # Action priority matrix
        action_matrix = {
            ("broadcast_glyph", "confirm"): ("EXECUTE", 0.9),
            ("broadcast_glyph", "hold"): ("HOLD", 0.6),
            ("compress", "confirm"): ("HOLD", 0.4),
            ("compress", "hold"): ("VAULT", 0.8),
            ("hold", "confirm"): ("HOLD", 0.7),
            ("hold", "hold"): ("VAULT", 0.9)
        }
        
        action_key = (alif_action, aleph_action)
        
        if action_key in action_matrix:
            base_action, base_confidence = action_matrix[action_key]
        else:
            base_action, base_confidence = ("HOLD", 0.5)
        
        # Adjust confidence based on entropy and echo strength
        confidence_adjustment = 0.0
        
        if tick_context.entropy < 0.3:
            confidence_adjustment += 0.1
        elif tick_context.entropy > 0.8:
            confidence_adjustment -= 0.2
        
        if tick_context.echo_strength > 0.7:
            confidence_adjustment += 0.1
        elif tick_context.echo_strength < 0.4:
            confidence_adjustment -= 0.1
        
        final_confidence = max(0.0, min(1.0, base_confidence + confidence_adjustment))
        
        return base_action, final_confidence
    
    def _generate_market_simulation(self, tick_context: TickContext) -> Dict:
        """Generate simulated market data for analysis"""
        # Simple simulation - replace with real market data
        base_price = 50000.0
        price_variation = np.sin(tick_context.tick_id * 0.1) * 1000.0
        
        return {
            "price": base_price + price_variation,
            "volume": 100.0 + tick_context.entropy * 50.0,
            "order_book": {
                "bids": [[49500.0, 1.0], [49000.0, 2.0]],
                "asks": [[50500.0, 1.0], [51000.0, 2.0]]
            },
            "trades": [
                {"price": base_price, "volume": 0.1, "side": "buy"},
                {"price": base_price - 10, "volume": 0.2, "side": "sell"}
            ]
        }
    
    def _assess_risk(self, tick_context: TickContext, confidence: float, 
                    smart_money_score: float) -> str:
        """Assess overall risk level"""
        risk_score = 0.0
        
        # Factor in entropy (high entropy = high risk)
        risk_score += tick_context.entropy * 0.3
        
        # Factor in confidence (low confidence = high risk)
        risk_score += (1.0 - confidence) * 0.4
        
        # Factor in smart money (low smart money = high risk)
        risk_score += (1.0 - smart_money_score) * 0.2
        
        # Factor in drift (high drift = high risk)
        risk_score += tick_context.drift_score * 0.1
        
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.6:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _determine_priority(self, action: str, confidence: float, 
                          smart_money_score: float) -> str:
        """Determine execution priority"""
        if action == "EXECUTE" and confidence > 0.8 and smart_money_score > 0.7:
            return "HIGH"
        elif action == "EXECUTE" and confidence > 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _log_trading_decision(self, decision: TradingDecision):
        """Log trading decision for analysis"""
        log_entry = {
            "tick_id": decision.tick_id,
            "timestamp": decision.timestamp.isoformat(),
            "action": decision.action,
            "confidence": decision.confidence,
            "smart_money_score": decision.smart_money_score,
            "pattern_confidence": decision.pattern_confidence,
            "detonation_approved": decision.detonation_approved,
            "risk_assessment": decision.risk_assessment,
            "execution_priority": decision.execution_priority
        }
        
        # Use tick manager's stack checker for secure logging
        filename = f"decision_{decision.tick_id:06d}.json"
        self.tick_manager.stack_checker.secure_write(filename, log_entry)
    
    def _update_health_metrics(self):
        """Update system health metrics"""
        current_time = datetime.now()
        
        with self._lock:
            # Update uptime
            self.health_metrics.uptime = current_time - self.start_time
            
            # Calculate average tick time
            tick_times = list(self.tick_manager.tick_times)
            if tick_times:
                self.health_metrics.average_tick_time = sum(tick_times) / len(tick_times)
            
            # Update success/failure rates from decisions
            recent_decisions = [d for d in self.trading_decisions 
                             if current_time - d.timestamp < timedelta(minutes=5)]
            
            if recent_decisions:
                executed_decisions = [d for d in recent_decisions if d.action == "EXECUTE"]
                self.health_metrics.successful_trades = len(executed_decisions)
                self.health_metrics.failed_trades = len(recent_decisions) - len(executed_decisions)
    
    def _export_final_diagnostics(self):
        """Export comprehensive diagnostics on shutdown"""
        diagnostics = {
            "session_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_runtime": str(self.health_metrics.uptime)
            },
            "health_metrics": self.health_metrics.__dict__,
            "tick_manager_status": self.tick_manager.get_system_status(),
            "recent_decisions": [
                {
                    "tick_id": d.tick_id,
                    "action": d.action,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp.isoformat()
                } for d in self.trading_decisions[-50:]  # Last 50 decisions
            ]
        }
        
        # Add recovery stats if available
        if self.recovery_manager:
            recovery_results = self.recovery_manager.full_system_recovery_scan()
            diagnostics["recovery_stats"] = recovery_results
        
        # Export to file
        diagnostics_file = f"final_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(diagnostics_file, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        logger.info(f"Final diagnostics exported to {diagnostics_file}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self._lock:
            return {
                "enabled": self.enabled,
                "uptime": str(self.health_metrics.uptime),
                "health_metrics": self.health_metrics.__dict__,
                "tick_manager": self.tick_manager.get_system_status(),
                "active_threads": len(self.active_threads),
                "recent_decisions": len([d for d in self.trading_decisions 
                                       if datetime.now() - d.timestamp < timedelta(minutes=1)]),
                "cores_available": CORES_AVAILABLE
            }
    
    def get_recent_decisions(self, count: int = 10) -> List[TradingDecision]:
        """Get recent trading decisions"""
        with self._lock:
            return self.trading_decisions[-count:] if self.trading_decisions else []

# Factory function
def create_integrated_system(tick_interval: float = 1.0, 
                           log_directory: str = "logs",
                           enable_recovery: bool = True) -> IntegratedALIFALEPHSystem:
    """Create configured integrated ALIF/ALEPH system"""
    return IntegratedALIFALEPHSystem(
        tick_interval=tick_interval,
        log_directory=log_directory,
        enable_recovery=enable_recovery
    )

# Example usage and demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create integrated system
    system = create_integrated_system(tick_interval=1.0)
    
    # Register decision callback
    def decision_callback(decision: TradingDecision):
        print(f"[Decision {decision.tick_id}] {decision.action} "
              f"(confidence: {decision.confidence:.2f}, "
              f"risk: {decision.risk_assessment})")
    
    system.register_decision_callback(decision_callback)
    
    try:
        # Start system
        system.start_system()
        
        # Run for demonstration
        print("🚀 Integrated ALIF/ALEPH System Demo")
        print("📊 Running for 30 seconds...")
        
        for i in range(30):
            status = system.get_system_status()
            recent_decisions = system.get_recent_decisions(count=5)
            
            print(f"\n[{i+1}/30] System Status:")
            print(f"  Uptime: {status['uptime']}")
            print(f"  Total Ticks: {status['health_metrics']['total_ticks_processed']}")
            print(f"  Recent Decisions: {len(recent_decisions)}")
            print(f"  Success Rate: {status['health_metrics']['success_rate']:.1%}")
            
            time.sleep(1.0)
        
        print("\n✅ Demo completed successfully")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    finally:
        # Stop system
        system.stop_system()
        print("🔚 System shutdown complete") 