"""
CLI Dashboard for Schwabot System
Real-time monitoring of NCCOs and system state
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import logging
import json
import time
import os
from pathlib import Path
from .event_bus import EventBus
from .ncco_generator import NCCOGenerator
from .replay_engine import ReplayEngine

logger = logging.getLogger(__name__)

@dataclass
class DashboardState:
    """Container for dashboard state"""
    active: bool
    last_ncco: Optional[Dict]
    last_update: float
    metrics: Dict[str, Any]
    callbacks: Dict[str, List[callable]]

class CLIDashboard:
    """CLI dashboard for real-time monitoring"""
    
    def __init__(self, event_bus: EventBus, ncco_gen: NCCOGenerator, replay_engine: ReplayEngine, log_dir: str = "logs"):
        self.event_bus = event_bus
        self.ncco_gen = ncco_gen
        self.replay_engine = replay_engine
        self.state = DashboardState(
            active=False,
            last_ncco=None,
            last_update=0.0,
            metrics={},
            callbacks={}
        )
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event bus subscriptions"""
        metric_keys = [
            "price",
            "entropy_score",
            "coherence_score",
            "current_strategy",
            "ghost_hash",
            "paradox_phase",
            "velocity_class",
            "pattern_cluster",
            "liquidity_status",
            "smart_money_score",
            "panic_zone"
        ]
        
        for key in metric_keys:
            self.event_bus.subscribe(key, lambda state, k=key: self._handle_metric_update(k, state))
    
    def _handle_metric_update(self, key: str, state: Any) -> None:
        """
        Handle metric updates
        
        Args:
            key: Metric key
            state: Event state
        """
        with self._lock:
            self.state.metrics[key] = state.value
            self.state.last_update = datetime.now().timestamp()
    
    def start(self) -> None:
        """Start dashboard"""
        if self.state.active:
            return
        
        self.state.active = True
        self._stop_event.clear()
        
        # Start update thread
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
        
        self.logger.info("Started CLI dashboard")
    
    def stop(self) -> None:
        """Stop dashboard"""
        if not self.state.active:
            return
        
        self.state.active = False
        self._stop_event.set()
        self._update_thread.join()
        
        self.logger.info("Stopped CLI dashboard")
    
    def _update_loop(self) -> None:
        """Dashboard update loop"""
        while not self._stop_event.is_set():
            try:
                # Get latest NCCO
                recent_nccos = self.ncco_gen.get_recent_nccos(1)
                if recent_nccos:
                    with self._lock:
                        self.state.last_ncco = recent_nccos[0]
                
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Print dashboard
                self._print_dashboard()
                
                # Sleep
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error updating dashboard: {e}")
                continue
    
    def _print_dashboard(self) -> None:
        """Print dashboard to console"""
        with self._lock:
            # Print header
            print("=" * 80)
            print("Schwabot CLI Dashboard")
            print("=" * 80)
            print()
            
            # Print current metrics
            print("Current Metrics:")
            print("-" * 40)
            for key, value in self.state.metrics.items():
                print(f"{key:20}: {value}")
            print()
            
            # Print last NCCO
            if self.state.last_ncco:
                print("Last NCCO:")
                print("-" * 40)
                print(f"ID: {self.state.last_ncco['ncco_id']}")
                print(f"Type: {self.state.last_ncco['ncco_type']}")
                print(f"Time: {datetime.fromtimestamp(self.state.last_ncco['timestamp'])}")
                print(f"Price: ${self.state.last_ncco['price']:.2f}")
                print(f"Strategy: {self.state.last_ncco['strategy']}")
                print(f"Ghost Hash: {self.state.last_ncco['ghost_hash']}")
                print(f"Entropy: {self.state.last_ncco['entropy']:.2f}")
                print(f"Coherence: {self.state.last_ncco['coherence']:.2f}")
                print(f"Paradox Phase: {self.state.last_ncco['paradox_phase']}")
                print(f"Velocity: {self.state.last_ncco['velocity_class']}")
                print(f"Pattern: {self.state.last_ncco['pattern_cluster']}")
                print(f"Liquidity: {self.state.last_ncco['liquidity_status']}")
                print(f"Smart Money: {self.state.last_ncco['smart_money_score']:.2f}")
                print(f"Panic Zone: {self.state.last_ncco['panic_zone']}")
                print()
            
            # Print replay status
            replay_state = self.replay_engine.get_replay_state()
            if replay_state["active"]:
                print("Replay Status:")
                print("-" * 40)
                print(f"Active: Yes")
                print(f"Speed: {replay_state['speed']}x")
                print(f"Progress: {replay_state['progress']*100:.1f}%")
                print()
            
            # Print footer
            print("=" * 80)
            print(f"Last Update: {datetime.fromtimestamp(self.state.last_update)}")
            print("=" * 80)
    
    def register_callback(self, event: str, callback: callable) -> None:
        """
        Register callback for dashboard events
        
        Args:
            event: Event type
            callback: Callback function
        """
        with self._lock:
            if event not in self.state.callbacks:
                self.state.callbacks[event] = []
            self.state.callbacks[event].append(callback)
    
    def unregister_callback(self, event: str, callback: callable) -> None:
        """
        Unregister callback
        
        Args:
            event: Event type
            callback: Callback function to remove
        """
        with self._lock:
            if event in self.state.callbacks and callback in self.state.callbacks[event]:
                self.state.callbacks[event].remove(callback)
    
    def _notify_callbacks(self, event: str, data: Any) -> None:
        """
        Notify callbacks of event
        
        Args:
            event: Event type
            data: Event data
        """
        with self._lock:
            if event in self.state.callbacks:
                for callback in self.state.callbacks[event]:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in callback: {e}")
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """
        Get current dashboard state
        
        Returns:
            Dictionary of dashboard state
        """
        with self._lock:
            return {
                "active": self.state.active,
                "last_ncco": self.state.last_ncco,
                "last_update": self.state.last_update,
                "metrics": self.state.metrics
            } 