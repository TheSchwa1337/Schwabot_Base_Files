"""
Replay Engine for Schwabot System
Enables NCCO-based decision replay and analysis
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import logging
import json
import time
from pathlib import Path
from .event_bus import EventBus
from .ncco_generator import NCCOGenerator

@dataclass
class ReplayState:
    """Container for replay state"""
    active: bool
    current_ncco: Optional[Dict]
    start_time: float
    end_time: float
    speed: float
    callbacks: Dict[str, List[Callable]]

class ReplayEngine:
    """Engine for replaying NCCO-based decisions"""
    
    def __init__(self, event_bus: EventBus, ncco_gen: NCCOGenerator, log_dir: str = "logs"):
        self.event_bus = event_bus
        self.ncco_gen = ncco_gen
        self.state = ReplayState(
            active=False,
            current_ncco=None,
            start_time=0.0,
            end_time=0.0,
            speed=1.0,
            callbacks={}
        )
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / "replay_engine.log",
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ReplayEngine')
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback for replay events
        
        Args:
            event: Event type (e.g., "ncco", "state_change")
            callback: Callback function
        """
        with self._lock:
            if event not in self.state.callbacks:
                self.state.callbacks[event] = []
            self.state.callbacks[event].append(callback)
    
    def unregister_callback(self, event: str, callback: Callable) -> None:
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
    
    def replay_sequence(self, nccos: List[Dict], speed: float = 1.0) -> None:
        """
        Replay sequence of NCCOs
        
        Args:
            nccos: List of NCCOs to replay
            speed: Replay speed multiplier
        """
        if not nccos:
            return
        
        with self._lock:
            self.state.active = True
            self.state.speed = speed
            self.state.start_time = nccos[0]["timestamp"]
            self.state.end_time = nccos[-1]["timestamp"]
            self._stop_event.clear()
        
        # Start replay thread
        self._replay_thread = threading.Thread(target=self._replay_loop, args=(nccos,))
        self._replay_thread.daemon = True
        self._replay_thread.start()
        
        self.logger.info(f"Started replay of {len(nccos)} NCCOs at {speed}x speed")
    
    def _replay_loop(self, nccos: List[Dict]) -> None:
        """
        Replay loop
        
        Args:
            nccos: List of NCCOs to replay
        """
        last_timestamp = None
        
        for ncco in nccos:
            if self._stop_event.is_set():
                break
            
            # Calculate delay
            if last_timestamp is not None:
                delay = (ncco["timestamp"] - last_timestamp) / self.state.speed
                time.sleep(max(0, delay))
            
            # Update state
            with self._lock:
                self.state.current_ncco = ncco
            
            # Replay NCCO
            self._replay_ncco(ncco)
            
            last_timestamp = ncco["timestamp"]
        
        # End replay
        with self._lock:
            self.state.active = False
            self.state.current_ncco = None
        
        self._notify_callbacks("replay_end", None)
        self.logger.info("Replay completed")
    
    def _replay_ncco(self, ncco: Dict) -> None:
        """
        Replay single NCCO
        
        Args:
            ncco: NCCO to replay
        """
        # Update event bus
        self.event_bus.update("price", ncco["price"], "replay")
        self.event_bus.update("entropy_score", ncco["entropy"], "replay")
        self.event_bus.update("coherence_score", ncco["coherence"], "replay")
        self.event_bus.update("ghost_hash", ncco["ghost_hash"], "replay")
        self.event_bus.update("current_strategy", ncco["strategy"], "replay")
        self.event_bus.update("paradox_phase", ncco["paradox_phase"], "replay")
        self.event_bus.update("velocity_class", ncco["velocity_class"], "replay")
        self.event_bus.update("pattern_cluster", ncco["pattern_cluster"], "replay")
        self.event_bus.update("liquidity_status", ncco["liquidity_status"], "replay")
        self.event_bus.update("smart_money_score", ncco["smart_money_score"], "replay")
        self.event_bus.update("panic_zone", ncco["panic_zone"], "replay")
        
        # Notify callbacks
        self._notify_callbacks("ncco", ncco)
    
    def stop_replay(self) -> None:
        """Stop current replay"""
        with self._lock:
            if not self.state.active:
                return
            
            self._stop_event.set()
            self.state.active = False
            self.state.current_ncco = None
        
        self.logger.info("Replay stopped")
    
    def get_replay_state(self) -> Dict[str, Any]:
        """
        Get current replay state
        
        Returns:
            Dictionary of replay state
        """
        with self._lock:
            return {
                "active": self.state.active,
                "current_ncco": self.state.current_ncco,
                "start_time": self.state.start_time,
                "end_time": self.state.end_time,
                "speed": self.state.speed,
                "progress": (
                    (self.state.current_ncco["timestamp"] - self.state.start_time) /
                    (self.state.end_time - self.state.start_time)
                    if self.state.active and self.state.current_ncco
                    else 0.0
                )
            }
    
    def replay_ghost_hash(self, ghost_hash: str, speed: float = 1.0) -> None:
        """
        Replay all NCCOs with matching ghost hash
        
        Args:
            ghost_hash: Ghost hash to match
            speed: Replay speed multiplier
        """
        nccos = self.ncco_gen.get_nccos_by_ghost_hash(ghost_hash)
        if nccos:
            self.replay_sequence(nccos, speed)
        else:
            self.logger.warning(f"No NCCOs found for ghost hash: {ghost_hash}")
    
    def replay_time_range(self, start_time: float, end_time: float, speed: float = 1.0) -> None:
        """
        Replay NCCOs within time range
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            speed: Replay speed multiplier
        """
        nccos = [
            n for n in self.ncco_gen.load_all_nccos()
            if start_time <= n["timestamp"] <= end_time
        ]
        if nccos:
            self.replay_sequence(nccos, speed)
        else:
            self.logger.warning(f"No NCCOs found in time range: {start_time} - {end_time}")
    
    def replay_last_nccos(self, count: int = 100, speed: float = 1.0) -> None:
        """
        Replay most recent NCCOs
        
        Args:
            count: Number of NCCOs to replay
            speed: Replay speed multiplier
        """
        nccos = self.ncco_gen.get_recent_nccos(count)
        if nccos:
            self.replay_sequence(nccos, speed)
        else:
            self.logger.warning("No recent NCCOs found") 