"""
Panic Pause Manager for Schwabot System
Manages trading pauses during panic zones
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import logging
from pathlib import Path
from .event_bus import EventBus

@dataclass
class PauseState:
    """Container for pause state"""
    active: bool
    start_time: float
    end_time: float
    reason: str
    metrics: Dict[str, float]
    cooldown_end: float

class PanicPauseManager:
    """Manages trading pauses during panic zones"""
    
    def __init__(self, event_bus: EventBus, log_dir: str = "logs"):
        self.event_bus = event_bus
        self.state = PauseState(
            active=False,
            start_time=0.0,
            end_time=0.0,
            reason="",
            metrics={},
            cooldown_end=0.0
        )
        
        # Pause configuration
        self.min_pause_duration = 5.0  # seconds
        self.max_pause_duration = 30.0  # seconds
        self.cooldown_duration = 60.0  # seconds
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / "panic_pause.log",
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PanicPauseManager')
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Subscribe to event bus
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event bus subscriptions"""
        self.event_bus.subscribe("panic_zone", self._handle_panic_zone)
        self.event_bus.subscribe("entropy_score", self._handle_entropy_update)
        self.event_bus.subscribe("coherence_score", self._handle_coherence_update)
    
    def _handle_panic_zone(self, state: Any) -> None:
        """
        Handle panic zone event
        
        Args:
            state: Event state
        """
        with self._lock:
            if state.value and not self.state.active:
                self._start_pause("panic_zone", {
                    "entropy": self.event_bus.get("entropy_score", 0.0),
                    "coherence": self.event_bus.get("coherence_score", 0.0)
                })
            elif not state.value and self.state.active:
                self._end_pause()
    
    def _handle_entropy_update(self, state: Any) -> None:
        """
        Handle entropy score update
        
        Args:
            state: Event state
        """
        with self._lock:
            if state.value > 4.5 and not self.state.active:
                self._start_pause("high_entropy", {
                    "entropy": state.value,
                    "coherence": self.event_bus.get("coherence_score", 0.0)
                })
    
    def _handle_coherence_update(self, state: Any) -> None:
        """
        Handle coherence score update
        
        Args:
            state: Event state
        """
        with self._lock:
            if state.value < 0.4 and not self.state.active:
                self._start_pause("low_coherence", {
                    "entropy": self.event_bus.get("entropy_score", 0.0),
                    "coherence": state.value
                })
    
    def _start_pause(self, reason: str, metrics: Dict[str, float]) -> None:
        """
        Start a trading pause
        
        Args:
            reason: Pause reason
            metrics: Current metrics
        """
        current_time = datetime.now().timestamp()
        
        # Check cooldown
        if current_time < self.state.cooldown_end:
            self.logger.info(f"Pause requested but in cooldown until {self.state.cooldown_end}")
            return
        
        # Calculate pause duration
        duration = min(
            max(self.min_pause_duration, metrics.get("entropy", 0.0) * 5.0),
            self.max_pause_duration
        )
        
        # Update state
        self.state = PauseState(
            active=True,
            start_time=current_time,
            end_time=current_time + duration,
            reason=reason,
            metrics=metrics,
            cooldown_end=current_time + duration + self.cooldown_duration
        )
        
        # Update event bus
        self.event_bus.update("trading_enabled", False, "panic_pause", {
            "reason": reason,
            "duration": duration,
            "metrics": metrics
        })
        
        self.logger.warning(
            f"Started pause: {reason} for {duration:.1f}s "
            f"(entropy: {metrics.get('entropy', 0.0):.2f}, "
            f"coherence: {metrics.get('coherence', 0.0):.2f})"
        )
    
    def _end_pause(self) -> None:
        """End current trading pause"""
        with self._lock:
            if not self.state.active:
                return
            
            current_time = datetime.now().timestamp()
            
            # Update state
            self.state.active = False
            
            # Update event bus
            self.event_bus.update("trading_enabled", True, "panic_pause", {
                "reason": self.state.reason,
                "duration": current_time - self.state.start_time,
                "metrics": self.state.metrics
            })
            
            self.logger.info(
                f"Ended pause: {self.state.reason} "
                f"after {current_time - self.state.start_time:.1f}s"
            )
    
    def check_pause(self) -> None:
        """Check if current pause should end"""
        with self._lock:
            if not self.state.active:
                return
            
            current_time = datetime.now().timestamp()
            
            # Check if pause should end
            if current_time >= self.state.end_time:
                self._end_pause()
    
    def get_pause_state(self) -> Dict[str, Any]:
        """
        Get current pause state
        
        Returns:
            Dictionary of pause state
        """
        with self._lock:
            return {
                "active": self.state.active,
                "start_time": self.state.start_time,
                "end_time": self.state.end_time,
                "reason": self.state.reason,
                "metrics": self.state.metrics,
                "cooldown_end": self.state.cooldown_end,
                "time_remaining": max(0.0, self.state.end_time - datetime.now().timestamp()) if self.state.active else 0.0,
                "cooldown_remaining": max(0.0, self.state.cooldown_end - datetime.now().timestamp())
            }
    
    def force_end_pause(self) -> None:
        """Force end current pause"""
        with self._lock:
            if not self.state.active:
                return
            
            self._end_pause()
            self.logger.warning("Pause force ended")
    
    def is_paused(self) -> bool:
        """
        Check if system is currently paused
        
        Returns:
            True if system is paused
        """
        with self._lock:
            return self.state.active
    
    def get_cooldown_remaining(self) -> float:
        """
        Get remaining cooldown time
        
        Returns:
            Remaining cooldown time in seconds
        """
        with self._lock:
            return max(0.0, self.state.cooldown_end - datetime.now().timestamp()) 