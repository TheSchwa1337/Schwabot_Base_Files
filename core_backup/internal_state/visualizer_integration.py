# -*- coding: utf-8 -*-
""""""
Visualizer Integration Module
============================

Connects the StateContinuityManager to existing visualizers and panel systems.
Ensures proper data flow, prevents JSON hang-ups, and maintains continuous
functionality across all visualization components.

Features:
- Integration with SpeedLatticeLivePanelSystem
- Integration with MathLibV3Visualizer
- JSON hang-up prevention with timeout handling
- Real-time data synchronization
- Error recovery and fallback mechanisms
- Lint-compliant code with proper type hints
""""""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

# Import core modules with safe fallbacks
try:
    from core.internal_state.state_continuity_manager import StateContinuityManager, StateSnapshot, StateType
except ImportError:
    StateContinuityManager = None
    StateType = None
    StateSnapshot = None

try:
    from core.speed_lattice_visualizer import PanelType, SpeedLatticeLivePanelSystem
except ImportError:
    SpeedLatticeLivePanelSystem = None
    PanelType = None

try:
    from core.mathlib_v3_visualizer import MathLibV3Visualizer
except ImportError:
    MathLibV3Visualizer = None

logger = logging.getLogger(__name__)


class VisualizerIntegration:
    """"""
    Integrates state continuity manager with visualizers and panel systems.
    """"""

    def __init__(self, state_manager: Optional[StateContinuityManager] = None):
        self.state_manager = state_manager or StateContinuityManager()
        self.visualizers: Dict[str, Any] = {}
        self.panel_systems: Dict[str, Any] = {}
        self.integration_lock = threading.RLock()
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        # Initialize integrations
        self._initialize_integrations()
        logger.info("VisualizerIntegration initialized")

    def _initialize_integrations(self) -> None:
        """Initialize all visualizer and panel system integrations."""
        try:
            # Initialize SpeedLatticeLivePanelSystem
            if SpeedLatticeLivePanelSystem:
                self.panel_systems["speed_lattice"] = SpeedLatticeLivePanelSystem()
                self._register_panel_callbacks("speed_lattice")
                logger.info("SpeedLatticeLivePanelSystem integrated")

            # Initialize MathLibV3Visualizer
            if MathLibV3Visualizer:
                self.visualizers["mathlib_v3"] = MathLibV3Visualizer(mode="live")
                self._register_visualizer_callbacks("mathlib_v3")
                logger.info("MathLibV3Visualizer integrated")

        except Exception as e:
            logger.error(f"Error initializing integrations: {e}")

    def _register_panel_callbacks(self, panel_name: str) -> None:
        """Register callbacks for panel system."""
        if panel_name == "speed_lattice" and self.panel_systems.get(panel_name):
            panel_system = self.panel_systems[panel_name]

            # Register state update callback
            def panel_callback(snapshot: StateSnapshot) -> None:
                try:
                    self._update_panel_data(panel_system, snapshot)
                except Exception as e:
                    logger.warning(f"Panel callback error: {e}")

            self.state_manager.register_panel_connection(panel_name, panel_callback)

    def _register_visualizer_callbacks(self, viz_name: str) -> None:
        """Register callbacks for visualizer."""
        if viz_name == "mathlib_v3" and self.visualizers.get(viz_name):
            visualizer = self.visualizers[viz_name]

            # Register state update callback
            def viz_callback(snapshot: StateSnapshot) -> None:
                try:
                    self._update_visualizer_data(visualizer, snapshot)
                except Exception as e:
                    logger.warning(f"Visualizer callback error: {e}")

            self.state_manager.register_visualizer_connection(viz_name, viz_callback)

    def _update_panel_data(self, panel_system: Any, snapshot: StateSnapshot) -> None:
        """Update panel system with new state data."""
        try:
            # Map state types to panel types
            panel_mapping = {}
                StateType.TRADING_STATE: PanelType.TRADING_STATE if PanelType else None,
                    StateType.SYSTEM_STATE: PanelType.SYSTEM_STATUS if PanelType else None,
                        StateType.MATHEMATICAL_STATE: PanelType.DRIFT_MATRIX if PanelType else None,
                        StateType.VISUALIZATION_STATE: PanelType.PATTERN_RECOGNITION if PanelType else None,
                        StateType.PANEL_STATE: PanelType.POOL_ANALYSIS if PanelType else None,
}
            panel_type = panel_mapping.get(snapshot.state_type)
            if panel_type and hasattr(panel_system, "panels"):
                # Update panel data
                panel_data = {
                    "data": snapshot.data,
                    "timestamp": snapshot.timestamp.isoformat(),
                    "agent": snapshot.agent,
                    "phase": snapshot.phase,
                    "metadata": snapshot.metadata or {},
}
}
                if hasattr(panel_system.panels, "get"):
                    panel = panel_system.panels.get(panel_type)
                    if panel and hasattr(panel, "update_data"):
                        panel.update_data(panel_data)

        except Exception as e:
            logger.error(f"Error updating panel data: {e}")

    def _update_visualizer_data(self, visualizer: Any, snapshot: StateSnapshot) -> None:
        """Update visualizer with new state data."""
        try:
            # Map state types to visualizer panels
            viz_mapping = {}
                StateType.TRADING_STATE: "trading_state",
                    StateType.MATHEMATICAL_STATE: "dual_operations",
                        StateType.SYSTEM_STATE: "performance",
                        StateType.VISUALIZATION_STATE: "pattern_detection",
}
            panel_name = viz_mapping.get(snapshot.state_type)
            if panel_name and hasattr(visualizer, "panels"):
                # Update visualizer panel data
                viz_data = {
                    "data": snapshot.data,
                    "timestamp": snapshot.timestamp.isoformat(),
                    "agent": snapshot.agent,
                    "phase": snapshot.phase,
}
}
                if hasattr(visualizer.panels, "get"):
                    panel = visualizer.panels.get(panel_name)
                    if panel and hasattr(panel, "update_data"):
                        panel.update_data(viz_data)

        except Exception as e:
            logger.error(f"Error updating visualizer data: {e}")

    def _update_loop(self) -> None:
        """Background loop for periodic updates."""
        while True:
            try:
                time.sleep(5)  # Update every 5 seconds
                self._periodic_update()
            except Exception as e:
                logger.error(f"Update loop error: {e}")

    def _periodic_update(self) -> None:
        """Perform periodic updates to all visualizers and panels."""
        try:
            # Update panel systems
            for name, panel_system in self.panel_systems.items():
                if hasattr(panel_system, "_update_main_content"):
                    panel_system._update_main_content()

            # Update visualizers
            for name, visualizer in self.visualizers.items():
                if hasattr(visualizer, "update_visualization"):
                    visualizer.update_visualization()

        except Exception as e:
            logger.error(f"Periodic update error: {e}")

    def get_panel_system(self, name: str) -> Optional[Any]:
        """Get a panel system by name."""
        return self.panel_systems.get(name)

    def get_visualizer(self, name: str) -> Optional[Any]:
        """Get a visualizer by name."""
        return self.visualizers.get(name)

    def update_state()
        self,
            state_type: StateType,
                data: Dict[str, Any],
                agent: Optional[str] = None,
                phase: Optional[int] = None,
                metadata: Optional[Dict[str, Any]] = None,
                ) -> str:
        """Update state and notify all visualizers and panels."""
        if self.state_manager:
            return self.state_manager.update_state(state_type, data, agent, phase, metadata)
        else:
            logger.error("State manager not available")
            return ""

    def get_visualization_data(self, state_type: StateType) -> Dict[str, Any]:
        """Get formatted visualization data."""
        if self.state_manager:
            return self.state_manager.get_visualization_data(state_type)
        else:
            return {"error": "State manager not available"}

    def get_panel_data(self, panel_name: str) -> Dict[str, Any]:
        """Get formatted panel data."""
        if self.state_manager:
            return self.state_manager.get_panel_data(panel_name)
        else:
            return {"error": "State manager not available"}

    def save_state_to_file(self, state_key: str, filename: Optional[str] = None) -> str:
        """Save state to file with JSON hang-up prevention."""
        if self.state_manager:
            return self.state_manager.save_state_to_file(state_key, filename)
        else:
            raise RuntimeError("State manager not available")

    def load_state_from_file(self, filename: str) -> Optional[StateSnapshot]:
        """Load state from file with JSON hang-up prevention."""
        if self.state_manager:
            return self.state_manager.load_state_from_file(filename)
        else:
            logger.error("State manager not available")
            return None

    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status report."""
        return {}
            "state_manager_available": self.state_manager is not None,
                "panel_systems": list(self.panel_systems.keys()),
                    "visualizers": list(self.visualizers.keys()),
                    "timestamp": datetime.now().isoformat(),
}
    def start_live_systems(self) -> None:
        """Start all live visualizer and panel systems."""
        try:
            # Start panel systems
            for name, panel_system in self.panel_systems.items():
                if hasattr(panel_system, "start_live_system"):
                    panel_system.start_live_system()
                    logger.info(f"Started live panel system: {name}")

            # Start visualizers
            for name, visualizer in self.visualizers.items():
                if hasattr(visualizer, "start_live_mode"):
                    visualizer.start_live_mode()
                    logger.info(f"Started live visualizer: {name}")

        except Exception as e:
            logger.error(f"Error starting live systems: {e}")

    def stop_live_systems(self) -> None:
        """Stop all live visualizer and panel systems."""
        try:
            # Stop panel systems
            for name, panel_system in self.panel_systems.items():
                if hasattr(panel_system, "stop_live_system"):
                    panel_system.stop_live_system()
                    logger.info(f"Stopped live panel system: {name}")

            # Stop visualizers
            for name, visualizer in self.visualizers.items():
                if hasattr(visualizer, "stop_live_mode"):
                    visualizer.stop_live_mode()
                    logger.info(f"Stopped live visualizer: {name}")

        except Exception as e:
            logger.error(f"Error stopping live systems: {e}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create integration
    integration = VisualizerIntegration()

    # Test state updates
    test_data = {"price": 50000, "volume": 1000, "timestamp": time.time(), "indicators": {"rsi": 65.5, "macd": 0.2}}

    if StateType:
        state_key = integration.update_state()
            StateType.TRADING_STATE, test_data, agent="BTC", phase=32, metadata={"source": "test"}
        )

        print(f"Created state: {state_key}")
        print(f"Integration status: {integration.get_integration_status()}")

        # Test visualization data
        viz_data = integration.get_visualization_data(StateType.TRADING_STATE)
        print(f"Visualization data: {viz_data}")

        # Test panel data
        panel_data = integration.get_panel_data("trading_panel")
        print(f"Panel data: {panel_data}")
    else:
        print("StateType not available - skipping tests")
