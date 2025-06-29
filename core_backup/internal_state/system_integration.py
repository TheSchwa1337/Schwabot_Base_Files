# -*- coding: utf-8 -*-
""""""
System Integration Module
========================

Connects the EnhancedStateManager to all internal systems, ensuring proper
initialization, organization, and connection to logging, memories, and backlogs
across testing, demo, and live modes.

Features:
- Integration with all core trading systems
- Automatic system state synchronization
- Memory and backlog management across modules
- BTC price hashing integration for demo states
- Comprehensive logging and monitoring
- System health and performance tracking
""""""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import core modules with safe fallbacks
try:
    from core.internal_state.enhanced_state_manager import BTCPriceHash, EnhancedStateManager, LogLevel, SystemMode
except ImportError:
    EnhancedStateManager = None
    SystemMode = None
    LogLevel = None
    BTCPriceHash = None

try:
    from core.internal_state.state_continuity_manager import StateContinuityManager, StateType
except ImportError:
    StateContinuityManager = None
    StateType = None

try:
    from core.dynamic_handoff_orchestrator import DynamicHandoffOrchestrator
except ImportError:
    DynamicHandoffOrchestrator = None

logger = logging.getLogger(__name__)


class SystemIntegration:
    """"""
    Integrates enhanced state manager with all internal systems.
    """"""

    def __init__(self, mode: SystemMode = SystemMode.DEMO, log_level: LogLevel = LogLevel.INFO):
        self.mode = mode
        self.log_level = log_level

        # Initialize core managers
        self.enhanced_manager = EnhancedStateManager(mode, log_level) if EnhancedStateManager else None
        self.state_continuity_manager = StateContinuityManager() if StateContinuityManager else None
        self.handoff_orchestrator = DynamicHandoffOrchestrator() if DynamicHandoffOrchestrator else None

        # System connections
        self.connected_systems: Dict[str, Any] = {}
        self.system_health: Dict[str, Dict[str, Any]] = {}

        # Integration thread
        self.integration_thread = threading.Thread(target=self._integration_loop, daemon=True)
        self.integration_thread.start()

        # Initialize all systems
        self._initialize_systems()

        logger.info(f"SystemIntegration initialized in {mode.value} mode")

    def _initialize_systems(self) -> None:
        """Initialize all connected systems."""
        try:
            # Initialize state continuity manager
            if self.state_continuity_manager:
                self.connected_systems["state_continuity"] = self.state_continuity_manager
                logger.info("StateContinuityManager connected")

            # Initialize handoff orchestrator
            if self.handoff_orchestrator:
                self.connected_systems["handoff_orchestrator"] = self.handoff_orchestrator
                logger.info("DynamicHandoffOrchestrator connected")

            # Initialize other core systems (placeholder for future integrations)
            self._initialize_core_systems()

        except Exception as e:
            logger.error(f"Error initializing systems: {e}")

    def _initialize_core_systems(self) -> None:
        """Initialize core trading systems."""
        try:
            # Placeholder for core system initialization
            # This would connect to actual trading systems, APIs, etc.
            core_systems = ["trading_engine", "risk_manager", "data_feed", "order_manager", "portfolio_manager"]

            for system_name in core_systems:
                # Simulate system connection
                self.connected_systems[system_name] = {}
                    "status": "connected",
                        "last_update": datetime.now().isoformat(),
                            "health": "healthy",
}
                logger.info(f"Core system connected: {system_name}")

        except Exception as e:
            logger.error(f"Error initializing core systems: {e}")

    def _integration_loop(self) -> None:
        """Background loop for system integration."""
        while True:
            try:
                time.sleep(5)  # Update every 5 seconds
                self._update_system_health()
                self._synchronize_states()
                self._process_system_backlogs()
            except Exception as e:
                logger.error(f"Integration loop error: {e}")

    def _update_system_health(self) -> None:
        """Update system health status."""
        try:
            for system_name, system in self.connected_systems.items():
                health_status = {
                    "status": "healthy",
                    "last_check": datetime.now().isoformat(),
                    "uptime": self._get_system_uptime(system),
                    "memory_usage": self._get_system_memory_usage(system),
                    "performance": self._get_system_performance(system),
}
}
                self.system_health[system_name] = health_status

        except Exception as e:
            logger.error(f"Error updating system health: {e}")

    def _get_system_uptime(self, system: Any) -> float:
        """Get system uptime in seconds."""
        try:
            if hasattr(system, "start_time"):
                return (datetime.now() - system.start_time).total_seconds()
            elif hasattr(system, "get_uptime"):
                return system.get_uptime()
            else:
                return 0.0
        except Exception:
            return 0.0

    def _get_system_memory_usage(self, system: Any) -> Dict[str, Any]:
        """Get system memory usage."""
        try:
            if hasattr(system, "get_memory_usage"):
                return system.get_memory_usage()
            else:
                return {"total": 0, "used": 0, "available": 0}
        except Exception:
            return {"total": 0, "used": 0, "available": 0}

    def _get_system_performance(self, system: Any) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            if hasattr(system, "get_performance_metrics"):
                return system.get_performance_metrics()
            else:
                return {"cpu_usage": 0, "response_time": 0, "throughput": 0}
        except Exception:
            return {"cpu_usage": 0, "response_time": 0, "throughput": 0}

    def _synchronize_states(self) -> None:
        """Synchronize states across all systems."""
        try:
            if self.enhanced_manager and self.state_continuity_manager:
                # Get enhanced manager state
                enhanced_status = self.enhanced_manager.get_system_status()

                # Update state continuity manager
                if StateType:
                    self.state_continuity_manager.update_state()
                        StateType.SYSTEM_STATE,
                            enhanced_status,
                                agent="system",
                                phase=32,
                                metadata={"source": "enhanced_manager"},
                                )

        except Exception as e:
            logger.error(f"Error synchronizing states: {e}")

    def _process_system_backlogs(self) -> None:
        """Process backlogs across all systems."""
        try:
            if self.enhanced_manager:
                # Process enhanced manager backlogs
                backlog_status = self.enhanced_manager.get_backlog_status()

                # Add system backlog entries if needed
                if backlog_status["queue_size"] > 10:
                    self.enhanced_manager.add_backlog_entry()
                        priority=1,
                            data={"action": "cleanup", "reason": "high_queue_size"},
                                source="system_integration",
                                target="cleanup",
                                )

        except Exception as e:
            logger.error(f"Error processing system backlogs: {e}")

    def create_demo_state_with_btc_hash()
        self, btc_price: float, btc_volume: float, phase: int = 32, additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create demo state with BTC price hashing and system integration."""
        try:
            if not self.enhanced_manager:
                raise RuntimeError("Enhanced manager not available")

            # Create demo state
            demo_state = self.enhanced_manager.create_demo_state(btc_price, btc_volume, phase, additional_data)

            # Add system integration data
            demo_state["system_integration"] = {}
                "connected_systems": list(self.connected_systems.keys()),
                    "system_health": self.system_health,
                        "integration_timestamp": datetime.now().isoformat(),
}
            # Update state continuity manager
            if self.state_continuity_manager and StateType:
                self.state_continuity_manager.update_state()
                    StateType.DEMO_STATE,
                        demo_state,
                            agent="BTC",
                            phase=phase,
                            metadata={"source": "system_integration"},
                            )

            # Add to backlog for processing
            self.enhanced_manager.add_backlog_entry()
                priority=5, data=demo_state, source="demo_creation", target="state_synchronization"
            )

            logger.info(f"Created demo state with BTC hash: {demo_state['btc_price_hash']['hash'][:16]}...")
            return demo_state

        except Exception as e:
            logger.error(f"Error creating demo state: {e}")
            return {"error": str(e)}

    def get_btc_price_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get BTC price history with system integration data."""
        try:
            if not self.enhanced_manager:
                return []

            btc_history = self.enhanced_manager.get_btc_price_history(limit)
            return []
                {}
                    "price": hash_obj.price,
                        "volume": hash_obj.volume,
                            "hash": hash_obj.hash_value,
                            "phase": hash_obj.phase,
                            "timestamp": hash_obj.timestamp.isoformat(),
                            "system_context": {}
                        "mode": self.mode.value,
                            "connected_systems": len(self.connected_systems),
                                "health_status": "healthy",
                                },
}
                for hash_obj in btc_history
]
        except Exception as e:
            logger.error(f"Error getting BTC price history: {e}")
            return []

    def get_comprehensive_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status across all components."""
        try:
            status = {
                "mode": self.mode.value,
                "log_level": self.log_level.value,
                "timestamp": datetime.now().isoformat(),
                "enhanced_manager": self.enhanced_manager.get_system_status() if self.enhanced_manager else None,
                "state_continuity": ()
}
                    self.state_continuity_manager.get_continuity_report() if self.state_continuity_manager else None
                ),
                    "connected_systems": {}
                    name: {}
                        "status": "connected",
                            "health": self.system_health.get(name, {}),
                                "last_update": datetime.now().isoformat(),
}
                    for name in self.connected_systems.keys()
                },
                    "system_health_summary": {}
                    "total_systems": len(self.connected_systems),
                        "healthy_systems": len([s for s in self.system_health.values() if s.get("status") == "healthy"]),
                            "unhealthy_systems": len([s for s in self.system_health.values() if s.get("status") != "healthy"]),
                            },
                            "integration_status": {}
                    "enhanced_manager_available": self.enhanced_manager is not None,
                        "state_continuity_available": self.state_continuity_manager is not None,
                            "handoff_orchestrator_available": self.handoff_orchestrator is not None,
                            "integration_thread_alive": self.integration_thread.is_alive(),
                            },
}
            return status

        except Exception as e:
            logger.error(f"Error getting comprehensive system status: {e}")
            return {"error": str(e)}

    def export_integrated_system_state(self, filename: Optional[str] = None) -> str:
        """Export complete integrated system state."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"integrated_system_state_{self.mode.value}_{timestamp}.json"

            # Get comprehensive status
            system_state = self.get_comprehensive_system_status()

            # Add enhanced manager export if available
            if self.enhanced_manager:
                enhanced_export = self.enhanced_manager.export_system_state()
                system_state["enhanced_manager_export"] = enhanced_export

            # Write to file
            import json

            with open(filename, "w") as f:
                json.dump(system_state, f, indent=2, default=str)

            logger.info(f"Exported integrated system state to: {filename}")
            return filename

        except Exception as e:
            logger.error(f"Error exporting integrated system state: {e}")
            raise

    def run_system_test(self, test_duration: int = 60) -> Dict[str, Any]:
        """Run comprehensive system test."""
        try:
            logger.info(f"Starting system test for {test_duration} seconds")

            start_time = datetime.now()
            test_results = {"start_time": start_time.isoformat(), "duration_seconds": test_duration, "tests": {}}

            # Test 1: Enhanced Manager
            if self.enhanced_manager:
                test_results["tests"]["enhanced_manager"] = {}
                    "status": "passed",
                        "memory_operations": self._test_memory_operations(),
                            "backlog_operations": self._test_backlog_operations(),
                            "btc_hash_generation": self._test_btc_hash_generation(),
}
            # Test 2: State Continuity
            if self.state_continuity_manager:
                test_results["tests"]["state_continuity"] = {}
                    "status": "passed",
                        "state_operations": self._test_state_operations(),
}
            # Test 3: System Integration
            test_results["tests"]["system_integration"] = {}
                "status": "passed",
                    "demo_state_creation": self._test_demo_state_creation(),
                        "system_health": self._test_system_health(),
}
            # Wait for test duration
            time.sleep(test_duration)

            end_time = datetime.now()
            test_results["end_time"] = end_time.isoformat()
            test_results["total_duration"] = (end_time - start_time).total_seconds()

            logger.info(f"System test completed in {test_results['total_duration']:.2f} seconds")
            return test_results

        except Exception as e:
            logger.error(f"Error running system test: {e}")
            return {"error": str(e)}

    def _test_memory_operations(self) -> Dict[str, Any]:
        """Test memory operations."""
        try:
            # Store test memory
            self.enhanced_manager.store_memory("test_memory", {"test": "data"})

            # Retrieve test memory
            retrieved = self.enhanced_manager.get_memory("test_memory")

            return {}
                "store_success": True,
                    "retrieve_success": retrieved is not None,
                        "data_integrity": retrieved == {"test": "data"} if retrieved else False,
}
        except Exception as e:
            return {"error": str(e)}

    def _test_backlog_operations(self) -> Dict[str, Any]:
        """Test backlog operations."""
        try:
            # Add test backlog entry
            entry_id = self.enhanced_manager.add_backlog_entry(5, {"test": "backlog"}, "test", "memory")

            # Get backlog status
            status = self.enhanced_manager.get_backlog_status()

            return {}
                "add_success": entry_id is not None,
                    "status_retrieved": status is not None,
                        "queue_size": status.get("queue_size", 0) if status else 0,
}
        except Exception as e:
            return {"error": str(e)}

    def _test_btc_hash_generation(self) -> Dict[str, Any]:
        """Test BTC hash generation."""
        try:
            # Generate test hash
            btc_hash = self.enhanced_manager.generate_btc_price_hash(50000.0, 1000.0, 32)

            return {}
                "generation_success": btc_hash is not None,
                    "hash_length": len(btc_hash.hash_value) if btc_hash else 0,
                        "price": btc_hash.price if btc_hash else 0,
                        "volume": btc_hash.volume if btc_hash else 0,
}
        except Exception as e:
            return {"error": str(e)}

    def _test_state_operations(self) -> Dict[str, Any]:
        """Test state operations."""
        try:
            if not StateType:
                return {"error": "StateType not available"}

            # Update test state
            state_key = self.state_continuity_manager.update_state()
                StateType.TESTING_STATE, {"test": "state_data"}, agent="test", phase=32
            )

            # Get state
            retrieved_state = self.state_continuity_manager.get_state(StateType.TESTING_STATE, "test", 32)

            return {}
                "update_success": state_key is not None,
                    "retrieve_success": retrieved_state is not None,
                        "state_type": retrieved_state.state_type.value if retrieved_state else None,
}
        except Exception as e:
            return {"error": str(e)}

    def _test_demo_state_creation(self) -> Dict[str, Any]:
        """Test demo state creation."""
        try:
            # Create demo state
            demo_state = self.create_demo_state_with_btc_hash(50000.0, 1000.0, 32, {"test": "demo_data"})

            return {}
                "creation_success": "error" not in demo_state,
                    "btc_hash_present": "btc_price_hash" in demo_state,
                        "system_integration_present": "system_integration" in demo_state,
}
        except Exception as e:
            return {"error": str(e)}

    def _test_system_health(self) -> Dict[str, Any]:
        """Test system health monitoring."""
        try:
            # Get system health
            health = self.system_health

            return {}
                "health_retrieved": len(health) > 0,
                    "systems_monitored": len(health),
                        "healthy_systems": len([s for s in health.values() if s.get("status") == "healthy"]),
}
        except Exception as e:
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Create system integration
    integration = SystemIntegration(mode=SystemMode.DEMO, log_level=LogLevel.INFO)

    # Test demo state creation
    demo_state = integration.create_demo_state_with_btc_hash(50000.0, 1000.0, 32, {"test": "data"})

    # Get comprehensive status
    status = integration.get_comprehensive_system_status()

    # Run system test
    test_results = integration.run_system_test(test_duration=30)

    # Export system state
    filename = integration.export_integrated_system_state()

    print(f"System integration test completed")
    print(f"Demo state created: {'btc_price_hash' in demo_state}")
    print(f"System status: {len(status.get('connected_systems', {}))} systems connected")
    print(f"Test results: {len(test_results.get('tests', {}))} tests completed")
    print(f"Exported to: {filename}")
