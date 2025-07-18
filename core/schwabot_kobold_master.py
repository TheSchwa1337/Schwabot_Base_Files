#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Schwabot AI Master Integration
====================================

Master integration script that ties together all Schwabot AI components
and allows the visual layer to take over the Schwabot trading system.

Features:
- Complete system integration and orchestration
- Visual layer takeover of Schwabot AI system
- Real-time AI-powered trading analysis and visualization
- Hardware-optimized performance and resource management
- Comprehensive monitoring and control interface
- Integration with existing 47-day mathematical framework
"""

import asyncio
import json
import logging
import time
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import threading

from .hardware_auto_detector import HardwareAutoDetector
from .schwabot_ai_integration import Schwabot AIIntegration, AnalysisType
from .visual_layer_controller import VisualLayerController, VisualizationType, ChartTimeframe
from .tick_loader import TickLoader, TickPriority
from .signal_cache import SignalCache, SignalType, SignalPriority
from .registry_writer import RegistryWriter, ArchivePriority
from .json_server import JSONServer, PacketPriority
from .alpha256_encryption import Alpha256Encryption

logger = logging.getLogger(__name__)

class SchwabotSchwabotMaster:
    """Master integration controller for Schwabot Schwabot AI system."""
    
    def __init__(self, config_path: str = "config/schwabot_master_config.json"):
        """Initialize master integration controller."""
        self.config_path = Path(config_path)
        
        # Core system components
        self.hardware_detector = HardwareAutoDetector()
        self.alpha256 = Alpha256Encryption()
        
        # Schwabot AI integration components
        self.schwabot_integration = None
        self.visual_controller = None
        self.tick_loader = None
        self.signal_cache = None
        self.registry_writer = None
        self.json_server = None
        
        # System state
        self.running = False
        self.initialized = False
        self.system_info = None
        self.memory_config = None
        
        # Performance tracking
        self.stats = {
            "start_time": 0.0,
            "uptime_seconds": 0.0,
            "total_analyses": 0,
            "total_visualizations": 0,
            "total_signals": 0,
            "system_health": "unknown"
        }
        
        # Control interface
        self.control_thread = None
        self.shutdown_event = threading.Event()
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the complete Schwabot Schwabot AI system."""
        try:
            logger.info("🚀 Initializing Schwabot Schwabot AI Master System...")
            
            # Detect hardware capabilities
            self.system_info = self.hardware_detector.detect_hardware()
            self.memory_config = self.hardware_detector.generate_memory_config()
            
            logger.info(f"✅ Hardware detected: {self.system_info.platform}")
            logger.info(f"   RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
            logger.info(f"   Optimization: {self.system_info.optimization_mode.value}")
            
            # Load or create master configuration
            self._load_master_configuration()
            
            # Initialize core components
            self._initialize_core_components()
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.initialized = True
            logger.info("✅ Schwabot Schwabot AI Master System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def _load_master_configuration(self):
        """Load or create master configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("✅ Loaded existing master configuration")
            else:
                self.config = self._create_default_master_config()
                self._save_master_configuration()
                logger.info("✅ Created new master configuration")
                
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            self.config = self._create_default_master_config()
    
    def _create_default_master_config(self) -> Dict[str, Any]:
        """Create default master configuration."""
        return {
            "version": "1.0.0",
            "system_name": "Schwabot Schwabot AI Master",
            "hardware_auto_detected": True,
            "system_info": {
                "platform": self.system_info.platform if self.system_info else "unknown",
                "ram_gb": self.system_info.ram_gb if self.system_info else 8.0,
                "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "balanced"
            },
            "schwabot_integration": {
                "enabled": True,
                "schwabot_path": "schwabot_ai",
                "model_path": "",
                "port": 5001,
                "auto_start": True,
                "auto_load_model": True
            },
            "visual_layer": {
                "enabled": True,
                "output_dir": "visualizations",
                "enable_ai_analysis": True,
                "enable_pattern_recognition": True,
                "enable_real_time_rendering": True
            },
            "tick_loader": {
                "enabled": True,
                "max_queue_size": 10000,
                "enable_compression": True,
                "enable_encryption": True
            },
            "signal_cache": {
                "enabled": True,
                "cache_size": 10000,
                "enable_similarity_matching": True,
                "enable_confidence_scoring": True
            },
            "registry_writer": {
                "enabled": True,
                "base_path": "data/registry",
                "enable_compression": True,
                "enable_encryption": True,
                "enable_backup_rotation": True
            },
            "json_server": {
                "enabled": True,
                "host": "localhost",
                "port": 8080,
                "max_connections": 100,
                "enable_encryption": True
            },
            "system_control": {
                "enable_health_monitoring": True,
                "health_check_interval_seconds": 30,
                "enable_performance_tracking": True,
                "enable_auto_restart": True,
                "max_restart_attempts": 3
            }
        }
    
    def _save_master_configuration(self):
        """Save master configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save master configuration: {e}")
    
    def _initialize_core_components(self):
        """Initialize all core system components."""
        try:
            logger.info("🔧 Initializing core components...")
            
            # Initialize Schwabot AI integration
            if self.config["schwabot_integration"]["enabled"]:
                self.schwabot_integration = Schwabot AIIntegration(
                    schwabot_path=self.config["schwabot_integration"]["schwabot_path"],
                    model_path=self.config["schwabot_integration"]["model_path"],
                    port=self.config["schwabot_integration"]["port"]
                )
                logger.info("✅ Schwabot AI integration initialized")
            
            # Initialize visual layer controller
            if self.config["visual_layer"]["enabled"]:
                self.visual_controller = VisualLayerController(
                    output_dir=self.config["visual_layer"]["output_dir"]
                )
                logger.info("✅ Visual layer controller initialized")
            
            # Initialize tick loader
            if self.config["tick_loader"]["enabled"]:
                self.tick_loader = TickLoader()
                logger.info("✅ Tick loader initialized")
            
            # Initialize signal cache
            if self.config["signal_cache"]["enabled"]:
                self.signal_cache = SignalCache()
                logger.info("✅ Signal cache initialized")
            
            # Initialize registry writer
            if self.config["registry_writer"]["enabled"]:
                self.registry_writer = RegistryWriter(
                    base_path=self.config["registry_writer"]["base_path"]
                )
                logger.info("✅ Registry writer initialized")
            
            # Initialize JSON server
            if self.config["json_server"]["enabled"]:
                self.json_server = JSONServer(
                    host=self.config["json_server"]["host"],
                    port=self.config["json_server"]["port"],
                    max_connections=self.config["json_server"]["max_connections"]
                )
                logger.info("✅ JSON server initialized")
            
            logger.info("✅ All core components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Core component initialization failed: {e}")
            raise
    
    async def start_system(self):
        """Start the complete Schwabot Schwabot AI system."""
        try:
            if self.running:
                logger.warning("⚠️ System already running")
                return
            
            logger.info("🚀 Starting Schwabot Schwabot AI Master System...")
            self.running = True
            self.stats["start_time"] = time.time()
            
            # Start all core components
            await self._start_core_components()
            
            # Start control interface
            self._start_control_interface()
            
            # Start health monitoring
            if self.config["system_control"]["enable_health_monitoring"]:
                self._start_health_monitoring()
            
            logger.info("✅ Schwabot Schwabot AI Master System started successfully")
            
            # Main system loop
            await self._main_system_loop()
            
        except Exception as e:
            logger.error(f"❌ System startup failed: {e}")
            await self.stop_system()
            raise
    
    async def _start_core_components(self):
        """Start all core system components."""
        try:
            # Start Schwabot AI integration
            if self.schwabot_integration:
                await self.schwabot_integration.start_processing()
                logger.info("✅ Schwabot AI integration started")
            
            # Start visual layer controller
            if self.visual_controller:
                await self.visual_controller.start_processing()
                logger.info("✅ Visual layer controller started")
            
            # Start tick loader
            if self.tick_loader:
                await self.tick_loader.start_processing()
                logger.info("✅ Tick loader started")
            
            # Start signal cache
            if self.signal_cache:
                await self.signal_cache.start_processing()
                logger.info("✅ Signal cache started")
            
            # Start registry writer
            if self.registry_writer:
                await self.registry_writer.start_writing()
                logger.info("✅ Registry writer started")
            
            # Start JSON server
            if self.json_server:
                # Start JSON server in background
                asyncio.create_task(self.json_server.start_server())
                logger.info("✅ JSON server started")
            
        except Exception as e:
            logger.error(f"❌ Core component startup failed: {e}")
            raise
    
    def _start_control_interface(self):
        """Start control interface for system management."""
        try:
            self.control_thread = threading.Thread(target=self._control_interface_loop, daemon=True)
            self.control_thread.start()
            logger.info("✅ Control interface started")
            
        except Exception as e:
            logger.error(f"❌ Control interface startup failed: {e}")
    
    def _start_health_monitoring(self):
        """Start health monitoring system."""
        try:
            health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
            health_thread.start()
            logger.info("✅ Health monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Health monitoring startup failed: {e}")
    
    async def _main_system_loop(self):
        """Main system processing loop."""
        try:
            while self.running and not self.shutdown_event.is_set():
                # Update uptime
                self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
                
                # Process system tasks
                await self._process_system_tasks()
                
                # Sleep to prevent excessive CPU usage
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"❌ Main system loop error: {e}")
    
    async def _process_system_tasks(self):
        """Process system tasks and coordinate components."""
        try:
            # Process tick data if available
            if self.tick_loader and self.tick_loader.running:
                await self._process_tick_data()
            
            # Process AI analyses
            if self.schwabot_integration and self.schwabot_integration.running:
                await self._process_ai_analyses()
            
            # Process visualizations
            if self.visual_controller and self.visual_controller.running:
                await self._process_visualizations()
            
            # Update system statistics
            self._update_system_statistics()
            
        except Exception as e:
            logger.error(f"❌ System task processing failed: {e}")
    
    async def _process_tick_data(self):
        """Process incoming tick data."""
        try:
            # Get processed ticks
            ticks = await self.tick_loader.process_ticks()
            
            for tick in ticks:
                # Cache tick data
                if self.signal_cache:
                    await self.signal_cache.cache_signal(
                        SignalType.PRICE,
                        tick.symbol,
                        {
                            "price": tick.price,
                            "volume": tick.volume,
                            "timestamp": tick.timestamp
                        },
                        SignalPriority.HIGH if tick.priority == TickPriority.HIGH else SignalPriority.MEDIUM
                    )
                
                # Archive tick data
                if self.registry_writer:
                    await self.registry_writer.write_entry(
                        "tick_data",
                        {
                            "symbol": tick.symbol,
                            "price": tick.price,
                            "volume": tick.volume,
                            "timestamp": tick.timestamp,
                            "priority": tick.priority.value
                        },
                        ArchivePriority.HIGH if tick.priority == TickPriority.HIGH else ArchivePriority.MEDIUM
                    )
                
                self.stats["total_signals"] += 1
            
        except Exception as e:
            logger.error(f"❌ Tick data processing failed: {e}")
    
    async def _process_ai_analyses(self):
        """Process AI analyses using Schwabot AI."""
        try:
            # Get cached signals for analysis
            if self.signal_cache:
                signals = await self.signal_cache.get_signals_by_type(SignalType.PRICE, limit=10)
                
                for signal in signals:
                    # Perform AI analysis
                    response = await self.schwabot_integration.process_trading_analysis(
                        signal.data,
                        AnalysisType.TECHNICAL_ANALYSIS
                    )
                    
                    if response:
                        self.stats["total_analyses"] += 1
                        
                        # Cache AI analysis result
                        await self.signal_cache.cache_signal(
                            SignalType.COMPOSITE,
                            signal.symbol,
                            {
                                "ai_analysis": response.text,
                                "confidence": response.confidence_score,
                                "results": response.analysis_results
                            },
                            SignalPriority.HIGH
                        )
            
        except Exception as e:
            logger.error(f"❌ AI analysis processing failed: {e}")
    
    async def _process_visualizations(self):
        """Process visualizations and AI-enhanced charts."""
        try:
            # Get recent signals for visualization
            if self.signal_cache:
                signals = await self.signal_cache.get_signals_by_type(SignalType.PRICE, limit=100)
                
                if len(signals) >= 20:
                    # Group signals by symbol
                    symbol_data = {}
                    for signal in signals:
                        if signal.symbol not in symbol_data:
                            symbol_data[signal.symbol] = []
                        symbol_data[signal.symbol].append(signal.data)
                    
                    # Generate visualizations for each symbol
                    for symbol, data in symbol_data.items():
                        if len(data) >= 20:
                            # Generate price chart
                            visual_analysis = await self.visual_controller.generate_price_chart(
                                data, symbol, ChartTimeframe.MINUTE_5
                            )
                            
                            if visual_analysis:
                                # Perform AI analysis on chart
                                visual_analysis = await self.visual_controller.perform_ai_analysis(visual_analysis)
                                
                                # Save visualization
                                await self.visual_controller.save_visualization(visual_analysis)
                                
                                self.stats["total_visualizations"] += 1
            
        except Exception as e:
            logger.error(f"❌ Visualization processing failed: {e}")
    
    def _update_system_statistics(self):
        """Update system statistics."""
        try:
            # Update uptime
            self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
            
            # Update system health
            self.stats["system_health"] = self._assess_system_health()
            
        except Exception as e:
            logger.error(f"❌ Statistics update failed: {e}")
    
    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        try:
            health_checks = []
            
            # Check Schwabot AI integration
            if self.schwabot_integration:
                health_checks.append(self.schwabot_integration.schwabot_running)
            
            # Check visual controller
            if self.visual_controller:
                health_checks.append(self.visual_controller.running)
            
            # Check tick loader
            if self.tick_loader:
                health_checks.append(self.tick_loader.running)
            
            # Check signal cache
            if self.signal_cache:
                health_checks.append(self.signal_cache.running)
            
            # Check registry writer
            if self.registry_writer:
                health_checks.append(self.registry_writer.running)
            
            # Check JSON server
            if self.json_server:
                health_checks.append(self.json_server.running)
            
            if all(health_checks):
                return "healthy"
            elif any(health_checks):
                return "degraded"
            else:
                return "unhealthy"
                
        except Exception as e:
            logger.error(f"❌ Health assessment failed: {e}")
            return "unknown"
    
    def _control_interface_loop(self):
        """Control interface loop for system management."""
        try:
            while self.running and not self.shutdown_event.is_set():
                # Process control commands
                # This could be expanded to handle user input, API commands, etc.
                
                time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"❌ Control interface loop error: {e}")
    
    def _health_monitoring_loop(self):
        """Health monitoring loop."""
        try:
            interval = self.config["system_control"]["health_check_interval_seconds"]
            
            while self.running and not self.shutdown_event.is_set():
                # Perform health checks
                health = self._assess_system_health()
                
                if health == "unhealthy":
                    logger.warning("⚠️ System health check failed - attempting recovery")
                    asyncio.run(self._attempt_system_recovery())
                
                time.sleep(interval)
                
        except Exception as e:
            logger.error(f"❌ Health monitoring loop error: {e}")
    
    async def _attempt_system_recovery(self):
        """Attempt to recover system components."""
        try:
            logger.info("🔄 Attempting system recovery...")
            
            # Restart failed components
            if self.schwabot_integration and not self.schwabot_integration.schwabot_running:
                await self.schwabot_integration.start_schwabot_server()
            
            if self.visual_controller and not self.visual_controller.running:
                await self.visual_controller.start_processing()
            
            logger.info("✅ System recovery completed")
            
        except Exception as e:
            logger.error(f"❌ System recovery failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    async def stop_system(self):
        """Stop the complete Schwabot Schwabot AI system."""
        try:
            if not self.running:
                return
            
            logger.info("🛑 Stopping Schwabot Schwabot AI Master System...")
            self.running = False
            self.shutdown_event.set()
            
            # Stop all core components
            await self._stop_core_components()
            
            logger.info("✅ Schwabot Schwabot AI Master System stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ System shutdown failed: {e}")
    
    async def _stop_core_components(self):
        """Stop all core system components."""
        try:
            # Stop Schwabot AI integration
            if self.schwabot_integration:
                self.schwabot_integration.stop_processing()
            
            # Stop visual layer controller
            if self.visual_controller:
                self.visual_controller.stop_processing()
            
            # Stop tick loader
            if self.tick_loader:
                self.tick_loader.stop_processing()
            
            # Stop signal cache
            if self.signal_cache:
                self.signal_cache.stop_processing()
            
            # Stop registry writer
            if self.registry_writer:
                self.registry_writer.stop_writing()
            
            # Stop JSON server
            if self.json_server:
                self.json_server.stop_server()
            
        except Exception as e:
            logger.error(f"❌ Core component shutdown failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        try:
            status = {
                "system_info": {
                    "name": self.config["system_name"],
                    "version": self.config["version"],
                    "running": self.running,
                    "initialized": self.initialized,
                    "uptime_seconds": self.stats["uptime_seconds"],
                    "system_health": self.stats["system_health"]
                },
                "statistics": self.stats,
                "components": {
                    "schwabot_integration": {
                        "enabled": bool(self.schwabot_integration),
                        "running": self.schwabot_integration.running if self.schwabot_integration else False,
                        "stats": self.schwabot_integration.get_statistics() if self.schwabot_integration else {}
                    },
                    "visual_controller": {
                        "enabled": bool(self.visual_controller),
                        "running": self.visual_controller.running if self.visual_controller else False,
                        "stats": self.visual_controller.get_statistics() if self.visual_controller else {}
                    },
                    "tick_loader": {
                        "enabled": bool(self.tick_loader),
                        "running": self.tick_loader.running if self.tick_loader else False,
                        "stats": self.tick_loader.get_statistics() if self.tick_loader else {}
                    },
                    "signal_cache": {
                        "enabled": bool(self.signal_cache),
                        "running": self.signal_cache.running if self.signal_cache else False,
                        "stats": self.signal_cache.get_statistics() if self.signal_cache else {}
                    },
                    "registry_writer": {
                        "enabled": bool(self.registry_writer),
                        "running": self.registry_writer.running if self.registry_writer else False,
                        "stats": self.registry_writer.get_statistics() if self.registry_writer else {}
                    },
                    "json_server": {
                        "enabled": bool(self.json_server),
                        "running": self.json_server.running if self.json_server else False,
                        "stats": self.json_server.get_statistics() if self.json_server else {}
                    }
                },
                "hardware_info": {
                    "platform": self.system_info.platform if self.system_info else "unknown",
                    "ram_gb": self.system_info.ram_gb if self.system_info else 0.0,
                    "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "unknown"
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Status collection failed: {e}")
            return {"error": str(e)}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    """Main function for Schwabot Schwabot AI Master System."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('schwabot_schwabot_master.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Schwabot Schwabot AI Master System...")
    
    # Create master system
    master_system = SchwabotSchwabotMaster()
    
    try:
        # Start the system
        await master_system.start_system()
        
    except KeyboardInterrupt:
        logger.info("📡 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
    finally:
        # Stop the system
        await master_system.stop_system()
        
        # Print final status
        status = master_system.get_system_status()
        logger.info(f"📊 Final system status: {json.dumps(status, indent=2)}")
        
        logger.info("👋 Schwabot Schwabot AI Master System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 