"""
Enhanced Visual Architecture
===========================

Advanced visualization system that integrates with the practical visual controller
to provide real-time representation of:

- Multi-bit mapping visualization (4-bit → 42-bit phaser)
- RAM → storage tier transitions with smooth animations
- Profit vector smoothing and tick integration
- High-frequency allocation management (10,000+ per hour)
- Dynamic drift visualization and compensation
- Millisecond-level error handling and sequencing
- Adaptive optimization with back-tested log integration

Features:
- Tesseract visualization integration
- Real-time WebSocket streaming
- Glass-morphism UI with thermal awareness
- Dynamic bit-level representation
- Profit orbital visualization
- Storage pipeline visualization
"""

import asyncio
import logging
import json
import time
import math
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
import numpy as np

# WebSocket and UI imports
import websockets
from websockets.server import WebSocketServerProtocol

# Integration imports
from core.practical_visual_controller import (
    PracticalVisualController,
    ControlMode,
    MappingBitLevel,
    VisualState
)
from core.orbital_profit_navigator import (
    OrbitalProfitNavigator,
    OrbitalZone,
    ProfitTier,
    BitMappingLevel
)

logger = logging.getLogger(__name__)

class VisualizationMode(Enum):
    """Visualization display modes"""
    OVERVIEW = "overview"
    BIT_MAPPING = "bit_mapping"
    PROFIT_ORBITAL = "profit_orbital"
    STORAGE_PIPELINE = "storage_pipeline"
    PERFORMANCE_DRIFT = "performance_drift"
    HIGH_FREQUENCY = "high_frequency"
    TESSERACT = "tesseract"

class RenderQuality(Enum):
    """Rendering quality levels for performance optimization"""
    LOW = "low"           # 30 FPS, basic animations
    MEDIUM = "medium"     # 60 FPS, smooth animations
    HIGH = "high"         # 120 FPS, ultra-smooth
    ADAPTIVE = "adaptive" # Dynamic based on system load

@dataclass
class BitVisualizationState:
    """State for bit mapping visualization"""
    current_level: BitMappingLevel = BitMappingLevel.BIT_16
    target_level: BitMappingLevel = BitMappingLevel.BIT_16
    transition_progress: float = 0.0
    processing_intensity: float = 0.5
    thermal_influence: float = 0.0
    
    # Bit representation visualization
    bit_particles: List[Dict[str, Any]] = field(default_factory=list)
    connection_strengths: Dict[str, float] = field(default_factory=dict)
    processing_waves: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ProfitVectorState:
    """State for profit vector smoothing"""
    profit_points: List[Tuple[float, float]] = field(default_factory=list)
    smoothing_factor: float = 0.8
    vector_velocity: Tuple[float, float] = (0.0, 0.0)
    orbital_positions: Dict[OrbitalZone, Tuple[float, float]] = field(default_factory=dict)
    
    # Tick integration
    tick_frequency: float = 60.0  # Hz
    last_tick: float = 0.0
    accumulated_drift: Tuple[float, float] = (0.0, 0.0)

@dataclass
class StoragePipelineVisualization:
    """Visualization state for storage pipeline"""
    ram_particles: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_flow: Dict[str, float] = field(default_factory=dict)
    compression_ratios: Dict[str, float] = field(default_factory=dict)
    
    # Data flow animation
    flow_speed: float = 1.0
    particle_density: int = 100
    thermal_color_mapping: Dict[str, str] = field(default_factory=dict)

class EnhancedVisualArchitecture:
    """
    Enhanced visual architecture providing real-time visualization
    of all system components with advanced rendering capabilities.
    """
    
    def __init__(self,
                 practical_controller: PracticalVisualController,
                 orbital_navigator: Optional[OrbitalProfitNavigator] = None,
                 websocket_host: str = "localhost",
                 websocket_port: int = 8765):
        """
        Initialize the enhanced visual architecture
        
        Args:
            practical_controller: The practical visual controller instance
            orbital_navigator: Orbital profit navigator for profit visualization
            websocket_host: WebSocket server host
            websocket_port: WebSocket server port
        """
        self.practical_controller = practical_controller
        self.orbital_navigator = orbital_navigator
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        
        # Visualization states
        self.bit_visualization = BitVisualizationState()
        self.profit_vector = ProfitVectorState()
        self.storage_pipeline = StoragePipelineVisualization()
        
        # Current visualization mode
        self.current_mode = VisualizationMode.OVERVIEW
        self.render_quality = RenderQuality.ADAPTIVE
        
        # Performance tracking for adaptive optimization
        self.performance_metrics = {
            "frame_rate": 60.0,
            "render_time_ms": 16.67,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "drift_compensation": 0.0,
            "error_rate": 0.0
        }
        
        # High-frequency management
        self.hf_allocations = []
        self.max_hf_display = 10000  # Maximum allocations to visualize
        self.drift_threshold = 0.1   # Drift compensation threshold
        
        # WebSocket connections
        self.connected_clients = set()
        self.websocket_server = None
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        # Adaptive optimization
        self.optimization_history = []
        self.last_optimization = time.time()
        
        logger.info("EnhancedVisualArchitecture initialized")
    
    async def start_visualization(self) -> bool:
        """Start the enhanced visualization system"""
        try:
            logger.info("[PASS] Starting Enhanced Visual Architecture...")
            
            # Initialize visualization components
            await self._initialize_visualization_components()
            
            # Start WebSocket server
            await self._start_websocket_server()
            
            # Start background rendering tasks
            await self._start_background_tasks()
            
            # Initialize adaptive optimization
            await self._initialize_adaptive_optimization()
            
            self.is_running = True
            logger.info("[PASS] Enhanced Visual Architecture started")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Error starting visualization: {e}")
            return False
    
    async def stop_visualization(self) -> bool:
        """Stop the enhanced visualization system"""
        try:
            logger.info("[PASS] Stopping Enhanced Visual Architecture...")
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Close WebSocket server
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
            
            # Disconnect all clients
            for client in self.connected_clients.copy():
                await client.close()
            
            self.is_running = False
            logger.info("[PASS] Enhanced Visual Architecture stopped")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Error stopping visualization: {e}")
            return False
    
    async def update_bit_mapping_visualization(self,
                                             current_level: BitMappingLevel,
                                             target_level: BitMappingLevel,
                                             processing_intensity: float) -> None:
        """Update bit mapping visualization with smooth transitions"""
        # Update bit visualization state
        self.bit_visualization.current_level = current_level
        self.bit_visualization.target_level = target_level
        self.bit_visualization.processing_intensity = processing_intensity
        
        # Calculate transition progress
        if current_level != target_level:
            transition_speed = self._calculate_transition_speed()
            self.bit_visualization.transition_progress = min(1.0, 
                self.bit_visualization.transition_progress + transition_speed)
        else:
            self.bit_visualization.transition_progress = 0.0
        
        # Generate bit particles for visualization
        await self._generate_bit_particles(current_level, target_level)
        
        # Update connection strengths based on processing intensity
        await self._update_connection_strengths(processing_intensity)
        
        # Create processing waves for phaser level (42-bit)
        if current_level.value >= 42 or target_level.value >= 42:
            await self._generate_processing_waves()
        
        # Broadcast update to clients
        await self._broadcast_bit_mapping_update()
    
    async def update_profit_vector_smoothing(self,
                                           orbital_positions: Dict[OrbitalZone, Tuple[float, float]],
                                           profit_data: List[Dict[str, Any]]) -> None:
        """Update profit vector visualization with smoothing and tick integration"""
        current_time = time.time()
        dt = current_time - self.profit_vector.last_tick
        
        # Update tick frequency based on system load
        self._adaptive_tick_frequency()
        
        # Process profit data into vector points
        new_profit_points = []
        for data in profit_data:
            x = data.get("time_offset", 0.0)
            y = data.get("profit_amount", 0.0)
            new_profit_points.append((x, y))
        
        # Apply vector smoothing
        smoothed_points = await self._apply_vector_smoothing(
            new_profit_points, 
            self.profit_vector.smoothing_factor
        )
        
        # Update orbital positions with interpolation
        await self._update_orbital_positions(orbital_positions, dt)
        
        # Calculate vector velocity for momentum visualization
        await self._calculate_vector_velocity(smoothed_points, dt)
        
        # Manage drift accumulation
        await self._manage_profit_drift(dt)
        
        # Update visualization state
        self.profit_vector.profit_points = smoothed_points
        self.profit_vector.orbital_positions = orbital_positions
        self.profit_vector.last_tick = current_time
        
        # Broadcast update
        await self._broadcast_profit_vector_update()
    
    async def update_storage_pipeline_visualization(self,
                                                  pipeline_status: Dict[str, Any]) -> None:
        """Update storage pipeline visualization with flow animations"""
        # Update pipeline flow rates
        self.storage_pipeline.pipeline_flow = {
            "ram_to_mid": pipeline_status.get("ram_to_mid_flow", 0.0),
            "mid_to_long": pipeline_status.get("mid_to_long_flow", 0.0),
            "long_to_archive": pipeline_status.get("long_to_archive_flow", 0.0)
        }
        
        # Update compression ratios
        self.storage_pipeline.compression_ratios = {
            "mid_term": pipeline_status.get("mid_compression", 1.0),
            "long_term": pipeline_status.get("long_compression", 2.0),
            "archive": pipeline_status.get("archive_compression", 5.0)
        }
        
        # Generate RAM particles for visualization
        await self._generate_ram_particles(pipeline_status)
        
        # Update thermal color mapping
        await self._update_thermal_colors(pipeline_status.get("thermal_state", {}))
        
        # Adjust flow speed based on system performance
        await self._adjust_flow_speed(pipeline_status)
        
        # Broadcast update
        await self._broadcast_storage_pipeline_update()
    
    async def handle_high_frequency_allocations(self,
                                              allocations: List[Dict[str, Any]]) -> None:
        """Handle visualization of high-frequency allocations with drift management"""
        current_time = time.time()
        
        # Add new allocations with timestamps
        for allocation in allocations:
            allocation["timestamp"] = current_time
            allocation["visualization_id"] = f"hf_{len(self.hf_allocations)}_{current_time}"
        
        # Add to high-frequency allocation list
        self.hf_allocations.extend(allocations)
        
        # Manage allocation count to prevent memory overflow
        if len(self.hf_allocations) > self.max_hf_display:
            # Remove oldest allocations
            remove_count = len(self.hf_allocations) - self.max_hf_display
            self.hf_allocations = self.hf_allocations[remove_count:]
        
        # Calculate drift compensation
        drift_compensation = await self._calculate_drift_compensation(allocations)
        
        # Update performance metrics
        self.performance_metrics["drift_compensation"] = drift_compensation
        
        # Adjust visualization quality if needed
        if len(allocations) > 100:  # High-frequency threshold
            await self._adaptive_quality_adjustment()
        
        # Broadcast high-frequency update
        await self._broadcast_hf_allocation_update(allocations, drift_compensation)
    
    async def handle_millisecond_sequencing(self,
                                          sequence_events: List[Dict[str, Any]]) -> None:
        """Handle millisecond-level sequencing events with error compensation"""
        current_time = time.time() * 1000  # Convert to milliseconds
        
        processed_events = []
        error_events = []
        
        for event in sequence_events:
            try:
                # Add timing information
                event["processing_time"] = current_time
                event["sequence_id"] = f"seq_{current_time}_{len(processed_events)}"
                
                # Validate event integrity
                if await self._validate_sequence_event(event):
                    processed_events.append(event)
                else:
                    error_events.append(event)
                    
            except Exception as e:
                logger.error(f"Error processing sequence event: {e}")
                error_events.append({
                    "error": str(e),
                    "original_event": event,
                    "timestamp": current_time
                })
        
        # Update error rate metrics
        total_events = len(processed_events) + len(error_events)
        if total_events > 0:
            error_rate = len(error_events) / total_events
            self.performance_metrics["error_rate"] = error_rate
        
        # Handle error events with compensation
        if error_events:
            await self._handle_sequence_errors(error_events)
        
        # Broadcast sequencing update
        await self._broadcast_sequencing_update(processed_events, error_events)
    
    async def adaptive_optimization_cycle(self) -> None:
        """Perform adaptive optimization based on back-tested logs and performance"""
        current_time = time.time()
        
        # Check if optimization cycle is due
        if current_time - self.last_optimization < 5.0:  # 5-second minimum interval
            return
        
        # Gather performance data
        performance_data = await self._gather_performance_data()
        
        # Add to optimization history
        self.optimization_history.append({
            "timestamp": current_time,
            "performance": performance_data,
            "render_quality": self.render_quality.value,
            "frame_rate": self.performance_metrics["frame_rate"]
        })
        
        # Keep only recent history (last 100 entries)
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        # Analyze trends and optimize
        optimizations = await self._analyze_and_optimize(performance_data)
        
        # Apply optimizations
        if optimizations:
            await self._apply_optimizations(optimizations)
        
        self.last_optimization = current_time
        
        # Broadcast optimization update
        await self._broadcast_optimization_update(optimizations)
    
    def _calculate_transition_speed(self) -> float:
        """Calculate bit level transition speed based on system performance"""
        base_speed = 0.02  # 2% per frame at 60 FPS
        
        # Adjust based on performance
        if self.performance_metrics["frame_rate"] < 30:
            return base_speed * 0.5  # Slower transitions for low performance
        elif self.performance_metrics["frame_rate"] > 90:
            return base_speed * 2.0  # Faster transitions for high performance
        
        return base_speed
    
    async def _generate_bit_particles(self,
                                    current_level: BitMappingLevel,
                                    target_level: BitMappingLevel) -> None:
        """Generate particle visualization for bit levels"""
        # Clear existing particles
        self.bit_visualization.bit_particles.clear()
        
        # Generate particles based on bit level
        bit_count = max(current_level.value, target_level.value)
        
        for i in range(bit_count):
            # Calculate particle position in bit space
            angle = (2 * math.pi * i) / bit_count
            radius = 100 + (bit_count - 4) * 5  # Expand radius with bit level
            
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            
            # Particle properties
            particle = {
                "id": f"bit_{i}",
                "x": x,
                "y": y,
                "active": i < current_level.value,
                "transitioning": current_level != target_level and i < target_level.value,
                "intensity": self.bit_visualization.processing_intensity,
                "color": self._get_bit_color(i, bit_count),
                "size": self._get_bit_size(i, current_level)
            }
            
            self.bit_visualization.bit_particles.append(particle)
    
    async def _apply_vector_smoothing(self,
                                    new_points: List[Tuple[float, float]],
                                    smoothing_factor: float) -> List[Tuple[float, float]]:
        """Apply smoothing to profit vector points"""
        if not self.profit_vector.profit_points:
            return new_points
        
        smoothed_points = []
        old_points = self.profit_vector.profit_points
        
        # Interpolate between old and new points
        for i, (new_x, new_y) in enumerate(new_points):
            if i < len(old_points):
                old_x, old_y = old_points[i]
                
                # Apply exponential smoothing
                smooth_x = old_x * smoothing_factor + new_x * (1 - smoothing_factor)
                smooth_y = old_y * smoothing_factor + new_y * (1 - smoothing_factor)
                
                smoothed_points.append((smooth_x, smooth_y))
            else:
                smoothed_points.append((new_x, new_y))
        
        return smoothed_points
    
    async def _calculate_drift_compensation(self,
                                          allocations: List[Dict[str, Any]]) -> float:
        """Calculate drift compensation for high-frequency visualizations"""
        if len(allocations) < 10:
            return 0.0
        
        # Calculate allocation rate
        time_span = max(alloc.get("timestamp", 0) for alloc in allocations) - \
                   min(alloc.get("timestamp", 0) for alloc in allocations)
        
        if time_span == 0:
            return 0.0
        
        rate = len(allocations) / time_span
        
        # Calculate drift based on rate
        if rate > 1000:  # Very high frequency
            drift = min(0.5, rate / 10000)  # Cap at 50% compensation
        elif rate > 100:  # High frequency
            drift = rate / 5000
        else:
            drift = 0.0
        
        return drift
    
    async def _adaptive_quality_adjustment(self) -> None:
        """Adaptively adjust rendering quality based on system load"""
        current_fps = self.performance_metrics["frame_rate"]
        cpu_usage = self.performance_metrics["cpu_usage"]
        
        # Quality adjustment logic
        if current_fps < 30 or cpu_usage > 80:
            # Reduce quality
            if self.render_quality == RenderQuality.HIGH:
                self.render_quality = RenderQuality.MEDIUM
            elif self.render_quality == RenderQuality.MEDIUM:
                self.render_quality = RenderQuality.LOW
        elif current_fps > 90 and cpu_usage < 50:
            # Increase quality
            if self.render_quality == RenderQuality.LOW:
                self.render_quality = RenderQuality.MEDIUM
            elif self.render_quality == RenderQuality.MEDIUM:
                self.render_quality = RenderQuality.HIGH
        
        # Adjust particle density and animation quality
        await self._adjust_render_parameters()
    
    def _adaptive_tick_frequency(self) -> None:
        """Adaptively adjust tick frequency based on system performance"""
        current_fps = self.performance_metrics["frame_rate"]
        
        # Adjust tick frequency to match display capability
        if current_fps < 30:
            self.profit_vector.tick_frequency = 30.0
        elif current_fps < 60:
            self.profit_vector.tick_frequency = current_fps
        else:
            self.profit_vector.tick_frequency = min(120.0, current_fps)
    
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time UI communication"""
        async def handle_client(websocket: WebSocketServerProtocol, path: str):
            """Handle WebSocket client connections"""
            self.connected_clients.add(websocket)
            logger.info(f"🔌 Client connected: {websocket.remote_address}")
            
            try:
                # Send initial state
                await self._send_initial_state(websocket)
                
                # Handle client messages
                async for message in websocket:
                    await self._handle_client_message(websocket, message)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"🔌 Client disconnected: {websocket.remote_address}")
            finally:
                self.connected_clients.discard(websocket)
        
        # Start WebSocket server
        self.websocket_server = await websockets.serve(
            handle_client,
            self.websocket_host,
            self.websocket_port
        )
        
        logger.info(f"🌐 WebSocket server started on {self.websocket_host}:{self.websocket_port}")
    
    async def _broadcast_bit_mapping_update(self) -> None:
        """Broadcast bit mapping visualization update to all clients"""
        update_data = {
            "type": "bit_mapping_update",
            "timestamp": time.time(),
            "data": {
                "current_level": self.bit_visualization.current_level.value,
                "target_level": self.bit_visualization.target_level.value,
                "transition_progress": self.bit_visualization.transition_progress,
                "processing_intensity": self.bit_visualization.processing_intensity,
                "particles": self.bit_visualization.bit_particles,
                "connections": self.bit_visualization.connection_strengths,
                "waves": self.bit_visualization.processing_waves
            }
        }
        
        await self._broadcast_to_clients(update_data)
    
    async def _broadcast_profit_vector_update(self) -> None:
        """Broadcast profit vector visualization update to all clients"""
        update_data = {
            "type": "profit_vector_update",
            "timestamp": time.time(),
            "data": {
                "profit_points": self.profit_vector.profit_points,
                "orbital_positions": {
                    zone.value: pos for zone, pos in self.profit_vector.orbital_positions.items()
                },
                "vector_velocity": self.profit_vector.vector_velocity,
                "tick_frequency": self.profit_vector.tick_frequency,
                "accumulated_drift": self.profit_vector.accumulated_drift
            }
        }
        
        await self._broadcast_to_clients(update_data)
    
    async def _broadcast_hf_allocation_update(self,
                                            allocations: List[Dict[str, Any]],
                                            drift_compensation: float) -> None:
        """Broadcast high-frequency allocation update to clients"""
        # Prepare allocation data for visualization
        viz_allocations = []
        for alloc in allocations[-100:]:  # Send only recent 100 for performance
            viz_alloc = {
                "id": alloc.get("visualization_id"),
                "symbol": alloc.get("symbol"),
                "amount": alloc.get("amount", 0.0),
                "timestamp": alloc.get("timestamp"),
                "zone": alloc.get("zone", "unknown"),
                "profit_tier": alloc.get("profit_tier", "micro")
            }
            viz_allocations.append(viz_alloc)
        
        update_data = {
            "type": "hf_allocation_update",
            "timestamp": time.time(),
            "data": {
                "allocations": viz_allocations,
                "total_count": len(self.hf_allocations),
                "drift_compensation": drift_compensation,
                "render_quality": self.render_quality.value,
                "performance_metrics": self.performance_metrics
            }
        }
        
        await self._broadcast_to_clients(update_data)
    
    async def _broadcast_to_clients(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps(data, default=str)
        
        # Send to all clients (remove disconnected ones)
        disconnected_clients = set()
        
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    def get_visualization_status(self) -> Dict[str, Any]:
        """Get comprehensive visualization system status"""
        return {
            "is_running": self.is_running,
            "current_mode": self.current_mode.value,
            "render_quality": self.render_quality.value,
            "connected_clients": len(self.connected_clients),
            "performance_metrics": self.performance_metrics.copy(),
            "bit_visualization": {
                "current_level": self.bit_visualization.current_level.value,
                "target_level": self.bit_visualization.target_level.value,
                "transition_progress": self.bit_visualization.transition_progress,
                "particle_count": len(self.bit_visualization.bit_particles)
            },
            "profit_vector": {
                "point_count": len(self.profit_vector.profit_points),
                "tick_frequency": self.profit_vector.tick_frequency,
                "accumulated_drift": self.profit_vector.accumulated_drift
            },
            "hf_allocations": {
                "count": len(self.hf_allocations),
                "max_display": self.max_hf_display,
                "drift_threshold": self.drift_threshold
            },
            "optimization": {
                "history_length": len(self.optimization_history),
                "last_optimization": self.last_optimization
            }
        }

# Integration function
async def create_integrated_visualization(practical_controller: PracticalVisualController,
                                        orbital_navigator: Optional[OrbitalProfitNavigator] = None) -> EnhancedVisualArchitecture:
    """
    Create and initialize integrated visualization system
    
    Args:
        practical_controller: The practical visual controller
        orbital_navigator: Optional orbital profit navigator
        
    Returns:
        Initialized enhanced visual architecture
    """
    # Create visualization system
    visual_arch = EnhancedVisualArchitecture(
        practical_controller=practical_controller,
        orbital_navigator=orbital_navigator
    )
    
    # Start visualization
    success = await visual_arch.start_visualization()
    
    if not success:
        raise RuntimeError("Failed to start enhanced visual architecture")
    
    # Integrate with practical controller callbacks
    practical_controller._visualization_callback = visual_arch.handle_controller_updates
    
    logger.info("[PASS] Integrated visualization system created successfully")
    return visual_arch 