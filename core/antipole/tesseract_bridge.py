"""
Tesseract Visualizer Bridge
===========================

Bridges Anti-Pole Theory calculations with 4D Tesseract visualization system.
Converts mathematical states into visual glyph packets for real-time profit navigation.
"""

import json
import asyncio
import websockets
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from uuid import uuid4
import logging

from .vector import AntiPoleState, AntiPoleVector
from .zbe_controller import ThermalMetrics, ThermalState

logger = logging.getLogger(__name__)

@dataclass
class GlyphPacket:
    """4D Glyph data packet for Tesseract visualization"""
    id: str
    timestamp: float
    glyph_type: str
    coordinates: List[float]  # 4D coordinates [x, y, z, w]
    color: List[float]        # RGBA color [r, g, b, a]
    size: float
    intensity: float
    metadata: Dict[str, Any]

@dataclass 
class TesseractFrame:
    """Complete frame of Tesseract visualization data"""
    frame_id: str
    timestamp: datetime
    glyphs: List[GlyphPacket]
    camera_position: List[float]  # 4D camera position
    profit_tier: Optional[str]
    thermal_state: str
    system_health: Dict[str, float]

class TesseractVisualizer:
    """
    4D Tesseract Visualization Engine for Anti-Pole Theory
    
    Converts Anti-Pole mathematical states into visual representations
    that guide profit navigation and portfolio management decisions.
    """
    
    def __init__(self, websocket_port: int = 8765, max_glyphs: int = 1000):
        self.websocket_port = websocket_port
        self.max_glyphs = max_glyphs
        
        # Visualization state
        self.active_glyphs = []
        self.glyph_history = []
        self.current_frame = None
        
        # 4D space mapping
        self.space_bounds = {
            'x': (-10.0, 10.0),    # Price momentum axis
            'y': (-10.0, 10.0),    # Volume intensity axis
            'z': (-10.0, 10.0),    # Hash entropy axis
            'w': (-5.0, 5.0)       # Anti-pole drift axis
        }
        
        # Color schemes for different states
        self.color_schemes = {
            'COLD': [0.2, 0.4, 0.8, 0.7],      # Blue
            'COOL': [0.2, 0.8, 0.6, 0.8],      # Cyan
            'WARM': [0.8, 0.8, 0.2, 0.8],      # Yellow
            'HOT': [0.9, 0.5, 0.1, 0.9],       # Orange
            'CRITICAL': [0.9, 0.2, 0.2, 1.0],  # Red
            'EMERGENCY': [1.0, 0.0, 0.0, 1.0], # Bright Red
            'PLATINUM': [0.9, 0.9, 0.9, 1.0],  # Platinum
            'GOLD': [1.0, 0.8, 0.0, 1.0],      # Gold
            'SILVER': [0.7, 0.7, 0.8, 1.0],    # Silver
            'BRONZE': [0.8, 0.5, 0.2, 1.0],    # Bronze
        }
        
        # WebSocket connections
        self.connections = set()
        self.server = None
        
        # Performance tracking
        self.frames_rendered = 0
        self.last_render_time = None
        
        logger.info(f"TesseractVisualizer initialized on port {websocket_port}")

    def map_antipole_to_4d(self, state: AntiPoleState, btc_price: float, volume: float) -> List[float]:
        """
        Map Anti-Pole state to 4D Tesseract coordinates
        
        Returns [x, y, z, w] coordinates:
        x: Price momentum (normalized)
        y: Volume intensity (normalized) 
        z: Hash entropy (normalized)
        w: Anti-pole drift (normalized)
        """
        # X-axis: Price momentum (requires price history for calculation)
        # For now, use hash entropy as proxy
        x = self._normalize_to_range(state.hash_entropy, 0.0, 1.0, 
                                   self.space_bounds['x'][0], self.space_bounds['x'][1])
        
        # Y-axis: Volume intensity (log scale normalization)
        volume_normalized = np.log10(max(volume, 1)) / 10.0  # Rough normalization
        y = self._normalize_to_range(volume_normalized, 0.0, 1.0,
                                   self.space_bounds['y'][0], self.space_bounds['y'][1])
        
        # Z-axis: Hash entropy (direct mapping)
        z = self._normalize_to_range(state.hash_entropy, 0.0, 1.0,
                                   self.space_bounds['z'][0], self.space_bounds['z'][1])
        
        # W-axis: Anti-pole drift (centered around 0)
        w = self._normalize_to_range(state.delta_psi_bar, -1.0, 1.0,
                                   self.space_bounds['w'][0], self.space_bounds['w'][1])
        
        return [x, y, z, w]

    def _normalize_to_range(self, value: float, in_min: float, in_max: float, 
                           out_min: float, out_max: float) -> float:
        """Normalize value from input range to output range"""
        # Clamp input value to range
        value = max(in_min, min(in_max, value))
        
        # Normalize to 0-1
        normalized = (value - in_min) / (in_max - in_min) if in_max != in_min else 0.5
        
        # Scale to output range
        return out_min + normalized * (out_max - out_min)

    def create_antipole_glyph(self, state: AntiPoleState, btc_price: float, 
                             volume: float, thermal_metrics: Optional[ThermalMetrics] = None) -> GlyphPacket:
        """
        Create a glyph representing the current Anti-Pole state
        """
        glyph_id = str(uuid4())
        coordinates = self.map_antipole_to_4d(state, btc_price, volume)
        
        # Determine glyph type based on state
        if state.profit_tier:
            glyph_type = f"profit_{state.profit_tier.lower()}"
            base_color = self.color_schemes.get(state.profit_tier, [0.5, 0.5, 0.5, 0.8])
        elif thermal_metrics:
            glyph_type = f"thermal_{thermal_metrics.state.value.lower()}"
            base_color = self.color_schemes.get(thermal_metrics.state.value, [0.5, 0.5, 0.5, 0.8])
        else:
            glyph_type = "neutral"
            base_color = [0.5, 0.5, 0.5, 0.6]
        
        # Adjust color based on ICAP probability
        color = base_color.copy()
        color[3] = min(1.0, base_color[3] * (0.5 + state.icap_probability))  # Alpha based on ICAP
        
        # Calculate size based on confidence/intensity
        base_size = 1.0
        size_multiplier = 0.5 + (state.icap_probability * 1.5)  # 0.5 to 2.0 range
        size = base_size * size_multiplier
        
        # Calculate intensity (for glow effects)
        intensity = state.icap_probability * (2.0 if state.is_ready else 1.0)
        
        # Metadata for debugging and analysis
        metadata = {
            'delta_psi_bar': state.delta_psi_bar,
            'icap_probability': state.icap_probability,
            'hash_entropy': state.hash_entropy,
            'is_ready': state.is_ready,
            'profit_tier': state.profit_tier,
            'phase_lock': state.phase_lock,
            'btc_price': btc_price,
            'volume': volume,
            'thermal_coefficient': state.thermal_coefficient
        }
        
        if thermal_metrics:
            metadata.update({
                'thermal_load': thermal_metrics.thermal_load,
                'thermal_state': thermal_metrics.state.value,
                'cooldown_active': thermal_metrics.cooldown_active,
                'cpu_temp': thermal_metrics.cpu_temp,
                'cpu_usage': thermal_metrics.cpu_usage
            })
        
        return GlyphPacket(
            id=glyph_id,
            timestamp=state.timestamp.timestamp(),
            glyph_type=glyph_type,
            coordinates=coordinates,
            color=color,
            size=size,
            intensity=intensity,
            metadata=metadata
        )

    def create_profit_navigation_glyphs(self, state: AntiPoleState, 
                                       btc_price: float, volume: float) -> List[GlyphPacket]:
        """
        Create specialized glyphs for profit navigation guidance
        """
        glyphs = []
        
        # Main Anti-Pole glyph
        main_glyph = self.create_antipole_glyph(state, btc_price, volume)
        glyphs.append(main_glyph)
        
        # If ready state, create navigation trail
        if state.is_ready:
            trail_glyphs = self._create_navigation_trail(state, btc_price, volume)
            glyphs.extend(trail_glyphs)
        
        # If profit tier detected, create profit zone indicator
        if state.profit_tier:
            profit_zone = self._create_profit_zone_glyph(state, btc_price, volume)
            glyphs.append(profit_zone)
        
        # If phase lock, create stability indicator
        if state.phase_lock:
            stability_glyph = self._create_stability_glyph(state, btc_price, volume)
            glyphs.append(stability_glyph)
        
        return glyphs

    def _create_navigation_trail(self, state: AntiPoleState, 
                               btc_price: float, volume: float) -> List[GlyphPacket]:
        """Create trail glyphs showing profit navigation path"""
        trail = []
        base_coords = self.map_antipole_to_4d(state, btc_price, volume)
        
        # Create 5 trail points extending in the W dimension
        for i in range(5):
            offset = (i + 1) * 0.5
            trail_coords = base_coords.copy()
            trail_coords[3] += offset  # Extend in W dimension
            
            alpha = 0.8 - (i * 0.15)  # Fade out
            trail_color = [0.0, 1.0, 0.5, alpha]  # Green trail
            
            trail_glyph = GlyphPacket(
                id=f"trail_{state.timestamp.timestamp()}_{i}",
                timestamp=state.timestamp.timestamp(),
                glyph_type="navigation_trail",
                coordinates=trail_coords,
                color=trail_color,
                size=0.3 - (i * 0.05),
                intensity=alpha,
                metadata={'trail_index': i, 'parent_ready': True}
            )
            trail.append(trail_glyph)
        
        return trail

    def _create_profit_zone_glyph(self, state: AntiPoleState, 
                                 btc_price: float, volume: float) -> GlyphPacket:
        """Create profit zone indicator glyph"""
        coords = self.map_antipole_to_4d(state, btc_price, volume)
        
        # Larger glyph surrounding the main point
        zone_coords = coords.copy()
        
        tier_colors = {
            'PLATINUM': [0.9, 0.9, 0.9, 0.3],
            'GOLD': [1.0, 0.8, 0.0, 0.3],
            'SILVER': [0.7, 0.7, 0.8, 0.3],
            'BRONZE': [0.8, 0.5, 0.2, 0.3]
        }
        
        color = tier_colors.get(state.profit_tier, [0.5, 0.5, 0.5, 0.3])
        
        return GlyphPacket(
            id=f"profit_zone_{state.timestamp.timestamp()}",
            timestamp=state.timestamp.timestamp(),
            glyph_type="profit_zone",
            coordinates=zone_coords,
            color=color,
            size=3.0,  # Large zone
            intensity=0.5,
            metadata={'profit_tier': state.profit_tier, 'zone_type': 'profit'}
        )

    def _create_stability_glyph(self, state: AntiPoleState, 
                               btc_price: float, volume: float) -> GlyphPacket:
        """Create phase lock stability indicator"""
        coords = self.map_antipole_to_4d(state, btc_price, volume)
        
        # Ring around the main point
        stability_coords = coords.copy()
        
        return GlyphPacket(
            id=f"stability_{state.timestamp.timestamp()}",
            timestamp=state.timestamp.timestamp(),
            glyph_type="phase_lock",
            coordinates=stability_coords,
            color=[0.0, 0.8, 1.0, 0.8],  # Cyan stability
            size=1.5,
            intensity=0.9,
            metadata={'phase_lock': True, 'stability_factor': state.recursion_stability}
        )

    def update_frame(self, state: AntiPoleState, btc_price: float, volume: float,
                    thermal_metrics: Optional[ThermalMetrics] = None) -> TesseractFrame:
        """
        Update the current visualization frame with new Anti-Pole data
        """
        timestamp = datetime.now()
        frame_id = str(uuid4())
        
        # Create glyphs for this frame
        new_glyphs = self.create_profit_navigation_glyphs(state, btc_price, volume)
        
        # Add thermal glyph if available
        if thermal_metrics:
            thermal_glyph = self.create_antipole_glyph(state, btc_price, volume, thermal_metrics)
            thermal_glyph.glyph_type = "thermal_indicator"
            new_glyphs.append(thermal_glyph)
        
        # Update active glyphs list
        self.active_glyphs.extend(new_glyphs)
        
        # Remove old glyphs to maintain performance
        current_time = timestamp.timestamp()
        self.active_glyphs = [
            glyph for glyph in self.active_glyphs 
            if current_time - glyph.timestamp < 60.0  # Keep last 60 seconds
        ]
        
        # Limit total glyph count
        if len(self.active_glyphs) > self.max_glyphs:
            self.active_glyphs = self.active_glyphs[-self.max_glyphs:]
        
        # Calculate camera position (adaptive based on activity)
        camera_position = self._calculate_camera_position(state, thermal_metrics)
        
        # Create frame
        frame = TesseractFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            glyphs=self.active_glyphs.copy(),
            camera_position=camera_position,
            profit_tier=state.profit_tier,
            thermal_state=thermal_metrics.state.value if thermal_metrics else "UNKNOWN",
            system_health={
                'icap_probability': state.icap_probability,
                'delta_psi_bar': state.delta_psi_bar,
                'hash_entropy': state.hash_entropy,
                'thermal_load': thermal_metrics.thermal_load if thermal_metrics else 0.0,
                'is_ready': state.is_ready,
                'phase_lock': state.phase_lock
            }
        )
        
        self.current_frame = frame
        self.frames_rendered += 1
        self.last_render_time = timestamp
        
        return frame

    def _calculate_camera_position(self, state: AntiPoleState, 
                                  thermal_metrics: Optional[ThermalMetrics]) -> List[float]:
        """Calculate optimal 4D camera position for visualization"""
        # Base camera position
        camera = [0.0, 0.0, -15.0, 0.0]
        
        # Adjust based on anti-pole activity
        if state.is_ready:
            # Move closer when ready
            camera[2] = -10.0
            
        # Adjust based on profit tier
        if state.profit_tier in ['PLATINUM', 'GOLD']:
            # Zoom in for high-value opportunities
            camera[2] = -8.0
            
        # Adjust W position based on drift
        camera[3] = state.delta_psi_bar * 2.0
        
        return camera

    async def broadcast_frame(self, frame: TesseractFrame):
        """Broadcast frame to all connected WebSocket clients"""
        if not self.connections:
            return
        
        # Convert frame to JSON
        frame_data = {
            'type': 'tesseract_frame',
            'frame_id': frame.frame_id,
            'timestamp': frame.timestamp.isoformat(),
            'glyphs': [asdict(glyph) for glyph in frame.glyphs],
            'camera_position': frame.camera_position,
            'profit_tier': frame.profit_tier,
            'thermal_state': frame.thermal_state,
            'system_health': frame.system_health,
            'metadata': {
                'frames_rendered': self.frames_rendered,
                'active_glyph_count': len(frame.glyphs),
                'render_time': frame.timestamp.isoformat()
            }
        }
        
        message = json.dumps(frame_data)
        
        # Send to all connections
        disconnected = set()
        for websocket in self.connections:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                logger.warning(f"Failed to send frame to client: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.connections -= disconnected

    async def handle_websocket_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        logger.info(f"New WebSocket connection from {websocket.remote_address}")
        self.connections.add(websocket)
        
        try:
            # Send current frame immediately
            if self.current_frame:
                await self.broadcast_frame(self.current_frame)
            
            # Keep connection alive
            await websocket.wait_closed()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connections.discard(websocket)
            logger.info(f"WebSocket connection closed: {websocket.remote_address}")

    async def start_websocket_server(self):
        """Start the WebSocket server for real-time visualization"""
        self.server = await websockets.serve(
            self.handle_websocket_connection,
            "localhost",
            self.websocket_port
        )
        logger.info(f"🚀 Tesseract WebSocket server started on ws://localhost:{self.websocket_port}")

    async def stop_websocket_server(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Tesseract WebSocket server stopped")

    def get_visualization_statistics(self) -> Dict:
        """Get visualization performance statistics"""
        return {
            'frames_rendered': self.frames_rendered,
            'active_connections': len(self.connections),
            'active_glyphs': len(self.active_glyphs),
            'last_render_time': self.last_render_time.isoformat() if self.last_render_time else None,
            'max_glyphs': self.max_glyphs,
            'websocket_port': self.websocket_port,
            'current_frame_id': self.current_frame.frame_id if self.current_frame else None
        }

    def export_frame_data(self, filename: Optional[str] = None) -> str:
        """Export current frame data to JSON file"""
        if not self.current_frame:
            return ""
        
        filename = filename or f"tesseract_frame_{self.current_frame.frame_id}.json"
        
        frame_export = {
            'frame': asdict(self.current_frame),
            'statistics': self.get_visualization_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(frame_export, f, indent=2, default=str)
        
        logger.info(f"Frame data exported to {filename}")
        return filename 