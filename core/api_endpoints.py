#!/usr/bin/env python3
"""
FastAPI Endpoints for Schwabot UI Integration
=============================================

Critical missing piece: Real-time WebSocket streaming and API key management
that bridges the mathematical systems to the React dashboard.

Provides:
- WebSocket /ws/stream for live entropy, pattern, and hash data
- POST /api/register-key for secure API key handling with SHA-256
- REST endpoints for system configuration and monitoring
- Signal dispatch integration for Configuration_Hook_Fixes
"""

import asyncio
import json
import logging
import hashlib
import sqlite3
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Core system imports with graceful fallbacks
try:
    from .hash_recollection import HashRecollectionSystem
    from .sustainment_underlay_controller import SustainmentUnderlayController
    from .ui_state_bridge import UIStateBridge
    from .visual_integration_bridge import VisualIntegrationBridge
    CORE_SYSTEMS_AVAILABLE = True
except ImportError:
    CORE_SYSTEMS_AVAILABLE = False
    print("⚠️ Core systems not available - API will run in mock mode")

logger = logging.getLogger(__name__)

# Data models
class APIKeyRequest(BaseModel):
    api_key_hash: str  # SHA-256 hash from frontend
    exchange: str
    testnet: bool = True

class SystemConfigRequest(BaseModel):
    sustainment_threshold: Optional[float] = None
    update_interval: Optional[float] = None
    gpu_enabled: Optional[bool] = None

class StreamData(BaseModel):
    time: str
    entropy: float
    pattern: str
    confidence: float
    hash_count: int
    sustainment_index: float

# FastAPI app
app = FastAPI(title="Schwabot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_connections: Set[WebSocket] = set()
hash_system: Optional[HashRecollectionSystem] = None
sustainment_controller: Optional[SustainmentUnderlayController] = None
ui_bridge: Optional[UIStateBridge] = None
visual_bridge: Optional[VisualIntegrationBridge] = None

# Database for API keys
DB_PATH = Path.home() / ".schwabot" / "secrets.sqlite"

def init_database():
    """Initialize SQLite database for API key storage"""
    DB_PATH.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id TEXT PRIMARY KEY,
            key_hash BLOB NOT NULL,
            exchange TEXT NOT NULL,
            testnet BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP,
            active BOOLEAN DEFAULT TRUE
        )
    """)
    conn.commit()
    conn.close()

def store_api_key(key_hash: str, exchange: str, testnet: bool) -> str:
    """Store API key hash with salt in database"""
    # Generate key ID
    key_id = hashlib.sha256(f"{exchange}_{key_hash}_{time.time()}".encode()).hexdigest()[:16]
    
    # Salt and hash the key hash (double hashing for security)
    salt = hashlib.urandom(32)
    salted_hash = hashlib.pbkdf2_hmac('sha256', key_hash.encode(), salt, 100000)
    
    # Store in database
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        INSERT INTO api_keys (key_id, key_hash, exchange, testnet)
        VALUES (?, ?, ?, ?)
    """, (key_id, salt + salted_hash, exchange, testnet))
    conn.commit()
    conn.close()
    
    return key_id

def get_stream_data() -> StreamData:
    """Get current stream data from core systems"""
    if not CORE_SYSTEMS_AVAILABLE:
        # Mock data for development
        return StreamData(
            time=datetime.now().isoformat(),
            entropy=0.45 + 0.3 * (time.time() % 10) / 10,
            pattern="anti_pole_formation",
            confidence=0.85,
            hash_count=1247,
            sustainment_index=0.73
        )
    
    # Get real data from systems
    current_time = datetime.now().isoformat()
    
    # Hash recollection data
    hash_data = {}
    if hash_system:
        try:
            hash_data = hash_system.get_current_metrics()
        except:
            pass
    
    # Sustainment data
    sustainment_data = {}
    if sustainment_controller:
        try:
            sustainment_data = sustainment_controller.get_sustainment_status()
        except:
            pass
    
    # UI bridge data
    ui_data = {}
    if ui_bridge:
        try:
            ui_data = ui_bridge.get_ui_state()
        except:
            pass
    
    return StreamData(
        time=current_time,
        entropy=hash_data.get('entropy', 0.5),
        pattern=ui_data.get('active_pattern', 'unknown'),
        confidence=sustainment_data.get('sustainment_index', 0.5),
        hash_count=hash_data.get('total_hashes', 0),
        sustainment_index=sustainment_data.get('sustainment_index', 0.5)
    )

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    init_database()
    logger.info("🚀 Schwabot API initialized")

# WebSocket endpoint for real-time streaming
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time data stream for React dashboard"""
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"WebSocket client connected: {websocket.client}")
    
    try:
        while True:
            # Get current data
            stream_data = get_stream_data()
            
            # Send to client
            await websocket.send_text(stream_data.json())
            
            # Wait 250ms (4 FPS)
            await asyncio.sleep(0.25)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)

# API key registration endpoint
@app.post("/api/register-key")
async def register_api_key(request: APIKeyRequest):
    """Register API key (receives SHA-256 hash from frontend)"""
    try:
        # Validate hash format
        if len(request.api_key_hash) != 64:
            raise HTTPException(status_code=400, detail="Invalid API key hash format")
        
        # Store the key
        key_id = store_api_key(request.api_key_hash, request.exchange, request.testnet)
        
        # Trigger NCCO manager if available
        if CORE_SYSTEMS_AVAILABLE and hash_system:
            try:
                # This would trigger the NCCO manager to register the API
                pass
            except Exception as e:
                logger.warning(f"Failed to trigger NCCO manager: {e}")
        
        return {"status": "success", "key_id": key_id, "message": "API key registered successfully"}
        
    except Exception as e:
        logger.error(f"API key registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System status endpoint
@app.get("/api/status")
async def get_system_status():
    """Get current system status"""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "hash_system": hash_system is not None,
            "sustainment_controller": sustainment_controller is not None,
            "ui_bridge": ui_bridge is not None,
            "visual_bridge": visual_bridge is not None,
        },
        "active_connections": len(active_connections),
        "core_systems_available": CORE_SYSTEMS_AVAILABLE
    }

# System configuration endpoint
@app.post("/api/configure")
async def configure_system(request: SystemConfigRequest):
    """Configure system parameters"""
    changes = {}
    
    try:
        if request.sustainment_threshold is not None and sustainment_controller:
            sustainment_controller.s_crit = request.sustainment_threshold
            changes["sustainment_threshold"] = request.sustainment_threshold
        
        if request.update_interval is not None and visual_bridge:
            # Would update visual bridge interval
            changes["update_interval"] = request.update_interval
        
        if request.gpu_enabled is not None and hash_system:
            hash_system.gpu_enabled = request.gpu_enabled
            changes["gpu_enabled"] = request.gpu_enabled
        
        return {"status": "success", "changes": changes}
        
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Strategy validation endpoint
@app.get("/api/validate")
async def get_strategy_validation():
    """Get current strategy validation results"""
    if not sustainment_controller:
        return {"status": "unavailable", "message": "Sustainment controller not available"}
    
    try:
        status = sustainment_controller.get_sustainment_status()
        
        # Extract 8-principle scores
        principles = {
            "anticipation": status.get("anticipation_score", 0.5),
            "integration": status.get("integration_score", 0.5),
            "responsiveness": status.get("responsiveness_score", 0.5),
            "simplicity": status.get("simplicity_score", 0.5),
            "economy": status.get("economy_score", 0.5),
            "survivability": status.get("survivability_score", 0.5),
            "continuity": status.get("continuity_score", 0.5),
            "improvisation": status.get("improvisation_score", 0.5),
        }
        
        return {
            "status": "success",
            "sustainment_index": status.get("sustainment_index", 0.5),
            "principles": principles,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Signal dispatch hook implementation
class SignalHooks:
    """Signal dispatch system for Configuration_Hook_Fixes integration"""
    
    @staticmethod
    def emit(match: Dict[str, Any]) -> None:
        """Emit signal for high-confidence matches"""
        try:
            # This is where we'd integrate with:
            # - profit_vector_router (confidence boost)
            # - thermal_manager (get drift bias)
            # - sfsss_router, vault_router, cluster_mapper, drift_engine
            
            signal_data = {
                "type": "pattern_match",
                "confidence": match.get("confidence", 0.0),
                "pattern_type": match.get("pattern", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "data": match
            }
            
            # Broadcast to WebSocket clients
            asyncio.create_task(broadcast_signal(signal_data))
            
        except Exception as e:
            logger.error(f"Signal dispatch error: {e}")

async def broadcast_signal(signal_data: Dict[str, Any]) -> None:
    """Broadcast signal to all WebSocket connections"""
    if not active_connections:
        return
    
    message = json.dumps(signal_data)
    disconnected = set()
    
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            disconnected.add(connection)
    
    # Clean up disconnected
    active_connections -= disconnected

# Export data endpoint
@app.get("/api/export")
async def export_data():
    """Export historical data"""
    if visual_bridge:
        export_data = visual_bridge._prepare_export_data()
        return {"status": "success", "data": export_data}
    else:
        return {"status": "unavailable", "message": "Visual bridge not available"}

# System initialization
def initialize_systems(
    hash_system_instance: Optional[HashRecollectionSystem] = None,
    sustainment_instance: Optional[SustainmentUnderlayController] = None,
    ui_bridge_instance: Optional[UIStateBridge] = None,
    visual_bridge_instance: Optional[VisualIntegrationBridge] = None
) -> None:
    """Initialize system instances"""
    global hash_system, sustainment_controller, ui_bridge, visual_bridge
    
    hash_system = hash_system_instance
    sustainment_controller = sustainment_instance
    ui_bridge = ui_bridge_instance
    visual_bridge = visual_bridge_instance
    
    logger.info("System instances initialized")

# GPU synchronization implementation
def synchronize_gpu_cpu() -> Dict[str, Any]:
    """Implementation for _synchronize_gpu_cpu stub"""
    if not hash_system:
        return {"status": "no_hash_system"}
    
    try:
        # Move queue lengths & CuPy mem stats into Prometheus metrics
        metrics = {
            "queue_depth": getattr(hash_system, 'result_queue', []),
            "gpu_memory_usage": 0.0,  # Would get from CuPy
            "cpu_memory_usage": 0.0,
            "sync_timestamp": time.time()
        }
        
        # Flush pending GPU hashes back to CPU for clustering
        if hasattr(hash_system, 'gpu_enabled') and hash_system.gpu_enabled:
            # Synchronization logic here
            pass
        
        return {"status": "success", "metrics": metrics}
        
    except Exception as e:
        logger.error(f"GPU sync error: {e}")
        return {"status": "error", "error": str(e)}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 