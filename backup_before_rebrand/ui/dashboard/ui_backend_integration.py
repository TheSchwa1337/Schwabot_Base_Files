import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState

# Add project root to path to load core modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.api.data_models import OrderRequest
from core.api.integration_manager import ApiIntegrationManager
from core.clean_trading_pipeline import CleanTradingPipeline, create_trading_pipeline

# --- Application Setup ---
app = FastAPI(
    title="Schwabot UI Backend",
    description="Provides API and WebSocket endpoints for the Schwabot dashboard.",
    version="1.0.0"
)

# --- Global State ---
pipeline: CleanTradingPipeline = None
config: dict = {}
pipeline_api: ApiIntegrationManager = None

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            if connection.client_state == WebSocketState.CONNECTED:
                await connection.send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the trading pipeline and other resources on application startup.
    """
    global pipeline, config
    # Load configuration
    config_path = os.path.join(project_root, "config", "live_config.json") # Assumes a default config
    if not os.path.exists(config_path):
        # Create a default config if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_config = {"symbol": "BTC/USDT", "initial_capital": 10000.0}
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        config = default_config
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Initialize the trading pipeline
    pipeline = create_trading_pipeline(
        symbol=config.get("symbol", "BTC/USDT"),
        initial_capital=config.get("initial_capital", 10000.0)
    )
    await manager.broadcast("Schwabot backend initialized.")

    # Initialize the API Integration Manager
    global pipeline_api
    pipeline_api = ApiIntegrationManager(config_path=os.path.join(project_root, "config", "api_keys.json"))
    await pipeline_api.start()
    await manager.broadcast("API Integration Manager started.")

@app.get("/status")
async def get_status():
    """Returns the current status of the trading bot."""
    if not pipeline:
        return {"error": "Pipeline not initialized"}
    
    summary = pipeline.get_pipeline_summary()
    return {
        "mode": getattr(pipeline, "mode", "testing"),
        "symbol": pipeline.symbol,
        "current_capital": summary.get("state", {}).get("current_capital"),
        "total_trades": summary.get("state", {}).get("total_trades"),
        "last_trade_timestamp": pipeline.state.timestamp, # Simplified
        "memory_slots": len(pipeline.market_data_history), # Example metric
    }

@app.post("/set_mode/{mode}")
async def set_mode(mode: str):
    """Sets the operating mode of the trading pipeline (testing, demo, live)."""
    if not pipeline:
        return {"error": "Pipeline not initialized"}
    try:
        pipeline.set_mode(mode)
        message = f"Mode changed to {mode.upper()}"
        await manager.broadcast(message)
        return {"status": "success", "message": message}
    except ValueError as e:
        return {"status": "error", "message": str(e)}

# --- Removed manual trade execution endpoint (/trigger_trade) ---

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time status and log streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # We can receive messages here if needed, but for now it's a one-way stream
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("A client has disconnected.")

# New API endpoints for trading integration
@app.post("/api/place_order")
async def place_order(order: dict):
    """Place an order via the API Integration Manager."""
    if not pipeline_api:
        return {"error": "API Integration Manager not initialized"}
    exchange = order.get("exchange")
    order_req = OrderRequest(**order.get("order", {}))
    resp = await pipeline_api.place_order(exchange, order_req)
    return resp

@app.get("/api/system_status")
async def api_system_status():
    """Get system status from the API Integration Manager."""
    if not pipeline_api:
        return {"error": "API Integration Manager not initialized"}
    return pipeline_api.get_system_status()

@app.get("/api/market_data/{exchange}/{symbol}")
async def api_market_data(exchange: str, symbol: str):
    """Fetch market data via the API Integration Manager."""
    if not pipeline_api:
        return {"error": "API Integration Manager not initialized"}
    data = await pipeline_api.get_market_data(exchange, symbol)
    return data

@app.post("/process_market_data")
async def process_market_data_endpoint(market_data: dict):
    """Process incoming market data through the unified pipeline."""
    if not pipeline:
        return {"error": "Pipeline not initialized"}
    result = pipeline.process_market_data(
        market_data.get("symbol"),
        market_data.get("price"),
        market_data.get("volume"),
        market_data.get("granularity", 1),
        market_data.get("tick_index", 0)
    )
    await manager.broadcast(json.dumps({"type": "signal", "data": result}))
    return {"status": "success", "data": result}

# --- Main execution ---
if __name__ == "__main__":
    print("🚀 Starting Schwabot UI Backend Server...")
    # Note: running this directly is for development.
    # For production, use a process manager like Gunicorn or systemd.
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info") 