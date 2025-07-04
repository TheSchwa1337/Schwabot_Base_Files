import asyncio
import json
import os
import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
import uvicorn

# Add project root to path to load core modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.clean_trading_pipeline import create_trading_pipeline, CleanTradingPipeline

# --- Application Setup ---
app = FastAPI(
    title="Schwabot UI Backend",
    description="Provides API and WebSocket endpoints for the Schwabot dashboard.",
    version="1.0.0"
)

# --- Global State ---
pipeline: CleanTradingPipeline = None
config: dict = {}

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

@app.post("/trigger_trade")
async def trigger_trade(trade_data: dict):
    """Manually triggers a trade (placeholder)."""
    # trade_data expected to contain: action, symbol, quantity, hash_id
    hash_id = trade_data.get("hash_id")
    await manager.broadcast(f"Manual trade triggered with hash ID: {hash_id}. (Feature is a placeholder)")
    # TODO: Implement manual trade execution logic
    return {"status": "placeholder", "message": "Manual trade endpoint is not fully implemented."}

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

# --- Main execution ---
if __name__ == "__main__":
    print("🚀 Starting Schwabot UI Backend Server...")
    # Note: running this directly is for development.
    # For production, use a process manager like Gunicorn or systemd.
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info") 