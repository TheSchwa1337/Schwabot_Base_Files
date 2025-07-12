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

@app.post("/trigger_trade")
async def trigger_trade(trade_data: dict):
    """Manually triggers a trade with full execution logic."""
    try:
        # Validate required fields
        required_fields = ["action", "symbol", "quantity"]
        for field in required_fields:
            if field not in trade_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        action = trade_data.get("action").upper()
        symbol = trade_data.get("symbol")
        quantity = float(trade_data.get("quantity"))
        price = trade_data.get("price")
        exchange = trade_data.get("exchange", "binance")
        hash_id = trade_data.get("hash_id", f"manual_{int(time.time())}")
        
        # Validate action
        if action not in ["BUY", "SELL"]:
            raise HTTPException(status_code=400, detail="Action must be 'BUY' or 'SELL'")
        
        # Validate quantity
        if quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")
        
        # Get current price if not provided
        if not price and pipeline_api:
            try:
                market_data = await pipeline_api.get_market_data(exchange, symbol)
                price = market_data.get("price", 0)
                if not price:
                    raise HTTPException(status_code=400, detail="Could not fetch current price")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to fetch price: {str(e)}")
        
        # Create order request
        order_request = OrderRequest(
            symbol=symbol,
            side=action.lower(),
            type="market" if not price else "limit",
            amount=quantity,
            price=price,
            params={
                "hash_id": hash_id,
                "manual_trade": True,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Execute trade based on mode
        pipeline_mode = getattr(pipeline, "mode", "testing")
        
        if pipeline_mode == "live" and pipeline_api:
            # Live trading - execute through API
            try:
                result = await pipeline_api.place_order(exchange, order_request)
                success = result.get("success", False)
                
                if success:
                    await manager.broadcast(f"✅ LIVE TRADE EXECUTED: {action} {quantity} {symbol} @ {price}")
                    return {
                        "status": "success",
                        "message": f"Live trade executed: {action} {quantity} {symbol}",
                        "order_id": result.get("order_id"),
                        "hash_id": hash_id,
                        "execution_price": price,
                        "mode": "live"
                    }
                else:
                    await manager.broadcast(f"❌ LIVE TRADE FAILED: {result.get('error', 'Unknown error')}")
                    return {
                        "status": "error",
                        "message": f"Live trade failed: {result.get('error', 'Unknown error')}",
                        "hash_id": hash_id
                    }
                    
            except Exception as e:
                await manager.broadcast(f"❌ LIVE TRADE EXCEPTION: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Live trade exception: {str(e)}",
                    "hash_id": hash_id
                }
        
        else:
            # Demo/testing mode - simulate trade
            trade_result = {
                "order_id": f"demo_{int(time.time())}",
                "symbol": symbol,
                "side": action.lower(),
                "amount": quantity,
                "price": price,
                "status": "filled",
                "timestamp": datetime.now().isoformat(),
                "hash_id": hash_id
            }
            
            # Process through pipeline for tracking
            if pipeline:
                try:
                    pipeline.process_market_data(
                        symbol=symbol,
                        price=price,
                        volume=quantity,
                        granularity=1,
                        tick_index=0
                    )
                except Exception as e:
                    print(f"Warning: Failed to process trade through pipeline: {e}")
            
            await manager.broadcast(f"🎮 DEMO TRADE: {action} {quantity} {symbol} @ {price}")
            return {
                "status": "success",
                "message": f"Demo trade executed: {action} {quantity} {symbol}",
                "order_id": trade_result["order_id"],
                "hash_id": hash_id,
                "execution_price": price,
                "mode": "demo"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Trade execution failed: {str(e)}"
        await manager.broadcast(f"❌ {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "hash_id": trade_data.get("hash_id", "unknown")
        }

@app.post("/api/execute_signal")
async def execute_signal(signal_data: dict):
    """Execute a trading signal with mathematical validation."""
    try:
        # Extract signal data
        asset = signal_data.get("asset", "BTC/USDC")
        price = float(signal_data.get("price", 60000.0))
        quantity = float(signal_data.get("quantity", 0.1))
        mode = signal_data.get("mode", "demo")
        confidence = float(signal_data.get("confidence", 0.5))
        
        # Validate confidence
        if confidence < 0.0 or confidence > 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
        
        # Determine action based on confidence and price
        # This is a simple heuristic - in practice, you'd use more sophisticated logic
        current_price = 60000.0  # This should come from market data
        if price > current_price * 1.01:  # 1% above current
            action = "BUY"
        elif price < current_price * 0.99:  # 1% below current
            action = "SELL"
        else:
            action = "HOLD"
        
        if action == "HOLD":
            return {
                "status": "hold",
                "message": "Signal indicates hold position",
                "confidence": confidence,
                "price": price
            }
        
        # Execute the trade
        trade_data = {
            "action": action,
            "symbol": asset,
            "quantity": quantity,
            "price": price,
            "hash_id": f"signal_{int(time.time())}",
            "confidence": confidence,
            "mode": mode
        }
        
        return await trigger_trade(trade_data)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Signal execution failed: {str(e)}"
        await manager.broadcast(f"❌ {error_msg}")
        return {
            "status": "error",
            "message": error_msg
        }

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