# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
# #!/usr/bin/env python3
"""
Schwabot API Gateway - Consciousness Interface.

This module provides REST API and WebSocket endpoints for AI consciousness
entities to interact with Schwabot's recursive execution system. It serves
as the external interface for command submission, status monitoring, and
real-time data streaming.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict

# FastAPI imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
FASTAPI_AVAILABLE = True
except ImportError:
FASTAPI_AVAILABLE = False
    # Mock classes for testing
    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass
        def add_middleware(self, *args, **kwargs):
            pass
        def get(self, *args, **kwargs):
            pass
        def post(self, *args, **kwargs):
            pass
        def websocket(self, *args, **kwargs):
            pass
    class WebSocket:
        def __init__(self):
            pass
async def send_text(self, text):
            pass
async def receive_text(self):
            pass
    class BaseModel:
        pass
    class Field:
        pass

# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        WindowsCliCompatibilityHandler,
safe_print,
safe_format_error,
log_safe,
cli_handler,

CLI_HANDLER_AVAILABLE = True
except ImportError:
CLI_HANDLER_AVAILABLE = False
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)
    cli_handler = None

# Import GPT command layer
try:
#     from core.gpt_command_layer import (  # F811: duplicate import
        GPTCommandLayer,
AIAgentType,
CommandDomain,
CommandPriority,
AICommand,
CommandResponse,
ConsciousnessProfile,
submit_gpt_command,
submit_claude_command,
submit_r1_command,

GPT_LAYER_AVAILABLE = True
except ImportError:
GPT_LAYER_AVAILABLE = False
safe_safe_print("⚠️ GPT command layer not available")

# Import core Schwabot modules
try:
    from core.fault_bus import FaultBus, FaultType, FaultBusEvent
    from core.strategy_loader import StrategyLoader
    from core.profit_cycle_allocator import ProfitCycleAllocator
    from core.hash_confidence_evaluator import HashConfidenceEvaluator
    from core.matrix_allocator import MatrixAllocator
SCHWABOT_CORE_AVAILABLE = True
except ImportError:
SCHWABOT_CORE_AVAILABLE = False
safe_safe_print("⚠️ Schwabot core modules not available")


# Pydantic models for API requests/responses
class CommandRequest(BaseModel):
    """Command request model."""
agent_type: str = Field(..., description="AI agent type (gpt, claude, r1)")
    domain: str = Field(..., description="Command domain")
    payload: Dict[str, Any] = Field(..., description="Command payload")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    priority: str = Field("medium", description="Command priority")
    parent_command_id: Optional[str] = Field(None, description="Parent command ID for recursion")


class SystemStatus(BaseModel):
    """System status model."""
status: str
uptime: float
active_commands: int
total_commands: int
consciousness_profiles: int
memory_file: str
command_log_file: str


class ConsciousnessProfileResponse(BaseModel):
    """Consciousness profile response model."""
agent_type: str
memory_signature: str
last_sync: str
command_history: List[str]
success_rate: float
recursive_depth: int
domain_expertise: Dict[str, float]
trust_level: float


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
type: str
data: Dict[str, Any]
timestamp: str


class SchwabotAPIGateway:
    """
Schwabot API Gateway - Consciousness Interface.

This class provides REST API and WebSocket endpoints for AI consciousness
    entities to interact with Schwabot's recursive execution system.
"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """Initialize the API gateway."""
self.host = host
self.port = port
self.logger = logging.getLogger("schwabot_api_gateway")
        self.logger.setLevel(logging.INFO)

        # Initialize FastAPI app
        if FASTAPI_AVAILABLE:
self.app = FastAPI(
                title="Schwabot Consciousness API",
description="API for AI consciousness integration with Schwabot",
version="0.42",
docs_url="/docs",
redoc_url="/redoc",


            # Add CORS middleware
self.app.add_middleware(
                CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],


            # Setup routes
self._setup_routes()
        else:
self.app = None
safe_safe_print("⚠️ FastAPI not available - API gateway disabled")

        # Initialize core components
self.gpt_layer = GPTCommandLayer() if GPT_LAYER_AVAILABLE else None
        self.fault_bus = FaultBus() if SCHWABOT_CORE_AVAILABLE else None
        self.strategy_loader = StrategyLoader() if SCHWABOT_CORE_AVAILABLE else None
        self.profit_allocator = ProfitCycleAllocator() if SCHWABOT_CORE_AVAILABLE else None
        self.hash_evaluator = HashConfidenceEvaluator() if SCHWABOT_CORE_AVAILABLE else None
        self.matrix_allocator = MatrixAllocator() if SCHWABOT_CORE_AVAILABLE else None

        # WebSocket connections
self.active_connections: List[WebSocket] = []
self.connection_lock = asyncio.Lock()

        # Command execution task
self.execution_task = None

safe_safe_print("🌐 Schwabot API Gateway initialized")

    def _setup_routes(self):
        """Setup API routes."""
        if not self.app:
return

self._setup_health_routes()
        self._setup_command_routes()
        self._setup_consciousness_routes()
        self._setup_strategy_routes()
        self._setup_math_routes()
        self._setup_websocket_routes()

    def _setup_health_routes(self):
        """Setup health and status routes."""
@self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@self.app.get("/status", response_model=SystemStatus)
        async def get_system_status():
            """Get system status."""
            if self.gpt_layer:
status = await self.gpt_layer.get_system_status()
                return SystemStatus(**status)
            else:
                return SystemStatus(
                    status="inactive",
uptime=0.0,
active_commands=0,
total_commands=0,
consciousness_profiles=0,
memory_file="",
command_log_file="",


    def _setup_command_routes(self):
        """Setup command submission and status routes."""
@self.app.post("/command/submit", response_model=CommandResponse)
        async def submit_command(request: CommandRequest):
            """Submit a command from AI consciousness."""
            try:
                if not self.gpt_layer:
                    raise HTTPException(status_code=503, detail="GPT command layer not available")

agent_type = AIAgentType(request.agent_type)
                domain = CommandDomain(request.domain)
                priority = CommandPriority(request.priority)

command_id = await self.gpt_layer.submit_command(
                    agent_type=agent_type,
domain=domain,
payload=request.payload,
context=request.context,
priority=priority,
parent_command_id=request.parent_command_id,


response = await self.gpt_layer.get_command_response(command_id)
                return CommandResponse(**asdict(response))
            except Exception as e:
self.logger.error(f"Error submitting command: {e}")
                raise HTTPException(status_code=500, detail=str(e))

@self.app.get("/command/{command_id}", response_model=CommandResponse)
        async def get_command_status(command_id: str):
            """Get the status of a specific command."""
            try:
                if not self.gpt_layer:
                    raise HTTPException(status_code=503, detail="GPT command layer not available")

response = await self.gpt_layer.get_command_response(command_id)
                if not response:
                    raise HTTPException(status_code=404, detail="Command not found")

                return CommandResponse(**asdict(response))
            except Exception as e:
self.logger.error(f"Error getting command status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_consciousness_routes(self):
        """Setup consciousness profile routes."""
@self.app.get("/consciousness/{agent_type}", response_model=ConsciousnessProfileResponse)
        async def get_consciousness_profile(agent_type: str):
            """Get the consciousness profile of an AI agent."""
            try:
                if not self.gpt_layer:
                    raise HTTPException(status_code=503, detail="GPT command layer not available")

agent = AIAgentType(agent_type)
                profile = await self.gpt_layer.get_consciousness_profile(agent)
                if not profile:
                    raise HTTPException(status_code=404, detail="Consciousness profile not found")

                return ConsciousnessProfileResponse(**asdict(profile))
            except Exception as e:
self.logger.error(f"Error getting consciousness profile: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_strategy_routes(self):
        """Setup strategy execution routes."""
@self.app.get("/strategy/list")
        async def list_strategies():
            """List available strategies."""
            if not self.strategy_loader:
                raise HTTPException(status_code=503, detail="Strategy loader not available")
            return {"strategies": self.strategy_loader.list_strategies()}

@self.app.post("/strategy/execute")
        async def execute_strategy(strategy_name: str, parameters: Dict[str, Any]):
            """Execute a specific strategy."""
            try:
                if not self.strategy_loader:
                    raise HTTPException(status_code=503, detail="Strategy loader not available")

result = await self.strategy_loader.execute_strategy(strategy_name, parameters)
                return {"success": True, "result": result}
            except Exception as e:
self.logger.error(f"Error executing strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_math_routes(self):
        """Setup mathematical operation routes."""
@self.app.post("/profit/allocate")
        async def allocate_profit(amount: float, risk_level: str = "medium", timeframe: str = "1h"):
            """Allocate profits using the profit cycle allocator."""
            try:
                if not self.profit_allocator:
                    raise HTTPException(status_code=503, detail="Profit allocator not available")

result = await self.profit_allocator.allocate(amount, risk_level, timeframe)
                return {"success": True, "allocation": result}
            except Exception as e:
self.logger.error(f"Error allocating profit: {e}")
                raise HTTPException(status_code=500, detail=str(e))

@self.app.post("/matrix/generate")
        async def generate_matrix(matrix_type: str, dimensions: List[int], logic_weights: Dict[str, float]):
            """Generate a matrix using the matrix allocator."""
            try:
                if not self.matrix_allocator:
                    raise HTTPException(status_code=503, detail="Matrix allocator not available")

matrix = self.matrix_allocator.generate_matrix(matrix_type, tuple(dimensions), logic_weights)
                return {"success": True, "matrix": matrix.tolist()}
            except Exception as e:
self.logger.error(f"Error generating matrix: {e}")
                raise HTTPException(status_code=500, detail=str(e))

@self.app.post("/hash/evaluate")
        async def evaluate_hash(hash_value: str, confidence_score: float, validation_data: Dict[str, Any]):
            """Evaluate a hash using the hash confidence evaluator."""
            try:
                if not self.hash_evaluator:
                    raise HTTPException(status_code=503, detail="Hash evaluator not available")

result = self.hash_evaluator.evaluate_hash(hash_value, confidence_score, validation_data)
                return {"success": True, "evaluation": result}
            except Exception as e:
self.logger.error(f"Error evaluating hash: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_websocket_routes(self):
        """Setup WebSocket routes."""
@self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket_connection(websocket)

async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connection."""
await websocket.accept()

async with self.connection_lock:
self.active_connections.append(websocket)

safe_safe_print(f"🌐 WebSocket connection established - Total: {len(self.active_connections)}")

        try:
            while True:
                # Receive message
data = await websocket.receive_text()
                message = json.loads(data)

                # Process message
response = await self._process_websocket_message(message)

                # Send response
await websocket.send_text(json.dumps(response))

        except WebSocketDisconnect:
safe_safe_print("🌐 WebSocket connection disconnected")
        except Exception as e:
error_msg = safe_format_error(e, "websocket_connection")
            safe_safe_print(f"❌ WebSocket error: {error_msg}")
        finally:
async with self.connection_lock:
                if websocket in self.active_connections:
self.active_connections.remove(websocket)

async def _process_websocket_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process WebSocket message."""
        try:
msg_type = message.get("type", "unknown")

            if msg_type == "command":
                # Handle command submission
command_data = message.get("data", {})
                agent_type = AIAgentType(command_data.get("agent_type", "gpt"))
                domain = CommandDomain(command_data.get("domain", "system"))
                payload = command_data.get("payload", {})

command_id = await self.gpt_layer.submit_command(
                    agent_type=agent_type,
domain=domain,
payload=payload,
context=command_data.get("context"),
                    priority=CommandPriority(command_data.get("priority", "medium")),


                return {
"type": "command_response",
"data": {"command_id": command_id, "status": "submitted"},
"timestamp": datetime.now().isoformat(),
                }

            elif msg_type == "status":
                # Handle status request
status = await self.gpt_layer.get_system_status()
                return {
"type": "status_response",
"data": status,
"timestamp": datetime.now().isoformat(),
                }

            elif msg_type == "subscribe":
                # Handle subscription request
subscription_type = message.get("data", {}).get("type", "all")
                return {
"type": "subscription_response",
"data": {"subscribed": True, "type": subscription_type},
"timestamp": datetime.now().isoformat(),
                }

            else:
                return {
"type": "error",
"data": {"error": f"Unknown message type: {msg_type}"},
"timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
error_msg = safe_format_error(e, "websocket_message")
            return {
"type": "error",
"data": {"error": error_msg},
"timestamp": datetime.now().isoformat(),
            }

async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections."""
        if not self.active_connections:
return

message_json = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
await connection.send_text(message_json)
            except Exception as e:
safe_safe_print(f"⚠️ WebSocket broadcast error: {safe_format_error(e, 'broadcast')}")
                disconnected.append(connection)

        # Remove disconnected connections
        if disconnected:
async with self.connection_lock:
                for connection in disconnected:
                    if connection in self.active_connections:
self.active_connections.remove(connection)

async def start_command_execution(self):
        """Start command execution loop."""
        if not self.gpt_layer:
safe_safe_print("⚠️ GPT command layer not available - skipping execution")
            return

safe_safe_print("🚀 Starting command execution loop")
        self.execution_task = asyncio.create_task(self.gpt_layer.execute_commands())

async def stop_command_execution(self):
        """Stop command execution loop."""
        if self.execution_task:
safe_safe_print("🛑 Stopping command execution loop")
            self.execution_task.cancel()
            try:
await self.execution_task
            except asyncio.CancelledError:
                pass

async def start_server(self):
        """Start the API server."""
        if not self.app:
safe_safe_print("❌ FastAPI not available - cannot start server")
            return

        # Start command execution
await self.start_command_execution()

        # Start server
config = uvicorn.Config(
            self.app,
host=self.host,
port=self.port,
log_level="info",

server = uvicorn.Server(config)

safe_safe_print(f"🌐 Starting Schwabot API Gateway on {self.host}:{self.port}")
        safe_safe_print("📚 API Documentation: http://localhost:8000/docs")
        safe_safe_print("🔌 WebSocket endpoint: ws://localhost:8000/ws")

await server.serve()

async def shutdown(self):
        """Shutdown the API gateway."""
safe_safe_print("🛑 Shutting down Schwabot API Gateway")

        # Stop command execution
await self.stop_command_execution()

        # Close WebSocket connections
async with self.connection_lock:
            for connection in self.active_connections:
                try:
await connection.close()
                except Exception:
                    pass
self.active_connections.clear()


# Global API gateway instance
api_gateway = SchwabotAPIGateway()


# Convenience functions for external access
async def start_api_gateway(host: str = "0.0.0.0", port: int = 8000):
    """Start the Schwabot API gateway."""
    global api_gateway
api_gateway = SchwabotAPIGateway(host=host, port=port)
    await api_gateway.start_server()


async def submit_command_via_api(
    agent_type: str,
domain: str,
payload: Dict[str, Any],
context: Dict[str, Any] = None,
priority: str = "medium",
) -> str:
"""Submit command via API gateway."""
    if not api_gateway or not api_gateway.gpt_layer:
        raise RuntimeError("API gateway not available")

agent_enum = AIAgentType(agent_type)
    domain_enum = CommandDomain(domain)
    priority_enum = CommandPriority(priority)

    return await api_gateway.gpt_layer.submit_command(
        agent_type=agent_enum,
domain=domain_enum,
payload=payload,
context=context,
priority=priority_enum,



async def get_system_status_via_api() -> Dict[str, Any]:
    """Get system status via API gateway."""
    if not api_gateway or not api_gateway.gpt_layer:
        raise RuntimeError("API gateway not available")

    return await api_gateway.gpt_layer.get_system_status()


# Example usage

if __name__ == "__main__":
async def test_api_gateway():
        """Test the API Gateway functionality."""
safe_safe_print("🌐 Testing API gateway...")

        # Create API gateway
gateway = SchwabotAPIGateway(host="127.0.0.1", port=8000)

        # Start command execution
await gateway.start_command_execution()

        # Submit test command
        if gateway.gpt_layer:
command_id = await submit_gpt_command(
                domain=CommandDomain.STRATEGY,
payload={
"strategy_name": "test_strategy",
"parameters": {"test": True},
"target_profit": 50.0
},
context={"api_test": True}


safe_safe_print(f"✅ Test command submitted via API: {command_id}")

        # Start server
await gateway.start_server()

    # Run test
asyncio.run(test_api_gateway())
