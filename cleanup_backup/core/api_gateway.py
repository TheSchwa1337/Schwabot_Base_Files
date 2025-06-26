from utils.safe_print import safe_print, info, warn, error, success, debug
#!/usr/bin/env python3
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
        def __init__(self, *args, **kwargs): pass
        def add_middleware(self, *args, **kwargs): pass
        def get(self, *args, **kwargs): pass
        def post(self, *args, **kwargs): pass
        def websocket(self, *args, **kwargs): pass

    class WebSocket:
        def __init__(self): pass
        async def send_text(self, text): pass
        async def receive_text(self): pass

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
    )
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
    from core.gpt_command_layer import (
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
    )
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


class CommandResponse(BaseModel):
    """Command response model."""
    command_id: str
    success: bool
    result: Dict[str, Any]
    execution_time: float
    timestamp: str
    error_message: Optional[str] = None


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
                version="0.42f",
                docs_url="/docs",
                redoc_url="/redoc",
            )

            # Add CORS middleware
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

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

        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        # System status
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
                )

        # Submit command
        @self.app.post("/command/submit", response_model=CommandResponse)
        async def submit_command(request: CommandRequest):
            """Submit a command from AI consciousness."""
            try:
                if not self.gpt_layer:
                    raise HTTPException(status_code=503, detail="GPT command layer not available")

                # Convert string to enum
                agent_type = AIAgentType(request.agent_type)
                domain = CommandDomain(request.domain)
                priority = CommandPriority(request.priority)

                # Submit command
                command_id = await self.gpt_layer.submit_command(
                    agent_type=agent_type,
                    domain=domain,
                    payload=request.payload,
                    context=request.context,
                    priority=priority,
                    parent_command_id=request.parent_command_id,
                )

                # Get response
                response = await self.gpt_layer.get_command_status(command_id)
                if response:
                    return CommandResponse(
                        command_id=response.command_id,
                        success=response.success,
                        result=response.result,
                        execution_time=response.execution_time,
                        timestamp=response.timestamp.isoformat(),
                        error_message=response.error_message,
                    )
                else:
                    return CommandResponse(
                        command_id=command_id,
                        success=True,
                        result={"status": "queued"},
                        execution_time=0.0,
                        timestamp=datetime.now().isoformat(),
                    )

            except Exception as e:
                error_msg = safe_format_error(e, "submit_command")
                safe_safe_print(f"❌ Command submission error: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

        # Get command status
        @self.app.get("/command/{command_id}", response_model=CommandResponse)
        async def get_command_status(command_id: str):
            """Get status of a specific command."""
            try:
                if not self.gpt_layer:
                    raise HTTPException(status_code=503, detail="GPT command layer not available")

                response = await self.gpt_layer.get_command_status(command_id)
                if response:
                    return CommandResponse(
                        command_id=response.command_id,
                        success=response.success,
                        result=response.result,
                        execution_time=response.execution_time,
                        timestamp=response.timestamp.isoformat(),
                        error_message=response.error_message,
                    )
                else:
                    raise HTTPException(status_code=404, detail="Command not found")

            except HTTPException:
                raise
            except Exception as e:
                error_msg = safe_format_error(e, "get_command_status")
                safe_safe_print(f"❌ Command status error: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

        # Get consciousness profile
        @self.app.get("/consciousness/{agent_type}", response_model=ConsciousnessProfileResponse)
        async def get_consciousness_profile(agent_type: str):
            """Get consciousness profile for specific agent."""
            try:
                if not self.gpt_layer:
                    raise HTTPException(status_code=503, detail="GPT command layer not available")

                agent_enum = AIAgentType(agent_type)
                profile = await self.gpt_layer.get_consciousness_profile(agent_enum)

                if profile:
                    return ConsciousnessProfileResponse(
                        agent_type=profile.agent_type.value,
                        memory_signature=profile.memory_signature,
                        last_sync=profile.last_sync.isoformat(),
                        command_history=profile.command_history,
                        success_rate=profile.success_rate,
                        recursive_depth=profile.recursive_depth,
                        domain_expertise={domain.value: expertise for domain,
                                          expertise in profile.domain_expertise.items()},
                        trust_level=profile.trust_level,
                    )
                else:
                    raise HTTPException(status_code=404, detail="Consciousness profile not found")

            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid agent type: {agent_type}")
            except HTTPException:
                raise
            except Exception as e:
                error_msg = safe_format_error(e, "get_consciousness_profile")
                safe_safe_print(f"❌ Consciousness profile error: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

        # Strategy endpoints
        @self.app.get("/strategy/list")
        async def list_strategies():
            """List available strategies."""
            try:
                if not self.strategy_loader:
                    raise HTTPException(status_code=503, detail="Strategy loader not available")

                strategies = self.strategy_loader.list_strategies()
                return {"strategies": strategies}

            except Exception as e:
                error_msg = safe_format_error(e, "list_strategies")
                safe_safe_print(f"❌ Strategy list error: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

        @self.app.post("/strategy/execute")
        async def execute_strategy(strategy_name: str, parameters: Dict[str, Any]):
            """Execute a strategy."""
            try:
                if not self.strategy_loader:
                    raise HTTPException(status_code=503, detail="Strategy loader not available")

                strategy = self.strategy_loader.load_strategy(strategy_name)
                if strategy:
                    result = await strategy.execute(parameters)
                    return {"strategy_executed": strategy_name, "result": result}
                else:
                    raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")

            except HTTPException:
                raise
            except Exception as e:
                error_msg = safe_format_error(e, "execute_strategy")
                safe_safe_print(f"❌ Strategy execution error: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

        # Profit endpoints
        @self.app.post("/profit/allocate")
        async def allocate_profit(amount: float, risk_level: str = "medium", timeframe: str = "1h"):
            """Allocate profit cycle."""
            try:
                if not self.profit_allocator:
                    raise HTTPException(status_code=503, detail="Profit allocator not available")

                result = await self.profit_allocator.allocate_cycle(
                    amount=amount,
                    risk_level=risk_level,
                    timeframe=timeframe
                )

                return {"profit_allocated": amount, "result": result}

            except Exception as e:
                error_msg = safe_format_error(e, "allocate_profit")
                safe_safe_print(f"❌ Profit allocation error: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

        # Matrix endpoints
        @self.app.post("/matrix/generate")
        async def generate_matrix(matrix_type: str, dimensions: List[int], logic_weights: Dict[str, float]):
            """Generate a new matrix."""
            try:
                if not self.matrix_allocator:
                    raise HTTPException(status_code=503, detail="Matrix allocator not available")

                matrix = await self.matrix_allocator.generate_matrix(
                    matrix_type=matrix_type,
                    dimensions=dimensions,
                    logic_weights=logic_weights
                )

                return {"matrix_generated": matrix_type, "matrix": matrix}

            except Exception as e:
                error_msg = safe_format_error(e, "generate_matrix")
                safe_safe_print(f"❌ Matrix generation error: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

        # Hash endpoints
        @self.app.post("/hash/evaluate")
        async def evaluate_hash(hash_value: str, confidence_score: float, validation_data: Dict[str, Any]):
            """Evaluate a hash."""
            try:
                if not self.hash_evaluator:
                    raise HTTPException(status_code=503, detail="Hash evaluator not available")

                evaluation = await self.hash_evaluator.evaluate_hash(
                    hash_value=hash_value,
                    confidence_score=confidence_score,
                    validation_data=validation_data
                )

                return {"hash_evaluated": hash_value, "evaluation": evaluation}

            except Exception as e:
                error_msg = safe_format_error(e, "evaluate_hash")
                safe_safe_print(f"❌ Hash evaluation error: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication."""
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
                )

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
        )
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
    )


async def get_system_status_via_api() -> Dict[str, Any]:
    """Get system status via API gateway."""
    if not api_gateway or not api_gateway.gpt_layer:
        raise RuntimeError("API gateway not available")

    return await api_gateway.gpt_layer.get_system_status()


# Example usage
if __name__ == "__main__":
    async def test_api_gateway():
        """Test API gateway functionality."""
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
            )

            safe_safe_print(f"✅ Test command submitted via API: {command_id}")

        # Start server
        await gateway.start_server()

    # Run test
    asyncio.run(test_api_gateway())
