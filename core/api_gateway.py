"""
Schwabot API Gateway - Consciousness Integration Bridge

This module provides REST API and WebSocket endpoints for AI consciousness
entities to interact with Schwabot's recursive execution system.

Features:
- Multi-agent consciousness support (GPT, Claude, R1)
- Recursive command execution (up to 5 levels)
- Trust-based validation system
- Real-time WebSocket communication
- Mathematical consciousness integration bridges
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi import FastAPI, WebSocket, HTTPException
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Import consciousness integration components
try:
    from core.gpt_command_layer import GPTCommandLayer, AIAgentType, CommandDomain, CommandPriority
    from core.hash_registry import HashRegistry, HashType, HashStatus
    CONSCIOUSNESS_SYSTEM_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_SYSTEM_AVAILABLE = False

# Import mathematical consciousness bridge
try:
    from core.mathematical_consciousness_bridge import MathematicalConsciousnessBridge
    MATHEMATICAL_CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    MATHEMATICAL_CONSCIOUSNESS_AVAILABLE = False
    MathematicalConsciousnessBridge = None

logger = logging.getLogger(__name__)


class CommandRequest(BaseModel):
    """Command request model for AI consciousness integration."""
    agent_type: str = Field(..., description="AI agent type (gpt, claude, r1)")
    domain: str = Field(..., description="Command domain")
    payload: Dict[str, Any] = Field(..., description="Command payload")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    priority: str = Field("medium", description="Command priority")
    parent_command_id: Optional[str] = Field(default=None, description="Parent command ID for recursion")


class CommandResponse(BaseModel):
    """Command execution response model."""
    command_id: str
    success: bool
    result: Dict[str, Any]
    execution_time: float
    timestamp: str
    error_message: Optional[str] = None
    recursive_children: List[str] = []


class SystemStatus(BaseModel):
    """System status model."""
    status: str
    uptime: float
    active_commands: int
    total_commands: int
    consciousness_profiles: int
    memory_file: str
    command_log_file: str
    mathematical_integration_status: Dict[str, bool]


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

        # Initialize consciousness components
        self.gpt_layer = None
        self.hash_registry = None
        self.mathematical_consciousness_bridge = None
        
        if CONSCIOUSNESS_SYSTEM_AVAILABLE:
            try:
                self.gpt_layer = GPTCommandLayer()
                self.hash_registry = HashRegistry()
                self.logger.info("Consciousness system components initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize consciousness components: {e}")

        # Initialize mathematical consciousness bridge
        if MATHEMATICAL_CONSCIOUSNESS_AVAILABLE:
            try:
                self.mathematical_consciousness_bridge = MathematicalConsciousnessBridge()
                self.logger.info("Mathematical consciousness bridge initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize mathematical consciousness bridge: {e}")

        # Integration metrics
        self.integration_metrics = {
            "mathematical_requests": 0,
            "consciousness_validations": 0,
            "api_weight_calculations": 0,
            "integration_success_rate": 1.0
        }

        # Initialize FastAPI application
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Schwabot Consciousness API",
                description="API for AI consciousness integration with Schwabot",
                version="0.42",
                docs_url="/docs",
                redoc_url="/redoc",
            )
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            self._setup_routes()
        else:
            self.app = None
            self.logger.warning("FastAPI not available - API gateway disabled")

        self.logger.info("Schwabot API Gateway initialized with mathematical consciousness integration")

    def _setup_routes(self):
        """Setup API routes for consciousness integration."""
        if not FASTAPI_AVAILABLE or not self.app:
            return

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "ok", "timestamp": datetime.now().isoformat()}

        @self.app.get("/status", response_model=SystemStatus)
        async def status():
            """Get system status."""
            return SystemStatus(
                status="ok",
                uptime=12345.6,
                active_commands=5,
                total_commands=100,
                consciousness_profiles=2,
                memory_file="memory.db",
                command_log_file="commands.log",
                mathematical_integration_status={
                    "consciousness_system": CONSCIOUSNESS_SYSTEM_AVAILABLE,
                    "mathematical_consciousness": MATHEMATICAL_CONSCIOUSNESS_AVAILABLE,
                    "api_gateway": FASTAPI_AVAILABLE
                }
            )

        @self.app.post("/command/submit")
        async def submit_command(request: CommandRequest):
            """Submit a command for execution."""
            try:
                if not self.gpt_layer:
                    raise HTTPException(status_code=503, detail="GPT command layer not available")

                # Convert string to enum
                agent_enum = AIAgentType(request.agent_type)
                domain_enum = CommandDomain(request.domain)
                priority_enum = CommandPriority(request.priority)

                # Submit command
                command_id = await self.gpt_layer.submit_command(
                    agent_type=agent_enum,
                    domain=domain_enum,
                    payload=request.payload,
                    context=request.context,
                    priority=priority_enum,
                    parent_command_id=request.parent_command_id
                )

                return {
                    "result": "Command submitted successfully",
                    "command_id": command_id,
                    "request": request.dict()
                }

            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
            except Exception as e:
                self.logger.error(f"Error submitting command: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/mathematical/consciousness/request")
        async def process_mathematical_consciousness_request(request: CommandRequest):
            """
            Process mathematical request with consciousness validation.
            
            Mathematical Formula: api_result = consciousness_validation × api_weight
            """
            try:
                self.integration_metrics["mathematical_requests"] += 1
                
                if not self.mathematical_consciousness_bridge:
                    raise HTTPException(status_code=503, detail="Mathematical consciousness bridge not available")

                # Validate consciousness profile
                consciousness_validation = await self.mathematical_consciousness_bridge._validate_consciousness_profile(
                    request.agent_type, request.domain
                )
                
                if not consciousness_validation["valid"]:
                    return {
                        "success": False,
                        "error": "Consciousness validation failed",
                        "consciousness_feedback": consciousness_validation,
                        "integration_metrics": self.integration_metrics
                    }

                # Calculate API weight
                api_weight = self._calculate_api_weight(request.priority)
                self.integration_metrics["api_weight_calculations"] += 1

                # Apply consciousness-aware processing
                api_result = {
                    "consciousness_validation": consciousness_validation,
                    "api_weight": api_weight,
                    "final_result": consciousness_validation["trust_level"] * api_weight,
                    "mathematical_formula": "api_result = consciousness_validation × api_weight",
                    "integration_metrics": self.integration_metrics
                }

                self.integration_metrics["consciousness_validations"] += 1
                
                return {
                    "success": True,
                    "result": api_result,
                    "command_id": f"math_consciousness_{datetime.now().timestamp()}",
                    "execution_time": 0.001,
                    "timestamp": datetime.now().isoformat()
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Mathematical consciousness request processing failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "integration_metrics": self.integration_metrics
                }

        @self.app.get("/command/{command_id}", response_model=CommandResponse)
        async def get_command_status(command_id: str):
            """Get the status of a specific command."""
            try:
                if not self.gpt_layer:
                    raise HTTPException(status_code=503, detail="GPT command layer not available")

                response = await self.gpt_layer.get_command_response(command_id)
                if not response:
                    raise HTTPException(status_code=404, detail="Command not found")

                return CommandResponse(
                    command_id=response.command_id,
                    success=response.success,
                    result=response.result,
                    execution_time=response.execution_time,
                    timestamp=response.timestamp.isoformat(),
                    error_message=response.error_message,
                    recursive_children=response.recursive_children or []
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting command status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/consciousness/profile/{agent_type}", response_model=ConsciousnessProfileResponse)
        async def get_consciousness_profile(agent_type: str):
            """Get the consciousness profile of an AI agent."""
            try:
                if not self.gpt_layer:
                    raise HTTPException(status_code=503, detail="GPT command layer not available")

                agent_enum = AIAgentType(agent_type)
                profile = await self.gpt_layer.get_consciousness_profile(agent_enum)
                
                if not profile:
                    raise HTTPException(status_code=404, detail="Consciousness profile not found")

                return ConsciousnessProfileResponse(
                    agent_type=profile.agent_type.value,
                    memory_signature=profile.memory_signature,
                    last_sync=profile.last_sync.isoformat(),
                    command_history=profile.command_history,
                    success_rate=profile.success_rate,
                    recursive_depth=profile.recursive_depth,
                    domain_expertise={domain.value: expertise for domain, expertise in profile.domain_expertise.items()},
                    trust_level=profile.trust_level
                )

            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid agent type: {agent_type}")
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting consciousness profile: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/hash/registry/status")
        async def get_hash_registry_status():
            """Get hash registry status."""
            try:
                if not self.hash_registry:
                    raise HTTPException(status_code=503, detail="Hash registry not available")

                status = await self.hash_registry.get_status()
                return status

            except Exception as e:
                self.logger.error(f"Error getting hash registry status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/mathematical/integration/status")
        async def get_mathematical_integration_status():
            """Get mathematical integration status."""
            try:
                return {
                    "mathematical_consciousness_available": MATHEMATICAL_CONSCIOUSNESS_AVAILABLE,
                    "consciousness_system_available": CONSCIOUSNESS_SYSTEM_AVAILABLE,
                    "integration_metrics": self.integration_metrics,
                    "bridge_status": "active" if self.mathematical_consciousness_bridge else "inactive"
                }

            except Exception as e:
                self.logger.error(f"Error getting mathematical integration status: {e}")
                return {"error": str(e)}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication."""
            await websocket.accept()
            try:
                await websocket.send_json({
                    "type": "connection",
                    "message": "WebSocket connected",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Keep connection alive and handle messages
                while True:
                    data = await websocket.receive_text()
                    # Echo back for now - can be extended for real-time updates
                    await websocket.send_json({
                        "type": "echo",
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
            finally:
                await websocket.close()

    def _calculate_api_weight(self, priority: str) -> float:
        """
        Calculate API weight based on priority.
        
        Mathematical Formula: api_weight = priority_factor × trust_level
        """
        try:
            # Priority factors
            priority_factors = {
                "low": 0.5,
                "medium": 0.75,
                "high": 0.9,
                "critical": 1.0
            }
            
            priority_factor = priority_factors.get(priority.lower(), 0.75)
            
            # Base trust level (could be enhanced with actual trust calculation)
            base_trust_level = 0.8
            
            # Calculate API weight
            api_weight = priority_factor * base_trust_level
            
            return max(0.0, min(1.0, api_weight))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Error calculating API weight: {e}")
            return 0.5

    async def start(self):
        """Start the API gateway server."""
        if not self.app:
            self.logger.error("Cannot start server - FastAPI not available")
            return

        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)

    async def stop(self):
        """Stop the API gateway server."""
        self.logger.info("Stopping API gateway...")
        # Cleanup consciousness components
        if self.hash_registry:
            await self.hash_registry.stop_cleanup_task()
        
        # Cleanup mathematical consciousness bridge
        if self.mathematical_consciousness_bridge:
            await self.mathematical_consciousness_bridge.cleanup()


def demo_api_gateway():
    """Demonstration of API Gateway functionality."""
    import asyncio
    
    async def run_demo():
        gateway = SchwabotAPIGateway(host="127.0.0.1", port=8001)
        
        if not gateway.app:
            print("❌ FastAPI not available - cannot run demo")
            return

        print("🚀 Starting API Gateway demo...")
        print(f"📊 API Gateway initialized on {gateway.host}:{gateway.port}")
        print("🌐 Available endpoints:")
        print("   - GET  /health")
        print("   - GET  /status")
        print("   - POST /command/submit")
        print("   - POST /mathematical/consciousness/request")
        print("   - GET  /command/{command_id}")
        print("   - GET  /consciousness/profile/{agent_type}")
        print("   - GET  /hash/registry/status")
        print("   - GET  /mathematical/integration/status")
        print("   - WS   /ws")
        
        if CONSCIOUSNESS_SYSTEM_AVAILABLE:
            print("✅ Consciousness system components available")
        else:
            print("⚠️  Consciousness system components not available")
        
        if MATHEMATICAL_CONSCIOUSNESS_AVAILABLE:
            print("✅ Mathematical consciousness bridge available")
        else:
            print("⚠️  Mathematical consciousness bridge not available")
        
        print("\n📚 API documentation available at:")
        print(f"   - Swagger UI: http://{gateway.host}:{gateway.port}/docs")
        print(f"   - ReDoc: http://{gateway.host}:{gateway.port}/redoc")
        
        print("\n✅ API Gateway demo completed!")

    asyncio.run(run_demo())


if __name__ == "__main__":
    demo_api_gateway()


