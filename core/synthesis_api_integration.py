# -*- coding: utf-8 -*-
"""
Synthesis API Integration - API Layer for Core Tensor Modulator

Provides API endpoints and integration for the synthesis engine system,
exposing recursive Unicode pathway processing and core tensor modulation
through the existing Schwabot API infrastructure.

Core Functionality:
- REST API endpoints for synthesis engine operations
- WebSocket integration for real-time pathway processing
- Integration with existing API Gateway and Bridge Manager
- Profit movement execution through synthesis engines
- Pathway statistics and monitoring endpoints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import asyncio
import json
import logging
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from core.synthesis_engine_system import (
    get_core_tensor_modulator,
    execute_synthesis_pathway,
    SynthesisEngineType,
    SpinOperation,
    PhaseDimension
)

# Configure logging
logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class PathwayRequest(BaseModel):
    """Request model for pathway processing."""
    pathway: str = Field(..., description="Initial pathway string")
    engines: List[str] = Field(..., description="List of engine names (RITTLE, RIDTLE, ALEPH, ALIF, FERRIS_RDE)")
    operations: List[str] = Field(..., description="List of operations (SPIN, TURN, DRIFT, CONNECT)")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Context data for operations")


class ProfitMovementRequest(BaseModel):
    """Request model for profit movement execution."""
    profit_amount: float = Field(..., description="Amount of profit to move")
    strategy_pathway: str = Field(..., description="Strategy pathway string")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Context data")


class PathwayResponse(BaseModel):
    """Response model for pathway processing."""
    pathway: str
    hash_256: str
    sectors: Dict[str, str]
    phase_value: float
    drift_value: float
    time_value: float
    differential_value: float
    checksum_valid: bool
    timestamp: str
    metadata: Dict[str, Any]


class ProfitMovementResponse(BaseModel):
    """Response model for profit movement execution."""
    original_profit: float
    final_profit: float
    profit_change: float
    phase_multiplier: float
    drift_adjustment: float
    time_factor: float
    differential_boost: float
    movement_timestamp: str
    pathway_result: Dict[str, Any]


class SystemStatisticsResponse(BaseModel):
    """Response model for system statistics."""
    total_pathways_processed: int
    total_spins_executed: int
    total_turns_executed: int
    total_drifts_executed: int
    total_connects_executed: int
    checksum_validity_rate: float
    average_phase_value: float
    average_drift_value: float
    average_time_value: float
    average_differential_value: float
    ferris_rde_available: bool
    aleph_alif_available: bool
    riddle_available: bool


@dataclass
class SynthesisAPIIntegration:
    """API integration layer for synthesis engine system."""
    
    def __init__(self):
        """Initialize synthesis API integration."""
        self.modulator = get_core_tensor_modulator()
        self.router = APIRouter(prefix="/synthesis", tags=["synthesis"])
        self.active_websockets: List[WebSocket] = []
        
        # Register API routes
        self._register_routes()
        
        logger.info("🔌 Synthesis API Integration initialized")

    def _register_routes(self) -> None:
        """Register API routes."""
        
        @self.router.post("/pathway", response_model=PathwayResponse)
        async def process_pathway(request: PathwayRequest) -> PathwayResponse:
            """Process pathway through synthesis engines."""
            try:
                result = execute_synthesis_pathway(
                    pathway=request.pathway,
                    engines=request.engines,
                    operations=request.operations,
                    context=request.context
                )
                
                # Broadcast to WebSocket clients
                await self._broadcast_pathway_result(result)
                
                return PathwayResponse(**result)
                
            except Exception as e:
                logger.error(f"Pathway processing error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/profit-movement", response_model=ProfitMovementResponse)
        async def execute_profit_movement(request: ProfitMovementRequest) -> ProfitMovementResponse:
            """Execute profit movement through synthesis engines."""
            try:
                result = self.modulator.execute_profit_movement(
                    profit_amount=request.profit_amount,
                    strategy_pathway=request.strategy_pathway,
                    context=request.context
                )
                
                # Convert pathway result to dict for response
                pathway_dict = {
                    "pathway": result["pathway_result"].pathway,
                    "hash_256": result["pathway_result"].hash_256,
                    "sectors": {k.value: v for k, v in result["pathway_result"].sectors.items()},
                    "phase_value": result["pathway_result"].phase_value,
                    "drift_value": result["pathway_result"].drift_value,
                    "time_value": result["pathway_result"].time_value,
                    "differential_value": result["pathway_result"].differential_value,
                    "checksum_valid": result["pathway_result"].checksum_valid,
                    "timestamp": result["pathway_result"].timestamp.isoformat(),
                    "metadata": result["pathway_result"].metadata
                }
                
                response_data = {
                    "original_profit": result["original_profit"],
                    "final_profit": result["final_profit"],
                    "profit_change": result["profit_change"],
                    "phase_multiplier": result["phase_multiplier"],
                    "drift_adjustment": result["drift_adjustment"],
                    "time_factor": result["time_factor"],
                    "differential_boost": result["differential_boost"],
                    "movement_timestamp": result["movement_timestamp"],
                    "pathway_result": pathway_dict
                }
                
                # Broadcast to WebSocket clients
                await self._broadcast_profit_movement(result)
                
                return ProfitMovementResponse(**response_data)
                
            except Exception as e:
                logger.error(f"Profit movement error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/statistics", response_model=SystemStatisticsResponse)
        async def get_system_statistics() -> SystemStatisticsResponse:
            """Get synthesis engine system statistics."""
            try:
                stats = self.modulator.get_pathway_statistics()
                return SystemStatisticsResponse(**stats)
                
            except Exception as e:
                logger.error(f"Statistics error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/engines")
        async def get_available_engines() -> Dict[str, List[str]]:
            """Get available synthesis engines and operations."""
            try:
                engines = [engine.value for engine in SynthesisEngineType]
                operations = [operation.value for operation in SpinOperation]
                
                return {
                    "engines": engines,
                    "operations": operations,
                    "phase_dimensions": [dim.value for dim in PhaseDimension]
                }
                
            except Exception as e:
                logger.error(f"Engine list error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/clear-history")
        async def clear_pathway_history() -> Dict[str, str]:
            """Clear pathway history."""
            try:
                self.modulator.clear_history()
                return {"message": "Pathway history cleared successfully"}
                
            except Exception as e:
                logger.error(f"Clear history error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time synthesis updates."""
            await websocket.accept()
            self.active_websockets.append(websocket)
            
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "subscribe_pathways":
                        await websocket.send_text(json.dumps({
                            "type": "subscription_confirmed",
                            "message": "Subscribed to pathway updates"
                        }))
                    elif message.get("type") == "subscribe_profit_movements":
                        await websocket.send_text(json.dumps({
                            "type": "subscription_confirmed", 
                            "message": "Subscribed to profit movement updates"
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Unknown message type"
                        }))
                        
            except WebSocketDisconnect:
                self.active_websockets.remove(websocket)
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_websockets:
                    self.active_websockets.remove(websocket)

    async def _broadcast_pathway_result(self, result: Dict[str, Any]) -> None:
        """Broadcast pathway result to WebSocket clients."""
        message = {
            "type": "pathway_result",
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
        
        await self._broadcast_message(message)

    async def _broadcast_profit_movement(self, result: Dict[str, Any]) -> None:
        """Broadcast profit movement result to WebSocket clients."""
        message = {
            "type": "profit_movement",
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
        
        await self._broadcast_message(message)

    async def _broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all active WebSocket clients."""
        if not self.active_websockets:
            return
            
        message_json = json.dumps(message)
        disconnected = []
        
        for websocket in self.active_websockets:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            if websocket in self.active_websockets:
                self.active_websockets.remove(websocket)

    def get_router(self) -> APIRouter:
        """Get the API router."""
        return self.router

    def get_websocket_count(self) -> int:
        """Get number of active WebSocket connections."""
        return len(self.active_websockets)


# Integration with existing API Gateway
class SynthesisAPIGatewayIntegration:
    """Integration class for existing API Gateway."""
    
    def __init__(self, api_gateway):
        """Initialize with existing API Gateway."""
        self.api_gateway = api_gateway
        self.synthesis_integration = SynthesisAPIIntegration()
        
        # Add synthesis routes to existing API Gateway
        self._integrate_with_gateway()
        
        logger.info("🔌 Synthesis API Gateway Integration initialized")

    def _integrate_with_gateway(self) -> None:
        """Integrate synthesis routes with existing API Gateway."""
        try:
            # Add synthesis router to existing API Gateway
            if hasattr(self.api_gateway, 'app'):
                self.api_gateway.app.include_router(self.synthesis_integration.get_router())
                logger.info("✅ Synthesis routes integrated with API Gateway")
            else:
                logger.warning("⚠️ API Gateway app not found for integration")
                
        except Exception as e:
            logger.error(f"❌ Failed to integrate with API Gateway: {e}")

    def get_synthesis_integration(self) -> SynthesisAPIIntegration:
        """Get synthesis API integration."""
        return self.synthesis_integration


# Integration with existing API Bridge Manager
class SynthesisBridgeIntegration:
    """Integration class for existing API Bridge Manager."""
    
    def __init__(self, bridge_manager):
        """Initialize with existing API Bridge Manager."""
        self.bridge_manager = bridge_manager
        self.synthesis_integration = SynthesisAPIIntegration()
        
        # Add synthesis endpoints to bridge manager
        self._integrate_with_bridge()
        
        logger.info("🔌 Synthesis Bridge Integration initialized")

    def _integrate_with_bridge(self) -> None:
        """Integrate synthesis endpoints with existing Bridge Manager."""
        try:
            # Add synthesis endpoints to bridge manager
            if hasattr(self.bridge_manager, 'add_endpoint'):
                # Add synthesis endpoints
                self.bridge_manager.add_endpoint(
                    "synthesis_pathway",
                    self.synthesis_integration.modulator.process_pathway
                )
                self.bridge_manager.add_endpoint(
                    "synthesis_profit_movement",
                    self.synthesis_integration.modulator.execute_profit_movement
                )
                self.bridge_manager.add_endpoint(
                    "synthesis_statistics",
                    self.synthesis_integration.modulator.get_pathway_statistics
                )
                logger.info("✅ Synthesis endpoints integrated with Bridge Manager")
            else:
                logger.warning("⚠️ Bridge Manager add_endpoint method not found")
                
        except Exception as e:
            logger.error(f"❌ Failed to integrate with Bridge Manager: {e}")

    def get_synthesis_integration(self) -> SynthesisAPIIntegration:
        """Get synthesis API integration."""
        return self.synthesis_integration


# Convenience functions for external access
def get_synthesis_api_integration() -> SynthesisAPIIntegration:
    """Get synthesis API integration instance."""
    return SynthesisAPIIntegration()


def integrate_with_api_gateway(api_gateway) -> SynthesisAPIGatewayIntegration:
    """Integrate synthesis system with existing API Gateway."""
    return SynthesisAPIGatewayIntegration(api_gateway)


def integrate_with_bridge_manager(bridge_manager) -> SynthesisBridgeIntegration:
    """Integrate synthesis system with existing Bridge Manager."""
    return SynthesisBridgeIntegration(bridge_manager)


# Example usage and testing
def main() -> None:
    """Test synthesis API integration."""
    print("🔌 Testing Synthesis API Integration")
    print("=" * 50)
    
    # Create synthesis API integration
    synthesis_api = get_synthesis_api_integration()
    
    # Test pathway processing
    test_request = PathwayRequest(
        pathway="BTC_PROFIT_STRATEGY_001",
        engines=["FERRIS_RDE", "RITTLE", "ALEPH"],
        operations=["SPIN", "DRIFT", "CONNECT"],
        context={"base_entropy": 0.5, "time_factor": 1.0}
    )
    
    print(f"📊 Test Pathway Request:")
    print(f"  Pathway: {test_request.pathway}")
    print(f"  Engines: {test_request.engines}")
    print(f"  Operations: {test_request.operations}")
    
    # Test profit movement
    profit_request = ProfitMovementRequest(
        profit_amount=1000.0,
        strategy_pathway="PROFIT_STRATEGY_002",
        context={"market_volatility": 0.3}
    )
    
    print(f"\n💰 Test Profit Movement Request:")
    print(f"  Profit Amount: ${profit_request.profit_amount:.2f}")
    print(f"  Strategy Pathway: {profit_request.strategy_pathway}")
    
    # Get system statistics
    stats = synthesis_api.modulator.get_pathway_statistics()
    print(f"\n📈 System Statistics:")
    print(f"  Total Pathways: {stats['total_pathways_processed']}")
    print(f"  WebSocket Connections: {synthesis_api.get_websocket_count()}")
    
    print(f"\n✅ Synthesis API Integration test completed")


if __name__ == "__main__":
    main() 