#!/usr/bin/env python3
"""
Simplified Schwabot API
======================

Clean, simplified API that addresses user concerns:
- JSON-based configuration instead of complex YAML
- Demo mode functionality for testing strategies without live trading
- Simplified visual interface
- Auto-generated defaults that work out of the box
- Prevention of minor errors becoming major errors

Key Features:
- Single unified API endpoint
- Built-in demo mode with synthetic data
- Simple JSON configuration
- Real-time WebSocket streaming
- Clean error handling
- Automatic fallback mechanisms
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import secrets
import hashlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our unified math system
try:
    from schwabot_unified_math_v2 import (
        UnifiedQuantumTradingController,
        calculate_btc_processor_metrics,
        SustainmentMetrics
    )
    MATH_SYSTEM_AVAILABLE = True
except ImportError:
    MATH_SYSTEM_AVAILABLE = False
    print("⚠️ Unified math system not available - running in mock mode")

logger = logging.getLogger(__name__)

# ===== SIMPLIFIED CONFIGURATION =====

@dataclass
class SimplifiedConfig:
    """Simplified JSON-based configuration"""
    # Trading settings
    demo_mode: bool = True
    live_trading_enabled: bool = False
    position_size_limit: float = 0.1
    
    # API settings
    api_port: int = 8000
    websocket_update_interval: float = 0.5  # seconds
    
    # Risk management
    max_drawdown: float = 0.05  # 5%
    stop_loss: float = 0.02     # 2%
    
    # Demo settings
    demo_speed_multiplier: float = 1.0  # 1x = real-time, 10x = 10x speed
    synthetic_data_enabled: bool = True
    
    # System settings
    sustainment_threshold: float = 0.65
    confidence_threshold: float = 0.70
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimplifiedConfig':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def get_default(cls) -> 'SimplifiedConfig':
        """Get default configuration that works out of the box"""
        return cls()

# ===== API MODELS =====

class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any]

class TradingRequest(BaseModel):
    action: str  # 'start', 'stop', 'pause'
    demo_mode: bool = True

class DemoRequest(BaseModel):
    scenario: str = "trending_market"  # trending_market, volatile_market, crash_test
    speed_multiplier: float = 1.0
    duration_minutes: float = 10.0

# ===== SIMPLIFIED API =====

class SimplifiedAPI:
    """Simplified Schwabot API with demo functionality"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize simplified API"""
        self.config_path = config_path or str(Path.home() / ".schwabot" / "simple_config.json")
        self.config = self._load_config()
        
        # Initialize trading controller
        if MATH_SYSTEM_AVAILABLE:
            self.trading_controller = UnifiedQuantumTradingController()
        else:
            self.trading_controller = None
        
        # Demo state
        self.demo_active = False
        self.demo_data = []
        self.demo_start_time = None
        
        # System state
        self.trading_active = False
        self.connected_clients = set()
        self.system_stats = {
            'uptime_seconds': 0,
            'trades_executed': 0,
            'total_profit': 0.0,
            'current_drawdown': 0.0,
            'errors_count': 0,
            'last_error': None
        }
        
        # Create FastAPI app
        self.app = self._create_app()
        
        logger.info("SimplifiedAPI initialized")
    
    def _load_config(self) -> SimplifiedConfig:
        """Load configuration from JSON file"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                return SimplifiedConfig.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        # Create default config and save it
        config = SimplifiedConfig.get_default()
        self._save_config(config)
        return config
    
    def _save_config(self, config: SimplifiedConfig) -> None:
        """Save configuration to JSON file"""
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Simplified Schwabot API",
            description="Clean, simplified API for Schwabot trading system",
            version="1.0.0"
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes"""
        
        @app.get("/")
        async def root():
            """API root endpoint"""
            return {
                "name": "Simplified Schwabot API",
                "version": "1.0.0",
                "status": "running",
                "demo_mode": self.config.demo_mode,
                "endpoints": {
                    "config": "/config",
                    "status": "/status", 
                    "trading": "/trading",
                    "demo": "/demo",
                    "websocket": "/ws"
                }
            }
        
        @app.get("/config")
        async def get_config():
            """Get current configuration"""
            return {
                "config": self.config.to_dict(),
                "config_path": self.config_path
            }
        
        @app.post("/config")
        async def update_config(request: ConfigUpdateRequest):
            """Update configuration"""
            try:
                # Validate and update config
                new_config = SimplifiedConfig.from_dict(request.config)
                self.config = new_config
                self._save_config(new_config)
                
                return {
                    "success": True,
                    "message": "Configuration updated successfully",
                    "config": new_config.to_dict()
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")
        
        @app.get("/status")
        async def get_status():
            """Get system status"""
            current_time = datetime.now(timezone.utc)
            
            # Get trading metrics if available
            trading_metrics = {}
            if self.trading_controller and MATH_SYSTEM_AVAILABLE:
                try:
                    # Mock market state for status
                    market_state = {
                        'latencies': [25.0],
                        'operations': [150],
                        'profit_deltas': [0.02],
                        'resource_costs': [1.0],
                        'utility_values': [0.8],
                        'predictions': [50000],
                        'subsystem_scores': [0.8, 0.75, 0.9, 0.85],
                        'system_states': [0.8],
                        'uptime_ratio': 0.99,
                        'iteration_states': [[0.8, 0.7]]
                    }
                    
                    result = self.trading_controller.evaluate_trade_opportunity(
                        price=50000.0, volume=1500.0, market_state=market_state
                    )
                    
                    trading_metrics = {
                        'should_execute': result['should_execute'],
                        'confidence': result['confidence'],
                        'sustainment_index': result['sustainment_metrics']['sustainment_index'],
                        'position_size': result['position_size']
                    }
                except Exception as e:
                    logger.error(f"Error getting trading metrics: {e}")
            
            return {
                "timestamp": current_time.isoformat(),
                "system": {
                    "status": "running",
                    "demo_mode": self.config.demo_mode,
                    "trading_active": self.trading_active,
                    "demo_active": self.demo_active,
                    "connected_clients": len(self.connected_clients),
                    "math_system_available": MATH_SYSTEM_AVAILABLE
                },
                "stats": self.system_stats,
                "trading_metrics": trading_metrics,
                "config": self.config.to_dict()
            }
        
        @app.post("/trading")
        async def control_trading(request: TradingRequest):
            """Control trading operations"""
            try:
                if request.action == "start":
                    if not request.demo_mode and not self.config.live_trading_enabled:
                        raise HTTPException(
                            status_code=403, 
                            detail="Live trading is disabled. Enable in configuration or use demo mode."
                        )
                    
                    self.trading_active = True
                    mode = "demo" if request.demo_mode else "live"
                    
                    return {
                        "success": True,
                        "message": f"Trading started in {mode} mode",
                        "trading_active": True,
                        "demo_mode": request.demo_mode
                    }
                
                elif request.action == "stop":
                    self.trading_active = False
                    return {
                        "success": True,
                        "message": "Trading stopped",
                        "trading_active": False
                    }
                
                elif request.action == "pause":
                    # Pause functionality
                    return {
                        "success": True,
                        "message": "Trading paused",
                        "trading_active": self.trading_active
                    }
                
                else:
                    raise HTTPException(status_code=400, detail="Invalid action. Use 'start', 'stop', or 'pause'")
                    
            except Exception as e:
                self.system_stats['errors_count'] += 1
                self.system_stats['last_error'] = str(e)
                raise HTTPException(status_code=500, detail=f"Trading control error: {e}")
        
        @app.post("/demo")
        async def start_demo(request: DemoRequest):
            """Start demo trading session"""
            try:
                self.demo_active = True
                self.demo_start_time = datetime.now(timezone.utc)
                self.demo_data = []
                
                # Generate demo scenario data
                demo_data = self._generate_demo_scenario(
                    request.scenario, 
                    request.duration_minutes,
                    request.speed_multiplier
                )
                
                return {
                    "success": True,
                    "message": f"Demo started: {request.scenario}",
                    "scenario": request.scenario,
                    "duration_minutes": request.duration_minutes,
                    "speed_multiplier": request.speed_multiplier,
                    "data_points": len(demo_data),
                    "demo_active": True
                }
                
            except Exception as e:
                self.system_stats['errors_count'] += 1
                self.system_stats['last_error'] = str(e)
                raise HTTPException(status_code=500, detail=f"Demo start error: {e}")
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data"""
            await websocket.accept()
            self.connected_clients.add(websocket)
            logger.info(f"Client connected: {websocket.client}")
            
            try:
                while True:
                    # Get current data
                    data = await self._get_realtime_data()
                    
                    # Send to client
                    await websocket.send_text(json.dumps(data))
                    
                    # Wait based on config
                    await asyncio.sleep(self.config.websocket_update_interval)
                    
            except WebSocketDisconnect:
                self.connected_clients.discard(websocket)
                logger.info("Client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connected_clients.discard(websocket)
    
    def _generate_demo_scenario(self, scenario: str, duration_minutes: float, speed_multiplier: float) -> List[Dict]:
        """Generate synthetic data for demo scenarios"""
        data_points = int(duration_minutes * 60 / self.config.websocket_update_interval)
        demo_data = []
        
        base_price = 50000.0
        base_volume = 1500.0
        
        for i in range(data_points):
            if scenario == "trending_market":
                # Upward trend with noise
                price_change = 0.001 + 0.0005 * (i / data_points)  # Gradual increase
                volatility = 0.01
            elif scenario == "volatile_market":
                # High volatility
                price_change = 0.0
                volatility = 0.03
            elif scenario == "crash_test":
                # Market crash scenario
                if i < data_points * 0.7:
                    price_change = 0.001  # Initial rise
                    volatility = 0.01
                else:
                    price_change = -0.005  # Sharp decline
                    volatility = 0.05
            else:
                # Default stable market
                price_change = 0.0
                volatility = 0.005
            
            # Add random noise
            import random
            noise = random.gauss(0, volatility)
            price = base_price * (1 + price_change + noise)
            volume = base_volume * (1 + random.gauss(0, 0.1))
            
            demo_data.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'price': price,
                'volume': volume,
                'scenario': scenario
            })
        
        self.demo_data = demo_data
        return demo_data
    
    async def _get_realtime_data(self) -> Dict[str, Any]:
        """Get real-time data for WebSocket streaming"""
        current_time = datetime.now(timezone.utc)
        
        # Demo mode data
        if self.demo_active and self.demo_data:
            # Get next demo data point
            elapsed = (current_time - self.demo_start_time).total_seconds()
            index = int(elapsed / self.config.websocket_update_interval) % len(self.demo_data)
            demo_point = self.demo_data[index]
        else:
            demo_point = None
        
        # Trading metrics
        trading_data = {}
        if self.trading_controller and MATH_SYSTEM_AVAILABLE:
            try:
                # Use demo data if available, otherwise mock data
                price = demo_point['price'] if demo_point else 50000.0
                volume = demo_point['volume'] if demo_point else 1500.0
                
                # Mock market state
                market_state = {
                    'latencies': [25.0],
                    'operations': [150],
                    'profit_deltas': [0.02],
                    'resource_costs': [1.0],
                    'utility_values': [0.8],
                    'predictions': [price],
                    'subsystem_scores': [0.8, 0.75, 0.9, 0.85],
                    'system_states': [0.8],
                    'uptime_ratio': 0.99,
                    'iteration_states': [[0.8, 0.7]]
                }
                
                result = self.trading_controller.evaluate_trade_opportunity(
                    price=price, volume=volume, market_state=market_state
                )
                
                trading_data = {
                    'price': price,
                    'volume': volume,
                    'should_execute': result['should_execute'],
                    'confidence': result['confidence'],
                    'sustainment_index': result['sustainment_metrics']['sustainment_index'],
                    'position_size': result['position_size'],
                    'hurst_exponent': result['fractal_metrics']['hurst_exponent'],
                    'hausdorff_dimension': result['fractal_metrics']['hausdorff_dimension']
                }
            except Exception as e:
                logger.error(f"Error calculating trading data: {e}")
                trading_data = {'error': str(e)}
        
        return {
            'timestamp': current_time.isoformat(),
            'system': {
                'demo_active': self.demo_active,
                'trading_active': self.trading_active,
                'connected_clients': len(self.connected_clients)
            },
            'demo_data': demo_point,
            'trading_data': trading_data,
            'stats': self.system_stats
        }
    
    def run(self, host: str = "127.0.0.1", port: Optional[int] = None):
        """Run the simplified API server"""
        if port is None:
            port = self.config.api_port
        
        logger.info(f"Starting Simplified Schwabot API on {host}:{port}")
        logger.info(f"Demo mode: {self.config.demo_mode}")
        logger.info(f"Live trading: {self.config.live_trading_enabled}")
        
        uvicorn.run(self.app, host=host, port=port)

# ===== CONVENIENCE FUNCTIONS =====

def create_simplified_api(config_path: Optional[str] = None) -> SimplifiedAPI:
    """Create simplified API instance"""
    return SimplifiedAPI(config_path=config_path)

def run_demo_api(scenario: str = "trending_market", port: int = 8000):
    """Quick function to run demo API"""
    api = create_simplified_api()
    api.config.demo_mode = True
    api.config.api_port = port
    api.run(port=port)

if __name__ == "__main__":
    # Run in demo mode by default
    print("🚀 Starting Simplified Schwabot API in Demo Mode")
    print("📊 Visit: http://localhost:8000")
    print("🔌 WebSocket: ws://localhost:8000/ws")
    
    api = create_simplified_api()
    api.run() 