# -*- coding: utf-8 -*-
"""
Mathematical Relay Visualization API
===================================

FastAPI-based visualization server that provides real-time access to
mathematical relay system data for D3.js and other visualization tools.

Features:
- Real-time data streaming via WebSocket
- RESTful API endpoints for historical data
- Multiple export formats (D3.js, Plotly, TradingView)
- Integration with MathematicalBacklogManager
- Cross-platform compatibility
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False

from core.ccxt_trading_executor import CCXTTradingExecutor, TradingPair
from core.mathematical_backlog_manager import MathematicalBacklogManager
from core.mathematical_relay_sequencer import MathematicalRelaySequencer

logger = logging.getLogger(__name__)


class VisualizationDataModel(BaseModel):
    """Pydantic model for visualization data validation."""

    sequences: List[Dict[str, Any]]
    market_data: List[Dict[str, Any]]
    trade_results: List[Dict[str, Any]]
    portfolio_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class MathematicalVisualizationAPI:
    """FastAPI-based visualization server for mathematical relay system."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000, static_dir: str = "static", enable_cors: bool = True):

        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for visualization API")

        self.host = host
        self.port = port
        self.static_dir = static_dir

        # Initialize core components
        self.backlog_manager = MathematicalBacklogManager()
        self.trading_executor = CCXTTradingExecutor()
        self.sequencer = MathematicalRelaySequencer()

        # WebSocket connections
        self.active_connections: List[WebSocket] = []

        # Create FastAPI app
        self.app = FastAPI(
            title="Mathematical Relay Visualization API",
            description="Real-time visualization API for mathematical relay trading system",
            version="1.0.0",
        )

        # Setup CORS
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Setup static files
        try:
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        except Exception as e:
            logger.warning(f"Static directory {static_dir} not found: {e}")

        # Setup routes
        self._setup_routes()

        logger.info(f"MathematicalVisualizationAPI initialized on {host}:{port}")

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve the main dashboard."""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Mathematical Relay Dashboard</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <script src="https://d3js.org/d3-time.v3.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .chart-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                    .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                    .status.connected { background-color: #d4edda; color: #155724; }
                    .status.disconnected { background-color: #f8d7da; color: #721c24; }
                </style>
            </head>
            <body>
                <h1>Mathematical Relay System Dashboard</h1>
                <div id="connection-status" class="status disconnected">Disconnected</div>
                
                <div class="chart-container">
                    <h3>Sequence Timing Analysis</h3>
                    <div id="sequence-timing-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Portfolio Performance</h3>
                    <div id="portfolio-performance-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Bit-Depth Switching</h3>
                    <div id="bit-depth-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Market Data</h3>
                    <div id="market-data-chart"></div>
                </div>
                
                <script>
                    let ws = null;
                    let reconnectAttempts = 0;
                    const maxReconnectAttempts = 5;
                    
                    function connectWebSocket() {
                        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                        const wsUrl = `${protocol}//${window.location.host}/ws/real-time`;
                        
                        ws = new WebSocket(wsUrl);
                        
                        ws.onopen = function() {
                            document.getElementById('connection-status').className = 'status connected';
                            document.getElementById('connection-status').textContent = 'Connected';
                            reconnectAttempts = 0;
                        };
                        
                        ws.onmessage = function(event) {
                            const data = JSON.parse(event.data);
                            updateVisualizations(data);
                        };
                        
                        ws.onclose = function() {
                            document.getElementById('connection-status').className = 'status disconnected';
                            document.getElementById('connection-status').textContent = 'Disconnected';
                            
                            if (reconnectAttempts < maxReconnectAttempts) {
                                reconnectAttempts++;
                                setTimeout(connectWebSocket, 1000 * reconnectAttempts);
                            }
                        };
                        
                        ws.onerror = function(error) {
                            console.error('WebSocket error:', error);
                        };
                    }
                    
                    async function updateVisualizations(data) {
                        // Update sequence timing chart
                        updateSequenceTimingChart(data.sequences || []);
                        
                        // Update portfolio performance
                        updatePortfolioChart(data.trade_results || []);
                        
                        // Update bit-depth switching
                        updateBitDepthChart(data.sequences || []);
                        
                        // Update market data
                        updateMarketDataChart(data.market_data || []);
                    }
                    
                    function updateSequenceTimingChart(sequences) {
                        const container = document.getElementById('sequence-timing-chart');
                        container.innerHTML = '';
                        
                        if (sequences.length === 0) {
                            container.innerHTML = '<p>No sequence data available</p>';
                            return;
                        }
                        
                        const margin = {top: 20, right: 20, bottom: 30, left: 40};
                        const width = 800 - margin.left - margin.right;
                        const height = 400 - margin.top - margin.bottom;
                        
                        const svg = d3.select(container)
                            .append('svg')
                            .attr('width', width + margin.right + margin.left)
                            .attr('height', height + margin.top + margin.bottom)
                            .append('g')
                            .attr('transform', `translate(${margin.left},${margin.top})`);
                        
                        // Parse timestamps
                        const data = sequences.map(s => ({
                            time: new Date(s.start_time),
                            duration: s.total_duration_microseconds / 1000, // Convert to milliseconds
                            type: s.sequence_type
                        }));
                        
                        const x = d3.scaleTime()
                            .domain(d3.extent(data, d => d.time))
                            .range([0, width]);
                        
                        const y = d3.scaleLinear()
                            .domain([0, d3.max(data, d => d.duration)])
                            .range([height, 0]);
                        
                        // Add axes
                        svg.append('g')
                            .attr('transform', `translate(0,${height})`)
                            .call(d3.axisBottom(x));
                        
                        svg.append('g')
                            .call(d3.axisLeft(y));
                        
                        // Add dots
                        svg.selectAll('circle')
                            .data(data)
                            .enter()
                            .append('circle')
                            .attr('cx', d => x(d.time))
                            .attr('cy', d => y(d.duration))
                            .attr('r', 4)
                            .attr('fill', d => d.type === 'btc_price_hash' ? 'red' : 
                                               d.type === 'bit_depth_switch' ? 'blue' : 'green')
                            .append('title')
                            .text(d => `${d.type}: ${d.duration.toFixed(2)}ms`);
                    }
                    
                    function updatePortfolioChart(tradeResults) {
                        const container = document.getElementById('portfolio-performance-chart');
                        container.innerHTML = '';
                        
                        if (tradeResults.length === 0) {
                            container.innerHTML = '<p>No trade data available</p>';
                            return;
                        }
                        
                        const margin = {top: 20, right: 20, bottom: 30, left: 40};
                        const width = 800 - margin.left - margin.right;
                        const height = 400 - margin.top - margin.bottom;
                        
                        const svg = d3.select(container)
                            .append('svg')
                            .attr('width', width + margin.right + margin.left)
                            .attr('height', height + margin.top + margin.bottom)
                            .append('g')
                            .attr('transform', `translate(${margin.left},${margin.top})`);
                        
                        // Calculate cumulative profit
                        let cumulative = 0;
                        const data = tradeResults.map(t => {
                            if (t.profit_realized) {
                                cumulative += parseFloat(t.profit_realized);
                            }
                            return {
                                time: new Date(t.execution_time * 1000),
                                profit: cumulative
                            };
                        });
                        
                        const x = d3.scaleTime()
                            .domain(d3.extent(data, d => d.time))
                            .range([0, width]);
                        
                        const y = d3.scaleLinear()
                            .domain([d3.min(data, d => d.profit), d3.max(data, d => d.profit)])
                            .range([height, 0]);
                        
                        // Add axes
                        svg.append('g')
                            .attr('transform', `translate(0,${height})`)
                            .call(d3.axisBottom(x));
                        
                        svg.append('g')
                            .call(d3.axisLeft(y));
                        
                        // Add line
                        const line = d3.line()
                            .x(d => x(d.time))
                            .y(d => y(d.profit));
                        
                        svg.append('path')
                            .datum(data)
                            .attr('fill', 'none')
                            .attr('stroke', 'steelblue')
                            .attr('stroke-width', 2)
                            .attr('d', line);
                    }
                    
                    function updateBitDepthChart(sequences) {
                        const container = document.getElementById('bit-depth-chart');
                        container.innerHTML = '';
                        
                        const bitDepthSequences = sequences.filter(s => s.sequence_type === 'bit_depth_switch');
                        
                        if (bitDepthSequences.length === 0) {
                            container.innerHTML = '<p>No bit-depth switching data available</p>';
                            return;
                        }
                        
                        const margin = {top: 20, right: 20, bottom: 30, left: 40};
                        const width = 800 - margin.left - margin.right;
                        const height = 400 - margin.top - margin.bottom;
                        
                        const svg = d3.select(container)
                            .append('svg')
                            .attr('width', width + margin.right + margin.left)
                            .attr('height', height + margin.top + margin.bottom)
                            .append('g')
                            .attr('transform', `translate(${margin.left},${margin.top})`);
                        
                        const data = bitDepthSequences.map(s => ({
                            time: new Date(s.start_time),
                            bitDepth: s.bit_depth || 32,
                            duration: s.total_duration_microseconds / 1000
                        }));
                        
                        const x = d3.scaleTime()
                            .domain(d3.extent(data, d => d.time))
                            .range([0, width]);
                        
                        const y = d3.scaleLinear()
                            .domain([0, d3.max(data, d => d.bitDepth)])
                            .range([height, 0]);
                        
                        // Add axes
                        svg.append('g')
                            .attr('transform', `translate(0,${height})`)
                            .call(d3.axisBottom(x));
                        
                        svg.append('g')
                            .call(d3.axisLeft(y));
                        
                        // Add bars
                        svg.selectAll('rect')
                            .data(data)
                            .enter()
                            .append('rect')
                            .attr('x', d => x(d.time))
                            .attr('y', d => y(d.bitDepth))
                            .attr('width', 20)
                            .attr('height', d => height - y(d.bitDepth))
                            .attr('fill', 'orange')
                            .append('title')
                            .text(d => `Bit Depth: ${d.bitDepth}, Duration: ${d.duration.toFixed(2)}ms`);
                    }
                    
                    function updateMarketDataChart(marketData) {
                        const container = document.getElementById('market-data-chart');
                        container.innerHTML = '';
                        
                        if (marketData.length === 0) {
                            container.innerHTML = '<p>No market data available</p>';
                            return;
                        }
                        
                        const margin = {top: 20, right: 20, bottom: 30, left: 40};
                        const width = 800 - margin.left - margin.right;
                        const height = 400 - margin.top - margin.bottom;
                        
                        const svg = d3.select(container)
                            .append('svg')
                            .attr('width', width + margin.right + margin.left)
                            .attr('height', height + margin.top + margin.bottom)
                            .append('g')
                            .attr('transform', `translate(${margin.left},${margin.top})`);
                        
                        const data = marketData.map(m => ({
                            time: new Date(m.timestamp),
                            price: parseFloat(m.price),
                            pair: m.pair
                        }));
                        
                        const x = d3.scaleTime()
                            .domain(d3.extent(data, d => d.time))
                            .range([0, width]);
                        
                        const y = d3.scaleLinear()
                            .domain([d3.min(data, d => d.price), d3.max(data, d => d.price)])
                            .range([height, 0]);
                        
                        // Add axes
                        svg.append('g')
                            .attr('transform', `translate(0,${height})`)
                            .call(d3.axisBottom(x));
                        
                        svg.append('g')
                            .call(d3.axisLeft(y));
                        
                        // Add line
                        const line = d3.line()
                            .x(d => x(d.time))
                            .y(d => y(d.price));
                        
                        svg.append('path')
                            .datum(data)
                            .attr('fill', 'none')
                            .attr('stroke', 'green')
                            .attr('stroke-width', 2)
                            .attr('d', line);
                    }
                    
                    // Initial load
                    async function loadInitialData() {
                        try {
                            const response = await fetch('/api/dashboard-data');
                            const data = await response.json();
                            updateVisualizations(data);
                        } catch (error) {
                            console.error('Error loading initial data:', error);
                        }
                    }
                    
                    // Connect WebSocket and load initial data
                    connectWebSocket();
                    loadInitialData();
                    
                    // Refresh data every 30 seconds as fallback
                    setInterval(loadInitialData, 30000);
                </script>
            </body>
            </html>
            """

        @self.app.get("/api/dashboard-data")
        async def get_dashboard_data():
            """Get comprehensive dashboard data."""
            try:
                return self._export_d3_format()
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/sequences")
        async def get_sequences(limit: int = 1000, sequence_type: Optional[str] = None):
            """Get sequence data with optional filtering."""
            try:
                sequences = self.backlog_manager.retrieve_events("sequence_logs", limit=limit)
                if sequence_type:
                    sequences = [s for s in sequences if s.get("sequence_type") == sequence_type]
                return {"sequences": sequences, "count": len(sequences)}
            except Exception as e:
                logger.error(f"Error getting sequences: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/market-data")
        async def get_market_data(limit: int = 1000, pair: Optional[str] = None):
            """Get market data with optional filtering."""
            try:
                market_data = self.backlog_manager.retrieve_events("market_data", limit=limit)
                if pair:
                    market_data = [m for m in market_data if m.get("pair") == pair]
                return {"market_data": market_data, "count": len(market_data)}
            except Exception as e:
                logger.error(f"Error getting market data: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/trade-results")
        async def get_trade_results(limit: int = 1000):
            """Get trade results."""
            try:
                trade_results = self.backlog_manager.retrieve_events("trade_results", limit=limit)
                return {"trade_results": trade_results, "count": len(trade_results)}
            except Exception as e:
                logger.error(f"Error getting trade results: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/system-status")
        async def get_system_status():
            """Get overall system status."""
            try:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "backlog_manager": "active",
                    "trading_executor": "active",
                    "sequencer": "active",
                    "active_connections": len(self.active_connections),
                    "statistics": self.sequencer.get_sequencing_statistics(),
                }
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/ws/real-time")
        async def websocket_endpoint(websocket: WebSocket):
            """Real-time data streaming via WebSocket."""
            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                while True:
                    # Send real-time data every second
                    data = self._export_d3_format()
                    await websocket.send_text(json.dumps(data))
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

    def _export_d3_format(self) -> Dict[str, Any]:
        """Export data optimized for D3.js consumption."""
        try:
            # Get data from backlog manager
            sequences = self.backlog_manager.retrieve_events("sequence_logs", limit=1000)
            market_data = self.backlog_manager.retrieve_events("market_data", limit=1000)
            trade_results = self.backlog_manager.retrieve_events("trade_results", limit=1000)

            # Format sequences for D3.js
            formatted_sequences = []
            for seq in sequences:
                formatted_sequences.append(
                    {
                        "sequence_id": seq.get("sequence_id"),
                        "sequence_type": seq.get("sequence_type"),
                        "start_time": seq.get("start_time"),
                        "end_time": seq.get("end_time"),
                        "status": seq.get("status"),
                        "total_duration_microseconds": seq.get("total_duration_microseconds", 0),
                        "bit_depth": seq.get("bit_depth"),
                        "btc_price": seq.get("btc_price"),
                        "btc_hash": seq.get("btc_hash"),
                        "ferris_rde_result": seq.get("ferris_rde_result"),
                    }
                )

            # Format market data for D3.js
            formatted_market_data = []
            for md in market_data:
                formatted_market_data.append(
                    {
                        "timestamp": md.get("timestamp"),
                        "pair": md.get("pair"),
                        "price": md.get("price"),
                        "volume": md.get("volume", 0),
                    }
                )

            # Format trade results for D3.js
            formatted_trade_results = []
            for tr in trade_results:
                formatted_trade_results.append(
                    {
                        "signal_id": tr.get("signal_id"),
                        "pair": tr.get("pair"),
                        "strategy": tr.get("strategy"),
                        "executed": tr.get("executed", False),
                        "fill_price": tr.get("fill_price"),
                        "fill_amount": tr.get("fill_amount"),
                        "profit_realized": tr.get("profit_realized", 0),
                        "execution_time": tr.get("execution_time"),
                        "error_message": tr.get("error_message"),
                    }
                )

            return {
                "sequences": formatted_sequences,
                "market_data": formatted_market_data,
                "trade_results": formatted_trade_results,
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_sequences": len(formatted_sequences),
                    "total_market_data_points": len(formatted_market_data),
                    "total_trades": len(formatted_trade_results),
                    "format": "d3_optimized",
                },
            }
        except Exception as e:
            logger.error(f"Error exporting D3 format: {e}")
            return {
                "sequences": [],
                "market_data": [],
                "trade_results": [],
                "metadata": {"error": str(e), "export_timestamp": datetime.now().isoformat()},
            }

    def start(self):
        """Start the visualization server."""
        try:
            import uvicorn

            uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
        except ImportError:
            logger.error("Uvicorn not available. Install with: pip install uvicorn")
            raise
        except Exception as e:
            logger.error(f"Error starting visualization server: {e}")
            raise


# Global instance for easy access
visualization_api = None


def start_visualization_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the visualization server."""
    global visualization_api
    visualization_api = MathematicalVisualizationAPI(host=host, port=port)
    visualization_api.start()


if __name__ == "__main__":
    # Test the visualization API
    start_visualization_server()
