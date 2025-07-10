#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Flask API Server
Advanced Trading Intelligence System with Soulprint Registry Integration
"""

import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

from flask import Flask, jsonify, render_template_string, request

# Add core directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import automated trading routes
from api.automated_trading_routes import automated_trading

# Import live trading routes
from api.live_trading_routes import live_trading
from core.soulprint_registry import SoulprintRegistry
from core.strategy_logic import activate_strategy_for_hash
from core.unified_math_system import generate_unified_hash
from core.visual_execution_node import VisualExecutionNode

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web frontend

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Register blueprints
app.register_blueprint(live_trading, url_prefix='/api/live')
app.register_blueprint(automated_trading, url_prefix='/api/automated')

# Initialize Schwabot components
soulprint_registry = SoulprintRegistry("data/web_soulprint_registry.json")

# HTML Dashboard Template
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Schwabot Trading Terminal</title>
    <style>
        * {
            margin: 0; 
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px; 
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(0, 255, 0, 0.1);
            border: 2px solid #00ff00;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2.5rem;
            color: #00ff00;
            text-shadow: 0 0 20px rgba(0, 255, 0, 0.5);
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1rem;
            color: #00cc00;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .panel {
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid #00ff00;
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        .panel h3 {
            color: #00ff00;
            margin-bottom: 15px;
            font-size: 1.3rem;
            border-bottom: 1px solid #00ff00;
            padding-bottom: 10px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #00cc00;
            font-weight: 500;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #00ff00;
            border-radius: 5px;
            color: #00ff00;
            font-size: 14px;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #00ffff;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
        }
        
        .btn {
            background: linear-gradient(45deg, #003300, #006600);
            color: #00ff00; 
            border: 2px solid #00ff00; 
            padding: 12px 24px; 
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s ease;
            margin: 5px;
        }
        
        .btn:hover {
            background: linear-gradient(45deg, #00ff00, #00cc00);
            color: #000;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 255, 0, 0.3);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #004400, #008800);
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #440000, #880000);
            border-color: #ff0000;
            color: #ff6666;
        }
        
        .btn-secondary:hover {
            background: linear-gradient(45deg, #ff0000, #cc0000);
            color: #fff;
        }
        
        .output-panel {
            grid-column: 1 / -1;
            min-height: 300px;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .output {
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #00ff00;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.4;
            white-space: pre-wrap;
            min-height: 200px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online {
            background: #00ff00;
            box-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
        }
        
        .status-offline {
            background: #ff0000;
            box-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
        }
        
        .status-warning {
            background: #ffff00;
            box-shadow: 0 0 10px rgba(255, 255, 0, 0.5);
        }
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #00ff00;
        }
        
        .tab {
            padding: 10px 20px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff00;
            border-bottom: none;
            cursor: pointer;
            color: #00ff00;
            transition: all 0.3s ease;
        }
        
        .tab.active {
            background: #00ff00;
            color: #000;
        }
        
        .tab:hover {
            background: rgba(0, 255, 0, 0.2);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: rgba(0, 0, 0, 0.6);
            border: 1px solid #00ff00;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00ff00;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #00cc00;
            margin-top: 5px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 3px solid rgba(0, 255, 0, 0.3);
            border-top: 3px solid #00ff00;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .alert-success {
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid #00ff00;
            color: #00ff00;
        }
        
        .alert-error {
            background: rgba(255, 0, 0, 0.1);
            border: 1px solid #ff0000;
            color: #ff6666;
        }
        
        .alert-warning {
            background: rgba(255, 255, 0, 0.1);
            border: 1px solid #ffff00;
            color: #ffff00;
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .container {
                padding: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
    <div class="header">
            <h1>🌀 SCHWABOT TRADING TERMINAL</h1>
            <p>Advanced Trading Intelligence System v1.0.0 | <span class="status-indicator status-online"></span>System Online</p>
    </div>
    
        <div class="tabs">
            <div class="tab active" onclick="switchTab('trading')">🚀 Live Trading</div>
            <div class="tab" onclick="switchTab('matrix')">🔮 Matrix Analysis</div>
            <div class="tab" onclick="switchTab('strategy')">🧠 Strategy Center</div>
            <div class="tab" onclick="switchTab('automated')">🤖 Automated Trading</div>
            <div class="tab" onclick="switchTab('monitoring')">📊 System Monitor</div>
        </div>
        
        <!-- Trading Tab -->
        <div id="trading" class="tab-content active">
            <div class="dashboard-grid">
                <div class="panel">
                    <h3>🎯 Trade Configuration</h3>
                    <div class="form-group">
                        <label for="tradeMode">Trading Mode:</label>
                        <select id="tradeMode" onchange="updateTradeForm()">
                            <option value="hash">Hash-Based Trading</option>
                            <option value="strategy">Strategy-Based Trading</option>
                    </select>
                </div>
                    
                    <div id="hashConfig">
                        <div class="form-group">
                            <label for="hashVector">Hash Vector (JSON):</label>
                            <input type="text" id="hashVector" placeholder='[0.1, 0.2, 0.3, ...]' value="[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]">
                </div>
            </div>
                    
                    <div id="strategyConfig" style="display: none;">
                        <div class="form-group">
                            <label for="marketData">Market Data (JSON):</label>
                            <input type="text" id="marketData" placeholder='{"symbol": "BTC/USDC", "price": 65000}' value='{"symbol": "BTC/USDC", "price": 65000, "volume": 1000000000}'>
                </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="strategyName">Strategy:</label>
                        <select id="strategyName">
                            <option value="momentum">Momentum</option>
                        <option value="mean_reversion">Mean Reversion</option>
                        <option value="entropy_driven">Entropy Driven</option>
                    </select>
                </div>
                    
                    <button class="btn btn-primary" onclick="executeTrade()">🚀 Execute Trade</button>
                    <button class="btn btn-secondary" onclick="testRoute()">🧪 Test Route</button>
                </div>
                
                <div class="panel">
                    <h3>📊 Quick Actions</h3>
                    <button class="btn" onclick="getSystemStatus()">📋 System Status</button>
                    <button class="btn" onclick="visualizeMatrix()">🖼️ Matrix Visualization</button>
                    <button class="btn" onclick="matchMatrix()">🔍 Match Matrix</button>
                    <button class="btn" onclick="clearOutput()">🗑️ Clear Output</button>
                    
                    <div style="margin-top: 20px;">
                        <h4>🎲 Generate Test Data</h4>
                        <button class="btn" onclick="generateTestHash()">Generate Hash</button>
                        <button class="btn" onclick="generateTestMarketData()">Generate Market Data</button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Matrix Tab -->
        <div id="matrix" class="tab-content">
            <div class="dashboard-grid">
                <div class="panel">
                    <h3>🔮 Matrix Analysis</h3>
                    <div class="form-group">
                        <label for="matrixHash">Hash Vector for Matrix Matching:</label>
                        <input type="text" id="matrixHash" placeholder='[0.1, 0.2, 0.3, ...]' value="[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]">
                    </div>
                    <div class="form-group">
                        <label for="threshold">Similarity Threshold:</label>
                        <input type="number" id="threshold" value="0.8" min="0" max="1" step="0.1">
                    </div>
                    <button class="btn btn-primary" onclick="matchMatrix()">🔍 Find Matching Matrix</button>
                    <button class="btn" onclick="visualizeMatrix()">🖼️ Visualize Matrix</button>
        </div>
        
                <div class="panel">
                    <h3>📈 Matrix Metrics</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value" id="matrixCount">0</div>
                            <div class="metric-label">Available Matrices</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="matchRate">0%</div>
                            <div class="metric-label">Match Rate</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="avgSimilarity">0.0</div>
                            <div class="metric-label">Avg Similarity</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Strategy Tab -->
        <div id="strategy" class="tab-content">
            <div class="dashboard-grid">
                <div class="panel">
                    <h3>🧠 Strategy Management</h3>
                    <div class="form-group">
                        <label for="strategySelect">Select Strategy:</label>
                        <select id="strategySelect">
                            <option value="momentum">Momentum Strategy</option>
                            <option value="mean_reversion">Mean Reversion Strategy</option>
                            <option value="entropy_driven">Entropy Driven Strategy</option>
                        </select>
                    </div>
                    <button class="btn btn-primary" onclick="testStrategy()">🧪 Test Strategy</button>
                    <button class="btn" onclick="listStrategies()">📋 List Strategies</button>
                </div>
                
                <div class="panel">
                    <h3>📊 Strategy Performance</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value" id="strategySuccess">0%</div>
                            <div class="metric-label">Success Rate</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="avgConfidence">0.0</div>
                            <div class="metric-label">Avg Confidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="totalTrades">0</div>
                            <div class="metric-label">Total Trades</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Automated Trading Tab -->
        <div id="automated" class="tab-content">
            <div class="dashboard-grid">
                <div class="panel">
                    <h3>🤖 Automated Trading Control</h3>
                    <div class="form-group">
                        <label for="autoSymbol">Symbol to Track:</label>
                        <input type="text" id="autoSymbol" placeholder="BTC/USDC" value="BTC/USDC">
                    </div>
                    <button class="btn btn-primary" onclick="initializeAutomatedTrading()">🚀 Initialize System</button>
                    <button class="btn" onclick="addSymbolToTracking()">➕ Add Symbol</button>
                    <button class="btn" onclick="startAutoTrading()">▶️ Start Auto Trading</button>
                    <button class="btn btn-secondary" onclick="getCurrentPrices()">💰 Get Prices</button>
                </div>
                
                <div class="panel">
                    <h3>📈 Real-Time Prices</h3>
                    <div id="priceDisplay" style="font-family: monospace; font-size: 12px;">
                        <div>Loading prices...</div>
                    </div>
                    <button class="btn" onclick="analyzeSymbol()">🔍 Analyze Symbol</button>
                    <button class="btn" onclick="makeAutomatedDecision()">🧠 Make Decision</button>
                </div>
            </div>
            
            <div class="dashboard-grid">
                <div class="panel">
                    <h3>🏗️ Buy/Sell Walls</h3>
                    <div class="form-group">
                        <label for="wallSymbol">Symbol:</label>
                        <input type="text" id="wallSymbol" placeholder="BTC/USDC" value="BTC/USDC">
                    </div>
                    <div class="form-group">
                        <label for="wallQuantity">Quantity (USD):</label>
                        <input type="number" id="wallQuantity" value="1000" min="100" step="100">
                    </div>
                    <div class="form-group">
                        <label for="wallBatchCount">Batch Count (1-50):</label>
                        <input type="number" id="wallBatchCount" value="10" min="1" max="50">
                    </div>
                    <button class="btn btn-primary" onclick="createBuyWall()">🟢 Create Buy Wall</button>
                    <button class="btn btn-secondary" onclick="createSellWall()">🔴 Create Sell Wall</button>
                </div>
                
                <div class="panel">
                    <h3>🧺 Basket Trading</h3>
                    <div class="form-group">
                        <label for="basketSymbols">Symbols (comma-separated):</label>
                        <input type="text" id="basketSymbols" placeholder="BTC/USDC,ETH/USDC,SOL/USDC" value="BTC/USDC,ETH/USDC,SOL/USDC">
                    </div>
                    <div class="form-group">
                        <label for="basketValue">Total Value (USD):</label>
                        <input type="number" id="basketValue" value="3000" min="100" step="100">
                    </div>
                    <button class="btn" onclick="createBasketOrder()">🧺 Create Basket</button>
                </div>
            </div>
            
            <div class="dashboard-grid">
                <div class="panel">
                    <h3>📋 Active Orders</h3>
                    <div id="ordersDisplay" style="font-family: monospace; font-size: 12px;">
                        <div>Loading orders...</div>
                    </div>
                    <button class="btn" onclick="getAllOrders()">🔄 Refresh Orders</button>
                </div>
                
                <div class="panel">
                    <h3>📊 Portfolio & Learning</h3>
                    <div id="portfolioDisplay" style="font-family: monospace; font-size: 12px;">
                        <div>Loading portfolio...</div>
                    </div>
                    <button class="btn" onclick="getPortfolio()">💰 Get Portfolio</button>
                    <button class="btn" onclick="getLearningStatus()">🧠 Learning Status</button>
                </div>
            </div>
        </div>
        
        <!-- Monitoring Tab -->
        <div id="monitoring" class="tab-content">
            <div class="dashboard-grid">
                <div class="panel">
                    <h3>📊 System Health</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value" id="systemStatus">Online</div>
                            <div class="metric-label">System Status</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="apiVersion">1.0.0</div>
                            <div class="metric-label">API Version</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="uptime">0s</div>
                            <div class="metric-label">Uptime</div>
                        </div>
                    </div>
                    <button class="btn" onclick="getSystemStatus()">🔄 Refresh Status</button>
                </div>
                
                <div class="panel">
                    <h3>🔧 Component Status</h3>
                    <div id="componentStatus">
                        <div>🔄 Loading component status...</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Output Panel -->
        <div class="panel output-panel">
            <h3>📋 System Output</h3>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>Processing...</div>
            </div>
            <div class="output" id="output">
🌀 Schwabot Trading Terminal Ready
Enter parameters and execute trades...
            </div>
        </div>
    </div>

    <script>
        let startTime = Date.now();
        
        function updateUptime() {
            const uptime = Math.floor((Date.now() - startTime) / 1000);
            const hours = Math.floor(uptime / 3600);
            const minutes = Math.floor((uptime % 3600) / 60);
            const seconds = uptime % 60;
            document.getElementById('uptime').textContent = `${hours}h ${minutes}m ${seconds}s`;
        }
        
        setInterval(updateUptime, 1000);
        
        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function updateTradeForm() {
            const mode = document.getElementById('tradeMode').value;
            const hashConfig = document.getElementById('hashConfig');
            const strategyConfig = document.getElementById('strategyConfig');
            
            if (mode === 'hash') {
                hashConfig.style.display = 'block';
                strategyConfig.style.display = 'none';
            } else {
                hashConfig.style.display = 'none';
                strategyConfig.style.display = 'block';
            }
        }
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }
        
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }
        
        function log(message, type = 'info') {
            const output = document.getElementById('output');
            const timestamp = new Date().toLocaleTimeString();
            const prefix = type === 'error' ? '❌' : type === 'success' ? '✅' : type === 'warning' ? '⚠️' : 'ℹ️';
            output.textContent += `[${timestamp}] ${prefix} ${message}\\n`;
            output.scrollTop = output.scrollHeight;
        }
        
        function clearOutput() {
            document.getElementById('output').textContent = '🌀 Schwabot Trading Terminal Ready\\n';
        }
        
        function generateTestHash() {
            const hash = Array.from({length: 10}, () => Math.random().toFixed(3));
            document.getElementById('hashVector').value = `[${hash.join(', ')}]`;
            log('Generated test hash vector');
        }
        
        function generateTestMarketData() {
            const marketData = {
                symbol: "BTC/USDC",
                price: Math.floor(Math.random() * 100000) + 50000,
                volume: Math.floor(Math.random() * 1000000000) + 100000000,
                timestamp: Date.now()
            };
            document.getElementById('marketData').value = JSON.stringify(marketData);
            log('Generated test market data');
        }
        
        async function executeTrade() {
            showLoading();
            const mode = document.getElementById('tradeMode').value;
            const strategy = document.getElementById('strategyName').value;
            
            try {
                let response;
                
                if (mode === 'hash') {
                    const hashVector = JSON.parse(document.getElementById('hashVector').value);
                    response = await fetch('/api/live/trade/hash', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                            hash_vector: hashVector,
                            strategy_name: strategy
                    })
                });
                } else {
                    const marketData = JSON.parse(document.getElementById('marketData').value);
                    response = await fetch(`/api/live/trade/strategy/${strategy}`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            market_data: marketData
                        })
                    });
                }
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`Trade executed successfully!`, 'success');
                    log(`Strategy: ${data.strategy_used || data.strategy}`);
                    if (data.matrix_file) log(`Matrix: ${data.matrix_file}`);
                    if (data.trade_decision) log(`Decision: ${JSON.stringify(data.trade_decision, null, 2)}`);
                    if (data.decision) log(`Decision: ${JSON.stringify(data.decision, null, 2)}`);
                } else {
                    log(`Trade failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function testRoute() {
            showLoading();
            try {
                const response = await fetch('/api/live/test/route', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`Test route completed!`, 'success');
                    log(`Test type: ${data.test_type}`);
                    log(`Strategy tested: ${data.strategy_tested}`);
                    log(`Hash length: ${data.test_hash_length}`);
                } else {
                    log(`Test failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function matchMatrix() {
            showLoading();
            try {
                const hashVector = JSON.parse(document.getElementById('matrixHash').value);
                const threshold = parseFloat(document.getElementById('threshold').value);
                
                const response = await fetch('/api/live/matrix/match', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        hash_vector: hashVector,
                        threshold: threshold
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`Matrix match found!`, 'success');
                    log(`Matrix file: ${data.matrix_file}`);
                    log(`Threshold: ${data.threshold}`);
                    log(`Hash length: ${data.hash_vector_length}`);
                } else if (response.status === 404) {
                    log(`No matrix match found: ${data.message}`, 'warning');
                } else {
                    log(`Matrix matching failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function visualizeMatrix() {
            showLoading();
            try {
                const response = await fetch('/api/live/visualize/matrix');
                
                if (response.ok) {
                    log(`Matrix visualization generated!`, 'success');
                    log(`Content-Type: ${response.headers.get('Content-Type')}`);
                    log(`Content-Length: ${response.headers.get('Content-Length')} bytes`);
                    
                    // Create download link
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'matrix_visualization.png';
                    a.click();
                    window.URL.revokeObjectURL(url);
                } else {
                    log(`Visualization failed: ${response.status}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function getSystemStatus() {
            showLoading();
            try {
                const response = await fetch('/api/live/status');
                const data = await response.json();
                
                if (response.ok) {
                    log(`System status retrieved!`, 'success');
                    log(`Status: ${data.status}`);
                    log(`API Version: ${data.api_version}`);
                    log(`Strategies loaded: ${data.strategies_loaded}`);
                    
                    // Update component status
                    const componentStatus = document.getElementById('componentStatus');
                    componentStatus.innerHTML = '';
                    Object.entries(data.components).forEach(([component, status]) => {
                        const statusClass = status === 'available' ? 'status-online' : 'status-offline';
                        componentStatus.innerHTML += `<div><span class="status-indicator ${statusClass}"></span>${component}: ${status}</div>`;
                    });
                } else {
                    log(`Status check failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function testStrategy() {
            showLoading();
            try {
                const strategy = document.getElementById('strategySelect').value;
                const marketData = {
                    symbol: "BTC/USDC",
                    price: 65000,
                    volume: 1000000000,
                    timestamp: Date.now()
                };
                
                const response = await fetch(`/api/live/trade/strategy/${strategy}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        market_data: marketData
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`Strategy test completed!`, 'success');
                    log(`Strategy: ${data.strategy}`);
                    log(`Decision: ${JSON.stringify(data.decision, null, 2)}`);
                } else {
                    log(`Strategy test failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function listStrategies() {
            showLoading();
            try {
                log('Available strategies:', 'info');
                log('• Momentum Strategy', 'info');
                log('• Mean Reversion Strategy', 'info');
                log('• Entropy Driven Strategy', 'info');
                log('(Strategy listing completed)', 'success');
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        // Automated Trading Functions
        async function initializeAutomatedTrading() {
            showLoading();
            try {
                const response = await fetch('/api/automated/initialize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        exchange_config: {
                            name: 'coinbase',
                            sandbox: true
                        },
                        symbols: ['BTC/USDC', 'ETH/USDC', 'SOL/USDC']
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`Automated trading initialized!`, 'success');
                    log(`Exchange: ${data.exchange}`);
                    log(`Symbols: ${data.symbols_tracking.join(', ')}`);
                } else {
                    log(`Initialization failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function addSymbolToTracking() {
            showLoading();
            try {
                const symbol = document.getElementById('autoSymbol').value;
                const response = await fetch('/api/automated/add_symbol', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol: symbol })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`Added ${symbol} to tracking!`, 'success');
                } else {
                    log(`Failed to add symbol: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function getCurrentPrices() {
            showLoading();
            try {
                const response = await fetch('/api/automated/prices');
                const data = await response.json();
                
                if (response.ok) {
                    const priceDisplay = document.getElementById('priceDisplay');
                    priceDisplay.innerHTML = '';
                    
                    Object.entries(data.prices).forEach(([symbol, price]) => {
                        priceDisplay.innerHTML += `<div>${symbol}: $${price.toFixed(2)}</div>`;
                    });
                    
                    log(`Retrieved ${Object.keys(data.prices).length} prices`, 'success');
                } else {
                    log(`Failed to get prices: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function analyzeSymbol() {
            showLoading();
            try {
                const symbol = document.getElementById('autoSymbol').value;
                const response = await fetch(`/api/automated/analyze/${symbol}`);
                const data = await response.json();
                
                if (response.ok) {
                    log(`Analysis completed for ${symbol}!`, 'success');
                    log(`Momentum: ${JSON.stringify(data.analysis.momentum)}`);
                    log(`Volatility: ${data.analysis.volatility.toFixed(4)}`);
                    log(`Patterns found: ${data.analysis.patterns.length}`);
                } else {
                    log(`Analysis failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function makeAutomatedDecision() {
            showLoading();
            try {
                const symbol = document.getElementById('autoSymbol').value;
                const response = await fetch(`/api/automated/decision/${symbol}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    if (data.status === 'success') {
                        log(`Automated decision made!`, 'success');
                        log(`Action: ${data.decision.action}`);
                        log(`Confidence: ${(data.decision.confidence * 100).toFixed(1)}%`);
                        log(`Reasoning: ${data.decision.reasoning}`);
                    } else {
                        log(`No confident decision: ${data.message}`, 'warning');
                    }
                } else {
                    log(`Decision failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function startAutoTrading() {
            showLoading();
            try {
                const symbol = document.getElementById('autoSymbol').value;
                const response = await fetch(`/api/automated/auto_trade/${symbol}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        interval_seconds: 60
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`Auto trading started for ${symbol}!`, 'success');
                    log(`Check interval: ${data.interval_seconds} seconds`);
                } else {
                    log(`Auto trading failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function createBuyWall() {
            showLoading();
            try {
                const symbol = document.getElementById('wallSymbol').value;
                const quantity = parseFloat(document.getElementById('wallQuantity').value);
                const batchCount = parseInt(document.getElementById('wallBatchCount').value);
                
                const response = await fetch('/api/automated/create_buy_wall', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        symbol: symbol,
                        quantity: quantity,
                        batch_count: batchCount,
                        spread_seconds: 30
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`Buy wall created!`, 'success');
                    log(`Symbol: ${symbol}`);
                    log(`Quantity: $${quantity}`);
                    log(`Batch ID: ${data.batch_id}`);
                } else {
                    log(`Buy wall failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function createSellWall() {
            showLoading();
            try {
                const symbol = document.getElementById('wallSymbol').value;
                const quantity = parseFloat(document.getElementById('wallQuantity').value);
                const batchCount = parseInt(document.getElementById('wallBatchCount').value);
                
                const response = await fetch('/api/automated/create_sell_wall', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        symbol: symbol,
                        quantity: quantity,
                        batch_count: batchCount,
                        spread_seconds: 30
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`Sell wall created!`, 'success');
                    log(`Symbol: ${symbol}`);
                    log(`Quantity: $${quantity}`);
                    log(`Batch ID: ${data.batch_id}`);
                } else {
                    log(`Sell wall failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function createBasketOrder() {
            showLoading();
            try {
                const symbolsText = document.getElementById('basketSymbols').value;
                const symbols = symbolsText.split(',').map(s => s.trim());
                const totalValue = parseFloat(document.getElementById('basketValue').value);
                
                // Equal weights for simplicity
                const weights = [1.0 / symbols.length] * symbols.length;
                
                const response = await fetch('/api/automated/create_basket', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        symbols: symbols,
                        weights: weights,
                        value: totalValue,
                        strategy: 'basket'
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`Basket order created!`, 'success');
                    log(`Symbols: ${symbols.join(', ')}`);
                    log(`Total value: $${totalValue}`);
                    log(`Basket ID: ${data.basket_id}`);
                } else {
                    log(`Basket order failed: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function getAllOrders() {
            showLoading();
            try {
                const response = await fetch('/api/automated/orders');
                const data = await response.json();
                
                if (response.ok) {
                    const ordersDisplay = document.getElementById('ordersDisplay');
                    ordersDisplay.innerHTML = '';
                    
                    if (Object.keys(data.orders).length === 0) {
                        ordersDisplay.innerHTML = '<div>No active orders</div>';
                    } else {
                        Object.entries(data.orders).forEach(([orderId, order]) => {
                            ordersDisplay.innerHTML += `
                                <div style="margin-bottom: 10px; padding: 5px; border: 1px solid #00ff00;">
                                    <div><strong>ID:</strong> ${orderId}</div>
                                    <div><strong>Symbol:</strong> ${order.symbol}</div>
                                    <div><strong>Side:</strong> ${order.side}</div>
                                    <div><strong>Quantity:</strong> ${order.quantity}</div>
                                    <div><strong>Status:</strong> ${order.status}</div>
                                </div>
`;
                });
                    }
                
                    log(`Retrieved ${Object.keys(data.orders).length} active orders`, 'success');
                } else {
                    log(`Failed to get orders: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function getPortfolio() {
            showLoading();
            try {
                const response = await fetch('/api/automated/portfolio');
                const data = await response.json();
                
                if (response.ok) {
                    const portfolioDisplay = document.getElementById('portfolioDisplay');
                    portfolioDisplay.innerHTML = '';
                    
                    if (data.portfolio && data.portfolio.total) {
                        Object.entries(data.portfolio.total).forEach(([currency, balance]) => {
                            if (balance > 0) {
                                portfolioDisplay.innerHTML += `<div>${currency}: ${balance}</div>`;
                            }
                        });
                    } else {
                        portfolioDisplay.innerHTML = '<div>No portfolio data available</div>';
                    }
                    
                    log(`Portfolio retrieved`, 'success');
                } else {
                    log(`Failed to get portfolio: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        async function getLearningStatus() {
            showLoading();
            try {
                const response = await fetch('/api/automated/learning_status');
                const data = await response.json();
                
                if (response.ok) {
                    const status = data.learning_status;
                    log(`Learning status retrieved!`, 'success');
                    log(`Learned patterns: ${status.learned_patterns}`);
                    log(`Decision history: ${status.decision_history}`);
                    log(`Active strategies: ${status.active_strategies}`);
                } else {
                    log(`Failed to get learning status: ${data.error}`, 'error');
                }
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            getSystemStatus();
            log('Frontend initialized successfully', 'success');
        });
    </script>
</body>
</html>
'''


# SocketIO Event Handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected', 'message': 'Connected to Schwabot API'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")


@socketio.on('join_room')
def handle_join_room(data):
    """Handle client joining a room."""
    room = data.get('room', 'default')
    join_room(room)
    emit('room_joined', {'room': room, 'message': f'Joined room: {room}'})


@socketio.on('leave_room')
def handle_leave_room(data):
    """Handle client leaving a room."""
    room = data.get('room', 'default')
    leave_room(room)
    emit('room_left', {'room': room, 'message': f'Left room: {room}'})


@socketio.on('subscribe_to_updates')
def handle_subscribe_to_updates(data):
    """Handle client subscribing to real-time updates."""
    update_types = data.get('types', ['all'])
    room = data.get('room', 'default')
    join_room(room)
    emit(
        'subscribed',
        {'types': update_types, 'room': room, 'message': f'Subscribed to updates: {update_types}'},
    )


# Main route with enhanced real-time dashboard
@app.route('/')
def dashboard():
    """Enhanced dashboard with real-time functionality."""
    dashboard_html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Schwabot Trading Intelligence Dashboard</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                min-height: 100vh;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
                color: white;
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .status-bar {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 15px;
                margin-bottom: 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                color: white;
            }
            
            .connection-status {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #ff4757;
                animation: pulse 2s infinite;
            }
            
            .status-indicator.connected {
                background: #2ed573;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .card {
                background: rgba(255,255,255,0.95);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0,0,0,0.15);
            }
            
            .card h3 {
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.3rem;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #2c3e50;
            }
            
            .form-group input, .form-group select, .form-group textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s ease;
            }
            
            .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
                outline: none;
                border-color: #3498db;
            }
            
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: all 0.3s ease;
                width: 100%;
                margin-top: 10px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .progress-container {
                margin-top: 15px;
                display: none;
            }
            
            .progress-bar {
                width: 100%;
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                width: 0%;
                transition: width 0.3s ease;
            }
            
            .progress-text {
                text-align: center;
                margin-top: 8px;
                font-size: 12px;
                color: #666;
            }
            
            .results {
                margin-top: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                display: none;
            }
            
            .results h4 {
                margin-bottom: 10px;
                color: #2c3e50;
            }
            
            .results pre {
                background: #2c3e50;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 12px;
            }
            
            .realtime-feed {
                background: #2c3e50;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                max-height: 300px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
            
            .realtime-feed .event {
                margin-bottom: 8px;
                padding: 8px;
                background: rgba(255,255,255,0.1);
                border-radius: 4px;
                border-left: 3px solid #3498db;
            }
            
            .realtime-feed .event.trade { border-left-color: #2ed573; }
            .realtime-feed .event.error { border-left-color: #ff4757; }
            .realtime-feed .event.progress { border-left-color: #ffa502; }
            
            .timestamp {
                color: #95a5a6;
                font-size: 10px;
            }
            
            .event-type {
                font-weight: bold;
                color: #3498db;
            }
            
            .event-data {
                margin-top: 5px;
                color: #bdc3c7;
            }
            
            .system-info {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            
            .info-item {
                text-align: center;
                padding: 15px;
                background: rgba(52, 152, 219, 0.1);
                border-radius: 8px;
                border: 1px solid rgba(52, 152, 219, 0.2);
            }
            
            .info-value {
                font-size: 1.5rem;
                font-weight: bold;
                color: #3498db;
            }
            
            .info-label {
                font-size: 0.9rem;
                color: #7f8c8d;
                margin-top: 5px;
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .grid {
                    grid-template-columns: 1fr;
                }
                
                .status-bar {
                    flex-direction: column;
                    gap: 10px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Schwabot Trading Intelligence</h1>
                <p>Advanced AI-Powered Trading System with Real-Time Feedback</p>
            </div>
            
            <div class="status-bar">
                <div class="connection-status">
                    <div class="status-indicator" id="connectionStatus"></div>
                    <span id="connectionText">Connecting...</span>
                </div>
                <div>
                    <span id="lastUpdate">No updates yet</span>
                </div>
            </div>
            
            <div class="grid">
                <!-- Live Trading Card -->
                <div class="card">
                    <h3>🎯 Live Trading</h3>
                    <form id="tradeForm">
                        <div class="form-group">
                            <label for="hashVector">Hash Vector (JSON array):</label>
                            <textarea id="hashVector" rows="3" placeholder='[0.1, 0.2, 0.3, 0.4, 0.5]'>[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]</textarea>
                        </div>
                        <div class="form-group">
                            <label for="strategyName">Strategy:</label>
                            <select id="strategyName">
                                <option value="momentum">Momentum</option>
                                <option value="mean_reversion">Mean Reversion</option>
                                <option value="arbitrage">Arbitrage</option>
                            </select>
                        </div>
                        <button type="submit" class="btn" id="tradeBtn">Execute Trade</button>
                    </form>
                    <div class="progress-container" id="tradeProgress">
                        <div class="progress-bar">
                            <div class="progress-fill" id="tradeProgressFill"></div>
                        </div>
                        <div class="progress-text" id="tradeProgressText">Processing...</div>
                    </div>
                    <div class="results" id="tradeResults"></div>
                </div>
                
                <!-- Matrix Matching Card -->
                <div class="card">
                    <h3>🧮 Matrix Matching</h3>
                    <form id="matrixForm">
                        <div class="form-group">
                            <label for="matrixHash">Hash Vector:</label>
                            <textarea id="matrixHash" rows="3" placeholder='[0.1, 0.2, 0.3, 0.4, 0.5]'>[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]</textarea>
                        </div>
                        <div class="form-group">
                            <label for="threshold">Similarity Threshold:</label>
                            <input type="number" id="threshold" value="0.8" min="0" max="1" step="0.1">
                        </div>
                        <button type="submit" class="btn" id="matrixBtn">Find Matrix</button>
                    </form>
                    <div class="results" id="matrixResults"></div>
                </div>
                
                <!-- Strategy Testing Card -->
                <div class="card">
                    <h3>🧪 Strategy Testing</h3>
                    <form id="testForm">
                        <div class="form-group">
                            <label for="testStrategy">Test Strategy:</label>
                            <select id="testStrategy">
                                <option value="momentum">Momentum</option>
                                <option value="mean_reversion">Mean Reversion</option>
                                <option value="arbitrage">Arbitrage</option>
                            </select>
                        </div>
                        <button type="submit" class="btn" id="testBtn">Run Full Test</button>
                    </form>
                    <div class="progress-container" id="testProgress">
                        <div class="progress-bar">
                            <div class="progress-fill" id="testProgressFill"></div>
                        </div>
                        <div class="progress-text" id="testProgressText">Testing...</div>
                    </div>
                    <div class="results" id="testResults"></div>
                </div>
                
                <!-- System Status Card -->
                <div class="card">
                    <h3>📊 System Status</h3>
                    <div class="system-info" id="systemInfo">
                        <div class="info-item">
                            <div class="info-value" id="apiStatus">Loading...</div>
                            <div class="info-label">API Status</div>
                        </div>
                        <div class="info-item">
                            <div class="info-value" id="strategiesCount">-</div>
                            <div class="info-label">Strategies</div>
                        </div>
                        <div class="info-item">
                            <div class="info-value" id="uptime">-</div>
                            <div class="info-label">Uptime</div>
                        </div>
                    </div>
                    <button class="btn" id="refreshStatusBtn">Refresh Status</button>
                </div>
            </div>
            
            <!-- Real-Time Feed -->
            <div class="card">
                <h3>📡 Real-Time Event Feed</h3>
                <div class="realtime-feed" id="realtimeFeed">
                    <div class="event">
                        <span class="timestamp">Connecting to real-time feed...</span>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Initialize Socket.IO connection
            const socket = io();
            let isConnected = false;
            
            // Connection status management
            socket.on('connect', function() {
                isConnected = true;
                document.getElementById('connectionStatus').classList.add('connected');
                document.getElementById('connectionText').textContent = 'Connected';
                addEvent('Connected to Schwabot API', 'system');
            });
            
            socket.on('disconnect', function() {
                isConnected = false;
                document.getElementById('connectionStatus').classList.remove('connected');
                document.getElementById('connectionText').textContent = 'Disconnected';
                addEvent('Disconnected from Schwabot API', 'error');
            });
            
            // Real-time event handling
            socket.on('realtime_update', function(data) {
                const eventType = data.type;
                const eventData = data.data;
                const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
                
                addEvent(`${eventType}: ${JSON.stringify(eventData)}`, eventType);
                updateLastUpdate(timestamp);
                
                // Handle specific event types
                handleSpecificEvent(eventType, eventData);
            });
            
            function handleSpecificEvent(eventType, data) {
                switch(eventType) {
                    case 'trade_progress':
                        updateTradeProgress(data.progress, data.message);
                        break;
                    case 'trade_completed':
                        showTradeResults(data);
                        break;
                    case 'test_progress':
                        updateTestProgress(data.progress, data.message);
                        break;
                    case 'test_completed':
                        showTestResults(data);
                        break;
                    case 'status_update':
                        updateSystemStatus(data);
                        break;
                }
            }
            
            function addEvent(message, type = 'info') {
                const feed = document.getElementById('realtimeFeed');
                const event = document.createElement('div');
                event.className = `event ${type}`;
                
                const timestamp = new Date().toLocaleTimeString();
                event.innerHTML = `
                    <div class="timestamp">${timestamp}</div>
                    <div class="event-type">${type.toUpperCase()}</div>
                    <div class="event-data">${message}</div>
                `;
                
                feed.appendChild(event);
                feed.scrollTop = feed.scrollHeight;
                
                // Keep only last 50 events
                while (feed.children.length > 50) {
                    feed.removeChild(feed.firstChild);
                }
            }
            
            function updateLastUpdate(timestamp) {
                document.getElementById('lastUpdate').textContent = `Last update: ${timestamp}`;
            }
            
            // Trade form handling
            document.getElementById('tradeForm').addEventListener('submit', function(e) {
                e.preventDefault();
                if (!isConnected) {
                    alert('Not connected to server');
                    return;
                }
                
                const hashVector = JSON.parse(document.getElementById('hashVector').value);
                const strategyName = document.getElementById('strategyName').value;
                
                document.getElementById('tradeBtn').disabled = true;
                document.getElementById('tradeProgress').style.display = 'block';
                document.getElementById('tradeResults').style.display = 'none';
                
                fetch('/api/live/trade/hash', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        hash_vector: hashVector,
                        strategy_name: strategyName
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        addEvent(`Trade error: ${data.error}`, 'error');
                    } else {
                        addEvent(`Trade submitted: ${data.message}`, 'trade');
                    }
                })
                .catch(error => {
                    addEvent(`Trade request failed: ${error}`, 'error');
                })
                .finally(() => {
                    document.getElementById('tradeBtn').disabled = false;
                });
            });
            
            function updateTradeProgress(progress, message) {
                document.getElementById('tradeProgressFill').style.width = progress + '%';
                document.getElementById('tradeProgressText').textContent = message;
                
                if (progress >= 100) {
                    setTimeout(() => {
                        document.getElementById('tradeProgress').style.display = 'none';
                    }, 2000);
                }
            }
            
            function showTradeResults(data) {
                const results = document.getElementById('tradeResults');
                results.innerHTML = `
                    <h4>Trade Completed</h4>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                `;
                results.style.display = 'block';
            }
            
            // Matrix form handling
            document.getElementById('matrixForm').addEventListener('submit', function(e) {
                e.preventDefault();
                if (!isConnected) {
                    alert('Not connected to server');
                    return;
                }
                
                const hashVector = JSON.parse(document.getElementById('matrixHash').value);
                const threshold = parseFloat(document.getElementById('threshold').value);
                
                document.getElementById('matrixBtn').disabled = true;
                
                fetch('/api/live/matrix/match', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        hash_vector: hashVector,
                        threshold: threshold
                    })
                })
                .then(response => response.json())
                .then(data => {
                    const results = document.getElementById('matrixResults');
                    results.innerHTML = `
                        <h4>Matrix Match Result</h4>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    `;
                    results.style.display = 'block';
                })
                .catch(error => {
                    addEvent(`Matrix match failed: ${error}`, 'error');
                })
                .finally(() => {
                    document.getElementById('matrixBtn').disabled = false;
                });
            });
            
            // Test form handling
            document.getElementById('testForm').addEventListener('submit', function(e) {
                e.preventDefault();
                if (!isConnected) {
                    alert('Not connected to server');
                    return;
                }
                
                const strategyName = document.getElementById('testStrategy').value;
                
                document.getElementById('testBtn').disabled = true;
                document.getElementById('testProgress').style.display = 'block';
                document.getElementById('testResults').style.display = 'none';
                
                fetch('/api/live/test/route', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        strategy_name: strategyName
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        addEvent(`Test error: ${data.error}`, 'error');
                    } else {
                        addEvent(`Test submitted: ${data.message}`, 'test');
                    }
                })
                .catch(error => {
                    addEvent(`Test request failed: ${error}`, 'error');
                })
                .finally(() => {
                    document.getElementById('testBtn').disabled = false;
                });
            });
            
            function updateTestProgress(progress, message) {
                document.getElementById('testProgressFill').style.width = progress + '%';
                document.getElementById('testProgressText').textContent = message;
                
                if (progress >= 100) {
                    setTimeout(() => {
                        document.getElementById('testProgress').style.display = 'none';
                    }, 2000);
                }
            }
            
            function showTestResults(data) {
                const results = document.getElementById('testResults');
                results.innerHTML = `
                    <h4>Test Completed</h4>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                `;
                results.style.display = 'block';
            }
            
            // System status handling
            document.getElementById('refreshStatusBtn').addEventListener('click', function() {
                fetch('/api/live/status')
                .then(response => response.json())
                .then(data => {
                    updateSystemStatus(data);
                })
                .catch(error => {
                    addEvent(`Status check failed: ${error}`, 'error');
                });
            });
            
            function updateSystemStatus(data) {
                document.getElementById('apiStatus').textContent = data.status;
                document.getElementById('strategiesCount').textContent = data.strategies_loaded || 0;
                
                if (data.timestamp) {
                    const uptime = Math.floor((Date.now() / 1000 - data.timestamp) / 60);
                    document.getElementById('uptime').textContent = `${uptime}m`;
                }
            }
            
            // Initial status check
            setTimeout(() => {
                document.getElementById('refreshStatusBtn').click();
            }, 1000);
            
            // Subscribe to real-time updates
            socket.emit('subscribe_to_updates', {
                types: ['all'],
                room: 'dashboard'
            });
        </script>
    </body>
    </html>
    '''
    return dashboard_html


@app.route('/api/trading/signal', methods=['POST'])
def generate_trading_signal():
    """Generate a new trading signal with soulprint registration"""
    try:
        data = request.json
        asset = data.get('asset', 'BTC')
        price = float(data.get('price', 60000))
        volume = float(data.get('volume', 1000000000))
        strategy = data.get('strategy', 'momentum_breakout')

        # Create visual execution node
        visual_node = VisualExecutionNode(asset, price)
        result = visual_node.execute()

        # Create vector for soulprint registration
        vector = {
            'pair': f'{asset}/USDC',
            'entropy': 0.88,  # This would come from actual calculation
            'momentum': 0.04,
            'volatility': 0.19,
            'temporal_variance': 0.92,
            'volume': volume,
            'strategy': strategy,
        }

        # Register soulprint
        soulprint = soulprint_registry.register_soulprint(vector=vector, strategy_id=strategy, confidence=0.85)

        # Enhance result with soulprint
        enhanced_result = {
            'signal': result,
            'soulprint': soulprint,
            'vector': vector,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'registered',
        }

        return jsonify(enhanced_result)

    except Exception as e:
        app.logger.error(f"Error generating trading signal: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Signal generation failed: {str(e)}'}), 500


@app.route('/api/registry/stats', methods=['GET'])
def get_registry_stats():
    """Get soulprint registry statistics"""
    try:
        stats = soulprint_registry.get_registry_stats()
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error getting registry stats: {str(e)}")
        return jsonify({'error': f'Failed to get stats: {str(e)}'}), 500


@app.route('/api/registry/replayable', methods=['GET'])
def get_replayable_signals():
    """Get replayable soulprint signals"""
    try:
        min_confidence = float(request.args.get('min_confidence', 0.8))
        replayable = soulprint_registry.find_replayable(min_confidence=min_confidence)

        # Convert to serializable format
        serializable_replayable = []
        for entry in replayable:
            serializable_replayable.append(
                {
                    'soulprint': entry.soulprint,
                    'timestamp': entry.timestamp,
                    'vector': entry.vector,
                    'strategy_id': entry.strategy_id,
                    'confidence': entry.confidence,
                    'is_executed': entry.is_executed,
                    'profit_result': entry.profit_result,
                    'replayable': entry.replayable,
                }
            )

        return jsonify(serializable_replayable)
    except Exception as e:
        app.logger.error(f"Error getting replayable signals: {str(e)}")
        return jsonify({'error': f'Failed to get replayable signals: {str(e)}'}), 500


@app.route('/api/registry/similar', methods=['POST'])
def find_similar_soulprints():
    """Find similar soulprint patterns"""
    try:
        data = request.json
        target_vector = data.get('vector', {})
        threshold = float(data.get('threshold', 0.85))

        similar = soulprint_registry.get_similar_soulprints(target_vector, threshold)

        # Convert to serializable format
        serializable_similar = []
        for entry in similar:
            serializable_similar.append(
                {
                    'soulprint': entry.soulprint,
                    'timestamp': entry.timestamp,
                    'vector': entry.vector,
                    'strategy_id': entry.strategy_id,
                    'confidence': entry.confidence,
                }
            )

        return jsonify(serializable_similar)
    except Exception as e:
        app.logger.error(f"Error finding similar soulprints: {str(e)}")
        return jsonify({'error': f'Failed to find similar soulprints: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '1.0.0',
            'components': {
                'soulprint_registry': 'active',
                'visual_execution_node': 'active',
                'unified_math_system': 'active',
            },
        }
    )


@app.route("/trade", methods=["POST"])
def trade():
    data = request.json
    pair = data.get("pair", "BTC")
    price = data.get("price", 60000)

    # Build + Execute Signal
    node = VisualExecutionNode(pair, price)
    signal_packet = node.execute()

    # Persist soulprint
    soulprint_registry.register(pair, signal_packet["hash"])

    # Route through strategy core (A2)
    strategy_response = activate_strategy_for_hash(signal_packet["hash"], pair)

    return jsonify(
        {
            "input": signal_packet["visual_display"],
            "hash": signal_packet["hash"],
            "strategy": strategy_response,
        }
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🌀 Starting Schwabot Flask API Server with Real-Time SocketIO...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔧 API endpoints available at: http://localhost:5000/api/")
    print("📡 Real-time WebSocket events enabled")
    print("🛑 Press Ctrl+C to stop the server")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Run the Flask-SocketIO app with eventlet
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
