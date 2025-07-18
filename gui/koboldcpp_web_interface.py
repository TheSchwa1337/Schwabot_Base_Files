#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KoboldCPP Web Interface for Unified Mathematical Trading System
==============================================================

Secure Flask web interface that provides:
1. Real-time trading dashboard with KoboldCPP integration
2. Mathematical component visualization and control
3. Multi-cryptocurrency trading interface
4. Strategy mapping and bit phase logic controls
5. Tensor analysis and memory management
6. Secure API endpoints for trading operations
"""

import asyncio
import json
import logging
import os
import secrets
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np

# Import core system components
from core.koboldcpp_integration import KoboldCPPIntegration, CryptocurrencyType, AnalysisType
from core.unified_mathematical_trading_system import UnifiedMathematicalTradingSystem
from core.tensor_weight_memory import TensorWeightMemory
from core.strategy_mapper import StrategyMapper
from core.visual_decision_engine import VisualDecisionEngine
from core.soulprint_registry import SoulprintRegistry
from core.cascade_memory_architecture import CascadeMemoryArchitecture

logger = logging.getLogger(__name__)

class KoboldCPPWebInterface:
    """Secure web interface for KoboldCPP trading system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the web interface."""
        self.config = config
        self.app = Flask(__name__)
        self.app.secret_key = secrets.token_hex(32)
        
        # Security configuration
        self.app.config['SESSION_COOKIE_SECURE'] = True
        self.app.config['SESSION_COOKIE_HTTPONLY'] = True
        self.app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Core system components
        self.kobold_integration = KoboldCPPIntegration()
        self.trading_system = UnifiedMathematicalTradingSystem(config)
        self.tensor_memory = TensorWeightMemory()
        self.strategy_mapper = StrategyMapper()
        self.visual_engine = VisualDecisionEngine()
        self.soulprint_registry = SoulprintRegistry()
        self.cascade_memory = CascadeMemoryArchitecture()
        
        # User management
        self.users = {
            'admin': generate_password_hash('admin123'),  # Change in production
            'trader': generate_password_hash('trader123')
        }
        
        # Active sessions
        self.active_sessions = {}
        
        # Real-time data
        self.market_data_cache = {}
        self.trading_signals = []
        self.system_status = {}
        
        # Setup routes and event handlers
        self._setup_routes()
        self._setup_socketio_events()
        
        logger.info("🌐 KoboldCPP Web Interface initialized")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return render_template('koboldcpp_dashboard.html')
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """User login."""
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')
                
                if username in self.users and check_password_hash(self.users[username], password):
                    session['user_id'] = username
                    session['login_time'] = time.time()
                    self.active_sessions[username] = {
                        'login_time': time.time(),
                        'last_activity': time.time()
                    }
                    return redirect(url_for('index'))
                else:
                    return render_template('login.html', error='Invalid credentials')
            
            return render_template('login.html')
        
        @self.app.route('/logout')
        def logout():
            """User logout."""
            if 'user_id' in session:
                username = session['user_id']
                if username in self.active_sessions:
                    del self.active_sessions[username]
                session.clear()
            return redirect(url_for('login'))
        
        @self.app.route('/api/system/status')
        def get_system_status():
            """Get comprehensive system status."""
            if 'user_id' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            try:
                status = {
                    'koboldcpp': self.kobold_integration.get_statistics(),
                    'trading_system': self.trading_system.get_system_status(),
                    'tensor_memory': self.tensor_memory.get_statistics(),
                    'strategy_mapper': self.strategy_mapper.get_system_status(),
                    'visual_engine': self.visual_engine.get_statistics(),
                    'soulprint_registry': {
                        'total_decisions': len(self.soulprint_registry.get_all_decisions()),
                        'active': True
                    },
                    'cascade_memory': {
                        'total_cascades': len(self.cascade_memory.get_all_cascades()),
                        'active': True
                    },
                    'market_data': {
                        'cached_symbols': len(self.market_data_cache),
                        'last_update': time.time()
                    },
                    'trading_signals': {
                        'total_signals': len(self.trading_signals),
                        'recent_signals': self.trading_signals[-10:] if self.trading_signals else []
                    }
                }
                
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"❌ System status error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/market/analyze', methods=['POST'])
        def analyze_market_data():
            """Analyze market data using KoboldCPP."""
            if 'user_id' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            try:
                data = request.json
                symbol = data.get('symbol', 'BTC/USDC')
                price = data.get('price', 0.0)
                volume = data.get('volume', 0.0)
                volatility = data.get('volatility', 0.0)
                
                # Create market data
                market_data = {
                    'symbol': symbol,
                    'price': price,
                    'volume': volume,
                    'volatility': volatility,
                    'bit_phase': data.get('bit_phase', 8),
                    'timestamp': time.time()
                }
                
                # Analyze with KoboldCPP
                analysis_result = asyncio.run(self.kobold_integration.analyze_market_data(market_data))
                
                if analysis_result:
                    # Store in cache
                    self.market_data_cache[symbol] = {
                        'data': market_data,
                        'analysis': analysis_result,
                        'timestamp': time.time()
                    }
                    
                    # Add to trading signals
                    signal = {
                        'symbol': symbol,
                        'action': analysis_result.get('action', 'HOLD'),
                        'confidence': analysis_result.get('confidence', 0.0),
                        'timestamp': time.time(),
                        'analysis': analysis_result
                    }
                    self.trading_signals.append(signal)
                    
                    # Emit real-time update
                    self.socketio.emit('market_analysis', signal)
                    
                    return jsonify({
                        'success': True,
                        'analysis': analysis_result,
                        'signal': signal
                    })
                else:
                    return jsonify({'error': 'Analysis failed'}), 500
                    
            except Exception as e:
                logger.error(f"❌ Market analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/trading/execute', methods=['POST'])
        def execute_trade():
            """Execute a trade based on analysis."""
            if 'user_id' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            try:
                data = request.json
                symbol = data.get('symbol')
                action = data.get('action')
                amount = data.get('amount', 0.0)
                
                # Create trading decision
                decision_data = {
                    'symbol': symbol,
                    'action': action,
                    'amount': amount,
                    'user_id': session['user_id'],
                    'timestamp': time.time()
                }
                
                # Process through trading system
                result = asyncio.run(self.trading_system.process_market_data_comprehensive({
                    'symbol': symbol,
                    'price': self.market_data_cache.get(symbol, {}).get('data', {}).get('price', 0.0),
                    'action': action,
                    'amount': amount
                }))
                
                # Store decision
                self.soulprint_registry.store_decision(decision_data)
                
                # Emit real-time update
                self.socketio.emit('trade_executed', {
                    'symbol': symbol,
                    'action': action,
                    'amount': amount,
                    'timestamp': time.time(),
                    'user': session['user_id']
                })
                
                return jsonify({
                    'success': True,
                    'decision': asdict(result) if result else None
                })
                
            except Exception as e:
                logger.error(f"❌ Trade execution error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/strategy/map', methods=['POST'])
        def map_strategy():
            """Map strategy to bit phase."""
            if 'user_id' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            try:
                data = request.json
                strategy_id = data.get('strategy_id')
                confidence = data.get('confidence', 0.5)
                
                # Map strategy
                bit_phase = self.strategy_mapper.map_strategy_to_bit_phase(
                    strategy_id, confidence, data.get('metadata')
                )
                
                return jsonify({
                    'success': True,
                    'strategy_id': strategy_id,
                    'bit_phase': bit_phase,
                    'confidence': confidence
                })
                
            except Exception as e:
                logger.error(f"❌ Strategy mapping error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/tensor/analyze', methods=['POST'])
        def analyze_tensor():
            """Analyze tensor data."""
            if 'user_id' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            try:
                data = request.json
                symbol = data.get('symbol')
                tensor_data = data.get('tensor_data', {})
                
                # Store tensor score
                self.tensor_memory.store_tensor_score(
                    symbol, 
                    data.get('score', 0.0), 
                    tensor_data
                )
                
                # Get analysis
                score = self.tensor_memory.get_tensor_score(symbol)
                avg_score = self.tensor_memory.get_average_tensor_score(symbol)
                
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'current_score': score,
                    'average_score': avg_score,
                    'statistics': self.tensor_memory.get_statistics()
                })
                
            except Exception as e:
                logger.error(f"❌ Tensor analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/visual/analyze', methods=['POST'])
        def analyze_visual():
            """Analyze visual data."""
            if 'user_id' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            try:
                data = request.json
                symbol = data.get('symbol')
                visual_data = data.get('visual_data', {})
                
                # Analyze visual data
                analysis = self.visual_engine.analyze_visual_data(symbol, visual_data)
                
                if analysis:
                    return jsonify({
                        'success': True,
                        'symbol': symbol,
                        'pattern_type': analysis.pattern_type,
                        'confidence': analysis.confidence,
                        'metadata': analysis.metadata
                    })
                else:
                    return jsonify({'error': 'Visual analysis failed'}), 500
                    
            except Exception as e:
                logger.error(f"❌ Visual analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/cryptocurrency/supported')
        def get_supported_cryptocurrencies():
            """Get supported cryptocurrencies."""
            if 'user_id' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            cryptocurrencies = [
                {'symbol': 'BTC', 'name': 'Bitcoin', 'type': CryptocurrencyType.BTC.value},
                {'symbol': 'ETH', 'name': 'Ethereum', 'type': CryptocurrencyType.ETH.value},
                {'symbol': 'XRP', 'name': 'Ripple', 'type': CryptocurrencyType.XRP.value},
                {'symbol': 'SOL', 'name': 'Solana', 'type': CryptocurrencyType.SOL.value},
                {'symbol': 'USDC', 'name': 'USD Coin', 'type': CryptocurrencyType.USDC.value}
            ]
            
            return jsonify({
                'success': True,
                'cryptocurrencies': cryptocurrencies
            })
    
    def _setup_socketio_events(self):
        """Setup SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            if 'user_id' not in session:
                emit('error', {'message': 'Unauthorized'})
                return
            
            logger.info(f"🔌 Client connected: {session['user_id']}")
            emit('connected', {'user_id': session['user_id']})
            
            # Send current system status
            status = {
                'koboldcpp': self.kobold_integration.get_statistics(),
                'trading_system': self.trading_system.get_system_status(),
                'market_data': {
                    'cached_symbols': len(self.market_data_cache),
                    'last_update': time.time()
                }
            }
            emit('system_status', status)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            if 'user_id' in session:
                logger.info(f"🔌 Client disconnected: {session['user_id']}")
        
        @self.socketio.on('request_market_data')
        def handle_market_data_request(data):
            """Handle market data requests."""
            if 'user_id' not in session:
                emit('error', {'message': 'Unauthorized'})
                return
            
            symbol = data.get('symbol', 'BTC/USDC')
            if symbol in self.market_data_cache:
                emit('market_data', self.market_data_cache[symbol])
            else:
                emit('error', {'message': f'No data available for {symbol}'})
        
        @self.socketio.on('request_trading_signals')
        def handle_trading_signals_request(data):
            """Handle trading signals requests."""
            if 'user_id' not in session:
                emit('error', {'message': 'Unauthorized'})
                return
            
            limit = data.get('limit', 10)
            recent_signals = self.trading_signals[-limit:] if self.trading_signals else []
            emit('trading_signals', recent_signals)
    
    def start_server(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Start the Flask server."""
        try:
            logger.info(f"🚀 Starting KoboldCPP Web Interface on {host}:{port}")
            
            # Start KoboldCPP integration
            asyncio.run(self.kobold_integration.start_kobold_server())
            
            # Start trading system
            self.trading_system.start_flask_server(host='127.0.0.1', port=5001)
            
            # Start web interface
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=False
            )
            
        except Exception as e:
            logger.error(f"❌ Server startup failed: {e}")
            raise
    
    def stop_server(self):
        """Stop the Flask server."""
        try:
            # Stop KoboldCPP integration
            asyncio.run(self.kobold_integration.stop_kobold_server())
            
            # Stop trading system
            # Note: Trading system Flask server runs separately
            
            logger.info("✅ KoboldCPP Web Interface stopped")
            
        except Exception as e:
            logger.error(f"❌ Server shutdown error: {e}")

def create_koboldcpp_web_interface(config: Dict[str, Any]) -> KoboldCPPWebInterface:
    """Create and configure the web interface."""
    return KoboldCPPWebInterface(config)

if __name__ == "__main__":
    # Configuration
    config = {
        'koboldcpp': {
            'path': 'koboldcpp',
            'port': 5001,
            'model_path': ''
        },
        'trading_system': {
            'flask_port': 5002,
            'enable_multi_bot': True,
            'enable_consensus': True
        },
        'security': {
            'session_timeout': 3600,
            'max_login_attempts': 3
        }
    }
    
    # Create and start interface
    interface = create_koboldcpp_web_interface(config)
    interface.start_server(debug=True) 