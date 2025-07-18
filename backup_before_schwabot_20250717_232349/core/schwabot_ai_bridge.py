#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot AI Bridge - Schwabot Integration Layer
============================================

This bridge connects the Schwabot unified trading system to Schwabot AI's
existing Flask/HTTP interface, allowing the AI to access all trading
functionality through natural language conversation.

The bridge provides:
- Trading system access through AI conversation
- Real-time data analysis and visualization
- Strategy execution and monitoring
- Portfolio management through chat
- Market analysis and insights
- Automated trading operations
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import requests
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Import our unified system components
from .schwabot_unified_interface import SchwabotUnifiedInterface, InterfaceMode
from .schwabot_ai_integration import Schwabot AIIntegration, AnalysisType
from .visual_layer_controller import VisualLayerController
from .tick_loader import TickLoader
from .signal_cache import SignalCache
from .registry_writer import RegistryWriter
from .json_server import JSONServer
from .memory_stack.ai_command_sequencer import AICommandSequencer
from .memory_stack.execution_validator import ExecutionValidator
from .memory_stack.memory_key_allocator import MemoryKeyAllocator

logger = logging.getLogger(__name__)

class CommandType(Enum):
    """Types of commands the AI can execute."""
    TRADING_ANALYSIS = "trading_analysis"
    PORTFOLIO_STATUS = "portfolio_status"
    EXECUTE_TRADE = "execute_trade"
    MARKET_INSIGHT = "market_insight"
    STRATEGY_ACTIVATION = "strategy_activation"
    VISUALIZATION = "visualization"
    SYSTEM_STATUS = "system_status"
    DATA_ANALYSIS = "data_analysis"

@dataclass
class AICommand:
    """Represents a command from the AI."""
    command_type: CommandType
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    user_message: str = ""
    response: str = ""

@dataclass
class TradingContext:
    """Trading context for AI interactions."""
    current_symbols: List[str] = field(default_factory=list)
    active_strategies: List[str] = field(default_factory=list)
    portfolio_value: float = 0.0
    risk_level: str = "medium"
    last_analysis: Optional[Dict[str, Any]] = None

class Schwabot AIBridge:
    """Bridge between Schwabot AI and Schwabot unified system."""
    
    def __init__(self, schwabot_port: int = 5001, bridge_port: int = 5005):
        """Initialize the bridge."""
        self.schwabot_port = schwabot_port
        self.bridge_port = bridge_port
        self.schwabot_url = f"http://localhost:{schwabot_port}"
        
        # Initialize unified system
        self.unified_interface = SchwabotUnifiedInterface(InterfaceMode.FULL_INTEGRATION)
        
        # Initialize components
        self.schwabot_integration = Schwabot AIIntegration(port=schwabot_port)
        self.visual_controller = VisualLayerController()
        self.tick_loader = TickLoader()
        self.signal_cache = SignalCache()
        self.registry_writer = RegistryWriter()
        
        # Memory stack components
        self.command_sequencer = AICommandSequencer()
        self.execution_validator = ExecutionValidator()
        self.memory_allocator = MemoryKeyAllocator()
        
        # Trading context
        self.trading_context = TradingContext()
        
        # Command patterns for AI recognition
        self.command_patterns = {
            CommandType.TRADING_ANALYSIS: [
                r"analyze\s+(\w+/\w+)",
                r"what's\s+the\s+analysis\s+for\s+(\w+/\w+)",
                r"show\s+me\s+(\w+/\w+)\s+analysis",
                r"chart\s+(\w+/\w+)",
                r"technical\s+analysis\s+(\w+/\w+)"
            ],
            CommandType.PORTFOLIO_STATUS: [
                r"portfolio\s+status",
                r"show\s+portfolio",
                r"what's\s+my\s+portfolio",
                r"portfolio\s+value",
                r"current\s+holdings"
            ],
            CommandType.EXECUTE_TRADE: [
                r"buy\s+(\w+/\w+)\s+(\d+(?:\.\d+)?)",
                r"sell\s+(\w+/\w+)\s+(\d+(?:\.\d+)?)",
                r"trade\s+(\w+/\w+)\s+(buy|sell)\s+(\d+(?:\.\d+)?)",
                r"execute\s+(buy|sell)\s+(\w+/\w+)\s+(\d+(?:\.\d+)?)"
            ],
            CommandType.MARKET_INSIGHT: [
                r"market\s+insight",
                r"market\s+analysis",
                r"what's\s+happening\s+in\s+the\s+market",
                r"market\s+trends",
                r"market\s+overview"
            ],
            CommandType.STRATEGY_ACTIVATION: [
                r"activate\s+strategy\s+(\w+)",
                r"start\s+strategy\s+(\w+)",
                r"enable\s+(\w+)\s+strategy",
                r"run\s+strategy\s+(\w+)"
            ],
            CommandType.VISUALIZATION: [
                r"show\s+chart",
                r"visualize\s+(\w+/\w+)",
                r"display\s+(\w+/\w+)",
                r"chart\s+view",
                r"visual\s+analysis"
            ],
            CommandType.SYSTEM_STATUS: [
                r"system\s+status",
                r"status\s+check",
                r"how\s+is\s+the\s+system",
                r"system\s+health",
                r"performance\s+status"
            ],
            CommandType.DATA_ANALYSIS: [
                r"analyze\s+data",
                r"data\s+analysis",
                r"signal\s+analysis",
                r"pattern\s+recognition",
                r"market\s+signals"
            ]
        }
        
        # Initialize Flask app for bridge
        self.app = Flask(__name__)
        CORS(self.app)
        self._setup_routes()
        
        # Bridge state
        self.bridge_running = False
        self.conversation_history: List[Dict[str, Any]] = []
        
        logger.info("🔗 Schwabot AI Bridge initialized")
    
    def _setup_routes(self):
        """Setup Flask routes for the bridge."""
        
        @self.app.route('/bridge/chat', methods=['POST'])
        def chat_endpoint():
            """Handle chat messages and route to appropriate trading functions."""
            try:
                data = request.get_json()
                user_message = data.get('message', '')
                
                # Process the message through our bridge
                response = asyncio.run(self._process_user_message(user_message))
                
                return jsonify({
                    'response': response,
                    'timestamp': datetime.now().isoformat(),
                    'command_executed': True
                })
                
            except Exception as e:
                logger.error(f"❌ Chat endpoint error: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/bridge/status', methods=['GET'])
        def status_endpoint():
            """Get bridge and system status."""
            try:
                status = asyncio.run(self._get_system_status())
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"❌ Status endpoint error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/bridge/execute', methods=['POST'])
        def execute_endpoint():
            """Execute trading commands directly."""
            try:
                data = request.get_json()
                command_type = data.get('command_type')
                parameters = data.get('parameters', {})
                
                result = asyncio.run(self._execute_command(command_type, parameters))
                
                return jsonify({
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"❌ Execute endpoint error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/bridge/visualize', methods=['POST'])
        def visualize_endpoint():
            """Generate visualizations."""
            try:
                data = request.get_json()
                symbol = data.get('symbol', 'BTC/USD')
                timeframe = data.get('timeframe', '1h')
                
                result = asyncio.run(self._generate_visualization(symbol, timeframe))
                
                return jsonify({
                    'visualization': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"❌ Visualize endpoint error: {e}")
                return jsonify({'error': str(e)}), 500
    
    async def _process_user_message(self, message: str) -> str:
        """Process user message and execute appropriate trading functions."""
        try:
            # Add to conversation history
            self.conversation_history.append({
                'user': message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Detect command type
            command = await self._detect_command(message)
            
            if command:
                # Execute the command
                response = await self._execute_ai_command(command)
                
                # Add response to history
                self.conversation_history.append({
                    'ai': response,
                    'timestamp': datetime.now().isoformat(),
                    'command_type': command.command_type.value
                })
                
                return response
            else:
                # Fallback to general AI response
                response = await self._get_general_ai_response(message)
                
                # Add response to history
                self.conversation_history.append({
                    'ai': response,
                    'timestamp': datetime.now().isoformat(),
                    'command_type': 'general'
                })
                
                return response
                
        except Exception as e:
            logger.error(f"❌ Message processing error: {e}")
            return f"I encountered an error processing your request: {str(e)}"
    
    async def _detect_command(self, message: str) -> Optional[AICommand]:
        """Detect what type of command the user is requesting."""
        message_lower = message.lower()
        
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower)
                if match:
                    # Extract parameters based on command type
                    parameters = await self._extract_parameters(command_type, match, message)
                    
                    return AICommand(
                        command_type=command_type,
                        parameters=parameters,
                        confidence=0.8,
                        user_message=message
                    )
        
        return None
    
    async def _extract_parameters(self, command_type: CommandType, match, message: str) -> Dict[str, Any]:
        """Extract parameters from the matched command."""
        parameters = {}
        
        if command_type == CommandType.TRADING_ANALYSIS:
            if match.groups():
                parameters['symbol'] = match.group(1).upper()
        
        elif command_type == CommandType.EXECUTE_TRADE:
            if len(match.groups()) >= 3:
                parameters['action'] = match.group(2).lower()
                parameters['symbol'] = match.group(1).upper()
                parameters['amount'] = float(match.group(3))
            elif len(match.groups()) >= 2:
                parameters['action'] = match.group(1).lower()
                parameters['symbol'] = match.group(2).upper()
                parameters['amount'] = float(match.group(3)) if len(match.groups()) > 2 else 1.0
        
        elif command_type == CommandType.STRATEGY_ACTIVATION:
            if match.groups():
                parameters['strategy_name'] = match.group(1)
        
        elif command_type == CommandType.VISUALIZATION:
            if match.groups():
                parameters['symbol'] = match.group(1).upper()
        
        return parameters
    
    async def _execute_ai_command(self, command: AICommand) -> str:
        """Execute the AI command and return response."""
        try:
            if command.command_type == CommandType.TRADING_ANALYSIS:
                return await self._execute_trading_analysis(command.parameters)
            
            elif command.command_type == CommandType.PORTFOLIO_STATUS:
                return await self._execute_portfolio_status()
            
            elif command.command_type == CommandType.EXECUTE_TRADE:
                return await self._execute_trade(command.parameters)
            
            elif command.command_type == CommandType.MARKET_INSIGHT:
                return await self._execute_market_insight()
            
            elif command.command_type == CommandType.STRATEGY_ACTIVATION:
                return await self._execute_strategy_activation(command.parameters)
            
            elif command.command_type == CommandType.VISUALIZATION:
                return await self._execute_visualization(command.parameters)
            
            elif command.command_type == CommandType.SYSTEM_STATUS:
                return await self._execute_system_status()
            
            elif command.command_type == CommandType.DATA_ANALYSIS:
                return await self._execute_data_analysis()
            
            else:
                return "I'm not sure how to handle that command yet."
                
        except Exception as e:
            logger.error(f"❌ Command execution error: {e}")
            return f"I encountered an error executing that command: {str(e)}"
    
    async def _execute_trading_analysis(self, parameters: Dict[str, Any]) -> str:
        """Execute trading analysis for a symbol."""
        try:
            symbol = parameters.get('symbol', 'BTC/USD')
            
            # Get real-time data
            tick_data = await self.tick_loader.get_latest_ticks(symbol, limit=1000)
            
            if not tick_data:
                return f"I couldn't find recent data for {symbol}. Please check if the symbol is correct."
            
            # Perform AI analysis
            analysis_request = self.schwabot_integration.SchwabotRequest(
                prompt=f"Analyze the trading data for {symbol}. Provide technical analysis, trend direction, support/resistance levels, and trading recommendations.",
                max_length=512,
                temperature=0.7,
                analysis_type=AnalysisType.TECHNICAL_ANALYSIS
            )
            
            # Add market data context
            analysis_request.prompt += f"\n\nMarket Data Context:\n"
            analysis_request.prompt += f"Current Price: ${tick_data[-1]['price']:.2f}\n"
            analysis_request.prompt += f"24h Change: {((tick_data[-1]['price'] - tick_data[0]['price']) / tick_data[0]['price'] * 100):.2f}%\n"
            analysis_request.prompt += f"Volume: {tick_data[-1]['volume']:.2f}\n"
            
            # Get AI analysis
            analysis_response = await self.schwabot_integration.analyze_trading_data(analysis_request)
            
            if analysis_response:
                # Generate visualization
                await self._generate_visualization(symbol, '1h')
                
                return f"📊 **Analysis for {symbol}**\n\n{analysis_response.text}\n\n📈 I've also generated a chart for you to visualize the data."
            else:
                return f"I couldn't complete the analysis for {symbol}. Please try again."
                
        except Exception as e:
            logger.error(f"❌ Trading analysis error: {e}")
            return f"Error performing trading analysis: {str(e)}"
    
    async def _execute_portfolio_status(self) -> str:
        """Get current portfolio status."""
        try:
            # Get portfolio data from registry
            portfolio_data = await self.registry_writer.get_portfolio_data()
            
            if portfolio_data:
                total_value = portfolio_data.get('total_value', 0.0)
                positions = portfolio_data.get('positions', [])
                
                response = f"💼 **Portfolio Status**\n\n"
                response += f"Total Value: ${total_value:,.2f}\n"
                response += f"Active Positions: {len(positions)}\n\n"
                
                if positions:
                    response += "**Current Positions:**\n"
                    for position in positions[:5]:  # Show top 5
                        symbol = position.get('symbol', 'Unknown')
                        amount = position.get('amount', 0.0)
                        value = position.get('value', 0.0)
                        pnl = position.get('pnl', 0.0)
                        
                        response += f"• {symbol}: {amount:.4f} (${value:.2f}) | P&L: ${pnl:.2f}\n"
                
                return response
            else:
                return "I couldn't retrieve your portfolio data. The system might not be connected to your trading account."
                
        except Exception as e:
            logger.error(f"❌ Portfolio status error: {e}")
            return f"Error retrieving portfolio status: {str(e)}"
    
    async def _execute_trade(self, parameters: Dict[str, Any]) -> str:
        """Execute a trade."""
        try:
            action = parameters.get('action', 'buy')
            symbol = parameters.get('symbol', 'BTC/USD')
            amount = parameters.get('amount', 1.0)
            
            # Validate trade parameters
            if amount <= 0:
                return "Invalid trade amount. Please specify a positive amount."
            
            # Execute trade through unified interface
            trade_result = await self.unified_interface.execute_trade(
                symbol=symbol,
                action=action,
                amount=amount
            )
            
            if trade_result.get('success'):
                return f"✅ **Trade Executed Successfully**\n\n"
                return f"Action: {action.upper()}\n"
                return f"Symbol: {symbol}\n"
                return f"Amount: {amount}\n"
                return f"Price: ${trade_result.get('price', 0):.2f}\n"
                return f"Total: ${trade_result.get('total', 0):.2f}"
            else:
                return f"❌ Trade failed: {trade_result.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return f"Error executing trade: {str(e)}"
    
    async def _execute_market_insight(self) -> str:
        """Get market insights."""
        try:
            # Get market data for major symbols
            symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD']
            market_data = {}
            
            for symbol in symbols:
                ticks = await self.tick_loader.get_latest_ticks(symbol, limit=100)
                if ticks:
                    current_price = ticks[-1]['price']
                    prev_price = ticks[0]['price']
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    market_data[symbol] = {
                        'price': current_price,
                        'change_pct': change_pct,
                        'volume': ticks[-1]['volume']
                    }
            
            # Generate market insight
            insight_request = self.schwabot_integration.SchwabotRequest(
                prompt=f"Provide a brief market insight based on this data: {json.dumps(market_data, indent=2)}",
                max_length=256,
                temperature=0.7,
                analysis_type=AnalysisType.MARKET_ANALYSIS
            )
            
            insight_response = await self.schwabot_integration.analyze_trading_data(insight_request)
            
            if insight_response:
                response = "🌍 **Market Insights**\n\n"
                response += insight_response.text + "\n\n"
                response += "**Current Market Data:**\n"
                
                for symbol, data in market_data.items():
                    response += f"• {symbol}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)\n"
                
                return response
            else:
                return "I couldn't generate market insights at the moment. Please try again."
                
        except Exception as e:
            logger.error(f"❌ Market insight error: {e}")
            return f"Error generating market insights: {str(e)}"
    
    async def _execute_strategy_activation(self, parameters: Dict[str, Any]) -> str:
        """Activate a trading strategy."""
        try:
            strategy_name = parameters.get('strategy_name', '')
            
            if not strategy_name:
                return "Please specify which strategy you'd like to activate."
            
            # Activate strategy through unified interface
            result = await self.unified_interface.activate_strategy(strategy_name)
            
            if result.get('success'):
                return f"✅ **Strategy Activated**\n\nStrategy: {strategy_name}\nStatus: Active\n\nI'll now monitor the market and execute trades based on this strategy's signals."
            else:
                return f"❌ Strategy activation failed: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"❌ Strategy activation error: {e}")
            return f"Error activating strategy: {str(e)}"
    
    async def _execute_visualization(self, parameters: Dict[str, Any]) -> str:
        """Generate visualization for a symbol."""
        try:
            symbol = parameters.get('symbol', 'BTC/USD')
            
            # Generate chart
            chart_result = await self._generate_visualization(symbol, '1h')
            
            if chart_result:
                return f"📈 **Chart Generated**\n\nI've created a chart for {symbol} and saved it to your visualizations folder.\n\nChart includes:\n• Price action\n• Technical indicators\n• Support/resistance levels\n• Trading signals"
            else:
                return f"I couldn't generate a chart for {symbol}. Please try again."
                
        except Exception as e:
            logger.error(f"❌ Visualization error: {e}")
            return f"Error generating visualization: {str(e)}"
    
    async def _execute_system_status(self) -> str:
        """Get system status."""
        try:
            status = await self._get_system_status()
            
            response = "🔧 **System Status**\n\n"
            response += f"Bridge Status: {'🟢 Running' if self.bridge_running else '🔴 Stopped'}\n"
            response += f"Schwabot AI: {'🟢 Connected' if status['schwabot_status'] else '🔴 Disconnected'}\n"
            response += f"Trading System: {'🟢 Active' if status['trading_status'] else '🔴 Inactive'}\n"
            response += f"Visual Layer: {'🟢 Active' if status['visual_status'] else '🔴 Inactive'}\n"
            response += f"Memory Stack: {'🟢 Active' if status['memory_status'] else '🔴 Inactive'}\n\n"
            response += f"Uptime: {status['uptime']}\n"
            response += f"Total Commands: {status['total_commands']}\n"
            response += f"Active Strategies: {status['active_strategies']}"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ System status error: {e}")
            return f"Error retrieving system status: {str(e)}"
    
    async def _execute_data_analysis(self) -> str:
        """Perform data analysis."""
        try:
            # Get recent signals
            signals = await self.signal_cache.get_recent_signals(limit=10)
            
            if signals:
                response = "📊 **Data Analysis**\n\n"
                response += f"Recent Signals: {len(signals)}\n\n"
                
                for signal in signals[:5]:
                    symbol = signal.get('symbol', 'Unknown')
                    signal_type = signal.get('type', 'Unknown')
                    confidence = signal.get('confidence', 0.0)
                    timestamp = signal.get('timestamp', 'Unknown')
                    
                    response += f"• {symbol}: {signal_type} (Confidence: {confidence:.2f})\n"
                
                response += "\nI'm continuously analyzing market data for trading opportunities."
                return response
            else:
                return "No recent signals found. The system is monitoring for new opportunities."
                
        except Exception as e:
            logger.error(f"❌ Data analysis error: {e}")
            return f"Error performing data analysis: {str(e)}"
    
    async def _get_general_ai_response(self, message: str) -> str:
        """Get a general AI response when no specific command is detected."""
        try:
            # Create a context-aware prompt
            context = f"User message: {message}\n\n"
            context += "You are an AI trading assistant. Provide helpful, accurate information about trading, markets, or the Schwabot system. "
            context += "If the user is asking about trading, provide insights. If they're asking about the system, explain how it works. "
            context += "Keep responses concise and informative."
            
            request = self.schwabot_integration.SchwabotRequest(
                prompt=context,
                max_length=256,
                temperature=0.7,
                analysis_type=AnalysisType.GENERAL
            )
            
            response = await self.schwabot_integration.analyze_trading_data(request)
            
            if response:
                return response.text
            else:
                return "I'm here to help with trading and market analysis. You can ask me to analyze specific symbols, check your portfolio, execute trades, or get market insights."
                
        except Exception as e:
            logger.error(f"❌ General AI response error: {e}")
            return "I'm here to help with your trading needs. You can ask me to analyze markets, check your portfolio, or execute trades."
    
    async def _generate_visualization(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Generate visualization for a symbol."""
        try:
            # Get tick data
            tick_data = await self.tick_loader.get_latest_ticks(symbol, limit=1000)
            
            if not tick_data:
                return None
            
            # Generate chart using visual controller
            chart_result = await self.visual_controller.generate_price_chart(
                tick_data=tick_data,
                symbol=symbol,
                timeframe=timeframe
            )
            
            if chart_result:
                # Save visualization
                await self.visual_controller.save_visualization(chart_result)
                
                return {
                    'success': True,
                    'chart_path': chart_result.get('file_path', ''),
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Visualization generation error: {e}")
            return None
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Check Schwabot AI status
            schwabot_status = False
            try:
                response = requests.get(f"{self.schwabot_url}/api/extra/version", timeout=5)
                schwabot_status = response.status_code == 200
            except:
                pass
            
            # Get unified interface status
            unified_status = self.unified_interface.get_unified_status()
            
            return {
                'bridge_status': self.bridge_running,
                'schwabot_status': schwabot_status,
                'trading_status': unified_status.get('trading_system', {}).get('status', 'unknown'),
                'visual_status': unified_status.get('visual_layer', {}).get('status', 'unknown'),
                'memory_status': unified_status.get('memory_stack', {}).get('status', 'unknown'),
                'uptime': f"{unified_status.get('uptime_seconds', 0):.0f} seconds",
                'total_commands': len(self.conversation_history),
                'active_strategies': len(self.trading_context.active_strategies),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ System status error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_command(self, command_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command directly."""
        try:
            if command_type == 'trading_analysis':
                response = await self._execute_trading_analysis(parameters)
                return {'success': True, 'response': response}
            
            elif command_type == 'portfolio_status':
                response = await self._execute_portfolio_status()
                return {'success': True, 'response': response}
            
            elif command_type == 'execute_trade':
                response = await self._execute_trade(parameters)
                return {'success': True, 'response': response}
            
            elif command_type == 'market_insight':
                response = await self._execute_market_insight()
                return {'success': True, 'response': response}
            
            elif command_type == 'visualization':
                result = await self._generate_visualization(
                    parameters.get('symbol', 'BTC/USD'),
                    parameters.get('timeframe', '1h')
                )
                return {'success': True, 'result': result}
            
            else:
                return {'success': False, 'error': f'Unknown command type: {command_type}'}
                
        except Exception as e:
            logger.error(f"❌ Direct command execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def start_bridge(self):
        """Start the bridge server."""
        try:
            if self.bridge_running:
                logger.info("✅ Bridge already running")
                return
            
            # Start the Flask app
            self.app.run(
                host='0.0.0.0',
                port=self.bridge_port,
                debug=False,
                threaded=True
            )
            
            self.bridge_running = True
            logger.info(f"🚀 Schwabot AI Bridge started on port {self.bridge_port}")
            
        except Exception as e:
            logger.error(f"❌ Bridge startup failed: {e}")
    
    def stop_bridge(self):
        """Stop the bridge server."""
        try:
            self.bridge_running = False
            logger.info("🛑 Schwabot AI Bridge stopped")
            
        except Exception as e:
            logger.error(f"❌ Bridge shutdown error: {e}")

# Global bridge instance
_bridge = None

def get_bridge() -> Schwabot AIBridge:
    """Get the global bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = Schwabot AIBridge()
    return _bridge

def start_bridge():
    """Start the global bridge."""
    bridge = get_bridge()
    bridge.start_bridge()

def stop_bridge():
    """Stop the global bridge."""
    global _bridge
    if _bridge:
        _bridge.stop_bridge()
        _bridge = None 