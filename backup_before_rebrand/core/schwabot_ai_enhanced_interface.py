#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KoboldCPP Enhanced Interface - Schwabot Integration Extension
============================================================

This module extends KoboldCPP's existing Flask/HTTP interface to integrate
Schwabot's trading functionality directly into the conversation flow.

The enhanced interface provides:
- Trading commands through natural language
- Real-time market data integration
- Portfolio management via chat
- Strategy activation and monitoring
- Visual chart generation
- System status and health monitoring
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
from flask import Flask, request, jsonify, render_template_string, Response
from flask_cors import CORS

# Import our bridge and unified system
from .koboldcpp_bridge import KoboldCPPBridge, CommandType, AICommand
from .schwabot_unified_interface import SchwabotUnifiedInterface, InterfaceMode

logger = logging.getLogger(__name__)

class EnhancedMode(Enum):
    """Enhanced interface modes."""
    CONVERSATION_ONLY = "conversation_only"
    TRADING_ENABLED = "trading_enabled"
    FULL_INTEGRATION = "full_integration"
    VISUAL_LAYER = "visual_layer"

@dataclass
class ConversationContext:
    """Context for ongoing conversations."""
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    trading_context: Dict[str, Any] = field(default_factory=dict)
    last_command: Optional[AICommand] = None
    created_at: datetime = field(default_factory=datetime.now)

class KoboldCPPEnhancedInterface:
    """Enhanced interface that extends KoboldCPP's existing functionality."""
    
    def __init__(self, kobold_port: int = 5001, enhanced_port: int = 5006):
        """Initialize the enhanced interface."""
        self.kobold_port = kobold_port
        self.enhanced_port = enhanced_port
        self.kobold_url = f"http://localhost:{kobold_port}"
        
        # Initialize bridge
        self.bridge = KoboldCPPBridge(kobold_port=kobold_port, bridge_port=5005)
        
        # Initialize unified system
        self.unified_interface = SchwabotUnifiedInterface(InterfaceMode.FULL_INTEGRATION)
        
        # Enhanced interface state
        self.enhanced_mode = EnhancedMode.FULL_INTEGRATION
        self.conversations: Dict[str, ConversationContext] = {}
        self.enhanced_running = False
        
        # Initialize Flask app for enhanced interface
        self.app = Flask(__name__)
        CORS(self.app)
        self._setup_enhanced_routes()
        
        # Trading command patterns for enhanced recognition
        self.enhanced_patterns = {
            'price_check': [
                r"what's\s+the\s+price\s+of\s+(\w+/\w+)",
                r"price\s+of\s+(\w+/\w+)",
                r"how\s+much\s+is\s+(\w+/\w+)",
                r"(\w+/\w+)\s+price"
            ],
            'portfolio_summary': [
                r"portfolio\s+summary",
                r"my\s+portfolio",
                r"show\s+my\s+holdings",
                r"what\s+do\s+i\s+own"
            ],
            'market_overview': [
                r"market\s+overview",
                r"what's\s+happening\s+in\s+the\s+market",
                r"market\s+summary",
                r"crypto\s+market"
            ],
            'strategy_status': [
                r"strategy\s+status",
                r"active\s+strategies",
                r"what\s+strategies\s+are\s+running",
                r"strategy\s+performance"
            ],
            'risk_assessment': [
                r"risk\s+assessment",
                r"how\s+risky\s+is\s+(\w+/\w+)",
                r"risk\s+level",
                r"market\s+risk"
            ],
            'performance_metrics': [
                r"performance",
                r"how\s+am\s+i\s+doing",
                r"trading\s+performance",
                r"profit\s+loss"
            ]
        }
        
        logger.info("🔧 KoboldCPP Enhanced Interface initialized")
    
    def _setup_enhanced_routes(self):
        """Setup enhanced Flask routes that extend KoboldCPP's functionality."""
        
        @self.app.route('/enhanced/chat', methods=['POST'])
        def enhanced_chat_endpoint():
            """Enhanced chat endpoint with trading integration."""
            try:
                data = request.get_json()
                user_message = data.get('message', '')
                session_id = data.get('session_id', 'default')
                user_id = data.get('user_id', 'anonymous')
                
                # Get or create conversation context
                if session_id not in self.conversations:
                    self.conversations[session_id] = ConversationContext(
                        session_id=session_id,
                        user_id=user_id
                    )
                
                context = self.conversations[session_id]
                
                # Process message through enhanced interface
                response = asyncio.run(self._process_enhanced_message(user_message, context))
                
                return jsonify({
                    'response': response['text'],
                    'command_executed': response['command_executed'],
                    'visualization': response.get('visualization'),
                    'trading_data': response.get('trading_data'),
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session_id
                })
                
            except Exception as e:
                logger.error(f"❌ Enhanced chat endpoint error: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/enhanced/stream', methods=['POST'])
        def enhanced_stream_endpoint():
            """Streaming chat endpoint for real-time responses."""
            try:
                data = request.get_json()
                user_message = data.get('message', '')
                session_id = data.get('session_id', 'default')
                
                def generate():
                    """Generate streaming response."""
                    try:
                        # Process message
                        context = self.conversations.get(session_id)
                        if not context:
                            context = ConversationContext(session_id=session_id, user_id='anonymous')
                            self.conversations[session_id] = context
                        
                        response = asyncio.run(self._process_enhanced_message(user_message, context))
                        
                        # Stream the response
                        yield f"data: {json.dumps({'type': 'response', 'data': response})}\n\n"
                        
                        # If there's trading data, stream it separately
                        if response.get('trading_data'):
                            yield f"data: {json.dumps({'type': 'trading_data', 'data': response['trading_data']})}\n\n"
                        
                        # If there's visualization, stream it
                        if response.get('visualization'):
                            yield f"data: {json.dumps({'type': 'visualization', 'data': response['visualization']})}\n\n"
                        
                        yield f"data: {json.dumps({'type': 'end'})}\n\n"
                        
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                
                return Response(generate(), mimetype='text/plain')
                
            except Exception as e:
                logger.error(f"❌ Enhanced stream endpoint error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/enhanced/trading/execute', methods=['POST'])
        def trading_execute_endpoint():
            """Execute trading commands directly."""
            try:
                data = request.get_json()
                command_type = data.get('command_type')
                parameters = data.get('parameters', {})
                session_id = data.get('session_id', 'default')
                
                result = asyncio.run(self._execute_trading_command(command_type, parameters, session_id))
                
                return jsonify({
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"❌ Trading execute endpoint error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/enhanced/trading/status', methods=['GET'])
        def trading_status_endpoint():
            """Get trading system status."""
            try:
                status = asyncio.run(self._get_enhanced_status())
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"❌ Trading status endpoint error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/enhanced/visualize', methods=['POST'])
        def enhanced_visualize_endpoint():
            """Generate enhanced visualizations."""
            try:
                data = request.get_json()
                symbol = data.get('symbol', 'BTC/USD')
                timeframe = data.get('timeframe', '1h')
                chart_type = data.get('chart_type', 'price')
                
                result = asyncio.run(self._generate_enhanced_visualization(symbol, timeframe, chart_type))
                
                return jsonify({
                    'visualization': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"❌ Enhanced visualize endpoint error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/enhanced/portfolio', methods=['GET'])
        def portfolio_endpoint():
            """Get portfolio information."""
            try:
                session_id = request.args.get('session_id', 'default')
                portfolio_data = asyncio.run(self._get_portfolio_data(session_id))
                
                return jsonify({
                    'portfolio': portfolio_data,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"❌ Portfolio endpoint error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/enhanced/strategies', methods=['GET'])
        def strategies_endpoint():
            """Get available and active strategies."""
            try:
                strategies_data = asyncio.run(self._get_strategies_data())
                
                return jsonify({
                    'strategies': strategies_data,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"❌ Strategies endpoint error: {e}")
                return jsonify({'error': str(e)}), 500
    
    async def _process_enhanced_message(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Process message through enhanced interface with trading integration."""
        try:
            # Add to conversation history
            context.conversation_history.append({
                'user': message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Check for enhanced patterns first
            enhanced_response = await self._check_enhanced_patterns(message, context)
            if enhanced_response:
                return enhanced_response
            
            # Check for trading commands
            trading_response = await self._check_trading_commands(message, context)
            if trading_response:
                return trading_response
            
            # Fallback to bridge processing
            bridge_response = await self.bridge._process_user_message(message)
            
            # Add response to history
            context.conversation_history.append({
                'ai': bridge_response,
                'timestamp': datetime.now().isoformat(),
                'source': 'bridge'
            })
            
            return {
                'text': bridge_response,
                'command_executed': True,
                'source': 'bridge'
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced message processing error: {e}")
            return {
                'text': f"I encountered an error processing your request: {str(e)}",
                'command_executed': False,
                'error': str(e)
            }
    
    async def _check_enhanced_patterns(self, message: str, context: ConversationContext) -> Optional[Dict[str, Any]]:
        """Check for enhanced pattern matches."""
        message_lower = message.lower()
        
        for pattern_type, patterns in self.enhanced_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower)
                if match:
                    return await self._handle_enhanced_pattern(pattern_type, match, message, context)
        
        return None
    
    async def _handle_enhanced_pattern(self, pattern_type: str, match, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle enhanced pattern matches."""
        try:
            if pattern_type == 'price_check':
                symbol = match.group(1).upper()
                price_data = await self._get_current_price(symbol)
                
                response_text = f"💰 **{symbol} Price**\n\n"
                response_text += f"Current Price: ${price_data['price']:.2f}\n"
                response_text += f"24h Change: {price_data['change_24h']:+.2f}%\n"
                response_text += f"Volume: {price_data['volume']:.2f}\n"
                
                return {
                    'text': response_text,
                    'command_executed': True,
                    'trading_data': price_data,
                    'source': 'enhanced_price_check'
                }
            
            elif pattern_type == 'portfolio_summary':
                portfolio_data = await self._get_portfolio_data(context.session_id)
                
                response_text = "💼 **Portfolio Summary**\n\n"
                response_text += f"Total Value: ${portfolio_data['total_value']:,.2f}\n"
                response_text += f"Active Positions: {len(portfolio_data['positions'])}\n"
                response_text += f"Total P&L: ${portfolio_data['total_pnl']:,.2f}\n"
                
                if portfolio_data['positions']:
                    response_text += "\n**Top Holdings:**\n"
                    for position in portfolio_data['positions'][:3]:
                        response_text += f"• {position['symbol']}: {position['amount']:.4f} (${position['value']:.2f})\n"
                
                return {
                    'text': response_text,
                    'command_executed': True,
                    'trading_data': portfolio_data,
                    'source': 'enhanced_portfolio'
                }
            
            elif pattern_type == 'market_overview':
                market_data = await self._get_market_overview()
                
                response_text = "🌍 **Market Overview**\n\n"
                response_text += market_data['summary'] + "\n\n"
                response_text += "**Top Movers:**\n"
                
                for mover in market_data['top_movers']:
                    response_text += f"• {mover['symbol']}: {mover['change']:+.2f}%\n"
                
                return {
                    'text': response_text,
                    'command_executed': True,
                    'trading_data': market_data,
                    'source': 'enhanced_market'
                }
            
            elif pattern_type == 'strategy_status':
                strategies_data = await self._get_strategies_data()
                
                response_text = "📈 **Strategy Status**\n\n"
                response_text += f"Active Strategies: {len(strategies_data['active'])}\n"
                response_text += f"Available Strategies: {len(strategies_data['available'])}\n\n"
                
                if strategies_data['active']:
                    response_text += "**Active Strategies:**\n"
                    for strategy in strategies_data['active']:
                        response_text += f"• {strategy['name']}: {strategy['status']}\n"
                
                return {
                    'text': response_text,
                    'command_executed': True,
                    'trading_data': strategies_data,
                    'source': 'enhanced_strategies'
                }
            
            elif pattern_type == 'risk_assessment':
                if match.groups():
                    symbol = match.group(1).upper()
                    risk_data = await self._get_risk_assessment(symbol)
                else:
                    risk_data = await self._get_portfolio_risk()
                
                response_text = "⚠️ **Risk Assessment**\n\n"
                response_text += risk_data['summary'] + "\n\n"
                response_text += f"Risk Level: {risk_data['risk_level']}\n"
                response_text += f"Volatility: {risk_data['volatility']:.2f}%\n"
                
                return {
                    'text': response_text,
                    'command_executed': True,
                    'trading_data': risk_data,
                    'source': 'enhanced_risk'
                }
            
            elif pattern_type == 'performance_metrics':
                performance_data = await self._get_performance_metrics(context.session_id)
                
                response_text = "📊 **Performance Metrics**\n\n"
                response_text += f"Total Return: {performance_data['total_return']:+.2f}%\n"
                response_text += f"Win Rate: {performance_data['win_rate']:.1f}%\n"
                response_text += f"Average Trade: ${performance_data['avg_trade']:.2f}\n"
                response_text += f"Sharpe Ratio: {performance_data['sharpe_ratio']:.2f}\n"
                
                return {
                    'text': response_text,
                    'command_executed': True,
                    'trading_data': performance_data,
                    'source': 'enhanced_performance'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Enhanced pattern handling error: {e}")
            return {
                'text': f"I encountered an error processing that request: {str(e)}",
                'command_executed': False,
                'error': str(e)
            }
    
    async def _check_trading_commands(self, message: str, context: ConversationContext) -> Optional[Dict[str, Any]]:
        """Check for trading-specific commands."""
        try:
            # Use bridge command detection
            command = await self.bridge._detect_command(message)
            
            if command:
                # Execute through bridge
                response = await self.bridge._execute_ai_command(command)
                
                # Add to context
                context.last_command = command
                context.conversation_history.append({
                    'ai': response,
                    'timestamp': datetime.now().isoformat(),
                    'command_type': command.command_type.value
                })
                
                return {
                    'text': response,
                    'command_executed': True,
                    'source': 'trading_command'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Trading command check error: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for a symbol."""
        try:
            # Get latest tick data
            tick_data = await self.bridge.tick_loader.get_latest_ticks(symbol, limit=2)
            
            if tick_data and len(tick_data) >= 2:
                current_price = tick_data[-1]['price']
                prev_price = tick_data[0]['price']
                change_24h = ((current_price - prev_price) / prev_price) * 100
                volume = tick_data[-1]['volume']
                
                return {
                    'symbol': symbol,
                    'price': current_price,
                    'change_24h': change_24h,
                    'volume': volume,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'symbol': symbol,
                    'price': 0.0,
                    'change_24h': 0.0,
                    'volume': 0.0,
                    'error': 'No data available'
                }
                
        except Exception as e:
            logger.error(f"❌ Get current price error: {e}")
            return {
                'symbol': symbol,
                'price': 0.0,
                'change_24h': 0.0,
                'volume': 0.0,
                'error': str(e)
            }
    
    async def _get_portfolio_data(self, session_id: str) -> Dict[str, Any]:
        """Get portfolio data for a session."""
        try:
            # Get portfolio from registry
            portfolio_data = await self.bridge.registry_writer.get_portfolio_data()
            
            if portfolio_data:
                total_value = portfolio_data.get('total_value', 0.0)
                positions = portfolio_data.get('positions', [])
                total_pnl = sum(pos.get('pnl', 0.0) for pos in positions)
                
                return {
                    'total_value': total_value,
                    'total_pnl': total_pnl,
                    'positions': positions,
                    'position_count': len(positions),
                    'session_id': session_id
                }
            else:
                return {
                    'total_value': 0.0,
                    'total_pnl': 0.0,
                    'positions': [],
                    'position_count': 0,
                    'session_id': session_id,
                    'error': 'No portfolio data available'
                }
                
        except Exception as e:
            logger.error(f"❌ Get portfolio data error: {e}")
            return {
                'total_value': 0.0,
                'total_pnl': 0.0,
                'positions': [],
                'position_count': 0,
                'session_id': session_id,
                'error': str(e)
            }
    
    async def _get_market_overview(self) -> Dict[str, Any]:
        """Get market overview data."""
        try:
            # Get data for major symbols
            symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD']
            market_data = {}
            top_movers = []
            
            for symbol in symbols:
                price_data = await self._get_current_price(symbol)
                if 'error' not in price_data:
                    market_data[symbol] = price_data
                    top_movers.append({
                        'symbol': symbol,
                        'change': price_data['change_24h']
                    })
            
            # Sort by absolute change
            top_movers.sort(key=lambda x: abs(x['change']), reverse=True)
            
            # Generate summary using AI
            summary_request = self.bridge.kobold_integration.KoboldRequest(
                prompt=f"Provide a brief market summary based on this data: {json.dumps(market_data, indent=2)}",
                max_length=200,
                temperature=0.7,
                analysis_type=self.bridge.kobold_integration.AnalysisType.MARKET_ANALYSIS
            )
            
            summary_response = await self.bridge.kobold_integration.analyze_trading_data(summary_request)
            summary = summary_response.text if summary_response else "Market data is currently being analyzed."
            
            return {
                'summary': summary,
                'top_movers': top_movers[:5],
                'market_data': market_data
            }
            
        except Exception as e:
            logger.error(f"❌ Get market overview error: {e}")
            return {
                'summary': "Unable to retrieve market overview at this time.",
                'top_movers': [],
                'market_data': {},
                'error': str(e)
            }
    
    async def _get_strategies_data(self) -> Dict[str, Any]:
        """Get available and active strategies."""
        try:
            # Get strategies from unified interface
            strategies = await self.unified_interface.get_available_strategies()
            
            active_strategies = []
            available_strategies = []
            
            for strategy in strategies:
                if strategy.get('active', False):
                    active_strategies.append(strategy)
                else:
                    available_strategies.append(strategy)
            
            return {
                'active': active_strategies,
                'available': available_strategies,
                'total_active': len(active_strategies),
                'total_available': len(available_strategies)
            }
            
        except Exception as e:
            logger.error(f"❌ Get strategies data error: {e}")
            return {
                'active': [],
                'available': [],
                'total_active': 0,
                'total_available': 0,
                'error': str(e)
            }
    
    async def _get_risk_assessment(self, symbol: str) -> Dict[str, Any]:
        """Get risk assessment for a symbol."""
        try:
            # Get historical data for volatility calculation
            tick_data = await self.bridge.tick_loader.get_latest_ticks(symbol, limit=100)
            
            if tick_data:
                prices = [tick['price'] for tick in tick_data]
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                
                import statistics
                volatility = statistics.stdev(returns) * 100
                
                # Determine risk level
                if volatility < 2:
                    risk_level = "Low"
                elif volatility < 5:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
                return {
                    'symbol': symbol,
                    'risk_level': risk_level,
                    'volatility': volatility,
                    'summary': f"{symbol} shows {risk_level.lower()} risk with {volatility:.1f}% volatility."
                }
            else:
                return {
                    'symbol': symbol,
                    'risk_level': "Unknown",
                    'volatility': 0.0,
                    'summary': f"Insufficient data for {symbol} risk assessment."
                }
                
        except Exception as e:
            logger.error(f"❌ Get risk assessment error: {e}")
            return {
                'symbol': symbol,
                'risk_level': "Unknown",
                'volatility': 0.0,
                'summary': f"Error assessing risk for {symbol}.",
                'error': str(e)
            }
    
    async def _get_portfolio_risk(self) -> Dict[str, Any]:
        """Get overall portfolio risk assessment."""
        try:
            portfolio_data = await self._get_portfolio_data('default')
            
            if portfolio_data['positions']:
                # Calculate portfolio volatility
                total_value = portfolio_data['total_value']
                weighted_volatility = 0.0
                
                for position in portfolio_data['positions']:
                    symbol = position['symbol']
                    position_value = position['value']
                    weight = position_value / total_value if total_value > 0 else 0
                    
                    risk_data = await self._get_risk_assessment(symbol)
                    weighted_volatility += weight * risk_data['volatility']
                
                # Determine overall risk level
                if weighted_volatility < 2:
                    risk_level = "Low"
                elif weighted_volatility < 5:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
                return {
                    'risk_level': risk_level,
                    'volatility': weighted_volatility,
                    'summary': f"Portfolio shows {risk_level.lower()} risk with {weighted_volatility:.1f}% weighted volatility."
                }
            else:
                return {
                    'risk_level': "Low",
                    'volatility': 0.0,
                    'summary': "No active positions in portfolio."
                }
                
        except Exception as e:
            logger.error(f"❌ Get portfolio risk error: {e}")
            return {
                'risk_level': "Unknown",
                'volatility': 0.0,
                'summary': "Error assessing portfolio risk.",
                'error': str(e)
            }
    
    async def _get_performance_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get trading performance metrics."""
        try:
            # Get trading history from registry
            trading_history = await self.bridge.registry_writer.get_trading_history()
            
            if trading_history:
                total_trades = len(trading_history)
                winning_trades = len([t for t in trading_history if t.get('pnl', 0) > 0])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                total_pnl = sum(t.get('pnl', 0) for t in trading_history)
                avg_trade = total_pnl / total_trades if total_trades > 0 else 0
                
                # Calculate Sharpe ratio (simplified)
                returns = [t.get('pnl', 0) for t in trading_history]
                if returns:
                    import statistics
                    avg_return = statistics.mean(returns)
                    std_return = statistics.stdev(returns) if len(returns) > 1 else 0
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'total_return': total_pnl,
                    'avg_trade': avg_trade,
                    'sharpe_ratio': sharpe_ratio
                }
            else:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'win_rate': 0.0,
                    'total_return': 0.0,
                    'avg_trade': 0.0,
                    'sharpe_ratio': 0.0
                }
                
        except Exception as e:
            logger.error(f"❌ Get performance metrics error: {e}")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'avg_trade': 0.0,
                'sharpe_ratio': 0.0,
                'error': str(e)
            }
    
    async def _execute_trading_command(self, command_type: str, parameters: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute a trading command directly."""
        try:
            # Route to appropriate handler
            if command_type == 'buy' or command_type == 'sell':
                return await self._execute_trade_command(command_type, parameters, session_id)
            elif command_type == 'analyze':
                return await self._execute_analysis_command(parameters, session_id)
            elif command_type == 'strategy':
                return await self._execute_strategy_command(parameters, session_id)
            else:
                return {'success': False, 'error': f'Unknown command type: {command_type}'}
                
        except Exception as e:
            logger.error(f"❌ Execute trading command error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_trade_command(self, action: str, parameters: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute a trade command."""
        try:
            symbol = parameters.get('symbol', 'BTC/USD')
            amount = parameters.get('amount', 1.0)
            
            # Execute through unified interface
            result = await self.unified_interface.execute_trade(
                symbol=symbol,
                action=action,
                amount=amount
            )
            
            return {
                'success': result.get('success', False),
                'result': result,
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"❌ Execute trade command error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_analysis_command(self, parameters: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute an analysis command."""
        try:
            symbol = parameters.get('symbol', 'BTC/USD')
            
            # Get analysis through bridge
            response = await self.bridge._execute_trading_analysis({'symbol': symbol})
            
            return {
                'success': True,
                'response': response,
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"❌ Execute analysis command error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_strategy_command(self, parameters: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute a strategy command."""
        try:
            strategy_name = parameters.get('strategy_name', '')
            action = parameters.get('action', 'activate')
            
            if action == 'activate':
                result = await self.unified_interface.activate_strategy(strategy_name)
            elif action == 'deactivate':
                result = await self.unified_interface.deactivate_strategy(strategy_name)
            else:
                result = {'success': False, 'error': f'Unknown strategy action: {action}'}
            
            return {
                'success': result.get('success', False),
                'result': result,
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"❌ Execute strategy command error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_enhanced_visualization(self, symbol: str, timeframe: str, chart_type: str) -> Optional[Dict[str, Any]]:
        """Generate enhanced visualization."""
        try:
            # Use bridge visualization
            result = await self.bridge._generate_visualization(symbol, timeframe)
            
            if result:
                # Add enhanced metadata
                result['chart_type'] = chart_type
                result['enhanced'] = True
                result['generated_at'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced visualization error: {e}")
            return None
    
    async def _get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced interface status."""
        try:
            # Get bridge status
            bridge_status = await self.bridge._get_system_status()
            
            # Add enhanced interface status
            enhanced_status = {
                'enhanced_interface': {
                    'status': 'running' if self.enhanced_running else 'stopped',
                    'mode': self.enhanced_mode.value,
                    'port': self.enhanced_port,
                    'active_conversations': len(self.conversations)
                },
                'bridge': bridge_status,
                'timestamp': datetime.now().isoformat()
            }
            
            return enhanced_status
            
        except Exception as e:
            logger.error(f"❌ Get enhanced status error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def start_enhanced_interface(self):
        """Start the enhanced interface server."""
        try:
            if self.enhanced_running:
                logger.info("✅ Enhanced interface already running")
                return
            
            # Start the Flask app
            self.app.run(
                host='0.0.0.0',
                port=self.enhanced_port,
                debug=False,
                threaded=True
            )
            
            self.enhanced_running = True
            logger.info(f"🚀 KoboldCPP Enhanced Interface started on port {self.enhanced_port}")
            
        except Exception as e:
            logger.error(f"❌ Enhanced interface startup failed: {e}")
    
    def stop_enhanced_interface(self):
        """Stop the enhanced interface server."""
        try:
            self.enhanced_running = False
            logger.info("🛑 KoboldCPP Enhanced Interface stopped")
            
        except Exception as e:
            logger.error(f"❌ Enhanced interface shutdown error: {e}")

# Global enhanced interface instance
_enhanced_interface = None

def get_enhanced_interface() -> KoboldCPPEnhancedInterface:
    """Get the global enhanced interface instance."""
    global _enhanced_interface
    if _enhanced_interface is None:
        _enhanced_interface = KoboldCPPEnhancedInterface()
    return _enhanced_interface

def start_enhanced_interface():
    """Start the global enhanced interface."""
    interface = get_enhanced_interface()
    interface.start_enhanced_interface()

def stop_enhanced_interface():
    """Stop the global enhanced interface."""
    global _enhanced_interface
    if _enhanced_interface:
        _enhanced_interface.stop_enhanced_interface()
        _enhanced_interface = None 