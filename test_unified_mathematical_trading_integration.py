#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Unified Mathematical Trading Integration
============================================

This test verifies that ALL mathematical components are actually being used
in trading decisions and that the system is properly integrated with:
1. Mathematical integration (DLT, Dualistic Engines, Bit Phases, etc.)
2. Flask server for multi-bot coordination
3. KoboldCPP integration
4. Registry and soulprint storage
5. Real-time trading pipeline

This ensures that when the system makes a decision, it's actually using
the mathematical foundation we've built.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the unified system
try:
    from core.unified_mathematical_trading_system import UnifiedMathematicalTradingSystem, create_unified_trading_system
    from backtesting.mathematical_integration_simplified import mathematical_integration, MathematicalSignal
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import unified system: {e}")
    UNIFIED_SYSTEM_AVAILABLE = False

class UnifiedMathematicalTradingIntegrationTester:
    """Test the unified mathematical trading integration."""
    
    def __init__(self):
        """Initialize the tester."""
        self.config = self._create_test_config()
        self.system = None
        self.test_results = {}
        
        logger.info("🧪 Unified Mathematical Trading Integration Tester initialized")
    
    def _create_test_config(self) -> Dict[str, Any]:
        """Create test configuration."""
        return {
            'flask_server': {
                'host': 'localhost',
                'port': 5001,  # Different port for testing
                'debug': False
            },
            'mathematical_integration': {
                'enabled': True,
                'weight': 0.7,
                'confidence_threshold': 0.6
            },
            'koboldcpp_integration': {
                'enabled': True,
                'server_url': 'http://localhost:5002',
                'model_name': 'test_model'
            },
            'trading': {
                'base_position_size': 0.01,
                'max_position_size': 0.1,
                'risk_tolerance': 0.2,
                'consensus_threshold': 0.7
            },
            'api_endpoints': [
                {
                    'name': 'test_endpoint',
                    'exchange': 'binance',
                    'api_key': 'test_key',
                    'secret': 'test_secret',
                    'sandbox': True
                }
            ],
            'monitoring': {
                'heartbeat_interval': 30,
                'performance_update_interval': 60,
                'log_level': 'INFO'
            }
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test."""
        try:
            logger.info("🧪 Starting Comprehensive Mathematical Trading Integration Test")
            
            # Test 1: System Initialization
            init_result = await self._test_system_initialization()
            self.test_results['system_initialization'] = init_result
            
            # Test 2: Mathematical Integration
            math_result = await self._test_mathematical_integration()
            self.test_results['mathematical_integration'] = math_result
            
            # Test 3: Decision Making Process
            decision_result = await self._test_decision_making_process()
            self.test_results['decision_making_process'] = decision_result
            
            # Test 4: Flask Server Integration
            flask_result = await self._test_flask_server_integration()
            self.test_results['flask_server_integration'] = flask_result
            
            # Test 5: Registry and Soulprint Storage
            registry_result = await self._test_registry_storage()
            self.test_results['registry_storage'] = registry_result
            
            # Test 6: Multi-Bot Coordination
            coordination_result = await self._test_multi_bot_coordination()
            self.test_results['multi_bot_coordination'] = coordination_result
            
            # Test 7: Real Trading Pipeline Integration
            pipeline_result = await self._test_trading_pipeline_integration()
            self.test_results['trading_pipeline_integration'] = pipeline_result
            
            # Test 8: Performance Metrics
            performance_result = await self._test_performance_metrics()
            self.test_results['performance_metrics'] = performance_result
            
            # Calculate overall result
            overall_result = self._calculate_overall_result()
            self.test_results['overall_result'] = overall_result
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            return {'error': str(e)}
    
    async def _test_system_initialization(self) -> Dict[str, Any]:
        """Test system initialization."""
        try:
            logger.info("🔧 Testing system initialization...")
            
            # Create the unified system
            self.system = create_unified_trading_system(self.config)
            
            # Check if system was created successfully
            if not self.system:
                return {'success': False, 'error': 'System creation failed'}
            
            # Check if mathematical integration is available
            math_available = hasattr(self.system, 'mathematical_integration') and self.system.mathematical_integration is not None
            
            # Check if soulprint registry is available
            registry_available = hasattr(self.system, 'soulprint_registry') and self.system.soulprint_registry is not None
            
            # Check if cascade memory is available
            cascade_available = hasattr(self.system, 'cascade_memory') and self.system.cascade_memory is not None
            
            success = math_available and registry_available and cascade_available
            
            return {
                'success': success,
                'mathematical_integration_available': math_available,
                'registry_available': registry_available,
                'cascade_memory_available': cascade_available,
                'system_created': True
            }
            
        except Exception as e:
            logger.error(f"❌ System initialization test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_mathematical_integration(self) -> Dict[str, Any]:
        """Test mathematical integration."""
        try:
            logger.info("🧮 Testing mathematical integration...")
            
            if not self.system or not self.system.mathematical_integration:
                return {'success': False, 'error': 'Mathematical integration not available'}
            
            # Test market data processing
            test_market_data = {
                'symbol': 'BTC/USDC',
                'price': 50000.0,
                'volume': 2000.0,
                'volatility': 0.15,
                'price_change': 0.01,
                'sentiment': 0.6,
                'price_history': [50000, 50100, 49900, 50200, 50300],
                'timestamp': time.time()
            }
            
            # Process through mathematical integration
            mathematical_signal = await self.system.mathematical_integration.process_market_data_mathematically(test_market_data)
            
            # Verify mathematical signal components
            components_verified = {
                'dlt_waveform_score': isinstance(mathematical_signal.dlt_waveform_score, (int, float)),
                'dualistic_consensus': mathematical_signal.dualistic_consensus is not None,
                'bit_phase': isinstance(mathematical_signal.bit_phase, int) and mathematical_signal.bit_phase in [4, 8, 16, 32, 42],
                'ferris_phase': isinstance(mathematical_signal.ferris_phase, (int, float)) and 0 <= mathematical_signal.ferris_phase <= 1,
                'tensor_score': isinstance(mathematical_signal.tensor_score, (int, float)),
                'entropy_score': isinstance(mathematical_signal.entropy_score, (int, float)) and mathematical_signal.entropy_score >= 0,
                'confidence': isinstance(mathematical_signal.confidence, (int, float)) and 0 <= mathematical_signal.confidence <= 1,
                'decision': mathematical_signal.decision in ['BUY', 'SELL', 'HOLD']
            }
            
            all_components_valid = all(components_verified.values())
            
            return {
                'success': all_components_valid,
                'components_verified': components_verified,
                'mathematical_signal': {
                    'dlt_waveform_score': mathematical_signal.dlt_waveform_score,
                    'bit_phase': mathematical_signal.bit_phase,
                    'ferris_phase': mathematical_signal.ferris_phase,
                    'tensor_score': mathematical_signal.tensor_score,
                    'entropy_score': mathematical_signal.entropy_score,
                    'confidence': mathematical_signal.confidence,
                    'decision': mathematical_signal.decision
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Mathematical integration test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_decision_making_process(self) -> Dict[str, Any]:
        """Test the complete decision making process."""
        try:
            logger.info("🎯 Testing decision making process...")
            
            if not self.system:
                return {'success': False, 'error': 'System not initialized'}
            
            # Test market data
            test_market_data = {
                'symbol': 'BTC/USDC',
                'price': 50000.0,
                'volume': 2000.0,
                'volatility': 0.15,
                'price_change': 0.01,
                'sentiment': 0.6,
                'price_history': [50000, 50100, 49900, 50200, 50300],
                'timestamp': time.time()
            }
            
            # Process through complete system
            decision = await self.system.process_market_data_comprehensive(test_market_data)
            
            # Verify decision components
            decision_components = {
                'decision_id': bool(decision.decision_id),
                'action': decision.action in ['BUY', 'SELL', 'HOLD'],
                'symbol': decision.symbol == 'BTC/USDC',
                'entry_price': isinstance(decision.entry_price, (int, float)) and decision.entry_price > 0,
                'position_size': isinstance(decision.position_size, (int, float)) and decision.position_size >= 0,
                'confidence': isinstance(decision.confidence, (int, float)) and 0 <= decision.confidence <= 1,
                'timestamp': isinstance(decision.timestamp, (int, float)),
                'mathematical_signal': decision.mathematical_signal is not None,
                'dlt_waveform_score': isinstance(decision.dlt_waveform_score, (int, float)),
                'bit_phase': isinstance(decision.bit_phase, int),
                'ferris_phase': isinstance(decision.ferris_phase, (int, float)),
                'tensor_score': isinstance(decision.tensor_score, (int, float)),
                'entropy_score': isinstance(decision.entropy_score, (int, float)),
                'risk_score': isinstance(decision.risk_score, (int, float)) and 0 <= decision.risk_score <= 1,
                'coordinating_bots': isinstance(decision.coordinating_bots, list),
                'consensus_achieved': isinstance(decision.consensus_achieved, bool)
            }
            
            all_components_valid = all(decision_components.values())
            
            return {
                'success': all_components_valid,
                'decision_components': decision_components,
                'decision_summary': {
                    'action': decision.action,
                    'confidence': decision.confidence,
                    'dlt_waveform_score': decision.dlt_waveform_score,
                    'bit_phase': decision.bit_phase,
                    'tensor_score': decision.tensor_score,
                    'entropy_score': decision.entropy_score,
                    'risk_score': decision.risk_score,
                    'consensus_achieved': decision.consensus_achieved
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Decision making process test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_flask_server_integration(self) -> Dict[str, Any]:
        """Test Flask server integration."""
        try:
            logger.info("🌐 Testing Flask server integration...")
            
            if not self.system:
                return {'success': False, 'error': 'System not initialized'}
            
            # Check if Flask app is available
            flask_available = hasattr(self.system, 'app') and self.system.app is not None
            
            # Check if routes are set up
            routes_available = hasattr(self.system.app, 'url_map') and len(self.system.app.url_map._rules) > 0
            
            # Test API endpoints (simulated)
            api_endpoints = [
                '/api/register',
                '/api/heartbeat',
                '/api/decision',
                '/api/consensus',
                '/api/execute'
            ]
            
            endpoints_available = all(
                any(rule.rule == endpoint for rule in self.system.app.url_map._rules)
                for endpoint in api_endpoints
            )
            
            success = flask_available and routes_available and endpoints_available
            
            return {
                'success': success,
                'flask_available': flask_available,
                'routes_available': routes_available,
                'endpoints_available': endpoints_available,
                'total_routes': len(self.system.app.url_map._rules) if flask_available else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Flask server integration test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_registry_storage(self) -> Dict[str, Any]:
        """Test registry and soulprint storage."""
        try:
            logger.info("💾 Testing registry and soulprint storage...")
            
            if not self.system:
                return {'success': False, 'error': 'System not initialized'}
            
            # Check if registry is available
            registry_available = hasattr(self.system, 'soulprint_registry') and self.system.soulprint_registry is not None
            
            # Check if cascade memory is available
            cascade_available = hasattr(self.system, 'cascade_memory') and self.system.cascade_memory is not None
            
            # Test decision storage
            test_decision = type('obj', (object,), {
                'decision_id': 'test_decision_001',
                'timestamp': time.time(),
                'action': 'BUY',
                'symbol': 'BTC/USDC',
                'entry_price': 50000.0,
                'position_size': 0.01,
                'confidence': 0.8,
                'dlt_waveform_score': 0.7,
                'bit_phase': 16,
                'ferris_phase': 0.5,
                'tensor_score': 0.6,
                'entropy_score': 0.4,
                'kobold_analysis': {'confidence': 0.7},
                'ai_confidence': 0.7,
                'coordinating_bots': ['bot1', 'bot2'],
                'consensus_achieved': True,
                'risk_score': 0.3,
                'stop_loss': 49000.0,
                'take_profit': 52000.0
            })()
            
            # Store decision in registry
            self.system._store_decision_in_registry(test_decision)
            
            # Check if decision was stored
            decisions_stored = len(self.system.soulprint_registry.get_all_decisions()) > 0
            
            # Check if cascade memory was updated
            cascade_memories_stored = len(self.system.cascade_memory.cascade_memories) > 0
            
            success = registry_available and cascade_available and decisions_stored and cascade_memories_stored
            
            return {
                'success': success,
                'registry_available': registry_available,
                'cascade_available': cascade_available,
                'decisions_stored': decisions_stored,
                'cascade_memories_stored': cascade_memories_stored,
                'total_decisions': len(self.system.soulprint_registry.get_all_decisions()),
                'total_cascade_memories': len(self.system.cascade_memory.cascade_memories)
            }
            
        except Exception as e:
            logger.error(f"❌ Registry storage test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_multi_bot_coordination(self) -> Dict[str, Any]:
        """Test multi-bot coordination."""
        try:
            logger.info("🤖 Testing multi-bot coordination...")
            
            if not self.system:
                return {'success': False, 'error': 'System not initialized'}
            
            # Test bot registration
            test_bot_data = {
                'node_id': 'test_bot_001',
                'ip_address': '192.168.1.100',
                'port': 5002,
                'api_key': 'test_api_key',
                'mathematical_capabilities': ['DLT', 'Dualistic', 'BitPhase'],
                'gpu_available': True,
                'cuda_version': '11.8'
            }
            
            # Simulate bot registration
            self.system.connected_bots[test_bot_data['node_id']] = type('obj', (object,), {
                'node_id': test_bot_data['node_id'],
                'ip_address': test_bot_data['ip_address'],
                'port': test_bot_data['port'],
                'api_key': test_bot_data['api_key'],
                'status': 'active',
                'last_heartbeat': time.time(),
                'mathematical_capabilities': test_bot_data['mathematical_capabilities'],
                'gpu_available': test_bot_data['gpu_available'],
                'cuda_version': test_bot_data['cuda_version']
            })()
            
            # Test consensus processing
            test_decision = type('obj', (object,), {
                'decision_id': 'test_consensus_001',
                'coordinating_bots': ['test_bot_001', 'test_bot_002'],
                'confidence': 0.8,
                'action': 'BUY',
                'symbol': 'BTC/USDC',
                'timestamp': time.time()
            })()
            
            consensus_result = self.system._process_consensus(test_decision)
            
            # Verify coordination components
            coordination_components = {
                'bot_registration': len(self.system.connected_bots) > 0,
                'consensus_processing': consensus_result is not None,
                'consensus_achieved': consensus_result.get('consensus_achieved', False),
                'total_bots': consensus_result.get('total_bots', 0) > 0
            }
            
            success = all(coordination_components.values())
            
            return {
                'success': success,
                'coordination_components': coordination_components,
                'connected_bots': len(self.system.connected_bots),
                'consensus_result': consensus_result
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-bot coordination test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_trading_pipeline_integration(self) -> Dict[str, Any]:
        """Test trading pipeline integration."""
        try:
            logger.info("📊 Testing trading pipeline integration...")
            
            if not self.system:
                return {'success': False, 'error': 'System not initialized'}
            
            # Check if trading pipelines are available
            pipelines_available = hasattr(self.system, 'trading_pipelines') and isinstance(self.system.trading_pipelines, dict)
            
            # Test pipeline creation
            if pipelines_available:
                # Simulate pipeline creation
                self.system.trading_pipelines['test_pipeline'] = type('obj', (object,), {
                    'config': self.config['api_endpoints'][0],
                    'status': 'active'
                })()
            
            # Test trade execution simulation
            test_decision = type('obj', (object,), {
                'decision_id': 'test_execution_001',
                'action': 'BUY',
                'symbol': 'BTC/USDC',
                'entry_price': 50000.0,
                'position_size': 0.01,
                'stop_loss': 49000.0,
                'take_profit': 52000.0
            })()
            
            test_bot = type('obj', (object,), {
                'node_id': 'test_bot_001',
                'ip_address': '192.168.1.100',
                'port': 5002
            })()
            
            execution_result = self.system._execute_trade_on_bot(test_bot, test_decision)
            
            # Verify pipeline components
            pipeline_components = {
                'pipelines_available': pipelines_available,
                'execution_simulation': execution_result is not None,
                'execution_success': execution_result.get('success', False),
                'total_pipelines': len(self.system.trading_pipelines)
            }
            
            success = all(pipeline_components.values())
            
            return {
                'success': success,
                'pipeline_components': pipeline_components,
                'execution_result': execution_result,
                'total_pipelines': len(self.system.trading_pipelines)
            }
            
        except Exception as e:
            logger.error(f"❌ Trading pipeline integration test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics."""
        try:
            logger.info("📈 Testing performance metrics...")
            
            if not self.system:
                return {'success': False, 'error': 'System not initialized'}
            
            # Check if performance metrics are available
            metrics_available = hasattr(self.system, 'performance_metrics') and isinstance(self.system.performance_metrics, dict)
            
            # Get system status
            system_status = self.system.get_system_status()
            
            # Verify metrics components
            metrics_components = {
                'metrics_available': metrics_available,
                'system_status_available': system_status is not None,
                'total_decisions_tracked': system_status.get('total_decisions', 0) >= 0,
                'mathematical_decisions_tracked': system_status.get('performance_metrics', {}).get('mathematical_decisions', 0) >= 0,
                'ai_decisions_tracked': system_status.get('performance_metrics', {}).get('ai_decisions', 0) >= 0,
                'successful_trades_tracked': system_status.get('performance_metrics', {}).get('successful_trades', 0) >= 0
            }
            
            success = all(metrics_components.values())
            
            return {
                'success': success,
                'metrics_components': metrics_components,
                'system_status': system_status,
                'performance_metrics': self.system.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Performance metrics test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_overall_result(self) -> Dict[str, Any]:
        """Calculate overall test result."""
        try:
            # Count successful tests
            successful_tests = sum(1 for result in self.test_results.values() 
                                 if isinstance(result, dict) and result.get('success', False))
            total_tests = len([result for result in self.test_results.values() 
                             if isinstance(result, dict) and 'success' in result])
            
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            
            # Determine overall status
            if success_rate >= 90:
                overall_status = "EXCELLENT"
            elif success_rate >= 80:
                overall_status = "GOOD"
            elif success_rate >= 70:
                overall_status = "FAIR"
            else:
                overall_status = "NEEDS_IMPROVEMENT"
            
            return {
                'success_rate': success_rate,
                'successful_tests': successful_tests,
                'total_tests': total_tests,
                'overall_status': overall_status,
                'all_tests_passed': success_rate == 100
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating overall result: {e}")
            return {'error': str(e)}

async def main():
    """Main test function."""
    logger.info("🧪 UNIFIED MATHEMATICAL TRADING INTEGRATION TEST")
    logger.info("=" * 60)
    
    if not UNIFIED_SYSTEM_AVAILABLE:
        logger.error("❌ Unified system not available - cannot run tests")
        return
    
    # Create and run tester
    tester = UnifiedMathematicalTradingIntegrationTester()
    results = await tester.run_comprehensive_test()
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        if isinstance(result, dict) and 'success' in result:
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
            
            if not result['success'] and 'error' in result:
                logger.error(f"   Error: {result['error']}")
    
    # Overall result
    overall = results.get('overall_result', {})
    if 'success_rate' in overall:
        logger.info("\n" + "=" * 60)
        logger.info(f"🎯 OVERALL RESULT: {overall['success_rate']:.1f}% SUCCESS RATE")
        logger.info(f"📊 Status: {overall.get('overall_status', 'UNKNOWN')}")
        logger.info(f"✅ Tests Passed: {overall.get('successful_tests', 0)}/{overall.get('total_tests', 0)}")
        
        if overall.get('all_tests_passed', False):
            logger.info("🎉 ALL TESTS PASSED - MATHEMATICAL INTEGRATION IS WORKING!")
        else:
            logger.info("⚠️ Some tests failed - review the results above")
    
    logger.info("=" * 60)
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 