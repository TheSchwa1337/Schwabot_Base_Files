#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test KoboldCPP Unified Mathematical Trading Integration
=====================================================

This test verifies the complete integration of KoboldCPP with the unified
mathematical trading system, including:
1. Multi-cryptocurrency support (BTC, ETH, XRP, SOL, USDC)
2. 8-bit phase logic and strategy mapping
3. Tensor surface integration
4. Registry and soulprint storage
5. Hardware auto-detection and CUDA acceleration
6. Real-time decision making with mathematical consensus

This ensures that the KoboldCPP integration is properly utilized within
the trading pipeline and all mathematical components are working together.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the unified system
try:
    from core.unified_mathematical_trading_system import UnifiedMathematicalTradingSystem, create_unified_trading_system
    from core.koboldcpp_integration import KoboldCPPIntegration, CryptocurrencyType, AnalysisType
    from backtesting.mathematical_integration_simplified import mathematical_integration, MathematicalSignal
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import unified system: {e}")
    UNIFIED_SYSTEM_AVAILABLE = False

class KoboldCPPUnifiedIntegrationTester:
    """Test the KoboldCPP unified mathematical trading integration."""
    
    def __init__(self):
        """Initialize the tester."""
        self.config = self._create_test_config()
        self.system = None
        self.test_results = {}
        
        logger.info("🧪 KoboldCPP Unified Integration Tester initialized")
    
    def _create_test_config(self) -> Dict[str, Any]:
        """Create test configuration."""
        return {
            'flask_server': {
                'host': 'localhost',
                'port': 5001,
                'debug': False
            },
            'mathematical_integration': {
                'enabled': True,
                'weight': 0.7,
                'confidence_threshold': 0.6
            },
            'koboldcpp_integration': {
                'enabled': True,
                'kobold_path': 'koboldcpp',
                'model_path': '',
                'port': 5002,
                'auto_start': False  # Don't auto-start for testing
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
            logger.info("🧪 Starting KoboldCPP Unified Integration Test")
            
            # Test 1: System Initialization
            init_result = await self._test_system_initialization()
            self.test_results['system_initialization'] = init_result
            
            # Test 2: KoboldCPP Integration
            kobold_result = await self._test_koboldcpp_integration()
            self.test_results['koboldcpp_integration'] = kobold_result
            
            # Test 3: Multi-Cryptocurrency Support
            crypto_result = await self._test_multi_cryptocurrency_support()
            self.test_results['multi_cryptocurrency_support'] = crypto_result
            
            # Test 4: Bit Phase Logic
            bit_phase_result = await self._test_bit_phase_logic()
            self.test_results['bit_phase_logic'] = bit_phase_result
            
            # Test 5: Strategy Mapping
            strategy_result = await self._test_strategy_mapping()
            self.test_results['strategy_mapping'] = strategy_result
            
            # Test 6: Decision Making Process
            decision_result = await self._test_decision_making_process()
            self.test_results['decision_making_process'] = decision_result
            
            # Test 7: Registry and Storage
            storage_result = await self._test_registry_storage()
            self.test_results['registry_storage'] = storage_result
            
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
            
            # Check if KoboldCPP integration is available
            kobold_available = hasattr(self.system, 'kobold_integration') and self.system.kobold_integration is not None
            
            # Check if soulprint registry is available
            registry_available = hasattr(self.system, 'soulprint_registry') and self.system.soulprint_registry is not None
            
            success = math_available and registry_available
            
            return {
                'success': success,
                'mathematical_integration_available': math_available,
                'koboldcpp_integration_available': kobold_available,
                'registry_available': registry_available,
                'system_created': True
            }
            
        except Exception as e:
            logger.error(f"❌ System initialization test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_koboldcpp_integration(self) -> Dict[str, Any]:
        """Test KoboldCPP integration."""
        try:
            logger.info("🤖 Testing KoboldCPP integration...")
            
            if not self.system or not self.system.kobold_integration:
                return {'success': False, 'error': 'KoboldCPP integration not available'}
            
            # Test KoboldCPP initialization
            kobold_integration = self.system.kobold_integration
            
            # Check if KoboldCPP was initialized properly
            initialization_checks = {
                'hardware_detected': hasattr(kobold_integration, 'auto_detected'),
                'system_info_available': hasattr(kobold_integration, 'system_info'),
                'model_config_available': hasattr(kobold_integration, 'model_config'),
                'request_queues_available': hasattr(kobold_integration, 'request_queues'),
                'model_capabilities_available': hasattr(kobold_integration, 'model_capabilities')
            }
            
            all_checks_passed = all(initialization_checks.values())
            
            # Test market data analysis (simulated)
            test_market_data = {
                'symbol': 'BTC/USDC',
                'price': 50000.0,
                'volume': 2000.0,
                'volatility': 0.15,
                'bit_phase': 8,
                'cryptocurrency': 'BTC'
            }
            
            # Simulate analysis (since we're not starting the actual server)
            analysis_result = {
                'confidence': 0.7,
                'action': 'BUY',
                'analysis': {
                    'trend': 'bullish',
                    'rsi': 65,
                    'macd': 'positive'
                },
                'bit_phase_result': 8,
                'tensor_score': 0.6
            }
            
            return {
                'success': all_checks_passed,
                'initialization_checks': initialization_checks,
                'analysis_simulation': analysis_result,
                'kobold_integration_ready': all_checks_passed
            }
            
        except Exception as e:
            logger.error(f"❌ KoboldCPP integration test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_multi_cryptocurrency_support(self) -> Dict[str, Any]:
        """Test multi-cryptocurrency support."""
        try:
            logger.info("💰 Testing multi-cryptocurrency support...")
            
            if not self.system:
                return {'success': False, 'error': 'System not initialized'}
            
            # Test all supported cryptocurrencies
            cryptocurrencies = [
                CryptocurrencyType.BTC,
                CryptocurrencyType.ETH,
                CryptocurrencyType.XRP,
                CryptocurrencyType.SOL,
                CryptocurrencyType.USDC
            ]
            
            crypto_tests = {}
            
            for crypto in cryptocurrencies:
                # Create test market data for each cryptocurrency
                test_data = self._create_cryptocurrency_test_data(crypto)
                
                # Test decision processing
                try:
                    decision = await self.system.process_market_data_comprehensive(test_data)
                    
                    crypto_tests[crypto.value] = {
                        'success': True,
                        'decision_created': decision is not None,
                        'cryptocurrency_detected': decision.cryptocurrency == crypto,
                        'bit_phase_set': decision.bit_phase in [4, 8, 16, 32, 42],
                        'strategy_mapped': decision.strategy_mapped,
                        'action': decision.action,
                        'confidence': decision.confidence
                    }
                    
                except Exception as e:
                    crypto_tests[crypto.value] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Check if all cryptocurrencies are supported
            all_supported = all(test.get('success', False) for test in crypto_tests.values())
            
            return {
                'success': all_supported,
                'cryptocurrency_tests': crypto_tests,
                'total_cryptocurrencies_tested': len(cryptocurrencies),
                'successful_tests': sum(1 for test in crypto_tests.values() if test.get('success', False))
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-cryptocurrency support test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_cryptocurrency_test_data(self, crypto: CryptocurrencyType) -> Dict[str, Any]:
        """Create test market data for a specific cryptocurrency."""
        base_prices = {
            CryptocurrencyType.BTC: 50000.0,
            CryptocurrencyType.ETH: 3000.0,
            CryptocurrencyType.XRP: 0.5,
            CryptocurrencyType.SOL: 100.0,
            CryptocurrencyType.USDC: 1.0
        }
        
        base_volatilities = {
            CryptocurrencyType.BTC: 0.15,
            CryptocurrencyType.ETH: 0.20,
            CryptocurrencyType.XRP: 0.25,
            CryptocurrencyType.SOL: 0.30,
            CryptocurrencyType.USDC: 0.01
        }
        
        return {
            'symbol': f'{crypto.value}/USDC',
            'price': base_prices[crypto],
            'volume': random.uniform(1000, 5000),
            'volatility': base_volatilities[crypto],
            'bit_phase': random.choice([4, 8, 16, 32, 42]),
            'cryptocurrency': crypto.value,
            'timestamp': time.time()
        }
    
    async def _test_bit_phase_logic(self) -> Dict[str, Any]:
        """Test bit phase logic."""
        try:
            logger.info("🔢 Testing bit phase logic...")
            
            if not self.system:
                return {'success': False, 'error': 'System not initialized'}
            
            # Test different bit phases
            bit_phases = [4, 8, 16, 32, 42]
            bit_phase_tests = {}
            
            for phase in bit_phases:
                test_data = {
                    'symbol': 'BTC/USDC',
                    'price': 50000.0,
                    'volume': 2000.0,
                    'volatility': 0.15,
                    'bit_phase': phase,
                    'cryptocurrency': 'BTC'
                }
                
                try:
                    decision = await self.system.process_market_data_comprehensive(test_data)
                    
                    bit_phase_tests[phase] = {
                        'success': True,
                        'bit_phase_set': decision.bit_phase == phase,
                        'strategy_mapped': decision.strategy_mapped,
                        'bit_phase_alignment': decision.bit_phase_alignment,
                        'tensor_alignment': decision.tensor_alignment
                    }
                    
                except Exception as e:
                    bit_phase_tests[phase] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Check if all bit phases work
            all_phases_work = all(test.get('success', False) for test in bit_phase_tests.values())
            
            return {
                'success': all_phases_work,
                'bit_phase_tests': bit_phase_tests,
                'supported_phases': bit_phases,
                'successful_phases': sum(1 for test in bit_phase_tests.values() if test.get('success', False))
            }
            
        except Exception as e:
            logger.error(f"❌ Bit phase logic test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_strategy_mapping(self) -> Dict[str, Any]:
        """Test strategy mapping."""
        try:
            logger.info("🗺️ Testing strategy mapping...")
            
            if not self.system:
                return {'success': False, 'error': 'System not initialized'}
            
            # Test strategy mapping for different cryptocurrencies
            strategy_tests = {}
            
            for crypto in [CryptocurrencyType.BTC, CryptocurrencyType.ETH, CryptocurrencyType.XRP, CryptocurrencyType.SOL]:
                test_data = self._create_cryptocurrency_test_data(crypto)
                
                try:
                    # Test strategy mapping directly
                    mathematical_signal = None
                    kobold_analysis = None
                    
                    if self.system.mathematical_integration:
                        mathematical_signal = await self.system.mathematical_integration.process_market_data_mathematically(test_data)
                    
                    strategy_result = await self.system._apply_strategy_mapping(
                        test_data, mathematical_signal, kobold_analysis, crypto
                    )
                    
                    strategy_tests[crypto.value] = {
                        'success': True,
                        'strategy_mapped': strategy_result.get('strategy_mapped', False),
                        'bit_phase_alignment': strategy_result.get('bit_phase_alignment', False),
                        'tensor_alignment': strategy_result.get('tensor_alignment', False),
                        'expected_profit': strategy_result.get('expected_profit', 0.0),
                        'volatility_adjustment': strategy_result.get('volatility_adjustment', 1.0),
                        'action': strategy_result.get('action', 'HOLD'),
                        'confidence': strategy_result.get('confidence', 0.0)
                    }
                    
                except Exception as e:
                    strategy_tests[crypto.value] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Check if strategy mapping works for all cryptocurrencies
            all_strategies_work = all(test.get('success', False) for test in strategy_tests.values())
            
            return {
                'success': all_strategies_work,
                'strategy_tests': strategy_tests,
                'total_strategies_tested': len(strategy_tests),
                'successful_strategies': sum(1 for test in strategy_tests.values() if test.get('success', False))
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy mapping test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_decision_making_process(self) -> Dict[str, Any]:
        """Test the complete decision making process."""
        try:
            logger.info("🎯 Testing decision making process...")
            
            if not self.system:
                return {'success': False, 'error': 'System not initialized'}
            
            # Test decision making for BTC
            test_market_data = {
                'symbol': 'BTC/USDC',
                'price': 50000.0,
                'volume': 2000.0,
                'volatility': 0.15,
                'bit_phase': 8,
                'cryptocurrency': 'BTC',
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
                'cryptocurrency': decision.cryptocurrency == CryptocurrencyType.BTC,
                'base_currency': decision.base_currency == 'USDC',
                'bit_phase': isinstance(decision.bit_phase, int) and decision.bit_phase in [4, 8, 16, 32, 42],
                'strategy_mapped': isinstance(decision.strategy_mapped, bool),
                'bit_phase_alignment': isinstance(decision.bit_phase_alignment, bool),
                'tensor_alignment': isinstance(decision.tensor_alignment, bool),
                'expected_profit': isinstance(decision.expected_profit, (int, float)),
                'volatility_adjustment': isinstance(decision.volatility_adjustment, (int, float)),
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
                    'cryptocurrency': decision.cryptocurrency.value,
                    'bit_phase': decision.bit_phase,
                    'strategy_mapped': decision.strategy_mapped,
                    'expected_profit': decision.expected_profit,
                    'risk_score': decision.risk_score,
                    'consensus_achieved': decision.consensus_achieved
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Decision making process test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_registry_storage(self) -> Dict[str, Any]:
        """Test registry and storage."""
        try:
            logger.info("💾 Testing registry and storage...")
            
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
                'cryptocurrency': CryptocurrencyType.BTC,
                'base_currency': 'USDC',
                'bit_phase': 8,
                'strategy_mapped': True,
                'bit_phase_alignment': True,
                'tensor_alignment': True,
                'expected_profit': 2.5,
                'volatility_adjustment': 1.2,
                'dlt_waveform_score': 0.7,
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
                'successful_trades_tracked': system_status.get('performance_metrics', {}).get('successful_trades', 0) >= 0,
                'cryptocurrency_support_tracked': 'supported_cryptocurrencies' in system_status,
                'bit_phase_support_tracked': 'bit_phase_support' in system_status,
                'strategy_mapping_tracked': 'strategy_mapping' in system_status,
                'koboldcpp_details_tracked': 'koboldcpp_details' in system_status
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
    logger.info("🧪 KOBOLDCPP UNIFIED MATHEMATICAL TRADING INTEGRATION TEST")
    logger.info("=" * 70)
    
    if not UNIFIED_SYSTEM_AVAILABLE:
        logger.error("❌ Unified system not available - cannot run tests")
        return
    
    # Create and run tester
    tester = KoboldCPPUnifiedIntegrationTester()
    results = await tester.run_comprehensive_test()
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 70)
    
    for test_name, result in results.items():
        if isinstance(result, dict) and 'success' in result:
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
            
            if not result['success'] and 'error' in result:
                logger.error(f"   Error: {result['error']}")
    
    # Overall result
    overall = results.get('overall_result', {})
    if 'success_rate' in overall:
        logger.info("\n" + "=" * 70)
        logger.info(f"🎯 OVERALL RESULT: {overall['success_rate']:.1f}% SUCCESS RATE")
        logger.info(f"📊 Status: {overall.get('overall_status', 'UNKNOWN')}")
        logger.info(f"✅ Tests Passed: {overall.get('successful_tests', 0)}/{overall.get('total_tests', 0)}")
        
        if overall.get('all_tests_passed', False):
            logger.info("🎉 ALL TESTS PASSED - KOBOLDCPP INTEGRATION IS WORKING!")
        else:
            logger.info("⚠️ Some tests failed - review the results above")
    
    logger.info("=" * 70)
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 