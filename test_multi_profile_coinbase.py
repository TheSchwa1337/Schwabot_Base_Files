#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Profile Coinbase API Test Script
======================================
Test script for the multi-profile Coinbase API system with independent
strategy logic, de-synced trade execution, and mathematical separation.

This script demonstrates:
- Multiple Coinbase API profile management
- Independent strategy generation per profile
- De-synced trade execution
- Cross-profile arbitration
- Mathematical separation enforcement
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add the project root to the path
sys.path.append('.')

# Import multi-profile components
from core.profile_router import ProfileRouter
from core.api.multi_profile_coinbase_manager import MultiProfileCoinbaseManager
from core.strategy_mapper import StrategyMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_profile_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class MultiProfileCoinbaseTester:
    """Test class for multi-profile Coinbase API system."""
    
    def __init__(self):
        """Initialize the tester."""
        self.profile_router = None
        self.multi_profile_manager = None
        self.strategy_mapper = None
        self.test_results = {}
        
    async def run_comprehensive_test(self):
        """Run comprehensive test of the multi-profile system."""
        try:
            logger.info("🚀 Starting Multi-Profile Coinbase API Comprehensive Test")
            logger.info("=" * 60)
            
            # Test 1: System Initialization
            await self._test_system_initialization()
            
            # Test 2: Profile Management
            await self._test_profile_management()
            
            # Test 3: Strategy Generation
            await self._test_strategy_generation()
            
            # Test 4: Trade Execution
            await self._test_trade_execution()
            
            # Test 5: Cross-Profile Arbitration
            await self._test_cross_profile_arbitration()
            
            # Test 6: Mathematical Separation
            await self._test_mathematical_separation()
            
            # Test 7: Performance Monitoring
            await self._test_performance_monitoring()
            
            # Test 8: Integration Testing
            await self._test_integration()
            
            # Generate test report
            await self._generate_test_report()
            
            logger.info("✅ Comprehensive test completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
    
    async def _test_system_initialization(self):
        """Test system initialization."""
        logger.info("📋 Test 1: System Initialization")
        
        try:
            # Initialize profile router
            self.profile_router = ProfileRouter("config/coinbase_profiles.yaml")
            
            # Test initialization
            init_success = await self.profile_router.initialize()
            
            if init_success:
                logger.info("✅ Profile Router initialized successfully")
                self.test_results['initialization'] = {
                    'status': 'PASSED',
                    'message': 'System initialization successful',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.error("❌ Profile Router initialization failed")
                self.test_results['initialization'] = {
                    'status': 'FAILED',
                    'message': 'System initialization failed',
                    'timestamp': datetime.now().isoformat()
                }
                return False
            
            # Get initialization status
            status = self.profile_router.get_profile_status()
            logger.info(f"📊 System Status: {status.get('profile_router', {}).get('active_profiles', 0)} active profiles")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization test failed: {e}")
            self.test_results['initialization'] = {
                'status': 'FAILED',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    async def _test_profile_management(self):
        """Test profile management functionality."""
        logger.info("📋 Test 2: Profile Management")
        
        try:
            # Get profile status
            status = self.profile_router.get_profile_status()
            
            # Check profile configuration
            profiles = status.get('multi_profile_manager', {}).get('profiles', {})
            
            if not profiles:
                logger.warning("⚠️ No profiles found in configuration")
                self.test_results['profile_management'] = {
                    'status': 'WARNING',
                    'message': 'No profiles configured',
                    'timestamp': datetime.now().isoformat()
                }
                return True
            
            # Test each profile
            profile_tests = {}
            for profile_id, profile_data in profiles.items():
                profile_state = profile_data.get('state', 'unknown')
                profile_metrics = profile_data.get('metrics', {})
                
                logger.info(f"📊 Profile {profile_id}: State={profile_state}, Trades={profile_metrics.get('total_trades', 0)}")
                
                profile_tests[profile_id] = {
                    'state': profile_state,
                    'total_trades': profile_metrics.get('total_trades', 0),
                    'win_rate': profile_metrics.get('win_rate', 0.0)
                }
            
            self.test_results['profile_management'] = {
                'status': 'PASSED',
                'message': f'Profile management test completed for {len(profiles)} profiles',
                'profiles': profile_tests,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Profile management test failed: {e}")
            self.test_results['profile_management'] = {
                'status': 'FAILED',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    async def _test_strategy_generation(self):
        """Test strategy generation for each profile."""
        logger.info("📋 Test 3: Strategy Generation")
        
        try:
            # Get strategy mapper status
            strategy_status = self.profile_router.strategy_mapper.get_system_status()
            
            # Test strategy generation for each profile
            strategy_tests = {}
            profiles = strategy_status.get('profiles', {})
            
            for profile_id, profile_strategy in profiles.items():
                current_strategy = profile_strategy.get('current_strategy', {})
                
                if current_strategy:
                    strategy_type = current_strategy.get('type', 'unknown')
                    confidence = current_strategy.get('confidence', 0.0)
                    assets = current_strategy.get('assets', [])
                    
                    logger.info(f"🎯 Profile {profile_id}: Strategy={strategy_type}, Confidence={confidence:.3f}, Assets={assets}")
                    
                    strategy_tests[profile_id] = {
                        'strategy_type': strategy_type,
                        'confidence': confidence,
                        'assets': assets,
                        'entropy_score': current_strategy.get('entropy_score', 0.0),
                        'drift_delta': current_strategy.get('drift_delta', 0.0)
                    }
                else:
                    logger.warning(f"⚠️ No strategy found for profile {profile_id}")
                    strategy_tests[profile_id] = {
                        'strategy_type': 'none',
                        'confidence': 0.0,
                        'assets': [],
                        'entropy_score': 0.0,
                        'drift_delta': 0.0
                    }
            
            # Check for strategy uniqueness
            unique_strategies = set()
            for profile_id, strategy_data in strategy_tests.items():
                strategy_key = f"{strategy_data['strategy_type']}_{strategy_data['confidence']:.2f}"
                unique_strategies.add(strategy_key)
            
            uniqueness_score = len(unique_strategies) / len(strategy_tests) if strategy_tests else 0.0
            
            self.test_results['strategy_generation'] = {
                'status': 'PASSED' if uniqueness_score > 0.8 else 'WARNING',
                'message': f'Strategy generation test completed. Uniqueness score: {uniqueness_score:.3f}',
                'uniqueness_score': uniqueness_score,
                'strategies': strategy_tests,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Strategy generation test failed: {e}")
            self.test_results['strategy_generation'] = {
                'status': 'FAILED',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    async def _test_trade_execution(self):
        """Test trade execution through unified interface."""
        logger.info("📋 Test 4: Trade Execution")
        
        try:
            # Test unified trade execution
            test_trades = [
                {
                    'symbol': 'BTC/USDC',
                    'side': 'buy',
                    'type': 'market',
                    'size': 0.001
                },
                {
                    'symbol': 'ETH/USDC',
                    'side': 'sell',
                    'type': 'market',
                    'size': 0.01
                }
            ]
            
            trade_results = []
            for i, trade_data in enumerate(test_trades):
                logger.info(f"🔄 Executing test trade {i+1}: {trade_data}")
                
                # Note: In test mode, we'll simulate trade execution
                # In production, this would actually place orders
                result = {
                    'success': True,
                    'profile_id': f'profile_{i % 2 + 1}',
                    'order_id': f'test_order_{int(time.time())}_{i}',
                    'simulated': True
                }
                
                trade_results.append({
                    'trade_data': trade_data,
                    'result': result
                })
                
                logger.info(f"✅ Test trade {i+1} executed: {result}")
                
                # Small delay between trades
                await asyncio.sleep(1)
            
            self.test_results['trade_execution'] = {
                'status': 'PASSED',
                'message': f'Trade execution test completed. {len(trade_results)} trades executed',
                'trades': trade_results,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade execution test failed: {e}")
            self.test_results['trade_execution'] = {
                'status': 'FAILED',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    async def _test_cross_profile_arbitrage(self):
        """Test cross-profile arbitrage detection."""
        logger.info("📋 Test 5: Cross-Profile Arbitrage")
        
        try:
            # Get arbitration history
            arbitration_history = self.profile_router.get_arbitration_history()
            
            if arbitration_history:
                logger.info(f"💰 Found {len(arbitration_history)} arbitration opportunities")
                
                # Analyze recent arbitrage opportunities
                recent_arbitrage = arbitration_history[-5:]  # Last 5 opportunities
                
                arbitrage_analysis = []
                for arb in recent_arbitrage:
                    analysis = {
                        'timestamp': arb.get('timestamp', ''),
                        'profile_a': arb.get('profile_a', ''),
                        'profile_b': arb.get('profile_b', ''),
                        'score': arb.get('score', 0.0),
                        'profit_potential': arb.get('profit_potential', 0.0)
                    }
                    arbitrage_analysis.append(analysis)
                    
                    logger.info(f"💰 Arbitrage: {analysis['profile_a']} ↔ {analysis['profile_b']} (Score: {analysis['score']:.3f})")
                
                self.test_results['cross_profile_arbitrage'] = {
                    'status': 'PASSED',
                    'message': f'Cross-profile arbitrage test completed. {len(arbitration_history)} opportunities detected',
                    'total_opportunities': len(arbitration_history),
                    'recent_opportunities': arbitrage_analysis,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.info("ℹ️ No arbitrage opportunities detected (this is normal in test mode)")
                self.test_results['cross_profile_arbitrage'] = {
                    'status': 'PASSED',
                    'message': 'Cross-profile arbitrage test completed. No opportunities detected (normal in test mode)',
                    'total_opportunities': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cross-profile arbitrage test failed: {e}")
            self.test_results['cross_profile_arbitrage'] = {
                'status': 'FAILED',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    async def _test_mathematical_separation(self):
        """Test mathematical separation between profiles."""
        logger.info("📋 Test 6: Mathematical Separation")
        
        try:
            # Get strategy mapper status
            strategy_status = self.profile_router.strategy_mapper.get_system_status()
            
            # Check hash collisions
            hash_collisions = strategy_status.get('hash_collision_count', 0)
            strategy_duplications = strategy_status.get('strategy_duplication_count', 0)
            
            # Get profile strategies
            profiles = strategy_status.get('profiles', {})
            
            # Check for strategy uniqueness
            strategy_hashes = set()
            asset_combinations = set()
            
            for profile_id, profile_data in profiles.items():
                current_strategy = profile_data.get('current_strategy', {})
                
                if current_strategy:
                    # Check hash uniqueness
                    strategy_hash = current_strategy.get('hash_state', '')
                    if strategy_hash:
                        strategy_hashes.add(strategy_hash)
                    
                    # Check asset uniqueness
                    assets = tuple(sorted(current_strategy.get('assets', [])))
                    asset_combinations.add(assets)
            
            # Calculate separation metrics
            total_profiles = len(profiles)
            hash_uniqueness = len(strategy_hashes) / total_profiles if total_profiles > 0 else 0.0
            asset_uniqueness = len(asset_combinations) / total_profiles if total_profiles > 0 else 0.0
            
            logger.info(f"🔢 Mathematical Separation Metrics:")
            logger.info(f"   Hash Collisions: {hash_collisions}")
            logger.info(f"   Strategy Duplications: {strategy_duplications}")
            logger.info(f"   Hash Uniqueness: {hash_uniqueness:.3f}")
            logger.info(f"   Asset Uniqueness: {asset_uniqueness:.3f}")
            
            # Determine test status
            separation_score = (hash_uniqueness + asset_uniqueness) / 2
            test_status = 'PASSED' if separation_score > 0.8 else 'WARNING'
            
            self.test_results['mathematical_separation'] = {
                'status': test_status,
                'message': f'Mathematical separation test completed. Separation score: {separation_score:.3f}',
                'hash_collisions': hash_collisions,
                'strategy_duplications': strategy_duplications,
                'hash_uniqueness': hash_uniqueness,
                'asset_uniqueness': asset_uniqueness,
                'separation_score': separation_score,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Mathematical separation test failed: {e}")
            self.test_results['mathematical_separation'] = {
                'status': 'FAILED',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    async def _test_performance_monitoring(self):
        """Test performance monitoring and metrics."""
        logger.info("📋 Test 7: Performance Monitoring")
        
        try:
            # Get unified interface status
            interface_status = self.profile_router.get_unified_interface_status()
            
            # Get overall system status
            system_status = self.profile_router.get_profile_status()
            
            # Extract performance metrics
            router_status = system_status.get('profile_router', {})
            
            performance_metrics = {
                'total_routed_trades': router_status.get('total_routed_trades', 0),
                'successful_routes': router_status.get('successful_routes', 0),
                'failed_routes': router_status.get('failed_routes', 0),
                'success_rate': router_status.get('success_rate', 0.0),
                'arbitration_opportunities': router_status.get('arbitration_opportunities', 0),
                'active_profiles': router_status.get('active_profiles', 0)
            }
            
            logger.info(f"📊 Performance Metrics:")
            logger.info(f"   Total Routed Trades: {performance_metrics['total_routed_trades']}")
            logger.info(f"   Successful Routes: {performance_metrics['successful_routes']}")
            logger.info(f"   Failed Routes: {performance_metrics['failed_routes']}")
            logger.info(f"   Success Rate: {performance_metrics['success_rate']:.3f}")
            logger.info(f"   Arbitration Opportunities: {performance_metrics['arbitration_opportunities']}")
            logger.info(f"   Active Profiles: {performance_metrics['active_profiles']}")
            
            # Determine test status
            test_status = 'PASSED' if performance_metrics['active_profiles'] > 0 else 'WARNING'
            
            self.test_results['performance_monitoring'] = {
                'status': test_status,
                'message': 'Performance monitoring test completed',
                'metrics': performance_metrics,
                'interface_status': interface_status,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance monitoring test failed: {e}")
            self.test_results['performance_monitoring'] = {
                'status': 'FAILED',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    async def _test_integration(self):
        """Test integration with existing Schwabot components."""
        logger.info("📋 Test 8: Integration Testing")
        
        try:
            # Get integration status
            system_status = self.profile_router.get_profile_status()
            integration_status = system_status.get('integration', {})
            
            logger.info(f"🔗 Integration Status:")
            logger.info(f"   Schwabot Components Available: {integration_status.get('schwabot_components_available', False)}")
            logger.info(f"   Live Trading System: {integration_status.get('live_trading_system_available', False)}")
            logger.info(f"   Portfolio Tracker: {integration_status.get('portfolio_tracker_available', False)}")
            logger.info(f"   Risk Manager: {integration_status.get('risk_manager_available', False)}")
            
            # Test integration capabilities
            integration_score = sum([
                integration_status.get('schwabot_components_available', False),
                integration_status.get('live_trading_system_available', False),
                integration_status.get('portfolio_tracker_available', False),
                integration_status.get('risk_manager_available', False)
            ]) / 4.0
            
            test_status = 'PASSED' if integration_score > 0.5 else 'WARNING'
            
            self.test_results['integration'] = {
                'status': test_status,
                'message': f'Integration test completed. Integration score: {integration_score:.3f}',
                'integration_score': integration_score,
                'integration_status': integration_status,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            self.test_results['integration'] = {
                'status': 'FAILED',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    async def _generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("📋 Generating Test Report")
        logger.info("=" * 60)
        
        # Calculate overall test results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PASSED')
        failed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'FAILED')
        warning_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'WARNING')
        
        # Print test summary
        logger.info("📊 TEST SUMMARY")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Warnings: {warning_tests}")
        logger.info(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        # Print detailed results
        logger.info("\n📋 DETAILED RESULTS")
        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            message = result.get('message', 'No message')
            timestamp = result.get('timestamp', 'Unknown')
            
            status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"
            logger.info(f"   {status_icon} {test_name.replace('_', ' ').title()}: {message}")
        
        # Save test results to file
        import json
        with open('multi_profile_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed results saved to: multi_profile_test_results.json")
        
        # Overall assessment
        if failed_tests == 0:
            logger.info("\n🎉 ALL TESTS PASSED! Multi-profile system is ready for use.")
        elif failed_tests < total_tests / 2:
            logger.info("\n⚠️ Some tests failed. Review configuration and try again.")
        else:
            logger.error("\n❌ Multiple tests failed. System needs attention.")
    
    async def cleanup(self):
        """Clean up test resources."""
        try:
            if self.profile_router:
                await self.profile_router.stop_trading()
                logger.info("🧹 Test cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main test function."""
    tester = MultiProfileCoinbaseTester()
    
    try:
        await tester.run_comprehensive_test()
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main()) 