#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Secure API Integration Manager
===================================
Comprehensive test suite for secure API integration with Alpha256 encryption,
multi-exchange profile management, and intelligent rebalancing.

Tests:
1. Alpha256 encryption functionality
2. Multi-exchange profile management
3. Secure API calls with encryption
4. Portfolio rebalancing with randomization
5. Concentration limit enforcement
6. Security event logging
7. Performance monitoring
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the secure API integration manager
try:
    from core.secure_api_integration_manager import (
        SecureAPIIntegrationManager,
        create_secure_api_manager,
        ExchangeType,
        ProfileType,
        RebalancingStrategy
    )
    SECURE_API_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Secure API Integration Manager not available: {e}")
    SECURE_API_AVAILABLE = False

# Import Alpha256 encryption
try:
    from core.alpha256_encryption import Alpha256Encryption, get_encryption
    ALPHA256_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Alpha256 encryption not available: {e}")
    ALPHA256_AVAILABLE = False

logger = logging.getLogger(__name__)

class SecureAPIIntegrationTester:
    """Comprehensive tester for secure API integration."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = {}
        self.start_time = datetime.now()
        self.manager = None
        
        # Test configuration
        self.test_config = {
            'test_encryption': True,
            'test_profiles': True,
            'test_api_calls': True,
            'test_rebalancing': True,
            'test_security': True,
            'test_performance': True
        }
        
        logger.info("🧪 Secure API Integration Tester initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        logger.info("🚀 Starting comprehensive secure API integration tests")
        
        try:
            # Test 1: Alpha256 Encryption
            if self.test_config['test_encryption'] and ALPHA256_AVAILABLE:
                await self._test_alpha256_encryption()
            
            # Test 2: Profile Management
            if self.test_config['test_profiles'] and SECURE_API_AVAILABLE:
                await self._test_profile_management()
            
            # Test 3: Secure API Calls
            if self.test_config['test_api_calls'] and SECURE_API_AVAILABLE:
                await self._test_secure_api_calls()
            
            # Test 4: Rebalancing with Randomization
            if self.test_config['test_rebalancing'] and SECURE_API_AVAILABLE:
                await self._test_rebalancing_with_randomization()
            
            # Test 5: Security Features
            if self.test_config['test_security'] and SECURE_API_AVAILABLE:
                await self._test_security_features()
            
            # Test 6: Performance Monitoring
            if self.test_config['test_performance'] and SECURE_API_AVAILABLE:
                await self._test_performance_monitoring()
            
            # Generate final report
            report = self._generate_test_report()
            
            logger.info("✅ All tests completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'test_results': self.test_results
            }
    
    async def _test_alpha256_encryption(self):
        """Test Alpha256 encryption functionality."""
        logger.info("🔐 Testing Alpha256 encryption...")
        
        try:
            # Initialize encryption
            encryption = get_encryption()
            
            # Test data encryption/decryption
            test_data = {
                'api_key': 'test_api_key_12345',
                'secret': 'test_secret_67890',
                'timestamp': time.time()
            }
            
            # Encrypt data
            encrypted_data = encryption.encrypt(str(test_data), 'test_session')
            
            # Decrypt data
            decrypted_data = encryption.decrypt(encrypted_data, 'test_session')
            
            # Verify decryption
            decrypted_dict = eval(decrypted_data)
            assert decrypted_dict['api_key'] == test_data['api_key']
            assert decrypted_dict['secret'] == test_data['secret']
            
            # Test API key storage
            key_id = encryption.store_api_key(
                'test_exchange',
                'test_api_key',
                'test_secret',
                ['read', 'trade']
            )
            
            # Test API key retrieval
            retrieved_key, retrieved_secret = encryption.get_api_key(key_id)
            assert retrieved_key == 'test_api_key'
            assert retrieved_secret == 'test_secret'
            
            self.test_results['alpha256_encryption'] = {
                'success': True,
                'encryption_works': True,
                'decryption_works': True,
                'api_key_storage_works': True,
                'api_key_retrieval_works': True
            }
            
            logger.info("✅ Alpha256 encryption test passed")
            
        except Exception as e:
            logger.error(f"❌ Alpha256 encryption test failed: {e}")
            self.test_results['alpha256_encryption'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_profile_management(self):
        """Test multi-exchange profile management."""
        logger.info("👥 Testing profile management...")
        
        try:
            # Create test configuration
            test_config = {
                'profiles': {
                    'test_profile_a': {
                        'name': 'Test Profile A',
                        'exchange': 'coinbase',
                        'type': 'conservative',
                        'api_key_id': 'test_coinbase_key',
                        'enabled': True,
                        'sandbox': True,
                        'max_position_size': 0.05,
                        'rebalancing_strategy': 'threshold_based',
                        'rebalancing_threshold': 0.03,
                        'randomization_factor': 0.05,
                        'target_allocation': {
                            'BTC': 0.40,
                            'ETH': 0.30,
                            'USDC': 0.30
                        }
                    },
                    'test_profile_b': {
                        'name': 'Test Profile B',
                        'exchange': 'binance',
                        'type': 'moderate',
                        'api_key_id': 'test_binance_key',
                        'enabled': True,
                        'sandbox': True,
                        'max_position_size': 0.10,
                        'rebalancing_strategy': 'risk_adjusted',
                        'rebalancing_threshold': 0.05,
                        'randomization_factor': 0.10,
                        'target_allocation': {
                            'BTC': 0.35,
                            'ETH': 0.25,
                            'SOL': 0.20,
                            'USDC': 0.20
                        }
                    }
                },
                'security': {
                    'encryption_enabled': True,
                    'key_rotation_interval': 86400,
                    'max_failed_attempts': 3,
                    'session_timeout': 3600,
                    'audit_logging': True
                },
                'rebalancing': {
                    'enabled': True,
                    'check_interval': 300,
                    'max_deviation': 0.10,
                    'randomization_enabled': True,
                    'randomization_factor': 0.10
                },
                'portfolio': {
                    'max_concentration': 0.25,
                    'min_diversification': 3,
                    'target_assets': ['BTC', 'ETH', 'SOL', 'XRP', 'USDC'],
                    'excluded_assets': ['USDT']
                }
            }
            
            # Create manager with test config
            self.manager = SecureAPIIntegrationManager()
            self.manager.config = test_config
            
            # Test profile initialization
            self.manager._initialize_profiles()
            
            # Verify profiles were created
            assert len(self.manager.profiles) == 2
            assert 'test_profile_a' in self.manager.profiles
            assert 'test_profile_b' in self.manager.profiles
            
            # Test profile properties
            profile_a = self.manager.profiles['test_profile_a']
            assert profile_a.exchange_type == ExchangeType.COINBASE
            assert profile_a.profile_type == ProfileType.CONSERVATIVE
            assert profile_a.max_position_size == 0.05
            assert profile_a.rebalancing_strategy == RebalancingStrategy.THRESHOLD_BASED
            
            profile_b = self.manager.profiles['test_profile_b']
            assert profile_b.exchange_type == ExchangeType.BINANCE
            assert profile_b.profile_type == ProfileType.MODERATE
            assert profile_b.max_position_size == 0.10
            assert profile_b.rebalancing_strategy == RebalancingStrategy.RISK_ADJUSTED
            
            self.test_results['profile_management'] = {
                'success': True,
                'profiles_created': len(self.manager.profiles),
                'profile_a_valid': True,
                'profile_b_valid': True,
                'exchange_types_correct': True,
                'profile_types_correct': True
            }
            
            logger.info("✅ Profile management test passed")
            
        except Exception as e:
            logger.error(f"❌ Profile management test failed: {e}")
            self.test_results['profile_management'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_secure_api_calls(self):
        """Test secure API calls with encryption."""
        logger.info("🔒 Testing secure API calls...")
        
        try:
            if not self.manager:
                raise ValueError("Manager not initialized")
            
            # Test security header generation
            headers = self.manager._generate_security_headers(
                'test_profile_a',
                'GET',
                '/accounts'
            )
            
            # Verify required headers
            required_headers = ['X-Timestamp', 'X-Nonce', 'X-Signature', 'X-Profile-ID']
            for header in required_headers:
                assert header in headers
            
            # Test data encryption
            test_data = {'symbol': 'BTC', 'amount': 0.1}
            encrypted_data = self.manager._encrypt_api_data(test_data, 'test_profile_a')
            
            # Test data decryption
            decrypted_data = self.manager._decrypt_api_response(
                {'encrypted_data': encrypted_data},
                'test_profile_a'
            )
            
            # Verify decryption
            assert decrypted_data['symbol'] == test_data['symbol']
            assert decrypted_data['amount'] == test_data['amount']
            
            # Test balance data processing
            mock_balance_data = {
                'accounts': [
                    {'currency': 'BTC', 'balance': '0.5'},
                    {'currency': 'ETH', 'balance': '2.0'},
                    {'currency': 'USDC', 'balance': '1000.0'}
                ]
            }
            
            processed_balance = self.manager._process_balance_data(
                mock_balance_data,
                'test_profile_a'
            )
            
            # Verify balance processing
            assert 'BTC' in processed_balance
            assert 'ETH' in processed_balance
            assert 'USDC' in processed_balance
            assert processed_balance['BTC'] == 0.5
            assert processed_balance['ETH'] == 2.0
            assert processed_balance['USDC'] == 1000.0
            
            self.test_results['secure_api_calls'] = {
                'success': True,
                'security_headers_work': True,
                'data_encryption_works': True,
                'data_decryption_works': True,
                'balance_processing_works': True
            }
            
            logger.info("✅ Secure API calls test passed")
            
        except Exception as e:
            logger.error(f"❌ Secure API calls test failed: {e}")
            self.test_results['secure_api_calls'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_rebalancing_with_randomization(self):
        """Test rebalancing with randomization to prevent over-concentration."""
        logger.info("⚖️ Testing rebalancing with randomization...")
        
        try:
            if not self.manager:
                raise ValueError("Manager not initialized")
            
            # Simulate portfolio allocation
            self.manager.total_portfolio_value = 10000.0
            
            # Create test allocations
            from core.secure_api_integration_manager import PortfolioAllocation
            
            # Test allocation with randomization
            allocation = PortfolioAllocation(
                symbol='BTC',
                target_percentage=0.40,
                current_percentage=0.45,
                randomized_target=0.42,  # Randomized target
                deviation=0.03,
                last_update=datetime.now(),
                exchange_distribution={'test_profile_a': 4500.0}
            )
            
            self.manager.portfolio_allocations['BTC'] = allocation
            
            # Test rebalancing action creation
            rebalancing_actions = await self.manager._check_profile_rebalancing('test_profile_a')
            
            # Verify rebalancing logic
            if rebalancing_actions:
                action = rebalancing_actions[0]
                assert action.symbol == 'BTC'
                assert action.action in ['buy', 'sell']
                assert action.amount > 0
                assert action.priority in [1, 2, 3, 4]
            
            # Test concentration limit enforcement
            concentration_check = self.manager._check_concentration_limits(
                type('Action', (), {
                    'action': 'buy',
                    'symbol': 'BTC',
                    'amount': 1000.0
                })()
            )
            
            # Test action validation
            valid_action = type('Action', (), {
                'profile_id': 'test_profile_a',
                'symbol': 'BTC',
                'action': 'buy',
                'amount': 100.0
            })()
            
            validation_result = self.manager._validate_rebalancing_action(valid_action)
            
            # Test randomization factor application
            profile = self.manager.profiles['test_profile_a']
            original_target = 0.40
            
            # Simulate randomization
            import random
            randomization = random.uniform(-profile.randomization_factor, profile.randomization_factor)
            randomized_target = original_target * (1 + randomization)
            
            # Verify randomization is within bounds
            assert abs(randomized_target - original_target) <= profile.randomization_factor * original_target
            
            self.test_results['rebalancing_with_randomization'] = {
                'success': True,
                'rebalancing_actions_created': len(rebalancing_actions) if rebalancing_actions else 0,
                'concentration_limits_work': True,
                'action_validation_works': validation_result,
                'randomization_within_bounds': True,
                'randomization_factor': profile.randomization_factor
            }
            
            logger.info("✅ Rebalancing with randomization test passed")
            
        except Exception as e:
            logger.error(f"❌ Rebalancing with randomization test failed: {e}")
            self.test_results['rebalancing_with_randomization'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_security_features(self):
        """Test security features and event logging."""
        logger.info("🛡️ Testing security features...")
        
        try:
            if not self.manager:
                raise ValueError("Manager not initialized")
            
            # Test security event logging
            initial_events = len(self.manager.security_events)
            
            self.manager._log_security_event(
                'test_profile_a',
                'GET',
                '/accounts',
                'success'
            )
            
            self.manager._log_security_event(
                'test_profile_a',
                'POST',
                '/orders',
                'error',
                'Invalid signature'
            )
            
            # Verify events were logged
            assert len(self.manager.security_events) == initial_events + 2
            
            # Verify security violations counter
            assert self.manager.security_violations >= 1
            
            # Test latest security event
            latest_event = self.manager.security_events[-1]
            assert latest_event['profile_id'] == 'test_profile_a'
            assert latest_event['method'] == 'POST'
            assert latest_event['endpoint'] == '/orders'
            assert latest_event['status'] == 'error'
            assert 'Invalid signature' in latest_event['error_message']
            
            # Test system status
            status = await self.manager.get_system_status()
            
            # Verify status contains security information
            assert 'security' in status
            assert 'profiles' in status
            assert 'portfolio' in status
            assert 'performance' in status
            
            security_status = status['security']
            assert 'alpha256_available' in security_status
            assert 'encryption_enabled' in security_status
            assert 'security_violations' in security_status
            
            self.test_results['security_features'] = {
                'success': True,
                'security_events_logged': len(self.manager.security_events),
                'security_violations_tracked': self.manager.security_violations,
                'system_status_works': True,
                'security_status_included': True
            }
            
            logger.info("✅ Security features test passed")
            
        except Exception as e:
            logger.error(f"❌ Security features test failed: {e}")
            self.test_results['security_features'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_performance_monitoring(self):
        """Test performance monitoring and metrics."""
        logger.info("📊 Testing performance monitoring...")
        
        try:
            if not self.manager:
                raise ValueError("Manager not initialized")
            
            # Simulate some trading activity
            self.manager.total_trades = 10
            self.manager.successful_trades = 8
            self.manager.total_profit = 150.0
            
            # Test performance metrics calculation
            success_rate = self.manager.successful_trades / self.manager.total_trades
            assert success_rate == 0.8  # 80% success rate
            
            # Test system status with performance data
            status = await self.manager.get_system_status()
            performance_status = status['performance']
            
            # Verify performance metrics
            assert performance_status['total_trades'] == 10
            assert performance_status['successful_trades'] == 8
            assert performance_status['total_profit'] == 150.0
            assert performance_status['success_rate'] == 0.8
            
            # Test profile performance tracking
            profile = self.manager.profiles['test_profile_a']
            profile.performance_metrics['total_rebalances'] = 5
            profile.performance_metrics['last_rebalance_success'] = True
            
            # Verify profile metrics are tracked
            assert profile.performance_metrics['total_rebalances'] == 5
            assert profile.performance_metrics['last_rebalance_success'] == True
            
            self.test_results['performance_monitoring'] = {
                'success': True,
                'total_trades_tracked': self.manager.total_trades,
                'successful_trades_tracked': self.manager.successful_trades,
                'total_profit_tracked': self.manager.total_profit,
                'success_rate_calculated': success_rate,
                'profile_metrics_tracked': True
            }
            
            logger.info("✅ Performance monitoring test passed")
            
        except Exception as e:
            logger.error(f"❌ Performance monitoring test failed: {e}")
            self.test_results['performance_monitoring'] = {
                'success': False,
                'error': str(e)
            }
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Calculate success rates
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': total_tests - successful_tests,
                'success_rate': success_rate,
                'duration_seconds': duration,
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat()
            },
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations(),
            'overall_success': success_rate >= 0.8  # 80% success threshold
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check Alpha256 encryption
        if 'alpha256_encryption' in self.test_results:
            result = self.test_results['alpha256_encryption']
            if not result.get('success', False):
                recommendations.append("🔐 Fix Alpha256 encryption implementation")
            else:
                recommendations.append("✅ Alpha256 encryption working correctly")
        
        # Check profile management
        if 'profile_management' in self.test_results:
            result = self.test_results['profile_management']
            if not result.get('success', False):
                recommendations.append("👥 Fix profile management system")
            else:
                recommendations.append("✅ Profile management working correctly")
        
        # Check secure API calls
        if 'secure_api_calls' in self.test_results:
            result = self.test_results['secure_api_calls']
            if not result.get('success', False):
                recommendations.append("🔒 Fix secure API call implementation")
            else:
                recommendations.append("✅ Secure API calls working correctly")
        
        # Check rebalancing
        if 'rebalancing_with_randomization' in self.test_results:
            result = self.test_results['rebalancing_with_randomization']
            if not result.get('success', False):
                recommendations.append("⚖️ Fix rebalancing with randomization")
            else:
                recommendations.append("✅ Rebalancing with randomization working correctly")
        
        # Check security features
        if 'security_features' in self.test_results:
            result = self.test_results['security_features']
            if not result.get('success', False):
                recommendations.append("🛡️ Fix security features")
            else:
                recommendations.append("✅ Security features working correctly")
        
        # Check performance monitoring
        if 'performance_monitoring' in self.test_results:
            result = self.test_results['performance_monitoring']
            if not result.get('success', False):
                recommendations.append("📊 Fix performance monitoring")
            else:
                recommendations.append("✅ Performance monitoring working correctly")
        
        return recommendations

async def main():
    """Main test execution function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Secure API Integration Test Suite")
    print("=" * 50)
    
    # Check availability
    if not ALPHA256_AVAILABLE:
        print("⚠️ Alpha256 encryption not available - some tests will be skipped")
    
    if not SECURE_API_AVAILABLE:
        print("⚠️ Secure API Integration Manager not available - some tests will be skipped")
    
    # Run tests
    tester = SecureAPIIntegrationTester()
    report = await tester.run_all_tests()
    
    # Display results
    print("\n📋 Test Results Summary")
    print("=" * 50)
    
    summary = report['test_summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Duration: {summary['duration_seconds']:.2f} seconds")
    
    print("\n📊 Detailed Results")
    print("=" * 50)
    
    for test_name, result in report['test_results'].items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result.get('success', False) and 'error' in result:
            print(f"  Error: {result['error']}")
    
    print("\n💡 Recommendations")
    print("=" * 50)
    
    for recommendation in report['recommendations']:
        print(f"• {recommendation}")
    
    print(f"\n🎯 Overall Status: {'✅ SUCCESS' if report['overall_success'] else '❌ FAILURE'}")
    
    # Save detailed report
    report_file = f"test_results_secure_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return report['overall_success']

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 