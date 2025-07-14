#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Integration Test for Schwabot Trading System

Tests the complete integration of:
- Unified Mathematical Core (GPU/CPU, fallback)
- Internal AI Agent System
- Flask Communication Relay
- ZPE/ZBE calculations
- Agent consensus building
"""

# Standard library imports
import asyncio
import json
import logging
import time
from typing import Any, Dict

# Third-party imports
import numpy as np
import requests

# Internal imports
from core.unified_mathematical_core import ZBECalculation, ZPECalculation, get_unified_math_core

    create_agent_system, get_communication_hub, MarketData, TradingSuggestion
)
from core.flask_communication_relay import get_flask_relay

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """Comprehensive integration test suite for Schwabot trading system."""

    def __init__(self):
        self.math_core = get_unified_math_core()
        self.agents = create_agent_system()
        self.communication_hub = get_communication_hub()
        self.test_results = {}

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("🚀 Starting comprehensive integration tests...")

        test_results = {}
            'mathematical_core': self.test_mathematical_core(),
            'ai_agent_system': self.test_ai_agent_system(),
            'communication_hub': self.test_communication_hub(),
            'zpe_zbe_calculations': self.test_zpe_zbe_calculations(),
            'consensus_building': self.test_consensus_building(),
            'flask_relay': self.test_flask_relay(),
            'overall_success': True
        }

        # Check overall success
        for test_name, result in test_results.items():
            if test_name != 'overall_success' and not result.get('success', False):
                test_results['overall_success'] = False
                logger.error("❌ Test failed: {}".format(test_name))

        if test_results['overall_success']:
            logger.info("✅ All integration tests passed!")
        else:
            logger.error("❌ Some integration tests failed!")

        return test_results

    def test_mathematical_core(self) -> Dict[str, Any]:
        """Test the unified mathematical core."""
        logger.info("Testing unified mathematical core...")

        try:
            # Test GPU/CPU detection
            gpu_available = self.math_core.gpu_available
            logger.info("GPU available: {}".format(gpu_available))

            # Test matrix operations
            A = np.array([[1, 2], [3, 4]], dtype=np.float32)
            B = np.array([[5, 6], [7, 8]], dtype=np.float32)

            # Test multiplication
            result = self.math_core.matrix_operation(A, B, 'multiply')
            expected = np.array([[19, 22], [43, 50]])

            if not np.allclose(result, expected, atol=1e-6):
                raise ValueError("Matrix multiplication failed")

            # Test addition
            result = self.math_core.matrix_operation(A, B, 'add')
            expected = A + B

            if not np.allclose(result, expected, atol=1e-6):
                raise ValueError("Matrix addition failed")

            # Test system status
            status = self.math_core.get_system_status()
            if not isinstance(status, dict):
                raise ValueError("System status not returned as dict")

            logger.info("✅ Mathematical core tests passed")
            return {}
                'success': True,
                'gpu_available': gpu_available,
                'matrix_operations': 'passed',
                'system_status': 'passed'
            }

        except Exception as e:
            logger.error("❌ Mathematical core test failed: {}".format(e))
            return {'success': False, 'error': str(e)}

    def test_ai_agent_system(self) -> Dict[str, Any]:
        """Test the AI agent system."""
        logger.info("Testing AI agent system...")

        try:
            # Check agent creation
            if len(self.agents) == 0:
                raise ValueError("No agents created")

            logger.info("Created {} agents".format(len(self.agents)))

            # Test agent types
            agent_types = set()
            for agent in self.agents.values():
                agent_types.add(agent.agent_type.value)
                logger.info("Agent: {} ({})".format(agent.agent_id, agent.agent_type.value))

            expected_types = {'strategy', 'risk'}
            if not agent_types.issuperset(expected_types):
                raise ValueError("Missing expected agent types. Expected: {}, Got: {}".format())
                    expected_types, agent_types
                ))

            # Test agent performance metrics
            for agent in self.agents.values():
                metrics = agent.get_performance_metrics()
                required_keys = {'suggestions_made', 'successful_trades', 'total_pnl', 'accuracy_rate'}

                if not all(key in metrics for key in, required_keys):
                    raise ValueError("Missing required performance metrics for agent {}".format(agent.agent_id))

            logger.info("✅ AI agent system tests passed")
            return {}
                'success': True,
                'agent_count': len(self.agents),
                'agent_types': list(agent_types),
                'performance_metrics': 'passed'
            }

        except Exception as e:
            logger.error("❌ AI agent system test failed: {}".format(e))
            return {'success': False, 'error': str(e)}

    def test_communication_hub(self) -> Dict[str, Any]:
        """Test the communication hub."""
        logger.info("Testing communication hub...")

        try:
            # Check hub initialization
            if not hasattr(self.communication_hub, 'agents'):
                raise ValueError("Communication hub not properly initialized")

            # Check agent registration
            registered_agents = len(self.communication_hub.agents)
            expected_agents = len(self.agents)

            if registered_agents != expected_agents:
                raise ValueError("Agent registration mismatch. Expected: {}, Got: {}".format())
                    expected_agents, registered_agents
                ))

            logger.info("Communication hub has {} registered agents".format(registered_agents))

            # Test hub state
            if self.communication_hub.running:
                logger.warning("Communication hub is already running")

            logger.info("✅ Communication hub tests passed")
            return {}
                'success': True,
                'registered_agents': registered_agents,
                'hub_state': 'initialized'
            }

        except Exception as e:
            logger.error("❌ Communication hub test failed: {}".format(e))
            return {'success': False, 'error': str(e)}

    def test_zpe_zbe_calculations(self) -> Dict[str, Any]:
        """Test ZPE and ZBE calculations."""
        logger.info("Testing ZPE and ZBE calculations...")

        try:
            # Test ZPE calculation
            frequency = 1e12  # 1 THz
            zpe_result = self.math_core.calculate_zpe(frequency)

            if not isinstance(zpe_result, ZPECalculation):
                raise ValueError("ZPE calculation did not return ZPECalculation object")

            # Check ZPE values
            expected_energy = 0.5 * self.math_core.planck_constant * frequency
            if abs(zpe_result.energy - expected_energy) > 1e-20:
                raise ValueError("ZPE calculation incorrect")

            logger.info("ZPE calculation: E = {:.2e} J for f = {:.2e} Hz".format())
                zpe_result.energy, frequency
            ))

            # Test ZBE calculation
            probabilities = np.array([0.25, 0.25, 0.25, 0.25])
            zbe_result = self.math_core.calculate_zbe(probabilities)

            if not isinstance(zbe_result, ZBECalculation):
                raise ValueError("ZBE calculation did not return ZBECalculation object")

            # Check ZBE values (entropy should be 2.0 for uniform, distribution)
            expected_entropy = 2.0
            if abs(zbe_result.entropy - expected_entropy) > 1e-6:
                raise ValueError("ZBE calculation incorrect")

            logger.info("ZBE calculation: H = {:.6f} bits".format(zbe_result.entropy))

            logger.info("✅ ZPE/ZBE calculation tests passed")
            return {}
                'success': True,
                'zpe_calculation': 'passed',
                'zbe_calculation': 'passed',
                'zpe_energy': zpe_result.energy,
                'zbe_entropy': zbe_result.entropy
            }

        except Exception as e:
            logger.error("❌ ZPE/ZBE calculation test failed: {}".format(e))
            return {'success': False, 'error': str(e)}

    async def test_consensus_building(self) -> Dict[str, Any]:
        """Test consensus building among agents."""
        logger.info("Testing consensus building...")

        try:
            # Create test market data
            market_data = MarketData()
                symbol="BTCUSDT",
                price=50000.0,
                volume=1000.0,
                timestamp=time.time(),
                bid=49999.0,
                ask=50001.0,
                spread=2.0,
                volatility=0.2
            )

            # Get suggestions from all agents
            suggestions = []
            for agent in self.agents.values():
                context = {}
                    'symbol': market_data.symbol,
                    'market_data': market_data
                }

                suggestion = await agent.make_suggestion(context)
                suggestions.append(suggestion)
                logger.info("Agent {} suggests: {} (confidence: {:.3f})".format())
                    agent.agent_id, suggestion.action, suggestion.confidence
                ))

            # Build consensus
            consensus = await self.communication_hub.build_consensus(suggestions)

            if not isinstance(consensus, dict):
                raise ValueError("Consensus not returned as dict")

            required_keys = {'consensus', 'confidence', 'reasoning', 'suggestion_count'}
            if not all(key in consensus for key in, required_keys):
                raise ValueError("Missing required consensus keys")

            logger.info("Consensus: {} (confidence: {:.3f})".format())
                consensus['consensus'], consensus['confidence']
            ))

            logger.info("✅ Consensus building tests passed")
            return {}
                'success': True,
                'suggestions_count': len(suggestions),
                'consensus_action': consensus['consensus'],
                'consensus_confidence': consensus['confidence']
            }

        except Exception as e:
            logger.error("❌ Consensus building test failed: {}".format(e))
            return {'success': False, 'error': str(e)}

    def test_flask_relay(self) -> Dict[str, Any]:
        """Test Flask communication relay."""
        logger.info("Testing Flask communication relay...")

        try:
            # Test relay creation
            relay = get_flask_relay()

            if not hasattr(relay, 'app'):
                raise ValueError("Flask app not created")

            if not hasattr(relay, 'socketio'):
                raise ValueError("SocketIO not created")

            # Test configuration
            if not isinstance(relay.config, dict):
                raise ValueError("Relay configuration not a dict")

            required_config_keys = {'host', 'port', 'debug', 'update_interval'}
            if not all(key in relay.config for key in, required_config_keys):
                raise ValueError("Missing required configuration keys")

            logger.info("Flask relay configured for {}:{}".format())
                relay.config['host'], relay.config['port']
            ))

            logger.info("✅ Flask relay tests passed")
            return {}
                'success': True,
                'relay_created': True,
                'config_valid': True,
                'host': relay.config['host'],
                'port': relay.config['port']
            }

        except Exception as e:
            logger.error("❌ Flask relay test failed: {}".format(e))
            return {'success': False, 'error': str(e)}

    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 60)
        report.append("📊 SCHWABOT INTEGRATION TEST REPORT")
        report.append("=" * 60)
        report.append("")

        # Overall status
        overall_success = results.get('overall_success', False)
        status_icon = "✅" if overall_success else "❌"
        report.append("{} Overall Status: {}".format(status_icon, "PASSED" if overall_success else "FAILED"))
        report.append("")

        # Individual test results
        for test_name, result in results.items():
            if test_name == 'overall_success':
                continue

            success = result.get('success', False)
            icon = "✅" if success else "❌"
            report.append("{} {}: {}".format(icon, test_name.replace('_', ' ').title(),))
                                           "PASSED" if success else "FAILED"))

            if success and isinstance(result, dict):
                # Add key metrics
                for key, value in result.items():
                    if key != 'success':
                        report.append("   - {}: {}".format(key, value))
            elif not success:
                report.append("   - Error: {}".format(result.get('error', 'Unknown error')))

            report.append("")

        # System summary
        report.append("-" * 60)
        report.append("SYSTEM SUMMARY")
        report.append("-" * 60)

        math_result = results.get('mathematical_core', {})
        agent_result = results.get('ai_agent_system', {})
        consensus_result = results.get('consensus_building', {})

        report.append("GPU Acceleration: {}".format())
            "Available" if math_result.get('gpu_available', False) else "Not Available"
        ))
        report.append("AI Agents: {} ({})".format())
            agent_result.get('agent_count', 0),
            ', '.join(agent_result.get('agent_types', []))
        ))
        report.append("Consensus Building: {}".format())
            "Functional" if consensus_result.get('success', False) else "Failed"
        ))

        report.append("")
        report.append("=" * 60)

        return '\n'.join(report)

async def main():
    """Main test execution function."""
    print("🚀 Schwabot Integration Test Suite")
    print("=" * 50)

    # Create test suite
    test_suite = IntegrationTestSuite()

    # Run all tests
    results = test_suite.run_all_tests()

    # Run async tests
    results['consensus_building'] = await test_suite.test_consensus_building()

    # Generate and display report
    report = test_suite.generate_test_report(results)
    print(report)

    # Save results to file
    with open('integration_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n📄 Test results saved to: integration_test_results.json")

    # Return success status
    return results.get('overall_success', False)

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 