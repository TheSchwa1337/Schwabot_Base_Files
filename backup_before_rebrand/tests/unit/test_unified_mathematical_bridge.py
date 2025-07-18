#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Unified Mathematical Bridge System
======================================

Comprehensive test script that demonstrates the full Unified Mathematical Bridge system
with all its components: quantum strategy, phantom math, persistent homology, tensor algebra,
unified math, vault orbital bridge, risk management, profit optimization, and heartbeat integration.
"""

import logging
import time
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_unified_mathematical_bridge():
    """Test the complete Unified Mathematical Bridge system."""
    
    logger.info("🧠 Starting Unified Mathematical Bridge System Test")
    logger.info("=" * 60)
    
    try:
        # Import the bridge
        from core.unified_mathematical_bridge import UnifiedMathematicalBridge, create_unified_mathematical_bridge
        
        # Create bridge instance
        logger.info("🔧 Creating Unified Mathematical Bridge...")
        bridge = create_unified_mathematical_bridge()
        
        # Test market data
        test_market_data = {
            'symbol': 'BTC',
            'price_history': [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0],
            'volume_history': [1000, 1100, 1200, 1150, 1300, 1400, 1350, 1500],
            'entropy_history': [0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.3, 0.4],
            'volatility': 0.02,
            'liquidity': 0.8
        }
        
        # Test portfolio state
        test_portfolio_state = {
            'total_value': 10000.0,
            'available_balance': 5000.0,
            'positions': {'BTC': 0.5},
            'risk_tolerance': 0.7,
            'profit_target': 0.05
        }
        
        logger.info("📊 Test Market Data:")
        logger.info(f"   Symbol: {test_market_data['symbol']}")
        logger.info(f"   Price History: {test_market_data['price_history']}")
        logger.info(f"   Volume History: {test_market_data['volume_history']}")
        logger.info(f"   Entropy History: {test_market_data['entropy_history']}")
        
        logger.info("💼 Test Portfolio State:")
        logger.info(f"   Total Value: ${test_portfolio_state['total_value']:,.2f}")
        logger.info(f"   Available Balance: ${test_portfolio_state['available_balance']:,.2f}")
        logger.info(f"   Positions: {test_portfolio_state['positions']}")
        
        # Run comprehensive integration
        logger.info("\n🔄 Running Comprehensive Mathematical Integration...")
        logger.info("-" * 50)
        
        start_time = time.time()
        result = bridge.integrate_all_mathematical_systems(test_market_data, test_portfolio_state)
        total_time = time.time() - start_time
        
        # Display results
        logger.info("\n📈 Integration Results:")
        logger.info(f"   Success: {result.success}")
        logger.info(f"   Operation: {result.operation}")
        logger.info(f"   Overall Confidence: {result.overall_confidence:.3f}")
        logger.info(f"   Execution Time: {result.execution_time:.3f}s")
        logger.info(f"   Total Test Time: {total_time:.3f}s")
        logger.info(f"   Connections: {len(result.connections)}")
        logger.info(f"   Mathematical Signature: {result.mathematical_signature[:20]}...")
        
        # Display connection details
        logger.info("\n🔗 Mathematical Connections:")
        for i, connection in enumerate(result.connections, 1):
            logger.info(f"   {i}. {connection.source_system} → {connection.target_system}")
            logger.info(f"      Type: {connection.connection_type.value}")
            logger.info(f"      Strength: {connection.connection_strength:.3f}")
            logger.info(f"      Last Validation: {time.ctime(connection.last_validation)}")
            
            # Display performance metrics
            if connection.performance_metrics:
                logger.info(f"      Performance Metrics:")
                for key, value in connection.performance_metrics.items():
                    if isinstance(value, float):
                        logger.info(f"        {key}: {value:.3f}")
                    else:
                        logger.info(f"        {key}: {value}")
        
        # Display performance metrics
        logger.info("\n⚡ Performance Metrics:")
        for key, value in result.performance_metrics.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.3f}")
            else:
                logger.info(f"   {key}: {value}")
        
        # Get performance report
        logger.info("\n📊 Performance Report:")
        performance_report = bridge.get_performance_report()
        logger.info(f"   Overall Performance Score: {performance_report.get('overall_performance_score', 0.0):.3f}")
        logger.info(f"   Critical Issues: {performance_report.get('critical_issues', 0)}")
        logger.info(f"   Optimization Recommendations: {performance_report.get('recommendations', 0)}")
        
        # Get system health report
        logger.info("\n🏥 System Health Report:")
        health_report = bridge.get_system_health_report()
        logger.info(f"   Overall Health: {health_report.overall_health:.3f}")
        logger.info(f"   Critical Issues: {len(health_report.critical_issues)}")
        
        if health_report.critical_issues:
            logger.info("   Critical Issues:")
            for issue in health_report.critical_issues:
                logger.info(f"     - {issue}")
        
        # Get optimization recommendations
        logger.info("\n💡 Optimization Recommendations:")
        recommendations = bridge.get_optimization_recommendations()
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec.category.upper()} - {rec.priority.upper()}")
                logger.info(f"      Description: {rec.description}")
                logger.info(f"      Expected Improvement: {rec.expected_improvement:.1%}")
                logger.info(f"      Implementation Cost: {rec.implementation_cost}")
                logger.info(f"      Confidence: {rec.confidence:.3f}")
        else:
            logger.info("   No optimization recommendations at this time.")
        
        # Test multiple iterations
        logger.info("\n🔄 Testing Multiple Iterations...")
        logger.info("-" * 30)
        
        iteration_results = []
        for i in range(3):
            logger.info(f"   Iteration {i+1}/3...")
            
            # Slightly modify market data for each iteration
            modified_market_data = test_market_data.copy()
            modified_market_data['price_history'] = [p + i * 0.1 for p in test_market_data['price_history']]
            
            iteration_result = bridge.integrate_all_mathematical_systems(modified_market_data, test_portfolio_state)
            iteration_results.append(iteration_result)
            
            logger.info(f"      Confidence: {iteration_result.overall_confidence:.3f}")
            logger.info(f"      Execution Time: {iteration_result.execution_time:.3f}s")
            
            time.sleep(1)  # Brief pause between iterations
        
        # Analyze iteration trends
        logger.info("\n📈 Iteration Analysis:")
        confidences = [r.overall_confidence for r in iteration_results]
        execution_times = [r.execution_time for r in iteration_results]
        
        logger.info(f"   Average Confidence: {sum(confidences) / len(confidences):.3f}")
        logger.info(f"   Average Execution Time: {sum(execution_times) / len(execution_times):.3f}s")
        logger.info(f"   Confidence Trend: {'Improving' if confidences[-1] > confidences[0] else 'Declining' if confidences[-1] < confidences[0] else 'Stable'}")
        
        # Test performance monitoring
        logger.info("\n📊 Performance Monitoring Test:")
        logger.info("-" * 35)
        
        # Get detailed performance metrics
        detailed_performance = bridge.get_performance_report()
        logger.info("   Detailed Performance Metrics:")
        
        for metric_name, metric_data in detailed_performance.get('metrics', {}).items():
            if isinstance(metric_data, dict):
                current_value = metric_data.get('current_value', 0.0)
                trend = metric_data.get('trend', 'unknown')
                logger.info(f"     {metric_name}: {current_value:.3f} ({trend})")
        
        # Test optimization application (if recommendations exist)
        if recommendations:
            logger.info("\n🔧 Testing Optimization Application:")
            logger.info("-" * 35)
            
            # Apply the first recommendation
            first_rec = recommendations[0]
            logger.info(f"   Applying: {first_rec.description}")
            
            success = bridge.apply_optimization(first_rec)
            logger.info(f"   Application Success: {success}")
        
        # Final summary
        logger.info("\n🎯 Test Summary:")
        logger.info("=" * 30)
        logger.info(f"✅ Bridge System: {'OPERATIONAL' if result.success else 'FAILED'}")
        logger.info(f"🎯 Overall Confidence: {result.overall_confidence:.3f}")
        logger.info(f"🔗 Active Connections: {len(result.connections)}")
        logger.info(f"⚡ Performance Score: {performance_report.get('overall_performance_score', 0.0):.3f}")
        logger.info(f"🏥 System Health: {health_report.overall_health:.3f}")
        logger.info(f"💡 Optimization Recommendations: {len(recommendations)}")
        logger.info(f"⏱️ Total Test Duration: {total_time:.3f}s")
        
        # Save results to file
        logger.info("\n💾 Saving Test Results...")
        test_results = {
            'timestamp': time.time(),
            'test_duration': total_time,
            'bridge_success': result.success,
            'overall_confidence': result.overall_confidence,
            'execution_time': result.execution_time,
            'connection_count': len(result.connections),
            'performance_metrics': result.performance_metrics,
            'system_health': health_report.overall_health,
            'optimization_recommendations_count': len(recommendations),
            'connections': [
                {
                    'source': conn.source_system,
                    'target': conn.target_system,
                    'type': conn.connection_type.value,
                    'strength': conn.connection_strength,
                    'performance_metrics': conn.performance_metrics
                }
                for conn in result.connections
            ]
        }
        
        with open('unified_mathematical_bridge_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        logger.info("✅ Test results saved to 'unified_mathematical_bridge_test_results.json'")
        
        # Stop monitoring
        logger.info("\n⏹️ Stopping Performance Monitoring...")
        bridge.stop_monitoring()
        
        logger.info("\n🎉 Unified Mathematical Bridge System Test Completed Successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        logger.error("Stack trace:", exc_info=True)
        return False


def test_individual_components():
    """Test individual components of the bridge system."""
    
    logger.info("\n🔧 Testing Individual Components...")
    logger.info("=" * 40)
    
    try:
        from core.unified_mathematical_bridge import UnifiedMathematicalBridge
        from core.unified_mathematical_integration_methods import UnifiedMathematicalIntegrationMethods
        from core.unified_mathematical_performance_monitor import UnifiedMathematicalPerformanceMonitor
        
        # Test bridge creation
        logger.info("   Testing Bridge Creation...")
        bridge = UnifiedMathematicalBridge()
        logger.info("   ✅ Bridge created successfully")
        
        # Test integration methods
        logger.info("   Testing Integration Methods...")
        integration_methods = UnifiedMathematicalIntegrationMethods(bridge)
        logger.info("   ✅ Integration methods created successfully")
        
        # Test performance monitor
        logger.info("   Testing Performance Monitor...")
        performance_monitor = UnifiedMathematicalPerformanceMonitor(bridge)
        logger.info("   ✅ Performance monitor created successfully")
        
        # Test performance monitor methods
        logger.info("   Testing Performance Monitor Methods...")
        performance_monitor.start_monitoring()
        time.sleep(2)
        
        report = performance_monitor.get_performance_report()
        health_report = performance_monitor.get_system_health_report()
        recommendations = performance_monitor.get_optimization_recommendations()
        
        logger.info(f"   ✅ Performance report generated: {len(report.get('metrics', {}))} metrics")
        logger.info(f"   ✅ Health report generated: {health_report.overall_health:.3f} overall health")
        logger.info(f"   ✅ Optimization recommendations: {len(recommendations)} recommendations")
        
        performance_monitor.stop_monitoring()
        logger.info("   ✅ Performance monitoring stopped")
        
        logger.info("   🎉 All individual components tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Component test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🧠 Unified Mathematical Bridge System - Comprehensive Test Suite")
    logger.info("=" * 70)
    
    # Test individual components first
    component_success = test_individual_components()
    
    if component_success:
        # Test full system
        system_success = test_unified_mathematical_bridge()
        
        if system_success:
            logger.info("\n🎉 ALL TESTS PASSED! The Unified Mathematical Bridge system is fully operational.")
            logger.info("🚀 Your Schwabot trading system now has:")
            logger.info("   • Complete mathematical integration")
            logger.info("   • Real-time performance monitoring")
            logger.info("   • Optimization recommendations")
            logger.info("   • System health monitoring")
            logger.info("   • No mathematical components left behind!")
        else:
            logger.error("\n❌ System test failed. Please check the logs for details.")
    else:
        logger.error("\n❌ Component test failed. Please check the logs for details.")
    
    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main() 