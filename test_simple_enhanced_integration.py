#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 SIMPLE ENHANCED MATH-TO-TRADE INTEGRATION TEST
================================================

Simple test to verify the enhanced math-to-trade integration works.
"""

import asyncio
import logging
import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

logger = logging.getLogger(__name__)


async def test_enhanced_integration():
    """Simple test of enhanced integration"""
    try:
        logger.info("🧪 Testing Enhanced Math-to-Trade Integration")
        
        # Test basic import
        try:
            from core.enhanced_math_to_trade_integration import (
                EnhancedMathToTradeIntegration, 
                EnhancedMathematicalSignal,
                create_enhanced_math_to_trade_integration
            )
            logger.info("✅ Enhanced integration module imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import enhanced integration: {e}")
            return False
        
        # Test creation
        try:
            config = {
                "enable_all_modules": True,
                "signal_aggregation_method": "weighted_mean",
                "confidence_threshold": 0.6,
                "strength_threshold": 0.3
            }
            
            integration = create_enhanced_math_to_trade_integration(config)
            logger.info("✅ Enhanced integration created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create enhanced integration: {e}")
            return False
        
        # Test signal generation
        try:
            signal = await integration.process_market_data_comprehensive(
                price=50000.0,
                volume=1000.0,
                asset_pair="BTC/USD"
            )
            
            if signal:
                logger.info(f"✅ Enhanced signal generated: {signal.signal_type.value}")
                logger.info(f"   Confidence: {signal.confidence:.3f}")
                logger.info(f"   Strength: {signal.strength:.3f}")
                logger.info(f"   Mathematical Score: {signal.mathematical_score:.3f}")
                
                # Test summary
                summary = integration.get_signal_summary()
                logger.info(f"✅ Signal summary: {summary}")
                
                # Test metrics
                metrics = integration.get_performance_metrics()
                logger.info(f"✅ Performance metrics: {metrics}")
                
                return True
            else:
                logger.error("❌ No signal generated")
                return False
                
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def main():
    """Main test function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # Run test
    success = await test_enhanced_integration()
    
    if success:
        logger.info("🎉 Enhanced integration test PASSED")
        return 0
    else:
        logger.error("❌ Enhanced integration test FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 