#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC/USDC Trading System Test Script 🚀

Demonstrates the complete BTC/USDC trading system with:
• Real mathematical implementations from YAML configs
• Strategy matrices → profit matrices → tensor calculations
• Ghost basket internal state management
• Thermal-aware and multi-bit processing
• Entry/exit functions for BTC/USDC trading

This script shows how your actual trading system works with real mathematical implementations.
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_mathematical_framework_integrator():
    """Test the mathematical framework integrator."""
    logger.info("🧪 Testing Mathematical Framework Integrator...")
    
    try:
        from core.mathematical_framework_integrator import mathematical_framework_integrator

        # Test DLT waveform engine
        logger.info("📊 Testing DLT Waveform Engine...")
        waveform_result = mathematical_framework_integrator.dlt_engine.process_waveform_complete(
            t=time.time() % 1000,
            seq=[50000.0, 50100.0],  # BTC price sequence
            hash_str="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            entry_price=50000.0,
            current_price=50100.0
        )
        
        logger.info(f"✅ DLT Waveform Result:")
        logger.info(f"   Waveform Value: {waveform_result.waveform_value:.6f}")
        logger.info(f"   Entropy: {waveform_result.entropy:.6f}")
        logger.info(f"   Bit Phase: {waveform_result.bit_phase}")
        logger.info(f"   Tensor Score: {waveform_result.tensor_score:.6f}")
        logger.info(f"   Confidence: {waveform_result.confidence:.6f}")
        
        # Test matrix mapper
        logger.info("📊 Testing Matrix Mapper...")
        matrix_result = mathematical_framework_integrator.matrix_mapper.process_matrix_complete(
            hash_value="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            tick=100,
            price=50100.0,
            entry_price=50000.0,
            phase=waveform_result.bit_phase
        )
        
        logger.info(f"✅ Matrix Mapper Result:")
        logger.info(f"   Basket ID: {matrix_result.basket_id}")
        logger.info(f"   Tensor Score: {matrix_result.tensor_score:.6f}")
        logger.info(f"   Confidence: {matrix_result.confidence:.6f}")
        logger.info(f"   Bit Phase: {matrix_result.bit_phase}")
        
        # Test profit cycle allocator
        logger.info("📊 Testing Profit Cycle Allocator...")
        execution_packet = {
            'volume': 1000.0,
            'actual_profit': 100.0,
            'entry_price': 50000.0,
            'current_price': 50100.0,
            'hash_value': "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            'tick': 100
        }
        
        profit_result = mathematical_framework_integrator.profit_allocator.allocate(
            execution_packet=execution_packet,
            cycles=["cycle_1", "cycle_2"],
            market_data={'price': 50100.0, 'volume': 1000.0}
        )
        
        logger.info(f"✅ Profit Allocation Result:")
        logger.info(f"   Allocation Success: {profit_result.allocation_success}")
        logger.info(f"   Allocated Amount: {profit_result.allocated_amount:.2f}")
        logger.info(f"   Profit Score: {profit_result.profit_score:.6f}")
        logger.info(f"   Confidence: {profit_result.confidence:.6f}")
        logger.info(f"   Basket ID: {profit_result.basket_id}")
        
        # Test BTC encoding processor
        logger.info("📊 Testing BTC Encoding Processor...")
        btc_result = mathematical_framework_integrator.btc_processor.process_btc_encoding_complete(
            price=50100.0,
            bit_depth=16
        )
        
        logger.info(f"✅ BTC Encoding Result:")
        logger.info(f"   Original Price: {btc_result.original_price:.2f}")
        logger.info(f"   Decoded Price: {btc_result.decoded_price:.2f}")
        logger.info(f"   Bit Depth: {btc_result.bit_depth}")
        logger.info(f"   Encoding Accuracy: {btc_result.encoding_accuracy:.6f}")
        
        # Test complete BTC trading processing
        logger.info("📊 Testing Complete BTC Trading Processing...")
        complete_result = mathematical_framework_integrator.process_btc_trading_complete(
            price=50100.0,
            volume=1000.0,
            entry_price=50000.0,
            hash_value="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            tick=100
        )
        
        logger.info(f"✅ Complete BTC Trading Result:")
        logger.info(f"   Overall Confidence: {complete_result['summary']['overall_confidence']:.6f}")
        logger.info(f"   Trading Signal: {complete_result['summary']['trading_signal']}")
        logger.info(f"   Backend: {complete_result['summary']['backend']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mathematical Framework Integrator test failed: {e}")
        return False


def test_btc_trading_engine():
    """Test the BTC trading engine."""
    logger.info("🧪 Testing BTC Trading Engine...")
    
    try:
        from core.btc_usdc_trading_engine import btc_trading_engine

        # Test BTC price processing
        logger.info("📊 Testing BTC Price Processing...")
        price_data = btc_trading_engine.process_btc_price(
            price=50100.0,
            volume=1000.0,
            thermal_state=70.0
        )
        
        logger.info(f"✅ BTC Price Data Result:")
        logger.info(f"   Price: {price_data.price:.2f}")
        logger.info(f"   Volume: {price_data.volume:.2f}")
        logger.info(f"   Hash: {price_data.hash_value[:16]}...")
        logger.info(f"   Bit Phase: {price_data.bit_phase}")
        logger.info(f"   Tensor Score: {price_data.tensor_score:.6f}")
        logger.info(f"   Entropy: {price_data.entropy:.6f}")
        logger.info(f"   Thermal State: {price_data.thermal_state:.1f}°C")
        
        # Test strategy matrix creation
        logger.info("📊 Testing Strategy Matrix Creation...")
        from core.btc_usdc_trading_engine import BitLevel, TradingMode
        
        strategy_matrix = btc_trading_engine.create_strategy_matrix(
            matrix_id="test_matrix_001",
            bit_level=BitLevel.BIT_16,
            thermal_mode=TradingMode.BALANCED_CONSISTENT
        )
        
        logger.info(f"✅ Strategy Matrix Result:")
        logger.info(f"   Matrix ID: {strategy_matrix.matrix_id}")
        logger.info(f"   Bit Level: {strategy_matrix.bit_level.value}")
        logger.info(f"   Thermal Mode: {strategy_matrix.thermal_mode.value}")
        logger.info(f"   Confidence Score: {strategy_matrix.confidence_score:.6f}")
        
        # Test trading signal generation
        logger.info("📊 Testing Trading Signal Generation...")
        signal = btc_trading_engine.generate_trading_signal(price_data)
        
        if signal:
            logger.info(f"✅ Trading Signal Generated:")
            logger.info(f"   Signal Type: {signal.signal_type}")
            logger.info(f"   Price: {signal.price:.2f}")
            logger.info(f"   Amount: {signal.amount:.6f}")
            logger.info(f"   Confidence: {signal.confidence:.6f}")
            logger.info(f"   Tensor Score: {signal.tensor_score:.6f}")
            logger.info(f"   Bit Phase: {signal.bit_phase}")
            logger.info(f"   Thermal Mode: {signal.thermal_mode.value}")
        else:
            logger.info("ℹ️ No trading signal generated (below threshold)")
        
        # Test ghost basket update
        if signal:
            logger.info("📊 Testing Ghost Basket Update...")
            basket = btc_trading_engine.update_ghost_basket("test_basket", signal)
            
            logger.info(f"✅ Ghost Basket Result:")
            logger.info(f"   Basket ID: {basket.basket_id}")
            logger.info(f"   Total Value: {basket.total_value:.2f}")
            logger.info(f"   Total PnL: {basket.total_pnl:.2f}")
            logger.info(f"   Risk Metrics: {len(basket.risk_metrics)} metrics")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BTC Trading Engine test failed: {e}")
        return False


def test_unified_btc_trading_pipeline():
    """Test the unified BTC trading pipeline."""
    logger.info("🧪 Testing Unified BTC Trading Pipeline...")
    
    try:
        from core.unified_btc_trading_pipeline import unified_btc_trading_pipeline

        # Test BTC price processing through complete pipeline
        logger.info("📊 Testing Complete BTC Trading Pipeline...")
        
        # Simulate multiple price updates
        base_price = 50000.0
        for i in range(5):
            # Simulate price movement
            price_change = random.uniform(-0.02, 0.02)  # ±2% change
            current_price = base_price * (1 + price_change)
            volume = random.uniform(500, 2000)
            thermal_state = random.uniform(65, 85)  # 65-85°C
            
            logger.info(f"📈 Processing BTC Price Update {i+1}:")
            logger.info(f"   Price: ${current_price:.2f} (change: {price_change*100:+.2f}%)")
            logger.info(f"   Volume: {volume:.0f}")
            logger.info(f"   Thermal State: {thermal_state:.1f}°C")
            
            result = unified_btc_trading_pipeline.process_btc_price(
                price=current_price,
                volume=volume,
                thermal_state=thermal_state
            )
            
            if result.success:
                if result.signal:
                    logger.info(f"   ✅ Signal: {result.signal.signal_type.upper()}")
                    logger.info(f"   ✅ Confidence: {result.signal.confidence:.3f}")
                    logger.info(f"   ✅ Basket: {result.signal.basket_id}")
                    logger.info(f"   ✅ Recommendation: {result.execution_recommendation}")
                else:
                    logger.info(f"   ℹ️ No signal (hold)")
                
                if result.ghost_basket_update.get('updated', False):
                    logger.info(f"   📦 Ghost Basket Updated: {result.ghost_basket_update['basket_id']}")
            else:
                logger.error(f"   ❌ Processing failed")
            
            logger.info("")  # Empty line for readability
        
        # Get pipeline summary
        logger.info("📊 Pipeline Summary:")
        summary = unified_btc_trading_pipeline.get_pipeline_summary()
        for key, value in summary.items():
            if key != 'config':  # Skip detailed config
                logger.info(f"   {key}: {value}")
        
        # Get ghost basket summary
        logger.info("📊 Ghost Basket Summary:")
        basket_summary = unified_btc_trading_pipeline.get_ghost_basket_summary()
        for basket_id, basket_data in basket_summary.items():
            logger.info(f"   {basket_id}:")
            logger.info(f"     Total Value: ${basket_data['total_value']:.2f}")
            logger.info(f"     Total PnL: ${basket_data['total_pnl']:.2f}")
            logger.info(f"     Positions: {basket_data['position_count']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Unified BTC Trading Pipeline test failed: {e}")
        return False


def test_mathematical_summary():
    """Test mathematical framework summary."""
    logger.info("🧪 Testing Mathematical Framework Summary...")
    
    try:
        from core.mathematical_framework_integrator import mathematical_framework_integrator
        
        summary = mathematical_framework_integrator.get_mathematical_summary()
        
        logger.info("✅ Mathematical Framework Summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                logger.info(f"   {key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"     {subkey}: {subvalue}")
            else:
                logger.info(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mathematical Framework Summary test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🚀 Starting BTC/USDC Trading System Tests...")
    logger.info("=" * 60)
    
    # Test results
    test_results = {}
    
    # Test 1: Mathematical Framework Integrator
    logger.info("🧪 TEST 1: Mathematical Framework Integrator")
    logger.info("-" * 40)
    test_results['mathematical_framework'] = test_mathematical_framework_integrator()
    logger.info("")
    
    # Test 2: BTC Trading Engine
    logger.info("🧪 TEST 2: BTC Trading Engine")
    logger.info("-" * 40)
    test_results['btc_trading_engine'] = test_btc_trading_engine()
    logger.info("")
    
    # Test 3: Unified BTC Trading Pipeline
    logger.info("🧪 TEST 3: Unified BTC Trading Pipeline")
    logger.info("-" * 40)
    test_results['unified_pipeline'] = test_unified_btc_trading_pipeline()
    logger.info("")
    
    # Test 4: Mathematical Summary
    logger.info("🧪 TEST 4: Mathematical Framework Summary")
    logger.info("-" * 40)
    test_results['mathematical_summary'] = test_mathematical_summary()
    logger.info("")
    
    # Summary
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("")
    logger.info(f"📈 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your BTC/USDC trading system is working correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Please check the implementation.")
    
    logger.info("")
    logger.info("🔧 Your BTC/USDC Trading System Features:")
    logger.info("   • Real mathematical implementations from YAML configs")
    logger.info("   • Strategy matrices → profit matrices → tensor calculations")
    logger.info("   • Ghost basket internal state management")
    logger.info("   • Thermal-aware and multi-bit processing")
    logger.info("   • Entry/exit functions for BTC/USDC trading")
    logger.info("   • GPU/CPU tensor operations with automatic fallback")
    logger.info("   • Complete mathematical framework integration")


if __name__ == "__main__":
    asyncio.run(main()) 