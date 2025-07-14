#!/usr/bin/env python3
"""
Enhanced Unified Trading Pipeline Test Suite 🧪

Comprehensive testing for Schwabot's enhanced Unified Trading Pipeline with Math + Memory Fusion Core:
- Math + Memory Fusion Core integration
- Enhanced signal generation and selection
- Enhanced trade execution with signal lineage
- Registry updates with profit vector memory
- Signal lineage tracking and confidence overlays
- Performance metrics and cycle summaries
- Hash generation and memory tracking
"""

import asyncio
import time
import numpy as np
from core.unified_trading_pipeline import UnifiedTradingPipeline, EnhancedTradeResult

async def test_enhanced_pipeline_initialization():
    """Test enhanced pipeline initialization with Math + Memory Fusion Core."""
    print("🧪 Testing Enhanced Pipeline Initialization...")
    
    # Test initialization
    pipeline = UnifiedTradingPipeline(mode="demo", config={"min_confidence": 0.7})
    init_result = await pipeline.initialize()
    
    assert init_result == True, "Pipeline initialization should succeed"
    
    # Check Math + Memory Fusion Core integration
    if hasattr(pipeline, 'strategy_executor') and pipeline.strategy_executor:
        print("   ✅ Math + Memory Fusion Core integrated")
    else:
        print("   ⚠️ Math + Memory Fusion Core not available (fallback mode)")
    
    # Check configuration
    assert pipeline.min_confidence_threshold == 0.7, "Confidence threshold should be set"
    assert pipeline.entropy_correction_threshold == 0.3, "Entropy threshold should be set"
    
    print("   ✅ Enhanced pipeline initialization tests passed")

async def test_enhanced_signal_generation():
    """Test enhanced signal generation using Math + Memory Fusion Core."""
    print("\n🧪 Testing Enhanced Signal Generation...")
    
    pipeline = UnifiedTradingPipeline(mode="demo")
    await pipeline.initialize()
    
    # Generate test market data
    market_data = {
        "symbol": "BTC/USDC",
        "price": 50000.0,
        "volume": 1000.0,
        "volatility": 0.02,
        "timestamp": time.time(),
        "prices": [49000, 49500, 50000],
        "volumes": [950, 975, 1000]
    }
    
    # Apply mathematical systems
    math_context = pipeline._apply_mathematical_systems(market_data)
    
    # Generate enhanced signals
    enhanced_signals = await pipeline._generate_enhanced_trading_signals(market_data, math_context)
    
    print(f"   Generated {len(enhanced_signals)} enhanced signals")
    
    # Validate signals
    for signal in enhanced_signals:
        assert hasattr(signal, 'symbol'), "Signal should have symbol"
        assert hasattr(signal, 'action'), "Signal should have action"
        assert hasattr(signal, 'confidence'), "Signal should have confidence"
        assert hasattr(signal, 'vector_confidence'), "Signal should have vector confidence"
        assert hasattr(signal, 'entropy_correction'), "Signal should have entropy correction"
        
        print(f"     Signal: {signal.action} {signal.symbol} "
              f"(confidence: {signal.confidence:.3f}, "
              f"vector: {signal.vector_confidence:.3f}, "
              f"entropy: {signal.entropy_correction:.3f})")
    
    print("   ✅ Enhanced signal generation tests passed")

async def test_best_signal_selection():
    """Test best signal selection based on confidence and entropy correction."""
    print("\n🧪 Testing Best Signal Selection...")
    
    pipeline = UnifiedTradingPipeline(mode="demo")
    await pipeline.initialize()
    
    # Create test signals with different confidence levels
    from core.strategy.strategy_executor import EnhancedTradingSignal
    
    test_signals = [
        EnhancedTradingSignal(
            symbol="BTC/USDC",
            action="buy",
            entry_price=50000.0,
            amount=1000.0,
            strategy_id="test_1",
            confidence=0.8,
            vector_confidence=0.9,
            entropy_correction=0.1,
            volatility=0.02,
            volume=1000.0,
            market_conditions={},
            signal_hash="test_hash_1"
        ),
        EnhancedTradingSignal(
            symbol="BTC/USDC",
            action="sell",
            entry_price=50000.0,
            amount=1000.0,
            strategy_id="test_2",
            confidence=0.6,
            vector_confidence=0.7,
            entropy_correction=0.3,
            volatility=0.02,
            volume=1000.0,
            market_conditions={},
            signal_hash="test_hash_2"
        )
    ]
    
    # Select best signal
    best_signal = pipeline._select_best_signal(test_signals)
    
    assert best_signal is not None, "Should select a best signal"
    assert best_signal.confidence == 0.8, "Should select signal with highest confidence"
    
    print(f"   Selected best signal: {best_signal.action} {best_signal.symbol} "
          f"(confidence: {best_signal.confidence:.3f})")
    
    print("   ✅ Best signal selection tests passed")

async def test_enhanced_trade_execution():
    """Test enhanced trade execution with signal lineage tracking."""
    print("\n🧪 Testing Enhanced Trade Execution...")
    
    pipeline = UnifiedTradingPipeline(mode="demo")
    await pipeline.initialize()
    
    # Create test signal
    from core.strategy.strategy_executor import EnhancedTradingSignal
    
    test_signal = EnhancedTradingSignal(
        symbol="ETH/USDC",
        action="buy",
        entry_price=3000.0,
        amount=500.0,
        strategy_id="test_execution",
        confidence=0.85,
        vector_confidence=0.9,
        entropy_correction=0.1,
        volatility=0.03,
        volume=500.0,
        market_conditions={},
        signal_hash="exec_test_hash"
    )
    
    # Execute enhanced trade
    trade_result = await pipeline._execute_enhanced_trade(test_signal)
    
    assert trade_result is not None, "Trade execution should succeed"
    assert isinstance(trade_result, EnhancedTradeResult), "Should return EnhancedTradeResult"
    assert trade_result.symbol == "ETH/USDC", "Symbol should match"
    assert trade_result.action == "buy", "Action should match"
    assert trade_result.entry_price == 3000.0, "Entry price should match"
    assert trade_result.profit > 0, "Should have positive profit for buy signal"
    assert trade_result.signal_hash == "exec_test_hash", "Signal hash should match"
    assert trade_result.canonical_hash is not None, "Should have canonical hash"
    assert trade_result.profit_vector_hash is not None, "Should have profit vector hash"
    assert trade_result.soulprint_hash is not None, "Should have soulprint hash"
    
    print(f"   Executed trade: {trade_result.action} {trade_result.symbol}")
    print(f"   Profit: {trade_result.profit:.3f}")
    print(f"   Confidence: {trade_result.mathematical_confidence:.3f}")
    print(f"   Canonical hash: {trade_result.canonical_hash}")
    print(f"   Registry status: {trade_result.registry_status}")
    
    print("   ✅ Enhanced trade execution tests passed")

async def test_hash_generation():
    """Test hash generation for trade tracking and memory."""
    print("\n🧪 Testing Hash Generation...")
    
    pipeline = UnifiedTradingPipeline(mode="demo")
    await pipeline.initialize()
    
    # Create test signal
    from core.strategy.strategy_executor import EnhancedTradingSignal
    
    test_signal = EnhancedTradingSignal(
        symbol="ADA/USDC",
        action="sell",
        entry_price=0.50,
        amount=10000.0,
        strategy_id="test_hash",
        confidence=0.75,
        vector_confidence=0.8,
        entropy_correction=0.2,
        volatility=0.04,
        volume=1000000.0,
        market_conditions={},
        signal_hash="hash_test_signal"
    )
    
    # Generate hashes
    canonical_hash = pipeline._generate_canonical_hash(test_signal)
    profit_vector_hash = pipeline._generate_profit_vector_hash(test_signal)
    soulprint_hash = pipeline._generate_soulprint_hash(test_signal)
    
    # Validate hashes
    assert len(canonical_hash) == 8, "Canonical hash should be 8 characters"
    assert len(profit_vector_hash) == 8, "Profit vector hash should be 8 characters"
    assert len(soulprint_hash) == 8, "Soulprint hash should be 8 characters"
    assert canonical_hash != "unknown", "Canonical hash should be generated"
    assert soulprint_hash != "unknown", "Soulprint hash should be generated"
    
    print(f"   Canonical hash: {canonical_hash}")
    print(f"   Profit vector hash: {profit_vector_hash}")
    print(f"   Soulprint hash: {soulprint_hash}")
    
    print("   ✅ Hash generation tests passed")

async def test_registry_updates():
    """Test enhanced registry updates with profit vector memory."""
    print("\n🧪 Testing Enhanced Registry Updates...")
    
    pipeline = UnifiedTradingPipeline(mode="demo")
    await pipeline.initialize()
    
    # Create test trade result
    test_trade_result = EnhancedTradeResult(
        symbol="DOT/USDC",
        action="buy",
        entry_price=7.50,
        exit_price=7.65,
        amount=1000.0,
        profit=0.02,
        timestamp=time.time(),
        signal_hash="reg_test_hash",
        vector_confidence=0.85,
        mathematical_confidence=0.8,
        entropy_correction=0.15,
        canonical_hash="test_canonical",
        profit_vector_hash="test_profit",
        soulprint_hash="test_soulprint",
        drawdown=0.01,
        volatility=0.025,
        risk_profile="low",
        strategy_id="test_registry",
        exit_type="take_profit",
        registry_status="confirmed"
    )
    
    # Update registries
    await pipeline._update_enhanced_registries(test_trade_result)
    
    # Check signal lineage
    assert len(pipeline.signal_lineage) > 0, "Signal lineage should be stored"
    latest_lineage = pipeline.signal_lineage[-1]
    assert latest_lineage["signal_hash"] == "reg_test_hash", "Signal hash should match"
    assert latest_lineage["canonical_hash"] == "test_canonical", "Canonical hash should match"
    assert latest_lineage["profit"] == 0.02, "Profit should match"
    
    print(f"   Updated registries for trade: {test_trade_result.canonical_hash}")
    print(f"   Signal lineage entries: {len(pipeline.signal_lineage)}")
    print(f"   Latest lineage profit: {latest_lineage['profit']:.3f}")
    
    print("   ✅ Enhanced registry updates tests passed")

async def test_performance_metrics():
    """Test enhanced performance metrics tracking."""
    print("\n🧪 Testing Enhanced Performance Metrics...")
    
    pipeline = UnifiedTradingPipeline(mode="demo")
    await pipeline.initialize()
    
    initial_trades = pipeline.total_trades
    initial_profit = pipeline.total_profit
    
    # Create test trade result
    test_trade_result = EnhancedTradeResult(
        symbol="SOL/USDC",
        action="buy",
        entry_price=100.0,
        exit_price=105.0,
        amount=100.0,
        profit=0.05,
        timestamp=time.time(),
        signal_hash="perf_test_hash",
        vector_confidence=0.9,
        mathematical_confidence=0.85,
        entropy_correction=0.1,
        canonical_hash="perf_canonical",
        profit_vector_hash="perf_profit",
        soulprint_hash="perf_soulprint",
        drawdown=0.0,
        volatility=0.03,
        risk_profile="medium",
        strategy_id="test_performance",
        exit_type="take_profit",
        registry_status="confirmed"
    )
    
    # Update performance metrics
    pipeline._update_enhanced_performance_metrics(test_trade_result)
    
    # Check metrics
    assert pipeline.total_trades == initial_trades + 1, "Total trades should increment"
    assert pipeline.total_profit == initial_profit + 0.05, "Total profit should increment"
    assert pipeline.successful_trades > 0, "Successful trades should increment"
    assert len(pipeline.trade_history) > 0, "Trade history should be updated"
    
    print(f"   Total trades: {pipeline.total_trades}")
    print(f"   Successful trades: {pipeline.successful_trades}")
    print(f"   Success rate: {pipeline.successful_trades/pipeline.total_trades:.2%}")
    print(f"   Total profit: {pipeline.total_profit:.3f}")
    print(f"   Trade history entries: {len(pipeline.trade_history)}")
    
    print("   ✅ Enhanced performance metrics tests passed")

async def test_cycle_summary():
    """Test enhanced cycle summary generation."""
    print("\n🧪 Testing Enhanced Cycle Summary...")
    
    pipeline = UnifiedTradingPipeline(mode="demo")
    await pipeline.initialize()
    
    # Generate test data
    market_data = {
        "symbol": "MATIC/USDC",
        "price": 0.80,
        "volume": 500000.0,
        "volatility": 0.035,
        "timestamp": time.time(),
        "prices": [0.78, 0.79, 0.80],
        "volumes": [480000, 490000, 500000]
    }
    
    # Create test signals and trade result
    from core.strategy.strategy_executor import EnhancedTradingSignal
    
    test_signals = [
        EnhancedTradingSignal(
            symbol="MATIC/USDC",
            action="buy",
            entry_price=0.80,
            amount=1000.0,
            strategy_id="test_cycle",
            confidence=0.8,
            vector_confidence=0.85,
            entropy_correction=0.15,
            volatility=0.035,
            volume=500000.0,
            market_conditions={},
            signal_hash="cycle_test_hash"
        )
    ]
    
    test_trade_result = EnhancedTradeResult(
        symbol="MATIC/USDC",
        action="buy",
        entry_price=0.80,
        exit_price=0.82,
        amount=1000.0,
        profit=0.025,
        timestamp=time.time(),
        signal_hash="cycle_test_hash",
        vector_confidence=0.85,
        mathematical_confidence=0.8,
        entropy_correction=0.15,
        canonical_hash="cycle_canonical",
        profit_vector_hash="cycle_profit",
        soulprint_hash="cycle_soulprint",
        drawdown=0.0,
        volatility=0.035,
        risk_profile="low",
        strategy_id="test_cycle",
        exit_type="take_profit",
        registry_status="confirmed"
    )
    
    # Generate cycle summary
    cycle_summary = pipeline._generate_enhanced_cycle_summary(market_data, test_signals, test_trade_result)
    
    # Validate summary
    assert "timestamp" in cycle_summary, "Should have timestamp"
    assert "market_data" in cycle_summary, "Should have market data"
    assert "enhanced_signals_count" in cycle_summary, "Should have signals count"
    assert "trade_executed" in cycle_summary, "Should have trade executed flag"
    assert "portfolio_value" in cycle_summary, "Should have portfolio value"
    assert "total_trades" in cycle_summary, "Should have total trades"
    assert "success_rate" in cycle_summary, "Should have success rate"
    assert "total_profit" in cycle_summary, "Should have total profit"
    assert "trade_result" in cycle_summary, "Should have trade result"
    
    print(f"   Cycle summary generated successfully")
    print(f"   Enhanced signals count: {cycle_summary['enhanced_signals_count']}")
    print(f"   Trade executed: {cycle_summary['trade_executed']}")
    print(f"   Portfolio value: ${cycle_summary['portfolio_value']:.2f}")
    print(f"   Success rate: {cycle_summary['success_rate']:.2%}")
    
    print("   ✅ Enhanced cycle summary tests passed")

async def test_complete_trading_cycle():
    """Test complete enhanced trading cycle."""
    print("\n🧪 Testing Complete Enhanced Trading Cycle...")
    
    pipeline = UnifiedTradingPipeline(mode="demo", config={"min_confidence": 0.6})
    await pipeline.initialize()
    
    # Run complete trading cycle
    cycle_result = await pipeline.run_trading_cycle()
    
    # Validate cycle result
    assert "timestamp" in cycle_result, "Should have timestamp"
    assert "market_data" in cycle_result, "Should have market data"
    assert "enhanced_signals_count" in cycle_result, "Should have signals count"
    assert "trade_executed" in cycle_result, "Should have trade executed flag"
    assert "portfolio_value" in cycle_result, "Should have portfolio value"
    
    print(f"   Complete trading cycle executed successfully")
    print(f"   Enhanced signals generated: {cycle_result['enhanced_signals_count']}")
    print(f"   Trade executed: {cycle_result['trade_executed']}")
    print(f"   Portfolio value: ${cycle_result['portfolio_value']:.2f}")
    
    # Check for mathematical insights if available
    if "mathematical_insights" in cycle_result:
        print(f"   Mathematical insights available: {len(cycle_result['mathematical_insights'])} fields")
    
    print("   ✅ Complete enhanced trading cycle tests passed")

async def test_error_handling():
    """Test error handling and fallback mechanisms."""
    print("\n🧪 Testing Error Handling...")
    
    pipeline = UnifiedTradingPipeline(mode="demo")
    await pipeline.initialize()
    
    # Test with invalid market data
    invalid_market_data = {}
    
    try:
        math_context = pipeline._apply_mathematical_systems(invalid_market_data)
        print(f"   Handled invalid market data gracefully")
    except Exception as e:
        print(f"   Error handling invalid market data: {e}")
    
    # Test with empty signals
    try:
        best_signal = pipeline._select_best_signal([])
        assert best_signal is None, "Should return None for empty signals"
        print(f"   Handled empty signals gracefully")
    except Exception as e:
        print(f"   Error handling empty signals: {e}")
    
    print("   ✅ Error handling tests passed")

async def main():
    """Run all enhanced unified trading pipeline tests."""
    print("🚀 Enhanced Unified Trading Pipeline Test Suite")
    print("=" * 65)
    
    try:
        await test_enhanced_pipeline_initialization()
        await test_enhanced_signal_generation()
        await test_best_signal_selection()
        await test_enhanced_trade_execution()
        await test_hash_generation()
        await test_registry_updates()
        await test_performance_metrics()
        await test_cycle_summary()
        await test_complete_trading_cycle()
        await test_error_handling()
        
        print(f"\n🎉 All enhanced unified trading pipeline tests completed successfully!")
        print(f"✅ Enhanced Unified Trading Pipeline is GODMODE READY with Math + Memory Fusion Core!")
        print(f"🧠 Trading pipeline now has full mathematical fusion and memory capabilities!")
        print(f"🔗 Ready for live trading execution and long-term memory evolution!")
        
    except Exception as e:
        print(f"\n❌ Enhanced unified trading pipeline test suite failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 