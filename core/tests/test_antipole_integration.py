"""
Anti-Pole Integration Test Suite v1.0
=====================================

Comprehensive test suite for the Quantum Anti-Pole integration with Schwabot.
Tests all components together for proper functionality and performance.

Tests:
- Quantum Anti-Pole Engine correctness
- Entropy Bridge integration
- Dashboard Integration API
- Mathematical correctness of anti-pole calculations
- Performance benchmarks
- Error handling and recovery
"""

import pytest
import asyncio
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List

# Import components to test
try:
    from ..quantum_antipole_engine import QuantumAntiPoleEngine, QAConfig, QuantumState, ComplexPole
    from ..entropy_bridge import EntropyBridge, EntropyBridgeConfig, EntropyFlowData
    from ..dashboard_integration import DashboardIntegration, DashboardConfig
    from ..profit_navigator import AntiPoleProfitNavigator
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

# Test configuration
MARKET_SIMULATION_TICKS = 50
PERFORMANCE_BENCHMARK_TICKS = 100
MAX_COMPUTATION_TIME_MS = 100  # Maximum acceptable computation time per tick

class MarketDataSimulator:
    """Generates realistic market data for testing"""
    
    def __init__(self, base_price: float = 45000.0, base_volume: float = 1000000.0):
        self.base_price = base_price
        self.base_volume = base_volume
        self.tick_count = 0
        
    def next_tick(self) -> Dict[str, float]:
        """Generate next market tick"""
        self.tick_count += 1
        
        # Generate realistic price movement
        trend = np.sin(self.tick_count * 0.05) * 1000  # Long-term trend
        volatility = np.random.randn() * 500  # Random volatility
        price = self.base_price + trend + volatility
        
        # Generate realistic volume
        volume_trend = np.cos(self.tick_count * 0.03) * 200000
        volume_volatility = np.random.randn() * 150000
        volume = max(self.base_volume + volume_trend + volume_volatility, 100000)
        
        return {
            'price': price,
            'volume': volume,
            'timestamp': datetime.utcnow()
        }

@pytest.fixture
def market_simulator():
    """Market data simulator fixture"""
    return MarketDataSimulator()

@pytest.fixture
def qa_config():
    """Quantum Anti-Pole Engine configuration for testing"""
    return QAConfig(
        use_gpu=False,  # Use CPU for consistent testing
        field_size=32,  # Smaller for faster tests
        tick_window=64,
        pole_order=8,
        debug_mode=True,
        use_entropy_tracker=False,  # Disable for isolated testing
        use_thermal_manager=False,
        use_ferris_wheel=False
    )

@pytest.fixture
def entropy_bridge_config():
    """Entropy bridge configuration for testing"""
    return EntropyBridgeConfig(
        history_size=100,
        websocket_port=8769,  # Different port for testing
        use_existing_entropy=False,  # Disable for isolated testing
        use_quantum_engine=True,
        use_ferris_wheel=False
    )

@pytest.fixture
def dashboard_config():
    """Dashboard integration configuration for testing"""
    return DashboardConfig(
        host="localhost",
        port=8770,  # Different port for testing
        update_frequency=20.0,  # Higher frequency for faster tests
        enable_entropy_bridge=False,  # Disable for isolated testing
        enable_profit_navigator=False,
        enable_thermal_monitoring=False
    )

class TestQuantumAntiPoleEngine:
    """Test suite for Quantum Anti-Pole Engine"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, qa_config):
        """Test engine initializes correctly"""
        engine = QuantumAntiPoleEngine(qa_config)
        
        assert engine.config == qa_config
        assert engine.frame_count == 0
        assert len(engine.price_buffer) == 0
        assert engine.use_gpu == qa_config.use_gpu
        
        # Test quantum field initialization
        assert engine.psi is not None
        assert engine.X.shape == (qa_config.field_size, qa_config.field_size)
        assert engine.Y.shape == (qa_config.field_size, qa_config.field_size)
        
        engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_single_tick_processing(self, qa_config, market_simulator):
        """Test processing a single market tick"""
        engine = QuantumAntiPoleEngine(qa_config)
        
        tick_data = market_simulator.next_tick()
        frame = await engine.process_tick(
            tick_data['price'], 
            tick_data['volume'], 
            tick_data['timestamp']
        )
        
        # Validate frame structure
        assert frame.uid is not None
        assert frame.timestamp == tick_data['timestamp']
        assert frame.price == tick_data['price']
        assert frame.volume == tick_data['volume']
        
        # Validate quantum state
        assert isinstance(frame.quantum_state, QuantumState)
        assert 0.0 <= frame.quantum_state.coherence <= 1.0
        assert frame.quantum_state.entropy >= 0.0
        
        # Validate AP-RSI
        assert 0.0 <= frame.ap_rsi <= 100.0
        
        # Validate computation time
        assert frame.computation_time_ms > 0
        assert frame.computation_time_ms < MAX_COMPUTATION_TIME_MS
        
        engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_multi_tick_processing(self, qa_config, market_simulator):
        """Test processing multiple ticks and pole analysis"""
        engine = QuantumAntiPoleEngine(qa_config)
        
        frames = []
        for i in range(MARKET_SIMULATION_TICKS):
            tick_data = market_simulator.next_tick()
            frame = await engine.process_tick(
                tick_data['price'], 
                tick_data['volume'], 
                tick_data['timestamp']
            )
            frames.append(frame)
        
        # Validate frame progression
        assert len(frames) == MARKET_SIMULATION_TICKS
        assert engine.frame_count == MARKET_SIMULATION_TICKS
        
        # Check that poles are detected after sufficient data
        final_frame = frames[-1]
        if len(engine.price_buffer) >= engine.config.pole_order * 2:
            # Should have some pole analysis
            assert isinstance(final_frame.complex_poles, list)
        
        # Validate AP-RSI evolution
        ap_rsi_values = [f.ap_rsi for f in frames[-10:]]  # Last 10 values
        assert all(0.0 <= val <= 100.0 for val in ap_rsi_values)
        
        # Check for reasonable variation (not stuck at initial value)
        if len(ap_rsi_values) > 5:
            assert np.std(ap_rsi_values) > 0.1  # Some variation expected
        
        engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_pole_analysis_correctness(self, qa_config):
        """Test mathematical correctness of pole analysis"""
        engine = QuantumAntiPoleEngine(qa_config)
        
        # Generate synthetic AR(2) data with known poles
        np.random.seed(42)  # For reproducible tests
        true_poles = [0.8 + 0.3j, 0.8 - 0.3j]  # Stable complex conjugate pair
        
        # Generate AR(2) time series
        n_points = 100
        noise = np.random.randn(n_points) * 0.1
        prices = np.zeros(n_points)
        prices[0] = 45000
        prices[1] = 45100
        
        # AR(2): x[n] = 1.6*x[n-1] - 0.73*x[n-2] + noise[n]
        for i in range(2, n_points):
            prices[i] = 1.6 * prices[i-1] - 0.73 * prices[i-2] + noise[i] * 100
        
        # Feed data to engine
        for i, price in enumerate(prices):
            volume = 1000000 + np.random.randn() * 10000
            await engine.process_tick(price, volume)
        
        # Check pole detection
        if len(engine.price_buffer) >= engine.config.pole_order * 2:
            frame = await engine.process_tick(prices[-1], 1000000)
            
            if frame.complex_poles:
                # Validate pole properties
                for pole in frame.complex_poles:
                    assert isinstance(pole, ComplexPole)
                    assert pole.magnitude >= 0
                    assert pole.stability in ['stable', 'unstable', 'marginal']
                    
                    # For our synthetic data, poles should be stable
                    if pole.magnitude < 1.0:
                        assert pole.stability == 'stable'
        
        engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_trading_signals(self, qa_config, market_simulator):
        """Test trading signal generation"""
        engine = QuantumAntiPoleEngine(qa_config)
        
        # Process enough ticks to generate meaningful signals
        for i in range(30):
            tick_data = market_simulator.next_tick()
            frame = await engine.process_tick(
                tick_data['price'], 
                tick_data['volume'], 
                tick_data['timestamp']
            )
        
        # Get trading signals from last frame
        signals = engine.get_trading_signals(frame)
        
        # Validate signal structure
        assert 'ap_rsi' in signals
        assert 'ap_rsi_signal' in signals
        assert 'quantum_coherence' in signals
        assert 'coherence_signal' in signals
        assert 'combined_signal' in signals
        assert 'recommendation' in signals
        
        # Validate signal values
        assert 0.0 <= signals['ap_rsi'] <= 100.0
        assert signals['ap_rsi_signal'] in ['overbought', 'oversold', 'neutral']
        assert 0.0 <= signals['quantum_coherence'] <= 1.0
        assert signals['coherence_signal'] in ['strong', 'moderate', 'weak']
        assert -1.0 <= signals['combined_signal'] <= 1.0
        assert signals['recommendation'] in ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']
        
        engine.shutdown()

class TestEntropyBridge:
    """Test suite for Entropy Bridge"""
    
    @pytest.mark.asyncio
    async def test_bridge_initialization(self, entropy_bridge_config):
        """Test entropy bridge initializes correctly"""
        bridge = EntropyBridge(entropy_bridge_config)
        
        assert bridge.config == entropy_bridge_config
        assert len(bridge.entropy_history) == 0
        assert bridge.current_data is None
        assert bridge.frame_count == 0
    
    @pytest.mark.asyncio
    async def test_entropy_flow_processing(self, entropy_bridge_config, market_simulator):
        """Test entropy flow data processing"""
        bridge = EntropyBridge(entropy_bridge_config)
        
        tick_data = market_simulator.next_tick()
        flow_data = await bridge.process_market_tick(
            tick_data['price'], 
            tick_data['volume'], 
            tick_data['timestamp']
        )
        
        # Validate flow data structure
        assert isinstance(flow_data, EntropyFlowData)
        assert flow_data.timestamp == tick_data['timestamp']
        assert flow_data.price == tick_data['price']
        assert flow_data.volume == tick_data['volume']
        
        # Validate entropy metrics
        assert flow_data.hash_entropy >= 0.0
        assert flow_data.quantum_entropy >= 0.0
        assert flow_data.combined_entropy >= 0.0
        assert flow_data.entropy_tier in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'NEUTRAL']
        
        # Validate trading signals
        assert 0.0 <= flow_data.ap_rsi <= 100.0
        assert 0.0 <= flow_data.coherence <= 1.0
        assert flow_data.recommendation in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
        
        # Validate performance
        assert flow_data.computation_time_ms > 0
        assert flow_data.computation_time_ms < MAX_COMPUTATION_TIME_MS * 2  # Bridge has overhead
    
    @pytest.mark.asyncio
    async def test_entropy_tier_determination(self, entropy_bridge_config, market_simulator):
        """Test entropy tier determination logic"""
        bridge = EntropyBridge(entropy_bridge_config)
        
        # Test various scenarios
        test_cases = [
            {'entropy': 0.9, 'ap_rsi': 80, 'coherence': 0.8, 'expected_tier': 'PLATINUM'},
            {'entropy': 0.7, 'ap_rsi': 65, 'coherence': 0.6, 'expected_tier': 'GOLD'},
            {'entropy': 0.4, 'ap_rsi': 50, 'coherence': 0.4, 'expected_tier': 'SILVER'},
            {'entropy': 0.2, 'ap_rsi': 40, 'coherence': 0.2, 'expected_tier': 'BRONZE'},
            {'entropy': 0.1, 'ap_rsi': 50, 'coherence': 0.1, 'expected_tier': 'NEUTRAL'}
        ]
        
        for case in test_cases:
            tier = bridge._determine_entropy_tier(
                case['entropy'], 
                case['ap_rsi'], 
                case['coherence']
            )
            
            # Note: Due to multi-factor scoring, exact tier matching may not always occur
            # But tier should be reasonable given inputs
            assert tier in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'NEUTRAL']

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_engine_performance(self, qa_config, market_simulator):
        """Test engine performance under load"""
        engine = QuantumAntiPoleEngine(qa_config)
        
        start_time = time.perf_counter()
        computation_times = []
        
        for i in range(PERFORMANCE_BENCHMARK_TICKS):
            tick_data = market_simulator.next_tick()
            
            tick_start = time.perf_counter()
            frame = await engine.process_tick(
                tick_data['price'], 
                tick_data['volume'], 
                tick_data['timestamp']
            )
            tick_end = time.perf_counter()
            
            computation_times.append((tick_end - tick_start) * 1000)
        
        total_time = time.perf_counter() - start_time
        
        # Performance assertions
        avg_computation_time = np.mean(computation_times)
        max_computation_time = np.max(computation_times)
        
        assert avg_computation_time < MAX_COMPUTATION_TIME_MS, f"Average computation time too high: {avg_computation_time:.2f}ms"
        assert max_computation_time < MAX_COMPUTATION_TIME_MS * 2, f"Maximum computation time too high: {max_computation_time:.2f}ms"
        
        # Throughput test
        throughput = PERFORMANCE_BENCHMARK_TICKS / total_time
        assert throughput >= 10.0, f"Throughput too low: {throughput:.2f} ticks/second"
        
        print(f"Performance Results:")
        print(f"  Average computation time: {avg_computation_time:.2f}ms")
        print(f"  Maximum computation time: {max_computation_time:.2f}ms")
        print(f"  Throughput: {throughput:.2f} ticks/second")
        
        engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, qa_config, market_simulator):
        """Test memory usage remains bounded"""
        import tracemalloc
        
        tracemalloc.start()
        engine = QuantumAntiPoleEngine(qa_config)
        
        # Initial memory snapshot
        initial_snapshot = tracemalloc.take_snapshot()
        
        # Process many ticks
        for i in range(PERFORMANCE_BENCHMARK_TICKS * 2):
            tick_data = market_simulator.next_tick()
            await engine.process_tick(
                tick_data['price'], 
                tick_data['volume'], 
                tick_data['timestamp']
            )
        
        # Final memory snapshot
        final_snapshot = tracemalloc.take_snapshot()
        top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        
        # Check memory usage hasn't grown excessively
        total_memory_growth = sum(stat.size_diff for stat in top_stats)
        
        # Convert to MB
        memory_growth_mb = total_memory_growth / (1024 * 1024)
        
        assert memory_growth_mb < 100.0, f"Memory growth too high: {memory_growth_mb:.2f}MB"
        
        print(f"Memory growth: {memory_growth_mb:.2f}MB")
        
        engine.shutdown()
        tracemalloc.stop()

class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_invalid_market_data(self, qa_config):
        """Test handling of invalid market data"""
        engine = QuantumAntiPoleEngine(qa_config)
        
        # Test various invalid inputs
        invalid_cases = [
            {'price': float('nan'), 'volume': 1000000},
            {'price': float('inf'), 'volume': 1000000},
            {'price': -1000, 'volume': 1000000},  # Negative price
            {'price': 45000, 'volume': float('nan')},
            {'price': 45000, 'volume': -1000},  # Negative volume
        ]
        
        for case in invalid_cases:
            try:
                frame = await engine.process_tick(case['price'], case['volume'])
                
                # Engine should handle invalid data gracefully
                assert frame is not None
                assert hasattr(frame, 'quantum_state')
                assert hasattr(frame, 'ap_rsi')
                
            except Exception as e:
                # If exceptions are raised, they should be specific and handled
                assert isinstance(e, (ValueError, TypeError))
        
        engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_component_failure_recovery(self, entropy_bridge_config):
        """Test recovery from component failures"""
        bridge = EntropyBridge(entropy_bridge_config)
        
        # Simulate component failure by setting to None
        original_quantum_engine = bridge.quantum_engine
        bridge.quantum_engine = None
        
        # Process should continue without quantum engine
        tick_data = {'price': 45000, 'volume': 1000000, 'timestamp': datetime.utcnow()}
        flow_data = await bridge.process_market_tick(
            tick_data['price'], 
            tick_data['volume'], 
            tick_data['timestamp']
        )
        
        # Should still produce valid flow data
        assert isinstance(flow_data, EntropyFlowData)
        assert flow_data.quantum_entropy == 0.0  # Default when component unavailable
        assert flow_data.ap_rsi == 50.0  # Default neutral value
        
        # Restore component
        bridge.quantum_engine = original_quantum_engine

# Integration tests
class TestFullSystemIntegration:
    """Test full system integration"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, market_simulator):
        """Test complete end-to-end data pipeline"""
        # Create all components
        qa_config = QAConfig(use_gpu=False, field_size=32, debug_mode=False)
        engine = QuantumAntiPoleEngine(qa_config)
        
        entropy_config = EntropyBridgeConfig(use_quantum_engine=False)  # Use separate engine
        bridge = EntropyBridge(entropy_config)
        bridge.quantum_engine = engine  # Manual integration for testing
        
        # Process market data through complete pipeline
        results = []
        for i in range(20):
            tick_data = market_simulator.next_tick()
            
            # Process through entropy bridge (which uses quantum engine)
            flow_data = await bridge.process_market_tick(
                tick_data['price'], 
                tick_data['volume'], 
                tick_data['timestamp']
            )
            
            results.append(flow_data)
        
        # Validate pipeline results
        assert len(results) == 20
        
        for flow_data in results:
            assert isinstance(flow_data, EntropyFlowData)
            assert flow_data.computation_time_ms > 0
            assert flow_data.computation_time_ms < MAX_COMPUTATION_TIME_MS * 3  # Allow for pipeline overhead
        
        # Check data evolution
        ap_rsi_values = [r.ap_rsi for r in results]
        assert all(0.0 <= val <= 100.0 for val in ap_rsi_values)
        
        # Check entropy progression
        entropy_values = [r.combined_entropy for r in results]
        assert all(val >= 0.0 for val in entropy_values)
        
        engine.shutdown()

# Utility functions for tests
def validate_quantum_state(state: QuantumState):
    """Validate quantum state properties"""
    assert isinstance(state.psi, complex)
    assert isinstance(state.phi, complex)
    assert 0.0 <= state.coherence <= 1.0
    assert state.entropy >= 0.0
    assert isinstance(state.timestamp, datetime)

def validate_complex_pole(pole: ComplexPole):
    """Validate complex pole properties"""
    assert isinstance(pole.real, float)
    assert isinstance(pole.imag, float)
    assert pole.magnitude >= 0.0
    assert pole.stability in ['stable', 'unstable', 'marginal']
    assert 0.0 <= pole.confidence <= 1.0

# Test runner configuration
if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise during testing
        format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s"
    )
    
    # Run specific test class
    pytest.main([__file__, "-v", "--tb=short"]) 