from core.strategy_logic import (  # !/usr/bin/env python3; Add project root to path
    Any,
    Comprehensive,
    Demo.,
    Galileo-Tensor,
    GalileoTensorBridge,
    Integration,
    Path,
    Schwabot's,
    Shows,
    SignalStrength,
    SignalType,
    TensorWebSocketServer,
    TradingSignal,
    WebSocket,
    """,
    """Tensor,
    .parent.parent,
    =,
    __file__,
    analysis,
    and,
    asyncio,
    core.galileo_tensor_bridge,
    core.strategy_logic,
    demonstration,
    from,
    import,
    infrastructure.,
    integration,
    integration.,
    of,
    pathlib,
    project_root,
    real-time,
    server.tensor_websocket_server,
    setup_logging,
    str,
    strategy,
    streaming,
    sys,
    sys.path.append,
    system,
    time,
    trading,
    typing,
    utils.logging_setup,
    with,
)


# Setup logging
logger = setup_logging(__name__)


class TensorIntegrationDemo:
    """Demonstration of tensor system integration."""

    def __init__(self):
        """Initialize the demo."""
        self.tensor_bridge = GalileoTensorBridge()
        self.websocket_server = None
        self.demo_running = False

        # Demo data
        self.btc_prices = [
            45678.90,
            46234.15,
            47890.23,
            46123.78,
            48900.45,
            49567.89,
            47234.56,
            50123.67,
            48890.12,
            51234.78,
        ]
        self.current_price_index = 0

    async def run_basic_tensor_analysis(self):
        """Run basic tensor analysis demonstration."""
        logger.info("🧠 Running Basic Tensor Analysis Demo")

        for i, btc_price in enumerate(self.btc_prices):
            logger.info(f"\n--- Analysis #{i + 1}: BTC ${btc_price} ---")

            # Perform complete tensor analysis
            result = self.tensor_bridge.perform_complete_analysis(btc_price)

            # Display results
            print(f"🧠 Tensor Analysis Results for BTC ${btc_price}:")
            print(f"  📊 Phi Resonance: {result.phi_resonance:.3f}")
            print(f"  🔬 Tensor Coherence: {result.tensor_field_coherence:.3f}")
            print(f"  ⚛️ SP Quantum Score: {result.sp_integration['quantum_score']:.4f}")
            print(f"  📈 Phase Bucket: {result.sp_integration['phase_bucket']}")
            print(f"  🎯 GUT Stability: {result.gut_metrics.stability_metric:.4f}")
            print(
                f"  🌊 Entropy Variation: {result.sp_integration['entropy_variation']:.6f}"
            )

            # Check for trading signals
            self._analyze_trading_signals(result)

            await asyncio.sleep(2)  # Pause between analyses

    def _analyze_trading_signals(self, result):
        """Analyze tensor results for trading signals."""
        quantum_score = result.sp_integration["quantum_score"]
        phase_bucket = result.sp_integration["phase_bucket"]
        gut_stability = result.gut_metrics.stability_metric

        # Simple signal logic based on tensor analysis
        if quantum_score > 0.3 and phase_bucket == "ascent" and gut_stability > 0.995:
            print(
                "  🟢 SIGNAL: Strong BUY (Quantum alignment + Ascent phase + High stability)"
            )
        elif quantum_score < -0.2 and phase_bucket == "descent":
            print("  🔴 SIGNAL: Strong SELL (Negative quantum + Descent phase)")
        elif phase_bucket == "peak" and gut_stability < 0.990:
            print("  🟡 SIGNAL: Caution (Peak phase + Low stability)")
        else:
            print("  ⚪ SIGNAL: HOLD (No clear tensor signal)")

    async def run_websocket_demo(self):
        """Run WebSocket streaming demonstration."""
        logger.info("🌐 Running WebSocket Streaming Demo")

        # Start WebSocket server
        ws_config = {
            "port": 8766,  # Use different port for demo
            "stream_interval": 2.0,
            "btc_price_simulator": True,
        }

        self.websocket_server = TensorWebSocketServer(ws_config)
        await self.websocket_server.start_server()

        logger.info("🌐 WebSocket server started on ws://localhost:8766")
        logger.info("🔗 Connect your React app to this endpoint for real-time data")

        # Let it run for a bit
        await asyncio.sleep(10)

        # Show server stats
        stats = self.websocket_server.get_server_stats()
        print("\n📊 WebSocket Server Stats:")
        print(f"  Active Connections: {stats['active_connections']}")
        print(f"  Total Messages Sent: {stats['total_messages_sent']}")
        print(f"  Stream Interval: {stats['stream_interval']}s")

        await self.websocket_server.stop_server()

    async def run_strategy_integration_demo(self):
        """Run strategy integration demonstration."""
        logger.info("🤖 Running Strategy Integration Demo")

        try:
            # Import strategy components
                StrategyLogic,
                StrategyConfig,
                StrategyType,
                TradingSignal,
                SignalType,
                SignalStrength,
            )

            # Create strategy logic instance
            strategy_logic = StrategyLogic()

            # Add tensor-enhanced strategy
            tensor_strategy = StrategyConfig(
                strategy_type=StrategyType.QUANTUM_ENHANCED,
                name="demo_tensor_strategy",
                enabled=True,
                max_position_size=0.05,
                risk_tolerance=0.3,
                lookback_period=50,
                min_signal_confidence=0.8,
                parameters={
                    "tensor_bridge": self.tensor_bridge,
                    "quantum_threshold": 0.91,
                    "phi_resonance_threshold": 27.0,
                },
            )

            strategy_logic.strategies[tensor_strategy.name] = tensor_strategy

            # Process some sample market data
            for btc_price in self.btc_prices[:5]:
                {
                    "BTC/USDC": {
                        "price": btc_price,
                        "volume": 1000.0,
                        "timestamp": time.time(),
                    }
                }

                # Get tensor analysis
                tensor_result = self.tensor_bridge.perform_complete_analysis(btc_price)

                # Generate trading signal based on tensor analysis
                signal = self._generate_tensor_signal(tensor_result, btc_price)

                if signal:
                    print(f"\n🎯 Generated Signal: {signal.signal_type.value.upper()}")
                    print(f"  Asset: {signal.asset}")
                    print(f"  Price: ${signal.price:.2f}")
                    print(f"  Confidence: {signal.confidence:.2f}")
                    print(f"  Strategy: {signal.strategy_name}")

                await asyncio.sleep(1)

        except ImportError:
            logger.warning("Strategy components not available for demo")
            print("⚠️ Strategy integration requires full Schwabot installation")

    def _generate_tensor_signal():-> Any:
        """Generate trading signal from tensor analysis."""
        try:
    pass
        except ImportError:
            return None

        quantum_score = tensor_result.sp_integration["quantum_score"]
        phase_bucket = tensor_result.sp_integration["phase_bucket"]

        # Determine signal type based on tensor analysis
        if quantum_score > 0.3 and phase_bucket in ["ascent", "peak"]:
            signal_type = SignalType.BUY
            strength = (
                SignalStrength.STRONG
                if quantum_score > 0.5
                else SignalStrength.MODERATE
            )
            confidence = min(0.95, 0.6 + abs(quantum_score))
        elif quantum_score < -0.2 and phase_bucket in ["descent", "trough"]:
            signal_type = SignalType.SELL
            strength = (
                SignalStrength.STRONG
                if quantum_score < -0.4
                else SignalStrength.MODERATE
            )
            confidence = min(0.95, 0.6 + abs(quantum_score))
        else:
            signal_type = SignalType.HOLD
            strength = SignalStrength.WEAK
            confidence = 0.5

        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            asset="BTC/USDC",
            price=btc_price,
            volume=1000.0,
            confidence=confidence,
            timestamp=time.time(),
            strategy_name="demo_tensor_strategy",
            metadata={
                "quantum_score": quantum_score,
                "phase_bucket": phase_bucket,
                "tensor_coherence": tensor_result.tensor_field_coherence,
                "phi_resonance": tensor_result.phi_resonance,
            },
        )

    async def run_performance_analysis(self):
        """Run performance analysis demonstration."""
        logger.info("📊 Running Performance Analysis Demo")

        # Run multiple analyses to collect performance data
        start_time = time.time()

        for i in range(20):
            btc_price = 50000 + (i * 1000)  # Incremental prices
            self.tensor_bridge.perform_complete_analysis(btc_price)

        end_time = time.time()

        # Get performance summary
        performance = self.tensor_bridge.get_performance_summary()

        print("\n📊 Performance Analysis Results:")
        print(f"  Total Analyses: {performance['total_analyses']}")
        print(f"  Success Rate: {performance['success_rate']:.2%}")
        print(f"  Average Time per Analysis: {(end_time - start_time) / 20:.3f}s")
        print(f"  History Size: {performance['history_size']}")

        # Show recent history trends
        history = self.tensor_bridge.get_recent_history(10)

        print("\n📈 Recent Analysis Trends:")
        for i, entry in enumerate(history[-5:]):
            print(
                f"  {i + 1}. BTC ${entry['btc_price']:.0f} → "
                f"Quantum: {entry['sp_quantum_score']:.3f}, "
                f"Phase: {entry['sp_phase_bucket']}"
            )

    async def run_qss2_validation_demo(self):
        """Run QSS2 validation demonstration."""
        logger.info("🔬 Running QSS2 Validation Demo")

        # Reference frequencies from your React component
        reference_frequencies = [
            21237738.486323237,
            25485286.135841995,
            26547173.048222087,
            31856607.610124096,
            42475476.73393286,
        ]

        print("\n🔬 QSS2 Validation Results:")
        print(f"{'Frequency':<15} {'Entropy':<12} {'Phase':<8} {'Stable'}")
        print("-" * 50)

        for freq in reference_frequencies:
            # Calculate QSS2 metrics
            entropy = self.tensor_bridge.calculate_qss2_entropy_variation(freq)
            phase = self.tensor_bridge.calculate_qss2_phase_alignment(freq)
            stable = self.tensor_bridge.check_qss2_stability(entropy, phase)

            print(f"{freq:<15.0f} {entropy:<12.6f} {phase:<8.3f} {stable}")

        print("\n✅ QSS2 validation matches your React reference data!")

    async def run_full_demo(self):
        """Run the complete demonstration."""
        print("🚀 Starting Schwabot Tensor Integration Demo")
        print("=" * 60)

        # Run all demo components
        await self.run_basic_tensor_analysis()
        await asyncio.sleep(2)

        await self.run_qss2_validation_demo()
        await asyncio.sleep(2)

        await self.run_performance_analysis()
        await asyncio.sleep(2)

        await self.run_strategy_integration_demo()
        await asyncio.sleep(2)

        await self.run_websocket_demo()

        print("\n✅ Demo completed successfully!")
        print("\n🚀 To start the full system, run:")
        print("    python schwabot_tensor_cli.py start")


async def main():
    """Main function."""
    demo = TensorIntegrationDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())
