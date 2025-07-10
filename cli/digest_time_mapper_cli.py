#!/usr/bin/env python3
"""
Digest Time Mapper CLI - Phase Wheel & Temporal Socketing Interface
Provides CLI commands for processing millisecond BTC price ticks into 16-bit frames
and 256-bit SHA digests for quantum-enhanced trading strategy selection.

Commands:
  * init                    – initialize digest time mapper
  * process-tick <price>    – process a single price tick
  * generate-digest         – generate 256-bit SHA digest from frames
  * ferris-wheel <duration> – run continuous processing loop
  * analyze <digest>        – analyze temporal patterns in digest
  * status                  – show mapper statistics
  * stop                    – stop processing loop
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

from core.digest_time_mapper import DigestResult, DigestTimeMapper, PriceTick
from core.quantum_mathematical_bridge import QuantumMathematicalBridge
from core.vector_registry import StrategyVector, VectorRegistry

logger = logging.getLogger(__name__)


class DigestTimeMapperCLI:
    """CLI interface for Digest Time Mapper operations."""

    def __init__(self):
        self.mapper: Optional[DigestTimeMapper] = None
        self.vector_registry: Optional[VectorRegistry] = None
        self.quantum_bridge: Optional[QuantumMathematicalBridge] = None
        self.is_running = False

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def init(self, frame_window_ms: int = 1000, phase_period_ms: int = 60000) -> bool:
        """Initialize the digest time mapper and related components."""
        try:
            logger.info("Initializing Digest Time Mapper...")

            # Initialize mapper
            self.mapper = DigestTimeMapper(frame_window_ms=frame_window_ms, phase_period_ms=phase_period_ms)

            # Initialize vector registry
            self.vector_registry = VectorRegistry("data/digest_vector_registry.json")

            # Initialize quantum bridge
            self.quantum_bridge = QuantumMathematicalBridge(quantum_dimension=16)

            logger.info("✅ Digest Time Mapper initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Digest Time Mapper: {e}")
            return False

    def process_tick(self, price: float, volume: float = 0.0, bid: float = 0.0, ask: float = 0.0) -> Optional[str]:
        """Process a single price tick."""
        if not self.mapper:
            logger.error("❌ Mapper not initialized. Run 'init' first.")
            return None

        try:
            frame = self.mapper.process_millisecond_tick(price=price, volume=volume, bid=bid, ask=ask)

            if frame:
                return f"✅ Generated frame {frame.frame_index} (entropy: {frame.entropy_level:.3f})"
            else:
                return "⏳ Tick buffered (not enough data for frame yet)"

        except Exception as e:
            logger.error(f"❌ Error processing tick: {e}")
            return None

    def generate_digest(self, frame_count: int = 16) -> Optional[str]:
        """Generate 256-bit SHA digest from accumulated frames."""
        if not self.mapper:
            logger.error("❌ Mapper not initialized. Run 'init' first.")
            return None

        try:
            digest_result = self.mapper.generate_phase_wheel_digest(frame_count)

            if digest_result:
                # Create strategy vector for the digest
                strategy_vector = StrategyVector(
                    digest="",
                    strategy_id=f"digest_{digest_result.digest_hex[:8]}",
                    asset_focus="BTC",
                    entry_confidence=digest_result.entropy_score,
                    exit_confidence=digest_result.temporal_coherence,
                    position_size=0.5,
                    stop_loss_pct=2.0,
                    take_profit_pct=5.0,
                    rsi_band=50,
                    volatility_class=1,
                    entropy_band=digest_result.entropy_score,
                )

                # Register with vector registry
                if self.vector_registry:
                    self.vector_registry.register_digest(digest_result.digest, strategy_vector)

                return (
                    f"✅ Generated digest: {digest_result.digest_hex[:16]}...\n"
                    f"   Entropy: {digest_result.entropy_score:.3f}\n"
                    f"   Coherence: {digest_result.temporal_coherence:.3f}\n"
                    f"   Socket matches: {len(digest_result.socket_matches)}"
                )
            else:
                return "⏳ Not enough frames for digest generation"

        except Exception as e:
            logger.error(f"❌ Error generating digest: {e}")
            return None

    def ferris_wheel(self, duration_seconds: int = 60) -> bool:
        """Run the Ferris Wheel continuous processing loop."""
        if not self.mapper:
            logger.error("❌ Mapper not initialized. Run 'init' first.")
            return False

        try:
            logger.info(f"🎡 Starting Ferris Wheel loop for {duration_seconds} seconds...")

            # Simulate price stream
            def price_stream():
                import random

                base_price = 50000.0
                start_time = time.time()

                while time.time() - start_time < duration_seconds:
                    # Simulate realistic price movement
                    change = random.gauss(0, 50)  # Normal distribution
                    base_price += change
                    base_price = max(base_price, 1000.0)  # Minimum price

                    yield base_price, time.time()
                    time.sleep(0.01)  # 10ms intervals

            # Run the loop
            self.is_running = True
            digest_count = 0

            for digest_result in self.mapper.ferris_wheel_loop(price_stream()):
                if not self.is_running:
                    break

                digest_count += 1
                print(
                    f"🎡 Digest {digest_count}: {digest_result.digest_hex[:16]}... "
                    f"(entropy: {digest_result.entropy_score:.3f})"
                )

                # Register with vector registry
                if self.vector_registry:
                    strategy_vector = StrategyVector(
                        digest="",
                        strategy_id=f"ferris_{digest_result.digest_hex[:8]}",
                        asset_focus="BTC",
                        entry_confidence=digest_result.entropy_score,
                        exit_confidence=digest_result.temporal_coherence,
                        position_size=0.5,
                        stop_loss_pct=2.0,
                        take_profit_pct=5.0,
                        rsi_band=50,
                        volatility_class=1,
                        entropy_band=digest_result.entropy_score,
                    )
                    self.vector_registry.register_digest(digest_result.digest, strategy_vector)

            logger.info(f"🎡 Ferris Wheel completed: {digest_count} digests generated")
            return True

        except Exception as e:
            logger.error(f"❌ Error in Ferris Wheel loop: {e}")
            return False
        finally:
            self.is_running = False

    def analyze(self, digest_hex: str) -> Optional[str]:
        """Analyze temporal patterns in a digest."""
        if not self.mapper:
            logger.error("❌ Mapper not initialized. Run 'init' first.")
            return None

        try:
            # Convert hex to bytes
            digest_bytes = bytes.fromhex(digest_hex)

            # Perform temporal analysis
            analysis = self.mapper.temporal_socket_analysis(digest_bytes)

            if analysis:
                result = f"🔍 Temporal Analysis for {digest_hex[:16]}...\n"
                result += f"   Entropy: {analysis.get('digest_entropy', 0):.3f}\n"
                result += f"   Coherence: {analysis.get('coherence_score', 0):.3f}\n"
                result += f"   Pattern matches: {len(analysis.get('pattern_matches', []))}\n"

                # Add socket details
                for socket_id, socket_analysis in analysis.get('temporal_sockets', {}).items():
                    result += f"   {socket_id}: match={socket_analysis.get('match_score', 0):.3f}\n"

                return result
            else:
                return "❌ Analysis failed"

        except Exception as e:
            logger.error(f"❌ Error analyzing digest: {e}")
            return None

    def status(self) -> str:
        """Show mapper statistics and status."""
        if not self.mapper:
            return "❌ Mapper not initialized. Run 'init' first."

        try:
            stats = self.mapper.get_mapper_stats()

            result = "📊 Digest Time Mapper Status\n"
            result += "=" * 40 + "\n"
            result += f"Processing: {'🟢 Active' if stats['processing_active'] else '🔴 Inactive'}\n"
            result += f"Ticks processed: {stats['total_ticks_processed']}\n"
            result += f"Frames generated: {stats['total_frames_generated']}\n"
            result += f"Digests created: {stats['total_digests_created']}\n"
            result += f"Avg processing time: {stats['avg_processing_time']:.4f}s\n"
            result += f"Phase wheel position: {stats['phase_wheel_position']:.3f} rad\n"
            result += f"Temporal sockets: {stats['temporal_socket_count']}\n"
            result += f"Backend: {stats['backend_info']}\n"

            # Add buffer info
            buffers = stats['current_buffer_sizes']
            result += f"Tick buffer: {buffers['tick_buffer']}\n"
            result += f"Frame buffer: {buffers['frame_buffer']}\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            return f"❌ Status error: {e}"

    def stop(self) -> str:
        """Stop the processing loop."""
        if not self.mapper:
            return "❌ Mapper not initialized."

        try:
            self.mapper.stop_processing()
            self.is_running = False
            return "✅ Processing stopped"

        except Exception as e:
            logger.error(f"❌ Error stopping processing: {e}")
            return f"❌ Stop error: {e}"

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.mapper:
                self.mapper.stop_processing()

            if self.quantum_bridge:
                self.quantum_bridge.cleanup_quantum_resources()

            logger.info("🧹 Resources cleaned up")

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Digest Time Mapper CLI - Phase Wheel & Temporal Socketing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                           # Initialize mapper
  %(prog)s process-tick 50000             # Process a price tick
  %(prog)s generate-digest                # Generate digest from frames
  %(prog)s ferris-wheel 30                # Run 30-second processing loop
  %(prog)s analyze a1b2c3d4...            # Analyze digest patterns
  %(prog)s status                         # Show mapper status
  %(prog)s stop                           # Stop processing loop
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize digest time mapper')
    init_parser.add_argument(
        '--frame-window',
        type=int,
        default=1000,
        help='Frame window in milliseconds (default: 1000)',
    )
    init_parser.add_argument(
        '--phase-period',
        type=int,
        default=60000,
        help='Phase period in milliseconds (default: 60000)',
    )

    # Process tick command
    tick_parser = subparsers.add_parser('process-tick', help='Process a price tick')
    tick_parser.add_argument('price', type=float, help='Price value')
    tick_parser.add_argument('--volume', type=float, default=0.0, help='Volume')
    tick_parser.add_argument('--bid', type=float, default=0.0, help='Bid price')
    tick_parser.add_argument('--ask', type=float, default=0.0, help='Ask price')

    # Generate digest command
    digest_parser = subparsers.add_parser('generate-digest', help='Generate SHA digest')
    digest_parser.add_argument('--frame-count', type=int, default=16, help='Number of frames to use (default: 16)')

    # Ferris wheel command
    ferris_parser = subparsers.add_parser('ferris-wheel', help='Run continuous processing loop')
    ferris_parser.add_argument('duration', type=int, help='Duration in seconds')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze digest patterns')
    analyze_parser.add_argument('digest', help='Digest hex string')

    # Status command
    subparsers.add_parser('status', help='Show mapper status')

    # Stop command
    subparsers.add_parser('stop', help='Stop processing loop')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create CLI instance
    cli = DigestTimeMapperCLI()

    try:
        if args.command == 'init':
            success = cli.init(args.frame_window, args.phase_period)
            sys.exit(0 if success else 1)

        elif args.command == 'process-tick':
            result = cli.process_tick(args.price, args.volume, args.bid, args.ask)
            if result:
                print(result)
            else:
                sys.exit(1)

        elif args.command == 'generate-digest':
            result = cli.generate_digest(args.frame_count)
            if result:
                print(result)
            else:
                sys.exit(1)

        elif args.command == 'ferris-wheel':
            success = cli.ferris_wheel(args.duration)
            sys.exit(0 if success else 1)

        elif args.command == 'analyze':
            result = cli.analyze(args.digest)
            if result:
                print(result)
            else:
                sys.exit(1)

        elif args.command == 'status':
            print(cli.status())

        elif args.command == 'stop':
            print(cli.stop())

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        cli.stop()
    except Exception as e:
        logger.error(f"❌ CLI error: {e}")
        sys.exit(1)
    finally:
        cli.cleanup()


if __name__ == "__main__":
    main()
