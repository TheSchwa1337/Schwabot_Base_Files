#!/usr/bin/env python3
"""
Automated Trading Pipeline CLI - Unified Decision Engine Interface
Provides CLI commands for running the complete automated trading pipeline
with full transparency and decision explanation.

Commands:
  * init                    – initialize the trading pipeline
  * process-tick <price>    – process a single price tick
  * run-stream <duration>   – run continuous pipeline on simulated stream
  * explain                 – explain last trading decision
  * metrics                 – show pipeline performance metrics
  * decisions               – show recent trading decisions
  * stop                    – stop continuous pipeline
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

from core.automated_trading_pipeline import AutomatedTradingPipeline
from core.pure_profit_calculator import ProcessingMode, StrategyParameters

logger = logging.getLogger(__name__)


class AutomatedTradingPipelineCLI:
    """CLI interface for Automated Trading Pipeline operations."""

    def __init__(self):
        self.pipeline: Optional[AutomatedTradingPipeline] = None

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def init(
        self,
        risk_tolerance: float = 0.02,
        profit_target: float = 0.05,
        position_size: float = 0.1,
        processing_mode: str = "hybrid",
        registry_path: str = "data/trading_vector_registry.json",
    ) -> bool:
        """Initialize the automated trading pipeline."""
        try:
            logger.info("Initializing Automated Trading Pipeline...")

            # Create strategy parameters
            strategy_params = StrategyParameters(
                risk_tolerance=risk_tolerance,
                profit_target=profit_target,
                position_size=position_size,
                tensor_depth=4,
                hash_memory_depth=100,
            )

            # Parse processing mode
            mode_map = {
                "gpu": ProcessingMode.GPU_ACCELERATED,
                "cpu": ProcessingMode.CPU_FALLBACK,
                "hybrid": ProcessingMode.HYBRID,
                "safe": ProcessingMode.SAFE_MODE,
            }
            mode = mode_map.get(processing_mode.lower(), ProcessingMode.HYBRID)

            # Create pipeline
            self.pipeline = AutomatedTradingPipeline(
                registry_path=registry_path,
                profit_calculator_params=strategy_params,
                processing_mode=mode,
            )

            logger.info("✅ Automated Trading Pipeline initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline: {e}")
            return False

    def process_tick(self, price: float, volume: float = 0.0, bid: float = 0.0, ask: float = 0.0) -> Optional[str]:
        """Process a single price tick through the pipeline."""
        if not self.pipeline:
            return "❌ Pipeline not initialized. Run 'init' first."

        try:
            decision = self.pipeline.process_price_tick(price, volume, bid, ask)

            if decision:
                return (
                    f"🎯 TRADING DECISION MADE:\n"
                    f"   Strategy: {decision.strategy_vector.strategy_id}\n"
                    f"   Confidence: {decision.confidence_score:.3f}\n"
                    f"   Position Size: {decision.position_size:.2%}\n"
                    f"   Entry Price: ${decision.entry_price:,.2f}\n"
                    f"   Stop Loss: ${decision.stop_loss:,.2f}\n"
                    f"   Take Profit: ${decision.take_profit:,.2f}\n"
                    f"   Reason: {decision.decision_reason}"
                )
            else:
                return "⏳ No trading decision - conditions not met"

        except Exception as e:
            logger.error(f"❌ Error processing tick: {e}")
            return f"❌ Processing error: {e}"

    def run_stream(self, duration_seconds: int = 60, max_decisions: int = 10) -> str:
        """Run continuous pipeline on simulated price stream."""
        if not self.pipeline:
            return "❌ Pipeline not initialized. Run 'init' first."

        try:
            logger.info(f"🎯 Starting continuous pipeline for {duration_seconds} seconds...")

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

            # Run pipeline
            decisions = []
            for decision in self.pipeline.run_continuous_pipeline(price_stream(), max_decisions):
                decisions.append(decision)
                print(
                    f"🎯 Decision {len(decisions)}: {decision.strategy_vector.strategy_id} "
                    f"(confidence: {decision.confidence_score:.3f})"
                )

            return f"✅ Pipeline completed: {len(decisions)} decisions made in {duration_seconds}s"

        except Exception as e:
            logger.error(f"❌ Error in continuous pipeline: {e}")
            return f"❌ Pipeline error: {e}"

    def explain(self) -> str:
        """Explain the last trading decision."""
        if not self.pipeline:
            return "❌ Pipeline not initialized. Run 'init' first."

        try:
            return self.pipeline.explain_last_decision()

        except Exception as e:
            logger.error(f"❌ Error explaining decision: {e}")
            return f"❌ Explanation error: {e}"

    def metrics(self) -> str:
        """Show pipeline performance metrics."""
        if not self.pipeline:
            return "❌ Pipeline not initialized. Run 'init' first."

        try:
            metrics = self.pipeline.get_pipeline_metrics()

            result = "📊 PIPELINE METRICS\n"
            result += "=" * 40 + "\n"

            # Pipeline stats
            stats = metrics['pipeline_stats']
            result += f"Total ticks processed  : {stats['total_ticks_processed']}\n"
            result += f"Total digests generated: {stats['total_digests_generated']}\n"
            result += f"Total decisions made   : {stats['total_decisions_made']}\n"
            result += f"Average processing time: {stats['avg_processing_time']:.4f}s\n"
            result += f"Success rate           : {stats['success_rate']:.2%}\n"
            result += f"Total PnL              : ${stats['total_pnl']:,.2f}\n"

            # Component stats
            result += "\n🔧 COMPONENT STATS:\n"
            digest_stats = metrics['component_stats']['digest_mapper']
            result += f"  Digest Mapper:\n"
            result += f"    Ticks processed: {digest_stats.get('total_ticks_processed', 0)}\n"
            result += f"    Frames generated: {digest_stats.get('total_frames_generated', 0)}\n"
            result += f"    Digests created: {digest_stats.get('total_digests_created', 0)}\n"

            registry_stats = metrics['component_stats']['vector_registry']
            result += f"  Vector Registry:\n"
            result += f"    Total vectors: {registry_stats.get('total_vectors', 0)}\n"
            result += f"    Total searches: {registry_stats.get('total_searches', 0)}\n"
            result += f"    Match rate: {registry_stats.get('match_rate', 0):.2%}\n"

            profit_stats = metrics['component_stats']['profit_calculator']
            result += f"  Profit Calculator:\n"
            result += f"    Total calculations: {profit_stats.get('total_calculations', 0)}\n"
            result += f"    Avg calculation time: {profit_stats.get('avg_calculation_time', 0):.6f}s\n"
            result += f"    Error count: {profit_stats.get('error_count', 0)}\n"

            # Recent decisions
            if metrics['decision_history']['recent_decisions']:
                result += "\n🎯 RECENT DECISIONS:\n"
                for i, decision in enumerate(metrics['decision_history']['recent_decisions'][-5:], 1):
                    result += f"  {i}. {decision['strategy']} "
                    result += f"(confidence: {decision['confidence']:.3f}, "
                    result += f"size: {decision['position_size']:.2%})\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error getting metrics: {e}")
            return f"❌ Metrics error: {e}"

    def decisions(self, count: int = 10) -> str:
        """Show recent trading decisions."""
        if not self.pipeline:
            return "❌ Pipeline not initialized. Run 'init' first."

        try:
            if not self.pipeline.decision_history:
                return "❌ No decisions have been made yet."

            result = f"🎯 RECENT TRADING DECISIONS (Last {count})\n"
            result += "=" * 50 + "\n"

            recent_decisions = self.pipeline.decision_history[-count:]

            for i, decision in enumerate(recent_decisions, 1):
                result += f"{i}. {decision.strategy_vector.strategy_id}\n"
                result += f"   Time: {time.strftime('%H:%M:%S', time.localtime(decision.timestamp))}\n"
                result += f"   Confidence: {decision.confidence_score:.3f}\n"
                result += f"   Position Size: {decision.position_size:.2%}\n"
                result += f"   Entry: ${decision.entry_price:,.2f}\n"
                result += f"   Stop Loss: ${decision.stop_loss:,.2f}\n"
                result += f"   Take Profit: ${decision.take_profit:,.2f}\n"
                result += f"   Reason: {decision.decision_reason}\n"
                result += "-" * 30 + "\n"

            return result

        except Exception as e:
            logger.error(f"❌ Error getting decisions: {e}")
            return f"❌ Decisions error: {e}"

    def stop(self) -> str:
        """Stop the continuous pipeline."""
        if not self.pipeline:
            return "❌ Pipeline not initialized."

        try:
            self.pipeline.stop_pipeline()
            return "✅ Pipeline stopped"

        except Exception as e:
            logger.error(f"❌ Error stopping pipeline: {e}")
            return f"❌ Stop error: {e}"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automated Trading Pipeline CLI - Unified Decision Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                           # Initialize pipeline
  %(prog)s process-tick 50000             # Process single price tick
  %(prog)s run-stream 30                  # Run 30-second continuous pipeline
  %(prog)s explain                        # Explain last decision
  %(prog)s metrics                        # Show performance metrics
  %(prog)s decisions                      # Show recent decisions
  %(prog)s stop                           # Stop continuous pipeline
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize trading pipeline')
    init_parser.add_argument('--risk-tolerance', type=float, default=0.02, help='Risk tolerance (default: 0.02)')
    init_parser.add_argument('--profit-target', type=float, default=0.05, help='Profit target (default: 0.05)')
    init_parser.add_argument('--position-size', type=float, default=0.1, help='Position size (default: 0.1)')
    init_parser.add_argument(
        '--mode',
        choices=['gpu', 'cpu', 'hybrid', 'safe'],
        default='hybrid',
        help='Processing mode (default: hybrid)',
    )
    init_parser.add_argument(
        '--registry-path', default='data/trading_vector_registry.json', help='Vector registry path'
    )

    # Process tick command
    tick_parser = subparsers.add_parser('process-tick', help='Process single price tick')
    tick_parser.add_argument('price', type=float, help='Price value')
    tick_parser.add_argument('--volume', type=float, default=0.0, help='Volume')
    tick_parser.add_argument('--bid', type=float, default=0.0, help='Bid price')
    tick_parser.add_argument('--ask', type=float, default=0.0, help='Ask price')

    # Run stream command
    stream_parser = subparsers.add_parser('run-stream', help='Run continuous pipeline')
    stream_parser.add_argument('duration', type=int, help='Duration in seconds')
    stream_parser.add_argument('--max-decisions', type=int, default=10, help='Maximum decisions to make (default: 10)')

    # Explain command
    subparsers.add_parser('explain', help='Explain last trading decision')

    # Metrics command
    subparsers.add_parser('metrics', help='Show pipeline metrics')

    # Decisions command
    decisions_parser = subparsers.add_parser('decisions', help='Show recent decisions')
    decisions_parser.add_argument('--count', type=int, default=10, help='Number of decisions to show (default: 10)')

    # Stop command
    subparsers.add_parser('stop', help='Stop continuous pipeline')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create CLI instance
    cli = AutomatedTradingPipelineCLI()

    try:
        if args.command == 'init':
            success = cli.init(
                args.risk_tolerance,
                args.profit_target,
                args.position_size,
                args.mode,
                args.registry_path,
            )
            sys.exit(0 if success else 1)

        elif args.command == 'process-tick':
            result = cli.process_tick(args.price, args.volume, args.bid, args.ask)
            print(result)

        elif args.command == 'run-stream':
            result = cli.run_stream(args.duration, args.max_decisions)
            print(result)

        elif args.command == 'explain':
            print(cli.explain())

        elif args.command == 'metrics':
            print(cli.metrics())

        elif args.command == 'decisions':
            print(cli.decisions(args.count))

        elif args.command == 'stop':
            print(cli.stop())

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        cli.stop()
    except Exception as e:
        logger.error(f"❌ CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
