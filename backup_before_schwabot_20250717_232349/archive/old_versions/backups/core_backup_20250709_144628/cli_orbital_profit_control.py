"""Module for Schwabot trading system."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from core.orbital_profit_control_system import OrbitalProfitControlSystem

#!/usr/bin/env python3
"""CLI wrapper for the 🌌 Orbital Profit Control System"

This small command-line utility exposes the most important entry-points of
`core.orbital_profit_control_system.OrbitalProfitControlSystem` in an easy-to-use
argparse driven interface so it can be launched from `schwabot_unified_cli.py`.

    Supported sub-commands:
    init        – initialize and start the orbital system
    status      – print a JSON dump of current system status
    optimize    – run a single optimise cycle (optionally supplying market, data)
    stop        – gracefully stop the system (if, running)

        Example:
        python schwabot_unified_cli.py orbit init
        python schwabot_unified_cli.py orbit status
        python schwabot_unified_cli.py orbit optimize --market-data market.json

        Note: this wrapper is intentionally stateless – each invocation spins up a fresh
        instance when necessary.  For long-running control loops use the *monitor* tool
        or integrate the class programmatically.
        """

        # Project imports – lazy to avoid heavy cost when only requesting --help
        # ---------------------------------------------------------------------------
        # Helper utilities
        # ---------------------------------------------------------------------------


            def _load_market_data(path: str) -> Dict[str, Any]:
            """Load market-data JSON file or return an empty dict."""
            p = Path(path)
                if not p.is_file():
                print("⚠️  Market-data file not found: {0}".format(path), file=sys.stderr)
            return {}
                try:
                    with p.open("r", encoding="utf-8") as fp:
                return json.load(fp)
                    except Exception as exc:
                    print("❌ Failed to read market-data file: {0}".format(exc), file=sys.stderr)
                return {}


                # ---------------------------------------------------------------------------
                # Command implementations
                # ---------------------------------------------------------------------------


                    def cmd_init(args: argparse.Namespace) -> None:
                    """Initialise and start the orbital controller."""
                    system = OrbitalProfitControlSystem()
                    system.start_orbital_control()
                    print("✅ Orbital Profit Control System initialised and started.")


                        def cmd_status(args: argparse.Namespace) -> None:
                        """Print current system status as JSON."""
                        system = OrbitalProfitControlSystem()
                        status = system.get_system_status()
                        print(json.dumps(status, indent=2, default=str))


                            def cmd_optimize(args: argparse.Namespace) -> None:
                            """Run a single optimise cycle and show summary JSON."""
                            system = OrbitalProfitControlSystem()
                            system.start_orbital_control()

                            market_data: Dict[str, Any] = {}
                                if args.market_data:
                                market_data = _load_market_data(args.market_data)
                                result = system.optimize_profit_flow(market_data)
                                print(json.dumps(result, indent=2, default=str))


                                    def cmd_stop(args: argparse.Namespace) -> None:
                                    """Stop the system (no persistent daemon yet, but provided for, symmetry)."""
                                    # In stateless mode this is a no-op, but we keep the endpoint for future use.
                                    print("🚦 No running persistent instance detected – stop command acknowledged.")


                                    # ---------------------------------------------------------------------------
                                    # Main CLI dispatcher
                                    # ---------------------------------------------------------------------------


                                        def main() -> None:
                                        parser = argparse.ArgumentParser()
                                        prog = "orbit",
                                        description = "Orbital Profit Control System CLI wrapper",
                                        )
                                        sub = parser.add_subparsers(dest="command", required=True)

                                        sub.add_parser("init", help="Initialise and start the orbital system")
                                        sub.add_parser("status", help="Show system status (single, snapshot)")

                                        opt_parser = sub.add_parser("optimize", help="Run a single optimisation cycle")
                                        opt_parser.add_argument()
                                        "--market-data",
                                        type = str,
                                        help = "Optional path to a JSON file with market data overrides",
                                        )

                                        sub.add_parser("stop", help="Gracefully stop the orbital system (placeholder)")

                                        args = parser.parse_args()

                                        dispatch_table = {}
                                        "init": cmd_init,
                                        "status": cmd_status,
                                        "optimize": cmd_optimize,
                                        "stop": cmd_stop,
                                        }

                                        dispatch_fn = dispatch_table.get(args.command)
                                            if dispatch_fn is None:
                                            parser.error("Unknown command: {0}".format(args.command))
                                            dispatch_fn(args)


                                                if __name__ == "__main__":
                                                main()
