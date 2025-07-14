"""Module for Schwabot trading system."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from core.entropy_driven_risk_management import EntropyDrivenRiskManager

#!/usr/bin/env python3
"""CLI wrapper for the 🧬 Entropy-Driven Risk Management system"

Offers a thin command-line facade around
`core.entropy_driven_risk_management.EntropyDrivenRiskManager` so it can be
used via the unified Schwabot CLI.

    Commands:
    init        – initialise and (optionally) start active management
    status      – dump current system status JSON
    process     – run a single entropy-management cycle with market data
    stop        – stop the internal loops (placeholder)
    """


        def _load_market_data(path: str) -> Dict[str, Any]:
        p = Path(path)
            if not p.is_file():
            print("⚠️  Market-data file not found: {0}".format(path), file=sys.stderr)
        return {}
            try:
        return json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
            print("❌ Failed to read market-data file: {0}".format(exc), file=sys.stderr)
        return {}


        # ---------------------------------------------------------------------------
        # Command handlers
        # ---------------------------------------------------------------------------


            def cmd_init(args: argparse.Namespace) -> None:
            mgr = EntropyDrivenRiskManager()
                if args.active:
                mgr.start_entropy_management()
                print("✅ Entropy manager initialised and started.")
                    else:
                    print("✅ Entropy manager initialised (inactive).")


                        def cmd_status(args: argparse.Namespace) -> None:
                        mgr = EntropyDrivenRiskManager()
                        print(json.dumps(mgr.get_system_status(), indent=2, default=str))


                            def cmd_process(args: argparse.Namespace) -> None:
                            mgr = EntropyDrivenRiskManager()
                            market: Dict[str, Any] = {}
                                if args.market_data:
                                market = _load_market_data(args.market_data)
                                result = mgr.process_entropy_driven_management(market)
                                print(json.dumps(result, indent=2, default=str))


                                    def cmd_stop(args: argparse.Namespace) -> None:
                                    print("🚦 No persistent entropy manager daemon found – stop acknowledged.")


                                    # ---------------------------------------------------------------------------
                                    # Main entry
                                    # ---------------------------------------------------------------------------


                                        def main() -> None:
                                        parser = argparse.ArgumentParser(
                                        prog="entropy",
                                        description="Entropy-Driven Risk Management CLI wrapper",
                                        )
                                        sub = parser.add_subparsers(dest="command", required=True)

                                        init_p = sub.add_parser("init", help="Initialise the entropy manager")
                                        init_p.add_argument("--active", action="store_true", help="Start immediately")

                                        sub.add_parser("status", help="Show status snapshot")

                                        proc_p = sub.add_parser("process", help="Run one processing cycle")
                                        proc_p.add_argument("--market-data", type=str, help="JSON file with market data")

                                        sub.add_parser("stop", help="Stop the entropy manager (placeholder)")

                                        args = parser.parse_args()

                                        handlers = {
                                        "init": cmd_init,
                                        "status": cmd_status,
                                        "process": cmd_process,
                                        "stop": cmd_stop,
                                        }
                                        handlers[args.command](args)


                                            if __name__ == "__main__":
                                            main()
