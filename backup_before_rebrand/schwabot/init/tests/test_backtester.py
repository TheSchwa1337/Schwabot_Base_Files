"""
test_backtester.py
-------------------
Backtesting simulation harness that loads historical CSV candle data,
simulates entry/exit based on gates and risk rules, and logs PnL
for strategy evaluation.
"""

import pandas as pd

from core.risk_manager import RiskManager
from core.strategy_registry import StrategyRegistry
from core.vector_band_gatekeeper import confirm_long_trend, confirm_mid_vector, confirm_short_drift


def run_backtest(csv_file: str, strategy_id: str) -> None:
    df = pd.read_csv(csv_file)
    registry = StrategyRegistry()
    risk = RiskManager()
    entry_price = None
    trade_open = False

    for _idx, row in df.iterrows():
        # Build tick_blob from CSV row
        tick_blob = f"{row['symbol']},price={row['close']},time={row['timestamp']}"

        # Check primary trend gates
        short_ok = confirm_short_drift(tick_blob)
        mid_ok = confirm_mid_vector(tick_blob)
        long_ok = confirm_long_trend(tick_blob)
        gate_passed = short_ok and mid_ok and long_ok

        if gate_passed and not trade_open:
            entry_price = float(row['close'])
            trade_open = True
            risk.register_trade('sim_trade', entry_price, row['timestamp'])
            continue

        if trade_open:
            decision = risk.update_price('sim_trade', float(row['close']))

            if decision in ['STOP', 'LOCK', 'TTL_EXIT']:
                exit_price = float(row['close'])
                pnl = (exit_price - entry_price) / entry_price
                registry.register_result(strategy_id, pnl, str(row['timestamp']))
                print(f"Trade closed @ {exit_price} | PnL: {pnl:.4f} | Reason: {decision}")
                trade_open = False
                risk.cancel_trade('sim_trade')

    # Summary of top strategies
    top = registry.get_top_strategies()
    print("Top strategies:")
    for strat, info in top:
        print(f"  {strat}: score={info['score']}, trades={len(info['results'])}")


if __name__ == '__main__':
    # Example usage (ensure CSV has 'symbol','close','timestamp','volume')
    import sys

    if len(sys.argv) != 3:
        print("Usage: python test_backtester.py <csv_file> <strategy_id>")
    else:
        run_backtest(sys.argv[1], sys.argv[2])
