# run_backtest.py

from schwabot.core.adaptive_profit_chain import APCFSystem  # adjust import if needed

# Config (you may want to parse from args or yaml)
start_date = "2024-01-01"
end_date = "2024-03-01"
dataset_path = "data/market_data.parquet"
strategy_profile = "ferris_full"

# 1. Initialize system
apcf = APCFSystem()

# 2. Load data
data = apcf.load_backtest_data(dataset_path)

# 3. Initialize (if required)
apcf.initialize_system(data['price'].values, data['volume'].values)

# 4. Run backtest
results = apcf.run_backtest(data, start_date, end_date, strategy_profile)

# 5. Print summary
if 'final_metrics' in results:
    print("Backtest complete.\nFinal Metrics:")
    for basket, metrics in results["final_metrics"].items():
        print(f"\nBasket: {basket}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
else:
    print("No metrics found. Debug output:", results)
