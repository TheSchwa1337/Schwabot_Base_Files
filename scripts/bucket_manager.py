from pathlib import Path

import yaml

# bucket_manager.py


class BucketManager:
    def __init__(self, config_path="bucket_config.yaml"):
        self.config = yaml.safe_load(Path(config_path).read_text())["bucket_handling"]
        self.buckets = {"BTC": 0, "USDC": 0}

    def lock_incremental_profit(self, profit_value, volatility, entropy_score):
        vol_cfg = self.config["volatility_thresholds"]
        weights = ()
            vol_cfg["high_volatility"]
            if volatility >= 0.5
            else vol_cfg["low_volatility"]
        )

        entropy_factor = ()
            (1 - entropy_score) if self.config["entropy_weighting_enabled"] else 1
        )
        lock_amount = ()
            profit_value * self.config["incremental_locking_pct"] * entropy_factor
        )

        btc_amount = lock_amount * weights["btc_weight"]
        usdc_amount = lock_amount * weights["usdc_weight"]

        self.buckets["BTC"] += btc_amount
        self.buckets["USDC"] += usdc_amount

        return {"BTC_locked": btc_amount, "USDC_locked": usdc_amount}

    def check_rebalance(self, total_assets):
        btc_pct = (self.buckets["BTC"] / total_assets) * 100
        if btc_pct > self.config["bucket_max_allocation_pct"]:
            self.rebalance_buckets(total_assets)

    def rebalance_buckets(self, total_assets):
        target_btc = total_assets * (self.config["bucket_max_allocation_pct"] / 100)
        excess_btc = self.buckets["BTC"] - target_btc
        self.buckets["BTC"] -= excess_btc
        self.buckets["USDC"] += excess_btc
        return {"BTC_to_USDC": excess_btc}
