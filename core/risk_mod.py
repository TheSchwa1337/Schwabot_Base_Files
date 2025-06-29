import numpy as np


def rho_base(interval: float, bits: float, kT: float = 0.3, kD: float = 0.5) -> float:
""""""
Calculates the base rho (risk scalar) based on block interval and difficulty.
    
Args:
    interval (float): Time in seconds since the previous block.
    bits (float): The 'bits' field from the block (represents difficulty).
        kT (float): Coefficient for time interval impact.
        kD (float): Coefficient for difficulty shock impact.
        
    Returns:
    float: The base rho scalar.
    """"""
    # Normalize interval around 600 seconds (expected block time)
    dt_pen = np.tanh((interval - 600) / 100) # Centered at 600s, scaled by 100s
    
    # Simplistic difficulty shock: log(bits) can represent a relative measure of difficulty.
    # A more advanced model might compare current bits to a historical average or 2016-block retarget.
    diff_shock = np.log(bits + 1e-9) # Add epsilon to prevent log(0)

        # Rho increases with longer intervals (more volatility/stress) and certain difficulty shifts
    return 1 + kT * dt_pen + kD * np.tanh(diff_shock / 0.5) # Divided by 0.5 for scaling of tanh input

def pool_mod(hashrate_now: float, hashrate_ma: float, block_gap: float, orphan_rate: float) -> float:
""""""
Calculates a risk modifier based on mining pool metrics.
    
Args:
    hashrate_now (float): Current observed hashrate from the pool.
        hashrate_ma (float): Moving average of hashrate (for normalization).
    block_gap (float): Time since the last block notification from the pool.
    orphan_rate (float): Current orphan block rate percentage (0-1).
        
    Returns:
        float: A modifier for the rho scalar based on pool health.
    """"""
        # Inflow boost: positive if current hashrate is above moving average
    inflow_boost = np.tanh((hashrate_now - hashrate_ma) / (hashrate_ma + 1e-9))
    
        # Gap penalty: penalize if block gap is significantly longer than expected (e.g., 900s = 15 mins beyond avg)
    gap_pen = np.tanh(max(0, block_gap - 900) / 600)
    
    # Orphan penalty: penalize based on high orphan rates
    # Orphan rate is typically 0-1 (percentage as decimal), so scaling factor might be needed
        orphan_pen = orphan_rate * 2.0 # Scale to have more impact if needed

    # Combine modifiers: inflow boosts, gaps and orphans penalize
return 1 + 0.1 * inflow_boost - 0.2 * gap_pen - 0.2 * orphan_pen

def volatility_mod(current_vol: float, reference_vol: float, gamma: float = 64.0) -> float:
""""""
    Calculates a volatility-based modifier for risk.
Args:
    current_vol (float): Realized volatility over a short-term window (e.g., 1h).
    reference_vol (float): Realized volatility over a longer-term window (e.g., 1M).
        gamma (float): Scaling factor for the bias.
    Returns:
        float: A bias term (0-255) for risk nibble adjustment.
    """"""
    # Ensure reference_vol is not zero to prevent division errors
        if reference_vol == 0: return 0.0:

        # Positive bias if current vol is higher than reference, negative if lower
    bias = gamma * (current_vol - reference_vol) / reference_vol
return np.clip(bias, -gamma, gamma) # Clamp to a reasonable range

def performance_mod(sharpe_score: float, km: float = 0.4) -> float:
""""""
Calculates a performance-based momentum modifier.
Args:
    sharpe_score (float): Sharpe ratio of a strategy class.
        km (float): Coefficient for Sharpe score impact.
    Returns:
        float: A momentum term (multiplier for rho).
    """"""
return 1 + km * np.tanh(sharpe_score)

def onchain_macro_mod(miner_outflow_zscore: float, ko: float = 0.2) -> float:
""""""
    Calculates an on-chain macro-economic modifier for leverage/risk.
Args:
    miner_outflow_zscore (float): Z-score of miner outflow (e.g., from Glassnode).
        ko (float): Coefficient for miner outflow impact.
    Returns:
        float: A macro scalar (multiplier for rho).
    """"""
        # Lower rho if miners are dumping (high positive Z-score)
return 1 - ko * np.tanh(miner_outflow_zscore) 