# Multi-Profile Coinbase API System

## Overview

The Multi-Profile Coinbase API System is a sophisticated trading infrastructure that manages multiple Coinbase API profiles with independent strategy logic, de-synced trade execution, and mathematical separation to ensure unique profit opportunities for each profile.

## Mathematical Core

The system implements the following mathematical principles:

```
∀ t: H₁(t) ≠ H₂(t) ∨ A₁ ≠ A₂
Strategy_Profile(t, Pᵢ) = ƒ(Hashₜᵢ, Assetsᵢ, Holdingsᵢ, Profit_Zonesᵢ)
```

Where:
- `H₁(t)`, `H₂(t)` = Active strategy hash streams per profile over time
- `A₁`, `A₂` = Asset sets for each profile
- `Strategy_Profile(t, Pᵢ)` = Strategy function for profile i at time t
- `Hashₜᵢ` = Profile-specific hash trajectory
- `Assetsᵢ` = Profile-specific asset selection
- `Holdingsᵢ` = Current holdings for profile i
- `Profit_Zonesᵢ` = Profile-specific profit targets

## Key Features

### ✅ **Multi-Profile Management**
- **Isolated API Keys/Secrets per Profile**: Each profile has its own API credentials
- **Independent Strategy Logic**: Each profile operates with unique strategy paths
- **De-Synced Trade Execution**: Trades are executed with random delays to prevent synchronization
- **Cross-Profile Arbitration**: Detects and executes arbitrage opportunities between profiles

### ✅ **Mathematical Separation**
- **Hash Uniqueness Enforcement**: Ensures each profile has unique hash trajectories
- **Asset Uniqueness**: Prevents excessive asset overlap between profiles
- **Strategy Path Encoding**: Each strategy is mathematically encoded, not copied
- **Entropy-Guided Asset Selection**: 6th asset is selected based on entropy and volume scores

### ✅ **Advanced Strategy System**
- **5 Base Assets + 1 Random Asset**: Each profile trades 6 assets total
- **Hash-Based Strategy Selection**: Strategy selection is deterministic based on profile hash
- **Confidence and Signal Strength**: Each strategy has profile-specific confidence levels
- **Profit Zone Generation**: Unique profit targets for each profile

### ✅ **Performance Monitoring**
- **Real-Time Metrics**: Track performance across all profiles
- **Hash Collision Detection**: Monitor and resolve hash collisions
- **Strategy Duplication Prevention**: Ensure strategies remain unique
- **Cross-Profile Correlation Analysis**: Monitor profile independence

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Profile System                     │
├─────────────────────────────────────────────────────────────┤
│  Profile Router                                             │
│  ├── Multi-Profile Manager                                  │
│  ├── Strategy Mapper                                        │
│  └── Integration Layer                                      │
├─────────────────────────────────────────────────────────────┤
│  Profile A (Primary)        Profile B (Secondary)           │
│  ├── API Credentials A      ├── API Credentials B           │
│  ├── Strategy Logic A       ├── Strategy Logic B            │
│  ├── Asset Selection A      ├── Asset Selection B           │
│  └── Trade Execution A      └── Trade Execution B           │
├─────────────────────────────────────────────────────────────┤
│  Cross-Profile Arbitration & Synchronization                │
│  ├── Arbitrage Detection                                    │
│  ├── Hash Echo Validation                                   │
│  └── De-Sync Mechanisms                                     │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Profile Configuration (`config/coinbase_profiles.yaml`)

```yaml
profiles:
  profile_a:
    name: "Primary Trading Profile"
    enabled: true
    priority: 1
    
    api_credentials:
      api_key: "YOUR_COINBASE_API_KEY_PROFILE_A"
      secret: "YOUR_COINBASE_SECRET_PROFILE_A"
      passphrase: "YOUR_COINBASE_PASSPHRASE_PROFILE_A"
      sandbox: true  # Set to false for live trading
    
    strategy_config:
      base_assets: ["BTC", "ETH", "SOL", "USDC", "XRP"]
      random_asset_pool: ["ADA", "DOT", "LINK", "MATIC", "AVAX", "UNI", "ATOM", "LTC"]
      strategy_weights:
        volume_weighted_hash_oscillator: 0.30
        zygot_zalgo_entropy_dual_key_gate: 0.25
        multi_phase_strategy_weight_tensor: 0.20
        quantum_strategy_calculator: 0.15
        entropy_enhanced_trading_executor: 0.10
      
      confidence_threshold: 0.75
      signal_strength_threshold: 0.65
      max_position_size_pct: 8.0
      risk_adjustment_factor: 1.2
      
      entropy_threshold: 0.003
      hash_trajectory_weight: 0.85
      drift_delta_range: [0.001, 0.005]
    
    trading_params:
      trading_pairs: ["BTC/USDC", "ETH/USDC", "SOL/USDC", "XRP/USDC"]
      max_open_positions: 4
      order_timeout: 45
      retry_attempts: 3
      slippage_tolerance: 0.0015
    
    risk_management:
      stop_loss_pct: 2.5
      take_profit_pct: 6.0
      max_daily_loss_pct: 4.0
      max_drawdown_pct: 12.0
      volatility_threshold: 0.75
    
    profile_hash:
      base_hash: "profile_a_primary_trading"
      entropy_seed: 0.42
      hash_rotation_interval: 3600  # 1 hour
      hash_complexity_factor: 1.1
```

## Usage

### 1. Setup Configuration

1. Copy the configuration template:
   ```bash
   cp config/coinbase_profiles.yaml config/coinbase_profiles_local.yaml
   ```

2. Edit the configuration with your actual API credentials:
   ```yaml
   api_credentials:
     api_key: "your_actual_api_key"
     secret: "your_actual_secret"
     passphrase: "your_actual_passphrase"
     sandbox: true  # Set to false for live trading
   ```

### 2. Run the Test Script

```bash
python test_multi_profile_coinbase.py
```

This will run a comprehensive test of the multi-profile system including:
- System initialization
- Profile management
- Strategy generation
- Trade execution (simulated)
- Cross-profile arbitrage
- Mathematical separation
- Performance monitoring
- Integration testing

### 3. Use in Production

```python
import asyncio
from core.profile_router import ProfileRouter

async def main():
    # Initialize the profile router
    router = ProfileRouter("config/coinbase_profiles_local.yaml")
    
    # Initialize the system
    await router.initialize()
    
    # Start trading
    await router.start_trading()
    
    # Monitor status
    status = router.get_profile_status()
    print(f"Active profiles: {status['profile_router']['active_profiles']}")
    
    # Execute unified trade
    trade_result = await router.execute_unified_trade({
        'symbol': 'BTC/USDC',
        'side': 'buy',
        'type': 'market',
        'size': 0.001
    })
    
    print(f"Trade result: {trade_result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Strategy Types

The system supports 5 main strategy types:

1. **Volume Weighted Hash Oscillator**: Volume-based strategy with hash oscillation
2. **Zygot Zalgo Entropy Dual Key Gate**: Advanced entropy-based strategy
3. **Multi-Phase Strategy Weight Tensor**: Multi-phase tensor-based strategy
4. **Quantum Strategy Calculator**: Quantum-inspired strategy calculation
5. **Entropy Enhanced Trading Executor**: Entropy-enhanced execution strategy

Each strategy is selected based on the profile's hash and configured weights.

## Mathematical Separation Logic

### Hash Uniqueness
- Each profile generates unique hashes based on profile characteristics
- Hash rotation occurs at configurable intervals
- Hash collisions are detected and resolved automatically

### Asset Uniqueness
- Base assets are selected using hash-based logic
- Random asset is chosen from a profile-specific pool
- Asset overlap between profiles is minimized

### Strategy Uniqueness
- Strategy selection is deterministic based on profile hash
- Strategy parameters are adjusted based on hash characteristics
- Strategy duplication is detected and prevented

## Cross-Profile Arbitration

The system automatically detects arbitrage opportunities between profiles:

1. **Hash Similarity Analysis**: Compares hash trajectories between profiles
2. **Asset Overlap Detection**: Identifies asset overlap patterns
3. **Arbitrage Score Calculation**: Computes arbitrage opportunity scores
4. **Opportunity Execution**: Executes arbitrage when scores exceed thresholds

## Performance Monitoring

### Metrics Tracked
- Total trades per profile
- Win rates and profit/loss
- Hash collision counts
- Strategy duplication counts
- Cross-profile correlation
- Arbitrage opportunities

### Alerts
- Hash collision alerts
- Strategy duplication alerts
- Performance degradation alerts
- Arbitrage opportunity notifications

## Security Features

### API Key Management
- Encrypted storage of API credentials
- Profile isolation for security
- Separate memory spaces per profile
- Isolated error handling

### Audit Logging
- Profile action logging
- Cross-profile communication logging
- Trade execution logging
- Security event logging

## Integration with Existing Schwabot System

The multi-profile system integrates seamlessly with the existing Schwabot infrastructure:

- **Live Trading System**: Integrates with existing live trading components
- **Portfolio Tracker**: Shares portfolio data across profiles
- **Risk Manager**: Applies risk management across all profiles
- **Unified Interface**: Provides single interface for multi-profile operations

## Testing

### Running Tests

```bash
# Run comprehensive test
python test_multi_profile_coinbase.py

# Check test results
cat multi_profile_test_results.json
```

### Test Coverage

The test suite covers:
- ✅ System initialization
- ✅ Profile management
- ✅ Strategy generation
- ✅ Trade execution
- ✅ Cross-profile arbitrage
- ✅ Mathematical separation
- ✅ Performance monitoring
- ✅ Integration testing

## Troubleshooting

### Common Issues

1. **API Connection Failures**
   - Verify API credentials are correct
   - Check network connectivity
   - Ensure sandbox mode is appropriate

2. **Hash Collisions**
   - System automatically resolves collisions
   - Check hash rotation intervals
   - Monitor collision frequency

3. **Strategy Duplications**
   - Review strategy weights
   - Check hash generation logic
   - Monitor uniqueness scores

4. **Performance Issues**
   - Check profile configuration
   - Monitor system resources
   - Review trading parameters

### Log Files

- `multi_profile_test.log`: Test execution logs
- `multi_profile_test_results.json`: Detailed test results
- System logs: Standard Schwabot logging

## Advanced Configuration

### Custom Strategy Weights

```yaml
strategy_weights:
  volume_weighted_hash_oscillator: 0.40  # Increase weight
  zygot_zalgo_entropy_dual_key_gate: 0.30
  multi_phase_strategy_weight_tensor: 0.20
  quantum_strategy_calculator: 0.05
  entropy_enhanced_trading_executor: 0.05
```

### Hash Configuration

```yaml
profile_hash:
  base_hash: "custom_profile_hash"
  entropy_seed: 0.75  # Custom entropy seed
  hash_rotation_interval: 1800  # 30 minutes
  hash_complexity_factor: 1.5  # Higher complexity
```

### Risk Management

```yaml
risk_management:
  stop_loss_pct: 1.5  # Tighter stop loss
  take_profit_pct: 8.0  # Higher profit target
  max_daily_loss_pct: 3.0  # Lower daily loss limit
  max_drawdown_pct: 10.0  # Lower drawdown limit
  volatility_threshold: 0.85  # Higher volatility threshold
```

## Contributing

When contributing to the multi-profile system:

1. Maintain mathematical separation principles
2. Ensure hash uniqueness across profiles
3. Test strategy independence
4. Validate cross-profile arbitration logic
5. Update configuration documentation

## License

This multi-profile Coinbase API system is part of the Schwabot trading system and follows the same licensing terms.

## Support

For support and questions:
1. Check the troubleshooting section
2. Review test results and logs
3. Verify configuration settings
4. Consult the mathematical documentation

---

**Note**: This system is designed for advanced trading scenarios. Ensure you understand the risks and test thoroughly before using in production. 