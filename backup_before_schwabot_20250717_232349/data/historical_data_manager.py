import numpy as np
import pandas as pd

from . import (  # !/usr/bin/env python3
    BTC,
    BTC/USDC,
    ROUND_DOWN,
    Any,
    Comprehensive,
    Data,
    Decimal,
    Dict,
    Features:,
    Handles,
    Historical,
    Key,
    Manager,
    Multi-decimal,
    Optional,
    Pipeline.,
    Schwabot's,
    Trading,
    2,
    6,
    8,
    """Historical,
    -,
    analysis,
    and,
    data,
    datetime,
    decimal,
    decimals,
    for,
    from,
    generation.,
    hash,
    hashlib,
    historical,
    import,
    ingestion,
    logging,
    management,
    multi-decimal,
    multiple,
    pattern,
    precision,
    preprocessing,
    price,
    profit,
    sources,
    system,
    system.,
    timedelta,
    typing,
)

- SHA256 hash pattern generation for temporal memory
- Integration with QSC-GTS biological immune system
- Real-time historical context for trading decisions
"""


# Import Schwabot components
    BTC_USDC_DIR,
    BTC_USDC_COMBINED,
    BTC_USDC_HASH_MEMORY,
    BTC_USDC_PRECISION_ANALYSIS,
)

logger = logging.getLogger(__name__)


class HistoricalDataManager:
    """Comprehensive historical data manager for Schwabot."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize historical data manager.

        Args:
            config: Configuration parameters
        """
        self.config = config or self._default_config()

        # Data storage
        self.historical_data: Optional[pd.DataFrame] = None
        self.hash_memory: Optional[pd.DataFrame] = None
        self.precision_analysis: Optional[pd.DataFrame] = None

        # Performance tracking
        self.data_loaded = False
        self.last_update = None
        self.total_records = 0

        # Multi-decimal analysis
        self.decimal_configs = {
            "macro": 2,  # 2 decimals for macro trends
            "standard": 6,  # 6 decimals for standard trading
            "micro": 8,  # 8 decimals for micro-scalping
        }

        logger.info("📊 Historical Data Manager initialized")

    def _default_config():-> Dict[str, Any]:
        """Default configuration for historical data manager."""
        return {
            "data_sources": {"coingecko": True, "binance": True, "ccxt": True},
            "timeframe": "1m",  # 1-minute candles
            "start_date": "2021-01-01",
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "min_data_points": 100000,  # Minimum required data points
            "enable_hash_memory": True,
            "enable_precision_analysis": True,
            "hash_memory_window": 1000,  # Hash memory lookback window
            "precision_analysis_window": 500,  # Precision analysis window
            "cache_enabled": True,
            "auto_refresh": True,
            "refresh_interval": 3600,  # 1 hour
        }

    def load_historical_data():-> bool:
        """Load historical BTC/USDC data from multiple sources.

        Args:
            force_refresh: Force refresh of data even if cached

        Returns:
            True if data loaded successfully
        """
        try:
            # Check if preprocessed data exists
            if not force_refresh and BTC_USDC_COMBINED.exists():
                logger.info("📊 Loading preprocessed historical data...")
                self.historical_data = pd.read_parquet(BTC_USDC_COMBINED)
                self.data_loaded = True
                self.total_records = len(self.historical_data)
                self.last_update = datetime.now()

                logger.info(f"✅ Loaded {self.total_records:,} historical records")
                return True

            # Load from raw historical files
            logger.info("📊 Loading raw historical data files...")
            raw_data = self._load_raw_historical_files()

            if raw_data is None or len(raw_data) < self.config["min_data_points"]:
                logger.error(
                    f"❌ Insufficient historical data: {len(raw_data) if raw_data is not None else 0} records"
                )
                return False

            # Preprocess and save
            self.historical_data = self._preprocess_historical_data(raw_data)
            self._save_preprocessed_data()

            # Generate hash memory and precision analysis
            if self.config["enable_hash_memory"]:
                self._generate_hash_memory()

            if self.config["enable_precision_analysis"]:
                self._generate_precision_analysis()

            self.data_loaded = True
            self.total_records = len(self.historical_data)
            self.last_update = datetime.now()

            logger.info(
                f"✅ Successfully loaded and processed {self.total_records:,} historical records"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load historical data: {e}")
            return False

    def _load_raw_historical_files():-> Optional[pd.DataFrame]:
        """Load raw historical data files from BTC_USDC_DIR."""
        data_files = list(BTC_USDC_DIR.glob("*.csv"))

        if not data_files:
            logger.warning(f"⚠️ No CSV files found in {BTC_USDC_DIR}")
            logger.info(f"📁 Expected location: {BTC_USDC_DIR}")
            logger.info("📥 Please add BTC/USDC CSV files to this directory")
            return None

        logger.info(f"📁 Found {len(data_files)} historical data files")

        # Load and combine all CSV files
        dataframes = []
        for file_path in sorted(data_files):
            try:
                logger.info(f"📊 Loading {file_path.name}...")
                df = pd.read_csv(file_path)

                # Standardize column names
                df = self._standardize_columns(df)

                if not df.empty:
                    dataframes.append(df)
                    logger.info(f"✅ Loaded {len(df):,} records from {file_path.name}")

            except Exception as e:
                logger.error(f"❌ Failed to load {file_path.name}: {e}")
                continue

        if not dataframes:
            logger.error("❌ No valid data files loaded")
            return None

        # Combine all dataframes
        combined_data = pd.concat(dataframes, ignore_index=True)

        # Remove duplicates and sort by timestamp
        combined_data = combined_data.drop_duplicates(subset=["timestamp"])
        combined_data = combined_data.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"📊 Combined {len(combined_data):,} unique historical records")
        return combined_data

    def _standardize_columns():-> pd.DataFrame:
        """Standardize column names and data types."""
        # Expected column mappings
        column_mappings = {
            # Common variations
            "time": "timestamp",
            "date": "timestamp",
            "datetime": "timestamp",
            "price": "close",
            "last": "close",
            "amount": "volume",
            "vol": "volume",
            "volumeto": "volume",
            "volumefrom": "volume",
        }

        # Rename columns
        df = df.rename(columns=column_mappings)

        # Ensure required columns exist
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"⚠️ Missing columns in data: {missing_columns}")
            # Try to infer missing columns
            if "close" in df.columns and "open" not in df.columns:
                df["open"] = df["close"]
            if "close" in df.columns and "high" not in df.columns:
                df["high"] = df["close"]
            if "close" in df.columns and "low" not in df.columns:
                df["low"] = df["close"]
            if "volume" not in df.columns:
                df["volume"] = 1000.0  # Default volume

        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            if df["timestamp"].dtype == "object":
                # Try different timestamp formats
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                except:
                    try:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    except:
                        logger.error("❌ Cannot parse timestamp column")
                        return pd.DataFrame()

        # Convert numeric columns
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with NaN values
        df = df.dropna(subset=["timestamp", "close"])

        return df

    def _preprocess_historical_data():-> pd.DataFrame:
        """Preprocess historical data for Schwabot analysis."""
        logger.info("🔧 Preprocessing historical data...")

        # Ensure proper data types
        processed_data = raw_data.copy()

        # Add derived columns
        processed_data["price_change"] = processed_data["close"].pct_change()
        processed_data["price_change_abs"] = processed_data["price_change"].abs()
        processed_data["volatility"] = (
            processed_data["price_change"].rolling(window=20).std()
        )
        processed_data["volume_ma"] = processed_data["volume"].rolling(window=20).mean()
        processed_data["volume_ratio"] = (
            processed_data["volume"] / processed_data["volume_ma"]
        )

        # Add time-based features
        processed_data["hour"] = processed_data["timestamp"].dt.hour
        processed_data["day_of_week"] = processed_data["timestamp"].dt.dayofweek
        processed_data["month"] = processed_data["timestamp"].dt.month

        # Add technical indicators
        processed_data["sma_20"] = processed_data["close"].rolling(window=20).mean()
        processed_data["sma_50"] = processed_data["close"].rolling(window=50).mean()
        processed_data["rsi"] = self._calculate_rsi(processed_data["close"])

        # Remove NaN values from calculations
        processed_data = processed_data.dropna()

        logger.info(f"✅ Preprocessed {len(processed_data):,} records")
        return processed_data

    def _calculate_rsi():-> pd.Series:
        """Calculate RSI technical indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _generate_hash_memory():-> None:
        """Generate hash memory from historical data for pattern recognition."""
        logger.info("🔐 Generating hash memory from historical data...")

        if self.historical_data is None:
            logger.error("❌ No historical data available for hash memory generation")
            return

        hash_memory_data = []
        window_size = self.config["hash_memory_window"]

        for i in range(window_size, len(self.historical_data)):
            # Get window of data
            window_data = self.historical_data.iloc[i - window_size : i + 1]

            # Generate hash for this window
            hash_data = self._generate_window_hash(window_data)

            hash_memory_data.append(
                {
                    "timestamp": window_data.iloc[-1]["timestamp"],
                    "price": window_data.iloc[-1]["close"],
                    "hash_pattern": hash_data["hash_pattern"],
                    "hash_entropy": hash_data["hash_entropy"],
                    "price_volatility": hash_data["price_volatility"],
                    "volume_pattern": hash_data["volume_pattern"],
                    "trend_direction": hash_data["trend_direction"],
                    "pattern_strength": hash_data["pattern_strength"],
                }
            )

        self.hash_memory = pd.DataFrame(hash_memory_data)

        # Save hash memory
        self.hash_memory.to_parquet(BTC_USDC_HASH_MEMORY)

        logger.info(f"✅ Generated hash memory for {len(self.hash_memory):,} patterns")

    def _generate_window_hash():-> Dict[str, Any]:
        """Generate hash pattern for a window of historical data."""
        # Create hash input string
        price_sequence = window_data["close"].values
        volume_sequence = window_data["volume"].values

        # Multi-decimal price formatting
        latest_price = price_sequence[-1]
        price_2_decimal = self._format_price(latest_price, 2)
        price_6_decimal = self._format_price(latest_price, 6)
        price_8_decimal = self._format_price(latest_price, 8)

        # Create hash input
        hash_input = f"{price_2_decimal}_{price_6_decimal}_{price_8_decimal}_"
        hash_input += f"{price_sequence.mean():.2f}_{volume_sequence.mean():.2f}_"
        hash_input += f"{price_sequence.std():.4f}_{volume_sequence.std():.4f}"

        # Generate hash
        hash_pattern = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # Calculate hash entropy
        hash_bytes = bytes.fromhex(hash_pattern)
        hash_entropy = -sum(
            (b / 255.0) * np.log2((b / 255.0) + 1e-8) for b in hash_bytes
        )
        hash_entropy = min(1.0, hash_entropy / 8.0)

        # Calculate pattern metrics
        price_volatility = price_sequence.std() / price_sequence.mean()
        volume_pattern = volume_sequence.std() / volume_sequence.mean()
        trend_direction = 1 if price_sequence[-1] > price_sequence[0] else -1
        pattern_strength = hash_entropy * (1 + price_volatility)

        return {
            "hash_pattern": hash_pattern,
            "hash_entropy": hash_entropy,
            "price_volatility": price_volatility,
            "volume_pattern": volume_pattern,
            "trend_direction": trend_direction,
            "pattern_strength": pattern_strength,
        }

    def _generate_precision_analysis():-> None:
        """Generate multi-decimal precision analysis from historical data."""
        logger.info("📊 Generating multi-decimal precision analysis...")

        if self.historical_data is None:
            logger.error("❌ No historical data available for precision analysis")
            return

        precision_data = []
        window_size = self.config["precision_analysis_window"]

        for i in range(window_size, len(self.historical_data)):
            # Get window of data
            window_data = self.historical_data.iloc[i - window_size : i + 1]

            # Generate precision analysis
            precision_analysis = self._analyze_precision_levels(window_data)

            precision_data.append(
                {
                    "timestamp": window_data.iloc[-1]["timestamp"],
                    "price": window_data.iloc[-1]["close"],
                    **precision_analysis,
                }
            )

        self.precision_analysis = pd.DataFrame(precision_data)

        # Save precision analysis
        self.precision_analysis.to_parquet(BTC_USDC_PRECISION_ANALYSIS)

        logger.info(
            f"✅ Generated precision analysis for {len(self.precision_analysis):,} records"
        )

    def _analyze_precision_levels():-> Dict[str, Any]:
        """Analyze price data at multiple decimal precision levels."""
        latest_price = window_data["close"].iloc[-1]

        # Multi-decimal formatting
        price_2_decimal = self._format_price(latest_price, 2)
        price_6_decimal = self._format_price(latest_price, 6)
        price_8_decimal = self._format_price(latest_price, 8)

        # Generate hashes for each precision level
        timestamp = window_data["timestamp"].iloc[-1].timestamp()

        hash_2_decimal = self._hash_price(price_2_decimal, timestamp, "macro")
        hash_6_decimal = self._hash_price(price_6_decimal, timestamp, "standard")
        hash_8_decimal = self._hash_price(price_8_decimal, timestamp, "micro")

        # Calculate profit scores for each precision level
        macro_profit_score = self._calculate_profit_score(
            hash_2_decimal, "macro", window_data
        )
        standard_profit_score = self._calculate_profit_score(
            hash_6_decimal, "standard", window_data
        )
        micro_profit_score = self._calculate_profit_score(
            hash_8_decimal, "micro", window_data
        )

        # 16-bit tick mapping
        tick_16bit = self._map_to_16bit(latest_price)

        return {
            "price_2_decimal": price_2_decimal,
            "price_6_decimal": price_6_decimal,
            "price_8_decimal": price_8_decimal,
            "hash_2_decimal": hash_2_decimal,
            "hash_6_decimal": hash_6_decimal,
            "hash_8_decimal": hash_8_decimal,
            "macro_profit_score": macro_profit_score,
            "standard_profit_score": standard_profit_score,
            "micro_profit_score": micro_profit_score,
            "tick_16bit": tick_16bit,
            "price_volatility": window_data["close"].std()
            / window_data["close"].mean(),
            "volume_activity": window_data["volume"].mean(),
            "trend_strength": abs(
                window_data["close"].iloc[-1] - window_data["close"].iloc[0]
            )
            / window_data["close"].iloc[0],
        }

    def _format_price():-> str:
        """Format price with specific decimal precision."""
        quant = Decimal("1." + ("0" * decimals))
        d_price = Decimal(str(price)).quantize(quant, rounding=ROUND_DOWN)
        return f"{d_price:.{decimals}f}"

    def _hash_price():-> str:
        """Generate SHA256 hash for price with timestamp and prefix."""
        data = f"{prefix}_{price_str}_{timestamp:.3f}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _map_to_16bit():-> int:
        """Map BTC price to 16-bit integer (0-65535)."""
        min_price, max_price = 10000.0, 100000.0
        clamped_price = max(min_price, min(max_price, price))
        normalized = (clamped_price - min_price) / (max_price - min_price)
        return int(normalized * 65535)

    def _calculate_profit_score():-> float:
        """Calculate profit score based on hash pattern and historical context."""
        # Calculate hash entropy
        hash_bytes = bytes.fromhex(price_hash)
        entropy = -sum((b / 255.0) * np.log2((b / 255.0) + 1e-8) for b in hash_bytes)
        base_score = min(1.0, entropy / 8.0)

        # Apply precision-specific modifiers
        precision_modifiers = {
            "macro": 0.8,  # Conservative macro scoring
            "standard": 1.0,  # Standard scoring
            "micro": 1.2,  # Boosted micro scoring
        }

        # Apply volatility modifier
        volatility = window_data["close"].std() / window_data["close"].mean()
        volatility_modifier = min(1.5, 1.0 + volatility * 10)

        modified_score = (
            base_score * precision_modifiers[precision_level] * volatility_modifier
        )
        return min(1.0, modified_score)

    def _save_preprocessed_data():-> None:
        """Save preprocessed historical data."""
        if self.historical_data is not None:
            self.historical_data.to_parquet(BTC_USDC_COMBINED)
            logger.info(f"✅ Saved preprocessed data to {BTC_USDC_COMBINED}")

    def get_historical_context():-> Dict[str, Any]:
        """Get historical context for current trading decision.

        Args:
            current_price: Current BTC price
            lookback_days: Number of days to look back

        Returns:
            Historical context data
        """
        if not self.data_loaded or self.historical_data is None:
            return {}

        # Get recent historical data
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_data = self.historical_data[
            self.historical_data["timestamp"] >= cutoff_date
        ].copy()

        if recent_data.empty:
            return {}

        # Calculate historical metrics
        price_stats = {
            "mean": recent_data["close"].mean(),
            "std": recent_data["close"].std(),
            "min": recent_data["close"].min(),
            "max": recent_data["close"].max(),
            "current_percentile": (current_price - recent_data["close"].min())
            / (recent_data["close"].max() - recent_data["close"].min()),
        }

        # Get hash memory context
        hash_context = {}
        if self.hash_memory is not None:
            recent_hashes = self.hash_memory[
                self.hash_memory["timestamp"] >= cutoff_date
            ]
            if not recent_hashes.empty:
                hash_context = {
                    "avg_pattern_strength": recent_hashes["pattern_strength"].mean(),
                    "trend_direction": recent_hashes["trend_direction"].mean(),
                    "avg_hash_entropy": recent_hashes["hash_entropy"].mean(),
                }

        # Get precision analysis context
        precision_context = {}
        if self.precision_analysis is not None:
            recent_precision = self.precision_analysis[
                self.precision_analysis["timestamp"] >= cutoff_date
            ]
            if not recent_precision.empty:
                precision_context = {
                    "avg_macro_score": recent_precision["macro_profit_score"].mean(),
                    "avg_standard_score": recent_precision[
                        "standard_profit_score"
                    ].mean(),
                    "avg_micro_score": recent_precision["micro_profit_score"].mean(),
                    "avg_tick_16bit": recent_precision["tick_16bit"].mean(),
                }

        return {
            "price_statistics": price_stats,
            "hash_memory_context": hash_context,
            "precision_context": precision_context,
            "data_points": len(recent_data),
            "lookback_days": lookback_days,
        }

    def get_system_status():-> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "data_loaded": self.data_loaded,
            "total_records": self.total_records,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "hash_memory_available": self.hash_memory is not None,
            "precision_analysis_available": self.precision_analysis is not None,
            "hash_memory_records": (
                len(self.hash_memory) if self.hash_memory is not None else 0
            ),
            "precision_analysis_records": (
                len(self.precision_analysis)
                if self.precision_analysis is not None
                else 0
            ),
            "data_files": {
                "combined_data": BTC_USDC_COMBINED.exists(),
                "hash_memory": BTC_USDC_HASH_MEMORY.exists(),
                "precision_analysis": BTC_USDC_PRECISION_ANALYSIS.exists(),
            },
            "configuration": self.config,
        }


# Helper function for easy integration
def create_historical_data_manager():-> HistoricalDataManager:
    """Create and initialize historical data manager.

    Args:
        config: Optional configuration parameters

    Returns:
        Initialized historical data manager
    """
    manager = HistoricalDataManager(config)

    # Try to load existing data
    if not manager.load_historical_data():
        logger.warning(
            "⚠️ No historical data loaded. Please add CSV files to data/historical/btc_usdc/"
        )

    return manager
