from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Line Render Engine - Trading Chart and Technical Indicator Visualization for Schwabot
===================================================================================

This module implements the line rendering engine for Schwabot, providing
visualization capabilities for trading charts, technical indicators, and
market data. It supports multiple chart types, indicator overlays, and
real-time rendering with customizable styling.

Core Functionality:
- Chart rendering and visualization
- Technical indicator plotting
- Real-time data updates
- Customizable styling and themes
- Export capabilities
- Interactive chart elements
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)

class ChartType(Enum):
    LINE = "line"
    CANDLESTICK = "candlestick"
    BAR = "bar"
    AREA = "area"
    SCATTER = "scatter"

class IndicatorType(Enum):
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    STOCHASTIC = "stochastic"

@dataclass
class DataPoint:
    timestamp: datetime
    value: float
    volume: Optional[float] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChartStyle:
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    background_color: str = "#ffffff"
    grid_color: str = "#e0e0e0"
    text_color: str = "#000000"
    line_width: float = 2.0
    opacity: float = 1.0
    show_grid: bool = True
    show_legend: bool = True

@dataclass
class IndicatorConfig:
    indicator_type: IndicatorType
    period: int
    color: str
    line_width: float = 1.0
    opacity: float = 0.8
    parameters: Dict[str, Any] = field(default_factory=dict)

class LineRenderEngine:
    def __init__(self):
        self.charts: Dict[str, Dict[str, Any]] = {}
        self.indicators: Dict[str, List[IndicatorConfig]] = {}
        self.data_series: Dict[str, List[DataPoint]] = {}
        self.styles: Dict[str, ChartStyle] = {}
        self.render_callbacks: Dict[str, List[callable]] = {}
        logger.info("LineRenderEngine initialized")

    def create_chart(self, chart_id: str, chart_type: ChartType, 
                    title: str = "", style: Optional[ChartStyle] = None) -> None:
        """Create a new chart with specified type and styling."""
        if style is None:
            style = ChartStyle()
        
        self.charts[chart_id] = {
            "type": chart_type,
            "title": title,
            "data_series": [],
            "indicators": [],
            "created_at": datetime.now()
        }
        self.styles[chart_id] = style
        self.indicators[chart_id] = []
        self.render_callbacks[chart_id] = []
        logger.info(f"Chart created: {chart_id} ({chart_type.value})")

    def add_data_series(self, chart_id: str, series_name: str, 
                       data: List[DataPoint]) -> None:
        """Add a data series to a chart."""
        if chart_id not in self.charts:
            logger.error(f"Chart not found: {chart_id}")
            return
        
        self.data_series[series_name] = data
        self.charts[chart_id]["data_series"].append(series_name)
        logger.debug(f"Data series added: {series_name} to {chart_id}")

    def add_indicator(self, chart_id: str, indicator_config: IndicatorConfig) -> None:
        """Add a technical indicator to a chart."""
        if chart_id not in self.charts:
            logger.error(f"Chart not found: {chart_id}")
            return
        
        self.indicators[chart_id].append(indicator_config)
        logger.debug(f"Indicator added: {indicator_config.indicator_type.value} to {chart_id}")

    def calculate_indicator(self, data: List[DataPoint], 
                          indicator_config: IndicatorConfig) -> List[DataPoint]:
        """Calculate technical indicator values."""
        if not data:
            return []
        
        if indicator_config.indicator_type == IndicatorType.SMA:
            return self._calculate_sma(data, indicator_config.period)
        elif indicator_config.indicator_type == IndicatorType.EMA:
            return self._calculate_ema(data, indicator_config.period)
        elif indicator_config.indicator_type == IndicatorType.RSI:
            return self._calculate_rsi(data, indicator_config.period)
        elif indicator_config.indicator_type == IndicatorType.MACD:
            return self._calculate_macd(data, indicator_config.parameters)
        elif indicator_config.indicator_type == IndicatorType.BOLLINGER_BANDS:
            return self._calculate_bollinger_bands(data, indicator_config.period)
        else:
            logger.warning(f"Unsupported indicator type: {indicator_config.indicator_type}")
            return []

    def _calculate_sma(self, data: List[DataPoint], period: int) -> List[DataPoint]:
        """Calculate Simple Moving Average."""
        if len(data) < period:
            return []
        
        sma_data = []
        for i in range(period - 1, len(data)):
            values = [data[j].value for j in range(i - period + 1, i + 1)]
            sma_value = sum(values) / period
            sma_data.append(DataPoint(
                timestamp=data[i].timestamp,
                value=sma_value
            ))
        return sma_data

    def _calculate_ema(self, data: List[DataPoint], period: int) -> List[DataPoint]:
        """Calculate Exponential Moving Average."""
        if not data:
            return []
        
        ema_data = []
        multiplier = 2.0 / (period + 1)
        
        # First EMA is SMA
        first_values = [data[i].value for i in range(unified_math.min(period, len(data)))]
        ema = sum(first_values) / len(first_values)
        ema_data.append(DataPoint(timestamp=data[0].timestamp, value=ema))
        
        # Calculate subsequent EMAs
        for i in range(1, len(data)):
            ema = (data[i].value * multiplier) + (ema * (1 - multiplier))
            ema_data.append(DataPoint(timestamp=data[i].timestamp, value=ema))
        
        return ema_data

    def _calculate_rsi(self, data: List[DataPoint], period: int) -> List[DataPoint]:
        """Calculate Relative Strength Index."""
        if len(data) < period + 1:
            return []
        
        rsi_data = []
        gains = []
        losses = []
        
        # Calculate initial gains and losses
        for i in range(1, len(data)):
            change = data[i].value - data[i-1].value
            gains.append(unified_math.max(change, 0))
            losses.append(max(-change, 0))
        
        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Calculate RSI for first period
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_data.append(DataPoint(timestamp=data[period].timestamp, value=rsi))
        
        # Calculate subsequent RSIs
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_data.append(DataPoint(timestamp=data[i+1].timestamp, value=rsi))
        
        return rsi_data

    def _calculate_macd(self, data: List[DataPoint], parameters: Dict[str, Any]) -> List[DataPoint]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        fast_period = parameters.get("fast_period", 12)
        slow_period = parameters.get("slow_period", 26)
        signal_period = parameters.get("signal_period", 9)
        
        if len(data) < slow_period:
            return []
        
        # Calculate fast and slow EMAs
        fast_ema = self._calculate_ema(data, fast_period)
        slow_ema = self._calculate_ema(data, slow_period)
        
        # Calculate MACD line
        macd_data = []
        min_length = unified_math.min(len(fast_ema), len(slow_ema))
        
        for i in range(min_length):
            macd_value = fast_ema[i].value - slow_ema[i].value
            macd_data.append(DataPoint(
                timestamp=fast_ema[i].timestamp,
                value=macd_value
            ))
        
        return macd_data

    def _calculate_bollinger_bands(self, data: List[DataPoint], period: int) -> List[DataPoint]:
        """Calculate Bollinger Bands."""
        if len(data) < period:
            return []
        
        bands_data = []
        for i in range(period - 1, len(data)):
            values = [data[j].value for j in range(i - period + 1, i + 1)]
            sma = sum(values) / period
            variance = sum((x - sma) ** 2 for x in values) / period
            std_dev = unified_math.unified_math.sqrt(variance)
            
            upper_band = sma + (2 * std_dev)
            lower_band = sma - (2 * std_dev)
            
            bands_data.append(DataPoint(
                timestamp=data[i].timestamp,
                value=sma,  # Middle band (SMA)
                metadata={"upper": upper_band, "lower": lower_band}
            ))
        
        return bands_data

    def render_chart(self, chart_id: str) -> Dict[str, Any]:
        """Render a chart with all its data series and indicators."""
        if chart_id not in self.charts:
            logger.error(f"Chart not found: {chart_id}")
            return {}
        
        chart_info = self.charts[chart_id]
        style = self.styles[chart_id]
        
        # Collect all data points
        all_data = {}
        for series_name in chart_info["data_series"]:
            if series_name in self.data_series:
                all_data[series_name] = self.data_series[series_name]
        
        # Calculate indicators
        indicator_data = {}
        for indicator_config in self.indicators[chart_id]:
            for series_name in chart_info["data_series"]:
                if series_name in self.data_series:
                    indicator_values = self.calculate_indicator(
                        self.data_series[series_name], 
                        indicator_config
                    )
                    indicator_data[f"{indicator_config.indicator_type.value}_{series_name}"] = indicator_values
        
        # Prepare render data
        render_data = {
            "chart_id": chart_id,
            "chart_type": chart_info["type"].value,
            "title": chart_info["title"],
            "style": {
                "primary_color": style.primary_color,
                "secondary_color": style.secondary_color,
                "background_color": style.background_color,
                "grid_color": style.grid_color,
                "text_color": style.text_color,
                "line_width": style.line_width,
                "opacity": style.opacity,
                "show_grid": style.show_grid,
                "show_legend": style.show_legend
            },
            "data_series": all_data,
            "indicators": indicator_data,
            "timestamp": datetime.now()
        }
        
        # Trigger render callbacks
        for callback in self.render_callbacks[chart_id]:
            try:
                callback(render_data)
            except Exception as e:
                logger.error(f"Error in render callback: {e}")
        
        logger.debug(f"Chart rendered: {chart_id}")
        return render_data

    def add_render_callback(self, chart_id: str, callback: callable) -> None:
        """Add a callback function to be called when chart is rendered."""
        if chart_id in self.render_callbacks:
            self.render_callbacks[chart_id].append(callback)
            logger.debug(f"Render callback added for chart: {chart_id}")

    def update_data_point(self, series_name: str, timestamp: datetime, 
                         value: float, **kwargs) -> None:
        """Update a single data point in a series."""
        if series_name not in self.data_series:
            self.data_series[series_name] = []
        
        # Find existing point or add new one
        for point in self.data_series[series_name]:
            if point.timestamp == timestamp:
                point.value = value
                for key, val in kwargs.items():
                    setattr(point, key, val)
                break
        else:
            # Add new point
            new_point = DataPoint(timestamp=timestamp, value=value, **kwargs)
            self.data_series[series_name].append(new_point)
        
        logger.debug(f"Data point updated: {series_name} at {timestamp}")

    def get_chart_info(self, chart_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific chart."""
        if chart_id not in self.charts:
            return None
        
        chart_info = self.charts[chart_id].copy()
        chart_info["style"] = self.styles[chart_id]
        chart_info["indicator_count"] = len(self.indicators[chart_id])
        chart_info["data_series_count"] = len(chart_info["data_series"])
        return chart_info

    def list_charts(self) -> List[str]:
        """List all available charts."""
        return list(self.charts.keys())

def main() -> None:
    """Main function for testing and demonstration."""
    engine = LineRenderEngine()
    
    # Create a chart
    engine.create_chart("price_chart", ChartType.LINE, "BTC Price Chart")
    
    # Generate sample data
    import random
    from datetime import timedelta
    
    base_time = datetime.now()
    sample_data = []
    base_price = 50000.0
    
    for i in range(100):
        timestamp = base_time + timedelta(hours=i)
        price_change = random.uniform(-1000, 1000)
        base_price += price_change
        sample_data.append(DataPoint(
            timestamp=timestamp,
            value=base_price,
            volume=random.uniform(1000, 5000)
        ))
    
    # Add data series
    engine.add_data_series("price_chart", "BTC", sample_data)
    
    # Add indicators
    sma_config = IndicatorConfig(
        indicator_type=IndicatorType.SMA,
        period=20,
        color="#ff7f0e"
    )
    engine.add_indicator("price_chart", sma_config)
    
    # Render chart
    render_result = engine.render_chart("price_chart")
    safe_print(f"Chart rendered with {len(render_result.get('data_series', {}))} data series")
    safe_print(f"Chart info: {engine.get_chart_info('price_chart')}")

if __name__ == "__main__":
    main() 