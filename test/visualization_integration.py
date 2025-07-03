"""
Visualization Integration Test
Tests the integration between trading engine and visualization components.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

from core.unified_trade_router import UnifiedTradeRouter
from core.trading_engine_integration import (
    TradeSignal, 
    TradeExecution, 
    generate_trade_signal
)
from core.clean_unified_math import clean_unified_math

logger = logging.getLogger(__name__)


class TradingVisualizationData:
    """
    Data container for trading visualization.
    Formats trading data for various visualization platforms.
    """
    
    def __init__(self):
        self.router = UnifiedTradeRouter()
        self.data_buffer = []
        self.max_buffer_size = 1000
        
    def format_signal_for_chart(self, signal: TradeSignal) -> Dict[str, Any]:
        """Format trade signal for chart visualization."""
        
        return {
            "timestamp": signal.timestamp.isoformat(),
            "price": signal.price,
            "volume": signal.volume,
            "signal_strength": signal.signal_strength,
            "mathematical_score": signal.mathematical_score,
            "risk_score": signal.risk_score,
            "confidence": signal.confidence,
            "entropy": signal.entropy,
            "volatility": signal.volatility,
            "order_side": signal.order_side.value,
            "order_type": signal.order_type.value,
            "asset": signal.asset,
            "signal_id": signal.id,
            "metadata": signal.metadata
        }
    
    def format_execution_for_chart(self, execution: TradeExecution) -> Dict[str, Any]:
        """Format trade execution for chart visualization."""
        
        return {
            "timestamp": execution.timestamp.isoformat(),
            "execution_price": execution.execution_price,
            "volume": execution.volume,
            "latency": execution.latency,
            "realized_profit": execution.realized_profit,
            "performance_score": execution.performance_score,
            "order_side": execution.order_side.value,
            "order_type": execution.order_type.value,
            "asset": execution.asset,
            "execution_id": execution.id,
            "signal_id": execution.signal_id
        }
    
    def get_price_chart_data(self, signals: List[TradeSignal]) -> Dict[str, List]:
        """Generate price chart data from signals."""
        
        timestamps = []
        prices = []
        volumes = []
        signal_strengths = []
        mathematical_scores = []
        
        for signal in signals:
            timestamps.append(signal.timestamp.isoformat())
            prices.append(signal.price)
            volumes.append(signal.volume)
            signal_strengths.append(signal.signal_strength)
            mathematical_scores.append(signal.mathematical_score)
        
        return {
            "timestamps": timestamps,
            "prices": prices,
            "volumes": volumes,
            "signal_strengths": signal_strengths,
            "mathematical_scores": mathematical_scores
        }
    
    def get_performance_chart_data(self, executions: List[TradeExecution]) -> Dict[str, List]:
        """Generate performance chart data from executions."""
        
        timestamps = []
        profits = []
        performance_scores = []
        latencies = []
        
        for execution in executions:
            timestamps.append(execution.timestamp.isoformat())
            profits.append(execution.realized_profit or 0)
            performance_scores.append(execution.performance_score or 0)
            latencies.append(execution.latency)
        
        return {
            "timestamps": timestamps,
            "profits": profits,
            "performance_scores": performance_scores,
            "latencies": latencies
        }
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Generate dashboard metrics for real-time display."""
        
        metrics = self.router.get_performance_metrics()
        
        # Calculate additional metrics
        recent_signals = self.router.signal_history[-10:] if self.router.signal_history else []
        recent_executions = self.router.execution_log[-10:] if self.router.execution_log else []
        
        avg_signal_strength = 0
        avg_mathematical_score = 0
        avg_performance_score = 0
        
        if recent_signals:
            avg_signal_strength = sum(s.signal_strength for s in recent_signals) / len(recent_signals)
            avg_mathematical_score = sum(s.mathematical_score for s in recent_signals) / len(recent_signals)
        
        if recent_executions:
            valid_executions = [e for e in recent_executions if e.performance_score is not None]
            if valid_executions:
                avg_performance_score = sum(e.performance_score for e in valid_executions) / len(valid_executions)
        
        return {
            **metrics,
            "recent_avg_signal_strength": round(avg_signal_strength, 4),
            "recent_avg_mathematical_score": round(avg_mathematical_score, 4),
            "recent_avg_performance_score": round(avg_performance_score, 4),
            "last_update": datetime.utcnow().isoformat()
        }


class RealTimeDataStream:
    """
    Real-time data streaming for live visualization.
    Provides continuous updates for live trading dashboards.
    """
    
    def __init__(self, router: UnifiedTradeRouter):
        self.router = router
        self.subscribers = []
        self.is_running = False
        
    async def start_stream(self, update_interval: float = 1.0):
        """Start real-time data streaming."""
        
        self.is_running = True
        logger.info("🔄 Starting real-time data stream...")
        
        while self.is_running:
            try:
                # Generate current market data
                current_time = time.time()
                current_price = 50000 + (current_time % 3600) * 10  # Simulate price movement
                current_volume = 1.0 + (current_time % 3600) * 0.01
                
                # Generate signal and execution
                signal = self.router.route_trade_signal(
                    price=current_price,
                    volume=current_volume,
                    metadata={"stream": True, "timestamp": current_time}
                )
                
                execution = self.router.route_trade_execution(signal)
                
                # Prepare data for subscribers
                stream_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "signal": signal.to_dict(),
                    "execution": execution.to_dict(),
                    "metrics": self.router.get_performance_metrics()
                }
                
                # Notify subscribers
                await self.notify_subscribers(stream_data)
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(update_interval)
    
    def stop_stream(self):
        """Stop real-time data streaming."""
        self.is_running = False
        logger.info("⏹️ Stopped real-time data stream")
    
    async def notify_subscribers(self, data: Dict[str, Any]):
        """Notify all subscribers with new data."""
        
        for subscriber in self.subscribers:
            try:
                await subscriber(data)
            except Exception as e:
                logger.error(f"Subscriber notification error: {e}")
    
    def subscribe(self, callback):
        """Subscribe to real-time data updates."""
        self.subscribers.append(callback)
        logger.info(f"📡 New subscriber added. Total subscribers: {len(self.subscribers)}")
    
    def unsubscribe(self, callback):
        """Unsubscribe from real-time data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"📡 Subscriber removed. Total subscribers: {len(self.subscribers)}")


class ChartDataExporter:
    """
    Export chart data in various formats for different visualization platforms.
    """
    
    @staticmethod
    def export_to_json(data: Dict[str, Any], filename: str):
        """Export data to JSON file."""
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"📁 Chart data exported to {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to export chart data: {e}")
    
    @staticmethod
    def export_to_csv(data: Dict[str, List], filename: str):
        """Export data to CSV file."""
        
        try:
            import csv
            
            # Find the longest list to determine number of rows
            max_length = max(len(v) for v in data.values() if isinstance(v, list))
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(data.keys())
                
                # Write data rows
                for i in range(max_length):
                    row = []
                    for key in data.keys():
                        if isinstance(data[key], list) and i < len(data[key]):
                            row.append(data[key][i])
                        else:
                            row.append('')
                    writer.writerow(row)
            
            logger.info(f"📁 Chart data exported to {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to export chart data: {e}")
    
    @staticmethod
    def format_for_plotly(data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for Plotly charts."""
        
        return {
            "data": [
                {
                    "x": data.get("timestamps", []),
                    "y": data.get("prices", []),
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Price"
                },
                {
                    "x": data.get("timestamps", []),
                    "y": data.get("signal_strengths", []),
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Signal Strength",
                    "yaxis": "y2"
                }
            ],
            "layout": {
                "title": "Trading Signals and Performance",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Price"},
                "yaxis2": {"title": "Signal Strength", "overlaying": "y", "side": "right"}
            }
        }


def create_visualization_demo():
    """Create a demonstration of the visualization integration."""
    
    logger.info("🎨 Creating Visualization Demo...")
    
    # Initialize components
    router = UnifiedTradeRouter()
    viz_data = TradingVisualizationData()
    
    # Generate sample data
    sample_signals = []
    sample_executions = []
    
    for i in range(20):
        try:
            signal = router.route_trade_signal(
                price=50000 + (i * 100),
                volume=1.0 + (i * 0.05),
                metadata={"demo": True, "iteration": i}
            )
            sample_signals.append(signal)
            
            execution = router.route_trade_execution(signal)
            sample_executions.append(execution)
            
        except Exception as e:
            logger.error(f"Demo data generation error: {e}")
    
    # Generate visualization data
    price_data = viz_data.get_price_chart_data(sample_signals)
    performance_data = viz_data.get_performance_chart_data(sample_executions)
    dashboard_metrics = viz_data.get_dashboard_metrics()
    
    # Export data
    ChartDataExporter.export_to_json(price_data, "demo_price_data.json")
    ChartDataExporter.export_to_json(performance_data, "demo_performance_data.json")
    ChartDataExporter.export_to_json(dashboard_metrics, "demo_dashboard_metrics.json")
    
    # Export CSV for spreadsheet analysis
    ChartDataExporter.export_to_csv(price_data, "demo_price_data.csv")
    
    # Format for Plotly
    plotly_data = ChartDataExporter.format_for_plotly(price_data)
    ChartDataExporter.export_to_json(plotly_data, "demo_plotly_chart.json")
    
    logger.info("✅ Visualization demo created successfully!")
    
    return {
        "price_data": price_data,
        "performance_data": performance_data,
        "dashboard_metrics": dashboard_metrics,
        "plotly_data": plotly_data
    }


async def run_live_visualization_demo(duration_seconds: int = 30):
    """Run a live visualization demo with real-time data."""
    
    logger.info(f"🔄 Starting Live Visualization Demo for {duration_seconds} seconds...")
    
    router = UnifiedTradeRouter()
    stream = RealTimeDataStream(router)
    
    # Collect data during demo
    collected_data = []
    
    async def data_collector(data):
        collected_data.append(data)
        logger.info(f"📊 Received data update: {len(collected_data)} total")
    
    # Subscribe to data stream
    stream.subscribe(data_collector)
    
    # Start streaming
    stream_task = asyncio.create_task(stream.start_stream(update_interval=0.5))
    
    # Let it run for specified duration
    await asyncio.sleep(duration_seconds)
    
    # Stop streaming
    stream.stop_stream()
    await stream_task
    
    # Export collected data
    ChartDataExporter.export_to_json(collected_data, "live_demo_data.json")
    
    logger.info(f"✅ Live visualization demo completed! Collected {len(collected_data)} data points")
    
    return collected_data


if __name__ == "__main__":
    # Run visualization demo
    demo_data = create_visualization_demo()
    
    # Run live demo (uncomment to test)
    # asyncio.run(run_live_visualization_demo(10))
    
    print("\n🎨 Visualization Integration Demo Complete!")
    print("📁 Generated files:")
    print("  - demo_price_data.json")
    print("  - demo_performance_data.json") 
    print("  - demo_dashboard_metrics.json")
    print("  - demo_price_data.csv")
    print("  - demo_plotly_chart.json") 