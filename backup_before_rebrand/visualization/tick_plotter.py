#!/usr/bin/env python3
"""
Tick Plotter - Real-time Price Feed Visualization
=================================================

Real-time price feed visualization with entry/exit point overlays.
Provides live charting capabilities for trading analysis and monitoring.

Features:
- Real-time price feed plotting
- Entry/exit point visualization
- Technical indicator overlays
- Interactive charts
- Historical data playback
- Multi-timeframe support
"""

import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, Slider

logger = logging.getLogger(__name__)

class TickPlotter:
    """Real-time tick plotting and visualization system."""
    
    def __init__(self, max_ticks: int = 1000, update_interval: float = 1.0):
        self.max_ticks = max_ticks
        self.update_interval = update_interval
        
        # Data storage
        self.price_data = deque(maxlen=max_ticks)
        self.volume_data = deque(maxlen=max_ticks)
        self.timestamps = deque(maxlen=max_ticks)
        
        # Trading signals
        self.entry_points = []
        self.exit_points = []
        self.stop_losses = []
        self.take_profits = []
        
        # Chart objects
        self.fig = None
        self.ax_price = None
        self.ax_volume = None
        self.ani = None
        
        # Technical indicators
        self.ema_short = deque(maxlen=max_ticks)
        self.ema_long = deque(maxlen=max_ticks)
        self.rsi_values = deque(maxlen=max_ticks)
        self.macd_values = deque(maxlen=max_ticks)
        
        # Live data flag
        self.is_live = False
        self.live_thread = None
        
    def add_tick(self, price: float, volume: float = 0.0, timestamp: float = None):
        """Add a new tick to the plotter."""
        if timestamp is None:
            timestamp = time.time()
        
        self.price_data.append(price)
        self.volume_data.append(volume)
        self.timestamps.append(timestamp)
        
        # Update technical indicators
        self._update_indicators()
    
    def add_entry_signal(self, price: float, timestamp: float, signal_type: str = "BUY"):
        """Add entry signal to the plot."""
        self.entry_points.append({
            'price': price,
            'timestamp': timestamp,
            'type': signal_type
        })
    
    def add_exit_signal(self, price: float, timestamp: float, signal_type: str = "SELL"):
        """Add exit signal to the plot."""
        self.exit_points.append({
            'price': price,
            'timestamp': timestamp,
            'type': signal_type
        })
    
    def add_stop_loss(self, price: float, timestamp: float):
        """Add stop loss level."""
        self.stop_losses.append({
            'price': price,
            'timestamp': timestamp
        })
    
    def add_take_profit(self, price: float, timestamp: float):
        """Add take profit level."""
        self.take_profits.append({
            'price': price,
            'timestamp': timestamp
        })
    
    def _update_indicators(self):
        """Update technical indicators."""
        if len(self.price_data) < 2:
            return
        
        prices = list(self.price_data)
        
        # EMA calculation
        if len(prices) >= 12:
            ema_12 = self._calculate_ema(prices, 12)
            self.ema_short.append(ema_12)
        
        if len(prices) >= 26:
            ema_26 = self._calculate_ema(prices, 26)
            self.ema_long.append(ema_26)
        
        # RSI calculation
        if len(prices) >= 14:
            rsi = self._calculate_rsi(prices, 14)
            self.rsi_values.append(rsi)
        
        # MACD calculation
        if len(self.ema_short) >= 9 and len(self.ema_long) >= 9:
            macd = self.ema_short[-1] - self.ema_long[-1]
            self.macd_values.append(macd)
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1]
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def create_chart(self, figsize: Tuple[int, int] = (15, 10)):
        """Create the main chart."""
        self.fig, (self.ax_price, self.ax_volume) = plt.subplots(2, 1, figsize=figsize, 
                                                                  gridspec_kw={'height_ratios': [3, 1]})
        
        # Set up price chart
        self.ax_price.set_title('Real-time Price Feed with Entry/Exit Signals', fontsize=14, fontweight='bold')
        self.ax_price.set_ylabel('Price ($)')
        self.ax_price.grid(True, alpha=0.3)
        
        # Set up volume chart
        self.ax_volume.set_xlabel('Time')
        self.ax_volume.set_ylabel('Volume')
        self.ax_volume.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def update_chart(self, frame):
        """Update the chart with new data."""
        if not self.price_data:
            return
        
        # Clear previous plots
        self.ax_price.clear()
        self.ax_volume.clear()
        
        # Convert timestamps to datetime
        times = [datetime.fromtimestamp(ts) for ts in self.timestamps]
        prices = list(self.price_data)
        volumes = list(self.volume_data)
        
        # Plot price data
        self.ax_price.plot(times, prices, 'b-', linewidth=1, label='Price', alpha=0.8)
        
        # Plot technical indicators
        if len(self.ema_short) > 0:
            ema_times = times[-len(self.ema_short):]
            self.ax_price.plot(ema_times, list(self.ema_short), 'r-', linewidth=1, label='EMA(12)', alpha=0.7)
        
        if len(self.ema_long) > 0:
            ema_times = times[-len(self.ema_long):]
            self.ax_price.plot(ema_times, list(self.ema_long), 'g-', linewidth=1, label='EMA(26)', alpha=0.7)
        
        # Plot entry signals
        for entry in self.entry_points:
            entry_time = datetime.fromtimestamp(entry['timestamp'])
            color = 'green' if entry['type'] == 'BUY' else 'red'
            marker = '^' if entry['type'] == 'BUY' else 'v'
            self.ax_price.scatter(entry_time, entry['price'], color=color, marker=marker, 
                                s=100, zorder=5, label=f"{entry['type']} Entry")
        
        # Plot exit signals
        for exit_point in self.exit_points:
            exit_time = datetime.fromtimestamp(exit_point['timestamp'])
            color = 'red' if exit_point['type'] == 'SELL' else 'green'
            marker = 'v' if exit_point['type'] == 'SELL' else '^'
            self.ax_price.scatter(exit_time, exit_point['price'], color=color, marker=marker,
                                s=100, zorder=5, label=f"{exit_point['type']} Exit")
        
        # Plot stop losses
        for sl in self.stop_losses:
            sl_time = datetime.fromtimestamp(sl['timestamp'])
            self.ax_price.axhline(y=sl['price'], color='red', linestyle='--', alpha=0.5, label='Stop Loss')
        
        # Plot take profits
        for tp in self.take_profits:
            tp_time = datetime.fromtimestamp(tp['timestamp'])
            self.ax_price.axhline(y=tp['price'], color='green', linestyle='--', alpha=0.5, label='Take Profit')
        
        # Plot volume
        self.ax_volume.bar(times, volumes, alpha=0.6, color='gray')
        
        # Set labels and grid
        self.ax_price.set_title('Real-time Price Feed with Entry/Exit Signals', fontsize=14, fontweight='bold')
        self.ax_price.set_ylabel('Price ($)')
        self.ax_price.grid(True, alpha=0.3)
        self.ax_price.legend(loc='upper left')
        
        self.ax_volume.set_xlabel('Time')
        self.ax_volume.set_ylabel('Volume')
        self.ax_volume.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(self.ax_price.get_xticklabels(), rotation=45)
        plt.setp(self.ax_volume.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
    
    def start_live_plot(self):
        """Start live plotting with animation."""
        if self.fig is None:
            self.create_chart()
        
        self.is_live = True
        self.ani = animation.FuncAnimation(self.fig, self.update_chart, 
                                         interval=self.update_interval * 1000, 
                                         blit=False)
        plt.show()
    
    def stop_live_plot(self):
        """Stop live plotting."""
        self.is_live = False
        if self.ani:
            self.ani.event_source.stop()
    
    def create_candlestick_chart(self, ohlcv_data: pd.DataFrame, save_path: str = None):
        """Create candlestick chart with technical indicators."""
        # Prepare data for mplfinance
        ohlcv_data.index = pd.to_datetime(ohlcv_data.index)
        
        # Add technical indicators
        add_plots = []
        
        # EMA lines
        if len(ohlcv_data) >= 26:
            ema_12 = ohlcv_data['Close'].ewm(span=12).mean()
            ema_26 = ohlcv_data['Close'].ewm(span=26).mean()
            
            add_plots.append(mpf.make_addplot(ema_12, color='red', width=0.7))
            add_plots.append(mpf.make_addplot(ema_26, color='blue', width=0.7))
        
        # RSI
        if len(ohlcv_data) >= 14:
            rsi = self._calculate_rsi_series(ohlcv_data['Close'], 14)
            add_plots.append(mpf.make_addplot(rsi, panel=1, color='purple', width=0.7))
        
        # MACD
        if len(ohlcv_data) >= 26:
            macd, signal, hist = self._calculate_macd(ohlcv_data['Close'])
            add_plots.append(mpf.make_addplot(macd, panel=2, color='blue', width=0.7))
            add_plots.append(mpf.make_addplot(signal, panel=2, color='red', width=0.7))
            add_plots.append(mpf.make_addplot(hist, panel=2, type='bar', color='gray', alpha=0.5))
        
        # Create the chart
        fig, axes = mpf.plot(ohlcv_data, type='candle', style='charles',
                            title='Candlestick Chart with Technical Indicators',
                            ylabel='Price ($)',
                            volume=True,
                            addplot=add_plots,
                            returnfig=True,
                            figsize=(15, 10))
        
        # Add RSI panel labels
        if len(ohlcv_data) >= 14:
            axes[1].set_ylabel('RSI')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
        
        # Add MACD panel labels
        if len(ohlcv_data) >= 26:
            axes[2].set_ylabel('MACD')
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Candlestick chart saved to {save_path}")
        
        return fig
    
    def _calculate_rsi_series(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for a pandas series."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD for a pandas series."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        return macd, signal, hist
    
    def export_chart_data(self, output_path: str = "chart_data.json"):
        """Export chart data for external analysis."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'price_data': list(self.price_data),
            'volume_data': list(self.volume_data),
            'timestamps': list(self.timestamps),
            'entry_points': self.entry_points,
            'exit_points': self.exit_points,
            'stop_losses': self.stop_losses,
            'take_profits': self.take_profits,
            'technical_indicators': {
                'ema_short': list(self.ema_short),
                'ema_long': list(self.ema_long),
                'rsi': list(self.rsi_values),
                'macd': list(self.macd_values)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Chart data exported to {output_path}")
        return data

class LiveDataSimulator:
    """Simulate live data for testing the tick plotter."""
    
    def __init__(self, base_price: float = 50000.0, volatility: float = 0.02):
        self.base_price = base_price
        self.volatility = volatility
        self.current_price = base_price
        self.tick_count = 0
    
    def generate_tick(self) -> Tuple[float, float]:
        """Generate a simulated tick."""
        # Random price movement
        change = np.random.normal(0, self.volatility * self.current_price * 0.01)
        self.current_price += change
        
        # Ensure positive price
        self.current_price = max(self.current_price, 1000.0)
        
        # Generate volume
        volume = np.random.uniform(0.1, 10.0)
        
        self.tick_count += 1
        return self.current_price, volume
    
    def generate_entry_signal(self, probability: float = 0.01) -> bool:
        """Generate entry signal with given probability."""
        return np.random.random() < probability
    
    def generate_exit_signal(self, probability: float = 0.01) -> bool:
        """Generate exit signal with given probability."""
        return np.random.random() < probability

def main():
    """Main function for testing and demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tick Plotter Visualization")
    parser.add_argument("--live", action="store_true", help="Start live plotting")
    parser.add_argument("--simulate", action="store_true", help="Simulate live data")
    parser.add_argument("--duration", type=int, default=60, help="Simulation duration in seconds")
    parser.add_argument("--save-path", help="Save chart to file")
    
    args = parser.parse_args()
    
    # Initialize tick plotter
    plotter = TickPlotter(max_ticks=500, update_interval=0.5)
    
    if args.live or args.simulate:
        # Create chart
        plotter.create_chart()
        
        # Initialize data simulator
        simulator = LiveDataSimulator(base_price=50000.0, volatility=0.02)
        
        # Start live plotting
        plotter.start_live_plot()
        
        # Simulate data
        if args.simulate:
            start_time = time.time()
            
            while time.time() - start_time < args.duration:
                # Generate tick
                price, volume = simulator.generate_tick()
                plotter.add_tick(price, volume)
                
                # Generate signals
                if simulator.generate_entry_signal(0.005):
                    plotter.add_entry_signal(price, time.time(), "BUY")
                
                if simulator.generate_exit_signal(0.005):
                    plotter.add_exit_signal(price, time.time(), "SELL")
                
                time.sleep(0.5)
            
            # Stop plotting
            plotter.stop_live_plot()
            
            # Save data
            if args.save_path:
                plotter.export_chart_data(args.save_path.replace('.png', '_data.json'))
        
        plt.show()
    
    else:
        # Show interactive menu
        print("📈 Tick Plotter Visualization")
        print("=" * 40)
        print("1. Start Live Plotting")
        print("2. Simulate Live Data")
        print("3. Load Historical Data")
        print("4. Export Chart Data")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == "1":
            plotter.create_chart()
            plotter.start_live_plot()
            plt.show()
        elif choice == "2":
            duration = int(input("Enter simulation duration (seconds): "))
            plotter.create_chart()
            plotter.start_live_plot()
            
            simulator = LiveDataSimulator()
            start_time = time.time()
            
            while time.time() - start_time < duration:
                price, volume = simulator.generate_tick()
                plotter.add_tick(price, volume)
                
                if simulator.generate_entry_signal(0.005):
                    plotter.add_entry_signal(price, time.time(), "BUY")
                
                if simulator.generate_exit_signal(0.005):
                    plotter.add_exit_signal(price, time.time(), "SELL")
                
                time.sleep(0.5)
            
            plotter.stop_live_plot()
            plt.show()
        elif choice == "3":
            print("Historical data loading not implemented yet.")
        elif choice == "4":
            plotter.export_chart_data()

if __name__ == "__main__":
    main() 