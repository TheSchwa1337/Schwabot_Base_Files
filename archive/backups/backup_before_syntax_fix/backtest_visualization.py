from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class BacktestVisualizer:
    """Visualizes backtest results with interactive plots."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def create_performance_charts(
        self,
        portfolio_history: List[float],
        trade_history: List[Dict],
        metrics: Dict[str, float],
        save: bool = True,
    ):
        """Create comprehensive performance visualization."""

        # Convert trade history to DataFrame
        trades_df = pd.DataFrame(trade_history)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='s')

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)

        # 1. Portfolio Value Over Time
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_portfolio_value(ax1, portfolio_history)

        # 2. Trade Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_trade_distribution(ax2, trades_df)

        # 3. Drawdown Analysis
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_drawdown(ax3, portfolio_history)

        # 4. Performance Metrics
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_metrics_summary(ax4, metrics)

        # Adjust layout
        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.results_dir / "backtest_results_{0}.png".format(timestamp)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_portfolio_value(self, ax, portfolio_history: List[float]):
        """Plot portfolio value over time."""
        x = range(len(portfolio_history))
        y = portfolio_history

        ax.plot(x, y, linewidth=2)
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Value (USDC)')
        ax.grid(True)

        # Add initial and final values
        ax.text(0, portfolio_history[0], "${0}".format(portfolio_history[0]), verticalalignment='bottom')
        ax.text(
            len(portfolio_history) - 1,
            portfolio_history[-1],
            "${0}".format(portfolio_history[-1]),
            verticalalignment='bottom',
        )

    def _plot_trade_distribution(self, ax, trades_df: pd.DataFrame):
        """Plot trade size and profit distribution."""
        if not trades_df.empty:
            trades_df['profit'] = trades_df.apply(
                lambda x: x['value'] - x['fees'] if x['type'] == 'sell' else 0, axis=1
            )

            # Create profit distribution
            sns.histplot(data=trades_df[trades_df['type'] == 'sell'], x='profit', bins=30, ax=ax)
            ax.set_title('Trade Profit Distribution')
            ax.set_xlabel('Profit/Loss (USDC)')
            ax.set_ylabel('Count')

    def _plot_drawdown(self, ax, portfolio_history: List[float]):
        """Plot drawdown analysis."""
        portfolio_values = np.array(portfolio_history)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max

        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax.plot(range(len(drawdown)), drawdown, color='red', linewidth=1)
        ax.set_title('Drawdown Analysis')
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown %')
        ax.grid(True)

        # Add max drawdown annotation
        max_drawdown_idx = np.argmin(drawdown)
        ax.annotate(
            "Max Drawdown: {0:.2%}".format(drawdown[max_drawdown_idx]),
            xy=(max_drawdown_idx, drawdown[max_drawdown_idx]),
            xytext=(10, 10),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->'),
        )

    def _plot_metrics_summary(self, ax, metrics: Dict[str, float]):
        """Plot performance metrics summary."""
        metrics_to_show = {
            'Total Return': "{0:.2%}".format(metrics['total_return']),
            'Sharpe Ratio': "{0:.2f}".format(metrics['sharpe_ratio']),
            'Max Drawdown': "{0:.2%}".format(metrics['max_drawdown']),
            'Win Rate': "{0:.2%}".format(metrics['win_rate']),
        }

        # Create bar chart
        y_pos = np.arange(len(metrics_to_show))
        values = [float(v.strip('%').strip('x')) for v in metrics_to_show.values()]

        bars = ax.barh(y_pos, values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics_to_show.keys())
        ax.set_title('Performance Metrics Summary')

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                "{0}".format(list(metrics_to_show.values())[i]),
                ha='left',
                va='center',
                fontweight='bold',
            )

    def create_trade_analysis(self, trade_history: List[Dict], save: bool = True):
        """Create detailed trade analysis visualization."""
        if not trade_history:
            return

        trades_df = pd.DataFrame(trade_history)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='s')

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        # 1. Trade Volume Over Time
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_trade_volume(ax1, trades_df)

        # 2. Trade Size Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_trade_sizes(ax2, trades_df)

        # 3. Win/Loss Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_win_loss(ax3, trades_df)

        # 4. Profit Factor Analysis
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_profit_factor(ax4, trades_df)

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.results_dir / "trade_analysis_{0}.png".format(timestamp)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_trade_volume(self, ax, trades_df: pd.DataFrame):
        """Plot trading volume over time."""
        daily_volume = trades_df.resample('D', on='timestamp')['value'].sum()

        ax.plot(daily_volume.index, daily_volume.values, linewidth=2)
        ax.set_title('Daily Trading Volume')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume (USDC)')
        ax.grid(True)
        plt.xticks(rotation=45)

    def _plot_trade_sizes(self, ax, trades_df: pd.DataFrame):
        """Plot distribution of trade sizes."""
        sns.boxplot(data=trades_df, x='type', y='size', ax=ax)
        ax.set_title('Trade Size Distribution')
        ax.set_xlabel('Trade Type')
        ax.set_ylabel('Size')

    def _plot_win_loss(self, ax, trades_df: pd.DataFrame):
        """Plot win/loss analysis."""
        sell_trades = trades_df[trades_df['type'] == 'sell']
        wins = sell_trades[sell_trades['price'] > sell_trades['entry_price']]
        losses = sell_trades[sell_trades['price'] <= sell_trades['entry_price']]

        labels = ['Winning Trades', 'Losing Trades']
        sizes = [len(wins), len(losses)]
        colors = ['lightgreen', 'lightcoral']

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax.set_title('Win/Loss Distribution')

    def _plot_profit_factor(self, ax, trades_df: pd.DataFrame):
        """Plot profit factor analysis."""
        sell_trades = trades_df[trades_df['type'] == 'sell'].copy()
        sell_trades['profit'] = sell_trades['price'] - sell_trades['entry_price']

        winning_trades = sell_trades[sell_trades['profit'] > 0]['profit'].sum()
        losing_trades = abs(sell_trades[sell_trades['profit'] < 0]['profit'].sum())

        if losing_trades != 0:
            profit_factor = winning_trades / losing_trades
        else:
            profit_factor = float('inf')

        ax.bar(
            ['Winning Trades', 'Losing Trades'],
            [winning_trades, losing_trades],
            color=['lightgreen', 'lightcoral'],
        )
        ax.set_title("Profit Factor Analysis (PF: {0})".format(profit_factor))
        ax.set_ylabel('Total Profit/Loss (USDC)')

    def save_trade_log(self, trade_history: List[Dict]):
        """Save detailed trade log to CSV."""
        if not trade_history:
            return

        trades_df = pd.DataFrame(trade_history)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='s')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / "trade_log_{0}.csv".format(timestamp)
        trades_df.to_csv(filepath, index=False)
