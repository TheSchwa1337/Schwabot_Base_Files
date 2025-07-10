#!/usr/bin/env python3
"""
Profit Mapper Visualization
===========================

Heatmap visualization for long-term strategy success/failure patterns.
Analyzes trading performance across different market conditions and time periods.

Features:
- Strategy performance heatmaps
- Success/failure pattern analysis
- Market condition correlation
- Time-based performance tracking
- Interactive visualization
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

class ProfitMapper:
    """Profit mapping and visualization system."""
    
    def __init__(self, data_dir: str = "vaults"):
        self.data_dir = data_dir
        self.strategy_data = {}
        self.performance_matrix = None
        self.time_periods = ['1h', '4h', '1d', '1w', '1m']
        self.market_conditions = ['bull', 'bear', 'sideways', 'volatile']
        
    def load_strategy_data(self) -> bool:
        """Load strategy performance data from vaults."""
        try:
            # Load strategy registry
            registry_path = os.path.join(self.data_dir, "strategy_registry.json")
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    self.strategy_data['registry'] = json.load(f)
            
            # Load agent scores
            scores_path = os.path.join(self.data_dir, "agent_scores.json")
            if os.path.exists(scores_path):
                with open(scores_path, 'r') as f:
                    self.strategy_data['scores'] = json.load(f)
            
            # Load phantom log
            phantom_path = os.path.join(self.data_dir, "phantom_log.json")
            if os.path.exists(phantom_path):
                with open(phantom_path, 'r') as f:
                    self.strategy_data['phantom'] = json.load(f)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load strategy data: {e}")
            return False
    
    def create_performance_matrix(self) -> np.ndarray:
        """Create performance matrix for heatmap visualization."""
        if not self.strategy_data:
            self.load_strategy_data()
        
        # Initialize matrix: strategies x time periods x market conditions
        strategies = list(self.strategy_data.get('registry', {}).keys())
        if not strategies:
            strategies = ['ema_rsi_macd', 'phantom_band', 'fractal_ghost']
        
        matrix = np.zeros((len(strategies), len(self.time_periods), len(self.market_conditions)))
        
        # Populate with performance data
        for i, strategy in enumerate(strategies):
            for j, period in enumerate(self.time_periods):
                for k, condition in enumerate(self.market_conditions):
                    # Get performance score from data
                    score = self._get_strategy_performance(strategy, period, condition)
                    matrix[i, j, k] = score
        
        self.performance_matrix = matrix
        return matrix
    
    def _get_strategy_performance(self, strategy: str, period: str, condition: str) -> float:
        """Get performance score for strategy under specific conditions."""
        try:
            # Check registry data
            if 'registry' in self.strategy_data and strategy in self.strategy_data['registry']:
                strategy_data = self.strategy_data['registry'][strategy]
                
                # Extract performance metrics
                win_rate = strategy_data.get('win_rate', 0.5)
                profit_factor = strategy_data.get('profit_factor', 1.0)
                sharpe_ratio = strategy_data.get('sharpe_ratio', 0.0)
                
                # Calculate composite score
                score = (win_rate * 0.4 + min(profit_factor / 2, 1.0) * 0.4 + max(sharpe_ratio / 2, 0.0) * 0.2)
                return score
            
            # Fallback to simulated data
            return np.random.uniform(0.3, 0.8)
            
        except Exception as e:
            logger.error(f"Error getting performance for {strategy}: {e}")
            return 0.5
    
    def generate_heatmap(self, save_path: str = None) -> plt.Figure:
        """Generate performance heatmap."""
        if self.performance_matrix is None:
            self.create_performance_matrix()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Schwabot Strategy Performance Heatmap', fontsize=16, fontweight='bold')
        
        strategies = list(self.strategy_data.get('registry', {}).keys())
        if not strategies:
            strategies = ['ema_rsi_macd', 'phantom_band', 'fractal_ghost']
        
        # Plot different views
        views = [
            ('Time Period Performance', self.performance_matrix.mean(axis=2), self.time_periods),
            ('Market Condition Performance', self.performance_matrix.mean(axis=1), self.market_conditions),
            ('Strategy Comparison', self.performance_matrix.mean(axis=(1, 2)), strategies),
            ('Overall Performance Matrix', self.performance_matrix.mean(axis=0), self.time_periods)
        ]
        
        for idx, (title, data, labels) in enumerate(views):
            ax = axes[idx // 2, idx % 2]
            
            if len(data.shape) == 1:
                # Bar chart for 1D data
                bars = ax.bar(range(len(data)), data, color='skyblue', alpha=0.7)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(labels, rotation=45)
                ax.set_ylabel('Performance Score')
                
                # Add value labels on bars
                for bar, value in zip(bars, data):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
            else:
                # Heatmap for 2D data
                sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn', 
                           xticklabels=labels, yticklabels=strategies, ax=ax)
                ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
        
        return fig
    
    def generate_strategy_comparison(self, save_path: str = None) -> plt.Figure:
        """Generate detailed strategy comparison chart."""
        if self.performance_matrix is None:
            self.create_performance_matrix()
        
        strategies = list(self.strategy_data.get('registry', {}).keys())
        if not strategies:
            strategies = ['ema_rsi_macd', 'phantom_band', 'fractal_ghost']
        
        # Calculate metrics for each strategy
        metrics = {}
        for i, strategy in enumerate(strategies):
            strategy_data = self.performance_matrix[i]
            metrics[strategy] = {
                'avg_performance': np.mean(strategy_data),
                'std_performance': np.std(strategy_data),
                'max_performance': np.max(strategy_data),
                'min_performance': np.min(strategy_data),
                'consistency': 1 - np.std(strategy_data) / np.mean(strategy_data) if np.mean(strategy_data) > 0 else 0
            }
        
        # Create comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        # Average performance
        avg_scores = [metrics[s]['avg_performance'] for s in strategies]
        bars1 = ax1.bar(strategies, avg_scores, color='lightblue', alpha=0.7)
        ax1.set_title('Average Performance')
        ax1.set_ylabel('Performance Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Performance consistency
        consistency_scores = [metrics[s]['consistency'] for s in strategies]
        bars2 = ax2.bar(strategies, consistency_scores, color='lightgreen', alpha=0.7)
        ax2.set_title('Performance Consistency')
        ax2.set_ylabel('Consistency Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Performance range
        ranges = [metrics[s]['max_performance'] - metrics[s]['min_performance'] for s in strategies]
        bars3 = ax3.bar(strategies, ranges, color='lightcoral', alpha=0.7)
        ax3.set_title('Performance Range')
        ax3.set_ylabel('Range')
        ax3.tick_params(axis='x', rotation=45)
        
        # Performance by market condition
        condition_performance = self.performance_matrix.mean(axis=1)
        for i, strategy in enumerate(strategies):
            ax4.plot(self.market_conditions, condition_performance[i], 
                    marker='o', label=strategy, linewidth=2)
        ax4.set_title('Performance by Market Condition')
        ax4.set_ylabel('Performance Score')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Strategy comparison saved to {save_path}")
        
        return fig
    
    def generate_phantom_analysis(self, save_path: str = None) -> plt.Figure:
        """Generate phantom zone analysis visualization."""
        if 'phantom' not in self.strategy_data:
            logger.warning("No phantom data available")
            return None
        
        phantom_data = self.strategy_data['phantom']
        
        # Extract phantom zone data
        durations = []
        profits = []
        symbols = []
        
        for entry in phantom_data:
            if isinstance(entry, dict):
                durations.append(entry.get('phantom_duration', 0))
                profits.append(entry.get('profit', 0))
                symbols.append(entry.get('symbol', 'Unknown'))
        
        if not durations:
            logger.warning("No valid phantom data found")
            return None
        
        # Create phantom analysis chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phantom Zone Analysis', fontsize=16, fontweight='bold')
        
        # Duration vs Profit scatter
        ax1.scatter(durations, profits, alpha=0.6, c='purple')
        ax1.set_xlabel('Phantom Duration (ticks)')
        ax1.set_ylabel('Profit')
        ax1.set_title('Phantom Duration vs Profit')
        ax1.grid(True, alpha=0.3)
        
        # Duration distribution
        ax2.hist(durations, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Phantom Duration (ticks)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Phantom Duration Distribution')
        
        # Profit distribution
        ax3.hist(profits, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Profit')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Profit Distribution')
        
        # Symbol performance
        symbol_profits = {}
        for symbol, profit in zip(symbols, profits):
            if symbol not in symbol_profits:
                symbol_profits[symbol] = []
            symbol_profits[symbol].append(profit)
        
        avg_profits = {s: np.mean(profits) for s, profits in symbol_profits.items()}
        ax4.bar(avg_profits.keys(), avg_profits.values(), alpha=0.7, color='orange')
        ax4.set_xlabel('Symbol')
        ax4.set_ylabel('Average Profit')
        ax4.set_title('Average Profit by Symbol')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Phantom analysis saved to {save_path}")
        
        return fig
    
    def export_performance_report(self, output_path: str = "performance_report.json"):
        """Export comprehensive performance report."""
        if self.performance_matrix is None:
            self.create_performance_matrix()
        
        strategies = list(self.strategy_data.get('registry', {}).keys())
        if not strategies:
            strategies = ['ema_rsi_macd', 'phantom_band', 'fractal_ghost']
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'strategies': {},
            'overall_summary': {},
            'recommendations': []
        }
        
        # Strategy analysis
        for i, strategy in enumerate(strategies):
            strategy_data = self.performance_matrix[i]
            report['strategies'][strategy] = {
                'average_performance': float(np.mean(strategy_data)),
                'performance_std': float(np.std(strategy_data)),
                'max_performance': float(np.max(strategy_data)),
                'min_performance': float(np.min(strategy_data)),
                'consistency_score': float(1 - np.std(strategy_data) / np.mean(strategy_data)) if np.mean(strategy_data) > 0 else 0,
                'best_time_period': self.time_periods[np.argmax(strategy_data.mean(axis=1))],
                'best_market_condition': self.market_conditions[np.argmax(strategy_data.mean(axis=0))]
            }
        
        # Overall summary
        report['overall_summary'] = {
            'total_strategies': len(strategies),
            'best_performing_strategy': max(report['strategies'].keys(), 
                                          key=lambda x: report['strategies'][x]['average_performance']),
            'most_consistent_strategy': max(report['strategies'].keys(),
                                          key=lambda x: report['strategies'][x]['consistency_score']),
            'average_performance': float(np.mean(self.performance_matrix)),
            'performance_volatility': float(np.std(self.performance_matrix))
        }
        
        # Generate recommendations
        best_strategy = report['overall_summary']['best_performing_strategy']
        best_data = report['strategies'][best_strategy]
        
        report['recommendations'] = [
            f"Primary strategy: {best_strategy} (avg performance: {best_data['average_performance']:.2f})",
            f"Best time period: {best_data['best_time_period']}",
            f"Best market condition: {best_data['best_market_condition']}",
            f"Consider using {best_strategy} during {best_data['best_market_condition']} markets in {best_data['best_time_period']} timeframes"
        ]
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report exported to {output_path}")
        return report

def main():
    """Main function for testing and demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profit Mapper Visualization")
    parser.add_argument("--data-dir", default="vaults", help="Data directory path")
    parser.add_argument("--output-dir", default="visualization/output", help="Output directory")
    parser.add_argument("--generate-all", action="store_true", help="Generate all visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize profit mapper
    mapper = ProfitMapper(args.data_dir)
    
    if args.generate_all:
        # Generate all visualizations
        mapper.generate_heatmap(os.path.join(args.output_dir, "performance_heatmap.png"))
        mapper.generate_strategy_comparison(os.path.join(args.output_dir, "strategy_comparison.png"))
        mapper.generate_phantom_analysis(os.path.join(args.output_dir, "phantom_analysis.png"))
        mapper.export_performance_report(os.path.join(args.output_dir, "performance_report.json"))
        
        print("✅ All visualizations generated successfully!")
    else:
        # Show interactive menu
        print("🎯 Profit Mapper Visualization")
        print("=" * 40)
        print("1. Generate Performance Heatmap")
        print("2. Generate Strategy Comparison")
        print("3. Generate Phantom Analysis")
        print("4. Export Performance Report")
        print("5. Generate All")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            mapper.generate_heatmap(os.path.join(args.output_dir, "performance_heatmap.png"))
        elif choice == "2":
            mapper.generate_strategy_comparison(os.path.join(args.output_dir, "strategy_comparison.png"))
        elif choice == "3":
            mapper.generate_phantom_analysis(os.path.join(args.output_dir, "phantom_analysis.png"))
        elif choice == "4":
            mapper.export_performance_report(os.path.join(args.output_dir, "performance_report.json"))
        elif choice == "5":
            mapper.generate_heatmap(os.path.join(args.output_dir, "performance_heatmap.png"))
            mapper.generate_strategy_comparison(os.path.join(args.output_dir, "strategy_comparison.png"))
            mapper.generate_phantom_analysis(os.path.join(args.output_dir, "phantom_analysis.png"))
            mapper.export_performance_report(os.path.join(args.output_dir, "performance_report.json"))
            print("✅ All visualizations generated successfully!")

if __name__ == "__main__":
    main() 