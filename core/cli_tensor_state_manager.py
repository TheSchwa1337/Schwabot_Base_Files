#!/usr/bin/env python3
"""
Tensor State Manager CLI - Advanced Tensor State Control

Provides command-line interface for managing tensor states,
BTC price processing, and mathematical tensor operations.

Features:
- Inspect tensor states and matrices
- Process BTC price data through tensor pipeline
- Manage tensor memory and cache
- Monitor tensor performance metrics
- Control tensor operations and calculations
"""

import argparse
import asyncio
import json
import sys
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.advanced_tensor_algebra import AdvancedTensorAlgebra
from core.strategy_bit_mapper import StrategyBitMapper
from core.fractal_core import FractalCore
from core.unified_profit_vectorization_system import UnifiedProfitVectorizationSystem
from utils.safe_print import safe_print, info, warn, error, success


class TensorStateManagerCLI:
    """CLI interface for tensor state management operations."""
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.tensor_algebra = None
        self.strategy_mapper = None
        self.fractal_core = None
        self.profit_system = None
        self.is_initialized = False
        
    async def initialize_system(self):
        """Initialize the tensor state management system."""
        try:
            info("Initializing Tensor State Management System...")
            
            # Initialize core components
            self.tensor_algebra = AdvancedTensorAlgebra()
            self.strategy_mapper = StrategyBitMapper("data/matrices")
            self.fractal_core = FractalCore()
            self.profit_system = UnifiedProfitVectorizationSystem()
            
            # Test system components
            await self._test_system()
            
            self.is_initialized = True
            success("✅ Tensor State Management System initialized successfully")
            
        except Exception as e:
            error(f"❌ Failed to initialize system: {e}")
            return False
            
        return True
    
    async def _test_system(self):
        """Test the tensor state management system."""
        info("Testing tensor system components...")
        
        # Test tensor algebra
        test_matrix = np.random.random((3, 3))
        tensor_result = self.tensor_algebra.tensor_dot_fusion(test_matrix, test_matrix)
        info(f"Tensor algebra test: {tensor_result.shape if hasattr(tensor_result, 'shape') else 'success'}")
        
        # Test strategy mapper
        test_hash = np.random.random(64)
        strategy_result = self.strategy_mapper.select_strategy(test_hash)
        info(f"Strategy mapper test: {strategy_result.get('status', 'success')}")
        
        # Test fractal core
        fractal_result = self.fractal_core.analyze_fractal_pattern(test_matrix)
        info(f"Fractal core test: {fractal_result.get('status', 'success')}")
        
        # Test profit system
        market_data = {"price": 50000.0, "volume": 1000.0, "volatility": 0.15}
        profit_result = self.profit_system.calculate_unified_profit(market_data)
        info(f"Profit system test: {profit_result.profit_value:.6f}")
    
    async def show_tensor_status(self):
        """Display comprehensive tensor system status."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info("🧮 TENSOR STATE MANAGEMENT SYSTEM STATUS")
        info("=" * 55)
        
        # Tensor algebra status
        tensor_stats = self.tensor_algebra.get_statistics()
        info(f"📊 Tensor Algebra Statistics:")
        info(f"  Total Operations: {tensor_stats.get('total_operations', 0)}")
        info(f"  Matrix Operations: {tensor_stats.get('matrix_operations', 0)}")
        info(f"  Vector Operations: {tensor_stats.get('vector_operations', 0)}")
        info(f"  Average Operation Time: {tensor_stats.get('avg_operation_time', 0):.3f}s")
        
        # Strategy mapper status
        mapper_stats = self.strategy_mapper.get_statistics()
        info(f"🗺️  Strategy Mapper Statistics:")
        info(f"  Total Strategies: {mapper_stats.get('total_strategies', 0)}")
        info(f"  Cache Hits: {mapper_stats.get('cache_hits', 0)}")
        info(f"  Cache Misses: {mapper_stats.get('cache_misses', 0)}")
        info(f"  Average Match Time: {mapper_stats.get('avg_match_time', 0):.3f}s")
        
        # Fractal core status
        fractal_stats = self.fractal_core.get_statistics()
        info(f"🌀 Fractal Core Statistics:")
        info(f"  Pattern Analysis: {fractal_stats.get('pattern_analysis', 0)}")
        info(f"  Fractal Dimensions: {fractal_stats.get('fractal_dimensions', 0)}")
        info(f"  Entropy Calculations: {fractal_stats.get('entropy_calculations', 0)}")
        
        # Memory usage
        memory_info = self._get_memory_usage()
        info(f"💾 Memory Usage:")
        info(f"  Tensor Cache: {memory_info.get('tensor_cache', 0):.1f}MB")
        info(f"  Strategy Cache: {memory_info.get('strategy_cache', 0):.1f}MB")
        info(f"  Fractal Cache: {memory_info.get('fractal_cache', 0):.1f}MB")
        info(f"  Total Memory: {memory_info.get('total_memory', 0):.1f}MB")
    
    async def inspect_tensor_state(self, tensor_name: str):
        """Inspect a specific tensor state."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info(f"🔍 INSPECTING TENSOR: {tensor_name}")
        info("=" * 40)
        
        # Get tensor information
        tensor_info = self.tensor_algebra.get_tensor_info(tensor_name)
        
        if not tensor_info:
            warn(f"Tensor '{tensor_name}' not found.")
            return
        
        info(f"Shape: {tensor_info.get('shape', 'Unknown')}")
        info(f"Data Type: {tensor_info.get('dtype', 'Unknown')}")
        info(f"Memory Usage: {tensor_info.get('memory_usage', 0):.2f}MB")
        info(f"Last Modified: {tensor_info.get('last_modified', 'Unknown')}")
        info(f"Operations Count: {tensor_info.get('operations_count', 0)}")
        
        # Show tensor data (first few elements)
        tensor_data = tensor_info.get('data', None)
        if tensor_data is not None:
            info(f"Data Preview:")
            if hasattr(tensor_data, 'shape') and len(tensor_data.shape) <= 2:
                # Show first few elements
                preview = tensor_data.flatten()[:10]
                info(f"  First 10 elements: {preview}")
            else:
                info(f"  Tensor shape: {tensor_data.shape}")
    
    async def process_btc_tensor(self, price: float, volume: float, volatility: float):
        """Process BTC price data through the tensor pipeline."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info(f"₿ PROCESSING BTC THROUGH TENSOR PIPELINE")
        info("=" * 45)
        
        market_data = {
            "price": price,
            "volume": volume,
            "volatility": volatility,
            "timestamp": time.time()
        }
        
        info(f"BTC Price: ${price:,.2f}")
        info(f"Volume: {volume:,.0f}")
        info(f"Volatility: {volatility:.3f}")
        
        # Step 1: Create price tensor
        info("\n📊 Step 1: Creating Price Tensor...")
        price_tensor = self.tensor_algebra.create_price_tensor(market_data)
        info(f"Price tensor shape: {price_tensor.shape}")
        info(f"Price tensor norm: {np.linalg.norm(price_tensor):.6f}")
        
        # Step 2: Generate hash vector
        info("\n🔗 Step 2: Generating Hash Vector...")
        hash_vector = self.strategy_mapper.generate_hash_vector(price_tensor)
        info(f"Hash vector length: {len(hash_vector)}")
        info(f"Hash vector norm: {np.linalg.norm(hash_vector):.6f}")
        
        # Step 3: Select strategy
        info("\n🎯 Step 3: Selecting Strategy...")
        strategy_result = self.strategy_mapper.select_strategy(hash_vector)
        info(f"Selected strategy: {strategy_result.get('strategy_name', 'Unknown')}")
        info(f"Strategy confidence: {strategy_result.get('confidence', 0):.3f}")
        info(f"Strategy tier: {strategy_result.get('strategy_tier', 'Unknown')}")
        
        # Step 4: Fractal analysis
        info("\n🌀 Step 4: Fractal Analysis...")
        fractal_result = self.fractal_core.analyze_fractal_pattern(price_tensor)
        info(f"Fractal dimension: {fractal_result.get('fractal_dimension', 0):.3f}")
        info(f"Entropy: {fractal_result.get('entropy', 0):.3f}")
        info(f"Pattern complexity: {fractal_result.get('pattern_complexity', 0):.3f}")
        
        # Step 5: Profit calculation
        info("\n💰 Step 5: Profit Calculation...")
        profit_result = self.profit_system.calculate_unified_profit(market_data)
        info(f"Profit value: {profit_result.profit_value:.6f}")
        info(f"Confidence: {profit_result.confidence:.3f}")
        info(f"Integration mode: {profit_result.integration_mode.value}")
        
        # Step 6: Tensor fusion
        info("\n⚡ Step 6: Tensor Fusion...")
        fusion_result = self.tensor_algebra.tensor_dot_fusion(price_tensor, hash_vector.reshape(-1, 1))
        info(f"Fusion result shape: {fusion_result.shape}")
        info(f"Fusion magnitude: {np.linalg.norm(fusion_result):.6f}")
        
        # Summary
        info("\n📋 PROCESSING SUMMARY")
        info("=" * 25)
        info(f"Final Signal Strength: {strategy_result.get('signal_strength', 0):.3f}")
        info(f"Final Confidence: {profit_result.confidence:.3f}")
        info(f"Processing Time: {time.time() - market_data['timestamp']:.3f}s")
    
    async def show_tensor_cache(self, limit: int = 10):
        """Display tensor cache contents."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info(f"💾 TENSOR CACHE CONTENTS (Top {limit})")
        info("=" * 45)
        
        cache_entries = self.tensor_algebra.get_cache_entries(limit=limit)
        
        if not cache_entries:
            warn("No tensor cache entries found.")
            return
        
        for i, entry in enumerate(cache_entries, 1):
            info(f"{i}. Tensor: {entry.get('name', 'Unknown')}")
            info(f"   Shape: {entry.get('shape', 'Unknown')}")
            info(f"   Memory: {entry.get('memory_usage', 0):.2f}MB")
            info(f"   Last Used: {entry.get('last_used', 'Unknown')}")
            info(f"   Access Count: {entry.get('access_count', 0)}")
            info("")
    
    async def clear_tensor_cache(self):
        """Clear the tensor cache."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info("🧹 Clearing Tensor Cache...")
        
        # Clear various caches
        self.tensor_algebra.clear_cache()
        self.strategy_mapper.clear_cache()
        self.fractal_core.clear_cache()
        
        success("✅ Tensor cache cleared successfully")
    
    async def show_performance_metrics(self):
        """Display performance metrics for tensor operations."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info("⚡ TENSOR PERFORMANCE METRICS")
        info("=" * 35)
        
        # Tensor algebra performance
        tensor_perf = self.tensor_algebra.get_performance_metrics()
        info(f"📊 Tensor Algebra Performance:")
        info(f"  Average Operation Time: {tensor_perf.get('avg_operation_time', 0):.3f}s")
        info(f"  Total Operations: {tensor_perf.get('total_operations', 0)}")
        info(f"  Cache Hit Rate: {tensor_perf.get('cache_hit_rate', 0):.1f}%")
        info(f"  Memory Efficiency: {tensor_perf.get('memory_efficiency', 0):.1f}%")
        
        # Strategy mapper performance
        mapper_perf = self.strategy_mapper.get_performance_metrics()
        info(f"🗺️  Strategy Mapper Performance:")
        info(f"  Average Match Time: {mapper_perf.get('avg_match_time', 0):.3f}s")
        info(f"  Total Matches: {mapper_perf.get('total_matches', 0)}")
        info(f"  Match Accuracy: {mapper_perf.get('match_accuracy', 0):.1f}%")
        
        # Fractal core performance
        fractal_perf = self.fractal_core.get_performance_metrics()
        info(f"🌀 Fractal Core Performance:")
        info(f"  Average Analysis Time: {fractal_perf.get('avg_analysis_time', 0):.3f}s")
        info(f"  Total Analyses: {fractal_perf.get('total_analyses', 0)}")
        info(f"  Pattern Detection Rate: {fractal_perf.get('pattern_detection_rate', 0):.1f}%")
        
        # Overall system performance
        overall_perf = self._calculate_overall_performance()
        info(f"🎯 Overall System Performance:")
        info(f"  System Efficiency: {overall_perf.get('system_efficiency', 0):.1f}%")
        info(f"  Response Time: {overall_perf.get('response_time', 0):.3f}s")
        info(f"  Throughput: {overall_perf.get('throughput', 0):.1f} ops/sec")
    
    async def export_tensor_data(self, tensor_name: str, file_path: str):
        """Export tensor data to file."""
        if not self.is_initialized:
            error("System not initialized. Run 'init' first.")
            return
        
        info(f"📤 EXPORTING TENSOR: {tensor_name}")
        info("=" * 35)
        
        try:
            # Get tensor data
            tensor_data = self.tensor_algebra.get_tensor_data(tensor_name)
            
            if tensor_data is None:
                warn(f"Tensor '{tensor_name}' not found.")
                return
            
            # Export based on file extension
            file_path = Path(file_path)
            
            if file_path.suffix == '.npy':
                np.save(file_path, tensor_data)
                info(f"Saved as NumPy array: {file_path}")
            elif file_path.suffix == '.json':
                # Convert to JSON-serializable format
                json_data = {
                    "name": tensor_name,
                    "shape": tensor_data.shape,
                    "dtype": str(tensor_data.dtype),
                    "data": tensor_data.tolist()
                }
                with open(file_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                info(f"Saved as JSON: {file_path}")
            elif file_path.suffix == '.csv':
                np.savetxt(file_path, tensor_data, delimiter=',')
                info(f"Saved as CSV: {file_path}")
            else:
                error(f"Unsupported file format: {file_path.suffix}")
                return
            
            success(f"✅ Tensor exported successfully to {file_path}")
            
        except Exception as e:
            error(f"❌ Export failed: {e}")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        try:
            # Calculate approximate memory usage
            tensor_cache_size = len(self.tensor_algebra.get_cache_entries()) * 0.1  # MB per entry
            strategy_cache_size = len(self.strategy_mapper.get_cache_entries()) * 0.05  # MB per entry
            fractal_cache_size = len(self.fractal_core.get_cache_entries()) * 0.02  # MB per entry
            
            return {
                "tensor_cache": tensor_cache_size,
                "strategy_cache": strategy_cache_size,
                "fractal_cache": fractal_cache_size,
                "total_memory": tensor_cache_size + strategy_cache_size + fractal_cache_size
            }
        except:
            return {"tensor_cache": 0, "strategy_cache": 0, "fractal_cache": 0, "total_memory": 0}
    
    def _calculate_overall_performance(self) -> Dict[str, float]:
        """Calculate overall system performance metrics."""
        try:
            # Get individual performance metrics
            tensor_perf = self.tensor_algebra.get_performance_metrics()
            mapper_perf = self.strategy_mapper.get_performance_metrics()
            fractal_perf = self.fractal_core.get_performance_metrics()
            
            # Calculate overall metrics
            avg_response_time = (
                tensor_perf.get('avg_operation_time', 0) +
                mapper_perf.get('avg_match_time', 0) +
                fractal_perf.get('avg_analysis_time', 0)
            ) / 3
            
            total_operations = (
                tensor_perf.get('total_operations', 0) +
                mapper_perf.get('total_matches', 0) +
                fractal_perf.get('total_analyses', 0)
            )
            
            throughput = total_operations / max(avg_response_time, 0.001)
            
            system_efficiency = (
                tensor_perf.get('memory_efficiency', 0) +
                mapper_perf.get('match_accuracy', 0) +
                fractal_perf.get('pattern_detection_rate', 0)
            ) / 3
            
            return {
                "system_efficiency": system_efficiency,
                "response_time": avg_response_time,
                "throughput": throughput
            }
        except:
            return {"system_efficiency": 0, "response_time": 0, "throughput": 0}
    
    async def run_interactive_mode(self):
        """Run interactive CLI mode."""
        info("🎮 INTERACTIVE TENSOR STATE MANAGER CLI")
        info("=" * 45)
        info("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\n🧮 tensor> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    info("👋 Goodbye!")
                    break
                elif command == 'help':
                    self._show_help()
                elif command == 'status':
                    await self.show_tensor_status()
                elif command == 'cache':
                    await self.show_tensor_cache()
                elif command == 'clear-cache':
                    await self.clear_tensor_cache()
                elif command == 'performance':
                    await self.show_performance_metrics()
                elif command.startswith('inspect '):
                    tensor_name = command.split(' ', 1)[1]
                    await self.inspect_tensor_state(tensor_name)
                elif command.startswith('btc '):
                    parts = command.split()
                    if len(parts) >= 4:
                        await self.process_btc_tensor(float(parts[1]), float(parts[2]), float(parts[3]))
                    else:
                        error("Usage: btc <price> <volume> <volatility>")
                elif command.startswith('export '):
                    parts = command.split()
                    if len(parts) >= 3:
                        await self.export_tensor_data(parts[1], parts[2])
                    else:
                        error("Usage: export <tensor_name> <file_path>")
                else:
                    warn(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                info("\n👋 Goodbye!")
                break
            except Exception as e:
                error(f"Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        info("📖 AVAILABLE COMMANDS:")
        info("  status                    - Show system status")
        info("  cache                     - Show tensor cache")
        info("  clear-cache               - Clear tensor cache")
        info("  performance               - Show performance metrics")
        info("  inspect <tensor_name>     - Inspect specific tensor")
        info("  btc <price> <volume> <vol> - Process BTC through tensor pipeline")
        info("  export <tensor> <file>    - Export tensor data")
        info("  quit/exit                 - Exit CLI")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Tensor State Manager CLI - Advanced Tensor State Control")
    parser.add_argument("--init", action="store_true", help="Initialize the system")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--cache", action="store_true", help="Show tensor cache")
    parser.add_argument("--clear-cache", action="store_true", help="Clear tensor cache")
    parser.add_argument("--performance", action="store_true", help="Show performance metrics")
    parser.add_argument("--inspect", metavar="TENSOR", help="Inspect specific tensor")
    parser.add_argument("--btc", nargs=3, metavar=("PRICE", "VOLUME", "VOLATILITY"), type=float, help="Process BTC through tensor pipeline")
    parser.add_argument("--export", nargs=2, metavar=("TENSOR", "FILE"), help="Export tensor data")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    
    args = parser.parse_args()
    
    cli = TensorStateManagerCLI()
    
    # Initialize if requested or if any command needs it
    if args.init or any([args.status, args.cache, args.clear_cache, args.performance, args.inspect, args.btc, args.export, args.interactive]):
        if not await cli.initialize_system():
            return 1
    
    # Execute commands
    if args.status:
        await cli.show_tensor_status()
    elif args.cache:
        await cli.show_tensor_cache()
    elif args.clear_cache:
        await cli.clear_tensor_cache()
    elif args.performance:
        await cli.show_performance_metrics()
    elif args.inspect:
        await cli.inspect_tensor_state(args.inspect)
    elif args.btc:
        await cli.process_btc_tensor(args.btc[0], args.btc[1], args.btc[2])
    elif args.export:
        await cli.export_tensor_data(args.export[0], args.export[1])
    elif args.interactive:
        await cli.run_interactive_mode()
    elif not args.init:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 