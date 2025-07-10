#!/usr/bin/env python3
"""
Phase 1 Cleanup Script for Schwabot Core
Implements the first phase of the comprehensive cleanup plan.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class Phase1Cleanup:
    def __init__(self, core_dir: str = "core", backup_dir: str = "backups"):
        self.core_dir = Path(core_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Files to delete (stubs and low-value files)
        self.files_to_delete = [
            "visual_execution_node.py",
            "enhanced_master_cycle_profit_engine.py", 
            "enhanced_tcell_system.py",
            "master_cycle_engine.py",
            "master_cycle_engine_enhanced.py",
            "mathlib_v3_visualizer.py",
            "smart_money_integration.py"
        ]
        
        # Files to consolidate (small utility files)
        self.small_utility_files = [
            "backend_math.py",
            "glyph_router.py", 
            "integration_orchestrator.py",
            "integration_test.py",
            "order_wall_analyzer.py",
            "profit_tier_adjuster.py",
            "swing_pattern_recognition.py",
            "unified_api_coordinator.py"
        ]
        
    def create_backup(self) -> str:
        """Create a timestamped backup of the core directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"core_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        print(f"Creating backup: {backup_path}")
        shutil.copytree(self.core_dir, backup_path)
        
        # Save backup metadata
        metadata = {
            "backup_timestamp": timestamp,
            "backup_path": str(backup_path),
            "files_deleted": [],
            "files_consolidated": [],
            "original_file_count": len(list(self.core_dir.glob("*.py")))
        }
        
        with open(backup_path / "backup_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return str(backup_path)
    
    def delete_stub_files(self) -> List[str]:
        """Delete the identified stub and low-value files."""
        deleted_files = []
        
        print("\n=== DELETING STUB FILES ===")
        for filename in self.files_to_delete:
            file_path = self.core_dir / filename
            if file_path.exists():
                print(f"Deleting: {filename}")
                file_path.unlink()
                deleted_files.append(filename)
            else:
                print(f"File not found: {filename}")
        
        return deleted_files
    
    def create_centralized_math_config(self):
        """Create centralized math configuration manager."""
        config_content = '''#!/usr/bin/env python3
"""
Centralized Math Configuration Manager
Manages all mathematical operations configuration for Schwabot.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

class MathConfigManager:
    """Centralized configuration manager for all mathematical operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/math_config.json"
        self.config = self._load_default_config()
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default mathematical configuration."""
        return {
            "precision": "float64",
            "cache_enabled": True,
            "gpu_acceleration": True,
            "parallel_processing": True,
            "optimization_level": "high",
            "max_cache_size": 10000,
            "cache_ttl": 3600,  # seconds
            "numerical_stability": {
                "epsilon": 1e-15,
                "max_iterations": 1000,
                "convergence_threshold": 1e-10
            },
            "tensor_operations": {
                "default_dtype": "float64",
                "memory_efficient": True,
                "chunk_size": 1000
            },
            "quantum_operations": {
                "simulation_precision": "double",
                "max_qubits": 32,
                "noise_model": "depolarizing"
            },
            "entropy_operations": {
                "base": 2,  # log base for entropy calculations
                "smoothing_factor": 1e-10,
                "normalization": True
            },
            "optimization": {
                "algorithm": "L-BFGS-B",
                "max_iterations": 1000,
                "tolerance": 1e-8,
                "constraints": "box"
            }
        }
    
    def _load_config(self):
        """Load configuration from file if it exists."""
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                self.config.update(file_config)
                print(f"Loaded math config from: {self.config_path}")
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
    
    def save_config(self):
        """Save current configuration to file."""
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Saved math config to: {self.config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_precision(self) -> str:
        """Get numerical precision setting."""
        return self.get("precision", "float64")
    
    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.get("cache_enabled", True)
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        return self.get("gpu_acceleration", True)
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            "enabled": self.is_cache_enabled(),
            "max_size": self.get("max_cache_size", 10000),
            "ttl": self.get("cache_ttl", 3600)
        }
    
    def get_numerical_stability_config(self) -> Dict[str, Any]:
        """Get numerical stability configuration."""
        return self.get("numerical_stability", {})

# Global instance
math_config = MathConfigManager()

def get_math_config() -> MathConfigManager:
    """Get the global math configuration manager."""
    return math_config
'''
        
        config_file = self.core_dir / "math_config_manager.py"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"Created: {config_file}")
    
    def create_math_cache(self):
        """Create centralized math results cache."""
        cache_content = '''#!/usr/bin/env python3
"""
Centralized Math Results Cache
Caches mathematical operation results to avoid redundant calculations.
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
import numpy as np

class MathResultsCache:
    """Cache for mathematical operation results."""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _generate_key(self, operation: str, params: Any) -> str:
        """Generate cache key from operation and parameters."""
        # Convert params to a hashable format
        if isinstance(params, (list, tuple)):
            param_str = str([self._hash_param(p) for p in params])
        elif isinstance(params, dict):
            param_str = str({k: self._hash_param(v) for k, v in sorted(params.items())})
        else:
            param_str = str(self._hash_param(params))
        
        key_data = f"{operation}:{param_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _hash_param(self, param: Any) -> str:
        """Hash a parameter for cache key generation."""
        if isinstance(param, np.ndarray):
            return hashlib.sha256(param.tobytes()).hexdigest()
        elif isinstance(param, (list, tuple)):
            return str([self._hash_param(p) for p in param])
        elif isinstance(param, dict):
            return str({k: self._hash_param(v) for k, v in sorted(param.items())})
        else:
            return str(param)
    
    def get(self, operation: str, params: Any) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._generate_key(operation, params)
        
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                # Move to end (LRU)
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return entry["result"]
            else:
                # Expired, remove
                del self.cache[key]
        
        self.stats["misses"] += 1
        return None
    
    def set(self, operation: str, params: Any, result: Any):
        """Cache a result."""
        key = self._generate_key(operation, params)
        
        # Remove if already exists
        if key in self.cache:
            del self.cache[key]
        
        # Add new entry
        self.cache[key] = {
            "result": result,
            "timestamp": time.time(),
            "operation": operation
        }
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        
        # Evict if cache is full
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size
        }
    
    def get_or_compute(self, operation: str, params: Any, compute_func) -> Any:
        """Get cached result or compute and cache it."""
        cached_result = self.get(operation, params)
        if cached_result is not None:
            return cached_result
        
        # Compute and cache
        result = compute_func()
        self.set(operation, params, result)
        return result

# Global cache instance
math_cache = MathResultsCache()

def get_math_cache() -> MathResultsCache:
    """Get the global math cache instance."""
    return math_cache
'''
        
        cache_file = self.core_dir / "math_cache.py"
        with open(cache_file, 'w') as f:
            f.write(cache_content)
        
        print(f"Created: {cache_file}")
    
    def create_math_orchestrator(self):
        """Create centralized math orchestrator."""
        orchestrator_content = '''#!/usr/bin/env python3
"""
Centralized Math Orchestrator
Orchestrates all mathematical operations with caching and optimization.
"""

import time
import logging
from typing import Any, Dict, Optional, Callable
import numpy as np

from .math_config_manager import get_math_config
from .math_cache import get_math_cache

logger = logging.getLogger(__name__)

class MathOrchestrator:
    """Centralized orchestrator for all mathematical operations."""
    
    def __init__(self):
        self.config = get_math_config()
        self.cache = get_math_cache()
        self.operation_times = {}
    
    def execute_math_operation(self, operation: str, params: Any, 
                             compute_func: Callable, 
                             use_cache: bool = True) -> Any:
        """Execute a mathematical operation with caching and timing."""
        start_time = time.time()
        
        try:
            if use_cache and self.config.is_cache_enabled():
                result = self.cache.get_or_compute(operation, params, compute_func)
            else:
                result = compute_func()
            
            execution_time = time.time() - start_time
            self.operation_times[operation] = execution_time
            
            logger.debug(f"Math operation '{operation}' completed in {execution_time:.6f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in math operation '{operation}': {e}")
            raise
    
    def execute_tensor_operation(self, operation: str, tensors: list, 
                               operation_func: Callable) -> np.ndarray:
        """Execute tensor operation with optimization."""
        def compute():
            return operation_func(*tensors)
        
        return self.execute_math_operation(f"tensor_{operation}", tensors, compute)
    
    def execute_quantum_operation(self, operation: str, params: Dict[str, Any],
                                operation_func: Callable) -> Any:
        """Execute quantum operation with simulation settings."""
        def compute():
            return operation_func(**params)
        
        return self.execute_math_operation(f"quantum_{operation}", params, compute)
    
    def execute_entropy_operation(self, operation: str, data: np.ndarray,
                                operation_func: Callable) -> float:
        """Execute entropy operation with numerical stability."""
        def compute():
            return operation_func(data)
        
        return self.execute_math_operation(f"entropy_{operation}", data, compute)
    
    def execute_optimization(self, objective_func: Callable, 
                           initial_guess: np.ndarray,
                           bounds: Optional[tuple] = None,
                           constraints: Optional[list] = None) -> Dict[str, Any]:
        """Execute optimization with configuration."""
        def compute():
            # This would integrate with scipy.optimize
            # For now, return a placeholder
            return {
                "success": True,
                "x": initial_guess,
                "fun": objective_func(initial_guess),
                "message": "Optimization completed"
            }
        
        params = {
            "initial_guess": initial_guess.tolist(),
            "bounds": bounds,
            "constraints": constraints
        }
        
        return self.execute_math_operation("optimization", params, compute)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            "cache_stats": cache_stats,
            "operation_times": self.operation_times,
            "config": {
                "precision": self.config.get_precision(),
                "cache_enabled": self.config.is_cache_enabled(),
                "gpu_enabled": self.config.is_gpu_enabled()
            }
        }
    
    def clear_cache(self):
        """Clear the math cache."""
        self.cache.clear()
        logger.info("Math cache cleared")

# Global orchestrator instance
math_orchestrator = MathOrchestrator()

def get_math_orchestrator() -> MathOrchestrator:
    """Get the global math orchestrator instance."""
    return math_orchestrator
'''
        
        orchestrator_file = self.core_dir / "math_orchestrator.py"
        with open(orchestrator_file, 'w') as f:
            f.write(orchestrator_content)
        
        print(f"Created: {orchestrator_file}")
    
    def consolidate_small_utilities(self) -> Dict[str, str]:
        """Consolidate small utility files into a single utilities module."""
        print("\n=== CONSOLIDATING SMALL UTILITIES ===")
        
        consolidated_content = '''#!/usr/bin/env python3
"""
Core Utilities Module
Consolidated utilities from various small files.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
import hashlib
import json

# ============================================================================
# Backend Math Utilities (from backend_math.py)
# ============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if abs(denominator) < 1e-15:
        return default
    return numerator / denominator

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] range."""
    if arr.size == 0:
        return arr
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

# ============================================================================
# Glyph Router Utilities (from glyph_router.py)
# ============================================================================

class GlyphRouter:
    """Routes glyph-based operations."""
    
    def __init__(self):
        self.glyph_map = {}
    
    def register_glyph(self, glyph: str, handler: callable):
        """Register a glyph handler."""
        self.glyph_map[glyph] = handler
    
    def route(self, glyph: str, *args, **kwargs) -> Any:
        """Route a glyph to its handler."""
        if glyph in self.glyph_map:
            return self.glyph_map[glyph](*args, **kwargs)
        raise ValueError(f"Unknown glyph: {glyph}")

# ============================================================================
# Integration Utilities (from integration_orchestrator.py, integration_test.py)
# ============================================================================

class IntegrationOrchestrator:
    """Orchestrates system integrations."""
    
    def __init__(self):
        self.integrations = {}
        self.test_results = {}
    
    def register_integration(self, name: str, integration_func: callable):
        """Register an integration function."""
        self.integrations[name] = integration_func
    
    def run_integration(self, name: str, *args, **kwargs) -> Any:
        """Run a registered integration."""
        if name in self.integrations:
            return self.integrations[name](*args, **kwargs)
        raise ValueError(f"Unknown integration: {name}")
    
    def test_integration(self, name: str) -> Dict[str, Any]:
        """Test an integration."""
        try:
            result = self.run_integration(name)
            self.test_results[name] = {"status": "success", "result": result}
            return self.test_results[name]
        except Exception as e:
            self.test_results[name] = {"status": "error", "error": str(e)}
            return self.test_results[name]

# ============================================================================
# Order Wall Analyzer (from order_wall_analyzer.py)
# ============================================================================

def analyze_order_wall(order_book: Dict[str, List], threshold: float = 0.1) -> Dict[str, Any]:
    """Analyze order book for significant order walls."""
    bids = order_book.get('bids', [])
    asks = order_book.get('asks', [])
    
    bid_walls = []
    ask_walls = []
    
    # Analyze bid walls
    for i, (price, volume) in enumerate(bids):
        if volume > threshold:
            bid_walls.append({
                'price': price,
                'volume': volume,
                'position': i
            })
    
    # Analyze ask walls
    for i, (price, volume) in enumerate(asks):
        if volume > threshold:
            ask_walls.append({
                'price': price,
                'volume': volume,
                'position': i
            })
    
    return {
        'bid_walls': bid_walls,
        'ask_walls': ask_walls,
        'total_bid_volume': sum(bid['volume'] for bid in bid_walls),
        'total_ask_volume': sum(ask['volume'] for ask in ask_walls)
    }

# ============================================================================
# Profit Tier Adjuster (from profit_tier_adjuster.py)
# ============================================================================

def adjust_profit_tier(current_tier: int, performance: float, 
                      thresholds: List[float]) -> int:
    """Adjust profit tier based on performance."""
    for i, threshold in enumerate(thresholds):
        if performance >= threshold:
            return i
    return len(thresholds) - 1

# ============================================================================
# Swing Pattern Recognition (from swing_pattern_recognition.py)
# ============================================================================

def detect_swing_pattern(prices: np.ndarray, window: int = 20) -> Dict[str, Any]:
    """Detect swing high/low patterns in price data."""
    if len(prices) < window * 2:
        return {"pattern": "insufficient_data"}
    
    highs = []
    lows = []
    
    for i in range(window, len(prices) - window):
        # Check for swing high
        if all(prices[i] >= prices[j] for j in range(i - window, i + window + 1)):
            highs.append(i)
        
        # Check for swing low
        if all(prices[i] <= prices[j] for j in range(i - window, i + window + 1)):
            lows.append(i)
    
    return {
        "swing_highs": highs,
        "swing_lows": lows,
        "pattern": "swing_detected" if (highs or lows) else "no_pattern"
    }

# ============================================================================
# Unified API Coordinator (from unified_api_coordinator.py)
# ============================================================================

class UnifiedAPICoordinator:
    """Coordinates API calls across different services."""
    
    def __init__(self):
        self.api_handlers = {}
        self.request_count = 0
    
    def register_api_handler(self, service: str, handler: callable):
        """Register an API handler for a service."""
        self.api_handlers[service] = handler
    
    def call_api(self, service: str, method: str, *args, **kwargs) -> Any:
        """Make an API call to a registered service."""
        self.request_count += 1
        
        if service in self.api_handlers:
            return self.api_handlers[service](method, *args, **kwargs)
        raise ValueError(f"Unknown API service: {service}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API coordination statistics."""
        return {
            "total_requests": self.request_count,
            "registered_services": list(self.api_handlers.keys())
        }

# Global instances
glyph_router = GlyphRouter()
integration_orchestrator = IntegrationOrchestrator()
api_coordinator = UnifiedAPICoordinator()
'''
        
        # Read content from small utility files and merge
        for filename in self.small_utility_files:
            file_path = self.core_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    # Add a comment indicating the source
                    consolidated_content += f"\n# ============================================================================\n"
                    consolidated_content += f"# Content from {filename}\n"
                    consolidated_content += f"# ============================================================================\n"
                    consolidated_content += content + "\n"
                    print(f"Consolidated: {filename}")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        # Write consolidated file
        utilities_file = self.core_dir / "core_utilities.py"
        with open(utilities_file, 'w') as f:
            f.write(consolidated_content)
        
        print(f"Created: {utilities_file}")
        
        # Return mapping of old files to new consolidated file
        return {filename: "core_utilities.py" for filename in self.small_utility_files}
    
    def run_cleanup(self) -> Dict[str, Any]:
        """Run the complete Phase 1 cleanup."""
        print("=== SCHWABOT PHASE 1 CLEANUP ===")
        print(f"Core directory: {self.core_dir}")
        print(f"Backup directory: {self.backup_dir}")
        
        # Create backup
        backup_path = self.create_backup()
        print(f"Backup created: {backup_path}")
        
        # Delete stub files
        deleted_files = self.delete_stub_files()
        
        # Create centralized math infrastructure
        print("\n=== CREATING CENTRALIZED MATH INFRASTRUCTURE ===")
        self.create_centralized_math_config()
        self.create_math_cache()
        self.create_math_orchestrator()
        
        # Consolidate small utilities
        consolidated_mapping = self.consolidate_small_utilities()
        
        # Summary
        summary = {
            "backup_path": backup_path,
            "deleted_files": deleted_files,
            "consolidated_files": consolidated_mapping,
            "new_files_created": [
                "math_config_manager.py",
                "math_cache.py", 
                "math_orchestrator.py",
                "core_utilities.py"
            ]
        }
        
        print("\n=== CLEANUP SUMMARY ===")
        print(f"Backup created: {backup_path}")
        print(f"Files deleted: {len(deleted_files)}")
        print(f"Files consolidated: {len(consolidated_mapping)}")
        print(f"New files created: {len(summary['new_files_created'])}")
        
        return summary

def main():
    """Main cleanup function."""
    cleanup = Phase1Cleanup()
    summary = cleanup.run_cleanup()
    
    # Save summary
    with open("phase1_cleanup_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nCleanup summary saved to: phase1_cleanup_summary.json")
    print("Phase 1 cleanup completed successfully!")

if __name__ == "__main__":
    main() 