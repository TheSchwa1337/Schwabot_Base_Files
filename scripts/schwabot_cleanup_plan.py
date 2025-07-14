#!/usr/bin/env python3
"""
Schwabot Cleanup and Implementation Plan
========================================

Based on analysis results, this script will:
1. DELETE non-functional files
2. REFACTOR files to be clean orchestration layers
3. IMPLEMENT mathematical logic where needed
4. Ensure proper integration with core/ directory

Usage:
    python schwabot_cleanup_plan.py
"""

import os
import shutil
from pathlib import Path


class SchwabotCleanupPlan:
    """Execute the cleanup and implementation plan."""
    
    def __init__(self):
        self.schwabot_dir = Path("schwabot")
        self.backup_dir = Path("schwabot_backup")
        
    def create_backup(self):
        """Create backup of schwabot directory before cleanup."""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        shutil.copytree(self.schwabot_dir, self.backup_dir)
        print(f"✅ Backup created at: {self.backup_dir}")
    
    def delete_files(self):
        """Delete non-functional files."""
        files_to_delete = [
            "schwabot/init/core/strategy_bit_mapper.py"  # Empty file
        ]
        
        for file_path in files_to_delete:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                print(f"🗑️  Deleted: {file_path}")
    
    def refactor_orchestration_files(self):
        """Refactor files to be clean orchestration layers."""
        
        # 1. Fix schwa_engine.py - main orchestration file
        self._fix_schwa_engine()
        
        # 2. Fix strategy_layered_gatekeeper.py
        self._fix_strategy_gatekeeper()
        
        # 3. Fix trade_executor.py
        self._fix_trade_executor()
        
        # 4. Fix other orchestration files
        self._fix_remaining_files()
    
    def _fix_schwa_engine(self):
        """Fix the main Schwabot engine to be a clean orchestration layer."""
        engine_file = Path("schwabot/init/core/schwa_engine.py")
        
        # Read current content
        with open(engine_file, 'r') as f:
            content = f.read()
        
        # Fix imports to use core modules
        content = content.replace(
            "from .data_feed import DataFeed",
            "from core.real_time_market_data import RealTimeMarketData as DataFeed"
        )
        
        # Add proper type annotations and docstrings
        content = content.replace(
            "def get_system_status():",
            "def get_system_status() -> Dict[str, Any]:"
        )
        
        # Write back
        with open(engine_file, 'w') as f:
            f.write(content)
        
        print(f"🔧 Refactored: {engine_file}")
    
    def _fix_strategy_gatekeeper(self):
        """Fix strategy gatekeeper to use core mathematical modules."""
        gatekeeper_file = Path("schwabot/init/core/strategy_layered_gatekeeper.py")
        
        # Read current content
        with open(gatekeeper_file, 'r') as f:
            content = f.read()
        
        # Add proper imports from core
        imports = """
from core.tensor_score_utils import TensorScoreUtils
from core.profit_optimization_engine import ProfitOptimizationEngine
from core.math_orchestrator import MathOrchestrator
"""
        
        # Add imports after existing imports
        content = content.replace(
            "from .profit_bucket_registry import ProfitBucketRegistry",
            "from .profit_bucket_registry import ProfitBucketRegistry" + imports
        )
        
        # Write back
        with open(gatekeeper_file, 'w') as f:
            f.write(content)
        
        print(f"🔧 Refactored: {gatekeeper_file}")
    
    def _fix_trade_executor(self):
        """Fix trade executor to use core trading modules."""
        executor_file = Path("schwabot/init/core/trade_executor.py")
        
        # Read current content
        with open(executor_file, 'r') as f:
            content = f.read()
        
        # Add proper imports from core
        imports = """
from core.smart_order_executor import SmartOrderExecutor
from core.secure_exchange_manager import SecureExchangeManager
"""
        
        # Add imports after existing imports
        content = content.replace(
            "import time",
            "import time" + imports
        )
        
        # Write back
        with open(executor_file, 'w') as f:
            f.write(content)
        
        print(f"🔧 Refactored: {executor_file}")
    
    def _fix_remaining_files(self):
        """Fix remaining orchestration files."""
        files_to_fix = [
            "schwabot/init/core/agent_memory.py",
            "schwabot/init/core/data_feed.py", 
            "schwabot/init/core/profit_bucket_registry.py",
            "schwabot/init/core/risk_manager.py",
            "schwabot/init/core/hash_drift_sync.py",
            "schwabot/init/core/registry_vote_matrix.py",
            "schwabot/init/core/strategy_registry.py",
            "schwabot/init/core/symbol_router.py",
            "schwabot/init/core/vector_band_gatekeeper.py"
        ]
        
        for file_path in files_to_fix:
            self._add_proper_docstrings_and_types(Path(file_path))
    
    def _add_proper_docstrings_and_types(self, file_path: Path):
        """Add proper docstrings and type annotations to a file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Add module docstring if missing
            if not content.strip().startswith('"""'):
                module_name = file_path.stem
                docstring = f'"""{module_name} module - Orchestration layer for Schwabot trading system."""\n\n'
                content = docstring + content
            
            # Write back
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"📝 Added docstrings to: {file_path}")
            
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    
    def implement_mathematical_logic(self):
        """Implement any missing mathematical logic that differs from core."""
        print("🧮 Implementing mathematical logic...")
        
        # The analysis showed no mathematical differences from core,
        # so we just need to ensure proper integration
        
        # Create integration bridge
        self._create_integration_bridge()
    
    def _create_integration_bridge(self):
        """Create a clean integration bridge between schwabot and core."""
        bridge_content = '''"""
Schwabot-Core Integration Bridge
===============================

This module provides clean integration between the Schwabot orchestration layer
and the core mathematical trading engine.
"""

from typing import Dict, Any, Optional
from core.math_orchestrator import MathOrchestrator
from core.math_config_manager import MathConfigManager
from core.math_cache import MathResultCache

class SchwabotCoreBridge:
    """Bridge between Schwabot orchestration and core mathematical engine."""
    
    def __init__(self):
        """Initialize the integration bridge."""
        self.math_config = MathConfigManager()
        self.math_cache = MathResultCache()
        self.math_orchestrator = MathOrchestrator()
        
        # Initialize core systems
        self._initialize_core_systems()
    
    def _initialize_core_systems(self):
        """Initialize all core mathematical systems."""
        try:
            # Activate math infrastructure
            self.math_config.activate()
            self.math_cache.activate()
            self.math_orchestrator.activate()
            
            print("✅ Core mathematical systems initialized")
            
        except Exception as e:
            print(f"❌ Error initializing core systems: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all integrated systems."""
        return {
            "schwabot": "active",
            "core_math": self.math_orchestrator.get_status(),
            "cache": self.math_cache.get_status(),
            "config": self.math_config.get_status()
        }
    
    def execute_trading_cycle(self, symbol: str) -> Dict[str, Any]:
        """Execute a complete trading cycle using core mathematical engine."""
        try:
            # Use core mathematical engine for calculations
            result = self.math_orchestrator.execute_trading_cycle(symbol)
            return result
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}

# Global bridge instance
core_bridge = SchwabotCoreBridge()
'''
        
        bridge_file = Path("schwabot/init/core/core_bridge.py")
        with open(bridge_file, 'w') as f:
            f.write(bridge_content)
        
        print(f"🌉 Created integration bridge: {bridge_file}")
    
    def run_cleanup(self):
        """Execute the complete cleanup plan."""
        print("🚀 Starting Schwabot cleanup and implementation...")
        
        # Step 1: Create backup
        self.create_backup()
        
        # Step 2: Delete non-functional files
        self.delete_files()
        
        # Step 3: Refactor orchestration files
        self.refactor_orchestration_files()
        
        # Step 4: Implement mathematical logic
        self.implement_mathematical_logic()
        
        print("\n✅ Schwabot cleanup and implementation complete!")
        print("📁 Backup available at: schwabot_backup/")

def main():
    """Main cleanup execution."""
    plan = SchwabotCleanupPlan()
    plan.run_cleanup()

if __name__ == "__main__":
    main() 