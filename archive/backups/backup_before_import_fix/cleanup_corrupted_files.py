#!/usr/bin/env python3
"""
Cleanup script to remove corrupted files identified in flake8 analysis.
This script will safely delete the 60 files with syntax errors.
"""

import os
import shutil
from pathlib import Path

# List of corrupted files to delete (all have syntax errors)
CORRUPTED_FILES = [
    "core/ai_matrix_consensus.py",
    "core/algorithmic_portfolio_balancer.py",
    "core/antipole_router.py",
    "core/automated_flake8_math_audit.py",
    "core/backtest_visualization.py",
    "core/bio_profit_vectorization.py",
    "core/chrono_recursive_logic_function.py",
    "core/clean_math_foundation.py",
    "core/clean_profit_memory_echo.py",
    "core/clean_profit_vectorization.py",
    "core/clean_unified_math.py",
    "core/cli_entropy_manager.py",
    "core/cli_orbital_profit_control.py",
    "core/cli_tensor_state_manager.py",
    "core/consolidated_math_utils.py",
    "core/distributed_mathematical_processor.py",
    "core/enhanced_error_recovery_system.py",
    "core/enhanced_live_execution_mapper.py",
    "core/enhanced_profit_trading_strategy.py",
    "core/entropy_drift_tracker.py",
    "core/entropy_driven_risk_management.py",
    "core/entropy_enhanced_trading_executor.py",
    "core/entropy_signal_integration.py",
    "core/galileo_tensor_bridge.py",
    "core/glyph_phase_resolver.py",
    "core/gpu_handlers.py",
    "core/integrated_advanced_trading_system.py",
    "core/integration_orchestrator.py",
    "core/integration_test.py",
    "core/live_vector_simulator.py",
    "core/master_profit_coordination_system.py",
    "core/mathematical_optimization_bridge.py",
    "core/mathlib_v4.py",
    "core/math_implementation_fixer.py",
    "core/matrix_mapper.py",
    "core/matrix_math_utils.py",
    "core/orbital_profit_control_system.py",
    "core/phase3_batch_refactor.py",
    "core/production_deployment_manager.py",
    "core/profit_allocator.py",
    "core/profit_backend_dispatcher.py",
    "core/profit_decorators.py",
    "core/profit_matrix_feedback_loop.py",
    "core/profit_tier_adjuster.py",
    "core/pure_profit_calculator.py",
    "core/qsc_enhanced_profit_allocator.py",
    "core/qutrit_signal_matrix.py",
    "core/schwabot_mathematical_trading_engine.py",
    "core/slot_state_mapper.py",
    "core/swing_pattern_recognition.py",
    "core/system_integration.py",
    "core/system_state_profiler.py",
    "core/tensor_profit_audit.py",
    "core/tensor_recursion_solver.py",
    "core/tensor_weight_memory.py",
    "core/unified_mathematical_core.py",
    "core/unified_math_system.py",
    "core/unified_profit_vectorization_system.py",
    "core/unified_trading_pipeline.py",
    "core/vectorized_profit_orchestrator.py",
]

def create_backup_directory():
    """Create a backup directory for corrupted files."""
    backup_dir = Path("backup_corrupted_files")
    backup_dir.mkdir(exist_ok=True)
    return backup_dir

def backup_and_delete_files():
    """Backup corrupted files and then delete them."""
    backup_dir = create_backup_directory()
    
    deleted_count = 0
    backed_up_count = 0
    
    print(f"Starting cleanup of {len(CORRUPTED_FILES)} corrupted files...")
    print(f"Backup directory: {backup_dir.absolute()}")
    print()
    
    for file_path in CORRUPTED_FILES:
        path = Path(file_path)
        
        if not path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
            
        try:
            # Create backup
            backup_path = backup_dir / path.name
            shutil.copy2(path, backup_path)
            backed_up_count += 1
            print(f"✅ Backed up: {file_path}")
            
            # Delete original
            path.unlink()
            deleted_count += 1
            print(f"🗑️  Deleted: {file_path}")
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print()
    print(f"Cleanup completed:")
    print(f"  Files backed up: {backed_up_count}")
    print(f"  Files deleted: {deleted_count}")
    print(f"  Backup location: {backup_dir.absolute()}")
    
    return deleted_count, backed_up_count

def verify_cleanup():
    """Verify that all corrupted files have been removed."""
    print("\nVerifying cleanup...")
    
    remaining_corrupted = []
    for file_path in CORRUPTED_FILES:
        if Path(file_path).exists():
            remaining_corrupted.append(file_path)
    
    if remaining_corrupted:
        print(f"❌ {len(remaining_corrupted)} files still exist:")
        for file_path in remaining_corrupted:
            print(f"  - {file_path}")
    else:
        print("✅ All corrupted files have been successfully removed!")
    
    return len(remaining_corrupted) == 0

def main():
    """Main cleanup function."""
    print("=" * 60)
    print("SCHWABOT CORRUPTED FILES CLEANUP")
    print("=" * 60)
    print()
    print("This script will:")
    print("1. Create a backup of all corrupted files")
    print("2. Delete the corrupted files from the core directory")
    print("3. Verify the cleanup was successful")
    print()
    
    # Confirm before proceeding
    response = input("Do you want to proceed with the cleanup? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cleanup cancelled.")
        return
    
    print()
    
    # Perform cleanup
    deleted_count, backed_up_count = backup_and_delete_files()
    
    # Verify cleanup
    success = verify_cleanup()
    
    print()
    print("=" * 60)
    if success:
        print("🎉 CLEANUP SUCCESSFUL!")
        print(f"✅ {deleted_count} corrupted files removed")
        print(f"✅ {backed_up_count} files backed up")
        print("✅ Core system is now clean and ready for development")
    else:
        print("⚠️  CLEANUP INCOMPLETE")
        print("Some files may still need manual removal")
    print("=" * 60)

if __name__ == "__main__":
    main() 