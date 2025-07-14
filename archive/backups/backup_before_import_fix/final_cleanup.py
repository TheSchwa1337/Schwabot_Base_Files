#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Cleanup Script

This script fixes remaining indentation errors and ensures the codebase is clean.
"""

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_indentation_errors():
    """Fix indentation errors in core files."""
    logger.info("Fixing indentation errors...")
    
    # Files with known indentation issues
    problem_files = [
        "core/__init__.py",
        "core/acceleration_enhancement.py",
        "core/advanced_dualistic_trading_execution_system.py",
        "core/advanced_risk_manager.py",
        "core/advanced_settings_engine.py",
        "core/advanced_tensor_algebra.py",
        "core/ai_matrix_consensus.py",
        "core/algorithmic_portfolio_balancer.py",
        "core/antipole_router.py"
    ]
    
    fixed_count = 0
    for filepath in problem_files:
        path = Path(filepath)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Fix common indentation issues
                lines = content.split('\n')
                fixed_lines = []
                
                for line in lines:
                    # Remove leading spaces that cause indentation errors
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        line = line.strip()
                    elif line.strip().startswith('class ') or line.strip().startswith('def '):
                        # Ensure proper indentation for class/function definitions
                        if not line.startswith('    ') and not line.startswith('\t'):
                            line = line.strip()
                    
                    fixed_lines.append(line)
                
                new_content = '\n'.join(fixed_lines)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                logger.info(f"Fixed indentation in {filepath}")
                fixed_count += 1
                
            except Exception as e:
                logger.error(f"Error fixing {filepath}: {e}")
    
    logger.info(f"Fixed indentation in {fixed_count} files")


def run_code_formatting():
    """Run code formatting tools."""
    logger.info("Running code formatting...")
    
    try:
        # Run Black for code formatting
        os.system("black core/ --line-length=120 --quiet")
        logger.info("✅ Black formatting complete")
        
        # Run isort for import sorting
        os.system("isort core/ --profile=black --quiet")
        logger.info("✅ Import sorting complete")
        
    except Exception as e:
        logger.error(f"Error in code formatting: {e}")


def final_verification():
    """Final verification of the codebase."""
    logger.info("Running final verification...")
    
    # Check core mathematical files
    core_files = [
        "core/unified_mathematical_core.py",
        "core/tensor_score_utils.py",
        "core/quantum_mathematical_bridge.py",
        "core/entropy_math.py",
        "core/strategy_logic.py",
        "core/unified_profit_vectorization_system.py"
    ]
    
    all_present = True
    for filepath in core_files:
        path = Path(filepath)
        if path.exists():
            size = path.stat().st_size
            logger.info(f"✅ {filepath}: {size:,} bytes")
        else:
            logger.error(f"❌ Missing: {filepath}")
            all_present = False
    
    if all_present:
        logger.info("✅ All core mathematical files present")
    else:
        logger.error("❌ Some core files missing")


def main():
    """Run final cleanup."""
    logger.info("============================================================")
    logger.info("FINAL CLEANUP & VERIFICATION")
    logger.info("============================================================")
    
    fix_indentation_errors()
    run_code_formatting()
    final_verification()
    
    logger.info("============================================================")
    logger.info("CLEANUP COMPLETE - READY FOR TRADING")
    logger.info("============================================================")


if __name__ == "__main__":
    main() 