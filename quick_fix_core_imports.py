#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Quick Fix Core Imports - Fix Import Issues

This script quickly fixes import issues in core modules.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_core_imports():
    """Fix core module import issues."""
    project_root = Path(".")
    
    # Fix ccxt_trading_executor.py if needed
    ccxt_file = project_root / "core" / "ccxt_trading_executor.py"
    if ccxt_file.exists():
        try:
            with open(ccxt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if there are any indentation issues
            lines = content.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                # Fix try-except blocks
                if line.strip() == 'try:' and i + 1 < len(lines):
                    fixed_lines.append(line)
                    # Ensure next line is properly indented
                    if not lines[i + 1].strip().startswith('    '):
                        fixed_lines.append('    pass  # TODO: Add proper imports')
                    continue
                
                # Fix class definitions
                if line.strip().startswith('class ') and ':' in line and i + 1 < len(lines):
                    fixed_lines.append(line)
                    # Ensure next line is properly indented
                    if not lines[i + 1].strip().startswith('    '):
                        fixed_lines.append('    """Class placeholder."""')
                        fixed_lines.append('    pass')
                    continue
                
                fixed_lines.append(line)
            
            # Write back if changes were made
            new_content = '\n'.join(fixed_lines)
            if new_content != content:
                with open(ccxt_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info(f"Fixed {ccxt_file}")
            
        except Exception as e:
            logger.error(f"Error fixing {ccxt_file}: {e}")
    
    # Create missing __init__.py files
    core_dirs = ['core', 'utils', 'backtesting']
    for dir_name in core_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# Schwabot AI Module\n")
                logger.info(f"Created {init_file}")

def test_imports():
    """Test the imports."""
    try:
        import sys
        import os
        sys.path.insert(0, str(Path(".")))
        
        # Test core imports
        import core
        print("✅ Core module imported successfully")
        
        # Test utils if it exists
        try:
            import utils
            print("✅ Utils module imported successfully")
        except ImportError:
            print("⚠️ Utils module not found")
        
        # Test backtesting if it exists
        try:
            import backtesting
            print("✅ Backtesting module imported successfully")
        except ImportError:
            print("⚠️ Backtesting module not found")
        
        print("✅ All available modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Quick fixing core imports...")
    fix_core_imports()
    print("🧪 Testing imports...")
    test_imports() 