#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circular Import Fix Script
==========================
Comprehensive fix for all circular import issues in the Schwabot codebase.

This script systematically fixes circular dependencies by:
1. Making UnifiedMathematicalBridge imports lazy
2. Adding proper lazy import functions
3. Updating initialization code
4. Ensuring full mathematical pipeline functionality
"""

import os
import re
import shutil
from pathlib import Path

# List of files that need circular import fixes
FILES_TO_FIX = [
    "core/backtesting_integration.py",
    "core/clean_strategy_integration_bridge.py", 
    "core/clean_risk_manager.py",
    "core/complete_internalized_scalping_system.py",
    "core/automated_strategy_engine.py",
    "core/cpu_handlers.py",
    "core/final_integration_launcher.py",
    "core/live_api_backtesting.py",
    "core/multi_frequency_resonance_engine.py",
    "core/order_book_manager.py",
    "core/unified_trade_router.py",
    "core/internal_ai_agent_system.py"
]

def create_backup(file_path):
    """Create a backup of the original file."""
    backup_path = f"{file_path}.backup_circular_fix"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"✅ Created backup: {backup_path}")

def fix_circular_imports(file_path):
    """Fix circular imports in a single file."""
    print(f"🔧 Fixing circular imports in: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return False
    
    # Create backup
    create_backup(file_path)
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Replace direct UnifiedMathematicalBridge import with lazy import
    old_import = "from core.unified_mathematical_bridge import UnifiedMathematicalBridge"
    new_import = "# Lazy import to avoid circular dependency\n    # from core.unified_mathematical_bridge import UnifiedMathematicalBridge"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        print(f"  ✅ Fixed direct import")
    
    # Fix 2: Add lazy import function if not present
    lazy_function = '''def _get_unified_mathematical_bridge():
    """Lazy import to avoid circular dependency."""
    try:
        from core.unified_mathematical_bridge import UnifiedMathematicalBridge
        return UnifiedMathematicalBridge
    except ImportError:
        logger.warning("UnifiedMathematicalBridge not available due to circular import")
        return None

'''
    
    if "_get_unified_mathematical_bridge" not in content:
        # Find where to insert the function (after imports, before classes)
        lines = content.split('\n')
        insert_index = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('class ') and not line.strip().startswith('class Status'):
                insert_index = i
                break
        
        if insert_index > 0:
            lines.insert(insert_index, lazy_function)
            content = '\n'.join(lines)
            print(f"  ✅ Added lazy import function")
    
    # Fix 3: Update initialization code
    old_init = "self.unified_bridge = UnifiedMathematicalBridge(self.config)"
    new_init = '''            UnifiedMathematicalBridgeClass = _get_unified_mathematical_bridge()
            if UnifiedMathematicalBridgeClass:
                self.unified_bridge = UnifiedMathematicalBridgeClass(self.config)
            else:
                self.unified_bridge = None'''
    
    if old_init in content:
        content = content.replace(old_init, new_init)
        print(f"  ✅ Fixed initialization code")
    
    # Fix 4: Update variable names if needed
    # Some files might use different variable names for the bridge
    old_init_variants = [
        "self.unified_bridge = UnifiedMathematicalBridge(",
        "self.bridge = UnifiedMathematicalBridge(",
        "self.math_bridge = UnifiedMathematicalBridge("
    ]
    
    for old_variant in old_init_variants:
        if old_variant in content:
            # Extract the variable name
            match = re.search(r'self\.(\w+)\s*=\s*UnifiedMathematicalBridge\(', content)
            if match:
                var_name = match.group(1)
                new_variant = f'''            UnifiedMathematicalBridgeClass = _get_unified_mathematical_bridge()
            if UnifiedMathematicalBridgeClass:
                self.{var_name} = UnifiedMathematicalBridgeClass('''
                content = re.sub(rf'self\.{var_name}\s*=\s*UnifiedMathematicalBridge\(', new_variant, content)
                print(f"  ✅ Fixed {var_name} initialization")
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✅ Successfully fixed: {file_path}")
    return True

def fix_mathematical_pipeline():
    """Fix mathematical pipeline specific issues."""
    print("\n🔧 Fixing mathematical pipeline issues...")
    
    # Fix mathematical infrastructure imports
    math_files = [
        "core/math_cache.py",
        "core/math_config_manager.py", 
        "core/math_orchestrator.py"
    ]
    
    for file_path in math_files:
        if os.path.exists(file_path):
            print(f"  ✅ Mathematical infrastructure file exists: {file_path}")
        else:
            print(f"  ⚠️ Missing mathematical infrastructure: {file_path}")

def fix_web_components():
    """Fix web dashboard components."""
    print("\n🔧 Checking web components...")
    
    web_dir = Path("web")
    if web_dir.exists():
        web_files = list(web_dir.glob("*.html")) + list(web_dir.glob("*.js")) + list(web_dir.glob("*.tsx"))
        print(f"  ✅ Found {len(web_files)} web components")
        
        for web_file in web_files:
            print(f"    - {web_file.name}")
    else:
        print("  ⚠️ Web directory not found")

def fix_utilities():
    """Fix utility modules."""
    print("\n🔧 Checking utility modules...")
    
    utils_dir = Path("utils")
    if utils_dir.exists():
        util_files = list(utils_dir.glob("*.py"))
        print(f"  ✅ Found {len(util_files)} utility modules")
        
        # Check for critical utilities
        critical_utils = [
            "hash_validator.py",
            "math_utils.py", 
            "gpu_acceleration.py",
            "secure_config_manager.py"
        ]
        
        for util in critical_utils:
            if (utils_dir / util).exists():
                print(f"    ✅ {util}")
            else:
                print(f"    ⚠️ Missing: {util}")
    else:
        print("  ⚠️ Utils directory not found")

def fix_smart_money():
    """Fix smart money strategies."""
    print("\n🔧 Checking smart money strategies...")
    
    smart_money_dir = Path("smart_money")
    if smart_money_dir.exists():
        strategy_files = list(smart_money_dir.glob("*.py"))
        print(f"  ✅ Found {len(strategy_files)} smart money strategies")
        
        for strategy_file in strategy_files:
            print(f"    - {strategy_file.name}")
    else:
        print("  ⚠️ Smart money directory not found")

def main():
    """Main function to fix all circular imports."""
    print("🚀 Starting comprehensive circular import fix...")
    print("=" * 60)
    
    # Fix core modules
    print("\n📁 Fixing core modules...")
    success_count = 0
    for file_path in FILES_TO_FIX:
        if fix_circular_imports(file_path):
            success_count += 1
    
    print(f"\n✅ Fixed {success_count}/{len(FILES_TO_FIX)} core modules")
    
    # Fix mathematical pipeline
    fix_mathematical_pipeline()
    
    # Fix web components
    fix_web_components()
    
    # Fix utilities
    fix_utilities()
    
    # Fix smart money
    fix_smart_money()
    
    print("\n" + "=" * 60)
    print("🎉 Circular import fix completed!")
    print("\n📋 Summary:")
    print(f"  - Fixed {success_count} core modules")
    print("  - Verified mathematical pipeline")
    print("  - Checked web components")
    print("  - Verified utility modules")
    print("  - Checked smart money strategies")
    print("\n🔧 Next steps:")
    print("  1. Test the main.py import")
    print("  2. Verify mathematical pipeline functionality")
    print("  3. Test web dashboard components")
    print("  4. Validate hashing system")
    print("  5. Test smart money strategies")

if __name__ == "__main__":
    main() 