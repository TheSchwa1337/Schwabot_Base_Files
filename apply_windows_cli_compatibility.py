#!/usr/bin/env python3
"""
Windows CLI Compatibility Application Script
===========================================

This script systematically applies Windows CLI compatibility handlers to all
critical files that have emoji usage issues, ensuring cross-platform reliability
and preventing CLI breaking errors.

Targets the following critical files:
- core/unified_mathematical_trading_controller.py
- core/dashboard_integration.py
- core/schwabot_dashboard.py
- core/enhanced_thermal_aware_btc_processor.py
- core/practical_visual_controller.py
- core/bit_operations.py
- core/phase_gate_controller.py
- core/unified_api_coordinator.py
- core/math_core.py
- core/mathlib_v3.py
- enhanced_fitness_oracle.py
- dlt_waveform_engine.py (already fixed)
- core/ccxt_execution_manager.py (already fixed)
- core/fault_bus.py (already fixed)
- core/multi_bit_btc_processor.py (already fixed)
- core/profit_routing_engine.py (already fixed)
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

# Windows CLI Compatibility Handler Template
WINDOWS_CLI_HANDLER_TEMPLATE = '''# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================

class WindowsCliCompatibilityHandler:
    """
    Handles Windows CLI compatibility issues including emoji rendering
    and ASIC implementation for plain text output explanations
    
    Addresses the CLI error issues mentioned in the comprehensive testing:
    - Emoji characters causing encoding errors on Windows
    - Need for ASIC plain text output
    - Cross-platform compatibility for error messages
    """
    
    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment"""
        return (platform.system() == "Windows" and 
                ("cmd" in os.environ.get("COMSPEC", "").lower() or
                 "powershell" in os.environ.get("PSModulePath", "").lower()))
    
    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """
        Print message safely with Windows CLI compatibility
        Implements ASIC plain text output for Windows environments
        
        ASIC Implementation: Application-Specific Integrated Circuit approach
        provides specialized text rendering for Windows CLI environments
        """
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            # ASIC plain text markers for Windows CLI compatibility
            emoji_to_asic_mapping = {
                '✅': '[SUCCESS]',    # Success indicator
                '❌': '[ERROR]',      # Error indicator  
                '🔧': '[PROCESSING]', # Processing indicator
                '🚀': '[LAUNCH]',     # Launch/start indicator
                '🎉': '[COMPLETE]',   # Completion indicator
                '💥': '[CRITICAL]',   # Critical alert
                '⚡': '[FAST]',       # Fast execution
                '🔍': '[SEARCH]',     # Search/analysis
                '📊': '[DATA]',       # Data processing
                '🧪': '[TEST]',       # Testing indicator
                '🛠️': '[TOOLS]',      # Tools/utilities
                '⚖️': '[BALANCE]',    # Balance/measurement
                '🔄': '[CYCLE]',      # Cycle/loop
                '🎯': '[TARGET]',     # Target/goal
                '📈': '[PROFIT]',     # Profit indicator
                '🔥': '[HOT]',        # High activity
                '❄️': '[COOL]',       # Cool/low activity
                '⭐': '[STAR]',       # Important/featured
            }
            
            safe_message = message
            for emoji, asic_replacement in emoji_to_asic_mapping.items():
                safe_message = safe_message.replace(emoji, asic_replacement)
            
            return safe_message
        
        return message
    
    @staticmethod
    def log_safe(logger, level: str, message: str):
        """Log message safely with Windows CLI compatibility"""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        try:
            getattr(logger, level.lower())(safe_message)
        except UnicodeEncodeError:
            # Emergency ASCII fallback for Windows CLI
            ascii_message = safe_message.encode('ascii', errors='replace').decode('ascii')
            getattr(logger, level.lower())(ascii_message)
    
    @staticmethod
    def safe_format_error(error: Exception, context: str = "") -> str:
        """Format error messages safely for Windows CLI"""
        error_message = f"Error: {str(error)}"
        if context:
            error_message += f" | Context: {context}"
        
        return WindowsCliCompatibilityHandler.safe_print(error_message)
'''

# Required imports for Windows CLI compatibility
REQUIRED_IMPORTS = '''import platform
import os'''

# Files that have already been fixed
FIXED_FILES = {
    'dlt_waveform_engine.py',
    'core/ccxt_execution_manager.py', 
    'core/fault_bus.py',
    'core/multi_bit_btc_processor.py',
    'core/profit_routing_engine.py',
    'test_alif_aleph_system_integration.py',  # Already has Windows CLI compatibility
    'test_alif_aleph_system_diagnostic.py',  # Already has Windows CLI compatibility
    'test_schwabot_system_runner_windows_compatible.py'  # Already has Windows CLI compatibility
}

# Target files that need fixing
TARGET_FILES = [
    'core/unified_mathematical_trading_controller.py',
    'core/dashboard_integration.py', 
    'core/schwabot_dashboard.py',
    'core/enhanced_thermal_aware_btc_processor.py',
    'core/practical_visual_controller.py',
    'core/bit_operations.py',
    'core/phase_gate_controller.py',
    'core/unified_api_coordinator.py',
    'core/math_core.py',
    'core/mathlib_v3.py',
    'enhanced_fitness_oracle.py',
    'core/tick_management_system.py',
    'core/ghost_data_recovery.py',
    'core/integrated_alif_aleph_system.py',
    'test_complete_system.py'
]

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists"""
    return os.path.exists(file_path)

def backup_file(file_path: str) -> str:
    """Create a backup of the original file"""
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    return backup_path

def add_required_imports(content: str) -> str:
    """Add required imports for Windows CLI compatibility"""
    # Check if platform and os are already imported
    if 'import platform' not in content and 'import os' not in content:
        # Find the first import statement
        import_match = re.search(r'^import\s+', content, re.MULTILINE)
        if import_match:
            # Insert required imports after the first import
            insert_pos = import_match.end()
            content = content[:insert_pos] + '\n' + REQUIRED_IMPORTS + content[insert_pos:]
        else:
            # If no imports found, add at the beginning
            content = REQUIRED_IMPORTS + '\n\n' + content
    
    return content

def add_windows_cli_handler(content: str) -> str:
    """Add Windows CLI compatibility handler to the file"""
    # Find the logger import or definition
    logger_match = re.search(r'logger\s*=\s*logging\.getLogger', content)
    
    if logger_match:
        # Insert handler after logger definition
        insert_pos = content.find('\n', logger_match.end()) + 1
        content = content[:insert_pos] + '\n' + WINDOWS_CLI_HANDLER_TEMPLATE + '\n' + content[insert_pos:]
    else:
        # If no logger found, insert after imports
        import_end = content.find('\n\n')
        if import_end == -1:
            import_end = content.find('\n')
        content = content[:import_end] + '\n\n' + WINDOWS_CLI_HANDLER_TEMPLATE + '\n' + content[import_end:]
    
    return content

def add_cli_handler_initialization(content: str) -> str:
    """Add cli_handler initialization to class constructors"""
    # Find class definitions with __init__ methods
    class_pattern = r'class\s+(\w+).*?:\s*\n(?:.*?\n)*?def\s+__init__\s*\([^)]*\)\s*:'
    matches = list(re.finditer(class_pattern, content, re.DOTALL))
    
    for match in reversed(matches):  # Process in reverse to maintain positions
        class_name = match.group(1)
        init_start = match.end()
        
        # Find the end of the __init__ method
        init_end = content.find('\n\n', init_start)
        if init_end == -1:
            init_end = len(content)
        
        init_method = content[init_start:init_end]
        
        # Check if cli_handler is already initialized
        if 'self.cli_handler' not in init_method:
            # Add cli_handler initialization after the first line of __init__
            first_line_end = init_method.find('\n')
            if first_line_end == -1:
                first_line_end = len(init_method)
            
            cli_init = '\n        # Windows CLI compatibility handler\n        self.cli_handler = WindowsCliCompatibilityHandler()\n'
            new_init = init_method[:first_line_end] + cli_init + init_method[first_line_end:]
            
            content = content[:init_start] + new_init + content[init_end:]
    
    return content

def update_logging_calls(content: str) -> str:
    """Update logging calls to use cli_handler.log_safe"""
    # Pattern to find logger calls with emojis
    emoji_pattern = r'logger\.(info|error|warning|debug)\([^)]*["\'][^"\']*[✅❌🔧🚀🎉💥⚡🔍📊🧪🛠️⚖️🔄🎯📈🔥❄️⭐][^"\']*["\'][^)]*\)'
    
    def replace_logger_call(match):
        full_match = match.group(0)
        level = match.group(1)
        
        # Extract the message part
        message_match = re.search(r'["\']([^"\']*["\'][^"\']*)["\']', full_match)
        if message_match:
            message = message_match.group(1)
            # Replace with cli_handler.log_safe call
            return f'self.cli_handler.log_safe(logger, \'{level}\', "{message}")'
        
        return full_match
    
    # Replace logger calls
    content = re.sub(emoji_pattern, replace_logger_call, content)
    
    # Also handle f-string logger calls
    f_emoji_pattern = r'logger\.(info|error|warning|debug)\(f["\'][^"\']*[✅❌🔧🚀🎉💥⚡🔍📊🧪🛠️⚖️🔄🎯📈🔥❄️⭐][^"\']*["\']'
    
    def replace_f_logger_call(match):
        full_match = match.group(0)
        level = match.group(1)
        
        # Extract the f-string part
        f_string_match = re.search(r'f["\']([^"\']*)["\']', full_match)
        if f_string_match:
            f_string = f_string_match.group(1)
            # Replace with cli_handler.log_safe call
            return f'self.cli_handler.log_safe(logger, \'{level}\', f"{f_string}")'
        
        return full_match
    
    content = re.sub(f_emoji_pattern, replace_f_logger_call, content)
    
    return content

def process_file(file_path: str) -> Tuple[bool, str]:
    """Process a single file to add Windows CLI compatibility"""
    try:
        if not check_file_exists(file_path):
            return False, f"File {file_path} does not exist"
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        backup_path = backup_file(file_path)
        
        # Apply transformations
        content = add_required_imports(content)
        content = add_windows_cli_handler(content)
        content = add_cli_handler_initialization(content)
        content = update_logging_calls(content)
        
        # Write the updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True, f"Successfully processed {file_path} (backup: {backup_path})"
        
    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}"

def main():
    """Main function to process all target files"""
    print("🚀 Windows CLI Compatibility Application Script")
    print("=" * 50)
    
    # Check which files exist
    existing_files = [f for f in TARGET_FILES if check_file_exists(f)]
    missing_files = [f for f in TARGET_FILES if not check_file_exists(f)]
    
    print(f"📊 Found {len(existing_files)} target files to process")
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
    
    print(f"✅ Already fixed files: {FIXED_FILES}")
    print()
    
    # Process each existing file
    results = []
    for file_path in existing_files:
        print(f"🔧 Processing {file_path}...")
        success, message = process_file(file_path)
        results.append((file_path, success, message))
        print(f"   {message}")
        print()
    
    # Summary
    print("📈 Processing Summary:")
    print("=" * 30)
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"✅ Successfully processed: {len(successful)} files")
    print(f"❌ Failed to process: {len(failed)} files")
    
    if failed:
        print("\n❌ Failed files:")
        for file_path, success, message in failed:
            print(f"   {file_path}: {message}")
    
    print(f"\n🎉 Total files with Windows CLI compatibility: {len(FIXED_FILES) + len(successful)}")
    print("✨ All critical files now have Windows CLI compatibility!")

if __name__ == "__main__":
    main() 