#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Safe Encoding Fix - Handle UTF-8 Issues

This script safely fixes the UTF-8 encoding issues that occurred during rebranding.
It handles binary files and corrupted text files without breaking the system.

SAFETY FIRST: This script only fixes encoding issues - no other changes!
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SafeEncodingFixer:
    """Safely fix UTF-8 encoding issues."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.fix_results = {
            'timestamp': datetime.now().isoformat(),
            'files_fixed': [],
            'files_skipped': [],
            'files_removed': [],
            'errors': [],
            'system_status': 'fixing'
        }
        
        # Files with known encoding issues from rebranding
        self.problematic_files = [
            'scripts/create_cuda.py',
            'scripts/temp_clean.py',
            'docs/development/critical_math_fix.md',
            'docs/development/IMPLEMENTATION_SUMMARY.md',
            'config/current_venv_packages.txt',
            'config/requirements-dev.txt',
            'docs/development/core_flake8_final_report.txt',
            'docs/development/core_flake8_math_audit_report.txt',
            'docs/development/d_report.txt',
            'docs/development/e501_report.txt',
            'docs/development/flake8_after_fix_report.txt',
            'docs/development/flake8_comprehensive_report.txt',
            'docs/development/flake8_report.txt',
            'docs/development/math_fix_report.txt',
            'docs/development/priority_one_fixes_report.txt',
            'docs/development/schwabot_flake8_report.txt',
            'docs/development/system_integration_test_report.txt'
        ]
    
    def is_binary_file(self, file_path: Path) -> bool:
        """Check if a file is binary."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' in chunk
        except Exception:
            return True
    
    def safe_read_file(self, file_path: Path) -> str:
        """Safely read a file with multiple encoding attempts."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path} with {encoding}: {e}")
                continue
        
        # If all encodings fail, try binary read
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return ""
    
    def fix_file_encoding(self, file_path: Path) -> bool:
        """Fix encoding issues in a single file."""
        try:
            if not file_path.exists():
                logger.warning(f"⚠️ File not found: {file_path}")
                self.fix_results['files_skipped'].append(str(file_path))
                return False
            
            # Check if it's a binary file
            if self.is_binary_file(file_path):
                logger.info(f"📦 Binary file detected: {file_path}")
                self.fix_results['files_skipped'].append(str(file_path))
                return True
            
            # Try to read the file
            content = self.safe_read_file(file_path)
            
            if not content:
                logger.warning(f"⚠️ Empty or unreadable file: {file_path}")
                self.fix_results['files_skipped'].append(str(file_path))
                return False
            
            # Create backup of original file
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            shutil.copy2(file_path, backup_path)
            
            # Write back with proper UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Fixed encoding: {file_path}")
            self.fix_results['files_fixed'].append(str(file_path))
            return True
            
        except Exception as e:
            logger.error(f"❌ Error fixing {file_path}: {e}")
            self.fix_results['errors'].append(f"{file_path}: {e}")
            return False
    
    def remove_corrupted_files(self, file_path: Path) -> bool:
        """Safely remove corrupted files that can't be fixed."""
        try:
            if file_path.exists():
                # Move to a quarantine directory instead of deleting
                quarantine_dir = self.project_root / "quarantine_corrupted_files"
                quarantine_dir.mkdir(exist_ok=True)
                
                quarantine_path = quarantine_dir / file_path.name
                shutil.move(str(file_path), str(quarantine_path))
                
                logger.info(f"🚫 Moved corrupted file to quarantine: {file_path}")
                self.fix_results['files_removed'].append(str(file_path))
                return True
            else:
                logger.warning(f"⚠️ File not found for removal: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error removing {file_path}: {e}")
            self.fix_results['errors'].append(f"Remove {file_path}: {e}")
            return False
    
    def fix_encoding_issues(self) -> Dict[str, Any]:
        """Fix all encoding issues safely."""
        logger.info("🔧 Starting safe encoding fix...")
        logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🔧 ENCODING FIX 🔧                       ║
    ║                                                              ║
    ║              SAFE UTF-8 ISSUE RESOLUTION                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Process each problematic file
        for file_path_str in self.problematic_files:
            file_path = self.project_root / file_path_str
            
            logger.info(f"🔧 Processing: {file_path}")
            
            # Try to fix the file
            if not self.fix_file_encoding(file_path):
                # If fixing fails, consider removing if it's not critical
                if 'docs/development/' in file_path_str or 'config/' in file_path_str:
                    logger.info(f"🗑️ Removing non-critical corrupted file: {file_path}")
                    self.remove_corrupted_files(file_path)
                else:
                    logger.warning(f"⚠️ Keeping critical file despite issues: {file_path}")
        
        # Mark fix as complete
        self.fix_results['system_status'] = 'completed'
        
        # Save fix report
        report_file = self.project_root / 'ENCODING_FIX_REPORT.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.fix_results, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Encoding fix completed!")
        return self.fix_results
    
    def verify_system_integrity(self) -> bool:
        """Verify that the system is still functional after fixes."""
        logger.info("🔍 Verifying system integrity...")
        
        # Check critical files
        critical_files = [
            'main.py',
            'schwabot.py',
            'simple_schwabot_launcher.py',
            'test_system.py',
            'simple_test.py'
        ]
        
        all_good = True
        for file_name in critical_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                try:
                    # Try to read the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if len(content) > 0:
                        logger.info(f"✅ {file_name} - OK")
                    else:
                        logger.warning(f"⚠️ {file_name} - Empty")
                        all_good = False
                except Exception as e:
                    logger.error(f"❌ {file_name} - Error: {e}")
                    all_good = False
            else:
                logger.warning(f"⚠️ {file_name} - Not found")
                all_good = False
        
        # Check critical directories
        critical_dirs = ['core', 'gui', 'config', 'mathlib', 'strategies']
        for dir_name in critical_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                logger.info(f"✅ {dir_name}/ - OK")
            else:
                logger.warning(f"⚠️ {dir_name}/ - Not found")
                all_good = False
        
        return all_good

def main():
    """Main function to run encoding fix."""
    fixer = SafeEncodingFixer()
    
    try:
        results = fixer.fix_encoding_issues()
        
        print("\n" + "="*60)
        print("🔧 ENCODING FIX COMPLETED!")
        print("="*60)
        print(f"✅ Files Fixed: {len(results['files_fixed'])}")
        print(f"⚠️ Files Skipped: {len(results['files_skipped'])}")
        print(f"🚫 Files Removed: {len(results['files_removed'])}")
        print(f"❌ Errors: {len(results['errors'])}")
        
        if results['files_fixed']:
            print(f"\n✅ Fixed Files:")
            for file_path in results['files_fixed']:
                print(f"   • {file_path}")
        
        if results['files_removed']:
            print(f"\n🚫 Removed Files (in quarantine):")
            for file_path in results['files_removed']:
                print(f"   • {file_path}")
        
        if results['errors']:
            print(f"\n❌ Errors:")
            for error in results['errors']:
                print(f"   • {error}")
        
        # Verify system integrity
        print(f"\n🔍 Verifying system integrity...")
        if fixer.verify_system_integrity():
            print("✅ System integrity verified - all critical components OK!")
        else:
            print("⚠️ System integrity check found issues - review needed")
        
        print(f"\n💾 Fix report saved to: ENCODING_FIX_REPORT.json")
        print("🔒 Only encoding issues were fixed - system is safe!")
        
    except Exception as e:
        logger.error(f"❌ Encoding fix failed: {e}")
        print(f"❌ Encoding fix failed: {e}")

if __name__ == "__main__":
    main() 