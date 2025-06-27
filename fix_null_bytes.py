#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Null Byte Cleanup Script

Removes null bytes (\x00) from all Python files in the codebase.
This fixes the "source code string cannot contain null bytes" error
that prevents Flake8 from properly parsing files.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any


class NullByteCleaner:
    """Removes null bytes from Python files."""
    
    def __init__(self, root_dir: str = "."):
        """Initialize the cleaner."""
        self.root_dir = Path(root_dir)
        self.files_processed = 0
        self.files_with_null_bytes = 0
        self.total_null_bytes_removed = 0
        
    def clean_all_files(self) -> Dict[str, Any]:
        """Clean null bytes from all Python files."""
        print("🧹 Starting null byte cleanup...")
        
        # Find all Python files
        python_files = list(self.root_dir.rglob("*.py"))
        print(f"📁 Found {len(python_files)} Python files to scan")
        
        for file_path in python_files:
            try:
                self._clean_file(file_path)
            except Exception as e:
                print(f"❌ Error cleaning {file_path}: {e}")
        
        return self._generate_report()
    
    def _clean_file(self, file_path: Path):
        """Clean null bytes from a single file."""
        if not file_path.exists():
            return
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Check if file contains null bytes
            if b'\x00' in content:
                # Remove null bytes
                cleaned_content = content.replace(b'\x00', b'')
                
                # Write back cleaned content
                with open(file_path, 'wb') as f:
                    f.write(cleaned_content)
                
                null_bytes_removed = content.count(b'\x00')
                self.files_with_null_bytes += 1
                self.total_null_bytes_removed += null_bytes_removed
                
                print(f"✅ Cleaned {file_path}: removed {null_bytes_removed} null bytes")
            
            self.files_processed += 1
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate cleanup report."""
        report = {
            'status': 'completed',
            'files_processed': self.files_processed,
            'files_with_null_bytes': self.files_with_null_bytes,
            'total_null_bytes_removed': self.total_null_bytes_removed,
            'cleanup_percentage': (self.files_with_null_bytes / self.files_processed * 100) if self.files_processed > 0 else 0
        }
        
        return report


def main():
    """Main execution function."""
    print("🚀 PTNS Null Byte Cleaner")
    print("=" * 50)
    
    cleaner = NullByteCleaner()
    report = cleaner.clean_all_files()
    
    print("\n" + "=" * 50)
    print("📊 Cleanup Report")
    print("=" * 50)
    print(f"📁 Files Processed: {report['files_processed']}")
    print(f"🧹 Files with Null Bytes: {report['files_with_null_bytes']}")
    print(f"🗑️  Total Null Bytes Removed: {report['total_null_bytes_removed']}")
    print(f"📈 Cleanup Percentage: {report['cleanup_percentage']:.2f}%")
    
    if report['files_with_null_bytes'] > 0:
        print("\n✅ Null bytes have been removed. You can now run Flake8 again.")
    else:
        print("\n✅ No null bytes found. Files are already clean.")
    
    print("\n🎯 Next Steps:")
    print("1. Run flake8 to check for remaining errors")
    print("2. Address any syntax or style issues found")
    print("3. Test system functionality")
    
    return report


if __name__ == "__main__":
    main() 