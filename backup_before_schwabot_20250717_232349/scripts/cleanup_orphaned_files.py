#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 Schwabot Orphaned Files Cleanup Script

Identifies and handles orphaned files with syntax errors that are not part of the core trading system.
This script helps maintain a clean, production-ready codebase.

Usage:
    python scripts/cleanup_orphaned_files.py --dry-run    # Show what would be cleaned
    python scripts/cleanup_orphaned_files.py --clean      # Actually clean the files
    python scripts/cleanup_orphaned_files.py --archive    # Move to archive instead of delete
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrphanedFileCleaner:
    """Clean up orphaned files with syntax errors."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.archive_dir = self.project_root / "archive" / "orphaned_files"
        
        # Core directories that should be preserved
        self.core_dirs = {
            "core", "api", "tests", "config", "scripts", "data", 
            "logs", "reports", "docs", "web", "static", "templates",
            "strategies", "visualization", "backtesting", "models",
            "utils", "settings", "server", "runtime", "registry",
            "r1", "newmath", "ncco_core", "nano-core", "memory_stack",
            "mathlib", "init", "hash_recollection", "gui", "frontend",
            "fractals", "flask", "ferris_tick", "examples", "engine",
            "demo", "cli", "btc", "aleph_core", "agents", "profiles",
            "simulations", "vaults", "unified_trading_system_deployment",
            "ui", "tools", "smart_money", "cache", "results", "installers",
            "deployment", "secure", "test_matrices"
        }
        
        # Core files that should be preserved
        self.core_files = {
            "main.py", "__init__.py", ".flake8", ".gitignore", "LICENSE",
            "requirements.txt", "README.md", "setup.py", "pyproject.toml"
        }
        
        # Files that are known to be part of the core system
        self.essential_files = {
            "core/unified_mathematical_bridge.py",
            "core/risk_manager.py", 
            "core/automated_trading_engine.py",
            "core/btc_usdc_trading_integration.py",
            "core/enhanced_gpu_auto_detector.py",
            "core/pure_profit_calculator.py",
            "core/trade_registry.py",
            "core/lantern_core.py",
            "core/strategy_mapper.py",
            "core/exo_echo_signals.py",
            "api/flask_app.py",
            "api/live_trading_routes.py",
            "api/automated_trading_routes.py",
            "api/echo_signal_api.py"
        }
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip .git directory
            if ".git" in root:
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    python_files.append(file_path)
        
        return python_files
    
    def check_syntax(self, file_path: Path) -> bool:
        """Check if a Python file has valid syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), str(file_path), 'exec')
            return True
        except (SyntaxError, UnicodeDecodeError, IOError):
            return False
    
    def is_core_file(self, file_path: Path) -> bool:
        """Check if a file is part of the core system."""
        # Check if it's in a core directory
        for part in file_path.parts:
            if part in self.core_dirs:
                return True
        
        # Check if it's a core file
        if file_path.name in self.core_files:
            return True
        
        # Check if it's an essential file
        relative_path = file_path.relative_to(self.project_root)
        if str(relative_path) in self.essential_files:
            return True
        
        # Check if it's in the core directory
        if "core" in file_path.parts:
            return True
        
        return False
    
    def find_orphaned_files(self) -> Dict[str, List[Path]]:
        """Find orphaned files with syntax errors."""
        python_files = self.find_python_files()
        
        orphaned_files = {
            "syntax_errors": [],
            "potential_orphans": []
        }
        
        for file_path in python_files:
            # Check syntax
            if not self.check_syntax(file_path):
                if not self.is_core_file(file_path):
                    orphaned_files["syntax_errors"].append(file_path)
                else:
                    logger.warning(f"Core file with syntax error: {file_path}")
            
            # Check for potential orphans (files not in core directories)
            elif not self.is_core_file(file_path):
                orphaned_files["potential_orphans"].append(file_path)
        
        return orphaned_files
    
    def create_archive_directory(self):
        """Create archive directory if it doesn't exist."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Archive directory: {self.archive_dir}")
    
    def archive_file(self, file_path: Path) -> bool:
        """Archive a file instead of deleting it."""
        try:
            # Create relative path structure in archive
            relative_path = file_path.relative_to(self.project_root)
            archive_path = self.archive_dir / relative_path
            
            # Create parent directories
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file to archive
            shutil.move(str(file_path), str(archive_path))
            logger.info(f"Archived: {file_path} -> {archive_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to archive {file_path}: {e}")
            return False
    
    def delete_file(self, file_path: Path) -> bool:
        """Delete a file."""
        try:
            file_path.unlink()
            logger.info(f"Deleted: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False
    
    def cleanup(self, action: str = "dry-run"):
        """Perform the cleanup operation."""
        logger.info("🔍 Scanning for orphaned files...")
        
        orphaned_files = self.find_orphaned_files()
        
        if not orphaned_files["syntax_errors"] and not orphaned_files["potential_orphans"]:
            logger.info("✅ No orphaned files found!")
            return
        
        # Report findings
        logger.info(f"📊 Found {len(orphaned_files['syntax_errors'])} files with syntax errors")
        logger.info(f"📊 Found {len(orphaned_files['potential_orphans'])} potential orphaned files")
        
        if action == "dry-run":
            logger.info("\n🔍 DRY RUN - Files that would be cleaned:")
            
            if orphaned_files["syntax_errors"]:
                logger.info("\n❌ Files with syntax errors:")
                for file_path in orphaned_files["syntax_errors"]:
                    logger.info(f"   {file_path}")
            
            if orphaned_files["potential_orphans"]:
                logger.info("\n⚠️  Potential orphaned files:")
                for file_path in orphaned_files["potential_orphans"]:
                    logger.info(f"   {file_path}")
        
        elif action == "clean":
            logger.info("\n🗑️  Cleaning orphaned files...")
            
            # Handle files with syntax errors
            for file_path in orphaned_files["syntax_errors"]:
                self.delete_file(file_path)
            
            # Handle potential orphans (ask for confirmation)
            if orphaned_files["potential_orphans"]:
                logger.info(f"\n⚠️  Found {len(orphaned_files['potential_orphans'])} potential orphaned files")
                logger.info("These files have valid syntax but may not be part of the core system.")
                logger.info("Consider reviewing them manually before deletion.")
        
        elif action == "archive":
            logger.info("\n📦 Archiving orphaned files...")
            self.create_archive_directory()
            
            # Archive files with syntax errors
            for file_path in orphaned_files["syntax_errors"]:
                self.archive_file(file_path)
            
            # Archive potential orphans
            for file_path in orphaned_files["potential_orphans"]:
                self.archive_file(file_path)
        
        logger.info("\n✅ Cleanup operation completed!")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Clean up orphaned files in Schwabot")
    parser.add_argument(
        "--action", 
        choices=["dry-run", "clean", "archive"],
        default="dry-run",
        help="Action to perform (default: dry-run)"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    cleaner = OrphanedFileCleaner(args.project_root)
    cleaner.cleanup(args.action)

if __name__ == "__main__":
    main() 