from typing import Any
#!/usr/bin/env python3
"""
Cleanup Obsolete Files - Systematic Removal of Problematic Content
================================================================

This script removes obsolete markdown files, conversation logs, and other
content that's causing flake8 parsing issues and cluttering the codebase.

Based on the systematic analysis of what's causing persistent flake8 errors.
"""

import os
import shutil
from pathlib import Path
from typing import List, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ObsoleteFileCleaner:
    """Systematic cleaner for obsolete and problematic files"""

    def __init__(self, root_dir -> Any: str = ".") -> Any:
        self.root_dir = Path(root_dir)
        self.removed_files: List[str] = []
        self.removed_dirs: List[str] = []

        # Files that should be removed (conversation logs, obsolete docs)
        self.obsolete_files = {
            # Conversation and analysis files
            "cursor_review_recent_flake8_errors_and.md",
            "cursor_integrating_visual_synthesis_for.md",
            "cursor_ensuring_integration_and_impleme.md",
            "ErrorFlast8Issue.txt",
            "Blink_Aleph_Alif_Code.txt",

            # Obsolete documentation files
            "8_principles_sustainment.md",
            "ALTITUDE_DASHBOARD_INTEGRATION_GUIDE.md",
            "ALTITUDE_DASHBOARD_SUMMARY.md",
            "BARE_EXCEPT_HANDLING_FIRST_STEP_IMPLEMENTATION_COMPLETE.md",
            "BTC_PROCESSOR_CONTROL_README.md",
            "BTC_PROCESSOR_INTEGRATION_GUIDE.md",
            "CCXT_INTEGRATION_SUMMARY.md",
            "COMPLETE_ALIF_ALEPH_ANALYSIS.md",
            "COMPLIANCE_IMPROVEMENT_SUMMARY.md",
            "COMPREHENSIVE_ARCHITECTURE_INTEGRATION_SUMMARY.md",
            "COMPREHENSIVE_FLAKE8_SOLUTION_GUIDE.md",
            "CRITICAL_INTEGRATION_PLAN.md",
            "DEPENDENCY_FIX_SUMMARY.md",
            "DEPENDENCY_INTEGRATION_REPORT.md",
            "DEPLOYMENT_GUIDE.md",
            "DEPLOYMENT_STATUS.md",
            "DESCRIPTIVE_NAMING_FIXES_SUMMARY.md",
            "ENHANCEMENT_SUMMARY.md",
            "FLAKE8_FIX_SUMMARY.md",
            "FLAKE8_PROGRESS_SUMMARY.md",
            "GHOST_ARCHITECTURE_5TH_STEP_COMPLETE.md",
            "IMPLEMENTATION_COMPLETE.md",
            "INTEGRATION_NAMING_SCHEMA_COMPLIANCE_SUMMARY.md",
            "INTEGRATION_STATUS_REPORT.md",
            "INTELLIGENT_SYSTEMS_IMPLEMENTATION_SUMMARY.md",
            "MAGIC_NUMBER_OPTIMIZATION_REVOLUTION_COMPLETE.md",
            "MATHEMATICAL_IMPLEMENTATION_STATUS.md",
            "MATHEMATICAL_IMPLEMENTATION_SUMMARY.md",
            "MATHEMATICAL_INFRASTRUCTURE_STATUS.md",
            "NEWS_INTEGRATION_GUIDE.md",
            "PRIORITY3_IMPLEMENTATION_SUMMARY.md",
            "PRODUCTION_READINESS_REPORT.md",
            "README_COMPLETE_SYSTEM.md",
            "README_HASH_RECOLLECTION.md",
            "README_MATHEMATICAL_FRAMEWORK.md",
            "SANITY_CHECK_REPORT.md",
            "SCHWABOT_FIXES_SUMMARY.md",
            "SECR_IMPLEMENTATION_STATUS.md",
            "SETUP_GUIDE.md",
            "SIMPLIFIED_API_README.md",
            "STRATEGY_SUSTAINMENT_INTEGRATION_GUIDE.md",
            "SUSTAINMENT_IMPLEMENTATION_PLAN.md",
            "TALIB_INSTALLATION.md",
            "UNIFIED_SYSTEM_README.md",
            "VISUAL_ARCHITECTURE_GUIDE.md",
            "VISUAL_CORE_INTEGRATION_REPORT.md",
            "WINDOWS_CLI_COMPATIBILITY.md",
        }

        # Directories that should be removed
        self.obsolete_dirs = {
            ".pytest_cache",
            "__pycache__",
            ".venv",
            "node_modules",
            "build",
            "dist",
            ".git",
        }

        # File patterns that indicate problematic content
        self.problematic_patterns = [
            "```",  # Markdown code fences
            "Copy\nEdit",  # Cursor/GPT artifacts
            "Thought for",  # ChatGPT thinking patterns
            "Based on what you're seeing",  # Conversation artifacts
        ]

    def is_problematic_file(self, file_path: Path) -> bool:
        """Check if a file contains problematic content"""
        if not file_path.is_file():
            return False

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Check for problematic patterns
            for pattern in self.problematic_patterns:
                if pattern in content:
                    return True

            # Check if it's a Python file with markdown content
            if file_path.suffix == '.py' and ('```' in content or 'Copy\nEdit' in content):
                return True

        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")

        return False

    def cleanup_obsolete_files(self, dry_run: bool = True) -> None:
        """Remove obsolete files"""
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Cleaning up obsolete files...")

        for filename in self.obsolete_files:
            file_path = self.root_dir / filename
            if file_path.exists():
                if dry_run:
                    logger.info(f"Would remove: {file_path}")
                else:
                    try:
                        file_path.unlink()
                        self.removed_files.append(str(file_path))
                        logger.info(f"Removed: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing {file_path}: {e}")

    def cleanup_obsolete_directories(self, dry_run: bool = True) -> None:
        """Remove obsolete directories"""
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Cleaning up obsolete directories...")

        for dirname in self.obsolete_dirs:
            dir_path = self.root_dir / dirname
            if dir_path.exists() and dir_path.is_dir():
                if dry_run:
                    logger.info(f"Would remove: {dir_path}")
                else:
                    try:
                        shutil.rmtree(dir_path)
                        self.removed_dirs.append(str(dir_path))
                        logger.info(f"Removed: {dir_path}")
                    except Exception as e:
                        logger.error(f"Error removing {dir_path}: {e}")

    def cleanup_problematic_content(self, dry_run: bool = True) -> None:
        """Remove files with problematic content"""
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Cleaning up files with problematic content...")

        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file() and self.is_problematic_file(file_path):
                if dry_run:
                    logger.info(f"Would remove (problematic content): {file_path}")
                else:
                    try:
                        file_path.unlink()
                        self.removed_files.append(str(file_path))
                        logger.info(f"Removed (problematic): {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing {file_path}: {e}")

    def cleanup_empty_files(self, dry_run: bool = True) -> None:
        """Remove empty or nearly empty files"""
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Cleaning up empty files...")

        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file():
                try:
                    if file_path.stat().st_size == 0:
                        if dry_run:
                            logger.info(f"Would remove (empty): {file_path}")
                        else:
                            file_path.unlink()
                            self.removed_files.append(str(file_path))
                            logger.info(f"Removed (empty): {file_path}")
                except Exception as e:
                    logger.warning(f"Error checking {file_path}: {e}")

    def cleanup_placeholder_files(self, dry_run: bool = True) -> None:
        """Remove files that are just placeholders"""
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Cleaning up placeholder files...")

        placeholder_content = {'', ' ', '\n', '\n\n'}

        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()

                    if content in placeholder_content:
                        if dry_run:
                            logger.info(f"Would remove (placeholder): {file_path}")
                        else:
                            file_path.unlink()
                            self.removed_files.append(str(file_path))
                            logger.info(f"Removed (placeholder): {file_path}")

                except Exception as e:
                    logger.warning(f"Error checking {file_path}: {e}")

    def run_full_cleanup(self, dry_run: bool = True) -> None:
        """Run complete cleanup process"""
        logger.info("=" * 60)
        logger.info(f"{'DRY RUN: ' if dry_run else ''}SYSTEMATIC CLEANUP STARTING")
        logger.info("=" * 60)

        # Reset counters
        self.removed_files = []
        self.removed_dirs = []

        # Run all cleanup steps
        self.cleanup_obsolete_files(dry_run)
        self.cleanup_obsolete_directories(dry_run)
        self.cleanup_problematic_content(dry_run)
        self.cleanup_empty_files(dry_run)
        self.cleanup_placeholder_files(dry_run)

        # Summary
        logger.info("=" * 60)
        logger.info("CLEANUP SUMMARY:")
        logger.info(f"Files to be removed: {len(self.removed_files)}")
        logger.info(f"Directories to be removed: {len(self.removed_dirs)}")
        logger.info("=" * 60)

        if not dry_run:
            logger.info("✅ Cleanup completed successfully!")
            logger.info("Your codebase should now be free of problematic content.")
        else:
            logger.info("💡 Run with dry_run=False to actually perform the cleanup")


def main() -> Any:
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Clean up obsolete and problematic files")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would be removed without actually removing (default)")
    parser.add_argument("--execute", action="store_true",
                       help="Actually perform the cleanup")
    parser.add_argument("--root-dir", default=".", help="Root directory to clean")

    args = parser.parse_args()

    # Determine if this is a dry run
    dry_run = not args.execute

    # Create cleaner and run
    cleaner = ObsoleteFileCleaner(args.root_dir)
    cleaner.run_full_cleanup(dry_run=dry_run)

    if dry_run:
        print("\n💡 To actually perform the cleanup, run:")
        print("   python tools/cleanup_obsolete_files.py --execute")


if __name__ == "__main__":
    main()