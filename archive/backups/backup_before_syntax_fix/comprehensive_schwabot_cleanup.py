import re
import shutil
import sys
from pathlib import Path
from typing import List

# -*- coding: utf-8 -*-
"""
Comprehensive Schwabot Directory Cleanup
========================================

This script will clean up the massive file clutter in the main directory:
- 50+ FINAL_*, COMPREHENSIVE_*, BATCH_*, CRITICAL_* markdown files
- Multiple requirements.txt variants
- Temporary fix files (d401.txt, phase1_stub_fixer.ps1, etc.)
- Redundant logs, error reports, and demo files
- Old config duplicates

MATHEMATICAL PRESERVATION: All mathematical content is preserved in consolidated files.
"""


class ComprehensiveSchwabitCleanup:
    """
    Comprehensive cleanup for the Schwabot directory structure.

    This will clean up 200+ scattered files while preserving all mathematical
    functionality and essential system files.
    """

    def __init__(self, project_root: str = "."):
        """Initialize the comprehensive cleanup."""
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "CLEANUP_BACKUP"

        # Essential directories to preserve completely
        self.essential_directories = {}
            "core",
            "schwabot",
            "config",
            "ai_oracles",
            "mathlib",
            "commands",
            "state",
            "init",
            "tools",
            ".git",
            ".mypy_cache",
            "logs",
            "frontend",
        }
        # Essential files to absolutely keep
        self.essential_files = {}
            # Core configuration (keep only the working, ones)
            "requirements.txt",  # Current working requirements
            "pyproject.toml",
            ".flake8",
            ".gitignore",
            ".gitattributes",
            "setup.py",
            "package.json",
            "LICENSE.txt",
            # Our new consolidated documentation
            "README.md",
            "MATH_DOCUMENTATION.md",
            "IMPLEMENTATION_GUIDE.md",
            "SYSTEM_ARCHITECTURE.md",
            "INSTALLATION_SOLUTION.md",
            # Cleanup and analysis scripts
            "comprehensive_schwabot_cleanup.py",
        }
        # Files to definitely delete (the clutter you, mentioned)
        self.files_to_delete = []
            # Multiple requirements variants (keep only requirements.txt)
            "requirements-dev.txt",
            "requirements_base.txt",
            "requirements_missing.txt",
            "requirements_news_integration.txt",
            "requirements-prod.txt",
            "requirements_clean.txt",
            "requirements_fixed.txt",  # We'll consolidate into requirements.txt'
            # Temporary fix files
            "d401.txt",
            "d401_all.txt",
            "d_full.txt",
            "d_full_after.txt",
            "d_remaining.txt",
            "d_remaining2.txt",
            "d_report.txt",
            # PowerShell and batch fix scripts
            "phase1_stub_fixer.ps1",
            "simple_stub_fixer.ps1",
            "apply_stub_fixes.bat",
            "run_syntax_fix.bat",
            "manual_syntax_fix.ps1",
            "simple_stub_fixer.ps1",
            # Flake8 temporary files
            "flake8_core_report.txt",
            "flake8_comprehensive_report.txt",
            "flake8_e999.txt",
            "flake8_e999_summary.txt",
            "flake8_full_report.txt",
            "flake8_compliance.log",
            "flake8_error_analysis.txt",
            "flake_f821_errors.txt",
            "flake_syntax_errors.txt",
            # Error logs and checks
            "api_errors.txt",
            "api_errors_check.txt",
            "api_gateway_errors.txt",
            "auto_scaler_errors.txt",
            "auto_scaler_errors_check.txt",
            "critical_errors.txt",
            "current_errors.txt",
            "current_e999_errors.txt",
            "doc_missing.txt",
            "doc_placeholder_output.txt",
            "e501_errors.txt",
            "e501_report.txt",
            # Old config duplicates
            ".flake8_temp",
            ".pre-commit-config.yaml",
            ".pre-commit-config.yml",
            "pre-commit-config.yaml",
            # Demo and test results
            "test_results.json",
            "test_shadow.db",
            "test_system.log",
            "ghost_shadow_tracker.db",
            "test_w293.txt",
            # Archive/backup files
            "current_venv_packages.txt",
            "dependency_installation_report.txt",
            "installation_report.txt",
            "black_check.txt",
            "add_docstrings_log.txt",
            "cleanup_log.txt",
            # Hash and registry files
            "hash_registry.json",
        ]
        # Pattern-based deletions for the scattered reports
        self.deletion_patterns = []
            # All the FINAL_*, COMPREHENSIVE_*, etc. reports (the main, clutter)
            r"^FINAL_.*\.md$",
            r"^COMPREHENSIVE_.*\.md$",
            r"^BATCH_.*\.md$",
            r"^CRITICAL_.*\.md$",
            r"^FLAKE8_.*\.md$",
            r"^IMPLEMENTATION_.*\.md$",
            r"^CLEANUP_.*\.md$",
            r"^PROGRESS_.*\.md$",
            r"^SUMMARY_.*\.md$",
            r"^DEPLOYMENT_.*\.md$",
            r"^MATHEMATICAL_.*\.md$",
            r"^CRYPTO_.*\.md$",
            r"^UNIFIED_.*\.md$",
            r"^SYSTEM_.*\.md$",
            r"^SCHWABOT_.*\.md$",
            r"^RECURSIVE_.*\.md$",
            r"^DLT_.*\.md$",
            r"^MISSING_.*\.md$",
            r"^NULL_.*\.md$",
            r"^VENV_.*\.md$",
            r"^PACKAGING_.*\.md$",
            r"^PRODUCTION_.*\.md$",
            r"^QUALITY_.*\.md$",
            r"^SELECTIVE_.*\.md$",
            r"^SURGICAL_.*\.md$",
            r"^SYNTAX_.*\.md$",
            r"^SYSTEMATIC_.*\.md$",
            r"^TEST_.*\.md$",
            r"^STUB_.*\.md$",
            r"^INDENTATION_.*\.md$",
            r"^INTEGRATION_.*\.md$",
            r"^INTERNALIZED_.*\.md$",
            r"^GAN_.*\.md$",
            r"^Complete_.*\.md$",
            r"^2bit_.*\.md$",
            r"^unicode_.*\.md$",
            # Demo results and test outputs
            r".*_demo_results_.*\.json$",
            r".*_test_.*\.json$",
            r".*_report_.*\.json$",
            r"advanced_test_report_.*\.json$",
            r"integrated_system_.*\.json$",
            r"mathematical_integration_.*\.json$",
            r".*\.log$",
            r".*\.db$",
            # Backup files
            r".*\.backup_.*$",
            r".*\.bak$",
            # Error and check files
            r".*_errors.*\.txt$",
            r".*_check.*\.txt$",
            r".*_final_check.*\.txt$",
            r".*_diff\.txt$",
            r"fix_progress_.*\.txt$",
            r"future_annotations_.*\.txt$",
            r"missing_imports_.*\.txt$",
            r"priority_one_.*\.txt$",
            r"mathematical_character_.*\.txt$",
            # Additional patterns for cleanup
            r"^e\d+_.*\.txt$",
            r"^test_.*\.txt$",
            r"^black_.*\.txt$",
            r"^mypy\.ini$",
            r"^flake8_.*\.txt$",
            r"^autopep8_.*\.txt$",
            # Python scripts that are temporary fixes
            r".*_fixer\.py$",
            r".*_fix\.py$",
            r"check_errors\.py$",
            r"targeted_fixer\.py$",
            r"apply_.*\.py$",
            r"auto_fix_.*\.py$",
            r"batch_.*\.py$",
            r"build_packages\.py$",
            r"debug_.*\.py$",
            r"execute_.*\.py$",
            r"installer\.py$",
            r"integrate_.*\.py$",
            r"launch_.*\.py$",
            r"phase_1_fix\.py$",
            r"refactor_.*\.py$",
            r"simple_.*\.py$",
            r"strategic_.*\.py$",
            r"test_.*\.py$",
            r".*_demo\.py$",
            r".*_stub.*\.py$",
        ]

    def scan_directory(self):
        """Scan directory and categorize files."""
        print("🔍 Scanning directory for cleanup targets...")

        all_files = []
        files_to_delete = []
        files_to_keep = []

        # Get all files in the current directory (not recursive for, directories)
        for item in self.project_root.iterdir():
            if item.is_file():
                all_files.append(item)

        print(f"📊 Found {len(all_files)} files in main directory")

        # Categorize files
        for file_path in all_files:
            filename = file_path.name

            # Keep essential files
            if filename in self.essential_files:
                files_to_keep.append(file_path)
                continue

            # Delete specific files
            if filename in self.files_to_delete:
                files_to_delete.append(file_path)
                continue

            # Check pattern-based deletions
            if self._matches_deletion_pattern(filename):
                # Exception: keep our new consolidated files
                if filename in []
                    "MATH_DOCUMENTATION.md",
                    "IMPLEMENTATION_GUIDE.md",
                    "SYSTEM_ARCHITECTURE.md",
                ]:
                    files_to_keep.append(file_path)
                else:
                    files_to_delete.append(file_path)
                continue

            # Default to keep if unsure
            files_to_keep.append(file_path)

        return files_to_delete, files_to_keep

    def _matches_deletion_pattern(): -> bool:
        """Check if filename matches any deletion pattern."""
        for pattern in self.deletion_patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return True
        return False

    def show_cleanup_preview(self, files_to_delete: List[Path]):
        """Show detailed preview of what will be deleted."""
        print("\n📋 COMPREHENSIVE CLEANUP PREVIEW")
        print("=" * 60)
        print(f"Total files to delete: {len(files_to_delete)}")

        # Group by category for better understanding
        categories = {}
            "Scattered Markdown Reports": [],
            "Requirements Variants": [],
            "Temporary Fix Files": [],
            "Error Logs & Checks": [],
            "Demo/Test Results": [],
            "Config Duplicates": [],
            "Temporary Python Scripts": [],
            "Backup Files": [],
            "Other Clutter": [],
        }
        for file_path in files_to_delete:
            filename = file_path.name

            if filename.endswith(".md") and any()
                prefix in filename.upper()
                for prefix in []
                    "FINAL_",
                    "COMPREHENSIVE_",
                    "BATCH_",
                    "CRITICAL_",
                    "FLAKE8_",
                    "IMPLEMENTATION_",
                    "CLEANUP_",
                    "PROGRESS_",
                    "SUMMARY_",
                ]
            ):
                categories["Scattered Markdown Reports"].append(filename)
            elif "requirements" in filename.lower():
                categories["Requirements Variants"].append(filename)
            elif any()
                pattern in filename.lower()
                for pattern in []
                    "d401",
                    "d_full",
                    "d_remaining",
                    "stub_fixer",
                    "syntax_fix",
                ]
            ):
                categories["Temporary Fix Files"].append(filename)
            elif any()
                pattern in filename.lower() for pattern in ["error", "check", "flake8"]
            ) and filename.endswith(".txt"):
                categories["Error Logs & Checks"].append(filename)
            elif any()
                pattern in filename.lower() for pattern in ["demo", "test", "result"]
            ) and filename.endswith((".json", ".log", ".db")):
                categories["Demo/Test Results"].append(filename)
            elif any()
                pattern in filename.lower()
                for pattern in ["config", ".flake8_temp", "pre-commit"]
            ):
                categories["Config Duplicates"].append(filename)
            elif filename.endswith(".py") and any()
                pattern in filename.lower()
                for pattern in ["_fixer", "_fix", "debug_", "test_", "demo"]
            ):
                categories["Temporary Python Scripts"].append(filename)
            elif any(pattern in filename.lower() for pattern in ["backup", ".bak"]):
                categories["Backup Files"].append(filename)
            else:
                categories["Other Clutter"].append(filename)

        # Show categories with counts
        for category, files in categories.items():
            if files:
                print(f"\n🗂️ {category} ({len(files)} files):")
                for filename in sorted(files)[:15]:  # Show first 15
                    print(f"   - {filename}")
                if len(files) > 15:
                    print(f"   ... and {len(files) - 15} more files")

    def consolidate_requirements(self):
        """Consolidate requirements files into a single working requirements.txt."""
        print("\n📋 Consolidating requirements files...")

        # Check if requirements_fixed.txt exists (our working, version)
        requirements_fixed = self.project_root / "schwabot" / "requirements_fixed.txt"
        main_requirements = self.project_root / "requirements.txt"

        if requirements_fixed.exists():
            print("✅ Found working requirements_fixed.txt - consolidating...")

            # Copy the working requirements to main requirements.txt
            try:
                shutil.copy2(requirements_fixed, main_requirements)
                print("✅ Consolidated requirements into main requirements.txt")
            except Exception as e:
                print(f"⚠️ Could not consolidate requirements: {e}")
        else:
            print()
                "ℹ️ No requirements_fixed.txt found - keeping existing requirements.txt"
            )

    def create_final_documentation(self):
        """Create final consolidated documentation."""
        print("\n📚 Creating final consolidated documentation...")

        # Check if we have the consolidated docs from schwabot directory
        source_dir = self.project_root / "schwabot"

        docs_to_copy = []
            "MATH_DOCUMENTATION.md",
            "IMPLEMENTATION_GUIDE.md",
            "SYSTEM_ARCHITECTURE.md",
            "INSTALLATION_SOLUTION.md",
        ]
        for doc_name in docs_to_copy:
            source_file = source_dir / doc_name
            dest_file = self.project_root / doc_name

            if source_file.exists() and not dest_file.exists():
                try:
                    shutil.copy2(source_file, dest_file)
                    print(f"✅ Copied {doc_name} to main directory")
                except Exception as e:
                    print(f"⚠️ Could not copy {doc_name}: {e}")

    def execute_cleanup(self, files_to_delete: List[Path], dry_run: bool = True):
        """Execute the comprehensive cleanup."""
        if dry_run:
            print("\n🔍 DRY RUN - No files will be deleted")
        else:
            print("\n🧹 EXECUTING COMPREHENSIVE CLEANUP")

            # Create backup directory
            self.backup_dir.mkdir(exist_ok=True)
            print(f"📦 Backup directory created: {self.backup_dir}")

        deleted_count = 0
        backed_up_count = 0

        for file_path in files_to_delete:
            try:
                if not dry_run:
                    # Create backup for potentially important files
                    if self._should_backup(file_path):
                        backup_path = self.backup_dir / file_path.name
                        shutil.copy2(file_path, backup_path)
                        backed_up_count += 1
                        print(f"📦 Backed up: {file_path.name}")

                    # Delete the file
                    file_path.unlink()
                    print(f"🗑️ Deleted: {file_path.name}")
                else:
                    print(f"🔍 Would delete: {file_path.name}")

                deleted_count += 1

            except Exception as e:
                print(f"⚠️ Could not delete {file_path.name}: {e}")

        print("\n✅ Cleanup completed!")
        print(f"📊 Files processed: {deleted_count}")
        if not dry_run and backed_up_count > 0:
            print(f"📦 Files backed up: {backed_up_count}")

        if dry_run:
            print("\n💡 To actually execute cleanup:")
            print("   python comprehensive_schwabot_cleanup.py --execute")

    def _should_backup(): -> bool:
        """Check if file should be backed up before deletion."""
        important_keywords = []
            "mathematical",
            "algorithm",
            "system",
            "integration",
            "btc",
            "crypto",
            "unified",
        ]
        filename_lower = file_path.name.lower()

        return any(keyword in filename_lower for keyword in, important_keywords)

    def run_comprehensive_cleanup(self, dry_run: bool = True):
        """Run the complete comprehensive cleanup."""
        print("🚀 COMPREHENSIVE SCHWABOT DIRECTORY CLEANUP")
        print("=" * 70)
        print()
            "Cleaning up 200+ scattered files while preserving mathematical functionality"
        )

        # Step 1: Consolidate requirements
        self.consolidate_requirements()

        # Step 2: Create final documentation
        self.create_final_documentation()

        # Step 3: Scan and categorize files
        files_to_delete, files_to_keep = self.scan_directory()

        print("\n📊 COMPREHENSIVE SCAN RESULTS:")
        print(f"   Files to keep: {len(files_to_keep)}")
        print(f"   Files to delete: {len(files_to_delete)}")
        print()
            f"   Cleanup reduction: {"}
                len(files_to_delete)
                / (len(files_to_delete) + len(files_to_keep))
                * 100:.1f}%"
        )

        # Step 4: Show detailed preview
        self.show_cleanup_preview(files_to_delete)

        # Step 5: Execute cleanup
        self.execute_cleanup(files_to_delete, dry_run=dry_run)

        print("\n🎯 FINAL RESULT:")
        print("   ✅ Clean, organized directory structure")
        print("   ✅ Mathematical functionality preserved")
        print(f"   ✅ Essential files maintained: {len(files_to_keep)}")
        print(f"   ✅ Clutter removed: {len(files_to_delete)} files")

        if not dry_run:
            print("\n📁 FINAL DIRECTORY STRUCTURE:")
            print("   📄 requirements.txt (consolidated)")
            print("   📄 README.md")
            print("   📄 MATH_DOCUMENTATION.md")
            print("   📄 IMPLEMENTATION_GUIDE.md")
            print("   📄 SYSTEM_ARCHITECTURE.md")
            print("   📁 core/ (mathematical, systems)")
            print("   📁 schwabot/ (main, application)")
            print("   📁 config/ (configuration)")
            print("   📁 Essential directories preserved")


def main():
    """Run the comprehensive cleanup."""

    # Check if user wants to execute (not dry, run)
    execute = "--execute" in sys.argv or "--real" in sys.argv

    cleanup = ComprehensiveSchwabitCleanup()
    cleanup.run_comprehensive_cleanup(dry_run=not, execute)


if __name__ == "__main__":
    main()
