#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Import Fixer for Schwabot Trading System

This script fixes all import-related issues including:
- Missing standard library imports
- Missing third-party library imports
- Undefined name references
- Import organization

Usage:
    python fix_imports_comprehensive.py
"""

import ast

# Configure logging
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Fix Unicode encoding issues on Windows
if sys.platform == 'win32':
    import codecs

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/import_fix.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ImportFixer:
    """Comprehensive import fixer for Python files."""

    def __init__(self, backup_dir: str = "backup_before_import_fix"):
        """Initialize the import fixer."""
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.fixed_files = []
        self.failed_files = []

        # Standard library imports that are commonly missing
        self.standard_library_imports = {
            'ModuleType': 'types',
            'ABC': 'abc',
            'abstractmethod': 'abc',
            '_iter_modules': 'pkgutil',
            'Queue': 'multiprocessing',
            'contextmanager': 'contextlib',
            'wraps': 'functools',
            'deque': 'collections',
            'defaultdict': 'collections',
            'BytesIO': 'io',
        }

        # Third-party library imports
        self.third_party_imports = {
            'plt': 'matplotlib.pyplot',
            'la': 'numpy.linalg',
            'cp': 'cupy',
            'np': 'numpy',
        }

        # Custom imports that need to be resolved
        self.custom_imports = {
            'QuantumStaticCore': 'core.quantum_static_core',
            'GalileoTensorBridge': 'core.galileo_tensor_bridge',
            'safe_cuda_operation': 'core.gpu_handlers',
        }

    def backup_file(self, file_path: Path) -> None:
        """Create backup of file before modification."""
        try:
            backup_path = self.backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backed up {file_path} to {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup {file_path}: {e}")

    def find_undefined_names(self, content: str, file_path: Path) -> Set[str]:
        """Find undefined names in the file content."""
        undefined_names = set()

        try:
            # Parse the AST to find undefined names
            tree = ast.parse(content)

            # Collect all names used in the file
            used_names = set()
            imported_names = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.name.split('.')[-1])
                        if alias.asname:
                            imported_names.add(alias.asname)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            imported_names.add(alias.name)
                            if alias.asname:
                                imported_names.add(alias.asname)

            # Find undefined names (excluding builtins)
            builtins = set(dir(__builtins__))
            undefined_names = used_names - imported_names - builtins

            # Filter out names that are likely defined in the file
            defined_names = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined_names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_names.add(target.id)

            undefined_names -= defined_names

        except SyntaxError:
            # If we can't parse the file, use regex to find potential undefined names
            logger.warning(f"Could not parse {file_path}, using regex fallback")
            undefined_names = self._find_undefined_names_regex(content)

        return undefined_names

    def _find_undefined_names_regex(self, content: str) -> Set[str]:
        """Find undefined names using regex as fallback."""
        undefined_names = set()

        # Common undefined names from the error report
        common_undefined = {
            'ModuleType',
            'ABC',
            'abstractmethod',
            '_iter_modules',
            'Queue',
            'mp',
            'contextmanager',
            'wraps',
            'deque',
            'defaultdict',
            'plt',
            'BytesIO',
            'la',
            'cp',
            'np',
            'QuantumStaticCore',
            'GalileoTensorBridge',
            'safe_cuda_operation',
            'error',
            'success',
            'warn',
        }

        for name in common_undefined:
            if re.search(rf'\b{re.escape(name)}\b', content):
                undefined_names.add(name)

        return undefined_names

    def generate_import_statements(self, undefined_names: Set[str], file_path: Path) -> List[str]:
        """Generate import statements for undefined names."""
        imports = []

        # Group imports by module
        standard_imports = {}
        third_party_imports = {}
        custom_imports = {}

        for name in undefined_names:
            if name in self.standard_library_imports:
                module = self.standard_library_imports[name]
                if module not in standard_imports:
                    standard_imports[module] = []
                standard_imports[module].append(name)
            elif name in self.third_party_imports:
                module = self.third_party_imports[name]
                if module not in third_party_imports:
                    third_party_imports[module] = []
                third_party_imports[module].append(name)
            elif name in self.custom_imports:
                module = self.custom_imports[name]
                if module not in custom_imports:
                    custom_imports[module] = []
                custom_imports[module].append(name)
            else:
                # Handle special cases
                if name in ('error', 'success', 'warn'):
                    # These are likely logging functions
                    imports.append('import logging')
                elif name == 'mp':
                    # multiprocessing alias
                    imports.append('import multiprocessing as mp')

        # Generate standard library imports
        for module, names in standard_imports.items():
            if len(names) == 1:
                imports.append(f'from {module} import {names[0]}')
            else:
                imports.append(f'from {module} import {", ".join(sorted(names))}')

        # Generate third-party imports
        for module, names in third_party_imports.items():
            if len(names) == 1:
                imports.append(f'from {module} import {names[0]}')
            else:
                imports.append(f'from {module} import {", ".join(sorted(names))}')

        # Generate custom imports
        for module, names in custom_imports.items():
            if len(names) == 1:
                imports.append(f'from {module} import {names[0]}')
            else:
                imports.append(f'from {module} import {", ".join(sorted(names))}')

        return imports

    def insert_imports(self, content: str, imports: List[str], file_path: Path) -> str:
        """Insert import statements at the top of the file."""
        lines = content.split('\n')

        # Find the first non-comment, non-docstring line
        insert_index = 0

        # Skip shebang and encoding
        if lines and lines[0].startswith('#!'):
            insert_index = 1
        if lines and insert_index < len(lines) and lines[insert_index].startswith('# -*-'):
            insert_index += 1

        # Skip docstring
        if lines and insert_index < len(lines) and lines[insert_index].strip().startswith('"""'):
            insert_index += 1
            while insert_index < len(lines) and not lines[insert_index].strip().endswith('"""'):
                insert_index += 1
            insert_index += 1

        # Skip more comments
        while insert_index < len(lines) and lines[insert_index].strip().startswith('#'):
            insert_index += 1

        # Insert imports
        if imports:
            import_lines = [''] + imports + ['']  # Add blank lines around imports
            lines[insert_index:insert_index] = import_lines

        return '\n'.join(lines)

    def fix_file(self, file_path: Path) -> bool:
        """Fix import issues in a single file."""
        try:
            # Skip non-Python files
            if not file_path.suffix == '.py':
                return True

            # Skip backup and cache directories
            if any(part.startswith('.') or part == '__pycache__' for part in file_path.parts):
                return True

            logger.info(f"Processing {file_path}")

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Create backup
            self.backup_file(file_path)

            # Find undefined names
            undefined_names = self.find_undefined_names(content, file_path)

            if not undefined_names:
                logger.info(f"No undefined names found in {file_path}")
                return True

            logger.info(f"Found undefined names in {file_path}: {undefined_names}")

            # Generate import statements
            imports = self.generate_import_statements(undefined_names, file_path)

            if not imports:
                logger.info(f"No imports needed for {file_path}")
                return True

            # Insert imports
            fixed_content = self.insert_imports(content, imports, file_path)

            # Validate syntax
            try:
                ast.parse(fixed_content)
            except SyntaxError as e:
                logger.error(f"Syntax error after import fix in {file_path}: {e}")
                self.failed_files.append(str(file_path))
                return False

            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            logger.info(f"Fixed imports in {file_path}: {imports}")
            self.fixed_files.append(str(file_path))

            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.failed_files.append(str(file_path))
            return False

    def fix_directory(self, directory: Path) -> None:
        """Fix import issues in all Python files in a directory."""
        logger.info(f"Scanning directory: {directory}")

        for file_path in directory.rglob('*.py'):
            self.fix_file(file_path)

        # Also check for specific problematic files mentioned in the error report
        specific_files = [
            'core/api/cache_sync.py',
            'core/api/handlers/__init__.py',
            'core/api/handlers/base_handler.py',
            'core/btc_usdc_trading_integration.py',
            'core/distributed_mathematical_processor.py',
            'core/enhanced_error_recovery_system.py',
            'core/entropy_drift_tracker.py',
            'core/mathlib_v3_visualizer.py',
            'core/profit_backend_dispatcher.py',
            'core/qsc_enhanced_profit_allocator.py',
            'core/strategy_bit_mapper.py',
        ]

        for file_path_str in specific_files:
            file_path = Path(file_path_str)
            if file_path.exists():
                self.fix_file(file_path)

    def generate_report(self) -> None:
        """Generate a report of the fixes applied."""
        report_path = Path('import_fix_report.md')

        with open(report_path, 'w') as f:
            f.write("# Import Fix Report\n\n")
            f.write(f"## Summary\n")
            f.write(f"- Files processed: {len(self.fixed_files) + len(self.failed_files)}\n")
            f.write(f"- Files fixed: {len(self.fixed_files)}\n")
            f.write(f"- Files failed: {len(self.failed_files)}\n")
            f.write(f"- Backup directory: {self.backup_dir}\n\n")

            f.write("## Fixed Files\n")
            for file_path in self.fixed_files:
                f.write(f"- {file_path}\n")

            f.write("\n## Failed Files\n")
            for file_path in self.failed_files:
                f.write(f"- {file_path}\n")

        logger.info(f"Report generated: {report_path}")


def main():
    """Main function to run the import fixer."""
    logger.info("Starting comprehensive import fix")

    # Create fixer instance
    fixer = ImportFixer()

    # Fix core directory
    core_dir = Path('core')
    if core_dir.exists():
        fixer.fix_directory(core_dir)

    # Fix root directory Python files
    root_dir = Path('.')
    for file_path in root_dir.glob('*.py'):
        fixer.fix_file(file_path)

    # Generate report
    fixer.generate_report()

    logger.info("Import fix completed")
    logger.info(f"Fixed {len(fixer.fixed_files)} files")
    logger.info(f"Failed {len(fixer.failed_files)} files")


if __name__ == "__main__":
    main()
