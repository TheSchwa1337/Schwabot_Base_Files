from typing import Any
#!/usr/bin/env python3
"""
Best Practices Enforcer - Centralized Code Quality Management
============================================================

Implements the fault-tolerant patterns we've established:
1. Centralized import resolution (safe_import)
2. Centralized error handling (error_handler)
3. Type annotation enforcement (type_enforcer)
4. Windows CLI compatibility
5. Mathematical function patterns

This ensures all new code follows our established patterns and prevents
the HIGH/MEDIUM error cycles we've eliminated.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CodePattern:
    """Represents a code pattern that should be enforced"""
    name: str
    description: str
    pattern: str
    replacement: str
    severity: str = "MEDIUM"
    category: str = "best_practice"


@dataclass
class EnforcementResult:
    """Result of enforcing best practices on a file"""
    file_path: str
    patterns_applied: List[str] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    success: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


class BestPracticesEnforcer:
    """Centralized enforcer for all established best practices"""

    def __init__(self) -> Any:
        self._patterns: List[CodePattern] = []
        self._register_core_patterns()

    def _register_core_patterns(self) -> None:
        """Register all the core patterns we've established"""

        # Import Resolution Patterns
        self._patterns.extend([
            CodePattern(
                name="scattered_import_error_handling",
                description="Replace scattered try/except ImportError with safe_import",
                pattern=r'try:\s*import\s+(\w+).*?except\s+ImportError:.*?(\w+)\s*=\s*None',
                replacement=r'from core.import_resolver import safe_import\n\1_imports = safe_import("\1", ["\2"])\n\2 = \1_imports["\2"]',
                severity="HIGH",
                category="import_resolution"
            ),
            CodePattern(
                name="bare_except_blocks",
                description="Replace bare except with error_handler.safe_execute",
                pattern=r'try:\s*(.*?)\s*except:',
                replacement=r'from core.error_handler import safe_execute\nresult = safe_execute(lambda: \1, default_return=None)',
                severity="CRITICAL",
                category="error_handling"
            ),
            CodePattern(
                name="missing_type_annotations",
                description="Add type annotations to function parameters",
                pattern=r'def\s+(\w+)\s*\(([^)]*)\)\s*:',
                replacement=r'def \1(\2) -> Any:',
                severity="MEDIUM",
                category="type_annotations"
            ),
            CodePattern(
                name="windows_cli_unsafe_print",
                description="Replace print with Windows CLI-safe version",
                pattern=r'print\s*\(\s*["\']([^"\']*[🔧✅❌🟠🟡🟢📝🎯📊🎉⚠️💡])[^"\']*["\']\s*\)',
                replacement=r'from core.windows_cli_compatibility import safe_print\nsafe_print("\1")',
                severity="MEDIUM",
                category="windows_cli"
            ),
            CodePattern(
                name="mathematical_function_patterns",
                description="Ensure mathematical functions have proper type annotations",
                pattern=r'def\s+(
    calculate|compute|process|analyze|evaluate|estimate|predict|forecast|simulate|optimize|minimize|maximize)\s*\(([^)]*)\)\s*:',
                replacement=r'def \1(\2) -> Union[float, Dict[str, Any]]:',
                severity="MEDIUM",
                category="mathematical_functions"
            ),
        ])

    def enforce_on_file(self, file_path: str) -> EnforcementResult:
        """Enforce all best practices on a single file"""
        result = EnforcementResult(file_path=file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Apply each pattern
            for pattern in self._patterns:
                try:
                    if re.search(pattern.pattern, content, re.DOTALL | re.MULTILINE):
                        content = re.sub(pattern.pattern, pattern.replacement, content, flags=re.DOTALL | re.MULTILINE)
                        result.patterns_applied.append(pattern.name)
                        logger.info(f"Applied {pattern.name} to {file_path}")
                except Exception as e:
                    result.issues_found.append(f"Error applying {pattern.name}: {e}")
                    logger.warning(f"Error applying {pattern.name} to {file_path}: {e}")

            # Write back if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Updated {file_path} with best practices")

            # Validate the file still parses
            try:
                ast.parse(content)
            except SyntaxError as e:
                result.issues_found.append(f"Syntax error after applying patterns: {e}")
                result.success = False
                # Revert changes if syntax is broken
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                logger.error(f"Reverted {file_path} due to syntax error")

        except Exception as e:
            result.issues_found.append(f"File processing error: {e}")
            result.success = False
            logger.error(f"Error processing {file_path}: {e}")

        return result

    def enforce_on_directory(self, directory: str) -> List[EnforcementResult]:
        """Enforce best practices on all Python files in a directory"""
        results = []

        for py_file in Path(directory).rglob('*.py'):
            if py_file.is_file() and not self._should_skip_file(py_file):
                result = self.enforce_on_file(str(py_file))
                results.append(result)

        return results

    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped"""
        skip_patterns = [
            '.venv',
            'site-packages',
            '__pycache__',
            '.git',
            'node_modules',
            'venv',
            'env',
        ]

        return any(pattern in str(file_path) for pattern in skip_patterns)

    def add_custom_pattern(self, pattern: CodePattern) -> None:
        """Add a custom pattern for enforcement"""
        self._patterns.append(pattern)

    def get_patterns_by_category(self, category: str) -> List[CodePattern]:
        """Get patterns by category"""
        return [p for p in self._patterns if p.category == category]

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about patterns"""
        categories = {}
        severities = {}

        for pattern in self._patterns:
            categories[pattern.category] = categories.get(pattern.category, 0) + 1
            severities[pattern.severity] = severities.get(pattern.severity, 0) + 1

        return {
            'total_patterns': len(self._patterns),
            'categories': categories,
            'severities': severities
        }


class PreCommitHook:
    """Pre-commit hook infrastructure for automated best practices enforcement"""

    def __init__(self) -> Any:
        self.enforcer = BestPracticesEnforcer()

    def run_pre_commit_check(self, staged_files: List[str]) -> bool:
        """Run pre-commit checks on staged files"""
        logger.info("Running pre-commit best practices check...")

        all_passed = True
        results = []

        for file_path in staged_files:
            if file_path.endswith('.py'):
                result = self.enforcer.enforce_on_file(file_path)
                results.append(result)

                if not result.success:
                    all_passed = False
                    logger.error(f"Pre-commit check failed for {file_path}")
                    for issue in result.issues_found:
                        logger.error(f"  - {issue}")

        # Generate summary
        total_files = len(results)
        successful_files = sum(1 for r in results if r.success)
        total_patterns_applied = sum(len(r.patterns_applied) for r in results)

        logger.info(f"Pre-commit check summary:")
        logger.info(f"  - Files processed: {total_files}")
        logger.info(f"  - Files passed: {successful_files}")
        logger.info(f"  - Patterns applied: {total_patterns_applied}")

        return all_passed

    def create_git_hook_script(self, output_path: str = ".git/hooks/pre-commit") -> None:
        """Create a git pre-commit hook script"""
        hook_content = '''#!/bin/bash
# Pre-commit hook for best practices enforcement

# Get staged Python files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ -n "$STAGED_FILES" ]; then
    echo "Running best practices enforcement on staged Python files..."
    python -c "
import sys
from core.best_practices_enforcer import PreCommitHook

hook = PreCommitHook()
success = hook.run_pre_commit_check(sys.argv[1:])

if not success:
    print('❌ Pre-commit check failed. Please fix the issues above.')
    sys.exit(1)
else:
    print('✅ Pre-commit check passed.')
" $STAGED_FILES
fi
'''

        # Ensure the hooks directory exists
        hook_path = Path(output_path)
        hook_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the hook script
        with open(hook_path, 'w') as f:
            f.write(hook_content)

        # Make it executable
        hook_path.chmod(0o755)

        logger.info(f"Created pre-commit hook at {output_path}")


# Global instances for easy access
best_practices_enforcer = BestPracticesEnforcer()
pre_commit_hook = PreCommitHook()


def enforce_best_practices_on_file(file_path: str) -> EnforcementResult:
    """Convenience function for enforcing best practices on a file"""
    return best_practices_enforcer.enforce_on_file(file_path)


def enforce_best_practices_on_directory(directory: str) -> List[EnforcementResult]:
    """Convenience function for enforcing best practices on a directory"""
    return best_practices_enforcer.enforce_on_directory(directory)


def setup_pre_commit_hook() -> None:
    """Setup the pre-commit hook for the repository"""
    pre_commit_hook.create_git_hook_script()


def run_pre_commit_check(staged_files: List[str]) -> bool:
    """Run pre-commit check on staged files"""
    return pre_commit_hook.run_pre_commit_check(staged_files)