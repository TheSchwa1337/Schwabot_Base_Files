#!/usr/bin/env python3
"""
Systematic Codebase Fix Script
==============================

This script identifies and fixes systematic syntax issues across the Schwabot codebase:

1. Broken dictionary/list definitions
2. Missing typing imports
3. Malformed function calls
4. Incorrect indentation patterns
"""

import re
from pathlib import Path
from typing import Dict, List, Set


class CodebaseFixer:
    """Systematic codebase syntax fixer."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.fixed_files: Set[str] = set()
        self.error_patterns = {}
            "broken_dict": r"(\w+)\s*=\s*\{\}\s*\n\s*([^}]*\})",
            "broken_list": r"(\w+)\s*=\s*\[\]\s*\n\s*([^\]]*\])",
            "missing_typing": r"from typing import (?!.*List)",
            "broken_function_call": r"(\w+)\s*\(\s*\)\s*\n\s*([^)]*\))",
        }

    def scan_codebase(self) -> Dict[str, List[str]]:
        """Scan the codebase for Python files and identify issues."""
        issues = {}
            "broken_dicts": [],
            "broken_lists": [],
            "missing_typing": [],
            "broken_calls": [],
            "syntax_errors": [],
        }
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                file_issues = self._analyze_file(content, str(file_path))
                for issue_type, file_list in file_issues.items():
                    if file_list:
                        issues[issue_type].extend(file_list)

            except Exception as e:
                issues["syntax_errors"].append(f"{file_path}: {e}")

        return issues

    def _analyze_file(self, content: str, filepath: str) -> Dict[str, List[str]]:
        """Analyze a single file for issues."""
        issues = {}
            "broken_dicts": [],
            "broken_lists": [],
            "missing_typing": [],
            "broken_calls": [],
        }
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line_num = i + 1

            # Check for broken dictionary definitions
            if re.match(r"^\s*\w+\s*=\s*\{\}\s*$", line):
                # Look ahead for indented key-value pairs
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j]
                    if re.match(r'^\s+["\']\w+["\']\s*:', next_line):
                        issues["broken_dicts"].append(f"{filepath}:{line_num}")
                        break

            # Check for broken list definitions
            if re.match(r"^\s*\w+\s*=\s*\[\]\s*$", line):
                # Look ahead for indented items
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j]
                    if re.match(r'^\s+["\']\w+["\']', next_line):
                        issues["broken_lists"].append(f"{filepath}:{line_num}")
                        break

            # Check for missing typing imports
            if "from typing import" in line and "List" not in line:
                if any("List" in l for l in lines[i : i + 5]):
                    issues["missing_typing"].append(f"{filepath}:{line_num}")

            # Check for broken function calls
            if re.match(r"^\s*\w+\(\s*\)\s*$", line):
                # Look ahead for parameters
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j]
                    if re.match(r"^\s+\w+\s*=", next_line):
                        issues["broken_calls"].append(f"{filepath}:{line_num}")
                        break

        return issues

    def fix_file(self, filepath: str) -> bool:
        """Fix syntax issues in a single file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Fix 1: Broken dictionary definitions
            content = self._fix_broken_dicts(content)

            # Fix 2: Broken list definitions
            content = self._fix_broken_lists(content)

            # Fix 3: Missing typing imports
            content = self._fix_missing_typing(content)

            # Fix 4: Broken function calls
            content = self._fix_broken_calls(content)

            # Fix 5: Fix indentation issues
            content = self._fix_indentation(content)

            if content != original_content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                self.fixed_files.add(filepath)
                return True

            return False

        except Exception as e:
            print(f"Error fixing {filepath}: {e}")
            return False

    def _fix_broken_dicts(self, content: str) -> str:
        """Fix broken dictionary definitions."""
        lines = content.split("\n")
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this is a broken dict definition
            if re.match(r"^\s*\w+\s*=\s*\{\}\s*$", line):
                var_name = re.match(r"^\s*(\w+)\s*=\s*\{\}\s*$", line).group(1)

                # Collect indented key-value pairs
                dict_items = []
                j = i + 1
                while j < len(lines) and re.match(r'^\s+["\']\w+["\']\s*:', lines[j]):
                    dict_items.append(lines[j].strip())
                    j += 1

                if dict_items:
                    # Fix the dictionary definition
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(" " * indent + f"{var_name} = {{")}}

                    for item in dict_items:
                        fixed_lines.append(" " * (indent + 4) + item)

                    fixed_lines.append(" " * indent + "}")
                    i = j
                else:
                    fixed_lines.append(line)
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1

        return "\n".join(fixed_lines)

    def _fix_broken_lists(self, content: str) -> str:
        """Fix broken list definitions."""
        lines = content.split("\n")
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this is a broken list definition
            if re.match(r"^\s*\w+\s*=\s*\[\]\s*$", line):
                var_name = re.match(r"^\s*(\w+)\s*=\s*\[\]\s*$", line).group(1)

                # Collect indented items
                list_items = []
                j = i + 1
                while j < len(lines) and re.match(r'^\s+["\']\w+["\']', lines[j]):
                    list_items.append(lines[j].strip())
                    j += 1

                if list_items:
                    # Fix the list definition
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(" " * indent + f"{var_name} = [")]

                    for item in list_items:
                        fixed_lines.append(" " * (indent + 4) + item)

                    fixed_lines.append(" " * indent + "]")
                    i = j
                else:
                    fixed_lines.append(line)
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1

        return "\n".join(fixed_lines)

    def _fix_missing_typing(self, content: str) -> str:
        """Fix missing typing imports."""
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            if "from typing import" in line and "List" not in line:
                # Check if List is used in the file
                if "List" in content:
                    # Add List to the import
                    if line.strip().endswith(","):
                        fixed_lines.append(line + " List")
                    else:
                        fixed_lines.append(line + ", List")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def _fix_broken_calls(self, content: str) -> str:
        """Fix broken function calls."""
        lines = content.split("\n")
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this is a broken function call
            if re.match(r"^\s*\w+\(\s*\)\s*$", line):
                func_name = re.match(r"^\s*(\w+)\(\s*\)\s*$", line).group(1)

                # Collect parameters
                params = []
                j = i + 1
                while j < len(lines) and re.match(r"^\s+\w+\s*=", lines[j]):
                    params.append(lines[j].strip())
                    j += 1

                if params:
                    # Fix the function call
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(" " * indent + f"{func_name}("))

                    for param in params:
                        fixed_lines.append(" " * (indent + 4) + param)

                    fixed_lines.append(" " * indent + ")")
                    i = j
                else:
                    fixed_lines.append(line)
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1

        return "\n".join(fixed_lines)

    def _fix_indentation(self, content: str) -> str:
        """Fix general indentation issues."""
        # Fix common indentation problems
        content = re.sub(r"^\s*}\s*$", "}", content, flags=re.MULTILINE)
        content = re.sub(r"^\s*\]\s*$", "]", content, flags=re.MULTILINE)

        return content

    def fix_codebase(self) -> Dict[str, int]:
        """Fix all issues across the codebase."""
        print("🔍 Scanning codebase for issues...")
        issues = self.scan_codebase()

        print("\n📊 Issues found:")
        for issue_type, file_list in issues.items():
            print(f"  {issue_type}: {len(file_list)}")

        print("\n🔧 Fixing files...")

        # Get all Python files
        python_files = list(self.project_root.rglob("*.py"))

        fixed_count = 0
        for file_path in python_files:
            if self.fix_file(str(file_path)):
                fixed_count += 1
                print(f"  ✅ Fixed: {file_path}")

        print(f"\n✅ Fixed {fixed_count} files")
        return {"files_fixed": fixed_count, "total_files": len(python_files)}


def main():
    """Main execution function."""
    fixer = CodebaseFixer()

    print("🚀 Schwabot Codebase Fixer")
    print("=" * 50)

    # Fix the codebase
    results = fixer.fix_codebase()

    print("\n📈 Summary:")
    print(f"  Files fixed: {results['files_fixed']}")
    print(f"  Total files: {results['total_files']}")

    if results["files_fixed"] > 0:
        print("\n🎉 Codebase fixes completed successfully!")
    else:
        print("\nℹ️  No files needed fixing.")


if __name__ == "__main__":
    main()
