#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Flake8 Error Fix Script

Fixes remaining Flake8 errors in the PTNS codebase:
- Indentation errors (E999)
- Import organization (E402)
- Unused imports (F401)
- Comparison issues (E712)
- Whitespace issues (E226, E261)
- Missing newlines (W292)
"""

import os
import re
import ast
import tokenize
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple
import autopep8


class Flake8ErrorFixer:
    """Comprehensive Flake8 error fixing system."""

    def __init__(self, root_dir: str = "."):
        """Initialize the error fixer."""
        self.root_dir = Path(root_dir)
        self.fixed_files = 0
        self.total_errors_fixed = 0
        self.error_types_fixed = {
            'E999': 0,  # Indentation errors
            'E402': 0,  # Import not at top
            'F401': 0,  # Unused imports
            'E712': 0,  # Comparison to True
            'E226': 0,  # Missing whitespace around operator
            'E261': 0,  # At least two spaces before inline comment
            'W292': 0,  # No newline at end of file
        }

    def fix_all_errors(self) -> Dict[str, Any]:
        """Fix all Flake8 errors in the codebase."""
        print("🔧 Starting comprehensive Flake8 error fixing...")
        
        # Find all Python files
        python_files = list(self.root_dir.rglob("*.py"))
        print(f"📁 Found {len(python_files)} Python files to process")
        
        for file_path in python_files:
            try:
                self._fix_file_errors(file_path)
            except Exception as e:
                print(f"❌ Error fixing {file_path}: {e}")
        
        return self._generate_report()

    def _fix_file_errors(self, file_path: Path):
        """Fix all errors in a single file."""
        if not file_path.exists():
            return
        
        original_content = file_path.read_text(encoding='utf-8')
        fixed_content = original_content
        
        # Fix indentation errors first (most critical)
        fixed_content = self._fix_indentation_errors(fixed_content)
        
        # Fix import organization
        fixed_content = self._fix_import_organization(fixed_content)
        
        # Fix unused imports
        fixed_content = self._fix_unused_imports(fixed_content)
        
        # Fix comparison issues
        fixed_content = self._fix_comparison_issues(fixed_content)
        
        # Fix whitespace issues
        fixed_content = self._fix_whitespace_issues(fixed_content)
        
        # Fix missing newlines
        fixed_content = self._fix_missing_newlines(fixed_content)
        
        # Apply autopep8 for final formatting
        fixed_content = autopep8.fix_code(
            fixed_content,
            options={
                'aggressive': 1,
                'max_line_length': 120,
                'ignore': ['E203', 'W503']
            }
        )
        
        # Write back if changes were made
        if fixed_content != original_content:
            file_path.write_text(fixed_content, encoding='utf-8')
            self.fixed_files += 1
            print(f"✅ Fixed errors in {file_path}")

    def _fix_indentation_errors(self, content: str) -> str:
        """Fix indentation errors by standardizing to 4 spaces."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                fixed_lines.append('')
                continue
            
            # Count leading spaces/tabs
            leading_chars = len(line) - len(line.lstrip())
            
            if leading_chars > 0:
                # Convert tabs to spaces and standardize indentation
                indent_level = leading_chars // 4
                if leading_chars % 4 != 0:
                    indent_level += 1
                
                fixed_line = '    ' * indent_level + line.lstrip()
            else:
                fixed_line = line
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)

    def _fix_import_organization(self, content: str) -> str:
        """Move imports to top of file and organize them."""
        lines = content.split('\n')
        
        # Find all import lines
        import_lines = []
        non_import_lines = []
        in_import_section = True
        
        for line in lines:
            stripped = line.strip()
            
            # Check if this is an import line
            is_import = (
                stripped.startswith('import ') or
                stripped.startswith('from ') or
                stripped.startswith('# Import') or
                stripped.startswith('# -*- coding:') or
                stripped.startswith('"""') or
                stripped.startswith("'''") or
                stripped.startswith('from __future__')
            )
            
            # Check if we're still in the import section
            if in_import_section and not is_import and stripped and not stripped.startswith('#'):
                in_import_section = False
            
            if in_import_section and is_import:
                import_lines.append(line)
            else:
                non_import_lines.append(line)
        
        # Reorganize imports
        organized_imports = self._organize_imports(import_lines)
        
        # Reconstruct file
        result_lines = []
        
        # Add file header if present
        header_lines = []
        for line in import_lines:
            if line.strip().startswith('# -*-') or line.strip().startswith('"""') or line.strip().startswith("'''"):
                header_lines.append(line)
        
        result_lines.extend(header_lines)
        
        # Add organized imports
        result_lines.extend(organized_imports)
        
        # Add rest of content
        result_lines.extend(non_import_lines)
        
        return '\n'.join(result_lines)

    def _organize_imports(self, import_lines: List[str]) -> List[str]:
        """Organize imports by type and alphabetical order."""
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        future_imports = []
        
        for line in import_lines:
            stripped = line.strip()
            
            if stripped.startswith('from __future__'):
                future_imports.append(line)
            elif stripped.startswith('import ') or stripped.startswith('from '):
                # Determine import type
                if 'core.' in stripped or 'utils.' in stripped or 'tests.' in stripped:
                    local_imports.append(line)
                elif any(pkg in stripped for pkg in ['numpy', 'pandas', 'matplotlib', 'psutil', 'queue', 'threading']):
                    third_party_imports.append(line)
                else:
                    stdlib_imports.append(line)
            else:
                # Comments or other lines
                stdlib_imports.append(line)
        
        # Sort each category
        future_imports.sort()
        stdlib_imports.sort()
        third_party_imports.sort()
        local_imports.sort()
        
        # Combine with separators
        result = []
        if future_imports:
            result.extend(future_imports)
            result.append('')
        
        if stdlib_imports:
            result.extend(stdlib_imports)
            result.append('')
        
        if third_party_imports:
            result.extend(third_party_imports)
            result.append('')
        
        if local_imports:
            result.extend(local_imports)
            result.append('')
        
        return result

    def _fix_unused_imports(self, content: str) -> str:
        """Remove unused imports using AST analysis."""
        try:
            tree = ast.parse(content)
            used_names = set()
            
            # Find all used names
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)
            
            # Parse imports to find unused ones
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                stripped = line.strip()
                
                if stripped.startswith('import ') or stripped.startswith('from '):
                    # Check if this import is used
                    import_names = self._extract_import_names(stripped)
                    if any(name in used_names for name in import_names):
                        fixed_lines.append(line)
                    else:
                        # Comment out unused import
                        fixed_lines.append(f"# {line}  # FIXME: Unused import")
                        self.error_types_fixed['F401'] += 1
                else:
                    fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
            
        except SyntaxError:
            # If AST parsing fails, return original content
            return content

    def _extract_import_names(self, import_line: str) -> List[str]:
        """Extract imported names from import line."""
        names = []
        
        # Handle 'from x import y' format
        if import_line.startswith('from '):
            match = re.search(r'from \S+ import (.+)', import_line)
            if match:
                import_part = match.group(1)
                # Handle 'import a, b, c' format
                for name in import_part.split(','):
                    name = name.strip()
                    if ' as ' in name:
                        name = name.split(' as ')[1].strip()
                    names.append(name)
        
        # Handle 'import x' format
        elif import_line.startswith('import '):
            match = re.search(r'import (.+)', import_line)
            if match:
                import_part = match.group(1)
                for name in import_part.split(','):
                    name = name.strip()
                    if ' as ' in name:
                        name = name.split(' as ')[1].strip()
                    names.append(name)
        
        return names

    def _fix_comparison_issues(self, content: str) -> str:
        """Fix comparison to True/False issues."""
        # Fix == True comparisons
        content = re.sub(r'\b==\s*True\b', '', content)
        content = re.sub(r'\b==\s*False\b', '', content)
        
        # Fix != True comparisons
        content = re.sub(r'\b!=\s*True\b', '', content)
        content = re.sub(r'\b!=\s*False\b', '', content)
        
        return content

    def _fix_whitespace_issues(self, content: str) -> str:
        """Fix whitespace around operators and inline comments."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix missing whitespace around operators
            line = re.sub(r'(\w)([+\-*/=<>!&|])(\w)', r'\1 \2 \3', line)
            
            # Fix inline comment spacing
            if '#' in line:
                parts = line.split('#', 1)
                if len(parts) == 2:
                    code_part = parts[0].rstrip()
                    comment_part = parts[1]
                    
                    # Ensure at least two spaces before comment
                    if code_part and not code_part.endswith('  '):
                        code_part += '  '
                    
                    line = code_part + '#' + comment_part
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def _fix_missing_newlines(self, content: str) -> str:
        """Ensure file ends with newline."""
        if content and not content.endswith('\n'):
            content += '\n'
            self.error_types_fixed['W292'] += 1
        
        return content

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive fix report."""
        total_fixed = sum(self.error_types_fixed.values())
        
        report = {
            'status': 'completed',
            'files_fixed': self.fixed_files,
            'total_errors_fixed': total_fixed,
            'error_breakdown': self.error_types_fixed.copy(),
            'summary': {
                'E999_indentation': self.error_types_fixed['E999'],
                'E402_imports': self.error_types_fixed['E402'],
                'F401_unused': self.error_types_fixed['F401'],
                'E712_comparisons': self.error_types_fixed['E712'],
                'E226_whitespace': self.error_types_fixed['E226'],
                'E261_comments': self.error_types_fixed['E261'],
                'W292_newlines': self.error_types_fixed['W292']
            }
        }
        
        return report


def main():
    """Main execution function."""
    print("🚀 PTNS Flake8 Error Fixer")
    print("=" * 50)
    
    fixer = Flake8ErrorFixer()
    report = fixer.fix_all_errors()
    
    print("\n" + "=" * 50)
    print("📊 Fix Report")
    print("=" * 50)
    print(f"✅ Files Fixed: {report['files_fixed']}")
    print(f"🔧 Total Errors Fixed: {report['total_errors_fixed']}")
    print(f"📈 Error Breakdown:")
    
    for error_type, count in report['error_breakdown'].items():
        if count > 0:
            print(f"  - {error_type}: {count}")
    
    print("\n🎯 Next Steps:")
    print("1. Run flake8 again to verify fixes")
    print("2. Test system functionality")
    print("3. Commit changes if all tests pass")
    
    return report


if __name__ == "__main__":
    main() 