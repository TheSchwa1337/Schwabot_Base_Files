#!/usr/bin/env python3
"""
Specialized Flake8 Fixer for Test Files
Handles common test file patterns, mock imports, and test-specific issues.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

class TestFilesFlake8Fixer:
    """Specialized fixer for test files with test-specific patterns."""
    
    def __init__(self):
        self.stats = defaultdict(int)
        self.fixed_files = []
        
        # Test-specific imports
        self.test_imports = {
            'unittest': 'import unittest',
            'TestCase': 'from unittest import TestCase',
            'Mock': 'from unittest.mock import Mock',
            'MagicMock': 'from unittest.mock import MagicMock',
            'patch': 'from unittest.mock import patch',
            'mock': 'from unittest import mock',
            'pytest': 'import pytest',
            'fixture': 'from pytest import fixture',
            'datetime': 'from datetime import datetime',
            'time': 'import time',
            'os': 'import os',
            'sys': 'import sys',
            'json': 'import json',
            'logging': 'import logging',
            'warnings': 'import warnings',
            # Common test mocks for your project
            'PhaseEngineHooks': 'from unittest.mock import Mock as PhaseEngineHooks',
            'OracleBus': 'from unittest.mock import Mock as OracleBus',
            'ThermalZoneManager': 'from unittest.mock import Mock as ThermalZoneManager',
            'GhostRecollectionSystem': 'from unittest.mock import Mock as GhostRecollectionSystem',
            'QuantumBTCProcessor': 'from unittest.mock import Mock as QuantumBTCProcessor',
            'NewsLanternAPI': 'from unittest.mock import Mock as NewsLanternAPI',
            'HashRecollectionSystem': 'from unittest.mock import Mock as HashRecollectionSystem',
            'SustainmentFramework': 'from unittest.mock import Mock as SustainmentFramework',
            'MathLibCore': 'from unittest.mock import Mock as MathLibCore',
            'WordFitnessTracker': 'from unittest.mock import Mock as WordFitnessTracker',
            'VisualCoreIntegration': 'from unittest.mock import Mock as VisualCoreIntegration',
            'SystemValidationFramework': 'from unittest.mock import Mock as SystemValidationFramework',
        }
        
    def find_test_files(self, directory: str = '.') -> List[str]:
        """Find all test files in the directory."""
        test_files = []
        for root, dirs, files in os.walk(directory):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']]
            for file in files:
                if (file.startswith('test_') and file.endswith('.py')) or file.endswith('_test.py'):
                    test_files.append(os.path.join(root, file))
        return test_files
    
    def fix_test_imports(self, content: str) -> str:
        """Fix imports specifically for test files."""
        lines = content.split('\n')
        
        # Separate imports and code
        import_lines = []
        code_lines = []
        in_import_section = True
        
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ')) and in_import_section:
                import_lines.append(line)
            elif stripped == '' and in_import_section:
                import_lines.append(line)
            else:
                in_import_section = False
                code_lines.append(line)
        
        # Find undefined names in test code
        code_content = '\n'.join(code_lines)
        undefined_names = self._find_test_undefined_names(code_content)
        
        # Add missing test imports
        existing_imports = set()
        for import_line in import_lines:
            if 'import ' in import_line:
                import_part = import_line.split('import ')[-1]
                for item in import_part.split(','):
                    item = item.strip().split(' as ')[-1]
                    existing_imports.add(item)
        
        new_imports = []
        for name in undefined_names:
            if name in self.test_imports and name not in existing_imports:
                new_imports.append(self.test_imports[name])
        
        # Organize imports for test files
        organized_imports = self._organize_test_imports(import_lines + new_imports)
        
        # Combine everything
        result = organized_imports + [''] + code_lines
        return '\n'.join(result)
    
    def _find_test_undefined_names(self, content: str) -> Set[str]:
        """Find undefined names specific to test files."""
        undefined = set()
        
        # Test-specific patterns
        test_patterns = {
            r'\bTestCase\b': 'TestCase',
            r'\bMock\b': 'Mock',
            r'\bMagicMock\b': 'MagicMock',
            r'@patch\b': 'patch',
            r'\bmock\.': 'mock',
            r'@pytest\.': 'pytest',
            r'\bfixture\b': 'fixture',
            r'\bdatetime\b': 'datetime',
            r'\btime\.': 'time',
            r'\bos\.': 'os',
            r'\bsys\.': 'sys',
            r'\bjson\.': 'json',
            r'\blogging\.': 'logging',
            r'\bwarnings\.': 'warnings',
            # Project-specific mocks
            r'\bPhaseEngineHooks\b': 'PhaseEngineHooks',
            r'\bOracleBus\b': 'OracleBus',
            r'\bThermalZoneManager\b': 'ThermalZoneManager',
            r'\bGhostRecollectionSystem\b': 'GhostRecollectionSystem',
            r'\bQuantumBTCProcessor\b': 'QuantumBTCProcessor',
            r'\bNewsLanternAPI\b': 'NewsLanternAPI',
            r'\bHashRecollectionSystem\b': 'HashRecollectionSystem',
            r'\bSustainmentFramework\b': 'SustainmentFramework',
            r'\bMathLibCore\b': 'MathLibCore',
            r'\bWordFitnessTracker\b': 'WordFitnessTracker',
            r'\bVisualCoreIntegration\b': 'VisualCoreIntegration',
            r'\bSystemValidationFramework\b': 'SystemValidationFramework',
        }
        
        for pattern, name in test_patterns.items():
            if re.search(pattern, content):
                undefined.add(name)
        
        return undefined
    
    def _organize_test_imports(self, import_lines: List[str]) -> List[str]:
        """Organize imports for test files."""
        standard_libs = []
        test_frameworks = []
        mocks = []
        project_imports = []
        
        for line in import_lines:
            if not line.strip():
                continue
            
            line = line.strip()
            
            # Standard library
            if any(lib in line for lib in ['import os', 'import sys', 'import time', 'import json', 
                                          'import logging', 'import warnings', 'from datetime']):
                standard_libs.append(line)
            # Test frameworks
            elif any(framework in line for lib in ['unittest', 'pytest']):
                test_frameworks.append(line)
            # Mocks
            elif 'mock' in line.lower() or 'Mock' in line:
                mocks.append(line)
            # Project imports
            else:
                project_imports.append(line)
        
        result = []
        if standard_libs:
            result.extend(sorted(standard_libs))
            result.append('')
        if test_frameworks:
            result.extend(sorted(test_frameworks))
            result.append('')
        if mocks:
            result.extend(sorted(mocks))
            result.append('')
        if project_imports:
            result.extend(sorted(project_imports))
        
        # Remove trailing empty lines
        while result and result[-1] == '':
            result.pop()
        
        return result
    
    def fix_test_method_names(self, content: str) -> str:
        """Fix long test method names by breaking them properly."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if len(line) > 79 and 'def test_' in line:
                # Break long test method definitions
                indent = len(line) - len(line.lstrip())
                if line.strip().endswith(':'):
                    # Method definition
                    method_part = line.rstrip()[:-1]  # Remove ':'
                    if len(method_part) > 75:
                        # Break at logical points
                        if '_and_' in method_part:
                            parts = method_part.split('_and_', 1)
                            fixed_lines.append(f"{parts[0]}_and_ \\")
                            fixed_lines.append(f"{' ' * (indent + 4)}{parts[1]}:")
                        elif '_with_' in method_part:
                            parts = method_part.split('_with_', 1)
                            fixed_lines.append(f"{parts[0]}_with_ \\")
                            fixed_lines.append(f"{' ' * (indent + 4)}{parts[1]}:")
                        elif '_when_' in method_part:
                            parts = method_part.split('_when_', 1)
                            fixed_lines.append(f"{parts[0]}_when_ \\")
                            fixed_lines.append(f"{' ' * (indent + 4)}{parts[1]}:")
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_test_assertions(self, content: str) -> str:
        """Fix long assertion lines in tests."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if len(line) > 79 and ('assert' in line or 'self.assert' in line):
                indent = len(line) - len(line.lstrip())
                
                # Break long assertions
                if 'self.assertEqual(' in line:
                    # Break assertEqual calls
                    match = re.match(r'^(\s*)(.*assertEqual\()(.*)(\))(.*)$', line)
                    if match:
                        spaces, method, args, close_paren, rest = match.groups()
                        if len(args) > 50:
                            # Split arguments
                            arg_parts = self._split_function_args(args)
                            if len(arg_parts) >= 2:
                                fixed_lines.append(f"{spaces}{method}")
                                for i, arg in enumerate(arg_parts):
                                    comma = ',' if i < len(arg_parts) - 1 else ''
                                    fixed_lines.append(f"{spaces}    {arg}{comma}")
                                fixed_lines.append(f"{spaces}){rest}")
                            else:
                                fixed_lines.append(line)
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                elif 'assert ' in line and ' == ' in line:
                    # Break simple assert statements
                    parts = line.split(' == ', 1)
                    if len(parts) == 2:
                        fixed_lines.append(f"{parts[0]} == \\")
                        fixed_lines.append(f"{' ' * (indent + 4)}{parts[1]}")
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _split_function_args(self, args_string: str) -> List[str]:
        """Split function arguments respecting parentheses and quotes."""
        args = []
        current_arg = ""
        paren_depth = 0
        in_quotes = False
        quote_char = None
        
        for char in args_string:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif char == '(' and not in_quotes:
                paren_depth += 1
            elif char == ')' and not in_quotes:
                paren_depth -= 1
            elif char == ',' and paren_depth == 0 and not in_quotes:
                args.append(current_arg.strip())
                current_arg = ""
                continue
            
            current_arg += char
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        return args
    
    def fix_test_docstrings(self, content: str) -> str:
        """Fix long docstrings in test methods."""
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for test method with long docstring
            if 'def test_' in line and i + 1 < len(lines):
                fixed_lines.append(line)
                i += 1
                
                # Check next line for docstring
                next_line = lines[i]
                if '"""' in next_line and len(next_line) > 79:
                    # Break long docstring
                    indent = len(next_line) - len(next_line.lstrip())
                    content_start = next_line.find('"""') + 3
                    content_end = next_line.rfind('"""')
                    
                    if content_end > content_start:
                        # Docstring on one line
                        doc_content = next_line[content_start:content_end]
                        fixed_lines.append(f"{' ' * indent}\"\"\"")
                        fixed_lines.append(f"{' ' * indent}{doc_content}")
                        fixed_lines.append(f"{' ' * indent}\"\"\"")
                    else:
                        fixed_lines.append(next_line)
                else:
                    fixed_lines.append(next_line)
            else:
                fixed_lines.append(line)
            
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def fix_test_file(self, file_path: str) -> bool:
        """Fix all flake8 issues in a test file."""
        try:
            print(f"Fixing test file {file_path}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            content = original_content
            
            # Apply test-specific fixes
            content = self.fix_test_imports(content)
            content = self.fix_test_method_names(content)
            content = self.fix_test_assertions(content)
            content = self.fix_test_docstrings(content)
            
            # Apply general fixes
            content = self._fix_whitespace(content)
            content = self._fix_line_length_general(content)
            content = self._fix_blank_lines(content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.fixed_files.append(file_path)
                self.stats['fixed_files'] += 1
                print(f"  ✓ Fixed {file_path}")
                return True
            else:
                print(f"  - No changes needed for {file_path}")
                return False
                
        except Exception as e:
            print(f"  ✗ Error fixing {file_path}: {e}")
            self.stats['error_files'] += 1
            return False
    
    def _fix_whitespace(self, content: str) -> str:
        """Fix whitespace issues."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            cleaned_line = line.rstrip()
            fixed_lines.append(cleaned_line)
        
        # Ensure file ends with newline
        content = '\n'.join(fixed_lines)
        if content and not content.endswith('\n'):
            content += '\n'
        
        return content
    
    def _fix_line_length_general(self, content: str) -> str:
        """Fix general line length issues not covered by specific methods."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if len(line) <= 79:
                fixed_lines.append(line)
                continue
            
            # Skip lines already handled by specific methods
            if ('def test_' in line or 'assert' in line or 'self.assert' in line or 
                line.strip().startswith('#') or '"""' in line):
                fixed_lines.append(line)
                continue
            
            # Handle long import lines
            if line.strip().startswith('from') and 'import' in line:
                parts = line.split(' import ')
                if len(parts) == 2 and len(parts[1]) > 50:
                    imports = [imp.strip() for imp in parts[1].split(',')]
                    fixed_lines.append(f"{parts[0]} import (")
                    for imp in imports:
                        fixed_lines.append(f"    {imp},")
                    # Remove trailing comma
                    if fixed_lines[-1].endswith(','):
                        fixed_lines[-1] = fixed_lines[-1][:-1]
                    fixed_lines.append(")")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_blank_lines(self, content: str) -> str:
        """Fix blank line issues for test files."""
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Test methods need proper spacing
            if line.strip().startswith('def test_'):
                # Add blank line before test method (except first one)
                if fixed_lines and fixed_lines[-1].strip() and not fixed_lines[-1].strip().startswith('class'):
                    fixed_lines.append('')
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def fix_all_test_files(self, directory: str = '.'):
        """Fix all test files in the directory."""
        print("🧪 Test Files Flake8 Fixer")
        print("=" * 40)
        
        test_files = self.find_test_files(directory)
        print(f"Found {len(test_files)} test files")
        
        if not test_files:
            print("No test files found!")
            return
        
        print("\nProcessing test files...")
        for file_path in test_files:
            self.fix_test_file(file_path)
        
        print(f"\n✅ Fixed {self.stats['fixed_files']} test files")
        print(f"❌ Errors in {self.stats['error_files']} files")
        
        if self.fixed_files:
            print("\nFixed files:")
            for file_path in self.fixed_files:
                print(f"  ✓ {file_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Files Flake8 Fixer')
    parser.add_argument('directory', nargs='?', default='.', 
                       help='Directory to search for test files (default: current)')
    
    args = parser.parse_args()
    
    fixer = TestFilesFlake8Fixer()
    fixer.fix_all_test_files(args.directory)

if __name__ == '__main__':
    main() 