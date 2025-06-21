#!/usr/bin/env python3
"""Selective Syntax Fixer - Target Critical Files First.

This script systematically fixes E999 syntax errors in critical files
using the established patterns, prioritizing core functionality.
"""

import os
import re
from pathlib import Path


class SelectiveSyntaxFixer:
    """Selective syntax error fixer for critical files."""
    
    def __init__(self):
        self.fix_stats = {
            'files_processed': 0,
            'errors_fixed': 0,
            'unicode_fixes': 0,
            'docstring_fixes': 0,
            'syntax_fixes': 0
        }
        
        # Priority files to fix first (core functionality)
        self.priority_files = [
            'core/advanced_mathematical_core.py',
            'core/filters.py',
            'core/flux_compensator.py',
            'core/ghost_phase_integrator.py',
            'core/ghost_pipeline.py',
            'core/ghost_profit_tracker.py',
            'core/ghost_memory.py',
            'core/ghost_memory_router.py',
            'core/ghost_meta_layer_engine.py',
            'core/ghost_news_glyph_map.py',
            'core/ghost_news_vectorizer.py',
            'core/ghost_decay.py',
            'core/ghost_hash_decoder.py',
            'core/integration_orchestrator.py',
            'core/klein_bottle_integrator.py',
            'core/lantern/lexicon_engine.py',
            'core/lantern/profit_story_engine.py',
            'core/lantern/story_parser.py',
            'core/lantern/word_fitness_tracker.py',
            'core/matrix/operations.py',
            'core/matrix/transformations.py',
            'core/phantom/memory_state.py',
            'core/phantom/state_manager.py',
            'core/profit/calculator.py',
            'core/profit/optimizer.py',
            'core/glyph/processor.py',
            'core/glyph/pattern_recognition.py',
        ]
    
    def fix_stub_docstrings(self, content: str) -> str:
        """Fix malformed stub docstrings."""
        # Fix the specific pattern: """Stub main function."""."""
        if '"""Stub main function."""."""' in content:
            content = content.replace(
                '"""Stub main function."""."""',
                '"""Stub main function."""\n    pass\n'
            )
            self.fix_stats['docstring_fixes'] += 1
        
        # Fix other malformed patterns
        content = re.sub(
            r'"""([^"]*)\."""\."""',
            r'"""\1."""',
            content
        )
        
        return content
    
    def fix_unicode_characters(self, content: str) -> str:
        """Replace Unicode characters with ASCII equivalents."""
        unicode_replacements = {
            '∇': 'del',  # nabla
            '∈': 'in',   # element of
            '≤': '<=',   # less than or equal
            '≥': '>=',   # greater than or equal
            '⇒': '=>',   # implies
            '∫': 'int',  # integral
            '∂': 'd',    # partial derivative
            '·': '.',    # middle dot
            '–': '-',    # en dash
            '₍': '(',    # subscript left parenthesis
            '₎': ')',    # subscript right parenthesis
            '♦': '',     # diamond (remove)
            '×': 'x',    # multiplication
            'Δ': 'd',    # delta
            'Σ': 'sum',  # sigma
            'π': 'pi',   # pi
            'σ': 'sigma', # sigma
            'λ': 'lambda', # lambda
            'μ': 'mu',   # mu
            'α': 'alpha', # alpha
            'β': 'beta', # beta
            'γ': 'gamma', # gamma
            'δ': 'delta', # delta
            'ε': 'epsilon', # epsilon
            'θ': 'theta', # theta
            'φ': 'phi',  # phi
            'ψ': 'psi',  # psi
            'ω': 'omega', # omega
        }
        
        for unicode_char, ascii_replacement in unicode_replacements.items():
            if unicode_char in content:
                content = content.replace(unicode_char, ascii_replacement)
                self.fix_stats['unicode_fixes'] += 1
        
        return content
    
    def fix_unterminated_strings(self, content: str) -> str:
        """Fix unterminated triple-quoted strings."""
        # Fix pattern: """text without closing
        content = re.sub(
            r'"""([^"]*)\n\s*"""\s*def\s+',
            r'"""\1"""\n\ndef ',
            content
        )
        
        # Fix pattern: """text at end of line
        content = re.sub(
            r'"""([^"]*)\n\s*def\s+',
            r'"""\1"""\n\ndef ',
            content
        )
        
        # Fix pattern: """text without closing at end
        content = re.sub(
            r'"""([^"]*)\n\s*if\s+__name__',
            r'"""\1"""\n\nif __name__',
            content
        )
        
        # Fix pattern: """text without closing at end
        content = re.sub(
            r'"""([^"]*)\n\s*"""\s*"""',
            r'"""\1"""\n',
            content
        )
        
        return content
    
    def fix_invalid_syntax(self, content: str) -> str:
        """Fix invalid syntax patterns."""
        # Fix stray periods after function definitions
        content = re.sub(
            r'def\s+(\w+)\s*\([^)]*\)\s*:\s*\.',
            r'def \1(\2):',
            content
        )
        
        # Fix invalid decimal literals
        content = re.sub(
            r'(\d+)\.(\d+)\.(\d+)',
            r'\1.\2_\3',  # Replace with underscore
            content
        )
        
        # Fix unterminated string literals
        content = re.sub(
            r'(["\'])([^"\']*)\n',
            r'\1\2\1\n',
            content
        )
        
        # Fix malformed function definitions
        content = re.sub(
            r'def\s+(\w+)\s*\([^)]*\)\s*:\s*"""([^"]*)"""\s*"""',
            r'def \1(\2):\n    """\3"""',
            content
        )
        
        return content
    
    def fix_file(self, file_path: str) -> bool:
        """Fix all syntax errors in a single file."""
        try:
            if not os.path.exists(file_path):
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply all fixes
            content = self.fix_stub_docstrings(content)
            content = self.fix_unicode_characters(content)
            content = self.fix_unterminated_strings(content)
            content = self.fix_invalid_syntax(content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def fix_priority_files(self) -> None:
        """Fix priority files first."""
        print("Fixing priority files...")
        print("=" * 50)
        
        for file_path in self.priority_files:
            if self.fix_file(file_path):
                self.fix_stats['errors_fixed'] += 1
                print(f"✅ Fixed: {file_path}")
            self.fix_stats['files_processed'] += 1
    
    def find_and_fix_stub_files(self) -> None:
        """Find and fix all stub files with the common pattern."""
        print("\nFinding and fixing stub files...")
        print("=" * 50)
        
        stub_pattern = '"""Stub main function."""."""'
        fixed_count = 0
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if stub_pattern in content:
                            if self.fix_file(file_path):
                                print(f"✅ Fixed stub: {file_path}")
                                fixed_count += 1
                                
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        print(f"Fixed {fixed_count} stub files")
    
    def run_selective_fix(self) -> None:
        """Run the selective fix process."""
        print("Selective Syntax Fixer - Critical Files First")
        print("=" * 60)
        
        # Step 1: Fix priority files
        self.fix_priority_files()
        
        # Step 2: Fix stub files
        self.find_and_fix_stub_files()
        
        # Summary
        print(f"\nSummary:")
        print(f"  Files processed: {self.fix_stats['files_processed']}")
        print(f"  Files with fixes: {self.fix_stats['errors_fixed']}")
        print(f"  Unicode fixes: {self.fix_stats['unicode_fixes']}")
        print(f"  Docstring fixes: {self.fix_stats['docstring_fixes']}")
        print(f"  Syntax fixes: {self.fix_stats['syntax_fixes']}")
        print("\nSelective syntax fixing completed!")


def main():
    """Main function."""
    fixer = SelectiveSyntaxFixer()
    fixer.run_selective_fix()


if __name__ == "__main__":
    main() 