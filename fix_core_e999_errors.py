#!/usr/bin/env python3
"""
Targeted fixer for actual E999 errors in core directory.
Now with targeted repair for mismatched brackets/parentheses in type annotations and function signatures.
"""

import os
import re


def fix_missing_indented_blocks(content):
    """Fix missing indented blocks after try, class, def, etc."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    block_keywords = (
        'try:', 'class ', 'def ', 'if ', 'elif ', 'else:', 'except', 'finally:', 'for ', 'while ', 'with '
    )
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Check for block statements that need indented blocks
        if any(stripped.startswith(kw) for kw in block_keywords):
            fixed_lines.append(line)
            i += 1
            # Check if next line is not indented or is empty
            if i < len(lines):
                next_line = lines[i]
                if not next_line.strip() or (not next_line.startswith('    ') and not next_line.startswith('\t')):
                    fixed_lines.append('    pass')
        else:
            fixed_lines.append(line)
            i += 1
    return '\n'.join(fixed_lines)


def fix_mismatched_brackets_in_types(content):
    """Fix mismatched brackets/parentheses in type annotations and function signatures."""
    try:
        # Fix List[Type) -> List[Type]
        content = re.sub(r'(List|Tuple|Dict|Set|Optional|Union)\[([^\]\)]+)\)', r'\1[\2]', content)
        # Fix List[Type]] -> List[Type]
        content = re.sub(r'(List|Tuple|Dict|Set|Optional|Union)\[([^\]]+)\]\]', r'\1[\2]', content)
        # Fix function signatures: def func(...[...)] -> def func(...[...])
        content = re.sub(r'def ([a-zA-Z0-9_]+)\(([^\)]*\[[^\)]*)\)\]', r'def \1(\2)', content)
    except re.error as e:
        print(f"Regex error in fix_mismatched_brackets_in_types: {e}")
    return content


def fix_unmatched_brackets(content):
    """Fix unmatched or misplaced parentheses/brackets in lines."""
    # Fix common patterns: ( ... ] -> ( ... ) and similar
    content = re.sub(r'\(\s*\]', '()', content)
    content = re.sub(r'\[\s*\)', '[]', content)
    content = re.sub(r'{\s*\]', '{}', content)
    content = re.sub(r'\[\s*}', '[]', content)
    # Fix specific patterns: []) -> []
    content = re.sub(r'\[\s*\]\s*\)', '[]', content)
    content = re.sub(r'\(\s*\[\s*\]', '()', content)
    # Fix lines where a closing bracket is used instead of a parenthesis
    content = re.sub(r'\(([^\n\)]*)\]', r'(\1)', content)
    content = re.sub(r'\[([^\n\]]*)\)', r'[\1]', content)
    # Remove unmatched closing brackets at line ends
    content = re.sub(r'\)\s*\]$', ')', content)
    content = re.sub(r'\]\s*\)$', ']', content)
    # Remove unmatched closing brackets at line starts
    content = re.sub(r'^\]\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\)\s*', '', content, flags=re.MULTILINE)
    # Remove unmatched opening brackets at line ends
    content = re.sub(r'\(\s*$', '', content)
    content = re.sub(r'\[\s*$', '', content)
    return content


def normalize_indentation(content):
    """Stack-based normalization of indentation to 4 spaces for all blocks."""
    lines = content.split('\n')
    # Convert tabs to 4 spaces
    lines = [line.replace('\t', '    ') for line in lines]
    # Remove trailing whitespace
    lines = [line.rstrip() for line in lines]
    # Remove leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    fixed_lines = []
    indent_stack = [0]
    block_keywords = (
        'try:', 'class ', 'def ', 'if ', 'elif ', 'else:', 'except', 'finally:', 'for ', 'while ', 'with '
    )
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Determine current indent
        curr_indent = len(line) - len(line.lstrip())
        # If this is a block start, push indent
        if any(stripped.startswith(kw) for kw in block_keywords):
            # Set block indent to previous + 4
            block_indent = (indent_stack[-1] if indent_stack else 0) + 4
            fixed_lines.append(' ' * indent_stack[-1] + stripped)
            indent_stack.append(block_indent)
            i += 1
            # Fix all lines in the block
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip())
                # If next line is empty, just append
                if not next_stripped:
                    fixed_lines.append('')
                    i += 1
                    continue
                # If next line is a new block at same or lower indent, break
                if next_indent <= indent_stack[-2] and any(next_stripped.startswith(kw) for kw in block_keywords):
                    break
                # If next line is less indented than block, break
                if next_indent < indent_stack[-1]:
                    break
                # Otherwise, ensure correct indent
                fixed_lines.append(' ' * indent_stack[-1] + next_stripped)
                i += 1
            indent_stack.pop()
        else:
            # Top-level or non-block line
            fixed_lines.append(' ' * indent_stack[-1] + stripped)
            i += 1
    return '\n'.join(fixed_lines)


def fix_generator_expression_parentheses(content):
    """Fix generator expression parentheses issue."""
    # Fix the specific pattern in common.py
    content = re.sub(r'for\s+([^:]+)\s+in\s+([^:]+)\s+if\s+([^:]+):', r'for \1 in (\2 for \2 in \2 if \3):', content)
    # More general fix for generator expressions
    content = re.sub(r'for\s+([^:]+)\s+in\s+([^:]+)\s+if\s+([^:]+):', r'for \1 in (\2 if \3):', content)
    return content


def fix_file(filepath):
    """Fix all syntax errors in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        original_content = content
        # Apply all fixes
        content = fix_missing_indented_blocks(content)
        content = fix_mismatched_brackets_in_types(content)
        content = fix_unmatched_brackets(content)
        content = normalize_indentation(content)
        content = fix_generator_expression_parentheses(content)
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    """Main function to fix core directory E999 errors."""
    # Target specific files with actual errors
    target_files = [
        'core/adaptive_trainer.py',
        'core/analysis_engine.py',
        'core/auth_manager.py',
        'core/backup_manager.py',
        'core/btc_tick_matrix_initializer.py',
        'core/cache_store.py',
        'core/cli_matrix_visualizer.py',
        'core/common.py',
        'core/config.py',
        'core/config_manager.py',
        'core/constants.py',
        'core/data_processor.py',
        'core/database_manager.py',
        'core/decision_engine.py',
        'core/demo_backtrace_pipeline.py',
        'core/dlt_waveform_engine.py',
        'core/fix_critical_issues.py',
        'core/helpers.py',
        'core/model_predictor.py',
        'core/multi_bit_btc_processor.py',
        'core/network_manager.py',
        'core/optimization_engine.py',
        'core/orchestrator.py',
        'core/performance_monitor.py',
        'core/post_failure_recovery_intelligence_loop.py',
        'core/profit_routing_engine.py',
        'core/risk_manager.py',
        'core/startup.py',
        'core/strategy_manager.py',
        'core/temporal_execution_correction_layer.py',
        'core/transaction_handler.py',
        'core/utilities.py',
    ]
    print(f"Targeting {len(target_files)} files with E999 errors")
    fixed_count = 0
    for filepath in target_files:
        if os.path.exists(filepath):
            if fix_file(filepath):
                print(f"Fixed: {filepath}")
                fixed_count += 1
        else:
            print(f"File not found: {filepath}")
    print(f"\nFixed {fixed_count} files")
    print("Core E999 error fixing complete!")


if __name__ == "__main__":
    main()
