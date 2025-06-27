#!/usr/bin/env python3
"""
Critical Error Profile Analyzer

This script analyzes Flake8 output to identify critical error profiles
and prioritize files that need manual intervention.
"""

import re
import subprocess
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict

def run_flake8_analysis() -> str:
    """Run Flake8 and capture detailed output."""
    try:
        result = subprocess.run(
            ['flake8', 'core/', '--select=E999,F821,F541,E265,C901,W505', '--show-source'],
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: Flake8 analysis timed out"
    except Exception as e:
        return f"Error running Flake8: {str(e)}"

def parse_flake8_output(output: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse Flake8 output into structured data."""
    errors_by_file = defaultdict(list)
    
    for line in output.strip().split('\n'):
        if not line.strip():
            continue
            
        # Parse Flake8 output format: file:line:col: code message
        match = re.match(r'([^:]+):(\\d+):(\\d+):\\s*(\\w+)\\s+(.+)', line)
        if match:
            file_path, line_num, col_num, error_code, message = match.groups()
            
            errors_by_file[file_path].append({
                'line': int(line_num),
                'column': int(col_num),
                'code': error_code,
                'message': message.strip(),
                'severity': get_error_severity(error_code)
            })
    
    return dict(errors_by_file)

def get_error_severity(error_code: str) -> int:
    """Assign severity levels to error codes."""
    severity_map = {
        'E999': 5,  # Syntax errors - highest priority
        'F821': 4,  # Undefined names
        'F541': 3,  # F-string issues
        'E265': 2,  # Comment formatting
        'C901': 2,  # Function complexity
        'W505': 1,  # Doc line too long
    }
    return severity_map.get(error_code, 1)

def categorize_errors(errors_by_file: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Categorize errors by type and severity."""
    categories = {
        'critical_syntax': [],      # E999 errors
        'missing_blocks': [],       # Missing indented blocks
        'unmatched_brackets': [],   # Bracket/parenthesis issues
        'invalid_syntax': [],       # Invalid syntax patterns
        'unterminated_strings': [], # String literal issues
        'other_issues': []          # Other error types
    }
    
    for file_path, errors in errors_by_file.items():
        file_info = {
            'file': file_path,
            'error_count': len(errors),
            'errors': errors,
            'max_severity': max(e['severity'] for e in errors),
            'e999_count': len([e for e in errors if e['code'] == 'E999']),
            'patterns': analyze_error_patterns(errors)
        }
        
        # Categorize based on error patterns
        if file_info['e999_count'] > 0:
            if has_missing_blocks(errors):
                categories['missing_blocks'].append(file_info)
            elif has_unmatched_brackets(errors):
                categories['unmatched_brackets'].append(file_info)
            elif has_invalid_syntax(errors):
                categories['invalid_syntax'].append(file_info)
            elif has_unterminated_strings(errors):
                categories['unterminated_strings'].append(file_info)
            else:
                categories['critical_syntax'].append(file_info)
        else:
            categories['other_issues'].append(file_info)
    
    return categories

def analyze_error_patterns(errors: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze error patterns in a file."""
    patterns = defaultdict(int)
    
    for error in errors:
        message = error['message'].lower()
        
        if 'expected an indented block' in message:
            patterns['missing_indented_block'] += 1
        elif 'unmatched' in message:
            patterns['unmatched_brackets'] += 1
        elif 'invalid syntax' in message:
            patterns['invalid_syntax'] += 1
        elif 'unterminated' in message:
            patterns['unterminated_string'] += 1
        elif 'closing parenthesis' in message:
            patterns['parenthesis_mismatch'] += 1
        else:
            patterns['other'] += 1
    
    return dict(patterns)

def has_missing_blocks(errors: List[Dict[str, Any]]) -> bool:
    """Check if file has missing indented block errors."""
    return any('expected an indented block' in e['message'].lower() for e in errors)

def has_unmatched_brackets(errors: List[Dict[str, Any]]) -> bool:
    """Check if file has unmatched bracket errors."""
    return any('unmatched' in e['message'].lower() for e in errors)

def has_invalid_syntax(errors: List[Dict[str, Any]]) -> bool:
    """Check if file has invalid syntax errors."""
    return any('invalid syntax' in e['message'].lower() for e in errors)

def has_unterminated_strings(errors: List[Dict[str, Any]]) -> bool:
    """Check if file has unterminated string errors."""
    return any('unterminated' in e['message'].lower() for e in errors)

def generate_priority_list(categories: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate prioritized list of files to fix."""
    priority_list = []
    
    # Sort each category by error count and severity
    for category_name, files in categories.items():
        sorted_files = sorted(files, key=lambda x: (x['e999_count'], x['max_severity']), reverse=True)
        priority_list.extend(sorted_files)
    
    # Sort overall by priority
    priority_list.sort(key=lambda x: (x['e999_count'], x['max_severity']), reverse=True)
    
    return priority_list

def create_manual_fix_plan(priority_list: List[Dict[str, Any]]) -> str:
    """Create a manual fix plan for high-priority files."""
    plan = []
    
    # Top 20 files that need manual intervention
    top_files = priority_list[:20]
    
    for i, file_info in enumerate(top_files, 1):
        file_path = file_info['file']
        error_count = file_info['error_count']
        e999_count = file_info['e999_count']
        patterns = file_info['patterns']
        
        plan.append(f"{i:2d}. {file_path}")
        plan.append(f"     Errors: {error_count} total, {e999_count} E999")
        plan.append(f"     Patterns: {', '.join(f'{k}={v}' for k, v in patterns.items())}")
        
        # Add specific fix suggestions
        if 'missing_indented_block' in patterns:
            plan.append(f"     \\u2192 Add missing indented blocks after try/if/for/while/def statements")
        if 'unmatched_brackets' in patterns:
            plan.append(f"     \\u2192 Fix unmatched brackets/parentheses")
        if 'invalid_syntax' in patterns:
            plan.append(f"     \\u2192 Fix invalid syntax (likely stub file implementation)")
        if 'unterminated_string' in patterns:
            plan.append(f"     \\u2192 Fix unterminated string literals")
        
        plan.append("")
    
    return "\n".join(plan)

def main():
    """Main analysis function."""
    print("\\u1f50d Critical Error Profile Analysis")
    print("=" * 50)
    
    # Run Flake8 analysis
    print("Running Flake8 analysis...")
    flake8_output = run_flake8_analysis()
    
    if flake8_output.startswith("Error"):
        print(f"\\u274c {flake8_output}")
        return
    
    # Parse and categorize errors
    print("Parsing error data...")
    errors_by_file = parse_flake8_output(flake8_output)
    categories = categorize_errors(errors_by_file)
    
    # Generate priority list
    priority_list = generate_priority_list(categories)
    
    # Print summary
    print(f"\\n\\u1f4ca ERROR ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total files with errors: {len(errors_by_file)}")
    print(f"Total errors: {sum(len(errors) for errors in errors_by_file.values())}")
    
    print(f"\\n\\u1f4cb ERROR CATEGORIES:")
    for category, files in categories.items():
        if files:
            total_errors = sum(len(f['errors']) for f in files)
            e999_errors = sum(f['e999_count'] for f in files)
            print(f"  {category}: {len(files)} files, {total_errors} errors ({e999_errors} E999)")
    
    # Create manual fix plan
    print(f"\\n\\u1f3af MANUAL FIX PRIORITY LIST")
    print("=" * 50)
    manual_plan = create_manual_fix_plan(priority_list)
    print(manual_plan)
    
    # Save detailed analysis
    analysis_data = {
        'summary': {
            'total_files': len(errors_by_file),
            'total_errors': sum(len(errors) for errors in errors_by_file.values()),
            'categories': {k: len(v) for k, v in categories.items()}
        },
        'priority_list': priority_list[:50],  # Top 50 files
        'categories': categories
    }
    
    with open('critical_error_analysis.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"\\n\\u1f4be Detailed analysis saved to: critical_error_analysis.json")
    print(f"\\u1f389 Analysis complete! Focus on the top 20 files for maximum impact.")

if __name__ == "__main__":
    main() 