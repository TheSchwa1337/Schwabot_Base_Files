import ast
import sys
from typing import Any, Dict, List


def check_type_hints(file_path: str) -> Dict[str, List[str]]:
    """
    Check type hints in a Python file.

    Args:
        file_path: Path to the Python file to analyze

    Returns:
        Dictionary of potential type hinting improvements
    """
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())

    issues: Dict[str, List[str]] = {}
        'missing_return_hints': [],
        'missing_param_hints': [],
        'complex_functions': []
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check return type hints
            if not node.returns and not any(isinstance(d, ast.Str) for d in node.decorator_list):
                issues['missing_return_hints'].append(node.name)

            # Check parameter type hints
            for arg in node.args.args:
                if not arg.annotation:
                    issues['missing_param_hints'].append(f"{node.name}: {arg.arg}")

            # Check function complexity
            if len(node.body) > 20:  # Arbitrary complexity threshold
                issues['complex_functions'].append(node.name)

    return issues

def main():
    files_to_check = []
        'core/zpe_zbe_core.py',
        'core/unified_math_system.py'
    ]

    all_issues: Dict[str, Dict[str, List[str]]] = {}

    for file_path in files_to_check:
        try:
            file_issues = check_type_hints(file_path)
            all_issues[file_path] = file_issues
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Print results
    for file_path, issues in all_issues.items():
        print(f"\nAnalysis for {file_path}:")
        for issue_type, details in issues.items():
            if details:
                print(f"  {issue_type.replace('_', ' ').title()}:")
                for detail in details:
                    print(f"    - {detail}")

if __name__ == '__main__':
    main() 