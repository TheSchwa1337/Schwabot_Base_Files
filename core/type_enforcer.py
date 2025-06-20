#!/usr/bin/env python3
"""Type Enforcer.

Centralized type-annotation management utility. Systematically adds missing
annotations to eliminate medium-priority Flake8 issues and provides
intelligent type inference for mathematical and data-processing functions
with Windows CLI compatibility.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TypeEnforcer:
    """Add or correct type annotations in Python source files."""

    def __init__(self: Any) -> None:
        """Initialize type-pattern and function-pattern dictionaries."""
        self._type_patterns: Dict[str, str] = {
            # Mathematical patterns
            'price': 'float',
            'volume': 'float',
            'quantity': 'float',
            'amount': 'float',
            'rate': 'float',
            'percentage': 'float',
            'ratio': 'float',
            'delta': 'float',
            'offset': 'float',
            'threshold': 'float',
            'limit': 'float',
            'target': 'float',
            'waveform': 'List[float]',
            'entropy': 'float',
            'correlation': 'float',
            'volatility': 'float',
            'momentum': 'float',
            'oscillator': 'List[float]',
            'indicator': 'Dict[str, float]',
            'signal': 'Dict[str, Any]',
            'pattern': 'Dict[str, Any]',
            'analysis': 'Dict[str, Any]',
            'prediction': 'Dict[str, Any]',
            'forecast': 'Dict[str, Any]',
            'optimization': 'Dict[str, Any]',
            'calibration': 'Dict[str, Any]',
            'validation': 'Dict[str, Any]',

            # Trading-specific patterns
            'order': 'Dict[str, Any]',
            'trade': 'Dict[str, Any]',
            'position': 'Dict[str, Any]',
            'portfolio': 'Dict[str, Any]',
            'balance': 'Dict[str, float]',
            'profit': 'float',
            'loss': 'float',
            'pnl': 'float',
            'roi': 'float',
            'risk': 'float',
            'exposure': 'float',
            'leverage': 'float',

            # Data-structure patterns
            'data': 'Dict[str, Any]',
            'result': 'Dict[str, Any]',
            'config': 'Dict[str, Any]',
            'params': 'Dict[str, Any]',
            'kwargs': 'Dict[str, Any]',
            'args': 'List[Any]',
            'items': 'List[Any]',
            'values': 'List[Any]',
            'keys': 'List[str]',
            'names': 'List[str]',
            'symbols': 'List[str]',
            'tickers': 'List[str]',

            # Time patterns
            'timestamp': 'datetime',
            'time': 'datetime',
            'date': 'datetime',
            'period': 'str',
            'duration': 'int',

            # String patterns
            'name': 'str',
            'id': 'str',
            'type': 'str',
            'status': 'str',
            'message': 'str',
            'description': 'str',
            'path': 'str',
            'url': 'str',
            'symbol': 'str',
            'ticker': 'str',
            'currency': 'str',
            'format': 'str',

            # Boolean patterns
            'enabled': 'bool',
            'active': 'bool',
            'valid': 'bool',
            'success': 'bool',
            'ready': 'bool',
            'available': 'bool',
            'visible': 'bool',
            'debug': 'bool',
            'verbose': 'bool',

            # Integer patterns
            'count': 'int',
            'index': 'int',
            'size': 'int',
            'length': 'int',
            'max': 'int',
            'min': 'int',
            'value': 'int',
            'number': 'int',
            'tick': 'int',
            'step': 'int',
            'level': 'int',
        }

        self._function_patterns: Dict[str, str] = {
            # Mathematical functions
            'calculate': 'float',
            'compute': 'float',
            'process': 'Dict[str, Any]',
            'analyze': 'Dict[str, Any]',
            'evaluate': 'float',
            'estimate': 'float',
            'predict': 'float',
            'forecast': 'float',
            'simulate': 'Dict[str, Any]',
            'optimize': 'Dict[str, Any]',
            'minimize': 'float',
            'maximize': 'float',

            # Data processing functions
            'transform': 'List[Any]',
            'filter': 'List[Any]',
            'sort': 'List[Any]',
            'group': 'Dict[str, List[Any]]',
            'aggregate': 'Dict[str, Any]',
            'validate': 'bool',
            'verify': 'bool',
            'check': 'bool',
            'test': 'bool',

            # I/O functions
            'load': 'Dict[str, Any]',
            'save': 'bool',
            'read': 'str',
            'write': 'bool',
            'parse': 'Dict[str, Any]',
            'serialize': 'str',
            'deserialize': 'Dict[str, Any]',

            # Utility functions
            'format': 'str',
            'convert': 'Any',
            'encode': 'str',
            'decode': 'str',
            'hash': 'str',
            'encrypt': 'str',
            'decrypt': 'str',
        }

    def enforce_type_annotations(self: Any, file_path: str) -> Dict[str, int]:
        """Enforce type annotations in a file with Windows CLI compatibility."""
        stats: Dict[str, int] = {
            'functions_fixed': 0,
            'parameters_fixed': 0,
            'returns_fixed': 0,
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the file
            tree = ast.parse(content)

            # Apply fixes
            fixed_content = self._apply_type_fixes(content, tree, stats)

            # Write back if changes were made
            if fixed_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                logger.info(f"Applied type annotations to {file_path}")

            return stats

        except Exception as e:
            logger.error(f"Error enforcing types in {file_path}: {e}")
            return stats

    def _apply_type_fixes(self: Any, content: str, tree: ast.AST, stats: Dict[str, int]) -> str:
        """Apply type annotation fixes to the content"""
        lines = content.split('\n')

        # Sort nodes by line number (descending) to avoid line number shifts
        nodes = []
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                nodes.append(node)

        nodes.sort(key=lambda x: x.lineno, reverse=True)

        for node in nodes:
            if isinstance(node, ast.FunctionDef):
                lines = self._fix_function_annotations(node, lines, stats)

        return '\n'.join(lines)

    def _fix_function_annotations(
        self: Any,
        node: ast.FunctionDef,
        lines: List[str],
        stats: Dict[str, int]
    ) -> List[str]:
        """Fix type annotations for a function"""
        line_idx = node.lineno - 1  # Convert to 0-based index

        if line_idx >= len(lines):
            return lines

        # Fix function signature
        original_line = lines[line_idx]
        fixed_line = self._fix_function_signature(original_line, node, stats)

        if fixed_line != original_line:
            lines[line_idx] = fixed_line
            stats['functions_fixed'] += 1

        # Fix parameter annotations
        for arg in node.args.args:
            if arg.arg != 'self' and arg.annotation is None:
                param_line = self._find_parameter_line(lines, line_idx, arg.arg)
                if param_line is not None:
                    param_idx = param_line
                    param_type = self._infer_parameter_type(arg.arg)
                    if param_type:
                        lines = self._add_parameter_annotation(lines, param_idx, arg.arg, param_type)
                        stats['parameters_fixed'] += 1

        # Fix return type annotation
        if node.returns is None and node.name != '__init__':
            return_type = self._infer_return_type(node.name)
            if return_type:
                lines = self._add_return_annotation(lines, line_idx, return_type)
                stats['returns_fixed'] += 1

        return lines

    def _fix_function_signature(self: Any, line: str, node: ast.FunctionDef,
                              stats: Dict[str, int]) -> str:
        """Fix function signature with proper type annotations"""
        # Add missing parameter type annotations
        if 'self' in line and 'self:' not in line:
            line = line.replace('(self)', '(self: Any)')
            line = line.replace('(self,', '(self: Any,')

        # Add missing return type annotation
        if line.strip().endswith(':') and not '->' in line:
            return_type = self._infer_return_type(node.name)
            if return_type:
                line = line.rstrip(':') + f' -> {return_type}:'

        return line

    def _infer_parameter_type(self: Any, param_name: str) -> Optional[str]:
        """Infer parameter type based on name patterns"""
        # Check exact matches first
        if param_name in self._type_patterns:
            return self._type_patterns[param_name]

        # Check partial matches
        for pattern, type_name in self._type_patterns.items():
            if pattern in param_name.lower():
                return type_name

        # Default types for common patterns
        if param_name.endswith('_list'):
            return 'List[Any]'
        elif param_name.endswith('_dict'):
            return 'Dict[str, Any]'
        elif param_name.endswith('_str'):
            return 'str'
        elif param_name.endswith('_int'):
            return 'int'
        elif param_name.endswith('_float'):
            return 'float'
        elif param_name.endswith('_bool'):
            return 'bool'

        return 'Any'

    def _infer_return_type(self: Any, function_name: str) -> Optional[str]:
        """Infer return type based on function name patterns"""
        # Check exact matches first
        if function_name in self._function_patterns:
            return self._function_patterns[function_name]

        # Check partial matches
        for pattern, type_name in self._function_patterns.items():
            if pattern in function_name.lower():
                return type_name

        # Default return types for common patterns
        if function_name.startswith('get_'):
            return 'Any'
        elif function_name.startswith('set_'):
            return 'None'
        elif function_name.startswith('is_'):
            return 'bool'
        elif function_name.startswith('has_'):
            return 'bool'
        elif function_name.startswith('can_'):
            return 'bool'
        elif function_name.startswith('should_'):
            return 'bool'
        elif function_name.startswith('will_'):
            return 'bool'

        return 'Any'

    def _find_parameter_line(self: Any, lines: List[str], func_line: int, param_name: str) -> Optional[int]:
        """Find the line containing a parameter definition"""
        # Look for the parameter in the function signature
        func_line_content = lines[func_line]
        if param_name in func_line_content:
            return func_line

        return None

    def _add_parameter_annotation(self: Any, lines: List[str], line_idx: int,
                                param_name: str, param_type: str) -> List[str]:
        """Add type annotation to a parameter"""
        line = lines[line_idx]

        # Add annotation to parameter
        if f'{param_name},' in line:
            line = line.replace(f'{param_name},', f'{param_name}: {param_type},')
        elif f'{param_name})' in line:
            line = line.replace(f'{param_name})', f'{param_name}: {param_type})')
        elif f'{param_name}:' in line and ':' not in line.split(param_name)[1].split(',')[0]:
            # Parameter already has annotation
            pass
        else:
            line = line.replace(param_name, f'{param_name}: {param_type}')

        lines[line_idx] = line
        return lines

    def _add_return_annotation(self: Any, lines: List[str], line_idx: int,
                             return_type: str) -> List[str]:
        """Add return type annotation to a function"""
        line = lines[line_idx]

        if line.strip().endswith(':'):
            line = line.rstrip(':') + f' -> {return_type}:'
            lines[line_idx] = line

        return lines

    def add_custom_pattern(self: Any, name: str, type_name: str, pattern_type: str = 'parameter') -> None:
        """Add a custom type pattern"""
        if pattern_type == 'parameter':
            self._type_patterns[name] = type_name
        elif pattern_type == 'function':
            self._function_patterns[name] = type_name

    def get_statistics(self: Any) -> Dict[str, int]:
        """Get statistics about type patterns"""
        return {
            'parameter_patterns': len(self._type_patterns),
            'function_patterns': len(self._function_patterns),
        }


# Global instance for easy access
type_enforcer = TypeEnforcer()


def enforce_types_in_file(file_path: str) -> Dict[str, int]:
    """Convenience function for enforcing types in a file"""
    return type_enforcer.enforce_type_annotations(file_path)


def enforce_types_in_directory(directory: str) -> Dict[str, int]:
    """Enforce types in all Python files in a directory"""
    total_stats = {'functions_fixed': 0, 'parameters_fixed': 0, 'returns_fixed': 0}

    for py_file in Path(directory).rglob('*.py'):
        if py_file.is_file():
            stats = enforce_types_in_file(str(py_file))
            for key in total_stats:
                total_stats[key] += stats[key]

    return total_stats
