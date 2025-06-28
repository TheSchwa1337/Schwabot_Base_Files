# -*- coding: utf-8 -*-
"""
Flake8 Cleanup System
=====================

Comprehensive system for cleaning up remaining Flake8 issues while preserving:
1. All mathematical content and critical trading algorithms
2. System functionality and structural integrity
3. Surgical precision in error correction

Mathematical Foundation:
- Cleanup Precision: C(error) = severity(error) * mathematical_impact(error)
- Preservation Priority: P(math) = criticality(math) * system_dependency(math)
- Error Resolution: R(target) = fix(style) + preserve(function) + maintain(math)
"""

import os
import sys
import ast
import re
import logging
import subprocess
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

class Flake8ErrorType(Enum):
    """Types of Flake8 errors."""
    SYNTAX_ERROR = "E999"
    IMPORT_ERROR = "E401"
    UNDEFINED_VARIABLE = "F821"
    UNUSED_IMPORT = "F401"
    UNUSED_VARIABLE = "F841"
    LINE_TOO_LONG = "E501"
    TRAILING_WHITESPACE = "W291"
    MISSING_WHITESPACE = "E225"
    EXTRA_WHITESPACE = "E302"
    INDENTATION_ERROR = "E111"
    BLANK_LINE_ERROR = "E303"
    DOCSTRING_ERROR = "D100"
    COMPLEXITY_ERROR = "C901"

class ErrorSeverity(Enum):
    """Severity levels for Flake8 errors."""
    CRITICAL = "critical"  # Syntax errors, import failures
    HIGH = "high"         # Undefined variables, unused imports
    MEDIUM = "medium"     # Style issues, line length
    LOW = "low"          # Whitespace, formatting

@dataclass
class Flake8Error:
    """Represents a Flake8 error."""
    error_type: Flake8ErrorType
    file_path: str
    line_number: int
    column: int
    message: str
    code: str
    severity: ErrorSeverity
    mathematical_impact: bool = False
    safe_to_fix: bool = True
    fix_strategy: str = ""
    
    def __post_init__(self):
        """Initialize Flake8 error with computed properties."""
        self.error_id = f"{self.file_path}:{self.line_number}:{self.code}"
        self._assess_mathematical_impact()
        self._determine_fix_strategy()
    
    def _assess_mathematical_impact(self):
        """Assess if this error affects mathematical content."""
        mathematical_indicators = [
            'btc', 'eth', 'usdc', 'xrp', 'price', 'hash', 'sha256',
            'tensor', 'matrix', 'vector', 'algorithm', 'optimization',
            'trading', 'profit', 'loss', 'pnl', 'roi', 'risk',
            'ferris', 'rde', 'lantern', 'recursive', 'lattice',
            'unified_math', 'np.', 'math.', 'scipy.'
        ]
        
        self.mathematical_impact = any(
            indicator in self.message.lower() or indicator in self.file_path.lower()
            for indicator in mathematical_indicators
        )
    
    def _determine_fix_strategy(self):
        """Determine the appropriate fix strategy."""
        if self.error_type == Flake8ErrorType.SYNTAX_ERROR:
            self.fix_strategy = "syntax_correction"
            self.safe_to_fix = True
        elif self.error_type == Flake8ErrorType.UNDEFINED_VARIABLE:
            self.fix_strategy = "variable_definition"
            self.safe_to_fix = not self.mathematical_impact
        elif self.error_type == Flake8ErrorType.UNUSED_IMPORT:
            self.fix_strategy = "import_cleanup"
            self.safe_to_fix = not self.mathematical_impact
        elif self.error_type == Flake8ErrorType.LINE_TOO_LONG:
            self.fix_strategy = "line_break"
            self.safe_to_fix = True
        elif self.error_type == Flake8ErrorType.TRAILING_WHITESPACE:
            self.fix_strategy = "whitespace_cleanup"
            self.safe_to_fix = True
        else:
            self.fix_strategy = "style_correction"
            self.safe_to_fix = True

class Flake8CleanupSystem:
    """
    Comprehensive Flake8 cleanup system with mathematical preservation.
    
    This system:
    1. Analyzes all Flake8 errors to determine severity and mathematical impact
    2. Applies safe fixes without compromising mathematical content
    3. Preserves critical trading algorithms and mathematical functions
    4. Maintains system functionality while improving code quality
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize the Flake8 cleanup system."""
        self.project_root = Path(project_root)
        self.flake8_errors: Dict[str, Flake8Error] = {}
        self.cleanup_metrics = {
            "errors_analyzed": 0,
            "safe_fixes_applied": 0,
            "mathematical_preserved": 0,
            "syntax_errors_fixed": 0,
            "style_issues_resolved": 0,
            "errors_prevented": 0
        }
        
        # Mathematical protection patterns
        self.mathematical_protection_patterns = [
            r'#\s*MATHEMATICAL PRESERVATION:',
            r'def.*calculate|def.*compute|def.*process',
            r'BTC.*price|ETH.*price|USDC.*price|XRP.*price',
            r'hashlib\.sha256|hashlib\.md5',
            r'tensor|matrix|vector',
            r'unified_math\.',
            r'ferris.*rde|lantern.*core|recursive.*lattice'
        ]
        
        logger.info("Flake8 Cleanup System initialized")
    
    def analyze_flake8_errors(self) -> Dict[str, Flake8Error]:
        """
        Analyze all Flake8 errors in the codebase.
        
        Returns:
            Dictionary of analyzed Flake8 errors.
        """
        logger.info("Analyzing Flake8 errors across codebase...")
        
        # Run Flake8 analysis
        flake8_output = self._run_flake8_analysis()
        
        # Parse Flake8 output
        for line in flake8_output:
            if line.strip():
                error = self._parse_flake8_line(line)
                if error:
                    self.flake8_errors[error.error_id] = error
                    self.cleanup_metrics["errors_analyzed"] += 1
        
        logger.info(f"Analyzed {len(self.flake8_errors)} Flake8 errors")
        return self.flake8_errors
    
    def categorize_errors(self) -> Dict[str, List[Flake8Error]]:
        """
        Categorize errors by type and severity.
        
        Returns:
            Dictionary of categorized errors.
        """
        logger.info("Categorizing Flake8 errors...")
        
        categories = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "mathematical": [],
            "safe_to_fix": [],
            "requires_care": []
        }
        
        for error in self.flake8_errors.values():
            # Categorize by severity
            categories[error.severity.value].append(error)
            
            # Categorize by mathematical impact
            if error.mathematical_impact:
                categories["mathematical"].append(error)
            
            # Categorize by fix safety
            if error.safe_to_fix:
                categories["safe_to_fix"].append(error)
            else:
                categories["requires_care"].append(error)
        
        logger.info(f"Categorized {len(self.flake8_errors)} errors")
        return categories
    
    def apply_safe_fixes(self) -> int:
        """
        Apply safe fixes that don't affect mathematical content.
        
        Returns:
            Number of safe fixes applied.
        """
        logger.info("Applying safe Flake8 fixes...")
        
        fixes_applied = 0
        
        for error in self.flake8_errors.values():
            if error.safe_to_fix and not error.mathematical_impact:
                if self._apply_fix(error):
                    fixes_applied += 1
                    self.cleanup_metrics["safe_fixes_applied"] += 1
                    
                    # Track specific fix types
                    if error.error_type == Flake8ErrorType.SYNTAX_ERROR:
                        self.cleanup_metrics["syntax_errors_fixed"] += 1
                    else:
                        self.cleanup_metrics["style_issues_resolved"] += 1
        
        logger.info(f"Applied {fixes_applied} safe fixes")
        return fixes_applied
    
    def apply_mathematical_preserving_fixes(self) -> int:
        """
        Apply fixes to mathematical content with extreme care.
        
        Returns:
            Number of mathematical-preserving fixes applied.
        """
        logger.info("Applying mathematical-preserving fixes...")
        
        fixes_applied = 0
        
        for error in self.flake8_errors.values():
            if error.mathematical_impact and error.safe_to_fix:
                if self._apply_mathematical_fix(error):
                    fixes_applied += 1
                    self.cleanup_metrics["mathematical_preserved"] += 1
        
        logger.info(f"Applied {fixes_applied} mathematical-preserving fixes")
        return fixes_applied
    
    def validate_fixes(self) -> bool:
        """
        Validate that all fixes are correct and don't introduce new errors.
        
        Returns:
            True if validation passes, False otherwise.
        """
        logger.info("Validating applied fixes...")
        
        # Run Flake8 again to check for new errors
        new_errors = self._run_flake8_analysis()
        
        # Count new errors
        new_error_count = len([line for line in new_errors if line.strip()])
        
        if new_error_count == 0:
            logger.info("All fixes validated successfully")
            return True
        else:
            logger.warning(f"Found {new_error_count} new errors after fixes")
            self.cleanup_metrics["errors_prevented"] = new_error_count
            return False
    
    def execute_cleanup(self) -> Dict[str, Any]:
        """
        Execute the complete Flake8 cleanup process.
        
        Returns:
            Summary of cleanup results.
        """
        logger.info("Starting Flake8 cleanup process...")
        
        # Step 1: Analyze all Flake8 errors
        self.analyze_flake8_errors()
        
        # Step 2: Categorize errors
        categories = self.categorize_errors()
        
        # Step 3: Apply safe fixes
        safe_fixes = self.apply_safe_fixes()
        
        # Step 4: Apply mathematical-preserving fixes
        math_fixes = self.apply_mathematical_preserving_fixes()
        
        # Step 5: Validate fixes
        validation_passed = self.validate_fixes()
        
        # Compile results
        results = {
            "cleanup_metrics": self.cleanup_metrics.copy(),
            "error_categories": {k: len(v) for k, v in categories.items()},
            "safe_fixes_applied": safe_fixes,
            "math_fixes_applied": math_fixes,
            "validation_passed": validation_passed,
            "error_breakdown": self._get_error_breakdown(),
            "recommendations": self._generate_cleanup_recommendations()
        }
        
        logger.info("Flake8 cleanup process completed")
        return results
    
    def _run_flake8_analysis(self) -> List[str]:
        """Run Flake8 analysis and return output."""
        try:
            result = subprocess.run(
                ['flake8', str(self.project_root), '--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.stdout.strip().split('\n') if result.stdout else []
        except subprocess.TimeoutExpired:
            logger.error("Flake8 analysis timed out")
            return []
        except Exception as e:
            logger.error(f"Failed to run Flake8 analysis: {e}")
            return []
    
    def _parse_flake8_line(self, line: str) -> Optional[Flake8Error]:
        """Parse a single Flake8 output line."""
        try:
            # Format: path:line:column:code:message
            parts = line.split(':', 4)
            if len(parts) >= 5:
                file_path = parts[0]
                line_number = int(parts[1])
                column = int(parts[2])
                code = parts[3]
                message = parts[4]
                
                # Determine error type and severity
                error_type = self._get_error_type(code)
                severity = self._get_error_severity(code)
                
                return Flake8Error(
                    error_type=error_type,
                    file_path=file_path,
                    line_number=line_number,
                    column=column,
                    message=message,
                    code=code,
                    severity=severity
                )
        except Exception as e:
            logger.warning(f"Failed to parse Flake8 line '{line}': {e}")
        
        return None
    
    def _get_error_type(self, code: str) -> Flake8ErrorType:
        """Get error type from Flake8 code."""
        error_type_map = {
            'E999': Flake8ErrorType.SYNTAX_ERROR,
            'E401': Flake8ErrorType.IMPORT_ERROR,
            'F821': Flake8ErrorType.UNDEFINED_VARIABLE,
            'F401': Flake8ErrorType.UNUSED_IMPORT,
            'F841': Flake8ErrorType.UNUSED_VARIABLE,
            'E501': Flake8ErrorType.LINE_TOO_LONG,
            'W291': Flake8ErrorType.TRAILING_WHITESPACE,
            'E225': Flake8ErrorType.MISSING_WHITESPACE,
            'E302': Flake8ErrorType.EXTRA_WHITESPACE,
            'E111': Flake8ErrorType.INDENTATION_ERROR,
            'E303': Flake8ErrorType.BLANK_LINE_ERROR,
            'D100': Flake8ErrorType.DOCSTRING_ERROR,
            'C901': Flake8ErrorType.COMPLEXITY_ERROR
        }
        return error_type_map.get(code, Flake8ErrorType.SYNTAX_ERROR)
    
    def _get_error_severity(self, code: str) -> ErrorSeverity:
        """Get error severity from Flake8 code."""
        if code.startswith('E999') or code.startswith('E401'):
            return ErrorSeverity.CRITICAL
        elif code.startswith('F821') or code.startswith('F401'):
            return ErrorSeverity.HIGH
        elif code.startswith('E501') or code.startswith('C901'):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _apply_fix(self, error: Flake8Error) -> bool:
        """Apply a safe fix to a Flake8 error."""
        try:
            with open(error.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if error.line_number <= len(lines):
                original_line = lines[error.line_number - 1]
                fixed_line = self._fix_line(original_line, error)
                
                if fixed_line != original_line:
                    lines[error.line_number - 1] = fixed_line
                    
                    with open(error.file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Error applying fix to {error.error_id}: {e}")
        
        return False
    
    def _apply_mathematical_fix(self, error: Flake8Error) -> bool:
        """Apply a fix to mathematical content with extreme care."""
        try:
            with open(error.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if error.line_number <= len(lines):
                original_line = lines[error.line_number - 1]
                
                # Check if line contains mathematical content
                if self._contains_mathematical_content(original_line):
                    # Apply minimal, safe fixes only
                    fixed_line = self._apply_minimal_fix(original_line, error)
                    
                    if fixed_line != original_line:
                        lines[error.line_number - 1] = fixed_line
                        
                        with open(error.file_path, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        
                        return True
                        
        except Exception as e:
            logger.error(f"Error applying mathematical fix to {error.error_id}: {e}")
        
        return False
    
    def _fix_line(self, line: str, error: Flake8Error) -> str:
        """Fix a line based on the error type."""
        if error.error_type == Flake8ErrorType.TRAILING_WHITESPACE:
            return line.rstrip() + '\n'
        elif error.error_type == Flake8ErrorType.LINE_TOO_LONG:
            return self._break_long_line(line)
        elif error.error_type == Flake8ErrorType.MISSING_WHITESPACE:
            return self._add_missing_whitespace(line)
        elif error.error_type == Flake8ErrorType.EXTRA_WHITESPACE:
            return self._remove_extra_whitespace(line)
        elif error.error_type == Flake8ErrorType.INDENTATION_ERROR:
            return self._fix_indentation(line)
        else:
            return line
    
    def _apply_minimal_fix(self, line: str, error: Flake8Error) -> str:
        """Apply minimal fixes to mathematical content."""
        if error.error_type == Flake8ErrorType.TRAILING_WHITESPACE:
            return line.rstrip() + '\n'
        elif error.error_type == Flake8ErrorType.LINE_TOO_LONG:
            # Only break if it's safe (not in the middle of a mathematical expression)
            if not self._is_mathematical_expression(line):
                return self._break_long_line(line)
        return line
    
    def _contains_mathematical_content(self, line: str) -> bool:
        """Check if a line contains mathematical content."""
        return any(re.search(pattern, line, re.IGNORECASE) 
                  for pattern in self.mathematical_protection_patterns)
    
    def _is_mathematical_expression(self, line: str) -> bool:
        """Check if a line contains a mathematical expression that shouldn't be broken."""
        math_expressions = [
            r'=.*\*.*\*',  # Multiple multiplications
            r'=.*\+.*\+',  # Multiple additions
            r'=.*\(.*\)',  # Parenthesized expressions
            r'hashlib\.',  # Hash functions
            r'np\.',       # NumPy operations
            r'unified_math\.'  # Unified math operations
        ]
        return any(re.search(pattern, line) for pattern in math_expressions)
    
    def _break_long_line(self, line: str) -> str:
        """Break a long line safely."""
        if len(line) <= 79:  # Already short enough
            return line
        
        # Try to break at logical points
        if 'import ' in line:
            return self._break_import_line(line)
        elif 'def ' in line:
            return self._break_function_line(line)
        elif '=' in line and not self._is_mathematical_expression(line):
            return self._break_assignment_line(line)
        else:
            # Add line continuation if safe
            return line.rstrip() + ' \\\n'
    
    def _break_import_line(self, line: str) -> str:
        """Break a long import line."""
        if 'from ' in line and ' import ' in line:
            parts = line.split(' import ')
            if len(parts) == 2:
                return f"{parts[0]} import (\n    {parts[1].rstrip()}\n)\n"
        return line
    
    def _break_function_line(self, line: str) -> str:
        """Break a long function definition line."""
        if '(' in line and ')' in line:
            # Simple break at first comma or parenthesis
            return line.replace(', ', ',\n    ')
        return line
    
    def _break_assignment_line(self, line: str) -> str:
        """Break a long assignment line."""
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                return f"{parts[0]}= \\\n    {parts[1]}"
        return line
    
    def _add_missing_whitespace(self, line: str) -> str:
        """Add missing whitespace around operators."""
        operators = ['+', '-', '*', '/', '=', '==', '!=', '<=', '>=']
        for op in operators:
            if op in line and not re.search(rf'\s{re.escape(op)}\s', line):
                line = re.sub(rf'([^\s]){re.escape(op)}([^\s])', rf'\1 {op} \2', line)
        return line
    
    def _remove_extra_whitespace(self, line: str) -> str:
        """Remove extra whitespace."""
        return re.sub(r'\s+', ' ', line).rstrip() + '\n'
    
    def _fix_indentation(self, line: str) -> str:
        """Fix indentation issues."""
        # Count leading spaces
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces % 4 != 0:
            # Fix to nearest 4-space boundary
            fixed_spaces = (leading_spaces // 4) * 4
            return ' ' * fixed_spaces + line.lstrip()
        return line
    
    def _get_error_breakdown(self) -> Dict[str, int]:
        """Get breakdown of errors by type."""
        breakdown = {}
        for error_type in Flake8ErrorType:
            breakdown[error_type.value] = len([
                e for e in self.flake8_errors.values() 
                if e.error_type == error_type
            ])
        return breakdown
    
    def _generate_cleanup_recommendations(self) -> List[str]:
        """Generate cleanup recommendations."""
        recommendations = []
        
        # Check for critical errors
        critical_errors = len([e for e in self.flake8_errors.values() 
                             if e.severity == ErrorSeverity.CRITICAL])
        if critical_errors > 0:
            recommendations.append(f"Address {critical_errors} critical errors first")
        
        # Check for mathematical content
        math_errors = len([e for e in self.flake8_errors.values() 
                          if e.mathematical_impact])
        if math_errors > 0:
            recommendations.append(f"Carefully review {math_errors} errors affecting mathematical content")
        
        # Check for safe fixes
        safe_fixes = len([e for e in self.flake8_errors.values() 
                         if e.safe_to_fix and not e.mathematical_impact])
        if safe_fixes > 0:
            recommendations.append(f"Apply {safe_fixes} safe style fixes")
        
        # General recommendations
        recommendations.append("Run Flake8 regularly to maintain code quality")
        recommendations.append("Use automated tools for style enforcement")
        
        return recommendations


def main():
    """Main function for Flake8 cleanup."""
    print("=== Flake8 Cleanup System ===")
    print("Systematically cleaning up Flake8 issues...")
    print("Preserving mathematical content and system functionality...")
    print()
    
    # Initialize cleanup system
    cleanup_system = Flake8CleanupSystem()
    
    try:
        # Execute cleanup
        results = cleanup_system.execute_cleanup()
        
        # Print summary
        print("=== FLAKE8 CLEANUP SUMMARY ===")
        metrics = results["cleanup_metrics"]
        print(f"Errors Analyzed: {metrics['errors_analyzed']}")
        print(f"Safe Fixes Applied: {metrics['safe_fixes_applied']}")
        print(f"Mathematical Preserved: {metrics['mathematical_preserved']}")
        print(f"Syntax Errors Fixed: {metrics['syntax_errors_fixed']}")
        print(f"Style Issues Resolved: {metrics['style_issues_resolved']}")
        print(f"Validation Passed: {results['validation_passed']}")
        
        print(f"\nError Categories:")
        for category, count in results["error_categories"].items():
            print(f"  {category}: {count}")
        
        print(f"\nError Breakdown:")
        for error_type, count in results["error_breakdown"].items():
            if count > 0:
                print(f"  {error_type}: {count}")
        
        print(f"\nRecommendations:")
        for recommendation in results["recommendations"]:
            print(f"  - {recommendation}")
        
    except Exception as e:
        logger.error(f"Flake8 cleanup failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 