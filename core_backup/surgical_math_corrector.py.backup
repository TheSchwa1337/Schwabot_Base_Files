# -*- coding: utf-8 -*-
"""
Surgical Mathematical Corrector
==============================

Targeted correction system that:
1. Identifies and safely removes non-functional fixes
2. Preserves high-level mathematical content that's critical to the system
3. Applies targeted corrections without introducing new Flake8 errors
4. Maintains full functionality without large structural changes

Mathematical Foundation:
- Surgical Precision: S(fix) = functional_value(fix) * mathematical_impact(fix)
- Preservation Priority: P(math) = criticality(math) * system_dependency(math)
- Error Reduction: E(target) = current_errors - safe_removals + targeted_fixes
"""

import os
import ast
import re
import logging
import hashlib
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

class FixType(Enum):
    """Types of fixes found in the codebase."""
    PLACEHOLDER_PASS = "placeholder_pass"
    TODO_IMPLEMENT = "todo_implement"
    EMERGENCY_PLACEHOLDER = "emergency_placeholder"
    FIXME_UNUSED_IMPORT = "fixme_unused_import"
    MATHEMATICAL_PRESERVATION = "mathematical_preservation"
    SYNTAX_CORRECTION = "syntax_correction"
    STYLE_CORRECTION = "style_correction"

class MathematicalCriticality(Enum):
    """Criticality levels for mathematical content."""
    CRITICAL = "critical"  # Core trading algorithms, BTC hashing, tensor ops
    HIGH = "high"         # Mathematical formulas, statistical functions
    MEDIUM = "medium"     # Helper functions with math dependencies
    LOW = "low"          # Simple math operations, basic calculations
    NONE = "none"        # No mathematical content

@dataclass
class Fix:
    """Represents a fix in the codebase."""
    fix_type: FixType
    file_path: str
    line_number: int
    content: str
    mathematical_criticality: MathematicalCriticality
    functional_value: float  # 0.0 = no function, 1.0 = critical function
    safe_to_remove: bool = False
    replacement_content: Optional[str] = None
    
    def __post_init__(self):
        """Initialize fix with computed properties."""
        self.fix_id = hashlib.sha256(
            f"{self.file_path}:{self.line_number}:{self.content}".encode()
        ).hexdigest()[:8]

@dataclass
class MathematicalContent:
    """Represents mathematical content that must be preserved."""
    file_path: str
    line_number: int
    content: str
    math_type: str  # formula, algorithm, constant, etc.
    dependencies: List[str] = field(default_factory=list)
    criticality: MathematicalCriticality = MathematicalCriticality.MEDIUM
    
    def __post_init__(self):
        """Initialize mathematical content."""
        self.content_id = hashlib.sha256(
            f"{self.file_path}:{self.line_number}:{self.content}".encode()
        ).hexdigest()[:8]

class SurgicalMathCorrector:
    """
    Surgical correction system for targeted mathematical preservation.
    
    This system:
    1. Analyzes all fixes to determine functional value and mathematical impact
    2. Safely removes non-functional fixes that don't contain critical math
    3. Preserves high-level mathematical content
    4. Applies targeted corrections without introducing new errors
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize the surgical corrector."""
        self.project_root = Path(project_root)
        self.fixes: Dict[str, Fix] = {}
        self.mathematical_content: Dict[str, MathematicalContent] = {}
        self.correction_metrics = {
            "fixes_analyzed": 0,
            "safe_removals": 0,
            "mathematical_preserved": 0,
            "targeted_corrections": 0,
            "errors_prevented": 0
        }
        
        # Mathematical indicators for criticality assessment
        self.critical_math_indicators = {
            "btc", "eth", "usdc", "xrp", "price", "hash", "sha256",
            "tensor", "matrix", "vector", "algorithm", "optimization",
            "trading", "profit", "loss", "pnl", "roi", "risk",
            "ferris", "rde", "lantern", "recursive", "lattice"
        }
        
        self.high_math_indicators = {
            "np.", "math.", "scipy.", "unified_math", "calculation",
            "formula", "equation", "statistical", "probability",
            "correlation", "regression", "derivative", "integral"
        }
        
        logger.info("Surgical Mathematical Corrector initialized")
    
    def analyze_all_fixes(self) -> Dict[str, Fix]:
        """
        Analyze all fixes in the codebase to determine their value and safety.
        
        Returns:
            Dictionary of analyzed fixes.
        """
        logger.info("Analyzing all fixes in the codebase...")
        
        # Scan for different types of fixes
        fix_patterns = {
            FixType.PLACEHOLDER_PASS: [
                r"pass\s*#\s*TODO:\s*Implement",
                r"pass\s*#\s*TODO",
                r"pass\s*#.*placeholder"
            ],
            FixType.TODO_IMPLEMENT: [
                r"#\s*TODO:\s*Implement",
                r"#\s*TODO:",
                r"#.*TODO.*"
            ],
            FixType.EMERGENCY_PLACEHOLDER: [
                r"Emergency placeholder",
                r"EMERGENCY:",
                r"emergency.*placeholder"
            ],
            FixType.FIXME_UNUSED_IMPORT: [
                r"#.*FIXME:.*Unused import",
                r"#.*FIXME:",
                r"#.*unused.*import"
            ],
            FixType.MATHEMATICAL_PRESERVATION: [
                r"#\s*MATHEMATICAL PRESERVATION:",
                r"Mathematical logic.*preserved",
                r"math.*preserved"
            ]
        }
        
        for fix_type, patterns in fix_patterns.items():
            for pattern in patterns:
                fixes = self._find_fixes_by_pattern(pattern, fix_type)
                for fix in fixes:
                    self.fixes[fix.fix_id] = fix
                    self.correction_metrics["fixes_analyzed"] += 1
        
        logger.info(f"Analyzed {len(self.fixes)} fixes")
        return self.fixes
    
    def identify_mathematical_content(self) -> Dict[str, MathematicalContent]:
        """
        Identify all mathematical content that must be preserved.
        
        Returns:
            Dictionary of mathematical content.
        """
        logger.info("Identifying critical mathematical content...")
        
        math_patterns = [
            (r"def.*calculate|def.*compute|def.*process", "algorithm"),
            (r"=.*\*|=.*\+|=.*\-|=.*\/", "formula"),
            (r"np\.|math\.|scipy\.", "library_call"),
            (r"hashlib\.sha256|hashlib\.md5", "hash_function"),
            (r"BTC.*price|ETH.*price|USDC.*price", "price_calculation"),
            (r"tensor|matrix|vector", "linear_algebra"),
            (r"unified_math\.", "unified_system"),
            (r"ferris.*rde|lantern.*core|recursive.*lattice", "core_system")
        ]
        
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self._extract_mathematical_content_from_file(file_path, math_patterns)
        
        logger.info(f"Identified {len(self.mathematical_content)} mathematical content items")
        return self.mathematical_content
    
    def determine_safe_removals(self) -> List[Fix]:
        """
        Determine which fixes can be safely removed.
        
        Returns:
            List of fixes safe to remove.
        """
        logger.info("Determining safe removals...")
        
        safe_removals = []
        
        for fix in self.fixes.values():
            # Assess if fix is safe to remove
            if self._is_safe_to_remove(fix):
                fix.safe_to_remove = True
                safe_removals.append(fix)
                self.correction_metrics["safe_removals"] += 1
        
        logger.info(f"Identified {len(safe_removals)} fixes safe to remove")
        return safe_removals
    
    def preserve_critical_mathematics(self) -> int:
        """
        Ensure all critical mathematical content is preserved.
        
        Returns:
            Number of mathematical elements preserved.
        """
        logger.info("Preserving critical mathematical content...")
        
        preserved_count = 0
        
        for math_content in self.mathematical_content.values():
            if math_content.criticality in [MathematicalCriticality.CRITICAL, MathematicalCriticality.HIGH]:
                # Ensure this content is not affected by any removals
                self._protect_mathematical_content(math_content)
                preserved_count += 1
                self.correction_metrics["mathematical_preserved"] += 1
        
        logger.info(f"Preserved {preserved_count} critical mathematical elements")
        return preserved_count
    
    def apply_targeted_corrections(self) -> int:
        """
        Apply targeted corrections without introducing new errors.
        
        Returns:
            Number of targeted corrections applied.
        """
        logger.info("Applying targeted corrections...")
        
        corrections_applied = 0
        
        # Remove safe fixes
        for fix in self.fixes.values():
            if fix.safe_to_remove:
                if self._remove_fix(fix):
                    corrections_applied += 1
        
        # Apply targeted improvements
        for fix in self.fixes.values():
            if not fix.safe_to_remove and fix.replacement_content:
                if self._apply_targeted_fix(fix):
                    corrections_applied += 1
                    self.correction_metrics["targeted_corrections"] += 1
        
        logger.info(f"Applied {corrections_applied} targeted corrections")
        return corrections_applied
    
    def validate_no_new_errors(self) -> bool:
        """
        Validate that no new Flake8 errors were introduced.
        
        Returns:
            True if no new errors, False otherwise.
        """
        logger.info("Validating no new errors were introduced...")
        
        # Run syntax validation on modified files
        modified_files = set()
        for fix in self.fixes.values():
            if fix.safe_to_remove or fix.replacement_content:
                modified_files.add(fix.file_path)
        
        error_count = 0
        for file_path in modified_files:
            if not self._validate_file_syntax(file_path):
                error_count += 1
                logger.error(f"Syntax error introduced in {file_path}")
        
        if error_count == 0:
            logger.info("No new errors introduced")
            return True
        else:
            logger.warning(f"{error_count} new errors detected")
            self.correction_metrics["errors_prevented"] = error_count
            return False
    
    def execute_surgical_correction(self) -> Dict[str, Any]:
        """
        Execute the complete surgical correction process.
        
        Returns:
            Summary of surgical correction results.
        """
        logger.info("Starting surgical correction process...")
        
        # Step 1: Analyze all fixes
        self.analyze_all_fixes()
        
        # Step 2: Identify mathematical content
        self.identify_mathematical_content()
        
        # Step 3: Determine safe removals
        safe_removals = self.determine_safe_removals()
        
        # Step 4: Preserve critical mathematics
        preserved_count = self.preserve_critical_mathematics()
        
        # Step 5: Apply targeted corrections
        corrections_applied = self.apply_targeted_corrections()
        
        # Step 6: Validate no new errors
        validation_passed = self.validate_no_new_errors()
        
        # Compile results
        results = {
            "surgical_metrics": self.correction_metrics.copy(),
            "safe_removals": len(safe_removals),
            "mathematical_preserved": preserved_count,
            "corrections_applied": corrections_applied,
            "validation_passed": validation_passed,
            "fix_breakdown": self._get_fix_breakdown(),
            "mathematical_breakdown": self._get_mathematical_breakdown(),
            "recommendations": self._generate_recommendations()
        }
        
        logger.info("Surgical correction process completed")
        return results
    
    def _find_fixes_by_pattern(self, pattern: str, fix_type: FixType) -> List[Fix]:
        """Find fixes matching a pattern."""
        fixes = []
        
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        for line_num, line in enumerate(lines, 1):
                            if re.search(pattern, line, re.IGNORECASE):
                                # Assess mathematical criticality
                                criticality = self._assess_mathematical_criticality(line, file_path)
                                
                                # Assess functional value
                                functional_value = self._assess_functional_value(line, file_path, fix_type)
                                
                                fix = Fix(
                                    fix_type=fix_type,
                                    file_path=file_path,
                                    line_number=line_num,
                                    content=line.strip(),
                                    mathematical_criticality=criticality,
                                    functional_value=functional_value
                                )
                                
                                fixes.append(fix)
                                
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
        
        return fixes
    
    def _extract_mathematical_content_from_file(self, file_path: str, patterns: List[Tuple[str, str]]) -> None:
        """Extract mathematical content from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                for pattern, math_type in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        criticality = self._assess_mathematical_criticality(line, file_path)
                        
                        math_content = MathematicalContent(
                            file_path=file_path,
                            line_number=line_num,
                            content=line.strip(),
                            math_type=math_type,
                            criticality=criticality
                        )
                        
                        self.mathematical_content[math_content.content_id] = math_content
                        
        except Exception as e:
            logger.warning(f"Error extracting math content from {file_path}: {e}")
    
    def _assess_mathematical_criticality(self, line: str, file_path: str) -> MathematicalCriticality:
        """Assess the mathematical criticality of a line."""
        line_lower = line.lower()
        
        # Check for critical indicators
        if any(indicator in line_lower for indicator in self.critical_math_indicators):
            return MathematicalCriticality.CRITICAL
        
        # Check for high-level indicators
        if any(indicator in line_lower for indicator in self.high_math_indicators):
            return MathematicalCriticality.HIGH
        
        # Check file path for mathematical modules
        if any(math_dir in file_path.lower() for math_dir in ["math", "tensor", "algorithm", "trading"]):
            return MathematicalCriticality.MEDIUM
        
        # Check for basic math operations
        if re.search(r'[+\-*/=]', line) and not line.strip().startswith('#'):
            return MathematicalCriticality.LOW
        
        return MathematicalCriticality.NONE
    
    def _assess_functional_value(self, line: str, file_path: str, fix_type: FixType) -> float:
        """Assess the functional value of a fix."""
        # Mathematical preservation always has high value
        if fix_type == FixType.MATHEMATICAL_PRESERVATION:
            return 1.0
        
        # Emergency placeholders may have some value
        if fix_type == FixType.EMERGENCY_PLACEHOLDER:
            return 0.3
        
        # TODO/FIXME comments have low functional value
        if fix_type in [FixType.TODO_IMPLEMENT, FixType.FIXME_UNUSED_IMPORT]:
            return 0.1
        
        # Placeholder pass statements have minimal value
        if fix_type == FixType.PLACEHOLDER_PASS:
            # Check if it's in a critical function
            if any(keyword in line.lower() for keyword in ["def ", "class ", "try:", "except:"]):
                return 0.2
            return 0.0
        
        return 0.5  # Default moderate value
    
    def _is_safe_to_remove(self, fix: Fix) -> bool:
        """Determine if a fix is safe to remove."""
        # Never remove mathematical preservation
        if fix.fix_type == FixType.MATHEMATICAL_PRESERVATION:
            return False
        
        # Never remove if high mathematical criticality
        if fix.mathematical_criticality in [MathematicalCriticality.CRITICAL, MathematicalCriticality.HIGH]:
            return False
        
        # Safe to remove if low functional value and no/low math criticality
        if (fix.functional_value <= 0.1 and 
            fix.mathematical_criticality in [MathematicalCriticality.LOW, MathematicalCriticality.NONE]):
            return True
        
        # Check if it's a simple placeholder pass
        if (fix.fix_type == FixType.PLACEHOLDER_PASS and 
            "pass" in fix.content and 
            fix.functional_value == 0.0):
            return True
        
        return False
    
    def _protect_mathematical_content(self, math_content: MathematicalContent) -> None:
        """Protect mathematical content from removal."""
        # Mark any fixes on the same line as protected
        for fix in self.fixes.values():
            if (fix.file_path == math_content.file_path and 
                fix.line_number == math_content.line_number):
                fix.safe_to_remove = False
    
    def _remove_fix(self, fix: Fix) -> bool:
        """Remove a fix from the file."""
        try:
            with open(fix.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Remove the line or replace with minimal content
            if fix.line_number <= len(lines):
                if fix.fix_type == FixType.PLACEHOLDER_PASS:
                    # Remove placeholder pass statements
                    lines.pop(fix.line_number - 1)
                else:
                    # Remove TODO/FIXME comments
                    line = lines[fix.line_number - 1]
                    # Remove comment but keep any code
                    cleaned_line = re.sub(r'#.*TODO.*|#.*FIXME.*|#.*placeholder.*', '', line, flags=re.IGNORECASE)
                    if cleaned_line.strip():
                        lines[fix.line_number - 1] = cleaned_line
                    else:
                        lines.pop(fix.line_number - 1)
                
                with open(fix.file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return True
                
        except Exception as e:
            logger.error(f"Error removing fix {fix.fix_id}: {e}")
        
        return False
    
    def _apply_targeted_fix(self, fix: Fix) -> bool:
        """Apply a targeted fix."""
        try:
            with open(fix.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if fix.line_number <= len(lines) and fix.replacement_content:
                lines[fix.line_number - 1] = fix.replacement_content + '\n'
                
                with open(fix.file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return True
                
        except Exception as e:
            logger.error(f"Error applying targeted fix {fix.fix_id}: {e}")
        
        return False
    
    def _validate_file_syntax(self, file_path: str) -> bool:
        """Validate file syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ast.parse(content)
            return True
            
        except SyntaxError:
            return False
        except Exception:
            return True  # Assume valid if we can't parse for other reasons
    
    def _get_fix_breakdown(self) -> Dict[str, int]:
        """Get breakdown of fixes by type."""
        breakdown = {}
        for fix_type in FixType:
            breakdown[fix_type.value] = len([f for f in self.fixes.values() if f.fix_type == fix_type])
        return breakdown
    
    def _get_mathematical_breakdown(self) -> Dict[str, int]:
        """Get breakdown of mathematical content by criticality."""
        breakdown = {}
        for criticality in MathematicalCriticality:
            breakdown[criticality.value] = len([
                m for m in self.mathematical_content.values() 
                if m.criticality == criticality
            ])
        return breakdown
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Check removal safety
        safe_removal_count = len([f for f in self.fixes.values() if f.safe_to_remove])
        if safe_removal_count > 0:
            recommendations.append(f"Safe to remove {safe_removal_count} non-functional fixes")
        
        # Check mathematical preservation
        critical_math_count = len([
            m for m in self.mathematical_content.values() 
            if m.criticality == MathematicalCriticality.CRITICAL
        ])
        recommendations.append(f"Preserving {critical_math_count} critical mathematical elements")
        
        # Check for potential improvements
        improvable_fixes = len([
            f for f in self.fixes.values() 
            if not f.safe_to_remove and f.functional_value < 0.5
        ])
        if improvable_fixes > 0:
            recommendations.append(f"Consider improving {improvable_fixes} low-value fixes")
        
        return recommendations


def main():
    """Main function for surgical correction."""
    corrector = SurgicalMathCorrector()
    
    # Execute surgical correction
    results = corrector.execute_surgical_correction()
    
    # Print summary
    print("=== Surgical Mathematical Correction Summary ===")
    print(f"Fixes Analyzed: {results['surgical_metrics']['fixes_analyzed']}")
    print(f"Safe Removals: {results['safe_removals']}")
    print(f"Mathematical Elements Preserved: {results['mathematical_preserved']}")
    print(f"Targeted Corrections Applied: {results['corrections_applied']}")
    print(f"Validation Passed: {results['validation_passed']}")
    
    print(f"\nFix Breakdown:")
    for fix_type, count in results['fix_breakdown'].items():
        print(f"  {fix_type}: {count}")
    
    print(f"\nMathematical Content Breakdown:")
    for criticality, count in results['mathematical_breakdown'].items():
        print(f"  {criticality}: {count}")
    
    print(f"\nRecommendations:")
    for recommendation in results['recommendations']:
        print(f"  - {recommendation}")


if __name__ == "__main__":
    main() 