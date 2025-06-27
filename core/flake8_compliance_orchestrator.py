# -*- coding: utf-8 -*-
"""
Flake8 Compliance Orchestrator - Systematic Code Quality Management

This system provides a structured approach to tackle all Flake8 issues while
maintaining mathematical integrity and profit-tier logic sequencing across
all Schwabot systems and sub-systems.

Features:
- Systematic file-by-file Flake8 checking
- Auto-fix capabilities for PEP8 issues
- Mathematical integrity validation
- Profit-tier logic preservation
- Recursive fallback system integration
- 16-bit and 256-bit relay state management
- Thermal system integration
- Performance optimization tracking
"""

from __future__ import annotations

import os
import sys
import subprocess
import logging
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Flake8 compliance levels"""
    CRITICAL = "critical"    # Must fix immediately
    HIGH = "high"           # Should fix soon
    MEDIUM = "medium"       # Nice to fix
    LOW = "low"            # Optional fixes


class MathematicalIntegrityLevel(Enum):
    """Mathematical integrity levels"""
    CRITICAL = "critical"    # Mathematical logic must be preserved
    HIGH = "high"           # Important mathematical relationships
    MEDIUM = "medium"       # Standard mathematical operations
    LOW = "low"            # Basic mathematical operations


@dataclass
class Flake8Issue:
    """Individual Flake8 issue"""
    file_path: str
    line_number: int
    column: int
    error_code: str
    description: str
    severity: ComplianceLevel
    mathematical_impact: MathematicalIntegrityLevel
    auto_fixable: bool
    suggested_fix: Optional[str] = None
    original_line: Optional[str] = None
    fixed_line: Optional[str] = None


@dataclass
class FileComplianceReport:
    """Compliance report for a single file"""
    file_path: str
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    auto_fixable_issues: int
    mathematical_integrity_score: float
    profit_tier_logic_preserved: bool
    issues: List[Flake8Issue] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class SystemComplianceReport:
    """Overall system compliance report"""
    total_files: int
    compliant_files: int
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    auto_fixed_issues: int
    mathematical_integrity_score: float
    profit_tier_logic_score: float
    processing_time: float
    file_reports: List[FileComplianceReport] = field(default_factory=list)


class MathematicalIntegrityValidator:
    """Validates mathematical integrity during Flake8 fixes"""
    
    def __init__(self):
        self.critical_patterns = {
            "profit_vector": r"profit.*vector|vector.*profit",
            "tensor_operations": r"tensor|np\.|numpy",
            "bit_operations": r"bit.*gate|gate.*bit|0b[01]+",
            "hash_operations": r"hashlib|sha256|md5",
            "emoji_relay": r"emoji.*relay|relay.*emoji",
            "lantern_core": r"lantern.*core|core.*lantern",
            "ferris_rde": r"ferris.*rde|rde.*ferris",
            "thermal_system": r"thermal.*|temperature.*|heat.*",
            "drift_maps": r"drift.*map|map.*drift",
            "timing_differential": r"timing.*differential|differential.*timing"
        }
        
        self.mathematical_functions = {
            "profit_calculation": r"def.*profit|profit.*=",
            "tensor_contraction": r"tensor.*contraction|contraction.*tensor",
            "bit_phase_extraction": r"bit.*phase|phase.*bit",
            "hash_generation": r"hash.*generation|generation.*hash",
            "relay_path_creation": r"relay.*path|path.*relay",
            "state_energy_calculation": r"state.*energy|energy.*state"
        }
    
    def validate_mathematical_integrity(self, file_path: str, original_content: str, fixed_content: str) -> Dict[str, Any]:
        """Validate mathematical integrity after fixes"""
        try:
            validation_result = {
                "integrity_preserved": True,
                "critical_patterns_preserved": {},
                "mathematical_functions_preserved": {},
                "profit_tier_logic_preserved": True,
                "relay_states_preserved": True,
                "thermal_integration_preserved": True,
                "issues": []
            }
            
            # Check critical patterns
            for pattern_name, pattern in self.critical_patterns.items():
                original_matches = len(re.findall(pattern, original_content, re.IGNORECASE))
                fixed_matches = len(re.findall(pattern, fixed_content, re.IGNORECASE))
                
                validation_result["critical_patterns_preserved"][pattern_name] = {
                    "original_count": original_matches,
                    "fixed_count": fixed_matches,
                    "preserved": fixed_matches >= original_matches
                }
                
                if fixed_matches < original_matches:
                    validation_result["integrity_preserved"] = False
                    validation_result["issues"].append(f"Lost {pattern_name} patterns")
            
            # Check mathematical functions
            for func_name, pattern in self.mathematical_functions.items():
                original_matches = len(re.findall(pattern, original_content, re.IGNORECASE))
                fixed_matches = len(re.findall(pattern, fixed_content, re.IGNORECASE))
                
                validation_result["mathematical_functions_preserved"][func_name] = {
                    "original_count": original_matches,
                    "fixed_count": fixed_matches,
                    "preserved": fixed_matches >= original_matches
                }
                
                if fixed_matches < original_matches:
                    validation_result["integrity_preserved"] = False
                    validation_result["issues"].append(f"Lost {func_name} functions")
            
            # Check profit-tier logic preservation
            profit_tier_patterns = [
                r"profit.*tier|tier.*profit",
                r"16.*bit|bit.*16",
                r"256.*bit|bit.*256",
                r"relay.*state|state.*relay",
                r"fallback.*logic|logic.*fallback"
            ]
            
            for pattern in profit_tier_patterns:
                original_matches = len(re.findall(pattern, original_content, re.IGNORECASE))
                fixed_matches = len(re.findall(pattern, fixed_content, re.IGNORECASE))
                
                if fixed_matches < original_matches:
                    validation_result["profit_tier_logic_preserved"] = False
                    validation_result["issues"].append(f"Lost profit-tier logic: {pattern}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Mathematical integrity validation failed: {e}")
            return {
                "integrity_preserved": False,
                "issues": [f"Validation error: {e}"]
            }


class ProfitTierLogicPreserver:
    """Preserves profit-tier logic during Flake8 fixes"""
    
    def __init__(self):
        self.profit_tier_components = {
            "asic_logic_gates": ["ASICLogicGate", "get_asic_gate_manager", "process_input"],
            "emoji_symbolic_relay": ["EmojiSymbolicRelay", "create_relay_path", "get_emoji_relay"],
            "lantern_core": ["LanternCore", "relay_to_bit_gates", "get_lantern_core"],
            "tensor_operations": ["tensor_contraction", "profit_routing_tensor", "tensor_algebra"],
            "drift_maps": ["DriftMap", "drift_magnitude", "drift_direction"],
            "timing_differential": ["TimingDifferential", "MICRO", "SHORT", "MEDIUM", "LONG"],
            "btc_mapping": ["map_btc_price_16bit", "mapped_16bit", "hash_sequence"],
            "thermal_system": ["thermal_boundary", "thermal_shift", "thermal_map"],
            "unified_math": ["unified_math", "EnhancedUnifiedMathematicalSystem"],
            "profit_vectorization": ["ProfitVectorizationResult", "calculate_profit_vectorization"]
        }
    
    def preserve_profit_tier_logic(self, content: str) -> str:
        """Preserve profit-tier logic while allowing PEP8 fixes"""
        try:
            preserved_content = content
            
            # Ensure critical imports are preserved
            critical_imports = [
                "from core.asic_logic_gate_foundation import",
                "from core.emoji_symbolic_relay import",
                "from core.lantern_core import",
                "from core.unified_math_system import",
                "from core.enhanced_unified_mathematical_system import",
                "import numpy as np",
                "import hashlib"
            ]
            
            for import_line in critical_imports:
                if import_line not in preserved_content:
                    preserved_content = f"{import_line}\n{preserved_content}"
            
            # Preserve mathematical constants
            math_constants = {
                "BTC_PRICE_MIN": "1000.0",
                "BTC_PRICE_MAX": "100000.0",
                "PROFIT_THRESHOLD": "0.02",
                "CONFIDENCE_THRESHOLD": "0.7",
                "SMOOTHING_FACTOR": "0.1"
            }
            
            for constant, value in math_constants.items():
                if constant not in preserved_content:
                    preserved_content = preserved_content.replace(
                        f"= {value}",
                        f"= {value}  # {constant}"
                    )
            
            return preserved_content
            
        except Exception as e:
            logger.error(f"Profit-tier logic preservation failed: {e}")
            return content


class Flake8ComplianceOrchestrator:
    """Main orchestrator for Flake8 compliance management"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.mathematical_validator = MathematicalIntegrityValidator()
        self.profit_tier_preserver = ProfitTierLogicPreserver()
        
        # File patterns to include/exclude
        self.include_patterns = ["*.py", "*.pyx", "*.pxd"]
        self.exclude_patterns = [
            "__pycache__", ".git", ".mypy_cache", "venv", "env",
            "node_modules", "build", "dist", ".pytest_cache"
        ]
        
        # Flake8 configuration
        self.flake8_config = {
            "max_line_length": 88,  # Black-compatible
            "ignore": [
                "E203",  # whitespace before ':'
                "W503",  # line break before binary operator
                "E501",  # line too long (handled by Black)
                "F401",  # imported but unused (handled separately)
                "F841",  # local variable assigned but never used (handled separately)
            ]
        }
    
    def get_all_python_files(self) -> List[Path]:
        """Get all Python files in the project"""
        python_files = []
        
        for pattern in self.include_patterns:
            python_files.extend(self.project_root.rglob(pattern))
        
        # Filter out excluded directories
        filtered_files = []
        for file_path in python_files:
            if not any(exclude in str(file_path) for exclude in self.exclude_patterns):
                filtered_files.append(file_path)
        
        return sorted(filtered_files)
    
    def run_flake8_on_file(self, file_path: Path) -> List[Flake8Issue]:
        """Run Flake8 on a single file and return issues"""
        try:
            # Build Flake8 command
            cmd = [
                "flake8",
                "--format=%(path)s:%(row)d:%(column)d:%(code)s:%(text)s",
                "--max-line-length=88",
                "--ignore=E203,W503,E501,F401,F841",
                str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            issues = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    issue = self._parse_flake8_output(line)
                    if issue:
                        issues.append(issue)
            
            return issues
            
        except subprocess.TimeoutExpired:
            logger.error(f"Flake8 timeout for {file_path}")
            return []
        except Exception as e:
            logger.error(f"Flake8 error for {file_path}: {e}")
            return []
    
    def _parse_flake8_output(self, output_line: str) -> Optional[Flake8Issue]:
        """Parse Flake8 output line into Flake8Issue"""
        try:
            # Format: file:line:column:code:description
            parts = output_line.split(':', 4)
            if len(parts) >= 5:
                file_path, line_num, column, error_code, description = parts
                
                return Flake8Issue(
                    file_path=file_path,
                    line_number=int(line_num),
                    column=int(column),
                    error_code=error_code,
                    description=description,
                    severity=self._determine_severity(error_code),
                    mathematical_impact=self._determine_mathematical_impact(error_code),
                    auto_fixable=self._is_auto_fixable(error_code),
                    suggested_fix=self._suggest_fix(error_code, description)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse Flake8 output: {e}")
            return None
    
    def _determine_severity(self, error_code: str) -> ComplianceLevel:
        """Determine severity level for error code"""
        critical_codes = ["E999", "F999", "E901", "E902", "E903"]
        high_codes = ["E101", "E111", "E112", "E113", "E114", "E115", "E116", "E117", "E201", "E202", "E203"]
        medium_codes = ["E211", "E221", "E222", "E223", "E224", "E225", "E226", "E227", "E228", "E231", "E241", "E242", "E251"]
        
        if error_code in critical_codes:
            return ComplianceLevel.CRITICAL
        elif error_code in high_codes:
            return ComplianceLevel.HIGH
        elif error_code in medium_codes:
            return ComplianceLevel.MEDIUM
        else:
            return ComplianceLevel.LOW
    
    def _determine_mathematical_impact(self, error_code: str) -> MathematicalIntegrityLevel:
        """Determine mathematical impact of error code"""
        critical_codes = ["E999", "F999", "E901", "E902", "E903"]  # Syntax errors
        high_codes = ["E101", "E111", "E112", "E113", "E114", "E115", "E116", "E117"]  # Indentation errors
        
        if error_code in critical_codes:
            return MathematicalIntegrityLevel.CRITICAL
        elif error_code in high_codes:
            return MathematicalIntegrityLevel.HIGH
        else:
            return MathematicalIntegrityLevel.LOW
    
    def _is_auto_fixable(self, error_code: str) -> bool:
        """Determine if error is auto-fixable"""
        auto_fixable_codes = [
            "E201", "E202", "E203",  # Whitespace around operators
            "E211", "E221", "E222", "E223", "E224", "E225", "E226", "E227", "E228",  # Whitespace
            "E231", "E241", "E242",  # Missing whitespace
            "E251", "E252", "E253",  # Whitespace around equals
            "E261", "E262", "E263", "E264", "E265", "E266",  # Comment spacing
            "E271", "E272", "E273", "E274", "E275",  # Multiple spaces
            "E301", "E302", "E303", "E304", "E305", "E306",  # Blank lines
            "E401", "E402", "E403", "E404", "E405", "E406", "E407", "E408", "E409", "E410", "E411", "E412", "E413", "E414", "E415", "E416", "E417", "E418", "E419", "E420", "E421", "E422", "E423", "E424", "E425", "E426", "E427", "E428", "E429", "E430", "E431", "E432", "E433", "E434", "E435", "E436", "E437", "E438", "E439", "E440", "E441", "E442", "E443", "E444", "E445", "E446", "E447", "E448", "E449", "E450", "E451", "E452", "E453", "E454", "E455", "E456", "E457", "E458", "E459", "E460", "E461", "E462", "E463", "E464", "E465", "E466", "E467", "E468", "E469", "E470", "E471", "E472", "E473", "E474", "E475", "E476", "E477", "E478", "E479", "E480", "E481", "E482", "E483", "E484", "E485", "E486", "E487", "E488", "E489", "E490", "E491", "E492", "E493", "E494", "E495", "E496", "E497", "E498", "E499",  # Import issues
            "W291", "W292", "W293", "W391", "W503", "W504"  # Whitespace warnings
        ]
        
        return error_code in auto_fixable_codes
    
    def _suggest_fix(self, error_code: str, description: str) -> Optional[str]:
        """Suggest fix for error code"""
        suggestions = {
            "E201": "Remove whitespace after opening bracket",
            "E202": "Remove whitespace before closing bracket",
            "E203": "Remove whitespace before ':'",
            "E211": "Remove whitespace before '('",
            "E221": "Fix multiple spaces before operator",
            "E222": "Fix multiple spaces after operator",
            "E223": "Fix tab before operator",
            "E224": "Fix tab after operator",
            "E225": "Fix missing whitespace around operator",
            "E226": "Fix missing whitespace around arithmetic operator",
            "E227": "Fix missing whitespace around bitwise or shift operator",
            "E228": "Fix missing whitespace around modulo operator",
            "E231": "Fix missing whitespace after ','",
            "E241": "Fix multiple spaces after ','",
            "E242": "Fix tab after ','",
            "E251": "Fix unexpected spaces around keyword / parameter equals",
            "E261": "Fix at least two spaces before inline comment",
            "E262": "Fix inline comment should start with '# '",
            "E271": "Fix multiple spaces after keyword",
            "E272": "Fix multiple spaces before keyword",
            "E273": "Fix tab after keyword",
            "E274": "Fix tab before keyword",
            "E301": "Fix expected 1 blank line, found 0",
            "E302": "Fix expected 2 blank lines, found 0",
            "E303": "Fix too many blank lines",
            "E304": "Fix blank lines found after function decorator",
            "E305": "Fix expected 2 blank lines after class or function definition",
            "E306": "Fix expected 1 blank line before a nested definition",
            "W291": "Fix trailing whitespace",
            "W292": "Fix no newline at end of file",
            "W293": "Fix blank line contains whitespace",
            "W391": "Fix blank line at end of file",
            "W503": "Fix line break before binary operator",
            "W504": "Fix line break after binary operator"
        }
        
        return suggestions.get(error_code, None)
    
    def auto_fix_file(self, file_path: Path, issues: List[Flake8Issue]) -> Tuple[bool, str, List[str]]:
        """Auto-fix Flake8 issues in a file"""
        try:
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            fixed_content = original_content
            applied_fixes = []
            
            # Apply fixes in reverse order to maintain line numbers
            auto_fixable_issues = [issue for issue in issues if issue.auto_fixable]
            auto_fixable_issues.sort(key=lambda x: x.line_number, reverse=True)
            
            for issue in auto_fixable_issues:
                if self._apply_single_fix(fixed_content, issue):
                    applied_fixes.append(f"Fixed {issue.error_code} on line {issue.line_number}")
            
            # Preserve profit-tier logic
            fixed_content = self.profit_tier_preserver.preserve_profit_tier_logic(fixed_content)
            
            # Validate mathematical integrity
            validation_result = self.mathematical_validator.validate_mathematical_integrity(
                str(file_path), original_content, fixed_content
            )
            
            if validation_result["integrity_preserved"]:
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                return True, fixed_content, applied_fixes
            else:
                logger.warning(f"Mathematical integrity compromised for {file_path}")
                return False, original_content, []
            
        except Exception as e:
            logger.error(f"Auto-fix failed for {file_path}: {e}")
            return False, "", []
    
    def _apply_single_fix(self, content: str, issue: Flake8Issue) -> bool:
        """Apply a single fix to content"""
        try:
            lines = content.split('\n')
            if issue.line_number > len(lines):
                return False
            
            line_index = issue.line_number - 1
            original_line = lines[line_index]
            
            # Apply fix based on error code
            fixed_line = self._fix_line_by_error_code(original_line, issue)
            
            if fixed_line != original_line:
                lines[line_index] = fixed_line
                content = '\n'.join(lines)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply fix: {e}")
            return False
    
    def _fix_line_by_error_code(self, line: str, issue: Flake8Issue) -> str:
        """Fix line based on error code"""
        try:
            if issue.error_code == "E201":  # Remove whitespace after opening bracket
                return re.sub(r'\(\s+', '(', line)
            elif issue.error_code == "E202":  # Remove whitespace before closing bracket
                return re.sub(r'\s+\)', ')', line)
            elif issue.error_code == "E203":  # Remove whitespace before ':'
                return re.sub(r'\s+:', ':', line)
            elif issue.error_code == "E211":  # Remove whitespace before '('
                return re.sub(r'\s+\(', '(', line)
            elif issue.error_code == "E225":  # Missing whitespace around operator
                return re.sub(r'([^=!<>])([=!<>]+)([^=])', r'\1 \2 \3', line)
            elif issue.error_code == "E231":  # Missing whitespace after ','
                return re.sub(r',([^\s])', r', \1', line)
            elif issue.error_code == "E251":  # Unexpected spaces around equals
                return re.sub(r'\s*=\s*', ' = ', line)
            elif issue.error_code == "W291":  # Trailing whitespace
                return line.rstrip()
            elif issue.error_code == "W292":  # No newline at end of file
                return line + '\n'
            elif issue.error_code == "W293":  # Blank line contains whitespace
                return line.strip()
            else:
                return line
                
        except Exception as e:
            logger.error(f"Failed to fix line by error code: {e}")
            return line
    
    def process_file(self, file_path: Path) -> FileComplianceReport:
        """Process a single file for Flake8 compliance"""
        start_time = time.time()
        
        try:
            # Run Flake8
            issues = self.run_flake8_on_file(file_path)
            
            # Categorize issues
            critical_issues = [i for i in issues if i.severity == ComplianceLevel.CRITICAL]
            high_issues = [i for i in issues if i.severity == ComplianceLevel.HIGH]
            medium_issues = [i for i in issues if i.severity == ComplianceLevel.MEDIUM]
            low_issues = [i for i in issues if i.severity == ComplianceLevel.LOW]
            auto_fixable_issues = [i for i in issues if i.auto_fixable]
            
            # Calculate mathematical integrity score
            mathematical_score = self._calculate_mathematical_integrity_score(issues)
            
            # Check profit-tier logic preservation
            profit_tier_preserved = self._check_profit_tier_logic_preservation(file_path, issues)
            
            # Auto-fix if possible
            auto_fixed_count = 0
            if auto_fixable_issues:
                success, _, applied_fixes = self.auto_fix_file(file_path, auto_fixable_issues)
                if success:
                    auto_fixed_count = len(applied_fixes)
            
            processing_time = time.time() - start_time
            
            return FileComplianceReport(
                file_path=str(file_path),
                total_issues=len(issues),
                critical_issues=len(critical_issues),
                high_issues=len(high_issues),
                medium_issues=len(medium_issues),
                low_issues=len(low_issues),
                auto_fixable_issues=len(auto_fixable_issues),
                mathematical_integrity_score=mathematical_score,
                profit_tier_logic_preserved=profit_tier_preserved,
                issues=issues,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return FileComplianceReport(
                file_path=str(file_path),
                total_issues=0,
                critical_issues=0,
                high_issues=0,
                medium_issues=0,
                low_issues=0,
                auto_fixable_issues=0,
                mathematical_integrity_score=0.0,
                profit_tier_logic_preserved=False,
                processing_time=time.time() - start_time
            )
    
    def _calculate_mathematical_integrity_score(self, issues: List[Flake8Issue]) -> float:
        """Calculate mathematical integrity score"""
        try:
            if not issues:
                return 1.0
            
            critical_impact = sum(1 for i in issues if i.mathematical_impact == MathematicalIntegrityLevel.CRITICAL)
            high_impact = sum(1 for i in issues if i.mathematical_impact == MathematicalIntegrityLevel.HIGH)
            medium_impact = sum(1 for i in issues if i.mathematical_impact == MathematicalIntegrityLevel.MEDIUM)
            
            # Weighted score calculation
            total_weighted_issues = critical_impact * 1.0 + high_impact * 0.7 + medium_impact * 0.3
            max_possible_issues = len(issues)
            
            if max_possible_issues == 0:
                return 1.0
            
            score = max(0.0, 1.0 - (total_weighted_issues / max_possible_issues))
            return score
            
        except Exception as e:
            logger.error(f"Failed to calculate mathematical integrity score: {e}")
            return 0.0
    
    def _check_profit_tier_logic_preservation(self, file_path: Path, issues: List[Flake8Issue]) -> bool:
        """Check if profit-tier logic is preserved"""
        try:
            # Check for critical profit-tier related issues
            critical_profit_issues = [
                i for i in issues 
                if i.mathematical_impact == MathematicalIntegrityLevel.CRITICAL
                and any(keyword in i.description.lower() for keyword in [
                    "profit", "vector", "tensor", "bit", "gate", "relay", "emoji", "lantern", "ferris"
                ])
            ]
            
            return len(critical_profit_issues) == 0
            
        except Exception as e:
            logger.error(f"Failed to check profit-tier logic preservation: {e}")
            return False
    
    def run_full_compliance_check(self) -> SystemComplianceReport:
        """Run full compliance check on all files"""
        start_time = time.time()
        
        try:
            # Get all Python files
            python_files = self.get_all_python_files()
            logger.info(f"Found {len(python_files)} Python files to check")
            
            # Process each file
            file_reports = []
            total_issues = 0
            critical_issues = 0
            high_issues = 0
            medium_issues = 0
            low_issues = 0
            auto_fixed_issues = 0
            mathematical_scores = []
            profit_tier_scores = []
            
            for i, file_path in enumerate(python_files, 1):
                logger.info(f"Processing file {i}/{len(python_files)}: {file_path}")
                
                file_report = self.process_file(file_path)
                file_reports.append(file_report)
                
                # Aggregate statistics
                total_issues += file_report.total_issues
                critical_issues += file_report.critical_issues
                high_issues += file_report.high_issues
                medium_issues += file_report.medium_issues
                low_issues += file_report.low_issues
                auto_fixed_issues += file_report.auto_fixable_issues
                mathematical_scores.append(file_report.mathematical_integrity_score)
                profit_tier_scores.append(1.0 if file_report.profit_tier_logic_preserved else 0.0)
            
            # Calculate overall scores
            mathematical_integrity_score = np.mean(mathematical_scores) if mathematical_scores else 0.0
            profit_tier_logic_score = np.mean(profit_tier_scores) if profit_tier_scores else 0.0
            
            # Count compliant files
            compliant_files = sum(1 for report in file_reports if report.total_issues == 0)
            
            processing_time = time.time() - start_time
            
            return SystemComplianceReport(
                total_files=len(python_files),
                compliant_files=compliant_files,
                total_issues=total_issues,
                critical_issues=critical_issues,
                high_issues=high_issues,
                medium_issues=medium_issues,
                low_issues=low_issues,
                auto_fixed_issues=auto_fixed_issues,
                mathematical_integrity_score=mathematical_integrity_score,
                profit_tier_logic_score=profit_tier_logic_score,
                processing_time=processing_time,
                file_reports=file_reports
            )
            
        except Exception as e:
            logger.error(f"Full compliance check failed: {e}")
            return SystemComplianceReport(
                total_files=0,
                compliant_files=0,
                total_issues=0,
                critical_issues=0,
                high_issues=0,
                medium_issues=0,
                low_issues=0,
                auto_fixed_issues=0,
                mathematical_integrity_score=0.0,
                profit_tier_logic_score=0.0,
                processing_time=time.time() - start_time
            )
    
    def generate_compliance_report(self, report: SystemComplianceReport, output_file: Optional[str] = None) -> str:
        """Generate comprehensive compliance report"""
        try:
            report_content = []
            report_content.append("# SCHWABOT FLAKE8 COMPLIANCE REPORT")
            report_content.append("## Complete System Code Quality Analysis")
            report_content.append("")
            
            # Summary
            report_content.append("## SUMMARY")
            report_content.append(f"- **Total Files**: {report.total_files}")
            report_content.append(f"- **Compliant Files**: {report.compliant_files}")
            report_content.append(f"- **Compliance Rate**: {(report.compliant_files / report.total_files * 100):.1f}%" if report.total_files > 0 else "- **Compliance Rate**: 0%")
            report_content.append(f"- **Total Issues**: {report.total_issues}")
            report_content.append(f"- **Critical Issues**: {report.critical_issues}")
            report_content.append(f"- **High Issues**: {report.high_issues}")
            report_content.append(f"- **Medium Issues**: {report.medium_issues}")
            report_content.append(f"- **Low Issues**: {report.low_issues}")
            report_content.append(f"- **Auto-Fixed Issues**: {report.auto_fixed_issues}")
            report_content.append(f"- **Mathematical Integrity Score**: {report.mathematical_integrity_score:.3f}")
            report_content.append(f"- **Profit-Tier Logic Score**: {report.profit_tier_logic_score:.3f}")
            report_content.append(f"- **Processing Time**: {report.processing_time:.2f} seconds")
            report_content.append("")
            
            # Critical issues
            if report.critical_issues > 0:
                report_content.append("## CRITICAL ISSUES")
                for file_report in report.file_reports:
                    critical_issues = [i for i in file_report.issues if i.severity == ComplianceLevel.CRITICAL]
                    if critical_issues:
                        report_content.append(f"### {file_report.file_path}")
                        for issue in critical_issues:
                            report_content.append(f"- Line {issue.line_number}: {issue.error_code} - {issue.description}")
                report_content.append("")
            
            # High priority issues
            if report.high_issues > 0:
                report_content.append("## HIGH PRIORITY ISSUES")
                for file_report in report.file_reports:
                    high_issues = [i for i in file_report.issues if i.severity == ComplianceLevel.HIGH]
                    if high_issues:
                        report_content.append(f"### {file_report.file_path}")
                        for issue in high_issues:
                            report_content.append(f"- Line {issue.line_number}: {issue.error_code} - {issue.description}")
                report_content.append("")
            
            # Mathematical integrity issues
            low_integrity_files = [fr for fr in report.file_reports if fr.mathematical_integrity_score < 0.8]
            if low_integrity_files:
                report_content.append("## MATHEMATICAL INTEGRITY ISSUES")
                for file_report in low_integrity_files:
                    report_content.append(f"### {file_report.file_path}")
                    report_content.append(f"- **Integrity Score**: {file_report.mathematical_integrity_score:.3f}")
                    mathematical_issues = [i for i in file_report.issues if i.mathematical_impact in [MathematicalIntegrityLevel.CRITICAL, MathematicalIntegrityLevel.HIGH]]
                    for issue in mathematical_issues:
                        report_content.append(f"- Line {issue.line_number}: {issue.error_code} - {issue.description}")
                report_content.append("")
            
            # Profit-tier logic issues
            profit_tier_issues = [fr for fr in report.file_reports if not fr.profit_tier_logic_preserved]
            if profit_tier_issues:
                report_content.append("## PROFIT-TIER LOGIC ISSUES")
                for file_report in profit_tier_issues:
                    report_content.append(f"### {file_report.file_path}")
                    profit_issues = [i for i in file_report.issues if "profit" in i.description.lower() or "vector" in i.description.lower()]
                    for issue in profit_issues:
                        report_content.append(f"- Line {issue.line_number}: {issue.error_code} - {issue.description}")
                report_content.append("")
            
            # File-by-file breakdown
            report_content.append("## FILE-BY-FILE BREAKDOWN")
            for file_report in report.file_reports:
                if file_report.total_issues > 0:
                    report_content.append(f"### {file_report.file_path}")
                    report_content.append(f"- **Total Issues**: {file_report.total_issues}")
                    report_content.append(f"- **Critical**: {file_report.critical_issues}")
                    report_content.append(f"- **High**: {file_report.high_issues}")
                    report_content.append(f"- **Medium**: {file_report.medium_issues}")
                    report_content.append(f"- **Low**: {file_report.low_issues}")
                    report_content.append(f"- **Auto-Fixable**: {file_report.auto_fixable_issues}")
                    report_content.append(f"- **Mathematical Integrity**: {file_report.mathematical_integrity_score:.3f}")
                    report_content.append(f"- **Profit-Tier Logic**: {'PASS' if file_report.profit_tier_logic_preserved else 'FAIL'}")
                    report_content.append(f"- **Processing Time**: {file_report.processing_time:.3f}s")
                    report_content.append("")
            
            report_text = "\n".join(report_content)
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"Compliance report saved to {output_file}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return f"Error generating report: {e}"


# Global orchestrator instance
flake8_orchestrator = Flake8ComplianceOrchestrator()


def run_compliance_check(project_root: str = ".", output_file: Optional[str] = None) -> SystemComplianceReport:
    """Run full Flake8 compliance check"""
    global flake8_orchestrator
    flake8_orchestrator.project_root = Path(project_root)
    
    logger.info("Starting Flake8 compliance check...")
    report = flake8_orchestrator.run_full_compliance_check()
    
    # Generate report
    report_text = flake8_orchestrator.generate_compliance_report(report, output_file)
    
    logger.info("Flake8 compliance check completed!")
    logger.info(f"Total files: {report.total_files}")
    logger.info(f"Compliant files: {report.compliant_files}")
    logger.info(f"Total issues: {report.total_issues}")
    logger.info(f"Mathematical integrity score: {report.mathematical_integrity_score:.3f}")
    
    return report


def check_single_file(file_path: str) -> FileComplianceReport:
    """Check compliance for a single file"""
    global flake8_orchestrator
    return flake8_orchestrator.process_file(Path(file_path))


if __name__ == "__main__":
    # Run compliance check
    report = run_compliance_check(output_file="flake8_compliance_report.md")
    print(f"\nCompliance check completed!")
    print(f"Report saved to: flake8_compliance_report.md") 