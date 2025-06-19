"""
Comprehensive Windows CLI Compliance Audit
==========================================

This script audits all hybrid optimization files to ensure they meet the Windows CLI 
compatibility standards outlined in WINDOWS_CLI_COMPATIBILITY.md:

✅ No bare except statements (structured error handling only)
✅ Complete type annotations for all functions
✅ No wildcard imports (explicit imports only)  
✅ Windows CLI emoji compatibility handling
✅ Proper naming conventions (descriptive, not generic)
✅ Mathematical optimization components properly implemented

This is the final verification that our dual pipeline hybrid optimization system
follows all established coding standards and compatibility requirements.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class ComplianceIssue:
    """Represents a compliance issue found during audit"""
    file_path: str
    line_number: int
    issue_type: str
    description: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    recommendation: str

class WindowsCliComplianceAuditor:
    """
    Comprehensive auditor for Windows CLI compatibility and coding standards
    """
    
    def __init__(self) -> None:
        self.issues: List[ComplianceIssue] = []
        self.files_audited: int = 0
        self.emoji_pattern = re.compile(r'[🚨⚠️✅❌🔄💰📊🔧🎯⚡🔍📈🧠🛡️🔥🎉🌟🖥️🎮🖼️🌀📡💫🎨🧮🌡️⏱️⚙️💡🧪📋🔱]')
        
        # Files to audit (hybrid optimization system)
        self.target_files = [
            'core/hybrid_optimization_manager.py',
            'core/magic_number_optimization_engine.py', 
            'core/optimized_constants_wrapper.py',
            'demo_hybrid_dual_pipeline_showcase.py',
            'demo_magic_number_optimization_revolution.py',
            'demo_revolutionary_optimization_showcase.py',
            'test_magic_number_optimization.py'
        ]
    
    def safe_print(self, message: str) -> None:
        """Safe print with Windows CLI compatibility"""
        if os.name == 'nt' and not os.environ.get('TERM_PROGRAM'):
            # Replace emojis for Windows CLI
            message = re.sub(self.emoji_pattern, lambda m: f'[{m.group(0)}]', message)
        print(message)
    
    def audit_file(self, file_path: str) -> None:
        """Audit a single file for compliance issues"""
        if not Path(file_path).exists():
            self.issues.append(ComplianceIssue(
                file_path=file_path,
                line_number=0,
                issue_type="FILE_NOT_FOUND",
                description=f"Target file does not exist: {file_path}",
                severity="HIGH",
                recommendation="Ensure file exists or remove from audit list"
            ))
            return
        
        self.files_audited += 1
        self.safe_print(f"🔍 Auditing: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Parse AST for advanced analysis
            try:
                tree = ast.parse(content)
                self._audit_ast(file_path, tree)
            except SyntaxError as e:
                self.issues.append(ComplianceIssue(
                    file_path=file_path,
                    line_number=e.lineno or 0,
                    issue_type="SYNTAX_ERROR",
                    description=f"Syntax error prevents analysis: {e}",
                    severity="CRITICAL",
                    recommendation="Fix syntax error"
                ))
                return
            
            # Line-by-line analysis
            self._audit_lines(file_path, lines)
            
        except Exception as e:
            self.issues.append(ComplianceIssue(
                file_path=file_path,
                line_number=0,
                issue_type="READ_ERROR",
                description=f"Could not read file: {e}",
                severity="HIGH",
                recommendation="Check file permissions and encoding"
            ))
    
    def _audit_ast(self, file_path: str, tree: ast.AST) -> None:
        """Audit using AST analysis"""
        for node in ast.walk(tree):
            # Check for bare except statements
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:  # Bare except
                    self.issues.append(ComplianceIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type="BARE_EXCEPT",
                        description="Bare except statement found - catches all exceptions including SystemExit",
                        severity="CRITICAL",
                        recommendation="Replace with specific exception handling: except Exception as e:"
                    ))
            
            # Check for wildcard imports
            elif isinstance(node, ast.ImportFrom):
                if any(alias.name == '*' for alias in node.names):
                    self.issues.append(ComplianceIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type="WILDCARD_IMPORT",
                        description=f"Wildcard import found: from {node.module} import *",
                        severity="CRITICAL",
                        recommendation="Replace with specific imports"
                    ))
            
            # Check function type annotations
            elif isinstance(node, ast.FunctionDef):
                # Check return type annotation
                if node.returns is None and node.name != '__init__':
                    self.issues.append(ComplianceIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type="MISSING_RETURN_TYPE",
                        description=f"Function '{node.name}' missing return type annotation",
                        severity="MEDIUM",
                        recommendation="Add return type annotation: def func() -> ReturnType:"
                    ))
                
                # Check parameter type annotations
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != 'self':
                        self.issues.append(ComplianceIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            issue_type="MISSING_PARAM_TYPE",
                            description=f"Parameter '{arg.arg}' in function '{node.name}' missing type annotation",
                            severity="MEDIUM", 
                            recommendation="Add parameter type annotation: param: Type"
                        ))
    
    def _audit_lines(self, file_path: str, lines: List[str]) -> None:
        """Audit individual lines for compliance issues"""
        has_cli_compatibility = False
        
        for line_num, line in enumerate(lines, 1):
            # Check for emoji usage without Windows CLI handling
            if self.emoji_pattern.search(line):
                # Check if it's in a string that might be displayed
                if any(keyword in line for keyword in ['print', 'logger', 'log_', 'message']):
                    # Check if there's Windows CLI handling nearby
                    context_lines = lines[max(0, line_num-10):min(len(lines), line_num+10)]
                    if not any('WindowsCliCompatibility' in ctx_line or 'safe_' in ctx_line 
                             for ctx_line in context_lines):
                        self.issues.append(ComplianceIssue(
                            file_path=file_path,
                            line_number=line_num,
                            issue_type="EMOJI_WITHOUT_CLI_HANDLING",
                            description="Emoji usage without Windows CLI compatibility handling",
                            severity="MEDIUM",
                            recommendation="Use WindowsCliCompatibilityHandler.safe_log_message() or safe_print()"
                        ))
            
            # Check for Windows CLI compatibility implementation
            if 'WindowsCliCompatibility' in line or 'is_windows_cli' in line:
                has_cli_compatibility = True
            
            # Check for generic naming patterns
            if re.search(r'\b(test1|gap1|fix1|temp|tmp)\b', line) and 'def ' in line:
                self.issues.append(ComplianceIssue(
                    file_path=file_path,
                    line_number=line_num,
                    issue_type="GENERIC_NAMING",
                    description="Generic function/variable naming found",
                    severity="LOW",
                    recommendation="Use descriptive names that explain function purpose"
                ))
        
        # Check if file needs Windows CLI compatibility
        has_emojis = any(self.emoji_pattern.search(line) for line in lines)
        if has_emojis and not has_cli_compatibility and 'test_' not in file_path:
            self.issues.append(ComplianceIssue(
                file_path=file_path,
                line_number=0,
                issue_type="MISSING_CLI_COMPATIBILITY",
                description="File contains emojis but lacks Windows CLI compatibility handling",
                severity="HIGH",
                recommendation="Add WindowsCliCompatibilityHandler or use safe_print functions"
            ))
    
    def check_mathematical_components(self) -> None:
        """Verify mathematical optimization components are properly implemented"""
        required_components = {
            'core/magic_number_optimization_engine.py': [
                'golden_ratio', 'fibonacci', 'thermal', 'sustainment', 'PHI', 'OptimizationType'
            ],
            'core/hybrid_optimization_manager.py': [
                'ProcessingContext', 'OptimizationMode', 'get_smart_constant', 'dual_pipeline'
            ],
            'core/optimized_constants_wrapper.py': [
                'OPTIMIZED_CONSTANTS', 'enable_optimizations', 'optimization_status'
            ]
        }
        
        for file_path, components in required_components.items():
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for component in components:
                        if component.lower() not in content.lower():
                            self.issues.append(ComplianceIssue(
                                file_path=file_path,
                                line_number=0,
                                issue_type="MISSING_MATH_COMPONENT",
                                description=f"Required mathematical component '{component}' not found",
                                severity="HIGH",
                                recommendation=f"Implement {component} for complete optimization system"
                            ))
                except Exception as e:
                    self.issues.append(ComplianceIssue(
                        file_path=file_path,
                        line_number=0,
                        issue_type="MATH_COMPONENT_CHECK_FAILED",
                        description=f"Could not verify mathematical components: {e}",
                        severity="MEDIUM",
                        recommendation="Manually verify mathematical optimization components"
                    ))
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run complete compliance audit"""
        self.safe_print("🚀 STARTING COMPREHENSIVE WINDOWS CLI COMPLIANCE AUDIT")
        self.safe_print("=" * 60)
        self.safe_print("")
        
        # Audit each target file
        for file_path in self.target_files:
            self.audit_file(file_path)
        
        # Check mathematical components
        self.safe_print("\n🧮 Checking mathematical optimization components...")
        self.check_mathematical_components()
        
        # Generate report
        return self.generate_audit_report()
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        # Count issues by severity
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        issue_types = {}
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
        
        total_issues = len(self.issues)
        compliance_score = max(0, 100 - (severity_counts['CRITICAL'] * 25 + 
                                       severity_counts['HIGH'] * 10 + 
                                       severity_counts['MEDIUM'] * 5 + 
                                       severity_counts['LOW'] * 1))
        
        # Print summary
        self.safe_print("\n📊 AUDIT SUMMARY:")
        self.safe_print("=" * 40)
        self.safe_print(f"Files Audited: {self.files_audited}")
        self.safe_print(f"Total Issues Found: {total_issues}")
        self.safe_print(f"Compliance Score: {compliance_score:.1f}/100")
        self.safe_print("")
        
        # Print severity breakdown
        self.safe_print("🚨 ISSUES BY SEVERITY:")
        for severity, count in severity_counts.items():
            if count > 0:
                self.safe_print(f"   {severity}: {count}")
        
        if total_issues == 0:
            self.safe_print("✅ NO COMPLIANCE ISSUES FOUND!")
            self.safe_print("🎉 All files meet Windows CLI compatibility standards!")
        else:
            self.safe_print(f"\n⚠️ ISSUES BY TYPE:")
            for issue_type, count in issue_types.items():
                self.safe_print(f"   {issue_type}: {count}")
            
            # Print detailed issues
            self.safe_print(f"\n📋 DETAILED ISSUE REPORT:")
            for issue in self.issues:
                self.safe_print(f"\n🔸 {issue.severity} - {issue.issue_type}")
                self.safe_print(f"   File: {issue.file_path}:{issue.line_number}")
                self.safe_print(f"   Issue: {issue.description}")
                self.safe_print(f"   Fix: {issue.recommendation}")
        
        return {
            'files_audited': self.files_audited,
            'total_issues': total_issues,
            'compliance_score': compliance_score,
            'severity_counts': severity_counts,
            'issue_types': issue_types,
            'issues': [
                {
                    'file': issue.file_path,
                    'line': issue.line_number,
                    'type': issue.issue_type,
                    'severity': issue.severity,
                    'description': issue.description,
                    'recommendation': issue.recommendation
                }
                for issue in self.issues
            ]
        }

def main() -> None:
    """Main audit function"""
    auditor = WindowsCliComplianceAuditor()
    
    # Header
    auditor.safe_print("🔍 WINDOWS CLI COMPLIANCE AUDIT - HYBRID OPTIMIZATION SYSTEM")
    auditor.safe_print("============================================================")
    auditor.safe_print("")
    auditor.safe_print("Auditing hybrid dual pipeline optimization system for:")
    auditor.safe_print("✅ No bare except statements")
    auditor.safe_print("✅ Complete type annotations") 
    auditor.safe_print("✅ No wildcard imports")
    auditor.safe_print("✅ Windows CLI emoji compatibility")
    auditor.safe_print("✅ Proper naming conventions")
    auditor.safe_print("✅ Mathematical optimization components")
    auditor.safe_print("")
    
    # Run audit
    report = auditor.run_comprehensive_audit()
    
    # Final status
    if report['total_issues'] == 0:
        auditor.safe_print("\n🏆 AUDIT COMPLETE - FULL COMPLIANCE ACHIEVED!")
        auditor.safe_print("🎯 Hybrid optimization system meets all Windows CLI standards!")
        auditor.safe_print("✨ Ready for production deployment!")
    else:
        auditor.safe_print(f"\n⚠️ AUDIT COMPLETE - {report['total_issues']} ISSUES FOUND")
        auditor.safe_print(f"📊 Compliance Score: {report['compliance_score']:.1f}/100")
        auditor.safe_print("🔧 Review issues above and apply recommended fixes")
    
    return report

if __name__ == "__main__":
    main() 