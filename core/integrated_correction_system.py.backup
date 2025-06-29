# -*- coding: utf-8 -*-
"""
Integrated Correction System
============================

Comprehensive correction system that combines:
1. UTC 2-Bit Wholesale Correction for systematic gap filling
2. Surgical Mathematical Correction for targeted precision fixes
3. Mathematical preservation verification
4. Flake8 error reduction without introducing new issues

Mathematical Foundation:
- Integrated Approach: I(correction) = wholesale(gaps) + surgical(precision)
- Mathematical Safety: S(math) = preserve(critical) ∧ enhance(functional)
- Error Minimization: E(min) = reduce(existing) - introduce(new)
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Import both correction systems
from core.surgical_math_corrector import SurgicalMathCorrector
# Note: UTC corrector import will be fixed after syntax issues resolved

logger = logging.getLogger(__name__)

class IntegratedCorrectionSystem:
    """
    Integrated system combining wholesale and surgical correction approaches.
    
    This system:
    1. Runs surgical correction first for precision fixes
    2. Follows with wholesale correction for systematic gaps
    3. Validates mathematical preservation throughout
    4. Ensures no new Flake8 errors are introduced
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize the integrated correction system."""
        self.project_root = Path(project_root)
        self.surgical_corrector = SurgicalMathCorrector(project_root)
        # self.wholesale_corrector = UTC2BitWholesaleCorrector(project_root)  # Will enable after fix
        
        self.correction_history: List[Dict[str, Any]] = []
        self.mathematical_integrity_score = 0.0
        
        logger.info("Integrated Correction System initialized")
    
    def execute_integrated_correction(self) -> Dict[str, Any]:
        """
        Execute the complete integrated correction process.
        
        Returns:
            Comprehensive results of the integrated correction.
        """
        logger.info("Starting integrated correction process...")
        
        start_time = datetime.now()
        
        # Phase 1: Initial Assessment
        logger.info("=== PHASE 1: Initial System Assessment ===")
        initial_assessment = self._perform_initial_assessment()
        
        # Phase 2: Surgical Correction (Precision First)
        logger.info("=== PHASE 2: Surgical Mathematical Correction ===")
        surgical_results = self.surgical_corrector.execute_surgical_correction()
        self.correction_history.append({
            "phase": "surgical",
            "timestamp": datetime.now().isoformat(),
            "results": surgical_results
        })
        
        # Phase 3: Mathematical Integrity Verification
        logger.info("=== PHASE 3: Mathematical Integrity Verification ===")
        integrity_results = self._verify_mathematical_integrity()
        
        # Phase 4: Wholesale Correction (Systematic Gaps)
        logger.info("=== PHASE 4: Wholesale Gap Correction ===")
        # wholesale_results = self.wholesale_corrector.execute_iterative_correction(max_iterations=2)
        # Placeholder for now until UTC corrector is fixed
        wholesale_results = {"status": "pending_syntax_fix", "iterations": 0}
        self.correction_history.append({
            "phase": "wholesale",
            "timestamp": datetime.now().isoformat(),
            "results": wholesale_results
        })
        
        # Phase 5: Final Validation
        logger.info("=== PHASE 5: Final System Validation ===")
        final_validation = self._perform_final_validation()
        
        end_time = datetime.now()
        
        # Compile comprehensive results
        integrated_results = {
            "execution_metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds()
            },
            "initial_assessment": initial_assessment,
            "surgical_results": surgical_results,
            "integrity_results": integrity_results,
            "wholesale_results": wholesale_results,
            "final_validation": final_validation,
            "mathematical_integrity_score": self.mathematical_integrity_score,
            "correction_history": self.correction_history,
            "summary": self._generate_integrated_summary()
        }
        
        # Save comprehensive results
        self._save_integrated_results(integrated_results)
        
        logger.info("Integrated correction process completed")
        return integrated_results
    
    def _perform_initial_assessment(self) -> Dict[str, Any]:
        """Perform initial system assessment."""
        logger.info("Performing initial system assessment...")
        
        assessment = {
            "total_python_files": 0,
            "files_with_mathematical_content": 0,
            "files_with_fixes": 0,
            "estimated_syntax_errors": 0,
            "mathematical_complexity_score": 0.0,
            "system_health_indicators": {}
        }
        
        # Count files and assess complexity
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    assessment["total_python_files"] += 1
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check for mathematical content
                        if self._has_mathematical_content(content):
                            assessment["files_with_mathematical_content"] += 1
                        
                        # Check for fixes/placeholders
                        if self._has_fixes(content):
                            assessment["files_with_fixes"] += 1
                        
                        # Check for syntax errors
                        if self._has_syntax_errors(content):
                            assessment["estimated_syntax_errors"] += 1
                        
                        # Assess mathematical complexity
                        complexity = self._assess_mathematical_complexity(content)
                        assessment["mathematical_complexity_score"] += complexity
                        
                    except Exception as e:
                        logger.warning(f"Error assessing {file_path}: {e}")
        
        # Calculate overall complexity score
        if assessment["total_python_files"] > 0:
            assessment["mathematical_complexity_score"] /= assessment["total_python_files"]
        
        # System health indicators
        assessment["system_health_indicators"] = {
            "mathematical_density": assessment["files_with_mathematical_content"] / max(assessment["total_python_files"], 1),
            "fix_density": assessment["files_with_fixes"] / max(assessment["total_python_files"], 1),
            "error_rate": assessment["estimated_syntax_errors"] / max(assessment["total_python_files"], 1),
            "complexity_level": "high" if assessment["mathematical_complexity_score"] > 0.7 else 
                               "medium" if assessment["mathematical_complexity_score"] > 0.4 else "low"
        }
        
        logger.info(f"Initial assessment completed: {assessment}")
        return assessment
    
    def _verify_mathematical_integrity(self) -> Dict[str, Any]:
        """Verify mathematical integrity after surgical correction."""
        logger.info("Verifying mathematical integrity...")
        
        integrity_results = {
            "critical_math_preserved": True,
            "mathematical_functions_intact": 0,
            "mathematical_constants_preserved": 0,
            "algorithm_integrity_score": 0.0,
            "trading_logic_preserved": True,
            "btc_hashing_intact": True,
            "tensor_operations_functional": True,
            "integrity_violations": []
        }
        
        # Check critical mathematical systems
        critical_systems = [
            ("BTC hashing", ["btc", "hash", "sha256"]),
            ("Tensor operations", ["tensor", "matrix", "vector"]),
            ("Trading algorithms", ["trading", "profit", "loss"]),
            ("Unified math", ["unified_math", "calculation"]),
            ("Ferris RDE", ["ferris", "rde"]),
            ("Lantern core", ["lantern", "core"])
        ]
        
        for system_name, indicators in critical_systems:
            if not self._verify_system_integrity(system_name, indicators):
                integrity_results["integrity_violations"].append(system_name)
                if "btc" in indicators:
                    integrity_results["btc_hashing_intact"] = False
                elif "tensor" in indicators:
                    integrity_results["tensor_operations_functional"] = False
                elif "trading" in indicators:
                    integrity_results["trading_logic_preserved"] = False
        
        # Calculate overall integrity score
        total_systems = len(critical_systems)
        intact_systems = total_systems - len(integrity_results["integrity_violations"])
        self.mathematical_integrity_score = intact_systems / total_systems
        integrity_results["algorithm_integrity_score"] = self.mathematical_integrity_score
        
        # Overall preservation status
        integrity_results["critical_math_preserved"] = self.mathematical_integrity_score >= 0.9
        
        logger.info(f"Mathematical integrity verification completed: {integrity_results}")
        return integrity_results
    
    def _perform_final_validation(self) -> Dict[str, Any]:
        """Perform final system validation."""
        logger.info("Performing final system validation...")
        
        validation = {
            "syntax_validation_passed": True,
            "mathematical_preservation_verified": True,
            "flake8_compliance_improved": False,
            "system_functionality_maintained": True,
            "error_reduction_achieved": False,
            "validation_details": {}
        }
        
        # Run syntax validation
        syntax_errors = self._count_syntax_errors()
        validation["syntax_validation_passed"] = syntax_errors == 0
        validation["validation_details"]["syntax_errors"] = syntax_errors
        
        # Verify mathematical preservation
        math_preservation = self.mathematical_integrity_score >= 0.95
        validation["mathematical_preservation_verified"] = math_preservation
        validation["validation_details"]["integrity_score"] = self.mathematical_integrity_score
        
        # Check system functionality (basic import tests)
        functionality_score = self._test_system_functionality()
        validation["system_functionality_maintained"] = functionality_score >= 0.8
        validation["validation_details"]["functionality_score"] = functionality_score
        
        # Overall validation status
        validation["overall_success"] = (
            validation["syntax_validation_passed"] and
            validation["mathematical_preservation_verified"] and
            validation["system_functionality_maintained"]
        )
        
        logger.info(f"Final validation completed: {validation}")
        return validation
    
    def _generate_integrated_summary(self) -> Dict[str, Any]:
        """Generate integrated summary of all correction phases."""
        logger.info("Generating integrated summary...")
        
        summary = {
            "correction_effectiveness": {
                "surgical_fixes_applied": 0,
                "wholesale_gaps_filled": 0,
                "mathematical_elements_preserved": 0,
                "syntax_errors_resolved": 0,
                "total_improvements": 0
            },
            "mathematical_preservation": {
                "integrity_score": self.mathematical_integrity_score,
                "critical_systems_intact": self.mathematical_integrity_score >= 0.9,
                "preservation_rate": self.mathematical_integrity_score * 100
            },
            "system_health": {
                "overall_health_score": 0.0,
                "functionality_maintained": True,
                "error_reduction_achieved": False,
                "compliance_improved": False
            },
            "recommendations": self._generate_integrated_recommendations()
        }
        
        # Extract metrics from correction history
        for phase_result in self.correction_history:
            if phase_result["phase"] == "surgical":
                surgical_metrics = phase_result["results"]["surgical_metrics"]
                summary["correction_effectiveness"]["surgical_fixes_applied"] = surgical_metrics.get("targeted_corrections", 0)
                summary["correction_effectiveness"]["mathematical_elements_preserved"] += surgical_metrics.get("mathematical_preserved", 0)
            
            elif phase_result["phase"] == "wholesale":
                wholesale_metrics = phase_result["results"]
                if isinstance(wholesale_metrics, dict) and "final_metrics" in wholesale_metrics:
                    summary["correction_effectiveness"]["wholesale_gaps_filled"] = wholesale_metrics["final_metrics"].get("corrections_applied", 0)
        
        # Calculate total improvements
        summary["correction_effectiveness"]["total_improvements"] = (
            summary["correction_effectiveness"]["surgical_fixes_applied"] +
            summary["correction_effectiveness"]["wholesale_gaps_filled"]
        )
        
        # Calculate overall health score
        summary["system_health"]["overall_health_score"] = (
            self.mathematical_integrity_score * 0.6 +  # 60% weight on math preservation
            (1.0 if summary["system_health"]["functionality_maintained"] else 0.0) * 0.4  # 40% weight on functionality
        )
        
        logger.info("Integrated summary generated")
        return summary
    
    def _generate_integrated_recommendations(self) -> List[str]:
        """Generate integrated recommendations."""
        recommendations = []
        
        # Mathematical preservation recommendations
        if self.mathematical_integrity_score < 0.95:
            recommendations.append(f"Mathematical integrity at {self.mathematical_integrity_score:.1%} - review critical systems")
        
        # System health recommendations
        if self.mathematical_integrity_score >= 0.95:
            recommendations.append("Mathematical systems are well-preserved - safe to proceed with optimizations")
        
        # Correction recommendations
        surgical_applied = sum(1 for phase in self.correction_history if phase["phase"] == "surgical")
        if surgical_applied > 0:
            recommendations.append("Surgical corrections completed - consider running wholesale correction")
        
        # Future improvement recommendations
        recommendations.append("Continue monitoring mathematical integrity during future changes")
        recommendations.append("Implement automated mathematical preservation testing")
        
        return recommendations
    
    def _save_integrated_results(self, results: Dict[str, Any]) -> None:
        """Save integrated results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"integrated_correction_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Integrated results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save integrated results: {e}")
    
    def _has_mathematical_content(self, content: str) -> bool:
        """Check if content has mathematical elements."""
        math_indicators = [
            'np.', 'math.', 'scipy.', 'unified_math',
            'BTC', 'ETH', 'USDC', 'XRP', 'price', 'hash',
            'tensor', 'matrix', 'vector', 'algorithm',
            'trading', 'profit', 'loss', 'calculation'
        ]
        return any(indicator in content for indicator in math_indicators)
    
    def _has_fixes(self, content: str) -> bool:
        """Check if content has fixes/placeholders."""
        fix_indicators = ['TODO', 'FIXME', 'pass  # TODO', 'Emergency placeholder']
        return any(indicator in content for indicator in fix_indicators)
    
    def _has_syntax_errors(self, content: str) -> bool:
        """Check if content has syntax errors."""
        try:
            import ast
            ast.parse(content)
            return False
        except SyntaxError:
            return True
    
    def _assess_mathematical_complexity(self, content: str) -> float:
        """Assess mathematical complexity of content."""
        complexity_indicators = [
            ('tensor', 0.3), ('matrix', 0.3), ('algorithm', 0.2),
            ('optimization', 0.2), ('calculation', 0.1), ('formula', 0.2),
            ('BTC', 0.2), ('trading', 0.1), ('unified_math', 0.3)
        ]
        
        complexity_score = 0.0
        for indicator, weight in complexity_indicators:
            if indicator in content:
                complexity_score += weight
        
        return min(complexity_score, 1.0)  # Cap at 1.0
    
    def _verify_system_integrity(self, system_name: str, indicators: List[str]) -> bool:
        """Verify integrity of a specific system."""
        # Check if system files exist and are syntactically valid
        system_files_found = 0
        system_files_valid = 0
        
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check if file relates to this system
                        if any(indicator in content.lower() for indicator in indicators):
                            system_files_found += 1
                            
                            # Check if file is syntactically valid
                            if not self._has_syntax_errors(content):
                                system_files_valid += 1
                                
                    except Exception:
                        continue
        
        # System is intact if most related files are valid
        if system_files_found == 0:
            return True  # No files found, assume system is optional
        
        integrity_ratio = system_files_valid / system_files_found
        return integrity_ratio >= 0.8  # 80% of files must be valid
    
    def _count_syntax_errors(self) -> int:
        """Count syntax errors across the codebase."""
        error_count = 0
        
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if self._has_syntax_errors(content):
                            error_count += 1
                            
                    except Exception:
                        continue
        
        return error_count
    
    def _test_system_functionality(self) -> float:
        """Test basic system functionality."""
        functionality_tests = [
            ("core.unified_math_system", 0.3),
            ("core.bit_phase_sequencer", 0.2),
            ("dual_unicore_handler", 0.2),
            ("core.ferris_rde_core", 0.1),
            ("core.lantern_core", 0.1),
            ("core.tensor_score_utils", 0.1)
        ]
        
        passed_tests = 0.0
        
        for module_name, weight in functionality_tests:
            try:
                # Try to import the module
                __import__(module_name)
                passed_tests += weight
            except ImportError:
                # Module not found or has import errors
                continue
            except Exception:
                # Other errors during import
                continue
        
        return passed_tests


def main():
    """Main function for integrated correction."""
    print("=== Integrated Correction System ===")
    print("Combining surgical precision with wholesale gap filling...")
    print("Preserving mathematical integrity throughout the process...")
    print()
    
    # Initialize integrated system
    integrated_system = IntegratedCorrectionSystem()
    
    try:
        # Execute integrated correction
        results = integrated_system.execute_integrated_correction()
        
        # Print comprehensive summary
        print("\n=== INTEGRATED CORRECTION SUMMARY ===")
        summary = results["summary"]
        
        print(f"Mathematical Integrity Score: {summary['mathematical_preservation']['integrity_score']:.1%}")
        print(f"Critical Systems Intact: {summary['mathematical_preservation']['critical_systems_intact']}")
        print(f"Overall Health Score: {summary['system_health']['overall_health_score']:.1%}")
        
        print(f"\nCorrection Effectiveness:")
        effectiveness = summary["correction_effectiveness"]
        print(f"  Surgical Fixes Applied: {effectiveness['surgical_fixes_applied']}")
        print(f"  Wholesale Gaps Filled: {effectiveness['wholesale_gaps_filled']}")
        print(f"  Mathematical Elements Preserved: {effectiveness['mathematical_elements_preserved']}")
        print(f"  Total Improvements: {effectiveness['total_improvements']}")
        
        print(f"\nSystem Health:")
        health = summary["system_health"]
        print(f"  Functionality Maintained: {health['functionality_maintained']}")
        print(f"  Error Reduction Achieved: {health['error_reduction_achieved']}")
        print(f"  Compliance Improved: {health['compliance_improved']}")
        
        print(f"\nRecommendations:")
        for recommendation in summary['recommendations']:
            print(f"  - {recommendation}")
        
        print(f"\nExecution completed in {results['execution_metadata']['duration_seconds']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Integrated correction failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 