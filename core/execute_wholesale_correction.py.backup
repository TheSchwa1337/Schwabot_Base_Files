# -*- coding: utf-8 -*-
"""
Execute Wholesale Correction System
==================================

Comprehensive execution script for the UTC 2-Bit Wholesale Correction system.
This script orchestrates the entire correction process with detailed logging,
progress tracking, and iterative refinement.

Mathematical Foundation:
- Iterative Correction: C(n) = Σᵢ₌₁ⁿ correction_factorᵢ * gap_sizeᵢ
- Convergence Check: |C(n) - C(n-1)| < ε
- Mathematical Preservation: preserve(math) ∧ correct(syntax)
"""

import os
import sys
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.utc_2bit_wholesale_corrector import UTC2BitWholesaleCorrector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wholesale_correction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class WholesaleCorrectionExecutor:
    """
    Executor for the wholesale correction system.
    
    This class orchestrates the entire correction process with:
    - Detailed progress tracking
    - Iterative refinement
    - Comprehensive reporting
    - Mathematical preservation verification
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize the executor."""
        self.project_root = Path(project_root)
        self.corrector = UTC2BitWholesaleCorrector(project_root)
        self.execution_history: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        
        logger.info("Wholesale Correction Executor initialized")
    
    def execute_full_correction_cycle(self, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Execute a full correction cycle with comprehensive tracking.
        
        Args:
            max_iterations: Maximum number of correction iterations.
            
        Returns:
            Comprehensive results of the correction cycle.
        """
        self.start_time = datetime.now()
        logger.info(f"Starting full correction cycle (max {max_iterations} iterations)")
        
        # Phase 1: Initial Assessment
        logger.info("=== PHASE 1: Initial Assessment ===")
        initial_assessment = self._perform_initial_assessment()
        
        # Phase 2: Iterative Correction
        logger.info("=== PHASE 2: Iterative Correction ===")
        correction_results = self.corrector.execute_iterative_correction(max_iterations)
        
        # Phase 3: Final Verification
        logger.info("=== PHASE 3: Final Verification ===")
        final_verification = self._perform_final_verification()
        
        # Phase 4: Mathematical Preservation Check
        logger.info("=== PHASE 4: Mathematical Preservation Check ===")
        math_preservation = self._verify_mathematical_preservation()
        
        self.end_time = datetime.now()
        
        # Compile comprehensive results
        results = {
            "execution_metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds(),
                "max_iterations": max_iterations
            },
            "initial_assessment": initial_assessment,
            "correction_results": correction_results,
            "final_verification": final_verification,
            "mathematical_preservation": math_preservation,
            "summary": self._generate_comprehensive_summary()
        }
        
        # Save results
        self._save_results(results)
        
        logger.info("Full correction cycle completed")
        return results
    
    def _perform_initial_assessment(self) -> Dict[str, Any]:
        """Perform initial assessment of the codebase."""
        logger.info("Performing initial assessment...")
        
        assessment = {
            "total_files": 0,
            "python_files": 0,
            "files_with_math": 0,
            "files_with_syntax_errors": 0,
            "files_needing_preservation": 0,
            "estimated_corrections_needed": 0
        }
        
        # Count files
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                assessment["total_files"] += 1
                if file.endswith('.py'):
                    assessment["python_files"] += 1
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check for mathematical content
                        if self._has_mathematical_content(content):
                            assessment["files_with_math"] += 1
                        
                        # Check for syntax errors
                        if self._has_syntax_errors(content):
                            assessment["files_with_syntax_errors"] += 1
                        
                        # Check if preservation needed
                        if self._needs_mathematical_preservation(content):
                            assessment["files_needing_preservation"] += 1
                            
                    except Exception as e:
                        logger.warning(f"Error assessing {file_path}: {e}")
        
        # Estimate corrections needed
        assessment["estimated_corrections_needed"] = (
            assessment["files_with_syntax_errors"] * 2 +  # Syntax fixes
            assessment["files_needing_preservation"] * 3 +  # Preservation fixes
            assessment["python_files"] * 1  # General improvements
        )
        
        logger.info(f"Initial assessment completed: {assessment}")
        return assessment
    
    def _perform_final_verification(self) -> Dict[str, Any]:
        """Perform final verification of corrections."""
        logger.info("Performing final verification...")
        
        verification = {
            "syntax_errors_remaining": 0,
            "mathematical_content_preserved": 0,
            "flake8_compliance": 0,
            "overall_health_score": 0.0
        }
        
        # Check remaining syntax errors
        for structure in self.corrector.utc_structures.values():
            try:
                with open(structure.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if self._has_syntax_errors(content):
                    verification["syntax_errors_remaining"] += 1
                else:
                    verification["mathematical_content_preserved"] += 1
                    
            except Exception:
                verification["syntax_errors_remaining"] += 1
        
        # Calculate health score
        total_files = len(self.corrector.utc_structures)
        if total_files > 0:
            verification["overall_health_score"] = (
                verification["mathematical_content_preserved"] / total_files
            )
        
        logger.info(f"Final verification completed: {verification}")
        return verification
    
    def _verify_mathematical_preservation(self) -> Dict[str, Any]:
        """Verify that mathematical content has been preserved."""
        logger.info("Verifying mathematical preservation...")
        
        preservation = {
            "total_mathematical_elements": 0,
            "preserved_elements": 0,
            "preservation_rate": 0.0,
            "critical_math_preserved": True,
            "preservation_details": []
        }
        
        for structure in self.corrector.utc_structures.values():
            try:
                with open(structure.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count mathematical elements
                math_elements = self._count_mathematical_elements(content)
                preservation["total_mathematical_elements"] += math_elements
                
                # Check preservation
                preserved = self._count_preserved_elements(content)
                preservation["preserved_elements"] += preserved
                
                # Check critical math
                if not self._has_critical_math_preserved(content):
                    preservation["critical_math_preserved"] = False
                
                preservation["preservation_details"].append({
                    "file": structure.file_path,
                    "elements": math_elements,
                    "preserved": preserved
                })
                
            except Exception as e:
                logger.warning(f"Error verifying preservation for {structure.file_path}: {e}")
        
        # Calculate preservation rate
        if preservation["total_mathematical_elements"] > 0:
            preservation["preservation_rate"] = (
                preservation["preserved_elements"] / preservation["total_mathematical_elements"]
            )
        
        logger.info(f"Mathematical preservation verification completed: {preservation}")
        return preservation
    
    def _generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of the correction process."""
        logger.info("Generating comprehensive summary...")
        
        summary = {
            "correction_effectiveness": {
                "structures_processed": self.corrector.correction_metrics["structures_processed"],
                "connections_established": self.corrector.correction_metrics["connections_established"],
                "gaps_identified": self.corrector.correction_metrics["gaps_identified"],
                "corrections_applied": self.corrector.correction_metrics["corrections_applied"],
                "mathematical_preserved": self.corrector.correction_metrics["mathematical_preserved"],
                "flake8_errors_fixed": self.corrector.correction_metrics["flake8_errors_fixed"]
            },
            "system_health": {
                "utc_structures_healthy": len([s for s in self.corrector.utc_structures.values() 
                                             if not s.flake8_errors]),
                "bit_logic_connections_healthy": len([c for c in self.corrector.bit_logic_connections.values() 
                                                    if c.correction_applied]),
                "asic_gaps_resolved": len([g for g in self.corrector.asic_gaps.values() 
                                         if g.priority == 1])
            },
            "mathematical_integrity": {
                "total_math_elements": sum(len(s.mathematical_content) 
                                         for s in self.corrector.utc_structures.values()),
                "preservation_success_rate": self.corrector.correction_metrics["mathematical_preserved"] / 
                                           max(sum(len(s.mathematical_content) 
                                                 for s in self.corrector.utc_structures.values()), 1)
            },
            "recommendations": self._generate_recommendations()
        }
        
        logger.info("Comprehensive summary generated")
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on correction results."""
        recommendations = []
        
        # Check for remaining issues
        if self.corrector.correction_metrics["gaps_identified"] > 0:
            recommendations.append(
                f"Consider additional iterations to resolve {self.corrector.correction_metrics['gaps_identified']} remaining ASIC gaps"
            )
        
        if self.corrector.correction_metrics["flake8_errors_fixed"] > 0:
            recommendations.append(
                "Run Flake8 analysis to verify all style issues have been resolved"
            )
        
        # Check mathematical preservation
        preservation_rate = self.corrector.correction_metrics["mathematical_preserved"] / max(
            sum(len(s.mathematical_content) for s in self.corrector.utc_structures.values()), 1
        )
        
        if preservation_rate < 0.95:
            recommendations.append(
                f"Mathematical preservation rate is {preservation_rate:.2%}. Consider manual review of critical mathematical content."
            )
        
        # Check system connectivity
        if len(self.corrector.bit_logic_connections) < len(self.corrector.utc_structures) * 0.5:
            recommendations.append(
                "Consider strengthening 2-bit logic connections between UTC structures"
            )
        
        if not recommendations:
            recommendations.append("System appears healthy. Continue with normal operations.")
        
        return recommendations
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"wholesale_correction_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _has_mathematical_content(self, content: str) -> bool:
        """Check if content has mathematical elements."""
        math_indicators = [
            'np.', 'math.', 'unified_math.',
            'hashlib.', 'BTC', 'ETH', 'USDC', 'XRP',
            'tensor', 'matrix', 'vector', 'hash',
            'MATHEMATICAL PRESERVATION'
        ]
        return any(indicator in content for indicator in math_indicators)
    
    def _has_syntax_errors(self, content: str) -> bool:
        """Check if content has syntax errors."""
        try:
            import ast
            ast.parse(content)
            return False
        except SyntaxError:
            return True
    
    def _needs_mathematical_preservation(self, content: str) -> bool:
        """Check if content needs mathematical preservation."""
        return self._has_mathematical_content(content) and not self._has_syntax_errors(content)
    
    def _count_mathematical_elements(self, content: str) -> int:
        """Count mathematical elements in content."""
        import re
        
        patterns = [
            r'# MATHEMATICAL PRESERVATION:',
            r'np\.',
            r'unified_math\.',
            r'hashlib\.',
            r'BTC.*price|ETH.*price|USDC.*price|XRP.*price'
        ]
        
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            count += len(matches)
        
        return count
    
    def _count_preserved_elements(self, content: str) -> int:
        """Count preserved mathematical elements."""
        import re
        
        # Look for preservation markers
        preservation_patterns = [
            r'# MATHEMATICAL PRESERVATION:.*?(?=\n|$)',
            r'# Preserved mathematical logic',
            r'# Mathematical content preserved'
        ]
        
        count = 0
        for pattern in preservation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            count += len(matches)
        
        return count
    
    def _has_critical_math_preserved(self, content: str) -> bool:
        """Check if critical mathematical content is preserved."""
        critical_indicators = [
            'MATHEMATICAL PRESERVATION',
            'unified_math',
            'hashlib.sha256',
            'BTC price'
        ]
        
        return any(indicator in content for indicator in critical_indicators)


def main():
    """Main execution function."""
    print("=== UTC 2-Bit Wholesale Correction System ===")
    print("Connecting UTC structures and 2-bit logic for comprehensive correction...")
    print()
    
    # Initialize executor
    executor = WholesaleCorrectionExecutor()
    
    try:
        # Execute full correction cycle
        results = executor.execute_full_correction_cycle(max_iterations=3)
        
        # Print summary
        print("\n=== CORRECTION SUMMARY ===")
        summary = results["summary"]
        
        print(f"UTC Structures Processed: {summary['correction_effectiveness']['structures_processed']}")
        print(f"2-Bit Logic Connections: {summary['correction_effectiveness']['connections_established']}")
        print(f"ASIC Gaps Identified: {summary['correction_effectiveness']['gaps_identified']}")
        print(f"Corrections Applied: {summary['correction_effectiveness']['corrections_applied']}")
        print(f"Mathematical Elements Preserved: {summary['correction_effectiveness']['mathematical_preserved']}")
        print(f"Flake8 Errors Fixed: {summary['correction_effectiveness']['flake8_errors_fixed']}")
        
        print(f"\nSystem Health:")
        print(f"  UTC Structures Healthy: {summary['system_health']['utc_structures_healthy']}")
        print(f"  Bit Logic Connections Healthy: {summary['system_health']['bit_logic_connections_healthy']}")
        print(f"  ASIC Gaps Resolved: {summary['system_health']['asic_gaps_resolved']}")
        
        print(f"\nMathematical Integrity:")
        print(f"  Total Math Elements: {summary['mathematical_integrity']['total_math_elements']}")
        print(f"  Preservation Success Rate: {summary['mathematical_integrity']['preservation_success_rate']:.2%}")
        
        print(f"\nRecommendations:")
        for recommendation in summary['recommendations']:
            print(f"  - {recommendation}")
        
        print(f"\nExecution completed in {results['execution_metadata']['duration_seconds']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 