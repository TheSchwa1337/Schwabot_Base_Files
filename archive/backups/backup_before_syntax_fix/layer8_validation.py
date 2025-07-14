#!/usr/bin/env python3
"""
🧬🔐🤖🔀 LAYER 8 VALIDATION
===========================

Focused validation for Layer 8: Hash-Glyph Memory Compression + Cross-Agent Path Blending
Checks the integrity and functionality of all Layer 8 components.
"""

import importlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.append('.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Layer8Validator:
    """Validator specifically for Layer 8 components"""

    def __init__(self):
        self.layer8_components = []
            'core.hash_glyph_compression',
            'core.ai_matrix_consensus',
            'core.visual_decision_engine', 
            'core.loop_strategy_switcher'
        ]
        self.results = {}
        self.issues = []

    def validate_layer8(self) -> Dict[str, Any]:
        """Validate all Layer 8 components"""
        logger.info("🧬🔐🤖🔀 Starting Layer 8 validation...")

        for component in self.layer8_components:
            logger.info(f"Validating {component}...")
            self.results[component] = self._validate_component(component)

        # Test integration
        integration_result = self._test_layer8_integration()
        self.results['integration'] = integration_result

        return self._generate_summary()

    def _validate_component(self, component_name: str) -> Dict[str, Any]:
        """Validate a single Layer 8 component"""
        try:
            # Import the module
            module = importlib.import_module(component_name)

            # Check for required classes
            required_classes = self._get_required_classes(component_name)
            class_status = {}

            for class_name in required_classes:
                if hasattr(module, class_name):
                    class_status[class_name] = "✅ Found"

                    # Test instantiation
                    try:
                        if class_name == 'HashGlyphCompressor':
                            instance = module.create_hash_glyph_compressor()
                        elif class_name == 'AIMatrixConsensus':
                            instance = module.create_ai_matrix_consensus()
                        elif class_name == 'VisualDecisionEngine':
                            instance = module.create_visual_decision_engine()
                        elif class_name == 'StrategyLoopSwitcher':
                            instance = module.create_strategy_loop_switcher()
                        else:
                            instance = getattr(module, class_name)()

                        class_status[class_name] = "✅ Found and instantiable"

                        # Test basic functionality
                        if hasattr(instance, 'get_memory_stats'):
                            stats = instance.get_memory_stats()
                            class_status[f"{class_name}_stats"] = "✅ Stats method works"

                    except Exception as e:
                        class_status[class_name] = f"⚠️ Found but instantiation failed: {e}"
                        self.issues.append(f"{component_name}.{class_name}: {e}")
                else:
                    class_status[class_name] = "❌ Missing"
                    self.issues.append(f"{component_name}: Missing {class_name} class")

            return {}
                "status": "✅ Valid",
                "classes": class_status,
                "import_success": True
            }

        except ImportError as e:
            self.issues.append(f"{component_name}: Import failed - {e}")
            return {}
                "status": "❌ Import failed",
                "error": str(e),
                "import_success": False
            }
        except Exception as e:
            self.issues.append(f"{component_name}: Validation error - {e}")
            return {}
                "status": "❌ Validation error",
                "error": str(e),
                "import_success": False
            }

    def _get_required_classes(self, component_name: str) -> List[str]:
        """Get required classes for each component"""
        class_map = {}
            'core.hash_glyph_compression': ['HashGlyphCompressor', 'GlyphMemoryChunk'],
            'core.ai_matrix_consensus': ['AIMatrixConsensus', 'AgentVote', 'ConsensusResult'],
            'core.visual_decision_engine': ['VisualDecisionEngine'],
            'core.loop_strategy_switcher': ['StrategyLoopSwitcher', 'AssetTarget', 'StrategyResult']
        }
        return class_map.get(component_name, [])

    def _test_layer8_integration(self) -> Dict[str, Any]:
        """Test Layer 8 integration"""
        try:
            logger.info("Testing Layer 8 integration...")

            # Import all components
            from core.ai_matrix_consensus import create_ai_matrix_consensus
            from core.hash_glyph_compression import create_hash_glyph_compressor
            from core.loop_strategy_switcher import create_strategy_loop_switcher
            from core.visual_decision_engine import create_visual_decision_engine

            # Create instances
            compressor = create_hash_glyph_compressor()
            consensus = create_ai_matrix_consensus()
            engine = create_visual_decision_engine()
            switcher = create_strategy_loop_switcher()

            # Test basic functionality
            test_results = {}

            # Test hash-glyph compression
            import numpy as np
            test_matrix = np.array([[1, 0, 2], [0, 2, 1], [2, 1, 0]])
            test_vector = np.array([0.1, 0.4, 0.3])
            test_votes = {"R1": "execute", "Claude": "recycle"}

            hash_key = compressor.store("test", test_matrix, "🌘", test_vector, test_votes)
            test_results["compression_store"] = "✅ Store works"

            retrieved = compressor.retrieve("test", test_matrix)
            test_results["compression_retrieve"] = "✅ Retrieve works" if retrieved else "❌ Retrieve failed"

            # Test AI consensus
            consensus_result = consensus.vote("🌘", test_vector)
            test_results["consensus_vote"] = "✅ Vote works"

            # Test visual decision engine
            glyph, blended_vector, decision = engine.route_with_path_blending()
                "test", test_matrix, test_vector
            )
            test_results["path_blending"] = "✅ Path blending works"

            # Test strategy loop switcher
            market_data = {"timestamp": 1234567890, "btc_price": 50000}
            portfolio = {"BTC": 0.1, "ETH": 2.0}

            results = switcher.force_cycle_execution(market_data, portfolio)
            test_results["strategy_execution"] = f"✅ Strategy execution works ({len(results)} results)"

            return {}
                "status": "✅ Integration successful",
                "tests": test_results
            }

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            traceback.print_exc()
            return {}
                "status": "❌ Integration failed",
                "error": str(e)
            }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        total_components = len(self.layer8_components)
        successful_imports = sum(1 for r in self.results.values())
                               if isinstance(r, dict) and r.get('import_success', False))

        integration_success = (self.results.get('integration', {}).get('status', '') == '✅ Integration successful')

        summary = {}
            "layer8_status": "✅ FULLY OPERATIONAL" if (successful_imports == total_components and, integration_success) else "⚠️ NEEDS ATTENTION",
            "components_validated": total_components,
            "successful_imports": successful_imports,
            "integration_success": integration_success,
            "total_issues": len(self.issues),
            "component_results": self.results,
            "issues": self.issues
        }

        return summary

def main():
    """Main validation function"""
    print("🧬🔐🤖🔀 LAYER 8 VALIDATION")
    print("=" * 50)

    try:
        validator = Layer8Validator()
        results = validator.validate_layer8()

        # Print results
        print(f"\n📊 LAYER 8 STATUS: {results['layer8_status']}")
        print(f"Components validated: {results['components_validated']}")
        print(f"Successful imports: {results['successful_imports']}")
        print(f"Integration success: {results['integration_success']}")
        print(f"Total issues: {results['total_issues']}")

        # Print component results
        print(f"\n🔍 COMPONENT RESULTS:")
        for component, result in results['component_results'].items():
            if component != 'integration':
                status = result.get('status', 'Unknown')
                print(f"  {component}: {status}")

                if 'classes' in result:
                    for class_name, class_status in result['classes'].items():
                        if not class_name.endswith('_stats'):
                            print(f"    {class_name}: {class_status}")

        # Print integration results
        integration = results['component_results'].get('integration', {})
        print(f"\n🔄 INTEGRATION: {integration.get('status', 'Unknown')}")

        if 'tests' in integration:
            for test_name, test_result in integration['tests'].items():
                print(f"  {test_name}: {test_result}")

        # Print issues
        if results['issues']:
            print(f"\n⚠️ ISSUES FOUND:")
            for issue in results['issues']:
                print(f"  {issue}")

        # Overall assessment
        if results['layer8_status'] == "✅ FULLY OPERATIONAL":
            print(f"\n🎉 LAYER 8 IS FULLY OPERATIONAL!")
            print("All components are working correctly and integrated properly.")
            print("Ready for Layer 9 development.")
        else:
            print(f"\n⚠️ LAYER 8 NEEDS ATTENTION")
            print("Please address the issues above before proceeding.")

        # Save results
        with open('layer8_validation_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Validation report saved to layer8_validation_report.json")

        return results['layer8_status'] == "✅ FULLY OPERATIONAL"

    except Exception as e:
        logger.error(f"Layer 8 validation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 