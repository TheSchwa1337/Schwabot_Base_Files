import json
import sys
from pathlib import Path

from flake8_positional_corrector import Flake8PositionalCorrector, flake8_corrector

# -*- coding: utf-8 -*-
"""
Test MathLib Positional State System and Flake8 Corrections
==========================================================

Comprehensive test suite for the MathLib positional state system and Flake8
corrections, ensuring proper 32-bit phase orientation and mathematical
integrity preservation across all MathLib versions.

Test Coverage:
- Positional state initialization and management
- 32-bit phase orientation application
- Flake8 error detection and correction
- Mathematical formula preservation
- Dependency relationship validation
- UTF-8 compatibility and emoji handling
- Comprehensive reporting and logging
"""


# Add core directory to path
core_dir = Path(__file__).parent / "core"
sys.path.insert(0, str(core_dir))

try:
        MathLibPositionalStateSystem,
        MathLibVersion,
        BitPhase,
        positional_state_system,
    )
    except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the core modules are available")
    sys.exit(1)


def test_positional_state_initialization():
    """Test positional state system initialization."""
    print("🧪 Testing Positional State Initialization...")

    try:
        # Test state initialization
        assert len(positional_state_system.states) == 5, ()
            f"Expected 5 states, got {len(positional_state_system.states)}"
        )

        # Test each MathLib version
        for version in MathLibVersion:
            state = positional_state_system.get_positional_state(version)
            assert state is not None, f"State not found for {version.value}"
            assert state.version == version, f"Version mismatch for {version.value}"
            assert state.is_active, f"State not active for {version.value}"

        print("✅ Positional state initialization test passed")
        return True

    except Exception as e:
        print(f"❌ Positional state initialization test failed: {e}")
        return False


def test_32bit_phase_orientation():
    """Test 32-bit phase orientation application."""
    print("🧪 Testing 32-bit Phase Orientation...")

    try:
        # Test 32-bit phase orientation for each version
        for version in MathLibVersion:
            result = positional_state_system.apply_32bit_phase_orientation(version)

            assert "error" not in result, ()
                f"Error applying 32-bit phase to {version.value}"
            )
            assert result["bit_phase"] == 32, ()
                f"Expected 32-bit phase, got {result['bit_phase']} for {version.value}"
            )

            # Verify state was updated
            state = positional_state_system.get_positional_state(version)
            assert state.bit_phase == BitPhase.THIRTY_TWO_BIT, ()
                f"State not updated for {version.value}"
            )

        print("✅ 32-bit phase orientation test passed")
        return True

    except Exception as e:
        print(f"❌ 32-bit phase orientation test failed: {e}")
        return False


def test_dependency_relationships():
    """Test dependency relationships between MathLib versions."""
    print("🧪 Testing Dependency Relationships...")

    try:
        # Test dependency graph
        dependency_graph = positional_state_system.dependency_graph

        # V1 should have no dependencies
        assert len(dependency_graph[MathLibVersion.V1]) == 0, ()
            "V1 should have no dependencies"
        )

        # V2 should depend on V1
        assert MathLibVersion.V1 in dependency_graph[MathLibVersion.V2], ()
            "V2 should depend on V1"
        )

        # V3 should depend on V1 and V2
        v3_deps = dependency_graph[MathLibVersion.V3]
        assert MathLibVersion.V1 in v3_deps, "V3 should depend on V1"
        assert MathLibVersion.V2 in v3_deps, "V3 should depend on V2"

        # V4 should depend on V2 and V3
        v4_deps = dependency_graph[MathLibVersion.V4]
        assert MathLibVersion.V2 in v4_deps, "V4 should depend on V2"
        assert MathLibVersion.V3 in v4_deps, "V4 should depend on V3"

        # Unified should depend on all versions
        unified_deps = dependency_graph[MathLibVersion.UNIFIED]
        assert len(unified_deps) == 4, "Unified should depend on all 4 versions"

        print("✅ Dependency relationships test passed")
        return True

    except Exception as e:
        print(f"❌ Dependency relationships test failed: {e}")
        return False


def test_mathematical_formula_preservation():
    """Test mathematical formula preservation."""
    print("🧪 Testing Mathematical Formula Preservation...")

    try:
        # Test formula extraction
        test_content = """
# MATHEMATICAL PRESERVATION: Mathematical logic or formula preserved below
    def calculate_btc_price_hash(price_data):
    # BTC price hashing algorithm
    return hashlib.sha256(str(price_data).encode()).hexdigest()

# MATHEMATICAL PRESERVATION: Tensor operation preserved below
    def tensor_contraction(a, b):
    # Tensor contraction formula
    return np.tensordot(a, b, axes=1)
"""

        formulas = flake8_corrector._extract_mathematical_formulas(test_content)

        assert len(formulas) >= 2, f"Expected at least 2 formulas, got {len(formulas)}"
        assert any("BTC price hashing" in formula for formula in, formulas), ()
            "BTC price hashing formula not found"
        )
        assert any("tensor" in formula.lower() for formula in formulas), ()
            "Tensor operation formula not found"
        )

        print("✅ Mathematical formula preservation test passed")
        return True

    except Exception as e:
        print(f"❌ Mathematical formula preservation test failed: {e}")
        return False


def test_flake8_error_correction():
    """Test Flake8 error detection and correction."""
    print("🧪 Testing Flake8 Error Correction...")

    try:
        # Test content with Flake8 errors
        test_content = """
# MATHEMATICAL PRESERVATION: Mathematical logic preserved below
    def test_function(x,y):  # Missing spaces around comma
    result=x+y  # Missing spaces around operators
    return result

def another_function():
    text="Unmatched quote  # Unmatched quote"
    return text
"""

        # Test correction
        corrected_content, corrections = flake8_corrector._correct_content()
            test_content, MathLibVersion.V3
        )

        # Verify corrections were made
        assert len(corrections) > 0, "No corrections were made"

        # Verify mathematical preservation was maintained
        assert "# MATHEMATICAL PRESERVATION:" in corrected_content, ()
            "Mathematical preservation lost"
        )

        # Verify syntax was corrected
        assert "x, y" in corrected_content, "Comma spacing not corrected"
        assert "result = x + y" in corrected_content, "Operator spacing not corrected"

        print("✅ Flake8 error correction test passed")
        return True

    except Exception as e:
        print(f"❌ Flake8 error correction test failed: {e}")
        return False


def test_utf8_compatibility():
    """Test UTF-8 compatibility and emoji handling."""
    print("🧪 Testing UTF-8 Compatibility...")

    try:
        # Test emoji handling in reports
        report = positional_state_system.get_comprehensive_report()

        # Verify report structure
        assert "timestamp" in report, "Timestamp missing from report"
        assert "total_versions" in report, "Total versions missing from report"
        assert "versions" in report, "Versions missing from report"
        assert "dependency_graph" in report, "Dependency graph missing from report"
        assert "overall_compliance" in report, "Overall compliance missing from report"

        # Test UTF-8 encoding
        report_json = json.dumps(report, ensure_ascii=False)
        assert "🧮" not in report_json, "Emojis should not be in JSON output"

        print("✅ UTF-8 compatibility test passed")
        return True

    except Exception as e:
        print(f"❌ UTF-8 compatibility test failed: {e}")
        return False


def test_comprehensive_reporting():
    """Test comprehensive reporting functionality."""
    print("🧪 Testing Comprehensive Reporting...")

    try:
        # Generate comprehensive report
        report = positional_state_system.get_comprehensive_report()

        # Verify report structure
        assert report["total_versions"] == 5, ()
            f"Expected 5 versions, got {report['total_versions']}"
        )
        assert len(report["versions"]) == 5, ()
            f"Expected 5 version entries, got {len(report['versions'])}"
        )
        assert len(report["dependency_graph"]) == 5, ()
            f"Expected 5 dependency entries, got {len(report['dependency_graph'])}"
        )

        # Verify each version has required fields
        for version_name, version_data in report["versions"].items():
            required_fields = []
                "bit_phase",
                "dependencies",
                "mathematical_formulas_count",
                "flake8_errors_count",
                "compliance_score",
                "last_updated",
                "is_active",
            ]

            for field in required_fields:
                assert field in version_data, ()
                    f"Missing field '{field}' in version {version_name}"
                )

        # Test report saving
        test_report_path = "test_positional_state_report.json"
        positional_state_system.save_state_report(test_report_path)

        # Verify file was created
        assert Path(test_report_path).exists(), "Report file was not created"

        # Clean up
        Path(test_report_path).unlink()

        print("✅ Comprehensive reporting test passed")
        return True

    except Exception as e:
        print(f"❌ Comprehensive reporting test failed: {e}")
        return False


def test_version_determination():
    """Test MathLib version determination."""
    print("🧪 Testing Version Determination...")

    try:
        # Test version determination from filenames
        test_cases = []
            ("mathlib_v1.py", MathLibVersion.V1),
            ("mathlib_v2.py", MathLibVersion.V2),
            ("mathlib_v3.py", MathLibVersion.V3),
            ("mathlib_v4.py", MathLibVersion.V4),
            ("unified_math_system.py", MathLibVersion.UNIFIED),
        ]
        for filename, expected_version in test_cases:
            determined_version = flake8_corrector._determine_mathlib_version()
                filename, ""
            )
            assert determined_version == expected_version, ()
                f"Expected {expected_version.value} for {filename}, got {determined_version.value if determined_version else 'None'}"
            )

        # Test version determination from content
        content_tests = []
            ('version = "1.0"', MathLibVersion.V1),
            ("MathLibV2", MathLibVersion.V2),
            ("MathLibV3", MathLibVersion.V3),
            ("MathLibV4", MathLibVersion.V4),
            ("UnifiedMathSystem", MathLibVersion.UNIFIED),
        ]
        for content, expected_version in content_tests:
            determined_version = flake8_corrector._determine_mathlib_version()
                "test.py", content
            )
            assert determined_version == expected_version, ()
                f"Expected {expected_version.value} for content '{content}', got {determined_version.value if determined_version else 'None'}"
            )

        print("✅ Version determination test passed")
        return True

    except Exception as e:
        print(f"❌ Version determination test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and generate comprehensive report."""
    print("🧮 MathLib Positional State System - Comprehensive Test Suite")
    print("=" * 70)

    tests = []
        ("Positional State Initialization", test_positional_state_initialization),
        ("32-bit Phase Orientation", test_32bit_phase_orientation),
        ("Dependency Relationships", test_dependency_relationships),
        ("Mathematical Formula Preservation", test_mathematical_formula_preservation),
        ("Flake8 Error Correction", test_flake8_error_correction),
        ("UTF-8 Compatibility", test_utf8_compatibility),
        ("Comprehensive Reporting", test_comprehensive_reporting),
        ("Version Determination", test_version_determination),
    ]
    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
            if success:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = "CRASHED"

    # Generate test summary
    print(f"\n{'=' * 70}")
    print("📊 TEST SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed / total) * 100:.1f}%")

    print("\n📋 DETAILED RESULTS:")
    for test_name, result in results.items():
        status_emoji = "✅" if result == "PASSED" else "❌"
        print(f"  {status_emoji} {test_name}: {result}")

    # Generate comprehensive report
    try:
        report = positional_state_system.get_comprehensive_report()
        report["test_results"] = results
        report["test_summary"] = {}
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "success_rate": (passed / total) * 100,
        }
        # Save test report
        with open()
            "mathlib_positional_system_test_report.json", "w", encoding="utf-8"
        ) as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n📄 Test report saved to: mathlib_positional_system_test_report.json")

    except Exception as e:
        print(f"❌ Error saving test report: {e}")

    # Final status
    if passed == total:
        print()
            "\n🎉 ALL TESTS PASSED! MathLib Positional State System is working correctly."
        )
        return True
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
