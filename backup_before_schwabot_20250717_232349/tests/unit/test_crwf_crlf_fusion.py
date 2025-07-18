#!/usr/bin/env python3
"""
Test Script for CRWF ⇆ CRLF ⇆ Schwabot Profit Layer Fusion

This script demonstrates the complete integration of:
1. Chrono Resonance Weather Mapping (CRWF)
2. Chrono-Recursive Logic Function (CRLF)
3. Geo-Located Entropy Trigger System (GETS)
4. Enhanced Matrix Mapper
5. Profit Layer Integration

Features tested:
- Weather-entropy fusion with market logic
- Geo-located resonance mapping
- Whale activity detection
- Locationary mapping layers
- Real-time entropy resolution
- Enhanced matrix optimization
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

    ChronoResonanceWeatherMapper, WeatherDataPoint, GeoLocation, CRWFResponse,
    create_crwf_mapper
)
    ChronoRecursiveLogicFunction, CRLFResponse, CRLFTriggerState, create_crlf
)
    CRWFCRLFIntegration, WhaleActivity, ProfitVector, LocationaryMapping,
    create_crwf_crlf_integration
)
    EnhancedMatrixMapper, EnhancedMatrixEntry, create_enhanced_matrix_mapper
)

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CRWFCRLFFusionTester:
    """Comprehensive tester for CRWF-CRLF fusion system."""

    def __init__(self):
        """Initialize the fusion tester."""
        self.integration = create_crwf_crlf_integration()
        self.test_results = {}

        # Create temporary directory for matrix testing
        self.temp_dir = tempfile.mkdtemp()
        self.matrix_mapper = create_enhanced_matrix_mapper(self.temp_dir)

        logger.info("🧪 CRWF-CRLF Fusion Tester initialized")

    def test_basic_crwf_computation(self) -> Dict[str, Any]:
        """Test basic CRWF computation."""
        logger.info("🌤️ Testing Basic CRWF Computation")

        # Test location (Tiger, GA - user's root, node)'
        test_location = self.integration.crwf_mapper.get_location(34.8, -83.4, "Tiger, GA")

        # Create test weather data
        test_weather = WeatherDataPoint()
            timestamp=datetime.now(),
            latitude=34.8,
            longitude=-83.4,
            altitude=300.0,
            temperature=20.0,
            pressure=1013.25,
            humidity=60.0,
            wind_speed=5.0,
            wind_direction=180.0,
            schumann_frequency=7.83,
            geomagnetic_index=2.0,
            solar_flux=100.0
        )

        # Compute CRWF
        crwf_response = self.integration.crwf_mapper.compute_crwf(test_weather, test_location)

        results = {}
            'crwf_output': crwf_response.crwf_output,
            'entropy_score': crwf_response.entropy_score,
            'resonance_strength': crwf_response.resonance_strength,
            'weather_pattern': crwf_response.weather_pattern.value,
            'geo_alignment_score': crwf_response.geo_alignment_score,
            'temporal_resonance': crwf_response.temporal_resonance,
            'crlf_adjustment_factor': crwf_response.crlf_adjustment_factor
        }

        logger.info(f"✅ CRWF Output: {results['crwf_output']:.4f}")
        logger.info(f"✅ Entropy Score: {results['entropy_score']:.3f}")
        logger.info(f"✅ Weather Pattern: {results['weather_pattern']}")

        return results

    def test_crlf_integration(self) -> Dict[str, Any]:
        """Test CRLF integration with CRWF."""
        logger.info("🔮 Testing CRLF Integration")

        # Get test location
        test_location = self.integration.crwf_mapper.get_location(34.8, -83.4, "Tiger, GA")

        # Create test weather data
        test_weather = WeatherDataPoint()
            timestamp=datetime.now(),
            latitude=34.8,
            longitude=-83.4,
            altitude=300.0,
            temperature=20.0,
            pressure=1013.25,
            humidity=60.0,
            wind_speed=5.0,
            wind_direction=180.0,
            schumann_frequency=7.83,
            geomagnetic_index=2.0,
            solar_flux=100.0
        )

        # Compute CRWF
        crwf_response = self.integration.crwf_mapper.compute_crwf(test_weather, test_location)

        # Create test strategy vector for CRLF
        strategy_vector = np.array([0.6, 0.4, 0.3, 0.7])
        profit_curve = np.array([100, 105, 103, 108, 110, 107, 112])

        # Compute CRLF
        crlf_response = self.integration.crlf_function.compute_crlf()
            strategy_vector, profit_curve, crwf_response.entropy_score
        )

        # Integrate CRLF with CRWF
        enhanced_crlf = self.integration.integrate_crlf_with_crwf()
            test_weather, 1000000.0, crlf_response.crlf_output, test_location
        )

        results = {}
            'original_crlf_output': crlf_response.crlf_output,
            'enhanced_crlf_output': enhanced_crlf,
            'crlf_trigger_state': crlf_response.trigger_state.value,
            'crlf_confidence': crlf_response.confidence,
            'recursion_depth': crlf_response.recursion_depth,
            'enhancement_factor': enhanced_crlf / crlf_response.crlf_output if crlf_response.crlf_output != 0 else 1.0
        }

        logger.info(f"✅ Original CRLF: {results['original_crlf_output']:.4f}")
        logger.info(f"✅ Enhanced CRLF: {results['enhanced_crlf_output']:.4f}")
        logger.info(f"✅ Enhancement Factor: {results['enhancement_factor']:.3f}")

        return results

    def test_whale_activity_detection(self) -> Dict[str, Any]:
        """Test whale activity detection."""
        logger.info("🐋 Testing Whale Activity Detection")

        # Get test location
        test_location = self.integration.crwf_mapper.get_location(34.8, -83.4, "Tiger, GA")

        # Create test market data
        test_market_data = {}
            'volume': 2000000.0,  # $2M volume
            'price_change': 0.5,  # 5% price change
            'trade_count': 1000,
            'average_trade_size': 2000.0
        }

        # Scan for whale activity
        whale_activity = self.integration.scan_whale_activity(test_market_data, test_location)

        if whale_activity:
            results = {}
                'volume_spike': whale_activity.volume_spike,
                'momentum_vector': whale_activity.momentum_vector,
                'divergence_score': whale_activity.divergence_score,
                'whale_count': whale_activity.whale_count,
                'volume_entropy': whale_activity.volume_entropy,
                'momentum_alignment': whale_activity.momentum_alignment,
                'whale_confidence': whale_activity.whale_confidence
            }

            logger.info(f"✅ Volume Spike: {results['volume_spike']:.2f}x")
            logger.info(f"✅ Momentum Vector: {results['momentum_vector']:.4f}")
            logger.info(f"✅ Whale Confidence: {results['whale_confidence']:.3f}")

            return results
        else:
            logger.warning("❌ No whale activity detected")
            return {'error': 'No whale activity detected'}

    def test_profit_vector_computation(self) -> Dict[str, Any]:
        """Test profit vector computation with full fusion."""
        logger.info("💰 Testing Profit Vector Computation")

        # Get test location
        test_location = self.integration.crwf_mapper.get_location(34.8, -83.4, "Tiger, GA")

        # Create test weather data
        test_weather = WeatherDataPoint()
            timestamp=datetime.now(),
            latitude=34.8,
            longitude=-83.4,
            altitude=300.0,
            temperature=20.0,
            pressure=1013.25,
            humidity=60.0,
            wind_speed=5.0,
            wind_direction=180.0,
            schumann_frequency=7.83,
            geomagnetic_index=2.0,
            solar_flux=100.0
        )

        # Compute CRWF
        crwf_response = self.integration.crwf_mapper.compute_crwf(test_weather, test_location)

        # Create test strategy vector for CRLF
        strategy_vector = np.array([0.6, 0.4, 0.3, 0.7])
        profit_curve = np.array([100, 105, 103, 108, 110, 107, 112])

        # Compute CRLF
        crlf_response = self.integration.crlf_function.compute_crlf()
            strategy_vector, profit_curve, crwf_response.entropy_score
        )

        # Create test market data for whale activity
        test_market_data = {}
            'volume': 2000000.0,
            'price_change': 0.5,
            'trade_count': 1000,
            'average_trade_size': 2000.0
        }

        # Scan for whale activity
        whale_activity = self.integration.scan_whale_activity(test_market_data, test_location)

        # Compute profit vector
        profit_vector = self.integration.compute_profit_vector()
            crwf_response, crlf_response, whale_activity
        )

        results = {}
            'base_profit': profit_vector.base_profit,
            'crwf_enhanced_profit': profit_vector.crwf_enhanced_profit,
            'crlf_adjusted_profit': profit_vector.crlf_adjusted_profit,
            'whale_enhanced_profit': profit_vector.whale_enhanced_profit,
            'final_profit_vector': profit_vector.final_profit_vector,
            'confidence_score': profit_vector.confidence_score,
            'risk_adjustment': profit_vector.risk_adjustment,
            'weather_entropy_factor': profit_vector.weather_entropy_factor,
            'geo_resonance_factor': profit_vector.geo_resonance_factor,
            'recommendations': profit_vector.recommendations
        }

        logger.info(f"✅ Final Profit Vector: {results['final_profit_vector']:.4f}")
        logger.info(f"✅ Confidence Score: {results['confidence_score']:.3f}")
        logger.info(f"✅ Action: {results['recommendations'].get('action', 'unknown')}")

        return results

    def test_locationary_mapping(self) -> Dict[str, Any]:
        """Test locationary mapping creation."""
        logger.info("📍 Testing Locationary Mapping")

        # Create locationary mapping for Tiger, GA
        locationary_mapping = self.integration.create_locationary_mapping(34.8, -83.4, "Tiger, GA")

        results = {}
            'location_name': locationary_mapping.location.name,
            'coordinates': (locationary_mapping.location.latitude, locationary_mapping.location.longitude),
            'resonance_strength': locationary_mapping.resonance_strength,
            'phase_sync_risk': locationary_mapping.phase_sync_risk,
            'delay_factor': locationary_mapping.delay_factor,
            'pressure_weighted_timing': locationary_mapping.pressure_weighted_timing,
            'quantum_weather_alignment': locationary_mapping.quantum_weather_alignment,
            'cold_base_factor': locationary_mapping.cold_base_factor,
            'vault_sync_delay': locationary_mapping.vault_sync_delay
        }

        logger.info(f"✅ Location: {results['location_name']}")
        logger.info(f"✅ Resonance Strength: {results['resonance_strength']:.3f}")
        logger.info(f"✅ Quantum Weather Alignment: {results['quantum_weather_alignment']:.3f}")

        return results

    def test_enhanced_matrix_mapper(self) -> Dict[str, Any]:
        """Test enhanced matrix mapper with CRWF-CRLF integration."""
        logger.info("🔧 Testing Enhanced Matrix Mapper")

        # Get test location
        test_location = self.integration.crwf_mapper.get_location(34.8, -83.4, "Tiger, GA")

        # Create test hash vector
        test_hash_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Create test strategy weights
        strategy_weights = {}
            'momentum': 0.3,
            'scalping': 0.2,
            'mean_reversion': 0.3,
            'swing': 0.2
        }

        # Create geo-optimized matrix
        matrix_name = self.matrix_mapper.create_geo_optimized_matrix()
            test_hash_vector, test_location, strategy_weights
        )

        if matrix_name:
            # Test matrix matching
            match_result = self.matrix_mapper.match_hash_to_enhanced_matrix()
                test_hash_vector, test_location, threshold=0.5
            )

            # Get entropy aligned matrices
            aligned_matrices = self.matrix_mapper.get_entropy_aligned_matrices()
                test_location, min_entropy_score=0.5
            )

            results = {}
                'matrix_created': matrix_name,
                'match_found': match_result is not None,
                'match_score': match_result[2] if match_result else 0.0,
                'aligned_matrices_count': len(aligned_matrices),
                'matrix_performance': self.matrix_mapper.get_matrix_performance_summary()
            }

            logger.info(f"✅ Matrix Created: {results['matrix_created']}")
            logger.info(f"✅ Match Found: {results['match_found']}")
            logger.info(f"✅ Match Score: {results['match_score']:.3f}")

            return results
        else:
            logger.error("❌ Failed to create geo-optimized matrix")
            return {'error': 'Failed to create matrix'}

    def test_multiple_locations(self) -> Dict[str, Any]:
        """Test the system with multiple locations."""
        logger.info("🌍 Testing Multiple Locations")

        # Test locations around the world
        test_locations = []
            (34.8, -83.4, "Tiger, GA"),      # User's root node'
            (40.7128, -74.060, "New York, NY"),
            (51.5074, -0.1278, "London, UK"),
            (35.6762, 139.6503, "Tokyo, Japan"),
            (-33.8688, 151.2093, "Sydney, Australia")
        ]

        results = {}

        for lat, lon, name in test_locations:
            logger.info(f"📍 Testing location: {name}")

            # Get location
            location = self.integration.crwf_mapper.get_location(lat, lon, name)

            # Create test weather data
            weather = WeatherDataPoint()
                timestamp=datetime.now(),
                latitude=lat,
                longitude=lon,
                altitude=100.0,
                temperature=20.0,
                pressure=1013.25,
                humidity=60.0,
                wind_speed=5.0,
                wind_direction=180.0
            )

            # Compute CRWF
            crwf_response = self.integration.crwf_mapper.compute_crwf(weather, location)

            # Create locationary mapping
            locationary_mapping = self.integration.create_locationary_mapping(lat, lon, name)

            results[name] = {}
                'crwf_output': crwf_response.crwf_output,
                'entropy_score': crwf_response.entropy_score,
                'resonance_strength': crwf_response.resonance_strength,
                'geo_alignment_score': crwf_response.geo_alignment_score,
                'locationary_resonance': locationary_mapping.resonance_strength,
                'quantum_alignment': locationary_mapping.quantum_weather_alignment
            }

        logger.info(f"✅ Tested {len(test_locations)} locations")

        return results

    def test_performance_summary(self) -> Dict[str, Any]:
        """Test comprehensive performance summary."""
        logger.info("📊 Testing Performance Summary")

        # Get performance summaries from all components
        crwf_summary = self.integration.crwf_mapper.get_performance_summary()
        crlf_summary = self.integration.crlf_function.get_performance_summary()
        integration_summary = self.integration.get_performance_summary()
        matrix_summary = self.matrix_mapper.get_matrix_performance_summary()

        results = {}
            'crwf_performance': crwf_summary,
            'crlf_performance': crlf_summary,
            'integration_performance': integration_summary,
            'matrix_performance': matrix_summary
        }

        logger.info("✅ Performance summaries generated")

        return results

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of the entire CRWF-CRLF fusion system."""
        logger.info("🚀 Starting Comprehensive CRWF-CRLF Fusion Test")
        logger.info("=" * 80)

        try:
            # Run all tests
            self.test_results['basic_crwf'] = self.test_basic_crwf_computation()
            self.test_results['crlf_integration'] = self.test_crlf_integration()
            self.test_results['whale_activity'] = self.test_whale_activity_detection()
            self.test_results['profit_vector'] = self.test_profit_vector_computation()
            self.test_results['locationary_mapping'] = self.test_locationary_mapping()
            self.test_results['enhanced_matrix'] = self.test_enhanced_matrix_mapper()
            self.test_results['multiple_locations'] = self.test_multiple_locations()
            self.test_results['performance_summary'] = self.test_performance_summary()

            # Generate overall summary
            self.test_results['overall_summary'] = self._generate_overall_summary()

            logger.info("🎉 Comprehensive test completed successfully!")
            logger.info("=" * 80)

            return self.test_results

        except Exception as e:
            logger.error(f"❌ Test failed with error: {e}")
            return {'error': str(e)}

    def _generate_overall_summary(self) -> Dict[str, Any]:
        """Generate overall test summary."""
        summary = {}
            'total_tests': len(self.test_results),
            'successful_tests': len([r for r in self.test_results.values() if 'error' not in r]),
            'test_names': list(self.test_results.keys()),
            'timestamp': datetime.now().isoformat()
        }

        # Add key metrics
        if 'profit_vector' in self.test_results:
            summary['final_profit_vector'] = self.test_results['profit_vector'].get('final_profit_vector', 0.0)
            summary['confidence_score'] = self.test_results['profit_vector'].get('confidence_score', 0.0)

        if 'basic_crwf' in self.test_results:
            summary['crwf_output'] = self.test_results['basic_crwf'].get('crwf_output', 0.0)
            summary['entropy_score'] = self.test_results['basic_crwf'].get('entropy_score', 0.0)

        return summary

    def print_detailed_results(self):
        """Print detailed test results."""
        logger.info("\n📋 DETAILED TEST RESULTS")
        logger.info("=" * 80)

        for test_name, results in self.test_results.items():
            logger.info(f"\n🔍 {test_name.upper().replace('_', ' ')}:")
            logger.info("-" * 40)

            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {results}")


def main():
    """Main function to run the comprehensive test."""
    logger.info("🌀 CRWF-CRLF Fusion System Test")
    logger.info("=" * 80)

    # Create tester
    tester = CRWFCRLFFusionTester()

    # Run comprehensive test
    results = tester.run_comprehensive_test()

    # Print detailed results
    tester.print_detailed_results()

    # Print overall summary
    if 'overall_summary' in results:
        summary = results['overall_summary']
        logger.info(f"\n🎯 OVERALL SUMMARY:")
        logger.info(f"   Total Tests: {summary['total_tests']}")
        logger.info(f"   Successful: {summary['successful_tests']}")
        logger.info(f"   Final Profit Vector: {summary.get('final_profit_vector', 0.0):.4f}")
        logger.info(f"   Confidence Score: {summary.get('confidence_score', 0.0):.3f}")
        logger.info(f"   CRWF Output: {summary.get('crwf_output', 0.0):.4f}")
        logger.info(f"   Entropy Score: {summary.get('entropy_score', 0.0):.3f}")

    logger.info("\n🎉 Test completed!")


if __name__ == '__main__':
    main() 