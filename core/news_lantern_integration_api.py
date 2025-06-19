"""
News-Lantern Mathematical Integration API
========================================

Flask API endpoints for the comprehensive News-Lantern Mathematical Integration
framework. Provides control over mathematical sequencing, thermal management,
processing allocation, and CCXT trading integration.

Key Features:
- Complete tick sequence processing control
- Real-time thermal and allocation monitoring
- Mathematical consistency tracking
- Phase-based processing metrics
- CCXT integration for portfolio management
- Story evolution and hash correlation analytics
"""

from flask import Flask, request, jsonify, Blueprint
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time

from .news_lantern_mathematical_integration import (
    NewsLanternMathematicalIntegration,
    create_news_lantern_integration,
    ProcessingPhase,
    MathematicalTickState,
    SequenceMetrics
)
from .news_profit_mathematical_bridge import NewsProfitMathematicalBridge
from .lantern_news_intelligence_bridge import LanternNewsIntelligenceBridge
from .btc_processor_controller import BTCProcessorController
from .profit_cycle_navigator import ProfitCycleNavigator

logger = logging.getLogger(__name__)

# Create Blueprint for integration API
integration_api = Blueprint('news_lantern_integration', __name__)

# Global integration instance
integration: Optional[NewsLanternMathematicalIntegration] = None


def initialize_integration():
    """Initialize the News-Lantern Mathematical Integration framework"""
    global integration
    
    if integration is None:
        try:
            # Initialize component systems
            news_bridge = NewsProfitMathematicalBridge()
            lantern_bridge = LanternNewsIntelligenceBridge()
            btc_controller = BTCProcessorController()
            profit_navigator = ProfitCycleNavigator(None)
            
            # Create integrated framework
            integration = create_news_lantern_integration(
                news_bridge=news_bridge,
                lantern_bridge=lantern_bridge,
                btc_controller=btc_controller,
                profit_navigator=profit_navigator
            )
            
            logger.info("News-Lantern Mathematical Integration initialized")
            
        except Exception as e:
            logger.error(f"Error initializing integration: {e}")
            integration = create_news_lantern_integration()  # Fallback
    
    return integration


@integration_api.route('/api/integration/status', methods=['GET'])
def get_integration_status():
    """Get comprehensive integration status and health metrics"""
    try:
        integration = initialize_integration()
        status = integration.get_integration_status()
        
        # Add runtime information
        status['runtime_info'] = {
            'integration_initialized': integration is not None,
            'current_sequence_id': integration.current_sequence_id,
            'tick_history_length': len(integration.tick_history),
            'sequence_metrics_length': len(integration.sequence_metrics),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"Error getting integration status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@integration_api.route('/api/integration/initialize', methods=['POST'])
def initialize_integration_endpoint():
    """Initialize all integration components"""
    try:
        integration = initialize_integration()
        
        # Run async initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(integration.initialize_integration())
            
            return jsonify({
                'success': True,
                'message': 'Integration components initialized successfully',
                'initialized_at': datetime.now().isoformat()
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error initializing integration: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@integration_api.route('/api/integration/process-sequence', methods=['POST'])
def process_tick_sequence():
    """Process a complete tick sequence through all mathematical phases"""
    try:
        integration = initialize_integration()
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        news_data = data.get('news_data', [])
        market_data = data.get('market_data', {})
        
        # Validate market data
        if 'btc_price' not in market_data:
            market_data['btc_price'] = 42000.0  # Default BTC price
        if 'volume' not in market_data:
            market_data['volume'] = 1000.0  # Default volume
        
        # Process sequence asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            tick_state = loop.run_until_complete(
                integration.process_tick_sequence(news_data, market_data)
            )
            
            # Convert tick state to serializable format
            tick_state_dict = {
                'tick_id': tick_state.tick_id,
                'timestamp': tick_state.timestamp.isoformat(),
                'btc_price': tick_state.btc_price,
                'news_events_count': len(tick_state.news_events),
                'lantern_events_count': len(tick_state.lantern_events),
                'phase_state': {
                    'b4': tick_state.phase_state.b4,
                    'b8': tick_state.phase_state.b8,
                    'b42': tick_state.phase_state.b42,
                    'tier': tick_state.phase_state.tier,
                    'density': tick_state.phase_state.density,
                    'variance_short': tick_state.phase_state.variance_short,
                    'variance_mid': tick_state.phase_state.variance_mid,
                    'variance_long': tick_state.phase_state.variance_long
                },
                'thermal_state': {
                    'cpu_temp': tick_state.thermal_state.cpu_temp,
                    'gpu_temp': tick_state.thermal_state.gpu_temp,
                    'zone': tick_state.thermal_state.zone.value,
                    'load_cpu': tick_state.thermal_state.load_cpu,
                    'load_gpu': tick_state.thermal_state.load_gpu,
                    'drift_coefficient': tick_state.thermal_state.drift_coefficient
                },
                'processing_allocation': tick_state.processing_allocation,
                'story_coherence': tick_state.story_coherence,
                'profit_crystallization': tick_state.profit_crystallization,
                'hash_correlations': tick_state.hash_correlations,
                'entry_exit_signals': tick_state.entry_exit_signals
            }
            
            return jsonify({
                'success': True,
                'tick_state': tick_state_dict,
                'processing_summary': {
                    'total_processing_time': sum(integration.phase_timing.values()),
                    'phase_timings': {phase.value: timing for phase, timing in integration.phase_timing.items()},
                    'mathematical_consistency': integration.mathematical_consistency_score
                }
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error processing tick sequence: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@integration_api.route('/api/integration/thermal-status', methods=['GET'])
def get_thermal_status():
    """Get current thermal status and processing allocation"""
    try:
        integration = initialize_integration()
        
        # Get current thermal state
        thermal_state = integration.thermal_manager.get_current_state()
        
        if thermal_state:
            thermal_info = {
                'current_state': {
                    'cpu_temp': thermal_state.cpu_temp,
                    'gpu_temp': thermal_state.gpu_temp,
                    'thermal_zone': thermal_state.zone.value,
                    'cpu_load': thermal_state.load_cpu,
                    'gpu_load': thermal_state.load_gpu,
                    'drift_coefficient': thermal_state.drift_coefficient
                },
                'processing_recommendation': thermal_state.processing_recommendation,
                'thermal_efficiency': 1.0 - integration._calculate_thermal_stress(thermal_state),
                'thermal_scaling_enabled': integration.thermal_scaling_enabled,
                'dynamic_allocation_enabled': integration.dynamic_allocation_enabled
            }
        else:
            thermal_info = {
                'current_state': None,
                'thermal_available': False
            }
        
        # Add allocation targets
        thermal_info['allocation_targets'] = {
            'cpu_target': integration.cpu_allocation_target,
            'gpu_target': integration.gpu_allocation_target
        }
        
        # Add performance counters
        thermal_info['performance_counters'] = {
            'thermal_throttle_events': integration.thermal_throttle_events,
            'processed_sequences': integration.processed_sequences,
            'successful_crystallizations': integration.successful_crystallizations
        }
        
        return jsonify({
            'success': True,
            'thermal_info': thermal_info
        })
        
    except Exception as e:
        logger.error(f"Error getting thermal status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@integration_api.route('/api/integration/allocation', methods=['GET', 'POST'])
def handle_processing_allocation():
    """Get or update processing allocation settings"""
    try:
        integration = initialize_integration()
        
        if request.method == 'GET':
            # Get current allocation information
            allocation_info = {
                'current_targets': {
                    'cpu_allocation_target': integration.cpu_allocation_target,
                    'gpu_allocation_target': integration.gpu_allocation_target
                },
                'control_settings': {
                    'thermal_scaling_enabled': integration.thermal_scaling_enabled,
                    'dynamic_allocation_enabled': integration.dynamic_allocation_enabled,
                    'thermal_compensation_factor': integration.thermal_compensation_factor
                },
                'mathematical_parameters': {
                    'tick_sequence_length': integration.tick_sequence_length,
                    'story_evolution_rate': integration.story_evolution_rate,
                    'hash_correlation_threshold': integration.hash_correlation_threshold,
                    'profit_crystallization_threshold': integration.profit_crystallization_threshold
                },
                'recent_performance': {
                    'mathematical_consistency_score': integration.mathematical_consistency_score,
                    'avg_thermal_efficiency': _calculate_avg_thermal_efficiency(integration),
                    'avg_story_coherence': _calculate_avg_story_coherence(integration)
                }
            }
            
            return jsonify({
                'success': True,
                'allocation_info': allocation_info
            })
        
        elif request.method == 'POST':
            # Update allocation settings
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No allocation data provided'
                }), 400
            
            # Update configuration
            integration.update_integration_configuration(data)
            
            return jsonify({
                'success': True,
                'message': 'Processing allocation updated successfully',
                'updated_settings': {
                    'cpu_allocation_target': integration.cpu_allocation_target,
                    'gpu_allocation_target': integration.gpu_allocation_target,
                    'thermal_scaling_enabled': integration.thermal_scaling_enabled,
                    'dynamic_allocation_enabled': integration.dynamic_allocation_enabled
                }
            })
            
    except Exception as e:
        logger.error(f"Error handling processing allocation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@integration_api.route('/api/integration/sequence-metrics', methods=['GET'])
def get_sequence_metrics():
    """Get detailed sequence processing metrics"""
    try:
        integration = initialize_integration()
        
        # Get recent metrics
        limit = request.args.get('limit', 20, type=int)
        recent_metrics = integration.sequence_metrics[-limit:] if integration.sequence_metrics else []
        
        # Calculate aggregated metrics
        if recent_metrics:
            aggregated_metrics = {
                'avg_thermal_efficiency': sum(m.thermal_efficiency for m in recent_metrics) / len(recent_metrics),
                'avg_story_coherence': sum(m.story_coherence_score for m in recent_metrics) / len(recent_metrics),
                'avg_hash_correlation': sum(m.hash_correlation_strength for m in recent_metrics) / len(recent_metrics),
                'avg_profit_realization': sum(m.profit_realization_rate for m in recent_metrics) / len(recent_metrics),
                'avg_mathematical_consistency': sum(m.mathematical_consistency for m in recent_metrics) / len(recent_metrics),
                'avg_cpu_gpu_balance': sum(m.cpu_gpu_balance for m in recent_metrics) / len(recent_metrics)
            }
            
            # Calculate phase timing statistics
            phase_timing_stats = {}
            for phase in ProcessingPhase:
                phase_times = []
                for metric in recent_metrics:
                    if phase in metric.phase_processing_times:
                        phase_times.append(metric.phase_processing_times[phase])
                
                if phase_times:
                    phase_timing_stats[phase.value] = {
                        'avg_time': sum(phase_times) / len(phase_times),
                        'min_time': min(phase_times),
                        'max_time': max(phase_times),
                        'sample_count': len(phase_times)
                    }
        else:
            aggregated_metrics = {}
            phase_timing_stats = {}
        
        metrics_data = {
            'recent_metrics': [
                {
                    'thermal_efficiency': m.thermal_efficiency,
                    'story_coherence_score': m.story_coherence_score,
                    'hash_correlation_strength': m.hash_correlation_strength,
                    'profit_realization_rate': m.profit_realization_rate,
                    'mathematical_consistency': m.mathematical_consistency,
                    'cpu_gpu_balance': m.cpu_gpu_balance,
                    'memory_utilization': m.memory_utilization,
                    'phase_processing_times': {phase.value: time for phase, time in m.phase_processing_times.items()}
                } for m in recent_metrics
            ],
            'aggregated_metrics': aggregated_metrics,
            'phase_timing_statistics': phase_timing_stats,
            'total_sequences_processed': integration.processed_sequences,
            'total_successful_crystallizations': integration.successful_crystallizations,
            'mathematical_violations': integration.mathematical_violations
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics_data
        })
        
    except Exception as e:
        logger.error(f"Error getting sequence metrics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@integration_api.route('/api/integration/tick-history', methods=['GET'])
def get_tick_history():
    """Get recent tick processing history"""
    try:
        integration = initialize_integration()
        
        # Get query parameters
        limit = request.args.get('limit', 10, type=int)
        include_details = request.args.get('include_details', False, type=bool)
        
        # Get recent tick states
        recent_ticks = integration.tick_history[-limit:] if integration.tick_history else []
        
        tick_history_data = []
        for tick_state in recent_ticks:
            tick_data = {
                'tick_id': tick_state.tick_id,
                'timestamp': tick_state.timestamp.isoformat(),
                'btc_price': tick_state.btc_price,
                'story_coherence': tick_state.story_coherence,
                'profit_crystallization': tick_state.profit_crystallization,
                'thermal_zone': tick_state.thermal_state.zone.value,
                'phase_tier': tick_state.phase_state.tier,
                'phase_density': tick_state.phase_state.density,
                'correlations_count': len(tick_state.hash_correlations),
                'signals_count': len(tick_state.entry_exit_signals)
            }
            
            if include_details:
                tick_data['details'] = {
                    'news_events_count': len(tick_state.news_events),
                    'lantern_events_count': len(tick_state.lantern_events),
                    'processing_allocation': tick_state.processing_allocation,
                    'hash_correlations': tick_state.hash_correlations,
                    'entry_exit_signals': tick_state.entry_exit_signals,
                    'phase_state_full': {
                        'b4': tick_state.phase_state.b4,
                        'b8': tick_state.phase_state.b8,
                        'b42': tick_state.phase_state.b42,
                        'variance_short': tick_state.phase_state.variance_short,
                        'variance_mid': tick_state.phase_state.variance_mid,
                        'variance_long': tick_state.phase_state.variance_long
                    },
                    'thermal_state_full': {
                        'cpu_temp': tick_state.thermal_state.cpu_temp,
                        'gpu_temp': tick_state.thermal_state.gpu_temp,
                        'load_cpu': tick_state.thermal_state.load_cpu,
                        'load_gpu': tick_state.thermal_state.load_gpu,
                        'drift_coefficient': tick_state.thermal_state.drift_coefficient
                    }
                }
            
            tick_history_data.append(tick_data)
        
        return jsonify({
            'success': True,
            'tick_history': tick_history_data,
            'total_ticks_processed': len(integration.tick_history),
            'query_parameters': {
                'limit': limit,
                'include_details': include_details
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting tick history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@integration_api.route('/api/integration/crystallization-analysis', methods=['GET'])
def get_crystallization_analysis():
    """Get profit crystallization analysis and trends"""
    try:
        integration = initialize_integration()
        
        # Analyze recent tick states
        recent_ticks = integration.tick_history[-50:] if integration.tick_history else []
        
        if not recent_ticks:
            return jsonify({
                'success': True,
                'analysis': {
                    'no_data': True,
                    'message': 'No tick history available for analysis'
                }
            })
        
        # Calculate crystallization statistics
        crystallizations = [tick.profit_crystallization for tick in recent_ticks]
        story_coherences = [tick.story_coherence for tick in recent_ticks]
        phase_densities = [tick.phase_state.density for tick in recent_ticks]
        
        # Count successful crystallizations (above threshold)
        successful_crystallizations = [c for c in crystallizations if c > integration.profit_crystallization_threshold]
        
        # Analyze asset signals
        asset_signal_counts = {}
        total_signal_strength = {}
        
        for tick in recent_ticks:
            for asset, signals in tick.entry_exit_signals.items():
                if asset not in asset_signal_counts:
                    asset_signal_counts[asset] = {'STRONG': 0, 'MODERATE': 0, 'WEAK': 0, 'NONE': 0}
                    total_signal_strength[asset] = []
                
                signal_strength = signals.get('signal_strength', 'NONE')
                asset_signal_counts[asset][signal_strength] += 1
                
                if 'crystallization_score' in signals:
                    total_signal_strength[asset].append(signals['crystallization_score'])
        
        # Calculate analysis
        analysis = {
            'crystallization_statistics': {
                'total_ticks_analyzed': len(recent_ticks),
                'successful_crystallizations': len(successful_crystallizations),
                'success_rate': len(successful_crystallizations) / len(recent_ticks) if recent_ticks else 0.0,
                'avg_crystallization': sum(crystallizations) / len(crystallizations) if crystallizations else 0.0,
                'max_crystallization': max(crystallizations) if crystallizations else 0.0,
                'min_crystallization': min(crystallizations) if crystallizations else 0.0,
                'crystallization_threshold': integration.profit_crystallization_threshold
            },
            'story_coherence_analysis': {
                'avg_story_coherence': sum(story_coherences) / len(story_coherences) if story_coherences else 0.0,
                'max_story_coherence': max(story_coherences) if story_coherences else 0.0,
                'min_story_coherence': min(story_coherences) if story_coherences else 0.0,
                'coherence_correlation_with_crystallization': _calculate_correlation(story_coherences, crystallizations)
            },
            'phase_state_analysis': {
                'avg_phase_density': sum(phase_densities) / len(phase_densities) if phase_densities else 0.0,
                'density_correlation_with_crystallization': _calculate_correlation(phase_densities, crystallizations)
            },
            'asset_signal_analysis': {
                'asset_signal_counts': asset_signal_counts,
                'avg_signal_strength_by_asset': {
                    asset: sum(strengths) / len(strengths) if strengths else 0.0
                    for asset, strengths in total_signal_strength.items()
                }
            },
            'trend_analysis': {
                'recent_crystallization_trend': _calculate_trend(crystallizations[-10:]) if len(crystallizations) >= 10 else 0.0,
                'recent_coherence_trend': _calculate_trend(story_coherences[-10:]) if len(story_coherences) >= 10 else 0.0
            }
        }
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error getting crystallization analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@integration_api.route('/api/integration/api-integration-status', methods=['GET'])
def get_api_integration_status():
    """Get CCXT and portfolio API integration status"""
    try:
        integration = initialize_integration()
        
        api_status = {
            'ccxt_integration': {
                'enabled': integration.ccxt_enabled,
                'portfolio_assets': integration.portfolio_assets,
                'active_positions_count': len(integration.active_positions),
                'successful_crystallizations': integration.successful_crystallizations
            },
            'component_status': {
                'news_bridge_active': integration.news_bridge is not None,
                'lantern_bridge_active': integration.lantern_bridge is not None,
                'btc_controller_active': integration.btc_controller is not None,
                'profit_navigator_active': integration.profit_navigator is not None
            },
            'trading_parameters': {
                'profit_crystallization_threshold': integration.profit_crystallization_threshold,
                'hash_correlation_threshold': integration.hash_correlation_threshold,
                'thermal_compensation_factor': integration.thermal_compensation_factor
            }
        }
        
        # Test component connectivity
        component_health = {}
        
        try:
            if integration.btc_controller:
                # Test BTC controller
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    btc_status = loop.run_until_complete(integration.btc_controller.get_system_status())
                    component_health['btc_controller'] = {'status': 'healthy', 'connected': True}
                except Exception as e:
                    component_health['btc_controller'] = {'status': 'error', 'error': str(e)}
                finally:
                    loop.close()
            else:
                component_health['btc_controller'] = {'status': 'not_initialized'}
        except Exception as e:
            component_health['btc_controller'] = {'status': 'error', 'error': str(e)}
        
        # Add component health to status
        api_status['component_health'] = component_health
        
        return jsonify({
            'success': True,
            'api_status': api_status
        })
        
    except Exception as e:
        logger.error(f"Error getting API integration status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@integration_api.route('/api/integration/test-sequence', methods=['POST'])
def test_integration_sequence():
    """Test the integration with mock data"""
    try:
        integration = initialize_integration()
        
        # Generate comprehensive mock data
        mock_news_data = [
            {
                'id': 'test_integration_btc',
                'title': 'Bitcoin Reaches New Mathematical Correlation Patterns',
                'content': 'Advanced mathematical analysis reveals new correlation patterns in Bitcoin price movements with institutional adoption.',
                'source': 'Mathematical Integration Test',
                'timestamp': datetime.now().isoformat(),
                'keywords': ['bitcoin', 'mathematical', 'correlation', 'institutional', 'adoption']
            },
            {
                'id': 'test_integration_thermal',
                'title': 'Thermal-Aware Processing Optimizes Cryptocurrency Trading',
                'content': 'New thermal management algorithms improve processing efficiency for high-frequency trading systems.',
                'source': 'Thermal Integration Test',
                'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                'keywords': ['thermal', 'processing', 'optimization', 'trading', 'efficiency']
            },
            {
                'id': 'test_integration_lantern',
                'title': 'Lantern Core Story Evolution Drives Market Predictions',
                'content': 'Advanced story evolution algorithms from Lantern Core show promising results in market prediction accuracy.',
                'source': 'Lantern Integration Test',
                'timestamp': (datetime.now() - timedelta(minutes=10)).isoformat(),
                'keywords': ['lantern', 'story', 'evolution', 'prediction', 'market']
            }
        ]
        
        mock_market_data = {
            'btc_price': 43250.75,
            'eth_price': 2650.40,
            'xrp_price': 0.62,
            'volume': 15000.0,
            'volatility': 0.025,
            'timestamp': time.time()
        }
        
        # Process test sequence
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            start_time = time.time()
            tick_state = loop.run_until_complete(
                integration.process_tick_sequence(mock_news_data, mock_market_data)
            )
            processing_time = time.time() - start_time
            
            # Create comprehensive test results
            test_results = {
                'test_execution': {
                    'success': True,
                    'processing_time_seconds': processing_time,
                    'tick_id': tick_state.tick_id,
                    'timestamp': tick_state.timestamp.isoformat()
                },
                'mathematical_processing': {
                    'phase_state': {
                        'b4_pattern': tick_state.phase_state.b4,
                        'b8_pattern': tick_state.phase_state.b8,
                        'b42_pattern': tick_state.phase_state.b42,
                        'tier': tick_state.phase_state.tier,
                        'density': tick_state.phase_state.density
                    },
                    'thermal_state': {
                        'zone': tick_state.thermal_state.zone.value,
                        'thermal_efficiency': 1.0 - integration._calculate_thermal_stress(tick_state.thermal_state),
                        'processing_allocation': tick_state.processing_allocation
                    }
                },
                'lantern_integration': {
                    'story_coherence': tick_state.story_coherence,
                    'lantern_events_processed': len(tick_state.lantern_events),
                    'news_events_processed': len(tick_state.news_events)
                },
                'hash_correlation': {
                    'correlations_calculated': len(tick_state.hash_correlations),
                    'max_correlation': max(tick_state.hash_correlations.values()) if tick_state.hash_correlations else 0.0,
                    'avg_correlation': sum(tick_state.hash_correlations.values()) / len(tick_state.hash_correlations) if tick_state.hash_correlations else 0.0
                },
                'profit_crystallization': {
                    'crystallization_score': tick_state.profit_crystallization,
                    'threshold_met': tick_state.profit_crystallization > integration.profit_crystallization_threshold,
                    'entry_exit_signals': tick_state.entry_exit_signals
                },
                'phase_timing': {
                    phase.value: timing for phase, timing in integration.phase_timing.items()
                },
                'mathematical_consistency': integration.mathematical_consistency_score
            }
            
            return jsonify({
                'success': True,
                'test_results': test_results,
                'mock_data_used': {
                    'news_items': len(mock_news_data),
                    'market_data_keys': list(mock_market_data.keys())
                }
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error testing integration sequence: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'test_execution': {
                'success': False,
                'error_details': str(e)
            }
        }), 500


# Helper functions
def _calculate_avg_thermal_efficiency(integration: NewsLanternMathematicalIntegration) -> float:
    """Calculate average thermal efficiency from recent metrics"""
    recent_metrics = integration.sequence_metrics[-10:] if integration.sequence_metrics else []
    if recent_metrics:
        return sum(m.thermal_efficiency for m in recent_metrics) / len(recent_metrics)
    return 0.0


def _calculate_avg_story_coherence(integration: NewsLanternMathematicalIntegration) -> float:
    """Calculate average story coherence from recent metrics"""
    recent_metrics = integration.sequence_metrics[-10:] if integration.sequence_metrics else []
    if recent_metrics:
        return sum(m.story_coherence_score for m in recent_metrics) / len(recent_metrics)
    return 0.0


def _calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate correlation coefficient between two lists"""
    try:
        import numpy as np
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        correlation = np.corrcoef(x, y)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    except Exception:
        return 0.0


def _calculate_trend(values: List[float]) -> float:
    """Calculate trend (slope) of values over time"""
    try:
        import numpy as np
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]  # Linear slope
        return float(trend)
    except Exception:
        return 0.0


# Error handlers
@integration_api.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@integration_api.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# Blueprint registration function
def register_integration_api(app: Flask):
    """Register the integration API blueprint with Flask app"""
    app.register_blueprint(integration_api)
    logger.info("News-Lantern Mathematical Integration API endpoints registered")


# Standalone test server
if __name__ == "__main__":
    from flask import Flask
    
    app = Flask(__name__)
    register_integration_api(app)
    
    @app.route('/')
    def index():
        return jsonify({
            'message': 'News-Lantern Mathematical Integration API',
            'endpoints': [
                '/api/integration/status',
                '/api/integration/initialize', 
                '/api/integration/process-sequence',
                '/api/integration/thermal-status',
                '/api/integration/allocation',
                '/api/integration/sequence-metrics',
                '/api/integration/tick-history',
                '/api/integration/crystallization-analysis',
                '/api/integration/api-integration-status',
                '/api/integration/test-sequence'
            ]
        })
    
    app.run(debug=True, port=5001) 