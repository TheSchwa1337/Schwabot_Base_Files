"""
News-Profit Mathematical Bridge API Endpoints
=============================================

Flask API endpoints for controlling and monitoring the news-to-profit
mathematical pipeline. Integrates with the settings panel for real-time
configuration and status monitoring.

Endpoints:
- /api/news-profit/status - Get system status
- /api/news-profit/config - Get/Update configuration
- /api/news-profit/process - Process news data
- /api/news-profit/correlations - Get hash correlations
- /api/news-profit/timings - Get profit timings
- /api/news-profit/execute - Execute trades
"""

from flask import Flask, request, jsonify, Blueprint
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from .news_profit_mathematical_bridge import (
    NewsProfitMathematicalBridge,
    create_news_profit_bridge
)
from .profit_cycle_navigator import ProfitCycleNavigator
from .btc_processor_controller import BTCProcessorController
from .fractal_controller import EnhancedFractalController

logger = logging.getLogger(__name__)

# Create Blueprint for news-profit API
news_profit_api = Blueprint('news_profit_api', __name__)

# Global bridge instance
bridge: Optional[NewsProfitMathematicalBridge] = None


def initialize_bridge():
    """Initialize the news-profit bridge with integrated components"""
    global bridge
    
    if bridge is None:
        try:
            # Initialize component controllers
            profit_navigator = ProfitCycleNavigator(None)  # Will be configured
            btc_controller = BTCProcessorController()
            fractal_controller = EnhancedFractalController()
            
            # Create integrated bridge
            bridge = create_news_profit_bridge(
                profit_navigator=profit_navigator,
                btc_controller=btc_controller,
                fractal_controller=fractal_controller
            )
            
            logger.info("News-Profit Mathematical Bridge initialized")
            
        except Exception as e:
            logger.error(f"Error initializing bridge: {e}")
            bridge = create_news_profit_bridge()  # Fallback without components
    
    return bridge


@news_profit_api.route('/api/news-profit/status', methods=['GET'])
def get_system_status():
    """Get current system status and performance metrics"""
    try:
        bridge = initialize_bridge()
        status = bridge.get_system_status()
        
        # Add runtime information
        status['runtime_info'] = {
            'bridge_initialized': bridge is not None,
            'profit_navigator_active': bridge.profit_navigator is not None,
            'btc_controller_active': bridge.btc_controller is not None,
            'fractal_controller_active': bridge.fractal_controller is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/config', methods=['GET', 'POST'])
def handle_configuration():
    """Get or update system configuration"""
    try:
        bridge = initialize_bridge()
        
        if request.method == 'GET':
            # Return current configuration
            config = {
                'correlation_threshold': bridge.correlation_threshold,
                'hash_window_minutes': bridge.hash_window_minutes,
                'profit_crystallization_threshold': bridge.profit_crystallization_threshold,
                'processing_parameters': {
                    'max_keywords': 10,
                    'entropy_classes': 4,
                    'hash_correlation_methods': ['hamming', 'bit_pattern', 'temporal']
                },
                'integration_settings': {
                    'profit_navigator_enabled': bridge.profit_navigator is not None,
                    'btc_controller_enabled': bridge.btc_controller is not None,
                    'fractal_controller_enabled': bridge.fractal_controller is not None
                }
            }
            
            return jsonify({
                'success': True,
                'config': config
            })
        
        elif request.method == 'POST':
            # Update configuration
            new_config = request.get_json()
            
            if not new_config:
                return jsonify({
                    'success': False,
                    'error': 'No configuration data provided'
                }), 400
            
            # Update bridge configuration
            bridge.update_configuration(new_config)
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated successfully',
                'updated_config': new_config
            })
            
    except Exception as e:
        logger.error(f"Error handling configuration: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/process', methods=['POST'])
def process_news_data():
    """Process raw news data through the mathematical pipeline"""
    try:
        bridge = initialize_bridge()
        
        # Get news data from request
        data = request.get_json()
        
        if not data or 'news_items' not in data:
            return jsonify({
                'success': False,
                'error': 'No news items provided'
            }), 400
        
        news_items = data['news_items']
        
        if not isinstance(news_items, list):
            return jsonify({
                'success': False,
                'error': 'News items must be a list'
            }), 400
        
        # Process asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                bridge.process_complete_pipeline(news_items)
            )
            
            return jsonify({
                'success': True,
                'results': results
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error processing news data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/correlations', methods=['GET'])
def get_hash_correlations():
    """Get current hash correlations"""
    try:
        bridge = initialize_bridge()
        
        # Get filter parameters
        min_correlation = request.args.get('min_correlation', 0.0, type=float)
        limit = request.args.get('limit', 50, type=int)
        
        # Filter correlations
        filtered_correlations = {
            signature: score 
            for signature, score in bridge.event_hash_correlations.items()
            if score >= min_correlation
        }
        
        # Sort by correlation strength
        sorted_correlations = dict(
            sorted(filtered_correlations.items(), 
                  key=lambda x: x[1], reverse=True)[:limit]
        )
        
        # Add correlation analysis
        correlation_stats = {
            'total_correlations': len(bridge.event_hash_correlations),
            'filtered_correlations': len(filtered_correlations),
            'average_correlation': sum(bridge.event_hash_correlations.values()) / 
                                 max(len(bridge.event_hash_correlations), 1),
            'max_correlation': max(bridge.event_hash_correlations.values()) 
                             if bridge.event_hash_correlations else 0.0,
            'profitable_correlations': bridge.profitable_correlations
        }
        
        return jsonify({
            'success': True,
            'correlations': sorted_correlations,
            'stats': correlation_stats,
            'filter_applied': {
                'min_correlation': min_correlation,
                'limit': limit
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting correlations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/timings', methods=['GET'])
def get_profit_timings():
    """Get current profit timings"""
    try:
        bridge = initialize_bridge()
        
        # Get filter parameters
        min_confidence = request.args.get('min_confidence', 0.0, type=float)
        active_only = request.args.get('active_only', True, type=bool)
        
        # Process timings
        timing_data = []
        
        for signature_hash, timing in bridge.profit_timings.items():
            if timing.confidence < min_confidence:
                continue
            
            # Check if timing is still active
            now = datetime.now()
            is_active = timing.entry_time <= now <= timing.exit_time
            
            if active_only and not is_active:
                continue
            
            timing_info = {
                'signature_hash': signature_hash,
                'entry_time': timing.entry_time.isoformat(),
                'exit_time': timing.exit_time.isoformat(),
                'confidence': timing.confidence,
                'profit_expectation': timing.profit_expectation,
                'risk_factor': timing.risk_factor,
                'hash_correlation_strength': timing.hash_correlation_strength,
                'is_active': is_active,
                'time_remaining': (timing.exit_time - now).total_seconds() if is_active else 0
            }
            
            timing_data.append(timing_info)
        
        # Sort by confidence
        timing_data.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate timing statistics
        timing_stats = {
            'total_timings': len(bridge.profit_timings),
            'active_timings': len([t for t in timing_data if t['is_active']]),
            'average_confidence': sum(t['confidence'] for t in timing_data) / max(len(timing_data), 1),
            'average_profit_expectation': sum(t['profit_expectation'] for t in timing_data) / max(len(timing_data), 1)
        }
        
        return jsonify({
            'success': True,
            'timings': timing_data,
            'stats': timing_stats,
            'filter_applied': {
                'min_confidence': min_confidence,
                'active_only': active_only
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting profit timings: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/execute', methods=['POST'])
def execute_trades():
    """Execute profit trades based on current timings"""
    try:
        bridge = initialize_bridge()
        
        # Get execution parameters
        data = request.get_json() or {}
        
        # Execution filters
        min_confidence = data.get('min_confidence', 0.5)
        max_risk = data.get('max_risk', 0.5)
        dry_run = data.get('dry_run', True)
        
        # Get valid timings for execution
        valid_timings = []
        now = datetime.now()
        
        for timing in bridge.profit_timings.values():
            if (timing.confidence >= min_confidence and 
                timing.risk_factor <= max_risk and
                timing.entry_time <= now <= timing.exit_time):
                valid_timings.append(timing)
        
        if not valid_timings:
            return jsonify({
                'success': True,
                'message': 'No valid timings for execution',
                'valid_timings': 0,
                'dry_run': dry_run
            })
        
        # Execute trades
        if dry_run:
            # Simulate execution
            execution_results = []
            for i, timing in enumerate(valid_timings):
                execution_results.append({
                    'trade_id': f"dry_run_{i}",
                    'confidence': timing.confidence,
                    'profit_expectation': timing.profit_expectation,
                    'risk_factor': timing.risk_factor,
                    'simulated': True
                })
        else:
            # Real execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                execution_results = loop.run_until_complete(
                    bridge.execute_profit_cycles(valid_timings)
                )
            finally:
                loop.close()
        
        return jsonify({
            'success': True,
            'message': f'Executed {len(execution_results)} trades',
            'execution_results': execution_results,
            'valid_timings': len(valid_timings),
            'dry_run': dry_run
        })
        
    except Exception as e:
        logger.error(f"Error executing trades: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/test', methods=['POST'])
def test_pipeline():
    """Test the pipeline with mock data"""
    try:
        bridge = initialize_bridge()
        
        # Generate mock news data
        mock_news = [
            {
                'id': 'test_btc_surge',
                'title': 'Bitcoin Surges Past $45,000 After Institutional Adoption News',
                'content': 'Major financial institutions announce significant cryptocurrency allocations following regulatory clarity.',
                'source': 'Bloomberg',
                'timestamp': (datetime.now() - timedelta(minutes=10)).isoformat()
            },
            {
                'id': 'test_trump_crypto',
                'title': 'Trump Announces Pro-Cryptocurrency Policy Framework',
                'content': 'Former president outlines comprehensive digital currency policy supporting blockchain innovation.',
                'source': 'Coindesk',
                'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat()
            },
            {
                'id': 'test_musk_bitcoin',
                'title': 'Elon Musk Tweets Support for Bitcoin Mining Sustainability',
                'content': 'Tesla CEO discusses renewable energy adoption in cryptocurrency mining operations.',
                'source': 'Twitter',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # Process through pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                bridge.process_complete_pipeline(mock_news)
            )
            
            return jsonify({
                'success': True,
                'test_results': results,
                'mock_data_used': len(mock_news),
                'message': 'Test pipeline completed successfully'
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error testing pipeline: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/analytics', methods=['GET'])
def get_analytics():
    """Get analytics and performance metrics"""
    try:
        bridge = initialize_bridge()
        
        # Calculate performance analytics
        analytics = {
            'processing_metrics': {
                'total_events_processed': bridge.processed_events,
                'profitable_correlations': bridge.profitable_correlations,
                'successful_trades': bridge.successful_trades,
                'success_rate': bridge.successful_trades / max(bridge.profitable_correlations, 1),
                'correlation_rate': bridge.profitable_correlations / max(bridge.processed_events, 1)
            },
            'correlation_analysis': {
                'active_correlations': len(bridge.event_hash_correlations),
                'average_correlation': sum(bridge.event_hash_correlations.values()) / 
                                     max(len(bridge.event_hash_correlations), 1),
                'correlation_distribution': _calculate_correlation_distribution(bridge),
                'threshold_performance': _analyze_threshold_performance(bridge)
            },
            'timing_analysis': {
                'active_timings': len(bridge.profit_timings),
                'average_confidence': _calculate_average_confidence(bridge),
                'profit_expectation_stats': _calculate_profit_stats(bridge),
                'risk_distribution': _calculate_risk_distribution(bridge)
            },
            'mathematical_insights': {
                'hash_signature_diversity': len(bridge.mathematical_signatures),
                'entropy_class_distribution': _calculate_entropy_distribution(bridge),
                'keyword_frequency': _analyze_keyword_frequency(bridge)
            }
        }
        
        return jsonify({
            'success': True,
            'analytics': analytics,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating analytics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/thermal', methods=['GET'])
def get_thermal_status():
    """Get current thermal status and management information"""
    try:
        bridge = initialize_bridge()
        
        thermal_status = {
            'current_state': bridge.get_current_thermal_state(),
            'thermal_integration_enabled': bridge.thermal_integration_enabled,
            'thermal_throttle_events': bridge.thermal_throttle_events,
            'thermal_history': list(bridge.thermal_history)[-10:] if hasattr(bridge, 'thermal_history') else [],
            'thermal_thresholds': {
                'cpu_threshold': bridge.thermal_cpu_threshold,
                'gpu_threshold': bridge.thermal_gpu_threshold,
                'emergency_threshold': bridge.thermal_emergency_threshold
            },
            'thermal_settings': {
                'thermal_scaling_enabled': bridge.thermal_scaling_enabled,
                'dynamic_allocation_enabled': bridge.dynamic_allocation_enabled
            }
        }
        
        return jsonify({
            'success': True,
            'thermal_status': thermal_status
        })
        
    except Exception as e:
        logger.error(f"Error getting thermal status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/allocation', methods=['GET', 'POST'])
def handle_allocation_settings():
    """Get or update CPU/GPU allocation settings"""
    try:
        bridge = initialize_bridge()
        
        if request.method == 'GET':
            # Get current allocation settings
            allocation_info = {
                'current_allocation': {
                    'cpu_percentage': bridge.cpu_allocation_percentage,
                    'gpu_percentage': bridge.gpu_allocation_percentage,
                    'processing_mode': bridge.processing_mode
                },
                'dynamic_allocation': bridge._calculate_dynamic_allocation() if hasattr(bridge, '_calculate_dynamic_allocation') else (bridge.cpu_allocation_percentage, bridge.gpu_allocation_percentage),
                'allocation_history': list(bridge.allocation_adjustments)[-10:] if hasattr(bridge, 'allocation_adjustments') else [],
                'processing_modes': ["hybrid", "cpu_only", "gpu_preferred", "thermal_aware"],
                'allocation_switches': bridge.allocation_switches if hasattr(bridge, 'allocation_switches') else 0,
                'thermal_management': {
                    'thermal_scaling_enabled': bridge.thermal_scaling_enabled,
                    'dynamic_allocation_enabled': bridge.dynamic_allocation_enabled
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
            
            # Update CPU allocation percentage
            if 'cpu_allocation_percentage' in data:
                bridge.set_cpu_allocation_percentage(data['cpu_allocation_percentage'])
            
            # Update processing mode
            if 'processing_mode' in data:
                bridge.set_processing_mode(data['processing_mode'])
            
            # Update thermal settings
            if 'thermal_scaling_enabled' in data:
                bridge.thermal_scaling_enabled = data['thermal_scaling_enabled']
            
            if 'dynamic_allocation_enabled' in data:
                bridge.dynamic_allocation_enabled = data['dynamic_allocation_enabled']
            
            # Update thermal thresholds
            if 'thermal_cpu_threshold' in data:
                bridge.thermal_cpu_threshold = data['thermal_cpu_threshold']
            
            if 'thermal_gpu_threshold' in data:
                bridge.thermal_gpu_threshold = data['thermal_gpu_threshold']
            
            return jsonify({
                'success': True,
                'message': 'Allocation settings updated successfully',
                'updated_settings': {
                    'cpu_percentage': bridge.cpu_allocation_percentage,
                    'gpu_percentage': bridge.gpu_allocation_percentage,
                    'processing_mode': bridge.processing_mode,
                    'thermal_scaling_enabled': bridge.thermal_scaling_enabled,
                    'dynamic_allocation_enabled': bridge.dynamic_allocation_enabled
                }
            })
            
    except Exception as e:
        logger.error(f"Error handling allocation settings: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/processing-load', methods=['GET'])
def get_processing_load():
    """Get current processing load and performance metrics"""
    try:
        bridge = initialize_bridge()
        
        # Get recent processing load data
        recent_loads = list(bridge.processing_load_history)[-20:] if hasattr(bridge, 'processing_load_history') else []
        
        # Calculate load statistics
        if recent_loads:
            cpu_ops = [load.get('cpu_operations', 0) for load in recent_loads]
            gpu_ops = [load.get('gpu_operations', 0) for load in recent_loads]
            total_ops = [load.get('operation_count', 0) for load in recent_loads]
            
            load_stats = {
                'avg_cpu_operations': sum(cpu_ops) / len(cpu_ops) if cpu_ops else 0,
                'avg_gpu_operations': sum(gpu_ops) / len(gpu_ops) if gpu_ops else 0,
                'avg_total_operations': sum(total_ops) / len(total_ops) if total_ops else 0,
                'cpu_utilization_ratio': sum(cpu_ops) / max(sum(total_ops), 1),
                'gpu_utilization_ratio': sum(gpu_ops) / max(sum(total_ops), 1),
                'thermal_throttle_frequency': sum(1 for load in recent_loads if load.get('thermal_throttled', False)) / len(recent_loads) if recent_loads else 0
            }
        else:
            load_stats = {
                'avg_cpu_operations': 0,
                'avg_gpu_operations': 0,
                'avg_total_operations': 0,
                'cpu_utilization_ratio': 0,
                'gpu_utilization_ratio': 0,
                'thermal_throttle_frequency': 0
            }
        
        processing_load_info = {
            'recent_loads': recent_loads,
            'load_statistics': load_stats,
            'current_allocation': {
                'cpu_percentage': bridge.cpu_allocation_percentage,
                'gpu_percentage': bridge.gpu_allocation_percentage
            },
            'thermal_state': bridge.get_current_thermal_state(),
            'performance_counters': {
                'thermal_throttle_events': bridge.thermal_throttle_events if hasattr(bridge, 'thermal_throttle_events') else 0,
                'allocation_switches': bridge.allocation_switches if hasattr(bridge, 'allocation_switches') else 0
            }
        }
        
        return jsonify({
            'success': True,
            'processing_load': processing_load_info
        })
        
    except Exception as e:
        logger.error(f"Error getting processing load: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/cooldown', methods=['GET', 'POST'])
def handle_cooldown_management():
    """Get or update GPU cooldown management settings"""
    try:
        bridge = initialize_bridge()
        
        if request.method == 'GET':
            # Get cooldown status
            cooldown_info = {
                'cooldown_manager_available': bridge.cooldown_manager is not None if hasattr(bridge, 'cooldown_manager') else False,
                'gpu_manager_available': bridge.gpu_manager is not None if hasattr(bridge, 'gpu_manager') else False,
                'thermal_integration_enabled': bridge.thermal_integration_enabled,
                'current_thermal_state': bridge.get_current_thermal_state(),
                'processing_mode': bridge.processing_mode,
                'thermal_throttle_events': bridge.thermal_throttle_events if hasattr(bridge, 'thermal_throttle_events') else 0
            }
            
            # Add cooldown manager status if available
            if hasattr(bridge, 'cooldown_manager') and bridge.cooldown_manager:
                try:
                    cooldown_info['cooldown_status'] = {
                        'active': True,
                        'current_phase': 'active',  # Would get from actual manager
                        'time_remaining': 300,      # Would get from actual manager
                        'cycle_count': 5            # Would get from actual manager
                    }
                except Exception:
                    cooldown_info['cooldown_status'] = {'active': False}
            
            return jsonify({
                'success': True,
                'cooldown_info': cooldown_info
            })
        
        elif request.method == 'POST':
            # Update cooldown settings
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No cooldown data provided'
                }), 400
            
            # For now, we can update thermal thresholds that affect cooldown
            if 'thermal_emergency_threshold' in data:
                bridge.thermal_emergency_threshold = data['thermal_emergency_threshold']
            
            if 'processing_mode' in data:
                bridge.set_processing_mode(data['processing_mode'])
            
            return jsonify({
                'success': True,
                'message': 'Cooldown settings updated',
                'updated_mode': bridge.processing_mode
            })
            
    except Exception as e:
        logger.error(f"Error handling cooldown management: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@news_profit_api.route('/api/news-profit/advanced-settings', methods=['GET', 'POST'])
def handle_advanced_settings():
    """Get or update advanced mathematical bridge settings"""
    try:
        bridge = initialize_bridge()
        
        if request.method == 'GET':
            # Get comprehensive advanced settings
            advanced_settings = {
                'processing_allocation': {
                    'cpu_allocation_percentage': bridge.cpu_allocation_percentage,
                    'gpu_allocation_percentage': bridge.gpu_allocation_percentage,
                    'processing_mode': bridge.processing_mode,
                    'dynamic_allocation_enabled': bridge.dynamic_allocation_enabled,
                    'thermal_scaling_enabled': bridge.thermal_scaling_enabled
                },
                'thermal_management': {
                    'thermal_integration_enabled': bridge.thermal_integration_enabled,
                    'thermal_cpu_threshold': bridge.thermal_cpu_threshold,
                    'thermal_gpu_threshold': bridge.thermal_gpu_threshold,
                    'thermal_emergency_threshold': bridge.thermal_emergency_threshold,
                    'current_thermal_state': bridge.get_current_thermal_state()
                },
                'mathematical_parameters': {
                    'correlation_threshold': bridge.correlation_threshold,
                    'hash_window_minutes': bridge.hash_window_minutes,
                    'profit_crystallization_threshold': bridge.profit_crystallization_threshold
                },
                'performance_metrics': {
                    'thermal_throttle_events': bridge.thermal_throttle_events if hasattr(bridge, 'thermal_throttle_events') else 0,
                    'allocation_switches': bridge.allocation_switches if hasattr(bridge, 'allocation_switches') else 0,
                    'processed_events': bridge.processed_events,
                    'successful_trades': bridge.successful_trades
                },
                'system_integration': {
                    'profit_navigator_active': bridge.profit_navigator is not None,
                    'btc_controller_active': bridge.btc_controller is not None,
                    'fractal_controller_active': bridge.fractal_controller is not None,
                    'thermal_manager_active': bridge.thermal_manager is not None if hasattr(bridge, 'thermal_manager') else False,
                    'gpu_manager_active': bridge.gpu_manager is not None if hasattr(bridge, 'gpu_manager') else False
                }
            }
            
            return jsonify({
                'success': True,
                'advanced_settings': advanced_settings
            })
        
        elif request.method == 'POST':
            # Update advanced settings
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No advanced settings data provided'
                }), 400
            
            # Update configuration through existing method
            bridge.update_configuration(data)
            
            return jsonify({
                'success': True,
                'message': 'Advanced settings updated successfully',
                'updated_at': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error handling advanced settings: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _calculate_correlation_distribution(bridge) -> Dict:
    """Calculate correlation score distribution"""
    if not bridge.event_hash_correlations:
        return {'ranges': {}, 'total': 0}
    
    ranges = {
        '0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, 
        '0.6-0.8': 0, '0.8-1.0': 0
    }
    
    for score in bridge.event_hash_correlations.values():
        if score < 0.2:
            ranges['0.0-0.2'] += 1
        elif score < 0.4:
            ranges['0.2-0.4'] += 1
        elif score < 0.6:
            ranges['0.4-0.6'] += 1
        elif score < 0.8:
            ranges['0.6-0.8'] += 1
        else:
            ranges['0.8-1.0'] += 1
    
    return {'ranges': ranges, 'total': len(bridge.event_hash_correlations)}


def _analyze_threshold_performance(bridge) -> Dict:
    """Analyze performance at different correlation thresholds"""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    performance = {}
    
    for threshold in thresholds:
        qualifying_correlations = sum(
            1 for score in bridge.event_hash_correlations.values() 
            if score >= threshold
        )
        performance[str(threshold)] = {
            'qualifying_correlations': qualifying_correlations,
            'percentage_of_total': qualifying_correlations / max(len(bridge.event_hash_correlations), 1) * 100
        }
    
    return performance


def _calculate_average_confidence(bridge) -> float:
    """Calculate average confidence of profit timings"""
    if not bridge.profit_timings:
        return 0.0
    
    return sum(timing.confidence for timing in bridge.profit_timings.values()) / len(bridge.profit_timings)


def _calculate_profit_stats(bridge) -> Dict:
    """Calculate profit expectation statistics"""
    if not bridge.profit_timings:
        return {'average': 0.0, 'max': 0.0, 'min': 0.0}
    
    expectations = [timing.profit_expectation for timing in bridge.profit_timings.values()]
    
    return {
        'average': sum(expectations) / len(expectations),
        'max': max(expectations),
        'min': min(expectations)
    }


def _calculate_risk_distribution(bridge) -> Dict:
    """Calculate risk factor distribution"""
    if not bridge.profit_timings:
        return {'low': 0, 'medium': 0, 'high': 0}
    
    distribution = {'low': 0, 'medium': 0, 'high': 0}
    
    for timing in bridge.profit_timings.values():
        if timing.risk_factor < 0.3:
            distribution['low'] += 1
        elif timing.risk_factor < 0.7:
            distribution['medium'] += 1
        else:
            distribution['high'] += 1
    
    return distribution


def _calculate_entropy_distribution(bridge) -> Dict:
    """Calculate entropy class distribution"""
    if not bridge.mathematical_signatures:
        return {'0': 0, '1': 0, '2': 0, '3': 0}
    
    distribution = {'0': 0, '1': 0, '2': 0, '3': 0}
    
    for signature in bridge.mathematical_signatures.values():
        distribution[str(signature.entropy_class)] += 1
    
    return distribution


def _analyze_keyword_frequency(bridge) -> Dict:
    """Analyze keyword frequency across events"""
    keyword_count = {}
    
    # This would analyze keywords from processed events
    # For now, return placeholder data
    return {
        'bitcoin': 15, 'trump': 8, 'institutional': 12,
        'regulation': 6, 'mining': 4, 'price': 10
    }


# Error handlers
@news_profit_api.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@news_profit_api.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# Blueprint registration function
def register_news_profit_api(app: Flask):
    """Register the news-profit API blueprint with Flask app"""
    app.register_blueprint(news_profit_api)
    logger.info("News-Profit Mathematical Bridge API endpoints registered")


# Standalone test server
if __name__ == "__main__":
    from flask import Flask
    
    app = Flask(__name__)
    register_news_profit_api(app)
    
    @app.route('/')
    def index():
        return jsonify({
            'message': 'News-Profit Mathematical Bridge API',
            'endpoints': [
                '/api/news-profit/status',
                '/api/news-profit/config',
                '/api/news-profit/process',
                '/api/news-profit/correlations',
                '/api/news-profit/timings',
                '/api/news-profit/execute',
                '/api/news-profit/test',
                '/api/news-profit/analytics'
            ]
        })
    
    app.run(debug=True, port=5000) 