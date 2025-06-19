#!/usr/bin/env python3
"""
Integrated Profit Correlation System Demonstration
=================================================

This script demonstrates the complete integrated system including:
- Critical error handling with automatic recovery
- Enhanced GPU hash processing with thermal management
- News profit mathematical bridge
- Complete pipeline from news events to profit opportunities

Usage:
    python examples/integrated_system_demo.py
"""

import asyncio
import logging
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import random
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our integrated system components
try:
    from core.integrated_profit_correlation_system import IntegratedProfitCorrelationSystem
    from core.critical_error_handler import ErrorCategory, ErrorSeverity
    from core.enhanced_gpu_hash_processor import ProcessingMode
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    logger.info("Make sure you're running from the project root directory")
    exit(1)

class SystemDemonstrator:
    """Demonstrates the integrated profit correlation system"""
    
    def __init__(self):
        """Initialize the demonstrator"""
        self.system = None
        self.demo_config = self._load_demo_config()
        
    def _load_demo_config(self) -> Dict[str, Any]:
        """Load demonstration configuration"""
        config_path = Path("config/integrated_system_config.yaml")
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info("✅ Loaded system configuration")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        # Return basic config if file not found
        logger.info("Using default configuration")
        return {
            'integrated_system': {
                'processing_queue_size': 1000,
                'result_history_size': 5000,
                'max_processing_workers': 2,
                'correlation_batch_size': 25,
                'profit_threshold_basis_points': 25.0,
                'risk_tolerance': 0.7,
                'monitoring_interval_seconds': 5.0
            },
            'error_handler': {
                'max_error_history': 1000,
                'auto_recovery_enabled': True,
                'max_recovery_attempts': 3
            },
            'gpu_processor': {
                'gpu_queue_size': 500,
                'cpu_queue_size': 1000,
                'batch_size_gpu': 50,
                'batch_size_cpu': 25,
                'thermal_monitoring_interval': 5.0
            }
        }

    def _generate_mock_news_events(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate mock news events for demonstration"""
        
        # Sample keywords and entities
        crypto_keywords = ['bitcoin', 'btc', 'crypto', 'blockchain', 'ethereum', 'defi']
        market_actions = ['surge', 'crash', 'pump', 'dump', 'rally', 'dip', 'breakout']
        entities = ['tesla', 'microstrategy', 'blackrock', 'sec', 'fed', 'musk', 'trump']
        
        events = []
        base_time = time.time()
        
        for i in range(count):
            # Generate random keywords
            keywords = random.sample(crypto_keywords, random.randint(1, 3))
            keywords.extend(random.sample(market_actions, random.randint(0, 2)))
            keywords.extend(random.sample(entities, random.randint(0, 1)))
            
            event = {
                'id': f"news_event_{i}_{int(base_time)}",
                'timestamp': base_time - random.randint(0, 3600),  # Within last hour
                'keywords': keywords,
                'headline': f"Breaking: {' '.join(keywords[:3]).title()} Market Movement",
                'source': random.choice(['reuters', 'bloomberg', 'coindesk', 'cointelegraph']),
                'corroboration_count': random.randint(1, 5),
                'trust_score': random.uniform(0.3, 0.9),
                'hash': f"hash_{i}_{int(base_time)}",
                'profit_potential': random.uniform(0.0, 100.0)  # basis points
            }
            
            events.append(event)
        
        return events

    async def demonstrate_system_startup(self):
        """Demonstrate system startup and initialization"""
        logger.info("🚀 Demonstrating Integrated Profit Correlation System Startup")
        logger.info("=" * 60)
        
        # Initialize the system
        logger.info("📋 Initializing system with configuration...")
        self.system = IntegratedProfitCorrelationSystem(self.demo_config)
        
        # Start the system
        logger.info("⚡ Starting integrated system...")
        await self.system.start_system()
        
        # Get initial status
        status = self.system.get_system_status()
        logger.info(f"✅ System started successfully!")
        logger.info(f"   - System Running: {status['system_running']}")
        logger.info(f"   - Active Workers: {status['active_workers']}")
        logger.info(f"   - GPU Available: {status['gpu_processor_status'].get('gpu_available', False)}")
        logger.info(f"   - Processing Mode: {status['gpu_processor_status'].get('current_mode', 'unknown')}")
        
        return True

    async def demonstrate_news_processing(self):
        """Demonstrate processing news events through the complete pipeline"""
        logger.info("\n📰 Demonstrating News Event Processing Pipeline")
        logger.info("=" * 60)
        
        # Generate mock news events
        logger.info("📝 Generating mock news events...")
        news_events = self._generate_mock_news_events(20)
        logger.info(f"   Generated {len(news_events)} news events")
        
        # Process events through the pipeline
        logger.info("⚙️  Processing events through integrated pipeline...")
        start_time = time.time()
        
        results = await self.system.process_news_events(news_events)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Pipeline processing completed in {processing_time:.3f}s")
        
        # Analyze results
        successful_results = [r for r in results if r.pipeline_successful]
        profit_opportunities = [r for r in results if r.profit_opportunity_identified]
        
        logger.info(f"📊 Processing Results:")
        logger.info(f"   - Total Events: {len(news_events)}")
        logger.info(f"   - Successful Processing: {len(successful_results)}")
        logger.info(f"   - Profit Opportunities: {len(profit_opportunities)}")
        logger.info(f"   - Success Rate: {len(successful_results)/len(news_events)*100:.1f}%")
        logger.info(f"   - Opportunity Rate: {len(profit_opportunities)/len(news_events)*100:.1f}%")
        
        # Show sample results
        if profit_opportunities:
            logger.info("💰 Sample Profit Opportunities:")
            for i, opportunity in enumerate(profit_opportunities[:3]):
                logger.info(f"   {i+1}. Event: {opportunity.event_id}")
                logger.info(f"      Profit Potential: {opportunity.estimated_profit_potential:.1f}bp")
                logger.info(f"      Risk Assessment: {opportunity.risk_assessment:.3f}")
                logger.info(f"      Processing Mode: {opportunity.processing_mode.value}")
                logger.info(f"      Thermal State: {opportunity.thermal_state}")
        
        return results

    async def demonstrate_error_handling(self):
        """Demonstrate error handling and recovery mechanisms"""
        logger.info("\n🚨 Demonstrating Error Handling and Recovery")
        logger.info("=" * 60)
        
        # Get error handler reference
        error_handler = self.system.error_handler
        
        # Simulate various types of errors
        logger.info("⚠️  Simulating different error scenarios...")
        
        # 1. Simulate GPU computation error
        logger.info("   Simulating GPU computation error...")
        await error_handler.handle_critical_error(
            ErrorCategory.GPU_HASH_COMPUTATION,
            'demonstration',
            Exception("Simulated GPU memory overflow"),
            {
                'gpu_temperature': 78.0,
                'memory_usage': 0.95,
                'batch_size': 100
            }
        )
        
        # 2. Simulate thermal management error
        logger.info("   Simulating thermal management error...")
        await error_handler.handle_critical_error(
            ErrorCategory.THERMAL_MANAGEMENT,
            'demonstration',
            Exception("GPU temperature exceeded threshold"),
            {
                'gpu_temperature': 85.0,
                'thermal_zone': 'critical'
            }
        )
        
        # 3. Simulate news correlation error
        logger.info("   Simulating news correlation error...")
        await error_handler.handle_critical_error(
            ErrorCategory.NEWS_CORRELATION,
            'demonstration',
            Exception("Correlation calculation failed"),
            {
                'correlation_count': 50,
                'hash_correlation_strength': 0.7
            }
        )
        
        # Get error statistics
        error_stats = error_handler.get_error_statistics()
        logger.info("📈 Error Handler Statistics:")
        logger.info(f"   - Total Errors: {error_stats['total_errors']}")
        logger.info(f"   - Error Patterns: {error_stats['error_patterns']}")
        logger.info(f"   - Recovery Success Rates: {error_stats['recovery_success_rates']}")
        logger.info(f"   - Estimated Profit Impact: {error_stats['estimated_profit_impact']:.1f}bp")

    async def demonstrate_performance_monitoring(self):
        """Demonstrate system performance monitoring"""
        logger.info("\n📊 Demonstrating Performance Monitoring")
        logger.info("=" * 60)
        
        # Get comprehensive system status
        status = self.system.get_system_status()
        
        # Display performance metrics
        metrics = status['performance_metrics']
        logger.info("🔍 System Performance Metrics:")
        logger.info(f"   - Events Processed: {metrics['total_news_events_processed']}")
        logger.info(f"   - Correlations Calculated: {metrics['total_correlations_calculated']}")
        logger.info(f"   - Profit Opportunities: {metrics['total_profit_opportunities_identified']}")
        logger.info(f"   - Avg Processing Time: {metrics['avg_processing_time_per_event']:.3f}s")
        logger.info(f"   - GPU Utilization: {metrics['gpu_utilization_rate']:.1%}")
        logger.info(f"   - Error Recovery Rate: {metrics['error_recovery_success_rate']:.1%}")
        logger.info(f"   - System Uptime: {metrics['system_uptime']/60:.1f} minutes")
        
        # Display GPU processor status
        gpu_status = status['gpu_processor_status']
        logger.info("🖥️  GPU Processor Status:")
        logger.info(f"   - Current Mode: {gpu_status.get('current_mode', 'unknown')}")
        logger.info(f"   - Thermal Zone: {gpu_status.get('thermal_zone', 'unknown')}")
        logger.info(f"   - GPU Temperature: {gpu_status.get('gpu_temperature', 0):.1f}°C")
        logger.info(f"   - CPU Temperature: {gpu_status.get('cpu_temperature', 0):.1f}°C")
        logger.info(f"   - GPU Available: {gpu_status.get('gpu_available', False)}")
        
        # Display resource usage
        logger.info("💾 Resource Utilization:")
        logger.info(f"   - Processing Queue: {status['processing_queue_size']} items")
        logger.info(f"   - Active Workers: {status['active_workers']}")
        logger.info(f"   - Result History: {status['result_history_size']} results")

    async def demonstrate_thermal_management(self):
        """Demonstrate thermal management and adaptive processing"""
        logger.info("\n🌡️  Demonstrating Thermal Management")
        logger.info("=" * 60)
        
        gpu_processor = self.system.gpu_processor
        
        # Get current thermal state
        thermal_state = gpu_processor.thermal_state
        logger.info("🔥 Current Thermal State:")
        logger.info(f"   - CPU Temperature: {thermal_state.cpu_temp:.1f}°C")
        logger.info(f"   - GPU Temperature: {thermal_state.gpu_temp:.1f}°C")
        logger.info(f"   - Thermal Zone: {thermal_state.zone.value}")
        logger.info(f"   - Throttle Factor: {thermal_state.throttle_factor:.2f}")
        logger.info(f"   - Emergency Shutdown: {thermal_state.emergency_shutdown_triggered}")
        
        # Show processing recommendations
        recommendations = thermal_state.processing_recommendation
        logger.info("⚡ Processing Recommendations:")
        logger.info(f"   - CPU Allocation: {recommendations.get('cpu', 0):.1%}")
        logger.info(f"   - GPU Allocation: {recommendations.get('gpu', 0):.1%}")
        
        # Demonstrate adaptive processing
        logger.info("🔄 Adaptive Processing Demonstration:")
        logger.info(f"   - Current Mode: {gpu_processor.current_mode.value}")
        logger.info(f"   - GPU Available: {gpu_processor.gpu_available}")
        
        # Show how thermal state affects processing
        if thermal_state.zone.value in ['hot', 'critical']:
            logger.info("⚠️  Thermal throttling active - performance may be reduced")
        elif thermal_state.zone.value == 'warm':
            logger.info("⚡ Thermal management active - optimizing performance")
        else:
            logger.info("✅ Thermal state optimal - full performance available")

    async def demonstrate_system_shutdown(self):
        """Demonstrate clean system shutdown"""
        logger.info("\n🛑 Demonstrating System Shutdown")
        logger.info("=" * 60)
        
        logger.info("📊 Final system statistics before shutdown...")
        status = self.system.get_system_status()
        
        final_metrics = status['performance_metrics']
        logger.info("📈 Final Performance Summary:")
        logger.info(f"   - Total Events Processed: {final_metrics['total_news_events_processed']}")
        logger.info(f"   - Total Profit Opportunities: {final_metrics['total_profit_opportunities_identified']}")
        logger.info(f"   - System Uptime: {final_metrics['system_uptime']/60:.1f} minutes")
        
        # Shutdown system
        logger.info("🔌 Shutting down integrated system...")
        await self.system.stop_system()
        logger.info("✅ System shutdown completed successfully")

    async def run_complete_demonstration(self):
        """Run the complete system demonstration"""
        logger.info("🎬 Starting Integrated Profit Correlation System Demonstration")
        logger.info("=" * 80)
        
        try:
            # 1. System startup
            await self.demonstrate_system_startup()
            
            # 2. News processing pipeline
            await self.demonstrate_news_processing()
            
            # 3. Error handling
            await self.demonstrate_error_handling()
            
            # 4. Performance monitoring
            await self.demonstrate_performance_monitoring()
            
            # 5. Thermal management
            await self.demonstrate_thermal_management()
            
            # Brief pause to let system run
            logger.info("\n⏳ Running system for 30 seconds to demonstrate monitoring...")
            await asyncio.sleep(30)
            
            # 6. System shutdown
            await self.demonstrate_system_shutdown()
            
            logger.info("\n🎉 Demonstration completed successfully!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Demonstration failed: {e}")
            
            # Attempt cleanup
            if self.system and self.system.system_running:
                try:
                    await self.system.stop_system()
                except:
                    pass
            
            raise

async def main():
    """Main demonstration function"""
    demonstrator = SystemDemonstrator()
    await demonstrator.run_complete_demonstration()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        exit(1) 