"""
System Clock Sequencer v1.0
===========================

Central timing coordination for all Schwabot subsystems.
Provides deterministic cadence for entropy updates, GAN scans, 
batch order processing, dashboard updates, and vault maintenance.

Integration Points:
- HashAffinityVault for periodic analysis
- StrategyExecutionMapper for signal generation
- ThermalZoneManager for cooling cycles
- AdvancedTestHarness for validation pulses
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# Core system imports
from .hash_affinity_vault import HashAffinityVault
from .strategy_execution_mapper import StrategyExecutionMapper

# Optional imports with fallbacks
try:
    from .thermal_zone_manager import ThermalZoneManager
except ImportError:
    ThermalZoneManager = None

try:
    from .entropy_bridge import EntropyBridge
except ImportError:
    EntropyBridge = None

try:
    from .dashboard_integration import DashboardIntegration
except ImportError:
    DashboardIntegration = None

# Mock thermal manager for missing dependency
class MockThermalZoneManager:
    def get_thermal_state(self):
        return {'gpu_utilization': 0.3, 'cpu_utilization': 0.2, 'gpu_temperature': 65, 'cpu_temperature': 55}

class ScheduleFrequency(Enum):
    """Available scheduling frequencies"""
    EVERY_SECOND = "1s"
    EVERY_5_SECONDS = "5s"
    EVERY_15_SECONDS = "15s"
    EVERY_30_SECONDS = "30s"
    EVERY_MINUTE = "1m"
    EVERY_5_MINUTES = "5m"
    EVERY_15_MINUTES = "15m"
    EVERY_HOUR = "1h"
    EVERY_4_HOURS = "4h"
    EVERY_DAY = "24h"

@dataclass
class ScheduledTask:
    """Represents a scheduled system task"""
    task_id: str
    name: str
    function: Callable
    frequency: ScheduleFrequency
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    enabled: bool
    critical: bool  # Critical tasks must complete successfully
    max_runtime: int  # Maximum runtime in seconds
    error_count: int = 0
    success_count: int = 0

class SystemClockSequencer:
    """
    Central timing coordinator for all Schwabot operations.
    Ensures deterministic execution of periodic tasks with error handling.
    """
    
    def __init__(self, vault: HashAffinityVault,
                 strategy_mapper: StrategyExecutionMapper,
                 thermal_manager=None,
                 entropy_bridge=None,
                 dashboard=None):
        """
        Initialize system clock with core components
        
        Args:
            vault: Hash affinity vault for periodic analysis
            strategy_mapper: Strategy execution engine
            thermal_manager: Thermal state management (optional)
            entropy_bridge: Entropy processing bridge (optional)
            dashboard: Dashboard integration (optional)
        """
        self.vault = vault
        self.strategy_mapper = strategy_mapper
        
        # Use provided components or create mocks
        self.thermal_manager = thermal_manager or MockThermalZoneManager()
        self.entropy_bridge = entropy_bridge
        self.dashboard = dashboard
        
        # Scheduling state
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.cycle_count = 0
        self.last_cycle_time = None
        self.avg_cycle_time = 0.0
        self.system_health_score = 1.0
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default task schedule
        self._setup_default_schedule()
    
    def _setup_default_schedule(self):
        """Setup default scheduled tasks for core system operations"""
        
        # High-frequency tasks (every few seconds)
        self.register_task(
            "thermal_monitoring",
            "Monitor thermal state and adjust performance",
            self._monitor_thermal_state,
            ScheduleFrequency.EVERY_5_SECONDS,
            critical=True,
            max_runtime=2
        )
        
        self.register_task(
            "vault_analysis",
            "Analyze vault patterns and detect anomalies",
            self._analyze_vault_patterns,
            ScheduleFrequency.EVERY_15_SECONDS,
            critical=True,
            max_runtime=5
        )
        
        self.register_task(
            "signal_generation",
            "Generate trading signals from recent patterns",
            self._generate_trading_signals,
            ScheduleFrequency.EVERY_30_SECONDS,
            critical=False,
            max_runtime=10
        )
        
        # Medium-frequency tasks (every few minutes)
        self.register_task(
            "entropy_processing",
            "Process entropy data and update bridges",
            self._process_entropy_data,
            ScheduleFrequency.EVERY_5_MINUTES,
            critical=False,
            max_runtime=30
        )
        
        self.register_task(
            "vault_maintenance",
            "Perform vault cleanup and optimization",
            self._maintain_vault,
            ScheduleFrequency.EVERY_15_MINUTES,
            critical=False,
            max_runtime=60
        )
        
        self.register_task(
            "dashboard_update",
            "Update dashboard with latest metrics",
            self._update_dashboard,
            ScheduleFrequency.EVERY_MINUTE,
            critical=False,
            max_runtime=5
        )
        
        # Low-frequency tasks (hourly/daily)
        self.register_task(
            "system_health_check",
            "Comprehensive system health assessment",
            self._assess_system_health,
            ScheduleFrequency.EVERY_HOUR,
            critical=True,
            max_runtime=120
        )
        
        self.register_task(
            "performance_optimization",
            "Optimize system performance and cleanup",
            self._optimize_performance,
            ScheduleFrequency.EVERY_4_HOURS,
            critical=False,
            max_runtime=300
        )
    
    def register_task(self, task_id: str, name: str, function: Callable,
                     frequency: ScheduleFrequency, critical: bool = False,
                     max_runtime: int = 60) -> bool:
        """Register a new scheduled task"""
        
        if task_id in self.scheduled_tasks:
            self.logger.warning(f"Task {task_id} already exists, overwriting")
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            function=function,
            frequency=frequency,
            last_run=None,
            next_run=datetime.utcnow(),
            enabled=True,
            critical=critical,
            max_runtime=max_runtime
        )
        
        self.scheduled_tasks[task_id] = task
        self.logger.info(f"Registered task: {name} ({frequency.value})")
        
        return True
    
    def _get_frequency_seconds(self, frequency: ScheduleFrequency) -> int:
        """Convert frequency enum to seconds"""
        frequency_map = {
            ScheduleFrequency.EVERY_SECOND: 1,
            ScheduleFrequency.EVERY_5_SECONDS: 5,
            ScheduleFrequency.EVERY_15_SECONDS: 15,
            ScheduleFrequency.EVERY_30_SECONDS: 30,
            ScheduleFrequency.EVERY_MINUTE: 60,
            ScheduleFrequency.EVERY_5_MINUTES: 300,
            ScheduleFrequency.EVERY_15_MINUTES: 900,
            ScheduleFrequency.EVERY_HOUR: 3600,
            ScheduleFrequency.EVERY_4_HOURS: 14400,
            ScheduleFrequency.EVERY_DAY: 86400
        }
        return frequency_map.get(frequency, 60)
    
    async def start(self):
        """Start the system clock sequencer"""
        if self.running:
            self.logger.warning("Clock sequencer already running")
            return
        
        self.running = True
        self.logger.info("🕐 Starting System Clock Sequencer")
        
        # Start main scheduling loop
        try:
            await self._main_loop()
        except Exception as e:
            self.logger.error(f"Clock sequencer failed: {e}")
            self.running = False
            raise
    
    async def stop(self):
        """Stop the system clock sequencer"""
        self.logger.info("🛑 Stopping System Clock Sequencer")
        self.running = False
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
    
    async def _main_loop(self):
        """Main scheduling loop"""
        
        while self.running:
            cycle_start = time.perf_counter()
            current_time = datetime.utcnow()
            
            # Check which tasks need to run
            tasks_to_run = []
            for task in self.scheduled_tasks.values():
                if task.enabled and (task.next_run is None or current_time >= task.next_run):
                    tasks_to_run.append(task)
            
            # Execute tasks
            if tasks_to_run:
                await self._execute_tasks(tasks_to_run, current_time)
            
            # Update cycle metrics
            cycle_time = (time.perf_counter() - cycle_start) * 1000
            self.cycle_count += 1
            self.last_cycle_time = cycle_time
            
            # Update average cycle time
            if self.avg_cycle_time == 0:
                self.avg_cycle_time = cycle_time
            else:
                self.avg_cycle_time = 0.9 * self.avg_cycle_time + 0.1 * cycle_time
            
            # Sleep until next cycle (aim for 1-second granularity)
            await asyncio.sleep(1.0)
    
    async def _execute_tasks(self, tasks: List[ScheduledTask], current_time: datetime):
        """Execute a list of scheduled tasks"""
        
        # Separate critical and non-critical tasks
        critical_tasks = [t for t in tasks if t.critical]
        non_critical_tasks = [t for t in tasks if not t.critical]
        
        # Execute critical tasks first (synchronously)
        for task in critical_tasks:
            await self._execute_single_task(task, current_time)
        
        # Execute non-critical tasks (can be async)
        if non_critical_tasks:
            await asyncio.gather(*[
                self._execute_single_task(task, current_time)
                for task in non_critical_tasks
            ], return_exceptions=True)
    
    async def _execute_single_task(self, task: ScheduledTask, current_time: datetime):
        """Execute a single scheduled task with error handling"""
        
        task_start = time.perf_counter()
        
        try:
            self.logger.debug(f"Executing task: {task.name}")
            
            # Execute task with timeout
            if asyncio.iscoroutinefunction(task.function):
                await asyncio.wait_for(task.function(), timeout=task.max_runtime)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor, 
                    task.function
                )
            
            # Update task success metrics
            task.success_count += 1
            task.last_run = current_time
            
            # Schedule next run
            frequency_seconds = self._get_frequency_seconds(task.frequency)
            task.next_run = current_time + timedelta(seconds=frequency_seconds)
            
            task_time = (time.perf_counter() - task_start) * 1000
            self.logger.debug(f"Task {task.name} completed in {task_time:.1f}ms")
            
        except asyncio.TimeoutError:
            task.error_count += 1
            self.logger.error(f"Task {task.name} timed out after {task.max_runtime}s")
            
        except Exception as e:
            task.error_count += 1
            self.logger.error(f"Task {task.name} failed: {e}")
            
            # Disable task if it fails too often
            if task.error_count > 5:
                task.enabled = False
                self.logger.warning(f"Disabled task {task.name} due to repeated failures")
    
    # Core system task implementations
    
    async def _monitor_thermal_state(self):
        """Monitor thermal state and adjust performance"""
        thermal_state = self.thermal_manager.get_thermal_state()
        
        # Check for thermal issues
        if thermal_state.get('gpu_temperature', 0) > 85:
            self.logger.warning("GPU temperature high, reducing processing load")
            # Could trigger thermal throttling here
        
        if thermal_state.get('cpu_temperature', 0) > 80:
            self.logger.warning("CPU temperature high, reducing processing load")
    
    async def _analyze_vault_patterns(self):
        """Analyze vault patterns and detect anomalies"""
        if len(self.vault.vault) < 10:
            return  # Not enough data
        
        # Detect anomalies
        anomalies = self.vault.detect_anomalies()
        
        if anomalies:
            self.logger.info(f"Detected {len(anomalies)} anomalies in vault data")
            
            # Could trigger alert system here
            for anomaly in anomalies[:3]:  # Log first 3
                self.logger.debug(f"Anomaly: {anomaly['tick_id']} - Z-score: {anomaly['z_score']:.2f}")
    
    async def _generate_trading_signals(self):
        """Generate trading signals from recent patterns"""
        if len(self.vault.recent_ticks) < 5:
            return  # Need minimum data
        
        # Get latest tick signature
        latest_tick = list(self.vault.recent_ticks)[-1]
        
        # Generate signal through strategy mapper
        signal = await self.strategy_mapper.generate_trade_signal(latest_tick)
        
        if signal:
            self.logger.info(f"Generated signal: {signal.strategy_type.value} {signal.side} "
                           f"confidence={signal.confidence:.3f}")
            
            # Could execute signal here or queue for batch processing
            # For now, just log the signal generation
    
    async def _process_entropy_data(self):
        """Process entropy data and update bridges"""
        if not self.entropy_bridge:
            return
        
        try:
            # Process latest entropy data
            entropy_stats = await self.entropy_bridge.get_entropy_statistics()
            
            if entropy_stats.get('total_entropy_events', 0) > 0:
                self.logger.debug(f"Processed entropy events: {entropy_stats['total_entropy_events']}")
                
        except Exception as e:
            self.logger.warning(f"Entropy processing failed: {e}")
    
    async def _maintain_vault(self):
        """Perform vault cleanup and optimization"""
        vault_stats = self.vault.export_comprehensive_state()
        
        # Check vault utilization
        utilization = vault_stats['vault_utilization']
        
        if utilization > 0.9:
            self.logger.info(f"Vault utilization high: {utilization:.1%}")
            # Could trigger cleanup here
        
        # Check for backend performance issues
        backend_performance = vault_stats['backend_performance']
        for backend, perf in backend_performance.items():
            error_rate = perf['error_count'] / max(perf['total_ticks'], 1)
            if error_rate > 0.1:  # More than 10% error rate
                self.logger.warning(f"Backend {backend} has high error rate: {error_rate:.1%}")
    
    async def _update_dashboard(self):
        """Update dashboard with latest metrics"""
        if not self.dashboard:
            return
        
        try:
            # Compile dashboard data
            dashboard_data = {
                'system_health': self.system_health_score,
                'vault_stats': self.vault.export_comprehensive_state(),
                'execution_stats': self.strategy_mapper.get_execution_statistics(),
                'thermal_state': self.thermal_manager.get_thermal_state(),
                'clock_metrics': {
                    'cycle_count': self.cycle_count,
                    'avg_cycle_time_ms': self.avg_cycle_time,
                    'last_cycle_time_ms': self.last_cycle_time
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Broadcast update
            await self.dashboard.broadcast_update(dashboard_data)
            
        except Exception as e:
            self.logger.warning(f"Dashboard update failed: {e}")
    
    async def _assess_system_health(self):
        """Comprehensive system health assessment"""
        health_factors = []
        
        # Check task success rates
        total_success = sum(task.success_count for task in self.scheduled_tasks.values())
        total_errors = sum(task.error_count for task in self.scheduled_tasks.values())
        
        if total_success + total_errors > 0:
            task_success_rate = total_success / (total_success + total_errors)
            health_factors.append(task_success_rate)
        
        # Check vault health
        vault_stats = self.vault.export_comprehensive_state()
        if vault_stats['total_ticks'] > 0:
            anomaly_rate = len(vault_stats['recent_anomalies']) / min(vault_stats['total_ticks'], 100)
            vault_health = max(0.0, 1.0 - anomaly_rate)
            health_factors.append(vault_health)
        
        # Check thermal health
        thermal_state = self.thermal_manager.get_thermal_state()
        gpu_health = 1.0 - min(thermal_state.get('gpu_utilization', 0.0), 1.0)
        cpu_health = 1.0 - min(thermal_state.get('cpu_utilization', 0.0), 1.0)
        health_factors.extend([gpu_health, cpu_health])
        
        # Calculate overall health score
        if health_factors:
            self.system_health_score = sum(health_factors) / len(health_factors)
        else:
            self.system_health_score = 0.5  # Neutral when no data
        
        self.logger.info(f"System health score: {self.system_health_score:.3f}")
        
        # Alert if health is low
        if self.system_health_score < 0.7:
            self.logger.warning("System health score is low - investigate issues")
    
    async def _optimize_performance(self):
        """Optimize system performance and cleanup"""
        self.logger.info("Performing system optimization")
        
        # Reset error counts for tasks that have recovered
        for task in self.scheduled_tasks.values():
            if task.error_count > 0 and task.success_count > task.error_count * 2:
                old_errors = task.error_count
                task.error_count = max(0, task.error_count - 1)
                self.logger.debug(f"Reduced error count for {task.name}: {old_errors} -> {task.error_count}")
        
        # Re-enable tasks that have been disabled but might work now
        disabled_tasks = [task for task in self.scheduled_tasks.values() if not task.enabled]
        for task in disabled_tasks:
            if task.error_count <= 3:  # Give another chance
                task.enabled = True
                task.error_count = 0
                self.logger.info(f"Re-enabled task: {task.name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        task_status = {}
        for task_id, task in self.scheduled_tasks.items():
            task_status[task_id] = {
                'name': task.name,
                'enabled': task.enabled,
                'critical': task.critical,
                'frequency': task.frequency.value,
                'success_count': task.success_count,
                'error_count': task.error_count,
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'next_run': task.next_run.isoformat() if task.next_run else None
            }
        
        return {
            'running': self.running,
            'cycle_count': self.cycle_count,
            'avg_cycle_time_ms': self.avg_cycle_time,
            'last_cycle_time_ms': self.last_cycle_time,
            'system_health_score': self.system_health_score,
            'total_tasks': len(self.scheduled_tasks),
            'enabled_tasks': len([t for t in self.scheduled_tasks.values() if t.enabled]),
            'tasks': task_status
        } 