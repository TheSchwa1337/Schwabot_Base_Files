"""
Schwabot Genesis Core
Central nervous system for the fractal-hive, managing global registry and system health.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging
from pathlib import Path
import json
import yaml
import threading
import hashlib
import psutil

from .pod_management import PodNode, PodConfig
from .fitness_oracle import FitnessOracle, FitnessScore
from .evolution_engine import EvolutionEngine, MutationProposal

logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """System health metrics"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    active_pods: int
    total_pods: int
    last_update: datetime
    critical_alerts: List[str]

@dataclass
class GlobalRegistry:
    """Global registry of system components"""
    pod_versions: Dict[str, str]  # pod_id -> version_hash
    mutation_history: List[Dict[str, Any]]
    system_events: List[Dict[str, Any]]
    last_snapshot: datetime

class GenesisCore:
    """Central nervous system for the fractal-hive"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the genesis core.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.health = SystemHealth(
            cpu_usage=0.0,
            memory_usage=0.0,
            gpu_usage=None,
            active_pods=0,
            total_pods=0,
            last_update=datetime.now(),
            critical_alerts=[]
        )
        
        self.registry = GlobalRegistry(
            pod_versions={},
            mutation_history=[],
            system_events=[],
            last_snapshot=datetime.now()
        )
        
        # Load configuration
        try:
            if config_path:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {}
            logger.info("Genesis core initialized with configuration")
        except Exception as e:
            logger.error(f"Failed to load genesis core config: {e}")
            self.config = {}
        
        # Thread safety
        self.registry_lock = threading.Lock()
        self.health_lock = threading.Lock()
        
        # Component references
        self.fitness_oracle: Optional[FitnessOracle] = None
        self.evolution_engine: Optional[EvolutionEngine] = None
        
    def register_pod(self, pod: PodNode) -> bool:
        """Register a pod in the global registry"""
        try:
            with self.registry_lock:
                # Calculate version hash
                version_hash = self._calculate_pod_hash(pod)
                
                # Store in registry
                self.registry.pod_versions[pod.id] = version_hash
                
                # Record system event
                self._record_system_event(
                    'pod_registered',
                    {
                        'pod_id': pod.id,
                        'version_hash': version_hash,
                        'parent_id': pod.parent_id
                    }
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Pod registration failed: {str(e)}")
            return False
            
    def _calculate_pod_hash(self, pod: PodNode) -> str:
        """Calculate immutable hash of pod's current state"""
        state = {
            'id': pod.id,
            'config': pod.config.__dict__,
            'metrics': pod.metrics.__dict__,
            'mutation_history': pod.mutation_history
        }
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        
    def register_mutation(self, mutation: MutationProposal) -> bool:
        """Register a mutation in the global registry"""
        try:
            with self.registry_lock:
                # Store mutation
                self.registry.mutation_history.append({
                    'mutation': mutation.__dict__,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Record system event
                self._record_system_event(
                    'mutation_registered',
                    {
                        'pod_id': mutation.pod_id,
                        'mutation_type': mutation.mutation_type,
                        'confidence': mutation.confidence
                    }
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Mutation registration failed: {str(e)}")
            return False
            
    def _record_system_event(self, event_type: str, details: Dict[str, Any]):
        """Record a system event"""
        self.registry.system_events.append({
            'type': event_type,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
    def update_health(self, pod_count: int, active_pod_count: int):
        """Update system health metrics"""
        try:
            with self.health_lock:
                # Update basic metrics
                self.health.cpu_usage = psutil.cpu_percent()
                self.health.memory_usage = psutil.virtual_memory().percent
                self.health.active_pods = active_pod_count
                self.health.total_pods = pod_count
                self.health.last_update = datetime.now()
                
                # Check for critical conditions
                self._check_critical_conditions()
                
        except Exception as e:
            logger.error(f"Health update failed: {str(e)}")
            
    def _check_critical_conditions(self):
        """Check for critical system conditions"""
        critical_alerts = []
        
        # Check CPU usage
        if self.health.cpu_usage > self.config.get('max_cpu_percent', 80.0):
            critical_alerts.append(f"High CPU usage: {self.health.cpu_usage}%")
            
        # Check memory usage
        if self.health.memory_usage > self.config.get('max_memory_percent', 70.0):
            critical_alerts.append(f"High memory usage: {self.health.memory_usage}%")
            
        # Update critical alerts
        self.health.critical_alerts = critical_alerts
        
        # Record critical events
        if critical_alerts:
            self._record_system_event(
                'critical_alert',
                {'alerts': critical_alerts}
            )
            
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        with self.registry_lock:
            with self.health_lock:
                return {
                    'health': self.health.__dict__,
                    'registry': {
                        'pod_count': len(self.registry.pod_versions),
                        'mutation_count': len(self.registry.mutation_history),
                        'event_count': len(self.registry.system_events)
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
    def take_snapshot(self) -> bool:
        """Take a snapshot of the system state"""
        try:
            with self.registry_lock:
                # Update snapshot time
                self.registry.last_snapshot = datetime.now()
                
                # Record snapshot event
                self._record_system_event(
                    'system_snapshot',
                    {
                        'pod_count': len(self.registry.pod_versions),
                        'mutation_count': len(self.registry.mutation_history)
                    }
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Snapshot failed: {str(e)}")
            return False
            
    def get_pod_lineage(self, pod_id: str) -> List[Dict[str, Any]]:
        """Get the lineage of a pod"""
        try:
            with self.registry_lock:
                lineage = []
                current_id = pod_id
                
                while current_id in self.registry.pod_versions:
                    # Get pod version
                    version_hash = self.registry.pod_versions[current_id]
                    
                    # Find mutation that created this version
                    mutation = next(
                        (m for m in reversed(self.registry.mutation_history)
                         if m['mutation']['pod_id'] == current_id),
                        None
                    )
                    
                    lineage.append({
                        'pod_id': current_id,
                        'version_hash': version_hash,
                        'mutation': mutation['mutation'] if mutation else None,
                        'timestamp': mutation['timestamp'] if mutation else None
                    })
                    
                    # Get parent ID from mutation
                    if mutation and 'parent_id' in mutation['mutation']:
                        current_id = mutation['mutation']['parent_id']
                    else:
                        break
                        
                return lineage
                
        except Exception as e:
            logger.error(f"Lineage retrieval failed: {str(e)}")
            return []
            
    def get_mutation_history(self, pod_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get mutation history, optionally filtered by pod"""
        try:
            with self.registry_lock:
                if pod_id is None:
                    return self.registry.mutation_history
                    
                return [
                    m for m in self.registry.mutation_history
                    if m['mutation']['pod_id'] == pod_id
                ]
                
        except Exception as e:
            logger.error(f"Mutation history retrieval failed: {str(e)}")
            return []
            
    def get_system_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get system events with optional filtering"""
        try:
            with self.registry_lock:
                events = self.registry.system_events
                
                # Apply filters
                if event_type:
                    events = [e for e in events if e['type'] == event_type]
                    
                if start_time:
                    events = [
                        e for e in events
                        if datetime.fromisoformat(e['timestamp']) >= start_time
                    ]
                    
                if end_time:
                    events = [
                        e for e in events
                        if datetime.fromisoformat(e['timestamp']) <= end_time
                    ]
                    
                return events
                
        except Exception as e:
            logger.error(f"System events retrieval failed: {str(e)}")
            return []
            
    def register_component(self, component_type: str, component: Any):
        """Register a system component"""
        if component_type == 'fitness_oracle':
            self.fitness_oracle = component
        elif component_type == 'evolution_engine':
            self.evolution_engine = component
        else:
            logger.warning(f"Unknown component type: {component_type}")
            
    def get_component(self, component_type: str) -> Optional[Any]:
        """Get a registered system component"""
        if component_type == 'fitness_oracle':
            return self.fitness_oracle
        elif component_type == 'evolution_engine':
            return self.evolution_engine
        else:
            logger.warning(f"Unknown component type: {component_type}")
            return None 