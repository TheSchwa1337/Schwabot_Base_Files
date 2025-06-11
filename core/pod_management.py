"""
Schwabot Fractal-Hive Pod Management System
Implements the core pod lifecycle management and evolution engine.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import hashlib
import json
import logging
from pathlib import Path
import threading
import psutil
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class PodMetrics:
    """Performance metrics for a trading pod"""
    p_and_l: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    regime_robustness: float = 0.0
    novelty_score: float = 0.0
    resource_usage: float = 0.0
    last_update: datetime = datetime.now()

@dataclass
class PodConfig:
    """Configuration for a trading pod"""
    strategy_template: str
    risk_parameters: Dict[str, float]
    resource_limits: Dict[str, float]
    mutation_boundaries: Dict[str, Any]
    version: str
    parent_id: Optional[str] = None
    creation_time: datetime = datetime.now()

class PodNode:
    """Represents a node in the pod evolution tree"""
    def __init__(
        self,
        pod_id: str,
        config: PodConfig,
        parent_id: Optional[str] = None
    ):
        self.id = pod_id
        self.parent_id = parent_id
        self.config = config
        self.metrics = PodMetrics()
        self.children: List[PodNode] = []
        self.mutation_history: List[Dict[str, Any]] = []
        self.is_active = False
        self.last_health_check = datetime.now()
        
    def calculate_hash(self) -> str:
        """Calculate immutable hash of pod's current state"""
        state = {
            'id': self.id,
            'config': self.config.__dict__,
            'metrics': self.metrics.__dict__,
            'mutation_history': self.mutation_history
        }
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        
    def update_metrics(self, new_metrics: Dict[str, float]):
        """Update pod performance metrics"""
        for key, value in new_metrics.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
        self.metrics.last_update = datetime.now()
        self.metrics.resource_usage = psutil.cpu_percent(interval=0.1) + psutil.virtual_memory().percent
        
    def add_child(self, child: 'PodNode'):
        """Add a child pod node"""
        self.children.append(child)
        
    def record_mutation(self, mutation_type: str, details: Dict[str, Any]):
        """Record a mutation event"""
        self.mutation_history.append({
            'type': mutation_type,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'parent_hash': self.calculate_hash()
        })

    def snapshot_metric_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.metrics.__dict__, sort_keys=True).encode()).hexdigest()

class PodLifecycleManager:
    """Manages the lifecycle of trading pods"""
    
    def __init__(
        self,
        max_pods: int = 100,
        resource_threshold: float = 0.8,
        mutation_rate: float = 0.05
    ):
        self.max_pods = max_pods
        self.resource_threshold = resource_threshold
        self.mutation_rate = mutation_rate
        self.pods: Dict[str, PodNode] = {}
        self.active_pods: Set[str] = set()
        self.pod_lock = threading.Lock()
        
    def spawn_pod(self, config: PodConfig) -> Optional[PodNode]:
        """Spawn a new trading pod"""
        with self.pod_lock:
            if len(self.pods) >= self.max_pods:
                logger.warning("Maximum pod limit reached")
                return None
                
            pod_id = f"pod_{len(self.pods)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            pod = PodNode(pod_id, config, config.parent_id)
            self.pods[pod_id] = pod
            return pod
            
    def clone_pod(self, pod_id: str) -> Optional[PodNode]:
        """Clone an existing pod with mutations"""
        with self.pod_lock:
            if pod_id not in self.pods:
                logger.error(f"Pod {pod_id} not found")
                return None
                
            parent = self.pods[pod_id]
            new_config = PodConfig(
                strategy_template=parent.config.strategy_template,
                risk_parameters=parent.config.risk_parameters.copy(),
                resource_limits=parent.config.resource_limits.copy(),
                mutation_boundaries=parent.config.mutation_boundaries.copy(),
                version=f"{parent.config.version}_clone",
                parent_id=pod_id
            )
            
            # Apply mutations based on resource usage
            if parent.metrics.resource_usage > 0.8:
                self.mutation_rate = 0.1
            else:
                self.mutation_rate = 0.05

            self._apply_mutations(new_config)
            
            return self.spawn_pod(new_config)
            
    def _apply_mutations(self, config: PodConfig):
        """Apply random mutations to pod configuration"""
        if np.random.random() < self.mutation_rate:
            # Mutate risk parameters
            for param in config.risk_parameters:
                if np.random.random() < 0.3:  # 30% chance to mutate each parameter
                    current = config.risk_parameters[param]
                    mutation = np.random.normal(0, 0.1)  # Small random change
                    config.risk_parameters[param] = max(0, min(1, current + mutation))
                    
    def prune_pod(self, pod_id: str):
        """Remove a pod from the system"""
        with self.pod_lock:
            if pod_id in self.pods:
                pod = self.pods[pod_id]
                if pod.is_active:
                    self.deactivate_pod(pod_id)
                del self.pods[pod_id]
                
    def activate_pod(self, pod_id: str) -> bool:
        """Activate a pod for trading"""
        with self.pod_lock:
            if pod_id not in self.pods:
                return False
                
            pod = self.pods[pod_id]
            if not pod.is_active:
                pod.is_active = True
                self.active_pods.add(pod_id)
                return True
            return False
            
    def deactivate_pod(self, pod_id: str) -> bool:
        """Deactivate a pod from trading"""
        with self.pod_lock:
            if pod_id not in self.pods:
                return False
                
            pod = self.pods[pod_id]
            if pod.is_active:
                pod.is_active = False
                self.active_pods.remove(pod_id)
                return True
            return False
            
    def check_resources(self) -> bool:
        """Check if system resources are within acceptable limits"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        return (cpu_percent < self.resource_threshold * 100 and 
                memory_percent < self.resource_threshold * 100)
                
    def get_pod_lineage(self, pod_id: str) -> List[PodNode]:
        """Get the lineage of a pod (ancestors)"""
        lineage = []
        current_id = pod_id
        
        while current_id in self.pods:
            pod = self.pods[current_id]
            lineage.append(pod)
            current_id = pod.parent_id
            if not current_id:
                break
                
        return lineage
        
    def get_pod_metrics(self, pod_id: str) -> Optional[PodMetrics]:
        """Get current metrics for a pod"""
        if pod_id in self.pods:
            return self.pods[pod_id].metrics
        return None 

    def auto_schedule_pods(self):
        for pod in self.pods.values():
            if pod.metrics.win_rate > 0.6 and not pod.is_active:
                self.activate_pod(pod.id)
            elif pod.metrics.max_drawdown > 0.25:
                self.deactivate_pod(pod.id) 

    def update_metrics(self, pod: PodNode, new_metrics: Dict[str, float]):
        for key, value in new_metrics.items():
            if hasattr(pod.metrics, key):
                setattr(pod.metrics, key, value)
        pod.metrics.last_update = datetime.now()
        pod.metrics.resource_usage = psutil.cpu_percent(interval=0.1) + psutil.virtual_memory().percent 

    def save_pod_to_disk(self, pod: PodNode, path: Path):
        with open(path / f"{pod.id}.json", "w") as f:
            json.dump({
                "config": pod.config.__dict__,
                "metrics": pod.metrics.__dict__,
                "mutations": pod.mutation_history
            }, f, indent=4)

    def load_pod_from_disk(self, path: Path) -> PodNode:
        with open(path, "r") as f:
            data = json.load(f)
        config = PodConfig(**data["config"])
        pod = PodNode(path.stem, config)
        pod.metrics = PodMetrics(**data["metrics"])
        pod.mutation_history = data["mutations"]
        return pod

    def generate_pod_evolution_graph(self):
        G = nx.DiGraph()
        for pod in self.pods.values():
            G.add_node(pod.id, metrics=pod.metrics.__dict__)
            if pod.parent_id:
                G.add_edge(pod.parent_id, pod.id)
        return G 

    def evaluate_phase_alignment(self, pod: PodNode, phase_handler: Any) -> bool:
        return pod.snapshot_metric_hash() in phase_handler.active_hashes 

    def crossover_pods(self, pod_a: PodNode, pod_b: PodNode) -> Optional[PodNode]:
        new_risks = {}
        for key in pod_a.config.risk_parameters:
            new_risks[key] = (pod_a.config.risk_parameters[key] + pod_b.config.risk_parameters.get(key, 0)) / 2

        new_config = PodConfig(
            strategy_template=pod_a.config.strategy_template,
            risk_parameters=new_risks,
            resource_limits=pod_a.config.resource_limits,
            mutation_boundaries=pod_a.config.mutation_boundaries,
            version=f"{pod_a.config.version}_x_{pod_b.config.version}",
            parent_id=pod_a.id
        )
        return self.spawn_pod(new_config) 

    def render_graph(self):
        G = self.generate_pod_evolution_graph()
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=700, font_size=8)
        plt.show() 