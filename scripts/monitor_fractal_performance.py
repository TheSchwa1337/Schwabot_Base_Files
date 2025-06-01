"""
Forever Fractal Performance Monitor
=================================

Monitors the performance of the Forever Fractal framework integration,
tracking metrics across RittleGEMM and NCCO components.
"""

import yaml
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime

from core.fractal_core import ForeverFractalCore
from core.triplet_matcher import TripletMatcher
from core.rittle_gemm import RittleGEMM
from ncco_core.ncco import NCCOCore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fractal_performance.log'),
        logging.StreamHandler()
    ]
)

class FractalPerformanceMonitor:
    """Monitors Forever Fractal performance metrics"""
    
    def __init__(self):
        # Load configuration
        config_path = Path(__file__).parent.parent / 'core' / 'validation_config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.fractal_core = ForeverFractalCore(
            decay_power=self.config['vector_quantization']['decay_power'],
            terms=self.config['vector_quantization']['terms'],
            dimension=self.config['vector_quantization']['dimension']
        )
        self.triplet_matcher = TripletMatcher(
            fractal_core=self.fractal_core,
            epsilon=self.config['vector_quantization']['epsilon_q'],
            min_coherence=self.config['triplet_coherence']['min_coherence']
        )
        self.rittle = RittleGEMM(ring_size=self.config['integration']['rittle']['ring_size'])
        self.ncco = NCCOCore()
        
        # Initialize metrics
        self.metrics = {
            'vector_quantization': [],
            'triplet_coherence': [],
            'profit_flow': [],
            'recursive_stability': [],
            'rittle_metrics': [],
            'ncco_metrics': [],
            'system_metrics': []
        }
        
    def monitor_vector_quantization(self) -> Dict:
        """Monitor vector quantization performance"""
        start_time = time.time()
        
        # Generate test vectors
        vectors = []
        for _ in range(100):
            t = time.time()
            vector = self.fractal_core.generate_fractal_vector(t)
            vectors.append(vector)
            
        # Calculate metrics
        quantization_time = time.time() - start_time
        vector_spacing = []
        for i in range(len(vectors)-1):
            for j in range(i+1, len(vectors)):
                diff = np.linalg.norm(np.array(vectors[i]) - np.array(vectors[j]))
                vector_spacing.append(diff)
                
        return {
            'timestamp': datetime.now().isoformat(),
            'quantization_time': quantization_time,
            'min_spacing': min(vector_spacing),
            'avg_spacing': np.mean(vector_spacing),
            'max_spacing': max(vector_spacing)
        }
        
    def monitor_triplet_coherence(self) -> Dict:
        """Monitor triplet coherence performance"""
        start_time = time.time()
        
        # Generate test triplets
        triplets = []
        for _ in range(50):
            states = []
            for _ in range(3):
                t = time.time()
                vector = self.fractal_core.generate_fractal_vector(t)
                states.append(vector)
            triplets.append(states)
            
        # Calculate metrics
        coherence_time = time.time() - start_time
        coherence_scores = []
        for triplet in triplets:
            match = self.triplet_matcher.find_matching_triplet(triplet)
            if match:
                coherence_scores.append(match.coherence)
                
        return {
            'timestamp': datetime.now().isoformat(),
            'coherence_time': coherence_time,
            'min_coherence': min(coherence_scores) if coherence_scores else 0,
            'avg_coherence': np.mean(coherence_scores) if coherence_scores else 0,
            'max_coherence': max(coherence_scores) if coherence_scores else 0
        }
        
    def monitor_profit_flow(self) -> Dict:
        """Monitor profit flow performance"""
        start_time = time.time()
        
        # Simulate profit flow
        bucket_map = {}
        last_profit = 0
        profit_deltas = []
        
        for _ in range(100):
            t = time.time()
            vector = self.fractal_core.generate_fractal_vector(t)
            profit_delta = np.sum(vector) - last_profit
            last_profit = np.sum(vector)
            profit_deltas.append(profit_delta)
            
            bucket_id = hash(str(vector)) % 100
            if bucket_id not in bucket_map:
                bucket_map[bucket_id] = 0
            bucket_map[bucket_id] += profit_delta
            
        return {
            'timestamp': datetime.now().isoformat(),
            'flow_time': time.time() - start_time,
            'min_delta': min(profit_deltas),
            'avg_delta': np.mean(profit_deltas),
            'max_delta': max(profit_deltas),
            'bucket_count': len(bucket_map)
        }
        
    def monitor_recursive_stability(self) -> Dict:
        """Monitor recursive stability performance"""
        start_time = time.time()
        
        # Generate test vectors
        vectors = []
        entropy_history = []
        
        for _ in range(50):
            t = time.time()
            vector = self.fractal_core.generate_fractal_vector(t)
            vectors.append(vector)
            
            entropy = -np.sum(np.abs(vector) * np.log2(np.abs(vector) + 1e-10))
            entropy_history.append(entropy)
            
        return {
            'timestamp': datetime.now().isoformat(),
            'stability_time': time.time() - start_time,
            'min_entropy': min(entropy_history),
            'avg_entropy': np.mean(entropy_history),
            'max_entropy': max(entropy_history),
            'entropy_increase': entropy_history[-1] - entropy_history[0]
        }
        
    def monitor_rittle_metrics(self) -> Dict:
        """Monitor RittleGEMM integration metrics"""
        start_time = time.time()
        
        # Generate test data
        profit = 0.02
        ret = 0.01
        vol = 0.5
        drift = 0.1
        exec_profit = 0.015
        rebuy = 1
        price = 100.0
        
        # Update RittleGEMM
        self.rittle.update(profit, ret, vol, drift, exec_profit, rebuy, price)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'update_time': time.time() - start_time,
            'profit': self.rittle.get('R1')[-1],
            'volume': self.rittle.get('R2')[-1],
            'drift': self.rittle.get('R7')[-1]
        }
        
    def monitor_ncco_metrics(self) -> Dict:
        """Monitor NCCO integration metrics"""
        start_time = time.time()
        
        # Generate test pattern
        pattern = np.random.rand(3)
        
        # Process through NCCO
        result = self.ncco.process_pattern(pattern)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'process_time': time.time() - start_time,
            'coherence': result['coherence'],
            'profit_bucket': result['profit_bucket']
        }
        
    def monitor_system_metrics(self) -> Dict:
        """Monitor system performance metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        try:
            # Collect metrics
            self.metrics['vector_quantization'].append(self.monitor_vector_quantization())
            self.metrics['triplet_coherence'].append(self.monitor_triplet_coherence())
            self.metrics['profit_flow'].append(self.monitor_profit_flow())
            self.metrics['recursive_stability'].append(self.monitor_recursive_stability())
            self.metrics['rittle_metrics'].append(self.monitor_rittle_metrics())
            self.metrics['ncco_metrics'].append(self.monitor_ncco_metrics())
            self.metrics['system_metrics'].append(self.monitor_system_metrics())
            
            # Check performance thresholds
            self._check_performance_thresholds()
            
            # Log metrics
            self._log_metrics()
            
        except Exception as e:
            logging.error(f"Error in monitoring cycle: {e}")
            
    def _check_performance_thresholds(self):
        """Check if metrics exceed performance thresholds"""
        thresholds = self.config['performance']
        
        # Check system metrics
        system_metrics = self.metrics['system_metrics'][-1]
        if system_metrics['cpu_percent'] > thresholds['max_cpu_usage']:
            logging.warning(f"CPU usage exceeds threshold: {system_metrics['cpu_percent']}%")
        if system_metrics['memory_percent'] > thresholds['max_memory_usage']:
            logging.warning(f"Memory usage exceeds threshold: {system_metrics['memory_percent']}%")
            
        # Check vector quantization
        vector_metrics = self.metrics['vector_quantization'][-1]
        if vector_metrics['quantization_time'] > thresholds['max_latency'] / 1000:
            logging.warning(f"Vector quantization latency exceeds threshold: {vector_metrics['quantization_time']}s")
            
        # Check triplet coherence
        triplet_metrics = self.metrics['triplet_coherence'][-1]
        if triplet_metrics['coherence_time'] > thresholds['max_latency'] / 1000:
            logging.warning(f"Triplet coherence latency exceeds threshold: {triplet_metrics['coherence_time']}s")
            
    def _log_metrics(self):
        """Log current metrics"""
        for category, metrics in self.metrics.items():
            if metrics:
                latest = metrics[-1]
                logging.info(f"{category.upper()} Metrics:")
                for key, value in latest.items():
                    if key != 'timestamp':
                        logging.info(f"  {key}: {value}")

def main():
    """Main monitoring loop"""
    monitor = FractalPerformanceMonitor()
    
    while True:
        monitor.run_monitoring_cycle()
        time.sleep(60)  # Run every minute

if __name__ == '__main__':
    main() 