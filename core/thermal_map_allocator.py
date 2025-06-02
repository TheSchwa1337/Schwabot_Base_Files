"""
Thermal Map Allocator
===================

Implements thermal-constrained memory allocation for Schwabot's trading intelligence.
Manages memory allocation based on thermal states and system constraints.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import psutil

@dataclass
class MemoryRegion:
    """Container for memory region metrics"""
    start_address: int
    size: int
    thermal_state: float
    access_frequency: float
    last_access: datetime
    priority: float
    coherence_score: float

class ThermalMapAllocator:
    """Manages thermal-aware memory allocation"""
    
    def __init__(
        self,
        max_thermal_threshold: float = 0.9,
        min_coherence_threshold: float = 0.5,
        memory_quota: float = 0.8  # Percentage of system memory to use
    ):
        self.max_thermal_threshold = max_thermal_threshold
        self.min_coherence_threshold = min_coherence_threshold
        self.memory_quota = memory_quota
        
        # Initialize memory regions
        self.memory_regions: Dict[str, List[MemoryRegion]] = {}
        
        # Initialize thermal map
        self.thermal_map: Dict[str, float] = {}
        
        # Initialize performance metrics
        self.performance_metrics = {
            'total_allocations': 0,
            'failed_allocations': 0,
            'thermal_violations': 0,
            'coherence_violations': 0,
            'avg_thermal_state': 0.0,
            'avg_coherence_score': 0.0,
            'memory_utilization': 0.0
        }
        
        # Initialize system memory info
        self.system_memory = psutil.virtual_memory()
        self.max_memory = int(self.system_memory.total * self.memory_quota)
        self.used_memory = 0
        self.max_memory_usage = 0

    def allocate_memory(
        self,
        key: str,
        size: int,
        priority: float = 1.0
    ) -> Optional[MemoryRegion]:
        """Allocate memory with thermal constraints"""
        if size > self.max_memory_usage:
            print(f"Error: Not enough memory to allocate {size} bytes.")
            return None
        
        # Check if allocation would exceed quota
        if self.used_memory + size > self.max_memory:
            self.performance_metrics['failed_allocations'] += 1
            return None
        
        # Calculate thermal state for new region
        thermal_state = self._calculate_thermal_state(key, size)
        
        # Check thermal constraints
        if thermal_state > self.max_thermal_threshold:
            self.performance_metrics['thermal_violations'] += 1
            return None
        
        # Calculate coherence score
        coherence_score = self._calculate_coherence_score(key)
        
        # Check coherence constraints
        if coherence_score < self.min_coherence_threshold:
            self.performance_metrics['coherence_violations'] += 1
            return None
        
        # Create memory region
        region = MemoryRegion(
            start_address=self.used_memory,
            size=size,
            thermal_state=thermal_state,
            access_frequency=0.0,
            last_access=datetime.now(),
            priority=priority,
            coherence_score=coherence_score
        )
        
        # Update memory tracking
        if key not in self.memory_regions:
            self.memory_regions[key] = []
        self.memory_regions[key].append(region)
        self.used_memory += size
        
        # Update thermal map
        self.thermal_map[key] = thermal_state
        
        # Update performance metrics
        self.performance_metrics['total_allocations'] += 1
        self._update_performance_metrics()
        
        # Update max usage
        self.max_memory_usage += size
        
        return region

    def deallocate_memory(
        self,
        key: str
    ) -> bool:
        """Deallocate memory region"""
        if key not in self.memory_regions:
            print(f"Error: Region '{key}' does not exist.")
            return False
        
        # Deallocate memory and update max usage
        regions = self.memory_regions.pop(key)
        size = sum(region.size for region in regions)
        self.used_memory -= size
        
        # Update thermal map
        if not self.memory_regions:
            del self.thermal_map
        else:
            self.thermal_map[key] = self._calculate_average_thermal_state(key)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Update max usage
        self.max_memory_usage -= size
        
        return True

    def update_region_metrics(
        self,
        key: str,
        region: MemoryRegion,
        access_frequency: float
    ):
        """Update metrics for a memory region"""
        if key not in self.memory_regions or region not in self.memory_regions[key]:
            return
        
        # Update region metrics
        region.access_frequency = access_frequency
        region.last_access = datetime.now()
        
        # Recalculate thermal state
        region.thermal_state = self._calculate_thermal_state(key, region.size)
        
        # Update thermal map
        self.thermal_map[key] = self._calculate_average_thermal_state(key)
        
        # Update performance metrics
        self._update_performance_metrics()

    def get_memory_stats(self) -> Dict:
        """Get memory allocation statistics"""
        return {
            'total_allocations': self.performance_metrics['total_allocations'],
            'failed_allocations': self.performance_metrics['failed_allocations'],
            'thermal_violations': self.performance_metrics['thermal_violations'],
            'coherence_violations': self.performance_metrics['coherence_violations'],
            'avg_thermal_state': self.performance_metrics['avg_thermal_state'],
            'avg_coherence_score': self.performance_metrics['avg_coherence_score'],
            'memory_utilization': self.performance_metrics['memory_utilization'],
            'used_memory': self.used_memory,
            'max_memory': self.max_memory,
            'available_memory': self.max_memory - self.used_memory
        }

    def _calculate_thermal_state(
        self,
        key: str,
        size: int
    ) -> float:
        """Calculate thermal state for memory region"""
        # Base thermal state on size and existing thermal map
        base_thermal = min(1.0, size / self.max_memory)
        
        # Adjust based on existing thermal state
        if key in self.thermal_map:
            existing_thermal = self.thermal_map[key]
            return (base_thermal + existing_thermal) / 2
        
        return base_thermal

    def _calculate_coherence_score(self, key: str) -> float:
        """Calculate memory coherence score"""
        if key not in self.memory_regions:
            return 1.0
        
        regions = self.memory_regions[key]
        if not regions:
            return 1.0
        
        # Calculate coherence based on region distribution
        total_size = sum(region.size for region in regions)
        max_gap = 0
        
        # Sort regions by address
        sorted_regions = sorted(regions, key=lambda r: r.start_address)
        
        # Find maximum gap between regions
        for i in range(len(sorted_regions) - 1):
            gap = sorted_regions[i + 1].start_address - (
                sorted_regions[i].start_address + sorted_regions[i].size
            )
            max_gap = max(max_gap, gap)
        
        # Calculate coherence score
        if max_gap == 0:
            return 1.0
        
        return 1.0 / (1.0 + max_gap / total_size)

    def _calculate_average_thermal_state(self, key: str) -> float:
        """Calculate average thermal state for a key"""
        if key not in self.memory_regions:
            return 0.0
        
        regions = self.memory_regions[key]
        if not regions:
            return 0.0
        
        return sum(region.thermal_state for region in regions) / len(regions)

    def _update_performance_metrics(self):
        """Update performance metrics"""
        total_regions = sum(len(regions) for regions in self.memory_regions.values())
        if total_regions == 0:
            return
        
        # Calculate averages
        total_thermal = sum(
            region.thermal_state
            for regions in self.memory_regions.values()
            for region in regions
        )
        total_coherence = sum(
            region.coherence_score
            for regions in self.memory_regions.values()
            for region in regions
        )
        
        self.performance_metrics['avg_thermal_state'] = total_thermal / total_regions
        self.performance_metrics['avg_coherence_score'] = total_coherence / total_regions
        self.performance_metrics['memory_utilization'] = self.used_memory / self.max_memory

    def get_memory_usage(self) -> int:
        return self.max_memory_usage

    def print_memory_regions(self):
        for region, regions in self.memory_regions.items():
            for region in regions:
                print(f"Region: {region}, Size: {region.size} bytes")

# Example usage
if __name__ == "__main__":
    allocator = ThermalMapAllocator()
    
    # Allocate memory
    if allocator.allocate_memory('tensor_data', 1024 * 1024):  # 1MB
        print("Memory allocated successfully.")
    
    # Deallocate memory
    if allocator.deallocate_memory('tensor_data'):
        print("Memory deallocated successfully.")
    
    # Print current memory usage
    print(f"Current memory usage: {allocator.get_memory_usage()} bytes")
    
    # Print all memory regions
    allocator.print_memory_regions()
    
    # Test memory allocation
    region1 = allocator.allocate_memory('market_data', 512 * 1024)  # 512KB
    print(f"Allocated region 1: {region1 is not None}")
    
    # Update metrics
    if region1:
        allocator.update_region_metrics('market_data', region1, 0.8)
    
    # Get stats
    stats = allocator.get_memory_stats()
    print("\nMemory stats:")
    print(stats)
    
    # Deallocate
    if region1:
        allocator.deallocate_memory('market_data')
    
    # Get updated stats
    stats = allocator.get_memory_stats()
    print("\nUpdated memory stats:")
    print(stats) 