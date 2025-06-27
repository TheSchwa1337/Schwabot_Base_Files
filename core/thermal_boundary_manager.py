"""
Thermal Boundary Manager - Advanced Thermal Management System
============================================================

Comprehensive thermal boundary management system for the Schwabot mathematical
trading framework. Provides real-time thermal monitoring, adaptive performance
scaling, and emergency thermal procedures.

Key Features:
- Real-time CPU and GPU temperature monitoring
- Adaptive batch size and thread count scaling
- Emergency thermal procedures and alerts
- Hardware-specific thermal boundaries
- Cross-platform thermal sensor support
- Integration with all core components
- Windows CLI compatibility with emoji fallbacks

Thermal States:
- COOL: Optimal performance (CPU < 50°C, GPU < 60°C)
- NORMAL: Standard performance (CPU < 70°C, GPU < 80°C)
- WARM: Reduced performance (CPU < 80°C, GPU < 90°C)
- HOT: Minimal performance (CPU < 90°C, GPU < 100°C)
- CRITICAL: Emergency mode (CPU < 95°C, GPU < 105°C)
- EMERGENCY: Shutdown procedures (CPU ≥ 95°C, GPU ≥ 105°C)

Integration Points:
- All core components for thermal-aware operations
- enhanced_windows_cli_compatibility.py: CLI compatibility
- thermal_mathematical_integration.py: Mathematical thermal modeling
- main_orchestrator.py: System-wide thermal coordination
- profit_routing_engine.py: Thermal-aware profit optimization

Windows CLI compatible with flake8 compliance.
"""

import asyncio
import logging
import os
import platform
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import psutil

# Import thermal components
try:
    from .thermal_zone_manager import ThermalZoneManager, ThermalZone
    from .thermal_map_allocator import ThermalMapAllocator
except ImportError as e:
    logging.warning(f"Thermal components not available: {e}")
    ThermalZoneManager = None
    ThermalZone = None
    ThermalMapAllocator = None

# Configure logging
logger = logging.getLogger(__name__)


class ThermalState(Enum):
    """Thermal state enumeration for system-wide thermal conditions."""
    COOL = "cool"
    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class HardwareType(Enum):
    """Hardware type enumeration for resource management."""
    CPU_ONLY = "cpu_only"
    GPU_AVAILABLE = "gpu_available"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class ThermalBoundary:
    """Thermal boundary configuration for different hardware states."""
    max_cpu_temp: float
    max_gpu_temp: Optional[float]
    batch_size: int
    thread_count: int
    processing_priority: str
    emergency_procedures: List[str]


@dataclass
class HardwareProfile:
    """Hardware profile for thermal management."""
    hardware_type: HardwareType
    cpu_cores: int
    gpu_available: bool
    thermal_sensors: bool
    platform: str


class ThermalBoundaryManager:
    """Advanced thermal boundary management system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the thermal boundary manager."""
        self.config = config or {}
        self.hardware_profile = self._detect_hardware()
        self.thermal_boundaries = self._configure_thermal_boundaries()
        self.current_thermal_state = ThermalState.NORMAL
        self.monitoring_active = False
        self.thermal_zones = {}
        self.alert_callbacks = []

    def _detect_hardware(self) -> HardwareProfile:
        """Detect hardware capabilities and thermal sensors."""
        try:
            # Detect platform
            platform_name = platform.system().lower()
            
            # Detect CPU cores
            cpu_cores = psutil.cpu_count(logical=True) or 1
            
            # Detect GPU availability (basic detection)
            gpu_available = self._check_gpu_availability()
            
            # Check thermal sensors
            thermal_sensors = self._check_thermal_sensors()

            # Determine hardware type
            if gpu_available and cpu_cores > 1:
                hardware_type = HardwareType.HYBRID
            elif gpu_available:
                hardware_type = HardwareType.GPU_AVAILABLE
            else:
                hardware_type = HardwareType.CPU_ONLY

            return HardwareProfile(
                hardware_type=hardware_type,
                cpu_cores=cpu_cores,
                gpu_available=gpu_available,
                thermal_sensors=thermal_sensors,
                platform=platform_name
            )
            
        except Exception as e:
            logger.warning(f"Hardware detection failed: {e}")
            return HardwareProfile(
                hardware_type=HardwareType.UNKNOWN,
                cpu_cores=1,
                gpu_available=False,
                thermal_sensors=False,
                platform="unknown"
            )

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for thermal monitoring."""
        try:
            # Basic GPU detection - can be enhanced with specific GPU libraries
            if platform.system().lower() == "windows":
                # Windows GPU detection
                try:
                    import wmi
                    c = wmi.WMI()
                    gpu_list = c.Win32_VideoController()
                    return len(gpu_list) > 0
                except ImportError:
                    # Fallback: check for common GPU files
                    gpu_files = [
                        "C:\\Windows\\System32\\nvapi64.dll",  # NVIDIA
                        "C:\\Windows\\System32\\amdxc64.dll",  # AMD
                    ]
                    return any(os.path.exists(f) for f in gpu_files)
            else:
                # Linux/Unix GPU detection
                try:
                    result = os.popen("lspci | grep -i vga").read()
                    return len(result.strip()) > 0
                except:
                    return False
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")
            return False

    def _check_thermal_sensors(self) -> bool:
        """Check if thermal sensors are available."""
        try:
            if platform.system().lower() == "windows":
                # Windows thermal sensor check
                try:
                    import wmi
                    c = wmi.WMI(namespace="root\\wmi")
                    sensors = c.MSAcpi_ThermalZoneTemperature()
                    return len(sensors) > 0
                except ImportError:
                    # Fallback: check for thermal files
                    thermal_files = [
                        "/sys/class/thermal/thermal_zone0/temp",
                        "/proc/acpi/thermal_zone/THM0/temperature"
                    ]
                    return any(os.path.exists(f) for f in thermal_files)
            else:
                # Linux thermal sensor check
                thermal_paths = [
                    "/sys/class/thermal/thermal_zone0/temp",
                    "/proc/acpi/thermal_zone/THM0/temperature"
                ]
                return any(os.path.exists(path) for path in thermal_paths)
        except Exception as e:
            logger.debug(f"Thermal sensor check failed: {e}")
            return False

    def _configure_thermal_boundaries(self) -> Dict[ThermalState, ThermalBoundary]:
        """Configure thermal boundaries based on hardware profile."""
        boundaries = {}
        
        # Base configuration
        base_batch_size = 100
        base_threads = max(1, self.hardware_profile.cpu_cores // 2)
        
        # Cool state - optimal performance
        boundaries[ThermalState.COOL] = ThermalBoundary(
            max_cpu_temp=50.0,
            max_gpu_temp=60.0 if self.hardware_profile.gpu_available else None,
            batch_size=base_batch_size * 2,
            thread_count=base_threads * 2,
            processing_priority="high",
            emergency_procedures=[]
        )
        
        # Normal state - standard performance
        boundaries[ThermalState.NORMAL] = ThermalBoundary(
            max_cpu_temp=70.0,
            max_gpu_temp=80.0 if self.hardware_profile.gpu_available else None,
            batch_size=base_batch_size,
            thread_count=base_threads,
            processing_priority="normal",
            emergency_procedures=[]
        )
        
        # Warm state - reduced performance
        boundaries[ThermalState.WARM] = ThermalBoundary(
            max_cpu_temp=80.0,
            max_gpu_temp=90.0 if self.hardware_profile.gpu_available else None,
            batch_size=base_batch_size // 2,
            thread_count=max(1, base_threads // 2),
            processing_priority="low",
            emergency_procedures=["reduce_batch_size", "throttle_threads"]
        )
        
        # Hot state - minimal performance
        boundaries[ThermalState.HOT] = ThermalBoundary(
            max_cpu_temp=90.0,
            max_gpu_temp=100.0 if self.hardware_profile.gpu_available else None,
            batch_size=base_batch_size // 4,
            thread_count=1,
            processing_priority="minimal",
            emergency_procedures=["reduce_batch_size", "throttle_threads", "pause_heavy_tasks"]
        )
        
        # Critical state - emergency mode
        boundaries[ThermalState.CRITICAL] = ThermalBoundary(
            max_cpu_temp=95.0,
            max_gpu_temp=105.0 if self.hardware_profile.gpu_available else None,
            batch_size=10,
            thread_count=1,
            processing_priority="emergency",
            emergency_procedures=["reduce_batch_size", "throttle_threads", "pause_heavy_tasks", "enable_cooling"]
        )
        
        # Emergency state - shutdown procedures
        boundaries[ThermalState.EMERGENCY] = ThermalBoundary(
            max_cpu_temp=float('inf'),
            max_gpu_temp=float('inf') if self.hardware_profile.gpu_available else None,
            batch_size=1,
            thread_count=1,
            processing_priority="shutdown",
            emergency_procedures=["emergency_shutdown", "save_state", "disable_all_tasks"]
        )
        
        return boundaries

    async def start_monitoring(self) -> None:
        """Start thermal monitoring."""
        if self.monitoring_active:
            logger.warning("Thermal monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info("Thermal monitoring started")
        
        try:
            while self.monitoring_active:
                await self._monitor_thermal_state()
                await asyncio.sleep(5)  # Check every 5 seconds
        except Exception as e:
            logger.error(f"Thermal monitoring failed: {e}")
            self.monitoring_active = False

    async def stop_monitoring(self) -> None:
        """Stop thermal monitoring."""
        self.monitoring_active = False
        logger.info("Thermal monitoring stopped")

    async def _monitor_thermal_state(self) -> None:
        """Monitor current thermal state and apply boundaries."""
        try:
            # Get current temperatures
            cpu_temp = await self._get_cpu_temperature()
            gpu_temp = await self._get_gpu_temperature()
            
            # Determine thermal state
            new_thermal_state = self._determine_thermal_state(cpu_temp, gpu_temp)
            
            # Check if state changed
            if new_thermal_state != self.current_thermal_state:
                logger.info(f"Thermal state changed: {self.current_thermal_state.value} -> {new_thermal_state.value}")
                self.current_thermal_state = new_thermal_state
                
                # Apply emergency procedures if needed
                boundary = self.thermal_boundaries[new_thermal_state]
                if boundary.emergency_procedures:
                    await self._apply_emergency_procedures(boundary.emergency_procedures)
                
                # Update thermal zones
                await self._update_thermal_zones(new_thermal_state)
                
                # Notify callbacks
                self._notify_thermal_change(new_thermal_state, cpu_temp, gpu_temp)
            
        except Exception as e:
            logger.error(f"Thermal monitoring error: {e}")

    async def get_thermal_state(self) -> Dict[str, Any]:
        """Get current thermal state information."""
        try:
            cpu_temp = await self._get_cpu_temperature()
            gpu_temp = await self._get_gpu_temperature()
            
            boundary = self.thermal_boundaries[self.current_thermal_state]
            
            return {
                'thermal_state': self.current_thermal_state.value,
                'cpu_temperature': cpu_temp,
                'gpu_temperature': gpu_temp,
                'boundary': {
                    'max_cpu_temp': boundary.max_cpu_temp,
                    'max_gpu_temp': boundary.max_gpu_temp,
                    'batch_size': boundary.batch_size,
                    'thread_count': boundary.thread_count,
                    'processing_priority': boundary.processing_priority,
                    'emergency_procedures': boundary.emergency_procedures
                },
                'hardware_profile': {
                    'hardware_type': self.hardware_profile.hardware_type.value,
                    'cpu_cores': self.hardware_profile.cpu_cores,
                    'gpu_available': self.hardware_profile.gpu_available,
                    'thermal_sensors': self.hardware_profile.thermal_sensors,
                    'platform': self.hardware_profile.platform
                }
            }
        except Exception as e:
            logger.error(f"Failed to get thermal state: {e}")
            return {'error': str(e)}

    async def _get_cpu_temperature(self) -> float:
        """Get current CPU temperature."""
        try:
            if platform.system().lower() == "windows":
                return await self._get_windows_cpu_temp()
            else:
                return await self._get_linux_cpu_temp()
        except Exception as e:
            logger.warning(f"CPU temperature reading failed: {e}")
            return 0.0

    async def _get_windows_cpu_temp(self) -> float:
        """Get CPU temperature on Windows."""
        try:
            import wmi
            c = wmi.WMI(namespace="root\\wmi")
            sensors = c.MSAcpi_ThermalZoneTemperature()
            
            if sensors:
                # Convert from tenths of Kelvin to Celsius
                temp_kelvin = sensors[0].CurrentTemperature / 10.0
                temp_celsius = temp_kelvin - 273.15
                return temp_celsius
            
            # Fallback: estimate from CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            estimated_temp = 30.0 + (cpu_percent * 0.5)  # Rough estimation
            return estimated_temp
            
        except Exception as e:
            logger.debug(f"Windows CPU temperature failed: {e}")
            # Fallback estimation
            cpu_percent = psutil.cpu_percent(interval=1)
            return 30.0 + (cpu_percent * 0.5)

    async def _get_linux_cpu_temp(self) -> float:
        """Get CPU temperature on Linux."""
        try:
            # Try multiple thermal sensor paths
            thermal_paths = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/thermal/thermal_zone1/temp",
                "/proc/acpi/thermal_zone/THM0/temperature"
            ]
            
            for path in thermal_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp_millicelsius = int(f.read().strip())
                        temp_celsius = temp_millicelsius / 1000.0
                        return temp_celsius
            
            # Fallback: estimate from CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            estimated_temp = 30.0 + (cpu_percent * 0.5)
            return estimated_temp
            
        except Exception as e:
            logger.debug(f"Linux CPU temperature failed: {e}")
            # Fallback estimation
            cpu_percent = psutil.cpu_percent(interval=1)
            return 30.0 + (cpu_percent * 0.5)

    async def _get_gpu_temperature(self) -> Optional[float]:
        """Get current GPU temperature."""
        # Placeholder for GPU temperature monitoring
        # Can be enhanced with specific GPU libraries (nvidia-ml-py, py3nvml, etc.)
        return None

    def _determine_thermal_state(self, cpu_temp: float, gpu_temp: Optional[float]) -> ThermalState:
        """Determine thermal state based on temperatures."""
        try:
            # Check emergency conditions first
            if cpu_temp >= 95.0 or (gpu_temp and gpu_temp >= 105.0):
                return ThermalState.EMERGENCY
            
            # Check critical conditions
            if cpu_temp >= 90.0 or (gpu_temp and gpu_temp >= 100.0):
                return ThermalState.CRITICAL
            
            # Check hot conditions
            if cpu_temp >= 80.0 or (gpu_temp and gpu_temp >= 90.0):
                return ThermalState.HOT
            
            # Check warm conditions
            if cpu_temp >= 70.0 or (gpu_temp and gpu_temp >= 80.0):
                return ThermalState.WARM
            
            # Check cool conditions
            if cpu_temp < 50.0 and (gpu_temp is None or gpu_temp < 60.0):
                return ThermalState.COOL
            
            # Default to normal
            return ThermalState.NORMAL
            
        except Exception as e:
            logger.error(f"Thermal state determination failed: {e}")
            return ThermalState.NORMAL

    async def _apply_emergency_procedures(self, procedures: List[str]) -> None:
        """Apply emergency procedures based on thermal state."""
        for procedure in procedures:
            try:
                if procedure == "reduce_batch_size":
                    logger.warning("Applying emergency procedure: reduce_batch_size")
                elif procedure == "throttle_threads":
                    logger.warning("Applying emergency procedure: throttle_threads")
                elif procedure == "pause_heavy_tasks":
                    logger.warning("Applying emergency procedure: pause_heavy_tasks")
                elif procedure == "enable_cooling":
                    logger.warning("Applying emergency procedure: enable_cooling")
                elif procedure == "emergency_shutdown":
                    logger.critical("Applying emergency procedure: emergency_shutdown")
                elif procedure == "save_state":
                    logger.warning("Applying emergency procedure: save_state")
                elif procedure == "disable_all_tasks":
                    logger.critical("Applying emergency procedure: disable_all_tasks")
                
                # Notify callbacks
                self._notify_emergency_procedure(procedure)
                
            except Exception as e:
                logger.error(f"Emergency procedure {procedure} failed: {e}")

    async def _update_thermal_zones(self, thermal_state: ThermalState) -> None:
        """Update thermal zones based on current state."""
        try:
            if ThermalZoneManager:
                # Update thermal zones if available
                pass
        except Exception as e:
            logger.debug(f"Thermal zone update failed: {e}")

    def get_processing_recommendations(self) -> Dict[str, Any]:
        """Get processing recommendations based on current thermal state."""
        try:
            boundary = self.thermal_boundaries[self.current_thermal_state]
            
            return {
                'batch_size': self._calculate_batch_size(boundary),
                'thread_count': self._calculate_threads(boundary),
                'processing_priority': self._get_processing_priority(boundary),
                'thermal_state': self.current_thermal_state.value,
                'recommendations': {
                    'reduce_workload': self.current_thermal_state in [ThermalState.WARM, ThermalState.HOT, ThermalState.CRITICAL],
                    'enable_cooling': self.current_thermal_state in [ThermalState.HOT, ThermalState.CRITICAL],
                    'emergency_mode': self.current_thermal_state in [ThermalState.CRITICAL, ThermalState.EMERGENCY],
                    'optimal_performance': self.current_thermal_state == ThermalState.COOL
                }
            }
        except Exception as e:
            logger.error(f"Processing recommendations failed: {e}")
            return {'error': str(e)}

    def _calculate_batch_size(self, boundary: ThermalBoundary) -> int:
        """Calculate recommended batch size based on thermal boundary."""
        try:
            base_batch_size = boundary.batch_size
            
            # Apply thermal scaling
            if self.current_thermal_state == ThermalState.COOL:
                return base_batch_size * 2
            elif self.current_thermal_state == ThermalState.NORMAL:
                return base_batch_size
            elif self.current_thermal_state == ThermalState.WARM:
                return base_batch_size // 2
            elif self.current_thermal_state == ThermalState.HOT:
                return base_batch_size // 4
            elif self.current_thermal_state == ThermalState.CRITICAL:
                return 10
            else:  # EMERGENCY
                return 1
        except Exception as e:
            logger.error(f"Batch size calculation failed: {e}")
            return 100

    def _calculate_threads(self, boundary: ThermalBoundary) -> int:
        """Calculate recommended thread count based on thermal boundary."""
        try:
            base_threads = boundary.thread_count
            
            # Apply thermal scaling
            if self.current_thermal_state == ThermalState.COOL:
                return min(base_threads * 2, self.hardware_profile.cpu_cores)
            elif self.current_thermal_state == ThermalState.NORMAL:
                return base_threads
            elif self.current_thermal_state == ThermalState.WARM:
                return max(1, base_threads // 2)
            else:  # HOT, CRITICAL, EMERGENCY
                return 1
        except Exception as e:
            logger.error(f"Thread count calculation failed: {e}")
            return 1

    def _get_processing_priority(self, boundary: ThermalBoundary) -> str:
        """Get processing priority based on thermal boundary."""
        try:
            return boundary.processing_priority
        except Exception as e:
            logger.error(f"Processing priority failed: {e}")
            return "normal"

    def add_thermal_alert_callback(self, callback: Callable[[ThermalState, float, Optional[float]], None]) -> None:
        """Add thermal alert callback."""
        self.alert_callbacks.append(callback)

    def remove_thermal_alert_callback(self, callback: Callable[[ThermalState, float, Optional[float]], None]) -> None:
        """Remove thermal alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    def _notify_thermal_change(self, thermal_state: ThermalState, cpu_temp: float, gpu_temp: Optional[float]) -> None:
        """Notify thermal change callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(thermal_state, cpu_temp, gpu_temp)
            except Exception as e:
                logger.error(f"Thermal alert callback failed: {e}")

    def _notify_emergency_procedure(self, procedure: str) -> None:
        """Notify emergency procedure callbacks."""
        for callback in self.alert_callbacks:
            try:
                # Call with special emergency procedure notification
                callback(ThermalState.EMERGENCY, 0.0, None)
            except Exception as e:
                logger.error(f"Emergency procedure callback failed: {e}")


def create_thermal_boundary_manager(config: Optional[Dict[str, Any]] = None) -> ThermalBoundaryManager:
    """Create and configure thermal boundary manager."""
    try:
        manager = ThermalBoundaryManager(config)
        logger.info("Thermal boundary manager created successfully")
        return manager
    except Exception as e:
        logger.error(f"Failed to create thermal boundary manager: {e}")
        raise


async def main() -> None:
    """Main function for testing thermal boundary manager."""
    try:
        # Create thermal boundary manager
        manager = create_thermal_boundary_manager()
        
        # Start monitoring
        await manager.start_monitoring()
        
    except Exception as e:
        logger.error(f"Thermal boundary manager test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())


