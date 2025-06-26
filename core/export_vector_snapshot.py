# -*- coding: utf-8 -*-\n# Import safe print for Windows compatibility
try:
from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    pass
    pass
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
    pass
    pass
def safe_print(message):


    pass
    pass
    print(message)
def info(message):


    pass
    pass
    print(f"[INFO] {message}")
def warn(message):


    pass
    pass
    print(f"[WARN] {message}")
def error(message):


    pass
    pass
    print(f"[ERROR] {message}")
def success(message):


    pass
    pass
    print(f"[SUCCESS] {message}")
def debug(message):


    pass
    pass
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""
Vector State Export Engine - Schwabot UROS v1.0
==============================================

Replaces stub export_vector_snapshot() with proper vector state export
that dumps DLT waveform, profit vector, and basket mapping data for analysis.

Features:
- Export DLT waveform data and entropy calculations
- Export tensor scoring and basket mapping data
- Export profit vector and allocation history
- Export bit phase resolution data
- Generate comprehensive state snapshots
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
# from core.unified_math_system import unified_math  # F811: duplicate import
import hashlib
import os
import glob
import pickle
import gzip

logger = logging.getLogger(__name__)

class ExportFormat(Enum):


    """Export format types."""
JSON = "json"
PICKLE = "pickle"
COMPRESSED = "compressed"
CSV = "csv"

class SnapshotType(Enum):


    """Snapshot type categories."""
DLT_WAVEFORM = "dlt_waveform"
TENSOR_SCORING = "tensor_scoring"
PROFIT_VECTOR = "profit_vector"
BASKET_MAPPING = "basket_mapping"
BIT_PHASE = "bit_phase"
COMPLETE_STATE = "complete_state"

@dataclass
class VectorSnapshot:


    """Vector state snapshot."""
snapshot_id: str
timestamp: datetime
snapshot_type: SnapshotType
data: Dict[str, Any]
metadata: Dict[str, Any] = field(default_factory=dict)
    export_format: ExportFormat = ExportFormat.JSON

@dataclass
class DLTWaveformData:


    """DLT waveform export data."""
waveform_name: str
timestamp: datetime
sequence_data: List[float]
entropy_level: float
phase_analysis: Dict[str, Any]
frequency_components: List[float]
power_spectrum: List[float]
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TensorScoringData:


    """Tensor scoring export data."""
timestamp: datetime
tensor_scores: List[float]
bit_phases: List[int]
basket_mappings: List[str]
strategy_decisions: List[str]
confidence_scores: List[float]
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProfitVectorData:


    """Profit vector export data."""
timestamp: datetime
profit_amounts: List[float]
allocation_distributions: List[Dict[str, float]]
rebalance_events: List[Dict[str, Any]]
performance_metrics: Dict[str, float]
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BasketMappingData:


    """Basket mapping export data."""
timestamp: datetime
hash_values: List[str]
basket_ids: List[str]
bit_phase_resolutions: List[int]
tensor_routes: List[str]
allocation_weights: List[float]
metadata: Dict[str, Any] = field(default_factory=dict)

class VectorStateExporter:


    """
Vector state export engine for comprehensive data export.

Mathematical Foundation:
- Waveform Analysis: FFT decomposition and entropy calculation
- Tensor Scoring: T = (current - entry) / entry * (phase + 1)
    - Profit Vectorization: P = Σ(allocations * weights * performance)
    - Basket Mapping: B = hash_to_basket(hash, bit_phase)
    - State Compression: S = compress(data, format, compression_level)
    """

def __init__(self, config_path: str = "./config/vector_export_config.json"):


    pass
    pass
        self.config_path = config_path

        # Export configuration
self.export_path = "./exports/vector_snapshots/"
self.compression_level = 6
self.max_file_size = 100 * 1024 * 1024  # 100MB

        # Data storage
self.snapshots: Dict[str, VectorSnapshot] = {}
self.export_history: List[Dict[str, Any]] = []

        # Integration with other components
self.dlt_engine = None
self.tensor_matcher = None
self.bit_phase_engine = None
self.matrix_mapper = None
self.profit_allocator = None

        # Load configuration
self._load_configuration()
        self._ensure_export_directories()
        logger.info("Vector State Exporter initialized")

def _load_configuration(self) -> None:


    pass
    pass
        """Load vector export configuration."""
        try:
            # Default configuration
config = {
"export_settings": {
"default_format": "json",
"compression_level": 6,
"max_file_size_mb": 100,
"auto_cleanup_days": 30
},
"data_retention": {
"snapshots_to_keep": 1000,
"max_age_days": 90,
"archive_old_data": True
},
"export_paths": {
"base_path": "./exports/vector_snapshots/",
"dlt_waveforms": "./exports/dlt_waveforms/",
"tensor_scores": "./exports/tensor_scores/",
"profit_vectors": "./exports/profit_vectors/",
"basket_mappings": "./exports/basket_mappings/"
}
}

logger.info("Vector export configuration loaded")

        except Exception as e:
logger.error(f"Error loading configuration: {e}")

def _ensure_export_directories(self) -> None:


    pass
    pass
        """Ensure export directories exist."""
        try:
directories = [
self.export_path,
"./exports/dlt_waveforms/",
"./exports/tensor_scores/",
"./exports/profit_vectors/",
"./exports/basket_mappings/",
"./exports/archives/"
]

            for directory in directories:
os.makedirs(directory, exist_ok=True)

logger.info("Export directories ensured")

        except Exception as e:
logger.error(f"Error ensuring export directories: {e}")

def export_vector_snapshot(self, snapshot_type: SnapshotType,


                             data: Dict[str, Any],
export_format: ExportFormat = ExportFormat.JSON,
compress: bool = False) -> str:
"""
Export vector state snapshot.

Parameters:
-----------
snapshot_type : SnapshotType
Type of snapshot to export
data : Dict[str, Any]
Data to export
export_format : ExportFormat
Format for export
compress : bool
Whether to compress the export

Returns:
--------
str
Path to exported file
"""
        try:
            # Generate snapshot ID
snapshot_id = f"{snapshot_type.value}_{int(time.time())}"

            # Create snapshot
snapshot = VectorSnapshot(
                snapshot_id=snapshot_id,
timestamp=datetime.now(),
                snapshot_type=snapshot_type,
data=data,
export_format=export_format,
metadata={
'compressed': compress,
'data_size': len(str(data)),
                    'export_format': export_format.value
}


            # Store snapshot
self.snapshots[snapshot_id] = snapshot

            # Export based on type
            if snapshot_type == SnapshotType.DLT_WAVEFORM:
export_path = self._export_dlt_waveform(snapshot, compress)
            elif snapshot_type == SnapshotType.TENSOR_SCORING:
export_path = self._export_tensor_scoring(snapshot, compress)
            elif snapshot_type == SnapshotType.PROFIT_VECTOR:
export_path = self._export_profit_vector(snapshot, compress)
            elif snapshot_type == SnapshotType.BASKET_MAPPING:
export_path = self._export_basket_mapping(snapshot, compress)
            elif snapshot_type == SnapshotType.BIT_PHASE:
export_path = self._export_bit_phase(snapshot, compress)
            elif snapshot_type == SnapshotType.COMPLETE_STATE:
export_path = self._export_complete_state(snapshot, compress)
            else:
export_path = self._export_generic(snapshot, compress)

            # Record export
self.export_history.append({
                'snapshot_id': snapshot_id,
'timestamp': datetime.now().isoformat(),
                'type': snapshot_type.value,
'format': export_format.value,
'compressed': compress,
'file_path': export_path,
'file_size': os.path.getsize(export_path) if os.path.exists(export_path) else 0
            })

logger.info(f"Vector snapshot exported: {export_path}")
            return export_path

        except Exception as e:
logger.error(f"Error exporting vector snapshot: {e}")
            return ""

def _export_dlt_waveform(self, snapshot: VectorSnapshot, compress: bool) -> str:


    pass
    pass
        """Export DLT waveform data."""
        try:
            # Create DLT waveform data structure
waveform_data = DLTWaveformData(
                waveform_name=snapshot.data.get('waveform_name', 'unknown'),
                timestamp=snapshot.timestamp,
sequence_data=snapshot.data.get('sequence_data', []),
                entropy_level=snapshot.data.get('entropy_level', 0.0),
                phase_analysis=snapshot.data.get('phase_analysis', {}),
                frequency_components=snapshot.data.get('frequency_components', []),
                power_spectrum=snapshot.data.get('power_spectrum', []),
                metadata=snapshot.data.get('metadata', {})


            # Prepare export data
export_data = {
'waveform_name': waveform_data.waveform_name,
'timestamp': waveform_data.timestamp.isoformat(),
                'sequence_data': waveform_data.sequence_data,
'entropy_level': waveform_data.entropy_level,
'phase_analysis': waveform_data.phase_analysis,
'frequency_components': waveform_data.frequency_components,
'power_spectrum': waveform_data.power_spectrum,
'metadata': waveform_data.metadata,
'snapshot_metadata': snapshot.metadata
}

            # Export to file
filename = f"dlt_waveform_{snapshot.snapshot_id}"
            return self._write_export_file(export_data, filename, snapshot.export_format, compress)

        except Exception as e:
logger.error(f"Error exporting DLT waveform: {e}")
            return ""

def _export_tensor_scoring(self, snapshot: VectorSnapshot, compress: bool) -> str:


    pass
    pass
        """Export tensor scoring data."""
        try:
            # Create tensor scoring data structure
tensor_data = TensorScoringData(
                timestamp=snapshot.timestamp,
tensor_scores=snapshot.data.get('tensor_scores', []),
                bit_phases=snapshot.data.get('bit_phases', []),
                basket_mappings=snapshot.data.get('basket_mappings', []),
                strategy_decisions=snapshot.data.get('strategy_decisions', []),
                confidence_scores=snapshot.data.get('confidence_scores', []),
                metadata=snapshot.data.get('metadata', {})


            # Prepare export data
export_data = {
'timestamp': tensor_data.timestamp.isoformat(),
                'tensor_scores': tensor_data.tensor_scores,
'bit_phases': tensor_data.bit_phases,
'basket_mappings': tensor_data.basket_mappings,
'strategy_decisions': tensor_data.strategy_decisions,
'confidence_scores': tensor_data.confidence_scores,
'metadata': tensor_data.metadata,
'snapshot_metadata': snapshot.metadata
}

            # Export to file
filename = f"tensor_scoring_{snapshot.snapshot_id}"
            return self._write_export_file(export_data, filename, snapshot.export_format, compress)

        except Exception as e:
logger.error(f"Error exporting tensor scoring: {e}")
            return ""

def _export_profit_vector(self, snapshot: VectorSnapshot, compress: bool) -> str:


    pass
    pass
        """Export profit vector data."""
        try:
            # Create profit vector data structure
profit_data = ProfitVectorData(
                timestamp=snapshot.timestamp,
profit_amounts=snapshot.data.get('profit_amounts', []),
                allocation_distributions=snapshot.data.get('allocation_distributions', []),
                rebalance_events=snapshot.data.get('rebalance_events', []),
                performance_metrics=snapshot.data.get('performance_metrics', {}),
                metadata=snapshot.data.get('metadata', {})


            # Prepare export data
export_data = {
'timestamp': profit_data.timestamp.isoformat(),
                'profit_amounts': profit_data.profit_amounts,
'allocation_distributions': profit_data.allocation_distributions,
'rebalance_events': profit_data.rebalance_events,
'performance_metrics': profit_data.performance_metrics,
'metadata': profit_data.metadata,
'snapshot_metadata': snapshot.metadata
}

            # Export to file
filename = f"profit_vector_{snapshot.snapshot_id}"
            return self._write_export_file(export_data, filename, snapshot.export_format, compress)

        except Exception as e:
logger.error(f"Error exporting profit vector: {e}")
            return ""

def _export_basket_mapping(self, snapshot: VectorSnapshot, compress: bool) -> str:


    pass
    pass
        """Export basket mapping data."""
        try:
            # Create basket mapping data structure
basket_data = BasketMappingData(
                timestamp=snapshot.timestamp,
hash_values=snapshot.data.get('hash_values', []),
                basket_ids=snapshot.data.get('basket_ids', []),
                bit_phase_resolutions=snapshot.data.get('bit_phase_resolutions', []),
                tensor_routes=snapshot.data.get('tensor_routes', []),
                allocation_weights=snapshot.data.get('allocation_weights', []),
                metadata=snapshot.data.get('metadata', {})


            # Prepare export data
export_data = {
'timestamp': basket_data.timestamp.isoformat(),
                'hash_values': basket_data.hash_values,
'basket_ids': basket_data.basket_ids,
'bit_phase_resolutions': basket_data.bit_phase_resolutions,
'tensor_routes': basket_data.tensor_routes,
'allocation_weights': basket_data.allocation_weights,
'metadata': basket_data.metadata,
'snapshot_metadata': snapshot.metadata
}

            # Export to file
filename = f"basket_mapping_{snapshot.snapshot_id}"
            return self._write_export_file(export_data, filename, snapshot.export_format, compress)

        except Exception as e:
logger.error(f"Error exporting basket mapping: {e}")
            return ""

def _export_bit_phase(self, snapshot: VectorSnapshot, compress: bool) -> str:


    pass
    pass
        """Export bit phase resolution data."""
        try:
            # Prepare export data
export_data = {
'timestamp': snapshot.timestamp.isoformat(),
                'bit_phase_data': snapshot.data.get('bit_phase_data', {}),
                'resolution_history': snapshot.data.get('resolution_history', []),
                'phase_statistics': snapshot.data.get('phase_statistics', {}),
                'hash_mappings': snapshot.data.get('hash_mappings', {}),
                'metadata': snapshot.data.get('metadata', {}),
                'snapshot_metadata': snapshot.metadata
}

            # Export to file
filename = f"bit_phase_{snapshot.snapshot_id}"
            return self._write_export_file(export_data, filename, snapshot.export_format, compress)

        except Exception as e:
logger.error(f"Error exporting bit phase: {e}")
            return ""

def _export_complete_state(self, snapshot: VectorSnapshot, compress: bool) -> str:


    pass
    pass
        """Export complete system state."""
        try:
            # Gather data from all components
complete_data = {
'timestamp': snapshot.timestamp.isoformat(),
                'dlt_waveform_data': self._gather_dlt_data(),
                'tensor_scoring_data': self._gather_tensor_data(),
                'profit_vector_data': self._gather_profit_data(),
                'basket_mapping_data': self._gather_basket_data(),
                'bit_phase_data': self._gather_bit_phase_data(),
                'system_metrics': self._gather_system_metrics(),
                'snapshot_metadata': snapshot.metadata
}

            # Export to file
filename = f"complete_state_{snapshot.snapshot_id}"
            return self._write_export_file(complete_data, filename, snapshot.export_format, compress)

        except Exception as e:
logger.error(f"Error exporting complete state: {e}")
            return ""

def _export_generic(self, snapshot: VectorSnapshot, compress: bool) -> str:


    pass
    pass
        """Export generic data."""
        try:
export_data = {
'timestamp': snapshot.timestamp.isoformat(),
                'data': snapshot.data,
'metadata': snapshot.metadata
}

filename = f"generic_{snapshot.snapshot_id}"
            return self._write_export_file(export_data, filename, snapshot.export_format, compress)

        except Exception as e:
logger.error(f"Error exporting generic data: {e}")
            return ""

def _write_export_file(self, data: Dict[str, Any], filename: str,]


                          export_format: ExportFormat, compress: bool) -> str:
"""Write export data to file."""
        try:
            # Determine file extension
            if export_format == ExportFormat.JSON:
extension = ".json"
                if compress:
extension = ".json.gz"
            elif export_format == ExportFormat.PICKLE:
extension = ".pkl"
                if compress:
extension = ".pkl.gz"
            elif export_format == ExportFormat.CSV:
extension = ".csv"
                if compress:
extension = ".csv.gz"
            else:
extension = ".dat"
                if compress:
extension = ".dat.gz"

            # Create file path
file_path = os.path.join(self.export_path, f"{filename}{extension}")

            # Write data based on format
            if export_format == ExportFormat.JSON:
                if compress:
                    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, default=str)
                else:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)

            elif export_format == ExportFormat.PICKLE:
                if compress:
                    with gzip.open(file_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            elif export_format == ExportFormat.CSV:
                # Convert to CSV format (simplified)
                csv_data = self._convert_to_csv(data)
                if compress:
                    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                        f.write(csv_data)
                else:
                    with open(file_path, 'w') as f:
                        f.write(csv_data)

            return file_path

        except Exception as e:
logger.error(f"Error writing export file: {e}")
            return ""

def _convert_to_csv(self, data: Dict[str, Any]) -> str:


    pass
    pass
        """Convert data to CSV format."""
        try:
            # Simplified CSV conversion
csv_lines = []

            # Add header
            if data:
headers = list(data.keys())
                csv_lines.append(','.join(headers))

                # Add data row
values = []
                for header in headers:
value = data[header]
                    if isinstance(value, (dict, list)):
                        value = str(value)
                    values.append(str(value))
                csv_lines.append(','.join(values))

            return '\n'.join(csv_lines)

        except Exception as e:
logger.error(f"Error converting to CSV: {e}")
            return ""

def _gather_dlt_data(self) -> Dict[str, Any]:


    pass
    pass
        """Gather DLT waveform data from engine."""
        try:
            if not self.dlt_engine:
                return {}

            # Gather waveform data
waveform_data = {
'active_waveforms': getattr(self.dlt_engine, 'active_waveforms', {}),
                'entropy_history': getattr(self.dlt_engine, 'entropy_history', []),
                'phase_analysis': getattr(self.dlt_engine, 'phase_analysis', {}),
                'frequency_data': getattr(self.dlt_engine, 'frequency_data', {})
            }

            return waveform_data

        except Exception as e:
logger.error(f"Error gathering DLT data: {e}")
            return {}

def _gather_tensor_data(self) -> Dict[str, Any]:


    pass
    pass
        """Gather tensor scoring data from matcher."""
        try:
            if not self.tensor_matcher:
                return {}

            # Gather tensor data
tensor_data = {
'match_history': getattr(self.tensor_matcher, 'match_history', []),
                'phase_weight_history': getattr(self.tensor_matcher, 'phase_weight_history', []),
                'strategy_mappings': getattr(self.tensor_matcher, 'strategy_mappings', {})
            }

            return tensor_data

        except Exception as e:
logger.error(f"Error gathering tensor data: {e}")
            return {}

def _gather_profit_data(self) -> Dict[str, Any]:


    pass
    pass
        """Gather profit vector data from allocator."""
        try:
            if not self.profit_allocator:
                return {}

            # Gather profit data
profit_data = {
'allocation_history': getattr(self.profit_allocator, 'allocation_history', []),
                'matrix_metrics': getattr(self.profit_allocator, 'matrix_metrics', {}),
                'profit_tracking': getattr(self.profit_allocator, 'profit_tracking', {})
            }

            return profit_data

        except Exception as e:
logger.error(f"Error gathering profit data: {e}")
            return {}

def _gather_basket_data(self) -> Dict[str, Any]:


    pass
    pass
        """Gather basket mapping data from mapper."""
        try:
            if not self.matrix_mapper:
                return {}

            # Gather basket data
basket_data = {
'hash_registry': getattr(self.matrix_mapper, 'hash_registry', {}),
                'basket_mappings': getattr(self.matrix_mapper, 'basket_mappings', {}),
                'tensor_routes': getattr(self.matrix_mapper, 'tensor_routes', {})
            }

            return basket_data

        except Exception as e:
logger.error(f"Error gathering basket data: {e}")
            return {}

def _gather_bit_phase_data(self) -> Dict[str, Any]:


    pass
    pass
        """Gather bit phase data from engine."""
        try:
            if not self.bit_phase_engine:
                return {}

            # Gather bit phase data
bit_phase_data = {
'phase_history': getattr(self.bit_phase_engine, 'phase_history', []),
                'supported_modes': getattr(self.bit_phase_engine, 'supported_modes', []),
                'resolution_stats': getattr(self.bit_phase_engine, 'resolution_stats', {})
            }

            return bit_phase_data

        except Exception as e:
logger.error(f"Error gathering bit phase data: {e}")
            return {}

def _gather_system_metrics(self) -> Dict[str, Any]:


    pass
    pass
        """Gather system performance metrics."""
        try:
metrics = {
'timestamp': datetime.now().isoformat(),
                'snapshot_count': len(self.snapshots),
                'export_count': len(self.export_history),
                'total_export_size': sum(exp.get('file_size', 0) for exp in self.export_history),
                'system_memory': self._get_memory_usage(),
                'disk_usage': self._get_disk_usage()
            }

            return metrics

        except Exception as e:
logger.error(f"Error gathering system metrics: {e}")
            return {}

def _get_memory_usage(self) -> Dict[str, Any]:


    pass
    pass
        """Get memory usage information."""
        try:
import psutil
memory = psutil.virtual_memory()
            return {
'total': memory.total,
'available': memory.available,
'used': memory.used,
'percent': memory.percent
}
        except ImportError:
    pass
    pass
            return {'error': 'psutil not available'}

def _get_disk_usage(self) -> Dict[str, Any]:


    pass
    pass
        """Get disk usage information."""
        try:
disk = psutil.disk_usage(self.export_path)
            return {
'total': disk.total,
'used': disk.used,
'free': disk.free,
'percent': disk.percent
}
        except ImportError:
    pass
    pass
            return {'error': 'psutil not available'}

def set_dlt_engine(self, dlt_engine) -> None:


    pass
    pass
        """Set DLT engine for integration."""
self.dlt_engine = dlt_engine
logger.info("DLT engine integrated with vector exporter")

def set_tensor_matcher(self, tensor_matcher) -> None:


    pass
    pass
        """Set tensor matcher for integration."""
self.tensor_matcher = tensor_matcher
logger.info("Tensor matcher integrated with vector exporter")

def set_bit_phase_engine(self, bit_engine) -> None:


    pass
    pass
        """Set bit phase engine for integration."""
self.bit_phase_engine = bit_engine
logger.info("Bit phase engine integrated with vector exporter")

def set_matrix_mapper(self, matrix_mapper) -> None:


    pass
    pass
        """Set matrix mapper for integration."""
self.matrix_mapper = matrix_mapper
logger.info("Matrix mapper integrated with vector exporter")

def set_profit_allocator(self, profit_allocator) -> None:


    pass
    pass
        """Set profit allocator for integration."""
self.profit_allocator = profit_allocator
logger.info("Profit allocator integrated with vector exporter")

def get_export_history(self, limit: int = 100) -> List[Dict[str, Any]]:


    pass
    pass
        """Get recent export history."""
        return self.export_history[-limit:] if self.export_history else []

def cleanup_old_exports(self, days: int = 30) -> int:


    pass
    pass
        """Clean up old export files."""
        try:
cutoff_time = datetime.now() - timedelta(days=days)
            deleted_count = 0

            for file_path in glob.glob(os.path.join(self.export_path, "*")):
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff_time:
os.remove(file_path)
                        deleted_count += 1

logger.info(f"Cleaned up {deleted_count} old export files")
            return deleted_count

        except Exception as e:
logger.error(f"Error cleaning up old exports: {e}")
            return 0

if __name__ == "__main__":
    pass
    pass
    # Test vector state exporter
exporter = VectorStateExporter()

    # Test DLT waveform export
dlt_data = {
'waveform_name': 'test_waveform',
'sequence_data': [1.0, 1.1, 0.9, 1.2, 0.8, 1.3],
'entropy_level': 4.5,
'phase_analysis': {'phase_1': 0.3, 'phase_2': 0.7},
'frequency_components': [0.1, 0.2, 0.3],
'power_spectrum': [0.01, 0.04, 0.09]
}

export_path = exporter.export_vector_snapshot(
        SnapshotType.DLT_WAVEFORM, dlt_data, ExportFormat.JSON, compress=False

safe_print(f"✅ DLT Waveform exported to: {export_path}")

    # Test tensor scoring export
tensor_data = {
'tensor_scores': [0.1, 0.2, 0.3, 0.4],
'bit_phases': [8, 16, 32, 64],
'basket_mappings': ['basket_1', 'basket_2', 'basket_3', 'basket_4'],
'strategy_decisions': ['buy', 'hold', 'sell', 'rebalance'],
'confidence_scores': [0.8, 0.6, 0.9, 0.7]
}

export_path = exporter.export_vector_snapshot(
        SnapshotType.TENSOR_SCORING, tensor_data, ExportFormat.JSON, compress=True

safe_print(f"✅ Tensor Scoring exported to: {export_path}")

    # Test complete state export
complete_data = {
'system_state': 'operational',
'component_count': 5,
'active_processes': 3
}

export_path = exporter.export_vector_snapshot(
        SnapshotType.COMPLETE_STATE, complete_data, ExportFormat.PICKLE, compress=False

safe_print(f"✅ Complete State exported to: {export_path}")

    # Get export history
history = exporter.get_export_history()
    safe_print(f"📊 Export History: {len(history)} exports")

    for export in history[-3:]:  # Last 3 exports
safe_print(f"  - {export['type']}: {export['file_path']} ({export['file_size']} bytes)")
