"""
Ghost Data Recovery System
=========================
Handles corrupted logs, misaligned vectors, string corruption from panic states,
and drift differential recovery across ALIF/ALEPH shadow cores.
"""

import json
import re
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class CorruptionMetrics:
    """Metrics for tracking data corruption patterns"""
    corrupted_files: int = 0
    recovered_files: int = 0
    quarantined_files: int = 0
    string_mismatches: int = 0
    vector_realignments: int = 0
    shadow_desync_events: int = 0
    panic_recoveries: int = 0
    last_corruption_time: Optional[datetime] = None

@dataclass
class VectorIntegrityState:
    """State tracking for vector plot integrity"""
    valid_vectors: int = 0
    corrupted_vectors: int = 0
    recovered_vectors: int = 0
    last_valid_vector: Optional[List[float]] = None
    integrity_score: float = 1.0
    drift_compensation: float = 0.0

class GhostDataDecontaminator:
    """Decontaminates and recovers corrupted log data"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_dir = Path(log_directory)
        self.recovery_dir = self.log_dir / "recovered"
        self.quarantine_dir = self.log_dir / "quarantine"
        
        # Create directories
        for dir_path in [self.recovery_dir, self.quarantine_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.corruption_metrics = CorruptionMetrics()
        self.recovery_patterns = self._load_recovery_patterns()
    
    def _load_recovery_patterns(self) -> Dict[str, str]:
        """Load regex patterns for common corruption types"""
        return {
            'incomplete_json': r'^\{.*[^}]$',
            'malformed_string': r'[^\x20-\x7E\n\r\t]',
            'broken_timestamp': r'"timestamp":\s*"[^"]*[^0-9TZ:-]',
            'invalid_entropy': r'"entropy":\s*([^0-9.-]|[0-9.]+[^0-9.,\s}])',
            'hash_corruption': r'"hash":\s*"0x[^0-9a-fA-F]',
            'vector_corruption': r'\[[^\]]*[^0-9.,\s\[\]]-'
        }
    
    def scan_and_recover_logs(self, directory: Optional[str] = None) -> Dict[str, int]:
        """Scan directory for corrupted logs and attempt recovery"""
        scan_dir = Path(directory) if directory else self.log_dir
        results = {"scanned": 0, "corrupted": 0, "recovered": 0, "quarantined": 0}
        
        for log_file in scan_dir.glob("*.json"):
            results["scanned"] += 1
            
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Check for corruption
                corruption_type = self._detect_corruption(content)
                
                if corruption_type:
                    results["corrupted"] += 1
                    self.corruption_metrics.corrupted_files += 1
                    
                    # Attempt recovery
                    recovered_data = self._attempt_recovery(content, corruption_type)
                    
                    if recovered_data:
                        self._save_recovered_file(log_file.name, recovered_data)
                        results["recovered"] += 1
                        self.corruption_metrics.recovered_files += 1
                    else:
                        self._quarantine_file(log_file, corruption_type)
                        results["quarantined"] += 1
                        self.corruption_metrics.quarantined_files += 1
                
            except Exception as e:
                logger.error(f"Error processing {log_file}: {e}")
                self._quarantine_file(log_file, f"processing_error_{e}")
                results["quarantined"] += 1
        
        return results
    
    def _detect_corruption(self, content: str) -> Optional[str]:
        """Detect type of corruption in log content"""
        for corruption_type, pattern in self.recovery_patterns.items():
            if re.search(pattern, content):
                return corruption_type
        
        # Check JSON validity
        try:
            json.loads(content)
        except json.JSONDecodeError:
            return "invalid_json"
        
        return None
    
    def _attempt_recovery(self, content: str, corruption_type: str) -> Optional[Dict]:
        """Attempt to recover corrupted data based on corruption type"""
        try:
            if corruption_type == "incomplete_json":
                return self._recover_incomplete_json(content)
            elif corruption_type == "malformed_string":
                return self._recover_malformed_strings(content)
            elif corruption_type == "broken_timestamp":
                return self._recover_timestamp(content)
            elif corruption_type == "invalid_entropy":
                return self._recover_entropy_values(content)
            elif corruption_type == "hash_corruption":
                return self._recover_hash_values(content)
            elif corruption_type == "vector_corruption":
                return self._recover_vector_data(content)
            elif corruption_type == "invalid_json":
                return self._recover_json_structure(content)
            
        except Exception as e:
            logger.error(f"Recovery failed for {corruption_type}: {e}")
        
        return None
    
    def _recover_incomplete_json(self, content: str) -> Optional[Dict]:
        """Recover incomplete JSON by adding missing closing braces"""
        # Count opening and closing braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        
        if open_braces > close_braces:
            # Add missing closing braces
            missing = open_braces - close_braces
            recovered_content = content + '}' * missing
            
            try:
                return json.loads(recovered_content)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _recover_malformed_strings(self, content: str) -> Optional[Dict]:
        """Recover strings with invalid characters"""
        # Remove or replace invalid characters
        cleaned_content = re.sub(r'[^\x20-\x7E\n\r\t]', '', content)
        
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _recover_timestamp(self, content: str) -> Optional[Dict]:
        """Recover corrupted timestamps"""
        # Replace with current timestamp if corrupted
        current_time = datetime.now().isoformat()
        
        # Find and replace malformed timestamp
        recovered = re.sub(
            r'"timestamp":\s*"[^"]*"',
            f'"timestamp": "{current_time}"',
            content
        )
        
        try:
            return json.loads(recovered)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _recover_entropy_values(self, content: str) -> Optional[Dict]:
        """Recover corrupted entropy values"""
        # Replace invalid entropy with default value
        recovered = re.sub(
            r'"entropy":\s*[^0-9.,\s}]+',
            '"entropy": 0.5',
            content
        )
        
        try:
            return json.loads(recovered)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _recover_hash_values(self, content: str) -> Optional[Dict]:
        """Recover corrupted hash values"""
        # Generate new hash if corrupted
        new_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        recovered = re.sub(
            r'"hash":\s*"[^"]*"',
            f'"hash": "0x{new_hash}"',
            content
        )
        
        try:
            return json.loads(recovered)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _recover_vector_data(self, content: str) -> Optional[Dict]:
        """Recover corrupted vector data"""
        # Replace corrupted vectors with zeros or interpolated values
        recovered = re.sub(
            r'\[[^\]]*[^0-9.,\s\[\]]-[^\]]*\]',
            '[0.0, 0.0, 0.0]',
            content
        )
        
        try:
            return json.loads(recovered)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _recover_json_structure(self, content: str) -> Optional[Dict]:
        """Attempt to recover basic JSON structure"""
        # Try to extract key-value pairs using regex
        extracted_data = {}
        
        # Extract common fields
        patterns = {
            'tick_id': r'"tick_id":\s*(\d+)',
            'entropy': r'"entropy":\s*([\d.]+)',
            'timestamp': r'"timestamp":\s*"([^"]*)"',
            'echo_strength': r'"echo_strength":\s*([\d.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                try:
                    if key in ['tick_id']:
                        extracted_data[key] = int(match.group(1))
                    elif key in ['entropy', 'echo_strength']:
                        extracted_data[key] = float(match.group(1))
                    else:
                        extracted_data[key] = match.group(1)
                except ValueError:
                    pass
        
        return extracted_data if extracted_data else None
    
    def _save_recovered_file(self, filename: str, data: Dict):
        """Save recovered data to recovery directory"""
        recovery_path = self.recovery_dir / f"recovered_{filename}"
        
        with open(recovery_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Recovered file saved: {recovery_path}")
    
    def _quarantine_file(self, file_path: Path, reason: str):
        """Move corrupted file to quarantine"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_path = self.quarantine_dir / f"{timestamp}_{reason}_{file_path.name}"
        
        try:
            file_path.rename(quarantine_path)
            logger.warning(f"File quarantined: {quarantine_path}")
        except Exception as e:
            logger.error(f"Failed to quarantine {file_path}: {e}")

class VectorDriftCorrector:
    """Corrects vector drift and misalignment in plot data"""
    
    def __init__(self):
        self.integrity_state = VectorIntegrityState()
        self.correction_history = []
        self.baseline_vectors = {}
        
    def validate_vector(self, vector: List[float], vector_id: str) -> bool:
        """Validate if vector is corrupted"""
        if not vector or len(vector) == 0:
            return False
        
        # Check for NaN or infinite values
        if any(not np.isfinite(v) for v in vector):
            return False
        
        # Check for unrealistic values (beyond expected range)
        if any(abs(v) > 1000.0 for v in vector):
            return False
        
        return True
    
    def correct_vector_drift(self, corrupted_vector: List[float], vector_id: str) -> List[float]:
        """Correct drift in corrupted vector using interpolation"""
        if vector_id in self.baseline_vectors:
            baseline = self.baseline_vectors[vector_id]
            
            # Use weighted interpolation between baseline and last valid
            if self.integrity_state.last_valid_vector:
                weight = 0.7  # Favor recent valid data
                corrected = [
                    weight * lv + (1 - weight) * bv
                    for lv, bv in zip(self.integrity_state.last_valid_vector, baseline)
                ]
            else:
                corrected = baseline.copy()
            
            self.integrity_state.recovered_vectors += 1
            self.correction_history.append({
                "vector_id": vector_id,
                "timestamp": datetime.now(),
                "correction_type": "drift_interpolation"
            })
            
            return corrected
        
        # Generate synthetic vector if no baseline
        synthetic_vector = self._generate_synthetic_vector(len(corrupted_vector))
        
        self.correction_history.append({
            "vector_id": vector_id,
            "timestamp": datetime.now(),
            "correction_type": "synthetic_generation"
        })
        
        return synthetic_vector
    
    def _generate_synthetic_vector(self, length: int) -> List[float]:
        """Generate synthetic vector data when no baseline available"""
        # Generate smooth synthetic data
        t = np.linspace(0, 2 * np.pi, length)
        synthetic = 0.5 + 0.3 * np.sin(t) + 0.1 * np.cos(2 * t)
        return synthetic.tolist()
    
    def update_baseline(self, vector: List[float], vector_id: str):
        """Update baseline vector for future corrections"""
        if self.validate_vector(vector, vector_id):
            self.baseline_vectors[vector_id] = vector.copy()
            self.integrity_state.last_valid_vector = vector.copy()
            self.integrity_state.valid_vectors += 1
            
            # Update integrity score
            total_vectors = (self.integrity_state.valid_vectors + 
                           self.integrity_state.corrupted_vectors)
            if total_vectors > 0:
                self.integrity_state.integrity_score = (
                    self.integrity_state.valid_vectors / total_vectors
                )

class ShadowDesyncRecovery:
    """Handles shadow layer desynchronization and chart drift"""
    
    def __init__(self):
        self.shadow_state = {
            "alif_shadow": {"last_sync": datetime.now(), "drift": 0.0},
            "aleph_shadow": {"last_sync": datetime.now(), "drift": 0.0}
        }
        self.desync_threshold = timedelta(seconds=5.0)
        self.recovery_count = 0
        
    def detect_shadow_desync(self, alif_time: datetime, aleph_time: datetime) -> bool:
        """Detect if shadow layers are desynced"""
        time_diff = abs((alif_time - aleph_time).total_seconds())
        return time_diff > self.desync_threshold.total_seconds()
    
    def recover_shadow_sync(self, alif_data: Dict, aleph_data: Dict) -> Tuple[Dict, Dict]:
        """Recover synchronized shadow data"""
        if self.detect_shadow_desync(
            alif_data.get("timestamp", datetime.now()),
            aleph_data.get("timestamp", datetime.now())
        ):
            # Synchronize to the more recent timestamp
            latest_time = max(
                alif_data.get("timestamp", datetime.now()),
                aleph_data.get("timestamp", datetime.now())
            )
            
            # Update both with synchronized timestamp
            alif_data["timestamp"] = latest_time
            aleph_data["timestamp"] = latest_time
            
            # Apply drift compensation
            self._apply_drift_compensation(alif_data, aleph_data)
            
            self.recovery_count += 1
            logger.info(f"Shadow desync recovered (count: {self.recovery_count})")
        
        return alif_data, aleph_data
    
    def _apply_drift_compensation(self, alif_data: Dict, aleph_data: Dict):
        """Apply drift compensation to synchronized data"""
        # Calculate drift compensation based on historical patterns
        alif_drift = self.shadow_state["alif_shadow"]["drift"]
        aleph_drift = self.shadow_state["aleph_shadow"]["drift"]
        
        # Apply compensation to entropy and echo values
        if "entropy" in alif_data:
            alif_data["entropy"] = max(0.0, min(1.0, 
                alif_data["entropy"] - alif_drift * 0.1))
        
        if "echo_strength" in aleph_data:
            aleph_data["echo_strength"] = max(0.0, min(1.0,
                aleph_data["echo_strength"] - aleph_drift * 0.1))

class GhostDataRecoveryManager:
    """Central manager for all ghost data recovery operations"""
    
    def __init__(self, log_directory: str = "logs"):
        self.decontaminator = GhostDataDecontaminator(log_directory)
        self.vector_corrector = VectorDriftCorrector()
        self.shadow_recovery = ShadowDesyncRecovery()
        
        # Recovery statistics
        self.recovery_stats = {
            "session_start": datetime.now(),
            "total_recoveries": 0,
            "recovery_success_rate": 0.0,
            "last_recovery": None
        }
        
    def full_system_recovery_scan(self) -> Dict[str, Any]:
        """Perform comprehensive recovery scan"""
        logger.info("Starting full system recovery scan...")
        
        # Step 1: Scan and recover corrupted logs
        log_results = self.decontaminator.scan_and_recover_logs()
        
        # Step 2: Check vector integrity
        vector_status = {
            "valid_vectors": self.vector_corrector.integrity_state.valid_vectors,
            "corrupted_vectors": self.vector_corrector.integrity_state.corrupted_vectors,
            "recovered_vectors": self.vector_corrector.integrity_state.recovered_vectors,
            "integrity_score": self.vector_corrector.integrity_state.integrity_score
        }
        
        # Step 3: Shadow desync recovery count
        shadow_status = {
            "recovery_count": self.shadow_recovery.recovery_count,
            "desync_threshold": self.shadow_recovery.desync_threshold.total_seconds()
        }
        
        # Update recovery statistics
        total_issues = (log_results["corrupted"] + 
                       vector_status["corrupted_vectors"])
        total_recovered = (log_results["recovered"] + 
                          vector_status["recovered_vectors"])
        
        if total_issues > 0:
            self.recovery_stats["recovery_success_rate"] = total_recovered / total_issues
        
        self.recovery_stats["total_recoveries"] = total_recovered
        self.recovery_stats["last_recovery"] = datetime.now()
        
        return {
            "log_recovery": log_results,
            "vector_status": vector_status,
            "shadow_status": shadow_status,
            "recovery_stats": self.recovery_stats,
            "corruption_metrics": self.decontaminator.corruption_metrics.__dict__
        }
    
    def emergency_recovery_protocol(self) -> bool:
        """Emergency recovery protocol for critical system failures"""
        logger.warning("Initiating emergency recovery protocol...")
        
        try:
            # 1. Stop all active logging
            # 2. Backup current state
            # 3. Clear corrupted data
            # 4. Restore from last known good state
            # 5. Re-initialize with safe defaults
            
            recovery_result = self.full_system_recovery_scan()
            
            # Check if recovery was successful
            success_rate = recovery_result["recovery_stats"]["recovery_success_rate"]
            
            if success_rate > 0.7:  # 70% recovery rate threshold
                logger.info(f"Emergency recovery successful (rate: {success_rate:.1%})")
                return True
            else:
                logger.error(f"Emergency recovery failed (rate: {success_rate:.1%})")
                return False
                
        except Exception as e:
            logger.error(f"Emergency recovery protocol failed: {e}")
            return False

# Factory function
def create_ghost_recovery_manager(log_directory: str = "logs") -> GhostDataRecoveryManager:
    """Create configured ghost data recovery manager"""
    return GhostDataRecoveryManager(log_directory)

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create recovery manager
    recovery_manager = create_ghost_recovery_manager()
    
    # Perform recovery scan
    results = recovery_manager.full_system_recovery_scan()
    
    print("🔧 Ghost Data Recovery Results:")
    print(f"📄 Log Recovery: {results['log_recovery']}")
    print(f"📊 Vector Status: {results['vector_status']}")
    print(f"👻 Shadow Status: {results['shadow_status']}")
    print(f"📈 Recovery Stats: {results['recovery_stats']}") 