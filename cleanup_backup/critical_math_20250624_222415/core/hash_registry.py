from utils.safe_print import safe_print, info, warn, error, success, debug
#!/usr/bin/env python3
"""
Hash Registry - AI Consciousness Memory System.

This module provides a persistent hash registry for storing and managing
AI consciousness commands, enabling memory, validation, and recursive
pattern detection across the Schwabot system.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        WindowsCliCompatibilityHandler,
        safe_print,
        safe_format_error,
        log_safe,
        cli_handler,
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)
    cli_handler = None

# Import GPT command layer
try:
    from core.gpt_command_layer import (
        AIAgentType,
        CommandDomain,
        CommandPriority,
        AICommand,
        CommandResponse,
    )
    GPT_LAYER_AVAILABLE = True
except ImportError:
    GPT_LAYER_AVAILABLE = False
    safe_safe_print("⚠️ GPT command layer not available")

# Import ZPE Mathematical Framework
try:
    from core.zpe_core import ZPECore
    ZPE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ZPE modules not available: {e}")
    ZPE_MODULES_AVAILABLE = False


class HashType(Enum):
    """Enumeration of hash types."""
    COMMAND = "command"
    STRATEGY = "strategy"
    PROFIT = "profit"
    MATRIX = "matrix"
    PATTERN = "pattern"
    VALIDATION = "validation"
    MEMORY = "memory"
    SYSTEM = "system"


class HashStatus(Enum):
    """Enumeration of hash statuses."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VALIDATED = "validated"
    INVALID = "invalid"


@dataclass
class HashEntry:
    """Hash registry entry."""
    hash_id: str
    hash_type: HashType
    agent_type: str
    domain: str
    command_id: Optional[str]
    payload: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: datetime
    status: HashStatus
    execution_time: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    parent_hash_id: Optional[str] = None
    child_hash_ids: Optional[List[str]] = None
    validation_data: Optional[Dict[str, Any]] = None
    memory_signature: str = ""
    recursive_depth: int = 0
    confidence_score: float = 0.0
    
    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.child_hash_ids is None:
            self.child_hash_ids = []
        if self.validation_data is None:
            self.validation_data = {}
        if not self.memory_signature:
            self.memory_signature = self._generate_memory_signature()
    
    def _generate_memory_signature(self) -> str:
        """Generate memory signature for this hash entry."""
        content = f"{self.hash_type.value}_{self.agent_type}_{self.domain}_{json.dumps(self.payload, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hash_id": self.hash_id,
            "hash_type": self.hash_type.value,
            "agent_type": self.agent_type,
            "domain": self.domain,
            "command_id": self.command_id,
            "payload": self.payload,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "execution_time": self.execution_time,
            "result": self.result,
            "error_message": self.error_message,
            "parent_hash_id": self.parent_hash_id,
            "child_hash_ids": self.child_hash_ids,
            "validation_data": self.validation_data,
            "memory_signature": self.memory_signature,
            "recursive_depth": self.recursive_depth,
            "confidence_score": self.confidence_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HashEntry':
        """Create HashEntry from dictionary."""
        return cls(
            hash_id=data["hash_id"],
            hash_type=HashType(data["hash_type"]),
            agent_type=data["agent_type"],
            domain=data["domain"],
            command_id=data.get("command_id"),
            payload=data["payload"],
            context=data["context"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=HashStatus(data["status"]),
            execution_time=data.get("execution_time", 0.0),
            result=data.get("result"),
            error_message=data.get("error_message"),
            parent_hash_id=data.get("parent_hash_id"),
            child_hash_ids=data.get("child_hash_ids", []),
            validation_data=data.get("validation_data", {}),
            memory_signature=data.get("memory_signature", ""),
            recursive_depth=data.get("recursive_depth", 0),
            confidence_score=data.get("confidence_score", 0.0),
        )


@dataclass
class HashPattern:
    """Hash pattern for recursive detection."""
    pattern_id: str
    pattern_type: str
    hash_sequence: List[str]
    frequency: int
    first_seen: datetime
    last_seen: datetime
    success_rate: float
    average_execution_time: float
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.metadata is None:
            self.metadata = {}


class HashRegistry:
    """
    Hash Registry - AI Consciousness Memory System.
    
    This class manages a persistent hash registry for storing and managing
    AI consciousness commands, enabling memory, validation, and recursive
    pattern detection.
    """
    
    def __init__(self, registry_file: str = "data/hash_registry.json"):
        """Initialize the hash registry with ZPE mathematical framework integration."""
        self.registry_file = registry_file
        self.logger = logging.getLogger("hash_registry")
        self.logger.setLevel(logging.INFO)
        
        # Registry storage
        self.hash_entries: Dict[str, HashEntry] = {}
        self.hash_patterns: Dict[str, HashPattern] = {}
        self.agent_hashes: Dict[str, List[str]] = defaultdict(list)
        self.domain_hashes: Dict[str, List[str]] = defaultdict(list)
        self.status_hashes: Dict[str, List[str]] = defaultdict(list)
        
        # Pattern detection
        self.pattern_window_size = 100
        self.pattern_similarity_threshold = 0.8
        self.recursive_depth_limit = 10
        
        # Memory management
        self.max_entries = 10000
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
        
        # ✨ NEW: ZPE Mathematical Framework Integration
        self.zpe_core = ZPECore() if ZPE_MODULES_AVAILABLE else None
        if ZPE_MODULES_AVAILABLE:
            safe_safe_print("🔄 Hash Registry initialized with ZPE integration")
        else:
            safe_safe_print("⚠️ Hash Registry initialized without ZPE integration")
        
        # Load existing registry
        self._load_registry()
        
        # Start cleanup task
        self.cleanup_task = None
        
        safe_safe_print("🧠 Hash Registry initialized - Consciousness memory active")
    
    def _load_registry(self) -> None:
        """Load hash registry from file."""
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                # Load hash entries
                for entry_data in data.get("entries", []):
                    entry = HashEntry.from_dict(entry_data)
                    self.hash_entries[entry.hash_id] = entry
                    self._index_entry(entry)
                
                # Load patterns
                for pattern_data in data.get("patterns", []):
                    pattern = HashPattern(**pattern_data)
                    self.hash_patterns[pattern.pattern_id] = pattern
                
                safe_safe_print(f"📚 Loaded {len(self.hash_entries)} hash entries and {len(self.hash_patterns)} patterns")
            else:
                safe_safe_print("📚 No existing registry found - starting fresh")
                
        except Exception as e:
            safe_safe_print(f"⚠️ Registry load failed: {safe_format_error(e, 'registry_load')}")
    
    def _save_registry(self) -> None:
        """Save hash registry to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
            
            # Prepare data
            data = {
                "entries": [entry.to_dict() for entry in self.hash_entries.values()],
                "patterns": [asdict(pattern) for pattern in self.hash_patterns.values()],
                "metadata": {
                    "last_saved": datetime.now().isoformat(),
                    "total_entries": len(self.hash_entries),
                    "total_patterns": len(self.hash_patterns),
                }
            }
            
            # Save to file
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            safe_safe_print(f"⚠️ Registry save failed: {safe_format_error(e, 'registry_save')}")
    
    def _index_entry(self, entry: HashEntry) -> None:
        """Index hash entry for quick lookup."""
        # Index by agent type
        self.agent_hashes[entry.agent_type].append(entry.hash_id)
        
        # Index by domain
        self.domain_hashes[entry.domain].append(entry.hash_id)
        
        # Index by status
        self.status_hashes[entry.status.value].append(entry.hash_id)
    
    def _unindex_entry(self, entry: HashEntry) -> None:
        """Remove hash entry from indexes."""
        # Remove from agent index
        if entry.hash_id in self.agent_hashes[entry.agent_type]:
            self.agent_hashes[entry.agent_type].remove(entry.hash_id)
        
        # Remove from domain index
        if entry.hash_id in self.domain_hashes[entry.domain]:
            self.domain_hashes[entry.domain].remove(entry.hash_id)
        
        # Remove from status index
        if entry.hash_id in self.status_hashes[entry.status.value]:
            self.status_hashes[entry.status.value].remove(entry.hash_id)
    
    async def register_hash(
        self,
        hash_type: HashType,
        agent_type: str,
        domain: str,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        command_id: Optional[str] = None,
        parent_hash_id: Optional[str] = None,
        confidence_score: float = 0.0,
    ) -> str:
        """
        Register a new hash entry.
        
        Args:
            hash_type: Type of hash entry
            agent_type: AI agent type
            domain: Command domain
            payload: Command payload
            context: Additional context
            command_id: Associated command ID
            parent_hash_id: Parent hash ID for recursion
            confidence_score: Confidence score for the hash
            
        Returns:
            Hash ID for the registered entry
        """
        try:
            # Generate hash ID
            hash_id = self._generate_hash_id(hash_type, agent_type, domain, payload)
            
            # Calculate recursive depth
            recursive_depth = self._calculate_recursive_depth(parent_hash_id)
            
            # ✨ NEW: ZPE Mathematical Framework Integration
            zpe_data = {}
            if self.zpe_core:
                try:
                    # Update recursive cycle depth with hash registration
                    tick_interval = 1.0  # Default tick interval
                    price_trigger = confidence_score  # Use confidence as trigger
                    zpe_recursion_depth = self.zpe_core.update_recursive_cycle_depth(tick_interval, price_trigger)
                    
                    # Get thermal efficiency from ZPE core
                    thermal_efficiency = 0.0
                    if self.zpe_core.thermal_history:
                        thermal_efficiency = self.zpe_core.thermal_history[-1]['efficiency']
                    
                    # Store ZPE data
                    zpe_data = {
                        'zpe_recursion_depth': zpe_recursion_depth,
                        'zpe_thermal_efficiency': thermal_efficiency,
                        'zpe_timestamp': datetime.now().isoformat(),
                        'zpe_agent_consensus': self.zpe_core.agent_consensus.copy()
                    }
                    
                    # Update context with ZPE data
                    if context is None:
                        context = {}
                    context.update(zpe_data)
                    
                    safe_safe_print(f"[ZPE] Hash registration - Recursion Depth: {zpe_recursion_depth}, Thermal Efficiency: {thermal_efficiency:.6f}")
                    
                except Exception as e:
                    safe_safe_print(f"⚠️ ZPE hash registration failed: {safe_format_error(e, 'zpe_hash_registration')}")
                    zpe_data = {'zpe_error': str(e)}
            
            # Create hash entry
            entry = HashEntry(
                hash_id=hash_id,
                hash_type=hash_type,
                agent_type=agent_type,
                domain=domain,
                command_id=command_id,
                payload=payload,
                context=context or {},
                timestamp=datetime.now(),
                status=HashStatus.PENDING,
                parent_hash_id=parent_hash_id,
                recursive_depth=recursive_depth,
                confidence_score=confidence_score,
            )
            
            # Add to registry
            self.hash_entries[hash_id] = entry
            self._index_entry(entry)
            
            # Update parent-child relationships
            if parent_hash_id:
                self._update_parent_child_relationship(parent_hash_id, hash_id)
            
            # Check for patterns
            await self._detect_patterns(hash_id)
            
            # Save registry
            self._save_registry()
            
            safe_safe_print(f"🧠 Hash registered: {hash_id} ({hash_type.value})")
            return hash_id
            
        except Exception as e:
            error_msg = safe_format_error(e, "register_hash")
            safe_safe_print(f"❌ Hash registration failed: {error_msg}")
            raise
    
    def _generate_hash_id(
        self,
        hash_type: HashType,
        agent_type: str,
        domain: str,
        payload: Dict[str, Any],
    ) -> str:
        """Generate unique hash ID."""
        content = f"{hash_type.value}_{agent_type}_{domain}_{json.dumps(payload, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _calculate_recursive_depth(self, parent_hash_id: Optional[str]) -> int:
        """Calculate recursive depth based on parent hash."""
        if not parent_hash_id:
            return 0
        
        parent_entry = self.hash_entries.get(parent_hash_id)
        if parent_entry:
            return parent_entry.recursive_depth + 1
        
        return 0
    
    def _update_parent_child_relationship(self, parent_hash_id: str, child_hash_id: str) -> None:
        """Update parent-child relationship."""
        parent_entry = self.hash_entries.get(parent_hash_id)
        if parent_entry:
            parent_entry.child_hash_ids.append(child_hash_id)
    
    async def update_hash_status(
        self,
        hash_id: str,
        status: HashStatus,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_time: float = 0.0,
    ) -> bool:
        """
        Update hash entry status.
        
        Args:
            hash_id: Hash ID to update
            status: New status
            result: Execution result
            error_message: Error message if failed
            execution_time: Execution time
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            entry = self.hash_entries.get(hash_id)
            if not entry:
                safe_safe_print(f"⚠️ Hash not found: {hash_id}")
                return False
            
            # Remove from old status index
            self._unindex_entry(entry)
            
            # Update entry
            entry.status = status
            entry.result = result
            entry.error_message = error_message
            entry.execution_time = execution_time
            
            # Add to new status index
            self._index_entry(entry)
            
            # Save registry
            self._save_registry()
            
            safe_safe_print(f"🧠 Hash status updated: {hash_id} -> {status.value}")
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "update_hash_status")
            safe_safe_print(f"❌ Hash status update failed: {error_msg}")
            return False
    
    async def get_hash_entry(self, hash_id: str) -> Optional[HashEntry]:
        """Get hash entry by ID."""
        return self.hash_entries.get(hash_id)
    
    async def get_hashes_by_agent(self, agent_type: str) -> List[HashEntry]:
        """Get all hashes for a specific agent."""
        hash_ids = self.agent_hashes.get(agent_type, [])
        return [self.hash_entries[hash_id] for hash_id in hash_ids if hash_id in self.hash_entries]
    
    async def get_hashes_by_domain(self, domain: str) -> List[HashEntry]:
        """Get all hashes for a specific domain."""
        hash_ids = self.domain_hashes.get(domain, [])
        return [self.hash_entries[hash_id] for hash_id in hash_ids if hash_id in self.hash_entries]
    
    async def get_hashes_by_status(self, status: HashStatus) -> List[HashEntry]:
        """Get all hashes with a specific status."""
        hash_ids = self.status_hashes.get(status.value, [])
        return [self.hash_entries[hash_id] for hash_id in hash_ids if hash_id in self.hash_entries]
    
    async def get_recent_hashes(self, limit: int = 100) -> List[HashEntry]:
        """Get recent hash entries."""
        entries = list(self.hash_entries.values())
        entries.sort(key=lambda x: x.timestamp, reverse=True)
        return entries[:limit]
    
    async def get_hash_family(self, hash_id: str) -> Dict[str, List[HashEntry]]:
        """Get hash family (parent and children)."""
        entry = self.hash_entries.get(hash_id)
        if not entry:
            return {}
        
        family = {
            "parent": None,
            "children": [],
            "siblings": [],
        }
        
        # Get parent
        if entry.parent_hash_id:
            family["parent"] = self.hash_entries.get(entry.parent_hash_id)
        
        # Get children
        for child_id in entry.child_hash_ids:
            child_entry = self.hash_entries.get(child_id)
            if child_entry:
                family["children"].append(child_entry)
        
        # Get siblings (same parent)
        if entry.parent_hash_id:
            parent_entry = self.hash_entries.get(entry.parent_hash_id)
            if parent_entry:
                for sibling_id in parent_entry.child_hash_ids:
                    if sibling_id != hash_id:
                        sibling_entry = self.hash_entries.get(sibling_id)
                        if sibling_entry:
                            family["siblings"].append(sibling_entry)
        
        return family
    
    async def _detect_patterns(self, hash_id: str) -> None:
        """Detect patterns in hash sequences."""
        try:
            # Get recent hashes for pattern analysis
            recent_hashes = await self.get_recent_hashes(self.pattern_window_size)
            
            # Look for repeating sequences
            for i in range(len(recent_hashes) - 2):
                for j in range(i + 3, len(recent_hashes)):
                    sequence = recent_hashes[i:j]
                    
                    # Check if this sequence appears elsewhere
                    if self._is_pattern_sequence(sequence, recent_hashes):
                        pattern_id = self._generate_pattern_id(sequence)
                        
                        if pattern_id not in self.hash_patterns:
                            # Create new pattern
                            pattern = HashPattern(
                                pattern_id=pattern_id,
                                pattern_type="sequence",
                                hash_sequence=[h.hash_id for h in sequence],
                                frequency=1,
                                first_seen=sequence[0].timestamp,
                                last_seen=sequence[-1].timestamp,
                                success_rate=self._calculate_sequence_success_rate(sequence),
                                average_execution_time=self._calculate_sequence_avg_time(sequence),
                                confidence_score=self._calculate_sequence_confidence(sequence),
                            )
                            self.hash_patterns[pattern_id] = pattern
                        else:
                            # Update existing pattern
                            pattern = self.hash_patterns[pattern_id]
                            pattern.frequency += 1
                            pattern.last_seen = sequence[-1].timestamp
                            pattern.success_rate = self._calculate_sequence_success_rate(sequence)
                            pattern.average_execution_time = self._calculate_sequence_avg_time(sequence)
                            pattern.confidence_score = self._calculate_sequence_confidence(sequence)
            
        except Exception as e:
            safe_safe_print(f"⚠️ Pattern detection failed: {safe_format_error(e, 'pattern_detection')}")
    
    def _is_pattern_sequence(self, sequence: List[HashEntry], all_hashes: List[HashEntry]) -> bool:
        """Check if a sequence appears as a pattern."""
        if len(sequence) < 3:
            return False
        
        # Create sequence signature
        sequence_signature = self._create_sequence_signature(sequence)
        
        # Look for this signature in other parts of the hash list
        for i in range(len(all_hashes) - len(sequence) + 1):
            if i == 0:  # Skip the original sequence
                continue
            
            test_sequence = all_hashes[i:i + len(sequence)]
            test_signature = self._create_sequence_signature(test_sequence)
            
            if self._compare_signatures(sequence_signature, test_signature):
                return True
        
        return False
    
    def _create_sequence_signature(self, sequence: List[HashEntry]) -> str:
        """Create signature for a sequence of hashes."""
        signature_parts = []
        for entry in sequence:
            part = f"{entry.hash_type.value}_{entry.agent_type}_{entry.domain}"
            signature_parts.append(part)
        return "|".join(signature_parts)
    
    def _compare_signatures(self, sig1: str, sig2: str) -> bool:
        """Compare two sequence signatures."""
        return sig1 == sig2
    
    def _generate_pattern_id(self, sequence: List[HashEntry]) -> str:
        """Generate pattern ID from sequence."""
        signature = self._create_sequence_signature(sequence)
        return hashlib.sha256(signature.encode()).hexdigest()[:16]
    
    def _calculate_sequence_success_rate(self, sequence: List[HashEntry]) -> float:
        """Calculate success rate for a sequence."""
        if not sequence:
            return 0.0
        
        successful = sum(1 for entry in sequence if entry.status == HashStatus.COMPLETED)
        return successful / len(sequence)
    
    def _calculate_sequence_avg_time(self, sequence: List[HashEntry]) -> float:
        """Calculate average execution time for a sequence."""
        if not sequence:
            return 0.0
        
        total_time = sum(entry.execution_time for entry in sequence)
        return total_time / len(sequence)
    
    def _calculate_sequence_confidence(self, sequence: List[HashEntry]) -> float:
        """Calculate confidence score for a sequence."""
        if not sequence:
            return 0.0
        
        avg_confidence = sum(entry.confidence_score for entry in sequence)
        success_rate = self._calculate_sequence_success_rate(sequence)
        
        return (avg_confidence + success_rate) / 2.0
    
    async def get_patterns(self, pattern_type: Optional[str] = None) -> List[HashPattern]:
        """Get hash patterns."""
        patterns = list(self.hash_patterns.values())
        
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        # Sort by frequency and confidence
        patterns.sort(key=lambda x: (x.frequency, x.confidence_score), reverse=True)
        return patterns
    
    async def validate_hash(self, hash_id: str, validation_data: Dict[str, Any]) -> bool:
        """Validate a hash entry."""
        try:
            entry = self.hash_entries.get(hash_id)
            if not entry:
                return False
            
            # Update validation data
            entry.validation_data.update(validation_data)
            
            # Check validation rules
            is_valid = self._apply_validation_rules(entry)
            
            # Update status
            new_status = HashStatus.VALIDATED if is_valid else HashStatus.INVALID
            await self.update_hash_status(hash_id, new_status)
            
            return is_valid
            
        except Exception as e:
            safe_safe_print(f"❌ Hash validation failed: {safe_format_error(e, 'hash_validation')}")
            return False
    
    def _apply_validation_rules(self, entry: HashEntry) -> bool:
        """Apply validation rules to hash entry."""
        # Rule 1: Check recursive depth
        if entry.recursive_depth > self.recursive_depth_limit:
            return False
        
        # Rule 2: Check confidence score
        if entry.confidence_score < 0.3:
            return False
        
        # Rule 3: Check payload structure
        if not self._validate_payload_structure(entry):
            return False
        
        # Rule 4: Check for known patterns
        if self._is_known_failure_pattern(entry):
            return False
        
        return True
    
    def _validate_payload_structure(self, entry: HashEntry) -> bool:
        """Validate payload structure."""
        # Basic structure validation
        if not isinstance(entry.payload, dict):
            return False
        
        # Domain-specific validation
        if entry.domain == "strategy":
            required_fields = ["strategy_name", "parameters"]
            return all(field in entry.payload for field in required_fields)
        
        elif entry.domain == "profit":
            required_fields = ["allocation_amount", "risk_level"]
            return all(field in entry.payload for field in required_fields)
        
        return True
    
    def _is_known_failure_pattern(self, entry: HashEntry) -> bool:
        """Check if entry matches known failure patterns."""
        # Check against existing patterns
        for pattern in self.hash_patterns.values():
            if pattern.success_rate < 0.2:  # Low success rate pattern
                if entry.hash_id in pattern.hash_sequence:
                    return True
        
        return False
    
    async def cleanup_old_entries(self) -> None:
        """Clean up old hash entries."""
        try:
            current_time = time.time()
            
            # Check if cleanup is needed
            if current_time - self.last_cleanup < self.cleanup_interval:
                return
            
            # Remove old entries
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days
            old_entries = [
                hash_id for hash_id, entry in self.hash_entries.items()
                if entry.timestamp < cutoff_time
            ]
            
            for hash_id in old_entries:
                entry = self.hash_entries[hash_id]
                self._unindex_entry(entry)
                del self.hash_entries[hash_id]
            
            # Enforce max entries limit
            if len(self.hash_entries) > self.max_entries:
                # Remove oldest entries
                entries = list(self.hash_entries.items())
                entries.sort(key=lambda x: x[1].timestamp)
                
                excess_count = len(entries) - self.max_entries
                for i in range(excess_count):
                    hash_id, entry = entries[i]
                    self._unindex_entry(entry)
                    del self.hash_entries[hash_id]
            
            self.last_cleanup = current_time
            self._save_registry()
            
            safe_safe_print(f"🧹 Cleaned up {len(old_entries)} old hash entries")
            
        except Exception as e:
            safe_safe_print(f"⚠️ Cleanup failed: {safe_format_error(e, 'cleanup')}")
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        try:
            stats = {
                "total_entries": len(self.hash_entries),
                "total_patterns": len(self.hash_patterns),
                "entries_by_agent": {agent: len(hashes) for agent, hashes in self.agent_hashes.items()},
                "entries_by_domain": {domain: len(hashes) for domain, hashes in self.domain_hashes.items()},
                "entries_by_status": {status: len(hashes) for status, hashes in self.status_hashes.items()},
                "recent_activity": len(await self.get_recent_hashes(24)),  # Last 24 entries
                "registry_file": self.registry_file,
                "last_cleanup": datetime.fromtimestamp(self.last_cleanup).isoformat(),
            }
            
            return stats
            
        except Exception as e:
            safe_safe_print(f"❌ Stats calculation failed: {safe_format_error(e, 'stats')}")
            return {}
    
    async def start_cleanup_task(self) -> None:
        """Start the cleanup task."""
        async def cleanup_loop() -> None:
            """Cleanup loop for old entries."""
            while True:
                try:
                    await self.cleanup_old_entries()
                    await asyncio.sleep(self.cleanup_interval)
                except Exception as e:
                    safe_safe_print(f"⚠️ Cleanup task error: {safe_format_error(e, 'cleanup_task')}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
        safe_safe_print("🧹 Cleanup task started")
    
    async def stop_cleanup_task(self) -> None:
        """Stop cleanup task."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            safe_safe_print("🧹 Cleanup task stopped")


# Global hash registry instance
hash_registry = HashRegistry()


# Convenience functions for external access
async def register_hash_entry(
    hash_type: str,
    agent_type: str,
    domain: str,
    payload: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    command_id: Optional[str] = None,
    parent_hash_id: Optional[str] = None,
    confidence_score: float = 0.0,
) -> str:
    """Register a new hash entry."""
    hash_type_enum = HashType(hash_type)
    return await hash_registry.register_hash(
        hash_type=hash_type_enum,
        agent_type=agent_type,
        domain=domain,
        payload=payload,
        context=context,
        command_id=command_id,
        parent_hash_id=parent_hash_id,
        confidence_score=confidence_score,
    )


async def get_hash_entry(hash_id: str) -> Optional[HashEntry]:
    """Get hash entry by ID."""
    return await hash_registry.get_hash_entry(hash_id)


async def update_hash_status(
    hash_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    execution_time: float = 0.0,
) -> bool:
    """Update hash entry status."""
    status_enum = HashStatus(status)
    return await hash_registry.update_hash_status(
        hash_id=hash_id,
        status=status_enum,
        result=result,
        error_message=error_message,
        execution_time=execution_time,
    )


async def get_registry_stats() -> Dict[str, Any]:
    """Get registry statistics."""
    return await hash_registry.get_registry_stats()


# Example usage
if __name__ == "__main__":
    async def test_hash_registry():
        """Test hash registry functionality."""
        safe_safe_print("🧠 Testing hash registry...")
        
        # Register test hashes
        hash_id1 = await register_hash_entry(
            hash_type="command",
            agent_type="gpt",
            domain="strategy",
            payload={"strategy_name": "test_strategy", "parameters": {"test": True}},
            context={"test": True},
            confidence_score=0.8,
        )
        
        hash_id2 = await register_hash_entry(
            hash_type="command",
            agent_type="gpt",
            domain="profit",
            payload={"allocation_amount": 100.0, "risk_level": "medium"},
            context={"test": True},
            parent_hash_id=hash_id1,
            confidence_score=0.7,
        )
        
        # Update status
        await update_hash_status(hash_id1, "completed", {"result": "success"}, execution_time=1.5)
        await update_hash_status(hash_id2, "failed", error_message="Test error", execution_time=0.5)
        
        # Get stats
        stats = await get_registry_stats()
        safe_safe_print(f"📊 Registry stats: {stats}")
        
        # Start cleanup task
        await hash_registry.start_cleanup_task()
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Stop cleanup task
        await hash_registry.stop_cleanup_task()
    
    # Run test
    asyncio.run(test_hash_registry()) 