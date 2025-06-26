from utils.safe_print import safe_print, info, warn, error, success, debug
#!/usr/bin/env python3
"""
File Integrity Checker - Core File System Integrity Pipeline Component
====================================================================

This module provides file integrity checking functionality for the Schwabot system.
It validates file integrity, checks file consistency, and maintains file system
health across the trading pipeline.

Core Functionality:
- File integrity validation
- Checksum verification
- File consistency checking
- Corrupted file detection
- File system health monitoring
"""

import hashlib
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class IntegrityCheckResult:
    """Result of file integrity check operation."""
    file_path: str
    is_valid: bool
    checksum: str
    check_time: datetime
    file_size: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class FileIntegrityChecker:
    """Core file integrity checking system for Schwabot."""

    def __init__(self):
        """Initialize the file integrity checker."""
        self.check_history: List[IntegrityCheckResult] = []
        self.file_checksums: Dict[str, str] = {}
        self.corrupted_files: List[str] = []
        self.check_count = 0

        logger.info("File Integrity Checker initialized")

    def calculate_file_checksum(self, file_path: str, algorithm: str = "sha256") -> str:
        """Calculate checksum for a file."""
        try:
            hash_func = getattr(hashlib, algorithm)()

            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)

            return hash_func.hexdigest()

        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""

    def check_file_integrity(self, file_path: str, expected_checksum: Optional[str] = None) -> IntegrityCheckResult:
        """Check integrity of a file."""
        try:
            if not os.path.exists(file_path):
                return IntegrityCheckResult(
                    file_path=file_path,
                    is_valid=False,
                    checksum="",
                    check_time=datetime.now(),
                    file_size=0,
                    error_message="File does not exist"
                )

            # Get file size
            file_size = os.path.getsize(file_path)

            # Calculate current checksum
            current_checksum = self.calculate_file_checksum(file_path)

            if not current_checksum:
                return IntegrityCheckResult(
                    file_path=file_path,
                    is_valid=False,
                    checksum="",
                    check_time=datetime.now(),
                    file_size=file_size,
                    error_message="Failed to calculate checksum"
                )

            # Check if we have a stored checksum
            stored_checksum = self.file_checksums.get(file_path)

            # Determine validity
            if expected_checksum:
                is_valid = current_checksum == expected_checksum
            elif stored_checksum:
                is_valid = current_checksum == stored_checksum
            else:
                # First time checking this file, store the checksum
                is_valid = True
                self.file_checksums[file_path] = current_checksum

            # Update stored checksum if valid
            if is_valid:
                self.file_checksums[file_path] = current_checksum
            else:
                if file_path not in self.corrupted_files:
                    self.corrupted_files.append(file_path)

            result = IntegrityCheckResult(
                file_path=file_path,
                is_valid=is_valid,
                checksum=current_checksum,
                check_time=datetime.now(),
                file_size=file_size,
                metadata={
                    "stored_checksum": stored_checksum,
                    "expected_checksum": expected_checksum
                }
            )

            self.check_history.append(result)
            self.check_count += 1

            logger.debug(f"File integrity check completed: {file_path} - {is_valid}")
            return result

        except Exception as e:
            logger.error(f"File integrity check error for {file_path}: {e}")
            return IntegrityCheckResult(
                file_path=file_path,
                is_valid=False,
                checksum="",
                check_time=datetime.now(),
                file_size=0,
                error_message=str(e)
            )

    def check_directory_integrity(self, directory_path: str, recursive: bool = True) -> List[IntegrityCheckResult]:
        """Check integrity of all files in a directory."""
        results = []

        try:
            directory = Path(directory_path)
            if not directory.exists():
                logger.error(f"Directory does not exist: {directory_path}")
                return results

            # Get all files in directory
            if recursive:
                files = list(directory.rglob("*"))
            else:
                files = list(directory.glob("*"))

            # Filter for files only
            files = [f for f in files if f.is_file()]

            logger.info(f"Checking integrity of {len(files)} files in {directory_path}")

            for file_path in files:
                result = self.check_file_integrity(str(file_path))
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Directory integrity check error: {e}")
            return results

    def detect_corrupted_files(self, directory_path: str) -> List[str]:
        """Detect corrupted files in a directory."""
        corrupted = []

        try:
            directory = Path(directory_path)
            if not directory.exists():
                return corrupted

            # Check all files in directory
            results = self.check_directory_integrity(directory_path)

            # Find corrupted files
            for result in results:
                if not result.is_valid:
                    corrupted.append(result.file_path)

            logger.info(f"Detected {len(corrupted)} corrupted files in {directory_path}")
            return corrupted

        except Exception as e:
            logger.error(f"Corrupted file detection error: {e}")
            return corrupted

    def repair_corrupted_file(self, file_path: str, backup_path: Optional[str] = None) -> bool:
        """Attempt to repair a corrupted file."""
        try:
            if file_path not in self.corrupted_files:
                logger.warning(f"File not marked as corrupted: {file_path}")
                return False

            # Try to restore from backup
            if backup_path and os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, file_path)
                logger.info(f"Restored {file_path} from backup")

                # Re-check integrity
                result = self.check_file_integrity(file_path)
                if result.is_valid:
                    self.corrupted_files.remove(file_path)
                    return True
                else:
                    logger.error(f"File still corrupted after backup restoration: {file_path}")
                    return False
            else:
                logger.error(f"No backup available for corrupted file: {file_path}")
                return False

        except Exception as e:
            logger.error(f"File repair error: {e}")
            return False

    def get_integrity_statistics(self) -> Dict[str, Any]:
        """Get integrity check statistics."""
        total_checks = len(self.check_history)
        valid_files = sum(1 for result in self.check_history if result.is_valid)
        success_rate = valid_files / total_checks if total_checks > 0 else 0.0

        return {
            "total_checks": total_checks,
            "valid_files": valid_files,
            "corrupted_files": len(self.corrupted_files),
            "success_rate": success_rate,
            "tracked_files": len(self.file_checksums)
        }

    def export_checksums(self, output_file: str) -> bool:
        """Export checksums to a file."""
        try:
            with open(output_file, 'w') as f:
                for file_path, checksum in self.file_checksums.items():
                    f.write(f"{file_path}:{checksum}\n")

            logger.info(f"Checksums exported to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Checksum export error: {e}")
            return False

    def import_checksums(self, input_file: str) -> bool:
        """Import checksums from a file."""
        try:
            with open(input_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        file_path, checksum = line.strip().split(':', 1)
                        self.file_checksums[file_path] = checksum

            logger.info(f"Checksums imported from {input_file}")
            return True

        except Exception as e:
            logger.error(f"Checksum import error: {e}")
            return False


def main() -> None:
    """Main function for testing file integrity checking."""
    checker = FileIntegrityChecker()

    # Test file integrity check
    test_file = __file__  # Check this file
    result = checker.check_file_integrity(test_file)
    safe_print(f"File integrity check result: {result.is_valid}")

    # Get statistics
    stats = checker.get_integrity_statistics()
    safe_print(f"Integrity statistics: {stats}")


if __name__ == "__main__":
    main()
