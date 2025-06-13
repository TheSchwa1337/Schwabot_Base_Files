"""
Config Checksum Verifier
======================

Validates YAML configuration files against stored checksums
to detect unauthorized modifications or corruption.
"""

from typing import Dict, Optional, Tuple
import hashlib
import json
from pathlib import Path
import yaml
import os
from jsonschema import validate, ValidationError
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ChecksumVerifier:
    def __init__(self, checksum_file: str = "config_checksums.json", config_dir: Optional[Path] = None):
        self.root = Path(__file__).resolve().parent
        self.config_dir = config_dir or self.root / "config"
        self.checksum_path = self.config_dir / checksum_file
        self.checksums: Dict[str, str] = {}
        self._load_checksums()
        
    def _load_checksums(self):
        """Load stored checksums"""
        if self.checksum_path.exists():
            try:
                with self.checksum_path.open("r") as f:
                    self.checksums = json.load(f)
                logger.info(f"Loaded {len(self.checksums)} checksums from {self.checksum_path}")
            except Exception as e:
                logger.warning(f"Failed to load checksums: {e}")
        else:
            logger.warning("No existing checksum file found")
                
    def _save_checksums(self):
        """Save current checksums"""
        try:
            with self.checksum_path.open("w") as f:
                json.dump(self.checksums, f, indent=2)
            logger.info(f"Saved {len(self.checksums)} checksums")
        except Exception as e:
            logger.error(f"Failed to save checksums: {e}")
            
    def compute_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum of a file"""
        try:
            with file_path.open("rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Could not compute checksum for {file_path}: {e}")
            return ""
            
    def verify_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Verify a file against its stored checksum"""
        if not file_path.exists():
            return False, "File does not exist"
            
        current_checksum = self.compute_checksum(file_path)
        relative_path = str(file_path.relative_to(self.root))
        stored_checksum = self.checksums.get(relative_path)
        
        if not stored_checksum:
            return False, "No stored checksum found"
            
        if current_checksum != stored_checksum:
            return False, "Checksum mismatch"
            
        return True, None
        
    def update_checksum(self, file_path: Path):
        """Update stored checksum for a file"""
        checksum = self.compute_checksum(file_path)
        relative_path = str(file_path.relative_to(self.root))
        if checksum:
            self.checksums[relative_path] = checksum
            self._save_checksums()
            
    def verify_all_configs(self) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Verify all YAML config files"""
        results = {}
        for yaml_file in self.config_dir.glob("*.yaml"):
            results[str(yaml_file.name)] = self.verify_file(yaml_file)
            
        return results 

    def validate_config_schema(self, config_path: Path) -> Tuple[bool, Optional[str]]:
        try:
            with config_path.open("r") as f:
                config = yaml.safe_load(f)
            schema = {
                "type": "object",
                "properties": {
                    "fractal": {
                        "type": "object",
                        "properties": {
                            "decay_power": {"type": "number"}
                        },
                        "required": ["decay_power"]
                    }
                },
                "required": ["fractal"]
            }
            validate(instance=config, schema=schema)
            return True, None
        except (yaml.YAMLError, ValidationError) as e:
            return False, str(e)

    def load_and_verify(self, file_path: Path, auto_update: bool = False) -> Optional[dict]:
        ok, err = self.verify_file(file_path)
        if not ok and auto_update:
            logger.warning(f"{file_path.name}: {err}, updating checksum...")
            self.update_checksum(file_path)
        elif not ok:
            logger.error(f"{file_path.name}: {err}")
            return None

        with file_path.open("r") as f:
            return yaml.safe_load(f)

# Minimal unittest hook
def test_checksum_verification():
    verifier = ChecksumVerifier()
    results = verifier.verify_all_configs()
    for file, (status, reason) in results.items():
        assert status, f"{file} failed checksum: {reason}"

if __name__ == "__main__":
    test_checksum_verification() 