"""
SHA Key Basket Mapper
===================

Manages mappings between basket IDs and their SHA-256 hashes.
Provides secure identification and tracking of trading baskets.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class SHAMapper:
    """Maps basket IDs to SHA-256 hashes"""
    
    def __init__(self, map_path: Optional[Path] = None):
        """Initialize the SHA mapper"""
        self.map_path = map_path or Path("data/sha_key_basket_map.json")
        self.basket_to_sha: Dict[str, str] = {}
        self.sha_to_basket: Dict[str, str] = {}
        self._load_mappings()
        
    def _load_mappings(self) -> None:
        """Load existing mappings from file"""
        try:
            if self.map_path.exists():
                with open(self.map_path, 'r') as f:
                    data = json.load(f)
                    self.basket_to_sha = data.get("basket_to_sha", {})
                    self.sha_to_basket = data.get("sha_to_basket", {})
                    
                logger.info(f"Loaded {len(self.basket_to_sha)} basket mappings")
                
        except Exception as e:
            logger.error(f"Error loading SHA mappings: {e}")
            raise
            
    def _save_mappings(self) -> None:
        """Save mappings to file"""
        try:
            self.map_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "basket_to_sha": self.basket_to_sha,
                "sha_to_basket": self.sha_to_basket,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.map_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving SHA mappings: {e}")
            raise
            
    def get_sha_key(self, basket_id: str) -> str:
        """Get SHA key for a basket ID"""
        if basket_id not in self.basket_to_sha:
            # Generate new SHA key
            sha_key = self._generate_sha_key(basket_id)
            self.basket_to_sha[basket_id] = sha_key
            self.sha_to_basket[sha_key] = basket_id
            self._save_mappings()
            
        return self.basket_to_sha[basket_id]
        
    def get_basket_id(self, sha_key: str) -> Optional[str]:
        """Get basket ID for a SHA key"""
        return self.sha_to_basket.get(sha_key)
        
    def _generate_sha_key(self, basket_id: str) -> str:
        """Generate SHA-256 key for a basket ID"""
        # Include timestamp for uniqueness
        timestamp = datetime.now().isoformat()
        data = f"{basket_id}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()
        
    def get_all_mappings(self) -> Dict[str, Dict[str, str]]:
        """Get all basket-SHA mappings"""
        return {
            "basket_to_sha": self.basket_to_sha.copy(),
            "sha_to_basket": self.sha_to_basket.copy()
        }
        
    def validate_mapping(self, basket_id: str, sha_key: str) -> bool:
        """Validate a basket ID and SHA key mapping"""
        return (
            basket_id in self.basket_to_sha and
            self.basket_to_sha[basket_id] == sha_key and
            sha_key in self.sha_to_basket and
            self.sha_to_basket[sha_key] == basket_id
        )
        
    def get_basket_history(self, basket_id: str) -> List[Dict[str, Any]]:
        """Get history of SHA keys for a basket"""
        history = []
        
        try:
            history_path = Path("logs/basket_history.json")
            if history_path.exists():
                with open(history_path, 'r') as f:
                    data = json.load(f)
                    history = data.get(basket_id, [])
                    
        except Exception as e:
            logger.error(f"Error loading basket history: {e}")
            
        return history
        
    def log_basket_change(self, basket_id: str, old_sha: Optional[str], new_sha: str) -> None:
        """Log a basket SHA key change"""
        try:
            history_path = Path("logs/basket_history.json")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing history
            history = {}
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                    
            # Add new entry
            if basket_id not in history:
                history[basket_id] = []
                
            history[basket_id].append({
                "timestamp": datetime.now().isoformat(),
                "old_sha": old_sha,
                "new_sha": new_sha
            })
            
            # Save updated history
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging basket change: {e}")
            
    def get_basket_statistics(self) -> Dict[str, Any]:
        """Get statistics about basket mappings"""
        return {
            "total_baskets": len(self.basket_to_sha),
            "total_sha_keys": len(self.sha_to_basket),
            "last_updated": datetime.now().isoformat()
        } 