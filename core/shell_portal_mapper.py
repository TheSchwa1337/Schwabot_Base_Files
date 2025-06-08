"""
Shell Portal Mapper
=================

Maps trigger routes to shell entry points and handles bounce-back logic
for failed triggers.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from datetime import datetime

@dataclass
class ShellRoute:
    """Represents a shell route configuration"""
    entry_point: str
    primary_route: str
    fallback_route: Optional[str] = None
    bounce_back_map: Optional[Dict[str, str]] = None
    cooldown_zone: str = "zpe"  # zpe or zbe

class ShellPortalMapper:
    """Maps trigger routes to shell entry points"""
    
    def __init__(self, config_path: str = "config/shell_routes.json"):
        self.config_path = Path(config_path)
        self.routes: Dict[str, ShellRoute] = {}
        self.load_config()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ShellPortalMapper')
        
    def load_config(self) -> None:
        """Load shell route configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    for route_id, data in config.items():
                        self.routes[route_id] = ShellRoute(
                            entry_point=data['entry_point'],
                            primary_route=data['primary_route'],
                            fallback_route=data.get('fallback_route'),
                            bounce_back_map=data.get('bounce_back_map'),
                            cooldown_zone=data.get('cooldown_zone', 'zpe')
                        )
        except Exception as e:
            self.logger.error(f"Failed to load shell routes: {e}")
            
    def get_route(self, trigger_id: str) -> Optional[ShellRoute]:
        """Get shell route for a trigger"""
        return self.routes.get(trigger_id)
        
    def get_entry_point(self, trigger_id: str) -> Optional[str]:
        """Get shell entry point for a trigger"""
        route = self.get_route(trigger_id)
        return route.entry_point if route else None
        
    def get_fallback_route(self, trigger_id: str) -> Optional[str]:
        """Get fallback route for a trigger"""
        route = self.get_route(trigger_id)
        return route.fallback_route if route else None
        
    def get_bounce_back(self, trigger_id: str, failure_type: str) -> Optional[str]:
        """Get bounce-back route for a failed trigger"""
        route = self.get_route(trigger_id)
        if route and route.bounce_back_map:
            return route.bounce_back_map.get(failure_type)
        return None
        
    def get_cooldown_zone(self, trigger_id: str) -> str:
        """Get cooldown zone for a trigger"""
        route = self.get_route(trigger_id)
        return route.cooldown_zone if route else 'zpe'
        
    def register_route(self, trigger_id: str, route: ShellRoute) -> None:
        """Register a new shell route"""
        self.routes[trigger_id] = route
        self._save_config()
        
    def _save_config(self) -> None:
        """Save current route configuration"""
        try:
            config = {}
            for trigger_id, route in self.routes.items():
                config[trigger_id] = {
                    'entry_point': route.entry_point,
                    'primary_route': route.primary_route,
                    'fallback_route': route.fallback_route,
                    'bounce_back_map': route.bounce_back_map,
                    'cooldown_zone': route.cooldown_zone
                }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save shell routes: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize mapper
    mapper = ShellPortalMapper()
    
    # Example route lookup
    route = mapper.get_route("0x8845")
    if route:
        print(f"Entry point: {route.entry_point}")
        print(f"Primary route: {route.primary_route}")
        print(f"Cooldown zone: {route.cooldown_zone}") 