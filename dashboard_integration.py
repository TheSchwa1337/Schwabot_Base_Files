"""
Dashboard Integration
====================

Provides dashboard bridge functionality for Schwabot configuration and status display.
Connects with the configuration system and provides real-time status monitoring.
"""

from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from dataclasses import asdict

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not available. Install with: pip install rich")

from config.config_utils import get_profile_params_from_yaml, load_yaml_config, ConfigError

logger = logging.getLogger(__name__)

class DashboardBridge:
    """
    Dashboard bridge for Schwabot configuration and status management
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize dashboard bridge
        
        Args:
            config_path: Optional path to main configuration file
        """
        self.config_path = config_path or Path(__file__).resolve().parent / "config" / "fractal_core.yaml"
        self.console = Console() if RICH_AVAILABLE else None
        self.profile = {}
        self.status_data = {}
        self.last_update = None
        
        # Load initial configuration
        self.reload_config()
        
        logger.info(f"Dashboard bridge initialized with config: {self.config_path}")
    
    def reload_config(self) -> None:
        """Reload configuration from file"""
        try:
            self.profile = get_profile_params_from_yaml(self.config_path)
            self.last_update = datetime.now()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            self.profile = {}
    
    def get_profile(self) -> Dict[str, Any]:
        """Get current profile configuration"""
        return self.profile.copy()
    
    def get_as_dict(self) -> Dict[str, Any]:
        """Get profile as dictionary (legacy compatibility)"""
        return self.get_profile()
    
    def update_status(self, status_data: Dict[str, Any]) -> None:
        """
        Update status data for dashboard display
        
        Args:
            status_data: Dictionary containing current status information
        """
        self.status_data.update(status_data)
        self.status_data['last_update'] = datetime.now().isoformat()
    
    def display_profile(self) -> None:
        """Display profile configuration using Rich formatting"""
        if not RICH_AVAILABLE:
            self._display_profile_simple()
            return
        
        self.console.rule("[bold blue]Fractal Profile Configuration")
        
        if not self.profile:
            self.console.print("[red]No profile data available[/red]")
            return
        
        # Create profile table
        table = Table(title="Profile Parameters")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Type", style="green")
        
        def add_nested_items(data: Dict[str, Any], prefix: str = ""):
            """Recursively add nested dictionary items to table"""
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    add_nested_items(value, full_key)
                else:
                    table.add_row(
                        full_key,
                        str(value),
                        type(value).__name__
                    )
        
        add_nested_items(self.profile)
        self.console.print(table)
        
        if self.last_update:
            self.console.print(f"\n[dim]Last updated: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    
    def _display_profile_simple(self) -> None:
        """Simple text display for when Rich is not available"""
        print("\n" + "="*50)
        print("FRACTAL PROFILE CONFIGURATION")
        print("="*50)
        
        if not self.profile:
            print("No profile data available")
            return
        
        def print_nested(data: Dict[str, Any], indent: int = 0):
            """Recursively print nested dictionary"""
            for key, value in data.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_nested(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value} ({type(value).__name__})")
        
        print_nested(self.profile)
        
        if self.last_update:
            print(f"\nLast updated: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
    
    def display_status(self) -> None:
        """Display current system status"""
        if not RICH_AVAILABLE:
            self._display_status_simple()
            return
        
        self.console.rule("[bold green]System Status")
        
        if not self.status_data:
            self.console.print("[yellow]No status data available[/yellow]")
            return
        
        # Create status panels
        panels = []
        
        # System metrics panel
        if any(key in self.status_data for key in ['tick_counter', 'pattern_history_size', 'shell_history_size']):
            metrics_text = ""
            if 'tick_counter' in self.status_data:
                metrics_text += f"Tick Counter: {self.status_data['tick_counter']}\n"
            if 'pattern_history_size' in self.status_data:
                metrics_text += f"Pattern History: {self.status_data['pattern_history_size']}\n"
            if 'shell_history_size' in self.status_data:
                metrics_text += f"Shell History: {self.status_data['shell_history_size']}\n"
            
            panels.append(Panel(metrics_text.strip(), title="System Metrics", border_style="blue"))
        
        # Strategy panel
        if any(key in self.status_data for key in ['active_strategy', 'vault_locked', 're_entry_trigger']):
            strategy_text = ""
            if 'active_strategy' in self.status_data:
                strategy_text += f"Active Strategy: {self.status_data['active_strategy']}\n"
            if 'vault_locked' in self.status_data:
                vault_status = "🔒 LOCKED" if self.status_data['vault_locked'] else "🔓 UNLOCKED"
                strategy_text += f"Vault Status: {vault_status}\n"
            if 're_entry_trigger' in self.status_data:
                trigger_status = "✅ ACTIVE" if self.status_data['re_entry_trigger'] else "❌ INACTIVE"
                strategy_text += f"Re-entry Trigger: {trigger_status}\n"
            
            panels.append(Panel(strategy_text.strip(), title="Strategy Status", border_style="green"))
        
        # Debug panel
        if any(key in self.status_data for key in ['test_mode', 'verbose_logging']):
            debug_text = ""
            if 'test_mode' in self.status_data:
                test_status = "🧪 ENABLED" if self.status_data['test_mode'] else "❌ DISABLED"
                debug_text += f"Test Mode: {test_status}\n"
            if 'verbose_logging' in self.status_data:
                verbose_status = "📝 ENABLED" if self.status_data['verbose_logging'] else "❌ DISABLED"
                debug_text += f"Verbose Logging: {verbose_status}\n"
            
            panels.append(Panel(debug_text.strip(), title="Debug Status", border_style="yellow"))
        
        # Display panels
        for panel in panels:
            self.console.print(panel)
        
        # Last update info
        if 'last_update' in self.status_data:
            self.console.print(f"\n[dim]Status last updated: {self.status_data['last_update']}[/dim]")
    
    def _display_status_simple(self) -> None:
        """Simple text display for status when Rich is not available"""
        print("\n" + "="*50)
        print("SYSTEM STATUS")
        print("="*50)
        
        if not self.status_data:
            print("No status data available")
            return
        
        for key, value in self.status_data.items():
            if key == 'last_update':
                continue
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        if 'last_update' in self.status_data:
            print(f"\nLast updated: {self.status_data['last_update']}")
        print("="*50)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for external display systems
        
        Returns:
            Dictionary containing all dashboard data
        """
        return {
            'profile': self.profile,
            'status': self.status_data,
            'config_path': str(self.config_path),
            'last_config_update': self.last_update.isoformat() if self.last_update else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_dashboard_json(self, output_path: Optional[Path] = None) -> Path:
        """
        Export dashboard data to JSON file
        
        Args:
            output_path: Optional path for output file
            
        Returns:
            Path to the exported JSON file
        """
        if output_path is None:
            output_path = Path("dashboard_export.json")
        
        dashboard_data = self.get_dashboard_data()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info(f"Dashboard data exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            raise
    
    def load_multiple_configs(self, config_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
        """
        Load multiple configuration files for comparison
        
        Args:
            config_paths: List of paths to configuration files
            
        Returns:
            Dictionary mapping config names to their data
        """
        configs = {}
        
        for config_path in config_paths:
            try:
                config_name = config_path.stem
                config_data = load_yaml_config(config_path)
                configs[config_name] = config_data
                logger.info(f"Loaded config: {config_name}")
            except Exception as e:
                logger.error(f"Error loading config {config_path}: {e}")
                configs[config_path.stem] = {"error": str(e)}
        
        return configs
    
    def compare_configs(self, config_paths: List[Path]) -> None:
        """
        Display comparison of multiple configuration files
        
        Args:
            config_paths: List of paths to configuration files to compare
        """
        configs = self.load_multiple_configs(config_paths)
        
        if not RICH_AVAILABLE:
            self._compare_configs_simple(configs)
            return
        
        self.console.rule("[bold purple]Configuration Comparison")
        
        # Create comparison table
        table = Table(title="Config Comparison")
        table.add_column("Parameter", style="cyan")
        
        for config_name in configs.keys():
            table.add_column(config_name, style="magenta")
        
        # Get all unique keys across all configs
        all_keys = set()
        for config_data in configs.values():
            if isinstance(config_data, dict) and "error" not in config_data:
                all_keys.update(self._get_all_keys(config_data))
        
        # Add rows for each parameter
        for key in sorted(all_keys):
            row = [key]
            for config_name, config_data in configs.items():
                if isinstance(config_data, dict) and "error" not in config_data:
                    value = self._get_nested_value(config_data, key)
                    row.append(str(value) if value is not None else "N/A")
                else:
                    row.append("ERROR")
            table.add_row(*row)
        
        self.console.print(table)
    
    def _compare_configs_simple(self, configs: Dict[str, Dict[str, Any]]) -> None:
        """Simple text comparison when Rich is not available"""
        print("\n" + "="*80)
        print("CONFIGURATION COMPARISON")
        print("="*80)
        
        for config_name, config_data in configs.items():
            print(f"\n--- {config_name.upper()} ---")
            if isinstance(config_data, dict) and "error" not in config_data:
                self._print_config_simple(config_data)
            else:
                print(f"ERROR: {config_data.get('error', 'Unknown error')}")
        
        print("="*80)
    
    def _print_config_simple(self, config: Dict[str, Any], indent: int = 0) -> None:
        """Print configuration in simple text format"""
        for key, value in config.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_config_simple(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    def _get_all_keys(self, data: Dict[str, Any], prefix: str = "") -> List[str]:
        """Get all keys from nested dictionary"""
        keys = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            if isinstance(value, dict):
                keys.extend(self._get_all_keys(value, full_key))
        return keys
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = key_path.split('.')
        value = data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None

# Convenience functions
def create_dashboard(config_path: Optional[str] = None) -> DashboardBridge:
    """
    Create a dashboard bridge instance
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        DashboardBridge instance
    """
    path = Path(config_path) if config_path else None
    return DashboardBridge(path)

def quick_status_display(status_data: Dict[str, Any]) -> None:
    """
    Quick status display function
    
    Args:
        status_data: Status data to display
    """
    dashboard = DashboardBridge()
    dashboard.update_status(status_data)
    dashboard.display_status()

def quick_profile_display(config_path: Optional[str] = None) -> None:
    """
    Quick profile display function
    
    Args:
        config_path: Optional path to configuration file
    """
    dashboard = create_dashboard(config_path)
    dashboard.display_profile()

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    dashboard = DashboardBridge()
    
    # Display profile
    dashboard.display_profile()
    
    # Update and display status
    example_status = {
        'tick_counter': 1234,
        'pattern_history_size': 500,
        'shell_history_size': 250,
        'active_strategy': 'momentum_cascade',
        'vault_locked': False,
        're_entry_trigger': True,
        'test_mode': True,
        'verbose_logging': False
    }
    
    dashboard.update_status(example_status)
    dashboard.display_status()
    
    # Export dashboard data
    export_path = dashboard.export_dashboard_json()
    print(f"\nDashboard data exported to: {export_path}") 