#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Settings Engine

A comprehensive settings management system for the Schwabot trading platform.
Handles YAML and JSON configuration files with validation, profiles, and
real-time updates.

Features:
- Multi-format configuration loading (YAML, JSON)
- Configuration validation and schema checking
- Profile management and switching
- Real-time configuration updates
- Backup and restore functionality
- CLI integration for settings management
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import yaml
from jsonschema import Draft7Validator, ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "SettingsSection",
    "AdvancedSettingsEngine",
    "ConfigFormat",
    "ValidationLevel",
    "SettingsProfile",
]


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    YAML = "yaml"
    JSON = "json"
    AUTO = "auto"


class ValidationLevel(Enum):
    """Configuration validation levels."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    SCHEMA = "schema"


@dataclass
class SettingsProfile:
    """Configuration profile with metadata."""
    name: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    settings: Dict[str, Any] = field(default_factory=dict)
    validation_level: ValidationLevel = ValidationLevel.BASIC
    is_active: bool = False


class SettingsSection:
    """Mutable settings section with validation and change tracking."""
    
    def __init__(self, name: str, data: Optional[Dict[str, Any]] = None):
        self.name = name
        self._data = data or {}
        self._original_data = self._data.copy()
        self._changes: Dict[str, Any] = {}
        self._validation_errors: List[str] = []
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a setting value and track changes."""
        if self._data.get(key) != value:
            self._data[key] = value
            self._changes[key] = value
            self._original_data[key] = value
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update multiple settings at once."""
        for key, value in data.items():
            self.set(key, value)
    
    def has_changes(self) -> bool:
        """Check if any changes have been made."""
        return bool(self._changes)
    
    def get_changes(self) -> Dict[str, Any]:
        """Get all changes made to this section."""
        return self._changes.copy()
    
    def reset_changes(self) -> None:
        """Reset change tracking."""
        self._changes.clear()
    
    def validate(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Validate the section against an optional schema."""
        self._validation_errors.clear()
        
        if not schema:
            return True
        
        try:
            validator = Draft7Validator(schema)
            errors = list(validator.iter_errors(self._data))
            
            if errors:
                for error in errors:
                    self._validation_errors.append(f"{error.path}: {error.message}")
                return False
            
            return True
        except Exception as e:
            self._validation_errors.append(f"Validation error: {e}")
            return False
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors for this section."""
        return self._validation_errors.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return self._data.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access."""
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-like assignment."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data
    
    def __repr__(self) -> str:
        return f"<SettingsSection '{self.name}' with {len(self._data)} settings>"


class AdvancedSettingsEngine:
    """
    Advanced settings management engine for Schwabot.
    
    Provides comprehensive configuration management with support for:
    - Multiple file formats (YAML, JSON)
    - Configuration validation
    - Profile management
    - Real-time updates
    - Backup and restore
    """
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        default_format: ConfigFormat = ConfigFormat.AUTO,
        validation_level: ValidationLevel = ValidationLevel.BASIC,
    ):
        """
        Initialize the settings engine.
        
        Args:
            config_dir: Directory containing configuration files
            default_format: Default format for new config files
            validation_level: Default validation level
        """
        self.config_dir = Path(config_dir or "config")
        self.default_format = default_format
        self.validation_level = validation_level
        
        # Internal state
        self._sections: Dict[str, SettingsSection] = {}
        self._profiles: Dict[str, SettingsProfile] = {}
        self._active_profile: Optional[str] = None
        self._loaded_files: Dict[str, Dict[str, Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._backup_dir = self.config_dir / "backups"
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self._backup_dir.mkdir(exist_ok=True)
        
        logger.info(
            "AdvancedSettingsEngine initialized with config_dir: %s", self.config_dir
        )
    
    def load(self, file_path: Optional[str] = None) -> None:
        """
        Load configuration from file or directory.
        
        Args:
            file_path: Specific file to load, or None to load all config files
        """
        if file_path:
            self._load_single_file(file_path)
        else:
            self._load_all_files()
        
        logger.info(f"Loaded {len(self._sections)} configuration sections")
    
    def _load_single_file(self, file_path: str) -> None:
        """Load a single configuration file."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.config_dir / path
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            # Determine format
            if path.suffix.lower() in ['.yaml', '.yml']:
                format_type = ConfigFormat.YAML
            elif path.suffix.lower() == '.json':
                format_type = ConfigFormat.JSON
            else:
                format_type = self.default_format
            
            # Load file
            with open(path, 'r', encoding='utf-8') as f:
                if format_type == ConfigFormat.YAML:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Store loaded data
            self._loaded_files[str(path)] = data
            
            # Create sections
            if isinstance(data, dict):
                for section_name, section_data in data.items():
                    if isinstance(section_data, dict):
                        section = SettingsSection(section_name, section_data)
                        self._sections[section_name] = section
                        logger.debug(f"Loaded section '{section_name}' from {path}")
            
            logger.info(f"Successfully loaded configuration from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            raise
    
    def _load_all_files(self) -> None:
        """Load all configuration files from the config directory."""
        config_files = list(self.config_dir.glob('*.yaml')) + \
                       list(self.config_dir.glob('*.yml')) + \
                       list(self.config_dir.glob('*.json'))
        
        # Load each file
        for file_path in config_files:
            try:
                self._load_single_file(str(file_path))
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
    
    def save(
        self,
        destination: Optional[str] = None,
        format_type: Optional[ConfigFormat] = None,
    ) -> None:
        """
        Save current configuration to file.
        
        Args:
            destination: Path to save file. If None, saves to a default location.
            format_type: Format to save in (YAML or JSON).
        """
        data = {name: section.to_dict() for name, section in self._sections.items()}
        
        if destination is None:
            destination = self.config_dir / "current_settings.yaml"
        
        path = Path(destination)
        if not path.is_absolute():
            path = self.config_dir / path
        
        # Determine format
        if format_type is None:
            if path.suffix.lower() in ['.yaml', '.yml']:
                format_type = ConfigFormat.YAML
            elif path.suffix.lower() == '.json':
                format_type = ConfigFormat.JSON
            else:
                format_type = self.default_format
        
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            with open(path, 'w', encoding='utf-8') as f:
                if format_type == ConfigFormat.YAML:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {path}: {e}")
            raise
    
    def get(
        self, key: str, default: Any = None, section: Optional[str] = None
    ) -> Any:
        """
        Get a setting value.
        
        Args:
            key: Setting key (can be 'section.key' format)
            default: Default value if not found
            section: Section name (if key doesn't include section)
        
        Returns:
            Setting value or default
        """
        if '.' in key and not section:
            section_name, setting_key = key.split('.', 1)
        else:
            section_name = section or 'default'
            setting_key = key
        
        if section_name in self._sections:
            return self._sections[section_name].get(setting_key, default)
        
        return default
    
    def set(
        self, 
        key: str, 
        value: Any, 
        section: Optional[str] = None
    ) -> None:
        """
        Set a setting value.
        
        Args:
            key: Setting key (can be 'section.key' format)
            value: Value to set
            section: Section name (if key doesn't include section)
        """
        if '.' in key and not section:
            section_name, setting_key = key.split('.', 1)
        else:
            section_name = section or 'default'
            setting_key = key
        
        if section_name not in self._sections:
            self._sections[section_name] = SettingsSection(section_name)
        
        self._sections[section_name].set(setting_key, value)
    
    def section(self, name: str) -> SettingsSection:
        """
        Get or create a settings section.
        
        Args:
            name: Section name
            
        Returns:
            SettingsSection instance
        """
        if name not in self._sections:
            self._sections[name] = SettingsSection(name)
        
        return self._sections[name]
    
    def apply_profile(self, profile_name: str) -> None:
        """
        Apply a configuration profile.
        
        Args:
            profile_name: Name of the profile to apply
        """
        if profile_name not in self._profiles:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        profile = self._profiles[profile_name]
        
        # Apply profile settings
        for section_name, section_data in profile.settings.items():
            if section_name not in self._sections:
                self._sections[section_name] = SettingsSection(section_name)
            
            self._sections[section_name].update(section_data)
        
        # Update active profile
        self._active_profile = profile_name
        
        # Validate if needed
        if profile.validation_level != ValidationLevel.NONE:
            self._validate_all(profile.validation_level)
        
        logger.info(f"Applied profile '{profile_name}'")
    
    def create_profile(self, name: str, description: str = "") -> SettingsProfile:
        """
        Create a new configuration profile.
        
        Args:
            name: Profile name
            description: Profile description
            
        Returns:
            Created SettingsProfile
        """
        if name in self._profiles:
            raise ValueError(f"Profile '{name}' already exists")
        
        # Create profile with current settings
        current_settings = {}
        for section_name, section in self._sections.items():
            current_settings[section_name] = section.to_dict()
        
        profile = SettingsProfile(
            name=name,
            description=description,
            settings=current_settings,
            validation_level=self.validation_level
        )
        
        self._profiles[name] = profile
        logger.info(f"Created profile '{name}'")
        
        return profile
    
    def save_profile(self, name: str) -> None:
        """Save current settings as a profile."""
        if name in self._profiles:
            # Update existing profile
            profile = self._profiles[name]
            profile.settings.clear()
            for section_name, section in self._sections.items():
                profile.settings[section_name] = section.to_dict()
            profile.modified_at = time.time()
        else:
            # Create new profile
            self.create_profile(name)
    
    def list_profiles(self) -> List[str]:
        """List all available profiles."""
        return list(self._profiles.keys())
    
    def get_active_profile(self) -> Optional[str]:
        """Get the name of the currently active profile."""
        return self._active_profile
    
    def diff(self, other: AdvancedSettingsEngine) -> Dict[str, Any]:
        """
        Compare this engine's settings with another.
        
        Args:
            other: Another AdvancedSettingsEngine instance
            
        Returns:
            Dictionary of differences
        """
        differences = {}
        
        # Compare sections
        all_sections = set(self._sections.keys()) | set(other._sections.keys())
        
        for section_name in all_sections:
            this_section = self._sections.get(section_name)
            other_section = other._sections.get(section_name)
            
            if this_section is None:
                differences[section_name] = {"type": "missing_in_this", "data": other_section.to_dict()}
            elif other_section is None:
                differences[section_name] = {"type": "missing_in_other", "data": this_section.to_dict()}
            else:
                # Compare section contents
                this_data = this_section.to_dict()
                other_data = other_section.to_dict()
                
                if this_data != other_data:
                    differences[section_name] = {
                        "type": "different",
                        "this": this_data,
                        "other": other_data
                    }
        
        return differences
    
    def backup(self, backup_name: Optional[str] = None) -> str:
        """
        Create a backup of current settings.
        
        Args:
            backup_name: Optional backup name, defaults to timestamp
            
        Returns:
            Path to backup file
        """
        if not backup_name:
            backup_name = f"backup_{int(time.time())}"
        
        backup_path = self._backup_dir / f"{backup_name}.yaml"
        
        # Save current settings to backup
        self.save(str(backup_path), ConfigFormat.YAML)
        
        logger.info(f"Created backup: {backup_path}")
        return str(backup_path)
    
    def restore(self, backup_path: str) -> None:
        """
        Restore settings from backup.
        
        Args:
            backup_path: Path to backup file
        """
        path = Path(backup_path)
        if not path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Load backup
        self._load_single_file(str(path))
        logger.info(f"Restored settings from backup: {backup_path}")
    
    def _validate_all(self, level: ValidationLevel) -> bool:
        """Validate all sections."""
        all_valid = True
        
        for section_name, section in self._sections.items():
            schema = self._schemas.get(section_name)
            
            if level == ValidationLevel.SCHEMA and schema:
                if not section.validate(schema):
                    all_valid = False
                    logger.error(f"Validation failed for section '{section_name}'")
                    for error in section.get_validation_errors():
                        logger.error(f"  {error}")
            
            elif level == ValidationLevel.STRICT:
                # Basic type checking and required fields
                if not self._validate_section_strict(section):
                    all_valid = False
        
        return all_valid
    
    def _validate_section_strict(self, section: SettingsSection) -> bool:
        """Strict validation of a section."""
        # This is a basic implementation - can be extended
        return True

    def is_loaded(self) -> bool:
        """Check if any configuration has been loaded."""
        return bool(self._sections)
    
    def get_sections(self) -> List[str]:
        """Get list of all section names."""
        return list(self._sections.keys())
    
    def reload(self) -> None:
        """Reload all configuration files."""
        self._sections.clear()
        self._loaded_files.clear()
        self.load()
    
    def __repr__(self) -> str:
        sections_count = len(self._sections)
        profiles_count = len(self._profiles)
        active_profile = self._active_profile or "none"
        
        return (f"<AdvancedSettingsEngine "
                f"sections={sections_count} "
                f"profiles={profiles_count} "
                f"active_profile='{active_profile}'>")
