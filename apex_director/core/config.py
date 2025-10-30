"""
APEX DIRECTOR Configuration Management
Centralized configuration for all system components
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class BackendConfig:
    """Represents the configuration for a backend service.

    Attributes:
        name: The name of the backend.
        enabled: A boolean indicating whether the backend is enabled.
        priority: The priority of the backend in the fallback chain.
        timeout: The request timeout in seconds.
        max_retries: The maximum number of retries on failure.
        rate_limit: The rate limit in requests per minute.
        cost_per_image: The estimated cost per image.
        quality_level: The quality level of the backend (1-5).
        resolution_cap: A tuple representing the maximum resolution.
        capabilities: A list of the backend's capabilities.
    """
    name: str
    enabled: bool
    priority: int  # Lower number = higher priority
    timeout: int = 300  # seconds
    max_retries: int = 3
    rate_limit: int = 60  # requests per minute
    cost_per_image: float = 0.0
    quality_level: int = 1  # 1-5 scale
    resolution_cap: tuple = (1024, 1024)
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ["image_generation"]


@dataclass
class AssetConfig:
    """Represents the configuration for asset management.

    Attributes:
        base_dir: The base directory for storing assets.
        subdirs: A dictionary mapping asset types to subdirectories.
        max_file_size: The maximum allowed file size in bytes.
        supported_formats: A list of supported file formats.
        metadata_backup: A boolean indicating whether to create metadata
            backups.
        auto_cleanup: A boolean indicating whether to automatically clean up
            orphaned files.
    """
    base_dir: str = "assets"
    subdirs: Dict[str, str] = None
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    supported_formats: List[str] = None
    metadata_backup: bool = True
    auto_cleanup: bool = False
    
    def __post_init__(self):
        if self.subdirs is None:
            self.subdirs = {
                "images": "images",
                "metadata": "metadata",
                "cache": "cache",
                "exports": "exports"
            }
        if self.supported_formats is None:
            self.supported_formats = [".png", ".jpg", ".jpeg", ".webp", ".json"]


@dataclass
class OrchestratorConfig:
    """Represents the configuration for the main orchestrator.

    Attributes:
        max_concurrent_jobs: The maximum number of jobs to process
            concurrently.
        checkpoint_interval: The interval in seconds for automatic
            checkpointing.
        failure_threshold: The number of failures before a job is marked as
            failed.
        auto_retry: A boolean indicating whether to automatically retry
            failed jobs.
        retry_delay: The delay in seconds between retries.
        cleanup_on_exit: A boolean indicating whether to perform cleanup on
            exit.
        health_check_interval: The interval in seconds for system health
            checks.
    """
    max_concurrent_jobs: int = 5
    checkpoint_interval: int = 300  # seconds
    failure_threshold: int = 3
    auto_retry: bool = True
    retry_delay: int = 30  # seconds
    cleanup_on_exit: bool = False
    health_check_interval: int = 60  # seconds


@dataclass
class EstimatorConfig:
    """Represents the configuration for the cost and time estimator.

    Attributes:
        cache_estimates: A boolean indicating whether to cache estimates.
        estimate_tolerance: A tolerance factor to add to time estimates.
        historical_data_retention: The number of days to retain historical
            data.
        cost_buffer: A buffer to add to cost estimates.
    """
    cache_estimates: bool = True
    estimate_tolerance: float = 0.2  # 20% tolerance
    historical_data_retention: int = 30  # days
    cost_buffer: float = 0.1  # 10% buffer


class ConfigManager:
    """A centralized class for managing system configuration.

    This class handles loading, saving, and providing access to configuration
    settings for all components of the system.
    """
    
    DEFAULT_CONFIG = {
        "backends": {
            "nano_banana": {
                "name": "Nano Banana",
                "enabled": True,
                "priority": 1,
                "timeout": 180,
                "max_retries": 2,
                "rate_limit": 30,
                "cost_per_image": 0.01,
                "quality_level": 1,
                "resolution_cap": [512, 512],
                "capabilities": ["image_generation", "fast_generation"]
            },
            "imagen": {
                "name": "Google Imagen",
                "enabled": True,
                "priority": 2,
                "timeout": 300,
                "max_retries": 3,
                "rate_limit": 60,
                "cost_per_image": 0.05,
                "quality_level": 3,
                "resolution_cap": [1024, 1024],
                "capabilities": ["image_generation", "high_quality"]
            },
            "minimax": {
                "name": "MiniMax",
                "enabled": True,
                "priority": 3,
                "timeout": 240,
                "max_retries": 3,
                "rate_limit": 45,
                "cost_per_image": 0.03,
                "quality_level": 4,
                "resolution_cap": [1024, 1024],
                "capabilities": ["image_generation", "fast_generation", "high_quality"]
            },
            "sdxl": {
                "name": "Stable Diffusion XL",
                "enabled": True,
                "priority": 4,
                "timeout": 600,
                "max_retries": 2,
                "rate_limit": 20,
                "cost_per_image": 0.08,
                "quality_level": 5,
                "resolution_cap": [1024, 1024],
                "capabilities": ["image_generation", "ultra_quality", "detailed"]
            }
        },
        "assets": {
            "base_dir": "assets",
            "max_file_size": 52428800,
            "supported_formats": [".png", ".jpg", ".jpeg", ".webp", ".json"],
            "metadata_backup": True,
            "auto_cleanup": False
        },
        "orchestrator": {
            "max_concurrent_jobs": 5,
            "checkpoint_interval": 300,
            "failure_threshold": 3,
            "auto_retry": True,
            "retry_delay": 30,
            "cleanup_on_exit": False,
            "health_check_interval": 60
        },
        "estimator": {
            "cache_estimates": True,
            "estimate_tolerance": 0.2,
            "historical_data_retention": 30,
            "cost_buffer": 0.1
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initializes the ConfigManager.

        Args:
            config_path: An optional path to the configuration file.
        """
        self.config_path = config_path or "apex_config.json"
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Loads the configuration from a file or uses the default.

        Returns:
            A dictionary containing the configuration.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._deep_merge(self.DEFAULT_CONFIG, config)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration")
        return self.DEFAULT_CONFIG.copy()
    
    def _save_config(self):
        """Saves the current configuration to a file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save config to {self.config_path}: {e}")
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merges two dictionaries.

        Args:
            base: The base dictionary.
            update: The dictionary with updates.

        Returns:
            The merged dictionary.
        """
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _validate_config(self):
        """Validates the configuration structure."""
        required_sections = ['backends', 'assets', 'orchestrator', 'estimator']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate backend configurations
        if 'backends' in self._config:
            for backend_name, backend_config in self._config['backends'].items():
                try:
                    BackendConfig(**backend_config)
                except Exception as e:
                    raise ValueError(f"Invalid backend config for {backend_name}: {e}")
    
    def get_backend_configs(self) -> List[BackendConfig]:
        """Gets a list of backend configurations.

        Returns:
            A list of BackendConfig objects, sorted by priority.
        """
        configs = []
        for name, config in self._config['backends'].items():
            config['name'] = name  # Ensure name is set
            configs.append(BackendConfig(**config))
        return sorted(configs, key=lambda x: x.priority)
    
    def get_asset_config(self) -> AssetConfig:
        """Gets the asset management configuration.

        Returns:
            An AssetConfig object.
        """
        return AssetConfig(**self._config['assets'])
    
    def get_orchestrator_config(self) -> OrchestratorConfig:
        """Gets the orchestrator configuration.

        Returns:
            An OrchestratorConfig object.
        """
        return OrchestratorConfig(**self._config['orchestrator'])
    
    def get_estimator_config(self) -> EstimatorConfig:
        """Gets the estimator configuration.

        Returns:
            An EstimatorConfig object.
        """
        return EstimatorConfig(**self._config['estimator'])
    
    def update_backend_config(self, backend_name: str, updates: Dict[str, Any]):
        """Updates the configuration for a specific backend.

        Args:
            backend_name: The name of the backend to update.
            updates: A dictionary of configuration updates.
        """
        if backend_name not in self._config['backends']:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        self._config['backends'][backend_name].update(updates)
        self._validate_config()
        self._save_config()
    
    def enable_backend(self, backend_name: str, enabled: bool = True):
        """Enables or disables a backend.

        Args:
            backend_name: The name of the backend.
            enabled: True to enable the backend, False to disable it.
        """
        if backend_name in self._config['backends']:
            self._config['backends'][backend_name]['enabled'] = enabled
            self._save_config()
    
    def set_cost_for_backend(self, backend_name: str, cost_per_image: float):
        """Updates the cost per image for a specific backend.

        Args:
            backend_name: The name of the backend.
            cost_per_image: The new cost per image.
        """
        if backend_name in self._config['backends']:
            self._config['backends'][backend_name]['cost_per_image'] = cost_per_image
            self._save_config()
    
    def get_enabled_backends(self) -> List[str]:
        """Gets a list of the names of all enabled backends.

        Returns:
            A list of strings, where each string is the name of an enabled
            backend.
        """
        return [name for name, config in self._config['backends'].items() 
                if config.get('enabled', True)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Returns the entire configuration as a dictionary.

        Returns:
            A dictionary representing the current configuration.
        """
        return self._config.copy()
    
    def reset_to_defaults(self):
        """Resets the configuration to the default settings."""
        self._config = self.DEFAULT_CONFIG.copy()
        self._save_config()


# Global configuration instance
config_manager = ConfigManager()


# Convenience functions for accessing configuration
def get_config() -> ConfigManager:
    """Gets the global instance of the ConfigManager.

    Returns:
        The global ConfigManager instance.
    """
    return config_manager


def get_enabled_backends() -> List[str]:
    """A convenience function to get the names of all enabled backends.

    Returns:
        A list of strings, where each string is the name of an enabled
        backend.
    """
    return config_manager.get_enabled_backends()


def get_backend_config(backend_name: str) -> BackendConfig:
    """A convenience function to get the configuration for a specific backend.

    Args:
        backend_name: The name of the backend.

    Returns:
        A BackendConfig object.
    """
    backend_configs = config_manager.get_backend_configs()
    for config in backend_configs:
        if config.name == backend_name:
            return config
    raise ValueError(f"Backend not found: {backend_name}")


def get_asset_config() -> AssetConfig:
    """A convenience function to get the asset management configuration.

    Returns:
        An AssetConfig object.
    """
    return config_manager.get_asset_config()


def get_orchestrator_config() -> OrchestratorConfig:
    """A convenience function to get the orchestrator configuration.

    Returns:
        An OrchestratorConfig object.
    """
    return config_manager.get_orchestrator_config()


def get_estimator_config() -> EstimatorConfig:
    """A convenience function to get the estimator configuration.

    Returns:
        An EstimatorConfig object.
    """
    return config_manager.get_estimator_config()