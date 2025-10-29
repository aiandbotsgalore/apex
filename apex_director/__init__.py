"""
APEX DIRECTOR - Core System Architecture & Backend Abstraction

A comprehensive image generation orchestration system with:
- Multi-backend support with automatic fallback
- Asset management and organization  
- Checkpoint and resume capabilities
- Cost and time estimation
- Professional error handling and logging

Author: APEX DIRECTOR Team
Version: 1.0.0
"""

from .core import *

__version__ = "1.0.0"
__author__ = "APEX DIRECTOR Team"

# Package metadata
__all__ = [
    "APEXOrchestrator",
    "BackendManager", 
    "AssetManager",
    "Estimator",
    "CheckpointManager",
    "ConfigManager",
    "audio"
]

# Package-level convenience functions
def get_orchestrator():
    """Get the main orchestrator instance"""
    from .core import get_orchestrator
    return get_orchestrator()

def get_backend_manager():
    """Get the backend manager instance"""  
    from .core import get_backend_manager
    return get_backend_manager()

def get_asset_manager():
    """Get the asset manager instance"""
    from .core import get_asset_manager
    return get_asset_manager()

def get_estimator():
    """Get the estimation engine instance"""
    from .core import get_estimator
    return get_estimator()

def get_checkpoint_manager():
    """Get the checkpoint manager instance"""
    from .core import get_checkpoint_manager
    return get_checkpoint_manager()

def get_config():
    """Get the configuration manager instance"""
    from .core import get_config
    return get_config()

# Initialize package
def initialize():
    """Initialize the APEX DIRECTOR system"""
    # Create necessary directories
    import os
    from pathlib import Path
    
    directories = [
        "assets",
        "assets/images", 
        "assets/metadata",
        "assets/cache",
        "assets/exports",
        "assets/projects",
        "assets/thumbnails",
        "assets/variants",
        "assets/checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("APEX DIRECTOR initialized successfully")

# Auto-initialize when imported
initialize()