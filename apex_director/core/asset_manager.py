"""
APEX DIRECTOR Asset Management System
Handles file organization, metadata management, and asset lifecycle
"""

import os
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid
from dataclasses import dataclass, asdict
import logging

from .config import get_asset_config

logger = logging.getLogger(__name__)


@dataclass
class AssetMetadata:
    """Complete metadata for an asset"""
    id: str
    filename: str
    file_path: str
    file_size: int
    file_hash: str
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    created_at: datetime = None
    job_id: Optional[str] = None
    prompt: Optional[str] = None
    backend_used: Optional[str] = None
    generation_params: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    quality_score: Optional[float] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None
    parent_id: Optional[str] = None  # For variants/derivatives
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.tags is None:
            self.tags = []
        if self.generation_params is None:
            self.generation_params = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if isinstance(self.created_at, datetime):
            data['created_at'] = self.created_at.isoformat()
        if isinstance(self.last_used, datetime):
            data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssetMetadata':
        """Create from dictionary"""
        # Handle datetime conversion
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_used' in data and isinstance(data['last_used'], str):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


@dataclass
class ProjectInfo:
    """Project-level metadata and settings"""
    id: str
    name: str
    description: str = ""
    created_at: datetime = None
    updated_at: datetime = None
    status: str = "active"
    total_jobs: int = 0
    completed_jobs: int = 0
    total_cost: float = 0.0
    settings: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.settings is None:
            self.settings = {}


class AssetManager:
    """Central asset management system"""
    
    def __init__(self, custom_config: Optional[Any] = None):
        self.config = custom_config or get_asset_config()
        self.base_dir = Path(self.config.base_dir)
        self.metadata_db = self.base_dir / "metadata" / "assets.json"
        self.projects_db = self.base_dir / "metadata" / "projects.json"
        self.assets_index: Dict[str, AssetMetadata] = {}
        self.projects_index: Dict[str, ProjectInfo] = {}
        
        # Initialize directory structure
        self._initialize_directories()
        
        # Load existing metadata
        self._load_metadata()
    
    def _initialize_directories(self):
        """Create required directory structure"""
        directories = [
            self.base_dir,
            self.base_dir / "images",
            self.base_dir / "metadata",
            self.base_dir / "cache",
            self.base_dir / "exports",
            self.base_dir / "projects",
            self.base_dir / "thumbnails",
            self.base_dir / "variants"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files in empty directories
            if not any(directory.iterdir()):
                (directory / ".gitkeep").touch()
    
    def _load_metadata(self):
        """Load asset and project metadata from disk"""
        try:
            if self.metadata_db.exists():
                with open(self.metadata_db, 'r') as f:
                    data = json.load(f)
                    self.assets_index = {
                        asset_id: AssetMetadata.from_dict(asset_data)
                        for asset_id, asset_data in data.items()
                    }
                logger.info(f"Loaded metadata for {len(self.assets_index)} assets")
            
            if self.projects_db.exists():
                with open(self.projects_db, 'r') as f:
                    data = json.load(f)
                    self.projects_index = {
                        project_id: ProjectInfo(**project_data)
                        for project_id, project_data in data.items()
                    }
                logger.info(f"Loaded metadata for {len(self.projects_index)} projects")
        
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save asset and project metadata to disk"""
        try:
            # Save assets
            assets_data = {
                asset_id: asset.to_dict()
                for asset_id, asset in self.assets_index.items()
            }
            
            with open(self.metadata_db, 'w') as f:
                json.dump(assets_data, f, indent=2)
            
            # Save projects
            projects_data = {
                project_id: project.__dict__
                for project_id, project in self.projects_index.items()
            }
            
            with open(self.projects_db, 'w') as f:
                json.dump(projects_data, f, indent=2)
            
            # Create backup if enabled
            if self.config.metadata_backup:
                self._create_metadata_backup()
                
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _create_metadata_backup(self):
        """Create backup of metadata files"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Backup assets metadata
            if self.metadata_db.exists():
                backup_path = self.metadata_db.parent / f"assets_backup_{timestamp}.json"
                shutil.copy2(self.metadata_db, backup_path)
            
            # Backup projects metadata
            if self.projects_db.exists():
                backup_path = self.projects_db.parent / f"projects_backup_{timestamp}.json"
                shutil.copy2(self.projects_db, backup_path)
                
        except Exception as e:
            logger.warning(f"Failed to create metadata backup: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def _get_unique_filename(self, original_filename: str, subdir: str = "images") -> Tuple[str, Path]:
        """Generate unique filename to avoid conflicts"""
        base_name = Path(original_filename).stem
        extension = Path(original_filename).suffix
        counter = 1
        
        while True:
            if counter == 1:
                filename = f"{base_name}{extension}"
            else:
                filename = f"{base_name}_{counter}{extension}"
            
            file_path = self.base_dir / subdir / filename
            
            if not file_path.exists():
                return filename, file_path
            
            counter += 1
    
    def register_asset(self, 
                      file_path: str,
                      asset_type: str = "image",
                      job_id: Optional[str] = None,
                      prompt: Optional[str] = None,
                      backend_used: Optional[str] = None,
                      generation_params: Optional[Dict[str, Any]] = None,
                      tags: Optional[List[str]] = None) -> str:
        """
        Register a new asset in the system
        
        Returns:
            Asset ID
        """
        try:
            file_path = Path(file_path)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Check for duplicates
            for existing_asset in self.assets_index.values():
                if existing_asset.file_hash == file_hash:
                    logger.info(f"Duplicate asset detected: {existing_asset.id}")
                    return existing_asset.id
            
            # Generate unique filename
            filename, unique_path = self._get_unique_filename(file_path.name, asset_type)
            
            # Copy file to organized location
            if file_path != unique_path:
                shutil.copy2(file_path, unique_path)
            
            # Create metadata
            asset_id = str(uuid.uuid4())
            metadata = AssetMetadata(
                id=asset_id,
                filename=filename,
                file_path=str(unique_path),
                file_size=unique_path.stat().st_size,
                file_hash=file_hash,
                format=unique_path.suffix.lower(),
                job_id=job_id,
                prompt=prompt,
                backend_used=backend_used,
                generation_params=generation_params or {},
                tags=tags or []
            )
            
            # Add to index
            self.assets_index[asset_id] = metadata
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Registered asset: {asset_id} ({filename})")
            return asset_id
            
        except Exception as e:
            logger.error(f"Failed to register asset: {e}")
            raise
    
    def get_asset(self, asset_id: str) -> Optional[AssetMetadata]:
        """Retrieve asset metadata"""
        return self.assets_index.get(asset_id)
    
    def get_asset_path(self, asset_id: str) -> Optional[Path]:
        """Get physical file path for asset"""
        asset = self.get_asset(asset_id)
        return Path(asset.file_path) if asset else None
    
    def search_assets(self, 
                     query: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     backend: Optional[str] = None,
                     format_filter: Optional[List[str]] = None,
                     date_from: Optional[datetime] = None,
                     date_to: Optional[datetime] = None,
                     limit: int = 100) -> List[AssetMetadata]:
        """
        Search assets by various criteria
        
        Args:
            query: Text search in filename, prompt, and tags
            tags: Filter by tags (AND operation)
            backend: Filter by backend used
            format_filter: Filter by file formats
            date_from: Filter from date
            date_to: Filter to date
            limit: Maximum results
            
        Returns:
            List of matching assets
        """
        results = []
        
        for asset in self.assets_index.values():
            # Text query filter
            if query:
                search_text = f"{asset.filename} {asset.prompt or ''} {' '.join(asset.tags)}".lower()
                if query.lower() not in search_text:
                    continue
            
            # Tags filter
            if tags:
                if not all(tag in asset.tags for tag in tags):
                    continue
            
            # Backend filter
            if backend and asset.backend_used != backend:
                continue
            
            # Format filter
            if format_filter and asset.format not in format_filter:
                continue
            
            # Date filters
            if date_from and asset.created_at < date_from:
                continue
            if date_to and asset.created_at > date_to:
                continue
            
            results.append(asset)
            
            if len(results) >= limit:
                break
        
        return results
    
    def update_asset(self, asset_id: str, updates: Dict[str, Any]) -> bool:
        """Update asset metadata"""
        if asset_id not in self.assets_index:
            return False
        
        asset = self.assets_index[asset_id]
        
        # Handle datetime fields
        for key, value in updates.items():
            if key in ['created_at', 'last_used'] and isinstance(value, str):
                updates[key] = datetime.fromisoformat(value)
        
        for key, value in updates.items():
            setattr(asset, key, value)
        
        self._save_metadata()
        logger.info(f"Updated asset metadata: {asset_id}")
        return True
    
    def delete_asset(self, asset_id: str, remove_file: bool = True) -> bool:
        """Delete asset and optionally remove physical file"""
        if asset_id not in self.assets_index:
            return False
        
        asset = self.assets_index[asset_id]
        
        try:
            # Remove file if requested
            if remove_file:
                file_path = Path(asset.file_path)
                if file_path.exists():
                    file_path.unlink()
            
            # Remove from index
            del self.assets_index[asset_id]
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Deleted asset: {asset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete asset {asset_id}: {e}")
            return False
    
    def create_project(self, name: str, description: str = "", settings: Optional[Dict[str, Any]] = None) -> str:
        """Create a new project"""
        project_id = str(uuid.uuid4())
        
        project = ProjectInfo(
            id=project_id,
            name=name,
            description=description,
            settings=settings or {}
        )
        
        self.projects_index[project_id] = project
        self._save_metadata()
        
        # Create project directory
        project_dir = self.base_dir / "projects" / name
        project_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created project: {project_id} ({name})")
        return project_id
    
    def get_project(self, project_id: str) -> Optional[ProjectInfo]:
        """Get project information"""
        return self.projects_index.get(project_id)
    
    def get_assets_by_project(self, project_name: str) -> List[AssetMetadata]:
        """Get all assets for a project"""
        # This would need project tags in assets for proper implementation
        return [asset for asset in self.assets_index.values() 
                if project_name in asset.tags]
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_files = len(self.assets_index)
        total_size = sum(asset.file_size for asset in self.assets_index.values())
        format_breakdown = {}
        
        for asset in self.assets_index.values():
            fmt = asset.format
            if fmt not in format_breakdown:
                format_breakdown[fmt] = {"count": 0, "size": 0}
            format_breakdown[fmt]["count"] += 1
            format_breakdown[fmt]["size"] += asset.file_size
        
        # Calculate directory sizes
        dir_stats = {}
        for subdir in ["images", "thumbnails", "variants"]:
            dir_path = self.base_dir / subdir
            if dir_path.exists():
                total_size_bytes = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                file_count = len([f for f in dir_path.rglob('*') if f.is_file()])
                dir_stats[subdir] = {
                    "file_count": file_count,
                    "size_mb": total_size_bytes / (1024 * 1024)
                }
        
        return {
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "format_breakdown": format_breakdown,
            "directory_stats": dir_stats,
            "projects_count": len(self.projects_index)
        }
    
    def cleanup_orphaned_files(self) -> Dict[str, int]:
        """Clean up files not referenced in metadata"""
        removed = {"files": 0, "size_mb": 0}
        
        try:
            # Get all files referenced in metadata
            referenced_files = {asset.file_path for asset in self.assets_index.values()}
            
            # Check files in asset directories
            for subdir in ["images", "thumbnails", "variants"]:
                dir_path = self.base_dir / subdir
                if not dir_path.exists():
                    continue
                
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file() and file_path.name != ".gitkeep":
                        if str(file_path) not in referenced_files:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            removed["files"] += 1
                            removed["size_mb"] += file_size / (1024 * 1024)
                            logger.info(f"Removed orphaned file: {file_path}")
            
            return removed
            
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")
            return removed
    
    def export_metadata(self, output_path: str, format: str = "json") -> bool:
        """Export asset metadata to file"""
        try:
            output_path = Path(output_path)
            
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump({
                        asset_id: asset.to_dict()
                        for asset_id, asset in self.assets_index.items()
                    }, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported metadata to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metadata: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on asset management system"""
        issues = []
        
        try:
            # Check metadata files
            if not self.metadata_db.exists():
                issues.append("Assets metadata file missing")
            if not self.projects_db.exists():
                issues.append("Projects metadata file missing")
            
            # Check directory structure
            required_dirs = ["images", "metadata", "cache", "exports"]
            for subdir in required_dirs:
                dir_path = self.base_dir / subdir
                if not dir_path.exists():
                    issues.append(f"Directory missing: {subdir}")
            
            # Check for inconsistencies
            orphaned_files = 0
            missing_files = 0
            
            for asset in self.assets_index.values():
                if not Path(asset.file_path).exists():
                    missing_files += 1
            
            return {
                "status": "healthy" if not issues else "degraded",
                "issues": issues,
                "stats": self.get_storage_stats(),
                "inconsistencies": {
                    "missing_files": missing_files,
                    "orphaned_files": orphaned_files
                }
            }
            
        except Exception as e:
            return {
                "status": "critical",
                "issues": [f"Health check failed: {e}"],
                "stats": {},
                "inconsistencies": {}
            }


# Global asset manager instance
_asset_manager: Optional[AssetManager] = None


def get_asset_manager() -> AssetManager:
    """Get the global asset manager instance"""
    global _asset_manager
    if _asset_manager is None:
        _asset_manager = AssetManager()
    return _asset_manager