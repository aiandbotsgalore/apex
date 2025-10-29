"""
Unit tests for APEX DIRECTOR Asset Manager

Tests asset organization, metadata management, and storage utilities.
"""

import pytest
import json
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil
import time

from apex_director.core.asset_manager import AssetManager, Project


class TestAssetManager:
    """Test suite for AssetManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def asset_manager(self, temp_dir):
        """Create asset manager instance"""
        return AssetManager(base_dir=temp_dir)
    
    def test_initialization(self, asset_manager, temp_dir):
        """Test asset manager initialization"""
        assert asset_manager.base_dir == temp_dir
        assert (temp_dir / "images").exists()
        assert (temp_dir / "metadata").exists()
        assert (temp_dir / "projects").exists()
    
    def test_create_project(self, asset_manager):
        """Test project creation"""
        project = asset_manager.create_project(
            name="Test Project",
            description="A test project"
        )
        
        assert project.name == "Test Project"
        assert project.description == "A test project"
        assert project.id is not None
        
        # Check project metadata file
        metadata_file = Path(project.path) / "project.json"
        assert metadata_file.exists()
        
        with open(metadata_file) as f:
            metadata = json.load(f)
            assert metadata["name"] == "Test Project"
            assert metadata["description"] == "A test project"
    
    def test_get_project(self, asset_manager):
        """Test project retrieval"""
        # Create a project first
        project = asset_manager.create_project(
            name="Test Project",
            description="A test project"
        )
        
        # Retrieve the project
        retrieved_project = asset_manager.get_project(project.id)
        assert retrieved_project.id == project.id
        assert retrieved_project.name == project.name
    
    def test_list_projects(self, asset_manager):
        """Test project listing"""
        # Create multiple projects
        project1 = asset_manager.create_project("Project 1", "First project")
        project2 = asset_manager.create_project("Project 2", "Second project")
        project3 = asset_manager.create_project("Project 3", "Third project")
        
        projects = asset_manager.list_projects()
        assert len(projects) == 3
        
        project_ids = [p.id for p in projects]
        assert project1.id in project_ids
        assert project2.id in project_ids
        assert project3.id in project_ids
    
    def test_store_asset(self, asset_manager):
        """Test asset storage"""
        # Create a test asset
        test_content = b"This is a test image content"
        asset_data = {
            "content": test_content,
            "filename": "test.png",
            "content_type": "image/png",
            "metadata": {"source": "test"}
        }
        
        asset_path = asset_manager.store_asset(
            asset_data=asset_data,
            project_id=None
        )
        
        assert asset_path.exists()
        assert asset_path.name == "test.png"
        
        # Verify file content
        with open(asset_path, 'rb') as f:
            stored_content = f.read()
            assert stored_content == test_content
    
    def test_store_asset_with_hash(self, asset_manager):
        """Test asset storage with content hash"""
        test_content = b"Test content for hashing"
        asset_data = {
            "content": test_content,
            "filename": "hashed.png"
        }
        
        asset_path = asset_manager.store_asset(asset_data=asset_data)
        
        # Check that hash directory was created
        content_hash = hashlib.sha256(test_content).hexdigest()
        assert content_hash in str(asset_path)
    
    def test_get_asset_metadata(self, asset_manager):
        """Test asset metadata retrieval"""
        # Store an asset
        test_content = b"Test content"
        asset_data = {
            "content": test_content,
            "filename": "metadata_test.png",
            "metadata": {"test_key": "test_value"}
        }
        
        asset_path = asset_manager.store_asset(asset_data=asset_data)
        
        # Retrieve metadata
        metadata = asset_manager.get_asset_metadata(asset_path)
        assert metadata["filename"] == "metadata_test.png"
        assert metadata["test_key"] == "test_value"
    
    def test_search_assets(self, asset_manager):
        """Test asset search functionality"""
        # Store multiple assets with different properties
        assets = [
            {
                "content": b"Image 1",
                "filename": "landscape.png",
                "metadata": {"category": "nature", "tags": ["landscape", "mountains"]}
            },
            {
                "content": b"Image 2",
                "filename": "portrait.png", 
                "metadata": {"category": "people", "tags": ["portrait", "face"]}
            },
            {
                "content": b"Image 3",
                "filename": "cityscape.png",
                "metadata": {"category": "city", "tags": ["urban", "buildings"]}
            }
        ]
        
        for asset_data in assets:
            asset_manager.store_asset(asset_data=asset_data)
        
        # Search by category
        nature_assets = asset_manager.search_assets(category="nature")
        assert len(nature_assets) == 1
        assert "landscape" in nature_assets[0]
        
        # Search by tags
        portrait_assets = asset_manager.search_assets(tags=["portrait"])
        assert len(portrait_assets) == 1
        assert "portrait" in portrait_assets[0]
        
        # Search by filename pattern
        landscape_assets = asset_manager.search_assets(filename_pattern="land*")
        assert len(landscape_assets) == 1
    
    def test_find_duplicates(self, asset_manager):
        """Test duplicate asset detection"""
        # Store duplicate content
        duplicate_content = b"Duplicate content"
        
        asset1_data = {
            "content": duplicate_content,
            "filename": "duplicate1.png"
        }
        
        asset2_data = {
            "content": duplicate_content,
            "filename": "duplicate2.png"
        }
        
        path1 = asset_manager.store_asset(asset_data=asset1_data)
        path2 = asset_manager.store_asset(asset_data=asset2_data)
        
        duplicates = asset_manager.find_duplicates()
        
        # Should find one duplicate pair
        assert len(duplicates) >= 1
    
    def test_get_storage_statistics(self, asset_manager):
        """Test storage statistics"""
        # Store some assets
        for i in range(5):
            content = f"Content {i}".encode()
            asset_data = {
                "content": content,
                "filename": f"file_{i}.txt"
            }
            asset_manager.store_asset(asset_data=asset_data)
        
        stats = asset_manager.get_storage_statistics()
        
        assert "total_files" in stats
        assert "total_size_bytes" in stats
        assert "total_size_mb" in stats
        assert "directory_breakdown" in stats
        
        assert stats["total_files"] == 5
        assert stats["total_size_bytes"] > 0
    
    def test_cleanup_temp_files(self, asset_manager):
        """Test temporary file cleanup"""
        # Create temporary files
        temp_dir = asset_manager.temp_dir
        temp_file1 = temp_dir / "temp1.tmp"
        temp_file2 = temp_dir / "temp2.tmp"
        
        temp_file1.write_text("temp content 1")
        temp_file2.write_text("temp content 2")
        
        # Verify files exist
        assert temp_file1.exists()
        assert temp_file2.exists()
        
        # Cleanup old files (create them with old timestamp)
        old_time = time.time() - (25 * 60 * 60)  # 25 hours ago
        temp_file1.touch(old_time)
        temp_file2.touch(old_time)
        
        # Run cleanup
        cleaned = asset_manager.cleanup_temp_files(max_age_hours=24)
        
        # Files should be cleaned
        assert cleaned >= 2
    
    def test_organize_assets(self, asset_manager):
        """Test asset organization"""
        # Store assets with different content types
        assets = [
            {"content": b"png content", "filename": "image.png", "content_type": "image/png"},
            {"content": b"jpg content", "filename": "photo.jpg", "content_type": "image/jpeg"},
            {"content": b"json content", "filename": "data.json", "content_type": "application/json"}
        ]
        
        asset_paths = []
        for asset_data in assets:
            path = asset_manager.store_asset(asset_data=asset_data)
            asset_paths.append(path)
        
        # Organize by content type
        organized = asset_manager.organize_assets(organized_paths)
        
        assert organized is True
        
        # Check that files are organized
        image_dir = asset_manager.base_dir / "images"
        assert image_dir.exists()
    
    def test_update_asset_metadata(self, asset_manager):
        """Test asset metadata updates"""
        # Store an asset
        asset_data = {
            "content": b"Test content",
            "filename": "update_test.png"
        }
        
        asset_path = asset_manager.store_asset(asset_data=asset_data)
        
        # Update metadata
        updates = {
            "description": "Updated description",
            "version": "1.1",
            "custom_field": "custom_value"
        }
        
        asset_manager.update_asset_metadata(asset_path, updates)
        
        # Verify updates
        metadata = asset_manager.get_asset_metadata(asset_path)
        assert metadata["description"] == "Updated description"
        assert metadata["version"] == "1.1"
        assert metadata["custom_field"] == "custom_value"
    
    def test_backup_project(self, asset_manager):
        """Test project backup"""
        # Create a project with assets
        project = asset_manager.create_project("Backup Project", "Test backup")
        
        # Add some assets to project
        asset_data = {
            "content": b"Project asset",
            "filename": "project_asset.png"
        }
        
        asset_manager.store_asset(asset_data=asset_data, project_id=project.id)
        
        # Create backup
        backup_path = asset_manager.backup_project(project.id)
        
        assert backup_path.exists()
        assert backup_path.name.startswith("backup_")
    
    def test_export_project(self, asset_manager):
        """Test project export"""
        # Create a project with assets
        project = asset_manager.create_project("Export Project", "Test export")
        
        # Add assets
        for i in range(3):
            asset_data = {
                "content": f"Asset {i}".encode(),
                "filename": f"asset_{i}.txt"
            }
            asset_manager.store_asset(asset_data=asset_data, project_id=project.id)
        
        # Export project
        export_path = asset_manager.export_project(project.id, format="zip")
        
        assert export_path.exists()
        assert export_path.suffix == ".zip"
    
    def test_import_assets(self, asset_manager):
        """Test bulk asset import"""
        import tempfile
        
        # Create temporary files to import
        import_dir = Path(tempfile.mkdtemp())
        try:
            # Create test files
            (import_dir / "file1.txt").write_text("content1")
            (import_dir / "file2.txt").write_text("content2")
            (import_dir / "file3.txt").write_text("content3")
            
            # Import assets
            imported_paths = asset_manager.import_assets(
                source_dir=import_dir,
                metadata={"import_source": "test"}
            )
            
            assert len(imported_paths) == 3
            
            # Check assets were imported
            for imported_path in imported_paths:
                assert imported_path.exists()
        finally:
            shutil.rmtree(import_dir)


@pytest.mark.asyncio
async def test_asset_manager_async_operations():
    """Test asynchronous asset operations"""
    with tempfile.TemporaryDirectory() as temp_dir:
        asset_manager = AssetManager(base_dir=Path(temp_dir))
        
        # Async asset storage
        async def store_async_asset():
            asset_data = {
                "content": b"Async test content",
                "filename": "async_test.png"
            }
            return asset_manager.store_asset(asset_data=asset_data)
        
        asset_path = await store_async_asset()
        assert asset_path.exists()
        
        # Async metadata retrieval
        async def get_async_metadata():
            return asset_manager.get_asset_metadata(asset_path)
        
        metadata = await get_async_metadata()
        assert metadata["filename"] == "async_test.png"


if __name__ == "__main__":
    pytest.main([__file__])
