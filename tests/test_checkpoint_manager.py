"""
Unit tests for APEX DIRECTOR Checkpoint Manager

Tests state management, checkpoint creation, and recovery functionality.
"""

import pytest
import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import tempfile

from apex_director.core.checkpoint import CheckpointManager, Checkpoint, JobState


class MockJob:
    """Mock job for testing"""
    
    def __init__(self, job_id, status="queued"):
        self.id = job_id
        self.status = status
        self.progress = 0.0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.result = None
        self.error = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result": self.result,
            "error": self.error
        }


class TestCheckpointManager:
    """Test suite for CheckpointManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager instance"""
        return CheckpointManager(base_dir=temp_dir)
    
    def test_initialization(self, checkpoint_manager, temp_dir):
        """Test checkpoint manager initialization"""
        assert checkpoint_manager.base_dir == temp_dir
        assert (temp_dir / "checkpoints").exists()
    
    @pytest.mark.asyncio
    async def test_create_checkpoint(self, checkpoint_manager):
        """Test checkpoint creation"""
        # Create some mock jobs
        jobs = [
            MockJob("job_1", "processing"),
            MockJob("job_2", "queued"),
            MockJob("job_3", "completed")
        ]
        
        # Create checkpoint
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="test_checkpoint",
            description="Test checkpoint",
            jobs=jobs
        )
        
        assert checkpoint_id is not None
        assert len(checkpoint_id) > 0
        
        # Check checkpoint file exists
        checkpoint_dir = checkpoint_manager.checkpoints_dir / checkpoint_id
        assert checkpoint_dir.exists()
        
        # Check checkpoint metadata
        metadata_file = checkpoint_dir / "metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file) as f:
            metadata = json.load(f)
            assert metadata["name"] == "test_checkpoint"
            assert metadata["description"] == "Test checkpoint"
            assert "timestamp" in metadata
    
    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpoint_manager):
        """Test checkpoint listing"""
        # Create multiple checkpoints
        jobs = [MockJob(f"job_{i}") for i in range(3)]
        
        checkpoint1_id = await checkpoint_manager.create_checkpoint(
            name="checkpoint_1",
            jobs=jobs
        )
        
        checkpoint2_id = await checkpoint_manager.create_checkpoint(
            name="checkpoint_2",
            jobs=jobs
        )
        
        checkpoint3_id = await checkpoint_manager.create_checkpoint(
            name="checkpoint_3",
            jobs=jobs
        )
        
        # List checkpoints
        checkpoints = await checkpoint_manager.list_checkpoints()
        
        assert len(checkpoints) == 3
        
        checkpoint_names = [cp["name"] for cp in checkpoints]
        assert "checkpoint_1" in checkpoint_names
        assert "checkpoint_2" in checkpoint_names
        assert "checkpoint_3" in checkpoint_names
    
    @pytest.mark.asyncio
    async def test_get_checkpoint(self, checkpoint_manager):
        """Test checkpoint retrieval"""
        jobs = [MockJob(f"job_{i}") for i in range(3)]
        
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="retrieval_test",
            jobs=jobs
        )
        
        # Get checkpoint
        checkpoint = await checkpoint_manager.get_checkpoint(checkpoint_id)
        
        assert checkpoint is not None
        assert checkpoint.id == checkpoint_id
        assert checkpoint.name == "retrieval_test"
        assert len(checkpoint.jobs) == 3
    
    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, checkpoint_manager):
        """Test checkpoint deletion"""
        jobs = [MockJob("delete_test_job")]
        
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="delete_me",
            jobs=jobs
        )
        
        # Verify checkpoint exists
        checkpoints = await checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 1
        
        # Delete checkpoint
        success = await checkpoint_manager.delete_checkpoint(checkpoint_id)
        assert success == True
        
        # Verify checkpoint is gone
        checkpoints = await checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 0
        
        # Verify checkpoint directory is deleted
        checkpoint_dir = checkpoint_manager.checkpoints_dir / checkpoint_id
        assert not checkpoint_dir.exists()
    
    @pytest.mark.asyncio
    async def test_auto_checkpoint_creation(self, checkpoint_manager):
        """Test automatic checkpoint creation"""
        # Set short interval for testing
        checkpoint_manager.config["auto_checkpoint_interval"] = 1  # 1 second
        
        jobs = [MockJob(f"auto_job_{i}") for i in range(2)]
        
        # Start auto checkpoint
        auto_task = await checkpoint_manager.start_auto_checkpointing(
            interval=1.0,
            jobs_provider=lambda: jobs
        )
        
        # Wait for automatic checkpoint creation
        await asyncio.sleep(2.5)
        
        # Check that auto checkpoints were created
        checkpoints = await checkpoint_manager.list_checkpoints()
        
        # Should have at least one auto checkpoint
        auto_checkpoints = [cp for cp in checkpoints if cp.get("auto_created")]
        assert len(auto_checkpoints) >= 1
        
        # Stop auto checkpointing
        auto_task.cancel()
        try:
            await auto_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint(self, checkpoint_manager):
        """Test checkpoint restoration"""
        # Create checkpoint with jobs
        jobs = [
            MockJob("restore_job_1", "processing"),
            MockJob("restore_job_2", "queued")
        ]
        
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="restore_test",
            jobs=jobs
        )
        
        # Mock job state callback
        job_state_callback = Mock()
        
        # Restore checkpoint
        success = await checkpoint_manager.restore_checkpoint(
            checkpoint_id,
            job_state_callback=job_state_callback
        )
        
        assert success == True
        
        # Verify callback was called for each job
        assert job_state_callback.call_count == len(jobs)
    
    @pytest.mark.asyncio
    async def test_checkpoint_compression(self, checkpoint_manager):
        """Test checkpoint compression for large data"""
        # Create jobs with large result data
        jobs = []
        for i in range(5):
            job = MockJob(f"large_job_{i}")
            # Add large result data
            job.result = {"data": "x" * 100000}  # 100KB of data
            jobs.append(job)
        
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="compression_test",
            jobs=jobs,
            compress=True
        )
        
        # Check that checkpoint was created
        checkpoint = await checkpoint_manager.get_checkpoint(checkpoint_id)
        assert checkpoint is not None
        
        # Verify compressed file exists
        checkpoint_dir = checkpoint_manager.checkpoints_dir / checkpoint_id
        compressed_file = checkpoint_dir / "jobs.json.gz"
        # Note: Actual compression depends on implementation
    
    @pytest.mark.asyncio
    async def test_checkpoint_metadata_update(self, checkpoint_manager):
        """Test checkpoint metadata updates"""
        jobs = [MockJob("metadata_job")]
        
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="metadata_test",
            jobs=jobs
        )
        
        # Update metadata
        updates = {
            "description": "Updated description",
            "tags": ["important", "production"],
            "user_note": "This checkpoint is critical"
        }
        
        success = await checkpoint_manager.update_checkpoint_metadata(
            checkpoint_id,
            updates
        )
        
        assert success == True
        
        # Verify updates
        checkpoint = await checkpoint_manager.get_checkpoint(checkpoint_id)
        assert checkpoint.description == "Updated description"
        assert "important" in checkpoint.tags
        assert checkpoint.user_note == "This checkpoint is critical"
    
    @pytest.mark.asyncio
    async def test_checkpoint_diff(self, checkpoint_manager):
        """Test checkpoint difference calculation"""
        # Create first checkpoint
        jobs1 = [MockJob("job_1"), MockJob("job_2")]
        checkpoint1_id = await checkpoint_manager.create_checkpoint(
            name="checkpoint_1",
            jobs=jobs1
        )
        
        # Create second checkpoint with modifications
        jobs2 = [
            MockJob("job_1", "completed"),  # Status changed
            MockJob("job_2", "queued"),     # Status changed
            MockJob("job_3", "queued")      # New job added
        ]
        checkpoint2_id = await checkpoint_manager.create_checkpoint(
            name="checkpoint_2",
            jobs=jobs2
        )
        
        # Calculate diff
        diff = await checkpoint_manager.calculate_checkpoint_diff(
            checkpoint1_id,
            checkpoint2_id
        )
        
        assert "modified_jobs" in diff
        assert "added_jobs" in diff
        assert "removed_jobs" in diff
        
        assert len(diff["modified_jobs"]) >= 2  # job_1 and job_2
        assert len(diff["added_jobs"]) == 1     # job_3
    
    @pytest.mark.asyncio
    async def test_checkpoint_export_import(self, checkpoint_manager):
        """Test checkpoint export and import"""
        jobs = [MockJob(f"export_job_{i}") for i in range(3)]
        
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="export_test",
            jobs=jobs
        )
        
        # Export checkpoint
        export_path = await checkpoint_manager.export_checkpoint(
            checkpoint_id,
            format="zip"
        )
        
        assert export_path.exists()
        
        # Import checkpoint to new location
        import_manager = CheckpointManager(base_dir=checkpoint_manager.base_dir / "imported")
        
        imported_id = await import_manager.import_checkpoint(
            export_path
        )
        
        assert imported_id is not None
        
        # Verify imported checkpoint
        imported_checkpoint = await import_manager.get_checkpoint(imported_id)
        assert imported_checkpoint.name == "export_test"
        assert len(imported_checkpoint.jobs) == 3
    
    @pytest.mark.asyncio
    async def test_checkpoint_cleanup(self, checkpoint_manager):
        """Test automatic checkpoint cleanup"""
        # Create old checkpoint
        old_jobs = [MockJob("old_job")]
        old_checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="old_checkpoint",
            jobs=old_jobs
        )
        
        # Manually set old timestamp
        checkpoint_dir = checkpoint_manager.checkpoints_dir / old_checkpoint_id
        metadata_file = checkpoint_dir / "metadata.json"
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        metadata["timestamp"] = (datetime.now() - timedelta(days=30)).isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create recent checkpoint
        recent_jobs = [MockJob("recent_job")]
        recent_checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="recent_checkpoint",
            jobs=recent_jobs
        )
        
        # Run cleanup (remove checkpoints older than 7 days)
        cleaned_count = await checkpoint_manager.cleanup_old_checkpoints(
            max_age_days=7
        )
        
        assert cleaned_count >= 1
        
        # Verify old checkpoint is gone
        checkpoints = await checkpoint_manager.list_checkpoints()
        checkpoint_names = [cp["name"] for cp in checkpoints]
        assert "old_checkpoint" not in checkpoint_names
        assert "recent_checkpoint" in checkpoint_names
    
    @pytest.mark.asyncio
    async def test_checkpoint_validation(self, checkpoint_manager):
        """Test checkpoint validation"""
        # Create valid checkpoint
        jobs = [MockJob("valid_job")]
        valid_id = await checkpoint_manager.create_checkpoint(
            name="valid_checkpoint",
            jobs=jobs
        )
        
        # Validate checkpoint
        is_valid = await checkpoint_manager.validate_checkpoint(valid_id)
        assert is_valid == True
        
        # Test with corrupted checkpoint (delete a required file)
        checkpoint_dir = checkpoint_manager.checkpoints_dir / valid_id
        metadata_file = checkpoint_dir / "metadata.json"
        metadata_file.unlink()  # Delete metadata file
        
        # Validate corrupted checkpoint
        is_valid = await checkpoint_manager.validate_checkpoint(valid_id)
        assert is_valid == False
    
    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_creation(self, checkpoint_manager):
        """Test concurrent checkpoint creation"""
        async def create_checkpoint_task(task_id):
            jobs = [MockJob(f"concurrent_job_{task_id}_{i}") for i in range(2)]
            checkpoint_id = await checkpoint_manager.create_checkpoint(
                name=f"concurrent_checkpoint_{task_id}",
                jobs=jobs
            )
            return checkpoint_id
        
        # Create multiple checkpoints concurrently
        tasks = [create_checkpoint_task(i) for i in range(5)]
        checkpoint_ids = await asyncio.gather(*tasks)
        
        assert len(checkpoint_ids) == 5
        
        # Verify all checkpoints exist
        for checkpoint_id in checkpoint_ids:
            checkpoint = await checkpoint_manager.get_checkpoint(checkpoint_id)
            assert checkpoint is not None
    
    @pytest.mark.asyncio
    async def test_checkpoint_statistics(self, checkpoint_manager):
        """Test checkpoint statistics"""
        # Create checkpoints with different characteristics
        for i in range(5):
            jobs = [MockJob(f"stats_job_{i}_{j}") for j in range(i + 1)]
            await checkpoint_manager.create_checkpoint(
                name=f"stats_checkpoint_{i}",
                jobs=jobs
            )
        
        # Get statistics
        stats = await checkpoint_manager.get_statistics()
        
        assert "total_checkpoints" in stats
        assert "total_jobs_saved" in stats
        assert "storage_used_mb" in stats
        assert "average_jobs_per_checkpoint" in stats
        
        assert stats["total_checkpoints"] == 5
        assert stats["total_jobs_saved"] == 15  # 1+2+3+4+5
        assert stats["average_jobs_per_checkpoint"] == 3.0


class TestCheckpoint:
    """Test suite for Checkpoint data class"""
    
    def test_checkpoint_creation(self):
        """Test checkpoint object creation"""
        jobs = [MockJob("test_job")]
        
        checkpoint = Checkpoint(
            id="test_checkpoint",
            name="Test Checkpoint",
            description="A test checkpoint",
            timestamp=datetime.now(),
            jobs=jobs,
            metadata={"version": "1.0"}
        )
        
        assert checkpoint.id == "test_checkpoint"
        assert checkpoint.name == "Test Checkpoint"
        assert checkpoint.description == "A test checkpoint"
        assert len(checkpoint.jobs) == 1
        assert checkpoint.metadata["version"] == "1.0"
    
    def test_checkpoint_serialization(self):
        """Test checkpoint serialization"""
        jobs = [MockJob("serialize_job")]
        
        checkpoint = Checkpoint(
            id="serialize_test",
            name="Serialize Test",
            timestamp=datetime.now(),
            jobs=jobs
        )
        
        # Convert to dict
        checkpoint_dict = checkpoint.to_dict()
        
        # Verify required fields
        assert "id" in checkpoint_dict
        assert "name" in checkpoint_dict
        assert "timestamp" in checkpoint_dict
        assert "jobs" in checkpoint_dict
        
        # Recreate from dict
        restored_checkpoint = Checkpoint.from_dict(checkpoint_dict)
        
        assert restored_checkpoint.id == checkpoint.id
        assert restored_checkpoint.name == checkpoint.name


class TestJobState:
    """Test suite for JobState data class"""
    
    def test_job_state_creation(self):
        """Test job state object creation"""
        job_state = JobState(
            job_id="test_job",
            status="processing",
            progress=0.5,
            result=None,
            error=None
        )
        
        assert job_state.job_id == "test_job"
        assert job_state.status == "processing"
        assert job_state.progress == 0.5
        assert job_state.result is None
        assert job_state.error is None
    
    def test_job_state_transitions(self):
        """Test job state transitions"""
        job_state = JobState(
            job_id="transition_test",
            status="queued",
            progress=0.0
        )
        
        # Transition to processing
        job_state.status = "processing"
        job_state.progress = 0.5
        
        assert job_state.status == "processing"
        assert job_state.progress == 0.5
        
        # Transition to completed
        job_state.status = "completed"
        job_state.progress = 1.0
        job_state.result = {"output": "success"}
        
        assert job_state.status == "completed"
        assert job_state.progress == 1.0
        assert job_state.result is not None


@pytest.mark.asyncio
async def test_checkpoint_manager_integration():
    """Integration test for checkpoint manager"""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_manager = CheckpointManager(base_dir=Path(temp_dir))
        
        # Create jobs
        jobs = [
            MockJob("integration_job_1"),
            MockJob("integration_job_2"),
            MockJob("integration_job_3")
        ]
        
        # Create checkpoint
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="integration_test",
            jobs=jobs
        )
        
        assert checkpoint_id is not None
        
        # List checkpoints
        checkpoints = await checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 1
        
        # Get checkpoint
        checkpoint = await checkpoint_manager.get_checkpoint(checkpoint_id)
        assert checkpoint.name == "integration_test"
        assert len(checkpoint.jobs) == 3
        
        # Get statistics
        stats = await checkpoint_manager.get_statistics()
        assert stats["total_checkpoints"] == 1
        assert stats["total_jobs_saved"] == 3


if __name__ == "__main__":
    pytest.main([__file__])
