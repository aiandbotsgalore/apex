"""
Unit tests for APEX DIRECTOR Orchestrator

Tests the main orchestrator that coordinates all system components.
"""

import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil

from apex_director.core.orchestrator import APEXOrchestrator
from apex_director.core.asset_manager import AssetManager
from apex_director.core.checkpoint import CheckpointManager


class TestAPEXOrchestrator:
    """Test suite for APEXOrchestrator"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    async def orchestrator(self, temp_dir):
        """Create orchestrator instance with test configuration"""
        config = {
            "orchestrator": {
                "max_concurrent_jobs": 3,
                "checkpoint_interval": 60,
                "auto_retry": True,
                "retry_attempts": 3
            },
            "backends": {
                "test_backend": {
                    "enabled": True,
                    "priority": 1,
                    "cost_per_image": 0.01
                }
            }
        }
        
        orchestrator = APEXOrchestrator(
            config=config,
            base_dir=temp_dir
        )
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.status == "initialized"
        assert orchestrator.max_concurrent_jobs == 3
        assert orchestrator._running == False
    
    @pytest.mark.asyncio
    async def test_start_stop(self, orchestrator):
        """Test starting and stopping the orchestrator"""
        await orchestrator.start()
        assert orchestrator._running == True
        assert orchestrator.status == "running"
        
        await orchestrator.stop()
        assert orchestrator._running == False
        assert orchestrator.status == "stopped"
    
    @pytest.mark.asyncio
    async def test_job_submission(self, orchestrator):
        """Test job submission"""
        job_request = {
            "id": "test_job_001",
            "type": "image_generation",
            "prompt": "A cinematic sunset",
            "params": {"width": 512, "height": 512}
        }
        
        job_id = await orchestrator.submit_job(job_request)
        assert job_id == "test_job_001"
        
        # Check job is in queue
        assert job_id in orchestrator._job_queue
    
    @pytest.mark.asyncio
    async def test_job_status(self, orchestrator):
        """Test job status tracking"""
        job_id = "test_job_status"
        job_request = {
            "id": job_id,
            "type": "image_generation",
            "prompt": "A test image"
        }
        
        await orchestrator.submit_job(job_request)
        
        status = await orchestrator.get_job_status(job_id)
        assert status["id"] == job_id
        assert status["status"] in ["queued", "processing", "completed", "failed"]
    
    @pytest.mark.asyncio
    async def test_batch_job_submission(self, orchestrator):
        """Test batch job submission"""
        jobs = [
            {"id": f"batch_job_{i}", "type": "image_generation", "prompt": f"Test {i}"}
            for i in range(5)
        ]
        
        job_ids = await orchestrator.submit_batch(jobs)
        assert len(job_ids) == 5
        
        # Check all jobs are in queue
        for job_id in job_ids:
            assert job_id in orchestrator._job_queue
    
    @pytest.mark.asyncio
    async def test_job_cancellation(self, orchestrator):
        """Test job cancellation"""
        job_id = "cancel_test_job"
        job_request = {
            "id": job_id,
            "type": "image_generation",
            "prompt": "A test image"
        }
        
        await orchestrator.submit_job(job_request)
        
        # Cancel the job
        success = await orchestrator.cancel_job(job_id)
        assert success == True
        
        # Check job is cancelled
        status = await orchestrator.get_job_status(job_id)
        assert status["status"] == "cancelled"
    
    @pytest.mark.asyncio
    async def test_system_statistics(self, orchestrator):
        """Test system statistics collection"""
        stats = await orchestrator.get_system_stats()
        
        assert "jobs" in stats
        assert "performance" in stats
        assert "system" in stats
        
        assert "total_jobs" in stats["jobs"]
        assert "active_jobs" in stats["jobs"]
        assert "completed_jobs" in stats["jobs"]
        assert "failed_jobs" in stats["jobs"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator):
        """Test error handling and recovery"""
        # Submit an invalid job
        invalid_job = {
            "id": "invalid_job",
            "type": "invalid_type",
            "prompt": "A test"
        }
        
        job_id = await orchestrator.submit_job(invalid_job)
        assert job_id == "invalid_job"
        
        # Job should eventually fail
        await asyncio.sleep(0.1)
        status = await orchestrator.get_job_status(job_id)
        assert status["status"] in ["failed", "cancelled"]
    
    @pytest.mark.asyncio
    async def test_concurrent_job_processing(self, orchestrator):
        """Test processing multiple jobs concurrently"""
        # Submit multiple jobs
        job_ids = []
        for i in range(3):
            job_id = f"concurrent_job_{i}"
            job_request = {
                "id": job_id,
                "type": "image_generation",
                "prompt": f"Concurrent test {i}"
            }
            await orchestrator.submit_job(job_request)
            job_ids.append(job_id)
        
        # All jobs should be in queue
        for job_id in job_ids:
            assert job_id in orchestrator._job_queue
        
        # Check statistics
        stats = await orchestrator.get_system_stats()
        assert stats["jobs"]["active_jobs"] >= 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation(self, orchestrator):
        """Test checkpoint creation"""
        # Submit some jobs
        for i in range(3):
            job_id = f"checkpoint_job_{i}"
            job_request = {
                "id": job_id,
                "type": "image_generation",
                "prompt": f"Checkpoint test {i}"
            }
            await orchestrator.submit_job(job_request)
        
        # Create checkpoint
        checkpoint_id = await orchestrator.create_checkpoint("test_checkpoint")
        assert checkpoint_id is not None
        
        # List checkpoints
        checkpoints = await orchestrator.list_checkpoints()
        assert len(checkpoints) >= 1
    
    @pytest.mark.asyncio
    async def test_configuration_updates(self, orchestrator):
        """Test runtime configuration updates"""
        # Update configuration
        new_config = {
            "orchestrator": {
                "max_concurrent_jobs": 5
            }
        }
        
        await orchestrator.update_configuration(new_config)
        assert orchestrator.max_concurrent_jobs == 5
    
    def test_orchestrator_factory(self):
        """Test orchestrator factory methods"""
        # Test get_orchestrator function
        from apex_director import get_orchestrator
        orchestrator = get_orchestrator()
        assert orchestrator is not None
        assert isinstance(orchestrator, APEXOrchestrator)


@pytest.mark.asyncio
async def test_orchestrator_integration():
    """Integration test for full orchestrator workflow"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            "orchestrator": {
                "max_concurrent_jobs": 2,
                "checkpoint_interval": 30
            }
        }
        
        orchestrator = APEXOrchestrator(
            config=config,
            base_dir=Path(temp_dir)
        )
        
        await orchestrator.initialize()
        
        # Submit a job
        job_request = {
            "id": "integration_test",
            "type": "image_generation",
            "prompt": "Integration test image"
        }
        
        job_id = await orchestrator.submit_job(job_request)
        assert job_id == "integration_test"
        
        # Check status
        status = await orchestrator.get_job_status(job_id)
        assert status["id"] == job_id
        
        # Get system stats
        stats = await orchestrator.get_system_stats()
        assert stats["jobs"]["total_jobs"] >= 1
        
        await orchestrator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
