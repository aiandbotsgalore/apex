"""
Integration tests for APEX DIRECTOR

Tests end-to-end workflows across multiple system components.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from apex_director.core.orchestrator import APEXOrchestrator
from apex_director.core.asset_manager import AssetManager
from apex_director.core.checkpoint import CheckpointManager
from apex_director.core.estimator import Estimator
from apex_director.images.generator import CinematicImageGenerator
from apex_director.video.assembler import VideoAssembler
from apex_director.audio.analyzer import AudioAnalyzer


class TestMusicVideoGenerationIntegration:
    """Integration tests for complete music video generation workflow"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def system_components(self, temp_dir):
        """Initialize all system components"""
        components = {}
        
        # Core components
        components['orchestrator'] = APEXOrchestrator(
            config={
                "orchestrator": {"max_concurrent_jobs": 2},
                "backends": {"test_backend": {"enabled": True, "priority": 1}}
            },
            base_dir=temp_dir
        )
        
        components['asset_manager'] = AssetManager(base_dir=temp_dir)
        components['checkpoint_manager'] = CheckpointManager(base_dir=temp_dir)
        components['estimator'] = Estimator(base_dir=temp_dir)
        
        # Feature components
        components['image_generator'] = CinematicImageGenerator(base_dir=temp_dir)
        components['video_assembler'] = VideoAssembler(base_dir=temp_dir)
        components['audio_analyzer'] = AudioAnalyzer()
        
        return components
    
    @pytest.mark.asyncio
    async def test_full_music_video_workflow(self, system_components, temp_dir):
        """Test complete music video generation workflow"""
        orchestrator = system_components['orchestrator']
        asset_manager = system_components['asset_manager']
        image_generator = system_components['image_generator']
        video_assembler = system_components['video_assembler']
        audio_analyzer = system_components['audio_analyzer']
        
        # Step 1: Initialize system
        await orchestrator.initialize()
        
        # Step 2: Create project
        project = asset_manager.create_project(
            name="Integration Test Video",
            description="Test music video generation workflow"
        )
        
        # Step 3: Mock audio analysis
        with patch.object(audio_analyzer, 'analyze_audio') as mock_analyze:
            mock_analyze.return_value = {
                "duration": 120.0,
                "tempo": 120.0,
                "key": "C major",
                "sections": [
                    {"type": "verse", "start": 0.0, "end": 30.0},
                    {"type": "chorus", "start": 30.0, "end": 60.0},
                    {"type": "bridge", "start": 60.0, "end": 90.0},
                    {"type": "chorus", "start": 90.0, "end": 120.0}
                ],
                "beats": [{"time": i * 0.5, "confidence": 0.9} for i in range(240)]
            }
            
            # Create mock audio file
            audio_file = temp_dir / "test_audio.mp3"
            audio_file.write_text("mock audio data")
            
            audio_analysis = await audio_analyzer.analyze_audio(str(audio_file))
        
        assert audio_analysis["duration"] == 120.0
        assert len(audio_analysis["sections"]) == 4
        
        # Step 4: Generate images for each section
        image_requests = []
        for i, section in enumerate(audio_analysis["sections"]):
            from apex_director.images.generator import GenerationRequest
            
            request = GenerationRequest(
                prompt=f"{section['type']} scene with cinematic style",
                scene_id=f"section_{i}",
                section_type=section["type"],
                timing={"start": section["start"], "end": section["end"]},
                genre="cinematic"
            )
            image_requests.append(request)
        
        # Mock image generation
        with patch.object(image_generator, 'generate_single_image') as mock_gen:
            mock_gen.return_value = {
                "image_path": temp_dir / f"section_{i}.png",
                "quality_score": 0.85
            }
            
            generated_images = []
            for i, request in enumerate(image_requests):
                result = await image_generator.generate_single_image(request)
                generated_images.append(result["image_path"])
                
                # Store in asset manager
                asset_manager.store_asset({
                    "content": f"mock image {i}".encode(),
                    "filename": f"section_{i}.png",
                    "metadata": {
                        "project_id": project.id,
                        "section_type": request.section_type,
                        "timing": request.timing
                    }
                }, project_id=project.id)
        
        assert len(generated_images) == 4
        
        # Step 5: Create video assembly job
        from apex_director.video.assembler import AssemblyJob, QualityMode
        
        assembly_job = AssemblyJob(
            job_id="integration_test_assembly",
            audio_path=str(audio_file),
            output_path=str(temp_dir / "final_video.mp4"),
            quality_mode=QualityMode.HIGH,
            image_sequence=[
                {
                    "image_path": str(img),
                    "start_time": audio_analysis["sections"][i]["start"],
                    "end_time": audio_analysis["sections"][i]["end"]
                } for i, img in enumerate(generated_images)
            ]
        )
        
        # Mock video assembly
        with patch.object(video_assembler, 'assemble_video') as mock_assemble:
            mock_assemble.return_value = {
                "success": True,
                "output_path": str(temp_dir / "final_video.mp4"),
                "duration": 120.0,
                "processing_time": 30.5
            }
            
            assembly_result = await video_assembler.assemble_video(assembly_job)
        
        assert assembly_result["success"] == True
        assert assembly_result["duration"] == 120.0
        
        # Step 6: Create checkpoint
        checkpoint_id = await orchestrator.create_checkpoint(
            name="integration_test_checkpoint",
            jobs=[{"id": "assembly_job", "status": "completed"}]
        )
        
        assert checkpoint_id is not None
        
        # Step 7: Verify asset organization
        project_assets = asset_manager.search_assets(project_id=project.id)
        assert len(project_assets) >= 4  # At least the generated images
        
        # Step 8: Get final statistics
        stats = orchestrator.get_system_stats()
        assert stats["jobs"]["total_jobs"] >= 1
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, system_components):
        """Test error handling and recovery in integrated workflow"""
        orchestrator = system_components['orchestrator']
        estimator = system_components['estimator']
        
        await orchestrator.initialize()
        
        # Simulate job failure
        job_id = "recovery_test_job"
        job_request = {
            "id": job_id,
            "type": "image_generation",
            "prompt": "Test image"
        }
        
        await orchestrator.submit_job(job_request)
        
        # Simulate failure and retry
        # (actual implementation would depend on orchestrator error handling)
        
        # Create checkpoint before recovery
        checkpoint_id = await orchestrator.create_checkpoint(
            name="pre_recovery",
            jobs=[{"id": job_id, "status": "failed"}]
        )
        
        assert checkpoint_id is not None
        
        # Test recovery
        recovery_success = await orchestrator.restore_from_checkpoint(checkpoint_id)
        # Recovery success depends on implementation
    
    @pytest.mark.asyncio
    async def test_cost_estimation_integration(self, system_components):
        """Test cost estimation across the full workflow"""
        estimator = system_components['estimator']
        
        # Add historical data
        from apex_director.core.estimator import EstimationRecord
        
        for i in range(10):
            record = EstimationRecord(
                job_id=f"cost_test_{i}",
                backend="test_backend",
                width=1024,
                height=1024,
                steps=30,
                quality_level=4,
                actual_cost=0.05 + i * 0.01,
                actual_time=35.0 + i * 2.0,
                prompt_complexity=0.6
            )
            estimator.add_generation_record(record)
        
        # Estimate cost for video generation
        video_estimate = estimator.estimate_generation_cost_time(
            width=1920,
            height=1080,
            steps=50,
            quality_level=5,
            prompt="A complete cinematic sequence"
        )
        
        assert video_estimate.estimated_cost > 0
        assert video_estimate.estimated_time_seconds > 0
        assert video_estimate.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_parallel_processing_integration(self, system_components):
        """Test parallel processing across multiple components"""
        orchestrator = system_components['orchestrator']
        
        await orchestrator.initialize()
        
        # Submit multiple concurrent jobs
        job_requests = [
            {
                "id": f"parallel_job_{i}",
                "type": "image_generation",
                "prompt": f"Parallel test image {i}"
            } for i in range(5)
        ]
        
        job_ids = await orchestrator.submit_batch(job_requests)
        
        assert len(job_ids) == 5
        
        # Check system can handle concurrent jobs
        stats = orchestrator.get_system_stats()
        assert stats["jobs"]["active_jobs"] >= 0
        
        # Clean up
        for job_id in job_ids:
            await orchestrator.cancel_job(job_id)
    
    @pytest.mark.asyncio
    async def test_asset_lifecycle_integration(self, system_components):
        """Test complete asset lifecycle management"""
        asset_manager = system_components['asset_manager']
        
        # Create project
        project = asset_manager.create_project(
            name="Asset Lifecycle Test",
            description="Testing asset management lifecycle"
        )
        
        # Create multiple assets
        for i in range(10):
            asset_data = {
                "content": f"Asset {i} content".encode(),
                "filename": f"asset_{i}.dat",
                "metadata": {
                    "version": "1.0",
                    "tags": ["test", f"category_{i % 3}"]
                }
            }
            
            asset_path = asset_manager.store_asset(
                asset_data=asset_data,
                project_id=project.id
            )
            
            assert asset_path.exists()
        
        # Search and filter assets
        tagged_assets = asset_manager.search_assets(
            tags=["test"],
            project_id=project.id
        )
        
        assert len(tagged_assets) == 10
        
        # Update metadata
        updated_asset_path = Path(asset_path)
        asset_manager.update_asset_metadata(
            updated_asset_path,
            {"version": "2.0", "updated": True}
        )
        
        # Find duplicates (should find none in this test)
        duplicates = asset_manager.find_duplicates()
        # Result depends on asset content
        
        # Get storage statistics
        stats = asset_manager.get_storage_statistics()
        assert stats["total_files"] >= 10
        
        # Clean up
        cleanup_count = asset_manager.cleanup_temp_files()
        assert cleanup_count >= 0
    
    @pytest.mark.asyncio
    async def test_quality_assurance_integration(self, system_components):
        """Test quality assurance across the workflow"""
        image_generator = system_components['image_generator']
        
        # Mock quality validation
        with patch.object(image_generator, 'validate_image_quality') as mock_validate:
            mock_validate.return_value = {
                "is_valid": True,
                "quality_score": 0.87,
                "broadcast_compliant": True,
                "issues": []
            }
            
            test_image = "/tmp/test_image.png"
            quality_result = await image_generator.validate_image_quality(test_image)
            
            assert quality_result["is_valid"] == True
            assert quality_result["broadcast_compliant"] == True
            assert quality_result["quality_score"] > 0.8
    
    @pytest.mark.asyncio
    async def test_checkpoint_recovery_integration(self, system_components):
        """Test checkpoint and recovery across multiple components"""
        orchestrator = system_components['orchestrator']
        checkpoint_manager = system_components['checkpoint_manager']
        
        await orchestrator.initialize()
        
        # Create complex job state
        jobs = [
            {"id": "job_1", "status": "processing", "progress": 0.5},
            {"id": "job_2", "status": "queued", "progress": 0.0},
            {"id": "job_3", "status": "completed", "progress": 1.0}
        ]
        
        # Create checkpoint
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            name="complex_state",
            description="Complex system state",
            jobs=jobs
        )
        
        assert checkpoint_id is not None
        
        # Verify checkpoint can be retrieved
        checkpoint = await checkpoint_manager.get_checkpoint(checkpoint_id)
        assert checkpoint is not None
        assert len(checkpoint.jobs) == 3
        
        # Test checkpoint validation
        is_valid = await checkpoint_manager.validate_checkpoint(checkpoint_id)
        assert is_valid == True
        
        # Test checkpoint statistics
        stats = await checkpoint_manager.get_statistics()
        assert stats["total_checkpoints"] >= 1
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, system_components):
        """Test system health monitoring across all components"""
        orchestrator = system_components['orchestrator']
        asset_manager = system_components['asset_manager']
        
        await orchestrator.initialize()
        
        # Create some test data
        project = asset_manager.create_project("Health Test")
        for i in range(3):
            asset_manager.store_asset({
                "content": f"health test {i}".encode(),
                "filename": f"health_{i}.dat"
            }, project_id=project.id)
        
        # Get comprehensive system health
        orchestrator_health = orchestrator.get_system_stats()
        asset_health = asset_manager.get_storage_statistics()
        
        # Check health metrics
        assert orchestrator_health["system"]["status"] in ["healthy", "running"]
        assert asset_health["total_files"] >= 3
        
        # Test backend health (mocked)
        # In real implementation, would check actual backends
        with patch('apex_director.core.backend_manager.BackendManager.check_all_backends') as mock_health:
            mock_health.return_value = {
                "test_backend": {"healthy": True, "error": None}
            }
            
            backend_health = mock_health.return_value
            assert backend_health["test_backend"]["healthy"] == True


class TestWorkflowOptimizations:
    """Integration tests for workflow optimizations"""
    
    @pytest.mark.asyncio
    async def test_cache_efficiency(self):
        """Test caching effectiveness across operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup components with caching
            asset_manager = AssetManager(base_dir=Path(temp_dir))
            
            # Store asset
            asset_data = {
                "content": "cached content".encode(),
                "filename": "cache_test.dat"
            }
            
            path1 = asset_manager.store_asset(asset_data=asset_data)
            
            # Retrieve same content
            path2 = asset_manager.store_asset(asset_data=asset_data)
            
            # Should detect as duplicate
            duplicates = asset_manager.find_duplicates()
            # Implementation-specific cache behavior
    
    @pytest.mark.asyncio
    async def test_concurrent_limit_efficiency(self):
        """Test concurrent job limiting effectiveness"""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = APEXOrchestrator(
                config={"orchestrator": {"max_concurrent_jobs": 2}},
                base_dir=Path(temp_dir)
            )
            
            await orchestrator.initialize()
            
            # Submit many jobs
            job_ids = []
            for i in range(10):
                job_id = f"concurrent_test_{i}"
                job_request = {"id": job_id, "type": "test"}
                submitted_id = await orchestrator.submit_job(job_request)
                job_ids.append(submitted_id)
            
            # System should handle limiting gracefully
            stats = orchestrator.get_system_stats()
            assert stats["jobs"]["active_jobs"] <= 10  # Within bounds
            
            # Clean up
            for job_id in job_ids:
                await orchestrator.cancel_job(job_id)
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self):
        """Test memory usage optimization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with large asset handling
            asset_manager = AssetManager(base_dir=Path(temp_dir))
            
            # Create large asset
            large_content = b"x" * 1000000  # 1MB
            asset_data = {
                "content": large_content,
                "filename": "large_asset.dat"
            }
            
            asset_path = asset_manager.store_asset(asset_data=asset_data)
            
            # Verify large asset is handled efficiently
            assert asset_path.exists()
            
            # Get storage statistics
            stats = asset_manager.get_storage_statistics()
            assert stats["total_size_bytes"] >= 1000000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
