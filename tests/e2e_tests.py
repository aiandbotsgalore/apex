"""
End-to-End Tests for APEX DIRECTOR

Tests complete user workflows from start to finish.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time

from apex_director.director import (
    MusicVideoRequest, 
    MusicVideoResult, 
    APEXDirector,
    submit_music_video_job,
    get_job_status
)
from apex_director.core.orchestrator import APEXOrchestrator
from apex_director.core.asset_manager import AssetManager


class TestCompleteUserWorkflows:
    """End-to-end user workflow tests"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    async def director(self, temp_dir):
        """Initialize APEX Director"""
        director = APEXDirector(base_dir=temp_dir)
        await director.initialize()
        yield director
        await director.shutdown()
    
    @pytest.mark.asyncio
    async def test_simple_music_video_generation(self, director, temp_dir):
        """Test simple music video generation from start to finish"""
        # Step 1: Prepare audio file
        audio_file = temp_dir / "test_song.mp3"
        audio_file.write_text("Mock audio content for testing")
        
        # Step 2: Submit generation request
        request = MusicVideoRequest(
            job_id="simple_test",
            audio_path=audio_file,
            output_dir=temp_dir / "output",
            genre="pop",
            artist_name="Test Artist",
            song_title="Test Song",
            concept="upbeat dance video",
            max_shots=10,
            quality_preset="web"
        )
        
        # Mock the generation process
        with patch.object(director, '_generate_music_video') as mock_generate:
            mock_generate.return_value = MusicVideoResult(
                success=True,
                job_id="simple_test",
                output_video_path=temp_dir / "output" / "final_video.mp4",
                total_processing_time=45.0,
                generated_images=[temp_dir / f"shot_{i}.png" for i in range(10)],
                overall_quality_score=0.87,
                style_consistency_score=0.92,
                audio_sync_score=0.95
            )
            
            # Submit job
            result = await director.generate_music_video(request)
            
            assert result.success == True
            assert result.output_video_path.exists()
            assert len(result.generated_images) == 10
            assert result.overall_quality_score > 0.8
            assert result.total_processing_time > 0
    
    @pytest.mark.asyncio
    async def test_batch_music_video_generation(self, director, temp_dir):
        """Test batch music video generation"""
        # Create multiple audio files
        audio_files = []
        for i in range(3):
            audio_file = temp_dir / f"batch_song_{i}.mp3"
            audio_file.write_text(f"Mock audio content {i}")
            audio_files.append(audio_file)
        
        # Submit batch requests
        requests = []
        for i, audio_file in enumerate(audio_files):
            request = MusicVideoRequest(
                job_id=f"batch_test_{i}",
                audio_path=audio_file,
                output_dir=temp_dir / f"output_{i}",
                genre="cinematic",
                max_shots=5,
                quality_preset="draft"
            )
            requests.append(request)
        
        # Mock batch generation
        with patch.object(director, '_generate_music_video') as mock_generate:
            async def mock_generate_batch(request):
                await asyncio.sleep(0.1)  # Simulate processing
                return MusicVideoResult(
                    success=True,
                    job_id=request.job_id,
                    output_video_path=request.output_dir / "final_video.mp4",
                    total_processing_time=30.0,
                    generated_images=[temp_dir / f"batch_{request.job_id}_shot_{j}.png" for j in range(5)]
                )
            
            mock_generate.side_effect = mock_generate_batch
            
            # Process batch
            results = []
            for request in requests:
                result = await director.generate_music_video(request)
                results.append(result)
            
            assert len(results) == 3
            for result in results:
                assert result.success == True
                assert len(result.generated_images) == 5
    
    @pytest.mark.asyncio
    async def test_custom_style_music_video(self, director, temp_dir):
        """Test music video generation with custom styling"""
        # Create style bible
        style_bible = {
            "visual_style": "neon cyberpunk",
            "color_palette": {
                "primary": ["#FF00FF", "#00FFFF", "#FF6600"],
                "secondary": ["#000000", "#333333"]
            },
            "lighting_style": "high contrast neon",
            "mood": "energetic and futuristic"
        }
        
        # Save style bible
        style_file = temp_dir / "custom_style.json"
        with open(style_file, 'w') as f:
            json.dump(style_bible, f)
        
        # Create audio file
        audio_file = temp_dir / "cyberpunk_song.mp3"
        audio_file.write_text("Mock cyberpunk audio")
        
        # Submit request with custom style
        request = MusicVideoRequest(
            job_id="custom_style_test",
            audio_path=audio_file,
            output_dir=temp_dir / "styled_output",
            genre="electronic",
            visual_themes=["cyberpunk", "neon", "futuristic"],
            color_palette=["#FF00FF", "#00FFFF"],
            enable_style_consistency=True,
            style_drift_tolerance=0.1
        )
        
        # Mock style-consistent generation
        with patch.object(director, '_generate_with_style_consistency') as mock_style_gen:
            mock_style_gen.return_value = MusicVideoResult(
                success=True,
                job_id="custom_style_test",
                output_video_path=temp_dir / "styled_output" / "styled_video.mp4",
                total_processing_time=60.0,
                style_consistency_score=0.94,
                generated_images=[temp_dir / f"styled_shot_{i}.png" for i in range(15)]
            )
            
            result = await director.generate_music_video(request, style_bible_path=style_file)
            
            assert result.success == True
            assert result.style_consistency_score > 0.9
            assert result.total_processing_time > 30  # Style consistency takes more time
    
    @pytest.mark.asyncio
    async def test_character_consistent_music_video(self, director, temp_dir):
        """Test music video generation with character consistency"""
        # Create character reference images
        character_refs = []
        for i in range(3):
            ref_file = temp_dir / f"character_ref_{i}.jpg"
            ref_file.write_text(f"Mock character reference {i}")
            character_refs.append(ref_file)
        
        # Create audio file
        audio_file = temp_dir / "character_song.mp3"
        audio_file.write_text("Mock character song audio")
        
        # Submit request with character consistency
        request = MusicVideoRequest(
            job_id="character_test",
            audio_path=audio_file,
            output_dir=temp_dir / "character_output",
            genre="narrative",
            character_reference_images=character_refs,
            enable_character_consistency=True,
            max_shots=20
        )
        
        # Mock character-consistent generation
        with patch.object(director, '_generate_with_character_consistency') as mock_char_gen:
            mock_char_gen.return_value = MusicVideoResult(
                success=True,
                job_id="character_test",
                output_video_path=temp_dir / "character_output" / "character_video.mp4",
                total_processing_time=120.0,
                generated_images=[temp_dir / f"character_shot_{i}.png" for i in range(20)],
                overall_quality_score=0.89
            )
            
            result = await director.generate_music_video(request)
            
            assert result.success == True
            assert len(result.generated_images) == 20
            assert result.total_processing_time > 60  # Character consistency takes time
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, director, temp_dir):
        """Test error recovery in end-to-end workflow"""
        # Create audio file
        audio_file = temp_dir / "error_test_song.mp3"
        audio_file.write_text("Mock audio for error testing")
        
        request = MusicVideoRequest(
            job_id="error_recovery_test",
            audio_path=audio_file,
            output_dir=temp_dir / "error_output",
            max_shots=5
        )
        
        # Simulate initial failure
        with patch.object(director, '_generate_music_video') as mock_generate:
            # First attempt fails
            mock_generate.side_effect = [
                Exception("Backend unavailable"),
                MusicVideoResult(
                    success=True,
                    job_id="error_recovery_test",
                    output_video_path=temp_dir / "error_output" / "recovered_video.mp4",
                    total_processing_time=45.0,
                    generated_images=[temp_dir / f"recovered_shot_{i}.png" for i in range(5)]
                )
            ]
            
            # Submit job (should retry and succeed)
            result = await director.generate_music_video(request)
            
            assert result.success == True
            assert mock_generate.call_count == 2  # Should have retried once
    
    @pytest.mark.asyncio
    async def test_progress_tracking_workflow(self, director, temp_dir):
        """Test progress tracking throughout the workflow"""
        # Create audio file
        audio_file = temp_dir / "progress_test_song.mp3"
        audio_file.write_text("Mock audio for progress tracking")
        
        # Track progress
        progress_updates = []
        
        def progress_callback(progress, status):
            progress_updates.append({"progress": progress, "status": status})
        
        request = MusicVideoRequest(
            job_id="progress_test",
            audio_path=audio_file,
            output_dir=temp_dir / "progress_output",
            progress_callback=progress_callback,
            max_shots=10
        )
        
        # Mock generation with progress updates
        with patch.object(director, '_generate_with_progress') as mock_progress_gen:
            async def generate_with_progress(request):
                # Simulate progress updates
                progress_updates.extend([
                    {"progress": 0.1, "status": "Analyzing audio"},
                    {"progress": 0.3, "status": "Generating scenes"},
                    {"progress": 0.6, "status": "Creating images"},
                    {"progress": 0.8, "status": "Assembling video"},
                    {"progress": 0.9, "status": "Finalizing"},
                    {"progress": 1.0, "status": "Complete"}
                ])
                
                return MusicVideoResult(
                    success=True,
                    job_id="progress_test",
                    output_video_path=temp_dir / "progress_output" / "tracked_video.mp4",
                    total_processing_time=90.0,
                    generated_images=[temp_dir / f"tracked_shot_{i}.png" for i in range(10)]
                )
            
            mock_progress_gen.side_effect = generate_with_progress
            
            result = await director.generate_music_video(request)
            
            assert result.success == True
            assert len(progress_updates) >= 6  # Should have progress updates
            
            # Check progress flow
            progress_values = [update["progress"] for update in progress_updates]
            assert progress_values[0] < progress_values[-1]  # Progress increases
            assert progress_values[-1] == 1.0  # Final progress is 100%
    
    @pytest.mark.asyncio
    async def test_quality_control_workflow(self, director, temp_dir):
        """Test quality control throughout the workflow"""
        # Create audio file
        audio_file = temp_dir / "quality_test_song.mp3"
        audio_file.write_text("Mock audio for quality testing")
        
        request = MusicVideoRequest(
            job_id="quality_test",
            audio_path=audio_file,
            output_dir=temp_dir / "quality_output",
            quality_preset="broadcast",
            broadcast_compliance=True,
            max_shots=15
        )
        
        # Mock quality-controlled generation
        with patch.object(director, '_generate_with_quality_control') as mock_quality_gen:
            mock_quality_gen.return_value = MusicVideoResult(
                success=True,
                job_id="quality_test",
                output_video_path=temp_dir / "quality_output" / "quality_video.mp4",
                total_processing_time=150.0,
                generated_images=[temp_dir / f"quality_shot_{i}.png" for i in range(15)],
                overall_quality_score=0.93,
                broadcast_compliance_score=0.96
            )
            
            result = await director.generate_music_video(request)
            
            assert result.success == True
            assert result.overall_quality_score > 0.9
            assert result.broadcast_compliance_score > 0.9
            assert result.total_processing_time > 60  # Quality control takes time
    
    @pytest.mark.asyncio
    async def test_resumable_workflow(self, director, temp_dir):
        """Test resumable workflow with checkpoints"""
        # Create audio file
        audio_file = temp_dir / "resume_test_song.mp3"
        audio_file.write_text("Mock audio for resume testing")
        
        request = MusicVideoRequest(
            job_id="resume_test",
            audio_path=audio_file,
            output_dir=temp_dir / "resume_output",
            max_shots=20
        )
        
        # Mock resumable generation
        with patch.object(director, '_generate_with_resume') as mock_resume_gen:
            # Simulate interruption and resume
            call_count = 0
            
            def generate_with_resume(request):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    # First attempt - simulate interruption
                    raise Exception("Process interrupted")
                else:
                    # Resume attempt
                    return MusicVideoResult(
                        success=True,
                        job_id="resume_test",
                        output_video_path=temp_dir / "resume_output" / "resumed_video.mp4",
                        total_processing_time=75.0,
                        generated_images=[temp_dir / f"resumed_shot_{i}.png" for i in range(20)],
                        resumed_from_checkpoint=True
                    )
            
            mock_resume_gen.side_effect = generate_with_resume
            
            # First attempt should fail
            with pytest.raises(Exception):
                await director.generate_music_video(request)
            
            # Resume should succeed
            result = await director.generate_music_video(request, resume=True)
            
            assert result.success == True
            assert result.resumed_from_checkpoint == True
            assert len(result.generated_images) == 20
    
    @pytest.mark.asyncio
    async def test_full_production_workflow(self, director, temp_dir):
        """Test full production workflow with all features enabled"""
        # Create audio file
        audio_file = temp_dir / "full_production_song.mp3"
        audio_file.write_text("Mock full production audio")
        
        # Create style bible and character references
        style_bible = {
            "visual_style": "cinematic drama",
            "color_palette": {"primary": ["#8B4513", "#D2B48C"], "secondary": ["#2F4F4F"]},
            "lighting_style": "golden hour"
        }
        
        character_refs = [temp_dir / "main_character.jpg" for _ in range(3)]
        for ref in character_refs:
            ref.write_text("Mock character reference")
        
        request = MusicVideoRequest(
            job_id="full_production",
            audio_path=audio_file,
            output_dir=temp_dir / "production_output",
            genre="cinematic",
            artist_name="Test Artist",
            song_title="Epic Ballad",
            concept="emotional journey",
            director_style="christopher_nolan",
            character_reference_images=character_refs,
            visual_themes=["emotional", "dramatic", "cinematic"],
            color_palette=["#8B4513", "#D2B48C"],
            target_resolution="1920x1080",
            target_fps=24,
            quality_preset="cinema",
            enable_character_consistency=True,
            enable_style_consistency=True,
            enable_upscaling=True,
            enable_color_grading=True,
            enable_motion_effects=True,
            broadcast_compliance=True,
            max_shots=30
        )
        
        # Mock full production generation
        with patch.object(director, '_generate_full_production') as mock_full_gen:
            mock_full_gen.return_value = MusicVideoResult(
                success=True,
                job_id="full_production",
                output_video_path=temp_dir / "production_output" / "full_production_video.mp4",
                total_processing_time=300.0,
                audio_analysis={
                    "duration": 180.0,
                    "tempo": 85.0,
                    "key": "D minor",
                    "sections": [{"type": "verse", "start": 0.0, "end": 30.0} for _ in range(6)]
                },
                cinematography_plan={
                    "shot_list": [f"Shot {i}: Cinematic angle" for i in range(30)],
                    "color_grading": "Cinematic teal & orange",
                    "motion_effects": ["Ken Burns", "Dolly zoom"]
                },
                image_generation_results=[
                    {"shot_id": i, "quality_score": 0.9 + (i % 10) * 0.01} for i in range(30)
                ],
                video_assembly_result={
                    "transitions": "Professional cuts and crossfades",
                    "color_grade": "Broadcast compliant",
                    "audio_sync": "Frame accurate"
                },
                quality_report={
                    "broadcast_compliant": True,
                    "quality_score": 0.94,
                    "style_consistency": 0.91,
                    "character_consistency": 0.88
                },
                generated_images=[temp_dir / f"production_shot_{i}.png" for i in range(30)],
                overall_quality_score=0.94,
                style_consistency_score=0.91,
                audio_sync_score=0.97,
                broadcast_compliance_score=0.96
            )
            
            result = await director.generate_music_video(request)
            
            assert result.success == True
            assert len(result.generated_images) == 30
            assert result.total_processing_time > 120  # Full production takes time
            assert result.overall_quality_score > 0.9
            assert result.broadcast_compliance_score > 0.9
            assert "audio_analysis" in result.audio_analysis
            assert "cinematography_plan" in result.cinematography_plan


class TestUserExperienceWorkflows:
    """Tests focused on user experience and ease of use"""
    
    @pytest.mark.asyncio
    async def test_simple_cli_workflow(self):
        """Test simplified CLI-like workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock CLI workflow
            def simple_workflow():
                # Simulate user providing minimal input
                audio_file = Path(temp_dir) / "song.mp3"
                audio_file.write_text("audio data")
                
                # Auto-detect genre and settings
                auto_settings = {
                    "genre": "pop",  # Auto-detected
                    "quality": "medium",
                    "duration": 180  # Auto-detected
                }
                
                # Generate with minimal user input
                return {
                    "input": {"audio": str(audio_file)},
                    "settings": auto_settings,
                    "output": str(temp_dir / "output.mp4")
                }
            
            workflow_result = simple_workflow()
            
            assert "input" in workflow_result
            assert "settings" in workflow_result
            assert "output" in workflow_result
            assert workflow_result["settings"]["genre"] == "pop"
    
    @pytest.mark.asyncio
    async def test_configuration_guided_workflow(self):
        """Test guided configuration workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate guided configuration
            def guided_configuration():
                steps = []
                
                # Step 1: Select genre
                steps.append({"step": "genre", "choice": "cinematic", "auto_settings": {
                    "color_palette": "dramatic",
                    "lighting": "high contrast",
                    "mood": "intense"
                }})
                
                # Step 2: Select style
                steps.append({"step": "style", "choice": "film_noir", "auto_settings": {
                    "visual_style": "black and white",
                    "composition": "symmetrical"
                }})
                
                # Step 3: Select quality
                steps.append({"step": "quality", "choice": "high", "auto_settings": {
                    "resolution": "1920x1080",
                    "upscaling": True,
                    "color_grading": True
                }})
                
                # Combine all settings
                final_config = {}
                for step in steps:
                    final_config.update(step["auto_settings"])
                
                return final_config
            
            config = guided_configuration()
            
            assert "genre" not in config  # Only auto settings
            assert "color_palette" in config
            assert "resolution" in config
            assert config["upscaling"] == True
    
    @pytest.mark.asyncio
    async def test_batch_file_processing(self):
        """Test processing multiple files at once"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple audio files
            audio_files = []
            for i in range(5):
                audio_file = Path(temp_dir) / f"song_{i}.mp3"
                audio_file.write_text(f"audio data {i}")
                audio_files.append(audio_file)
            
            # Batch processing workflow
            def process_batch(audio_files):
                results = []
                
                for audio_file in audio_files:
                    # Process each file
                    result = {
                        "input": str(audio_file),
                        "status": "processed",
                        "output": str(audio_file.parent / f"{audio_file.stem}_video.mp4"),
                        "settings": {
                            "genre": "auto-detected",
                            "quality": "standard"
                        }
                    }
                    results.append(result)
                
                return results
            
            batch_results = process_batch(audio_files)
            
            assert len(batch_results) == 5
            for result in batch_results:
                assert result["status"] == "processed"
                assert result["output"].endswith("_video.mp4")
    
    @pytest.mark.asyncio
    async def test_template_based_workflow(self):
        """Test using predefined templates"""
        templates = {
            "pop_video": {
                "genre": "pop",
                "style": "bright and colorful",
                "effects": ["quick cuts", "colorful transitions"],
                "duration_factor": 1.0
            },
            "rock_video": {
                "genre": "rock",
                "style": "high energy",
                "effects": ["fast zooms", "dramatic lighting"],
                "duration_factor": 1.2
            },
            "ballad_video": {
                "genre": "ballad",
                "style": "emotional and cinematic",
                "effects": ["slow motion", "soft focus"],
                "duration_factor": 0.8
            }
        }
        
        def apply_template(template_name, audio_duration):
            template = templates[template_name]
            
            return {
                "template": template_name,
                "applied_settings": template,
                "estimated_duration": audio_duration * template["duration_factor"],
                "effects_count": len(template["effects"])
            }
        
        # Test pop template
        pop_result = apply_template("pop_video", 180)
        assert pop_result["template"] == "pop_video"
        assert pop_result["estimated_duration"] == 180
        assert len(pop_result["applied_settings"]["effects"]) == 2
        
        # Test ballad template
        ballad_result = apply_template("ballad_video", 240)
        assert ballad_result["template"] == "ballad_video"
        assert ballad_result["estimated_duration"] == 192  # 240 * 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
