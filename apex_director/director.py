"""
APEX DIRECTOR - Master Orchestrator
Complete end-to-end music video generation system

Integrates:
- Audio Analysis Engine  
- Cinematography & Narrative System
- Cinematic Image Generation Pipeline
- Video Assembly & Post-Production Engine
- Quality Assurance Framework
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import numpy as np

from .audio.analyzer import AudioAnalysisEngine
from .cinematography import CinematographyDirector
from .images.generator import CinematicImageGenerator, GenerationRequest
from .video.assembler import VideoAssembler, AssemblyConfig
from .qa.validator import QualityValidator
from .core.orchestrator import APEXOrchestrator
from .core.asset_manager import AssetManager
from .core.checkpoint import CheckpointManager
from .core.estimator import Estimator


@dataclass
class MusicVideoRequest:
    """Complete music video generation request"""
    job_id: str
    audio_path: Path
    output_dir: Path
    
    # Creative direction
    genre: str = "pop"
    artist_name: str = ""
    song_title: str = ""
    concept: str = ""
    director_style: str = ""
    
    # Technical specifications
    target_resolution: str = "1920x1080"  # HD, UHD_4K
    target_fps: int = 24
    target_duration: Optional[float] = None  # Auto-detect from audio
    quality_preset: str = "broadcast"  # draft, web, broadcast, cinema
    
    # Generation preferences
    character_reference_images: List[Path] = field(default_factory=list)
    style_reference_images: List[Path] = field(default_factory=list)
    color_palette: List[str] = field(default_factory=list)
    visual_themes: List[str] = field(default_factory=list)
    
    # Processing options
    max_shots: int = 50
    shots_per_minute: float = 10.0
    enable_character_consistency: bool = True
    enable_style_consistency: bool = True
    enable_upscaling: bool = True
    enable_color_grading: bool = True
    enable_motion_effects: bool = True
    
    # Quality assurance
    enable_qa: bool = True
    broadcast_compliance: bool = True
    style_drift_tolerance: float = 0.15
    
    # Callbacks
    progress_callback: Optional[Callable[[float, str], None]] = None
    status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


@dataclass
class MusicVideoResult:
    """Complete music video generation result"""
    success: bool
    job_id: str
    output_video_path: Optional[Path] = None
    
    # Generation metadata
    total_processing_time: float = 0.0
    audio_analysis: Dict[str, Any] = field(default_factory=dict)
    cinematography_plan: Dict[str, Any] = field(default_factory=dict)
    image_generation_results: List[Dict[str, Any]] = field(default_factory=list)
    video_assembly_result: Dict[str, Any] = field(default_factory=dict)
    quality_report: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    overall_quality_score: float = 0.0
    style_consistency_score: float = 0.0
    audio_sync_score: float = 0.0
    broadcast_compliance_score: float = 0.0
    
    # Asset inventory
    generated_images: List[Path] = field(default_factory=list)
    intermediate_files: List[Path] = field(default_factory=list)
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_json(self) -> str:
        """Export result as JSON"""
        result_dict = {
            "success": self.success,
            "job_id": self.job_id,
            "output_video_path": str(self.output_video_path) if self.output_video_path else None,
            "total_processing_time": self.total_processing_time,
            "overall_quality_score": self.overall_quality_score,
            "style_consistency_score": self.style_consistency_score,
            "audio_sync_score": self.audio_sync_score,
            "broadcast_compliance_score": self.broadcast_compliance_score,
            "generated_images": [str(p) for p in self.generated_images],
            "errors": self.errors,
            "warnings": self.warnings
        }
        return json.dumps(result_dict, indent=2)


class APEXDirectorMaster:
    """
    Master orchestrator for complete music video generation
    Coordinates all APEX DIRECTOR subsystems
    """
    
    def __init__(self, workspace_dir: Path = Path("apex_workspace")):
        self.workspace_dir = workspace_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize workspace
        self._initialize_workspace()
        
        # Initialize subsystems
        self.audio_analyzer = AudioAnalysisEngine()
        self.cinematography_director = CinematographyDirector()
        self.image_generator = CinematicImageGenerator()
        self.video_assembler = VideoAssembler(AssemblyConfig())
        self.quality_validator = QualityValidator()
        
        # Core systems
        self.asset_manager = AssetManager(str(workspace_dir / "assets"))
        self.checkpoint_manager = CheckpointManager(str(workspace_dir / "checkpoints"))
        self.estimator = Estimator()
        
        # Processing state
        self.current_job: Optional[MusicVideoRequest] = None
        self.processing_stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "average_processing_time": 0.0
        }
    
    def _initialize_workspace(self):
        """Initialize workspace directory structure"""
        subdirs = [
            "assets", "checkpoints", "audio_analysis", "cinematography",
            "generated_images", "video_assembly", "quality_reports", 
            "final_outputs", "logs", "temp"
        ]
        
        for subdir in subdirs:
            (self.workspace_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    async def generate_music_video(self, request: MusicVideoRequest) -> MusicVideoResult:
        """
        Master function to generate complete music video
        Orchestrates the entire pipeline from audio to final video
        """
        start_time = time.time()
        self.current_job = request
        
        result = MusicVideoResult(success=False, job_id=request.job_id)
        
        try:
            self.logger.info(f"Starting music video generation for job {request.job_id}")
            self._update_status("initializing", {"job_id": request.job_id})
            
            # Create job-specific workspace
            job_workspace = self.workspace_dir / f"job_{request.job_id}"
            job_workspace.mkdir(exist_ok=True)
            
            # 1. AUDIO ANALYSIS PHASE
            self.logger.info("Phase 1: Audio Analysis")
            self._update_status("audio_analysis", {"phase": 1, "total_phases": 5})
            self._update_progress(0.05, "Analyzing audio structure...")
            
            audio_result = await self._analyze_audio(request, job_workspace)
            if not audio_result["success"]:
                result.errors.append(f"Audio analysis failed: {audio_result.get('error', 'Unknown error')}")
                return result
            
            result.audio_analysis = audio_result["data"]
            
            # 2. CINEMATOGRAPHY PLANNING PHASE  
            self.logger.info("Phase 2: Cinematography Planning")
            self._update_status("cinematography_planning", {"phase": 2, "total_phases": 5})
            self._update_progress(0.15, "Creating cinematic plan...")
            
            cinematography_result = await self._create_cinematography_plan(request, audio_result["data"], job_workspace)
            if not cinematography_result["success"]:
                result.errors.append(f"Cinematography planning failed: {cinematography_result.get('error', 'Unknown error')}")
                return result
            
            result.cinematography_plan = cinematography_result["data"]
            
            # 3. IMAGE GENERATION PHASE
            self.logger.info("Phase 3: Image Generation")
            self._update_status("image_generation", {"phase": 3, "total_phases": 5})
            self._update_progress(0.25, "Generating cinematic images...")
            
            image_result = await self._generate_images(request, cinematography_result["data"], job_workspace)
            if not image_result["success"]:
                result.errors.append(f"Image generation failed: {image_result.get('error', 'Unknown error')}")
                return result
            
            result.image_generation_results = image_result["data"]
            result.generated_images = image_result["image_paths"]
            
            # 4. VIDEO ASSEMBLY PHASE
            self.logger.info("Phase 4: Video Assembly") 
            self._update_status("video_assembly", {"phase": 4, "total_phases": 5})
            self._update_progress(0.70, "Assembling final video...")
            
            assembly_result = await self._assemble_video(request, audio_result["data"], image_result, job_workspace)
            if not assembly_result["success"]:
                result.errors.append(f"Video assembly failed: {assembly_result.get('error', 'Unknown error')}")
                return result
            
            result.video_assembly_result = assembly_result["data"]
            result.output_video_path = Path(assembly_result["output_path"])
            
            # 5. QUALITY ASSURANCE PHASE
            if request.enable_qa:
                self.logger.info("Phase 5: Quality Assurance")
                self._update_status("quality_assurance", {"phase": 5, "total_phases": 5})
                self._update_progress(0.90, "Validating quality...")
                
                qa_result = await self._validate_quality(request, result.output_video_path, job_workspace)
                result.quality_report = qa_result.get("data", {})
                
                # Extract quality scores
                if "scores" in result.quality_report:
                    scores = result.quality_report["scores"]
                    result.overall_quality_score = scores.get("overall_quality", 0.0)
                    result.style_consistency_score = scores.get("style_consistency", 0.0)
                    result.audio_sync_score = scores.get("audio_sync", 0.0)
                    result.broadcast_compliance_score = scores.get("broadcast_compliance", 0.0)
            
            # SUCCESS
            result.success = True
            result.total_processing_time = time.time() - start_time
            
            self._update_progress(1.0, "Music video generation completed!")
            self._update_status("completed", {"output_path": str(result.output_video_path)})
            
            # Save result metadata
            result_path = job_workspace / "generation_result.json"
            with open(result_path, 'w') as f:
                f.write(result.to_json())
            
            # Update statistics
            self.processing_stats["total_jobs"] += 1
            self.processing_stats["completed_jobs"] += 1
            self.processing_stats["average_processing_time"] = (
                (self.processing_stats["average_processing_time"] * (self.processing_stats["total_jobs"] - 1) + 
                 result.total_processing_time) / self.processing_stats["total_jobs"]
            )
            
            self.logger.info(f"Music video generation completed successfully: {result.output_video_path}")
            return result
            
        except Exception as e:
            result.errors.append(f"Unexpected error: {str(e)}")
            result.total_processing_time = time.time() - start_time
            
            self.processing_stats["total_jobs"] += 1
            self.processing_stats["failed_jobs"] += 1
            
            self.logger.error(f"Music video generation failed: {str(e)}")
            return result
    
    async def _analyze_audio(self, request: MusicVideoRequest, workspace: Path) -> Dict[str, Any]:
        """Phase 1: Comprehensive audio analysis"""
        try:
            analysis_result = await self.audio_analyzer.analyze_complete(
                audio_path=str(request.audio_path),
                output_dir=str(workspace / "audio_analysis")
            )
            
            return {
                "success": True,
                "data": analysis_result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_cinematography_plan(self, 
                                        request: MusicVideoRequest, 
                                        audio_analysis: Dict[str, Any], 
                                        workspace: Path) -> Dict[str, Any]:
        """Phase 2: Create cinematography and narrative plan"""
        try:
            # Extract audio sections and emotional arc
            audio_sections = audio_analysis.get("sections", [])
            audio_duration = audio_analysis.get("duration", 0.0)
            
            # Create emotional arc from spectral analysis
            spectral_data = audio_analysis.get("spectral", {})
            emotional_arc = self._extract_emotional_arc(spectral_data)
            
            # Create cinematography plan
            plan = await self.cinematography_director.create_cinematic_plan(
                audio_duration=audio_duration,
                audio_sections=audio_sections,
                genre=request.genre,
                emotional_arc=emotional_arc
            )
            
            # Save plan
            plan_path = workspace / "cinematography" / "cinematic_plan.json"
            plan_path.parent.mkdir(exist_ok=True)
            self.cinematography_director.export_cinematography_plan(plan_path)
            
            return {
                "success": True,
                "data": plan
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_images(self, 
                             request: MusicVideoRequest, 
                             cinematography_plan: Dict[str, Any], 
                             workspace: Path) -> Dict[str, Any]:
        """Phase 3: Generate cinematic images"""
        try:
            shot_sequence = cinematography_plan["shot_sequence"]
            image_requests = []
            
            # Create image generation requests for each shot
            for i, shot in enumerate(shot_sequence):
                # Create subject based on concept or generic description
                if request.concept:
                    subject = f"{request.concept}, {shot['emotional_tone']} scene"
                else:
                    subject = f"A {shot['emotional_tone']} music video scene"
                
                image_request = GenerationRequest(
                    prompt=subject,
                    scene_id=shot["shot_id"],
                    genre=request.genre,
                    director_style=request.director_style,
                    shot_type=shot["shot_type"],
                    emotional_tone=shot["emotional_tone"],
                    upscale=request.enable_upscaling,
                    output_dir=workspace / "generated_images" / f"shot_{i:03d}"
                )
                image_requests.append(image_request)
            
            # Generate images
            def progress_callback(current, total, message):
                progress = 0.25 + (current / total) * 0.4  # 25% to 65%
                self._update_progress(progress, f"Generating image {current}/{total}: {message}")
            
            results = await self.image_generator.generate_image_sequence(
                image_requests, progress_callback
            )
            
            # Collect image paths
            image_paths = []
            for result in results:
                if result.selected_variants:
                    image_paths.extend([v.image_path for v in result.selected_variants])
            
            return {
                "success": True,
                "data": [result.__dict__ for result in results],
                "image_paths": image_paths
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _assemble_video(self, 
                            request: MusicVideoRequest, 
                            audio_analysis: Dict[str, Any],
                            image_result: Dict[str, Any], 
                            workspace: Path) -> Dict[str, Any]:
        """Phase 4: Assemble final video"""
        try:
            # Configure assembly
            assembly_config = AssemblyConfig(
                target_fps=request.target_fps,
                resolution=self._parse_resolution(request.target_resolution),
                color_grading_enabled=request.enable_color_grading,
                motion_effects_enabled=request.enable_motion_effects,
                quality_preset=request.quality_preset
            )
            
            assembler = VideoAssembler(assembly_config)
            
            # Create style bible from request
            style_bible = self._create_style_bible(request)
            
            # Assemble video
            result = await assembler.assemble_video(
                job_id=request.job_id,
                audio_path=request.audio_path,
                images_dir=workspace / "generated_images",
                output_dir=workspace / "final_outputs",
                audio_analysis=audio_analysis,
                style_bible=style_bible
            )
            
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _validate_quality(self, 
                              request: MusicVideoRequest, 
                              video_path: Path, 
                              workspace: Path) -> Dict[str, Any]:
        """Phase 5: Quality assurance validation"""
        try:
            validation_result = await self.quality_validator.validate_complete_video(
                video_path=str(video_path),
                audio_path=str(request.audio_path),
                broadcast_compliance=request.broadcast_compliance,
                style_drift_tolerance=request.style_drift_tolerance
            )
            
            return {
                "success": True,
                "data": validation_result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
    
    def _extract_emotional_arc(self, spectral_data: Dict[str, Any]) -> List[str]:
        """Extract emotional arc from spectral analysis"""
        valence_data = spectral_data.get("valence", [])
        energy_data = spectral_data.get("energy", [])
        
        if not valence_data or not energy_data:
            return ["neutral"] * 5
        
        # Combine valence and energy to determine emotions
        emotions = []
        for i in range(len(valence_data)):
            valence = valence_data[i]
            energy = energy_data[i] if i < len(energy_data) else 0.5
            
            if valence > 0.6 and energy > 0.6:
                emotions.append("energetic")
            elif valence > 0.6 and energy < 0.4:
                emotions.append("peaceful")
            elif valence < 0.4 and energy > 0.6:
                emotions.append("dramatic")
            elif valence < 0.4 and energy < 0.4:
                emotions.append("sad")
            else:
                emotions.append("building")
        
        # Simplify to 5 main emotional phases
        if len(emotions) > 5:
            step = len(emotions) // 5
            emotions = [emotions[i * step] for i in range(5)]
        elif len(emotions) < 5:
            emotions.extend(["neutral"] * (5 - len(emotions)))
        
        return emotions[:5]
    
    def _parse_resolution(self, resolution_str: str) -> Tuple[int, int]:
        """Parse resolution string to tuple"""
        if resolution_str == "UHD_4K":
            return (3840, 2160)
        else:
            return (1920, 1080)  # Default to HD
    
    def _create_style_bible(self, request: MusicVideoRequest) -> Dict[str, Any]:
        """Create style bible from request parameters"""
        return {
            "genre": request.genre,
            "color_palette": request.color_palette,
            "visual_themes": request.visual_themes,
            "director_style": request.director_style,
            "character_consistency": request.enable_character_consistency,
            "style_consistency": request.enable_style_consistency
        }
    
    def _update_progress(self, progress: float, message: str):
        """Update progress callback"""
        if self.current_job and self.current_job.progress_callback:
            self.current_job.progress_callback(progress, message)
    
    def _update_status(self, status: str, data: Dict[str, Any]):
        """Update status callback"""
        if self.current_job and self.current_job.status_callback:
            self.current_job.status_callback(status, data)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()
    
    async def estimate_processing_time(self, request: MusicVideoRequest) -> Dict[str, Any]:
        """Estimate processing time and cost for request"""
        try:
            estimation = await self.estimator.estimate_job(
                audio_duration=request.target_duration or 180.0,  # Default 3 minutes
                num_shots=min(request.max_shots, int((request.target_duration or 180.0) * request.shots_per_minute / 60)),
                enable_upscaling=request.enable_upscaling,
                quality_preset=request.quality_preset,
                enable_qa=request.enable_qa
            )
            
            return estimation
        except Exception as e:
            return {
                "error": str(e),
                "estimated_time": 0.0,
                "estimated_cost": 0.0
            }


# Example usage and convenience functions
async def generate_music_video_simple(audio_path: str, 
                                    output_dir: str, 
                                    genre: str = "pop",
                                    concept: str = "") -> MusicVideoResult:
    """Simple interface for music video generation"""
    
    director = APEXDirectorMaster()
    
    request = MusicVideoRequest(
        job_id=f"simple_{int(time.time())}",
        audio_path=Path(audio_path),
        output_dir=Path(output_dir),
        genre=genre,
        concept=concept
    )
    
    result = await director.generate_music_video(request)
    return result


async def main():
    """Example usage"""
    
    director = APEXDirectorMaster()
    
    # Create a test request
    request = MusicVideoRequest(
        job_id="test_video_001",
        audio_path=Path("test_audio.mp3"),
        output_dir=Path("test_output"),
        genre="pop",
        concept="A dreamy journey through a magical forest",
        director_style="wes_anderson",
        quality_preset="broadcast"
    )
    
    # Add progress callback
    def progress_callback(progress: float, message: str):
        print(f"Progress: {progress*100:.1f}% - {message}")
    
    request.progress_callback = progress_callback
    
    # Generate music video
    result = await director.generate_music_video(request)
    
    if result.success:
        print(f"✅ Music video generated successfully!")
        print(f"Output: {result.output_video_path}")
        print(f"Processing time: {result.total_processing_time:.2f}s")
        print(f"Overall quality: {result.overall_quality_score:.3f}")
    else:
        print(f"❌ Generation failed:")
        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())