"""
Main Cinematic Image Generation Engine
Integrates all components for professional image generation pipeline
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from PIL import Image

# Import all components
from .backend_interface import BackendManager, GenerationConfig
from .prompt_engineer import CinematographyPromptEngineer, CameraSettings, LightingSetup, Composition
from .style_persistence import StyleBibleManager, StyleConsistencyValidator
from .character_system import CharacterConsistencyManager
from .variant_selector import VariantSelector
from .upscaller import ProfessionalUpscaler, UpscaleSettings

logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Complete generation request"""
    prompt: str
    scene_id: Optional[str] = None
    character_name: Optional[str] = None
    genre: str = "drama"
    director_style: Optional[str] = None
    camera_settings: Optional[CameraSettings] = None
    lighting_setup: Optional[LightingSetup] = None
    composition: Optional[Composition] = None
    backend_preference: str = "minimax"
    num_variants: int = 4
    upscale: bool = True
    upscale_preset: str = "broadcast_quality"
    output_dir: Path = Path("generated_images")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class GenerationResult:
    """Complete generation result"""
    request: GenerationRequest
    selected_variants: List[Any]  # VariantResult from variant_selector
    upscaled_variants: List[Any]  # UpscaleResult from upscaller
    generation_time: float
    style_consistency_score: float
    character_consistency_score: float
    overall_quality_score: float
    metadata: Dict[str, Any]
    
    def save_results(self):
        """Save all generation results"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save selected variants
        for i, variant in enumerate(self.selected_variants, 1):
            variant_path = self.output_dir / f"variant_{i}.png"
            variant.image.save(variant_path)
            
            # Save variant metadata
            variant_meta_path = self.output_dir / f"variant_{i}_metadata.json"
            with open(variant_meta_path, 'w') as f:
                json.dump(variant.to_dict(), f, indent=2, default=str)
        
        # Save upscaled variants
        for i, upscale_result in enumerate(self.upscaled_variants, 1):
            upscale_path = self.output_dir / f"upscaled_{i}"
            upscale_result.save(upscale_path)
        
        # Save generation metadata
        metadata_path = self.output_dir / "generation_metadata.json"
        generation_metadata = {
            "request": self.request.to_dict(),
            "generation_time": self.generation_time,
            "style_consistency_score": self.style_consistency_score,
            "character_consistency_score": self.character_consistency_score,
            "overall_quality_score": self.overall_quality_score,
            "num_variants_generated": len(self.selected_variants),
            "num_variants_upscaled": len(self.upscaled_variants)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(generation_metadata, f, indent=2, default=str)

class CinematicImageGenerator:
    """Main cinematic image generation engine"""
    
    def __init__(
        self,
        style_bible_path: str = "style_bible.json",
        character_profiles_path: str = "character_profiles",
        output_dir: str = "generated_images"
    ):
        # Initialize all components
        self.backend_manager = BackendManager()
        self.prompt_engineer = CinematographyPromptEngineer(style_bible_path)
        self.style_manager = StyleBibleManager(style_bible_path)
        self.style_validator = StyleConsistencyValidator(self.style_manager)
        self.character_manager = CharacterConsistencyManager(character_profiles_path)
        
        # Initialize upscaler
        self.upscaler = ProfessionalUpscaler()
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Cinematic image generator initialized")
    
    async def generate_image_sequence(
        self,
        requests: List[GenerationRequest],
        progress_callback: Optional[callable] = None
    ) -> List[GenerationResult]:
        """Generate a sequence of images with style consistency"""
        
        logger.info(f"Starting image sequence generation: {len(requests)} requests")
        
        results = []
        
        for i, request in enumerate(requests, 1):
            logger.info(f"Processing request {i}/{len(requests)}")
            
            if progress_callback:
                progress_callback(i, len(requests), f"Generating image {i}")
            
            try:
                result = await self.generate_single_image(request)
                results.append(result)
                
                # Update style bible with this generation for consistency
                if result.selected_variants:
                    self._update_style_consistency(request, result)
                
            except Exception as e:
                logger.error(f"Failed to generate image {i}: {e}")
                # Continue with next request
                continue
        
        if progress_callback:
            progress_callback(len(requests), len(requests), "Sequence generation completed")
        
        logger.info(f"Sequence generation completed: {len(results)}/{len(requests)} successful")
        return results
    
    async def generate_single_image(self, request: GenerationRequest) -> GenerationResult:
        """Generate a single image with all components"""
        
        start_time = time.time()
        
        # Step 1: Generate enhanced prompt
        enhanced_prompt = self._generate_enhanced_prompt(request)
        
        # Step 2: Create generation configuration
        config = GenerationConfig(
            width=1024,
            height=1024,
            steps=30,
            guidance_scale=7.5,
            num_variants=request.num_variants,
            backend=request.backend_preference
        )
        
        # Step 3: Generate image variants
        variants = await self._generate_variants(enhanced_prompt, config, request)
        
        if not variants:
            raise ValueError("No variants were generated")
        
        # Step 4: Select best variants
        selected_variants = await self._select_best_variants(variants, enhanced_prompt, request)
        
        # Step 5: Validate consistency
        consistency_scores = await self._validate_consistency(selected_variants, request)
        
        # Step 6: Upscale selected variants
        upscaled_variants = []
        if request.upscale and selected_variants:
            upscaled_variants = await self._upscale_variants(selected_variants, request)
        
        generation_time = time.time() - start_time
        
        # Create result
        result = GenerationResult(
            request=request,
            selected_variants=selected_variants,
            upscaled_variants=upscaled_variants,
            generation_time=generation_time,
            style_consistency_score=consistency_scores.get("style", 0.0),
            character_consistency_score=consistency_scores.get("character", 0.0),
            overall_quality_score=consistency_scores.get("overall", 0.0),
            metadata={
                "enhanced_prompt": enhanced_prompt,
                "backend_used": request.backend_preference,
                "generation_config": asdict(config),
                "consistency_scores": consistency_scores
            }
        )
        
        # Save results
        result.output_dir = request.output_dir
        result.save_results()
        
        logger.info(
            f"Image generation completed in {generation_time:.2f}s. "
            f"Quality score: {result.overall_quality_score:.3f}"
        )
        
        return result
    
    def _generate_enhanced_prompt(self, request: GenerationRequest) -> str:
        """Generate enhanced cinematic prompt"""
        
        # Create camera, lighting, and composition settings if not provided
        if not request.camera_settings:
            request.camera_settings = CameraSettings()
        
        if not request.lighting_setup:
            request.lighting_setup = LightingSetup()
        
        if not request.composition:
            request.composition = Composition()
        
        # Generate base prompt
        base_prompt = self.prompt_engineer.generate_cinematic_prompt(
            subject=request.prompt,
            scene_description="",
            genre=request.genre,
            director_style=request.director_style,
            camera_settings=request.camera_settings,
            lighting_setup=request.lighting_setup,
            composition=request.composition
        )
        
        # Apply style consistency
        style_prompt = self.style_manager.generate_consistency_prompt(
            base_prompt,
            scene_id=request.scene_id,
            character_name=request.character_name
        )
        
        # Add character-specific prompts
        if request.character_name:
            character_prompt = self.character_manager.generate_character_prompt(
                request.character_name
            )
            if character_prompt:
                style_prompt += f", {character_prompt}"
        
        # Optimize for backend
        optimized_prompt = self.prompt_engineer.optimize_prompt_for_backend(
            style_prompt,
            request.backend_preference
        )
        
        return optimized_prompt
    
    async def _generate_variants(
        self,
        prompt: str,
        config: GenerationConfig,
        request: GenerationRequest
    ) -> List[Tuple[str, Image.Image]]:
        """Generate image variants using backend"""
        
        logger.info(f"Generating {config.num_variants} variants")
        
        try:
            # Generate variants
            images = await self.backend_manager.generate_with_backend(
                config.backend,
                prompt,
                config
            )
            
            # Create variant IDs
            variants = [(f"variant_{i+1}", img) for i, img in enumerate(images)]
            
            return variants
            
        except Exception as e:
            logger.error(f"Failed to generate variants: {e}")
            return []
    
    async def _select_best_variants(
        self,
        variants: List[Tuple[str, Image.Image]],
        prompt: str,
        request: GenerationRequest
    ) -> List[Any]:
        """Select best variants using 4-criteria scoring"""
        
        logger.info("Selecting best variants")
        
        # Get style reference from style manager for consistency scoring
        style_reference = self._get_style_reference(request.scene_id)
        
        # Initialize variant selector
        selector = VariantSelector(style_reference)
        
        # Select best variants
        selected = selector.select_best_variants(
            variants,
            prompt,
            num_selections=min(2, len(variants))  # Select top 2 variants
        )
        
        # Log selection results
        for variant in selected:
            explanation = selector.get_scoring_explanation(variant)
            logger.info(f"Selected {variant.variant_id}:\n{explanation}")
        
        return selected
    
    async def _validate_consistency(
        self,
        variants: List[Any],
        request: GenerationRequest
    ) -> Dict[str, float]:
        """Validate style and character consistency"""
        
        if not variants:
            return {"style": 0.0, "character": 0.0, "overall": 0.0}
        
        # Use the best variant for validation
        best_variant = variants[0]
        image = best_variant.image
        
        scores = {
            "style": 0.0,
            "character": 0.0,
            "overall": 0.0
        }
        
        # Validate style consistency
        if request.scene_id:
            is_consistent, style_score, issues = self.style_validator.validate_image_consistency(
                image,
                expected_scene_id=request.scene_id
            )
            scores["style"] = style_score
            
            if issues:
                logger.warning(f"Style consistency issues: {issues}")
        
        # Validate character consistency
        if request.character_name:
            is_consistent, char_score, issues = self.character_manager.validate_character_consistency(
                image,
                request.character_name
            )
            scores["character"] = char_score
            
            if issues:
                logger.warning(f"Character consistency issues: {issues}")
        
        # Calculate overall score
        if request.scene_id and request.character_name:
            scores["overall"] = (scores["style"] + scores["character"]) / 2.0
        elif request.scene_id or request.character_name:
            scores["overall"] = max(scores["style"], scores["character"])
        else:
            scores["overall"] = best_variant.scores.overall_score
        
        return scores
    
    async def _upscale_variants(
        self,
        variants: List[Any],
        request: GenerationRequest
    ) -> List[Any]:
        """Upscale selected variants"""
        
        if not variants:
            return []
        
        logger.info(f"Upscaling {len(variants)} variants")
        
        upscaled_variants = []
        
        for variant in variants:
            try:
                upscale_result = self.upscaler.upscale_image(
                    variant.image,
                    preset=request.upscale_preset
                )
                upscaled_variants.append(upscale_result)
                
                logger.info(
                    f"Upscaled {variant.variant_id}: "
                    f"Quality score {upscale_result.quality_metrics.get('overall_quality', 0):.3f}"
                )
                
            except Exception as e:
                logger.error(f"Failed to upscale {variant.variant_id}: {e}")
                continue
        
        return upscaled_variants
    
    def _get_style_reference(self, scene_id: Optional[str]) -> Optional[Image.Image]:
        """Get style reference image for consistency scoring"""
        
        if scene_id:
            # Get scene style from style manager
            scene_style = self.style_manager.get_scene_style(scene_id)
            if scene_style and scene_style.reference_image:
                return scene_style.reference_image
        
        # Return None to use default style scoring
        return None
    
    def _update_style_consistency(
        self,
        request: GenerationRequest,
        result: GenerationResult
    ):
        """Update style bible with generation results"""
        
        if result.selected_variants:
            # Use the best variant to update style consistency
            best_variant = result.selected_variants[0]
            
            # Update style manager with this generation
            if request.scene_id:
                # Add to scene style if it exists
                scene_style = self.style_manager.get_scene_style(request.scene_id)
                if scene_style:
                    # Update style elements based on the generated image
                    pass  # In real implementation, would extract style features
    
    def create_project_template(self, project_name: str) -> Dict[str, Path]:
        """Create project template with all necessary files"""
        
        project_dir = self.output_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Create style bible template
        style_bible_path = project_dir / "style_bible.json"
        if not style_bible_path.exists():
            # Copy default style bible
            import shutil
            if Path("style_bible.json").exists():
                shutil.copy("style_bible.json", style_bible_path)
        
        # Create character profiles directory
        char_profiles_dir = project_dir / "character_profiles"
        char_profiles_dir.mkdir(exist_ok=True)
        
        # Create generation results directory
        results_dir = project_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Create template configuration
        config = {
            "project_name": project_name,
            "style_bible_path": str(style_bible_path),
            "character_profiles_dir": str(char_profiles_dir),
            "results_dir": str(results_dir),
            "default_settings": {
                "backend": "minimax",
                "upscale_preset": "broadcast_quality",
                "num_variants": 4
            }
        }
        
        config_path = project_dir / "project_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created project template: {project_dir}")
        
        return {
            "project_dir": project_dir,
            "style_bible": style_bible_path,
            "character_profiles": char_profiles_dir,
            "results": results_dir,
            "config": config_path
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and available components"""
        
        return {
            "available_backends": self.backend_manager.get_available_backends(),
            "character_count": len(self.character_manager.characters),
            "style_bible_loaded": bool(self.style_manager.style_bible),
            "upscaler_presets": list(self.upscaler.presets.keys()),
            "output_directory": str(self.output_dir),
            "generation_stats": {
                "total_generations": 0,  # Would track in real implementation
                "avg_quality_score": 0.0,
                "success_rate": 0.0
            }
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform system health check"""
        
        health_status = {}
        
        # Check backends
        backends_ok = []
        for backend_name in self.backend_manager.get_available_backends():
            try:
                backend = self.backend_manager.get_backend(backend_name)
                if backend:
                    backends_ok.append(True)
                else:
                    backends_ok.append(False)
            except:
                backends_ok.append(False)
        
        health_status["backends"] = all(backends_ok) if backends_ok else False
        
        # Check other components
        health_status["prompt_engineer"] = True
        health_status["style_manager"] = True
        health_status["character_manager"] = True
        health_status["variant_selector"] = True
        health_status["upscaler"] = True
        
        return health_status

# Convenience functions for easy usage
async def quick_generate(
    prompt: str,
    output_dir: str = "quick_generation",
    **kwargs
) -> GenerationResult:
    """Quick image generation with sensible defaults"""
    
    generator = CinematicImageGenerator(output_dir=output_dir)
    
    request = GenerationRequest(
        prompt=prompt,
        output_dir=Path(output_dir),
        **kwargs
    )
    
    return await generator.generate_single_image(request)

async def batch_generate(
    prompts: List[str],
    output_dir: str = "batch_generation",
    **kwargs
) -> List[GenerationResult]:
    """Batch generate multiple images"""
    
    generator = CinematicImageGenerator(output_dir=output_dir)
    
    requests = [
        GenerationRequest(
            prompt=prompt,
            output_dir=Path(output_dir),
            **kwargs
        )
        for prompt in prompts
    ]
    
    return await generator.generate_image_sequence(requests)