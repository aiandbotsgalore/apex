"""
Unit tests for APEX DIRECTOR Image Generation Pipeline

Tests cinematic image generation, backend integration, quality scoring, and style consistency.
"""

import pytest
import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import numpy as np

from apex_director.images.generator import CinematicImageGenerator, GenerationRequest
from apex_director.images.prompt_engineer import PromptEngineer
from apex_director.images.style_persistence import StylePersistenceManager
from apex_director.images.character_system import CharacterConsistencyManager
from apex_director.images.variant_selector import VariantSelector
from apex_director.images.upscaller import ProfessionalUpscaler


class MockImageBackend:
    """Mock image generation backend for testing"""
    
    def __init__(self, name="mock", should_fail=False):
        self.name = name
        self.should_fail = should_fail
        self.call_count = 0
    
    async def generate(self, prompt, **kwargs):
        self.call_count += 1
        
        if self.should_fail:
            raise Exception(f"{self.name} backend failed")
        
        # Return mock image data
        return {
            "images": [
                {
                    "url": f"http://example.com/{self.name}_image_{self.call_count}.png",
                    "filename": f"image_{self.call_count}.png"
                }
            ],
            "metadata": {
                "backend": self.name,
                "prompt": prompt,
                "parameters": kwargs
            }
        }


class TestCinematicImageGenerator:
    """Test suite for CinematicImageGenerator"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def image_generator(self, temp_dir):
        """Create image generator instance"""
        return CinematicImageGenerator(base_dir=temp_dir)
    
    def test_initialization(self, image_generator, temp_dir):
        """Test image generator initialization"""
        assert image_generator.base_dir == temp_dir
        assert image_generator.backend_manager is not None
        assert image_generator.style_manager is not None
        assert image_generator.character_manager is not None
    
    @pytest.mark.asyncio
    async def test_generate_single_image(self, image_generator):
        """Test single image generation"""
        # Mock backend
        mock_backend = MockImageBackend("test_backend")
        
        with patch.object(image_generator.backend_manager, 'generate_single_backend') as mock_gen:
            mock_gen.return_value = {
                "image_path": "/tmp/test_image.png",
                "metadata": {"backend": "test_backend"}
            }
            
            request = GenerationRequest(
                prompt="A cinematic sunset",
                scene_id="test_scene",
                genre="cinematic"
            )
            
            result = await image_generator.generate_single_image(request)
            
            assert result is not None
            assert "image_path" in result
    
    @pytest.mark.asyncio
    async def test_generate_image_sequence(self, image_generator):
        """Test image sequence generation with consistency"""
        # Mock backend responses
        with patch.object(image_generator.backend_manager, 'generate_sequence') as mock_gen:
            mock_gen.return_value = {
                "images": [
                    {"path": "/tmp/frame_1.png"},
                    {"path": "/tmp/frame_2.png"},
                    {"path": "/tmp/frame_3.png"}
                ],
                "consistency_score": 0.85
            }
            
            requests = [
                GenerationRequest(
                    prompt=f"Scene frame {i}",
                    scene_id="sequence_test",
                    sequence_position=i
                ) for i in range(3)
            ]
            
            results = await image_generator.generate_image_sequence(requests)
            
            assert len(results["images"]) == 3
            assert results["consistency_score"] > 0
    
    @pytest.mark.asyncio
    async def test_generate_with_style_consistency(self, image_generator):
        """Test generation with style consistency"""
        style_data = {
            "visual_style": "cinematic realism",
            "color_palette": ["#FF6B35", "#004E89", "#1A936F"],
            "lighting_style": "dramatic"
        }
        
        image_generator.style_manager.load_style_bible(style_data)
        
        with patch.object(image_generator.backend_manager, 'generate_with_style') as mock_gen:
            mock_gen.return_value = {
                "image_path": "/tmp/styled_image.png",
                "style_consistency_score": 0.9
            }
            
            request = GenerationRequest(
                prompt="A warrior in battle",
                scene_id="battle_scene",
                style_constraints=True
            )
            
            result = await image_generator.generate_single_image(request)
            
            assert result["style_consistency_score"] > 0.8
    
    @pytest.mark.asyncio
    async def test_generate_with_character_consistency(self, image_generator):
        """Test generation with character consistency"""
        # Create character profile
        character_id = await image_generator.character_manager.create_character(
            name="test_character",
            reference_images=["/tmp/ref1.jpg", "/tmp/ref2.jpg"],
            description="A brave warrior"
        )
        
        with patch.object(image_generator.backend_manager, 'generate_with_character') as mock_gen:
            mock_gen.return_value = {
                "image_path": "/tmp/character_image.png",
                "character_consistency_score": 0.88
            }
            
            request = GenerationRequest(
                prompt="The warrior walks through a forest",
                scene_id="forest_scene",
                character_name="test_character",
                maintain_character=True
            )
            
            result = await image_generator.generate_single_image(request)
            
            assert result["character_consistency_score"] > 0.8
    
    @pytest.mark.asyncio
    async def test_upscaling(self, image_generator):
        """Test professional upscaling"""
        # Mock upscaler
        with patch.object(image_generator.upscaler, 'upscale_image') as mock_upscale:
            mock_upscale.return_value = {
                "upscaled_path": "/tmp/upscaled_image.png",
                "scale_factor": 4,
                "quality_metrics": {"sharpness": 0.92}
            }
            
            test_image_path = "/tmp/test_image.png"
            result = await image_generator.upscale_image(
                test_image_path,
                preset="broadcast_quality"
            )
            
            assert result["scale_factor"] == 4
            assert result["quality_metrics"]["sharpness"] > 0.9
    
    @pytest.mark.asyncio
    async def test_variant_selection(self, image_generator):
        """Test multi-variant generation and selection"""
        # Mock backend to return multiple variants
        with patch.object(image_generator.backend_manager, 'generate_variants') as mock_gen:
            mock_variants = [
                {
                    "image_path": f"/tmp/variant_{i}.png",
                    "metadata": {"backend": "test"}
                } for i in range(4)
            ]
            mock_gen.return_value = mock_variants
            
            request = GenerationRequest(
                prompt="A mystical landscape",
                variant_count=4
            )
            
            result = await image_generator.generate_with_variants(request)
            
            assert "variants" in result
            assert "selected_variant" in result
            assert len(result["variants"]) == 4
    
    @pytest.mark.asyncio
    async def test_batch_generation(self, image_generator):
        """Test batch image generation"""
        requests = [
            GenerationRequest(
                prompt=f"Scene {i}",
                scene_id=f"batch_scene_{i}"
            ) for i in range(5)
        ]
        
        with patch.object(image_generator, 'generate_single_image') as mock_gen:
            mock_gen.return_value = {
                "image_path": f"/tmp/batch_image_{i}.png"
            }
            
            results = await image_generator.generate_batch(requests)
            
            assert len(results) == 5
            assert mock_gen.call_count == 5
    
    @pytest.mark.asyncio
    async def test_error_handling(self, image_generator):
        """Test error handling in generation"""
        # Mock backend to fail
        mock_backend = MockImageBackend("failing_backend", should_fail=True)
        
        with patch.object(image_generator.backend_manager, 'generate_single_backend') as mock_gen:
            mock_gen.side_effect = Exception("Backend unavailable")
            
            request = GenerationRequest(
                prompt="A test image",
                scene_id="error_test"
            )
            
            # Should handle error gracefully
            with pytest.raises(Exception):
                await image_generator.generate_single_image(request)
    
    @pytest.mark.asyncio
    async def test_backend_fallback(self, image_generator):
        """Test automatic backend fallback"""
        with patch.object(image_generator.backend_manager, 'generate_with_fallback') as mock_fallback:
            mock_fallback.return_value = {
                "image_path": "/tmp/fallback_image.png",
                "backend_used": "backup_backend"
            }
            
            request = GenerationRequest(
                prompt="Fallback test image",
                enable_fallback=True
            )
            
            result = await image_generator.generate_single_image(request)
            
            assert result["backend_used"] == "backup_backend"
    
    @pytest.mark.asyncio
    async def test_quality_validation(self, image_generator):
        """Test image quality validation"""
        # Mock quality validator
        with patch.object(image_generator.quality_validator, 'validate_image') as mock_validate:
            mock_validate.return_value = {
                "is_valid": True,
                "quality_score": 0.87,
                "issues": []
            }
            
            test_image_path = "/tmp/quality_test.png"
            result = await image_generator.validate_image_quality(test_image_path)
            
            assert result["is_valid"] == True
            assert result["quality_score"] > 0.8
    
    @pytest.mark.asyncio
    async def test_style_drift_detection(self, image_generator):
        """Test style drift detection"""
        # Load initial style
        style_data = {"visual_style": "cinematic", "color_palette": ["#FF6B35"]}
        image_generator.style_manager.update_style(style_data)
        
        # Mock style analyzer
        with patch.object(image_generator.style_manager, 'detect_style_drift') as mock_drift:
            mock_drift.return_value = {
                "drift_score": 0.12,
                "drift_detected": False,
                "recommendations": []
            }
            
            test_image_path = "/tmp/style_test.png"
            result = await image_generator.detect_style_drift(test_image_path)
            
            assert result["drift_score"] < 0.15  # Within tolerance
    
    def test_get_generation_statistics(self, image_generator):
        """Test generation statistics"""
        stats = image_generator.get_generation_statistics()
        
        assert "total_generations" in stats
        assert "success_rate" in stats
        assert "average_quality_score" in stats
        assert "backend_usage" in stats
        assert "common_issues" in stats


class TestPromptEngineer:
    """Test suite for PromptEngineer"""
    
    def test_initialization(self):
        """Test prompt engineer initialization"""
        engineer = PromptEngineer()
        assert engineer is not None
        assert hasattr(engineer, 'genre_prompts')
        assert hasattr(engineer, 'director_styles')
    
    def test_build_cinematic_prompt(self):
        """Test cinematic prompt building"""
        engineer = PromptEngineer()
        
        prompt_data = {
            "base_prompt": "A warrior",
            "genre": "fantasy",
            "director_style": "christopher_nolan",
            "camera_settings": {
                "lens": "50mm",
                "aperture": "f/2.8"
            },
            "lighting": {
                "key_light": "volumetric",
                "mood": "dramatic"
            }
        }
        
        enhanced_prompt = engineer.build_cinematic_prompt(prompt_data)
        
        assert len(enhanced_prompt) > len(prompt_data["base_prompt"])
        assert "cinematic" in enhanced_prompt.lower() or "film" in enhanced_prompt.lower()
    
    def test_optimize_for_backend(self):
        """Test backend-specific prompt optimization"""
        engineer = PromptEngineer()
        
        base_prompt = "A detailed landscape"
        
        # Optimize for different backends
        sdxl_prompt = engineer.optimize_for_backend(base_prompt, "sdxl")
        imagen_prompt = engineer.optimize_for_backend(base_prompt, "imagen")
        
        assert sdxl_prompt != base_prompt
        assert imagen_prompt != base_prompt
        assert sdxl_prompt != imagen_prompt  # Different optimizations
    
    def test_add_camera_settings(self):
        """Test camera settings addition"""
        engineer = PromptEngineer()
        
        base_prompt = "A portrait"
        camera_settings = {
            "lens": "85mm",
            "aperture": "f/1.4",
            "iso": "ISO 200"
        }
        
        enhanced_prompt = engineer.add_camera_settings(base_prompt, camera_settings)
        
        assert "85mm" in enhanced_prompt or "85 mm" in enhanced_prompt
        assert "f/1.4" in enhanced_prompt or "f1.4" in enhanced_prompt
    
    def test_add_lighting_setup(self):
        """Test lighting setup addition"""
        engineer = PromptEngineer()
        
        base_prompt = "A subject"
        lighting_setup = {
            "key_light": "soft_box",
            "fill_light": "reflector",
            "background": "neutral_gray"
        }
        
        enhanced_prompt = engineer.add_lighting_setup(base_prompt, lighting_setup)
        
        assert "lighting" in enhanced_prompt.lower() or "illumination" in enhanced_prompt.lower()
    
    def test_apply_composition_rules(self):
        """Test composition rule application"""
        engineer = PromptEngineer()
        
        base_prompt = "A landscape"
        composition_rules = {
            "rule_of_thirds": True,
            "leading_lines": True,
            "symmetry": False
        }
        
        enhanced_prompt = engineer.apply_composition_rules(base_prompt, composition_rules)
        
        # Should mention composition techniques
        composition_terms = ["rule of thirds", "leading lines", "composition"]
        assert any(term in enhanced_prompt.lower() for term in composition_terms)
    
    def test_get_genre_prompts(self):
        """Test genre-specific prompt retrieval"""
        engineer = PromptEngineer()
        
        fantasy_prompt = engineer.get_genre_prompts("fantasy")
        noir_prompt = engineer.get_genre_prompts("film_noir")
        
        assert isinstance(fantasy_prompt, list)
        assert isinstance(noir_prompt, list)
        assert len(fantasy_prompt) > 0
        assert len(noir_prompt) > 0
    
    def test_get_director_styles(self):
        """Test director style retrieval"""
        engineer = PromptEngineer()
        
        nolan_style = engineer.get_director_styles("christopher_nolan")
        anderson_style = engineer.get_director_styles("wes_anderson")
        
        assert "visual_style" in nolan_style
        assert "color_palette" in nolan_style
        assert "composition_style" in nolan_style


class TestStylePersistenceManager:
    """Test suite for StylePersistenceManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def style_manager(self, temp_dir):
        """Create style persistence manager"""
        return StylePersistenceManager(base_dir=temp_dir)
    
    def test_initialization(self, style_manager, temp_dir):
        """Test style manager initialization"""
        assert style_manager.base_dir == temp_dir
        assert (temp_dir / "style_bible.json").exists()
    
    def test_load_style_bible(self, style_manager):
        """Test loading style bible"""
        style_data = {
            "project_name": "Test Project",
            "visual_style": "cinematic realism",
            "color_palette": {
                "primary": ["#FF6B35", "#004E89"],
                "secondary": ["#1A936F", "#FFD23F"]
            }
        }
        
        style_manager.load_style_bible(style_data)
        
        assert style_manager.current_style is not None
        assert style_manager.current_style["project_name"] == "Test Project"
        assert style_manager.current_style["visual_style"] == "cinematic realism"
    
    def test_update_style(self, style_manager):
        """Test style updates"""
        # Initial style
        initial_style = {"visual_style": "natural"}
        style_manager.load_style_bible(initial_style)
        
        # Update with new style
        update_style = {
            "visual_style": "dramatic",
            "lighting": "high contrast"
        }
        
        style_manager.update_style(update_style)
        
        assert style_manager.current_style["visual_style"] == "dramatic"
        assert "lighting" in style_manager.current_style
    
    def test_extract_style_features(self, style_manager):
        """Test style feature extraction from images"""
        # Mock image analysis
        with patch.object(style_manager, 'analyze_image_style') as mock_analyze:
            mock_analyze.return_value = {
                "color_histogram": [0.1, 0.2, 0.3],
                "lighting_style": "soft",
                "texture_pattern": "smooth"
            }
            
            test_image = "/tmp/style_sample.png"
            features = style_manager.extract_style_features(test_image)
            
            assert "color_histogram" in features
            assert "lighting_style" in features
            assert "texture_pattern" in features
    
    def test_calculate_style_similarity(self, style_manager):
        """Test style similarity calculation"""
        style1 = {
            "color_palette": ["#FF6B35", "#004E89"],
            "lighting": "dramatic"
        }
        
        style2 = {
            "color_palette": ["#FF6B35", "#1A936F"],  # One similar color
            "lighting": "soft"
        }
        
        similarity = style_manager.calculate_style_similarity(style1, style2)
        
        assert 0 <= similarity <= 1
        assert similarity > 0  # Should have some similarity
    
    def test_detect_style_drift(self, style_manager):
        """Test style drift detection"""
        # Set baseline style
        baseline_style = {
            "color_palette": ["#FF6B35", "#004E89"],
            "lighting": "dramatic"
        }
        style_manager.set_baseline_style(baseline_style)
        
        # Mock current image style
        current_style = {
            "color_palette": ["#1A936F", "#FFD23F"],  # Different palette
            "lighting": "soft"  # Different lighting
        }
        
        drift_score = style_manager.detect_style_drift(current_style)
        
        assert 0 <= drift_score <= 1
        assert drift_score > 0  # Should detect some drift
    
    def test_save_style_profile(self, style_manager):
        """Test style profile saving"""
        style_profile = {
            "name": "Custom Style",
            "features": {
                "color_palette": ["#FF6B35"],
                "lighting": "dramatic"
            }
        }
        
        profile_id = style_manager.save_style_profile(style_profile)
        
        assert profile_id is not None
        
        # Verify profile was saved
        saved_profile = style_manager.get_style_profile(profile_id)
        assert saved_profile["name"] == "Custom Style"


class TestCharacterConsistencyManager:
    """Test suite for CharacterConsistencyManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def character_manager(self, temp_dir):
        """Create character consistency manager"""
        return CharacterConsistencyManager(base_dir=temp_dir)
    
    @pytest.mark.asyncio
    async def test_create_character_profile(self, character_manager):
        """Test character profile creation"""
        profile_id = await character_manager.create_character_profile(
            name="test_character",
            reference_images=["/tmp/ref1.jpg", "/tmp/ref2.jpg"],
            description="A brave warrior"
        )
        
        assert profile_id is not None
        
        # Verify profile was created
        profile = character_manager.get_character_profile(profile_id)
        assert profile["name"] == "test_character"
        assert profile["description"] == "A brave warrior"
    
    @pytest.mark.asyncio
    async def test_extract_character_features(self, character_manager):
        """Test character feature extraction"""
        # Mock face detection and encoding
        with patch.object(character_manager, 'encode_face') as mock_encode:
            mock_encode.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            
            reference_image = "/tmp/character_ref.jpg"
            features = await character_manager.extract_character_features(reference_image)
            
            assert "face_encoding" in features
            assert "facial_landmarks" in features
            assert "age_estimate" in features
    
    @pytest.mark.asyncio
    async def test_validate_character_consistency(self, character_manager):
        """Test character consistency validation"""
        # Create character profile
        profile_id = await character_manager.create_character_profile(
            name="validation_test",
            reference_images=["/tmp/ref.jpg"]
        )
        
        # Mock consistency check
        with patch.object(character_manager, 'compare_faces') as mock_compare:
            mock_compare.return_value = {
                "similarity_score": 0.87,
                "matches": True,
                "confidence": 0.92
            }
            
            test_image = "/tmp/generated_character.jpg"
            is_consistent, confidence = await character_manager.validate_consistency(
                test_image,
                profile_id
            )
            
            assert is_consistent == True
            assert confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_find_similar_characters(self, character_manager):
        """Test finding similar characters"""
        # Create multiple character profiles
        profile1_id = await character_manager.create_character_profile(
            name="character_1",
            reference_images=["/tmp/ref1.jpg"]
        )
        
        profile2_id = await character_manager.create_character_profile(
            name="character_2",
            reference_images=["/tmp/ref2.jpg"]
        )
        
        # Mock similarity calculation
        with patch.object(character_manager, 'calculate_character_similarity') as mock_sim:
            mock_sim.return_value = [
                {"profile_id": profile2_id, "similarity": 0.78}
            ]
            
            similar = await character_manager.find_similar_characters(profile1_id)
            
            assert len(similar) > 0
            assert similar[0]["similarity"] > 0.7
    
    def test_get_character_statistics(self, character_manager):
        """Test character statistics"""
        stats = character_manager.get_character_statistics()
        
        assert "total_characters" in stats
        assert "consistency_success_rate" in stats
        assert "average_confidence_score" in stats
        assert "common_issues" in stats


if __name__ == "__main__":
    pytest.main([__file__])
