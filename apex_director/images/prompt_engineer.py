"""
Advanced Cinematography Prompt Engineering
Professional film and TV production prompt generation system
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import random

logger = logging.getLogger(__name__)

@dataclass
class CameraSettings:
    """Camera and lens settings for cinematography"""
    lens: str = "50mm"  # 24mm, 35mm, 50mm, 85mm, 135mm
    aperture: str = "f/2.8"
    focal_length: Optional[int] = None
    iso: str = "ISO 800"
    shutter_speed: str = "1/125"
    
    def __post_init__(self):
        if self.focal_length is None:
            # Extract focal length from lens
            try:
                self.focal_length = int(self.lens.replace("mm", ""))
            except:
                self.focal_length = 50

@dataclass
class LightingSetup:
    """Professional lighting setup"""
    key_light: str = "soft_box"
    fill_light: str = "reflector"
    rim_light: str = "hair_light"
    background: str = "neutral"
    mood: str = "cinematic"
    
    def format_lighting_description(self) -> str:
        return f"{self.key_light}, {self.fill_light}, {self.rim_light}, {self.background} background, {self.mood} mood"

@dataclass
class Composition:
    """Cinematic composition rules"""
    rule_of_thirds: bool = True
    leading_lines: bool = False
    symmetry: bool = False
    depth_of_field: str = "shallow"  # shallow, medium, deep
    framing: str = "medium_shot"  # close_up, medium_shot, wide_shot, extreme_wide
    
    def format_composition(self) -> str:
        elements = []
        if self.rule_of_thirds:
            elements.append("rule of thirds")
        if self.leading_lines:
            elements.append("leading lines")
        if self.symmetry:
            elements.append("symmetrical composition")
        elements.append(f"{self.depth_of_field} depth of field")
        elements.append(f"{self.framing}")
        return ", ".join(elements)

class CinematographyPromptEngineer:
    """Advanced prompt engineering for cinematic image generation"""
    
    def __init__(self, style_bible_path: str = "style_bible.json"):
        self.style_bible_path = Path(style_bible_path)
        self.style_bible = self._load_style_bible()
        self._load_cinematography_knowledge()
    
    def _load_style_bible(self) -> Dict[str, Any]:
        """Load style bible for consistency"""
        if self.style_bible_path.exists():
            try:
                with open(self.style_bible_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load style_bible.json: {e}")
        return {}
    
    def _load_cinematography_knowledge(self):
        """Load cinematography knowledge base"""
        # Film genres and their visual characteristics
        self.genre_prompts = {
            "film_noir": {
                "lighting": "high contrast, dramatic shadows, chiaroscuro lighting",
                "color_palette": "black and white with selective color",
                "composition": "low angle shots, geometric compositions",
                "atmosphere": "moody, mysterious, foreboding"
            },
            "sci_fi": {
                "lighting": "neon lights, volumetric lighting, futuristic glow",
                "color_palette": "blue, purple, cyan, neon green",
                "composition": "wide shots, architectural elements, technology",
                "atmosphere": "futuristic, clean, technological"
            },
            "horror": {
                "lighting": "low-key lighting, harsh shadows, practical lights",
                "color_palette": "dark blues, deep reds, muted earth tones",
                "composition": "Dutch angles, tight shots, negative space",
                "atmosphere": "ominous, unsettling, claustrophobic"
            },
            "thriller": {
                "lighting": "high contrast, directional lighting",
                "color_palette": "desaturated, cool tones",
                "composition": "tight framing, close-ups, dramatic angles",
                "atmosphere": "tense, suspenseful, dynamic"
            },
            "romance": {
                "lighting": "soft, warm lighting, golden hour",
                "color_palette": "warm pastels, soft pinks, golden yellows",
                "composition": "medium shots, intimate framing",
                "atmosphere": "romantic, soft, dreamy"
            },
            "action": {
                "lighting": "high contrast, dynamic lighting",
                "color_palette": "vibrant, saturated colors",
                "composition": "wide shots, dynamic angles, motion blur",
                "atmosphere": "energetic, intense, fast-paced"
            }
        }
        
        # Director styles and their visual signatures
        self_director_styles = {
            "wes_anderson": {
                "composition": "symmetrical composition, center framing, formal composition",
                "color_palette": "vibrant pastel colors, desaturated backgrounds",
                "lighting": "even, diffused lighting, minimal shadows",
                "style": "whimsical, detailed production design"
            },
            "christopher_nolan": {
                "composition": "IMAX aspect ratios, wide shots, practical effects",
                "color_palette": "desaturated colors, high contrast",
                "lighting": "natural lighting, practical lights, dramatic shadows",
                "style": "intense, realistic, technical precision"
            },
            "guillermo_del_toro": {
                "composition": "low angle shots, gothic elements, ornate details",
                "color_palette": "deep reds, rich browns, metallic accents",
                "lighting": "dramatic candlelight, practical sources",
                "style": "gothic, fantastical, tactile"
            },
            "david_fincher": {
                "composition": "precise framing, Dutch angles, geometric compositions",
                "color_palette": "desaturated, cool tones, high contrast",
                "lighting": "practically lit, moody, controlled",
                "style": "sleek, modern, meticulously crafted"
            }
        }
        
        # Technical cinematography terms
        self.technical_terms = {
            "lens_types": {
                "24mm": "wide angle lens, expansive view, distortion",
                "35mm": "wide angle lens, environmental context",
                "50mm": "natural perspective, human eye view",
                "85mm": "portrait lens, shallow depth of field",
                "135mm": "telephoto lens, compression, background separation"
            },
            "aperture_effects": {
                "f/1.4": "very shallow depth of field, bokeh, dreamy background",
                "f/2.0": "shallow depth of field, subject isolation",
                "f/2.8": "shallow depth of field, professional standard",
                "f/4.0": "medium depth of field, subtle background blur",
                "f/8.0": "deep depth of field, sharp throughout"
            },
            "camera_movements": {
                "dolly_in": "approaching subject, building tension",
                "dolly_out": "revealing context, creating space",
                "track_shot": "following action, dynamic perspective",
                "crane_shot": "elevated perspective, sweeping movement",
                "handheld": "organic, documentary style, energy"
            }
        }
    
    def generate_cinematic_prompt(
        self,
        subject: str,
        scene_description: str,
        genre: str = "drama",
        director_style: Optional[str] = None,
        camera_settings: Optional[CameraSettings] = None,
        lighting_setup: Optional[LightingSetup] = None,
        composition: Optional[Composition] = None,
        style_bible_overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a comprehensive cinematic prompt"""
        
        # Start with base prompt
        prompt_parts = []
        
        # Subject and scene
        prompt_parts.append(f"{subject}, {scene_description}")
        
        # Apply style bible consistency
        if self.style_bible:
            prompt_parts.append(self._apply_style_bible_consistency())
        
        # Genre-specific elements
        if genre in self.genre_prompts:
            genre_data = self.genre_prompts[genre]
            for category, description in genre_data.items():
                prompt_parts.append(description)
        
        # Director style
        if director_style and director_style in self_director_styles:
            director_data = self_director_styles[director_style]
            for category, description in director_data.items():
                prompt_parts.append(description)
        
        # Camera settings
        if camera_settings:
            prompt_parts.append(self._format_camera_settings(camera_settings))
        
        # Lighting setup
        if lighting_setup:
            prompt_parts.append(lighting_setup.format_lighting_description())
        
        # Composition
        if composition:
            prompt_parts.append(composition.format_composition())
        
        # Apply style bible overrides
        if style_bible_overrides:
            for key, value in style_bible_overrides.items():
                prompt_parts.append(value)
        
        # Combine all elements
        final_prompt = ", ".join(prompt_parts)
        
        logger.info(f"Generated cinematic prompt: {final_prompt[:100]}...")
        return final_prompt
    
    def _apply_style_bible_consistency(self) -> str:
        """Apply style bible elements for consistency"""
        consistency_elements = []
        
        # Color palette consistency
        if "color_palette" in self.style_bible:
            consistency_elements.append(self.style_bible["color_palette"])
        
        # Visual style consistency
        if "visual_style" in self.style_bible:
            consistency_elements.append(self.style_bible["visual_style"])
        
        # Lighting style consistency
        if "lighting_style" in self.style_bible:
            consistency_elements.append(self.style_bible["lighting_style"])
        
        # Quality standards
        if "quality_standards" in self.style_bible:
            consistency_elements.append(self.style_bible["quality_standards"])
        
        return ", ".join(consistency_elements)
    
    def _format_camera_settings(self, camera: CameraSettings) -> str:
        """Format camera settings for prompt"""
        parts = []
        
        # Lens type
        if camera.lens in self.technical_terms["lens_types"]:
            parts.append(self.technical_terms["lens_types"][camera.lens])
        
        # Aperture effect
        if camera.aperture in self.technical_terms["aperture_effects"]:
            parts.append(self.technical_terms["aperture_effects"][camera.aperture])
        
        # Technical specs
        parts.append(f"{camera.lens} lens, {camera.aperture}, {camera.iso}")
        
        return ", ".join(parts)
    
    def generate_style_variations(
        self,
        base_prompt: str,
        num_variations: int = 4
    ) -> List[str]:
        """Generate style variations of a base prompt"""
        variations = []
        
        # Lighting variations
        lighting_styles = [
            "dramatic chiaroscuro lighting",
            "soft diffused lighting",
            "high contrast lighting",
            "natural window lighting",
            "practical candlelight",
            "neon accent lighting"
        ]
        
        # Color grading variations
        color_grades = [
            "warm color grading",
            "cool color grading",
            "desaturated palette",
            "vibrant saturated colors",
            "cinematic teal and orange",
            "black and white with selective color"
        ]
        
        # Composition variations
        compositions = [
            "rule of thirds composition",
            "symmetrical composition",
            "Dutch angle shot",
            "center framing",
            "leading lines composition",
            "negative space composition"
        ]
        
        for i in range(num_variations):
            variation = base_prompt
            
            # Add lighting variation
            if i < len(lighting_styles):
                variation += f", {lighting_styles[i]}"
            
            # Add color grade variation
            if i < len(color_grades):
                variation += f", {color_grades[i]}"
            
            # Add composition variation
            if i < len(compositions):
                variation += f", {compositions[i]}"
            
            variations.append(variation)
        
        return variations
    
    def optimize_prompt_for_backend(
        self,
        prompt: str,
        backend_name: str
    ) -> str:
        """Optimize prompt for specific backend requirements"""
        
        # Backend-specific optimizations
        backend_optimizations = {
            "sdxl": {
                "add": "masterpiece, best quality, ultra detailed",
                "remove": [],
                "modifications": {
                    "negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark"
                }
            },
            "minimax": {
                "add": "professional photography, cinematic quality",
                "remove": ["masterpiece", "best quality"],
                "modifications": {}
            },
            "imagen": {
                "add": "photorealistic, high resolution",
                "remove": ["masterpiece", "ultra detailed"],
                "modifications": {}
            },
            "nano_banana": {
                "add": "artistic, creative interpretation",
                "remove": [],
                "modifications": {}
            }
        }
        
        if backend_name not in backend_optimizations:
            return prompt
        
        optimization = backend_optimizations[backend_name]
        optimized_prompt = prompt
        
        # Add quality boosters
        if "add" in optimization:
            optimized_prompt += f", {', '.join(optimization['add'])}"
        
        # Remove incompatible terms
        if "remove" in optimization:
            for term in optimization["remove"]:
                optimized_prompt = optimized_prompt.replace(term, "").strip()
        
        return optimized_prompt
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, List[str]]:
        """Validate prompt for quality and compatibility"""
        warnings = []
        
        # Check prompt length
        if len(prompt) > 2000:
            warnings.append("Prompt may be too long for some backends")
        
        # Check for conflicting terms
        conflicting_pairs = [
            ("f/1.4", "deep depth of field"),
            ("wide shot", "shallow depth of field"),
            ("symmetrical", "Dutch angle"),
            ("black and white", "vibrant saturated colors")
        ]
        
        for term1, term2 in conflicting_pairs:
            if term1.lower() in prompt.lower() and term2.lower() in prompt.lower():
                warnings.append(f"Conflicting terms: '{term1}' and '{term2}'")
        
        # Check for quality terms
        quality_terms = ["cinematic", "professional", "masterpiece", "high quality"]
        has_quality = any(term.lower() in prompt.lower() for term in quality_terms)
        
        if not has_quality:
            warnings.append("Consider adding quality enhancement terms")
        
        return len(warnings) == 0, warnings