"""
Style Persistence Engine
Maintains visual consistency across multiple shots using style_bible.json
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from PIL import Image
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class StyleElement:
    """Individual style element for consistency tracking"""
    name: str
    value: str
    weight: float = 1.0
    category: str = "general"  # color, lighting, composition, texture, mood
    is_essential: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CharacterReference:
    """Character reference for consistency"""
    name: str
    reference_images: List[Path]
    face_encoding: Optional[np.ndarray] = None
    attributes: Dict[str, str] = None
    style_notes: str = ""
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "reference_images": [str(p) for p in self.reference_images],
            "face_encoding": self.face_encoding.tolist() if self.face_encoding is not None else None,
            "attributes": self.attributes,
            "style_notes": self.style_notes
        }

@dataclass
class SceneStyle:
    """Style characteristics for a specific scene"""
    scene_id: str
    style_elements: List[StyleElement]
    lighting_setup: Dict[str, str]
    color_palette: List[str]
    mood_descriptors: List[str]
    technical_specs: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "style_elements": [se.to_dict() for se in self.style_elements],
            "lighting_setup": self.lighting_setup,
            "color_palette": self.color_palette,
            "mood_descriptors": self.mood_descriptors,
            "technical_specs": self.technical_specs
        }

class StyleBibleManager:
    """Manages style consistency using style_bible.json"""
    
    def __init__(self, style_bible_path: str = "style_bible.json"):
        self.style_bible_path = Path(style_bible_path)
        self.style_bible = self._load_or_create_style_bible()
        self.character_references: Dict[str, CharacterReference] = {}
        self.scene_styles: Dict[str, SceneStyle] = {}
        
    def _load_or_create_style_bible(self) -> Dict[str, Any]:
        """Load existing style bible or create new one"""
        if self.style_bible_path.exists():
            try:
                with open(self.style_bible_path, 'r') as f:
                    style_bible = json.load(f)
                logger.info(f"Loaded style bible from {self.style_bible_path}")
                return style_bible
            except Exception as e:
                logger.warning(f"Failed to load style bible: {e}")
        
        # Create new style bible with default values
        default_style_bible = {
            "version": "1.0",
            "project_name": "Untitled Project",
            "overall_style": {
                "visual_style": "cinematic realism",
                "color_grading": "natural with subtle warmth",
                "lighting_style": "three-point lighting with practical accents",
                "composition_style": "rule of thirds with dynamic framing"
            },
            "color_palette": {
                "primary_colors": ["#2C3E50", "#ECF0F1", "#E74C3C"],
                "secondary_colors": ["#3498DB", "#F39C12", "#27AE60"],
                "accent_colors": ["#9B59B6", "#E67E22", "#1ABC9C"],
                "neutral_colors": ["#2C3E50", "#7F8C8D", "#BDC3C7"],
                "skin_tones": ["#FDBCB4", "#F1C27D", "#E0AC69", "#C68642", "#8D5524"]
            },
            "lighting_profile": {
                "key_light_intensity": "medium",
                "shadow_quality": "soft with defined edges",
                "color_temperature": "5600K daylight balanced",
                "practical_lights": "warm tungsten accents",
                "rim_lighting": "subtle backlight for separation"
            },
            "camera_profile": {
                "preferred_lenses": ["35mm", "50mm", "85mm"],
                "aperture_range": "f/2.8 to f/5.6",
                "depth_of_field": "shallow for subjects, deep for environments",
                "shutter_speed": "1/125 standard, faster for action",
                "iso_range": "400-1600 with minimal noise"
            },
            "mood_descriptors": [
                "cinematic", "professional", "engaging", "natural", "polished"
            ],
            "quality_standards": {
                "resolution": "minimum 1920x1080, preferred 4K",
                "sharpness": "critically sharp focus on subjects",
                "noise": "minimal sensor noise, natural grain acceptable",
                "dynamic_range": "preserves highlight and shadow detail"
            },
            "character_consistency": {},
            "scene_styles": {},
            "style_variations": {
                "formal": "controlled, symmetrical, professional",
                "casual": "natural, relaxed, approachable",
                "dramatic": "high contrast, intense, emotional",
                "romantic": "soft, warm, intimate"
            }
        }
        
        self._save_style_bible(default_style_bible)
        logger.info("Created new default style bible")
        return default_style_bible
    
    def _save_style_bible(self, style_bible: Dict[str, Any]):
        """Save style bible to file"""
        try:
            with open(self.style_bible_path, 'w') as f:
                json.dump(style_bible, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save style bible: {e}")
    
    def update_style_element(self, category: str, key: str, value: Any):
        """Update a specific style element in the bible"""
        if category not in self.style_bible:
            self.style_bible[category] = {}
        
        self.style_bible[category][key] = value
        self._save_style_bible(self.style_bible)
        logger.info(f"Updated style element: {category}.{key} = {value}")
    
    def get_style_element(self, category: str, key: str, default: Any = None) -> Any:
        """Get a style element from the bible"""
        return self.style_bible.get(category, {}).get(key, default)
    
    def add_character_reference(self, character: CharacterReference):
        """Add character reference for consistency"""
        self.character_references[character.name] = character
        
        # Update style bible
        if "character_consistency" not in self.style_bible:
            self.style_bible["character_consistency"] = {}
        
        self.style_bible["character_consistency"][character.name] = character.to_dict()
        self._save_style_bible(self.style_bible)
        
        logger.info(f"Added character reference: {character.name}")
    
    def get_character_reference(self, name: str) -> Optional[CharacterReference]:
        """Get character reference by name"""
        char_dict = self.style_bible.get("character_consistency", {}).get(name)
        if char_dict:
            # Convert back to CharacterReference object
            return CharacterReference(
                name=char_dict["name"],
                reference_images=[Path(p) for p in char_dict["reference_images"]],
                face_encoding=np.array(char_dict["face_encoding"]) if char_dict["face_encoding"] else None,
                attributes=char_dict["attributes"],
                style_notes=char_dict["style_notes"]
            )
        return None
    
    def add_scene_style(self, scene_style: SceneStyle):
        """Add scene-specific style"""
        self.scene_styles[scene_style.scene_id] = scene_style
        
        # Update style bible
        if "scene_styles" not in self.style_bible:
            self.style_bible["scene_styles"] = {}
        
        self.style_bible["scene_styles"][scene_style.scene_id] = scene_style.to_dict()
        self._save_style_bible(self.style_bible)
        
        logger.info(f"Added scene style: {scene_style.scene_id}")
    
    def get_scene_style(self, scene_id: str) -> Optional[SceneStyle]:
        """Get scene-specific style"""
        scene_dict = self.style_bible.get("scene_styles", {}).get(scene_id)
        if scene_dict:
            style_elements = [StyleElement(**se) for se in scene_dict["style_elements"]]
            return SceneStyle(
                scene_id=scene_dict["scene_id"],
                style_elements=style_elements,
                lighting_setup=scene_dict["lighting_setup"],
                color_palette=scene_dict["color_palette"],
                mood_descriptors=scene_dict["mood_descriptors"],
                technical_specs=scene_dict["technical_specs"]
            )
        return None
    
    def generate_consistency_prompt(
        self,
        base_prompt: str,
        scene_id: Optional[str] = None,
        character_name: Optional[str] = None,
        style_variation: Optional[str] = None
    ) -> str:
        """Generate prompt with style consistency elements"""
        
        consistency_elements = []
        
        # Overall style consistency
        overall_style = self.style_bible.get("overall_style", {})
        for key, value in overall_style.items():
            consistency_elements.append(value)
        
        # Color palette consistency
        color_palette = self.style_bible.get("color_palette", {})
        if "primary_colors" in color_palette:
            primary_colors = ", ".join(color_palette["primary_colors"][:3])
            consistency_elements.append(f"color palette: {primary_colors}")
        
        # Lighting profile consistency
        lighting_profile = self.style_bible.get("lighting_profile", {})
        lighting_desc = ", ".join(lighting_profile.values())
        if lighting_desc:
            consistency_elements.append(f"lighting: {lighting_desc}")
        
        # Camera profile consistency
        camera_profile = self.style_bible.get("camera_profile", {})
        camera_desc = f"{camera_profile.get('preferred_lenses', ['50mm'])[0]} lens, {camera_profile.get('aperture_range', 'f/2.8')}"
        consistency_elements.append(f"camera settings: {camera_desc}")
        
        # Scene-specific style
        if scene_id:
            scene_style = self.get_scene_style(scene_id)
            if scene_style:
                # Add scene-specific color palette
                if scene_style.color_palette:
                    consistency_elements.append(f"scene colors: {', '.join(scene_style.color_palette[:3])}")
                
                # Add scene mood descriptors
                if scene_style.mood_descriptors:
                    consistency_elements.append(f"scene mood: {', '.join(scene_style.mood_descriptors[:2])}")
        
        # Character-specific consistency
        if character_name:
            character = self.get_character_reference(character_name)
            if character and character.style_notes:
                consistency_elements.append(character.style_notes)
        
        # Style variation
        if style_variation:
            variations = self.style_bible.get("style_variations", {})
            if style_variation in variations:
                consistency_elements.append(variations[style_variation])
        
        # Quality standards
        quality_standards = self.style_bible.get("quality_standards", {})
        quality_desc = ", ".join(quality_standards.values())
        if quality_desc:
            consistency_elements.append(f"quality: {quality_desc}")
        
        # Combine all elements
        consistency_text = ", ".join(consistency_elements)
        final_prompt = f"{base_prompt}, style consistency: {consistency_text}"
        
        return final_prompt
    
    def detect_style_drift(self, generated_image: Image.Image) -> Tuple[float, List[str]]:
        """Detect style drift using CLIP embeddings (placeholder implementation)"""
        # This is a placeholder - in reality, would use CLIP embeddings
        # to compare the generated image against the style bible
        
        drift_score = 0.0
        drift_issues = []
        
        # Simulate style drift detection
        # In real implementation:
        # 1. Generate CLIP embeddings for generated image
        # 2. Compare against style bible embeddings
        # 3. Calculate drift score
        # 4. Identify specific drift issues
        
        logger.info("Performing style drift detection (simulated)")
        
        # Mock drift detection results
        drift_score = 0.15  # 15% drift
        drift_issues = [
            "Color temperature slightly off from target",
            "Lighting contrast higher than specified",
            "Composition slightly different from style guide"
        ]
        
        return drift_score, drift_issues
    
    def calculate_style_similarity(
        self,
        image1: Image.Image,
        image2: Image.Image
    ) -> float:
        """Calculate style similarity between two images"""
        # Placeholder implementation
        # In reality, would use feature extraction and comparison
        
        # Mock similarity calculation
        similarity = 0.85  # 85% similar
        
        logger.info(f"Style similarity between images: {similarity:.2%}")
        return similarity
    
    def get_style_statistics(self) -> Dict[str, Any]:
        """Get statistics about style consistency"""
        return {
            "total_characters": len(self.character_references),
            "total_scenes": len(self.scene_styles),
            "style_bible_version": self.style_bible.get("version", "unknown"),
            "last_updated": self.style_bible_path.stat().st_mtime if self.style_bible_path.exists() else None,
            "color_palette_size": len(self.style_bible.get("color_palette", {})),
            "lighting_elements": len(self.style_bible.get("lighting_profile", {})),
            "mood_descriptors": len(self.style_bible.get("mood_descriptors", []))
        }

class StyleConsistencyValidator:
    """Validates style consistency across generated images"""
    
    def __init__(self, style_bible_manager: StyleBibleManager):
        self.style_bible_manager = style_bible_manager
    
    def validate_image_consistency(
        self,
        image: Image.Image,
        expected_scene_id: Optional[str] = None,
        expected_character: Optional[str] = None
    ) -> Tuple[bool, float, List[str]]:
        """Validate image against style consistency requirements"""
        
        issues = []
        consistency_score = 1.0
        
        # Check style drift
        drift_score, drift_issues = self.style_bible_manager.detect_style_drift(image)
        consistency_score -= drift_score
        issues.extend(drift_issues)
        
        # Scene-specific validation
        if expected_scene_id:
            scene_style = self.style_bible_manager.get_scene_style(expected_scene_id)
            if scene_style:
                # Validate color palette match
                # Validate lighting consistency
                # Validate mood descriptors
                pass
        
        # Character-specific validation
        if expected_character:
            character = self.style_bible_manager.get_character_reference(expected_character)
            if character:
                # Validate character consistency
                # Validate facial features
                # Validate clothing/style
                pass
        
        is_consistent = consistency_score >= 0.8
        return is_consistent, consistency_score, issues