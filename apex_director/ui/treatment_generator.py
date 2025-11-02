"""
APEX DIRECTOR Creative Treatment Generation System
Generates creative treatments for music videos based on input parameters
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
import random

from .input_validator import ProcessedInput

logger = logging.getLogger(__name__)


class TreatmentType(Enum):
    """Types of creative treatments"""
    NARRATIVE = "narrative"
    PERFORMANCE = "performance"
    CONCEPTUAL = "conceptual"
    ABSTRACT = "abstract"
    DOCUMENTARY = "documentary"
    COMMERCIAL = "commercial"
    EXPERIMENTAL = "experimental"


class VisualComplexity(Enum):
    """Visual complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CINEMATIC = "cinematic"


@dataclass
class SceneDefinition:
    """Represents the definition of a single scene in the treatment.

    Attributes:
        id: The unique identifier for the scene.
        start_time: The start time of the scene in seconds.
        duration: The duration of the scene in seconds.
        scene_type: The type of scene (e.g., intro, verse, chorus).
        description: A visual description of the scene.
        mood: The emotional mood of the scene.
        color_palette: A list of colors in the scene's color palette.
        camera_movement: The camera movement for the scene.
        lighting_style: The lighting style for the scene.
        visual_elements: A list of visual elements in the scene.
        transition_to_next: The transition to the next scene.
        beat_sync_points: A list of specific beats to sync with.
    """
    id: str
    start_time: float  # Start time in seconds
    duration: float    # Duration in seconds
    scene_type: str    # intro, verse, chorus, bridge, outro
    description: str   # Visual description of the scene
    mood: str          # Emotional mood
    color_palette: List[str]
    camera_movement: str
    lighting_style: str
    visual_elements: List[str]
    transition_to_next: str
    beat_sync_points: List[float]  # Specific beats to sync with


@dataclass
class VisualTreatment:
    """Represents a complete visual treatment for a music video.

    Attributes:
        id: The unique identifier for the treatment.
        project_name: The name of the project.
        audio_duration: The duration of the audio in seconds.
        treatment_type: The type of creative treatment.
        visual_complexity: The level of visual complexity.
        overall_concept: The overall concept of the treatment.
        scenes: A list of scene definitions.
        color_scheme: A dictionary defining the color scheme.
        style_keywords: A list of style keywords.
        technical_specs: A dictionary of technical specifications.
        creation_timestamp: The timestamp when the treatment was created.
        version: The version of the treatment.
        notes: Additional notes for the treatment.
    """
    id: str
    project_name: str
    audio_duration: float
    treatment_type: TreatmentType
    visual_complexity: VisualComplexity
    overall_concept: str
    scenes: List[SceneDefinition]
    color_scheme: Dict[str, Any]
    style_keywords: List[str]
    technical_specs: Dict[str, Any]
    creation_timestamp: datetime
    version: str = "1.0"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the VisualTreatment to a dictionary.

        Returns:
            A dictionary representation of the VisualTreatment.
        """
        data = asdict(self)
        data['treatment_type'] = self.treatment_type.value
        data['visual_complexity'] = self.visual_complexity.value
        data['creation_timestamp'] = self.creation_timestamp.isoformat()
        return data


class TreatmentGenerator:
    """A system for generating creative treatments for music videos.

    This class provides functionality for:
    - Generating a complete visual treatment based on input parameters
    - Saving and loading treatments to and from files
    - Getting a summary of a treatment
    """
    
    def __init__(self):
        """Initializes the TreatmentGenerator."""
        self.style_templates = self._load_style_templates()
        self.color_palettes = self._load_color_palettes()
        self.camera_movements = self._load_camera_movements()
        self.visual_elements = self._load_visual_elements()
        
        logger.info("Treatment Generator initialized")
    
    def _load_style_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load style templates for different visual styles"""
        return {
            'cinematic': {
                'treatment_types': [TreatmentType.NARRATIVE, TreatmentType.PERFORMANCE],
                'visual_complexity': VisualComplexity.CINEMATIC,
                'mood_options': ['dramatic', 'epic', 'intimate', 'mysterious', 'romantic'],
                'lighting_styles': ['cinematic', 'golden hour', 'neon', 'moody', 'high contrast'],
                'camera_movements': ['smooth tracking', 'handheld', 'steadi cam', 'dolly push'],
                'color_schemes': ['warm cinematic', 'cool cinematic', 'film noir', 'desaturated']
            },
            'anime': {
                'treatment_types': [TreatmentType.NARRATIVE, TreatmentType.CONCEPTUAL],
                'visual_complexity': VisualComplexity.COMPLEX,
                'mood_options': ['energetic', 'dreamy', 'dramatic', 'cute', 'epic'],
                'lighting_styles': ['bright anime', 'cell shaded', 'watercolor', 'dramatic shadows'],
                'camera_movements': ['dynamic angles', 'close ups', 'wide shots', 'rotating'],
                'color_schemes': ['vibrant anime', 'pastel anime', 'dark anime', 'monochrome']
            },
            'realistic': {
                'treatment_types': [TreatmentType.NARRATIVE, TreatmentType.DOCUMENTARY],
                'visual_complexity': VisualComplexity.MODERATE,
                'mood_options': ['authentic', 'raw', 'natural', 'intimate', 'powerful'],
                'lighting_styles': ['natural light', 'studio lighting', 'candlelight', 'window light'],
                'camera_movements': ['static shots', 'slow push in', 'follow shots', 'interview style'],
                'color_schemes': ['natural tones', 'warm realistic', 'cool realistic', 'black and white']
            },
            'artistic': {
                'treatment_types': [TreatmentType.CONCEPTUAL, TreatmentType.EXPERIMENTAL],
                'visual_complexity': VisualComplexity.COMPLEX,
                'mood_options': ['expressive', 'surreal', 'beautiful', 'unconventional', 'emotional'],
                'lighting_styles': ['dramatic chiaroscuro', 'colored gels', 'practicals', 'mixed'],
                'camera_movements': ['unusual angles', 'filtered shots', 'creative framing'],
                'color_schemes': ['artistic palette', 'complementary colors', 'monochromatic']
            },
            'futuristic': {
                'treatment_types': [TreatmentType.CONCEPTUAL, TreatmentType.EXPERIMENTAL],
                'visual_complexity': VisualComplexity.COMPLEX,
                'mood_options': ['tech', 'neon', 'sleek', 'digital', 'otherworldly'],
                'lighting_styles': ['neon lighting', 'LED strips', 'holographic', 'UV blacklight'],
                'camera_movements': ['floating camera', 'digital glitch', 'rotation', 'zoom'],
                'color_schemes': ['neon cyberpunk', 'blue tech', 'rainbow neon', 'holographic']
            },
            'vintage': {
                'treatment_types': [TreatmentType.PERFORMANCE, TreatmentType.NARRATIVE],
                'visual_complexity': VisualComplexity.MODERATE,
                'mood_options': ['nostalgic', 'warm', 'sepia', 'classic', 'retro'],
                'lighting_styles': ['vintage lighting', 'practical lamps', 'softbox', 'film grain'],
                'camera_movements': ['vintage style', 'handheld', 'slow movements'],
                'color_schemes': ['sepia tones', 'vintage film', 'warm vintage', 'black and white vintage']
            },
            'minimalist': {
                'treatment_types': [TreatmentType.CONCEPTUAL, TreatmentType.PERFORMANCE],
                'visual_complexity': VisualComplexity.SIMPLE,
                'mood_options': ['clean', 'focused', 'simple', 'elegant', 'pure'],
                'lighting_styles': ['simple lighting', 'single source', 'soft lighting'],
                'camera_movements': ['static shots', 'slow zoom', 'minimal movement'],
                'color_schemes': ['monochrome', 'single color', 'limited palette', 'black and white']
            },
            'surreal': {
                'treatment_types': [TreatmentType.EXPERIMENTAL, TreatmentType.CONCEPTUAL],
                'visual_complexity': VisualComplexity.COMPLEX,
                'mood_options': ['unreal', 'dreamlike', 'impossible', 'distorted', 'magical'],
                'lighting_styles': ['impossible lighting', 'multiple sources', 'colored shadows'],
                'camera_movements': ['floating', 'disorienting', 'impossible angles'],
                'color_schemes': ['surreal palette', 'impossible colors', 'dream colors']
            }
        }
    
    def _load_color_palettes(self) -> Dict[str, List[str]]:
        """Load predefined color palettes"""
        return {
            'warm_cinematic': ['#8B4513', '#D2691E', '#FFD700', '#FF4500', '#800020'],
            'cool_cinematic': ['#191970', '#4169E1', '#708090', '#2F4F4F', '#87CEEB'],
            'neon_cyberpunk': ['#00FFFF', '#FF00FF', '#39FF14', '#FF073A', '#FFFF00'],
            'vintage_sepia': ['#D2B48C', '#8B7355', '#A0522D', '#CD853F', '#DEB887'],
            'pastel_anime': ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C'],
            'natural_tones': ['#8FBC8F', '#CD853F', '#F5DEB3', '#D2691E', '#696969'],
            'monochrome': ['#000000', '#404040', '#808080', '#C0C0C0', '#FFFFFF'],
            'complementary': ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF'],
            'earth_tones': ['#8B4513', '#CD853F', '#D2B48C', '#F4A460', '#A0522D'],
            'ocean_blues': ['#000080', '#0000CD', '#4169E1', '#1E90FF', '#87CEEB']
        }
    
    def _load_camera_movements(self) -> List[str]:
        """Load available camera movements"""
        return [
            'static shot',
            'slow zoom in',
            'slow zoom out',
            'tracking shot left',
            'tracking shot right',
            'handheld subtle',
            'handheld dramatic',
            'dolly push in',
            'dolly pull out',
            'crane up',
            'crane down',
            'tilt up',
            'tilt down',
            'pan left',
            'pan right',
            'rotating around subject',
            'floating camera',
            'follow subject left',
            'follow subject right',
            'close-up push in',
            'wide shot pull out'
        ]
    
    def _load_visual_elements(self) -> Dict[str, List[str]]:
        """Load visual elements by category"""
        return {
            'particles': ['sparkles', 'snow', 'rain', 'dust motes', 'fireflies', 'confetti'],
            'atmospheric': ['fog', 'smoke', 'steam', 'mist', 'clouds', 'aurora'],
            'geometric': ['lines', 'circles', 'triangles', 'spirals', 'fractals', 'grids'],
            'nature': ['flowers', 'leaves', 'water drops', 'waves', 'reflections', 'shadows'],
            'abstract': ['geometric shapes', 'color swirls', 'light beams', 'energy fields'],
            'urban': ['neon signs', 'streetlights', 'buildings', 'bridges', 'windows'],
            'fantasy': ['magical glow', 'crystal formations', 'ethereal mist', 'star fields'],
            'tech': ['digital displays', 'circuit patterns', 'data streams', 'holograms']
        }
    
    async def generate_treatment(self, 
                                processed_input: ProcessedInput,
                                custom_requirements: Optional[Dict[str, Any]] = None) -> VisualTreatment:
        """Generates a complete visual treatment.

        Args:
            processed_input: The validated and processed input data.
            custom_requirements: Optional custom requirements or overrides.

        Returns:
            A VisualTreatment object with the complete treatment.
        """
        try:
            logger.info(f"Generating treatment for project: {processed_input.project_name}")
            
            # Apply custom requirements if provided
            requirements = custom_requirements or {}
            
            # Determine treatment parameters
            treatment_type = self._determine_treatment_type(processed_input, requirements)
            visual_complexity = self._determine_visual_complexity(processed_input, requirements)
            style_template = self.style_templates.get(processed_input.visual_style, self.style_templates['cinematic'])
            
            # Generate overall concept
            overall_concept = self._generate_overall_concept(
                processed_input, treatment_type, style_template, requirements
            )
            
            # Create scene breakdown
            scenes = self._generate_scene_breakdown(
                processed_input, treatment_type, style_template, overall_concept
            )
            
            # Generate color scheme
            color_scheme = self._generate_color_scheme(processed_input, style_template, requirements)
            
            # Generate style keywords
            style_keywords = self._generate_style_keywords(
                processed_input, treatment_type, style_template
            )
            
            # Technical specifications
            technical_specs = self._generate_technical_specs(processed_input)
            
            treatment = VisualTreatment(
                id=str(uuid.uuid4()),
                project_name=processed_input.project_name,
                audio_duration=processed_input.duration_seconds,
                treatment_type=treatment_type,
                visual_complexity=visual_complexity,
                overall_concept=overall_concept,
                scenes=scenes,
                color_scheme=color_scheme,
                style_keywords=style_keywords,
                technical_specs=technical_specs,
                creation_timestamp=datetime.utcnow(),
                notes=self._generate_treatment_notes(processed_input, treatment_type)
            )
            
            logger.info(f"Treatment generated successfully: {len(scenes)} scenes")
            return treatment
            
        except Exception as e:
            logger.error(f"Error generating treatment: {e}")
            raise ValueError(f"Failed to generate creative treatment: {str(e)}")
    
    def _determine_treatment_type(self, 
                                  processed_input: ProcessedInput,
                                  requirements: Dict[str, Any]) -> TreatmentType:
        """Determine the most appropriate treatment type"""
        user_preference = requirements.get('treatment_type')
        if user_preference:
            try:
                return TreatmentType(user_preference)
            except ValueError:
                pass
        
        # Auto-determine based on concept description keywords
        description = processed_input.concept_description.lower()
        
        narrative_keywords = ['story', 'narrative', 'plot', 'character', 'journey']
        performance_keywords = ['performance', 'concert', 'live', 'band', 'singer']
        conceptual_keywords = ['concept', 'artistic', 'creative', 'abstract', 'metaphor']
        abstract_keywords = ['abstract', 'experimental', 'avant-garde', 'surreal']
        
        if any(keyword in description for keyword in narrative_keywords):
            return TreatmentType.NARRATIVE
        elif any(keyword in description for keyword in performance_keywords):
            return TreatmentType.PERFORMANCE
        elif any(keyword in description for keyword in abstract_keywords):
            return TreatmentType.ABSTRACT
        elif any(keyword in description for keyword in conceptual_keywords):
            return TreatmentType.CONCEPTUAL
        else:
            # Default based on visual style
            style_templates = self.style_templates.get(processed_input.visual_style, {})
            template_types = style_templates.get('treatment_types', [TreatmentType.CONCEPTUAL])
            return template_types[0] if template_types else TreatmentType.CONCEPTUAL
    
    def _determine_visual_complexity(self, 
                                   processed_input: ProcessedInput,
                                   requirements: Dict[str, Any]) -> VisualComplexity:
        """Determine appropriate visual complexity"""
        user_preference = requirements.get('visual_complexity')
        if user_preference:
            try:
                return VisualComplexity(user_preference)
            except ValueError:
                pass
        
        # Auto-determine based on audio duration and visual style
        duration = processed_input.duration_seconds
        
        # Shorter videos can handle higher complexity
        if duration < 60:  # Under 1 minute
            if processed_input.visual_style in ['anime', 'artistic', 'surreal', 'futuristic']:
                return VisualComplexity.COMPLEX
            else:
                return VisualComplexity.MODERATE
        elif duration > 300:  # Over 5 minutes
            return VisualComplexity.SIMPLE
        else:
            return VisualComplexity.MODERATE
    
    def _generate_overall_concept(self, 
                                processed_input: ProcessedInput,
                                treatment_type: TreatmentType,
                                style_template: Dict[str, Any],
                                requirements: Dict[str, Any]) -> str:
        """Generate overall conceptual description"""
        
        # Base concept from user description
        base_concept = processed_input.concept_description
        
        # Enhanced concept based on treatment type
        if treatment_type == TreatmentType.NARRATIVE:
            concept = f"A narrative-driven visual story: {base_concept}. The video follows a cohesive storyline that complements the musical journey, using cinematic techniques to create emotional depth and visual storytelling."
        
        elif treatment_type == TreatmentType.PERFORMANCE:
            concept = f"A dynamic performance-focused video: {base_concept}. The visuals showcase the music performance with dynamic angles, energy-capturing shots, and performance-enhancing visuals that sync with the musical rhythms."
        
        elif treatment_type == TreatmentType.CONCEPTUAL:
            concept = f"A conceptual artistic exploration: {base_concept}. The video uses symbolic imagery, creative visual metaphors, and artistic interpretation to express the music's themes and emotions."
        
        elif treatment_type == TreatmentType.ABSTRACT:
            concept = f"An abstract visual interpretation: {base_concept}. The video presents non-representational visuals that flow with the music, creating an immersive audiovisual experience through color, form, and motion."
        
        elif treatment_type == TreatmentType.DOCUMENTARY:
            concept = f"A documentary-style presentation: {base_concept}. The video captures authentic moments, behind-the-scenes elements, and real-world contexts that surround the music creation."
        
        elif treatment_type == TreatmentType.EXPERIMENTAL:
            concept = f"An experimental visual journey: {base_concept}. The video pushes creative boundaries with unconventional techniques, innovative visuals, and unique artistic expressions."
        
        else:
            concept = base_concept
        
        # Add style-specific enhancement
        visual_style = processed_input.visual_style
        if visual_style == 'cinematic':
            concept += " The aesthetic is cinematic with film-like quality, dramatic lighting, and professional cinematography."
        elif visual_style == 'anime':
            concept += " The visual style incorporates anime aesthetics with distinctive character designs, vibrant colors, and dynamic animation-inspired elements."
        elif visual_style == 'futuristic':
            concept += " The visuals embrace futuristic themes with high-tech elements, digital effects, and contemporary sci-fi aesthetics."
        
        return concept
    
    def _generate_scene_breakdown(self, 
                                processed_input: ProcessedInput,
                                treatment_type: TreatmentType,
                                style_template: Dict[str, Any],
                                overall_concept: str) -> List[SceneDefinition]:
        """Generate detailed scene breakdown"""
        
        duration = processed_input.duration_seconds
        
        # Determine number of scenes based on duration
        if duration < 60:
            scene_count = max(3, int(duration / 10))  # At least 3 scenes, ~10s each
        elif duration < 180:
            scene_count = max(5, int(duration / 15))  # ~15s each
        else:
            scene_count = max(8, int(duration / 20))  # ~20s each
        
        scenes = []
        current_time = 0
        scene_duration = duration / scene_count
        
        # Scene types distribution
        scene_types = ['intro', 'verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus', 'outro']
        
        for i in range(scene_count):
            scene_type = scene_types[i % len(scene_types)]
            
            # Adjust duration for intro/outro
            if scene_type == 'intro':
                duration = min(scene_duration * 0.7, 8)
            elif scene_type == 'outro':
                duration = min(scene_duration * 0.8, 10)
            else:
                duration = scene_duration
            
            scene = self._create_scene(
                scene_id=str(uuid.uuid4()),
                start_time=current_time,
                duration=duration,
                scene_type=scene_type,
                processed_input=processed_input,
                treatment_type=treatment_type,
                style_template=style_template,
                scene_index=i,
                total_scenes=scene_count
            )
            
            scenes.append(scene)
            current_time += duration
        
        return scenes
    
    def _create_scene(self, 
                     scene_id: str,
                     start_time: float,
                     duration: float,
                     scene_type: str,
                     processed_input: ProcessedInput,
                     treatment_type: TreatmentType,
                     style_template: Dict[str, Any],
                     scene_index: int,
                     total_scenes: int) -> SceneDefinition:
        """Create a single scene definition"""
        
        # Determine mood for this scene
        mood_options = style_template.get('mood_options', ['neutral'])
        mood = random.choice(mood_options)
        
        # Determine color palette
        color_schemes = style_template.get('color_schemes', ['neutral'])
        chosen_scheme = random.choice(color_schemes)
        color_palette = self.color_palettes.get(chosen_scheme, ['#FFFFFF', '#000000'])
        
        # Select camera movement
        camera_movements = style_template.get('camera_movements', ['static shot'])
        camera_movement = random.choice(camera_movements)
        
        # Select lighting style
        lighting_styles = style_template.get('lighting_styles', ['standard lighting'])
        lighting_style = random.choice(lighting_styles)
        
        # Select visual elements
        visual_elements = []
        for category, elements in self.visual_elements.items():
            if random.random() < 0.3:  # 30% chance for each category
                visual_elements.append(random.choice(elements))
        
        # Determine transition to next scene
        transition_options = [
            'cut', 'dissolve', 'fade to black', 'fade from black', 'wipe left', 
            'wipe right', 'zoom transition', 'match cut', 'rhythmic cut'
        ]
        transition_to_next = random.choice(transition_options)
        
        # Generate beat sync points (simplified)
        beat_sync_points = []
        beats_per_second = 2  # Assume 120 BPM
        for beat_time in range(int(start_time), int(start_time + duration), max(1, int(60/beats_per_second))):
            beat_sync_points.append(beat_time - start_time)
        
        # Generate scene description
        scene_description = self._generate_scene_description(
            scene_type, processed_input, treatment_type, mood, visual_elements
        )
        
        return SceneDefinition(
            id=scene_id,
            start_time=start_time,
            duration=duration,
            scene_type=scene_type,
            description=scene_description,
            mood=mood,
            color_palette=color_palette,
            camera_movement=camera_movement,
            lighting_style=lighting_style,
            visual_elements=visual_elements,
            transition_to_next=transition_to_next,
            beat_sync_points=beat_sync_points
        )
    
    def _generate_scene_description(self,
                                   scene_type: str,
                                   processed_input: ProcessedInput,
                                   treatment_type: TreatmentType,
                                   mood: str,
                                   visual_elements: List[str]) -> str:
        """Generate description for a scene"""
        
        base_description = processed_input.concept_description
        
        if scene_type == 'intro':
            description = f"Opening scene establishing the visual world: {base_description}. The mood is {mood}, setting the tone for the entire video with introductory visuals."
        elif scene_type == 'verse':
            description = f"Verse scene exploring the narrative: {base_description}. The mood is {mood}, with intimate or narrative-focused visuals that support the lyrical content."
        elif scene_type == 'chorus':
            description = f"Chorus scene with high energy visuals: {base_description}. The mood is {mood}, featuring dynamic and impactful visuals that match the musical climax."
        elif scene_type == 'bridge':
            description = f"Bridge scene providing contrast: {base_description}. The mood is {mood}, offering a different perspective or visual departure from verse/chorus."
        elif scene_type == 'outro':
            description = f"Closing scene bringing resolution: {base_description}. The mood is {mood}, providing visual closure and reflection on the overall narrative."
        else:
            description = f"Scene exploring {base_description} with {mood} mood."
        
        # Add visual elements if present
        if visual_elements:
            description += f" Incorporating: {', '.join(visual_elements)}."
        
        return description
    
    def _generate_color_scheme(self, 
                             processed_input: ProcessedInput,
                             style_template: Dict[str, Any],
                             requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate color scheme for the treatment"""
        
        # User color preference
        user_colors = requirements.get('color_palette')
        if user_colors and isinstance(user_colors, list):
            return {
                'primary': user_colors[:3],  # First 3 as primary
                'secondary': user_colors[3:6] if len(user_colors) > 3 else user_colors[:3],
                'accent': user_colors[6:] if len(user_colors) > 6 else user_colors[-1:] if user_colors else [],
                'type': 'custom',
                'scheme_name': 'Custom User Palette'
            }
        
        # Use style template colors
        color_schemes = style_template.get('color_schemes', ['neutral'])
        scheme_name = random.choice(color_schemes)
        palette = self.color_palettes.get(scheme_name, ['#FFFFFF', '#000000'])
        
        return {
            'primary': palette[:3],
            'secondary': palette[3:6] if len(palette) > 3 else palette[:3],
            'accent': palette[6:] if len(palette) > 6 else [palette[-1]] if palette else [],
            'type': scheme_name,
            'scheme_name': scheme_name.replace('_', ' ').title(),
            'description': f"Color palette based on {scheme_name} aesthetic"
        }
    
    def _generate_style_keywords(self, 
                                processed_input: ProcessedInput,
                                treatment_type: TreatmentType,
                                style_template: Dict[str, Any]) -> List[str]:
        """Generate style keywords for the treatment"""
        
        keywords = []
        
        # Visual style keywords
        visual_style = processed_input.visual_style
        keywords.append(visual_style)
        
        # Treatment type keywords
        keywords.append(treatment_type.value)
        
        # Style template specific keywords
        mood_options = style_template.get('mood_options', [])
        if mood_options:
            keywords.append(random.choice(mood_options))
        
        lighting_styles = style_template.get('lighting_styles', [])
        if lighting_styles:
            keywords.append(random.choice(lighting_styles))
        
        # Add some general aesthetic keywords
        aesthetic_keywords = [
            'cohesive', 'professional', 'cinematic', 'artistic', 'engaging',
            'memorable', 'atmospheric', 'stylized', 'polished', 'creative'
        ]
        keywords.extend(random.sample(aesthetic_keywords, min(3, len(aesthetic_keywords))))
        
        # Ensure uniqueness
        return list(set(keywords))
    
    def _generate_technical_specs(self, processed_input: ProcessedInput) -> Dict[str, Any]:
        """Generate technical specifications"""
        return {
            'output_resolution': processed_input.output_resolution,
            'frame_rate': processed_input.frame_rate,
            'aspect_ratio': f"{processed_input.output_resolution[0]}:{processed_input.output_resolution[1]}",
            'color_space': 'sRGB',
            'bit_depth': '8-bit',
            'codec': 'H.264',
            'render_quality': 'high',
            'estimated_render_time_hours': max(1, int(processed_input.duration_seconds / 60)),
            'storage_requirement_gb': max(1, (processed_input.duration_seconds * processed_input.frame_rate * 100) // (1024 * 1024))
        }
    
    def _generate_treatment_notes(self, 
                                 processed_input: ProcessedInput, 
                                 treatment_type: TreatmentType) -> str:
        """Generate production notes and suggestions"""
        notes = f"Creative Treatment Notes for {processed_input.project_name}\n\n"
        
        if treatment_type == TreatmentType.NARRATIVE:
            notes += "• Focus on consistent storyline throughout\n"
            notes += "• Ensure character continuity between scenes\n"
            notes += "• Use narrative pacing to match musical structure\n\n"
        
        elif treatment_type == TreatmentType.PERFORMANCE:
            notes += "• Capture energy and emotion of live performance\n"
            notes += "• Use dynamic camera angles to showcase different perspectives\n"
            notes += "• Ensure good audio sync with visual elements\n\n"
        
        elif treatment_type == TreatmentType.CONCEPTUAL:
            notes += "• Emphasize symbolic and metaphorical imagery\n"
            notes += "• Create visual metaphors that support musical themes\n"
            notes += "• Use creative transitions between concepts\n\n"
        
        # Style-specific notes
        if processed_input.visual_style == 'cinematic':
            notes += "• Maintain cinematic quality throughout\n"
            notes += "• Pay attention to lighting continuity\n"
            notes += "• Consider aspect ratio and framing for cinematic feel\n\n"
        elif processed_input.visual_style == 'anime':
            notes += "• Ensure art style consistency across all scenes\n"
            notes += "• Use vibrant colors and dynamic compositions\n"
            notes += "• Consider character design consistency\n\n"
        
        notes += f"• Total estimated scenes: To be determined during storyboarding\n"
        notes += f"• Recommended post-production time: {max(2, int(processed_input.duration_seconds / 60) + 4)} hours\n"
        notes += f"• Review checkpoints at: 25%, 50%, 75% completion\n"
        
        return notes
    
    def save_treatment(self, treatment: VisualTreatment, file_path: str) -> bool:
        """Saves a treatment to a JSON file.

        Args:
            treatment: The VisualTreatment object to save.
            file_path: The path to the file to save the treatment to.

        Returns:
            True if the treatment was successfully saved, False otherwise.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(treatment.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Treatment saved to: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save treatment: {e}")
            return False
    
    def load_treatment(self, file_path: str) -> Optional[VisualTreatment]:
        """Loads a treatment from a JSON file.

        Args:
            file_path: The path to the file to load the treatment from.

        Returns:
            A VisualTreatment object, or None if the treatment could not be loaded.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert scene data back to SceneDefinition objects
            scenes = []
            for scene_data in data.get('scenes', []):
                scene = SceneDefinition(**scene_data)
                scenes.append(scene)
            
            treatment = VisualTreatment(
                id=data['id'],
                project_name=data['project_name'],
                audio_duration=data['audio_duration'],
                treatment_type=TreatmentType(data['treatment_type']),
                visual_complexity=VisualComplexity(data['visual_complexity']),
                overall_concept=data['overall_concept'],
                scenes=scenes,
                color_scheme=data['color_scheme'],
                style_keywords=data['style_keywords'],
                technical_specs=data['technical_specs'],
                creation_timestamp=datetime.fromisoformat(data['creation_timestamp']),
                version=data.get('version', '1.0'),
                notes=data.get('notes', '')
            )
            
            logger.info(f"Treatment loaded from: {file_path}")
            return treatment
            
        except Exception as e:
            logger.error(f"Failed to load treatment: {e}")
            return None
    
    def get_treatment_summary(self, treatment: VisualTreatment) -> Dict[str, Any]:
        """Gets summary information about a treatment.

        Args:
            treatment: The VisualTreatment object to get the summary for.

        Returns:
            A dictionary of summary information.
        """
        return {
            'id': treatment.id,
            'project_name': treatment.project_name,
            'treatment_type': treatment.treatment_type.value,
            'visual_complexity': treatment.visual_complexity.value,
            'audio_duration': f"{treatment.audio_duration:.1f} seconds",
            'scene_count': len(treatment.scenes),
            'style_keywords': treatment.style_keywords,
            'color_scheme': treatment.color_scheme.get('scheme_name', 'Unknown'),
            'creation_date': treatment.creation_timestamp.strftime('%Y-%m-%d %H:%M'),
            'version': treatment.version,
            'estimated_render_time': treatment.technical_specs.get('estimated_render_time_hours', 'Unknown')
        }
