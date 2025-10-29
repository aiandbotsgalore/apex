"""
APEX DIRECTOR - Cinematography and Narrative System
Professional filmmaking techniques for music video generation
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import math
import random


class ShotType(Enum):
    """Professional shot types"""
    EXTREME_WIDE_SHOT = "extreme_wide_shot"
    WIDE_SHOT = "wide_shot"
    MEDIUM_WIDE_SHOT = "medium_wide_shot"
    MEDIUM_SHOT = "medium_shot"
    MEDIUM_CLOSE_UP = "medium_close_up"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"


class CameraMovement(Enum):
    """Professional camera movements"""
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    TRUCK_LEFT = "truck_left"
    TRUCK_RIGHT = "truck_right"
    CRANE_UP = "crane_up"
    CRANE_DOWN = "crane_down"
    HANDHELD = "handheld"
    STEADICAM = "steadicam"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


class LightingSetup(Enum):
    """Professional lighting setups"""
    THREE_POINT = "three_point"
    REMBRANDT = "rembrandt"
    BUTTERFLY = "butterfly"
    SPLIT = "split"
    RIM_LIGHT = "rim_light"
    HIGH_KEY = "high_key"
    LOW_KEY = "low_key"
    CHIAROSCURO = "chiaroscuro"
    NATURAL = "natural"
    DRAMATIC = "dramatic"


class NarrativeAct(Enum):
    """Three-act structure"""
    ACT_1_SETUP = "act_1_setup"
    ACT_2_CONFRONTATION = "act_2_confrontation"
    ACT_3_RESOLUTION = "act_3_resolution"


@dataclass
class CompositionRules:
    """Professional composition rules"""
    rule_of_thirds: bool = True
    leading_lines: bool = False
    depth_of_field: str = "medium"  # shallow, medium, deep
    symmetry: bool = False
    framing: str = "medium_shot"
    horizon_line: str = "center"  # upper_third, center, lower_third
    negative_space: float = 0.2  # 0-1
    focal_point: Tuple[float, float] = (0.33, 0.33)  # Rule of thirds


@dataclass
class CameraSpec:
    """Professional camera specifications"""
    lens_focal_length: int = 50  # mm
    aperture: str = "f/2.8"
    iso: str = "ISO 800"
    shutter_speed: str = "1/50s"
    sensor_size: str = "full_frame"
    aspect_ratio: str = "16:9"
    
    @property
    def depth_of_field_factor(self) -> float:
        """Calculate depth of field factor from aperture"""
        f_number = float(self.aperture.replace('f/', ''))
        return 1.0 / f_number


@dataclass
class LightingSpec:
    """Professional lighting specification"""
    key_light: Dict[str, Any] = field(default_factory=dict)
    fill_light: Dict[str, Any] = field(default_factory=dict)
    rim_light: Dict[str, Any] = field(default_factory=dict)
    background_light: Dict[str, Any] = field(default_factory=dict)
    practicals: List[Dict[str, Any]] = field(default_factory=list)
    overall_mood: str = "neutral"
    color_temperature: int = 5600  # Kelvin
    
    def __post_init__(self):
        """Initialize default lighting"""
        if not self.key_light:
            self.key_light = {
                "intensity": 100,
                "angle": 45,
                "height": 45,
                "softness": 0.5,
                "color": "#FFFFFF"
            }
        
        if not self.fill_light:
            self.fill_light = {
                "intensity": 40,
                "angle": -45,
                "height": 30,
                "softness": 0.8,
                "color": "#F0F0F0"
            }


@dataclass
class Shot:
    """Complete shot specification"""
    shot_id: str
    shot_type: ShotType
    composition: CompositionRules
    camera_spec: CameraSpec
    lighting_spec: LightingSpec
    movement: CameraMovement
    duration: float  # seconds
    narrative_function: str  # establishing, reaction, reveal, etc.
    emotional_tone: str  # happy, sad, tense, peaceful, etc.
    visual_motifs: List[str] = field(default_factory=list)
    
    @property
    def cinematic_description(self) -> str:
        """Generate cinematic description for image generation"""
        components = []
        
        # Shot type
        components.append(f"{self.shot_type.value}")
        
        # Camera specs
        components.append(f"{self.camera_spec.lens_focal_length}mm lens")
        components.append(f"{self.camera_spec.aperture}")
        
        # Lighting mood
        components.append(f"{self.lighting_spec.overall_mood} lighting")
        
        # Composition
        if self.composition.rule_of_thirds:
            components.append("rule of thirds composition")
        if self.composition.leading_lines:
            components.append("dynamic leading lines")
        
        # Movement
        if self.movement != CameraMovement.STATIC:
            components.append(f"{self.movement.value} camera movement")
        
        return ", ".join(components)


class CinematographyEngine:
    """Professional cinematography system"""
    
    def __init__(self):
        self.shot_library: Dict[str, Shot] = {}
        self.visual_motifs: Dict[str, List[str]] = {}
        self.lighting_presets: Dict[str, LightingSpec] = {}
        self.composition_templates: Dict[str, CompositionRules] = {}
        
        self._initialize_presets()
    
    def _initialize_presets(self):
        """Initialize professional presets"""
        
        # Lighting presets
        self.lighting_presets["three_point"] = LightingSpec(
            key_light={"intensity": 100, "angle": 45, "height": 45, "softness": 0.5},
            fill_light={"intensity": 40, "angle": -45, "height": 30, "softness": 0.8},
            rim_light={"intensity": 60, "angle": 180, "height": 60, "softness": 0.3},
            overall_mood="professional"
        )
        
        self.lighting_presets["rembrandt"] = LightingSpec(
            key_light={"intensity": 80, "angle": 45, "height": 45, "softness": 0.3},
            fill_light={"intensity": 20, "angle": -45, "height": 30, "softness": 0.9},
            overall_mood="dramatic"
        )
        
        self.lighting_presets["high_key"] = LightingSpec(
            key_light={"intensity": 70, "angle": 30, "height": 30, "softness": 0.8},
            fill_light={"intensity": 60, "angle": -30, "height": 30, "softness": 0.8},
            background_light={"intensity": 50, "angle": 0, "height": 0, "softness": 1.0},
            overall_mood="bright"
        )
        
        self.lighting_presets["low_key"] = LightingSpec(
            key_light={"intensity": 90, "angle": 60, "height": 50, "softness": 0.2},
            fill_light={"intensity": 15, "angle": -60, "height": 20, "softness": 0.9},
            overall_mood="dark"
        )
        
        # Composition templates
        self.composition_templates["classic"] = CompositionRules(
            rule_of_thirds=True,
            leading_lines=False,
            depth_of_field="medium",
            framing="medium_shot"
        )
        
        self.composition_templates["dynamic"] = CompositionRules(
            rule_of_thirds=True,
            leading_lines=True,
            depth_of_field="shallow",
            framing="medium_shot"
        )
        
        self.composition_templates["intimate"] = CompositionRules(
            rule_of_thirds=False,
            symmetry=True,
            depth_of_field="shallow",
            framing="close_up"
        )
    
    def create_shot(self,
                   shot_id: str,
                   shot_type: ShotType,
                   emotional_tone: str,
                   narrative_function: str,
                   lighting_preset: str = "three_point",
                   composition_template: str = "classic",
                   duration: float = 3.0) -> Shot:
        """Create a professional shot specification"""
        
        # Get presets
        lighting = self.lighting_presets.get(lighting_preset, self.lighting_presets["three_point"])
        composition = self.composition_templates.get(composition_template, self.composition_templates["classic"])
        
        # Camera specs based on shot type
        camera_spec = self._get_camera_spec_for_shot_type(shot_type)
        
        # Select movement based on emotional tone
        movement = self._select_movement_for_emotion(emotional_tone)
        
        shot = Shot(
            shot_id=shot_id,
            shot_type=shot_type,
            composition=composition,
            camera_spec=camera_spec,
            lighting_spec=lighting,
            movement=movement,
            duration=duration,
            narrative_function=narrative_function,
            emotional_tone=emotional_tone
        )
        
        self.shot_library[shot_id] = shot
        return shot
    
    def _get_camera_spec_for_shot_type(self, shot_type: ShotType) -> CameraSpec:
        """Get appropriate camera specs for shot type"""
        
        specs = {
            ShotType.EXTREME_WIDE_SHOT: CameraSpec(lens_focal_length=14, aperture="f/8.0"),
            ShotType.WIDE_SHOT: CameraSpec(lens_focal_length=24, aperture="f/5.6"),
            ShotType.MEDIUM_WIDE_SHOT: CameraSpec(lens_focal_length=35, aperture="f/4.0"),
            ShotType.MEDIUM_SHOT: CameraSpec(lens_focal_length=50, aperture="f/2.8"),
            ShotType.MEDIUM_CLOSE_UP: CameraSpec(lens_focal_length=85, aperture="f/2.0"),
            ShotType.CLOSE_UP: CameraSpec(lens_focal_length=100, aperture="f/1.8"),
            ShotType.EXTREME_CLOSE_UP: CameraSpec(lens_focal_length=135, aperture="f/1.4")
        }
        
        return specs.get(shot_type, CameraSpec())
    
    def _select_movement_for_emotion(self, emotional_tone: str) -> CameraMovement:
        """Select camera movement based on emotional tone"""
        
        movement_map = {
            "happy": [CameraMovement.DOLLY_IN, CameraMovement.CRANE_UP, CameraMovement.STEADICAM],
            "sad": [CameraMovement.DOLLY_OUT, CameraMovement.TILT_DOWN, CameraMovement.STATIC],
            "tense": [CameraMovement.HANDHELD, CameraMovement.ZOOM_IN, CameraMovement.CRANE_DOWN],
            "peaceful": [CameraMovement.STATIC, CameraMovement.STEADICAM, CameraMovement.PAN_RIGHT],
            "dramatic": [CameraMovement.CRANE_UP, CameraMovement.DOLLY_IN, CameraMovement.ZOOM_IN],
            "energetic": [CameraMovement.HANDHELD, CameraMovement.TRUCK_LEFT, CameraMovement.PAN_LEFT]
        }
        
        movements = movement_map.get(emotional_tone, [CameraMovement.STATIC])
        return random.choice(movements)
    
    def generate_cinematic_prompt(self, shot: Shot, subject: str) -> str:
        """Generate cinematic prompt for image generation"""
        
        prompt_parts = [
            f"{subject}",
            f"{shot.cinematic_description}",
            f"{shot.emotional_tone} mood",
            f"professional cinematography",
            f"high quality"
        ]
        
        return ", ".join(prompt_parts)


class NarrativeStructure:
    """Three-act narrative structure for music videos"""
    
    def __init__(self):
        self.acts: Dict[NarrativeAct, List[Dict[str, Any]]] = {
            NarrativeAct.ACT_1_SETUP: [],
            NarrativeAct.ACT_2_CONFRONTATION: [],
            NarrativeAct.ACT_3_RESOLUTION: []
        }
        
        self.total_duration: float = 0.0
        self.act_proportions = {
            NarrativeAct.ACT_1_SETUP: 0.25,      # 25%
            NarrativeAct.ACT_2_CONFRONTATION: 0.50,  # 50%
            NarrativeAct.ACT_3_RESOLUTION: 0.25   # 25%
        }
    
    def create_narrative_structure(self, 
                                 audio_duration: float,
                                 audio_sections: List[Dict[str, Any]],
                                 emotional_arc: List[str]) -> Dict[str, Any]:
        """Create three-act structure based on audio analysis"""
        
        self.total_duration = audio_duration
        
        # Calculate act durations
        act_durations = {
            act: duration * self.act_proportions[act] 
            for act, duration in [(act, audio_duration) for act in NarrativeAct]
        }
        
        # Map audio sections to acts
        act_sections = self._map_sections_to_acts(audio_sections, act_durations)
        
        # Create narrative beats
        narrative_beats = self._create_narrative_beats(act_sections, emotional_arc)
        
        return {
            "total_duration": self.total_duration,
            "act_durations": act_durations,
            "act_sections": act_sections,
            "narrative_beats": narrative_beats,
            "emotional_arc": emotional_arc
        }
    
    def _map_sections_to_acts(self, 
                            audio_sections: List[Dict[str, Any]], 
                            act_durations: Dict[NarrativeAct, float]) -> Dict[NarrativeAct, List[Dict[str, Any]]]:
        """Map audio sections to narrative acts"""
        
        act_sections = {act: [] for act in NarrativeAct}
        
        # Calculate act boundaries
        act_1_end = act_durations[NarrativeAct.ACT_1_SETUP]
        act_2_end = act_1_end + act_durations[NarrativeAct.ACT_2_CONFRONTATION]
        
        for section in audio_sections:
            start_time = section.get("start", 0)
            end_time = section.get("end", 0)
            mid_time = (start_time + end_time) / 2
            
            if mid_time <= act_1_end:
                act_sections[NarrativeAct.ACT_1_SETUP].append(section)
            elif mid_time <= act_2_end:
                act_sections[NarrativeAct.ACT_2_CONFRONTATION].append(section)
            else:
                act_sections[NarrativeAct.ACT_3_RESOLUTION].append(section)
        
        return act_sections
    
    def _create_narrative_beats(self, 
                              act_sections: Dict[NarrativeAct, List[Dict[str, Any]]], 
                              emotional_arc: List[str]) -> List[Dict[str, Any]]:
        """Create specific narrative beats within acts"""
        
        narrative_beats = []
        
        # Act 1: Setup beats
        act_1_beats = [
            {"beat": "opening_image", "function": "establish_world"},
            {"beat": "inciting_incident", "function": "start_journey"},
            {"beat": "first_plot_point", "function": "enter_new_world"}
        ]
        
        # Act 2: Confrontation beats
        act_2_beats = [
            {"beat": "rising_action", "function": "build_tension"},
            {"beat": "midpoint", "function": "major_change"},
            {"beat": "crisis", "function": "lowest_point"},
            {"beat": "climax", "function": "confrontation"}
        ]
        
        # Act 3: Resolution beats
        act_3_beats = [
            {"beat": "falling_action", "function": "resolve_conflict"},
            {"beat": "resolution", "function": "new_equilibrium"},
            {"beat": "final_image", "function": "emotional_conclusion"}
        ]
        
        # Combine all beats
        all_beats = act_1_beats + act_2_beats + act_3_beats
        
        # Add timing and emotional context
        for i, beat in enumerate(all_beats):
            progress = i / len(all_beats)
            emotional_index = min(int(progress * len(emotional_arc)), len(emotional_arc) - 1)
            
            beat.update({
                "timing": progress * self.total_duration,
                "emotional_tone": emotional_arc[emotional_index],
                "act_progress": progress
            })
            
            narrative_beats.append(beat)
        
        return narrative_beats


class VisualMotifSystem:
    """Visual motif system for thematic consistency"""
    
    def __init__(self):
        self.motifs: Dict[str, Dict[str, Any]] = {}
        self.color_palettes: Dict[str, List[str]] = {}
        self.symbolic_elements: Dict[str, List[str]] = {}
        
        self._initialize_motifs()
    
    def _initialize_motifs(self):
        """Initialize common visual motifs"""
        
        # Color motifs
        self.color_palettes = {
            "warm_passion": ["#FF6B35", "#F7931E", "#FFD23F", "#EE4B2B"],
            "cool_melancholy": ["#4A90E2", "#5C7CFA", "#748FFC", "#91A7FF"],
            "earth_grounded": ["#8B4513", "#D2691E", "#CD853F", "#F4A460"],
            "neon_futuristic": ["#FF0080", "#00FFFF", "#FFFF00", "#FF4500"],
            "monochrome_classic": ["#000000", "#333333", "#666666", "#FFFFFF"]
        }
        
        # Symbolic elements
        self.symbolic_elements = {
            "journey": ["roads", "bridges", "horizons", "pathways"],
            "love": ["flowers", "hearts", "intertwined hands", "warm light"],
            "struggle": ["stairs", "mountains", "storms", "shadows"],
            "freedom": ["birds", "open skies", "wide spaces", "flowing fabric"],
            "time": ["clocks", "seasons", "aging", "reflections"]
        }
        
        # Visual motifs by genre
        self.motifs = {
            "pop": {
                "color_palette": "warm_passion",
                "lighting_style": "high_key",
                "symbols": ["celebration", "energy", "movement"],
                "composition": "dynamic"
            },
            "ballad": {
                "color_palette": "cool_melancholy", 
                "lighting_style": "low_key",
                "symbols": ["intimacy", "reflection", "emotion"],
                "composition": "intimate"
            },
            "rock": {
                "color_palette": "monochrome_classic",
                "lighting_style": "dramatic",
                "symbols": ["power", "rebellion", "energy"],
                "composition": "dynamic"
            },
            "electronic": {
                "color_palette": "neon_futuristic",
                "lighting_style": "stylized",
                "symbols": ["technology", "movement", "rhythm"],
                "composition": "geometric"
            }
        }
    
    def get_motif_for_genre(self, genre: str) -> Dict[str, Any]:
        """Get visual motif configuration for genre"""
        return self.motifs.get(genre, self.motifs["pop"])
    
    def generate_color_palette(self, emotional_tone: str, intensity: float = 1.0) -> List[str]:
        """Generate color palette based on emotion and intensity"""
        
        emotion_color_map = {
            "happy": "warm_passion",
            "sad": "cool_melancholy", 
            "energetic": "neon_futuristic",
            "peaceful": "earth_grounded",
            "dramatic": "monochrome_classic"
        }
        
        palette_key = emotion_color_map.get(emotional_tone, "warm_passion")
        base_palette = self.color_palettes[palette_key]
        
        # Adjust intensity
        if intensity < 0.5:
            # Desaturate colors for lower intensity
            adjusted_palette = [self._desaturate_color(color, 0.5) for color in base_palette]
        else:
            adjusted_palette = base_palette
        
        return adjusted_palette
    
    def _desaturate_color(self, hex_color: str, factor: float) -> str:
        """Desaturate a hex color by a factor (0-1)"""
        # Simple desaturation - move toward gray
        # In a real implementation, this would use proper color space conversion
        return hex_color  # Placeholder


class CinematographyDirector:
    """Main cinematography director orchestrating all systems"""
    
    def __init__(self):
        self.cinematography_engine = CinematographyEngine()
        self.narrative_structure = NarrativeStructure()
        self.visual_motif_system = VisualMotifSystem()
        
        self.shot_sequence: List[Shot] = []
        self.narrative_plan: Optional[Dict[str, Any]] = None
        self.visual_motifs: Optional[Dict[str, Any]] = None
    
    async def create_cinematic_plan(self,
                                  audio_duration: float,
                                  audio_sections: List[Dict[str, Any]],
                                  genre: str,
                                  emotional_arc: List[str]) -> Dict[str, Any]:
        """Create complete cinematic plan for music video"""
        
        # 1. Create narrative structure
        self.narrative_plan = self.narrative_structure.create_narrative_structure(
            audio_duration, audio_sections, emotional_arc
        )
        
        # 2. Get visual motifs for genre
        self.visual_motifs = self.visual_motif_system.get_motif_for_genre(genre)
        
        # 3. Generate shot sequence
        self.shot_sequence = await self._generate_shot_sequence()
        
        # 4. Create comprehensive plan
        cinematic_plan = {
            "narrative_structure": self.narrative_plan,
            "visual_motifs": self.visual_motifs,
            "shot_sequence": [
                {
                    "shot_id": shot.shot_id,
                    "shot_type": shot.shot_type.value,
                    "emotional_tone": shot.emotional_tone,
                    "duration": shot.duration,
                    "cinematic_description": shot.cinematic_description,
                    "narrative_function": shot.narrative_function
                }
                for shot in self.shot_sequence
            ],
            "total_shots": len(self.shot_sequence),
            "average_shot_duration": audio_duration / len(self.shot_sequence) if self.shot_sequence else 0
        }
        
        return cinematic_plan
    
    async def _generate_shot_sequence(self) -> List[Shot]:
        """Generate shot sequence based on narrative beats"""
        
        if not self.narrative_plan:
            return []
        
        shots = []
        narrative_beats = self.narrative_plan["narrative_beats"]
        
        for i, beat in enumerate(narrative_beats):
            # Determine shot characteristics based on narrative beat
            shot_type = self._select_shot_type_for_beat(beat)
            emotional_tone = beat["emotional_tone"]
            narrative_function = beat["function"]
            
            # Calculate shot duration
            if i < len(narrative_beats) - 1:
                duration = narrative_beats[i + 1]["timing"] - beat["timing"]
            else:
                duration = self.narrative_plan["total_duration"] - beat["timing"]
            
            # Create shot
            shot = self.cinematography_engine.create_shot(
                shot_id=f"shot_{i:03d}_{beat['beat']}",
                shot_type=shot_type,
                emotional_tone=emotional_tone,
                narrative_function=narrative_function,
                duration=duration
            )
            
            shots.append(shot)
        
        return shots
    
    def _select_shot_type_for_beat(self, beat: Dict[str, Any]) -> ShotType:
        """Select appropriate shot type for narrative beat"""
        
        beat_shot_map = {
            "opening_image": ShotType.WIDE_SHOT,
            "inciting_incident": ShotType.MEDIUM_SHOT,
            "first_plot_point": ShotType.CLOSE_UP,
            "rising_action": ShotType.MEDIUM_WIDE_SHOT,
            "midpoint": ShotType.MEDIUM_CLOSE_UP,
            "crisis": ShotType.CLOSE_UP,
            "climax": ShotType.EXTREME_CLOSE_UP,
            "falling_action": ShotType.MEDIUM_SHOT,
            "resolution": ShotType.WIDE_SHOT,
            "final_image": ShotType.EXTREME_WIDE_SHOT
        }
        
        return beat_shot_map.get(beat["beat"], ShotType.MEDIUM_SHOT)
    
    def get_shot_specification(self, shot_id: str) -> Optional[Shot]:
        """Get detailed shot specification"""
        return self.cinematography_engine.shot_library.get(shot_id)
    
    def export_cinematography_plan(self, output_path: Path) -> None:
        """Export complete cinematography plan"""
        
        plan_data = {
            "narrative_structure": self.narrative_plan,
            "visual_motifs": self.visual_motifs,
            "shots": [
                {
                    "shot_id": shot.shot_id,
                    "shot_type": shot.shot_type.value,
                    "composition": {
                        "rule_of_thirds": shot.composition.rule_of_thirds,
                        "leading_lines": shot.composition.leading_lines,
                        "depth_of_field": shot.composition.depth_of_field,
                        "framing": shot.composition.framing
                    },
                    "camera_spec": {
                        "lens_focal_length": shot.camera_spec.lens_focal_length,
                        "aperture": shot.camera_spec.aperture,
                        "iso": shot.camera_spec.iso
                    },
                    "lighting_spec": {
                        "overall_mood": shot.lighting_spec.overall_mood,
                        "color_temperature": shot.lighting_spec.color_temperature
                    },
                    "movement": shot.movement.value,
                    "duration": shot.duration,
                    "emotional_tone": shot.emotional_tone,
                    "narrative_function": shot.narrative_function,
                    "cinematic_description": shot.cinematic_description
                }
                for shot in self.shot_sequence
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(plan_data, f, indent=2)


# Example usage
async def main():
    """Example cinematography system usage"""
    
    director = CinematographyDirector()
    
    # Mock audio analysis data
    audio_sections = [
        {"start": 0, "end": 30, "label": "verse"},
        {"start": 30, "end": 60, "label": "chorus"},
        {"start": 60, "end": 90, "label": "verse"},
        {"start": 90, "end": 120, "label": "chorus"},
        {"start": 120, "end": 150, "label": "bridge"},
        {"start": 150, "end": 180, "label": "chorus"}
    ]
    
    emotional_arc = ["peaceful", "building", "energetic", "climactic", "resolved"]
    
    # Create cinematic plan
    plan = await director.create_cinematic_plan(
        audio_duration=180.0,
        audio_sections=audio_sections,
        genre="pop",
        emotional_arc=emotional_arc
    )
    
    print(f"Created cinematic plan with {plan['total_shots']} shots")
    print(f"Average shot duration: {plan['average_shot_duration']:.2f}s")
    
    # Export plan
    director.export_cinematography_plan(Path("cinematography_plan.json"))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())