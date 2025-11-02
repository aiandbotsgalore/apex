"""
APEX DIRECTOR Storyboard Creation Module
Creates detailed storyboards based on visual treatments
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
import base64
import os
from pathlib import Path

from .treatment_generator import VisualTreatment, SceneDefinition, TreatmentType

logger = logging.getLogger(__name__)


class ShotType(Enum):
    """Types of camera shots"""
    EXTREME_WIDE = "extreme_wide"  # EWS
    WIDE = "wide"  # WS
    MEDIUM_WIDE = "medium_wide"  # MWS
    MEDIUM = "medium"  # MS
    MEDIUM_CLOSE = "medium_close"  # MCS
    CLOSE_UP = "close_up"  # CU
    EXTREME_CLOSE_UP = "extreme_close_up"  # ECU
    OVER_THE_SHOULDER = "over_the_shoulder"  # OTS
    TWO_SHOT = "two_shot"  # 2S
    POINT_OF_VIEW = "point_of_view"  # POV


class CameraAngle(Enum):
    """Camera angles"""
    EYE_LEVEL = "eye_level"
    HIGH_ANGLE = "high_angle"
    LOW_ANGLE = "low_angle"
    BIRD_EYE = "bird_eye"
    WORM_EYE = "worm_eye"
    DUTCH_ANGLE = "dutch_angle"


class Framing(Enum):
    """Shot framing"""
    ESTABLISHING = "establishing"
    SUPPORTING = "supporting"
    REVEAL = "reveal"
    TRANSITION = "transition"
    BEAT_FOCUS = "beat_focus"
    CLIMAX = "climax"


@dataclass
class ShotDefinition:
    """Represents an individual shot definition within a scene.

    Attributes:
        id: The unique identifier for the shot.
        shot_number: The number of the shot within the scene.
        start_time: The start time of the shot in seconds.
        duration: The duration of the shot in seconds.
        shot_type: The type of camera shot.
        camera_angle: The camera angle.
        framing: The framing of the shot.
        description: A description of the shot.
        visual_elements: A list of visual elements in the shot.
        camera_movement: The camera movement for the shot.
        lighting_notes: Notes about the lighting for the shot.
        transition_from_previous: The transition from the previous shot.
        sync_points: A list of audio sync points for the shot.
        notes: Additional notes for the shot.
    """
    id: str
    shot_number: int
    start_time: float
    duration: float
    shot_type: ShotType
    camera_angle: CameraAngle
    framing: Framing
    description: str
    visual_elements: List[str]
    camera_movement: str
    lighting_notes: str
    transition_from_previous: str
    sync_points: List[float]
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the ShotDefinition to a dictionary.

        Returns:
            A dictionary representation of the ShotDefinition.
        """
        data = asdict(self)
        data['shot_type'] = self.shot_type.value
        data['camera_angle'] = self.camera_angle.value
        data['framing'] = self.framing.value
        return data


@dataclass
class StoryboardScene:
    """Represents a complete storyboard for a single scene.

    Attributes:
        scene_id: The unique identifier for the scene.
        scene_type: The type of scene.
        start_time: The start time of the scene in seconds.
        duration: The duration of the scene in seconds.
        shots: A list of shot definitions in the scene.
        scene_overview: An overview of the scene.
        mood_notes: Notes about the mood of the scene.
        color_palette: A list of colors in the scene's color palette.
        audio_sync_notes: Notes about audio synchronization for the scene.
        special_requirements: A list of special requirements for the scene.
    """
    scene_id: str
    scene_type: str
    start_time: float
    duration: float
    shots: List[ShotDefinition]
    scene_overview: str
    mood_notes: str
    color_palette: List[str]
    audio_sync_notes: str
    special_requirements: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the StoryboardScene to a dictionary.

        Returns:
            A dictionary representation of the StoryboardScene.
        """
        data = asdict(self)
        data['shots'] = [shot.to_dict() for shot in self.shots]
        return data


@dataclass
class Storyboard:
    """Represents a complete storyboard for the entire video.

    Attributes:
        id: The unique identifier for the storyboard.
        project_name: The name of the project.
        treatment_id: The ID of the treatment this storyboard is based on.
        total_duration: The total duration of the storyboard in seconds.
        scenes: A list of storyboard scenes.
        technical_specs: A dictionary of technical specifications.
        creation_timestamp: The timestamp when the storyboard was created.
        version: The version of the storyboard.
        notes: Additional notes for the storyboard.
    """
    id: str
    project_name: str
    treatment_id: str
    total_duration: float
    scenes: List[StoryboardScene]
    technical_specs: Dict[str, Any]
    creation_timestamp: datetime
    version: str = "1.0"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the Storyboard to a dictionary.

        Returns:
            A dictionary representation of the Storyboard.
        """
        data = asdict(self)
        data['scenes'] = [scene.to_dict() for scene in self.scenes]
        data['creation_timestamp'] = self.creation_timestamp.isoformat()
        return data


class StoryboardCreator:
    """A system for creating and managing storyboards.

    This class provides functionality for:
    - Creating detailed storyboards from visual treatments
    - Exporting and importing storyboards
    - Generating production notes and summaries from storyboards
    """
    
    def __init__(self):
        """Initializes the StoryboardCreator."""
        self.shot_templates = self._load_shot_templates()
        self.transition_types = self._load_transition_types()
        self.camera_movements = self._load_camera_movements()
        
        logger.info("Storyboard Creator initialized")
    
    def _load_shot_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load shot type templates"""
        return {
            'intro': {
                'framing_priority': [Framing.ESTABLISHING, Framing.REVEAL],
                'shot_types': [ShotType.WIDE, ShotType.MEDIUM_WIDE, ShotType.MEDIUM],
                'camera_angles': [CameraAngle.EYE_LEVEL, CameraAngle.HIGH_ANGLE],
                'description': "Establishing shots that introduce the visual world"
            },
            'verse': {
                'framing_priority': [Framing.SUPPORTING, Framing.BEAT_FOCUS, Framing.ESTABLISHING],
                'shot_types': [ShotType.MEDIUM, ShotType.MEDIUM_CLOSE, ShotType.CLOSE_UP, ShotType.OTS],
                'camera_angles': [CameraAngle.EYE_LEVEL, CameraAngle.HIGH_ANGLE, CameraAngle.LOW_ANGLE],
                'description': "Supporting shots that build narrative and emotional connection"
            },
            'chorus': {
                'framing_priority': [Framing.BEAT_FOCUS, Framing.CLIMAX, Framing.SUPPORTING],
                'shot_types': [ShotType.WIDE, ShotType.MEDIUM_WIDE, ShotType.CLOSE_UP, ShotType.EXTREME_WIDE],
                'camera_angles': [CameraAngle.EYE_LEVEL, CameraAngle.LOW_ANGLE, CameraAngle.HIGH_ANGLE],
                'description': "High-energy shots that capture the musical climax"
            },
            'bridge': {
                'framing_priority': [Framing.REVEAL, Framing.TRANSITION, Framing.SUPPORTING],
                'shot_types': [ShotType.MEDIUM_CLOSE, ShotType.CLOSE_UP, ShotType.POINT_OF_VIEW, ShotType.WIDE],
                'camera_angles': [CameraAngle.EYE_LEVEL, CameraAngle.DUTCH_ANGLE, CameraAngle.BIRD_EYE],
                'description': "Contrasting shots that provide visual break and new perspective"
            },
            'outro': {
                'framing_priority': [Framing.TRANSITION, Framing.SUPPORTING, Framing.ESTABLISHING],
                'shot_types': [ShotType.MEDIUM_WIDE, ShotType.WIDE, ShotType.EXTREME_WIDE, ShotType.MEDIUM],
                'camera_angles': [CameraAngle.EYE_LEVEL, CameraAngle.HIGH_ANGLE],
                'description': "Closing shots that provide visual resolution and closure"
            }
        }
    
    def _load_transition_types(self) -> List[str]:
        """Load available transition types"""
        return [
            'hard cut',
            'dissolve 0.5s',
            'dissolve 1.0s',
            'fade to black',
            'fade from black',
            'fade to white',
            'fade from white',
            'wipe left',
            'wipe right',
            'wipe up',
            'wipe down',
            'zoom transition',
            'match cut',
            'rhythmic cut',
            'cross fade',
            'motion blur transition',
            'color wipe',
            'flash cut'
        ]
    
    def _load_camera_movements(self) -> List[str]:
        """Load camera movements"""
        return [
            'static',
            'slow zoom in',
            'slow zoom out',
            'zoom in (fast)',
            'zoom out (fast)',
            'pan left',
            'pan right',
            'tilt up',
            'tilt down',
            'track left',
            'track right',
            'dolly push in',
            'dolly pull out',
            'crane up',
            'crane down',
            'handheld subtle',
            'handheld dramatic',
            'arc left',
            'arc right',
            'reveal shot',
            'dolly track',
            'follow shot'
        ]
    
    async def create_storyboard(self, 
                               treatment: VisualTreatment,
                               custom_shots: Optional[List[Dict[str, Any]]] = None) -> Storyboard:
        """Creates a complete storyboard from a visual treatment.

        Args:
            treatment: The VisualTreatment object to create the storyboard from.
            custom_shots: Optional custom shot specifications.

        Returns:
            A Storyboard object with detailed shots.
        """
        try:
            logger.info(f"Creating storyboard for project: {treatment.project_name}")
            
            scenes = []
            
            for scene_idx, scene_def in enumerate(treatment.scenes):
                storyboard_scene = await self._create_storyboard_scene(
                    scene_def, treatment, scene_idx, len(treatment.scenes)
                )
                scenes.append(storyboard_scene)
            
            # Calculate total duration
            total_duration = sum(scene_def.duration for scene_def in treatment.scenes)
            
            storyboard = Storyboard(
                id=str(uuid.uuid4()),
                project_name=treatment.project_name,
                treatment_id=treatment.id,
                total_duration=total_duration,
                scenes=scenes,
                technical_specs=treatment.technical_specs,
                creation_timestamp=datetime.utcnow()
            )
            
            logger.info(f"Storyboard created with {len(scenes)} scenes, {sum(len(scene.shots) for scene in scenes)} total shots")
            return storyboard
            
        except Exception as e:
            logger.error(f"Error creating storyboard: {e}")
            raise ValueError(f"Failed to create storyboard: {str(e)}")
    
    async def _create_storyboard_scene(self, 
                                     scene_def: SceneDefinition,
                                     treatment: VisualTreatment,
                                     scene_index: int,
                                     total_scenes: int) -> StoryboardScene:
        """Create storyboard for a single scene"""
        
        # Determine number of shots based on scene duration and type
        scene_duration = scene_def.duration
        if scene_duration < 5:
            shot_count = 2
        elif scene_duration < 10:
            shot_count = 3
        elif scene_duration < 20:
            shot_count = 4
        else:
            shot_count = max(4, int(scene_duration / 5))
        
        # Get template for this scene type
        template = self.shot_templates.get(scene_def.scene_type, self.shot_templates['verse'])
        
        shots = []
        current_time = scene_def.start_time
        
        for shot_idx in range(shot_count):
            shot_duration = scene_duration / shot_count
            
            shot = self._create_shot(
                shot_id=str(uuid.uuid4()),
                shot_number=shot_idx + 1,
                start_time=current_time,
                duration=shot_duration,
                scene_type=scene_def.scene_type,
                scene_description=scene_def.description,
                visual_elements=scene_def.visual_elements,
                color_palette=scene_def.color_palette,
                mood=scene_def.mood,
                template=template,
                is_first_shot=(shot_idx == 0),
                is_last_shot=(shot_idx == shot_count - 1),
                total_shots=shot_count
            )
            
            shots.append(shot)
            current_time += shot_duration
        
        # Create scene overview
        scene_overview = self._generate_scene_overview(scene_def, treatment, scene_index)
        
        # Audio sync notes
        audio_sync_notes = self._generate_audio_sync_notes(scene_def, shot_count)
        
        storyboard_scene = StoryboardScene(
            scene_id=scene_def.id,
            scene_type=scene_def.scene_type,
            start_time=scene_def.start_time,
            duration=scene_def.duration,
            shots=shots,
            scene_overview=scene_overview,
            mood_notes=f"Mood: {scene_def.mood}. Lighting: {scene_def.lighting_style}. Camera: {scene_def.camera_movement}",
            color_palette=scene_def.color_palette,
            audio_sync_notes=audio_sync_notes,
            special_requirements=scene_def.visual_elements
        )
        
        return storyboard_scene
    
    def _create_shot(self,
                    shot_id: str,
                    shot_number: int,
                    start_time: float,
                    duration: float,
                    scene_type: str,
                    scene_description: str,
                    visual_elements: List[str],
                    color_palette: List[str],
                    mood: str,
                    template: Dict[str, Any],
                    is_first_shot: bool,
                    is_last_shot: bool,
                    total_shots: int) -> ShotDefinition:
        """Create a single shot definition"""
        
        # Determine shot type
        if is_first_shot and 'ESTABLISHING' in [f.value for f in template['framing_priority']]:
            # First shot often establishes
            shot_types = [ShotType.WIDE, ShotType.MEDIUM_WIDE, ShotType.EXTREME_WIDE]
        elif is_last_shot and total_shots > 2:
            # Last shot often reveals or transitions
            shot_types = [ShotType.MEDIUM_WIDE, ShotType.WIDE, ShotType.EXTREME_WIDE]
        else:
            # Middle shots
            shot_types = template['shot_types']
        
        shot_type = shot_types[shot_number % len(shot_types)]
        
        # Determine camera angle
        camera_angles = template['camera_angles']
        camera_angle = camera_angles[shot_number % len(camera_angles)]
        
        # Determine framing
        framing_priorities = template['framing_priority']
        framing = framing_priorities[shot_number % len(framing_priorities)]
        
        # Select camera movement
        if shot_type in [ShotType.EXTREME_WIDE, ShotType.WIDE]:
            camera_movement = random.choice(['static', 'slow zoom in', 'pan left', 'pan right'])
        elif shot_type in [ShotType.CLOSE_UP, ShotType.EXTREME_CLOSE_UP]:
            camera_movement = random.choice(['static', 'slow zoom in', 'tilt up', 'tilt down'])
        else:
            camera_movement = random.choice(self.camera_movements)
        
        # Generate description
        description = self._generate_shot_description(
            shot_type, camera_angle, scene_description, visual_elements, mood
        )
        
        # Lighting notes
        lighting_notes = self._generate_lighting_notes(shot_type, camera_angle, mood)
        
        # Transition from previous
        if is_first_shot:
            transition_from_previous = "scene start"
        else:
            transition_from_previous = random.choice(self.transition_types)
        
        # Sync points
        sync_points = []
        beats_per_second = 2  # Assume 120 BPM
        for sync_time in range(int(start_time), int(start_time + duration), max(1, int(60/beats_per_second))):
            sync_points.append(sync_time - start_time)
        
        # Notes
        notes = self._generate_shot_notes(shot_type, camera_angle, scene_type)
        
        return ShotDefinition(
            id=shot_id,
            shot_number=shot_number,
            start_time=start_time,
            duration=duration,
            shot_type=shot_type,
            camera_angle=camera_angle,
            framing=framing,
            description=description,
            visual_elements=visual_elements,
            camera_movement=camera_movement,
            lighting_notes=lighting_notes,
            transition_from_previous=transition_from_previous,
            sync_points=sync_points,
            notes=notes
        )
    
    def _generate_shot_description(self,
                                 shot_type: ShotType,
                                 camera_angle: CameraAngle,
                                 scene_description: str,
                                 visual_elements: List[str],
                                 mood: str) -> str:
        """Generate description for a shot"""
        
        shot_descriptions = {
            ShotType.EXTREME_WIDE: "Extreme wide shot establishing the environment and scale",
            ShotType.WIDE: "Wide shot showing subject in environmental context",
            ShotType.MEDIUM_WIDE: "Medium wide shot balancing subject and environment",
            ShotType.MEDIUM: "Medium shot showing subject from waist up",
            ShotType.MEDIUM_CLOSE: "Medium close shot showing subject from chest up",
            ShotType.CLOSE_UP: "Close-up shot focusing on subject's face or detail",
            ShotType.EXTREME_CLOSE_UP: "Extreme close-up highlighting specific details",
            ShotType.OVER_THE_SHOULDER: "Over-the-shoulder shot creating perspective connection",
            ShotType.TWO_SHOT: "Two-shot showing interaction between subjects",
            ShotType.POINT_OF_VIEW: "Point-of-view shot from character perspective"
        }
        
        angle_descriptions = {
            CameraAngle.EYE_LEVEL: "shot at natural eye level",
            CameraAngle.HIGH_ANGLE: "high angle looking down, creating vulnerability",
            CameraAngle.LOW_ANGLE: "low angle looking up, creating power",
            CameraAngle.BIRD_EYE: "bird's eye view from directly above",
            CameraAngle.WORM_EYE: "worm's eye view from below",
            CameraAngle.DUTCH_ANGLE: "tilted angle creating tension"
        }
        
        base_desc = shot_descriptions.get(shot_type, "Medium shot")
        angle_desc = angle_descriptions.get(camera_angle, "at eye level")
        
        description = f"{base_desc}, {angle_desc}"
        
        # Add mood and visual elements
        description += f". Mood: {mood}"
        
        if visual_elements:
            elements_str = ", ".join(visual_elements[:3])  # Limit to first 3
            description += f". Visual elements: {elements_str}"
        
        # Add scene context
        if scene_description:
            description += f". Scene context: {scene_description[:100]}"
        
        return description
    
    def _generate_lighting_notes(self, shot_type: ShotType, camera_angle: CameraAngle, mood: str) -> str:
        """Generate lighting notes for a shot"""
        
        # Base lighting on mood
        mood_lighting = {
            'dramatic': 'Strong directional lighting with deep shadows',
            'romantic': 'Soft, warm lighting with gentle highlights',
            'mysterious': 'Low key lighting with selective illumination',
            'energetic': 'High contrast lighting with vibrant colors',
            'intimate': 'Soft, close lighting with warm tones',
            'epic': 'Dramatic cinematic lighting with volumetric effects',
            'dreamy': 'Soft, diffused lighting with ethereal quality',
            'powerful': 'Strong, directional lighting with deep contrast'
        }
        
        base_lighting = mood_lighting.get(mood, 'Standard lighting setup')
        
        # Adjust for shot type
        if shot_type in [ShotType.CLOSE_UP, ShotType.EXTREME_CLOSE_UP]:
            base_lighting += '. Use soft fill to avoid harsh shadows on face'
        elif shot_type == ShotType.EXTREME_WIDE:
            base_lighting += '. Ensure even coverage across wide frame'
        
        return base_lighting
    
    def _generate_scene_overview(self, 
                               scene_def: SceneDefinition,
                               treatment: VisualTreatment,
                               scene_index: int) -> str:
        """Generate overview description for a scene"""
        
        scene_type = scene_def.scene_type
        duration = scene_def.duration
        
        if scene_type == 'intro':
            overview = f"Opening scene ({duration:.1f}s): Establishes the visual world and tone. "
            overview += f"This scene sets expectations and introduces key visual elements: {', '.join(scene_def.visual_elements[:3])}."
            
        elif scene_type == 'verse':
            overview = f"Verse scene ({duration:.1f}s): Develops narrative and emotional content. "
            overview += f"Focus on building story elements and character development with {scene_def.mood} mood."
            
        elif scene_type == 'chorus':
            overview = f"Chorus scene ({duration:.1f}s): High-energy climax moment. "
            overview += "This is the musical and visual peak, featuring dynamic shots and impactful imagery."
            
        elif scene_type == 'bridge':
            overview = f"Bridge scene ({duration:.1f}s): Provides contrast and musical variation. "
            overview += "Use this section for visual departure or new perspective with different mood."
            
        elif scene_type == 'outro':
            overview = f"Closing scene ({duration:.1f}s): Brings narrative resolution and visual closure. "
            overview += "This final scene should provide satisfying conclusion to the visual story."
        
        else:
            overview = f"Scene ({duration:.1f}s): {scene_def.description}"
        
        # Add color and mood notes
        overview += f" Color palette: {', '.join(scene_def.color_palette[:3])}. "
        overview += f"Lighting style: {scene_def.lighting_style}. "
        overview += f"Camera movement: {scene_def.camera_movement}."
        
        return overview
    
    def _generate_audio_sync_notes(self, scene_def: SceneDefinition, shot_count: int) -> str:
        """Generate notes about audio synchronization"""
        
        notes = "Audio sync points: "
        
        # Add beat sync points if available
        if scene_def.beat_sync_points:
            sync_times = [f"{t:.1f}s" for t in scene_def.beat_sync_points[:5]]  # First 5 sync points
            notes += f"Beat sync at {', '.join(sync_times)}. "
        
        # Add scene type specific notes
        if scene_def.scene_type == 'chorus':
            notes += "Emphasize sync with musical climax and rhythm changes. "
        elif scene_def.scene_type == 'verse':
            notes += "Maintain steady sync with verse lyrics and rhythm. "
        elif scene_def.scene_type == 'bridge':
            notes += "Sync with musical bridge elements and tempo changes. "
        
        # Shot-specific sync
        if shot_count > 1:
            notes += f"Plan {shot_count} shots within scene for rhythm-based cutting."
        
        return notes
    
    def _generate_shot_notes(self, shot_type: ShotType, camera_angle: CameraAngle, scene_type: str) -> str:
        """Generate production notes for a shot"""
        
        notes = []
        
        # Shot type notes
        if shot_type == ShotType.EXTREME_WIDE:
            notes.append("Ensure sufficient depth of field to keep all elements in focus")
        elif shot_type == ShotType.CLOSE_UP:
            notes.append("Pay special attention to actor's performance and micro-expressions")
        elif shot_type == ShotType.OVER_THE_SHOULDER:
            notes.append("Maintain proper framing and avoid blocking key visual elements")
        
        # Angle notes
        if camera_angle == CameraAngle.DUTCH_ANGLE:
            notes.append("Use dutch angle sparingly to avoid disorienting the viewer")
        elif camera_angle == CameraAngle.BIRD_EYE:
            notes.append("Ensure adequate lighting from above to avoid harsh shadows")
        
        # Scene type notes
        if scene_type == 'chorus':
            notes.append("High energy shots require dynamic camera work and quick cutting")
        elif scene_type == 'verse':
            notes.append("Maintain steady pacing and consistent mood throughout")
        
        return '; '.join(notes) if notes else "Standard shot requirements"
    
    def export_storyboard(self, storyboard: Storyboard, file_path: str, format: str = 'json') -> bool:
        """Exports a storyboard to a file.

        Args:
            storyboard: The Storyboard object to export.
            file_path: The path to the file to export the storyboard to.
            format: The format to export the storyboard in.

        Returns:
            True if the storyboard was successfully exported, False otherwise.
        """
        try:
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(storyboard.to_dict(), f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Storyboard exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export storyboard: {e}")
            return False
    
    def import_storyboard(self, file_path: str) -> Optional[Storyboard]:
        """Imports a storyboard from a file.

        Args:
            file_path: The path to the file to import the storyboard from.

        Returns:
            A Storyboard object, or None if the import failed.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert scene data
            scenes = []
            for scene_data in data.get('scenes', []):
                # Convert shots
                shots = []
                for shot_data in scene_data.get('shots', []):
                    shot = ShotDefinition(**shot_data)
                    shots.append(shot)
                
                scene = StoryboardScene(
                    scene_id=scene_data['scene_id'],
                    scene_type=scene_data['scene_type'],
                    start_time=scene_data['start_time'],
                    duration=scene_data['duration'],
                    shots=shots,
                    scene_overview=scene_data['scene_overview'],
                    mood_notes=scene_data['mood_notes'],
                    color_palette=scene_data['color_palette'],
                    audio_sync_notes=scene_data['audio_sync_notes'],
                    special_requirements=scene_data['special_requirements']
                )
                scenes.append(scene)
            
            storyboard = Storyboard(
                id=data['id'],
                project_name=data['project_name'],
                treatment_id=data['treatment_id'],
                total_duration=data['total_duration'],
                scenes=scenes,
                technical_specs=data['technical_specs'],
                creation_timestamp=datetime.fromisoformat(data['creation_timestamp']),
                version=data.get('version', '1.0'),
                notes=data.get('notes', '')
            )
            
            logger.info(f"Storyboard imported from: {file_path}")
            return storyboard
            
        except Exception as e:
            logger.error(f"Failed to import storyboard: {e}")
            return None
    
    def get_storyboard_summary(self, storyboard: Storyboard) -> Dict[str, Any]:
        """Gets a summary of a storyboard.

        Args:
            storyboard: The Storyboard object to get the summary for.

        Returns:
            A dictionary containing a summary of the storyboard.
        """
        total_shots = sum(len(scene.shots) for scene in storyboard.scenes)
        
        # Shot type distribution
        shot_type_counts = {}
        for scene in storyboard.scenes:
            for shot in scene.shots:
                shot_type = shot.shot_type.value
                shot_type_counts[shot_type] = shot_type_counts.get(shot_type, 0) + 1
        
        # Scene type distribution
        scene_type_counts = {}
        for scene in storyboard.scenes:
            scene_type = scene.scene_type
            scene_type_counts[scene_type] = scene_type_counts.get(scene_type, 0) + 1
        
        return {
            'project_name': storyboard.project_name,
            'total_duration': f"{storyboard.total_duration:.1f} seconds",
            'scene_count': len(storyboard.scenes),
            'total_shots': total_shots,
            'avg_shots_per_scene': total_shots / len(storyboard.scenes) if storyboard.scenes else 0,
            'shot_type_distribution': shot_type_counts,
            'scene_type_distribution': scene_type_counts,
            'creation_date': storyboard.creation_timestamp.strftime('%Y-%m-%d %H:%M'),
            'version': storyboard.version
        }
    
    def generate_production_notes(self, storyboard: Storyboard) -> str:
        """Generates comprehensive production notes from a storyboard.

        Args:
            storyboard: The Storyboard object to generate the notes from.

        Returns:
            A string of production notes.
        """
        
        notes = f"PRODUCTION NOTES - {storyboard.project_name}\n"
        notes += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        notes += f"Version: {storyboard.version}\n\n"
        
        notes += "OVERVIEW:\n"
        notes += f"• Total duration: {storyboard.total_duration:.1f} seconds\n"
        notes += f"• Total scenes: {len(storyboard.scenes)}\n"
        notes += f"• Total shots: {sum(len(scene.shots) for scene in storyboard.scenes)}\n\n"
        
        # Scene breakdown
        notes += "SCENE BREAKDOWN:\n"
        for scene_idx, scene in enumerate(storyboard.scenes):
            notes += f"\nScene {scene_idx + 1} ({scene.scene_type.upper()}):\n"
            notes += f"  • Duration: {scene.duration:.1f} seconds\n"
            notes += f"  • Shots: {len(scene.shots)}\n"
            notes += f"  • Color palette: {', '.join(scene.color_palette[:3])}\n"
            notes += f"  • Overview: {scene.scene_overview[:100]}...\n"
            
            # Shot details for complex scenes
            if len(scene.shots) > 3:
                notes += f"  • Shot breakdown: "
                shot_details = []
                for shot in scene.shots:
                    shot_details.append(f"{shot.shot_type.value} ({shot.duration:.1f}s)")
                notes += "; ".join(shot_details)
                notes += "\n"
        
        # Technical requirements
        notes += f"\nTECHNICAL REQUIREMENTS:\n"
        specs = storyboard.technical_specs
        notes += f"• Resolution: {specs.get('output_resolution', 'Unknown')}\n"
        notes += f"• Frame rate: {specs.get('frame_rate', 'Unknown')} fps\n"
        notes += f"• Aspect ratio: {specs.get('aspect_ratio', 'Unknown')}\n"
        notes += f"• Codec: {specs.get('codec', 'Unknown')}\n"
        
        # Special requirements
        all_special_requirements = []
        for scene in storyboard.scenes:
            all_special_requirements.extend(scene.special_requirements)
        
        if all_special_requirements:
            notes += f"\nSPECIAL EFFECTS/VISUAL ELEMENTS:\n"
            unique_requirements = list(set(all_special_requirements))
            for req in unique_requirements:
                notes += f"• {req}\n"
        
        notes += f"\nPOST-PRODUCTION NOTES:\n"
        notes += f"• Estimated render time: {specs.get('estimated_render_time_hours', 'Unknown')} hours\n"
        notes += f"• Storage requirement: {specs.get('storage_requirement_gb', 'Unknown')} GB\n"
        notes += f"• Review checkpoints: 25%, 50%, 75% completion\n"
        
        return notes
