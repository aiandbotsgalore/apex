"""
Professional Video Transitions - Broadcast-quality transitions with frame accuracy
Implements cut, crossfade, whip pan, match dissolve with precise timing
"""

import numpy as np
import cv2
import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import ffmpeg
from scipy import ndimage
from apex_director.video.timeline import Timeline, Clip, Transition


class TransitionType(Enum):
    """Professional transition types"""
    CUT = "cut"
    CROSSFADE = "crossfade"
    WHIP_PAN = "whip_pan"
    MATCH_DISSOLVE = "match_dissolve"
    DIP_TO_BLACK = "dip_to_black"
    DIP_TO_WHITE = "dip_to_white"
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    WIPE_UP = "wipe_up"
    WIPE_DOWN = "wipe_down"


@dataclass
class TransitionEffect:
    """Represents a professional transition effect with parameters.

    Attributes:
        type: The type of transition.
        duration_frames: The duration of the transition in frames.
        parameters: A dictionary of additional parameters for the transition.
    """
    type: TransitionType
    duration_frames: int
    parameters: Dict[str, Union[int, float, str, Tuple[int, int]]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        
        # Set default parameters based on transition type
        defaults = {
            TransitionType.CROSSFADE: {"curve": "linear"},
            TransitionType.WHIP_PAN: {"pan_speed": 15.0, "blur_amount": 3},
            TransitionType.MATCH_DISSOLVE: {"tolerance": 10, "blend_mode": "normal"},
            TransitionType.DIP_TO_BLACK: {"opacity": True},
            TransitionType.DIP_TO_WHITE: {"opacity": True},
            TransitionType.WIPE_LEFT: {"edge_feather": 0},
            TransitionType.WIPE_RIGHT: {"edge_feather": 0},
            TransitionType.WIPE_UP: {"edge_feather": 0},
            TransitionType.WIPE_DOWN: {"edge_feather": 0}
        }
        
        if self.type in defaults:
            for key, value in defaults[self.type].items():
                if key not in self.parameters:
                    self.parameters[key] = value


class TransitionEngine:
    """A professional transition rendering engine.

    This class provides functionality for applying various broadcast-quality
    transitions with frame accuracy.
    """
    
    def __init__(self, timeline: Timeline):
        """Initializes the TransitionEngine.

        Args:
            timeline: The timeline to apply transitions to.
        """
        self.timeline = timeline
        self.frame_width = timeline.resolution[0]
        self.frame_height = timeline.resolution[1]
        self.frame_rate = timeline.frame_rate
        self.color_space = "Rec.709"
        
    def apply_cut_transition(self, frame_a: np.ndarray, frame_b: np.ndarray, 
                           progress: float = 1.0) -> np.ndarray:
        """Applies a perfect frame cut with no transition.

        Args:
            frame_a: The first frame.
            frame_b: The second frame.
            progress: The progress of the transition (ignored for cuts).

        Returns:
            The second frame.
        """
        # Immediate cut at exact frame boundary
        return frame_b.copy()
    
    def apply_crossfade_transition(self, frame_a: np.ndarray, frame_b: np.ndarray, 
                                 progress: float, curve: str = "linear") -> np.ndarray:
        """Applies a professional crossfade with configurable curves.

        Args:
            frame_a: The first frame.
            frame_b: The second frame.
            progress: The progress of the transition.
            curve: The fade curve to use.

        Returns:
            The transitioned frame.
        """
        if progress <= 0:
            return frame_a.copy()
        elif progress >= 1:
            return frame_b.copy()
        
        # Select fade curve
        alpha = self._calculate_fade_curve(progress, curve)
        
        # Apply crossfade
        result = cv2.addWeighted(frame_a, 1 - alpha, frame_b, alpha, 0)
        return result
    
    def apply_whip_pan_transition(self, frame_a: np.ndarray, frame_b: np.ndarray, 
                                progress: float, pan_speed: float = 15.0, 
                                blur_amount: int = 3) -> np.ndarray:
        """Applies a whip pan transition with motion blur.

        Args:
            frame_a: The first frame.
            frame_b: The second frame.
            progress: The progress of the transition.
            pan_speed: The speed of the pan.
            blur_amount: The amount of motion blur to apply.

        Returns:
            The transitioned frame.
        """
        if progress <= 0:
            return frame_a.copy()
        elif progress >= 1:
            return frame_b.copy()
        
        # Calculate pan progress
        pan_progress = progress * pan_speed
        
        # Pan direction (left to right)
        offset_x = int(pan_progress * self.frame_width * 0.1)
        
        # Create motion effect
        if progress < 0.5:
            # First half: pan out of frame_a
            matrix = np.float32([[1, 0, -offset_x], [0, 1, 0]])
            panned_frame = cv2.warpAffine(frame_a, matrix, (self.frame_width, self.frame_height))
            
            # Add motion blur
            if blur_amount > 0:
                panned_frame = cv2.GaussianBlur(panned_frame, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
            
            # Blend with frame_b
            alpha = progress * 2
            result = cv2.addWeighted(panned_frame, 1 - alpha, frame_b, alpha, 0)
        else:
            # Second half: pan into frame_b
            reverse_progress = (progress - 0.5) * 2
            offset_x = int((1 - reverse_progress) * self.frame_width * 0.1)
            
            matrix = np.float32([[1, 0, offset_x], [0, 1, 0]])
            panned_frame = cv2.warpAffine(frame_b, matrix, (self.frame_width, self.frame_height))
            
            # Add motion blur
            if blur_amount > 0:
                panned_frame = cv2.GaussianBlur(panned_frame, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
            
            # Blend with frame_a
            alpha = reverse_progress
            result = cv2.addWeighted(frame_a, alpha, panned_frame, 1 - alpha, 0)
        
        return result
    
    def apply_match_dissolve_transition(self, frame_a: np.ndarray, frame_b: np.ndarray, 
                                      progress: float, tolerance: int = 10) -> np.ndarray:
        """Applies a match dissolve transition based on color similarity.

        Args:
            frame_a: The first frame.
            frame_b: The second frame.
            progress: The progress of the transition.
            tolerance: The color similarity tolerance.

        Returns:
            The transitioned frame.
        """
        if progress <= 0:
            return frame_a.copy()
        elif progress >= 1:
            return frame_b.copy()
        
        # Convert to LAB color space for better perceptual comparison
        lab_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2LAB)
        lab_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2LAB)
        
        # Calculate color difference map
        diff = np.abs(lab_a.astype(np.float32) - lab_b.astype(np.float32))
        color_diff = np.sum(diff, axis=2)
        
        # Create mask based on tolerance
        mask = color_diff < tolerance
        
        # Apply dissolve based on mask and progress
        alpha = progress
        
        # Blend similar regions more quickly
        mask_alpha = mask.astype(np.float32) * alpha
        mask_alpha = np.stack([mask_alpha, mask_alpha, mask_alpha], axis=2)
        
        result = frame_a * (1 - mask_alpha) + frame_b * mask_alpha
        
        return result.astype(np.uint8)
    
    def apply_dip_transition(self, frame_a: np.ndarray, frame_b: np.ndarray, 
                           progress: float, to_color: str = "black") -> np.ndarray:
        """Applies a dip to black or white transition.

        Args:
            frame_a: The first frame.
            frame_b: The second frame.
            progress: The progress of the transition.
            to_color: The color to dip to ("black" or "white").

        Returns:
            The transitioned frame.
        """
        if progress <= 0:
            return frame_a.copy()
        elif progress >= 1:
            return frame_b.copy()
        
        # Create dip color
        if to_color == "black":
            dip_color = np.zeros_like(frame_a)
        else:  # white
            dip_color = np.full_like(frame_a, 255)
        
        # Calculate dip progress
        if progress < 0.5:
            # Dip to color
            alpha = progress * 2
            result = cv2.addWeighted(frame_a, 1 - alpha, dip_color, alpha, 0)
        else:
            # Rise from color
            alpha = (progress - 0.5) * 2
            result = cv2.addWeighted(dip_color, 1 - alpha, frame_b, alpha, 0)
        
        return result
    
    def apply_wipe_transition(self, frame_a: np.ndarray, frame_b: np.ndarray, 
                            progress: float, direction: str = "left", 
                            edge_feather: int = 0) -> np.ndarray:
        """Applies a professional wipe transition.

        Args:
            frame_a: The first frame.
            frame_b: The second frame.
            progress: The progress of the transition.
            direction: The direction of the wipe.
            edge_feather: The amount of feather to apply to the edge of the wipe.

        Returns:
            The transitioned frame.
        """
        if progress <= 0:
            return frame_a.copy()
        elif progress >= 1:
            return frame_b.copy()
        
        # Create wipe mask
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        # Calculate wipe position
        wipe_progress = progress
        
        if direction == "left":
            wipe_x = int(self.frame_width * wipe_progress)
            mask[:, :wipe_x] = 1.0
        elif direction == "right":
            wipe_x = int(self.frame_width * (1 - wipe_progress))
            mask[:, wipe_x:] = 1.0
        elif direction == "up":
            wipe_y = int(self.frame_height * wipe_progress)
            mask[:wipe_y, :] = 1.0
        elif direction == "down":
            wipe_y = int(self.frame_height * (1 - wipe_progress))
            mask[wipe_y:, :] = 1.0
        
        # Apply feather to edge
        if edge_feather > 0:
            kernel = np.ones((edge_feather * 2 + 1, edge_feather * 2 + 1), np.float32)
            mask = cv2.filter2D(mask, -1, kernel / kernel.sum())
        
        # Expand mask to 3 channels
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        
        # Apply wipe
        result = frame_a * (1 - mask_3ch) + frame_b * mask_3ch
        
        return result.astype(np.uint8)
    
    def _calculate_fade_curve(self, progress: float, curve_type: str) -> float:
        """Calculate fade curve for professional appearance"""
        if curve_type == "linear":
            return progress
        elif curve_type == "ease_in":
            return progress * progress
        elif curve_type == "ease_out":
            return 1 - (1 - progress) * (1 - progress)
        elif curve_type == "ease_in_out":
            if progress < 0.5:
                return 2 * progress * progress
            else:
                return 1 - 2 * (1 - progress) * (1 - progress)
        elif curve_type == "sine":
            return 0.5 - 0.5 * math.cos(progress * math.pi)
        elif curve_type == "gamma":
            gamma = 2.2
            return math.pow(progress, 1 / gamma)
        else:
            return progress


class TransitionGenerator:
    """Generates FFmpeg commands for professional transitions."""
    
    @staticmethod
    def generate_ffmpeg_crossfade(input_a: str, input_b: str, output: str, 
                                duration: float, offset: float, curve: str = "linear") -> str:
        """Generates an FFmpeg command for a crossfade transition.

        Args:
            input_a: The first input file.
            input_b: The second input file.
            output: The output file.
            duration: The duration of the crossfade in seconds.
            offset: The offset of the crossfade in seconds.
            curve: The fade curve to use.

        Returns:
            The FFmpeg command as a string.
        """
        curve_filter = {
            "linear": "fade=t=linear:st=0:d={}".format(duration),
            "ease_in": "fade=t=in:st=0:d={}".format(duration),
            "ease_out": "fade=t=out:st=0:d={}".format(duration),
            "ease_in_out": "fade=t=in:out:st=0:d={}".format(duration),
            "sine": "fade=t=in:sin:st=0:d={}".format(duration)
        }
        
        crossfade_filter = curve_filter.get(curve, curve_filter["linear"])
        
        return (
            f"ffmpeg -i {input_a} -i {input_b} "
            f"-filter_complex "
            f"'[0:v][1:v]xfade=transition={crossfade_filter}:offset={offset}[v]' "
            f"-map '[v]' -map 0:a -map 1:a -c:v libx264 -c:a aac {output}"
        )
    
    @staticmethod
    def generate_ffmpeg_whip_pan(input_a: str, input_b: str, output: str, 
                               duration: float, offset: float) -> str:
        """Generates an FFmpeg command for a whip pan transition.

        Args:
            input_a: The first input file.
            input_b: The second input file.
            output: The output file.
            duration: The duration of the transition in seconds.
            offset: The offset of the transition in seconds.

        Returns:
            The FFmpeg command as a string.
        """
        return (
            f"ffmpeg -i {input_a} -i {input_b} "
            f"-filter_complex "
            f"'[0:v]scale=iw*1.2:ih*1.2,pad=iw*1.2:ih*1.2:(ow-iw)/2:(oh-ih)/2[p0];"
            f"[1:v]scale=iw*1.2:ih*1.2,pad=iw*1.2:ih*1.2:(ow-iw)/2:(oh-ih)/2[p1];"
            f"[p0][p1]blend=all_mode=screen:all_opacity=0.5[b]' "
            f"-map '[b]' -map 0:a -map 1:a -c:v libx264 -c:a aac {output}"
        )
    
    @staticmethod
    def generate_ffmpeg_wipe(input_a: str, input_b: str, output: str, 
                           duration: float, offset: float, direction: str = "left") -> str:
        """Generates an FFmpeg command for a wipe transition.

        Args:
            input_a: The first input file.
            input_b: The second input file.
            output: The output file.
            duration: The duration of the transition in seconds.
            offset: The offset of the transition in seconds.
            direction: The direction of the wipe.

        Returns:
            The FFmpeg command as a string.
        """
        wipe_type = {
            "left": "wipeleft",
            "right": "wiperight",
            "up": "wipeup",
            "down": "wipedown"
        }
        
        transition = wipe_type.get(direction, "wipeleft")
        
        return (
            f"ffmpeg -i {input_a} -i {input_b} "
            f"-filter_complex "
            f"'[0:v][1:v]xfade=transition={transition}:offset={offset}:duration={duration}[v]' "
            f"-map '[v]' -map 0:a -map 1:a -c:v libx264 -c:a aac {output}"
        )


class TransitionPreview:
    """Provides functionality for real-time transition preview and adjustment.

    This class allows for the generation of preview frames for transitions,
    as well as the real-time adjustment of transition parameters.
    """
    
    def __init__(self, timeline: Timeline):
        """Initializes the TransitionPreview.

        Args:
            timeline: The timeline to generate previews for.
        """
        self.timeline = timeline
        self.preview_resolution = (1920, 1080)
        self.preview_fps = 30.0
        
    def generate_transition_preview(self, transition: Transition, 
                                  sample_frame_a: np.ndarray, 
                                  sample_frame_b: np.ndarray) -> List[np.ndarray]:
        """Generates a sequence of preview frames for a given transition.

        Args:
            transition: The transition to generate a preview for.
            sample_frame_a: A sample frame from the first clip.
            sample_frame_b: A sample frame from the second clip.

        Returns:
            A list of frames representing the transition preview.
        """
        frames = []
        transition_engine = TransitionEngine(self.timeline)
        
        # Generate frames at transition resolution
        sample_frame_a = cv2.resize(sample_frame_a, self.preview_resolution)
        sample_frame_b = cv2.resize(sample_frame_b, self.preview_resolution)
        
        duration_frames = transition.duration_frames
        for i in range(duration_frames + 1):
            progress = i / duration_frames
            
            if transition.type == "cut":
                frame = transition_engine.apply_cut_transition(sample_frame_a, sample_frame_b, progress)
            elif transition.type == "crossfade":
                curve = transition.parameters.get("curve", "linear")
                frame = transition_engine.apply_crossfade_transition(sample_frame_a, sample_frame_b, progress, curve)
            elif transition.type == "whip_pan":
                pan_speed = transition.parameters.get("pan_speed", 15.0)
                blur_amount = transition.parameters.get("blur_amount", 3)
                frame = transition_engine.apply_whip_pan_transition(sample_frame_a, sample_frame_b, progress, pan_speed, blur_amount)
            elif transition.type == "match_dissolve":
                tolerance = transition.parameters.get("tolerance", 10)
                frame = transition_engine.apply_match_dissolve_transition(sample_frame_a, sample_frame_b, progress, tolerance)
            elif transition.type == "dip_to_black":
                frame = transition_engine.apply_dip_transition(sample_frame_a, sample_frame_b, progress, "black")
            elif transition.type == "dip_to_white":
                frame = transition_engine.apply_dip_transition(sample_frame_a, sample_frame_b, progress, "white")
            elif "wipe" in transition.type:
                direction = transition.type.replace("wipe_", "")
                edge_feather = transition.parameters.get("edge_feather", 0)
                frame = transition_engine.apply_wipe_transition(sample_frame_a, sample_frame_b, progress, direction, edge_feather)
            
            frames.append(frame)
        
        return frames
    
    def adjust_transition_parameters(self, transition: Transition, 
                                   parameter_updates: Dict[str, Union[int, float, str]]) -> None:
        """Adjusts the parameters of a transition in real-time.

        Args:
            transition: The transition to adjust.
            parameter_updates: A dictionary of parameter updates.
        """
        for key, value in parameter_updates.items():
            transition.parameters[key] = value
    
    def analyze_transition_impact(self, transition: Transition) -> Dict[str, float]:
        """Analyzes the visual impact of a transition.

        Args:
            transition: The transition to analyze.

        Returns:
            A dictionary of impact metrics.
        """
        # This would implement metrics like:
        # - Cut detection confidence
        # - Color matching score
        # - Motion smoothness
        # - Overall visual coherence
        
        impact_metrics = {
            "smoothness": 0.95,  # Placeholder
            "color_harmony": 0.87,  # Placeholder
            "motion_coherence": 0.92,  # Placeholder
            "visual_impact": 0.89  # Placeholder
        }
        
        return impact_metrics


# Utility functions for professional transition management
def validate_transition_timing(timeline: Timeline, transition: Transition) -> Dict[str, Union[bool, List[str]]]:
    """Validates the timing of a transition against broadcast standards.

    Args:
        timeline: The timeline containing the transition.
        transition: The transition to validate.

    Returns:
        A dictionary containing the validation status, errors, and warnings.
    """
    errors = []
    warnings = []
    
    # Check transition duration (minimum 1 frame, maximum 60 frames for most transitions)
    if transition.duration_frames < 1:
        errors.append(f"Transition duration too short: {transition.duration_frames} frames")
    elif transition.duration_frames > 60 and transition.type != "cut":
        warnings.append(f"Transition duration very long: {transition.duration_frames} frames")
    
    # Check frame accuracy
    if transition.start_frame % 1 != 0:
        errors.append("Transition start frame must be integer")
    
    # Check for overlapping transitions
    overlapping = False
    for other_trans in timeline.transitions:
        if other_trans != transition:
            # Check if transitions overlap in time
            trans_start = transition.start_frame
            trans_end = transition.start_frame + transition.duration_frames
            other_start = other_trans.start_frame
            other_end = other_trans.start_frame + other_trans.duration_frames
            
            if (trans_start < other_end and trans_end > other_start):
                overlapping = True
                errors.append(f"Transition overlaps with {other_trans.type} at frame {other_trans.start_frame}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def auto_generate_transitions(timeline: Timeline, style: str = "professional") -> List[Transition]:
    """Automatically generates a list of transitions for a timeline.

    Args:
        timeline: The timeline to generate transitions for.
        style: The style of transitions to generate.

    Returns:
        A list of generated transitions.
    """
    transitions = []
    
    for i in range(len(timeline.clips) - 1):
        clip_a = timeline.clips[i]
        clip_b = timeline.clips[i + 1]
        
        # Estimate transition position (at boundary between clips)
        # This is simplified - would need actual timeline layout
        
        if style == "minimal":
            # Just cuts
            transition = Transition(
                type="cut",
                start_frame=int(clip_a.in_frame + clip_a.frame_count - 1),
                duration_frames=1,
                source_a_clip=clip_a.id,
                source_b_clip=clip_b.id
            )
        elif style == "professional":
            # Mix of cuts and crossfades
            if i % 3 == 0:  # Every third transition
                transition = Transition(
                    type="crossfade",
                    start_frame=int(clip_a.in_frame + clip_a.frame_count - 15),
                    duration_frames=30,
                    source_a_clip=clip_a.id,
                    source_b_clip=clip_b.id,
                    parameters={"curve": "ease_in_out"}
                )
            else:
                transition = Transition(
                    type="cut",
                    start_frame=int(clip_a.in_frame + clip_a.frame_count - 1),
                    duration_frames=1,
                    source_a_clip=clip_a.id,
                    source_b_clip=clip_b.id
                )
        elif style == "cinematic":
            # Longer, more dramatic transitions
            transition = Transition(
                type="crossfade",
                start_frame=int(clip_a.in_frame + clip_a.frame_count - 30),
                duration_frames=60,
                source_a_clip=clip_a.id,
                source_b_clip=clip_b.id,
                parameters={"curve": "sine"}
            )
        
        transitions.append(transition)
    
    return transitions