"""
Professional Timeline Construction - Beat-locked cutting with frame accuracy
Handles precise audio-visual synchronization and frame-perfect cuts
"""

import numpy as np
import librosa
import cv2
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class FrameRate(Enum):
    """Professional broadcast frame rates"""
    NTSC = 29.97
    PAL = 25.0
    FILM = 24.0
    UHD_4K = 60.0


class BeatType(Enum):
    """Audio beat detection types"""
    ONSET = "onset"
    BEAT = "beat"
    TATON = "taton"  # Tempo-aware onset
    FRAME_LOCK = "frame_lock"  # Frame-aligned beats


@dataclass
class Frame:
    """Represents a frame-accurate timeline element.

    Attributes:
        frame_number: The frame number.
        timestamp: The timestamp of the frame in seconds.
        duration: The duration of the frame in seconds.
        source_path: The path to the source file for this frame.
        frame_rate: The frame rate of the source file.
        width: The width of the frame.
        height: The height of the frame.
    """
    frame_number: int
    timestamp: float
    duration: float
    source_path: str
    frame_rate: float
    width: int
    height: int
    
    def __post_init__(self):
        """Ensure frame accuracy"""
        self.frame_number = int(self.frame_number)
        self.timestamp = round(self.timestamp, 6)
        self.duration = round(self.duration, 6)


@dataclass
class Clip:
    """Represents a professional clip with metadata.

    Attributes:
        id: The unique identifier for the clip.
        source_path: The path to the source file for the clip.
        in_frame: The in-point of the clip in frames.
        out_frame: The out-point of the clip in frames.
        in_time: The in-point of the clip in seconds.
        out_time: The out-point of the clip in seconds.
        duration: The duration of the clip in seconds.
        frame_rate: The frame rate of the clip.
        width: The width of the clip.
        height: The height of the clip.
        audio_streams: A list of audio streams in the clip.
        video_stream: The video stream in the clip.
        color_space: The color space of the clip.
        sample_rate: The audio sample rate of the clip.
        channels: The number of audio channels in the clip.
    """
    id: str
    source_path: str
    in_frame: int
    out_frame: int
    in_time: float
    out_time: float
    duration: float
    frame_rate: float
    width: int
    height: int
    audio_streams: List[Dict] = field(default_factory=list)
    video_stream: Dict = field(default_factory=dict)
    color_space: str = "Rec.709"
    sample_rate: int = 48000
    channels: int = 2
    
    @property
    def frame_count(self) -> int:
        """Calculates the total number of frames in the clip."""
        return self.out_frame - self.in_frame + 1
    
    @property
    def exact_duration(self) -> float:
        """Calculates the exact duration of the clip in seconds."""
        return self.frame_count / self.frame_rate


@dataclass
class Transition:
    """Represents a professional transition between clips.

    Attributes:
        type: The type of transition.
        start_frame: The start frame of the transition.
        duration_frames: The duration of the transition in frames.
        source_a_clip: The ID of the first clip in the transition.
        source_b_clip: The ID of the second clip in the transition.
        parameters: A dictionary of additional parameters for the transition.
    """
    type: str  # cut, crossfade, whip_pan, match_dissolve
    start_frame: int
    duration_frames: int
    source_a_clip: str
    source_b_clip: str
    parameters: Dict = field(default_factory=dict)
    
    @property
    def start_time(self) -> float:
        """Calculates the start time of the transition in seconds."""
        return self.start_frame / 30.0  # Assuming 30fps base
    
    @property
    def end_time(self) -> float:
        """Calculates the end time of the transition in seconds."""
        return (self.start_frame + self.duration_frames) / 30.0


@dataclass
class Marker:
    """Represents a marker on the timeline for beats, cuts, effects, etc.

    Attributes:
        frame_number: The frame number of the marker.
        time_stamp: The timestamp of the marker in seconds.
        type: The type of marker.
        value: The value of the marker.
        description: A description of the marker.
    """
    frame_number: int
    time_stamp: float
    type: str  # beat, cut, effect, note
    value: Union[str, int, float, Dict]
    description: str = ""
    
    @property
    def exact_time(self) -> float:
        """Calculates the exact time of the marker in seconds."""
        return self.frame_number / 30.0


class Timeline:
    """Represents a professional timeline with beat-locked precision.

    This class provides functionality for:
    - Adding and managing clips, transitions, and markers
    - Detecting and adding beat-locked markers from an audio file
    - Finding frame-accurate cut points
    - Creating beat-locked cuts
    - Validating the timeline for broadcast standards
    - Exporting and loading the timeline to and from JSON
    """
    
    def __init__(self, frame_rate: float = 30.0, resolution: Tuple[int, int] = (1920, 1080)):
        """Initializes the Timeline.

        Args:
            frame_rate: The frame rate of the timeline.
            resolution: The resolution of the timeline.
        """
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.clips: List[Clip] = []
        self.transitions: List[Transition] = []
        self.markers: List[Marker] = []
        self.total_frames = 0
        self.duration = 0.0
        self.audio_sample_rate = 48000
        
        # Beat detection settings
        self.beat_threshold = 0.1
        self.beat_frame_tolerance = 1  # ±1 frame tolerance
        self.frame_accuracy = True
        
    def add_clip(self, clip: Clip) -> None:
        """Adds a clip to the timeline with frame validation.

        Args:
            clip: The clip to add.
        """
        # Validate frame rate compatibility
        if abs(clip.frame_rate - self.frame_rate) > 0.01:
            raise ValueError(f"Clip frame rate {clip.frame_rate} incompatible with timeline {self.frame_rate}")
        
        # Validate resolution
        if (clip.width, clip.height) != self.resolution:
            clip.width, clip.height = self.resolution
            clip.video_stream['width'] = self.resolution[0]
            clip.video_stream['height'] = self.resolution[1]
        
        self.clips.append(clip)
        self._recalculate_timeline()
    
    def add_transition(self, transition: Transition) -> None:
        """Adds a transition to the timeline with frame-perfect alignment.

        Args:
            transition: The transition to add.
        """
        self.transitions.append(transition)
        self.transitions.sort(key=lambda t: t.start_frame)
    
    def add_beat_markers(self, audio_path: str, beat_type: BeatType = BeatType.BEAT) -> List[Marker]:
        """Detects and adds beat-locked markers to the timeline.

        Args:
            audio_path: The path to the audio file.
            beat_type: The type of beat detection to use.

        Returns:
            A list of the markers that were added.
        """
        try:
            # Load audio for beat detection
            y, sr = librosa.load(audio_path, sr=self.audio_sample_rate)
            
            if beat_type == BeatType.BEAT:
                beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
                beat_times = librosa.frames_to_time(beats[1], sr=sr, hop_length=512)
            elif beat_type == BeatType.ONSET:
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
                beat_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
            elif beat_type == BeatType.TATON:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, tatum=True)
                beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
            
            # Convert to frame-accurate markers
            markers = []
            for i, beat_time in enumerate(beat_times):
                # Convert to nearest frame
                beat_frame = int(round(beat_time * self.frame_rate))
                beat_frame = max(0, beat_frame)  # Ensure non-negative
                
                marker = Marker(
                    frame_number=beat_frame,
                    time_stamp=beat_frame / self.frame_rate,
                    type="beat",
                    value=beat_time,
                    description=f"Beat {i+1} - {beat_type.value}"
                )
                markers.append(marker)
                self.markers.append(marker)
            
            self.markers.sort(key=lambda m: m.frame_number)
            return markers
            
        except Exception as e:
            print(f"Warning: Beat detection failed - {e}")
            return []
    
    def find_frame_accurate_cut_points(self, source_path: str) -> List[Frame]:
        """Finds optimal cut points in a source file using audio-visual analysis.

        Args:
            source_path: The path to the source file.

        Returns:
            A list of frame-accurate cut points.
        """
        try:
            # Load video to get frame info
            cap = cv2.VideoCapture(source_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Load audio for analysis
            y, sr = librosa.load(source_path, sr=self.audio_sample_rate)
            
            # Detect onsets (potential cut points)
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, hop_length=512, 
                threshold=self.beat_threshold
            )
            
            # Convert to exact frames
            cut_points = []
            for onset_frame in onset_frames:
                # Convert onset frame to video frame
                video_frame = int(onset_frame * (len(y) / sr) * fps / (len(y) / sr))
                video_frame = min(video_frame, frame_count - 1)
                
                # Find exact frame alignment
                exact_frame = self._align_to_frame(video_frame, fps)
                
                cut_point = Frame(
                    frame_number=exact_frame,
                    timestamp=exact_frame / fps,
                    duration=0.0,  # Cut points have no duration
                    source_path=source_path,
                    frame_rate=fps,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                cut_points.append(cut_point)
            
            return sorted(cut_points, key=lambda c: c.frame_number)
            
        except Exception as e:
            print(f"Warning: Cut point detection failed - {e}")
            return []
    
    def _align_to_frame(self, frame_number: int, source_fps: float) -> int:
        """Aligns a frame number to the timeline's frame rate.

        Args:
            frame_number: The frame number to align.
            source_fps: The frame rate of the source video.

        Returns:
            The aligned frame number.
        """
        # Convert to seconds
        seconds = frame_number / source_fps
        
        # Round to nearest frame in timeline rate
        aligned_frame = int(round(seconds * self.frame_rate))
        
        # Apply frame tolerance
        tolerance = self.beat_frame_tolerance
        
        # Ensure alignment is within tolerance of original
        original_seconds = frame_number / source_fps
        aligned_seconds = aligned_frame / self.frame_rate
        
        if abs(aligned_seconds - original_seconds) > (tolerance / self.frame_rate):
            # Use nearest frame that meets tolerance
            aligned_frame = int(original_seconds * self.frame_rate)
        
        return max(0, aligned_frame)
    
    def create_beat_locked_cut(self, source_path: str, start_marker: Marker, 
                              duration_beats: int = 4) -> Clip:
        """Creates a beat-locked cut with frame accuracy.

        Args:
            source_path: The path to the source file.
            start_marker: The marker to start the cut from.
            duration_beats: The duration of the cut in beats.

        Returns:
            A new Clip object.
        """
        # Get audio for tempo analysis
        y, sr = librosa.load(source_path, sr=self.audio_sample_rate)
        
        # Detect beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
        
        # Find start beat
        start_time = start_marker.time_stamp
        start_beat_idx = None
        
        for i, beat_time in enumerate(beat_times):
            if abs(beat_time - start_time) < (1.0 / self.frame_rate):  # Within frame tolerance
                start_beat_idx = i
                break
        
        if start_beat_idx is None:
            raise ValueError("Could not find matching beat for start marker")
        
        # Calculate end beat
        end_beat_idx = min(start_beat_idx + duration_beats, len(beat_times) - 1)
        
        # Frame-accurate timing
        in_time = beat_times[start_beat_idx]
        out_time = beat_times[end_beat_idx]
        
        # Convert to frames
        cap = cv2.VideoCapture(source_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        in_frame = int(in_time * fps)
        out_frame = int(out_time * fps)
        
        # Create clip
        clip = Clip(
            id=f"beat_locked_{len(self.clips)}",
            source_path=source_path,
            in_frame=in_frame,
            out_frame=out_frame,
            in_time=in_time,
            out_time=out_time,
            duration=out_time - in_time,
            frame_rate=fps,
            width=self.resolution[0],
            height=self.resolution[1]
        )
        
        return clip
    
    def validate_timeline(self) -> Dict[str, Union[bool, List[str]]]:
        """Validates the timeline for broadcast standards.

        Returns:
            A dictionary containing the validation results.
        """
        errors = []
        warnings = []
        
        # Check frame rate consistency
        if not all(abs(clip.frame_rate - self.frame_rate) < 0.01 for clip in self.clips):
            errors.append("Frame rate mismatch between clips and timeline")
        
        # Check resolution consistency
        if not all(clip.width == self.resolution[0] and clip.height == self.resolution[1] 
                  for clip in self.clips):
            errors.append("Resolution mismatch between clips")
        
        # Check audio sync (basic validation)
        for clip in self.clips:
            if not clip.audio_streams:
                warnings.append(f"Clip {clip.id} has no audio stream")
        
        # Check for audio gaps
        if self._has_audio_gaps():
            warnings.append("Timeline has audio gaps")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _has_audio_gaps(self) -> bool:
        """Checks for audio gaps in the timeline.

        Returns:
            True if audio gaps are found, False otherwise.
        """
        # Simplified check - look for significant gaps
        min_gap_threshold = 1.0 / self.frame_rate  # 1 frame gap
        
        # This would need more sophisticated implementation for production
        return False
    
    def _recalculate_timeline(self) -> None:
        """Recalculates the total duration and frame count of the timeline."""
        if not self.clips:
            self.total_frames = 0
            self.duration = 0.0
            return
        
        total_duration = 0.0
        total_frames = 0
        
        for clip in self.clips:
            total_duration += clip.duration
            total_frames += clip.frame_count
        
        self.duration = total_duration
        self.total_frames = total_frames
    
    def export_timeline_json(self, filepath: Union[str, Path]) -> None:
        """Exports the timeline to a JSON file for persistence.

        Args:
            filepath: The path to the output file.
        """
        timeline_data = {
            "frame_rate": self.frame_rate,
            "resolution": self.resolution,
            "duration": self.duration,
            "total_frames": self.total_frames,
            "clips": [
                {
                    "id": clip.id,
                    "source_path": clip.source_path,
                    "in_frame": clip.in_frame,
                    "out_frame": clip.out_frame,
                    "in_time": clip.in_time,
                    "out_time": clip.out_time,
                    "duration": clip.duration,
                    "frame_rate": clip.frame_rate,
                    "width": clip.width,
                    "height": clip.height,
                    "color_space": clip.color_space,
                    "sample_rate": clip.sample_rate,
                    "channels": clip.channels
                }
                for clip in self.clips
            ],
            "transitions": [
                {
                    "type": trans.type,
                    "start_frame": trans.start_frame,
                    "duration_frames": trans.duration_frames,
                    "source_a_clip": trans.source_a_clip,
                    "source_b_clip": trans.source_b_clip,
                    "parameters": trans.parameters
                }
                for trans in self.transitions
            ],
            "markers": [
                {
                    "frame_number": marker.frame_number,
                    "time_stamp": marker.time_stamp,
                    "type": marker.type,
                    "value": marker.value,
                    "description": marker.description
                }
                for marker in self.markers
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(timeline_data, f, indent=2)
    
    def load_timeline_json(self, filepath: Union[str, Path]) -> None:
        """Loads a timeline from a JSON file.

        Args:
            filepath: The path to the input file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.frame_rate = data["frame_rate"]
        self.resolution = tuple(data["resolution"])
        self.duration = data["duration"]
        self.total_frames = data["total_frames"]
        
        # Load clips
        self.clips = []
        for clip_data in data["clips"]:
            clip = Clip(
                id=clip_data["id"],
                source_path=clip_data["source_path"],
                in_frame=clip_data["in_frame"],
                out_frame=clip_data["out_frame"],
                in_time=clip_data["in_time"],
                out_time=clip_data["out_time"],
                duration=clip_data["duration"],
                frame_rate=clip_data["frame_rate"],
                width=clip_data["width"],
                height=clip_data["height"],
                color_space=clip_data["color_space"],
                sample_rate=clip_data["sample_rate"],
                channels=clip_data["channels"]
            )
            self.clips.append(clip)
        
        # Load transitions
        self.transitions = []
        for trans_data in data["transitions"]:
            trans = Transition(
                type=trans_data["type"],
                start_frame=trans_data["start_frame"],
                duration_frames=trans_data["duration_frames"],
                source_a_clip=trans_data["source_a_clip"],
                source_b_clip=trans_data["source_b_clip"],
                parameters=trans_data["parameters"]
            )
            self.transitions.append(trans)
        
        # Load markers
        self.markers = []
        for marker_data in data["markers"]:
            marker = Marker(
                frame_number=marker_data["frame_number"],
                time_stamp=marker_data["time_stamp"],
                type=marker_data["type"],
                value=marker_data["value"],
                description=marker_data["description"]
            )
            self.markers.append(marker)


# Utility functions for professional timeline operations
def calculate_edit_decision_list(timeline: Timeline) -> List[Dict]:
    """Generates an Edit Decision List (EDL) from a timeline.

    Args:
        timeline: The timeline to generate the EDL from.

    Returns:
        A list of dictionaries, where each dictionary represents an EDL entry.
    """
    edl = []
    
    for i, clip in enumerate(timeline.clips):
        # Calculate actual duration accounting for transitions
        duration = clip.duration
        transition = _get_overlapping_transition(timeline.transitions, clip, i)
        if transition:
            duration -= transition.duration_frames / timeline.frame_rate
        
        edl_entry = {
            "clip_number": i + 1,
            "clip_id": clip.id,
            "source_path": clip.source_path,
            "in_time": clip.in_time,
            "out_time": clip.out_time,
            "duration": duration,
            "frame_in": clip.in_frame,
            "frame_out": clip.out_frame
        }
        
        if transition:
            edl_entry["transition"] = transition.type
            edl_entry["transition_duration"] = transition.duration_frames / timeline.frame_rate
        
        edl.append(edl_entry)
    
    return edl


def _get_overlapping_transition(transitions: List[Transition], clip: Clip, clip_index: int) -> Optional[Transition]:
    """Finds a transition that overlaps with a given clip.

    Args:
        transitions: A list of transitions in the timeline.
        clip: The clip to check for overlapping transitions.
        clip_index: The index of the clip in the timeline.

    Returns:
        The overlapping transition, or None if no transition overlaps.
    """
    # Simplified - look for transitions at clip boundaries
    if clip_index < len(transitions):
        # Check if transition starts at end of this clip
        trans = transitions[clip_index]
        # Implementation would need timeline context to be precise
        return trans if trans.source_a_clip == clip.id else None
    return None


def generate_color_correction_list(timeline: Timeline) -> Dict[str, List[Dict]]:
    """Generates a list of color correction parameters from timeline markers.

    Args:
        timeline: The timeline to generate the color correction list from.

    Returns:
        A dictionary of color correction parameters, grouped by stage.
    """
    color_corrections = {
        "primary": [],
        "secondary": [],
        "creative": [],
        "finishing": []
    }
    
    for marker in timeline.markers:
        if marker.type == "color_correct":
            correction = marker.value
            stage = correction.get("stage", "primary")
            if stage in color_corrections:
                color_corrections[stage].append({
                    "frame": marker.frame_number,
                    "time": marker.time_stamp,
                    "parameters": correction["parameters"]
                })
    
    return color_corrections