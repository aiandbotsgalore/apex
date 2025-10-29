"""
Apex Director Professional Video Assembly System
Broadcast-quality video processing with frame-accurate timing
"""

from .timeline import Timeline, Clip, Transition, Marker, Frame, FrameRate, BeatType
from .transitions import TransitionEngine, TransitionType, TransitionEffect, TransitionGenerator
from .color_grader import ColorGrader, ColorCorrection, ColorSpace, ColorCurve, LUT
from .motion import MotionEngine, MotionType, CameraMovement, Keyframe, ParallaxLayer
from .exporter import BroadcastExporter, ExportSettings, VideoResolution, VideoCodec
from .assembler import VideoAssembler, AssemblyJob, ProcessingResult, AssemblyMode, QualityMode

__version__ = "1.0.0"
__all__ = [
    "Timeline", "Clip", "Transition", "Marker", "Frame", "FrameRate", "BeatType",
    "TransitionEngine", "TransitionType", "TransitionEffect", "TransitionGenerator",
    "ColorGrader", "ColorCorrection", "ColorSpace", "ColorCurve", "LUT",
    "MotionEngine", "MotionType", "CameraMovement", "Keyframe", "ParallaxLayer",
    "BroadcastExporter", "ExportSettings", "VideoResolution", "VideoCodec",
    "VideoAssembler", "AssemblyJob", "ProcessingResult", "AssemblyMode", "QualityMode"
]
