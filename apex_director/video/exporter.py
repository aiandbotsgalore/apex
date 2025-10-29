"""
Broadcast-Quality Video Export Engine
Exact FFmpeg specifications for 1080p/4K with professional audio/video encoding
"""

import os
import subprocess
import json
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from apex_director.video.timeline import Timeline
from apex_director.video.transitions import TransitionEngine
from apex_director.video.color_grader import ColorGrader
from apex_director.video.motion import MotionEngine


class VideoFormat(Enum):
    """Professional video formats"""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    MKV = "mkv"


class VideoResolution(Enum):
    """Professional video resolutions"""
    HD_1080P = (1920, 1080)
    UHD_4K = (3840, 2160)
    DCI_4K = (4096, 2160)
    HD_720P = (1280, 720)


class VideoCodec(Enum):
    """Professional video codecs"""
    H264 = "libx264"
    H265 = "libx265"
    PRORES = "prores_ks"
    DNXHR = "dnxhd"
    RAV1E = "librav1e"


class AudioCodec(Enum):
    """Professional audio codecs"""
    AAC = "aac"
    PCM = "pcm_s16le"
    AC3 = "ac3"
    DTS = "dts"
    OPUS = "libopus"


class ColorSpace(Enum):
    """Professional color spaces for export"""
    REC_709 = "bt709"
    REC_2020 = "bt2020"
    REC_601 = "bt601"


@dataclass
class ExportSettings:
    """Professional export settings"""
    video_codec: VideoCodec = VideoCodec.H264
    audio_codec: AudioCodec = AudioCodec.AAC
    video_format: VideoFormat = VideoFormat.MP4
    resolution: VideoResolution = VideoResolution.HD_1080P
    frame_rate: float = 30.0
    bitrate_video: int = 25000  # kbps
    bitrate_audio: int = 192  # kbps
    color_space: ColorSpace = ColorSpace.REC_709
    chroma_subsampling: str = "4:2:0"
    bit_depth: int = 8
    profile: str = "high"  # H.264 profile
    level: str = "4.0"  # H.264 level
    preset: str = "slow"  # Encoding preset
    tune: str = "film"  # Encoding tune
    gop_size: int = 30  # Group of Pictures size
    b_frames: int = 3  # B-frames count
    reference_frames: int = 3
    cabac: bool = True  # Context-Adaptive Binary Arithmetic Coding
    deinterlace: bool = False
    interlaced: bool = False
    top_field_first: bool = True
    
    # Audio settings
    sample_rate: int = 48000
    channels: int = 2
    audio_channels_layout: str = "stereo"
    
    # Quality settings
    crf: int = 18  # Constant Rate Factor (0-51, lower = better quality)
    two_pass: bool = True
    lookahead: int = 32  # VBV lookahead
    vbv_bufsize: int = 10000  # VBV buffer size
    
    # Professional features
    closed_gop: bool = True
    strict_gop: bool = False
    timecode: bool = False
    burn_in_timecode: bool = False
    loudness_normalize: bool = True
    peak_normalize: bool = False
    
    # Metadata
    title: str = ""
    artist: str = ""
    album: str = ""
    date: str = ""
    comment: str = ""
    
    def __post_init__(self):
        # Auto-adjust settings based on resolution
        if self.resolution == VideoResolution.UHD_4K:
            self.bitrate_video = 50000
            self.crf = 20
            self.level = "5.1"
        elif self.resolution == VideoResolution.DCI_4K:
            self.bitrate_video = 60000
            self.crf = 20
            self.level = "5.1"


@dataclass
class FFmpegCommand:
    """FFmpeg command builder for professional export"""
    inputs: List[str] = field(default_factory=list)
    input_options: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    output_options: List[str] = field(default_factory=list)
    output_file: str = ""
    overwrite: bool = True
    
    def build_command(self) -> List[str]:
        """Build FFmpeg command array"""
        cmd = ["ffmpeg"]
        
        # Input options
        for opt in self.input_options:
            cmd.extend(["-i", opt])
        
        # Input files
        for inp in self.inputs:
            cmd.append("-i")
            cmd.append(inp)
        
        # Filter complex
        if self.filters:
            filter_complex = ";".join(self.filters)
            cmd.extend(["-filter_complex", filter_complex])
        
        # Output options
        for opt in self.output_options:
            cmd.extend(opt)
        
        # Output file
        cmd.append(self.output_file)
        
        return cmd


class BroadcastExporter:
    """Professional broadcast-quality video exporter"""
    
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.export_settings = ExportSettings()
        self.output_path = "output"
        self.temp_path = "temp"
        
        # Ensure directories exist
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)
    
    def export_video(self, output_filename: str, settings: Optional[ExportSettings] = None) -> Dict[str, Union[bool, str, float]]:
        """Export video with professional settings"""
        if settings:
            self.export_settings = settings
        
        output_path = os.path.join(self.output_path, output_filename)
        
        # Build FFmpeg command
        ffmpeg_cmd = self._build_export_command(output_path)
        
        # Execute export
        try:
            print(f"Starting export to {output_path}")
            
            # Run FFmpeg
            process = subprocess.run(
                ffmpeg_cmd.build_command(),
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get file info
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            return {
                "success": True,
                "output_path": output_path,
                "file_size": file_size,
                "duration": self.timeline.duration,
                "message": "Export completed successfully"
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"FFmpeg error: {e.stderr}",
                "returncode": e.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Export failed: {str(e)}"
            }
    
    def _build_export_command(self, output_path: str) -> FFmpegCommand:
        """Build comprehensive FFmpeg command for broadcast export"""
        cmd = FFmpegCommand()
        
        # Input setup - in a real implementation, this would handle actual input files
        # For now, we'll create a command structure that demonstrates the professional settings
        
        if self.export_settings.two_pass:
            # First pass
            cmd_pass1 = self._build_pass_command(output_path, pass_number=1)
            return cmd_pass1
        else:
            # Single pass
            cmd = self._build_single_pass_command(output_path)
            return cmd
    
    def _build_single_pass_command(self, output_path: str) -> FFmpegCommand:
        """Build single-pass export command"""
        cmd = FFmpegCommand()
        
        # Video codec settings
        if self.export_settings.video_codec == VideoCodec.H264:
            cmd.output_options.extend([
                "-c:v", "libx264",
                "-preset", self.export_settings.preset,
                "-profile:v", self.export_settings.profile,
                "-level:v", self.export_settings.level,
                "-crf", str(self.export_settings.crf),
                "-b:v", f"{self.export_settings.bitrate_video}k"
            ])
        elif self.export_settings.video_codec == VideoCodec.H265:
            cmd.output_options.extend([
                "-c:v", "libx265",
                "-preset", self.export_settings.preset,
                "-crf", str(self.export_settings.crf),
                "-b:v", f"{self.export_settings.bitrate_video}k"
            ])
        
        # Resolution and frame rate
        cmd.output_options.extend([
            "-s", f"{self.export_settings.resolution[0]}x{self.export_settings.resolution[1]}",
            "-r", str(self.export_settings.frame_rate)
        ])
        
        # Color space and chroma
        cmd.output_options.extend([
            "-colorspace", self.export_settings.color_space.value,
            "-pix_fmt", self._get_pixel_format()
        ])
        
        # GOP and B-frames
        cmd.output_options.extend([
            "-g", str(self.export_settings.gop_size),
            "-bf", str(self.export_settings.b_frames),
            "-ref:v", str(self.export_settings.reference_frames)
        ])
        
        # Audio settings
        if self.export_settings.audio_codec == AudioCodec.AAC:
            cmd.output_options.extend([
                "-c:a", "aac",
                "-b:a", f"{self.export_settings.bitrate_audio}k",
                "-ar", str(self.export_settings.sample_rate),
                "-ac", str(self.export_settings.channels),
                "-acodec:a", self.export_settings.audio_codec.value
            ])
        elif self.export_settings.audio_codec == AudioCodec.PCM:
            cmd.output_options.extend([
                "-c:a", "pcm_s16le",
                "-ar", str(self.export_settings.sample_rate),
                "-ac", str(self.export_settings.channels)
            ])
        
        # Professional features
        if self.export_settings.cabac:
            cmd.output_options.append("-cabac")
        
        if self.export_settings.closed_gop:
            cmd.output_options.append("-sc_threshold")
            cmd.output_options.append("0")  # Disable scene change detection
        
        # VBV settings for quality control
        if self.export_settings.video_codec in [VideoCodec.H264, VideoCodec.H265]:
            cmd.output_options.extend([
                "-maxrate", f"{int(self.export_settings.bitrate_video * 1.2)}k",
                "-bufsize", f"{self.export_settings.vbv_bufsize}k",
                "-lookahead", str(self.export_settings.lookahead)
            ])
        
        # Metadata
        if self.export_settings.title:
            cmd.output_options.extend(["-metadata", f"title={self.export_settings.title}"])
        if self.export_settings.artist:
            cmd.output_options.extend(["-metadata", f"artist={self.export_settings.artist}"])
        
        # Format-specific options
        if self.export_settings.video_format == VideoFormat.MP4:
            cmd.output_options.extend([
                "-movflags", "+faststart",  # Web optimization
                "-metadata:g", "compatible_brands=mp42"
            ])
        elif self.export_settings.video_format == VideoFormat.MOV:
            cmd.output_options.extend([
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p"  # Ensure compatibility
            ])
        
        # Audio normalization
        if self.export_settings.loudness_normalize:
            cmd.filters.append("loudnorm=I=-23:LRA=7:TP=-1")
        
        # Deinterlacing if needed
        if self.export_settings.deinterlace:
            cmd.filters.append("yadif=mode=1:parity=auto")
        
        cmd.output_file = output_path
        return cmd
    
    def _build_pass_command(self, output_path: str, pass_number: int) -> FFmpegCommand:
        """Build multi-pass export command"""
        cmd = self._build_single_pass_command(output_path)
        
        if pass_number == 1:
            # First pass - analysis only
            cmd.output_options.extend([
                "-an",  # No audio in first pass
                "-f", "null", "/dev/null"  # Discard output
            ])
        else:
            # Second pass - actual encoding
            cmd.output_options.extend([
                "-pass", "2",
                "-passlogfile", f"{self.temp_path}/ffmpeg2pass"
            ])
        
        return cmd
    
    def _get_pixel_format(self) -> str:
        """Get appropriate pixel format based on settings"""
        if self.export_settings.bit_depth == 10:
            if self.export_settings.video_codec == VideoCodec.H265:
                return "yuv420p10le"
            else:
                return "yuv420p"  # H.264 doesn't support 10-bit in yuv420
        else:
            return "yuv420p"
    
    def export_pro_res(self, output_path: str, profile: str = "422") -> Dict[str, Union[bool, str]]:
        """Export in ProRes format for professional workflows"""
        settings = ExportSettings()
        settings.video_codec = VideoCodec.PRORES
        settings.profile = profile  # "proxy", "lt", "422", "4444"
        settings.bitrate_video = 0  # ProRes uses constant quality
        
        cmd = FFmpegCommand()
        cmd.output_options.extend([
            "-c:v", "prores_ks",
            "-profile:v", profile,
            "-pix_fmt", "yuv422p10le",
            "-s", f"{self.export_settings.resolution[0]}x{self.export_settings.resolution[1]}",
            "-r", str(self.export_settings.frame_rate)
        ])
        
        return self._execute_ffmpeg_command(cmd, output_path)
    
    def export_dnxhd(self, output_path: str, profile: str = "120") -> Dict[str, Union[bool, str]]:
        """Export in DNxHD format for Avid workflows"""
        settings = ExportSettings()
        settings.video_codec = VideoCodec.DNXHR
        settings.profile = profile  # "36", "120", "145", "220"
        
        cmd = FFmpegCommand()
        cmd.output_options.extend([
            "-c:v", "dnxhd",
            "-b:v", f"{profile}M",
            "-pix_fmt", "yuv422p",
            "-s", f"{self.export_settings.resolution[0]}x{self.export_settings.resolution[1]}",
            "-r", str(self.export_settings.frame_rate)
        ])
        
        return self._execute_ffmpeg_command(cmd, output_path)
    
    def _execute_ffmpeg_command(self, cmd: FFmpegCommand, output_path: str) -> Dict[str, Union[bool, str]]:
        """Execute FFmpeg command and return results"""
        cmd.output_file = output_path
        
        try:
            process = subprocess.run(
                cmd.build_command(),
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                "success": True,
                "output_path": output_path,
                "message": "Export completed successfully"
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"FFmpeg error: {e.stderr}",
                "returncode": e.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Export failed: {str(e)}"
            }


class QualityAnalyzer:
    """Analyze exported video for quality metrics"""
    
    def __init__(self):
        self.ffprobe_path = "ffprobe"
    
    def analyze_video_quality(self, video_path: str) -> Dict[str, Union[float, int, bool]]:
        """Analyze video quality metrics"""
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        
        try:
            # Get video info with ffprobe
            probe_cmd = [
                self.ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            video_info = json.loads(result.stdout)
            
            # Analyze video stream
            video_stream = None
            audio_stream = None
            
            for stream in video_info["streams"]:
                if stream["codec_type"] == "video":
                    video_stream = stream
                elif stream["codec_type"] == "audio":
                    audio_stream = stream
            
            if not video_stream:
                return {"error": "No video stream found"}
            
            # Calculate quality metrics
            quality_metrics = {
                "resolution": f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}",
                "frame_rate": float(video_stream.get('r_frame_rate', '0/1').split('/')[0]) / 
                            max(1, int(video_stream.get('r_frame_rate', '0/1').split('/')[1])),
                "video_codec": video_stream.get('codec_name', 'unknown'),
                "bitrate": int(float(video_stream.get('bit_rate', 0))) if video_stream.get('bit_rate') else 0,
                "duration": float(video_info["format"].get('duration', 0)),
                "file_size": int(video_info["format"].get('size', 0)),
                "has_audio": audio_stream is not None,
                "audio_codec": audio_stream.get('codec_name', 'none') if audio_stream else 'none',
                "audio_sample_rate": int(audio_stream.get('sample_rate', 0)) if audio_stream else 0,
                "audio_channels": int(audio_stream.get('channels', 0)) if audio_stream else 0
            }
            
            # Broadcast standards compliance check
            quality_metrics["broadcast_compliant"] = self._check_broadcast_compliance(quality_metrics)
            
            return quality_metrics
            
        except subprocess.CalledProcessError as e:
            return {"error": f"FFprobe error: {e.stderr}"}
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _check_broadcast_compliance(self, metrics: Dict) -> bool:
        """Check if video meets broadcast standards"""
        compliance_checks = []
        
        # Frame rate check (broadcast standards)
        frame_rate = metrics.get("frame_rate", 0)
        broadcast_frame_rates = [23.976, 24, 25, 29.97, 30, 50, 59.94, 60]
        compliance_checks.append(abs(frame_rate - round(frame_rate)) < 0.01 and 
                               frame_rate in broadcast_frame_rates)
        
        # Resolution check
        resolution_str = metrics.get("resolution", "0x0")
        if "x" in resolution_str:
            width, height = map(int, resolution_str.split("x"))
            # Check for valid broadcast resolutions
            valid_resolutions = [
                (1920, 1080), (1280, 720), (3840, 2160), (4096, 2160)
            ]
            compliance_checks.append((width, height) in valid_resolutions)
        
        # Bitrate check
        bitrate = metrics.get("bitrate", 0)
        # Minimum bitrate for HD content
        compliance_checks.append(bitrate >= 5000000)  # 5 Mbps minimum
        
        # Audio check
        audio_sample_rate = metrics.get("audio_sample_rate", 0)
        compliance_checks.append(audio_sample_rate in [44100, 48000])
        
        return all(compliance_checks)


class ExportManager:
    """Manage multiple export formats and presets"""
    
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.exporter = BroadcastExporter(timeline)
        self.quality_analyzer = QualityAnalyzer()
    
    def export_multiple_formats(self, base_filename: str) -> Dict[str, Dict]:
        """Export multiple formats from single timeline"""
        results = {}
        
        # Broadcast formats
        broadcast_formats = [
            ("broadcast_h264.mp4", self._get_broadcast_h264_settings()),
            ("broadcast_h265.mp4", self._get_broadcast_h265_settings()),
            ("broadcast_prores.mov", self._get_prores_settings()),
            ("broadcast_dnxhd.mov", self._get_dnxhd_settings())
        ]
        
        for filename, settings in broadcast_formats:
            result = self.exporter.export_video(filename, settings)
            results[filename] = result
            
            if result["success"]:
                # Analyze quality
                if "output_path" in result:
                    quality = self.quality_analyzer.analyze_video_quality(result["output_path"])
                    results[filename]["quality_analysis"] = quality
        
        return results
    
    def _get_broadcast_h264_settings(self) -> ExportSettings:
        """Get H.264 settings for broadcast"""
        settings = ExportSettings()
        settings.video_codec = VideoCodec.H264
        settings.video_format = VideoFormat.MP4
        settings.resolution = VideoResolution.HD_1080P
        settings.frame_rate = 29.97
        settings.crf = 18
        settings.profile = "high"
        settings.level = "4.0"
        settings.preset = "slow"
        settings.gop_size = 30
        settings.b_frames = 3
        settings.color_space = ColorSpace.REC_709
        return settings
    
    def _get_broadcast_h265_settings(self) -> ExportSettings:
        """Get H.265 settings for broadcast"""
        settings = ExportSettings()
        settings.video_codec = VideoCodec.H265
        settings.video_format = VideoFormat.MP4
        settings.resolution = VideoResolution.UHD_4K
        settings.frame_rate = 29.97
        settings.crf = 20
        settings.preset = "medium"
        settings.gop_size = 30
        settings.color_space = ColorSpace.REC_2020
        settings.bit_depth = 10
        return settings
    
    def _get_prores_settings(self) -> ExportSettings:
        """Get ProRes settings"""
        settings = ExportSettings()
        settings.video_codec = VideoCodec.PRORES
        settings.video_format = VideoFormat.MOV
        settings.resolution = VideoResolution.HD_1080P
        settings.frame_rate = 29.97
        settings.profile = "422"
        settings.bit_depth = 10
        return settings
    
    def _get_dnxhd_settings(self) -> ExportSettings:
        """Get DNxHD settings"""
        settings = ExportSettings()
        settings.video_codec = VideoCodec.DNXHR
        settings.video_format = VideoFormat.MOV
        settings.resolution = VideoResolution.HD_1080P
        settings.frame_rate = 29.97
        settings.profile = "220"
        return settings


# Utility functions for professional export
def validate_export_settings(settings: ExportSettings) -> Dict[str, Union[bool, List[str]]]:
    """Validate export settings for broadcast standards"""
    errors = []
    warnings = []
    
    # Frame rate validation
    valid_frame_rates = [23.976, 24, 25, 29.97, 30, 50, 59.94, 60]
    if settings.frame_rate not in valid_frame_rates:
        errors.append(f"Frame rate {settings.frame_rate} not in broadcast standards")
    
    # Resolution validation
    valid_resolutions = [
        VideoResolution.HD_720P,
        VideoResolution.HD_1080P,
        VideoResolution.UHD_4K,
        VideoResolution.DCI_4K
    ]
    if settings.resolution not in valid_resolutions:
        errors.append(f"Resolution {settings.resolution} not supported")
    
    # Bitrate validation
    if settings.bitrate_video < 1000:  # Minimum 1 Mbps
        warnings.append(f"Very low video bitrate: {settings.bitrate_video} kbps")
    
    # Audio validation
    valid_audio_rates = [44100, 48000]
    if settings.sample_rate not in valid_audio_rates:
        errors.append(f"Audio sample rate {settings.sample_rate} not in broadcast standards")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def create_preset_configs() -> Dict[str, ExportSettings]:
    """Create preset configurations for common workflows"""
    presets = {}
    
    # Web delivery preset
    web_preset = ExportSettings()
    web_preset.video_codec = VideoCodec.H264
    web_preset.video_format = VideoFormat.MP4
    web_preset.resolution = VideoResolution.HD_1080P
    web_preset.frame_rate = 30.0
    web_preset.crf = 23
    web_preset.bitrate_video = 8000
    web_preset.preset = "medium"
    web_preset.two_pass = False
    presets["web"] = web_preset
    
    # Broadcast preset
    broadcast_preset = ExportSettings()
    broadcast_preset.video_codec = VideoCodec.H264
    broadcast_preset.video_format = VideoFormat.MP4
    broadcast_preset.resolution = VideoResolution.HD_1080P
    broadcast_preset.frame_rate = 29.97
    broadcast_preset.crf = 18
    broadcast_preset.bitrate_video = 25000
    broadcast_preset.preset = "slow"
    broadcast_preset.two_pass = True
    presets["broadcast"] = broadcast_preset
    
    # Cinema preset
    cinema_preset = ExportSettings()
    cinema_preset.video_codec = VideoCodec.H265
    cinema_preset.video_format = VideoFormat.MP4
    cinema_preset.resolution = VideoResolution.UHD_4K
    cinema_preset.frame_rate = 24.0
    cinema_preset.crf = 16
    cinema_preset.bitrate_video = 50000
    cinema_preset.preset = "slow"
    cinema_preset.bit_depth = 10
    cinema_preset.color_space = ColorSpace.REC_2020
    presets["cinema"] = cinema_preset
    
    # Archive preset
    archive_preset = ExportSettings()
    archive_preset.video_codec = VideoCodec.PRORES
    archive_preset.video_format = VideoFormat.MOV
    archive_preset.resolution = VideoResolution.HD_1080P
    archive_preset.frame_rate = 29.97
    archive_preset.profile = "422"
    archive_preset.bit_depth = 10
    presets["archive"] = archive_preset
    
    return presets