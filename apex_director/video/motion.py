"""
Professional Motion Effects - Camera movements and Ken Burns effect
Implements smooth camera movements and parallax effects for cinematic quality
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import math
from apex_director.video.timeline import Timeline


class MotionType(Enum):
    """Professional motion types"""
    KEN_BURNS_ZOOM = "ken_burns_zoom"
    KEN_BURNS_PAN = "ken_burns_pan"
    KEN_BURNS_COMBINED = "ken_burns_combined"
    PARALLAX_2D = "parallax_2d"
    PARALLAX_3D = "parallax_3d"
    DOLLY_ZOOM = "dolly_zoom"
    TRUCKING = "trucking"
    BOOMING = "booming"
    PANNING = "panning"
    TILTING = "tilting"


@dataclass
class Keyframe:
    """Motion keyframe with interpolation support"""
    time: float  # Seconds
    position_x: float = 0.0
    position_y: float = 0.0
    zoom: float = 1.0
    rotation: float = 0.0  # Degrees
    interpolation: str = "linear"  # linear, ease_in, ease_out, ease_in_out, bezier


@dataclass
class CameraMovement:
    """Professional camera movement definition"""
    motion_type: MotionType
    start_time: float
    duration: float
    keyframes: List[Keyframe] = field(default_factory=list)
    parameters: Dict[str, Union[float, int, str]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.keyframes:
            # Create default keyframes
            self.keyframes = [
                Keyframe(time=0.0, interpolation="ease_out"),
                Keyframe(time=self.duration, interpolation="ease_in")
            ]


@dataclass
class ParallaxLayer:
    """Parallax layer for pseudo-3D effect"""
    depth: float = 0.0  # 0.0 (foreground) to 1.0 (background)
    movement_factor: float = 1.0  # How much this layer moves
    blur_strength: float = 0.0  # Depth of field blur
    opacity: float = 1.0
    mask: Optional[np.ndarray] = None


class MotionEngine:
    """Professional motion effects engine"""
    
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.frame_width = timeline.resolution[0]
        self.frame_height = timeline.resolution[1]
        self.frame_rate = timeline.frame_rate
        
        # Motion tracking
        self.camera_movements: List[CameraMovement] = []
        self.parallax_layers: List[ParallaxLayer] = []
        
        # Quality settings
        self.enable_interpolation = True
        self.enable_edge_feathering = True
        self.motion_blur_enabled = True
    
    def add_camera_movement(self, movement: CameraMovement) -> None:
        """Add camera movement to timeline"""
        self.camera_movements.append(movement)
        self.camera_movements.sort(key=lambda m: m.start_time)
    
    def add_parallax_layer(self, layer: ParallaxLayer) -> None:
        """Add parallax layer for depth effect"""
        self.parallax_layers.append(layer)
        self.parallax_layers.sort(key=lambda l: l.depth, reverse=True)
    
    def apply_ken_burns_effect(self, frame: np.ndarray, movement: CameraMovement, 
                             current_time: float) -> np.ndarray:
        """Apply Ken Burns effect (pan and zoom)"""
        if movement.motion_type not in [MotionType.KEN_BURNS_ZOOM, 
                                       MotionType.KEN_BURNS_PAN, 
                                       MotionType.KEN_BURNS_COMBINED]:
            return frame
        
        # Calculate interpolated parameters
        params = self._interpolate_movement(movement, current_time)
        
        # Apply Ken Burns effect
        if movement.motion_type == MotionType.KEN_BURNS_ZOOM:
            return self._apply_ken_burns_zoom(frame, params)
        elif movement.motion_type == MotionType.KEN_BURNS_PAN:
            return self._apply_ken_burns_pan(frame, params)
        else:  # COMBINED
            return self._apply_ken_burns_combined(frame, params)
    
    def apply_parallax_effect(self, frame: np.ndarray, layers: List[ParallaxLayer], 
                            movement_vector: Tuple[float, float]) -> np.ndarray:
        """Apply 2D/3D parallax effect"""
        if not layers:
            return frame
        
        result = np.zeros_like(frame)
        
        # Process layers from background to foreground
        for layer in sorted(layers, key=lambda l: l.depth):
            layer_frame = self._extract_layer_from_frame(frame, layer)
            
            # Calculate movement based on depth and parallax
            parallax_offset = self._calculate_parallax_offset(
                movement_vector, layer.depth, layer.movement_factor
            )
            
            # Apply movement with edge handling
            moved_layer = self._apply_layer_movement(layer_frame, parallax_offset, layer.blur_strength)
            
            # Apply opacity
            moved_layer = moved_layer * layer.opacity
            
            # Composite with result
            result = self._composite_layers(result, moved_layer)
        
        return np.clip(result, 0, 1)
    
    def apply_dolly_zoom_effect(self, frame: np.ndarray, zoom_factor: float, 
                              current_time: float, speed: float = 1.0) -> np.ndarray:
        """Apply dolly zoom (Vertigo effect)"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Calculate zoom and crop parameters
        target_width = int(width / zoom_factor)
        target_height = int(height / zoom_factor)
        
        # Ensure crop dimensions are valid
        target_width = max(1, min(target_width, width))
        target_height = max(1, min(target_height, height))
        
        # Calculate crop position (centered)
        crop_x = center_x - target_width // 2
        crop_y = center_y - target_height // 2
        
        # Ensure crop stays within frame bounds
        crop_x = max(0, min(crop_x, width - target_width))
        crop_y = max(0, min(crop_y, height - target_height))
        
        # Crop and resize
        cropped = frame[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
        result = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Apply dolly zoom motion characteristics
        if self.motion_blur_enabled:
            result = self._apply_motion_blur(result, speed, current_time)
        
        return result
    
    def _apply_ken_burns_zoom(self, frame: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply Ken Burns zoom effect"""
        zoom = params.get("zoom", 1.0)
        center_x = params.get("position_x", 0.5) * self.frame_width
        center_y = params.get("position_y", 0.5) * self.frame_height
        
        height, width = frame.shape[:2]
        
        # Calculate zoomed dimensions
        new_width = int(width / zoom)
        new_height = int(height / zoom)
        
        # Ensure valid dimensions
        new_width = max(1, min(new_width, width))
        new_height = max(1, min(new_height, height))
        
        # Calculate crop position (centered on focal point)
        crop_x = int(center_x - new_width / 2)
        crop_y = int(center_y - new_height / 2)
        
        # Ensure crop stays within bounds
        crop_x = max(0, min(crop_x, width - new_width))
        crop_y = max(0, min(crop_y, height - new_height))
        
        # Crop and resize to original size
        cropped = frame[crop_y:crop_y + new_height, crop_x:crop_x + new_width]
        result = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
        
        return result
    
    def _apply_ken_burns_pan(self, frame: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply Ken Burns pan effect"""
        offset_x = params.get("position_x", 0.0) * self.frame_width * 0.2
        offset_y = params.get("position_y", 0.0) * self.frame_height * 0.2
        
        height, width = frame.shape[:2]
        
        # Create transformation matrix
        matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        
        # Apply transformation with edge handling
        result = cv2.warpAffine(frame, matrix, (width, height), 
                              borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    def _apply_ken_burns_combined(self, frame: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply combined Ken Burns pan and zoom"""
        # First apply zoom
        zoomed = self._apply_ken_burns_zoom(frame, params)
        
        # Then apply pan
        result = self._apply_ken_burns_pan(zoomed, params)
        
        return result
    
    def _calculate_parallax_offset(self, movement_vector: Tuple[float, float], 
                                  depth: float, movement_factor: float) -> Tuple[float, float]:
        """Calculate parallax offset based on depth and movement"""
        base_x, base_y = movement_vector
        
        # Foreground moves more than background
        parallax_x = base_x * (1.0 - depth) * movement_factor
        parallax_y = base_y * (1.0 - depth) * movement_factor
        
        return (parallax_x, parallax_y)
    
    def _extract_layer_from_frame(self, frame: np.ndarray, layer: ParallaxLayer) -> np.ndarray:
        """Extract content for parallax layer"""
        if layer.mask is not None:
            # Apply custom mask
            mask_3ch = np.stack([layer.mask, layer.mask, layer.mask], axis=2)
            return frame * mask_3ch
        else:
            # Return full frame (would need segmentation for real implementation)
            return frame.copy()
    
    def _apply_layer_movement(self, layer_frame: np.ndarray, offset: Tuple[float, float], 
                            blur_strength: float) -> np.ndarray:
        """Apply movement to parallax layer"""
        height, width = layer_frame.shape[:2]
        offset_x, offset_y = offset
        
        # Create transformation matrix
        matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        
        # Apply movement
        moved_layer = cv2.warpAffine(layer_frame, matrix, (width, height),
                                   borderMode=cv2.BORDER_REFLECT)
        
        # Apply depth blur if specified
        if blur_strength > 0:
            moved_layer = cv2.GaussianBlur(moved_layer, 
                                         (int(blur_strength * 10) * 2 + 1, 
                                          int(blur_strength * 10) * 2 + 1), 0)
        
        return moved_layer
    
    def _composite_layers(self, base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Composite layers with proper blending"""
        # Simple alpha composite
        return base + overlay
    
    def _apply_motion_blur(self, frame: np.ndarray, speed: float, current_time: float) -> np.ndarray:
        """Apply motion blur effect"""
        # Calculate blur based on movement speed
        blur_amount = max(0, min(int(speed * 3), 15))  # Max 15 pixels blur
        
        if blur_amount > 0:
            # Create motion blur kernel in direction of movement
            kernel_size = blur_amount * 2 + 1
            kernel = np.ones((1, kernel_size), np.float32) / kernel_size
            result = cv2.filter2D(frame, -1, kernel)
        else:
            result = frame
        
        return result
    
    def _interpolate_movement(self, movement: CameraMovement, current_time: float) -> Dict[str, float]:
        """Interpolate camera movement parameters"""
        # Find surrounding keyframes
        if len(movement.keyframes) < 2:
            return {
                "position_x": 0.0,
                "position_y": 0.0,
                "zoom": 1.0,
                "rotation": 0.0
            }
        
        # Sort keyframes by time
        keyframes = sorted(movement.keyframes, key=lambda k: k.time)
        
        # Find interpolation segment
        if current_time <= keyframes[0].time:
            start_key = keyframes[0]
            end_key = keyframes[1]
            progress = 0.0
        elif current_time >= keyframes[-1].time:
            start_key = keyframes[-2]
            end_key = keyframes[-1]
            progress = 1.0
        else:
            # Find segment containing current_time
            for i in range(len(keyframes) - 1):
                if keyframes[i].time <= current_time <= keyframes[i + 1].time:
                    start_key = keyframes[i]
                    end_key = keyframes[i + 1]
                    segment_duration = end_key.time - start_key.keyframes[0].time
                    progress = (current_time - start_key.time) / segment_duration if segment_duration > 0 else 0.0
                    break
        
        # Apply interpolation method
        interpolation = start_key.interpolation
        if interpolation == "ease_in":
            progress = progress * progress
        elif interpolation == "ease_out":
            progress = 1 - (1 - progress) * (1 - progress)
        elif interpolation == "ease_in_out":
            if progress < 0.5:
                progress = 2 * progress * progress
            else:
                progress = 1 - 2 * (1 - progress) * (1 - progress)
        
        # Interpolate parameters
        result = {
            "position_x": start_key.position_x + (end_key.position_x - start_key.position_x) * progress,
            "position_y": start_key.position_y + (end_key.position_y - start_key.position_y) * progress,
            "zoom": start_key.zoom + (end_key.zoom - start_key.zoom) * progress,
            "rotation": start_key.rotation + (end_key.rotation - start_key.rotation) * progress
        }
        
        return result
    
    def create_parallax_layers(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> List[ParallaxLayer]:
        """Create parallax layers from depth map or image analysis"""
        layers = []
        
        if depth_map is not None:
            # Use provided depth map
            height, width = depth_map.shape
            layer_count = 5  # Create 5 depth layers
            
            for i in range(layer_count):
                depth_min = i / layer_count
                depth_max = (i + 1) / layer_count
                
                # Create mask for this depth range
                mask = (depth_map >= depth_min) & (depth_map < depth_max)
                
                layer = ParallaxLayer(
                    depth=(depth_min + depth_max) / 2,
                    movement_factor=1.0 - (depth_min + depth_max) / 2,
                    blur_strength=(depth_min + depth_max) / 2 * 0.5,
                    opacity=1.0,
                    mask=mask
                )
                layers.append(layer)
        else:
            # Automatic layer creation based on image analysis
            layers = self._analyze_image_for_layers(frame)
        
        return layers
    
    def _analyze_image_for_layers(self, frame: np.ndarray) -> List[ParallaxLayer]:
        """Automatically analyze image to create parallax layers"""
        layers = []
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Detect edges (likely foreground objects)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect regions using watershed or similar
        # This is simplified - would need more sophisticated analysis
        
        # Create layers based on edge density and distance from center
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        
        # Distance from center
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        normalized_distance = distance / max_distance
        
        # Create 3 parallax layers
        for i, depth in enumerate([0.3, 0.6, 0.9]):  # Foreground, mid, background
            # Create mask based on distance and edge density
            if depth < 0.5:  # Foreground
                mask = (normalized_distance < 0.4) | (edges > 0)
            elif depth < 0.8:  # Mid-ground
                mask = (normalized_distance >= 0.2) & (normalized_distance <= 0.7)
            else:  # Background
                mask = normalized_distance > 0.5
            
            layer = ParallaxLayer(
                depth=depth,
                movement_factor=1.0 - depth,
                blur_strength=depth * 0.3,
                opacity=0.9,
                mask=mask
            )
            layers.append(layer)
        
        return layers


class CameraTracking:
    """Advanced camera tracking for motion effects"""
    
    def __init__(self):
        self.tracked_features = []
        self.motion_history = []
    
    def detect_motion_in_sequence(self, frame_sequence: List[np.ndarray]) -> List[Dict[str, float]]:
        """Detect motion between frames for automatic camera effects"""
        motions = []
        
        for i in range(1, len(frame_sequence)):
            prev_frame = frame_sequence[i - 1]
            curr_frame = frame_sequence[i]
            
            # Detect optical flow
            motion = self._calculate_optical_flow(prev_frame, curr_frame)
            motions.append(motion)
        
        return motions
    
    def _calculate_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, float]:
        """Calculate optical flow between two frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor((frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor((frame2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, 
            corners1=np.array([[]], dtype=np.float32),
            corners2=np.array([[]], dtype=np.float32)
        )[0]
        
        # Analyze motion
        motion_x = 0.0
        motion_y = 0.0
        motion_magnitude = 0.0
        
        if flow is not None and len(flow) > 0:
            # Calculate average motion
            motion_vectors = []
            for point in flow:
                if point is not None:
                    motion_vectors.append(point[0])
            
            if motion_vectors:
                motion_x = np.mean([mv[0] for mv in motion_vectors])
                motion_y = np.mean([mv[1] for mv in motion_vectors])
                motion_magnitude = np.sqrt(motion_x**2 + motion_y**2)
        
        return {
            "motion_x": motion_x,
            "motion_y": motion_y,
            "magnitude": motion_magnitude
        }


class MotionPreview:
    """Real-time motion effect preview"""
    
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.preview_resolution = (1920, 1080)
        self.preview_fps = 30.0
        
    def generate_motion_preview(self, movement: CameraMovement, 
                              sample_frame: np.ndarray) -> List[np.ndarray]:
        """Generate preview frames for motion effect"""
        frames = []
        motion_engine = MotionEngine(self.timeline)
        
        # Resize sample frame for preview
        sample_frame = cv2.resize(sample_frame, self.preview_resolution)
        
        # Generate frames across movement duration
        total_frames = int(movement.duration * self.preview_fps)
        
        for i in range(total_frames):
            current_time = i / self.preview_fps
            current_frame = sample_frame.copy()
            
            # Apply motion effect
            if movement.motion_type == MotionType.KEN_BURNS_ZOOM:
                result = motion_engine.apply_ken_burns_effect(
                    current_frame, movement, current_time
                )
            elif movement.motion_type == MotionType.DOLLY_ZOOM:
                # Calculate zoom based on time and movement parameters
                zoom_factor = movement.parameters.get("zoom_factor", 1.0)
                speed = movement.parameters.get("speed", 1.0)
                result = motion_engine.apply_dolly_zoom_effect(
                    current_frame, zoom_factor, current_time, speed
                )
            else:
                result = current_frame
            
            frames.append(result)
        
        return frames
    
    def adjust_motion_parameters(self, movement: CameraMovement, 
                               parameter_updates: Dict[str, Union[float, int, str]]) -> None:
        """Adjust motion parameters in real-time"""
        for key, value in parameter_updates.items():
            movement.parameters[key] = value


# Utility functions for professional motion effects
def validate_motion_effect(movement: CameraMovement, timeline: Timeline) -> Dict[str, Union[bool, List[str]]]:
    """Validate motion effect for broadcast standards"""
    errors = []
    warnings = []
    
    # Check duration
    if movement.duration <= 0:
        errors.append("Motion effect duration must be positive")
    
    # Check zoom limits (avoid excessive zoom that causes quality loss)
    for keyframe in movement.keyframes:
        if keyframe.zoom > 5.0:
            warnings.append(f"High zoom factor detected: {keyframe.zoom}")
        elif keyframe.zoom < 0.2:
            warnings.append(f"Low zoom factor detected: {keyframe.zoom}")
    
    # Check motion speed
    speed_check = _calculate_motion_speed(movement)
    if speed_check > 100:
        warnings.append(f"High motion speed detected: {speed_check}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def _calculate_motion_speed(movement: CameraMovement) -> float:
    """Calculate motion speed for validation"""
    if len(movement.keyframes) < 2:
        return 0.0
    
    total_distance = 0.0
    total_time = 0.0
    
    for i in range(1, len(movement.keyframes)):
        prev_key = movement.keyframes[i - 1]
        curr_key = movement.keyframes[i]
        
        # Calculate distance in parameter space
        distance = math.sqrt(
            (curr_key.position_x - prev_key.position_x)**2 +
            (curr_key.position_y - prev_key.position_y)**2 +
            (curr_key.zoom - prev_key.zoom)**2
        )
        
        total_distance += distance
        total_time += curr_key.time - prev_key.time
    
    return total_distance / total_time if total_time > 0 else 0.0


def auto_generate_ken_burns(movement: CameraMovement, duration: float = 4.0) -> CameraMovement:
    """Automatically generate Ken Burns effect"""
    movement.duration = duration
    
    # Create keyframes for typical Ken Burns
    keyframes = [
        Keyframe(time=0.0, position_x=0.3, position_y=0.3, zoom=1.2, 
                interpolation="ease_out"),
        Keyframe(time=duration * 0.7, position_x=0.7, position_y=0.7, zoom=1.0,
                interpolation="linear"),
        Keyframe(time=duration, position_x=0.8, position_y=0.8, zoom=1.1,
                interpolation="ease_in")
    ]
    
    movement.keyframes = keyframes
    return movement