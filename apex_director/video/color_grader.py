"""
4-Stage Professional Color Grading Pipeline
Broadcast-quality color correction with Rec.709/Rec.2020 support
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import colorsys
from apex_director.video.timeline import Timeline


class ColorSpace(Enum):
    """Professional color spaces"""
    REC_709 = "Rec.709"
    REC_2020 = "Rec.2020"
    DCI_P3 = "DCI-P3"
    REC_601 = "Rec.601"


class ColorCurve(Enum):
    """Professional color curves"""
    LINEAR = "linear"
    LOG = "log"
    CINEON = "cineon"
    RED_LOG = "red_log"
    ALEXA_V3_LOG_C = "alexa_v3_log_c"


@dataclass
class ColorCorrection:
    """Represents a set of individual color correction parameters.

    Attributes:
        exposure: The exposure adjustment in stops.
        contrast: The contrast adjustment.
        brightness: The brightness adjustment.
        saturation: The saturation adjustment.
        hue: The hue adjustment in degrees.
        gamma: The gamma adjustment.
        shadows: The shadows adjustment.
        midtones: The midtones adjustment.
        highlights: The highlights adjustment.
        blacks: The blacks adjustment.
        whites: The whites adjustment.
        temperature: The white balance temperature in Kelvin.
        tint: The white balance tint.
    """
    exposure: float = 0.0  # -2.0 to +2.0 stops
    contrast: float = 0.0  # -100 to +100
    brightness: float = 0.0  # -100 to +100
    saturation: float = 0.0  # -100 to +100
    hue: float = 0.0  # -180 to +180 degrees
    gamma: float = 1.0  # 0.1 to 10.0
    
    # Advanced controls
    shadows: float = 0.0  # -100 to +100
    midtones: float = 0.0  # -100 to +100
    highlights: float = 0.0  # -100 to +100
    blacks: float = 0.0  # -100 to +100
    whites: float = 0.0  # -100 to +100
    
    # White balance
    temperature: float = 5600.0  # 1000K to 10000K
    tint: float = 0.0  # -100 to +100


@dataclass
class SkinToneMask:
    """Represents the parameters for isolating skin tones.

    Attributes:
        enabled: Whether the skin tone mask is enabled.
        y_min: The minimum Y value in the YCbCr color space.
        y_max: The maximum Y value in the YCbCr color space.
        cb_min: The minimum Cb value in the YCbCr color space.
        cb_max: The maximum Cb value in the YCbCr color space.
        cr_min: The minimum Cr value in the YCbCr color space.
        cr_max: The maximum Cr value in the YCbCr color space.
        softness: The softness of the mask.
    """
    enabled: bool = False
    y_min: float = 0.0  # YCbCr Y channel min
    y_max: float = 1.0  # YCbCr Y channel max
    cb_min: float = 0.0  # YCbCr Cb channel min
    cb_max: float = 1.0  # YCbCr Cb channel max
    cr_min: float = 0.0  # YCbCr Cr channel min
    cr_max: float = 1.0  # YCbCr Cr channel max
    softness: float = 0.5  # 0.0 to 1.0


@dataclass
class SelectiveSaturation:
    """Represents the parameters for selective color desaturation.

    Attributes:
        enabled: Whether selective saturation is enabled.
        target_colors: A list of HSV color ranges to target.
        desaturation_amount: The amount of desaturation to apply.
        softness: The softness of the desaturation effect.
    """
    enabled: bool = False
    target_colors: List[Tuple[float, float, float]] = field(default_factory=list)  # HSV ranges
    desaturation_amount: float = 1.0  # 0.0 to 1.0
    softness: float = 0.5


@dataclass
class LUT:
    """Represents a Look-Up Table (LUT) for creative grading.

    Attributes:
        name: The name of the LUT.
        file_path: The path to the LUT file.
        type: The type of LUT ("3d" or "1d").
        cube_size: The size of the LUT cube.
        data: The LUT data as a NumPy array.
    """
    name: str = ""
    file_path: Optional[str] = None
    type: str = "3d"  # "3d" or "1d"
    cube_size: int = 33
    data: Optional[np.ndarray] = None


@dataclass
class FilmGrain:
    """Represents the parameters for film grain simulation.

    Attributes:
        enabled: Whether film grain is enabled.
        intensity: The intensity of the film grain.
        grain_type: The type of film grain ("camera", "film", or "digital").
        seed: The random seed for the film grain.
    """
    enabled: bool = False
    intensity: float = 0.1  # 0.0 to 1.0
    grain_type: str = "camera"  # "camera", "film", "digital"
    seed: int = 42


@dataclass
class Vignette:
    """Represents the parameters for a vignette effect.

    Attributes:
        enabled: Whether the vignette is enabled.
        amount: The amount of the vignette.
        midpoint: The midpoint of the vignette.
        roundness: The roundness of the vignette.
        feather: The feather of the vignette.
    """
    enabled: bool = False
    amount: float = 0.3  # 0.0 to 1.0
    midpoint: float = 0.7  # 0.0 to 1.0
    roundness: float = 1.0  # 0.0 to 2.0
    feather: float = 0.3  # 0.0 to 1.0


@dataclass
class ChromaticAberration:
    """Represents the parameters for chromatic aberration simulation.

    Attributes:
        enabled: Whether chromatic aberration is enabled.
        amount: The amount of chromatic aberration.
        channel_bias: The channel to bias the aberration towards.
    """
    enabled: bool = False
    amount: float = 0.1  # 0.0 to 1.0
    channel_bias: str = "blue"  # "red", "green", "blue"


class ColorGrader:
    """A 4-stage professional color grading pipeline.

    This class provides a comprehensive color grading solution with support for:
    - Primary and secondary color correction
    - Creative grading with Look-Up Tables (LUTs)
    - Finishing effects such as film grain, sharpening, and vignettes
    - Broadcast-quality color spaces (Rec.709, Rec.2020)
    """
    
    def __init__(self, timeline: Timeline):
        """Initializes the ColorGrader.

        Args:
            timeline: The timeline to be graded.
        """
        self.timeline = timeline
        self.color_space = ColorSpace.REC_709
        self.curve = ColorCurve.LINEAR
        
        # Stage corrections
        self.primary_correction = ColorCorrection()
        self.secondary_correction = ColorCorrection()
        self.creative_lut = LUT()
        self.finishing_effects = {
            "film_grain": FilmGrain(),
            "vignette": Vignette(),
            "chromatic_aberration": ChromaticAberration(),
            "sharpening": 0.0  # 0.0 to 1.0
        }
        
        # Masks
        self.skin_tone_mask = SkinToneMask()
        self.selective_saturation = SelectiveSaturation()
        
        # Caching
        self._conversion_matrices = {}
        self._setup_color_conversion()
    
    def stage_1_primary_correction(self, frame: np.ndarray) -> np.ndarray:
        """Performs stage 1 primary color correction.

        This stage includes adjustments for exposure, white balance, and contrast.

        Args:
            frame: The input frame to be corrected.

        Returns:
            The corrected frame.
        """
        result = frame.astype(np.float32) / 255.0
        
        # Exposure adjustment (logarithmic)
        if self.primary_correction.exposure != 0:
            exposure_factor = 2 ** self.primary_correction.exposure
            result = np.clip(result * exposure_factor, 0, 1)
        
        # Contrast adjustment
        if self.primary_correction.contrast != 0:
            contrast_factor = (259 * (self.primary_correction.contrast + 255)) / (255 * (259 - self.primary_correction.contrast))
            result = np.clip(contrast_factor * (result - 0.5) + 0.5, 0, 1)
        
        # Brightness adjustment
        if self.primary_correction.brightness != 0:
            brightness_offset = self.primary_correction.brightness / 255.0
            result = np.clip(result + brightness_offset, 0, 1)
        
        # Gamma correction
        if self.primary_correction.gamma != 1.0:
            gamma_inv = 1.0 / self.primary_correction.gamma
            result = np.power(result, gamma_inv)
        
        # White balance adjustment
        if self.primary_correction.temperature != 5600.0 or self.primary_correction.tint != 0:
            result = self._apply_white_balance(result, self.primary_correction.temperature, self.primary_correction.tint)
        
        return np.clip(result, 0, 1)
    
    def stage_2_secondary_correction(self, frame: np.ndarray) -> np.ndarray:
        """Performs stage 2 secondary color correction.

        This stage includes adjustments for skin tone isolation and selective
        desaturation.

        Args:
            frame: The input frame to be corrected.

        Returns:
            The corrected frame.
        """
        result = frame.copy()
        
        # Apply skin tone mask if enabled
        if self.skin_tone_mask.enabled:
            mask = self._create_skin_tone_mask(result)
            result = self._apply_masked_correction(result, mask, self.secondary_correction)
        
        # Apply selective saturation
        if self.selective_saturation.enabled:
            result = self._apply_selective_saturation(result)
        
        # Advanced tonal adjustments
        result = self._apply_tonal_adjustments(result, self.secondary_correction)
        
        return result
    
    def stage_3_creative_grade(self, frame: np.ndarray) -> np.ndarray:
        """Performs stage 3 creative color grading.

        This stage includes the application of Look-Up Tables (LUTs) and cinematic
        color curves.

        Args:
            frame: The input frame to be graded.

        Returns:
            The graded frame.
        """
        result = frame.copy()
        
        # Apply LUT if specified
        if self.creative_lut.data is not None or self.creative_lut.file_path:
            result = self._apply_lut(result, self.creative_lut)
        
        # Apply cinematic curves
        result = self._apply_cinematic_curves(result)
        
        # Color space conversion for creative grade
        result = self._apply_creative_color_grading(result)
        
        return result
    
    def stage_4_finishing(self, frame: np.ndarray) -> np.ndarray:
        """Performs stage 4 finishing effects.

        This stage includes the application of film grain, sharpening, vignettes,
        and chromatic aberration.

        Args:
            frame: The input frame to be finished.

        Returns:
            The finished frame.
        """
        result = frame.copy()
        
        # Film grain
        if self.finishing_effects["film_grain"].enabled:
            result = self._apply_film_grain(result, self.finishing_effects["film_grain"])
        
        # Sharpening
        if self.finishing_effects["sharpening"] > 0:
            result = self._apply_sharpening(result, self.finishing_effects["sharpening"])
        
        # Vignette
        if self.finishing_effects["vignette"].enabled:
            result = self._apply_vignette(result, self.finishing_effects["vignette"])
        
        # Chromatic aberration
        if self.finishing_effects["chromatic_aberration"].enabled:
            result = self._apply_chromatic_aberration(result, self.finishing_effects["chromatic_aberration"])
        
        # Final color space conversion
        result = self._final_color_space_conversion(result)
        
        return result
    
    def grade_frame(self, frame: np.ndarray) -> np.ndarray:
        """Applies the complete 4-stage grading pipeline to a frame.

        Args:
            frame: The input frame to be graded.

        Returns:
            The graded frame.
        """
        # Stage 1: Primary correction
        graded = self.stage_1_primary_correction(frame)
        
        # Stage 2: Secondary correction
        graded = self.stage_2_secondary_correction(graded)
        
        # Stage 3: Creative grade
        graded = self.stage_3_creative_grade(graded)
        
        # Stage 4: Finishing
        graded = self.stage_4_finishing(graded)
        
        return np.clip(graded, 0, 1)
    
    def _apply_white_balance(self, frame: np.ndarray, temperature: float, tint: float) -> np.ndarray:
        """Apply white balance adjustment"""
        # Convert RGB to XYZ color space
        xyz = self._rgb_to_xyz(frame)
        
        # Apply temperature adjustment
        temp_factor = self._calculate_temperature_factor(temperature)
        xyz[:, :, 0] *= temp_factor["r"]
        xyz[:, :, 1] *= temp_factor["g"]
        xyz[:, :, 2] *= temp_factor["b"]
        
        # Apply tint adjustment
        tint_factor = self._calculate_tint_factor(tint)
        xyz[:, :, 0] *= tint_factor["r"]
        xyz[:, :, 1] *= tint_factor["g"]
        xyz[:, :, 2] *= tint_factor["b"]
        
        # Convert back to RGB
        result = self._xyz_to_rgb(xyz)
        
        return result
    
    def _create_skin_tone_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create skin tone isolation mask"""
        # Convert to YCbCr color space
        ycbcr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        ycbcr = ycbcr.astype(np.float32) / 255.0
        
        # Create mask based on skin tone ranges
        y = ycbcr[:, :, 0]
        cb = ycbcr[:, :, 1]
        cr = ycbcr[:, :, 2]
        
        mask = (
            (y >= self.skin_tone_mask.y_min) & (y <= self.skin_tone_mask.y_max) &
            (cb >= self.skin_tone_mask.cb_min) & (cb <= self.skin_tone_mask.cb_max) &
            (cr >= self.skin_tone_mask.cr_min) & (cr <= self.skin_tone_mask.cr_max)
        ).astype(np.float32)
        
        # Apply softness
        if self.skin_tone_mask.softness > 0:
            mask = gaussian_filter(mask, sigma=self.skin_tone_mask.softness * 10)
        
        return mask
    
    def _apply_masked_correction(self, frame: np.ndarray, mask: np.ndarray, correction: ColorCorrection) -> np.ndarray:
        """Apply color correction to masked regions"""
        # Apply corrections to frame
        corrected = self._apply_color_correction(frame, correction)
        
        # Blend based on mask
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        result = frame * (1 - mask_3ch) + corrected * mask_3ch
        
        return result
    
    def _apply_selective_saturation(self, frame: np.ndarray) -> np.ndarray:
        """Apply selective desaturation based on color ranges"""
        hsv = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32) / 255.0
        
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # Create desaturation mask
        desat_mask = np.zeros_like(s)
        
        for target_hue_range in self.selective_saturation.target_colors:
            hue_min, hue_max = target_hue_range[0], target_hue_range[1]
            sat_min, sat_max = target_hue_range[2], target_hue_range[3]
            
            # Handle hue wraparound
            if hue_min > hue_max:
                hue_mask = (h >= hue_min) | (h <= hue_max)
            else:
                hue_mask = (h >= hue_min) & (h <= hue_max)
            
            sat_mask = (s >= sat_min) & (s <= sat_max)
            
            desat_mask = np.maximum(desat_mask, hue_mask & sat_mask)
        
        # Apply desaturation
        new_s = s * (1 - desat_mask * self.selective_saturation.desaturation_amount)
        
        # Recombine
        hsv[:, :, 1] = new_s
        result = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        result = result.astype(np.float32) / 255.0
        
        return result
    
    def _apply_tonal_adjustments(self, frame: np.ndarray, correction: ColorCorrection) -> np.ndarray:
        """Apply advanced tonal adjustments (shadows, midtones, highlights)"""
        # Convert to HSL for tonal control
        hsl = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2HLS)
        hsl = hsl.astype(np.float32) / 255.0
        
        l = hsl[:, :, 1]
        
        # Create masks for different tonal ranges
        shadows_mask = (l < 0.5).astype(np.float32)
        midtones_mask = ((l >= 0.3) & (l <= 0.7)).astype(np.float32)
        highlights_mask = (l > 0.5).astype(np.float32)
        
        # Apply adjustments
        adjusted_l = l.copy()
        if correction.shadows != 0:
            adjusted_l += shadows_mask * (correction.shadows / 100.0) * 0.5
        
        if correction.midtones != 0:
            adjusted_l += midtones_mask * (correction.midtones / 100.0) * 0.5
        
        if correction.highlights != 0:
            adjusted_l += highlights_mask * (correction.highlights / 100.0) * 0.5
        
        adjusted_l = np.clip(adjusted_l, 0, 1)
        hsl[:, :, 1] = adjusted_l
        
        # Convert back to RGB
        result = cv2.cvtColor((hsl * 255).astype(np.uint8), cv2.COLOR_HLS2RGB)
        result = result.astype(np.float32) / 255.0
        
        return result
    
    def _apply_lut(self, frame: np.ndarray, lut: LUT) -> np.ndarray:
        """Apply Look-Up Table"""
        if lut.data is None:
            # Load LUT from file if not loaded
            lut = self._load_lut_from_file(lut)
        
        if lut.data is not None:
            result = self._apply_3d_lut(frame, lut.data)
        else:
            result = frame
        
        return result
    
    def _apply_cinematic_curves(self, frame: np.ndarray) -> np.ndarray:
        """Apply cinematic color curves"""
        # Apply S-curve for cinematic look
        result = frame.copy()
        
        # Contrast curve
        if self.primary_correction.contrast > 0:
            # Enhance contrast with S-curve
            curve = np.array([
                0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
                0.22, 0.25, 0.28, 0.32, 0.36, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                0.70, 0.75, 0.80, 0.84, 0.88, 0.92, 0.96, 0.98, 1.0
            ])
            
            result = self._apply_lut_curve(result, curve)
        
        return result
    
    def _apply_film_grain(self, frame: np.ndarray, grain: FilmGrain) -> np.ndarray:
        """Apply film grain simulation"""
        np.random.seed(grain.seed)
        
        if grain.grain_type == "camera":
            # Digital camera grain
            noise = np.random.normal(0, grain.intensity * 0.1, frame.shape)
            result = frame + noise
            
        elif grain.grain_type == "film":
            # Film grain (more pronounced in shadows)
            noise = np.random.normal(0, grain.intensity * 0.15, frame.shape)
            # Bias grain towards darker areas
            luminance = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            luminance = luminance.astype(np.float32) / 255.0
            shadow_factor = 1 - luminance
            result = frame + noise * shadow_factor
            
        else:  # digital
            # Digital noise (subtle)
            noise = np.random.normal(0, grain.intensity * 0.05, frame.shape)
            result = frame + noise
        
        return np.clip(result, 0, 1)
    
    def _apply_sharpening(self, frame: np.ndarray, amount: float) -> np.ndarray:
        """Apply professional sharpening"""
        # Unsharp mask
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
        sharpened = frame + (frame - blurred) * amount
        
        return np.clip(sharpened, 0, 1)
    
    def _apply_vignette(self, frame: np.ndarray, vignette: Vignette) -> np.ndarray:
        """Apply vignette effect"""
        height, width = frame.shape[:2]
        
        # Create vignette mask
        center_x, center_y = width / 2, height / 2
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        y, x = np.ogrid[:height, :width]
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize distance
        normalized_distance = distance_from_center / max_distance
        
        # Apply vignette
        vignette_amount = vignette.amount * (normalized_distance / vignette.midpoint) ** (roundness / 2.0)
        vignette_mask = np.exp(-vignette_amount * 10)
        vignette_mask = np.clip(vignette_mask, 1 - vignette.amount, 1.0)
        
        # Apply feather
        if vignette.feather > 0:
            vignette_mask = gaussian_filter(vignette_mask, sigma=vignette.feather * 20)
        
        # Apply to frame
        result = frame * vignette_mask[:, :, np.newaxis]
        
        return result
    
    def _apply_chromatic_aberration(self, frame: np.ndarray, aberration: ChromaticAberration) -> np.ndarray:
        """Apply chromatic aberration simulation"""
        height, width = frame.shape[:2]
        
        # Create channel offsets
        if aberration.channel_bias == "blue":
            # Blue shift to the right and down
            offset_x = int(aberration.amount * 5)
            offset_y = int(aberration.amount * 3)
        elif aberration.channel_bias == "red":
            # Red shift to the left and up
            offset_x = -int(aberration.amount * 5)
            offset_y = -int(aberration.amount * 3)
        else:  # green
            # Green shift (minimal)
            offset_x = 0
            offset_y = 0
        
        # Create shifted versions of each channel
        frame_shift = np.zeros_like(frame)
        
        for channel in range(3):
            shifted_channel = np.zeros_like(frame[:, :, channel])
            
            if offset_x != 0 or offset_y != 0:
                M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                shifted_channel = cv2.warpAffine(frame[:, :, channel], M, (width, height))
            
            frame_shift[:, :, channel] = shifted_channel
        
        # Blend with original
        result = frame * (1 - aberration.amount) + frame_shift * aberration.amount
        
        return result
    
    def _apply_color_correction(self, frame: np.ndarray, correction: ColorCorrection) -> np.ndarray:
        """Apply general color correction"""
        result = frame.copy()
        
        # Saturation
        if correction.saturation != 0:
            hsv = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32) / 255.0
            hsv[:, :, 1] *= (1 + correction.saturation / 100.0)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
            result = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
            result = result.astype(np.float32) / 255.0
        
        # Hue shift
        if correction.hue != 0:
            hsv = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32) / 255.0
            hsv[:, :, 0] = (hsv[:, :, 0] + correction.hue / 360.0) % 1.0
            result = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
            result = result.astype(np.float32) / 255.0
        
        return result
    
    def _setup_color_conversion(self):
        """Setup color space conversion matrices"""
        # Rec.709 RGB to XYZ matrix
        self._conversion_matrices["rec709_to_xyz"] = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        # XYZ to Rec.709 RGB matrix
        self._conversion_matrices["xyz_to_rec709"] = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ])
    
    def _rgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to XYZ color space"""
        return np.dot(rgb, self._conversion_matrices["rec709_to_xyz"].T)
    
    def _xyz_to_rgb(self, xyz: np.ndarray) -> np.ndarray:
        """Convert XYZ to RGB color space"""
        return np.dot(xyz, self._conversion_matrices["xyz_to_rec709"].T)
    
    def _calculate_temperature_factor(self, temperature: float) -> Dict[str, float]:
        """Calculate white balance factors for temperature"""
        # Simplified temperature calculation
        normalized_temp = temperature / 5600.0
        
        if normalized_temp > 1.0:
            # Cooler (higher temperature)
            r_factor = 1.0 + (normalized_temp - 1.0) * 0.5
            b_factor = 1.0 - (normalized_temp - 1.0) * 0.3
        else:
            # Warmer (lower temperature)
            r_factor = 1.0 - (1.0 - normalized_temp) * 0.2
            b_factor = 1.0 + (1.0 - normalized_temp) * 0.4
        
        return {"r": r_factor, "g": 1.0, "b": b_factor}
    
    def _calculate_tint_factor(self, tint: float) -> Dict[str, float]:
        """Calculate white balance factors for tint"""
        # Simplified tint calculation
        normalized_tint = tint / 100.0
        
        # Adjust green-magenta balance
        if normalized_tint > 0:
            # Towards magenta
            r_factor = 1.0 + normalized_tint * 0.1
            b_factor = 1.0 + normalized_tint * 0.1
            g_factor = 1.0 - normalized_tint * 0.05
        else:
            # Towards green
            r_factor = 1.0 + normalized_tint * 0.05
            b_factor = 1.0 + normalized_tint * 0.05
            g_factor = 1.0 - normalized_tint * 0.1
        
        return {"r": r_factor, "g": g_factor, "b": b_factor}
    
    def _apply_creative_color_grading(self, frame: np.ndarray) -> np.ndarray:
        """Apply creative color grading transformations"""
        # This would implement more sophisticated creative grading
        # like teal & orange, dark & moody, etc.
        
        # Example: Teal & Orange look
        hsv = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32) / 255.0
        
        # Enhance orange tones in highlights
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # Boost saturation in orange range (0.05-0.15 in HSV)
        orange_mask = ((h >= 0.05) & (h <= 0.15) & (v > 0.7))
        s[orange_mask] *= 1.2
        
        # Add teal tones in shadows (0.5-0.6 in HSV)
        teal_mask = ((h >= 0.5) & (h <= 0.6) & (v < 0.5))
        h[teal_mask] = 0.55  # Teal hue
        
        hsv[:, :, 1] = np.clip(s, 0, 1)
        hsv[:, :, 0] = h
        
        result = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        result = result.astype(np.float32) / 255.0
        
        return result
    
    def _final_color_space_conversion(self, frame: np.ndarray) -> np.ndarray:
        """Final color space conversion for output"""
        if self.color_space == ColorSpace.REC_709:
            # Rec.709 is the standard output format
            return frame
        elif self.color_space == ColorSpace.REC_2020:
            # Convert to Rec.2020
            # This would require a more complex conversion
            return frame  # Placeholder
        else:
            return frame
    
    def _apply_lut_curve(self, frame: np.ndarray, curve: np.ndarray) -> np.ndarray:
        """Apply 1D LUT curve to frame"""
        result = frame.copy()
        
        # Apply curve to each channel
        for channel in range(3):
            result[:, :, channel] = curve[np.minimum((result[:, :, channel] * 255).astype(np.int32), 255)]
        
        result = result / 255.0
        return result
    
    def _load_lut_from_file(self, lut: LUT) -> LUT:
        """Load LUT from file"""
        if lut.file_path and lut.file_path.endswith('.cube'):
            try:
                # Parse .cube file format
                lut_data = self._parse_cube_file(lut.file_path)
                lut.data = lut_data
            except Exception as e:
                print(f"Warning: Could not load LUT from {lut.file_path}: {e}")
        
        return lut
    
    def _parse_cube_file(self, file_path: str) -> np.ndarray:
        """Parse .cube LUT file format"""
        # Simplified .cube parser
        # This would need a full implementation for production use
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        size = 33  # Default size
        for line in lines:
            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[1])
                break
        
        # Parse LUT data
        lut_data = []
        for line in lines:
            if len(line.strip().split()) == 3:
                try:
                    r, g, b = map(float, line.strip().split())
                    lut_data.append([r, g, b])
                except ValueError:
                    continue
        
        lut_array = np.array(lut_data).reshape(size, size, size, 3)
        return lut_array
    
    def _apply_3d_lut(self, frame: np.ndarray, lut_data: np.ndarray) -> np.ndarray:
        """Apply 3D LUT to frame"""
        # Simplified 3D LUT application
        # For production, this would need interpolation
        
        # Quantize frame values to LUT indices
        height, width = frame.shape[:2]
        size = lut_data.shape[0]
        
        # Normalize to LUT range [0, size-1]
        frame_indices = np.minimum((frame * (size - 1)).astype(np.int32), size - 1)
        
        result = np.zeros_like(frame)
        for i in range(height):
            for j in range(width):
                r_idx, g_idx, b_idx = frame_indices[i, j]
                result[i, j] = lut_data[r_idx, g_idx, b_idx]
        
        return result


# Utility functions for color grading
def validate_color_grading(frame: np.ndarray) -> Dict[str, Union[bool, List[str]]]:
    """Validates the output of the color grading process.

    Args:
        frame: The graded frame to be validated.

    Returns:
        A dictionary containing the validation results.
    """
    errors = []
    warnings = []
    
    # Check for clipping
    if np.any(frame < 0):
        errors.append("Negative color values detected (underexposure)")
    
    if np.any(frame > 1):
        errors.append("Color values > 1.0 detected (overexposure)")
    
    # Check dynamic range
    if frame.max() - frame.min() < 0.1:
        warnings.append("Low dynamic range detected")
    
    # Check saturation levels
    hsv = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32) / 255.0
    if np.mean(hsv[:, :, 1]) < 0.1:
        warnings.append("Low average saturation detected")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def auto_color_balance(frame: np.ndarray) -> ColorCorrection:
    """Performs automatic white balance and color correction.

    Args:
        frame: The input frame to be balanced.

    Returns:
        A ColorCorrection object with the calculated adjustments.
    """
    # Calculate gray world assumption
    mean_r = np.mean(frame[:, :, 0])
    mean_g = np.mean(frame[:, :, 1])
    mean_b = np.mean(frame[:, :, 2])
    
    # Normalize to mid-gray (0.5)
    gray_target = 0.5
    
    correction = ColorCorrection()
    correction.exposure = 0.0  # Will be calculated separately
    
    # White balance adjustments
    if mean_r > 0 and mean_g > 0 and mean_b > 0:
        correction.temperature = 5600.0 * (mean_b / mean_r)  # Simplified
        correction.tint = ((mean_g / ((mean_r + mean_b) / 2)) - 1.0) * 100
    
    return correction


def analyze_histogram(frame: np.ndarray) -> Dict[str, float]:
    """Analyzes the color histogram of a frame.

    Args:
        frame: The input frame to be analyzed.

    Returns:
        A dictionary of histogram analysis results.
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32) / 255.0
    
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    analysis = {
        "mean_hue": float(np.mean(h)),
        "mean_saturation": float(np.mean(s)),
        "mean_brightness": float(np.mean(v)),
        "hue_std": float(np.std(h)),
        "saturation_std": float(np.std(s)),
        "brightness_std": float(np.std(v)),
        "shadows_percentage": float(np.sum(v < 0.3) / v.size),
        "midtones_percentage": float(np.sum((v >= 0.3) & (v < 0.7)) / v.size),
        "highlights_percentage": float(np.sum(v >= 0.7) / v.size)
    }
    
    return analysis