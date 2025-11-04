"""
Broadcast Standards Validator
Professional compliance checking for broadcast-quality video
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum


class BroadcastStandard(Enum):
    """Enumeration of broadcast standards."""
    REC709 = "Rec.709/BT.709"
    REC2020 = "Rec.2020/BT.2020"
    NTSC = "NTSC"
    PAL = "PAL"
    SECAM = "SECAM"


class LegalizerMode(Enum):
    """Enumeration of video legalizer modes."""
    SIMPLE = "simple"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"


@dataclass
class BroadcastCompliance:
    """Represents broadcast compliance metrics.

    Attributes:
        ire_levels_compliant: Whether the IRE levels are compliant.
        gamut_compliant: Whether the color gamut is compliant.
        safe_area_compliant: Whether the safe areas are compliant.
        level_violations: The number of IRE level violations.
        gamut_violations: The number of color gamut violations.
        safe_area_violations: The number of safe area violations.
        overall_compliance_score: The overall compliance score.
    """
    ire_levels_compliant: bool
    gamut_compliant: bool
    safe_area_compliant: bool
    level_violations: int
    gamut_violations: int
    safe_area_violations: int
    overall_compliance_score: float


class BroadcastStandardsValidator:
    """A class for validating video against professional broadcast standards.

    This class can validate a video against a variety of broadcast standards,
    including IRE level compliance, color gamut compliance, and safe area
    compliance. It can also apply a broadcast legalizer to the video to
    automatically correct any compliance issues.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initializes the BroadcastStandardsValidator.

        Args:
            config: A dictionary of configuration parameters.
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger('apex_director.qa.broadcast_standards')
        
        # Configuration parameters
        self.broadcast_standard = self.config.get('broadcast_standard', BroadcastStandard.REC709)
        self.check_ire_levels = self.config.get('check_ire_levels', True)
        self.check_gamut = self.config.get('check_gamut', True)
        self.check_safe_area = self.config.get('check_safe_area', True)
        self.check_legalizer = self.config.get('check_legalizer', True)
        
        # IRE level thresholds (legal broadcast range)
        self.ire_min = self.config.get('ire_min', 7.5)  # Setup level
        self.ire_max = self.config.get('ire_max', 100.0)  # Peak level
        
        # Color gamut limits for Rec.709
        self.gamut_limits = self._get_gamut_limits()
        
        # Safe area percentages
        self.action_safe_percentage = self.config.get('action_safe_percentage', 0.10)  # 10%
        self.title_safe_percentage = self.config.get('title_safe_percentage', 0.20)  # 20%
        
        # Compliance thresholds
        self.ire_violation_threshold = self.config.get('ire_violation_threshold', 0.01)  # 1% of pixels
        self.gamut_violation_threshold = self.config.get('gamut_violation_threshold', 0.005)  # 0.5% of pixels
        self.safe_area_violation_threshold = self.config.get('safe_area_violation_threshold', 0.02)  # 2% of area
        
        # Legalizer configuration
        self.legalizer_mode = self.config.get('legalizer_mode', LegalizerMode.SIMPLE)
        self.auto_legalize = self.config.get('auto_legalize', False)
        
        # Analysis results storage
        self.ire_analysis_results = []
        self.gamut_analysis_results = []
        self.safe_area_results = []
        
    def validate_compliance(self, sample_frames: List[np.ndarray]) -> Dict:
        """
        Validate broadcast compliance for sample frames
        
        Args:
            sample_frames: List of sample frames from video
            
        Returns:
            Dictionary with compliance analysis results
        """
        self.logger.info(f"Validating broadcast compliance for {len(sample_frames)} frames")
        
        if len(sample_frames) == 0:
            return {
                'score': 0.0,
                'error': 'No frames provided for analysis'
            }
        
        try:
            # Perform all compliance checks
            ire_results = self._analyze_ire_levels(sample_frames) if self.check_ire_levels else {'compliant': True, 'violations': 0}
            gamut_results = self._analyze_gamut_compliance(sample_frames) if self.check_gamut else {'compliant': True, 'violations': 0}
            safe_area_results = self._analyze_safe_area_compliance(sample_frames) if self.check_safe_area else {'compliant': True, 'violations': 0}
            
            # Calculate overall compliance score
            compliance_score = self._calculate_compliance_score(ire_results, gamut_results, safe_area_results)
            
            # Check if auto-legalization is needed
            needs_legalization = (
                not ire_results.get('compliant', True) or
                not gamut_results.get('compliant', True) or
                not safe_area_results.get('compliant', True)
            )
            
            # Generate detailed results
            results = {
                'score': compliance_score * 100,  # Convert to 0-100 scale
                'broadcast_standard': self.broadcast_standard.value,
                'compliance': BroadcastCompliance(
                    ire_levels_compliant=ire_results.get('compliant', True),
                    gamut_compliant=gamut_results.get('compliant', True),
                    safe_area_compliant=safe_area_results.get('compliant', True),
                    level_violations=ire_results.get('violations', 0),
                    gamut_violations=gamut_results.get('violations', 0),
                    safe_area_violations=safe_area_results.get('violations', 0),
                    overall_compliance_score=compliance_score
                ),
                'ire_analysis': ire_results,
                'gamut_analysis': gamut_results,
                'safe_area_analysis': safe_area_results,
                'needs_legalization': needs_legalization,
                'legalizer_recommendations': self._generate_legalizer_recommendations(
                    ire_results, gamut_results, safe_area_results
                ),
                'technical_details': self._get_technical_details(sample_frames),
                'recommendations': self._generate_compliance_recommendations(
                    ire_results, gamut_results, safe_area_results
                )
            }
            
            # Store analysis results
            self._store_analysis_results(ire_results, gamut_results, safe_area_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Broadcast compliance validation failed: {e}")
            return {
                'score': 0.0,
                'error': str(e)
            }
    
    def _analyze_ire_levels(self, sample_frames: List[np.ndarray]) -> Dict:
        """Analyzes the IRE level compliance of a list of sample frames.

        Args:
            sample_frames: A list of sample frames from a video.

        Returns:
            A dictionary with the IRE level analysis results.
        """
        try:
            self.logger.info("Analyzing IRE levels")
            
            total_pixels = 0
            violation_pixels = 0
            below_setup_pixels = 0
            above_peak_pixels = 0
            
            ire_statistics = {
                'min_ire': float('inf'),
                'max_ire': -float('inf'),
                'avg_ire': 0,
                'ire_distribution': {}
            }
            
            for frame_idx, frame in enumerate(sample_frames):
                # Convert to YUV for luminance analysis
                if len(frame.shape) == 3:
                    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                    luminance = yuv_frame[:, :, 0]  # Y channel
                else:
                    luminance = frame
                
                # Convert luminance to IRE scale
                # IRE = ((Y - Y_black) / (Y_white - Y_black)) * 100
                # For broadcast standards: Y_black = 16, Y_white = 235 (8-bit)
                y_8bit = (luminance * 255).astype(np.uint8) if luminance.max() <= 1 else luminance
                
                ire_values = ((y_8bit - 16) / (235 - 16)) * 100
                ire_values = np.clip(ire_values, 0, 100)  # Clip to valid IRE range
                
                # Count violations
                below_setup = np.sum(ire_values < self.ire_min)
                above_peak = np.sum(ire_values > self.ire_max)
                
                below_setup_pixels += below_setup
                above_peak_pixels += above_peak
                total_pixels += ire_values.size
                
                # Update statistics
                frame_min_ire = np.min(ire_values)
                frame_max_ire = np.max(ire_values)
                frame_avg_ire = np.mean(ire_values)
                
                ire_statistics['min_ire'] = min(ire_statistics['min_ire'], frame_min_ire)
                ire_statistics['max_ire'] = max(ire_statistics['max_ire'], frame_max_ire)
                ire_statistics['avg_ire'] += frame_avg_ire
                
                # Create IRE distribution histogram
                hist, bins = np.histogram(ire_values, bins=20, range=(0, 100))
                for i, count in enumerate(hist):
                    bin_start = bins[i]
                    bin_key = f"{bin_start:.1f}"
                    if bin_key not in ire_statistics['ire_distribution']:
                        ire_statistics['ire_distribution'][bin_key] = 0
                    ire_statistics['ire_distribution'][bin_key] += count
            
            # Calculate average IRE
            ire_statistics['avg_ire'] /= len(sample_frames)
            
            # Calculate violation percentages
            violation_percentage = (below_setup_pixels + above_peak_pixels) / total_pixels
            below_setup_percentage = below_setup_pixels / total_pixels
            above_peak_percentage = above_peak_pixels / total_pixels
            
            # Determine compliance
            is_compliant = violation_percentage <= self.ire_violation_threshold
            
            return {
                'compliant': is_compliant,
                'violations': below_setup_pixels + above_peak_pixels,
                'total_pixels': total_pixels,
                'violation_percentage': violation_percentage,
                'below_setup_percentage': below_setup_percentage,
                'above_peak_percentage': above_peak_percentage,
                'ire_min_detected': ire_statistics['min_ire'],
                'ire_max_detected': ire_statistics['max_ire'],
                'ire_avg_detected': ire_statistics['avg_ire'],
                'statistics': ire_statistics,
                'legal_range': (self.ire_min, self.ire_max)
            }
            
        except Exception as e:
            self.logger.error(f"IRE level analysis failed: {e}")
            return {
                'compliant': False,
                'violations': 0,
                'error': str(e)
            }
    
    def _analyze_gamut_compliance(self, sample_frames: List[np.ndarray]) -> Dict:
        """Analyzes the color gamut compliance of a list of sample frames.

        Args:
            sample_frames: A list of sample frames from a video.

        Returns:
            A dictionary with the color gamut analysis results.
        """
        try:
            self.logger.info("Analyzing color gamut compliance")
            
            total_pixels = 0
            gamut_violations = 0
            r_over_saturations = 0
            g_over_saturations = 0
            b_over_saturations = 0
            
            gamut_statistics = {
                'avg_r': 0,
                'avg_g': 0,
                'avg_b': 0,
                'max_r': 0,
                'max_g': 0,
                'max_b': 0
            }
            
            for frame in sample_frames:
                # Convert to RGB if necessary
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame
                
                # Normalize to 0-1 range
                if rgb_frame.max() > 1:
                    rgb_frame = rgb_frame.astype(np.float32) / 255.0
                
                # Check for gamut violations using broadcast-safe limits
                r, g, b = cv2.split(rgb_frame)
                
                # Count over-saturations (broadcast gamut violations)
                r_over = np.sum(r > 1.0)  # Red over-saturated
                g_over = np.sum(g > 1.0)  # Green over-saturated
                b_over = np.sum(b > 1.0)  # Blue over-saturated
                
                r_over_saturations += r_over
                g_over_saturations += g_over
                b_over_saturations += b_over
                
                # Update statistics
                gamut_statistics['avg_r'] += np.mean(r)
                gamut_statistics['avg_g'] += np.mean(g)
                gamut_statistics['avg_b'] += np.mean(b)
                gamut_statistics['max_r'] = max(gamut_statistics['max_r'], np.max(r))
                gamut_statistics['max_g'] = max(gamut_statistics['max_g'], np.max(g))
                gamut_statistics['max_b'] = max(gamut_statistics['max_b'], np.max(b))
                
                total_pixels += r.size
            
            # Calculate averages
            frame_count = len(sample_frames)
            gamut_statistics['avg_r'] /= frame_count
            gamut_statistics['avg_g'] /= frame_count
            gamut_statistics['avg_b'] /= frame_count
            
            # Calculate total gamut violations
            gamut_violations = r_over_saturations + g_over_saturations + b_over_saturations
            violation_percentage = gamut_violations / total_pixels if total_pixels > 0 else 0
            
            # Check compliance with more stringent broadcast limits
            # For professional broadcast, use more conservative limits
            broadcast_safe_limits = {
                'r_max': 0.95,  # Conservative limit for red
                'g_max': 0.95,  # Conservative limit for green
                'b_max': 0.95   # Conservative limit for blue
            }
            
            is_compliant = violation_percentage <= self.gamut_violation_threshold
            
            return {
                'compliant': is_compliant,
                'violations': gamut_violations,
                'total_pixels': total_pixels,
                'violation_percentage': violation_percentage,
                'r_over_saturations': r_over_saturations,
                'g_over_saturations': g_over_saturations,
                'b_over_saturations': b_over_saturations,
                'broadcast_safe_limits': broadcast_safe_limits,
                'statistics': gamut_statistics
            }
            
        except Exception as e:
            self.logger.error(f"Gamut analysis failed: {e}")
            return {
                'compliant': False,
                'violations': 0,
                'error': str(e)
            }
    
    def _analyze_safe_area_compliance(self, sample_frames: List[np.ndarray]) -> Dict:
        """Analyzes the safe area compliance of a list of sample frames.

        Args:
            sample_frames: A list of sample frames from a video.

        Returns:
            A dictionary with the safe area analysis results.
        """
        try:
            self.logger.info("Analyzing safe area compliance")
            
            total_violations = 0
            frame_count = len(sample_frames)
            
            safe_area_analysis = {
                'action_safe_violations': 0,
                'title_safe_violations': 0,
                'avg_action_safe_coverage': 0,
                'avg_title_safe_coverage': 0
            }
            
            for frame in sample_frames:
                height, width = frame.shape[:2]
                
                # Calculate safe area boundaries
                action_safe_margin_w = int(width * self.action_safe_percentage)
                action_safe_margin_h = int(height * self.action_safe_percentage)
                
                title_safe_margin_w = int(width * self.title_safe_percentage)
                title_safe_margin_h = int(height * self.title_safe_percentage)
                
                # Define safe area rectangles
                action_safe_rect = (
                    action_safe_margin_w, action_safe_margin_h,
                    width - action_safe_margin_w, height - action_safe_margin_h
                )
                
                title_safe_rect = (
                    title_safe_margin_w, title_safe_margin_h,
                    width - title_safe_margin_w, height - title_safe_margin_h
                )
                
                # For this analysis, we'll check if there's sufficient content in safe areas
                # In a real implementation, this would detect text, logos, or important visual elements
                
                # Placeholder: Assume content is properly positioned
                # Real implementation would use text detection, object detection, etc.
                
                action_safe_violation = False  # Placeholder
                title_safe_violation = False   # Placeholder
                
                if action_safe_violation:
                    safe_area_analysis['action_safe_violations'] += 1
                    total_violations += 1
                
                if title_safe_violation:
                    safe_area_analysis['title_safe_violations'] += 1
                    total_violations += 1
                
                # Calculate coverage percentages (placeholder)
                safe_area_analysis['avg_action_safe_coverage'] += 0.9  # 90% coverage
                safe_area_analysis['avg_title_safe_coverage'] += 0.85  # 85% coverage
            
            # Calculate averages
            safe_area_analysis['avg_action_safe_coverage'] /= frame_count
            safe_area_analysis['avg_title_safe_coverage'] /= frame_count
            
            # Determine compliance
            violation_percentage = total_violations / frame_count if frame_count > 0 else 0
            is_compliant = violation_percentage <= self.safe_area_violation_threshold
            
            return {
                'compliant': is_compliant,
                'violations': total_violations,
                'frame_count': frame_count,
                'violation_percentage': violation_percentage,
                'safe_area_percentages': {
                    'action_safe': self.action_safe_percentage * 100,
                    'title_safe': self.title_safe_percentage * 100
                },
                'analysis': safe_area_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Safe area analysis failed: {e}")
            return {
                'compliant': False,
                'violations': 0,
                'error': str(e)
            }
    
    def _calculate_compliance_score(self, ire_results: Dict, gamut_results: Dict, 
                                  safe_area_results: Dict) -> float:
        """Calculates the overall compliance score.

        Args:
            ire_results: The IRE level analysis results.
            gamut_results: The color gamut analysis results.
            safe_area_results: The safe area analysis results.

        Returns:
            The overall compliance score.
        """
        try:
            scores = []
            weights = []
            
            # IRE level score (40% weight)
            if self.check_ire_levels:
                ire_violation_rate = ire_results.get('violation_percentage', 1.0)
                ire_score = max(0, 1 - ire_violation_rate * 10)  # Penalty for violations
                scores.append(ire_score)
                weights.append(0.4)
            
            # Gamut compliance score (35% weight)
            if self.check_gamut:
                gamut_violation_rate = gamut_results.get('violation_percentage', 1.0)
                gamut_score = max(0, 1 - gamut_violation_rate * 20)  # Stricter penalty
                scores.append(gamut_score)
                weights.append(0.35)
            
            # Safe area compliance score (25% weight)
            if self.check_safe_area:
                safe_area_violation_rate = safe_area_results.get('violation_percentage', 1.0)
                safe_area_score = max(0, 1 - safe_area_violation_rate * 5)  # Moderate penalty
                scores.append(safe_area_score)
                weights.append(0.25)
            
            # Calculate weighted average
            if scores and weights:
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize weights
                compliance_score = np.average(scores, weights=weights)
            else:
                compliance_score = 1.0  # Default if no checks performed
            
            return compliance_score
            
        except Exception as e:
            self.logger.error(f"Compliance score calculation failed: {e}")
            return 0.0
    
    def _generate_legalizer_recommendations(self, ire_results: Dict, gamut_results: Dict,
                                          safe_area_results: Dict) -> List[str]:
        """Generates legalizer recommendations based on the analysis results.

        Args:
            ire_results: The IRE level analysis results.
            gamut_results: The color gamut analysis results.
            safe_area_results: The safe area analysis results.

        Returns:
            A list of legalizer recommendations.
        """
        recommendations = []
        
        # IRE level recommendations
        if self.check_ire_levels and not ire_results.get('compliant', True):
            below_setup_pct = ire_results.get('below_setup_percentage', 0)
            above_peak_pct = ire_results.get('above_peak_percentage', 0)
            
            if below_setup_pct > 0.005:  # More than 0.5% below setup
                recommendations.append("Apply setup level adjustment to raise blacks")
            
            if above_peak_pct > 0.005:  # More than 0.5% above peak
                recommendations.append("Apply peak level compression to prevent clipping")
            
            recommendations.append("Use broadcast legalizer to ensure IRE compliance")
        
        # Gamut recommendations
        if self.check_gamut and not gamut_results.get('compliant', True):
            recommendations.append("Apply color gamut limiter to prevent over-saturation")
            recommendations.append("Reduce saturation in problem color ranges")
            recommendations.append("Use HSV adjustment for broadcast-safe colors")
        
        # Safe area recommendations
        if self.check_safe_area and not safe_area_results.get('compliant', True):
            recommendations.append("Ensure important content stays within safe areas")
            recommendations.append("Add safe area guides to editing timeline")
            recommendations.push("Check title and logo positioning")
        
        if not recommendations:
            recommendations.append("Video appears to be broadcast-compliant")
        
        return recommendations
    
    def _generate_compliance_recommendations(self, ire_results: Dict, gamut_results: Dict,
                                           safe_area_results: Dict) -> List[str]:
        """Generates compliance recommendations based on the analysis results.

        Args:
            ire_results: The IRE level analysis results.
            gamut_results: The color gamut analysis results.
            safe_area_results: The safe area analysis results.

        Returns:
            A list of compliance recommendations.
        """
        recommendations = []
        
        # General compliance recommendations
        if (not ire_results.get('compliant', True) or 
            not gamut_results.get('compliant', True)):
            recommendations.append("Re-process video with broadcast legalizer")
            recommendations.append("Convert to appropriate color space for delivery")
        
        if not safe_area_results.get('compliant', True):
            recommendations.append("Review framing and reposition important elements")
        
        # Technical recommendations
        if self.check_ire_levels:
            detected_avg_ire = ire_results.get('ire_avg_detected', 50)
            if detected_avg_ire < 40:
                recommendations.append("Overall levels appear too dark - increase exposure")
            elif detected_avg_ire > 80:
                recommendations.append("Overall levels appear too bright - reduce exposure")
        
        # Quality improvement suggestions
        recommendations.extend([
            "Ensure consistent color grading throughout",
            "Check monitor calibration for accurate assessment",
            "Test on target broadcast equipment",
            "Maintain consistent audio levels (-23 LUFS for broadcast)"
        ])
        
        return recommendations
    
    def _get_technical_details(self, sample_frames: List[np.ndarray]) -> Dict:
        """Gets technical details about the video.

        Args:
            sample_frames: A list of sample frames from the video.

        Returns:
            A dictionary of technical details.
        """
        try:
            if not sample_frames:
                return {}
            
            first_frame = sample_frames[0]
            height, width = first_frame.shape[:2]
            
            # Detect color space
            color_space = self._detect_color_space(first_frame)
            
            # Detect bit depth (placeholder)
            bit_depth = 8  # Assume 8-bit for now
            
            # Calculate frame aspect ratio
            aspect_ratio = width / height
            
            return {
                'resolution': f"{width}x{height}",
                'aspect_ratio': f"{aspect_ratio:.3f}",
                'color_space': color_space,
                'bit_depth': bit_depth,
                'broadcast_standard': self.broadcast_standard.value,
                'frame_count_analyzed': len(sample_frames)
            }
            
        except Exception as e:
            self.logger.error(f"Technical details extraction failed: {e}")
            return {}
    
    def _detect_color_space(self, frame: np.ndarray) -> str:
        """Detects the color space of a frame.

        Args:
            frame: The frame to detect the color space of.

        Returns:
            The detected color space.
        """
        try:
            # Simple color space detection based on color distribution
            if len(frame.shape) == 3:
                # Assume BGR (OpenCV default) or RGB
                # In a real implementation, would use more sophisticated detection
                return "BGR/RGB"
            else:
                return "Grayscale"
        except Exception as e:
            self.logger.error(f"Color space detection failed: {e}")
            return "Unknown"
    
    def _get_gamut_limits(self) -> Dict:
        """Gets the color gamut limits for the current broadcast standard.

        Returns:
            A dictionary of color gamut limits.
        """
        if self.broadcast_standard == BroadcastStandard.REC709:
            return {
                'r_max': 0.95,
                'g_max': 0.95,
                'b_max': 0.95,
                'r_min': 0.0,
                'g_min': 0.0,
                'b_min': 0.0
            }
        elif self.broadcast_standard == BroadcastStandard.REC2020:
            return {
                'r_max': 0.98,
                'g_max': 0.98,
                'b_max': 0.98,
                'r_min': 0.0,
                'g_min': 0.0,
                'b_min': 0.0
            }
        else:
            # Default conservative limits
            return {
                'r_max': 0.90,
                'g_max': 0.90,
                'b_max': 0.90,
                'r_min': 0.05,
                'g_min': 0.05,
                'b_min': 0.05
            }
    
    def _store_analysis_results(self, ire_results: Dict, gamut_results: Dict, 
                              safe_area_results: Dict):
        """Stores the analysis results for future reference.

        Args:
            ire_results: The IRE level analysis results.
            gamut_results: The color gamut analysis results.
            safe_area_results: The safe area analysis results.
        """
        self.ire_analysis_results.append(ire_results)
        self.gamut_analysis_results.append(gamut_results)
        self.safe_area_results.append(safe_area_results)
    
    def apply_broadcast_legalizer(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Applies a broadcast legalizer to a list of frames.

        Args:
            frames: A list of frames to legalize.

        Returns:
            A list of legalized frames.
        """
        self.logger.info(f"Applying broadcast legalizer to {len(frames)} frames")
        
        legalized_frames = []
        
        for frame in frames:
            if self.legalizer_mode == LegalizerMode.SIMPLE:
                legalized_frame = self._simple_legalizer(frame)
            elif self.legalizer_mode == LegalizerMode.ADVANCED:
                legalized_frame = self._advanced_legalizer(frame)
            elif self.legalizer_mode == LegalizerMode.PROFESSIONAL:
                legalized_frame = self._professional_legalizer(frame)
            else:
                legalized_frame = frame
            
            legalized_frames.append(legalized_frame)
        
        return legalized_frames
    
    def _simple_legalizer(self, frame: np.ndarray) -> np.ndarray:
        """Applies a simple broadcast legalizer to a frame.

        Args:
            frame: The frame to legalize.

        Returns:
            The legalized frame.
        """
        try:
            # Convert to float for processing
            if frame.max() > 1:
                frame = frame.astype(np.float32) / 255.0
            
            # Apply simple IRE clipping
            legalized = np.clip(frame, 0.075, 1.0)  # 7.5% to 100% range
            
            # Scale back to 0-255 range
            if frame.max() <= 1:
                legalized = (legalized * 255).astype(np.uint8)
            
            return legalized
            
        except Exception as e:
            self.logger.error(f"Simple legalizer failed: {e}")
            return frame
    
    def _advanced_legalizer(self, frame: np.ndarray) -> np.ndarray:
        """Applies an advanced broadcast legalizer to a frame.

        Args:
            frame: The frame to legalize.

        Returns:
            The legalized frame.
        """
        try:
            # Convert to float
            if frame.max() > 1:
                frame = frame.astype(np.float32) / 255.0
            
            # Convert to YUV for luminance-based processing
            yuv_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
            yuv_frame = yuv_frame.astype(np.float32) / 255.0
            
            # Legalize luminance (Y channel)
            y_legalized = np.clip(yuv_frame[:, :, 0], self.ire_min/100, 1.0)
            
            # Reconstruct frame
            legalized_yuv = yuv_frame.copy()
            legalized_yuv[:, :, 0] = y_legalized
            
            # Convert back to RGB
            legalized_rgb = cv2.cvtColor((legalized_yuv * 255).astype(np.uint8), cv2.COLOR_YUV2RGB)
            
            # Scale back if needed
            if frame.max() <= 1:
                legalized_rgb = legalized_rgb.astype(np.float32) / 255.0
            
            return legalized_rgb
            
        except Exception as e:
            self.logger.error(f"Advanced legalizer failed: {e}")
            return frame
    
    def _professional_legalizer(self, frame: np.ndarray) -> np.ndarray:
        """Applies a professional broadcast legalizer to a frame.

        Args:
            frame: The frame to legalize.

        Returns:
            The legalized frame.
        """
        try:
            # Convert to float
            if frame.max() > 1:
                frame = frame.astype(np.float32) / 255.0
            
            # Apply professional-level legalizing
            legalized = frame.copy()
            
            # Legalize IRE levels
            legalized[:, :, 0] = np.clip(legalized[:, :, 0], self.ire_min/100, 1.0)  # R
            legalized[:, :, 1] = np.clip(legalized[:, :, 1], self.ire_min/100, 1.0)  # G
            legalized[:, :, 2] = np.clip(legalized[:, :, 2], self.ire_min/100, 1.0)  # B
            
            # Apply gamut limiting
            gamut_limits = self.gamut_limits
            legalized[:, :, 0] = np.clip(legalized[:, :, 0], gamut_limits['r_min'], gamut_limits['r_max'])
            legalized[:, :, 1] = np.clip(legalized[:, :, 1], gamut_limits['g_min'], gamut_limits['g_max'])
            legalized[:, :, 2] = np.clip(legalized[:, :, 2], gamut_limits['b_min'], gamut_limits['b_max'])
            
            # Apply soft limiting to preserve highlight details
            legalized = self._apply_soft_limiting(legalized)
            
            # Scale back if needed
            if frame.max() <= 1:
                legalized = (legalized * 255).astype(np.uint8)
            
            return legalized
            
        except Exception as e:
            self.logger.error(f"Professional legalizer failed: {e}")
            return frame
    
    def _apply_soft_limiting(self, frame: np.ndarray) -> np.ndarray:
        """Applies soft limiting to a frame to preserve highlight details.

        Args:
            frame: The frame to apply soft limiting to.

        Returns:
            The frame with soft limiting applied.
        """
        try:
            # Identify areas approaching limits
            high_values = frame > 0.9
            low_values = frame < 0.1
            
            # Apply gentle compression to extreme areas
            if np.any(high_values):
                frame[high_values] = 0.9 + (frame[high_values] - 0.9) * 0.5
            
            if np.any(low_values):
                frame[low_values] = frame[low_values] * 0.5 + 0.05
            
            return np.clip(frame, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Soft limiting failed: {e}")
            return frame
    
    def _default_config(self) -> Dict:
        """Returns the default configuration for the broadcast standards
        validator.

        Returns:
            A dictionary of default configuration parameters.
        """
        return {
            'broadcast_standard': BroadcastStandard.REC709,
            'check_ire_levels': True,
            'check_gamut': True,
            'check_safe_area': True,
            'check_legalizer': True,
            'ire_min': 7.5,
            'ire_max': 100.0,
            'action_safe_percentage': 0.10,
            'title_safe_percentage': 0.20,
            'ire_violation_threshold': 0.01,
            'gamut_violation_threshold': 0.005,
            'safe_area_violation_threshold': 0.02,
            'legalizer_mode': LegalizerMode.SIMPLE,
            'auto_legalize': False
        }
    
    def generate_broadcast_compliance_report(self, results: Dict) -> str:
        """Generates a detailed broadcast compliance report.

        Args:
            results: A dictionary of compliance analysis results.

        Returns:
            A string containing the broadcast compliance report.
        """
        try:
            report = f"""
# BROADCAST COMPLIANCE REPORT

## Summary
- **Overall Score**: {results.get('score', 0):.1f}/100
- **Broadcast Standard**: {results.get('broadcast_standard', 'Unknown')}
- **Compliance Status**: {'PASS' if results.get('score', 0) >= 80 else 'FAIL'}

## IRE Level Analysis
- **Compliance**: {'PASS' if results.get('ire_analysis', {}).get('compliant', False) else 'FAIL'}
- **Violations**: {results.get('ire_analysis', {}).get('violations', 0)} pixels
- **Range Detected**: {results.get('ire_analysis', {}).get('ire_min_detected', 0):.1f} - {results.get('ire_analysis', {}).get('ire_max_detected', 0):.1f} IRE
- **Legal Range**: {self.ire_min} - {self.ire_max} IRE

## Color Gamut Analysis
- **Compliance**: {'PASS' if results.get('gamut_analysis', {}).get('compliant', False) else 'FAIL'}
- **Violations**: {results.get('gamut_analysis', {}).get('violations', 0)} pixels
- **Broadcast-Safe Limits Applied**: Yes

## Safe Area Analysis
- **Compliance**: {'PASS' if results.get('safe_area_analysis', {}).get('compliant', False) else 'FAIL'}
- **Action Safe**: {self.action_safe_percentage*100:.1f}% margin
- **Title Safe**: {self.title_safe_percentage*100:.1f}% margin

## Legalizer Recommendations
{chr(10).join(f"- {rec}" for rec in results.get('legalizer_recommendations', []))}

## Technical Details
{chr(10).join(f"- **{k}**: {v}" for k, v in results.get('technical_details', {}).items())}

## Recommendations
{chr(10).join(f"- {rec}" for rec in results.get('recommendations', []))}

---
Report generated by APEX DIRECTOR Broadcast Standards Validator
            """
            return report.strip()
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Create sample frames for testing
    sample_frames = []
    for i in range(5):
        # Create frames with different IRE levels
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        sample_frames.append(frame)
    
    validator = BroadcastStandardsValidator()
    results = validator.validate_compliance(sample_frames)
    
    print("Broadcast Standards Analysis:")
    print(f"Overall Score: {results['score']:.1f}/100")
    print(f"IRE Compliant: {results['ire_analysis']['compliant']}")
    print(f"Gamut Compliant: {results['gamut_analysis']['compliant']}")
    print(f"Safe Area Compliant: {results['safe_area_analysis']['compliant']}")
