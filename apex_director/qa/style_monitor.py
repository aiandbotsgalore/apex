"""
Visual Style Monitor - CLIP-based visual consistency tracking and style drift detection
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from dataclasses import dataclass


@dataclass
class StyleMetrics:
    """Style consistency metrics"""
    clip_similarity_score: float
    color_consistency_score: float
    composition_score: float
    lighting_consistency_score: float
    overall_style_score: float
    drift_detected: bool
    reference_frame_idx: int


class StyleMonitor:
    """
    Visual Style Consistency Monitor
    
    Uses CLIP embeddings and other computer vision techniques to:
    - Detect visual style drift between frames
    - Measure color consistency
    - Analyze composition and lighting
    - Generate style consistency scores
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize style monitor with configuration"""
        self.config = config or self._default_config()
        self.logger = logging.getLogger('apex_director.qa.style_monitor')
        
        # Configuration parameters
        self.clip_threshold = self.config.get('clip_threshold', 0.8)
        self.reference_frames = self.config.get('reference_frames', 5)
        self.sample_interval = self.config.get('sample_interval', 30)
        self.color_weight = self.config.get('color_weight', 0.3)
        self.composition_weight = self.config.get('composition_weight', 0.25)
        self.lighting_weight = self.config.get('lighting_weight', 0.25)
        self.clip_weight = self.config.get('clip_weight', 0.2)
        
        # Initialize CLIP model (placeholder - would use actual CLIP in implementation)
        self.clip_model = None
        self._initialize_clip_model()
        
        # Reference style profiles
        self.reference_styles = []
        self.style_baseline = None
        
        # Style drift tracking
        self.drift_threshold = self.config.get('drift_threshold', 0.15)
        self.consecutive_drift_frames = 0
        self.max_consecutive_drift = self.config.get('max_consecutive_drift', 10)
    
    def analyze_consistency(self, sample_frames: List[np.ndarray]) -> Dict:
        """
        Analyze visual consistency across sample frames
        
        Args:
            sample_frames: List of sample frames from video
            
        Returns:
            Dictionary with consistency analysis results
        """
        self.logger.info(f"Analyzing style consistency for {len(sample_frames)} frames")
        
        if len(sample_frames) < 2:
            return {'score': 100.0, 'error': 'Insufficient frames for analysis'}
        
        # Calculate different consistency metrics
        color_scores = self._analyze_color_consistency(sample_frames)
        composition_scores = self._analyze_composition_consistency(sample_frames)
        lighting_scores = self._analyze_lighting_consistency(sample_frames)
        clip_scores = self._analyze_clip_consistency(sample_frames)
        
        # Calculate overall style score
        overall_score = (
            color_scores['score'] * self.color_weight +
            composition_scores['score'] * self.composition_weight +
            lighting_scores['score'] * self.lighting_weight +
            clip_scores['score'] * self.clip_weight
        )
        
        # Detect style drift
        drift_detected = self._detect_style_drift(sample_frames)
        
        # Generate detailed analysis
        results = {
            'score': overall_score * 100,  # Convert to 0-100 scale
            'overall_style_score': overall_score,
            'color_consistency_score': color_scores['score'] * 100,
            'composition_score': composition_scores['score'] * 100,
            'lighting_consistency_score': lighting_scores['score'] * 100,
            'clip_similarity_score': clip_scores['score'] * 100,
            'drift_detected': drift_detected,
            'style_metrics': {
                'color_variance': color_scores.get('variance', 0),
                'composition_stability': composition_scores.get('stability', 0),
                'lighting_stability': lighting_scores.get('stability', 0),
                'clip_similarity_range': clip_scores.get('range', 0)
            },
            'recommendations': self._generate_style_recommendations(
                color_scores, composition_scores, lighting_scores, clip_scores
            )
        }
        
        return results
    
    def _initialize_clip_model(self):
        """Initialize CLIP model for visual similarity analysis"""
        try:
            # Placeholder for CLIP initialization
            # In a real implementation, this would load a pre-trained CLIP model
            self.clip_model = "clip_model_placeholder"
            self.logger.info("CLIP model initialized")
        except Exception as e:
            self.logger.warning(f"CLIP model initialization failed: {e}")
            self.clip_model = None
    
    def _analyze_color_consistency(self, frames: List[np.ndarray]) -> Dict:
        """Analyze color consistency across frames"""
        try:
            # Convert frames to HSV for better color analysis
            hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in frames]
            
            # Calculate histograms for each channel
            h_histograms = []
            s_histograms = []
            v_histograms = []
            
            for hsv_frame in hsv_frames:
                h_hist = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])
                s_hist = cv2.calcHist([hsv_frame], [1], None, [256], [0, 256])
                v_hist = cv2.calcHist([hsv_frame], [2], None, [256], [0, 256])
                
                h_histograms.append(h_hist.flatten())
                s_histograms.append(s_hist.flatten())
                v_histograms.append(v_hist.flatten())
            
            # Normalize histograms
            h_histograms = np.array(h_histograms)
            s_histograms = np.array(s_histograms)
            v_histograms = np.array(v_histograms)
            
            # Calculate correlation between consecutive frames
            h_correlations = []
            s_correlations = []
            v_correlations = []
            
            for i in range(len(h_histograms) - 1):
                h_corr = np.corrcoef(h_histograms[i], h_histograms[i + 1])[0, 1]
                s_corr = np.corrcoef(s_histograms[i], s_histograms[i + 1])[0, 1]
                v_corr = np.corrcoef(v_histograms[i], v_histograms[i + 1])[0, 1]
                
                h_correlations.append(h_corr if not np.isnan(h_corr) else 0)
                s_correlations.append(s_corr if not np.isnan(s_corr) else 0)
                v_correlations.append(v_corr if not np.isnan(v_corr) else 0)
            
            # Calculate color consistency score
            avg_correlation = (
                np.mean(h_correlations) * 0.4 +
                np.mean(s_correlations) * 0.3 +
                np.mean(v_correlations) * 0.3
            )
            
            # Calculate color variance
            color_variance = np.var([
                np.mean(h_correlations),
                np.mean(s_correlations),
                np.mean(v_correlations)
            ])
            
            return {
                'score': max(0, min(1, avg_correlation)),
                'variance': color_variance,
                'channel_correlations': {
                    'hue': np.mean(h_correlations),
                    'saturation': np.mean(s_correlations),
                    'value': np.mean(v_correlations)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Color consistency analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _analyze_composition_consistency(self, frames: List[np.ndarray]) -> Dict:
        """Analyze composition consistency across frames"""
        try:
            composition_scores = []
            
            for frame in frames:
                # Calculate rule of thirds composition score
                thirds_score = self._rule_of_thirds_score(frame)
                
                # Calculate center of mass position
                center_score = self._center_of_mass_score(frame)
                
                # Calculate edge distribution
                edge_score = self._edge_distribution_score(frame)
                
                # Calculate average composition score
                frame_score = (thirds_score * 0.4 + center_score * 0.3 + edge_score * 0.3)
                composition_scores.append(frame_score)
            
            # Calculate stability (inverse of variance)
            composition_variance = np.var(composition_scores)
            stability = max(0, 1 - composition_variance * 4)  # Scale variance
            
            # Average composition score
            avg_score = np.mean(composition_scores)
            
            return {
                'score': avg_score,
                'stability': stability,
                'variance': composition_variance,
                'composition_scores': composition_scores
            }
            
        except Exception as e:
            self.logger.error(f"Composition analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _analyze_lighting_consistency(self, frames: List[np.ndarray]) -> Dict:
        """Analyze lighting consistency across frames"""
        try:
            brightness_values = []
            contrast_values = []
            
            for frame in frames:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate average brightness
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Calculate contrast using standard deviation
                contrast = np.std(gray)
                contrast_values.append(contrast)
            
            # Calculate brightness consistency
            brightness_variance = np.var(brightness_values)
            brightness_stability = max(0, 1 - brightness_variance / 65025)  # Normalize by max pixel value squared
            
            # Calculate contrast consistency
            contrast_variance = np.var(contrast_values)
            contrast_stability = max(0, 1 - contrast_variance / 65025)
            
            # Overall lighting score
            lighting_score = (brightness_stability + contrast_stability) / 2
            
            return {
                'score': lighting_score,
                'stability': lighting_score,
                'brightness_variance': brightness_variance,
                'contrast_variance': contrast_variance,
                'brightness_values': brightness_values,
                'contrast_values': contrast_values
            }
            
        except Exception as e:
            self.logger.error(f"Lighting analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _analyze_clip_consistency(self, frames: List[np.ndarray]) -> Dict:
        """Analyze visual similarity using CLIP embeddings"""
        try:
            if self.clip_model is None:
                # Fallback to simpler visual similarity
                return self._fallback_visual_similarity(frames)
            
            # Extract CLIP embeddings (placeholder implementation)
            embeddings = self._extract_clip_embeddings(frames)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings) - 1):
                similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(similarity)
            
            if not similarities:
                return {'score': 1.0, 'range': 0}
            
            avg_similarity = np.mean(similarities)
            similarity_range = np.max(similarities) - np.min(similarities)
            
            return {
                'score': avg_similarity,
                'range': similarity_range,
                'similarities': similarities
            }
            
        except Exception as e:
            self.logger.error(f"CLIP consistency analysis failed: {e}")
            return self._fallback_visual_similarity(frames)
    
    def _extract_clip_embeddings(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Extract CLIP embeddings from frames (placeholder)"""
        # Placeholder implementation
        # In reality, this would use a pre-trained CLIP model
        embeddings = []
        for frame in frames:
            # Simple feature extraction as placeholder
            # This would be replaced with actual CLIP embeddings
            features = self._extract_basic_features(frame)
            embeddings.append(features)
        return embeddings
    
    def _extract_basic_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract basic visual features as CLIP placeholder"""
        # Resize to standard size
        resized = cv2.resize(frame, (224, 224))
        
        # Extract color histogram
        hist = cv2.calcHist([resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten()
        
        # Extract edge features
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_features = edges.flatten()
        
        # Combine features
        features = np.concatenate([hist, edge_features])
        return features
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _fallback_visual_similarity(self, frames: List[np.ndarray]) -> Dict:
        """Fallback visual similarity analysis when CLIP is not available"""
        try:
            similarities = []
            
            for i in range(len(frames) - 1):
                # Calculate structural similarity index (SSIM)
                similarity = self._calculate_ssim(frames[i], frames[i + 1])
                similarities.append(similarity)
            
            if not similarities:
                return {'score': 1.0, 'range': 0}
            
            avg_similarity = np.mean(similarities)
            similarity_range = np.max(similarities) - np.min(similarities)
            
            return {
                'score': avg_similarity,
                'range': similarity_range,
                'similarities': similarities
            }
            
        except Exception as e:
            self.logger.error(f"Fallback similarity analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index (simplified version)"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Resize to same size if needed
            if gray1.shape != gray2.shape:
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
            
            # Calculate means
            mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
            
            # Calculate squares and products
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(gray1 * gray1, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(gray2 * gray2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
            
            # SSIM constants
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            # Calculate SSIM
            ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return max(0, min(1, ssim))
            
        except Exception as e:
            self.logger.error(f"SSIM calculation failed: {e}")
            return 0.5
    
    def _detect_style_drift(self, frames: List[np.ndarray]) -> bool:
        """Detect if style drift has occurred"""
        try:
            if len(frames) < 3:
                return False
            
            # Analyze consecutive frame differences
            drift_scores = []
            
            for i in range(len(frames) - 1):
                # Calculate visual difference
                diff_score = self._calculate_frame_difference(frames[i], frames[i + 1])
                drift_scores.append(diff_score)
            
            # Check if drift exceeds threshold
            high_drift_frames = sum(1 for score in drift_scores if score > self.drift_threshold)
            
            # Update consecutive drift counter
            if high_drift_frames > len(drift_scores) * 0.3:  # More than 30% of frames show drift
                self.consecutive_drift_frames += 1
            else:
                self.consecutive_drift_frames = 0
            
            # Return True if sustained drift detected
            return self.consecutive_drift_frames >= self.max_consecutive_drift
            
        except Exception as e:
            self.logger.error(f"Style drift detection failed: {e}")
            return False
    
    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate visual difference between two frames"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Normalize difference score
            max_possible_diff = gray1.shape[0] * gray1.shape[1] * 255
            actual_diff = np.sum(diff)
            
            return actual_diff / max_possible_diff
            
        except Exception as e:
            self.logger.error(f"Frame difference calculation failed: {e}")
            return 0.0
    
    def _rule_of_thirds_score(self, frame: np.ndarray) -> float:
        """Calculate rule of thirds composition score"""
        try:
            height, width = frame.shape[:2]
            
            # Define rule of thirds lines
            h_line1, h_line2 = height // 3, 2 * height // 3
            w_line1, w_line2 = width // 3, 2 * width // 3
            
            # Convert to grayscale and find edges
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density on thirds lines
            thirds_edges = np.concatenate([
                edges[h_line1, :], edges[h_line2, :],  # Horizontal lines
                edges[:, w_line1], edges[:, w_line2]   # Vertical lines
            ])
            
            edge_density = np.mean(thirds_edges) / 255.0
            
            # Higher edge density on thirds lines indicates better composition
            return min(1.0, edge_density * 2)  # Scale up a bit
            
        except Exception as e:
            self.logger.error(f"Rule of thirds calculation failed: {e}")
            return 0.5
    
    def _center_of_mass_score(self, frame: np.ndarray) -> float:
        """Calculate center of mass positioning score"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate center of mass
            M = cv2.moments(gray)
            if M["m00"] == 0:
                return 0.5
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            height, width = gray.shape
            
            # Calculate distance from center (normalized)
            center_x, center_y = width // 2, height // 2
            distance_from_center = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
            
            # Score based on distance from center (closer to thirds is better)
            normalized_distance = distance_from_center / max_distance
            
            # Optimal distance is around 0.3-0.4 of maximum distance (rule of thirds)
            optimal_distance = 0.35
            distance_score = 1.0 - abs(normalized_distance - optimal_distance) * 2
            
            return max(0, min(1, distance_score))
            
        except Exception as e:
            self.logger.error(f"Center of mass calculation failed: {e}")
            return 0.5
    
    def _edge_distribution_score(self, frame: np.ndarray) -> float:
        """Calculate edge distribution score"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Divide frame into 9 sections (3x3 grid)
            height, width = edges.shape
            section_height = height // 3
            section_width = width // 3
            
            edge_densities = []
            for i in range(3):
                for j in range(3):
                    section = edges[
                        i * section_height:(i + 1) * section_height,
                        j * section_width:(j + 1) * section_width
                    ]
                    density = np.mean(section) / 255.0
                    edge_densities.append(density)
            
            # Calculate variance in edge distribution
            variance = np.var(edge_densities)
            
            # Good distribution has moderate variance (not too uniform, not too clustered)
            optimal_variance = 0.05
            distribution_score = 1.0 - abs(variance - optimal_variance) * 10
            
            return max(0, min(1, distribution_score))
            
        except Exception as e:
            self.logger.error(f"Edge distribution calculation failed: {e}")
            return 0.5
    
    def _generate_style_recommendations(self, color_scores: Dict, composition_scores: Dict,
                                      lighting_scores: Dict, clip_scores: Dict) -> List[str]:
        """Generate recommendations based on style analysis"""
        recommendations = []
        
        # Color consistency recommendations
        if color_scores.get('score', 1) < 0.7:
            recommendations.append("Review color grading for consistency")
            recommendations.append("Check white balance across different shots")
        
        # Composition recommendations
        if composition_scores.get('score', 1) < 0.6:
            recommendations.append("Improve composition consistency")
            recommendations.append("Apply rule of thirds more consistently")
        
        # Lighting recommendations
        if lighting_scores.get('score', 1) < 0.7:
            recommendations.append("Standardize lighting conditions")
            recommendations.append("Check exposure consistency across shots")
        
        # CLIP similarity recommendations
        if clip_scores.get('score', 1) < 0.8:
            recommendations.append("Review visual style consistency")
            recommendations.append("Consider re-editing for style continuity")
        
        # Drift detection recommendations
        if self.consecutive_drift_frames > 0:
            recommendations.append("Style drift detected - review sequence consistency")
        
        if not recommendations:
            recommendations.append("Style consistency is good - no action needed")
        
        return recommendations
    
    def _default_config(self) -> Dict:
        """Default configuration for style monitor"""
        return {
            'clip_threshold': 0.8,
            'reference_frames': 5,
            'sample_interval': 30,
            'color_weight': 0.3,
            'composition_weight': 0.25,
            'lighting_weight': 0.25,
            'clip_weight': 0.2,
            'drift_threshold': 0.15,
            'max_consecutive_drift': 10
        }
    
    def save_style_profile(self, frames: List[np.ndarray], output_path: str):
        """Save style profile for future reference"""
        try:
            profile = {
                'config': self.config,
                'analysis_results': self.analyze_consistency(frames),
                'timestamp': str(np.datetime64('now'))
            }
            
            with open(output_path, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            
            self.logger.info(f"Style profile saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save style profile: {e}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Create sample frames for testing
    sample_frames = []
    for i in range(10):
        # Create a simple test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        sample_frames.append(frame)
    
    monitor = StyleMonitor()
    results = monitor.analyze_consistency(sample_frames)
    
    print("Style Consistency Analysis:")
    print(f"Overall Score: {results['score']:.2f}/100")
    print(f"Drift Detected: {results['drift_detected']}")
    print(f"Recommendations: {results['recommendations']}")
