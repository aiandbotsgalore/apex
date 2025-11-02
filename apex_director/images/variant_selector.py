"""
Multi-Variant Selection System
4-criteria scoring: CLIP aesthetic + composition + style + artifacts
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from PIL import Image, ImageFilter, ImageEnhance
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class QualityScores:
    """Individual quality scores for an image variant.

    Attributes:
        clip_aesthetic_score: The CLIP aesthetic quality score.
        composition_score: The composition quality score.
        style_consistency_score: The style consistency score.
        artifacts_score: The technical artifact score.
        overall_score: The weighted overall score.
        timestamp: The time when the scores were generated.
    """
    clip_aesthetic_score: float  # CLIP aesthetic quality
    composition_score: float     # Composition quality
    style_consistency_score: float  # Style consistency
    artifacts_score: float       # Technical artifact score
    overall_score: float         # Weighted overall score
    timestamp: float             # When scored
    
    def to_dict(self) -> Dict[str, float]:
        """Converts the QualityScores to a dictionary.

        Returns:
            A dictionary representation of the QualityScores.
        """
        return asdict(self)
    
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Checks if the overall score is above a certain threshold.

        Args:
            threshold: The threshold to use for the check.

        Returns:
            True if the overall score is above the threshold, False otherwise.
        """
        return self.overall_score >= threshold

@dataclass
class VariantResult:
    """The complete result of a variant selection.

    Attributes:
        variant_id: The ID of the variant.
        image: The variant image.
        scores: The quality scores for the variant.
        ranking: The rank of the variant in the selection.
        selection_reason: The reason why the variant was selected.
        technical_details: A dictionary of technical details about the variant.
    """
    variant_id: str
    image: Image.Image
    scores: QualityScores
    ranking: int  # Rank in selection (1 = best)
    selection_reason: str
    technical_details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the VariantResult to a dictionary.

        Returns:
            A dictionary representation of the VariantResult.
        """
        return {
            "variant_id": self.variant_id,
            "scores": self.scores.to_dict(),
            "ranking": self.ranking,
            "selection_reason": self.selection_reason,
            "technical_details": self.technical_details
        }

class CLIPScorer:
    """A class for scoring images using CLIP-based aesthetic and semantic
    scoring.

    This is a simulated implementation that uses image properties to generate
    scores. A real implementation would use a pre-trained CLIP model.
    """
    
    def __init__(self):
        """Initializes the CLIPScorer."""
        # Placeholder for CLIP model initialization
        # In real implementation, would load CLIP model
        self.model_loaded = False
        self.aesthetic_prompts = [
            "beautiful composition, professional photography, cinematic quality",
            "high aesthetic value, artistic, visually appealing",
            "perfect lighting, ideal framing, stunning visual appeal"
        ]
        logger.info("CLIP scorer initialized (simulated)")
    
    def score_aesthetic_quality(self, image: Image.Image) -> float:
        """Scores the aesthetic quality of an image using CLIP.

        This is a placeholder implementation. A real implementation would use
        CLIP to compare the image against aesthetic prompts.

        Args:
            image: The image to score.

        Returns:
            The aesthetic quality score.
        """
        # Placeholder implementation
        # In reality, would use CLIP to compare image against aesthetic prompts
        
        # Simulate aesthetic scoring based on image properties
        scores = []
        
        # Resolution score
        width, height = image.size
        resolution_score = min(1.0, (width * height) / (1024 * 1024))
        scores.append(resolution_score)
        
        # Sharpness score (edge detection)
        try:
            edge_score = self._calculate_sharpness_score(image)
            scores.append(edge_score)
        except:
            scores.append(0.8)  # Default if calculation fails
        
        # Color balance score
        color_score = self._calculate_color_balance_score(image)
        scores.append(color_score)
        
        # Combine scores
        aesthetic_score = np.mean(scores)
        
        logger.info(f"CLIP aesthetic score: {aesthetic_score:.3f}")
        return aesthetic_score
    
    def score_semantic_consistency(self, image: Image.Image, prompt: str) -> float:
        """Scores the semantic consistency between an image and a prompt.

        This is a placeholder implementation. A real implementation would
        compare the image embedding with the text prompt embedding.

        Args:
            image: The image to score.
            prompt: The prompt to compare against.

        Returns:
            The semantic consistency score.
        """
        # Placeholder for CLIP semantic scoring
        # Would compare image embedding with text prompt embedding
        
        # Mock semantic consistency score
        semantic_score = 0.85  # Simulated score
        
        logger.info(f"Semantic consistency score: {semantic_score:.3f}")
        return semantic_score
    
    def _calculate_sharpness_score(self, image: Image.Image) -> float:
        """Calculates a sharpness score using edge detection.

        Args:
            image: The image to calculate the sharpness of.

        Returns:
            The sharpness score.
        """
        try:
            # Convert to grayscale for edge detection
            gray = image.convert('L')
            
            # Apply edge detection filter
            edges = gray.filter(ImageFilter.FIND_EDGES)
            
            # Calculate variance of edge pixels
            edge_array = np.array(edges)
            sharpness = np.var(edge_array) / 255.0  # Normalize to 0-1
            
            # Normalize sharpness score
            sharpness_score = min(1.0, sharpness * 10)
            
            return sharpness_score
        except:
            return 0.8  # Default score
    
    def _calculate_color_balance_score(self, image: Image.Image) -> float:
        """Calculates a color balance and distribution score.

        Args:
            image: The image to calculate the color balance of.

        Returns:
            The color balance score.
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Calculate color histogram balance
            color_balance_scores = []
            
            for channel in range(3):  # R, G, B
                channel_data = img_array[:, :, channel]
                histogram = np.histogram(channel_data, bins=256, range=(0, 255))[0]
                
                # Calculate balance (how evenly distributed the colors are)
                balance = 1.0 - (np.std(histogram) / np.mean(histogram))
                color_balance_scores.append(max(0, balance))
            
            overall_balance = np.mean(color_balance_scores)
            return min(1.0, overall_balance)
        except:
            return 0.8  # Default score

class CompositionAnalyzer:
    """A class for analyzing the composition quality of an image.

    This class uses a set of composition rules to score an image. The rules
    include the rule of thirds, leading lines, symmetry, depth, and framing.
    """
    
    def __init__(self):
        """Initializes the CompositionAnalyzer."""
        self.composition_rules = {
            "rule_of_thirds": {"weight": 0.3, "description": "Subject placement on thirds grid"},
            "leading_lines": {"weight": 0.2, "description": "Lines guiding eye to subject"},
            "symmetry": {"weight": 0.2, "description": "Balanced composition"},
            "depth": {"weight": 0.15, "description": "Foreground, middle, background layers"},
            "framing": {"weight": 0.15, "description": "Subject framing and boundary elements"}
        }
    
    def score_composition(self, image: Image.Image) -> float:
        """Scores the composition quality of an image.

        Args:
            image: The image to score.

        Returns:
            The composition quality score.
        """
        composition_scores = []
        
        # Rule of thirds analysis
        thirds_score = self._analyze_rule_of_thirds(image)
        composition_scores.append(thirds_score * self.composition_rules["rule_of_thirds"]["weight"])
        
        # Leading lines analysis
        lines_score = self._analyze_leading_lines(image)
        composition_scores.append(lines_score * self.composition_rules["leading_lines"]["weight"])
        
        # Symmetry analysis
        symmetry_score = self._analyze_symmetry(image)
        composition_scores.append(symmetry_score * self.composition_rules["symmetry"]["weight"])
        
        # Depth analysis
        depth_score = self._analyze_depth(image)
        composition_scores.append(depth_score * self.composition_rules["depth"]["weight"])
        
        # Framing analysis
        framing_score = self._analyze_framing(image)
        composition_scores.append(framing_score * self.composition_rules["framing"]["weight"])
        
        total_score = sum(composition_scores)
        
        logger.info(f"Composition score: {total_score:.3f}")
        return total_score
    
    def _analyze_rule_of_thirds(self, image: Image.Image) -> float:
        """Analyzes the rule of thirds compliance of an image.

        This is a placeholder implementation. A real implementation would
        detect the subject's position relative to the thirds grid.

        Args:
            image: The image to analyze.

        Returns:
            The rule of thirds score.
        """
        # Placeholder for rule of thirds analysis
        # Would detect subject position relative to thirds grid
        
        width, height = image.size
        thirds_x = [width // 3, 2 * width // 3]
        thirds_y = [height // 3, 2 * height // 3]
        
        # Mock rule of thirds score
        rule_score = 0.75  # Simulated score
        
        return rule_score
    
    def _analyze_leading_lines(self, image: Image.Image) -> float:
        """Detects and analyzes leading lines in an image.

        This is a placeholder implementation. A real implementation would use
        edge detection and line detection algorithms.

        Args:
            image: The image to analyze.

        Returns:
            The leading lines score.
        """
        # Placeholder for leading lines detection
        # Would use edge detection and line detection algorithms
        
        # Mock leading lines score
        lines_score = 0.6  # Simulated score
        
        return lines_score
    
    def _analyze_symmetry(self, image: Image.Image) -> float:
        """Analyzes the compositional symmetry of an image.

        Args:
            image: The image to analyze.

        Returns:
            The symmetry score.
        """
        # Placeholder for symmetry analysis
        
        # Convert to grayscale for comparison
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Calculate horizontal symmetry
        height = img_array.shape[0]
        top_half = img_array[:height//2]
        bottom_half = img_array[height//2:][::-1]
        
        if len(top_half) == len(bottom_half):
            symmetry_diff = np.mean(np.abs(top_half - bottom_half))
            symmetry_score = 1.0 - (symmetry_diff / 255.0)
        else:
            symmetry_score = 0.7  # Default for asymmetric images
        
        return max(0, symmetry_score)
    
    def _analyze_depth(self, image: Image.Image) -> float:
        """Analyzes the depth and layering of an image.

        This is a placeholder implementation. A real implementation would use
        more sophisticated techniques to analyze depth.

        Args:
            image: The image to analyze.

        Returns:
            The depth score.
        """
        # Placeholder for depth analysis
        
        # Mock depth score based on sharpness variation
        try:
            # Calculate sharpness variation across image
            sharpness_map = self._calculate_sharpness_map(image)
            depth_score = np.std(sharpness_map) / np.mean(sharpness_map)
            depth_score = min(1.0, depth_score)  # Normalize
        except:
            depth_score = 0.8
        
        return depth_score
    
    def _analyze_framing(self, image: Image.Image) -> float:
        """Analyzes the subject framing of an image.

        This is a placeholder implementation. A real implementation would
        detect the subject and its framing.

        Args:
            image: The image to analyze.

        Returns:
            The framing score.
        """
        # Placeholder for framing analysis
        
        # Mock framing score
        framing_score = 0.8  # Simulated score
        
        return framing_score
    
    def _calculate_sharpness_map(self, image: Image.Image) -> np.ndarray:
        """Calculates a sharpness map of the image.

        Args:
            image: The image to calculate the sharpness map of.

        Returns:
            A NumPy array representing the sharpness map.
        """
        try:
            # Convert to grayscale
            gray = image.convert('L')
            
            # Calculate gradient (sharpness indicator)
            img_array = np.array(gray, dtype=np.float64)
            
            # Calculate gradients
            grad_x = np.abs(np.diff(img_array, axis=1))
            grad_y = np.abs(np.diff(img_array, axis=0))
            
            # Combine gradients
            gradient = np.sqrt(grad_x[:-1]**2 + grad_y[:, :-1]**2)
            
            return gradient / 255.0  # Normalize to 0-1
        except:
            return np.ones((image.height//4, image.width//4)) * 0.8

class StyleConsistencyScorer:
    """A class for scoring the style consistency of an image with a reference
    style.

    This class works by extracting style features from a reference image and
    then comparing those features to the features of a new image. The style
    features include the color palette, texture, and lighting characteristics.
    """
    
    def __init__(self, style_reference: Optional[Image.Image] = None):
        """Initializes the StyleConsistencyScorer.

        Args:
            style_reference: The reference image to use for style comparison.
        """
        self.style_reference = style_reference
        self.style_features = None
        if style_reference:
            self._extract_style_features(style_reference)
    
    def _extract_style_features(self, reference_image: Image.Image):
        """Extracts style features from a reference image.

        This is a placeholder implementation. A real implementation would
        extract more sophisticated style features.

        Args:
            reference_image: The reference image.
        """
        # Placeholder for style feature extraction
        # Would extract color palette, texture, lighting characteristics
        
        img_array = np.array(reference_image)
        self.style_features = {
            "color_palette": self._extract_color_palette(img_array),
            "texture_stats": self._calculate_texture_stats(img_array),
            "lighting_stats": self._calculate_lighting_stats(img_array)
        }
    
    def _extract_color_palette(self, img_array: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extracts the dominant colors from an image.

        Args:
            img_array: The image as a NumPy array.

        Returns:
            A list of the dominant colors.
        """
        # Simplified color palette extraction
        pixels = img_array.reshape(-1, 3)
        
        # Simple histogram approach
        color_bins = {}
        for pixel in pixels:
            # Quantize colors to reduce categories
            quantized = (pixel // 32) * 32
            key = tuple(quantized)
            color_bins[key] = color_bins.get(key, 0) + 1
        
        # Get top colors
        top_colors = sorted(color_bins.items(), key=lambda x: x[1], reverse=True)[:8]
        return [color[0] for color in top_colors]
    
    def _calculate_texture_stats(self, img_array: np.ndarray) -> Dict[str, float]:
        """Calculates texture statistics for an image.

        Args:
            img_array: The image as a NumPy array.

        Returns:
            A dictionary of texture statistics.
        """
        # Convert to grayscale for texture analysis
        gray = np.mean(img_array, axis=2)
        
        return {
            "contrast": np.std(gray),
            "uniformity": 1.0 / (1.0 + np.std(gray)),
            "entropy": self._calculate_entropy(gray)
        }
    
    def _calculate_lighting_stats(self, img_array: np.ndarray) -> Dict[str, float]:
        """Calculates lighting characteristics for an image.

        Args:
            img_array: The image as a NumPy array.

        Returns:
            A dictionary of lighting statistics.
        """
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        return {
            "brightness": brightness / 255.0,
            "contrast": contrast / 255.0
        }
    
    def _calculate_entropy(self, array: np.ndarray) -> float:
        """Calculates the entropy of an array.

        Args:
            array: The array to calculate the entropy of.

        Returns:
            The entropy of the array.
        """
        hist, _ = np.histogram(array.flatten(), bins=256, range=(0, 255))
        hist = hist[hist > 0]  # Remove zero entries
        hist = hist / np.sum(hist)
        
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def score_style_consistency(self, image: Image.Image) -> float:
        """Scores the style consistency of an image against a reference.

        Args:
            image: The image to score.

        Returns:
            The style consistency score.
        """
        if not self.style_features:
            return 0.8  # Default score if no reference
        
        img_array = np.array(image)
        
        # Score color palette consistency
        color_score = self._score_color_consistency(img_array)
        
        # Score texture consistency
        texture_score = self._score_texture_consistency(img_array)
        
        # Score lighting consistency
        lighting_score = self._score_lighting_consistency(img_array)
        
        # Combine scores
        overall_score = (color_score + texture_score + lighting_score) / 3.0
        
        logger.info(f"Style consistency score: {overall_score:.3f}")
        return overall_score
    
    def _score_color_consistency(self, img_array: np.ndarray) -> float:
        """Scores the color palette consistency of an image.

        Args:
            img_array: The image as a NumPy array.

        Returns:
            The color consistency score.
        """
        test_palette = self._extract_color_palette(img_array)
        reference_palette = self.style_features["color_palette"]
        
        if not reference_palette:
            return 0.8
        
        # Calculate color distance between palettes
        distances = []
        for ref_color in reference_palette[:5]:  # Compare top 5 colors
            min_distance = float('inf')
            for test_color in test_palette[:5]:
                distance = np.linalg.norm(np.array(ref_color) - np.array(test_color))
                min_distance = min(min_distance, distance)
            distances.append(min_distance)
        
        # Convert distance to similarity score
        avg_distance = np.mean(distances)
        color_score = max(0, 1.0 - (avg_distance / 255.0))
        
        return color_score
    
    def _score_texture_consistency(self, img_array: np.ndarray) -> float:
        """Scores the texture consistency of an image.

        Args:
            img_array: The image as a NumPy array.

        Returns:
            The texture consistency score.
        """
        test_stats = self._calculate_texture_stats(img_array)
        ref_stats = self.style_features["texture_stats"]
        
        # Calculate similarity of texture statistics
        score = 0
        for key in ["contrast", "uniformity", "entropy"]:
            test_val = test_stats[key]
            ref_val = ref_stats[key]
            similarity = 1.0 - abs(test_val - ref_val)
            score += similarity
        
        return score / 3.0
    
    def _score_lighting_consistency(self, img_array: np.ndarray) -> float:
        """Scores the lighting consistency of an image.

        Args:
            img_array: The image as a NumPy array.

        Returns:
            The lighting consistency score.
        """
        test_stats = self._calculate_lighting_stats(img_array)
        ref_stats = self.style_features["lighting_stats"]
        
        # Calculate lighting similarity
        brightness_diff = abs(test_stats["brightness"] - ref_stats["brightness"])
        contrast_diff = abs(test_stats["contrast"] - ref_stats["contrast"])
        
        brightness_score = max(0, 1.0 - brightness_diff * 2)
        contrast_score = max(0, 1.0 - contrast_diff * 2)
        
        return (brightness_score + contrast_score) / 2.0

class ArtifactDetector:
    """A class for detecting technical artifacts and quality issues in an image.

    This class can detect a variety of artifacts, including noise, compression
    artifacts, color banding, and blurring.
    """
    
    def __init__(self):
        """Initializes the ArtifactDetector."""
        self.artifact_types = {
            "noise": {"weight": 0.3, "threshold": 0.1},
            "compression": {"weight": 0.25, "threshold": 0.15},
            "banding": {"weight": 0.25, "threshold": 0.1},
            "blurring": {"weight": 0.2, "threshold": 0.2}
        }
    
    def detect_artifacts(self, image: Image.Image) -> Dict[str, float]:
        """Detects various technical artifacts in an image.

        Args:
            image: The image to detect artifacts in.

        Returns:
            A dictionary of artifact scores.
        """
        artifact_scores = {}
        
        # Noise detection
        noise_score = self._detect_noise(image)
        artifact_scores["noise"] = noise_score
        
        # Compression artifacts
        compression_score = self._detect_compression_artifacts(image)
        artifact_scores["compression"] = compression_score
        
        # Color banding
        banding_score = self._detect_color_banding(image)
        artifact_scores["banding"] = banding_score
        
        # Excessive blurring
        blur_score = self._detect_blurring(image)
        artifact_scores["blurring"] = blur_score
        
        return artifact_scores
    
    def score_artifact_quality(self, image: Image.Image) -> float:
        """Scores the overall artifact quality of an image.

        A higher score indicates fewer artifacts.

        Args:
            image: The image to score.

        Returns:
            The artifact quality score.
        """
        artifacts = self.detect_artifacts(image)
        
        # Calculate weighted score
        total_score = 0
        total_weight = 0
        
        for artifact_type, score in artifacts.items():
            if artifact_type in self.artifact_types:
                weight = self.artifact_types[artifact_type]["weight"]
                total_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            quality_score = total_score / total_weight
        else:
            quality_score = 0.8
        
        logger.info(f"Artifact quality score: {quality_score:.3f}")
        return quality_score
    
    def _detect_noise(self, image: Image.Image) -> float:
        """Detects the noise level in an image.

        Args:
            image: The image to detect noise in.

        Returns:
            The noise quality score.
        """
        try:
            # Convert to grayscale
            gray = image.convert('L')
            img_array = np.array(gray, dtype=np.float64)
            
            # Calculate local variance (noise indicator)
            height, width = img_array.shape
            noise_score = 0
            
            for i in range(1, height-1, 5):  # Sample every 5 pixels
                for j in range(1, width-1, 5):
                    local_region = img_array[i-1:i+2, j-1:j+2]
                    local_var = np.var(local_region)
                    noise_score += local_var
            
            # Normalize noise score
            noise_score = noise_score / ((height * width) / 25)  # Account for sampling
            normalized_noise = min(1.0, noise_score / 1000.0)  # Adjust multiplier as needed
            
            # Return quality score (lower noise = higher quality)
            quality_score = max(0, 1.0 - normalized_noise)
            return quality_score
            
        except:
            return 0.8  # Default quality score
    
    def _detect_compression_artifacts(self, image: Image.Image) -> float:
        """Detects JPEG compression artifacts in an image.

        Args:
            image: The image to detect compression artifacts in.

        Returns:
            The compression artifact quality score.
        """
        try:
            # Look for block patterns and edge artifacts
            gray = image.convert('L')
            img_array = np.array(gray, dtype=np.float64)
            
            # Calculate 8x8 block variance (typical JPEG block size)
            block_size = 8
            height, width = img_array.shape
            artifact_score = 0
            blocks_checked = 0
            
            for i in range(0, height - block_size, block_size):
                for j in range(0, width - block_size, block_size):
                    block = img_array[i:i+block_size, j:j+block_size]
                    
                    # Check for uniform blocks (compression artifact)
                    block_var = np.var(block)
                    if block_var < 50:  # Very uniform block
                        artifact_score += 0.1
                    
                    blocks_checked += 1
            
            if blocks_checked > 0:
                compression_artifact = artifact_score / blocks_checked
                quality_score = max(0, 1.0 - compression_artifact)
            else:
                quality_score = 0.8
            
            return quality_score
            
        except:
            return 0.8
    
    def _detect_color_banding(self, image: Image.Image) -> float:
        """Detects color banding artifacts in an image.

        Args:
            image: The image to detect color banding in.

        Returns:
            The color banding quality score.
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image, dtype=np.float64)
            
            # Check for color gradients in each channel
            banding_scores = []
            
            for channel in range(3):
                channel_data = img_array[:, :, channel]
                
                # Calculate gradient magnitude
                grad_x = np.abs(np.diff(channel_data, axis=1))
                grad_y = np.abs(np.diff(channel_data, axis=0))
                
                # Low gradients indicate potential banding
                low_gradient_ratio = np.sum((grad_x < 5) & (grad_y < 5)) / grad_x.size
                banding_scores.append(low_gradient_ratio)
            
            avg_banding = np.mean(banding_scores)
            quality_score = max(0, 1.0 - avg_banding)
            
            return quality_score
            
        except:
            return 0.8
    
    def _detect_blurring(self, image: Image.Image) -> float:
        """Detects excessive blurring in an image.

        Args:
            image: The image to detect blurring in.

        Returns:
            The blurring quality score.
        """
        try:
            # Convert to grayscale
            gray = image.convert('L')
            img_array = np.array(gray, dtype=np.float64)
            
            # Calculate gradient magnitudes (sharpness indicator)
            grad_x = np.abs(np.diff(img_array, axis=1))
            grad_y = np.abs(np.diff(img_array, axis=0))
            gradient_magnitude = np.sqrt(grad_x[:-1]**2 + grad_y[:, :-1]**2)
            
            # Calculate average gradient
            avg_gradient = np.mean(gradient_magnitude)
            
            # Normalize and convert to blur score
            # Higher gradients = less blur = higher quality
            blur_score = min(1.0, avg_gradient / 50.0)  # Adjust denominator as needed
            quality_score = max(0, blur_score)
            
            return quality_score
            
        except:
            return 0.8

class VariantSelector:
    """The main variant selection system with 4-criteria scoring.

    This class uses a combination of CLIP-based aesthetic scoring, composition
    analysis, style consistency scoring, and artifact detection to select the
    best variants from a list of images.
    """
    
    def __init__(self, style_reference: Optional[Image.Image] = None):
        """Initializes the VariantSelector.

        Args:
            style_reference: The reference image to use for style comparison.
        """
        self.clip_scorer = CLIPScorer()
        self.composition_analyzer = CompositionAnalyzer()
        self.style_scorer = StyleConsistencyScorer(style_reference)
        self.artifact_detector = ArtifactDetector()
        
        # Scoring weights
        self.weights = {
            "clip_aesthetic": 0.35,
            "composition": 0.25,
            "style_consistency": 0.25,
            "artifacts": 0.15
        }
        
        logger.info("Variant selector initialized with 4-criteria scoring")
    
    def select_best_variants(
        self,
        variants: List[Tuple[str, Image.Image]],
        prompt: str,
        num_selections: int = 1
    ) -> List[VariantResult]:
        """Selects the best variants using 4-criteria scoring.

        Args:
            variants: A list of tuples, where each tuple contains the variant
                ID and the variant image.
            prompt: The prompt that was used to generate the variants.
            num_selections: The number of variants to select.

        Returns:
            A list of the best VariantResult objects.
        """
        
        logger.info(f"Evaluating {len(variants)} variants with 4-criteria scoring")
        
        scored_variants = []
        
        for variant_id, image in variants:
            # Score each variant
            scores = self._score_variant(image, prompt)
            
            # Calculate weighted overall score
            overall_score = (
                scores.clip_aesthetic_score * self.weights["clip_aesthetic"] +
                scores.composition_score * self.weights["composition"] +
                scores.style_consistency_score * self.weights["style_consistency"] +
                scores.artifacts_score * self.weights["artifacts"]
            )
            
            scores.overall_score = overall_score
            
            # Create variant result
            variant_result = VariantResult(
                variant_id=variant_id,
                image=image,
                scores=scores,
                ranking=0,  # Will be set after sorting
                selection_reason="",
                technical_details={
                    "individual_scores": {
                        "clip_aesthetic": scores.clip_aesthetic_score,
                        "composition": scores.composition_score,
                        "style_consistency": scores.style_consistency_score,
                        "artifacts": scores.artifacts_score
                    },
                    "scoring_weights": self.weights
                }
            )
            
            scored_variants.append(variant_result)
        
        # Sort by overall score
        scored_variants.sort(key=lambda x: x.scores.overall_score, reverse=True)
        
        # Set rankings and selection reasons
        for i, variant in enumerate(scored_variants[:num_selections]):
            variant.ranking = i + 1
            variant.selection_reason = self._generate_selection_reason(variant)
        
        # Log top selections
        for i, variant in enumerate(scored_variants[:num_selections]):
            logger.info(
                f"Rank {i+1}: {variant.variant_id} "
                f"(Score: {variant.scores.overall_score:.3f})"
            )
        
        return scored_variants[:num_selections]
    
    def _score_variant(self, image: Image.Image, prompt: str) -> QualityScores:
        """Scores a single variant with all criteria.

        Args:
            image: The image to score.
            prompt: The prompt that was used to generate the variant.

        Returns:
            A QualityScores object containing the scores for the variant.
        """
        
        # CLIP aesthetic scoring
        clip_score = self.clip_scorer.score_aesthetic_quality(image)
        semantic_score = self.clip_scorer.score_semantic_consistency(image, prompt)
        combined_clip_score = (clip_score + semantic_score) / 2.0
        
        # Composition scoring
        composition_score = self.composition_analyzer.score_composition(image)
        
        # Style consistency scoring
        style_score = self.style_scorer.score_style_consistency(image)
        
        # Artifact detection
        artifacts_score = self.artifact_detector.score_artifact_quality(image)
        
        return QualityScores(
            clip_aesthetic_score=combined_clip_score,
            composition_score=composition_score,
            style_consistency_score=style_score,
            artifacts_score=artifacts_score,
            overall_score=0.0,  # Will be calculated separately
            timestamp=time.time()
        )
    
    def _generate_selection_reason(self, variant: VariantResult) -> str:
        """Generates a human-readable reason for the selection of a variant.

        Args:
            variant: The VariantResult object.

        Returns:
            A string containing the selection reason.
        """
        
        reasons = []
        
        # Highlight best individual scores
        scores = variant.scores
        
        if scores.clip_aesthetic_score >= 0.9:
            reasons.append("excellent aesthetic quality")
        elif scores.clip_aesthetic_score >= 0.8:
            reasons.append("good aesthetic quality")
        
        if scores.composition_score >= 0.9:
            reasons.append("outstanding composition")
        elif scores.composition_score >= 0.8:
            reasons.append("strong composition")
        
        if scores.style_consistency_score >= 0.9:
            reasons.append("perfect style consistency")
        elif scores.style_consistency_score >= 0.8:
            reasons.append("good style consistency")
        
        if scores.artifacts_score >= 0.9:
            reasons.append("clean technical quality")
        elif scores.artifacts_score >= 0.8:
            reasons.append("acceptable technical quality")
        
        # Overall assessment
        if scores.overall_score >= 0.9:
            overall_reason = "exceptional overall quality"
        elif scores.overall_score >= 0.8:
            overall_reason = "high overall quality"
        elif scores.overall_score >= 0.7:
            overall_reason = "good overall quality"
        else:
            overall_reason = "acceptable quality"
        
        reasons.append(overall_reason)
        
        return "; ".join(reasons)
    
    def get_scoring_explanation(self, variant: VariantResult) -> str:
        """Gets a detailed explanation of the scoring for a variant.

        Args:
            variant: The VariantResult object.

        Returns:
            A string containing the scoring explanation.
        """
        
        explanation = [
            f"Variant: {variant.variant_id}",
            f"Overall Score: {variant.scores.overall_score:.3f}",
            "",
            "Individual Scores:",
            f"  CLIP Aesthetic: {variant.scores.clip_aesthetic_score:.3f}",
            f"  Composition: {variant.scores.composition_score:.3f}",
            f"  Style Consistency: {variant.scores.style_consistency_score:.3f}",
            f"  Artifact Quality: {variant.scores.artifacts_score:.3f}",
            "",
            "Selection Reason:",
            f"  {variant.selection_reason}",
            "",
            "Scoring Weights:",
            f"  CLIP Aesthetic: {self.weights['clip_aesthetic']:.2f}",
            f"  Composition: {self.weights['composition']:.2f}",
            f"  Style Consistency: {self.weights['style_consistency']:.2f}",
            f"  Artifact Quality: {self.weights['artifacts']:.2f}"
        ]
        
        return "\n".join(explanation)
    
    def update_scoring_weights(
        self,
        weights: Optional[Dict[str, float]] = None,
        clip_aesthetic: Optional[float] = None,
        composition: Optional[float] = None,
        style_consistency: Optional[float] = None,
        artifacts: Optional[float] = None
    ):
        """Updates the scoring weights.

        Args:
            weights: A dictionary of weights to update.
            clip_aesthetic: The new weight for the CLIP aesthetic score.
            composition: The new weight for the composition score.
            style_consistency: The new weight for the style consistency score.
            artifacts: The new weight for the artifacts score.
        """
        
        if weights:
            self.weights.update(weights)
        
        if clip_aesthetic is not None:
            self.weights["clip_aesthetic"] = clip_aesthetic
        
        if composition is not None:
            self.weights["composition"] = composition
        
        if style_consistency is not None:
            self.weights["style_consistency"] = style_consistency
        
        if artifacts is not None:
            self.weights["artifacts"] = artifacts
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            for key in self.weights:
                self.weights[key] /= total_weight
        
        logger.info(f"Updated scoring weights: {self.weights}")
    
    def save_scoring_results(
        self,
        results: List[VariantResult],
        output_path: Path
    ):
        """Saves the scoring results to a file.

        Args:
            results: A list of VariantResult objects.
            output_path: The path to save the results to.
        """
        
        try:
            results_data = {
                "timestamp": time.time(),
                "num_variants": len(results),
                "scoring_weights": self.weights,
                "results": [result.to_dict() for result in results]
            }
            
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Saved scoring results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save scoring results: {e}")