"""
Artifact Detector
Quality issue identification including face detection, text/watermark removal
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import re


class ArtifactType(Enum):
    """Types of artifacts to detect"""
    FACE = "face"
    TEXT = "text"
    WATERMARK = "watermark"
    LOGO = "logo"
    COPYRIGHT_NOTICE = "copyright_notice"
    BRANDING = "branding"
    SUBTITLE = "subtitle"
    CREDITS = "credits"
    FRAME_DROP = "frame_drop"
    COMPRESSION_ARTIFACT = "compression_artifact"
    NOISE = "noise"
    banding = "banding"
    BLOCKING = "blocking"
    RINGING = "ringing"


@dataclass
class ArtifactDetection:
    """Artifact detection result"""
    artifact_type: ArtifactType
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    severity: str = "low"  # low, medium, high
    description: str = ""
    action_required: str = ""


class ArtifactDetector:
    """
    Quality Issue Artifact Detector
    
    Identifies various types of artifacts in video content:
    - Face detection and privacy concerns
    - Text and watermark identification
    - Logo and branding detection
    - Compression and encoding artifacts
    - Technical quality issues
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize artifact detector"""
        self.config = config or self._default_config()
        self.logger = logging.getLogger('apex_director.qa.artifact_detector')
        
        # Configuration parameters
        self.detect_faces = self.config.get('detect_faces', True)
        self.detect_text = self.config.get('detect_text', True)
        self.detect_watermarks = self.config.get('detect_watermarks', True)
        self.detect_logos = self.config.get('detect_logos', True)
        self.detect_compression_artifacts = self.config.get('detect_compression_artifacts', True)
        
        # Detection thresholds
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.face_confidence_threshold = self.config.get('face_confidence_threshold', 0.8)
        self.text_confidence_threshold = self.config.get('text_confidence_threshold', 0.6)
        
        # Analysis parameters
        self.sample_interval = self.config.get('sample_interval', 30)  # Process every Nth frame
        self.max_faces_per_frame = self.config.get('max_faces_per_frame', 10)
        self.face_blur_threshold = self.config.get('face_blur_threshold', 100)  # Laplacian variance
        
        # Initialize detection models
        self.face_detector = None
        self.text_detector = None
        self.logo_detector = None
        
        self._initialize_detectors()
        
        # Storage for analysis results
        self.artifact_history = []
        self.face_registry = []
        self.text_occurrences = []
        
        # Watermark detection parameters
        self.watermark_patterns = self._initialize_watermark_patterns()
        self.branding_keywords = self._initialize_branding_keywords()
    
    def detect_artifacts(self, sample_frames: List[np.ndarray]) -> Dict:
        """
        Detect artifacts in sample frames
        
        Args:
            sample_frames: List of sample frames from video
            
        Returns:
            Dictionary with artifact detection results
        """
        self.logger.info(f"Detecting artifacts in {len(sample_frames)} frames")
        
        if len(sample_frames) == 0:
            return {
                'score': 0.0,
                'error': 'No frames provided for analysis'
            }
        
        try:
            all_artifacts = []
            
            # Process frames with sampling
            process_frames = sample_frames[::self.sample_interval]
            
            # Detect different types of artifacts
            if self.detect_faces:
                face_artifacts = self._detect_faces(process_frames)
                all_artifacts.extend(face_artifacts)
            
            if self.detect_text:
                text_artifacts = self._detect_text(process_frames)
                all_artifacts.extend(text_artifacts)
            
            if self.detect_watermarks:
                watermark_artifacts = self._detect_watermarks(process_frames)
                all_artifacts.extend(watermark_artifacts)
            
            if self.detect_logos:
                logo_artifacts = self._detect_logos(process_frames)
                all_artifacts.extend(logo_artifacts)
            
            if self.detect_compression_artifacts:
                compression_artifacts = self._detect_compression_artifacts(sample_frames)
                all_artifacts.extend(compression_artifacts)
            
            # Analyze noise and quality issues
            quality_artifacts = self._detect_quality_issues(sample_frames)
            all_artifacts.extend(quality_artifacts)
            
            # Calculate overall artifact score
            artifact_score = self._calculate_artifact_score(all_artifacts, len(sample_frames))
            
            # Categorize artifacts
            artifact_summary = self._categorize_artifacts(all_artifacts)
            
            # Generate recommendations
            recommendations = self._generate_artifact_recommendations(all_artifacts)
            
            results = {
                'score': artifact_score * 100,  # Convert to 0-100 scale (higher = fewer artifacts)
                'total_artifacts': len(all_artifacts),
                'faces_detected': len([a for a in all_artifacts if a.artifact_type == ArtifactType.FACE]),
                'text_detected': len([a for a in all_artifacts if a.artifact_type == ArtifactType.TEXT]),
                'watermarks_detected': len([a for a in all_artifacts if a.artifact_type == ArtifactType.WATERMARK]),
                'logos_detected': len([a for a in all_artifacts if a.artifact_type == ArtifactType.LOGO]),
                'compression_artifacts': len([a for a in all_artifacts if a.artifact_type == ArtifactType.COMPRESSION_ARTIFACT]),
                'quality_issues': len([a for a in all_artifacts if a.artifact_type in [ArtifactType.NOISE, ArtifactType.banding, ArtifactType.BLOCKING]]),
                'artifact_summary': artifact_summary,
                'detected_artifacts': [
                    {
                        'type': artifact.artifact_type.value,
                        'confidence': artifact.confidence,
                        'severity': artifact.severity,
                        'description': artifact.description,
                        'action_required': artifact.action_required,
                        'bounding_box': artifact.bounding_box
                    }
                    for artifact in all_artifacts
                ],
                'recommendations': recommendations,
                'privacy_concerns': self._assess_privacy_concerns(all_artifacts),
                'legal_concerns': self._assess_legal_concerns(all_artifacts)
            }
            
            # Store results
            self._store_artifact_results(results, all_artifacts)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Artifact detection failed: {e}")
            return {
                'score': 0.0,
                'error': str(e)
            }
    
    def _initialize_detectors(self):
        """Initialize detection models"""
        try:
            # Initialize face detector (Haar cascades as fallback)
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Text detection would use EAST text detector or similar in real implementation
            self.text_detector = "east_text_detector"  # Placeholder
            
            # Logo detection would use custom trained models in real implementation
            self.logo_detector = "custom_logo_detector"  # Placeholder
            
            self.logger.info("Artifact detectors initialized")
            
        except Exception as e:
            self.logger.warning(f"Detector initialization failed: {e}")
            self.face_detector = None
            self.text_detector = None
            self.logo_detector = None
    
    def _detect_faces(self, frames: List[np.ndarray]) -> List[ArtifactDetection]:
        """Detect faces in frames"""
        artifacts = []
        
        try:
            if self.face_detector is None:
                self.logger.warning("Face detector not available")
                return artifacts
            
            for frame_idx, frame in enumerate(frames):
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (x, y, w, h) in faces[:self.max_faces_per_frame]:
                    # Calculate confidence based on face size and clarity
                    face_area = w * h
                    frame_area = frame.shape[0] * frame.shape[1]
                    size_confidence = min(1.0, face_area / (frame_area * 0.1))  # 10% of frame is good size
                    
                    # Check face clarity
                    face_roi = gray[y:y+h, x:x+w]
                    blur_score = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                    clarity_confidence = min(1.0, blur_score / self.face_blur_threshold)
                    
                    # Combine confidences
                    confidence = (size_confidence + clarity_confidence) / 2
                    
                    if confidence >= self.face_confidence_threshold:
                        severity = self._determine_face_severity(confidence, face_area, frame_area)
                        
                        artifact = ArtifactDetection(
                            artifact_type=ArtifactType.FACE,
                            confidence=confidence,
                            bounding_box=(x, y, w, h),
                            severity=severity,
                            description=f"Face detected with {confidence:.2f} confidence",
                            action_required=self._get_face_action_required(severity)
                        )
                        artifacts.append(artifact)
                        
                        # Register face for tracking
                        self._register_face_detection(frame_idx, (x, y, w, h), confidence)
            
            self.logger.info(f"Detected {len(artifacts)} faces")
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
        
        return artifacts
    
    def _detect_text(self, frames: List[np.ndarray]) -> List[ArtifactDetection]:
        """Detect text in frames"""
        artifacts = []
        
        try:
            for frame_idx, frame in enumerate(frames):
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply text detection using morphological operations
                # In real implementation, would use EAST text detector or Tesseract
                text_regions = self._detect_text_regions_morphological(gray)
                
                for region in text_regions:
                    x, y, w, h = region
                    
                    # Extract text region for analysis
                    text_roi = frame[y:y+h, x:x+w]
                    
                    # Analyze text characteristics
                    confidence = self._analyze_text_confidence(text_roi, region)
                    
                    if confidence >= self.text_confidence_threshold:
                        # Determine text type
                        text_type = self._classify_text_type(text_roi, region)
                        
                        severity = self._determine_text_severity(text_type, confidence)
                        
                        artifact = ArtifactDetection(
                            artifact_type=ArtifactType.TEXT,
                            confidence=confidence,
                            bounding_box=region,
                            severity=severity,
                            description=f"{text_type} detected with {confidence:.2f} confidence",
                            action_required=self._get_text_action_required(text_type, severity)
                        )
                        artifacts.append(artifact)
                        
                        # Register text occurrence
                        self._register_text_occurrence(frame_idx, region, text_type)
            
            self.logger.info(f"Detected {len(artifacts)} text elements")
            
        except Exception as e:
            self.logger.error(f"Text detection failed: {e}")
        
        return artifacts
    
    def _detect_watermarks(self, frames: List[np.ndarray]) -> List[ArtifactDetection]:
        """Detect watermarks and logos"""
        artifacts = []
        
        try:
            for frame_idx, frame in enumerate(frames):
                # Detect potential watermark regions
                watermark_regions = self._detect_potential_watermarks(frame)
                
                for region in watermark_regions:
                    x, y, w, h = region
                    
                    # Analyze watermark characteristics
                    watermark_confidence = self._analyze_watermark_confidence(frame, region)
                    
                    if watermark_confidence >= 0.6:  # Lower threshold for watermarks
                        severity = self._determine_watermark_severity(watermark_confidence, region, frame.shape)
                        
                        # Classify watermark type
                        watermark_type = self._classify_watermark_type(frame, region)
                        
                        artifact = ArtifactDetection(
                            artifact_type=ArtifactType.WATERMARK,
                            confidence=watermark_confidence,
                            bounding_box=region,
                            severity=severity,
                            description=f"{watermark_type} detected with {watermark_confidence:.2f} confidence",
                            action_required=self._get_watermark_action_required(watermark_type, severity)
                        )
                        artifacts.append(artifact)
            
            self.logger.info(f"Detected {len(artifacts)} potential watermarks")
            
        except Exception as e:
            self.logger.error(f"Watermark detection failed: {e}")
        
        return artifacts
    
    def _detect_logos(self, frames: List[np.ndarray]) -> List[ArtifactDetection]:
        """Detect logos and branding elements"""
        artifacts = []
        
        try:
            for frame in frames:
                # Template matching for known logos would be done here
                # For now, detect corner/edge regions that might contain logos
                logo_regions = self._detect_potential_logos(frame)
                
                for region in logo_regions:
                    x, y, w, h = region
                    
                    # Analyze logo characteristics
                    logo_confidence = self._analyze_logo_confidence(frame, region)
                    
                    if logo_confidence >= 0.5:
                        severity = self._determine_logo_severity(logo_confidence, region, frame.shape)
                        
                        artifact = ArtifactDetection(
                            artifact_type=ArtifactType.LOGO,
                            confidence=logo_confidence,
                            bounding_box=region,
                            severity=severity,
                            description=f"Potential logo detected with {logo_confidence:.2f} confidence",
                            action_required=self._get_logo_action_required(severity)
                        )
                        artifacts.append(artifact)
            
            self.logger.info(f"Detected {len(artifacts)} potential logos")
            
        except Exception as e:
            self.logger.error(f"Logo detection failed: {e}")
        
        return artifacts
    
    def _detect_compression_artifacts(self, frames: List[np.ndarray]) -> List[ArtifactDetection]:
        """Detect compression and encoding artifacts"""
        artifacts = []
        
        try:
            for frame_idx, frame in enumerate(frames):
                # Detect blocking artifacts
                blocking_score = self._detect_blocking_artifacts(frame)
                if blocking_score > 0.3:
                    artifact = ArtifactDetection(
                        artifact_type=ArtifactType.BLOCKING,
                        confidence=blocking_score,
                        severity=self._determine_blocking_severity(blocking_score),
                        description=f"Blocking artifacts detected with {blocking_score:.2f} intensity",
                        action_required="Increase compression quality or bitrate"
                    )
                    artifacts.append(artifact)
                
                # Detect banding artifacts
                banding_score = self._detect_banding_artifacts(frame)
                if banding_score > 0.25:
                    artifact = ArtifactDetection(
                        artifact_type=ArtifactType.banding,
                        confidence=banding_score,
                        severity=self._determine_banding_severity(banding_score),
                        description=f"Banding artifacts detected with {banding_score:.2f} intensity",
                        action_required="Increase color depth or apply dithering"
                    )
                    artifacts.append(artifact)
                
                # Detect ringing artifacts
                ringing_score = self._detect_ringing_artifacts(frame)
                if ringing_score > 0.2:
                    artifact = ArtifactDetection(
                        artifact_type=ArtifactType.RINGING,
                        confidence=ringing_score,
                        severity=self._determine_ringing_severity(ringing_score),
                        description=f"Ringing artifacts detected with {ringing_score:.2f} intensity",
                        action_required="Apply anti-aliasing or reduce compression"
                    )
                    artifacts.append(artifact)
            
            self.logger.info(f"Detected {len(artifacts)} compression artifacts")
            
        except Exception as e:
            self.logger.error(f"Compression artifact detection failed: {e}")
        
        return artifacts
    
    def _detect_quality_issues(self, frames: List[np.ndarray]) -> List[ArtifactDetection]:
        """Detect general quality issues"""
        artifacts = []
        
        try:
            for frame in frames:
                # Detect noise
                noise_score = self._detect_noise(frame)
                if noise_score > 0.4:
                    artifact = ArtifactDetection(
                        artifact_type=ArtifactType.NOISE,
                        confidence=noise_score,
                        severity=self._determine_noise_severity(noise_score),
                        description=f"High noise level detected: {noise_score:.2f}",
                        action_required="Apply denoising filter"
                    )
                    artifacts.append(artifact)
                
                # Detect motion blur
                blur_score = self._detect_motion_blur(frame)
                if blur_score > 0.3:
                    artifact = ArtifactDetection(
                        artifact_type=ArtifactType.COMPRESSION_ARTIFACT,  # Reuse type for blur
                        confidence=blur_score,
                        severity=self._determine_blur_severity(blur_score),
                        description=f"Motion blur detected: {blur_score:.2f}",
                        action_required="Check camera settings or apply deblurring"
                    )
                    artifacts.append(artifact)
            
            self.logger.info(f"Detected {len(artifacts)} quality issues")
            
        except Exception as e:
            self.logger.error(f"Quality issue detection failed: {e}")
        
        return artifacts
    
    def _detect_text_regions_morphological(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions using morphological operations"""
        try:
            # Apply morphological operations to detect text-like structures
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Apply closing to connect text components
            closed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter based on text-like characteristics
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # Text typically has small height, reasonable width
                if (10 < w < 400 and 5 < h < 100 and 
                    0.5 < aspect_ratio < 10 and area > 50):
                    text_regions.append((x, y, w, h))
            
            return text_regions
            
        except Exception as e:
            self.logger.error(f"Morphological text detection failed: {e}")
            return []
    
    def _detect_potential_watermarks(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential watermark regions"""
        try:
            # Look for semi-transparent overlays in corners and edges
            regions = []
            height, width = frame.shape[:2]
            
            # Define potential watermark regions (corners and edges)
            corner_regions = [
                (0, 0, width//6, height//6),  # Top-left
                (width - width//6, 0, width//6, height//6),  # Top-right
                (0, height - height//6, width//6, height//6),  # Bottom-left
                (width - width//6, height - height//6, width//6, height//6)  # Bottom-right
            ]
            
            edge_regions = [
                (width//3, 0, width//3, height//8),  # Top edge
                (width//3, height - height//8, width//3, height//8),  # Bottom edge
                (0, height//3, width//8, height//3),  # Left edge
                (width - width//8, height//3, width//8, height//3)  # Right edge
            ]
            
            potential_regions = corner_regions + edge_regions
            
            for region in potential_regions:
                x, y, w, h = region
                roi = frame[y:y+h, x:x+w]
                
                # Check for watermark characteristics
                if self._is_potential_watermark_region(roi):
                    regions.append(region)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Potential watermark detection failed: {e}")
            return []
    
    def _detect_potential_logos(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential logo regions"""
        try:
            regions = []
            height, width = frame.shape[:2]
            
            # Look for high-contrast regions that might contain logos
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter for logo-like characteristics
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # Logos are typically small and compact
                if (20 < w < 200 and 20 < h < 200 and 
                    0.3 < aspect_ratio < 3 and area > 200):
                    regions.append((x, y, w, h))
            
            return regions[:10]  # Limit to top 10 regions
            
        except Exception as e:
            self.logger.error(f"Potential logo detection failed: {e}")
            return []
    
    def _detect_blocking_artifacts(self, frame: np.ndarray) -> float:
        """Detect blocking artifacts in compressed video"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Look for 8x8 block patterns (typical JPEG block size)
            block_size = 8
            height, width = gray.shape
            
            blocking_score = 0
            block_count = 0
            
            for i in range(0, height - block_size, block_size):
                for j in range(0, width - block_size, block_size):
                    # Calculate variance within block
                    block = gray[i:i+block_size, j:j+block_size]
                    block_var = np.var(block)
                    
                    # Calculate variance difference with neighboring blocks
                    if i + block_size < height and j + block_size < width:
                        right_block = gray[i:i+block_size, j+block_size:j+2*block_size]
                        bottom_block = gray[i+block_size:i+2*block_size, j:j+block_size]
                        
                        if right_block.size > 0 and bottom_block.size > 0:
                            right_var = np.var(right_block)
                            bottom_var = np.var(bottom_block)
                            
                            # High variance difference suggests blocking
                            var_diff = abs(block_var - right_var) + abs(block_var - bottom_var)
                            blocking_score += var_diff / 1000  # Normalize
                            block_count += 1
            
            if block_count > 0:
                blocking_score /= block_count
            
            return min(1.0, blocking_score)
            
        except Exception as e:
            self.logger.error(f"Blocking artifact detection failed: {e}")
            return 0.0
    
    def _detect_banding_artifacts(self, frame: np.ndarray) -> float:
        """Detect color banding artifacts"""
        try:
            # Convert to different color spaces to detect banding
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Analyze hue channel for banding
            hue = hsv[:, :, 0]
            
            # Calculate gradient in hue channel
            grad_x = cv2.Sobel(hue, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(hue, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Low gradients in smooth areas suggest banding
            smooth_areas = gradient_magnitude < 5
            banding_score = np.mean(smooth_areas)
            
            return min(1.0, banding_score)
            
        except Exception as e:
            self.logger.error(f"Banding artifact detection failed: {e}")
            return 0.0
    
    def _detect_ringing_artifacts(self, frame: np.ndarray) -> float:
        """Detect ringing artifacts around edges"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Apply Laplacian to detect ringing
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate ringing score based on high-frequency content near edges
            edge_mask = edges > 0
            ringing_regions = np.abs(laplacian) > np.std(laplacian)
            
            overlap = edge_mask & ringing_regions
            ringing_score = np.sum(overlap) / np.sum(edge_mask) if np.sum(edge_mask) > 0 else 0
            
            return min(1.0, ringing_score * 2)  # Scale up
            
        except Exception as e:
            self.logger.error(f"Ringing artifact detection failed: {e}")
            return 0.0
    
    def _detect_noise(self, frame: np.ndarray) -> float:
        """Detect noise level in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur and compare with original
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Calculate difference
            diff = cv2.absdiff(gray, blurred)
            
            # Noise score based on high-frequency content
            noise_score = np.mean(diff) / 255.0
            
            return min(1.0, noise_score * 3)  # Scale up
            
        except Exception as e:
            self.logger.error(f"Noise detection failed: {e}")
            return 0.0
    
    def _detect_motion_blur(self, frame: np.ndarray) -> float:
        """Detect motion blur"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Laplacian to detect blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # High variance indicates sharp image, low variance indicates blur
            blur_threshold = 100
            blur_score = max(0, 1 - laplacian_var / blur_threshold)
            
            return min(1.0, blur_score)
            
        except Exception as e:
            self.logger.error(f"Motion blur detection failed: {e}")
            return 0.0
    
    def _is_potential_watermark_region(self, roi: np.ndarray) -> bool:
        """Check if region has watermark characteristics"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            
            # Check for semi-transparent characteristics
            alpha = cv2.calcHist([roi], [0], None, [256], [0, 256])
            alpha_smoothness = 1 - np.std(alpha[50:200]) / np.mean(alpha[50:200])
            
            # Check for corner positioning (already filtered in calling function)
            # Check for specific watermark patterns
            
            watermark_score = (
                alpha_smoothness * 0.4 +
                self._check_watermark_patterns(roi) * 0.6
            )
            
            return watermark_score > 0.3
            
        except Exception as e:
            self.logger.error(f"Watermark region analysis failed: {e}")
            return False
    
    def _check_watermark_patterns(self, roi: np.ndarray) -> float:
        """Check for watermark patterns"""
        try:
            # Look for semi-transparent text patterns
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to find text-like elements
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Score based on number and size of text-like elements
            score = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 500:  # Typical text size
                    score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"Watermark pattern check failed: {e}")
            return 0.0
    
    def _analyze_text_confidence(self, roi: np.ndarray, region: Tuple[int, int, int, int]) -> float:
        """Analyze text detection confidence"""
        try:
            # Simple confidence based on region characteristics
            x, y, w, h = region
            aspect_ratio = w / h if h > 0 else 0
            
            # Text typically has reasonable aspect ratio
            ratio_score = 1.0 if 0.5 < aspect_ratio < 10 else 0.5
            
            # Size score
            area = w * h
            size_score = 1.0 if 100 < area < 10000 else 0.3
            
            # Contrast score
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 50)
            
            confidence = (ratio_score + size_score + contrast_score) / 3
            return confidence
            
        except Exception as e:
            self.logger.error(f"Text confidence analysis failed: {e}")
            return 0.0
    
    def _analyze_watermark_confidence(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> float:
        """Analyze watermark detection confidence"""
        try:
            x, y, w, h = region
            roi = frame[y:y+h, x:x+w]
            
            # Semi-transparency check
            transparency_score = self._check_transparency(roi)
            
            # Corner position score
            height, width = frame.shape[:2]
            corner_score = 1.0 if (
                (x < width/3 and y < height/3) or
                (x > 2*width/3 and y < height/3) or
                (x < width/3 and y > 2*height/3) or
                (x > 2*width/3 and y > 2*height/3)
            ) else 0.5
            
            # Pattern consistency score
            pattern_score = self._check_watermark_patterns(roi)
            
            confidence = (transparency_score + corner_score + pattern_score) / 3
            return confidence
            
        except Exception as e:
            self.logger.error(f"Watermark confidence analysis failed: {e}")
            return 0.0
    
    def _analyze_logo_confidence(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> float:
        """Analyze logo detection confidence"""
        try:
            x, y, w, h = region
            roi = frame[y:y+h, x:x+w]
            
            # High contrast check
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 100)
            
            # Edge density check
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(1.0, edge_density * 5)
            
            # Symmetry check (logos often have some symmetry)
            symmetry_score = self._check_symmetry(roi)
            
            confidence = (contrast_score + edge_score + symmetry_score) / 3
            return confidence
            
        except Exception as e:
            self.logger.error(f"Logo confidence analysis failed: {e}")
            return 0.0
    
    def _check_transparency(self, roi: np.ndarray) -> float:
        """Check for semi-transparency characteristics"""
        try:
            # Look for mid-range values that suggest transparency
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Check for significant content in mid-range (128±50)
            mid_range_content = np.sum(hist[78:178]) / np.sum(hist)
            
            return min(1.0, mid_range_content * 2)
            
        except Exception as e:
            self.logger.error(f"Transparency check failed: {e}")
            return 0.0
    
    def _check_symmetry(self, roi: np.ndarray) -> float:
        """Check for symmetry (common in logos)"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Check horizontal symmetry
            if w > 2:
                left_half = gray[:, :w//2]
                right_half = cv2.flip(gray[:, w-w//2:w], 1)
                
                if left_half.shape == right_half.shape:
                    diff = cv2.absdiff(left_half, right_half)
                    h_symmetry = 1 - np.mean(diff) / 255.0
                else:
                    h_symmetry = 0
            else:
                h_symmetry = 0
            
            # Check vertical symmetry
            if h > 2:
                top_half = gray[:h//2, :]
                bottom_half = cv2.flip(gray[h-h//2:h, :], 0)
                
                if top_half.shape == bottom_half.shape:
                    diff = cv2.absdiff(top_half, bottom_half)
                    v_symmetry = 1 - np.mean(diff) / 255.0
                else:
                    v_symmetry = 0
            else:
                v_symmetry = 0
            
            return max(h_symmetry, v_symmetry)
            
        except Exception as e:
            self.logger.error(f"Symmetry check failed: {e}")
            return 0.0
    
    def _classify_text_type(self, roi: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """Classify type of text detected"""
        try:
            x, y, w, h = region
            aspect_ratio = w / h if h > 0 else 0
            
            # Simple classification based on size and position
            if aspect_ratio > 5:
                return "Subtitle"
            elif w < 100 and h < 50:
                return "Logo Text"
            elif y < 100:
                return "Title"
            elif y > roi.shape[0] - 100:
                return "Credits"
            else:
                return "Text"
                
        except Exception as e:
            self.logger.error(f"Text type classification failed: {e}")
            return "Unknown Text"
    
    def _classify_watermark_type(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """Classify type of watermark"""
        try:
            x, y, w, h = region
            height, width = frame.shape[:2]
            
            # Classify based on position and characteristics
            if x < width/3 and y < height/3:
                return "Corner Watermark"
            elif y < height/4:
                return "Top Banner"
            elif y > 3*height/4:
                return "Bottom Banner"
            else:
                return "Semi-transparent Overlay"
                
        except Exception as e:
            self.logger.error(f"Watermark type classification failed: {e}")
            return "Unknown Watermark"
    
    def _determine_face_severity(self, confidence: float, face_area: int, frame_area: int) -> str:
        """Determine severity of face detection"""
        try:
            size_ratio = face_area / frame_area
            
            if confidence > 0.9 and size_ratio > 0.05:
                return "high"
            elif confidence > 0.7 and size_ratio > 0.02:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Face severity determination failed: {e}")
            return "low"
    
    def _determine_text_severity(self, text_type: str, confidence: float) -> str:
        """Determine severity of text detection"""
        try:
            if text_type in ["Copyright Notice", "Credits"] and confidence > 0.8:
                return "high"
            elif confidence > 0.8:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Text severity determination failed: {e}")
            return "low"
    
    def _determine_watermark_severity(self, confidence: float, region: Tuple[int, int, int, int], 
                                    frame_shape: Tuple[int, int, int]) -> str:
        """Determine severity of watermark"""
        try:
            if confidence > 0.8:
                return "high"
            elif confidence > 0.6:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Watermark severity determination failed: {e}")
            return "low"
    
    def _determine_logo_severity(self, confidence: float, region: Tuple[int, int, int, int], 
                               frame_shape: Tuple[int, int, int]) -> str:
        """Determine severity of logo detection"""
        try:
            if confidence > 0.8:
                return "medium"
            elif confidence > 0.5:
                return "low"
            else:
                return "minimal"
                
        except Exception as e:
            self.logger.error(f"Logo severity determination failed: {e}")
            return "low"
    
    def _determine_blocking_severity(self, score: float) -> str:
        """Determine severity of blocking artifacts"""
        if score > 0.7:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _determine_banding_severity(self, score: float) -> str:
        """Determine severity of banding artifacts"""
        if score > 0.5:
            return "high"
        elif score > 0.3:
            return "medium"
        else:
            return "low"
    
    def _determine_ringing_severity(self, score: float) -> str:
        """Determine severity of ringing artifacts"""
        if score > 0.4:
            return "high"
        elif score > 0.2:
            return "medium"
        else:
            return "low"
    
    def _determine_noise_severity(self, score: float) -> str:
        """Determine severity of noise"""
        if score > 0.7:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _determine_blur_severity(self, score: float) -> str:
        """Determine severity of motion blur"""
        if score > 0.7:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_face_action_required(self, severity: str) -> str:
        """Get action required for face detection"""
        actions = {
            "high": "Review privacy compliance and obtain proper releases",
            "medium": "Ensure faces are properly licensed or blurred",
            "low": "Verify face usage is authorized"
        }
        return actions.get(severity, "Review face detection")
    
    def _get_text_action_required(self, text_type: str, severity: str) -> str:
        """Get action required for text detection"""
        if "copyright" in text_type.lower():
            return "Ensure copyright notices are accurate and current"
        elif "credits" in text_type.lower():
            return "Verify credit information is complete and correct"
        elif text_type == "Subtitle":
            return "Review subtitle timing and accuracy"
        else:
            return "Verify text usage rights and licensing"
    
    def _get_watermark_action_required(self, watermark_type: str, severity: str) -> str:
        """Get action required for watermark detection"""
        if severity == "high":
            return "Remove unauthorized watermarks or obtain proper licensing"
        else:
            return "Verify watermark ownership and licensing"
    
    def _get_logo_action_required(self, severity: str) -> str:
        """Get action required for logo detection"""
        return "Verify logo usage rights and proper attribution"
    
    def _calculate_artifact_score(self, artifacts: List[ArtifactDetection], total_frames: int) -> float:
        """Calculate overall artifact score (0-1, higher = better)"""
        try:
            if not artifacts:
                return 1.0  # Perfect score for no artifacts
            
            # Base score
            score = 1.0
            
            # Penalty for each artifact type
            for artifact in artifacts:
                penalty = 0
                
                # Severity-based penalties
                if artifact.severity == "high":
                    penalty = 0.15
                elif artifact.severity == "medium":
                    penalty = 0.08
                elif artifact.severity == "low":
                    penalty = 0.03
                else:
                    penalty = 0.01
                
                # Type-based adjustments
                if artifact.artifact_type == ArtifactType.FACE:
                    penalty *= 2.0  # Faces are more critical
                elif artifact.artifact_type == ArtifactType.COPYRIGHT_NOTICE:
                    penalty *= 1.5
                elif artifact.artifact_type in [ArtifactType.COMPRESSION_ARTIFACT, ArtifactType.NOISE]:
                    penalty *= 0.5  # Technical issues less critical
                
                score -= penalty
            
            # Normalize by number of frames (more frames = more opportunities for issues)
            frame_penalty = len(artifacts) / (total_frames * 10)  # Assume 10% artifact rate is bad
            score -= frame_penalty
            
            return max(0, min(1, score))
            
        except Exception as e:
            self.logger.error(f"Artifact score calculation failed: {e}")
            return 0.5
    
    def _categorize_artifacts(self, artifacts: List[ArtifactDetection]) -> Dict:
        """Categorize artifacts by type and severity"""
        try:
            summary = {
                'by_type': {},
                'by_severity': {'high': 0, 'medium': 0, 'low': 0, 'minimal': 0},
                'total_count': len(artifacts)
            }
            
            for artifact in artifacts:
                # By type
                artifact_type = artifact.artifact_type.value
                if artifact_type not in summary['by_type']:
                    summary['by_type'][artifact_type] = 0
                summary['by_type'][artifact_type] += 1
                
                # By severity
                severity = artifact.severity
                if severity in summary['by_severity']:
                    summary['by_severity'][severity] += 1
                else:
                    summary['by_severity']['low'] += 1
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Artifact categorization failed: {e}")
            return {}
    
    def _generate_artifact_recommendations(self, artifacts: List[ArtifactDetection]) -> List[str]:
        """Generate recommendations based on detected artifacts"""
        try:
            recommendations = []
            
            # Privacy recommendations
            faces = [a for a in artifacts if a.artifact_type == ArtifactType.FACE]
            if faces:
                recommendations.append("Review privacy compliance for detected faces")
                recommendations.append("Obtain proper talent releases if faces are identifiable")
            
            # Copyright recommendations
            text_items = [a for a in artifacts if a.artifact_type == ArtifactType.TEXT]
            if text_items:
                recommendations.append("Verify all text content is properly licensed")
                recommendations.append("Ensure copyright notices are current and accurate")
            
            # Technical recommendations
            compression_artifacts = [a for a in artifacts if a.artifact_type == ArtifactType.COMPRESSION_ARTIFACT]
            if compression_artifacts:
                recommendations.append("Increase compression quality or bitrate")
                recommendations.append("Consider using different encoding settings")
            
            # Branding recommendations
            watermarks = [a for a in artifacts if a.artifact_type == ArtifactType.WATERMARK]
            if watermarks:
                recommendations.append("Verify watermark ownership and licensing")
                recommendations.append("Remove unauthorized watermarks if present")
            
            if not recommendations:
                recommendations.append("No significant artifacts detected")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return ["Artifact analysis failed - manual review recommended"]
    
    def _assess_privacy_concerns(self, artifacts: List[ArtifactDetection]) -> Dict:
        """Assess privacy-related concerns"""
        try:
            faces = [a for a in artifacts if a.artifact_type == ArtifactType.FACE]
            
            concerns = {
                'faces_detected': len(faces),
                'high_risk_faces': len([f for f in faces if f.severity == "high"]),
                'privacy_score': 1.0 - (len(faces) * 0.1),  # Lower score with more faces
                'action_required': "Review privacy compliance"
            }
            
            if len(faces) == 0:
                concerns['privacy_score'] = 1.0
                concerns['action_required'] = "No privacy concerns detected"
            
            return concerns
            
        except Exception as e:
            self.logger.error(f"Privacy assessment failed: {e}")
            return {'error': str(e)}
    
    def _assess_legal_concerns(self, artifacts: List[ArtifactDetection]) -> Dict:
        """Assess legal-related concerns"""
        try:
            concerns = {
                'copyright_text': len([a for a in artifacts if a.artifact_type == ArtifactType.COPYRIGHT_NOTICE]),
                'watermarks': len([a for a in artifacts if a.artifact_type == ArtifactType.WATERMARK]),
                'logos': len([a for a in artifacts if a.artifact_type == ArtifactType.LOGO]),
                'legal_risk_score': 0.5,  # Neutral baseline
                'requires_review': False
            }
            
            # Calculate legal risk based on detected items
            risk_factors = 0
            if concerns['copyright_text'] > 0:
                risk_factors += 0.2
            if concerns['watermarks'] > 0:
                risk_factors += 0.3
            if concerns['logos'] > 0:
                risk_factors += 0.2
            
            concerns['legal_risk_score'] = 1.0 - risk_factors
            concerns['requires_review'] = risk_factors > 0.5
            
            return concerns
            
        except Exception as e:
            self.logger.error(f"Legal assessment failed: {e}")
            return {'error': str(e)}
    
    def _initialize_watermark_patterns(self) -> List[str]:
        """Initialize watermark detection patterns"""
        return [
            r"©\s*\d{4}",  # Copyright symbol with year
            r"All rights reserved",
            r"Proprietary",
            r"Confidential",
            r"WATERMARK",
            r"©",
            r"™",
            r"®"
        ]
    
    def _initialize_branding_keywords(self) -> List[str]:
        """Initialize branding keywords for detection"""
        return [
            "YouTube", "Netflix", "Amazon", "Disney", "Warner", "Universal",
            "Sony", "Paramount", "Fox", "HBO", "BBC", "CNN",
            "Adobe", "Microsoft", "Apple", "Google", "Facebook",
            "TM", "®", "™"
        ]
    
    def _register_face_detection(self, frame_idx: int, bounding_box: Tuple[int, int, int, int], confidence: float):
        """Register face detection for tracking"""
        self.face_registry.append({
            'frame_idx': frame_idx,
            'bounding_box': bounding_box,
            'confidence': confidence,
            'timestamp': str(np.datetime64('now'))
        })
    
    def _register_text_occurrence(self, frame_idx: int, region: Tuple[int, int, int, int], text_type: str):
        """Register text occurrence"""
        self.text_occurrences.append({
            'frame_idx': frame_idx,
            'region': region,
            'text_type': text_type,
            'timestamp': str(np.datetime64('now'))
        })
    
    def _store_artifact_results(self, results: Dict, artifacts: List[ArtifactDetection]):
        """Store artifact analysis results"""
        self.artifact_history.append({
            'timestamp': str(np.datetime64('now')),
            'results': results,
            'artifact_count': len(artifacts)
        })
    
    def _default_config(self) -> Dict:
        """Default configuration for artifact detector"""
        return {
            'detect_faces': True,
            'detect_text': True,
            'detect_watermarks': True,
            'detect_logos': True,
            'detect_compression_artifacts': True,
            'min_confidence': 0.7,
            'face_confidence_threshold': 0.8,
            'text_confidence_threshold': 0.6,
            'sample_interval': 30,
            'max_faces_per_frame': 10,
            'face_blur_threshold': 100
        }


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Create sample frames for testing
    sample_frames = []
    for i in range(5):
        # Create frames with some artifacts
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        sample_frames.append(frame)
    
    detector = ArtifactDetector()
    results = detector.detect_artifacts(sample_frames)
    
    print("Artifact Detection Results:")
    print(f"Overall Score: {results['score']:.1f}/100")
    print(f"Total Artifacts: {results['total_artifacts']}")
    print(f"Faces Detected: {results['faces_detected']}")
    print(f"Text Detected: {results['text_detected']}")
    print(f"Watermarks Detected: {results['watermarks_detected']}")
    print(f"Recommendations: {results['recommendations']}")
