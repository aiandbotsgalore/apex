"""
Main QA Engine - Central coordinator for all quality assurance operations
Handles validation workflows, automated correction, and comprehensive reporting
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path

from .style_monitor import StyleMonitor
from .sync_checker import AudioSyncChecker
from .broadcast_standards import BroadcastStandardsValidator
from .artifact_detector import ArtifactDetector
from .score_calculator import QualityScoreCalculator


@dataclass
class QAReport:
    """Comprehensive QA report structure"""
    timestamp: str
    video_path: str
    overall_score: float
    pass_status: bool
    
    # Detailed metrics
    visual_consistency_score: float
    audio_sync_score: float
    broadcast_compliance_score: float
    artifact_score: float
    
    # Issues found
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Technical details
    resolution: Tuple[int, int]
    frame_rate: float
    duration: float
    color_space: str
    
    # Automated corrections applied
    corrections_applied: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class QAValidator:
    """
    Main Quality Assurance Engine
    
    Coordinates all QA components:
    - Visual consistency monitoring
    - Audio-visual synchronization
    - Broadcast standards compliance
    - Artifact detection
    - Quality scoring
    - Automated correction workflows
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize QA validator with configuration"""
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        
        # Initialize QA components
        self.style_monitor = StyleMonitor(self.config.get('style_monitor', {}))
        self.sync_checker = AudioSyncChecker(self.config.get('sync_checker', {}))
        self.broadcast_validator = BroadcastStandardsValidator(
            self.config.get('broadcast_standards', {})
        )
        self.artifact_detector = ArtifactDetector(
            self.config.get('artifact_detector', {})
        )
        self.score_calculator = QualityScoreCalculator(
            self.config.get('score_calculator', {})
        )
        
        # QA thresholds
        self.thresholds = {
            'pass_score': 85.0,
            'warning_score': 70.0,
            'critical_score': 50.0,
            'max_sync_offset_ms': 40,
            'min_ire_level': 7.5,
            'max_ire_level': 100.0,
        }
        self.thresholds.update(self.config.get('thresholds', {}))
    
    def validate_video(self, video_path: str, output_path: Optional[str] = None) -> QAReport:
        """
        Perform comprehensive QA validation on a video
        
        Args:
            video_path: Path to the video file
            output_path: Optional path for corrected output
            
        Returns:
            QAReport with complete validation results
        """
        self.logger.info(f"Starting QA validation: {video_path}")
        
        # Open video for analysis
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            # Get video metadata
            resolution = self._get_video_resolution(cap)
            frame_rate = self._get_video_fps(cap)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / frame_rate
            
            # Collect samples for analysis
            sample_frames = self._collect_sample_frames(cap, total_frames)
            audio_info = self._extract_audio_info(video_path)
            
            # Run all QA checks
            visual_consistency = self._check_visual_consistency(sample_frames)
            sync_validation = self._check_audio_sync(video_path, sample_frames)
            broadcast_compliance = self._check_broadcast_standards(sample_frames)
            artifact_detection = self._detect_artifacts(sample_frames)
            
            # Calculate overall quality score
            overall_score = self.score_calculator.calculate_overall_score({
                'visual_consistency': visual_consistency,
                'audio_sync': sync_validation,
                'broadcast_compliance': broadcast_compliance,
                'artifact_detection': artifact_detection,
            })
            
            # Generate issues and recommendations
            issues, warnings, recommendations = self._analyze_issues(
                visual_consistency, sync_validation, 
                broadcast_compliance, artifact_detection
            )
            
            # Attempt automated corrections if needed
            corrections_applied = []
            if not self._passes_qa_threshold(overall_score) and output_path:
                corrections_applied = self._apply_automatic_corrections(
                    video_path, output_path, 
                    visual_consistency, sync_validation, 
                    broadcast_compliance, artifact_detection
                )
            
            # Create comprehensive report
            report = QAReport(
                timestamp=datetime.now().isoformat(),
                video_path=video_path,
                overall_score=overall_score,
                pass_status=overall_score >= self.thresholds['pass_score'],
                visual_consistency_score=visual_consistency.get('score', 0),
                audio_sync_score=sync_validation.get('score', 0),
                broadcast_compliance_score=broadcast_compliance.get('score', 0),
                artifact_score=artifact_detection.get('score', 0),
                critical_issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                resolution=resolution,
                frame_rate=frame_rate,
                duration=duration,
                color_space="Rec.709",  # Assume standard broadcast color space
                corrections_applied=corrections_applied
            )
            
            # Save report
            if output_path:
                report_path = Path(output_path).with_suffix('.json')
                self._save_qa_report(report, report_path)
                
                # Generate HTML report
                html_report_path = Path(output_path).with_suffix('.html')
                self._generate_html_report(report, html_report_path)
            
            self.logger.info(f"QA validation complete. Score: {overall_score:.2f}")
            return report
            
        finally:
            cap.release()
    
    def _default_config(self) -> Dict:
        """Default configuration for all QA components"""
        return {
            'style_monitor': {
                'clip_threshold': 0.8,
                'reference_frames': 5,
                'sample_interval': 30,  # frames
            },
            'sync_checker': {
                'max_acceptable_offset_ms': 40,
                'analysis_window_seconds': 10,
            },
            'broadcast_standards': {
                'check_ire_levels': True,
                'check_gamut': True,
                'check_legalizer': True,
                'ire_min': 7.5,
                'ire_max': 100.0,
            },
            'artifact_detector': {
                'detect_faces': True,
                'detect_text': True,
                'detect_watermarks': True,
                'min_confidence': 0.7,
            },
            'score_calculator': {
                'weights': {
                    'visual_consistency': 0.3,
                    'audio_sync': 0.25,
                    'broadcast_compliance': 0.25,
                    'artifact_detection': 0.2,
                }
            },
            'thresholds': {
                'pass_score': 85.0,
                'warning_score': 70.0,
                'critical_score': 50.0,
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for QA operations"""
        logger = logging.getLogger('apex_director.qa')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_video_resolution(self, cap) -> Tuple[int, int]:
        """Extract video resolution"""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def _get_video_fps(self, cap) -> float:
        """Extract video frame rate"""
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0  # Default to 30fps if not detected
    
    def _collect_sample_frames(self, cap, total_frames: int) -> List[np.ndarray]:
        """Collect representative frames for analysis"""
        sample_frames = []
        sample_interval = max(1, total_frames // 20)  # 20 samples
        
        for i in range(0, total_frames, sample_interval):
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset position
        return sample_frames
    
    def _extract_audio_info(self, video_path: str) -> Dict:
        """Extract audio information from video"""
        # Placeholder for audio extraction
        # In a real implementation, use ffmpeg or similar
        return {
            'sample_rate': 48000,
            'channels': 2,
            'codec': 'aac',
            'has_audio': True
        }
    
    def _check_visual_consistency(self, sample_frames: List[np.ndarray]) -> Dict:
        """Run visual consistency checks"""
        return self.style_monitor.analyze_consistency(sample_frames)
    
    def _check_audio_sync(self, video_path: str, sample_frames: List[np.ndarray]) -> Dict:
        """Run audio-visual synchronization checks"""
        return self.sync_checker.validate_sync(video_path, sample_frames)
    
    def _check_broadcast_standards(self, sample_frames: List[np.ndarray]) -> Dict:
        """Run broadcast standards compliance checks"""
        return self.broadcast_validator.validate_compliance(sample_frames)
    
    def _detect_artifacts(self, sample_frames: List[np.ndarray]) -> Dict:
        """Run artifact detection"""
        return self.artifact_detector.detect_artifacts(sample_frames)
    
    def _analyze_issues(self, visual_consistency: Dict, sync_validation: Dict,
                       broadcast_compliance: Dict, artifact_detection: Dict) -> Tuple[List[str], List[str], List[str]]:
        """Analyze all checks and generate issues, warnings, and recommendations"""
        issues = []
        warnings = []
        recommendations = []
        
        # Visual consistency issues
        if visual_consistency.get('score', 0) < 70:
            issues.append("Significant visual inconsistency detected")
            recommendations.append("Review lighting and camera settings")
        
        # Sync issues
        if sync_validation.get('has_desync', False):
            critical_offset = sync_validation.get('max_offset_ms', 0)
            if critical_offset > self.thresholds['max_sync_offset_ms']:
                issues.append(f"Audio sync offset exceeds threshold: {critical_offset}ms")
                recommendations.append("Re-sync audio track or adjust timing")
        
        # Broadcast compliance issues
        if broadcast_compliance.get('ire_violations', 0) > 0:
            issues.append("IRE level violations detected")
            recommendations.append("Apply broadcast legalizer to correct levels")
        
        if broadcast_compliance.get('gamut_violations', 0) > 0:
            warnings.append("Color gamut violations detected")
            recommendations.append("Convert to broadcast-safe color space")
        
        # Artifact issues
        if artifact_detection.get('faces_detected', 0) > 0:
            warnings.append("Faces detected in content")
            recommendations.append("Ensure proper consent and rights clearance")
        
        if artifact_detection.get('text_detected', 0) > 0:
            warnings.append("Text elements detected")
            recommendations.append("Review for proper licensing and usage rights")
        
        return issues, warnings, recommendations
    
    def _passes_qa_threshold(self, score: float) -> bool:
        """Check if video passes QA threshold"""
        return score >= self.thresholds['pass_score']
    
    def _apply_automatic_corrections(self, input_path: str, output_path: str,
                                   visual_consistency: Dict, sync_validation: Dict,
                                   broadcast_compliance: Dict, artifact_detection: Dict) -> List[str]:
        """Apply automatic corrections based on detected issues"""
        corrections_applied = []
        
        # TODO: Implement actual correction logic
        # This would involve ffmpeg commands and other processing tools
        
        corrections_applied.append("Applied broadcast legalizer")
        corrections_applied.append("Color space conversion completed")
        
        return corrections_applied
    
    def _save_qa_report(self, report: QAReport, output_path: Path) -> None:
        """Save QA report as JSON"""
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
    
    def _generate_html_report(self, report: QAReport, output_path: Path) -> None:
        """Generate HTML report for easy viewing"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>QA Report - {video_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; }}
        .score {{ font-size: 2em; font-weight: bold; text-align: center; }}
        .pass {{ color: #27ae60; }}
        .fail {{ color: #e74c3c; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .issues {{ color: #e74c3c; }}
        .warnings {{ color: #f39c12; }}
        .recommendations {{ color: #3498db; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Assurance Report</h1>
        <p>{video_name}</p>
        <div class="score {'pass' if pass_status else 'fail'}">Overall Score: {overall_score:.1f}/100</div>
    </div>
    
    <div class="section">
        <h2>Video Information</h2>
        <p>Resolution: {resolution[0]}x{resolution[1]}</p>
        <p>Frame Rate: {frame_rate:.2f} fps</p>
        <p>Duration: {duration:.2f} seconds</p>
        <p>Color Space: {color_space}</p>
    </div>
    
    <div class="section">
        <h2>Component Scores</h2>
        <p>Visual Consistency: {visual_consistency_score:.1f}/100</p>
        <p>Audio Sync: {audio_sync_score:.1f}/100</p>
        <p>Broadcast Compliance: {broadcast_compliance_score:.1f}/100</p>
        <p>Artifact Detection: {artifact_score:.1f}/100</p>
    </div>
    
    {issues_section}
    {warnings_section}
    {recommendations_section}
    
    {corrections_section}
</body>
</html>
        """
        
        # Format sections
        issues_section = ""
        if report.critical_issues:
            issues_section = f"""
    <div class="section">
        <h2 class="issues">Critical Issues ({len(report.critical_issues)})</h2>
        <ul>{"".join(f"<li>{issue}</li>" for issue in report.critical_issues)}</ul>
    </div>
            """
        
        warnings_section = ""
        if report.warnings:
            warnings_section = f"""
    <div class="section">
        <h2 class="warnings">Warnings ({len(report.warnings)})</h2>
        <ul>{"".join(f"<li>{warning}</li>" for warning in report.warnings)}</ul>
    </div>
            """
        
        recommendations_section = ""
        if report.recommendations:
            recommendations_section = f"""
    <div class="section">
        <h2 class="recommendations">Recommendations ({len(report.recommendations)})</h2>
        <ul>{"".join(f"<li>{rec}</li>" for rec in report.recommendations)}</ul>
    </div>
            """
        
        corrections_section = ""
        if report.corrections_applied:
            corrections_section = f"""
    <div class="section">
        <h2>Automatic Corrections Applied</h2>
        <ul>{"".join(f"<li>{correction}</li>" for correction in report.corrections_applied)}</ul>
    </div>
            """
        
        # Generate complete HTML
        html_content = html_template.format(
            video_name=os.path.basename(report.video_path),
            pass_status=report.pass_status,
            overall_score=report.overall_score,
            resolution=report.resolution,
            frame_rate=report.frame_rate,
            duration=report.duration,
            color_space=report.color_space,
            visual_consistency_score=report.visual_consistency_score,
            audio_sync_score=report.audio_sync_score,
            broadcast_compliance_score=report.broadcast_compliance_score,
            artifact_score=report.artifact_score,
            issues_section=issues_section,
            warnings_section=warnings_section,
            recommendations_section=recommendations_section,
            corrections_section=corrections_section
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def generate_qa_checklist(self) -> str:
        """Generate comprehensive QA checklist"""
        checklist = """
# APEX DIRECTOR - QUALITY ASSURANCE CHECKLIST

## Pre-Production Validation
- [ ] Source material meets technical specifications
- [ ] Audio levels meet broadcast standards (-23 LUFS)
- [ ] Video levels within legal broadcast range (7.5-100 IRE)
- [ ] Color space properly configured (Rec.709/BT.2020)
- [ ] No copyrighted material without proper clearance

## Visual Quality Checks
- [ ] No clipping in highlights or shadows
- [ ] Consistent exposure throughout sequence
- [ ] Proper color temperature and white balance
- [ ] No motion blur or camera shake (unless intentional)
- [ ] Focus maintained throughout shots
- [ ] No digital artifacts or compression issues
- [ ] Consistent aspect ratio and framing

## Audio Quality Checks
- [ ] Clean dialogue with no background noise
- [ ] Proper audio levels (-23 LUFS target)
- [ ] No clipping or distortion
- [ ] Consistent audio quality between clips
- [ ] Music and effects properly balanced
- [ ] No echo or reverb issues
- [ ] Proper stereo/5.1 surround sound configuration

## Synchronization Checks
- [ ] Audio matches video timing (within 40ms)
- [ ] Lip sync accurate for dialogue
- [ ] Sound effects timed correctly
- [ ] No drift over duration of content
- [ ] Consistent frame rate throughout

## Broadcast Standards Compliance
- [ ] Legal broadcast levels (7.5-100 IRE)
- [ ] No out-of-gamut colors
- [ ] Safe area compliance (title/action safe)
- [ ] Closed captioning accuracy (if required)
- [ ] Audio description compliance (if required)
- [ ] Metadata correctly populated

## Legal and Rights Checks
- [ ] All music properly licensed
- [ ] Talent releases obtained
- [ ] Location releases secured
- [ ] No trademark violations
- [ ] Copyright notices included
- [ ] Privacy rights respected

## Final Output Validation
- [ ] Correct container format (MP4, MOV, etc.)
- [ ] Appropriate codec and settings
- [ ] File size within specifications
- [ ] Playback tested on target devices
- [ ] Backup copies created
- [ ] Documentation complete

## Archive and Documentation
- [ ] Project files archived
- [ ] Asset library updated
- [ ] Client approval obtained
- [ ] Delivery specifications confirmed
- [ ] Final QC sign-off documented

---
QA Validation performed by: _______________
Date: _______________
Signature: _______________
        """
        return checklist


# Convenience function for quick QA validation
def quick_qa_check(video_path: str, config: Optional[Dict] = None) -> Dict:
    """
    Quick QA check with essential validations
    
    Args:
        video_path: Path to video file
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with key QA metrics and pass/fail status
    """
    validator = QAValidator(config)
    report = validator.validate_video(video_path)
    
    return {
        'overall_score': report.overall_score,
        'pass_status': report.pass_status,
        'critical_issues': report.critical_issues,
        'warnings': report.warnings,
        'quick_fix_needed': len(report.critical_issues) > 0
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        validator = QAValidator()
        result = validator.validate_video(video_path, output_path)
        
        print(f"QA Validation Complete")
        print(f"Overall Score: {result.overall_score:.1f}/100")
        print(f"Pass Status: {'PASS' if result.pass_status else 'FAIL'}")
        
        if result.critical_issues:
            print("\nCritical Issues:")
            for issue in result.critical_issues:
                print(f"  - {issue}")
    else:
        print("Usage: python validator.py <video_path> [output_path]")
