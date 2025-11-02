"""
APEX DIRECTOR QA Framework Example Usage
Demonstrates comprehensive QA workflow
"""

import os
import sys
import logging
from pathlib import Path

# Add the apex_director module to Python path
sys.path.insert(0, str(Path(__file__).parent))

from apex_director.qa import (
    QAValidator, StyleMonitor, AudioSyncChecker, 
    BroadcastStandardsValidator, ArtifactDetector,
    QualityScoreCalculator, QualityReportGenerator
)


def setup_logging():
    """Sets up logging for QA operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('qa_analysis.log')
        ]
    )


def create_sample_video_data():
    """Creates sample video data for demonstration purposes.

    Returns:
        A list of sample video frames.
    """
    import numpy as np
    import cv2
    
    print("Creating sample video frames for demonstration...")
    
    # Create sample frames
    sample_frames = []
    for i in range(30):  # 30 sample frames
        # Create a frame with some variation
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(frame, (320, 240), 50, (255, 0, 0), 3)
        
        sample_frames.append(frame)
    
    # Save a sample frame for reference
    cv2.imwrite('sample_frame.jpg', sample_frames[0])
    print(f"Created {len(sample_frames)} sample frames")
    
    return sample_frames


def demonstrate_qa_workflow():
    """Demonstrates the complete QA workflow."""
    print("=" * 60)
    print("APEX DIRECTOR QA FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    
    # Setup
    setup_logging()
    sample_frames = create_sample_video_data()
    
    # Create QA configuration
    qa_config = {
        'style_monitor': {
            'clip_threshold': 0.8,
            'reference_frames': 5,
            'sample_interval': 30,
        },
        'sync_checker': {
            'max_acceptable_offset_ms': 40,
            'analysis_window_seconds': 10,
        },
        'broadcast_standards': {
            'check_ire_levels': True,
            'check_gamut': True,
            'check_safe_area': True,
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
    
    print("\n1. INITIALIZING QA COMPONENTS")
    print("-" * 40)
    
    # Initialize all QA components
    try:
        style_monitor = StyleMonitor(qa_config.get('style_monitor', {}))
        print("✓ Style Monitor initialized")
        
        sync_checker = AudioSyncChecker(qa_config.get('sync_checker', {}))
        print("✓ Audio Sync Checker initialized")
        
        broadcast_validator = BroadcastStandardsValidator(qa_config.get('broadcast_standards', {}))
        print("✓ Broadcast Standards Validator initialized")
        
        artifact_detector = ArtifactDetector(qa_config.get('artifact_detector', {}))
        print("✓ Artifact Detector initialized")
        
        score_calculator = QualityScoreCalculator(qa_config.get('score_calculator', {}))
        print("✓ Quality Score Calculator initialized")
        
        report_generator = QualityReportGenerator()
        print("✓ Report Generator initialized")
        
        validator = QAValidator(qa_config)
        print("✓ Main QA Validator initialized")
        
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")
        return
    
    print("\n2. RUNNING INDIVIDUAL QA COMPONENTS")
    print("-" * 40)
    
    # Demonstrate individual component analysis
    try:
        # Style consistency analysis
        print("\nVisual Style Consistency Analysis:")
        visual_consistency = style_monitor.analyze_consistency(sample_frames)
        print(f"  Overall Style Score: {visual_consistency['score']:.1f}/100")
        print(f"  Style Drift Detected: {visual_consistency.get('drift_detected', False)}")
        print(f"  Recommendations: {len(visual_consistency.get('recommendations', []))}")
        
        # Audio sync analysis (simulated)
        print("\nAudio-Video Synchronization Analysis:")
        sync_results = sync_checker.validate_sync("demo_video.mp4", sample_frames)
        print(f"  Sync Score: {sync_results['score']:.1f}/100")
        print(f"  Time Offset: {sync_results.get('time_offset_ms', 0):.1f}ms")
        print(f"  Has Desync: {sync_results.get('has_desync', False)}")
        
        # Broadcast standards analysis
        print("\nBroadcast Standards Compliance Analysis:")
        broadcast_compliance = broadcast_validator.validate_compliance(sample_frames)
        print(f"  Compliance Score: {broadcast_compliance['score']:.1f}/100")
        print(f"  IRE Levels Compliant: {broadcast_compliance.get('ire_analysis', {}).get('compliant', True)}")
        print(f"  Gamut Compliant: {broadcast_compliance.get('gamut_analysis', {}).get('compliant', True)}")
        
        # Artifact detection
        print("\nArtifact Detection Analysis:")
        artifact_detection = artifact_detector.detect_artifacts(sample_frames)
        print(f"  Artifact Score: {artifact_detection['score']:.1f}/100")
        print(f"  Total Artifacts: {artifact_detection['total_artifacts']}")
        print(f"  Faces Detected: {artifact_detection['faces_detected']}")
        print(f"  Text Elements: {artifact_detection['text_detected']}")
        
    except Exception as e:
        print(f"✗ Individual component analysis failed: {e}")
    
    print("\n3. CALCULATING COMPREHENSIVE QUALITY SCORE")
    print("-" * 40)
    
    # Combine all results for comprehensive scoring
    try:
        qa_results = {
            'visual_consistency': visual_consistency,
            'audio_sync': sync_results,
            'broadcast_compliance': broadcast_compliance,
            'artifact_detection': artifact_detection
        }
        
        # Calculate overall quality score
        overall_score = score_calculator.calculate_overall_score(qa_results)
        print(f"Overall Quality Score: {overall_score:.1f}/100")
        
        # Calculate detailed breakdown
        detailed_breakdown = score_calculator.calculate_detailed_score_breakdown(qa_results)
        print(f"Quality Level: {detailed_breakdown.overall_score:.1f}")
        print(f"Confidence Level: {detailed_breakdown.confidence_level:.1%}")
        print(f"Quality Trend: {detailed_breakdown.score_trend}")
        
        print("\nComponent Scores:")
        for component, score in detailed_breakdown.component_scores.items():
            print(f"  {component.replace('_', ' ').title()}: {score:.1f}/100")
        
    except Exception as e:
        print(f"✗ Quality scoring failed: {e}")
    
    print("\n4. GENERATING QA REPORT")
    print("-" * 40)
    
    # Generate comprehensive QA report
    try:
        # Create a mock QA report for demonstration
        mock_qa_report = type('MockQAReport', (), {
            'timestamp': '2024-01-15T10:30:00',
            'video_path': 'demo_video.mp4',
            'overall_score': overall_score,
            'pass_status': overall_score >= 85.0,
            'visual_consistency_score': visual_consistency['score'],
            'audio_sync_score': sync_results['score'],
            'broadcast_compliance_score': broadcast_compliance['score'],
            'artifact_score': artifact_detection['score'],
            'critical_issues': detailed_breakdown.critical_issues,
            'warnings': ['Sample warning for demonstration'],
            'recommendations': detailed_breakdown.improvement_opportunities,
            'resolution': (1920, 1080),
            'frame_rate': 30.0,
            'duration': 120.0,
            'color_space': 'Rec.709',
            'corrections_applied': ['Applied broadcast legalizer']
        })()
        
        # Generate reports in multiple formats
        print("Generating comprehensive reports...")
        report_paths = report_generator.generate_comprehensive_report(
            mock_qa_report, detailed_breakdown, "demo_qa_reports"
        )
        
        print("Generated reports:")
        for format_type, path in report_paths.items():
            if path:
                print(f"  {format_type.upper()}: {path}")
        
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
    
    print("\n5. QA WORKFLOW COMPLETION")
    print("-" * 40)
    
    # Final summary
    print(f"✓ QA Analysis Complete")
    print(f"✓ Quality Score: {overall_score:.1f}/100")
    print(f"✓ Delivery Status: {'APPROVED' if overall_score >= 85 else 'REQUIRES REVISION'}")
    print(f"✓ Reports Generated: {len([p for p in report_paths.values() if p])}")
    
    print("\n6. ADDITIONAL QA FEATURES")
    print("-" * 40)
    
    # Demonstrate additional features
    try:
        # Generate QA checklist
        checklist = validator.generate_qa_checklist()
        print(f"✓ QA Checklist generated ({len(checklist)} characters)")
        
        # Quick QA check
        quick_result = validator.quick_qa_check("demo_video.mp4", qa_config)
        print(f"✓ Quick QA Check: {quick_result['overall_score']:.1f}/100")
        print(f"  Pass Status: {quick_result['pass_status']}")
        print(f"  Critical Issues: {len(quick_result['critical_issues'])}")
        
        # Broadcast compliance report
        compliance_report = broadcast_validator.generate_broadcast_compliance_report(broadcast_compliance)
        print(f"✓ Broadcast Compliance Report generated ({len(compliance_report)} characters)")
        
        # Quality report
        quality_report = score_calculator.generate_quality_report(qa_results)
        print(f"✓ Quality Report generated ({len(quality_report)} characters)")
        
    except Exception as e:
        print(f"✗ Additional features demonstration failed: {e}")
    
    print("\n" + "=" * 60)
    print("QA FRAMEWORK DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    print("\nNext Steps:")
    print("1. Review generated reports in demo_qa_reports/ directory")
    print("2. Implement QA pipeline in your video production workflow")
    print("3. Customize QA configuration for your specific requirements")
    print("4. Set up automated QA processing for batch operations")
    print("5. Integrate with your existing video editing tools")


def demonstrate_individual_components():
    """Demonstrates the individual QA components in isolation."""
    print("\n" + "=" * 60)
    print("INDIVIDUAL COMPONENT DEMONSTRATION")
    print("=" * 60)
    
    sample_frames = create_sample_video_data()
    
    # Style Monitor Demo
    print("\n1. STYLE MONITOR DEMO")
    print("-" * 30)
    try:
        monitor = StyleMonitor()
        result = monitor.analyze_consistency(sample_frames)
        print(f"CLIP Similarity Score: {result.get('clip_similarity_score', 0):.1f}")
        print(f"Color Consistency: {result.get('color_consistency_score', 0):.1f}")
        print(f"Style Drift: {'Detected' if result.get('drift_detected') else 'Not Detected'}")
    except Exception as e:
        print(f"Style Monitor Demo Failed: {e}")
    
    # Sync Checker Demo
    print("\n2. AUDIO SYNC CHECKER DEMO")
    print("-" * 30)
    try:
        checker = AudioSyncChecker()
        result = checker.validate_sync("test_video.mp4", sample_frames)
        print(f"Sync Score: {result.get('score', 0):.1f}")
        print(f"Time Offset: {result.get('time_offset_ms', 0):.1f}ms")
        print(f"Has Audio: {result.get('has_audio', False)}")
    except Exception as e:
        print(f"Sync Checker Demo Failed: {e}")
    
    # Broadcast Standards Demo
    print("\n3. BROADCAST STANDARDS DEMO")
    print("-" * 30)
    try:
        validator = BroadcastStandardsValidator()
        result = validator.validate_compliance(sample_frames)
        print(f"Compliance Score: {result.get('score', 0):.1f}")
        print(f"IRE Compliant: {result.get('ire_analysis', {}).get('compliant', False)}")
        print(f"Needs Legalizer: {result.get('needs_legalization', False)}")
    except Exception as e:
        print(f"Broadcast Standards Demo Failed: {e}")
    
    # Artifact Detector Demo
    print("\n4. ARTIFACT DETECTOR DEMO")
    print("-" * 30)
    try:
        detector = ArtifactDetector()
        result = detector.detect_artifacts(sample_frames)
        print(f"Artifact Score: {result.get('score', 0):.1f}")
        print(f"Total Artifacts: {result.get('total_artifacts', 0)}")
        print(f"Privacy Score: {result.get('privacy_concerns', {}).get('privacy_score', 1.0):.2f}")
    except Exception as e:
        print(f"Artifact Detector Demo Failed: {e}")
    
    # Score Calculator Demo
    print("\n5. SCORE CALCULATOR DEMO")
    print("-" * 30)
    try:
        calculator = QualityScoreCalculator()
        
        # Create sample QA results
        sample_results = {
            'visual_consistency': {'score': 85},
            'audio_sync': {'score': 90},
            'broadcast_compliance': {'score': 88},
            'artifact_detection': {'score': 92}
        }
        
        overall_score = calculator.calculate_overall_score(sample_results)
        print(f"Overall Score: {overall_score:.1f}")
        
        breakdown = calculator.calculate_detailed_score_breakdown(sample_results)
        print(f"Quality Level: {breakdown.overall_score:.1f}")
        print(f"Confidence: {breakdown.confidence_level:.1%}")
    except Exception as e:
        print(f"Score Calculator Demo Failed: {e}")


def main():
    """The main function for the demonstration."""
    print("APEX DIRECTOR QA FRAMEWORK")
    print("Comprehensive Quality Assurance System")
    print("Version 1.0.0")
    print("=" * 60)
    
    try:
        # Run comprehensive workflow demonstration
        demonstrate_qa_workflow()
        
        # Run individual component demonstrations
        demonstrate_individual_components()
        
        print("\n✓ All demonstrations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\n\nDemonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
