"""
APEX DIRECTOR - Quality Assurance Framework

Comprehensive QA system for professional video production:
- Visual consistency monitoring with CLIP similarity
- Audio-visual synchronization validation
- Broadcast standards compliance checking
- Artifact detection and identification
- Automated quality scoring and reporting

Modules:
- validator: Main QA engine and workflow coordinator
- style_monitor: Visual consistency tracking and style drift detection
- sync_checker: Frame-accurate audio-visual synchronization
- broadcast_standards: Professional broadcast compliance validation
- artifact_detector: Quality issue identification and classification
- score_calculator: Comprehensive quality metrics and scoring
"""

from .validator import QAValidator, QAReport, quick_qa_check
from .style_monitor import StyleMonitor, StyleMetrics
from .sync_checker import AudioSyncChecker, SyncMetrics
from .broadcast_standards import BroadcastStandardsValidator, BroadcastCompliance, BroadcastStandard, LegalizerMode
from .artifact_detector import ArtifactDetector, ArtifactDetection, ArtifactType
from .score_calculator import QualityScoreCalculator, QualityScoreBreakdown, QualityMetric
from .metrics_collector import MetricsCollector, QualityMetrics
from .metrics_report import MetricsReport, ReportConfig
from .metrics_viz import MetricsVisualizer

__version__ = "1.0.0"
__author__ = "APEX DIRECTOR Team"

# QA Framework configuration
QA_CONFIG = {
    "version": __version__,
    "modules": [
        "validator",
        "style_monitor", 
        "sync_checker",
        "broadcast_standards",
        "artifact_detector",
        "score_calculator",
        "metrics_collector",
        "metrics_report",
        "metrics_viz"
    ],
    "supported_standards": [
        "Rec.709/BT.709",
        "Rec.2020/BT.2020", 
        "NTSC",
        "PAL",
        "SECAM"
    ],
    "features": [
        "Visual style drift detection",
        "Frame-accurate sync validation",
        "IRE level compliance checking",
        "Color gamut validation",
        "Safe area verification",
        "Face detection and privacy assessment",
        "Text and watermark identification",
        "Compression artifact detection",
        "Automated quality scoring",
        "Broadcast legalizer application",
        "Comprehensive metrics collection",
        "Quality report generation",
        "Interactive dashboard visualization",
        "Performance trend analysis"
    ]
}

def create_qa_validator(config=None):
    """Create and return a QA validator instance with optional configuration"""
    return QAValidator(config)

def quick_qua_check(video_path, config=None):
    """Perform quick QA validation on a video file"""
    return quick_qa_check(video_path, config)

# Default quality thresholds
DEFAULT_QUALITY_THRESHOLDS = {
    "excellent": 90.0,
    "good": 80.0,
    "acceptable": 70.0,
    "needs_improvement": 50.0,
    "unacceptable": 30.0
}

# Broadcast standards reference
BROADCAST_STANDARDS = {
    "rec709": {
        "ire_range": (7.5, 100.0),
        "color_space": "BT.709",
        "gamut_limits": {
            "r_max": 0.95,
            "g_max": 0.95, 
            "b_max": 0.95
        }
    },
    "rec2020": {
        "ire_range": (7.5, 100.0),
        "color_space": "BT.2020",
        "gamut_limits": {
            "r_max": 0.98,
            "g_max": 0.98,
            "b_max": 0.98
        }
    }
}

__all__ = [
    'QAValidator',
    'QAReport', 
    'StyleMonitor',
    'AudioSyncChecker',
    'BroadcastStandardsValidator',
    'ArtifactDetector',
    'QualityScoreCalculator',
    'StyleMetrics',
    'SyncMetrics',
    'BroadcastCompliance',
    'ArtifactDetection',
    'QualityScoreBreakdown',
    'BroadcastStandard',
    'LegalizerMode',
    'ArtifactType',
    'QualityMetric',
    'MetricsCollector',
    'QualityMetrics',
    'MetricsReport',
    'ReportConfig',
    'MetricsVisualizer',
    'create_qa_validator',
    'quick_qa_check',
    'QA_CONFIG',
    'DEFAULT_QUALITY_THRESHOLDS',
    'BROADCAST_STANDARDS'
]
