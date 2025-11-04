"""
APEX DIRECTOR Quality Metrics Collector

Collects and analyzes quality metrics for the music video generation system.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

from ..images.style_persistence import StylePersistence
from ..images.variant_selector import VariantSelector
from ..audio.analyzer import AudioAnalyzer
from .style_monitor import StyleMonitor
from .sync_checker import SyncChecker
from .broadcast_standards import BroadcastStandards

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Represents a collection of quality metrics for a specific project stage.

    Attributes:
        timestamp: The timestamp of when the metrics were collected.
        project_id: The ID of the project.
        stage: The stage of the project for which the metrics were collected.
        metrics: A dictionary of the collected metrics.
        metadata: A dictionary of metadata associated with the metrics.
    """
    timestamp: datetime
    project_id: str
    stage: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the QualityMetrics to a dictionary for JSON serialization.

        Returns:
            A dictionary representation of the QualityMetrics.
        """
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricsCollector:
    """A class for collecting and analyzing quality metrics."""
    
    def __init__(self):
        """Initializes the MetricsCollector."""
        self.metrics_history: List[QualityMetrics] = []
        self.session_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Initialize analysis components
        self.style_monitor = StyleMonitor()
        self.sync_checker = SyncChecker()
        self.broadcast_standards = BroadcastStandards()
        
    async def collect_image_quality_metrics(
        self,
        project_id: str,
        images: List[str],
        style_reference: Optional[str] = None
    ) -> Dict[str, float]:
        """Collects image quality metrics.

        Args:
            project_id: The ID of the project.
            images: A list of image paths.
            style_reference: The path to a style reference image.

        Returns:
            A dictionary of image quality metrics.
        """
        metrics = {}
        
        try:
            # Style consistency scores
            if style_reference and len(images) > 1:
                consistency_scores = []
                for image in images:
                    score = await self.style_monitor.calculate_style_similarity(
                        image, style_reference
                    )
                    consistency_scores.append(score)
                
                metrics['style_consistency_avg'] = np.mean(consistency_scores)
                metrics['style_consistency_std'] = np.std(consistency_scores)
                metrics['style_consistency_min'] = min(consistency_scores)
                metrics['style_consistency_max'] = max(consistency_scores)
            
            # Visual quality metrics
            aesthetic_scores = []
            for image in images:
                score = await self._calculate_aesthetic_score(image)
                aesthetic_scores.append(score)
            
            metrics['aesthetic_score_avg'] = np.mean(aesthetic_scores)
            metrics['aesthetic_score_std'] = np.std(aesthetic_scores)
            metrics['image_count'] = len(images)
            
            # Additional quality checks
            artifact_count = 0
            for image in images:
                artifacts = await self._detect_artifacts(image)
                artifact_count += len(artifacts)
            
            metrics['artifact_count'] = artifact_count
            metrics['artifact_rate'] = artifact_count / len(images) if images else 0
            
        except Exception as e:
            logger.error(f"Error collecting image quality metrics: {e}")
            metrics['error'] = 1.0
            
        return metrics
    
    async def collect_video_quality_metrics(
        self,
        project_id: str,
        video_path: str,
        reference_audio: Optional[str] = None
    ) -> Dict[str, float]:
        """Collects video quality metrics.

        Args:
            project_id: The ID of the project.
            video_path: The path to the video file.
            reference_audio: The path to a reference audio file.

        Returns:
            A dictionary of video quality metrics.
        """
        metrics = {}
        
        try:
            # Audio-visual synchronization
            if reference_audio:
                sync_score = await self.sync_checker.check_sync(
                    video_path, reference_audio
                )
                metrics['sync_score'] = sync_score
            
            # Broadcast standards compliance
            broadcast_score = await self.broadcast_standards.validate_video(
                video_path
            )
            metrics['broadcast_compliance'] = broadcast_score
            
            # Technical quality metrics
            technical_metrics = await self._analyze_video_technical_quality(
                video_path
            )
            metrics.update(technical_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting video quality metrics: {e}")
            metrics['error'] = 1.0
            
        return metrics
    
    async def collect_workflow_metrics(
        self,
        project_id: str,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Collects workflow performance metrics.

        Args:
            project_id: The ID of the project.
            workflow_data: A dictionary of workflow data.

        Returns:
            A dictionary of workflow performance metrics.
        """
        metrics = {}
        
        try:
            # Processing time metrics
            for stage, timing in workflow_data.get('stage_timings', {}).items():
                metrics[f'{stage}_time'] = timing
            
            # Success/failure rates
            total_operations = workflow_data.get('total_operations', 1)
            failed_operations = workflow_data.get('failed_operations', 0)
            
            metrics['success_rate'] = (total_operations - failed_operations) / total_operations
            metrics['failure_rate'] = failed_operations / total_operations
            
            # Resource usage
            metrics['peak_memory_mb'] = workflow_data.get('peak_memory_mb', 0)
            metrics['avg_cpu_usage'] = workflow_data.get('avg_cpu_usage', 0)
            
            # Quality scores
            overall_quality = workflow_data.get('overall_quality_score', 0)
            metrics['overall_quality'] = overall_quality
            
        except Exception as e:
            logger.error(f"Error collecting workflow metrics: {e}")
            metrics['error'] = 1.0
            
        return metrics
    
    async def collect_comprehensive_metrics(
        self,
        project_id: str,
        stage: str,
        data: Dict[str, Any]
    ) -> QualityMetrics:
        """Collects comprehensive metrics for a project stage.

        Args:
            project_id: The ID of the project.
            stage: The stage of the project.
            data: A dictionary of data for the stage.

        Returns:
            A QualityMetrics object.
        """
        metrics = {}
        
        # Collect stage-specific metrics
        if 'images' in data:
            image_metrics = await self.collect_image_quality_metrics(
                project_id, data['images'], data.get('style_reference')
            )
            metrics.update(image_metrics)
        
        if 'video_path' in data:
            video_metrics = await self.collect_video_quality_metrics(
                project_id, data['video_path'], data.get('reference_audio')
            )
            metrics.update(video_metrics)
        
        if 'workflow_data' in data:
            workflow_metrics = await self.collect_workflow_metrics(
                project_id, data['workflow_data']
            )
            metrics.update(workflow_metrics)
        
        # Store metrics
        quality_metrics = QualityMetrics(
            timestamp=datetime.now(),
            project_id=project_id,
            stage=stage,
            metrics=metrics,
            metadata=data.get('metadata', {})
        )
        
        self.metrics_history.append(quality_metrics)
        
        return quality_metrics
    
    async def _calculate_aesthetic_score(self, image_path: str) -> float:
        """Calculates the aesthetic score for an image.

        This is a placeholder implementation. A real implementation would
        integrate with CLIP or similar aesthetic models.

        Args:
            image_path: The path to the image.

        Returns:
            The aesthetic score.
        """
        # This would integrate with CLIP or similar aesthetic models
        # For now, return a placeholder
        return 0.75
    
    async def _detect_artifacts(self, image_path: str) -> List[str]:
        """Detects visual artifacts in an image.

        This is a placeholder implementation. A real implementation would
        integrate with artifact detection models.

        Args:
            image_path: The path to the image.

        Returns:
            A list of detected artifacts.
        """
        # This would integrate with artifact detection models
        # For now, return empty list
        return []
    
    async def _analyze_video_technical_quality(self, video_path: str) -> Dict[str, float]:
        """Analyzes the technical quality of a video.

        This is a placeholder implementation. A real implementation would
        integrate with FFmpeg analysis.

        Args:
            video_path: The path to the video.

        Returns:
            A dictionary of technical quality metrics.
        """
        # This would integrate with FFmpeg analysis
        # For now, return placeholder metrics
        return {
            'bitrate_kbps': 5000,
            'resolution': 1920,
            'fps': 30,
            'compression_efficiency': 0.85
        }
    
    def get_project_metrics(self, project_id: str) -> List[QualityMetrics]:
        """Gets all metrics for a specific project.

        Args:
            project_id: The ID of the project.

        Returns:
            A list of QualityMetrics objects.
        """
        return [m for m in self.metrics_history if m.project_id == project_id]
    
    def get_summary_stats(self, project_id: str) -> Dict[str, Any]:
        """Gets summary statistics for a project.

        Args:
            project_id: The ID of the project.

        Returns:
            A dictionary of summary statistics.
        """
        project_metrics = self.get_project_metrics(project_id)
        
        if not project_metrics:
            return {}
        
        # Aggregate all metrics
        all_metric_names = set()
        for metrics in project_metrics:
            all_metric_names.update(metrics.metrics.keys())
        
        summary = {}
        for metric_name in all_metric_names:
            values = [
                m.metrics.get(metric_name, 0)
                for m in project_metrics
                if metric_name in m.metrics
            ]
            
            if values:
                summary[metric_name] = {
                    'avg': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return summary
    
    def export_metrics(self, output_path: Path) -> None:
        """Exports the metrics to a JSON file.

        Args:
            output_path: The path to the output file.
        """
        export_data = {
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'export_timestamp': datetime.now().isoformat(),
            'total_sessions': len(set(m.project_id for m in self.metrics_history))
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {output_path}")
