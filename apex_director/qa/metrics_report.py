"""
APEX DIRECTOR Quality Metrics Report Generator

Generates comprehensive quality reports and dashboards.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import json
from jinja2 import Template

from .metrics_collector import MetricsCollector, QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    include_charts: bool = True
    include_summary: bool = True
    include_details: bool = True
    include_recommendations: bool = True
    format: str = "html"  # html, json, markdown


class MetricsReport:
    """Quality metrics report generator"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.report_templates = self._load_templates()
    
    def generate_project_report(
        self,
        project_id: str,
        output_path: Path,
        config: Optional[ReportConfig] = None
    ) -> Path:
        """Generate comprehensive project quality report"""
        if config is None:
            config = ReportConfig()
        
        # Get project metrics
        project_metrics = self.metrics_collector.get_project_metrics(project_id)
        
        if not project_metrics:
            logger.warning(f"No metrics found for project {project_id}")
            return output_path
        
        # Generate report content
        report_content = self._generate_report_content(project_id, project_metrics, config)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Project report generated: {output_path}")
        return output_path
    
    def generate_summary_report(
        self,
        projects: List[str],
        output_path: Path,
        time_range_days: int = 30
    ) -> Path:
        """Generate summary report across multiple projects"""
        cutoff_date = datetime.now() - timedelta(days=time_range_days)
        
        # Filter metrics by time range
        recent_metrics = [
            m for m in self.metrics_collector.metrics_history
            if m.timestamp >= cutoff_date and m.project_id in projects
        ]
        
        if not recent_metrics:
            logger.warning("No recent metrics found for summary report")
            return output_path
        
        # Generate summary report
        report_content = self._generate_summary_content(projects, recent_metrics)
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Summary report generated: {output_path}")
        return output_path
    
    def _generate_report_content(
        self,
        project_id: str,
        project_metrics: List[QualityMetrics],
        config: ReportConfig
    ) -> str:
        """Generate report content using templates"""
        
        summary_stats = self.metrics_collector.get_summary_stats(project_id)
        
        template_data = {
            'project_id': project_id,
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics_count': len(project_metrics),
            'stages': list(set(m.stage for m in project_metrics)),
            'summary_stats': summary_stats,
            'latest_metrics': project_metrics[-1].metrics if project_metrics else {},
            'include_charts': config.include_charts,
            'include_summary': config.include_summary,
            'include_details': config.include_details,
            'include_recommendations': config.include_recommendations
        }
        
        return self.report_templates['project_report'].render(**template_data)
    
    def _generate_summary_content(
        self,
        projects: List[str],
        metrics: List[QualityMetrics]
    ) -> str:
        """Generate summary report content"""
        
        # Calculate overall statistics
        all_stages = set(m.stage for m in metrics)
        all_projects = set(m.project_id for m in metrics)
        
        stage_performance = {}
        for stage in all_stages:
            stage_metrics = [m for m in metrics if m.stage == stage]
            avg_scores = [
                m.metrics.get('overall_quality', 0) 
                for m in stage_metrics
                if 'overall_quality' in m.metrics
            ]
            
            if avg_scores:
                stage_performance[stage] = {
                    'avg_score': sum(avg_scores) / len(avg_scores),
                    'sample_count': len(avg_scores)
                }
        
        template_data = {
            'projects': projects,
            'total_projects': len(projects),
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'time_range_days': (datetime.now() - metrics[0].timestamp).days,
            'stage_performance': stage_performance,
            'total_operations': len(metrics)
        }
        
        return self.report_templates['summary_report'].render(**template_data)
    
    def _load_templates(self) -> Dict[str, Template]:
        """Load Jinja2 templates for report generation"""
        
        project_template = """
<!DOCTYPE html>
<html>
<head>
    <title>APEX DIRECTOR Quality Report - {{ project_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .metric-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric-card { background: #f9f9f9; padding: 10px; border-radius: 3px; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #333; }
        .metric-label { color: #666; font-size: 0.9em; }
        .score-good { color: #28a745; }
        .score-warning { color: #ffc107; }
        .score-error { color: #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <h1>APEX DIRECTOR Quality Report</h1>
        <p><strong>Project:</strong> {{ project_id }}</p>
        <p><strong>Generated:</strong> {{ generation_time }}</p>
        <p><strong>Data Points:</strong> {{ metrics_count }}</p>
    </div>
    
    {% if include_summary %}
    <div class="metric-section">
        <h2>Summary Statistics</h2>
        <div class="metric-grid">
            {% for metric_name, stats in summary_stats.items() %}
            <div class="metric-card">
                <div class="metric-label">{{ metric_name.replace('_', ' ').title() }}</div>
                <div class="metric-value">{{ "%.2f"|format(stats.avg) }}</div>
                <div>Samples: {{ stats.count }} | Range: {{ "%.2f"|format(stats.min) }} - {{ "%.2f"|format(stats.max) }}</div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
    
    {% if include_details %}
    <div class="metric-section">
        <h2>Latest Metrics</h2>
        <div class="metric-grid">
            {% for metric_name, value in latest_metrics.items() %}
            <div class="metric-card">
                <div class="metric-label">{{ metric_name.replace('_', ' ').title() }}</div>
                <div class="metric-value {% if value >= 0.8 %}score-good{% elif value >= 0.6 %}score-warning{% else %}score-error{% endif %}">
                    {{ "%.3f"|format(value) if value is number else value }}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
    
    {% if include_recommendations %}
    <div class="metric-section">
        <h2>Quality Recommendations</h2>
        <ul>
            {% for metric_name, stats in summary_stats.items() %}
                {% if stats.avg < 0.7 %}
                    <li><strong>{{ metric_name.replace('_', ' ').title() }}:</strong> Score is below recommended threshold. Consider optimization.</li>
                {% endif %}
            {% endfor %}
        </ul>
    </div>
    {% endif %}
</body>
</html>
"""
        
        summary_template = """
<!DOCTYPE html>
<html>
<head>
    <title>APEX DIRECTOR Quality Summary Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .metric-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric-card { background: #f9f9f9; padding: 10px; border-radius: 3px; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #333; }
        .metric-label { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>APEX DIRECTOR Quality Summary</h1>
        <p><strong>Projects:</strong> {{ total_projects }}</p>
        <p><strong>Generated:</strong> {{ generation_time }}</p>
        <p><strong>Time Range:</strong> Last {{ time_range_days }} days</p>
        <p><strong>Total Operations:</strong> {{ total_operations }}</p>
    </div>
    
    <div class="metric-section">
        <h2>Stage Performance</h2>
        <div class="metric-grid">
            {% for stage, performance in stage_performance.items() %}
            <div class="metric-card">
                <div class="metric-label">{{ stage.replace('_', ' ').title() }}</div>
                <div class="metric-value">{{ "%.2f"|format(performance.avg_score) }}</div>
                <div>Samples: {{ performance.sample_count }}</div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""
        
        return {
            'project_report': Template(project_template),
            'summary_report': Template(summary_template)
        }
