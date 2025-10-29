"""
APEX DIRECTOR Quality Metrics Visualization

Provides visualization utilities for quality metrics and dashboards.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from io import BytesIO

from .metrics_collector import MetricsCollector, QualityMetrics

logger = logging.getLogger(__name__)


class MetricsVisualizer:
    """Creates visualizations for quality metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.color_palette = sns.color_palette("husl", 12)
        
        # Configure matplotlib for high-quality output
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
    def create_timeline_chart(
        self,
        project_id: str,
        metric_name: str,
        output_path: Path,
        time_range_days: int = 30
    ) -> Path:
        """Create timeline chart for a specific metric"""
        
        # Get project metrics
        project_metrics = self.metrics_collector.get_project_metrics(project_id)
        
        # Filter by time range
        cutoff_date = datetime.now() - timedelta(days=time_range_days)
        filtered_metrics = [
            m for m in project_metrics 
            if m.timestamp >= cutoff_date and metric_name in m.metrics
        ]
        
        if not filtered_metrics:
            logger.warning(f"No data found for metric {metric_name}")
            return output_path
        
        # Extract data
        timestamps = [m.timestamp for m in filtered_metrics]
        values = [m.metrics[metric_name] for m in filtered_metrics]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(timestamps, values, marker='o', linewidth=2, markersize=6)
        ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, time_range_days // 10)))
        plt.xticks(rotation=45)
        
        # Add value annotations
        for i, (timestamp, value) in enumerate(zip(timestamps, values)):
            if i % max(1, len(timestamps) // 10) == 0:  # Annotate every 10th point
                ax.annotate(f'{value:.2f}', (timestamp, value), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Timeline chart saved: {output_path}")
        return output_path
    
    def create_summary_dashboard(
        self,
        project_ids: List[str],
        output_path: Path,
        time_range_days: int = 30
    ) -> Path:
        """Create comprehensive summary dashboard"""
        
        # Collect data for all projects
        all_metrics_data = []
        cutoff_date = datetime.now() - timedelta(days=time_range_days)
        
        for project_id in project_ids:
            project_metrics = self.metrics_collector.get_project_metrics(project_id)
            recent_metrics = [
                m for m in project_metrics if m.timestamp >= cutoff_date
            ]
            
            for metric in recent_metrics:
                all_metrics_data.append({
                    'project_id': project_id,
                    'timestamp': metric.timestamp,
                    'stage': metric.stage,
                    **metric.metrics
                })
        
        if not all_metrics_data:
            logger.warning("No data found for dashboard creation")
            return output_path
        
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics_data)
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('APEX DIRECTOR Quality Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Overall quality score over time
        self._plot_timeline_by_project(axes[0, 0], df, 'overall_quality', 'Overall Quality Score')
        
        # 2. Style consistency distribution
        self._plot_metric_distribution(axes[0, 1], df, 'style_consistency_avg', 'Style Consistency')
        
        # 3. Sync score by project
        self._plot_project_comparison(axes[0, 2], df, 'sync_score', 'Audio-Visual Sync Score')
        
        # 4. Broadcast compliance
        self._plot_stage_performance(axes[1, 0], df, 'broadcast_compliance', 'Broadcast Compliance')
        
        # 5. Aesthetic scores
        self._plot_metric_trend(axes[1, 1], df, 'aesthetic_score_avg', 'Aesthetic Score Trend')
        
        # 6. Success rate by stage
        self._plot_success_rates(axes[1, 2], df, 'success_rate', 'Success Rate by Stage')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary dashboard saved: {output_path}")
        return output_path
    
    def create_quality_heatmap(
        self,
        project_ids: List[str],
        output_path: Path,
        time_range_days: int = 30
    ) -> Path:
        """Create quality metrics heatmap"""
        
        # Get all metric names across projects
        metric_names = set()
        cutoff_date = datetime.now() - timedelta(days=time_range_days)
        
        for project_id in project_ids:
            project_metrics = self.metrics_collector.get_project_metrics(project_id)
            for metric in project_metrics:
                if metric.timestamp >= cutoff_date:
                    metric_names.update(metric.metrics.keys())
        
        if not metric_names:
            logger.warning("No metrics found for heatmap")
            return output_path
        
        # Create heatmap data matrix
        heatmap_data = []
        stage_names = ['Audio Analysis', 'Image Generation', 'Video Assembly', 'Export']
        
        for project_id in project_ids:
            project_row = [project_id]
            project_metrics = self.metrics_collector.get_project_metrics(project_id)
            
            for stage in stage_names:
                stage_metrics = [m for m in project_metrics 
                               if m.stage.lower().replace(' ', '_') in stage.lower().replace(' ', '_')
                               and m.timestamp >= cutoff_date]
                
                if stage_metrics:
                    avg_quality = np.mean([m.metrics.get('overall_quality', 0) for m in stage_metrics])
                    project_row.append(avg_quality)
                else:
                    project_row.append(0)
            
            heatmap_data.append(project_row)
        
        # Create heatmap
        if heatmap_data:
            df = pd.DataFrame(heatmap_data, columns=['Project'] + stage_names)
            df = df.set_index('Project')
            
            plt.figure(figsize=(12, max(6, len(project_ids) * 0.5)))
            sns.heatmap(df, annot=True, cmap='RdYlGn', center=0.5, 
                       vmin=0, vmax=1, cbar_kws={'label': 'Quality Score'})
            plt.title('Quality Scores by Project and Stage')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Quality heatmap saved: {output_path}")
        return output_path
    
    def create_image_to_base64(self, plot: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string for embedding in HTML"""
        buffer = BytesIO()
        plot.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return image_base64
    
    def _plot_timeline_by_project(self, ax, df, metric, title):
        """Plot timeline for multiple projects"""
        projects = df['project_id'].unique()
        colors = sns.color_palette("husl", len(projects))
        
        for i, project in enumerate(projects):
            project_data = df[df['project_id'] == project].sort_values('timestamp')
            if metric in project_data.columns:
                ax.plot(project_data['timestamp'], project_data[metric], 
                       label=project, color=colors[i], marker='o', alpha=0.7)
        
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_metric_distribution(self, ax, df, metric, title):
        """Plot distribution of a metric"""
        if metric in df.columns:
            sns.histplot(data=df, x=metric, ax=ax, bins=20, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_project_comparison(self, ax, df, metric, title):
        """Plot comparison between projects"""
        if metric in df.columns:
            sns.boxplot(data=df, x='project_id', y=metric, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Project')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_stage_performance(self, ax, df, metric, title):
        """Plot performance by stage"""
        if metric in df.columns:
            sns.barplot(data=df, x='stage', y=metric, ax=ax, estimator=np.mean)
            ax.set_title(title)
            ax.set_xlabel('Stage')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_metric_trend(self, ax, df, metric, title):
        """Plot trend of a metric over time"""
        if metric in df.columns:
            daily_avg = df.groupby(df['timestamp'].dt.date)[metric].mean().reset_index()
            ax.plot(daily_avg['timestamp'], daily_avg[metric], marker='o', linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Score')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_success_rates(self, ax, df, metric, title):
        """Plot success rates by stage"""
        if metric in df.columns:
            stage_success = df.groupby('stage')[metric].mean().reset_index()
            bars = ax.bar(range(len(stage_success)), stage_success[metric])
            
            # Color bars based on performance
            for i, bar in enumerate(bars):
                if stage_success[metric].iloc[i] >= 0.8:
                    bar.set_color('#28a745')
                elif stage_success[metric].iloc[i] >= 0.6:
                    bar.set_color('#ffc107')
                else:
                    bar.set_color('#dc3545')
            
            ax.set_title(title)
            ax.set_xlabel('Stage')
            ax.set_ylabel('Success Rate')
            ax.set_xticks(range(len(stage_success)))
            ax.set_xticklabels(stage_success['stage'], rotation=45)
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
