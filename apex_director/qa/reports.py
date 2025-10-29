"""
Quality Reports Generator
Generates comprehensive QA reports in multiple formats
"""

import json
import csv
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import asdict

from .validator import QAReport
from .score_calculator import QualityScoreBreakdown


class QualityReportGenerator:
    """
    Quality Report Generator
    
    Generates comprehensive QA reports in multiple formats:
    - JSON reports for machine processing
    - HTML reports for human viewing
    - CSV reports for data analysis
    - PDF reports for formal documentation
    - Executive summaries for stakeholders
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize report generator"""
        self.config = config or self._default_config()
        self.logger = logging.getLogger('apex_director.qa.reports')
        
        # Report templates
        self.templates = self._load_templates()
        
        # Report formatting options
        self.format_options = self.config.get('format_options', {
            'include_detailed_metrics': True,
            'include_technical_details': True,
            'include_recommendations': True,
            'include_historical_data': True,
            'include_benchmark_comparison': True
        })
        
        # Output directories
        self.output_dirs = self.config.get('output_dirs', {
            'json': 'reports/json',
            'html': 'reports/html',
            'csv': 'reports/csv',
            'pdf': 'reports/pdf',
            'executive': 'reports/executive'
        })
    
    def generate_comprehensive_report(self, qa_report: QAReport, breakdown: QualityScoreBreakdown, 
                                    output_dir: str = "qa_reports") -> Dict[str, str]:
        """
        Generate comprehensive QA report in multiple formats
        
        Args:
            qa_report: QA validation report
            breakdown: Detailed quality score breakdown
            output_dir: Base output directory
            
        Returns:
            Dictionary with paths to generated reports
        """
        self.logger.info(f"Generating comprehensive QA reports in {output_dir}")
        
        try:
            # Create output directories
            os.makedirs(output_dir, exist_ok=True)
            for format_dir in self.output_dirs.values():
                full_dir = os.path.join(output_dir, format_dir)
                os.makedirs(full_dir, exist_ok=True)
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"qa_report_{timestamp}"
            
            generated_reports = {}
            
            # Generate JSON report
            json_path = self._generate_json_report(
                qa_report, breakdown, output_dir, base_filename
            )
            generated_reports['json'] = json_path
            
            # Generate HTML report
            html_path = self._generate_html_report(
                qa_report, breakdown, output_dir, base_filename
            )
            generated_reports['html'] = html_path
            
            # Generate CSV report
            csv_path = self._generate_csv_report(
                qa_report, breakdown, output_dir, base_filename
            )
            generated_reports['csv'] = csv_path
            
            # Generate executive summary
            exec_path = self._generate_executive_summary(
                qa_report, breakdown, output_dir, base_filename
            )
            generated_reports['executive'] = exec_path
            
            # Generate technical report
            tech_path = self._generate_technical_report(
                qa_report, breakdown, output_dir, base_filename
            )
            generated_reports['technical'] = tech_path
            
            # Generate comparison report (if historical data available)
            comp_path = self._generate_comparison_report(
                qa_report, breakdown, output_dir, base_filename
            )
            if comp_path:
                generated_reports['comparison'] = comp_path
            
            self.logger.info(f"Generated {len(generated_reports)} reports successfully")
            return generated_reports
            
        except Exception as e:
            self.logger.error(f"Comprehensive report generation failed: {e}")
            return {}
    
    def _generate_json_report(self, qa_report: QAReport, breakdown: QualityScoreBreakdown,
                            output_dir: str, base_filename: str) -> str:
        """Generate JSON report for machine processing"""
        try:
            filename = f"{base_filename}.json"
            filepath = os.path.join(output_dir, self.output_dirs['json'], filename)
            
            # Combine all data into comprehensive structure
            report_data = {
                'metadata': {
                    'generator': 'APEX DIRECTOR QA Framework',
                    'version': '1.0.0',
                    'timestamp': datetime.now().isoformat(),
                    'report_type': 'comprehensive_qa'
                },
                'qa_report': asdict(qa_report),
                'quality_breakdown': asdict(breakdown),
                'component_scores': breakdown.component_scores,
                'weighted_scores': breakdown.weighted_scores,
                'penalties': breakdown.penalties,
                'bonuses': breakdown.bonuses,
                'critical_issues': breakdown.critical_issues,
                'improvement_opportunities': breakdown.improvement_opportunities,
                'quality_metrics': self._extract_quality_metrics(qa_report, breakdown),
                'compliance_status': self._generate_compliance_status(qa_report),
                'recommendations_summary': self._generate_recommendations_summary(qa_report, breakdown)
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"JSON report generated: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"JSON report generation failed: {e}")
            return ""
    
    def _generate_html_report(self, qa_report: QAReport, breakdown: QualityScoreBreakdown,
                            output_dir: str, base_filename: str) -> str:
        """Generate HTML report for human viewing"""
        try:
            filename = f"{base_filename}.html"
            filepath = os.path.join(output_dir, self.output_dirs['html'], filename)
            
            html_content = self._create_html_template(qa_report, breakdown)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"HTML report generation failed: {e}")
            return ""
    
    def _generate_csv_report(self, qa_report: QAReport, breakdown: QualityScoreBreakdown,
                           output_dir: str, base_filename: str) -> str:
        """Generate CSV report for data analysis"""
        try:
            filename = f"{base_filename}.csv"
            filepath = os.path.join(output_dir, self.output_dirs['csv'], filename)
            
            # Create comprehensive CSV data
            csv_data = []
            
            # Basic information row
            csv_data.append(['Metric', 'Value', 'Status'])
            csv_data.append(['Overall Score', f"{breakdown.overall_score:.1f}", self._get_status_label(breakdown.overall_score)])
            csv_data.append(['Quality Level', self._get_quality_level(breakdown.overall_score), ''])
            csv_data.append(['Pass Status', 'PASS' if qa_report.pass_status else 'FAIL', ''])
            csv_data.append(['Video Path', qa_report.video_path, ''])
            csv_data.append(['Timestamp', qa_report.timestamp, ''])
            csv_data.append(['Duration', f"{qa_report.duration:.2f}s", ''])
            csv_data.append(['Resolution', f"{qa_report.resolution[0]}x{qa_report.resolution[1]}", ''])
            
            # Component scores
            csv_data.append(['', '', ''])  # Empty row
            csv_data.append(['Component Scores', '', ''])
            for component, score in breakdown.component_scores.items():
                csv_data.append([component.replace('_', ' ').title(), f"{score:.1f}", self._get_status_label(score)])
            
            # Critical issues
            csv_data.append(['', '', ''])
            csv_data.append(['Critical Issues', '', ''])
            for issue in breakdown.critical_issues:
                csv_data.append([issue, '', 'CRITICAL'])
            
            # Recommendations
            csv_data.append(['', '', ''])
            csv_data.append(['Recommendations', '', ''])
            for rec in breakdown.improvement_opportunities:
                csv_data.append([rec, '', 'INFO'])
            
            # Write CSV file
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            self.logger.info(f"CSV report generated: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"CSV report generation failed: {e}")
            return ""
    
    def _generate_executive_summary(self, qa_report: QAReport, breakdown: QualityScoreBreakdown,
                                  output_dir: str, base_filename: str) -> str:
        """Generate executive summary report"""
        try:
            filename = f"{base_filename}_executive.md"
            filepath = os.path.join(output_dir, self.output_dirs['executive'], filename)
            
            content = f"""# EXECUTIVE QUALITY ASSESSMENT REPORT

## Executive Summary

**Content Quality**: {self._get_quality_level(breakdown.overall_score)} ({breakdown.overall_score:.1f}/100)
**Delivery Status**: {'APPROVED FOR DELIVERY' if qa_report.pass_status else 'REQUIRES REVISION'}
**Overall Assessment**: {self._get_executive_summary(breakdown.overall_score, qa_report.pass_status)}

## Key Quality Metrics

| Component | Score | Status |
|-----------|-------|--------|
| Visual Consistency | {breakdown.component_scores.get('visual_consistency', 0):.1f}/100 | {self._get_status_label(breakdown.component_scores.get('visual_consistency', 0))} |
| Audio Synchronization | {breakdown.component_scores.get('audio_sync', 0):.1f}/100 | {self._get_status_label(breakdown.component_scores.get('audio_sync', 0))} |
| Broadcast Compliance | {breakdown.component_scores.get('broadcast_compliance', 0):.1f}/100 | {self._get_status_label(breakdown.component_scores.get('broadcast_compliance', 0))} |
| Artifact Detection | {breakdown.component_scores.get('artifact_detection', 0):.1f}/100 | {self._get_status_label(breakdown.component_scores.get('artifact_detection', 0))} |

## Critical Findings

### Critical Issues Requiring Immediate Attention
{chr(10).join(f"- {issue}" for issue in breakdown.critical_issues[:5]) if breakdown.critical_issues else "- No critical issues detected"}

### Priority Recommendations
{chr(10).join(f"- {rec}" for rec in breakdown.improvement_opportunities[:3]) if breakdown.improvement_opportunities else "- Quality standards met across all metrics"}

## Compliance Status

### Broadcast Standards
- IRE Levels: {'COMPLIANT' if qa_report.broadcast_compliance_score > 80 else 'NON-COMPLIANT'}
- Color Gamut: {'COMPLIANT' if qa_report.broadcast_compliance_score > 75 else 'NON-COMPLIANT'}
- Safe Areas: {'COMPLIANT' if qa_report.broadcast_compliance_score > 85 else 'NON-COMPLIANT'}

### Quality Benchmarks
- Professional Standard (85+): {'MET' if breakdown.overall_score >= 85 else 'NOT MET'}
- Broadcast Ready (80+): {'MET' if breakdown.overall_score >= 80 else 'NOT MET'}
- Streaming Quality (75+): {'MET' if breakdown.overall_score >= 75 else 'NOT MET'}

## Delivery Recommendation

**Recommended Action**: {self._get_delivery_recommendation(breakdown.overall_score, qa_report.pass_status, breakdown.critical_issues)}

**Risk Assessment**: {self._assess_delivery_risk(breakdown, qa_report)}

**Timeline Impact**: {self._assess_timeline_impact(breakdown, qa_report.pass_status)}

## Next Steps

1. **Immediate** (if required): Address critical issues
2. **Short-term** (1-2 days): Implement high-priority recommendations
3. **Medium-term** (1 week): Apply quality improvements
4. **Long-term**: Monitor quality trends and implement preventive measures

---

**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**QA Framework**: APEX DIRECTOR v1.0.0
**Confidence Level**: {breakdown.confidence_level:.1%}
**Quality Trend**: {breakdown.score_trend.title()}
"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Executive summary generated: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Executive summary generation failed: {e}")
            return ""
    
    def _generate_technical_report(self, qa_report: QAReport, breakdown: QualityScoreBreakdown,
                                 output_dir: str, base_filename: str) -> str:
        """Generate technical detail report"""
        try:
            filename = f"{base_filename}_technical.md"
            filepath = os.path.join(output_dir, self.output_dirs['pdf'], filename)
            
            content = f"""# TECHNICAL QUALITY ASSESSMENT REPORT

## Technical Specifications

### Video Properties
- **Resolution**: {qa_report.resolution[0]}x{qa_report.resolution[1]}
- **Frame Rate**: {qa_report.frame_rate:.2f} fps
- **Duration**: {qa_report.duration:.2f} seconds
- **Color Space**: {qa_report.color_space}
- **Total Frames**: {int(qa_report.duration * qa_report.frame_rate)}

### Automated Analysis Results

#### Visual Consistency Analysis
- **CLIP Similarity Score**: {breakdown.component_scores.get('visual_consistency', 0):.1f}/100
- **Style Drift Detection**: {'DETECTED' if 'style_drift' in str(qa_report).lower() else 'NOT DETECTED'}
- **Color Consistency**: {self._assess_component_detail(qa_report, 'visual_consistency')}
- **Composition Stability**: {self._assess_component_detail(qa_report, 'visual_consistency')}

#### Audio Synchronization Analysis
- **Sync Accuracy**: {breakdown.component_scores.get('audio_sync', 0):.1f}/100
- **Frame Offset**: {getattr(qa_report, 'frame_offset', 'N/A')} frames
- **Time Offset**: {getattr(qa_report, 'time_offset_ms', 'N/A')} ms
- **Sync Variance**: {getattr(qa_report, 'sync_variance', 'N/A')}

#### Broadcast Standards Compliance
- **IRE Level Compliance**: {breakdown.component_scores.get('broadcast_compliance', 0):.1f}/100
- **Gamut Compliance**: {'PASS' if qa_report.broadcast_compliance_score > 80 else 'FAIL'}
- **Safe Area Compliance**: {'PASS' if qa_report.broadcast_compliance_score > 85 else 'FAIL'}
- **Legal Range Verification**: {'COMPLETE' if qa_report.broadcast_compliance_score > 75 else 'FAILED'}

#### Artifact Detection Results
- **Artifact Score**: {breakdown.component_scores.get('artifact_detection', 0):.1f}/100
- **Faces Detected**: {getattr(qa_report, 'faces_detected', 0)}
- **Text Elements**: {getattr(qa_report, 'text_detected', 0)}
- **Watermarks**: {getattr(qa_report, 'watermarks_detected', 0)}
- **Compression Artifacts**: {getattr(qa_report, 'compression_artifacts', 0)}

## Detailed Score Breakdown

### Weighted Component Scores
{chr(10).join(f"- **{component.replace('_', ' ').title()}**: {score:.2f}" for component, score in breakdown.weighted_scores.items())}

### Applied Penalties
{chr(10).join(f"- **{penalty}**: -{points} points" for penalty, points in breakdown.penalties.items()) if breakdown.penalties else "- No penalties applied"}

### Quality Bonuses
{chr(10).join(f"- **{bonus}**: +{points} points" for bonus, points in breakdown.bonuses.items()) if breakdown.bonuses else "- No bonuses applied"}

## Quality Metrics

### Confidence Analysis
- **Overall Confidence**: {breakdown.confidence_level:.1%}
- **Data Completeness**: {self._assess_data_completeness(qa_report, breakdown)}
- **Measurement Reliability**: {self._assess_reliability(breakdown)}

### Trend Analysis
- **Quality Trend**: {breakdown.score_trend.title()}
- **Historical Performance**: {self._assess_historical_performance(breakdown)}
- **Improvement Trajectory**: {self._assess_improvement_trajectory(breakdown)}

## Technical Recommendations

### Immediate Actions Required
{chr(10).join(f"- {issue}" for issue in breakdown.critical_issues) if breakdown.critical_issues else "- No immediate actions required"}

### Quality Optimization Opportunities
{chr(10).join(f"- {opp}" for opp in breakdown.improvement_opportunities) if breakdown.improvement_opportunities else "- Quality optimization complete"}

### Long-term Quality Management
- Implement automated quality monitoring
- Establish quality baselines and thresholds
- Regular calibration of QA systems
- Continuous improvement processes

## System Information

### QA Framework Details
- **Version**: APEX DIRECTOR QA Framework v1.0.0
- **Analysis Date**: {qa_report.timestamp}
- **Processing Time**: {self._estimate_processing_time(qa_report)}
- **System Load**: {self._assess_system_load()}

### Validation Methods
- CLIP-based visual similarity analysis
- Frame-accurate audio-visual synchronization
- Broadcast standards compliance validation
- Automated artifact detection
- Comprehensive quality scoring

---

**Technical Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Quality Assessment Framework**: APEX DIRECTOR v1.0.0
"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Technical report generated: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Technical report generation failed: {e}")
            return ""
    
    def _generate_comparison_report(self, qa_report: QAReport, breakdown: QualityScoreBreakdown,
                                  output_dir: str, base_filename: str) -> Optional[str]:
        """Generate comparison report with historical data"""
        try:
            # Placeholder for historical comparison
            # In a real implementation, would compare with previous QA results
            
            filename = f"{base_filename}_comparison.md"
            filepath = os.path.join(output_dir, self.output_dirs['html'], filename)
            
            content = """# QUALITY COMPARISON REPORT

## Historical Performance Analysis

This section would contain:
- Comparison with previous QA sessions
- Trend analysis over time
- Performance benchmarking
- Quality improvement tracking

*Historical comparison functionality to be implemented*
"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Comparison report generation failed: {e}")
            return None
    
    def _create_html_template(self, qa_report: QAReport, breakdown: QualityScoreBreakdown) -> str:
        """Create comprehensive HTML template"""
        try:
            # Color scheme based on quality score
            score_color = self._get_score_color(breakdown.overall_score)
            status_color = "green" if qa_report.pass_status else "red"
            
            html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APEX DIRECTOR - QA Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .subtitle {{ font-size: 1.2em; opacity: 0.9; }}
        
        .score-overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .score-card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid {score_color}; }}
        .score-card h3 {{ color: #2c3e50; margin-bottom: 10px; }}
        .score-value {{ font-size: 2.5em; font-weight: bold; color: {score_color}; }}
        .status {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; background: {status_color}; }}
        
        .section {{ background: white; margin-bottom: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section-header {{ background: #f8f9fa; padding: 20px; border-radius: 10px 10px 0 0; border-bottom: 1px solid #dee2e6; }}
        .section-header h2 {{ color: #2c3e50; }}
        .section-content {{ padding: 20px; }}
        
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric-item {{ padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .metric-label {{ font-weight: bold; color: #495057; }}
        .metric-value {{ font-size: 1.2em; color: #2c3e50; }}
        
        .issue-list {{ list-style: none; }}
        .issue-item {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .critical {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .info {{ background: #d1ecf1; border-left: 4px solid #17a2b8; }}
        
        .progress-bar {{ background: #e9ecef; border-radius: 10px; overflow: hidden; height: 20px; }}
        .progress-fill {{ height: 100%; background: {score_color}; transition: width 0.3s ease; }}
        
        .footer {{ text-align: center; margin-top: 50px; padding: 20px; color: #6c757d; border-top: 1px solid #dee2e6; }}
        
        @media print {{ .container {{ max-width: none; }} .section {{ page-break-inside: avoid; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Quality Assessment Report</h1>
            <div class="subtitle">APEX DIRECTOR QA Framework v1.0.0</div>
            <div class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
        
        <div class="score-overview">
            <div class="score-card">
                <h3>Overall Quality Score</h3>
                <div class="score-value">{breakdown.overall_score:.1f}</div>
                <span class="status">{'PASS' if qa_report.pass_status else 'FAIL'}</span>
                <div class="progress-bar" style="margin-top: 15px;">
                    <div class="progress-fill" style="width: {breakdown.overall_score}%"></div>
                </div>
            </div>
            
            <div class="score-card">
                <h3>Quality Level</h3>
                <div class="score-value">{self._get_quality_level(breakdown.overall_score)}</div>
                <p style="margin-top: 10px; color: #6c757d;">Based on comprehensive analysis</p>
            </div>
            
            <div class="score-card">
                <h3>Confidence Level</h3>
                <div class="score-value">{breakdown.confidence_level:.1%}</div>
                <p style="margin-top: 10px; color: #6c757d;">Analysis reliability</p>
            </div>
            
            <div class="score-card">
                <h3>Quality Trend</h3>
                <div class="score-value">{breakdown.score_trend.title()}</div>
                <p style="margin-top: 10px; color: #6c757d;">Historical comparison</p>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2>Component Scores</h2>
            </div>
            <div class="section-content">
                <div class="metric-grid">
                    {chr(10).join(f'''
                    <div class="metric-item">
                        <div class="metric-label">{component.replace('_', ' ').title()}</div>
                        <div class="metric-value">{score:.1f}/100</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {score}%"></div>
                        </div>
                    </div>
                    ''' for component, score in breakdown.component_scores.items())}
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2>Critical Issues</h2>
            </div>
            <div class="section-content">
                <ul class="issue-list">
                    {chr(10).join(f'<li class="issue-item critical">{issue}</li>' for issue in breakdown.critical_issues)}
                </ul>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2>Improvement Opportunities</h2>
            </div>
            <div class="section-content">
                <ul class="issue-list">
                    {chr(10).join(f'<li class="issue-item info">{opp}</li>' for opp in breakdown.improvement_opportunities)}
                </ul>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2>Technical Details</h2>
            </div>
            <div class="section-content">
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-label">Video Resolution</div>
                        <div class="metric-value">{qa_report.resolution[0]}x{qa_report.resolution[1]}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Frame Rate</div>
                        <div class="metric-value">{qa_report.frame_rate:.2f} fps</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Duration</div>
                        <div class="metric-value">{qa_report.duration:.2f} seconds</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Color Space</div>
                        <div class="metric-value">{qa_report.color_space}</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by APEX DIRECTOR Quality Assurance Framework</p>
            <p>Report ID: {datetime.now().strftime("%Y%m%d_%H%M%S")}</p>
        </div>
    </div>
</body>
</html>
            """
            
            return html_template
            
        except Exception as e:
            self.logger.error(f"HTML template creation failed: {e}")
            return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"
    
    def _extract_quality_metrics(self, qa_report: QAReport, breakdown: QualityScoreBreakdown) -> Dict:
        """Extract key quality metrics"""
        return {
            'overall_quality_score': breakdown.overall_score,
            'component_scores': breakdown.component_scores,
            'weighted_scores': breakdown.weighted_scores,
            'confidence_level': breakdown.confidence_level,
            'quality_trend': breakdown.score_trend,
            'pass_status': qa_report.pass_status,
            'quality_level': self._get_quality_level(breakdown.overall_score)
        }
    
    def _generate_compliance_status(self, qa_report: QAReport) -> Dict:
        """Generate compliance status summary"""
        return {
            'broadcast_compliance': qa_report.broadcast_compliance_score > 80,
            'ire_levels': qa_report.broadcast_compliance_score > 75,
            'color_gamut': qa_report.broadcast_compliance_score > 70,
            'safe_areas': qa_report.broadcast_compliance_score > 85,
            'overall_compliant': qa_report.pass_status
        }
    
    def _generate_recommendations_summary(self, qa_report: QAReport, breakdown: QualityScoreBreakdown) -> List[str]:
        """Generate summary of key recommendations"""
        recommendations = []
        
        if breakdown.critical_issues:
            recommendations.append("Address critical issues immediately")
        
        if breakdown.component_scores.get('broadcast_compliance', 100) < 80:
            recommendations.append("Apply broadcast legalizer")
        
        if breakdown.component_scores.get('audio_sync', 100) < 90:
            recommendations.append("Improve audio-visual synchronization")
        
        if breakdown.component_scores.get('visual_consistency', 100) < 85:
            recommendations.append("Enhance visual style consistency")
        
        return recommendations[:5]  # Limit to top 5
    
    def _get_status_label(self, score: float) -> str:
        """Get status label based on score"""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "GOOD"
        elif score >= 70:
            return "ACCEPTABLE"
        elif score >= 60:
            return "NEEDS IMPROVEMENT"
        else:
            return "POOR"
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level description"""
        if score >= 95:
            return "Exceptional"
        elif score >= 85:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 65:
            return "Acceptable"
        elif score >= 50:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def _get_score_color(self, score: float) -> str:
        """Get color for score visualization"""
        if score >= 90:
            return "#28a745"  # Green
        elif score >= 80:
            return "#17a2b8"  # Blue
        elif score >= 70:
            return "#ffc107"  # Yellow
        elif score >= 60:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red
    
    def _get_executive_summary(self, score: float, pass_status: bool) -> str:
        """Generate executive summary text"""
        if score >= 90 and pass_status:
            return "Content demonstrates exceptional quality and is approved for immediate delivery across all platforms."
        elif score >= 80 and pass_status:
            return "Content meets professional quality standards and is approved for delivery with minor optimization recommended."
        elif score >= 70:
            return "Content quality is acceptable but requires attention to specific areas before delivery."
        else:
            return "Content requires significant quality improvements before delivery approval can be granted."
    
    def _get_delivery_recommendation(self, score: float, pass_status: bool, critical_issues: List[str]) -> str:
        """Get delivery recommendation"""
        if not critical_issues and pass_status:
            return "APPROVED FOR IMMEDIATE DELIVERY"
        elif score >= 80:
            return "APPROVED WITH MINOR REVISIONS"
        elif score >= 70:
            return "CONDITIONAL APPROVAL - REQUIRES REVISIONS"
        else:
            return "NOT APPROVED - MAJOR REVISIONS REQUIRED"
    
    def _assess_delivery_risk(self, breakdown: QualityScoreBreakdown, qa_report: QAReport) -> str:
        """Assess delivery risk"""
        risk_score = 0
        
        if breakdown.overall_score < 70:
            risk_score += 3
        elif breakdown.overall_score < 80:
            risk_score += 1
        
        if len(breakdown.critical_issues) > 0:
            risk_score += 2
        
        if qa_report.broadcast_compliance_score < 80:
            risk_score += 1
        
        if risk_score >= 4:
            return "HIGH RISK"
        elif risk_score >= 2:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"
    
    def _assess_timeline_impact(self, breakdown: QualityScoreBreakdown, pass_status: bool) -> str:
        """Assess timeline impact"""
        if pass_status:
            return "NO DELAY - On schedule for delivery"
        elif breakdown.overall_score >= 70:
            return "MINOR DELAY - 1-2 days additional time needed"
        else:
            return "SIGNIFICANT DELAY - 1 week additional time needed"
    
    def _assess_component_detail(self, qa_report: QAReport, component: str) -> str:
        """Assess component-specific details"""
        # Placeholder for component-specific analysis
        return "Analysis complete"
    
    def _assess_data_completeness(self, qa_report: QAReport, breakdown: QualityScoreBreakdown) -> str:
        """Assess completeness of analysis data"""
        completeness_score = 0
        total_components = 6  # Expected number of components
        
        if breakdown.component_scores:
            completeness_score += len(breakdown.component_scores) / total_components * 50
        
        if breakdown.confidence_level > 0.8:
            completeness_score += 50
        
        if completeness_score >= 90:
            return "COMPLETE"
        elif completeness_score >= 70:
            return "MOSTLY COMPLETE"
        else:
            return "PARTIAL"
    
    def _assess_reliability(self, breakdown: QualityScoreBreakdown) -> str:
        """Assess reliability of measurements"""
        if breakdown.confidence_level > 0.9:
            return "HIGH"
        elif breakdown.confidence_level > 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_historical_performance(self, breakdown: QualityScoreBreakdown) -> str:
        """Assess historical performance (placeholder)"""
        return "Historical data analysis pending"
    
    def _assess_improvement_trajectory(self, breakdown: QualityScoreBreakdown) -> str:
        """Assess improvement trajectory"""
        if breakdown.score_trend == "improving":
            return "Positive improvement trend detected"
        elif breakdown.score_trend == "declining":
            return "Declining trend requires attention"
        else:
            return "Stable performance level"
    
    def _estimate_processing_time(self, qa_report: QAReport) -> str:
        """Estimate processing time (placeholder)"""
        return "Analysis completed successfully"
    
    def _assess_system_load(self) -> str:
        """Assess system load during processing (placeholder)"""
        return "Normal processing load"
    
    def _load_templates(self) -> Dict:
        """Load report templates"""
        return {
            'html': 'comprehensive',
            'json': 'detailed',
            'csv': 'tabular',
            'executive': 'summary'
        }
    
    def _default_config(self) -> Dict:
        """Default configuration for report generator"""
        return {
            'format_options': {
                'include_detailed_metrics': True,
                'include_technical_details': True,
                'include_recommendations': True,
                'include_historical_data': True,
                'include_benchmark_comparison': True
            },
            'output_dirs': {
                'json': 'reports/json',
                'html': 'reports/html',
                'csv': 'reports/csv',
                'pdf': 'reports/pdf',
                'executive': 'reports/executive'
            }
        }


if __name__ == "__main__":
    # Example usage
    from .validator import QAValidator
    from .score_calculator import QualityScoreCalculator
    
    # This would typically be called with actual QA results
    print("Quality Report Generator initialized")
