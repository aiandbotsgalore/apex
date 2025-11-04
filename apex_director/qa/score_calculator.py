"""
Quality Score Calculator
Comprehensive metrics system for video quality assessment
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum


class QualityMetric(Enum):
    """Enumeration of quality metric categories."""
    VISUAL_CONSISTENCY = "visual_consistency"
    AUDIO_SYNC = "audio_sync"
    BROADCAST_COMPLIANCE = "broadcast_compliance"
    ARTIFACT_DETECTION = "artifact_detection"
    TECHNICAL_QUALITY = "technical_quality"
    CONTENT_QUALITY = "content_quality"
    DELIVERY_READINESS = "delivery_readiness"


@dataclass
class QualityScoreBreakdown:
    """Represents a detailed breakdown of a quality score.

    Attributes:
        overall_score: The overall quality score.
        component_scores: A dictionary of scores for each quality component.
        weighted_scores: A dictionary of weighted scores for each quality
            component.
        penalties: A dictionary of penalties applied to the score.
        bonuses: A dictionary of bonuses applied to the score.
        confidence_level: The confidence level of the score.
        score_trend: The trend of the score (improving, declining, or stable).
        critical_issues: A list of critical issues found.
        improvement_opportunities: A list of improvement opportunities.
    """
    overall_score: float
    component_scores: Dict[str, float]
    weighted_scores: Dict[str, float]
    penalties: Dict[str, float]
    bonuses: Dict[str, float]
    confidence_level: float
    score_trend: str  # improving, declining, stable
    critical_issues: List[str]
    improvement_opportunities: List[str]


class QualityScoreCalculator:
    """A class for calculating a comprehensive quality score for a video.

    This class provides a sophisticated quality metrics system that includes:
    - Multi-dimensional quality scoring
    - Component-based analysis
    - Weighted scoring systems
    - Trend analysis
    - Predictive quality assessment
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initializes the QualityScoreCalculator.

        Args:
            config: A dictionary of configuration parameters.
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger('apex_director.qa.score_calculator')
        
        # Scoring weights for different components
        self.component_weights = self.config.get('weights', {
            'visual_consistency': 0.25,
            'audio_sync': 0.20,
            'broadcast_compliance': 0.20,
            'artifact_detection': 0.15,
            'technical_quality': 0.10,
            'content_quality': 0.07,
            'delivery_readiness': 0.03
        })
        
        # Normalize weights
        total_weight = sum(self.component_weights.values())
        if total_weight > 0:
            self.component_weights = {k: v / total_weight for k, v in self.component_weights.items()}
        
        # Penalty thresholds
        self.penalty_thresholds = self.config.get('penalty_thresholds', {
            'critical_failure': 50.0,
            'major_issue': 70.0,
            'minor_issue': 85.0
        })
        
        # Bonus criteria
        self.bonus_criteria = self.config.get('bonus_criteria', {
            'excellent_quality': 90.0,
            'broadcast_ready': 85.0,
            'minimal_artifacts': 80.0
        })
        
        # Quality benchmarks
        self.quality_benchmarks = self.config.get('quality_benchmarks', {
            'broadcast_tv': 80.0,
            'streaming': 75.0,
            'social_media': 70.0,
            'professional': 85.0,
            'premium': 90.0
        })
        
        # Scoring history for trend analysis
        self.score_history = []
        self.component_history = []
        
        # Quality validation rules
        self.validation_rules = self._initialize_validation_rules()
        
    def calculate_overall_score(self, qa_results: Dict) -> float:
        """Calculates the overall quality score from QA results.

        Args:
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            The overall quality score (0-100).
        """
        try:
            self.logger.info("Calculating overall quality score")
            
            # Extract component scores
            component_scores = self._extract_component_scores(qa_results)
            
            # Apply penalties and bonuses
            adjusted_scores = self._apply_quality_adjustments(component_scores, qa_results)
            
            # Calculate weighted overall score
            overall_score = self._calculate_weighted_score(adjusted_scores)
            
            # Apply quality bonuses
            final_score = self._apply_quality_bonuses(overall_score, qa_results)
            
            # Store for trend analysis
            self._store_score_for_trend_analysis(final_score, component_scores, qa_results)
            
            return min(100.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Overall score calculation failed: {e}")
            return 0.0
    
    def calculate_detailed_score_breakdown(self, qa_results: Dict) -> QualityScoreBreakdown:
        """Calculates a detailed quality score breakdown.

        Args:
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            A QualityScoreBreakdown object with a detailed analysis of the
            quality score.
        """
        try:
            # Extract basic metrics
            component_scores = self._extract_component_scores(qa_results)
            adjusted_scores = self._apply_quality_adjustments(component_scores, qa_results)
            weighted_scores = self._calculate_weighted_component_scores(adjusted_scores)
            
            # Calculate penalties
            penalties = self._calculate_penalties(component_scores, qa_results)
            
            # Calculate bonuses
            bonuses = self._calculate_bonuses(component_scores, qa_results)
            
            # Calculate overall score
            overall_score = self._calculate_weighted_score(adjusted_scores)
            final_score = self._apply_quality_bonuses(overall_score, qa_results)
            
            # Determine confidence level
            confidence_level = self._calculate_confidence_level(qa_results)
            
            # Analyze score trend
            score_trend = self._analyze_score_trend()
            
            # Identify critical issues
            critical_issues = self._identify_critical_issues(component_scores, qa_results)
            
            # Identify improvement opportunities
            improvement_opportunities = self._identify_improvement_opportunities(component_scores)
            
            return QualityScoreBreakdown(
                overall_score=final_score,
                component_scores=component_scores,
                weighted_scores=weighted_scores,
                penalties=penalties,
                bonuses=bonuses,
                confidence_level=confidence_level,
                score_trend=score_trend,
                critical_issues=critical_issues,
                improvement_opportunities=improvement_opportunities
            )
            
        except Exception as e:
            self.logger.error(f"Detailed score breakdown calculation failed: {e}")
            return QualityScoreBreakdown(
                overall_score=0.0,
                component_scores={},
                weighted_scores={},
                penalties={},
                bonuses={},
                confidence_level=0.0,
                score_trend="unknown",
                critical_issues=["Score calculation failed"],
                improvement_opportunities=["Review QA system configuration"]
            )
    
    def _extract_component_scores(self, qa_results: Dict) -> Dict[str, float]:
        """Extracts component scores from QA results.

        Args:
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            A dictionary of component scores.
        """
        try:
            scores = {}
            
            # Visual consistency score
            if 'visual_consistency' in qa_results:
                scores['visual_consistency'] = qa_results['visual_consistency'].get('score', 0)
            else:
                scores['visual_consistency'] = 0.0
            
            # Audio sync score
            if 'audio_sync' in qa_results:
                scores['audio_sync'] = qa_results['audio_sync'].get('score', 0)
            else:
                scores['audio_sync'] = 0.0
            
            # Broadcast compliance score
            if 'broadcast_compliance' in qa_results:
                scores['broadcast_compliance'] = qa_results['broadcast_compliance'].get('score', 0)
            else:
                scores['broadcast_compliance'] = 0.0
            
            # Artifact detection score (inverse - higher score = fewer artifacts)
            if 'artifact_detection' in qa_results:
                artifact_score = qa_results['artifact_detection'].get('score', 0)
                scores['artifact_detection'] = artifact_score
            else:
                scores['artifact_detection'] = 0.0
            
            # Technical quality score (derived from other metrics)
            scores['technical_quality'] = self._calculate_technical_quality_score(qa_results)
            
            # Content quality score (based on detected content characteristics)
            scores['content_quality'] = self._calculate_content_quality_score(qa_results)
            
            # Delivery readiness score
            scores['delivery_readiness'] = self._calculate_delivery_readiness_score(qa_results)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Component score extraction failed: {e}")
            return {}
    
    def _calculate_technical_quality_score(self, qa_results: Dict) -> float:
        """Calculates the technical quality score.

        Args:
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            The technical quality score.
        """
        try:
            # Base technical score on broadcast compliance and artifact detection
            broadcast_score = qa_results.get('broadcast_compliance', {}).get('score', 0)
            artifact_score = qa_results.get('artifact_detection', {}).get('score', 0)
            
            # Combine with additional technical factors
            visual_consistency = qa_results.get('visual_consistency', {}).get('score', 0)
            sync_score = qa_results.get('audio_sync', {}).get('score', 0)
            
            # Weighted combination
            technical_score = (
                broadcast_score * 0.4 +
                artifact_score * 0.3 +
                visual_consistency * 0.2 +
                sync_score * 0.1
            )
            
            return technical_score
            
        except Exception as e:
            self.logger.error(f"Technical quality score calculation failed: {e}")
            return 0.0
    
    def _calculate_content_quality_score(self, qa_results: Dict) -> float:
        """Calculates the content quality score.

        Args:
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            The content quality score.
        """
        try:
            # Base content quality on detected artifacts and consistency
            artifact_score = qa_results.get('artifact_detection', {}).get('score', 0)
            visual_score = qa_results.get('visual_consistency', {}).get('score', 0)
            
            # Adjust for content-specific issues
            artifacts = qa_results.get('artifact_detection', {})
            content_penalty = 0
            
            # Penalty for faces without privacy compliance
            if artifacts.get('faces_detected', 0) > 0:
                privacy_concerns = artifacts.get('privacy_concerns', {})
                if privacy_concerns.get('privacy_score', 1.0) < 0.7:
                    content_penalty += 10
            
            # Penalty for unauthorized watermarks
            if artifacts.get('watermarks_detected', 0) > 0:
                legal_concerns = artifacts.get('legal_concerns', {})
                if legal_concerns.get('requires_review', False):
                    content_penalty += 15
            
            # Calculate final content score
            content_score = min(100, (artifact_score + visual_score) / 2 - content_penalty)
            
            return max(0, content_score)
            
        except Exception as e:
            self.logger.error(f"Content quality score calculation failed: {e}")
            return 0.0
    
    def _calculate_delivery_readiness_score(self, qa_results: Dict) -> float:
        """Calculates the delivery readiness score.

        Args:
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            The delivery readiness score.
        """
        try:
            # Base readiness on meeting delivery standards
            broadcast_score = qa_results.get('broadcast_compliance', {}).get('score', 0)
            artifact_score = qa_results.get('artifact_detection', {}).get('score', 0)
            sync_score = qa_results.get('audio_sync', {}).get('score', 0)
            
            # Check for delivery-blocking issues
            readiness_factors = []
            
            # Broadcast compliance
            if broadcast_score >= 85:
                readiness_factors.append(1.0)
            elif broadcast_score >= 70:
                readiness_factors.append(0.7)
            else:
                readiness_factors.append(0.3)
            
            # Audio sync
            if sync_score >= 90:
                readiness_factors.append(1.0)
            elif sync_score >= 80:
                readiness_factors.append(0.8)
            else:
                readiness_factors.append(0.4)
            
            # Artifact levels
            if artifact_score >= 80:
                readiness_factors.append(1.0)
            elif artifact_score >= 60:
                readiness_factors.append(0.6)
            else:
                readiness_factors.append(0.2)
            
            # Calculate average readiness
            readiness_score = np.mean(readiness_factors) * 100
            
            return readiness_score
            
        except Exception as e:
            self.logger.error(f"Delivery readiness score calculation failed: {e}")
            return 0.0
    
    def _apply_quality_adjustments(self, component_scores: Dict[str, float], qa_results: Dict) -> Dict[str, float]:
        """Applies quality adjustments to component scores.

        Args:
            component_scores: A dictionary of component scores.
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            A dictionary of adjusted component scores.
        """
        try:
            adjusted_scores = component_scores.copy()
            
            # Apply validation rules
            for rule_name, rule in self.validation_rules.items():
                if rule['enabled']:
                    penalty = self._apply_validation_rule(rule, qa_results)
                    if penalty > 0:
                        # Apply penalty to relevant components
                        for component in rule['components']:
                            if component in adjusted_scores:
                                adjusted_scores[component] -= penalty
            
            # Ensure scores stay within bounds
            for component in adjusted_scores:
                adjusted_scores[component] = max(0, min(100, adjusted_scores[component]))
            
            return adjusted_scores
            
        except Exception as e:
            self.logger.error(f"Quality adjustment failed: {e}")
            return component_scores
    
    def _calculate_weighted_score(self, adjusted_scores: Dict[str, float]) -> float:
        """Calculates the weighted overall score.

        Args:
            adjusted_scores: A dictionary of adjusted component scores.

        Returns:
            The weighted overall score.
        """
        try:
            weighted_sum = 0
            total_weight = 0
            
            for component, score in adjusted_scores.items():
                weight = self.component_weights.get(component, 0)
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Weighted score calculation failed: {e}")
            return 0.0
    
    def _calculate_weighted_component_scores(self, adjusted_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculates the weighted scores for each component.

        Args:
            adjusted_scores: A dictionary of adjusted component scores.

        Returns:
            A dictionary of weighted component scores.
        """
        try:
            weighted_scores = {}
            
            for component, score in adjusted_scores.items():
                weight = self.component_weights.get(component, 0)
                weighted_scores[component] = score * weight
            
            return weighted_scores
            
        except Exception as e:
            self.logger.error(f"Weighted component score calculation failed: {e}")
            return {}
    
    def _apply_quality_bonuses(self, base_score: float, qa_results: Dict) -> float:
        """Applies quality bonuses for exceptional performance.

        Args:
            base_score: The base score to apply bonuses to.
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            The score with bonuses applied.
        """
        try:
            bonus_score = base_score
            
            # Excellence bonus
            if base_score >= self.bonus_criteria['excellent_quality']:
                bonus_score += 5
            
            # Broadcast ready bonus
            if self._is_broadcast_ready(qa_results):
                bonus_score += 3
            
            # Minimal artifacts bonus
            artifacts = qa_results.get('artifact_detection', {})
            if artifacts.get('total_artifacts', 100) < 5:  # Less than 5 artifacts
                bonus_score += 2
            
            # Perfect sync bonus
            sync_results = qa_results.get('audio_sync', {})
            if sync_results.get('time_offset_ms', 100) < 10:  # Less than 10ms offset
                bonus_score += 2
            
            return min(100, bonus_score)
            
        except Exception as e:
            self.logger.error(f"Quality bonus application failed: {e}")
            return base_score
    
    def _calculate_penalties(self, component_scores: Dict[str, float], qa_results: Dict) -> Dict[str, float]:
        """Calculates penalties for quality issues.

        Args:
            component_scores: A dictionary of component scores.
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            A dictionary of penalties.
        """
        try:
            penalties = {}
            
            # Critical failure penalty
            if component_scores.get('broadcast_compliance', 100) < self.penalty_thresholds['critical_failure']:
                penalties['broadcast_compliance_failure'] = 20
            
            # Major sync issue penalty
            sync_results = qa_results.get('audio_sync', {})
            if sync_results.get('time_offset_ms', 0) > 50:  # More than 50ms offset
                penalties['major_sync_issue'] = 15
            
            # High artifact penalty
            artifacts = qa_results.get('artifact_detection', {})
            if artifacts.get('total_artifacts', 0) > 20:
                penalties['excessive_artifacts'] = 10
            
            # Privacy issue penalty
            if artifacts.get('faces_detected', 0) > 0:
                privacy_score = artifacts.get('privacy_concerns', {}).get('privacy_score', 1.0)
                if privacy_score < 0.5:
                    penalties['privacy_concern'] = 25
            
            return penalties
            
        except Exception as e:
            self.logger.error(f"Penalty calculation failed: {e}")
            return {}
    
    def _calculate_bonuses(self, component_scores: Dict[str, float], qa_results: Dict) -> Dict[str, float]:
        """Calculates bonuses for exceptional performance.

        Args:
            component_scores: A dictionary of component scores.
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            A dictionary of bonuses.
        """
        try:
            bonuses = {}
            
            # Excellence bonuses
            if component_scores.get('visual_consistency', 0) > 95:
                bonuses['excellent_consistency'] = 5
            
            if component_scores.get('audio_sync', 0) > 95:
                bonuses['perfect_sync'] = 5
            
            if component_scores.get('broadcast_compliance', 0) > 95:
                bonuses['broadcast_excellence'] = 3
            
            # Compliance bonuses
            if self._is_broadcast_ready(qa_results):
                bonuses['broadcast_ready'] = 5
            
            # Low artifact bonus
            artifacts = qa_results.get('artifact_detection', {})
            if artifacts.get('total_artifacts', 100) == 0:
                bonuses['artifact_free'] = 5
            
            return bonuses
            
        except Exception as e:
            self.logger.error(f"Bonus calculation failed: {e}")
            return {}
    
    def _calculate_confidence_level(self, qa_results: Dict) -> float:
        """Calculates the confidence level in the quality assessment.

        Args:
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            The confidence level of the quality assessment.
        """
        try:
            confidence_factors = []
            
            # Factor 1: Completeness of analysis
            expected_components = ['visual_consistency', 'audio_sync', 'broadcast_compliance', 'artifact_detection']
            analyzed_components = sum(1 for comp in expected_components if comp in qa_results)
            completeness = analyzed_components / len(expected_components)
            confidence_factors.append(completeness)
            
            # Factor 2: Consistency between measurements
            if len(qa_results) > 1:
                scores = []
                for component_results in qa_results.values():
                    if isinstance(component_results, dict) and 'score' in component_results:
                        scores.append(component_results['score'])
                
                if len(scores) > 1:
                    score_variance = np.var(scores)
                    consistency = max(0, 1 - score_variance / 1000)  # Normalize variance
                    confidence_factors.append(consistency)
            
            # Factor 3: Quality of individual measurements
            measurement_confidence = 0
            measurements_count = 0
            
            for component_results in qa_results.values():
                if isinstance(component_results, dict):
                    # Check for confidence indicators
                    if 'confidence' in component_results:
                        measurement_confidence += component_results['confidence']
                        measurements_count += 1
                    elif 'has_error' not in component_results:  # Assume reasonable confidence if no error
                        measurement_confidence += 0.8
                        measurements_count += 1
            
            if measurements_count > 0:
                avg_measurement_confidence = measurement_confidence / measurements_count
                confidence_factors.append(avg_measurement_confidence)
            
            # Calculate overall confidence
            if confidence_factors:
                return np.mean(confidence_factors)
            else:
                return 0.5  # Neutral confidence
                
        except Exception as e:
            self.logger.error(f"Confidence level calculation failed: {e}")
            return 0.5
    
    def _analyze_score_trend(self) -> str:
        """Analyzes the score trend over time.

        Returns:
            The score trend (improving, declining, or stable).
        """
        try:
            if len(self.score_history) < 3:
                return "stable"
            
            recent_scores = self.score_history[-3:]  # Last 3 scores
            score_values = [score['overall_score'] for score in recent_scores]
            
            if len(score_values) >= 3:
                # Simple trend detection
                if score_values[-1] > score_values[-2] > score_values[-3]:
                    return "improving"
                elif score_values[-1] < score_values[-2] < score_values[-3]:
                    return "declining"
                else:
                    return "stable"
            else:
                return "stable"
                
        except Exception as e:
            self.logger.error(f"Score trend analysis failed: {e}")
            return "unknown"
    
    def _identify_critical_issues(self, component_scores: Dict[str, float], qa_results: Dict) -> List[str]:
        """Identifies critical quality issues.

        Args:
            component_scores: A dictionary of component scores.
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            A list of critical quality issues.
        """
        try:
            critical_issues = []
            
            # Check component scores
            for component, score in component_scores.items():
                if score < self.penalty_thresholds['critical_failure']:
                    critical_issues.append(f"Critical failure in {component}: {score:.1f}")
            
            # Check specific quality issues
            broadcast_results = qa_results.get('broadcast_compliance', {})
            if not broadcast_results.get('compliance', BroadcastCompliance).ire_levels_compliant:
                critical_issues.append("IRE level violations detected")
            
            sync_results = qa_results.get('audio_sync', {})
            if sync_results.get('has_desync', False):
                offset = sync_results.get('time_offset_ms', 0)
                if offset > 100:  # More than 100ms offset
                    critical_issues.append(f"Major audio sync issue: {offset:.1f}ms")
            
            artifacts = qa_results.get('artifact_detection', {})
            if artifacts.get('faces_detected', 0) > 0:
                privacy_score = artifacts.get('privacy_concerns', {}).get('privacy_score', 1.0)
                if privacy_score < 0.3:
                    critical_issues.append("High privacy risk: faces without proper consent")
            
            if not critical_issues:
                critical_issues.append("No critical issues detected")
            
            return critical_issues
            
        except Exception as e:
            self.logger.error(f"Critical issue identification failed: {e}")
            return ["Issue analysis failed"]
    
    def _identify_improvement_opportunities(self, component_scores: Dict[str, float]) -> List[str]:
        """Identifies improvement opportunities.

        Args:
            component_scores: A dictionary of component scores.

        Returns:
            A list of improvement opportunities.
        """
        try:
            opportunities = []
            
            # Find lowest-scoring components
            sorted_components = sorted(component_scores.items(), key=lambda x: x[1])
            
            for component, score in sorted_components[:3]:  # Top 3 improvement opportunities
                if score < 80:
                    if component == 'visual_consistency':
                        opportunities.append("Improve visual style consistency across shots")
                    elif component == 'audio_sync':
                        opportunities.append("Refine audio-visual synchronization")
                    elif component == 'broadcast_compliance':
                        opportunities.append("Apply broadcast legalizer for compliance")
                    elif component == 'artifact_detection':
                        opportunities.append("Reduce artifacts through quality enhancement")
                    elif component == 'technical_quality':
                        opportunities.append("Improve technical quality parameters")
                    elif component == 'content_quality':
                        opportunities.append("Enhance content quality and clarity")
                    elif component == 'delivery_readiness':
                        opportunities.append("Optimize for delivery standards")
            
            # Add general opportunities
            min_score = min(component_scores.values()) if component_scores else 0
            if min_score < 70:
                opportunities.append("Overall quality improvement needed across all components")
            
            if not opportunities:
                opportunities.append("Quality is already at a good level")
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Improvement opportunity identification failed: {e}")
            return ["Improvement analysis failed"]
    
    def _is_broadcast_ready(self, qa_results: Dict) -> bool:
        """Checks if the video is broadcast ready.

        Args:
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            True if the video is broadcast ready, False otherwise.
        """
        try:
            # Check key broadcast requirements
            broadcast_score = qa_results.get('broadcast_compliance', {}).get('score', 0)
            sync_score = qa_results.get('audio_sync', {}).get('score', 0)
            artifact_score = qa_results.get('artifact_detection', {}.get('score', 0))
            
            # Broadcast ready criteria
            return (
                broadcast_score >= 85 and
                sync_score >= 90 and
                artifact_score >= 80
            )
            
        except Exception as e:
            self.logger.error(f"Broadcast readiness check failed: {e}")
            return False
    
    def _apply_validation_rule(self, rule: Dict, qa_results: Dict) -> float:
        """Applies a validation rule and returns the penalty amount.

        This is a placeholder implementation.

        Args:
            rule: The validation rule to apply.
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            The penalty amount.
        """
        try:
            # This would contain the logic for specific validation rules
            # For now, return 0 (no penalty)
            return 0
            
        except Exception as e:
            self.logger.error(f"Validation rule application failed: {e}")
            return 0
    
    def _store_score_for_trend_analysis(self, overall_score: float, component_scores: Dict[str, float], qa_results: Dict):
        """Stores the score for trend analysis.

        Args:
            overall_score: The overall quality score.
            component_scores: A dictionary of component scores.
            qa_results: A dictionary containing the results from all QA
                components.
        """
        try:
            self.score_history.append({
                'timestamp': str(np.datetime64('now')),
                'overall_score': overall_score,
                'component_scores': component_scores.copy()
            })
            
            # Keep only last 50 scores for memory efficiency
            if len(self.score_history) > 50:
                self.score_history = self.score_history[-50:]
            
            # Store component history
            self.component_history.append({
                'timestamp': str(np.datetime64('now')),
                'components': component_scores.copy()
            })
            
            if len(self.component_history) > 50:
                self.component_history = self.component_history[-50:]
                
        except Exception as e:
            self.logger.error(f"Score storage for trend analysis failed: {e}")
    
    def _initialize_validation_rules(self) -> Dict:
        """Initializes the quality validation rules.

        Returns:
            A dictionary of quality validation rules.
        """
        return {
            'minimum_ire_compliance': {
                'enabled': True,
                'components': ['broadcast_compliance'],
                'penalty': 15,
                'description': 'IRE level must be within broadcast standards'
            },
            'sync_tolerance': {
                'enabled': True,
                'components': ['audio_sync'],
                'penalty': 20,
                'description': 'Audio sync must be within acceptable tolerance'
            },
            'artifact_threshold': {
                'enabled': True,
                'components': ['artifact_detection'],
                'penalty': 10,
                'description': 'Artifact levels must be below threshold'
            },
            'consistency_requirement': {
                'enabled': True,
                'components': ['visual_consistency'],
                'penalty': 8,
                'description': 'Visual consistency must meet standards'
            }
        }
    
    def get_quality_benchmark_scores(self) -> Dict[str, float]:
        """Gets the scores against different quality benchmarks.

        Returns:
            A dictionary of benchmark scores.
        """
        try:
            current_overall = self.score_history[-1]['overall_score'] if self.score_history else 0
            
            benchmark_comparison = {}
            for benchmark, threshold in self.quality_benchmarks.items():
                benchmark_comparison[benchmark] = {
                    'threshold': threshold,
                    'meets_standard': current_overall >= threshold,
                    'score_gap': max(0, threshold - current_overall),
                    'performance': 'exceeds' if current_overall >= threshold else 'below'
                }
            
            return benchmark_comparison
            
        except Exception as e:
            self.logger.error(f"Benchmark score calculation failed: {e}")
            return {}
    
    def generate_quality_report(self, qa_results: Dict) -> str:
        """Generates a comprehensive quality report.

        Args:
            qa_results: A dictionary containing the results from all QA
                components.

        Returns:
            A string containing the quality report.
        """
        try:
            breakdown = self.calculate_detailed_score_breakdown(qa_results)
            benchmarks = self.get_quality_benchmark_scores()
            
            report = f"""
# COMPREHENSIVE QUALITY REPORT

## Overall Quality Assessment
- **Overall Score**: {breakdown.overall_score:.1f}/100
- **Quality Level**: {self._get_quality_level(breakdown.overall_score)}
- **Trend**: {breakdown.score_trend}
- **Confidence**: {breakdown.confidence_level:.1%}

## Component Score Breakdown
{chr(10).join(f"- **{component.replace('_', ' ').title()}**: {score:.1f}/100" for component, score in breakdown.component_scores.items())}

## Weighted Component Scores
{chr(10).join(f"- **{component.replace('_', ' ').title()}**: {score:.1f}/100" for component, score in breakdown.weighted_scores.items())}

## Quality Benchmarks
{chr(10).join(f"- **{benchmark.replace('_', ' ').title()}**: {data['performance']} (needs {data['score_gap']:.1f} points)" for benchmark, data in benchmarks.items())}

## Critical Issues
{chr(10).join(f"- {issue}" for issue in breakdown.critical_issues)}

## Improvement Opportunities
{chr(10).join(f"- {opportunity}" for opportunity in breakdown.improvement_opportunities)}

## Quality Assessment Summary
{self._generate_quality_summary(breakdown)}

---
Report generated by APEX DIRECTOR Quality Score Calculator
            """
            return report.strip()
            
        except Exception as e:
            self.logger.error(f"Quality report generation failed: {e}")
            return f"Quality report generation failed: {e}"
    
    def _get_quality_level(self, score: float) -> str:
        """Gets a quality level description based on a score.

        Args:
            score: The score to get the quality level for.

        Returns:
            The quality level description.
        """
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
    
    def _generate_quality_summary(self, breakdown: QualityScoreBreakdown) -> str:
        """Generates a quality assessment summary.

        Args:
            breakdown: The detailed quality score breakdown.

        Returns:
            A string containing the quality assessment summary.
        """
        try:
            level = self._get_quality_level(breakdown.overall_score)
            
            if breakdown.overall_score >= 85:
                summary = f"Video demonstrates {level.lower()} quality and is ready for professional use."
            elif breakdown.overall_score >= 75:
                summary = f"Video shows {level.lower()} quality with room for minor improvements."
            elif breakdown.overall_score >= 65:
                summary = f"Video has {level.lower()} quality but requires attention to several areas."
            else:
                summary = f"Video quality needs significant improvement before professional use."
            
            # Add trend information
            if breakdown.score_trend == "improving":
                summary += " Quality is trending upward."
            elif breakdown.score_trend == "declining":
                summary += " Quality is trending downward - review recent changes."
            
            # Add confidence level information
            if breakdown.confidence_level > 0.8:
                summary += " High confidence in assessment results."
            elif breakdown.confidence_level < 0.5:
                summary += " Low confidence - manual review recommended."
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Quality summary generation failed: {e}")
            return "Quality assessment summary unavailable."
    
    def _default_config(self) -> Dict:
        """Returns the default configuration for the quality score calculator.

        Returns:
            A dictionary of default configuration parameters.
        """
        return {
            'weights': {
                'visual_consistency': 0.25,
                'audio_sync': 0.20,
                'broadcast_compliance': 0.20,
                'artifact_detection': 0.15,
                'technical_quality': 0.10,
                'content_quality': 0.07,
                'delivery_readiness': 0.03
            },
            'penalty_thresholds': {
                'critical_failure': 50.0,
                'major_issue': 70.0,
                'minor_issue': 85.0
            },
            'bonus_criteria': {
                'excellent_quality': 90.0,
                'broadcast_ready': 85.0,
                'minimal_artifacts': 80.0
            },
            'quality_benchmarks': {
                'broadcast_tv': 80.0,
                'streaming': 75.0,
                'social_media': 70.0,
                'professional': 85.0,
                'premium': 90.0
            }
        }


# Additional utility functions for quality assessment

def calculate_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        image1: The first image.
        image2: The second image.

    Returns:
        The PSNR value.
    """
    try:
        mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    except:
        return 0.0


def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculates the Structural Similarity Index (SSIM) between two images.

    This is a simplified implementation.

    Args:
        image1: The first image.
        image2: The second image.

    Returns:
        The SSIM value.
    """
    try:
        # Simplified SSIM calculation
        mu1 = cv2.GaussianBlur(image1.astype(float), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(image2.astype(float), (11, 11), 1.5)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(image1 * image1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(image2 * image2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(image1 * image2, (11, 11), 1.5) - mu1_mu2
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return max(0, min(1, ssim))
    except:
        return 0.0


def calculate_vmaf_score(reference: np.ndarray, test: np.ndarray) -> float:
    """Calculates the VMAF (Video Multi-method Assessment Fusion) score.

    This is a simplified implementation that uses SSIM and PSNR as proxies.

    Args:
        reference: The reference image.
        test: The test image.

    Returns:
        The VMAF score.
    """
    try:
        # Simplified VMAF calculation
        # In a real implementation, would use the actual VMAF library
        
        # Calculate SSIM as proxy
        ssim_score = calculate_ssim(reference, test)
        
        # Calculate PSNR as proxy
        psnr_score = calculate_psnr(reference, test)
        psnr_normalized = min(1.0, psnr_score / 40)  # Normalize PSNR to 0-1
        
        # Combine scores (VMAF formula approximation)
        vmaf_score = (0.5 * ssim_score + 0.5 * psnr_normalized) * 100
        
        return min(100, max(0, vmaf_score))
    except:
        return 0.0


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Create sample QA results
    sample_qa_results = {
        'visual_consistency': {'score': 85.5},
        'audio_sync': {'score': 92.3, 'time_offset_ms': 15.2},
        'broadcast_compliance': {'score': 88.7},
        'artifact_detection': {'score': 91.2, 'total_artifacts': 3}
    }
    
    calculator = QualityScoreCalculator()
    
    # Calculate overall score
    overall_score = calculator.calculate_overall_score(sample_qa_results)
    print(f"Overall Quality Score: {overall_score:.1f}/100")
    
    # Calculate detailed breakdown
    breakdown = calculator.calculate_detailed_score_breakdown(sample_qa_results)
    print(f"Component Scores: {breakdown.component_scores}")
    print(f"Critical Issues: {breakdown.critical_issues}")
    print(f"Improvement Opportunities: {breakdown.improvement_opportunities}")
    
    # Generate comprehensive report
    report = calculator.generate_quality_report(sample_qa_results)
    print("\n" + report)
