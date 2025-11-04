"""
APEX DIRECTOR Cost & Time Estimator
Predicts resource requirements for image generation jobs
"""

import time
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .config import get_config, EstimatorConfig
from .backend_manager import get_backend_manager

logger = logging.getLogger(__name__)


@dataclass
class HistoricalRecord:
    """Represents a historical data point for a completed generation job.

    Attributes:
        timestamp: The timestamp when the job was completed.
        backend: The name of the backend used for the job.
        width: The width of the generated image.
        height: The height of the generated image.
        steps: The number of generation steps used.
        quality_level: The quality level of the generation.
        actual_time: The actual time taken to complete the job, in seconds.
        actual_cost: The actual cost of the job.
        success: A boolean indicating whether the job was successful.
        prompt_length: The length of the prompt used.
        complexity_score: A score representing the complexity of the request.
    """
    timestamp: datetime
    backend: str
    width: int
    height: int
    steps: int
    quality_level: int
    actual_time: float
    actual_cost: float
    success: bool
    prompt_length: int = 0
    complexity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the HistoricalRecord to a dictionary.

        Returns:
            A dictionary representation of the historical record.
        """
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoricalRecord':
        """Creates a HistoricalRecord instance from a dictionary.

        Args:
            data: A dictionary containing historical record data.

        Returns:
            An instance of HistoricalRecord.
        """
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class CostEstimate:
    """Represents an estimated cost and time for a generation request.

    Attributes:
        estimate_id: A unique identifier for this estimate.
        job_params: The parameters of the job for which the estimate was
            generated.
        estimated_cost: The estimated cost of the job.
        estimated_time_seconds: The estimated time to complete the job, in
            seconds.
        confidence_score: A score from 0.0 to 1.0 representing the
            confidence in the estimate.
        factors: A dictionary of factors that influenced the estimate.
        backend_suggestions: A list of recommended backends for the job.
        created_at: The timestamp when the estimate was created.
        expires_at: The timestamp when the estimate expires.
    """
    estimate_id: str
    job_params: Dict[str, Any]
    estimated_cost: float
    estimated_time_seconds: float
    confidence_score: float  # 0.0 to 1.0
    factors: Dict[str, float]
    backend_suggestions: List[str]  # Recommended backends
    created_at: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the CostEstimate to a dictionary.

        Returns:
            A dictionary representation of the cost estimate.
        """
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CostEstimate':
        """Creates a CostEstimate instance from a dictionary.

        Args:
            data: A dictionary containing cost estimate data.

        Returns:
            An instance of CostEstimate.
        """
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'expires_at' in data:
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)


class EstimationEngine:
    """The core engine for estimating job costs and times.

    This class uses historical data and algorithmic models to predict the
    resource requirements for generation jobs.
    """
    
    def __init__(self, config: EstimatorConfig):
        """Initializes the EstimationEngine.

        Args:
            config: The configuration for the estimator.
        """
        self.config = config
        self.historical_data: List[HistoricalRecord] = []
        self.cache: Dict[str, CostEstimate] = {}
        self.cache_ttl = 3600  # 1 hour in seconds
        
        # Load historical data
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Loads historical performance data from a file."""
        try:
            data_dir = Path("assets/metadata")
            historical_file = data_dir / "estimation_history.json"
            
            if historical_file.exists():
                with open(historical_file, 'r') as f:
                    data = json.load(f)
                    self.historical_data = [
                        HistoricalRecord.from_dict(record)
                        for record in data
                    ]
                logger.info(f"Loaded {len(self.historical_data)} historical records")
        
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
            self.historical_data = []
    
    def _save_historical_data(self):
        """Saves the historical performance data to a file."""
        try:
            data_dir = Path("assets/metadata")
            data_dir.mkdir(parents=True, exist_ok=True)
            historical_file = data_dir / "estimation_history.json"
            
            data = [record.to_dict() for record in self.historical_data]
            
            with open(historical_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save historical data: {e}")
    
    def _generate_estimate_id(self, job_params: Dict[str, Any]) -> str:
        """Generates a unique ID for an estimate based on job parameters.

        This ID is used as a cache key.

        Args:
            job_params: The parameters of the job.

        Returns:
            A unique hexadecimal ID string.
        """
        # Create a stable hash from job parameters
        key_parts = []
        for key in ['backend', 'width', 'height', 'steps', 'quality_level']:
            key_parts.append(f"{key}:{job_params.get(key, 'default')}")
        
        import hashlib
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _is_estimate_valid(self, estimate: CostEstimate) -> bool:
        """Checks if a cached estimate is still valid.

        Args:
            estimate: The CostEstimate to check.

        Returns:
            True if the estimate is still valid, False otherwise.
        """
        return datetime.utcnow() < estimate.expires_at
    
    def _calculate_complexity_score(self, prompt: str, width: int, height: int, steps: int) -> float:
        """Calculates a complexity score for a generation request.

        The score is based on factors like image size, number of steps, and
        the complexity of the prompt.

        Args:
            prompt: The text prompt.
            width: The width of the image.
            height: The height of the image.
            steps: The number of generation steps.

        Returns:
            A float representing the complexity score.
        """
        # Base complexity factors
        base_score = 1.0
        
        # Image size complexity
        size_complexity = (width * height) / (512 * 512)
        
        # Steps complexity
        steps_complexity = steps / 20.0  # Normalize around 20 steps
        
        # Prompt complexity (length and complexity indicators)
        prompt_length_factor = min(len(prompt) / 100, 2.0)  # Cap at 2x for very long prompts
        
        # Complexity keywords
        complex_keywords = [
            'detailed', 'complex', 'intricate', 'multiple', 'various',
            'crowd', 'landscape', 'architecture', 'mechanical', 'abstract'
        ]
        keyword_factor = sum(1 for keyword in complex_keywords if keyword.lower() in prompt.lower())
        
        # Final complexity score
        complexity = base_score * (1 + size_complexity * 0.5) * steps_complexity * (1 + keyword_factor * 0.2)
        
        return min(complexity, 5.0)  # Cap at 5.0
    
    def _get_similar_records(self, job_params: Dict[str, Any], limit: int = 50) -> List[HistoricalRecord]:
        """Retrieves historical records that are similar to a given job.

        Similarity is scored based on backend, resolution, steps, and quality
        level.

        Args:
            job_params: The parameters of the job to find similar records
                for.
            limit: The maximum number of similar records to return.

        Returns:
            A list of the most similar HistoricalRecord objects.
        """
        similar_records = []
        
        for record in self.historical_data:
            if not record.success:
                continue
            
            # Score based on similarity
            score = 0.0
            max_score = 0.0
            
            # Backend similarity
            if job_params.get('backend') == record.backend:
                score += 3.0
            max_score += 3.0
            
            # Resolution similarity
            target_resolution = job_params.get('width', 512) * job_params.get('height', 512)
            record_resolution = record.width * record.height
            resolution_diff = abs(target_resolution - record_resolution) / target_resolution
            resolution_score = max(0, 1 - resolution_diff)
            score += resolution_score * 2.0
            max_score += 2.0
            
            # Steps similarity
            target_steps = job_params.get('steps', 20)
            steps_diff = abs(target_steps - record.steps) / max(target_steps, record.steps)
            steps_score = max(0, 1 - steps_diff)
            score += steps_score * 1.5
            max_score += 1.5
            
            # Quality level similarity
            target_quality = job_params.get('quality_level', 3)
            quality_diff = abs(target_quality - record.quality_level)
            quality_score = max(0, 1 - quality_diff / 2.0)  # Allow some difference
            score += quality_score * 1.0
            max_score += 1.0
            
            # Normalize score
            if max_score > 0:
                normalized_score = score / max_score
                if normalized_score > 0.5:  # Only include reasonably similar records
                    similar_records.append((record, normalized_score))
        
        # Sort by similarity score and return top matches
        similar_records.sort(key=lambda x: x[1], reverse=True)
        return [record for record, score in similar_records[:limit]]
    
    def _calculate_baseline_estimate(self, job_params: Dict[str, Any]) -> Tuple[float, float]:
        """Calculates baseline time and cost estimates for a job.

        This method provides an initial estimate based on backend
        configurations and job parameters, without considering historical
        data.

        Args:
            job_params: The parameters of the job.

        Returns:
            A tuple containing the estimated time and cost.
        """
        backend_manager = get_backend_manager()
        
        # Get backend configuration
        backend_name = job_params.get('backend', '')
        backend_config = None
        for config in backend_manager.configs:
            if config.name == backend_name:
                backend_config = config
                break
        
        if not backend_config:
            # Use average of all backends
            all_costs = [config.cost_per_image for config in backend_manager.configs if config.enabled]
            avg_cost = sum(all_costs) / len(all_costs) if all_costs else 0.05
            base_cost = avg_cost
            base_time = 5.0  # Default 5 seconds
        else:
            base_cost = backend_config.cost_per_image
            base_time = 10.0  # Default baseline time
        
        # Adjust for parameters
        width = job_params.get('width', 512)
        height = job_params.get('height', 512)
        steps = job_params.get('steps', 20)
        quality_level = job_params.get('quality_level', 3)
        prompt = job_params.get('prompt', '')
        
        # Size adjustment
        size_multiplier = (width * height) / (512 * 512)
        
        # Steps adjustment
        steps_multiplier = steps / 20.0
        
        # Quality adjustment
        quality_multiplier = quality_level / 3.0
        
        # Complexity adjustment
        complexity = self._calculate_complexity_score(prompt, width, height, steps)
        
        # Final estimates
        estimated_time = base_time * size_multiplier * steps_multiplier * quality_multiplier * complexity
        estimated_cost = base_cost * size_multiplier * steps_multiplier * quality_multiplier
        
        return estimated_time, estimated_cost
    
    def _calculate_confidence_score(self, similar_records: List[HistoricalRecord], 
                                   baseline_time: float, baseline_cost: float) -> float:
        """Calculates a confidence score for an estimate.

        The score is based on the quantity and consistency of similar
        historical data.

        Args:
            similar_records: A list of historical records similar to the
                current job.
            baseline_time: The baseline time estimate.
            baseline_cost: The baseline cost estimate.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        if not similar_records:
            return 0.3  # Low confidence with no data
        
        # Base confidence on number of similar records
        data_confidence = min(len(similar_records) / 20, 1.0)  # Full confidence at 20+ records
        
        # Check consistency of historical data
        if len(similar_records) >= 3:
            times = [record.actual_time for record in similar_records]
            costs = [record.actual_cost for record in similar_records]
            
            # Calculate coefficient of variation (lower = more consistent)
            time_cv = statistics.stdev(times) / statistics.mean(times) if times else 1.0
            cost_cv = statistics.stdev(costs) / statistics.mean(costs) if costs else 1.0
            
            # Consistency score (lower variation = higher consistency)
            consistency_score = max(0, 1 - (time_cv + cost_cv) / 2)
        else:
            consistency_score = 0.5
        
        # Time of day factor (avoid estimates made during unusual hours)
        current_hour = datetime.utcnow().hour
        time_factor = 0.9 if 8 <= current_hour <= 22 else 0.7
        
        # Final confidence score
        confidence = data_confidence * 0.6 + consistency_score * 0.3 + time_factor * 0.1
        return min(confidence, 0.95)  # Cap at 95%
    
    def estimate_generation(self, job_params: Dict[str, Any], prompt: str) -> CostEstimate:
        """Generates a comprehensive cost and time estimate for a job.

        This method combines baseline calculations with historical data to
        provide a refined estimate with a confidence score.

        Args:
            job_params: The parameters of the job.
            prompt: The text prompt for the job.

        Returns:
            A CostEstimate object with the detailed estimate.
        """
        
        # Add prompt to job parameters
        job_params_with_prompt = job_params.copy()
        job_params_with_prompt['prompt'] = prompt
        
        # Check cache first
        estimate_id = self._generate_estimate_id(job_params_with_prompt)
        
        if self.config.cache_estimates and estimate_id in self.cache:
            cached_estimate = self.cache[estimate_id]
            if self._is_estimate_valid(cached_estimate):
                logger.debug(f"Using cached estimate: {estimate_id}")
                return cached_estimate
        
        # Get similar historical records
        similar_records = self._get_similar_records(job_params_with_prompt)
        
        # Calculate baseline estimates
        baseline_time, baseline_cost = self._calculate_baseline_estimate(job_params_with_prompt)
        
        # Adjust based on historical data if available
        if similar_records:
            # Weight historical average with baseline
            historical_times = [record.actual_time for record in similar_records]
            historical_costs = [record.actual_cost for record in similar_records]
            
            avg_historical_time = statistics.mean(historical_times)
            avg_historical_cost = statistics.mean(historical_costs)
            
            # Weight historical data more heavily as we get more records
            weight = min(len(similar_records) / 10, 0.8)  # Max 80% weight to historical
            
            final_time = baseline_time * (1 - weight) + avg_historical_time * weight
            final_cost = baseline_cost * (1 - weight) + avg_historical_cost * weight
        else:
            final_time = baseline_time
            final_cost = baseline_cost
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(similar_records, baseline_time, baseline_cost)
        
        # Apply tolerance buffer
        time_buffer = final_time * self.config.estimate_tolerance
        cost_buffer = final_cost * self.config.cost_buffer
        
        final_time += time_buffer
        final_cost += cost_buffer
        
        # Get backend suggestions
        backend_manager = get_backend_manager()
        suggested_backends = []
        
        # Score backends based on estimated efficiency
        for config in backend_manager.configs:
            if not config.enabled:
                continue
            
            # Calculate efficiency score for this backend
            backend_params = job_params_with_prompt.copy()
            backend_params['backend'] = config.name
            
            backend_time, backend_cost = self._calculate_baseline_estimate(backend_params)
            
            # Efficiency score (lower time and cost = higher score)
            efficiency_score = 1.0 / (backend_time + backend_cost + 0.1)
            suggested_backends.append((config.name, efficiency_score))
        
        # Sort by efficiency and take top 3
        suggested_backends.sort(key=lambda x: x[1], reverse=True)
        backend_suggestions = [backend for backend, score in suggested_backends[:3]]
        
        # Create estimate
        estimate = CostEstimate(
            estimate_id=estimate_id,
            job_params=job_params_with_prompt,
            estimated_cost=final_cost,
            estimated_time_seconds=final_time,
            confidence_score=confidence_score,
            factors={
                "complexity_score": self._calculate_complexity_score(prompt, 
                                                                    job_params.get('width', 512),
                                                                    job_params.get('height', 512),
                                                                    job_params.get('steps', 20)),
                "historical_weight": min(len(similar_records) / 10, 0.8),
                "baseline_time": baseline_time,
                "baseline_cost": baseline_cost,
                "buffer_applied": True
            },
            backend_suggestions=backend_suggestions,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=self.cache_ttl)
        )
        
        # Cache the estimate
        if self.config.cache_estimates:
            self.cache[estimate_id] = estimate
        
        logger.info(f"Generated estimate {estimate_id}: ${final_cost:.4f} in {final_time:.1f}s (confidence: {confidence_score:.2f})")
        
        return estimate
    
    def add_historical_record(self, record: HistoricalRecord):
        """Adds a new historical record to the dataset.

        Args:
            record: The HistoricalRecord to add.
        """
        self.historical_data.append(record)
        
        # Keep only recent records to manage memory
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.historical_data_retention)
        self.historical_data = [
            record for record in self.historical_data 
            if record.timestamp > cutoff_date
        ]
        
        # Save updated data
        self._save_historical_data()
        
        # Clear relevant cache entries
        self._clear_relevant_cache(record)
    
    def _clear_relevant_cache(self, record: HistoricalRecord):
        """Clears cache entries that might be affected by new data.

        For simplicity, this method currently clears the entire cache.

        Args:
            record: The new HistoricalRecord that was added.
        """
        # Simple approach: clear all cache (could be optimized)
        if self.config.cache_estimates:
            self.cache.clear()
    
    def batch_estimate(self, requests: List[Dict[str, Any]]) -> List[CostEstimate]:
        """Generates estimates for multiple requests efficiently.

        Args:
            requests: A list of job request dictionaries.

        Returns:
            A list of CostEstimate objects corresponding to the requests.
        """
        estimates = []
        
        for request in requests:
            prompt = request.get('prompt', '')
            job_params = {k: v for k, v in request.items() if k != 'prompt'}
            estimate = self.estimate_generation(job_params, prompt)
            estimates.append(estimate)
        
        return estimates
    
    def get_estimation_statistics(self) -> Dict[str, Any]:
        """Gets statistics about the estimation engine's performance.

        Returns:
            A dictionary of statistics, including the number of historical
            records, data variance, and cache performance.
        """
        if not self.historical_data:
            return {
                "total_records": 0,
                "average_accuracy": 0.0,
                "average_time_variance": 0.0,
                "average_cost_variance": 0.0,
                "cache_hit_rate": 0.0
            }
        
        # Calculate statistics
        successful_records = [r for r in self.historical_data if r.success]
        
        if not successful_records:
            return {"total_records": len(self.historical_data)}
        
        # Average values
        avg_time = statistics.mean(r.actual_time for r in successful_records)
        avg_cost = statistics.mean(r.actual_cost for r in successful_records)
        
        # Variances (coefficient of variation)
        times = [r.actual_time for r in successful_records]
        costs = [r.actual_cost for r in successful_records]
        
        time_variance = statistics.stdev(times) / avg_time if avg_time > 0 else 0
        cost_variance = statistics.stdev(costs) / avg_cost if avg_cost > 0 else 0
        
        # Backend distribution
        backend_counts = {}
        for record in successful_records:
            backend_counts[record.backend] = backend_counts.get(record.backend, 0) + 1
        
        # Cache hit rate (simulated)
        cache_hits = len(self.cache)
        total_requests = len(self.historical_data) + cache_hits
        cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "total_records": len(self.historical_data),
            "successful_records": len(successful_records),
            "average_time_seconds": avg_time,
            "average_cost": avg_cost,
            "time_variance": time_variance,
            "cost_variance": cost_variance,
            "backend_distribution": backend_counts,
            "cache_size": len(self.cache),
            "cache_hit_rate": cache_hit_rate
        }


# Global estimation engine instance
_estimation_engine: Optional[EstimationEngine] = None


def get_estimator() -> EstimationEngine:
    """Gets the global instance of the EstimationEngine.

    This function implements a singleton pattern to ensure that only one
    instance of the estimation engine exists.

    Returns:
        The global EstimationEngine instance.
    """
    global _estimation_engine
    if _estimation_engine is None:
        config = get_config().get_estimator_config()
        _estimation_engine = EstimationEngine(config)
    return _estimation_engine


# Convenience functions
def estimate_generation_cost_time(job_params: Dict[str, Any], prompt: str) -> CostEstimate:
    """A convenience function to estimate generation cost and time.

    Args:
        job_params: The parameters of the job.
        prompt: The text prompt for the job.

    Returns:
        A CostEstimate object with the detailed estimate.
    """
    return get_estimator().estimate_generation(job_params, prompt)


def batch_estimate_generation(requests: List[Dict[str, Any]]) -> List[CostEstimate]:
    """A convenience function to estimate multiple generations in a batch.

    Args:
        requests: A list of job request dictionaries.

    Returns:
        A list of CostEstimate objects.
    """
    return get_estimator().batch_estimate(requests)


def add_generation_record(record: HistoricalRecord):
    """A convenience function to add a new historical record.

    Args:
        record: The HistoricalRecord to add.
    """
    get_estimator().add_historical_record(record)