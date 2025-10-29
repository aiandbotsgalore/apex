"""
Unit tests for APEX DIRECTOR Estimator

Tests cost and time estimation functionality using historical data.
"""

import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from apex_director.core.estimator import Estimator, EstimationRecord, EstimationEngine


class MockEstimatorEngine:
    """Mock estimation engine for testing"""
    
    def __init__(self):
        self.historical_data = []
    
    def add_record(self, record):
        """Add historical record"""
        self.historical_data.append(record)
    
    def predict_cost(self, features):
        """Mock cost prediction"""
        if not self.historical_data:
            return 0.05  # Default estimate
        
        # Simple mock prediction based on resolution and complexity
        width, height = features.get("width", 512), features.get("height", 512)
        steps = features.get("steps", 20)
        quality = features.get("quality_level", 3)
        
        base_cost = (width * height) / 1000000 * 0.01
        step_cost = steps * 0.001
        quality_cost = quality * 0.01
        
        return base_cost + step_cost + quality_cost
    
    def predict_time(self, features):
        """Mock time prediction"""
        if not self.historical_data:
            return 30.0  # Default 30 seconds
        
        width, height = features.get("width", 512), features.get("height", 512)
        steps = features.get("steps", 20)
        
        base_time = (width * height) / 10000 * 0.1
        step_time = steps * 1.5
        
        return base_time + step_time
    
    def calculate_confidence(self, features):
        """Mock confidence calculation"""
        if not self.historical_data:
            return 0.5  # Low confidence without data
        
        # More data = higher confidence
        data_count = len(self.historical_data)
        min_data = 10
        
        if data_count >= min_data:
            return 0.85
        else:
            return 0.5 + (data_count / min_data) * 0.35


class TestEstimator:
    """Test suite for Estimator"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def estimator(self, temp_dir):
        """Create estimator instance"""
        return Estimator(base_dir=temp_dir)
    
    def test_initialization(self, estimator):
        """Test estimator initialization"""
        assert estimator.base_dir == estimator.base_dir
        assert estimator.engine is not None
    
    def test_add_generation_record(self, estimator):
        """Test adding generation records"""
        record = EstimationRecord(
            job_id="test_record",
            backend="test_backend",
            width=1024,
            height=1024,
            steps=30,
            quality_level=4,
            actual_cost=0.075,
            actual_time=45.2,
            prompt_complexity=0.8
        )
        
        estimator.add_generation_record(record)
        
        # Check record was added
        assert len(estimator.engine.historical_data) == 1
        assert estimator.engine.historical_data[0] == record
    
    def test_estimate_generation_cost_time(self, estimator):
        """Test cost and time estimation"""
        # Add some historical data
        for i in range(10):
            record = EstimationRecord(
                job_id=f"record_{i}",
                backend="test_backend",
                width=512 + i * 64,
                height=512 + i * 64,
                steps=20 + i * 2,
                quality_level=3,
                actual_cost=0.03 + i * 0.005,
                actual_time=25.0 + i * 2.0,
                prompt_complexity=0.5 + i * 0.05
            )
            estimator.add_generation_record(record)
        
        # Estimate for new parameters
        estimate = estimator.estimate_generation_cost_time(
            width=1024,
            height=1024,
            steps=30,
            quality_level=4,
            prompt="A detailed cinematic scene"
        )
        
        assert estimate.estimated_cost > 0
        assert estimate.estimated_time_seconds > 0
        assert estimate.confidence_score > 0
        assert estimate.confidence_score <= 1.0
    
    def test_estimate_batch(self, estimator):
        """Test batch estimation"""
        # Add historical data
        for i in range(5):
            record = EstimationRecord(
                job_id=f"batch_record_{i}",
                backend="test_backend",
                width=512,
                height=512,
                steps=20,
                quality_level=3,
                actual_cost=0.03,
                actual_time=25.0,
                prompt_complexity=0.6
            )
            estimator.add_generation_record(record)
        
        # Batch estimate
        requests = [
            {
                "width": 512,
                "height": 512,
                "steps": 20,
                "quality_level": 3,
                "prompt": "Image 1"
            },
            {
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "quality_level": 4,
                "prompt": "Image 2"
            }
        ]
        
        estimates = estimator.estimate_batch(requests)
        
        assert len(estimates) == 2
        for estimate in estimates:
            assert estimate.estimated_cost > 0
            assert estimate.estimated_time_seconds > 0
    
    def test_update_estimates_with_actual(self, estimator):
        """Test updating estimates with actual results"""
        # Add initial record
        record1 = EstimationRecord(
            job_id="update_test_1",
            backend="test_backend",
            width=512,
            height=512,
            steps=20,
            quality_level=3,
            actual_cost=0.025,
            actual_time=22.5,
            prompt_complexity=0.5,
            estimated_cost=0.03,
            estimated_time=25.0,
            estimation_error_cost=0.005,
            estimation_error_time=2.5
        )
        
        estimator.add_generation_record(record1)
        
        # Add another record showing improvement
        record2 = EstimationRecord(
            job_id="update_test_2",
            backend="test_backend",
            width=512,
            height=512,
            steps=20,
            quality_level=3,
            actual_cost=0.028,
            actual_time=23.0,
            prompt_complexity=0.5,
            estimated_cost=0.027,
            estimated_time=23.2,
            estimation_error_cost=0.001,
            estimation_error_time=0.2
        )
        
        estimator.add_generation_record(record2)
        
        # Check estimation error improvement
        avg_error1 = record1.estimation_error_cost
        avg_error2 = record2.estimation_error_cost
        assert avg_error2 <= avg_error1  # Should improve or stay same
    
    def test_get_estimation_statistics(self, estimator):
        """Test estimation statistics"""
        # Add records with various backends and conditions
        backends = ["backend1", "backend2", "backend3"]
        
        for i in range(20):
            record = EstimationRecord(
                job_id=f"stats_record_{i}",
                backend=backends[i % 3],
                width=512 + (i % 3) * 256,
                height=512 + (i % 3) * 256,
                steps=20 + (i % 2) * 10,
                quality_level=3,
                actual_cost=0.025 + (i % 5) * 0.005,
                actual_time=20.0 + (i % 5) * 5.0,
                prompt_complexity=0.4 + (i % 5) * 0.15
            )
            estimator.add_generation_record(record)
        
        stats = estimator.get_estimation_statistics()
        
        assert "total_records" in stats
        assert "average_cost_error" in stats
        assert "average_time_error" in stats
        assert "backend_statistics" in stats
        assert "confidence_improvement" in stats
        
        assert stats["total_records"] == 20
        
        # Check backend statistics
        for backend in backends:
            assert backend in stats["backend_statistics"]
    
    def test_persistence(self, estimator):
        """Test saving and loading estimation data"""
        # Add some records
        for i in range(5):
            record = EstimationRecord(
                job_id=f"persist_record_{i}",
                backend="test_backend",
                width=512,
                height=512,
                steps=20,
                quality_level=3,
                actual_cost=0.025,
                actual_time=22.0,
                prompt_complexity=0.5
            )
            estimator.add_generation_record(record)
        
        # Save data
        save_path = estimator.save_estimation_data()
        assert save_path.exists()
        
        # Create new estimator and load data
        new_estimator = Estimator(base_dir=estimator.base_dir)
        load_success = new_estimator.load_estimation_data()
        
        assert load_success == True
        assert len(new_estimator.engine.historical_data) == 5
    
    def test_complexity_analysis(self, estimator):
        """Test prompt complexity analysis"""
        simple_prompt = "A cat"
        complex_prompt = "A majestic lion standing on a rocky cliff overlooking a vast savanna during golden hour, with dramatic volumetric lighting and cinematic composition"
        
        complexity_simple = estimator.analyze_prompt_complexity(simple_prompt)
        complexity_complex = estimator.analyze_prompt_complexity(complex_prompt)
        
        assert 0 <= complexity_simple <= 1
        assert 0 <= complexity_complex <= 1
        assert complexity_complex > complexity_simple
    
    def test_backend_specific_estimates(self, estimator):
        """Test estimates for specific backends"""
        # Add records for different backends
        backends_data = {
            "fast_backend": [
                EstimationRecord(
                    job_id=f"fast_{i}",
                    backend="fast_backend",
                    width=512, height=512, steps=20, quality_level=3,
                    actual_cost=0.02, actual_time=15.0,
                    prompt_complexity=0.5
                ) for i in range(10)
            ],
            "quality_backend": [
                EstimationRecord(
                    job_id=f"quality_{i}",
                    backend="quality_backend",
                    width=512, height=512, steps=30, quality_level=4,
                    actual_cost=0.05, actual_time=35.0,
                    prompt_complexity=0.5
                ) for i in range(10)
            ]
        }
        
        for backend, records in backends_data.items():
            for record in records:
                estimator.add_generation_record(record)
        
        # Get estimates for each backend
        fast_estimate = estimator.estimate_generation_cost_time(
            backend="fast_backend",
            width=512, height=512, steps=20, quality_level=3
        )
        
        quality_estimate = estimator.estimate_generation_cost_time(
            backend="quality_backend",
            width=512, height=512, steps=30, quality_level=4
        )
        
        # Quality backend should be more expensive and slower
        assert quality_estimate.estimated_cost > fast_estimate.estimated_cost
        assert quality_estimate.estimated_time_seconds > fast_estimate.estimated_time_seconds
    
    def test_confidence_scaling(self, estimator):
        """Test confidence score scaling with data amount"""
        # Estimate with no data
        estimate_no_data = estimator.estimate_generation_cost_time(
            width=512, height=512, steps=20
        )
        
        # Add minimum data for medium confidence
        for i in range(5):
            record = EstimationRecord(
                job_id=f"medium_conf_{i}",
                backend="test_backend",
                width=512, height=512, steps=20, quality_level=3,
                actual_cost=0.025, actual_time=22.0,
                prompt_complexity=0.5
            )
            estimator.add_generation_record(record)
        
        estimate_some_data = estimator.estimate_generation_cost_time(
            width=512, height=512, steps=20
        )
        
        # Add sufficient data for high confidence
        for i in range(15):  # Total 20 records
            record = EstimationRecord(
                job_id=f"high_conf_{i}",
                backend="test_backend",
                width=512, height=512, steps=20, quality_level=3,
                actual_cost=0.025, actual_time=22.0,
                prompt_complexity=0.5
            )
            estimator.add_generation_record(record)
        
        estimate_high_data = estimator.estimate_generation_cost_time(
            width=512, height=512, steps=20
        )
        
        # Confidence should increase with more data
        assert estimate_some_data.confidence_score > estimate_no_data.confidence_score
        assert estimate_high_data.confidence_score > estimate_some_data.confidence_score
        assert estimate_high_data.confidence_score > 0.8
    
    def test_error_metrics(self, estimator):
        """Test estimation error metrics"""
        # Add records with known estimation errors
        test_records = [
            EstimationRecord(
                job_id=f"error_test_{i}",
                backend="test_backend",
                width=512, height=512, steps=20, quality_level=3,
                actual_cost=0.025 + i * 0.01,  # Increasing actual costs
                actual_time=20.0 + i * 5.0,    # Increasing actual times
                prompt_complexity=0.5,
                estimated_cost=0.03,           # Constant estimate
                estimated_time=25.0,           # Constant estimate
                estimation_error_cost=abs(0.03 - (0.025 + i * 0.01)),
                estimation_error_time=abs(25.0 - (20.0 + i * 5.0))
            ) for i in range(5)
        ]
        
        for record in test_records:
            estimator.add_generation_record(record)
        
        # Get error statistics
        stats = estimator.get_estimation_statistics()
        
        assert "error_statistics" in stats
        assert "mean_absolute_error_cost" in stats["error_statistics"]
        assert "mean_absolute_error_time" in stats["error_statistics"]
        assert "root_mean_square_error_cost" in stats["error_statistics"]
        
        assert stats["error_statistics"]["mean_absolute_error_cost"] > 0
        assert stats["error_statistics"]["mean_absolute_error_time"] > 0


class TestEstimationRecord:
    """Test suite for EstimationRecord data class"""
    
    def test_estimation_record_creation(self):
        """Test creating estimation records"""
        record = EstimationRecord(
            job_id="test_job",
            backend="test_backend",
            width=1024,
            height=1024,
            steps=30,
            quality_level=4,
            actual_cost=0.075,
            actual_time=45.2,
            prompt_complexity=0.8
        )
        
        assert record.job_id == "test_job"
        assert record.backend == "test_backend"
        assert record.width == 1024
        assert record.height == 1024
        assert record.steps == 30
        assert record.quality_level == 4
        assert record.actual_cost == 0.075
        assert record.actual_time == 45.2
        assert record.prompt_complexity == 0.8
        
        # Default values
        assert record.estimated_cost is None
        assert record.estimated_time is None
        assert record.estimation_error_cost is None
        assert record.estimation_error_time is None
    
    def test_estimation_record_with_estimated_values(self):
        """Test creating record with estimated values"""
        record = EstimationRecord(
            job_id="test_job",
            backend="test_backend",
            width=1024,
            height=1024,
            steps=30,
            quality_level=4,
            actual_cost=0.075,
            actual_time=45.2,
            prompt_complexity=0.8,
            estimated_cost=0.07,
            estimated_time=42.0,
            estimation_error_cost=0.005,
            estimation_error_time=3.2
        )
        
        assert record.estimated_cost == 0.07
        assert record.estimated_time == 42.0
        assert record.estimation_error_cost == 0.005
        assert record.estimation_error_time == 3.2
    
    def test_record_serialization(self):
        """Test record serialization to JSON"""
        record = EstimationRecord(
            job_id="serialization_test",
            backend="test_backend",
            width=512,
            height=512,
            steps=20,
            quality_level=3,
            actual_cost=0.025,
            actual_time=22.0,
            prompt_complexity=0.6
        )
        
        # Convert to dict and then to JSON
        record_dict = record.__dict__
        json_str = json.dumps(record_dict)
        
        # Deserialize
        restored_dict = json.loads(json_str)
        restored_record = EstimationRecord(**restored_dict)
        
        # Check equality
        assert restored_record.job_id == record.job_id
        assert restored_record.actual_cost == record.actual_cost
        assert restored_record.actual_time == record.actual_time


if __name__ == "__main__":
    pytest.main([__file__])
