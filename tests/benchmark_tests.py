"""
Performance Benchmarking Tests for APEX DIRECTOR

Tests system performance under various loads and conditions.
"""

import pytest
import asyncio
import time
import psutil
import gc
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import statistics
import json
from typing import List, Dict, Any

from apex_director.core.orchestrator import APEXOrchestrator
from apex_director.core.asset_manager import AssetManager
from apex_director.core.estimator import Estimator


class PerformanceMetrics:
    """Performance metrics collector"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.cpu_usage = []
        self.throughput_metrics = []
    
    def start_measurement(self):
        """Start performance measurement"""
        self.start_time = time.time()
        gc.collect()  # Clean up before measurement
    
    def end_measurement(self):
        """End performance measurement"""
        self.end_time = time.time()
        return self.get_results()
    
    def record_memory(self):
        """Record current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
    
    def record_cpu(self):
        """Record current CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.append(cpu_percent)
    
    def record_throughput(self, metric_name: str, value: float):
        """Record throughput metric"""
        self.throughput_metrics.append({
            "metric": metric_name,
            "value": value,
            "timestamp": time.time()
        })
    
    def get_results(self) -> Dict[str, Any]:
        """Get performance results"""
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        return {
            "duration_seconds": duration,
            "memory_stats": {
                "peak_mb": max(self.memory_usage) if self.memory_usage else 0,
                "average_mb": statistics.mean(self.memory_usage) if self.memory_usage else 0,
                "final_mb": self.memory_usage[-1] if self.memory_usage else 0
            },
            "cpu_stats": {
                "peak_percent": max(self.cpu_usage) if self.cpu_usage else 0,
                "average_percent": statistics.mean(self.cpu_usage) if self.cpu_usage else 0
            },
            "throughput": self.throughput_metrics
        }


class TestOrchestratorPerformance:
    """Performance tests for orchestrator"""
    
    @pytest.mark.asyncio
    async def test_job_throughput(self):
        """Test job processing throughput"""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = APEXOrchestrator(
                config={"orchestrator": {"max_concurrent_jobs": 5}},
                base_dir=Path(temp_dir)
            )
            
            await orchestrator.initialize()
            
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Submit many jobs quickly
            job_count = 100
            job_ids = []
            
            for i in range(job_count):
                job_id = f"throughput_job_{i}"
                job_request = {
                    "id": job_id,
                    "type": "image_generation",
                    "prompt": f"Performance test image {i}"
                }
                
                submitted_id = await orchestrator.submit_job(job_request)
                job_ids.append(submitted_id)
                
                # Record metrics every 10 jobs
                if i % 10 == 0:
                    metrics.record_memory()
                    metrics.record_cpu()
            
            metrics.end_measurement()
            
            # Check throughput
            results = metrics.get_results()
            jobs_per_second = job_count / results["duration_seconds"]
            
            assert jobs_per_second > 10  # At least 10 jobs per second
            assert results["memory_stats"]["peak_mb"] < 500  # Reasonable memory usage
            
            # Cleanup
            for job_id in job_ids:
                await orchestrator.cancel_job(job_id)
    
    @pytest.mark.asyncio
    async def test_concurrent_job_processing_performance(self):
        """Test performance with concurrent jobs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = APEXOrchestrator(
                config={"orchestrator": {"max_concurrent_jobs": 10}},
                base_dir=Path(temp_dir)
            )
            
            await orchestrator.initialize()
            
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Submit jobs concurrently
            async def submit_job(i):
                job_id = f"concurrent_job_{i}"
                job_request = {
                    "id": job_id,
                    "type": "image_generation",
                    "prompt": f"Concurrent test {i}"
                }
                return await orchestrator.submit_job(job_request)
            
            # Submit 20 jobs concurrently
            tasks = [submit_job(i) for i in range(20)]
            job_ids = await asyncio.gather(*tasks)
            
            # Monitor system during processing
            for _ in range(5):
                metrics.record_memory()
                metrics.record_cpu()
                await asyncio.sleep(0.1)
            
            results = metrics.end_measurement()
            
            # Verify concurrent processing works
            assert len(job_ids) == 20
            assert results["cpu_stats"]["average_percent"] > 0
            
            # Cleanup
            for job_id in job_ids:
                await orchestrator.cancel_job(job_id)
    
    @pytest.mark.asyncio
    async def test_checkpoint_performance(self):
        """Test checkpoint creation and restoration performance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = APEXOrchestrator(
                config={"orchestrator": {"max_concurrent_jobs": 3}},
                base_dir=Path(temp_dir)
            )
            
            await orchestrator.initialize()
            
            # Create many jobs for checkpoint
            job_count = 50
            for i in range(job_count):
                job_id = f"checkpoint_job_{i}"
                job_request = {
                    "id": job_id,
                    "type": "image_generation",
                    "prompt": f"Checkpoint test {i}"
                }
                await orchestrator.submit_job(job_request)
            
            # Test checkpoint creation performance
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            checkpoint_id = await orchestrator.create_checkpoint(
                "performance_checkpoint"
            )
            
            checkpoint_time = metrics.end_measurement()
            
            assert checkpoint_id is not None
            assert checkpoint_time["duration_seconds"] < 5  # Should be fast
            
            # Test checkpoint restoration performance
            metrics.start_measurement()
            
            await orchestrator.restore_from_checkpoint(checkpoint_id)
            
            restore_time = metrics.end_measurement()
            
            assert restore_time["duration_seconds"] < 10  # Should be reasonably fast
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage under high load"""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = APEXOrchestrator(
                config={"orchestrator": {"max_concurrent_jobs": 20}},
                base_dir=Path(temp_dir)
            )
            
            await orchestrator.initialize()
            
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Submit many jobs to stress memory
            job_ids = []
            for i in range(100):
                job_id = f"memory_test_{i}"
                job_request = {
                    "id": job_id,
                    "type": "image_generation",
                    "prompt": f"Memory stress test {i}"
                }
                submitted_id = await orchestrator.submit_job(job_request)
                job_ids.append(submitted_id)
                
                # Record memory every 10 jobs
                if i % 10 == 0:
                    metrics.record_memory()
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Final memory measurement
            metrics.record_memory()
            results = metrics.end_measurement()
            
            # Check memory growth is reasonable
            memory_growth = results["memory_stats"]["final_mb"] - results["memory_stats"]["peak_mb"]
            assert memory_growth < 100  # Less than 100MB growth after peak
            
            # Cleanup
            for job_id in job_ids:
                await orchestrator.cancel_job(job_id)


class TestAssetManagerPerformance:
    """Performance tests for asset manager"""
    
    def test_storage_performance(self):
        """Test asset storage performance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_manager = AssetManager(base_dir=Path(temp_dir))
            
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Store many assets
            asset_count = 1000
            for i in range(asset_count):
                asset_data = {
                    "content": f"Asset {i} data".encode() * 10,  # ~100 bytes each
                    "filename": f"asset_{i}.dat",
                    "metadata": {
                        "index": i,
                        "tags": [f"tag_{j}" for j in range(i % 5)]
                    }
                }
                
                asset_path = asset_manager.store_asset(asset_data=asset_data)
                
                # Record metrics every 100 assets
                if i % 100 == 0:
                    metrics.record_memory()
                    metrics.record_throughput("assets_stored", i)
            
            results = metrics.end_measurement()
            
            # Check storage throughput
            assets_per_second = asset_count / results["duration_seconds"]
            assert assets_per_second > 100  # At least 100 assets per second
            
            # Check statistics
            stats = asset_manager.get_storage_statistics()
            assert stats["total_files"] == asset_count
    
    def test_search_performance(self):
        """Test asset search performance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_manager = AssetManager(base_dir=Path(temp_dir))
            
            # Create assets with various metadata
            for i in range(500):
                asset_data = {
                    "content": f"Search test {i}".encode(),
                    "filename": f"search_{i}.dat",
                    "metadata": {
                        "category": f"cat_{i % 10}",
                        "tags": [f"tag_{j}" for j in range(i % 3)],
                        "priority": i % 5
                    }
                }
                
                asset_manager.store_asset(asset_data=asset_data)
            
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Perform various searches
            search_queries = [
                {"category": "cat_1"},
                {"tags": ["tag_0"]},
                {"priority": 3},
                {"filename_pattern": "search_1*"}
            ]
            
            for query in search_queries:
                results = asset_manager.search_assets(**query)
                metrics.record_throughput("searches_completed", 1)
            
            results = metrics.end_measurement()
            
            # Check search performance
            searches_per_second = 1 / results["duration_seconds"]
            assert searches_per_second > 10  # At least 10 searches per second
    
    def test_duplicate_detection_performance(self):
        """Test duplicate detection performance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_manager = AssetManager(base_dir=Path(temp_dir))
            
            # Create many assets with duplicates
            for i in range(200):
                # Create duplicates by repeating content
                content = f"Duplicate content {i % 10}".encode()
                asset_data = {
                    "content": content,
                    "filename": f"duplicate_{i}.dat"
                }
                
                asset_manager.store_asset(asset_data=asset_data)
            
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Find duplicates
            duplicates = asset_manager.find_duplicates()
            
            results = metrics.end_measurement()
            
            # Check detection performance
            assert results["duration_seconds"] < 10  # Should complete within 10 seconds
            assert len(duplicates) > 0  # Should find some duplicates
    
    def test_storage_statistics_performance(self):
        """Test storage statistics performance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_manager = AssetManager(base_dir=Path(temp_dir))
            
            # Create many assets
            for i in range(1000):
                asset_data = {
                    "content": f"Stats test {i}".encode(),
                    "filename": f"stats_{i}.dat"
                }
                
                asset_manager.store_asset(asset_data=asset_data)
            
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Get statistics
            stats = asset_manager.get_storage_statistics()
            
            results = metrics.end_measurement()
            
            # Check statistics performance
            assert results["duration_seconds"] < 5  # Should be fast
            assert stats["total_files"] == 1000


class TestEstimatorPerformance:
    """Performance tests for estimator"""
    
    def test_estimation_speed(self):
        """Test estimation calculation speed"""
        with tempfile.TemporaryDirectory() as temp_dir:
            estimator = Estimator(base_dir=Path(temp_dir))
            
            # Add historical data
            for i in range(100):
                from apex_director.core.estimator import EstimationRecord
                
                record = EstimationRecord(
                    job_id=f"perf_test_{i}",
                    backend="test_backend",
                    width=512 + i * 64,
                    height=512 + i * 64,
                    steps=20 + (i % 3) * 10,
                    quality_level=3,
                    actual_cost=0.025 + i * 0.001,
                    actual_time=20.0 + i * 0.5,
                    prompt_complexity=0.5 + i * 0.01
                )
                
                estimator.add_generation_record(record)
            
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Perform many estimations
            estimation_count = 50
            for i in range(estimation_count):
                estimate = estimator.estimate_generation_cost_time(
                    width=1024,
                    height=1024,
                    steps=30,
                    quality_level=4,
                    prompt=f"Performance test estimation {i}"
                )
                
                assert estimate.estimated_cost > 0
                assert estimate.confidence_score > 0
                
                if i % 10 == 0:
                    metrics.record_throughput("estimations_completed", i)
            
            results = metrics.end_measurement()
            
            # Check estimation performance
            estimations_per_second = estimation_count / results["duration_seconds"]
            assert estimations_per_second > 100  # At least 100 estimations per second
    
    def test_batch_estimation_performance(self):
        """Test batch estimation performance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            estimator = Estimator(base_dir=Path(temp_dir))
            
            # Add historical data
            for i in range(50):
                from apex_director.core.estimator import EstimationRecord
                
                record = EstimationRecord(
                    job_id=f"batch_test_{i}",
                    backend="test_backend",
                    width=512,
                    height=512,
                    steps=20,
                    quality_level=3,
                    actual_cost=0.025,
                    actual_time=20.0,
                    prompt_complexity=0.6
                )
                
                estimator.add_generation_record(record)
            
            # Create large batch of estimation requests
            batch_requests = [
                {
                    "width": 512,
                    "height": 512,
                    "steps": 20,
                    "quality_level": 3,
                    "prompt": f"Batch estimation {i}"
                } for i in range(100)
            ]
            
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Perform batch estimation
            batch_estimates = estimator.estimate_batch(batch_requests)
            
            results = metrics.end_measurement()
            
            # Check batch performance
            assert len(batch_estimates) == 100
            assert results["duration_seconds"] < 30  # Should complete within 30 seconds
    
    def test_statistics_calculation_performance(self):
        """Test statistics calculation performance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            estimator = Estimator(base_dir=Path(temp_dir))
            
            # Add large amount of historical data
            for i in range(1000):
                from apex_director.core.estimator import EstimationRecord
                
                record = EstimationRecord(
                    job_id=f"stats_test_{i}",
                    backend=f"backend_{i % 5}",
                    width=512 + (i % 5) * 256,
                    height=512 + (i % 5) * 256,
                    steps=20 + (i % 3) * 15,
                    quality_level=3,
                    actual_cost=0.025 + i * 0.0001,
                    actual_time=20.0 + i * 0.02,
                    prompt_complexity=0.5 + i * 0.001
                )
                
                estimator.add_generation_record(record)
            
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Calculate statistics
            stats = estimator.get_estimation_statistics()
            
            results = metrics.end_measurement()
            
            # Check statistics performance
            assert results["duration_seconds"] < 10  # Should be fast
            assert stats["total_records"] == 1000
            assert "backend_statistics" in stats


class TestSystemResourceUsage:
    """Tests for system resource usage monitoring"""
    
    @pytest.mark.asyncio
    async def test_cpu_usage_monitoring(self):
        """Test CPU usage monitoring during operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = APEXOrchestrator(
                config={"orchestrator": {"max_concurrent_jobs": 10}},
                base_dir=Path(temp_dir)
            )
            
            await orchestrator.initialize()
            
            cpu_readings = []
            
            # Submit jobs and monitor CPU
            for i in range(20):
                job_id = f"cpu_test_{i}"
                job_request = {
                    "id": job_id,
                    "type": "image_generation",
                    "prompt": f"CPU monitoring test {i}"
                }
                
                await orchestrator.submit_job(job_request)
                
                # Record CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_readings.append(cpu_percent)
                
                await asyncio.sleep(0.1)
            
            # Check CPU usage is reasonable
            average_cpu = statistics.mean(cpu_readings)
            peak_cpu = max(cpu_readings)
            
            assert average_cpu < 80  # Average should not be too high
            assert peak_cpu < 95  # Peak should not be at 100%
            
            # Cleanup
            for i in range(20):
                await orchestrator.cancel_job(f"cpu_test_{i}")
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks during extended operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_manager = AssetManager(base_dir=Path(temp_dir))
            
            # Record baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss
            
            # Perform many operations
            for batch in range(10):
                # Create and process assets
                for i in range(50):
                    asset_data = {
                        "content": f"Memory test batch {batch} asset {i}".encode(),
                        "filename": f"memory_{batch}_{i}.dat"
                    }
                    
                    asset_path = asset_manager.store_asset(asset_data=asset_data)
                    
                    # Simulate processing
                    metadata = asset_manager.get_asset_metadata(asset_path)
                
                # Force garbage collection
                gc.collect()
                
                # Record memory after each batch
                current_memory = process.memory_info().rss
                memory_growth_mb = (current_memory - baseline_memory) / 1024 / 1024
                
                # Memory growth should be reasonable
                assert memory_growth_mb < 100  # Less than 100MB growth
    
    @pytest.mark.asyncio
    async def test_disk_usage_optimization(self):
        """Test disk usage optimization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_manager = AssetManager(base_dir=Path(temp_dir))
            
            # Record initial disk usage
            initial_usage = sum(f.stat().st_size for f in Path(temp_dir).rglob('*') if f.is_file())
            
            # Create many temporary files
            temp_files = []
            for i in range(100):
                temp_file = temp_dir / f"temp_{i}.tmp"
                temp_file.write_text(f"Temporary content {i}" * 100)  # ~3KB each
                temp_files.append(temp_file)
            
            # Run cleanup
            cleaned_count = asset_manager.cleanup_temp_files(max_age_hours=0)  # Clean all
            
            # Check disk usage improvement
            final_usage = sum(f.stat().st_size for f in Path(temp_dir).rglob('*') if f.is_file())
            
            # Should have cleaned up files
            assert cleaned_count >= 0
            assert final_usage <= initial_usage + (100 * 3000)  # Allow for metadata overhead


class TestScalabilityBenchmarks:
    """Tests for system scalability benchmarks"""
    
    @pytest.mark.asyncio
    async def test_job_queue_scalability(self):
        """Test job queue handling at scale"""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = APEXOrchestrator(
                config={"orchestrator": {"max_concurrent_jobs": 5}},
                base_dir=Path(temp_dir)
            )
            
            await orchestrator.initialize()
            
            # Test with increasing job counts
            job_counts = [10, 50, 100, 200]
            
            for job_count in job_counts:
                metrics = PerformanceMetrics()
                metrics.start_measurement()
                
                # Submit large number of jobs
                job_ids = []
                for i in range(job_count):
                    job_id = f"scale_test_{job_count}_{i}"
                    job_request = {
                        "id": job_id,
                        "type": "image_generation",
                        "prompt": f"Scalability test {job_count} - {i}"
                    }
                    
                    submitted_id = await orchestrator.submit_job(job_request)
                    job_ids.append(submitted_id)
                
                submission_time = metrics.end_measurement()
                
                # Check submission performance scales reasonably
                jobs_per_second = job_count / submission_time["duration_seconds"]
                
                # Performance should degrade gracefully, not crash
                assert jobs_per_second > 5  # At least 5 jobs per second
                
                # Verify system stability
                stats = orchestrator.get_system_stats()
                assert stats["jobs"]["active_jobs"] <= job_count
                
                # Cleanup
                for job_id in job_ids:
                    await orchestrator.cancel_job(job_id)
    
    @pytest.mark.asyncio
    async def test_asset_storage_scalability(self):
        """Test asset storage at scale"""
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_manager = AssetManager(base_dir=Path(temp_dir))
            
            # Test with increasing asset counts
            asset_counts = [100, 500, 1000, 2000]
            
            for asset_count in asset_counts:
                metrics = PerformanceMetrics()
                metrics.start_measurement()
                
                # Create large number of assets
                for i in range(asset_count):
                    asset_data = {
                        "content": f"Scale test asset {asset_count} - {i}".encode(),
                        "filename": f"scale_asset_{asset_count}_{i}.dat",
                        "metadata": {
                            "batch": asset_count,
                            "index": i,
                            "tags": [f"scale_tag_{j}" for j in range(i % 3)]
                        }
                    }
                    
                    asset_manager.store_asset(asset_data=asset_data)
                
                storage_time = metrics.end_measurement()
                
                # Check storage performance
                assets_per_second = asset_count / storage_time["duration_seconds"]
                assert assets_per_second > 50  # At least 50 assets per second
                
                # Verify statistics are correct
                stats = asset_manager.get_storage_statistics()
                assert stats["total_files"] >= asset_count
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_at_scale(self):
        """Test memory efficiency with large datasets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            estimator = Estimator(base_dir=Path(temp_dir))
            
            # Add increasingly large amounts of data
            record_counts = [100, 500, 1000, 2000]
            
            peak_memories = []
            
            for count in record_counts:
                process = psutil.Process()
                initial_memory = process.memory_info().rss
                
                # Add records
                for i in range(count):
                    from apex_director.core.estimator import EstimationRecord
                    
                    record = EstimationRecord(
                        job_id=f"memory_test_{count}_{i}",
                        backend="test_backend",
                        width=512,
                        height=512,
                        steps=20,
                        quality_level=3,
                        actual_cost=0.025,
                        actual_time=20.0,
                        prompt_complexity=0.6
                    )
                    
                    estimator.add_generation_record(record)
                
                # Record peak memory
                peak_memory = process.memory_info().rss
                peak_memories.append((count, peak_memory))
                
                # Perform operation to test current memory usage
                estimate = estimator.estimate_generation_cost_time(
                    width=512,
                    height=512,
                    steps=20
                )
                
                assert estimate.estimated_cost > 0
            
            # Check memory growth is sub-linear
            memory_growth = peak_memories[-1][1] - peak_memories[0][1]
            record_growth = peak_memories[-1][0] - peak_memories[0][0]
            
            # Memory per record should be reasonable and not grow excessively
            memory_per_record = memory_growth / record_growth
            assert memory_per_record < 1000  # Less than 1KB per record


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
