"""
Test Fixtures and Mock Data Generators for APEX DIRECTOR

Provides reusable fixtures, mock data, and test utilities.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid

# Import APEX DIRECTOR components
from apex_director.core.orchestrator import APEXOrchestrator
from apex_director.core.asset_manager import AssetManager
from apex_director.core.checkpoint import CheckpointManager, JobState
from apex_director.core.estimator import Estimator, EstimationRecord
from apex_director.images.generator import GenerationRequest
from apex_director.video.assembler import AssemblyJob
from apex_director.audio.analyzer import AudioAnalysis


class MockDataGenerator:
    """Generate realistic mock data for testing"""
    
    @staticmethod
    def generate_audio_analysis(duration: float = 120.0, tempo: float = 120.0) -> AudioAnalysis:
        """Generate mock audio analysis"""
        sections = []
        section_types = ["verse", "chorus", "bridge", "verse", "chorus", "outro"]
        current_time = 0.0
        
        for i, section_type in enumerate(section_types):
            section_duration = duration / len(section_types)
            section = {
                "type": section_type,
                "start": current_time,
                "end": current_time + section_duration,
                "confidence": 0.8 + (i % 3) * 0.05
            }
            sections.append(section)
            current_time += section_duration
        
        # Generate beat markers
        beat_interval = 60.0 / tempo  # seconds per beat
        beats = []
        for i in range(int(duration / beat_interval)):
            beats.append({
                "time": i * beat_interval,
                "confidence": 0.7 + (i % 5) * 0.05
            })
        
        return AudioAnalysis(
            duration=duration,
            tempo=tempo,
            key="C major",
            sections=sections,
            beats=beats
        )
    
    @staticmethod
    def generate_generation_requests(count: int = 5) -> List[GenerationRequest]:
        """Generate mock generation requests"""
        prompts = [
            "A cinematic sunset over mountains",
            "An abstract digital art piece",
            "A futuristic cityscape at night",
            "A peaceful forest with morning light",
            "A dramatic ocean wave",
            "A vintage car on a desert road",
            "A magical fairy tale castle",
            "A cyberpunk character portrait",
            "A cosmic nebula in space",
            "A cozy cabin in winter"
        ]
        
        requests = []
        for i in range(count):
            prompt = prompts[i % len(prompts)]
            request = GenerationRequest(
                prompt=prompt,
                scene_id=f"scene_{i}",
                genre="cinematic" if i % 2 == 0 else "abstract",
                director_style=["christopher_nolan", "wes_anderson", "guillermo_del_toro"][i % 3],
                camera_settings={
                    "lens": ["24mm", "50mm", "85mm"][i % 3],
                    "aperture": ["f/2.8", "f/4.0", "f/5.6"][i % 3]
                }
            )
            requests.append(request)
        
        return requests
    
    @staticmethod
    def generate_estimation_records(count: int = 50) -> List[EstimationRecord]:
        """Generate mock estimation records"""
        records = []
        backends = ["nano_banana", "imagen", "minimax", "sdxl"]
        
        for i in range(count):
            width = 512 + (i % 4) * 256
            height = 512 + (i % 4) * 256
            steps = 20 + (i % 3) * 15
            quality = 3 + (i % 3)
            
            base_cost = (width * height) / 1000000 * 0.01
            step_cost = steps * 0.001
            quality_cost = quality * 0.01
            actual_cost = base_cost + step_cost + quality_cost
            
            base_time = (width * height) / 10000 * 0.1
            step_time = steps * 1.2
            actual_time = base_time + step_time
            
            record = EstimationRecord(
                job_id=f"mock_record_{i}",
                backend=backends[i % len(backends)],
                width=width,
                height=height,
                steps=steps,
                quality_level=quality,
                actual_cost=actual_cost,
                actual_time=actual_time,
                prompt_complexity=0.5 + (i % 5) * 0.1
            )
            records.append(record)
        
        return records
    
    @staticmethod
    def generate_job_states(count: int = 10) -> List[JobState]:
        """Generate mock job states"""
        statuses = ["queued", "processing", "completed", "failed", "cancelled"]
        states = []
        
        for i in range(count):
            status = statuses[i % len(statuses)]
            progress = 0.0 if status == "queued" else (1.0 if status == "completed" else i / count)
            
            state = JobState(
                job_id=f"mock_job_{i}",
                status=status,
                progress=progress,
                result={"output": f"result_{i}"} if status == "completed" else None,
                error=f"error_{i}" if status == "failed" else None
            )
            states.append(state)
        
        return states
    
    @staticmethod
    def generate_assembly_job() -> AssemblyJob:
        """Generate mock assembly job"""
        return AssemblyJob(
            job_id="mock_assembly_job",
            audio_path="/tmp/mock_audio.mp3",
            output_path="/tmp/mock_output.mp4",
            image_sequence=[
                {
                    "image_path": f"/tmp/frame_{i}.png",
                    "start_time": i * 3.0,
                    "end_time": (i + 1) * 3.0
                } for i in range(10)
            ]
        )
    
    @staticmethod
    def generate_asset_data(count: int = 20) -> List[Dict[str, Any]]:
        """Generate mock asset data"""
        assets = []
        categories = ["nature", "urban", "abstract", "character", "landscape"]
        
        for i in range(count):
            asset = {
                "content": f"Mock asset content {i}".encode(),
                "filename": f"asset_{i}.png",
                "content_type": "image/png",
                "metadata": {
                    "category": categories[i % len(categories)],
                    "tags": [f"tag_{j}" for j in range(i % 3)],
                    "source": "mock_generator",
                    "quality_score": 0.8 + (i % 10) * 0.02,
                    "created_at": datetime.now().isoformat()
                }
            }
            assets.append(asset)
        
        return assets
    
    @staticmethod
    def generate_style_bible() -> Dict[str, Any]:
        """Generate mock style bible"""
        return {
            "project_name": "Mock Style Project",
            "overall_style": {
                "visual_style": "cinematic realism with dramatic flair",
                "color_grading": "natural with subtle warmth",
                "lighting_style": "three-point lighting with practical accents"
            },
            "color_palette": {
                "primary_colors": ["#2C3E50", "#ECF0F1", "#E74C3C"],
                "secondary_colors": ["#3498DB", "#F39C12", "#9B59B6"],
                "skin_tones": ["#FDBCB4", "#F1C27D", "#E0AC69"]
            },
            "camera_profile": {
                "preferred_lenses": ["35mm", "50mm", "85mm"],
                "aperture_range": "f/2.8 to f/5.6",
                "depth_of_field": "shallow to medium"
            },
            "lighting_setup": {
                "key_light": "soft box with diffusion",
                "fill_light": "reflector or secondary soft box",
                "rim_light": "practical lights or LED strip"
            }
        }


class MockBackend:
    """Mock backend for testing"""
    
    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.call_count = 0
        self.responses = []
    
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Mock image generation"""
        self.call_count += 1
        
        if self.should_fail:
            raise Exception(f"{self.name} backend failed")
        
        response = {
            "image_path": f"/tmp/{self.name}_output_{self.call_count}.png",
            "metadata": {
                "backend": self.name,
                "prompt": prompt,
                "parameters": kwargs,
                "generated_at": datetime.now().isoformat(),
                "quality_score": 0.85 + (self.call_count % 10) * 0.01
            }
        }
        
        self.responses.append(response)
        return response
    
    async def health_check(self) -> Dict[str, Any]:
        """Mock health check"""
        return {
            "healthy": not self.should_fail,
            "response_time": 0.1 + (self.call_count % 5) * 0.02,
            "error": f"{self.name} backend failed" if self.should_fail else None
        }


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_data_generator():
    """Provide mock data generator"""
    return MockDataGenerator()


@pytest.fixture
async def orchestrator(temp_dir):
    """Create orchestrator instance for testing"""
    config = {
        "orchestrator": {
            "max_concurrent_jobs": 5,
            "checkpoint_interval": 60,
            "auto_retry": True,
            "retry_attempts": 3
        },
        "backends": {
            "test_backend": {"enabled": True, "priority": 1},
            "fallback_backend": {"enabled": True, "priority": 2}
        }
    }
    
    orchestrator = APEXOrchestrator(config=config, base_dir=temp_dir)
    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.shutdown()


@pytest.fixture
def asset_manager(temp_dir):
    """Create asset manager instance for testing"""
    return AssetManager(base_dir=temp_dir)


@pytest.fixture
async def checkpoint_manager(temp_dir):
    """Create checkpoint manager instance for testing"""
    manager = CheckpointManager(base_dir=temp_dir)
    yield manager


@pytest.fixture
def estimator(temp_dir):
    """Create estimator instance for testing"""
    return Estimator(base_dir=temp_dir)


@pytest.fixture
def mock_backends():
    """Create mock backends for testing"""
    return {
        "primary": MockBackend("primary", should_fail=False),
        "fallback": MockBackend("fallback", should_fail=False),
        "failing": MockBackend("failing", should_fail=True)
    }


@pytest.fixture
def sample_audio_analysis(mock_data_generator):
    """Provide sample audio analysis"""
    return mock_data_generator.generate_audio_analysis()


@pytest.fixture
def sample_generation_requests(mock_data_generator):
    """Provide sample generation requests"""
    return mock_data_generator.generate_generation_requests(5)


@pytest.fixture
def sample_estimation_records(mock_data_generator):
    """Provide sample estimation records"""
    return mock_data_generator.generate_estimation_records(20)


@pytest.fixture
def sample_job_states(mock_data_generator):
    """Provide sample job states"""
    return mock_data_generator.generate_job_states(8)


@pytest.fixture
def sample_assembly_job(mock_data_generator):
    """Provide sample assembly job"""
    return mock_data_generator.generate_assembly_job()


@pytest.fixture
def sample_asset_data(mock_data_generator):
    """Provide sample asset data"""
    return mock_data_generator.generate_asset_data(15)


@pytest.fixture
def sample_style_bible(mock_data_generator):
    """Provide sample style bible"""
    return mock_data_generator.generate_style_bible()


@pytest.fixture
def sample_project(asset_manager):
    """Create a sample project"""
    return asset_manager.create_project(
        name="Test Project",
        description="A project for testing"
    )


@pytest.fixture
def sample_checkpoints(checkpoint_manager, sample_job_states):
    """Create sample checkpoints"""
    checkpoint_ids = []
    
    # Create multiple checkpoints
    for i in range(3):
        jobs = sample_job_states[:(i + 2)]  # Different number of jobs
        checkpoint_id = asyncio.create_task(
            checkpoint_manager.create_checkpoint(
                name=f"test_checkpoint_{i}",
                description=f"Test checkpoint {i}",
                jobs=jobs
            )
        )
        checkpoint_ids.append(checkpoint_id)
    
    # Wait for all to complete
    checkpoint_ids = asyncio.run(asyncio.gather(*checkpoint_ids))
    return checkpoint_ids


@pytest.fixture
def sample_estimates(estimator, sample_estimation_records):
    """Add sample estimation records to estimator"""
    for record in sample_estimation_records:
        estimator.add_generation_record(record)
    return estimator


@pytest.fixture
def performance_monitor():
    """Create performance monitoring utilities"""
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_usage = []
        
        def start(self):
            self.start_time = datetime.now()
            self.memory_usage.clear()
        
        def record_memory(self):
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage.append(memory_mb)
        
        def end(self):
            self.end_time = datetime.now()
        
        def get_results(self):
            duration = (self.end_time - self.start_time).total_seconds()
            return {
                "duration_seconds": duration,
                "peak_memory_mb": max(self.memory_usage) if self.memory_usage else 0,
                "average_memory_mb": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                "memory_measurements": len(self.memory_usage)
            }
    
    return PerformanceMonitor()


@pytest.fixture
def quality_simulator():
    """Simulate image quality metrics"""
    
    class QualitySimulator:
        @staticmethod
        def simulate_quality_score(resolution: int, complexity: float) -> float:
            """Simulate quality score based on resolution and complexity"""
            base_score = 0.7
            
            # Resolution factor
            if resolution >= 1024:
                base_score += 0.1
            
            # Complexity factor
            complexity_factor = complexity * 0.2
            
            # Add some randomness
            import random
            random_factor = random.uniform(-0.05, 0.05)
            
            return min(1.0, base_score + complexity_factor + random_factor)
        
        @staticmethod
        def simulate_broadcast_compliance(quality_score: float) -> Dict[str, Any]:
            """Simulate broadcast compliance checks"""
            return {
                "compliant": quality_score > 0.8,
                "quality_score": quality_score,
                "issues": [] if quality_score > 0.8 else ["Low quality score"],
                "recommendations": ["Improve image quality"] if quality_score <= 0.8 else []
            }
    
    return QualitySimulator()


@pytest.fixture
def audio_mock():
    """Mock audio processing"""
    
    class AudioMock:
        @staticmethod
        def mock_audio_analysis(duration: float = 180.0) -> Dict[str, Any]:
            """Mock audio analysis results"""
            return {
                "duration": duration,
                "tempo": 120.0,
                "key": "C major",
                "time_signature": "4/4",
                "sections": [
                    {"type": "intro", "start": 0.0, "end": 15.0},
                    {"type": "verse", "start": 15.0, "end": 45.0},
                    {"type": "chorus", "start": 45.0, "end": 75.0},
                    {"type": "verse", "start": 75.0, "end": 105.0},
                    {"type": "chorus", "start": 105.0, "end": 135.0},
                    {"type": "bridge", "start": 135.0, "end": 150.0},
                    {"type": "chorus", "start": 150.0, "end": 180.0}
                ],
                "beats": [{"time": i * 0.5, "confidence": 0.9} for i in range(int(duration * 2))],
                "spectral_features": {
                    "spectral_centroid": 2000.0,
                    "zero_crossing_rate": 0.1,
                    "mfcc": [1.0, 0.5, -0.2, 0.1] * 13
                }
            }
        
        @staticmethod
        def mock_beat_detection(audio_data: np.ndarray, sample_rate: int) -> List[float]:
            """Mock beat detection"""
            duration = len(audio_data) / sample_rate
            beat_times = []
            
            # Generate beats at regular intervals
            for i in range(int(duration * 2)):  # 120 BPM
                beat_times.append(i * 0.5)
            
            return beat_times
    
    return AudioMock()


@pytest.fixture
def video_mock():
    """Mock video processing"""
    
    class VideoMock:
        @staticmethod
        def mock_video_assembly(image_sequence: List[Path], audio_path: Path) -> Dict[str, Any]:
            """Mock video assembly results"""
            return {
                "success": True,
                "output_path": "/tmp/assembled_video.mp4",
                "duration": 120.0,
                "frame_count": int(120 * 30),  # 30 FPS
                "resolution": (1920, 1080),
                "processing_time": 45.0,
                "transitions": ["cut", "crossfade", "whip_pan"],
                "effects": ["color_grading", "motion_blur"]
            }
        
        @staticmethod
        def mock_color_grading() -> Dict[str, Any]:
            """Mock color grading results"""
            return {
                "primary_correction": {
                    "exposure": 0.1,
                    "contrast": 15.0,
                    "saturation": 10.0
                },
                "secondary_correction": {
                    "skin_tone_balance": True,
                    "selective_desaturation": False
                },
                "creative_grade": {
                    "lut_applied": "cinematic_teal_orange",
                    "film_grain": 0.1
                },
                "broadcast_compliance": True
            }
    
    return VideoMock()


@pytest.fixture
def file_system_mock(temp_dir):
    """Mock file system operations"""
    
    class FileSystemMock:
        def __init__(self, base_dir: Path):
            self.base_dir = base_dir
        
        def create_mock_files(self, count: int = 10, size_kb: int = 100):
            """Create mock files for testing"""
            files = []
            for i in range(count):
                file_path = self.base_dir / f"mock_file_{i}.dat"
                content = b"x" * (size_kb * 1024)
                file_path.write_bytes(content)
                files.append(file_path)
            return files
        
        def create_nested_structure(self, depth: int = 3, breadth: int = 5):
            """Create nested directory structure"""
            def create_level(current_dir: Path, current_depth: int):
                if current_depth >= depth:
                    return
                
                for i in range(breadth):
                    subdir = current_dir / f"level_{current_depth}_dir_{i}"
                    subdir.mkdir(exist_ok=True)
                    
                    # Create some files
                    for j in range(3):
                        file_path = subdir / f"file_{j}.txt"
                        file_path.write_text(f"Content of level {current_depth} file {j}")
                    
                    create_level(subdir, current_depth + 1)
            
            create_level(self.base_dir, 0)
            return self.base_dir
    
    return FileSystemMock(temp_dir)


@pytest.fixture
def network_mock():
    """Mock network operations"""
    
    class NetworkMock:
        @staticmethod
        async def mock_api_call(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Mock API call"""
            # Simulate network delay
            await asyncio.sleep(0.1)
            
            # Simulate different responses based on endpoint
            if "generate" in endpoint:
                return {
                    "success": True,
                    "image_url": "http://example.com/generated_image.png",
                    "processing_time": 2.5
                }
            elif "health" in endpoint:
                return {
                    "status": "healthy",
                    "response_time": 0.1,
                    "uptime": 3600.0
                }
            else:
                return {"success": True, "message": "Mock API response"}
        
        @staticmethod
        def simulate_network_error(error_type: str = "timeout"):
            """Simulate network errors"""
            if error_type == "timeout":
                raise asyncio.TimeoutError("Network request timed out")
            elif error_type == "connection":
                raise ConnectionError("Failed to connect to server")
            elif error_type == "server":
                raise Exception("Server returned 500 error")
            else:
                raise Exception(f"Unknown network error: {error_type}")
    
    return NetworkMock()


@pytest.fixture
def database_mock():
    """Mock database operations"""
    
    class DatabaseMock:
        def __init__(self):
            self.data = {}
            self.connection_count = 0
        
        def connect(self):
            """Mock database connection"""
            self.connection_count += 1
            return self
        
        def disconnect(self):
            """Mock database disconnection"""
            self.connection_count = max(0, self.connection_count - 1)
        
        def insert(self, table: str, record: Dict[str, Any]):
            """Mock record insertion"""
            if table not in self.data:
                self.data[table] = []
            
            record_id = str(uuid.uuid4())
            record["id"] = record_id
            self.data[table].append(record)
            return record_id
        
        def select(self, table: str, conditions: Dict[str, Any] = None) -> List[Dict[str, Any]]:
            """Mock record selection"""
            if table not in self.data:
                return []
            
            records = self.data[table]
            
            if conditions:
                filtered_records = []
                for record in records:
                    match = True
                    for key, value in conditions.items():
                        if record.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_records.append(record)
                return filtered_records
            
            return records
        
        def update(self, table: str, record_id: str, updates: Dict[str, Any]) -> bool:
            """Mock record update"""
            if table not in self.data:
                return False
            
            for record in self.data[table]:
                if record.get("id") == record_id:
                    record.update(updates)
                    return True
            
            return False
        
        def delete(self, table: str, record_id: str) -> bool:
            """Mock record deletion"""
            if table not in self.data:
                return False
            
            for i, record in enumerate(self.data[table]):
                if record.get("id") == record_id:
                    del self.data[table][i]
                    return True
            
            return False
    
    return DatabaseMock()


@pytest.fixture
def logging_mock():
    """Mock logging operations"""
    
    class LoggingMock:
        def __init__(self):
            self.logs = []
            self.log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        def log(self, level: str, message: str, **kwargs):
            """Mock logging"""
            if level not in self.log_levels:
                level = "INFO"
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "extra": kwargs
            }
            
            self.logs.append(log_entry)
            return log_entry
        
        def get_logs(self, level: str = None, message_contains: str = None) -> List[Dict[str, Any]]:
            """Get filtered logs"""
            filtered_logs = self.logs
            
            if level:
                filtered_logs = [log for log in filtered_logs if log["level"] == level]
            
            if message_contains:
                filtered_logs = [
                    log for log in filtered_logs 
                    if message_contains in log["message"]
                ]
            
            return filtered_logs
        
        def clear_logs(self):
            """Clear all logs"""
            self.logs.clear()
        
        def get_statistics(self) -> Dict[str, Any]:
            """Get logging statistics"""
            level_counts = {}
            for level in self.log_levels:
                level_counts[level] = len([log for log in self.logs if log["level"] == level])
            
            return {
                "total_logs": len(self.logs),
                "level_distribution": level_counts,
                "first_log_time": self.logs[0]["timestamp"] if self.logs else None,
                "last_log_time": self.logs[-1]["timestamp"] if self.logs else None
            }
    
    return LoggingMock()


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers"""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runperformance", action="store_true", default=False, help="run performance tests"
    )
    parser.addoption(
        "--runintegration", action="store_true", default=False, help="run integration tests"
    )


if __name__ == "__main__":
    # Allow running this file directly for testing fixtures
    pytest.main([__file__, "-v"])
