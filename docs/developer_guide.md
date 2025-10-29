# APEX DIRECTOR Developer Guide

Comprehensive developer documentation for extending and customizing APEX DIRECTOR.

## Table of Contents

- [System Architecture](#system-architecture)
- [Development Environment](#development-environment)
- [Core Components](#core-components)
- [Extending the System](#extending-the-system)
- [API Development](#api-development)
- [Testing Framework](#testing-framework)
- [Performance Optimization](#performance-optimization)
- [Debugging and Profiling](#debugging-and-profiling)
- [Contributing Guidelines](#contributing-guidelines)

---

## System Architecture

### High-Level Architecture

APEX DIRECTOR follows a modular, microservices-inspired architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    APEX DIRECTOR CORE                       │
├─────────────────────────────────────────────────────────────┤
│  Orchestrator  │  Asset Manager  │  Checkpoint Manager      │
│  Estimator     │  Config Manager │  Backend Manager        │
├─────────────────────────────────────────────────────────────┤
│                    FEATURE LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  Image Gen │  Video Assembly │  Audio Analysis │  QA        │
├─────────────────────────────────────────────────────────────┤
│                    BACKEND LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  NanoBanana │  Google Imagen  │  MiniMax     │  SDXL       │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions

```python
# Example component interaction flow
async def music_video_generation_workflow():
    # 1. Audio Analysis
    audio_analysis = await audio_analyzer.analyze(audio_path)
    
    # 2. Scene Planning
    scenes = await cinematography_director.plan_scenes(audio_analysis)
    
    # 3. Image Generation
    images = []
    for scene in scenes:
        image = await image_generator.generate(scene.request)
        images.append(image)
    
    # 4. Quality Validation
    validated_images = await quality_validator.validate(images)
    
    # 5. Video Assembly
    video = await video_assembler.assemble(validated_images, audio_analysis)
    
    # 6. Final QA
    final_video = await quality_validator.final_check(video)
    
    return final_video
```

### Design Patterns

#### 1. Factory Pattern
```python
# Backend factory for creating backend instances
class BackendFactory:
    @staticmethod
    def create_backend(backend_type: str, config: Dict[str, Any]) -> BackendInterface:
        if backend_type == "nano_banana":
            return NanoBananaBackend(config)
        elif backend_type == "imagen":
            return GoogleImagenBackend(config)
        elif backend_type == "minimax":
            return MiniMaxBackend(config)
        # ... other backends
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
```

#### 2. Strategy Pattern
```python
# Quality scoring strategy
class QualityScoringStrategy:
    def __init__(self, strategy_type: str):
        self.strategy = self._get_strategy(strategy_type)
    
    def _get_strategy(self, strategy_type: str):
        if strategy_type == "aesthetic":
            return AestheticScoringStrategy()
        elif strategy_type == "technical":
            return TechnicalScoringStrategy()
        elif strategy_type == "composite":
            return CompositeScoringStrategy()
    
    def score(self, image: Image) -> float:
        return self.strategy.calculate_score(image)
```

#### 3. Observer Pattern
```python
# Progress notification system
class ProgressNotifier:
    def __init__(self):
        self.observers = []
    
    def attach(self, observer: Callable[[float, str], None]):
        self.observers.append(observer)
    
    def notify(self, progress: float, status: str):
        for observer in self.observers:
            observer(progress, status)
```

### Data Flow

```
Audio Input → Audio Analysis → Scene Planning → Image Generation → 
Quality Validation → Video Assembly → Post-Processing → Final Output
```

---

## Development Environment

### Setup

#### Prerequisites
```bash
# Python 3.8+ required
python --version

# Git for version control
git --version

# Node.js (for development tools)
node --version
npm --version
```

#### Installation
```bash
# Clone repository
git clone https://github.com/apex-director/core.git
cd apex-director

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Development Tools
```bash
# Formatter (Black)
black --version

# Linter (Flake8)
flake8 --version

# Type checker (MyPy)
mypy --version

# Test runner (Pytest)
pytest --version
```

### Project Structure

```
apex_director/
├── apex_director/           # Main package
│   ├── core/               # Core system components
│   ├── images/             # Image generation module
│   ├── video/              # Video assembly module
│   ├── audio/              # Audio processing module
│   ├── cinematography/     # Cinematography system
│   ├── qa/                 # Quality assurance module
│   └── director.py         # Main orchestrator
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── performance/       # Performance tests
│   └── e2e/               # End-to-end tests
├── docs/                   # Documentation
├── examples/              # Usage examples
├── scripts/               # Development scripts
└── tools/                 # Development tools
```

### Code Standards

#### Python Style Guide
```python
# Follow PEP 8 with these exceptions
# - Maximum line length: 100 characters
# - Use Black for formatting
# - Use type hints everywhere

from typing import List, Dict, Optional, Union
from pathlib import Path
import asyncio

class ExampleClass:
    """Example class following APEX DIRECTOR standards."""
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Union[str, int]],
        optional_param: Optional[str] = None,
    ) -> None:
        self.name = name
        self.config = config
        self.optional_param = optional_param
        self._private_attribute = "internal"
    
    async def process_data(
        self,
        data: List[Dict[str, str]],
        output_path: Path,
    ) -> Dict[str, str]:
        """
        Process data and return results.
        
        Args:
            data: List of data dictionaries to process
            output_path: Path for output file
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            ValueError: If data is invalid
            FileNotFoundError: If output_path is invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        if not output_path.parent.exists():
            raise FileNotFoundError(f"Output directory not found: {output_path.parent}")
        
        # Process data
        results = []
        for item in data:
            processed = await self._process_item(item)
            results.append(processed)
        
        # Write results
        await self._write_results(results, output_path)
        
        return {
            "processed_count": len(results),
            "output_path": str(output_path),
            "status": "completed"
        }
    
    async def _process_item(self, item: Dict[str, str]) -> Dict[str, str]:
        """Private method for processing individual items."""
        return {"processed": item["data"].upper()}
    
    async def _write_results(
        self,
        results: List[Dict[str, str]],
        output_path: Path,
    ) -> None:
        """Private method for writing results to file."""
        import json
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
```

#### Documentation Standards
```python
def complex_function(
    param1: str,
    param2: int,
    param3: Optional[List[Dict[str, Any]]] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Brief description of what the function does.
    
    Longer description explaining the purpose, behavior,
    and any important considerations for users.
    
    Args:
        param1: Description of param1, including type and constraints
        param2: Description of param2, including valid ranges
        param3: Optional description. Default is None if optional.
        
    Returns:
        Description of return value, including type and structure
        
    Raises:
        ValueError: When param1 is invalid or empty
        TypeError: When param2 is not an integer
        FileNotFoundError: When required files are missing
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result)
        {'status': 'success', 'value': 'TEST42'}
        
    Note:
        Any additional notes about usage, performance, or edge cases.
    """
    pass
```

---

## Core Components

### Orchestrator

The central coordinator managing all system components.

```python
class APEXOrchestrator:
    """Main orchestrator for APEX DIRECTOR system."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        base_dir: Path,
        backend_manager: Optional[BackendManager] = None,
        asset_manager: Optional[AssetManager] = None,
    ) -> None:
        self.config = config
        self.base_dir = base_dir
        self.backend_manager = backend_manager or BackendManager(config)
        self.asset_manager = asset_manager or AssetManager(base_dir)
        self.job_queue = asyncio.Queue()
        self.active_jobs: Dict[str, Job] = {}
        self._shutdown_event = asyncio.Event()
    
    async def submit_job(self, job_request: Dict[str, Any]) -> str:
        """Submit a job for processing."""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            request=job_request,
            status="queued",
            created_at=datetime.now(),
        )
        
        await self.job_queue.put(job)
        self.active_jobs[job_id] = job
        
        # Start processing if not already running
        if not hasattr(self, '_processing_task'):
            self._processing_task = asyncio.create_task(self._process_queue())
        
        return job_id
    
    async def _process_queue(self) -> None:
        """Process jobs from the queue."""
        while not self._shutdown_event.is_set():
            try:
                job = await asyncio.wait_for(
                    self.job_queue.get(),
                    timeout=1.0
                )
                
                await self._process_job(job)
                self.job_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing job queue: {e}")
```

### Asset Manager

Manages all system assets with organized storage.

```python
class AssetManager:
    """Manages asset storage, organization, and metadata."""
    
    def __init__(self, base_dir: Path, config: Dict[str, Any] = None) -> None:
        self.base_dir = base_dir
        self.config = config or {}
        self.metadata_db = self._init_metadata_db()
        self._ensure_directories()
    
    def store_asset(
        self,
        asset_data: Dict[str, Any],
        project_id: Optional[str] = None,
    ) -> Path:
        """Store an asset with automatic organization."""
        # Generate unique filename
        filename = self._generate_filename(asset_data)
        
        # Determine storage location
        if project_id:
            storage_path = self.base_dir / "projects" / project_id / "assets"
        else:
            storage_path = self.base_dir / "assets" / "general"
        
        storage_path.mkdir(parents=True, exist_ok=True)
        asset_path = storage_path / filename
        
        # Write asset file
        with open(asset_path, "wb") as f:
            f.write(asset_data["content"])
        
        # Store metadata
        metadata = self._extract_metadata(asset_data)
        metadata["path"] = str(asset_path)
        metadata["created_at"] = datetime.now().isoformat()
        self.metadata_db[asset_path] = metadata
        
        return asset_path
    
    def search_assets(
        self,
        query: str = "",
        category: str = "",
        tags: Optional[List[str]] = None,
        **filters: Any,
    ) -> List[Path]:
        """Search for assets using various criteria."""
        results = []
        
        for asset_path, metadata in self.metadata_db.items():
            # Text search
            if query and query.lower() not in str(asset_path).lower():
                continue
            
            # Category filter
            if category and metadata.get("category") != category:
                continue
            
            # Tags filter
            if tags:
                asset_tags = set(metadata.get("tags", []))
                if not any(tag in asset_tags for tag in tags):
                    continue
            
            # Additional filters
            if not all(
                metadata.get(key) == value
                for key, value in filters.items()
            ):
                continue
            
            results.append(Path(asset_path))
        
        return results
```

---

## Extending the System

### Adding New Backends

Create custom backend by implementing the `BackendInterface`:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BackendInterface(ABC):
    """Interface for all image generation backends."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend."""
        pass
    
    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image from a prompt."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health status."""
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get backend capabilities."""
        pass

# Custom backend implementation
class CustomBackend(BackendInterface):
    """Custom backend for specialized image generation."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.api_key = config.get("api_key")
        self.endpoint = config.get("endpoint")
        self._session = None
    
    async def initialize(self) -> bool:
        """Initialize the custom backend."""
        try:
            # Initialize API session
            self._session = aiohttp.ClientSession()
            
            # Test connection
            health = await self.health_check()
            return health.get("healthy", False)
            
        except Exception as e:
            logger.error(f"Failed to initialize custom backend: {e}")
            return False
    
    async def generate_image(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        quality: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate image using custom backend."""
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "quality": quality,
            **kwargs
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self._session.post(
                f"{self.endpoint}/generate",
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                
                if response.status != 200:
                    raise Exception(f"Generation failed: {result}")
                
                return {
                    "image_path": result["image_url"],
                    "metadata": {
                        "backend": "custom",
                        "prompt": prompt,
                        "parameters": payload,
                        "generated_at": datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health."""
        try:
            async with self._session.get(
                f"{self.endpoint}/health"
            ) as response:
                if response.status == 200:
                    return {
                        "healthy": True,
                        "response_time": 0.1,
                        "status": "operational"
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get backend capabilities."""
        return {
            "max_resolution": (1024, 1024),
            "supported_formats": ["png", "jpg"],
            "quality_levels": [1, 2, 3, 4, 5],
            "special_features": ["custom_filters", "advanced_prompts"]
        }
    
    async def close(self) -> None:
        """Cleanup resources."""
        if self._session:
            await self._session.close()
```

### Custom Quality Scorers

```python
class CustomQualityScorer:
    """Custom quality scoring implementation."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.weights = config.get("weights", {})
    
    async def score_image(
        self,
        image_path: Path,
        criteria: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Score image based on custom criteria."""
        scores = {}
        
        # Custom scoring logic
        scores["composition"] = await self._score_composition(image_path)
        scores["color_harmony"] = await self._score_color_harmony(image_path)
        scores["technical_quality"] = await self._score_technical_quality(image_path)
        
        # Calculate weighted average
        total_weight = sum(self.weights.values())
        weighted_score = sum(
            score * self.weights.get(metric, 1.0)
            for metric, score in scores.items()
        ) / total_weight
        
        return {
            "overall_score": weighted_score,
            "individual_scores": scores,
            "recommendations": self._generate_recommendations(scores)
        }
    
    async def _score_composition(self, image_path: Path) -> float:
        """Score image composition."""
        # Implement custom composition scoring
        # Could use OpenCV, PIL, or ML models
        return 0.85  # Placeholder
    
    async def _score_color_harmony(self, image_path: Path) -> float:
        """Score color harmony."""
        # Implement custom color harmony scoring
        return 0.78  # Placeholder
    
    async def _score_technical_quality(self, image_path: Path) -> float:
        """Score technical image quality."""
        # Implement custom technical quality scoring
        return 0.92  # Placeholder
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        for metric, score in scores.items():
            if score < 0.7:
                recommendations.append(f"Improve {metric} (current: {score:.2f})")
        
        return recommendations
```

### Custom Video Effects

```python
class CustomVideoEffect:
    """Custom video effect implementation."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.effect_id = config.get("effect_id", str(uuid.uuid4()))
    
    async def apply_effect(
        self,
        video_path: Path,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply custom effect to video."""
        output_path = video_path.parent / f"{video_path.stem}_custom_effect.mp4"
        
        # Custom effect implementation
        ffmpeg_cmd = self._build_ffmpeg_command(video_path, output_path, parameters)
        
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Effect application failed: {stderr.decode()}")
        
        return {
            "success": True,
            "output_path": output_path,
            "effect_id": self.effect_id,
            "parameters": parameters
        }
    
    def _build_ffmpeg_command(
        self,
        input_path: Path,
        output_path: Path,
        parameters: Dict[str, Any],
    ) -> List[str]:
        """Build FFmpeg command for custom effect."""
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-vf", self._build_filter_graph(parameters),
            "-c:a", "copy",
            "-y",  # Overwrite output file
            str(output_path)
        ]
        
        return cmd
    
    def _build_filter_graph(self, parameters: Dict[str, Any]) -> str:
        """Build FFmpeg filter graph for custom effect."""
        # Implement custom filter graph based on parameters
        effects = []
        
        if parameters.get("color_grade"):
            effects.append("colorbalance=rs=0.1:gs=-0.05:bs=-0.1")
        
        if parameters.get("vignette"):
            effects.append("vignette")
        
        if parameters.get("sharpen"):
            amount = parameters.get("sharpen_amount", 1.0)
            effects.append(f"unsharp=5:5:{amount}")
        
        return ",".join(effects)
```

---

## API Development

### REST API Framework

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="APEX DIRECTOR API",
    description="Professional music video generation API",
    version="1.0.0"
)

class MusicVideoRequest(BaseModel):
    audio_path: str
    output_dir: str
    genre: Optional[str] = "cinematic"
    quality_preset: Optional[str] = "high"
    max_shots: Optional[int] = 15

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    stage: str
    output_path: Optional[str] = None

@app.post("/api/v1/generate", response_model=dict)
async def generate_music_video(
    request: MusicVideoRequest,
    background_tasks: BackgroundTasks
) -> dict:
    """Generate music video from audio file."""
    try:
        # Submit generation job
        job_id = await submit_music_video_job(
            audio_path=request.audio_path,
            output_dir=request.output_dir,
            genre=request.genre,
            quality_preset=request.quality_preset,
            max_shots=request.max_shots
        )
        
        return {
            "job_id": job_id,
            "status": "submitted",
            "message": "Generation job started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get status of generation job."""
    try:
        status = get_job_status(job_id)
        
        return JobStatusResponse(
            job_id=job_id,
            status=status["status"],
            progress=status["progress"],
            stage=status.get("current_stage", "unknown"),
            output_path=status.get("output_path")
        )
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/cancel/{job_id}")
async def cancel_job(job_id: str) -> dict:
    """Cancel running generation job."""
    try:
        success = await cancel_music_video_job(job_id)
        
        if success:
            return {"message": "Job cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Unable to cancel job")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check() -> dict:
    """API health check."""
    system_status = get_system_status()
    
    return {
        "status": "healthy",
        "system": system_status,
        "timestamp": datetime.now().isoformat()
    }
```

### GraphQL API

```python
import strawberry
from typing import List, Optional
from strawberry.types import Info

@strawberry.type
class MusicVideoResult:
    job_id: str
    status: str
    output_path: Optional[str] = None
    quality_score: Optional[float] = None

@strawberry.type
class GenerationRequest:
    audio_path: str
    output_dir: str
    genre: str = "cinematic"
    quality_preset: str = "high"

@strawberry.type
class Query:
    @strawberry.field
    def job_status(self, job_id: str) -> Optional[MusicVideoResult]:
        status = get_job_status(job_id)
        if not status:
            return None
        
        return MusicVideoResult(
            job_id=job_id,
            status=status["status"],
            output_path=status.get("output_path"),
            quality_score=status.get("quality_score")
        )
    
    @strawberry.field
    def system_status(self) -> dict:
        return get_system_status()

@strawberry.type
class Mutation:
    @strawberry.field
    async def generate_music_video(
        self,
        request: GenerationRequest,
        info: Info
    ) -> MusicVideoResult:
        job_id = await submit_music_video_job(
            audio_path=request.audio_path,
            output_dir=request.output_dir,
            genre=request.genre,
            quality_preset=request.quality_preset
        )
        
        return MusicVideoResult(
            job_id=job_id,
            status="submitted"
        )
    
    @strawberry.field
    async def cancel_generation(self, job_id: str) -> bool:
        return await cancel_music_video_job(job_id)

schema = strawberry.Schema(query=Query, mutation=Mutation)
```

---

## Testing Framework

### Unit Testing

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

class TestImageGenerator:
    """Test suite for image generator."""
    
    @pytest.fixture
    def mock_backend(self):
        """Create mock backend for testing."""
        backend = Mock()
        backend.generate_image = AsyncMock(return_value={
            "image_path": "/tmp/test.png",
            "metadata": {"backend": "mock"}
        })
        return backend
    
    @pytest.fixture
    def image_generator(self, mock_backend):
        """Create image generator with mocked backend."""
        from apex_director.images import CinematicImageGenerator
        
        generator = CinematicImageGenerator()
        generator.backend_manager.available_backends["mock"] = mock_backend
        return generator
    
    @pytest.mark.asyncio
    async def test_generate_single_image(self, image_generator, mock_backend):
        """Test single image generation."""
        from apex_director.images import GenerationRequest
        
        request = GenerationRequest(
            prompt="A test image",
            scene_id="test_scene"
        )
        
        result = await image_generator.generate_single_image(request)
        
        assert result is not None
        assert "image_path" in result
        mock_backend.generate_image.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_with_retry(self, image_generator):
        """Test generation retry logic."""
        # Mock backend that fails twice then succeeds
        call_count = 0
        
        async def failing_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {
                "image_path": "/tmp/retried.png",
                "metadata": {"retries": call_count}
            }
        
        with patch.object(
            image_generator.backend_manager,
            'generate_single_backend',
            side_effect=failing_generate
        ):
            request = GenerationRequest(prompt="retry test")
            result = await image_generator.generate_single_image(request)
            
            assert result is not None
            assert call_count == 3
```

### Integration Testing

```python
import pytest
import tempfile
from pathlib import Path

class TestSystemIntegration:
    """Integration tests for complete system workflows."""
    
    @pytest.fixture
    def test_environment(self):
        """Create test environment with temporary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Setup test directories
            (temp_path / "audio").mkdir()
            (temp_path / "output").mkdir()
            (temp_path / "assets").mkdir()
            
            yield temp_path
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, test_environment):
        """Test complete music video generation workflow."""
        # Create test audio file
        audio_file = test_environment / "audio" / "test.mp3"
        audio_file.write_text("mock audio data")
        
        # Submit generation request
        job_id = await submit_music_video_job(
            audio_path=str(audio_file),
            output_dir=str(test_environment / "output"),
            max_shots=5
        )
        
        # Wait for completion
        max_wait = 30  # seconds
        wait_interval = 1  # seconds
        waited = 0
        
        while waited < max_wait:
            status = get_job_status(job_id)
            
            if status["status"] == "completed":
                break
            elif status["status"] == "failed":
                pytest.fail(f"Job failed: {status.get('error', 'Unknown error')}")
            
            await asyncio.sleep(wait_interval)
            waited += wait_interval
        
        # Verify output
        output_dir = Path(status["output_path"]).parent
        assert output_dir.exists()
        
        # Check for expected files
        expected_files = [
            "final_video.mp4",
            "audio_analysis.json",
            "production_log.txt"
        ]
        
        for filename in expected_files:
            assert (output_dir / filename).exists()
```

### Performance Testing

```python
import pytest
import time
import psutil
from typing import List, Dict

class TestPerformance:
    """Performance tests for system components."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_batch_generation_performance(self):
        """Test performance with batch generation."""
        requests = [
            GenerationRequest(
                prompt=f"Test image {i}",
                scene_id=f"batch_scene_{i}"
            ) for i in range(20)
        ]
        
        # Measure performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        results = await batch_generate(requests)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        # Performance assertions
        processing_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB
        
        assert len(results) == 20
        assert processing_time < 300  # Should complete within 5 minutes
        assert memory_usage < 500  # Should use less than 500MB additional memory
    
    @pytest.mark.performance
    def test_asset_manager_scalability(self, tmp_path):
        """Test asset manager performance with large datasets."""
        asset_manager = AssetManager(base_dir=tmp_path)
        
        # Create many assets
        start_time = time.time()
        
        for i in range(1000):
            asset_data = {
                "content": f"Asset {i}".encode(),
                "filename": f"asset_{i}.dat",
                "metadata": {"index": i}
            }
            asset_manager.store_asset(asset_data)
        
        creation_time = time.time() - start_time
        
        # Test search performance
        start_time = time.time()
        
        results = asset_manager.search_assets(metadata_filter={"index": 500})
        
        search_time = time.time() - start_time
        
        # Performance assertions
        assert creation_time < 60  # Should create 1000 assets in under 1 minute
        assert search_time < 1  # Should search within 1 second
        assert len(results) == 1
```

---

## Performance Optimization

### Async Performance

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncImageProcessor:
    """Optimized async image processing."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(
        self,
        images: List[Path],
        processor_func: callable,
    ) -> List[Dict[str, Any]]:
        """Process images in parallel using thread pool."""
        tasks = []
        
        for image_path in images:
            task = asyncio.create_task(
                self._process_single_image(image_path, processor_func)
            )
            tasks.append(task)
        
        # Process in batches to avoid overwhelming the system
        batch_size = self.max_workers * 2
        results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
        
        return [r for r in results if not isinstance(r, Exception)]
    
    async def _process_single_image(
        self,
        image_path: Path,
        processor_func: callable,
    ) -> Dict[str, Any]:
        """Process single image in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            processor_func,
            image_path
        )
```

### Memory Optimization

```python
import weakref
from functools import lru_cache

class OptimizedAssetManager:
    """Memory-optimized asset manager."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._metadata_cache = {}
        self._cache_size_limit = 1000
        self._setup_cache_management()
    
    def _setup_cache_management(self):
        """Setup cache size management using weak references."""
        self._cache_refs = weakref.WeakValueDictionary()
    
    @lru_cache(maxsize=128)
    def get_cached_metadata(self, asset_path: str) -> Dict[str, Any]:
        """Get metadata with caching."""
        return self._load_metadata(Path(asset_path))
    
    def store_asset_optimized(
        self,
        asset_data: Dict[str, Any],
        use_cache: bool = True,
    ) -> Path:
        """Store asset with memory optimization."""
        # Store file first
        asset_path = self._store_file(asset_data)
        
        # Cache metadata if enabled and cache has space
        if use_cache and len(self._metadata_cache) < self._cache_size_limit:
            metadata = self._extract_metadata(asset_data)
            metadata["path"] = str(asset_path)
            self._metadata_cache[str(asset_path)] = metadata
        
        return asset_path
    
    def cleanup_cache(self):
        """Cleanup cache when memory pressure detected."""
        if len(self._metadata_cache) > self._cache_size_limit:
            # Remove oldest entries
            keys_to_remove = list(self._metadata_cache.keys())[:100]
            for key in keys_to_remove:
                del self._metadata_cache[key]
```

### Database Optimization

```python
import sqlite3
from contextlib import contextmanager

class OptimizedMetadataDB:
    """Optimized metadata database."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database with optimized schema."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    category TEXT,
                    tags TEXT,  -- JSON array
                    metadata TEXT,  -- JSON object
                    created_at TEXT,
                    accessed_at TEXT,
                    file_size INTEGER,
                    file_hash TEXT
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON assets(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON assets(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON assets(file_hash)")
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with optimizations."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Balanced durability/performance
        conn.execute("PRAGMA cache_size=10000")  # Increase cache size
        
        try:
            yield conn
        finally:
            conn.close()
    
    def insert_asset_optimized(self, asset_data: Dict[str, Any]) -> str:
        """Insert asset with optimized batching."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO assets 
                (path, filename, category, tags, metadata, created_at, file_size, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(asset_data["path"]),
                asset_data["filename"],
                asset_data.get("category"),
                json.dumps(asset_data.get("tags", [])),
                json.dumps(asset_data.get("metadata", {})),
                datetime.now().isoformat(),
                len(asset_data.get("content", b"")),
                self._calculate_file_hash(asset_data.get("content", b""))
            ))
            
            conn.commit()
            return cursor.lastrowid
```

---

## Debugging and Profiling

### Debug Logging

```python
import logging
from functools import wraps

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('apex_director_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def debug_profile(func):
    """Decorator for profiling function calls."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Starting {func_name}")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = await func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            logger.debug(f"Completed {func_name} in {end_time - start_time:.2f}s")
            logger.debug(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {func_name}: {e}")
            raise
    
    return wrapper

class DebugProfiler:
    """Debug profiler for tracking performance."""
    
    def __init__(self):
        self.profiles = {}
        self.active_profiles = {}
    
    def start_profile(self, name: str):
        """Start profiling a section of code."""
        self.active_profiles[name] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss,
            'calls': 0
        }
    
    def end_profile(self, name: str):
        """End profiling and record results."""
        if name not in self.active_profiles:
            logger.warning(f"No active profile found for {name}")
            return
        
        profile = self.active_profiles[name]
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        duration = end_time - profile['start_time']
        memory_delta = end_memory - profile['start_memory']
        
        if name not in self.profiles:
            self.profiles[name] = {
                'total_time': 0,
                'total_memory': 0,
                'call_count': 0,
                'avg_time': 0,
                'avg_memory': 0
            }
        
        p = self.profiles[name]
        p['total_time'] += duration
        p['total_memory'] += memory_delta
        p['call_count'] += 1
        p['avg_time'] = p['total_time'] / p['call_count']
        p['avg_memory'] = p['total_memory'] / p['call_count']
        
        del self.active_profiles[name]
    
    def get_report(self) -> Dict[str, Any]:
        """Get profiling report."""
        return {
            'active_profiles': list(self.active_profiles.keys()),
            'completed_profiles': self.profiles
        }
```

### Error Tracking

```python
import traceback
from datetime import datetime

class ErrorTracker:
    """Comprehensive error tracking and reporting."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.error_count = 0
        self.error_types = {}
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        severity: str = "error"
    ):
        """Log error with full context."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'severity': severity,
            'error_id': str(uuid.uuid4())
        }
        
        # Increment counters
        self.error_count += 1
        error_type = type(error).__name__
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Error ID: {error_info['error_id']}\n")
            f.write(f"Time: {error_info['timestamp']}\n")
            f.write(f"Type: {error_info['error_type']}\n")
            f.write(f"Message: {error_info['error_message']}\n")
            f.write(f"Severity: {error_info['severity']}\n")
            f.write(f"Context: {error_info['context']}\n")
            f.write(f"Traceback:\n{error_info['traceback']}")
        
        # Also log to standard logger
        logger.error(
            f"Error {error_info['error_id']}: {error_info['error_message']}",
            extra={'context': context, 'traceback': error_info['traceback']}
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        return {
            'total_errors': self.error_count,
            'error_types': self.error_types,
            'most_common_error': max(self.error_types, key=self.error_types.get) if self.error_types else None
        }

# Global error tracker
error_tracker = ErrorTracker(Path("errors.log"))

def handle_errors(func):
    """Decorator for automatic error tracking."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            context = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            error_tracker.log_error(e, context)
            raise
    
    return wrapper
```

---

## Contributing Guidelines

### Development Workflow

1. **Fork and Clone**: Fork the repository and clone locally
2. **Create Branch**: Create feature branch from `develop`
3. **Develop**: Implement changes with tests
4. **Test**: Run full test suite
5. **Review**: Submit pull request with detailed description

### Code Review Process

#### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated for new functionality
- [ ] Documentation updated
- [ ] All tests pass
- [ ] Performance impact considered
- [ ] Security implications reviewed

#### Review Criteria

```markdown
## Code Review Checklist

### Functionality
- [ ] Code does what it claims to do
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] Performance is acceptable

### Code Quality
- [ ] Code is readable and well-commented
- [ ] Functions are appropriately sized
- [ ] Dependencies are minimized
- [ ] No code duplication

### Testing
- [ ] Unit tests cover new functionality
- [ ] Integration tests pass
- [ ] Edge cases are tested
- [ ] Performance tests updated if needed

### Documentation
- [ ] Function docstrings are complete
- [ ] README updates if needed
- [ ] API documentation updated
- [ ] Examples provided for new features
```

### Commit Standards

```bash
# Format: type(scope): description

# Features
feat(images): add custom backend interface
feat(video): implement Ken Burns effect
feat(api): add GraphQL endpoint for job status

# Bug fixes
fix(orchestrator): handle job cancellation edge case
fix(asset_manager): prevent memory leak in search
fix(audio): fix beat detection for irregular rhythms

# Documentation
docs(api): update image generation examples
docs(user): add troubleshooting section
docs(dev): clarify extension guidelines

# Tests
test(quality): add broadcast compliance tests
test(performance): benchmark memory usage
test(integration): end-to-end workflow test

# Refactoring
refactor(core): simplify error handling
refactor(images): extract common validation logic
refactor(video): optimize color grading pipeline

# Performance
perf(orchestrator): reduce queue processing latency
perf(asset_manager): optimize search algorithm
perf(images): parallelize batch generation

# Security
security(api): add input validation
security(asset): implement file size limits
security(core): sanitize user inputs
```

### Release Process

```python
# Release workflow script
import subprocess
from pathlib import Path

def create_release(version: str):
    """Create a new release."""
    # 1. Update version
    update_version(version)
    
    # 2. Run tests
    run_tests()
    
    # 3. Build documentation
    build_docs()
    
    # 4. Create changelog
    create_changelog(version)
    
    # 5. Create release commit
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", f"Release {version}"])
    
    # 6. Create and push tag
    subprocess.run(["git", "tag", f"v{version}"])
    subprocess.run(["git", "push", "origin", f"v{version}"])

def update_version(version: str):
    """Update version in all relevant files."""
    version_file = Path("apex_director/__init__.py")
    content = version_file.read_text()
    content = content.replace(f'__version__ = "current"', f'__version__ = "{version}"')
    version_file.write_text(content)
```

This developer guide provides the foundation for contributing to APEX DIRECTOR. Whether you're adding new features, fixing bugs, or optimizing performance, these guidelines will help ensure high-quality contributions that maintain the system's reliability and extensibility.

*For questions about development, please refer to the [community resources](../README.md#community) or open an issue on GitHub.*
