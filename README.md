# APEX DIRECTOR - Core System Architecture & Backend Abstraction

A comprehensive image generation orchestration system built with professional-grade architecture for production use.

## 🏗️ Architecture Overview

APEX DIRECTOR implements a robust multi-layered architecture designed for scalability, reliability, and maintainability:

### Core Components

1. **Main Orchestrator** (`orchestrator.py`) - Central coordinator managing the entire pipeline
2. **Backend Abstraction Layer** (`backend_manager.py`) - Unified interface for multiple image generators
3. **Asset Management System** (`asset_manager.py`) - Structured organization for 100+ generated files
4. **Checkpoint & Resume System** (`checkpoint.py`) - State management for failure recovery
5. **Cost & Time Estimator** (`estimator.py`) - Resource commitment prediction
6. **Configuration Management** (`config.py`) - Centralized settings for all components

## 🚀 Key Features

### Multi-Backend Support
- **Nano Banana** → **Google Imagen** → **MiniMax** → **SDXL**
- Automatic fallback cascade for reliability
- Load balancing and performance optimization
- Individual backend health monitoring

### Professional Error Handling
- Comprehensive retry logic with exponential backoff
- Graceful degradation when backends fail
- Detailed error logging and recovery procedures
- System health monitoring and alerting

### Asset Management
- Structured directory organization (`assets/images`, `assets/metadata`, etc.)
- Complete metadata schema with hashes and provenance tracking
- Duplicate detection and asset deduplication
- Project-based organization with tags and search
- Storage statistics and cleanup utilities

### State Management
- Automatic checkpoint creation every 5 minutes
- Manual checkpoint triggers for critical operations
- Complete system state recovery from checkpoints
- Job queue persistence across restarts
- Historical data tracking for analytics

### Cost & Time Prediction
- Machine learning-based estimation using historical data
- Confidence scoring for predictions
- Multi-factor complexity analysis (resolution, steps, prompt complexity)
- Backend-specific cost models
- Batch estimation for efficient processing

## 📁 Directory Structure

```
apex_director/
├── core/
│   ├── __init__.py          # Core module exports
│   ├── orchestrator.py      # Main system controller
│   ├── backend_manager.py   # Multi-backend abstraction
│   ├── asset_manager.py     # File organization & metadata
│   ├── checkpoint.py        # State management
│   ├── estimator.py         # Cost/time prediction
│   └── config.py            # Configuration settings
├── schemas/
│   └── __init__.py          # JSON schemas for validation
└── __init__.py              # Package initialization

assets/
├── images/                  # Generated images
├── metadata/                # Asset metadata & projects
├── cache/                   # Temporary files & caches
├── exports/                 # Final deliverables
├── projects/               # Project-specific assets
├── thumbnails/             # Image thumbnails
├── variants/               # Asset variants
└── checkpoints/            # System checkpoints
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- AsyncIO support
- 2GB+ available disk space for assets

### Installation

```bash
# Clone or copy the APEX DIRECTOR codebase
cd apex_director

# Install dependencies
uv pip install -r requirements.txt

# Initialize the system
python -c "import apex_director; apex_director.initialize()"
```

### Configuration

The system creates `apex_config.json` automatically on first run. Key configuration options:

```json
{
  "backends": {
    "nano_banana": {
      "enabled": true,
      "priority": 1,
      "cost_per_image": 0.01,
      "quality_level": 1
    },
    "imagen": {
      "enabled": true, 
      "priority": 2,
      "cost_per_image": 0.05,
      "quality_level": 3
    }
  },
  "orchestrator": {
    "max_concurrent_jobs": 5,
    "checkpoint_interval": 300,
    "auto_retry": true
  }
}
```

## 💡 Usage Examples

### Basic Usage

```python
import asyncio
from apex_director import submit_generation_job, start_orchestrator, get_system_status

async def main():
    # Start the system
    await start_orchestrator()
    
    # Submit a generation job
    job_id = await submit_generation_job(
        prompt="A majestic lion in a savanna at sunset",
        priority=3,
        generation_params={
            "width": 512,
            "height": 512,
            "steps": 20,
            "quality_level": 3
        }
    )
    
    print(f"Job submitted: {job_id}")
    
    # Check system status
    status = get_system_status()
    print(f"Active jobs: {status['jobs']['active_jobs']}")
    print(f"Success rate: {status['performance']['success_rate']:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Batch Processing

```python
from apex_director import get_orchestrator

orchestrator = get_orchestrator()

# Submit multiple jobs
jobs = [
    {"prompt": "Mountain lake reflection", "priority": 4},
    {"prompt": "Cyberpunk city at night", "priority": 2},
    {"prompt": "Cozy forest cottage", "priority": 3}
]

job_ids = await orchestrator.submit_batch(jobs)
print(f"Batch submitted: {len(job_ids)} jobs")
```

### Cost Estimation

```python
from apex_director import get_estimator

estimator = get_estimator()

# Get cost and time estimate
estimate = estimator.estimate_generation({
    "width": 1024,
    "height": 1024,
    "steps": 50,
    "quality_level": 5
}, "A highly detailed fantasy landscape")

print(f"Estimated cost: ${estimate.estimated_cost:.4f}")
print(f"Estimated time: {estimate.estimated_time_seconds:.1f} seconds")
print(f"Confidence: {estimate.confidence_score:.2%}")
```

### Asset Management

```python
from apex_director import get_asset_manager

asset_manager = get_asset_manager()

# Create project
project_id = asset_manager.create_project(
    name="Demo Project",
    description="Example project"
)

# Search assets
results = asset_manager.search_assets(
    query="landscape",
    backend="imagen",
    format_filter=[".png", ".jpg"]
)

print(f"Found {len(results)} matching assets")

# Get storage statistics
stats = asset_manager.get_storage_stats()
print(f"Total files: {stats['total_files']}")
print(f"Storage used: {stats['total_size_mb']:.2f} MB")
```

### Checkpoint System

```python
from apex_director import (
    get_checkpoint_manager, 
    create_checkpoint, 
    restore_from_checkpoint
)

checkpoint_manager = get_checkpoint_manager()

# Manual checkpoint
checkpoint_id = await create_checkpoint("before_batch_processing")

# List available checkpoints
checkpoints = await checkpoint_manager.list_checkpoints()
for cp in checkpoints:
    print(f"{cp['id']}: {cp['timestamp']} ({cp['job_count']} jobs)")

# Restore from checkpoint
success = await restore_from_checkpoint(checkpoint_id)
if success:
    print("System restored from checkpoint")
```

## 🛠️ System Components

### Backend Manager
- **Purpose**: Abstract multiple image generation services
- **Features**: Automatic fallback, health monitoring, load balancing
- **Backends**: Nano Banana → Imagen → MiniMax → SDXL cascade

### Asset Manager  
- **Purpose**: Organize and track generated assets
- **Features**: Metadata management, duplicate detection, project organization
- **Storage**: Structured directory layout with automatic organization

### Estimator
- **Purpose**: Predict cost and time requirements
- **Features**: ML-based predictions, confidence scoring, historical data
- **Accuracy**: Improves over time with usage data

### Checkpoint Manager
- **Purpose**: Enable failure recovery and state persistence
- **Features**: Automatic checkpoints, manual triggers, recovery procedures
- **Reliability**: Ensures no job loss during system failures

### Orchestrator
- **Purpose**: Coordinate all system components
- **Features**: Job queuing, priority management, error handling
- **Scalability**: Multi-processor support with configurable concurrency

## 📊 Monitoring & Statistics

### System Health
```python
status = get_system_status()
print(f"Overall status: {status['orchestrator']['status']}")
print(f"Uptime: {status['orchestrator']['uptime_seconds']/3600:.1f} hours")
print(f"Backend availability: {status['backends']['availability_percent']:.1f}%")
```

### Performance Metrics
```python
orchestrator = get_orchestrator()
stats = orchestrator.get_statistics()
print(f"Jobs per hour: {stats['jobs_per_hour']:.1f}")
print(f"Average cost per job: ${stats['average_cost_per_job']:.4f}")
print(f"Success rate: {stats['success_rate']:.2%}")
```

## 🔒 Production Considerations

### Security
- Input validation for all prompts and parameters
- Resource usage limits to prevent abuse
- Secure API key management for backend services
- Asset access controls and permissions

### Reliability
- Comprehensive error handling and recovery
- Automatic retry logic with configurable limits
- Circuit breaker pattern for failing backends
- Graceful degradation when services are unavailable

### Scalability
- Horizontal scaling through job distribution
- Configurable concurrent job limits
- Efficient memory usage for large asset collections
- Database integration for high-volume deployments

### Monitoring
- Real-time system health monitoring
- Performance metrics collection
- Error tracking and alerting
- Audit trails for compliance

## 🧪 Testing

Run the comprehensive example:

```bash
python example_usage.py
```

This demonstrates:
- Basic job submission and processing
- Batch operations
- Cost estimation
- Asset management
- Checkpoint operations
- System monitoring

## 📚 API Reference

### Core Functions

#### Job Management
- `submit_generation_job()` - Submit individual job
- `submit_batch()` - Submit multiple jobs
- `cancel_job()` - Cancel queued/in-progress job
- `get_job_status()` - Get job status and progress

#### System Control  
- `start_orchestrator()` - Start the system
- `stop_orchestrator()` - Stop the system gracefully
- `get_system_status()` - Get comprehensive status

#### Asset Management
- `get_asset_manager()` - Access asset management
- `search_assets()` - Find assets by criteria
- `get_storage_stats()` - Storage utilization

#### Estimation
- `estimate_generation_cost_time()` - Predict job requirements
- `add_generation_record()` - Improve estimation accuracy

## 🤝 Contributing

The system is designed for extensibility:

1. **Add New Backends**: Implement `BackendInterface` 
2. **Custom Estimators**: Extend `EstimationEngine`
3. **Asset Formats**: Add support to `AssetManager`
4. **Storage Backends**: Implement alternative storage in `AssetManager`

## 📄 License

This is a production-ready system architecture implementation. See individual component documentation for specific API details.

## 🆘 Support

For issues, questions, or contributions, please refer to the component-specific documentation in each module.

---

**APEX DIRECTOR** - Professional image generation orchestration at scale.
