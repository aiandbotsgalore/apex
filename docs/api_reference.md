# APEX DIRECTOR API Reference

Comprehensive API documentation for the APEX DIRECTOR professional media production system.

## Table of Contents

- [Core System APIs](#core-system-apis)
- [Image Generation APIs](#image-generation-apis)
- [Video Assembly APIs](#video-assembly-apis)
- [Audio Processing APIs](#audio-processing-apis)
- [Asset Management APIs](#asset-management-apis)
- [Quality Assurance APIs](#quality-assurance-apis)
- [Configuration APIs](#configuration-apis)
- [Error Handling](#error-handling)

---

## Core System APIs

### APEXOrchestrator

The central coordinator managing the entire pipeline.

#### Class: APEXOrchestrator

**Constructor:**
```python
APEXOrchestrator(config: Dict[str, Any], base_dir: Path)
```

**Parameters:**
- `config` - System configuration dictionary
- `base_dir` - Base directory for system files

**Methods:**

##### `async initialize() -> bool`
Initialize the orchestrator and all components.
- **Returns:** `True` if initialization successful

##### `async start() -> None`
Start the orchestrator and begin processing jobs.

##### `async stop() -> None`
Stop the orchestrator gracefully.

##### `async submit_job(job_request: Dict[str, Any]) -> str`
Submit a single job for processing.
- **Parameters:**
  - `job_request` - Job configuration dictionary
- **Returns:** Job ID string

##### `async submit_batch(job_requests: List[Dict[str, Any]]) -> List[str]`
Submit multiple jobs in batch.
- **Parameters:**
  - `job_requests` - List of job configuration dictionaries
- **Returns:** List of job IDs

##### `async get_job_status(job_id: str) -> Dict[str, Any]`
Get the status and details of a specific job.
- **Parameters:**
  - `job_id` - Unique job identifier
- **Returns:** Job status dictionary

##### `async cancel_job(job_id: str) -> bool`
Cancel a queued or in-progress job.
- **Parameters:**
  - `job_id` - Job to cancel
- **Returns:** `True` if cancellation successful

##### `async create_checkpoint(name: str, description: str = "", jobs: List[Dict[str, Any]] = None) -> str`
Create a system checkpoint.
- **Parameters:**
  - `name` - Checkpoint name
  - `description` - Checkpoint description
  - `jobs` - List of job states to checkpoint
- **Returns:** Checkpoint ID

##### `async restore_from_checkpoint(checkpoint_id: str) -> bool`
Restore system from checkpoint.
- **Parameters:**
  - `checkpoint_id` - Checkpoint to restore from
- **Returns:** `True` if restoration successful

##### `async get_system_stats() -> Dict[str, Any]`
Get comprehensive system statistics.
- **Returns:** System statistics dictionary

---

## Image Generation APIs

### CinematicImageGenerator

Main image generation engine with multi-backend support.

#### Class: CinematicImageGenerator

**Constructor:**
```python
CinematicImageGenerator(base_dir: Path, config: Dict[str, Any] = None)
```

**Methods:**

##### `async generate_single_image(request: GenerationRequest) -> Dict[str, Any]`
Generate a single image with cinematic quality.
- **Parameters:**
  - `request` - Generation request configuration
- **Returns:** Generation result dictionary

##### `async generate_image_sequence(requests: List[GenerationRequest]) -> Dict[str, Any]`
Generate a sequence of images with style consistency.
- **Parameters:**
  - `requests` - List of generation requests
- **Returns:** Sequence generation result

##### `async generate_with_variants(request: GenerationRequest) -> Dict[str, Any]`
Generate multiple variants and select the best.
- **Parameters:**
  - `request` - Generation request with variant settings
- **Returns:** Variant selection result

##### `async generate_batch(requests: List[GenerationRequest]) -> List[Dict[str, Any]]`
Generate multiple images in batch.
- **Parameters:**
  - `requests` - List of generation requests
- **Returns:** List of generation results

##### `async upscale_image(image_path: Path, preset: str = "high_quality") -> Dict[str, Any]`
Upscale an image using professional algorithms.
- **Parameters:**
  - `image_path` - Path to input image
  - `preset` - Upscaling preset ("web", "high_quality", "broadcast")
- **Returns:** Upscaling result dictionary

##### `async validate_image_quality(image_path: Path) -> Dict[str, Any]`
Validate image quality against broadcast standards.
- **Parameters:**
  - `image_path` - Path to image to validate
- **Returns:** Quality validation result

### GenerationRequest

Configuration for image generation requests.

#### Class: GenerationRequest

**Constructor:**
```python
GenerationRequest(
    prompt: str,
    scene_id: str = "",
    genre: str = "cinematic",
    director_style: str = "",
    camera_settings: Dict[str, Any] = None,
    lighting_setup: Dict[str, Any] = None,
    composition: Dict[str, Any] = None,
    character_name: str = "",
    style_constraints: bool = False,
    maintain_character: bool = False,
    upscale: bool = False,
    upscale_preset: str = "high_quality"
)
```

**Attributes:**
- `prompt` - Text description of the desired image
- `scene_id` - Unique identifier for the scene
- `genre` - Visual genre ("cinematic", "abstract", "fantasy", etc.)
- `director_style` - Director style reference ("christopher_nolan", "wes_anderson", etc.)
- `camera_settings` - Camera parameters (lens, aperture, ISO, etc.)
- `lighting_setup` - Lighting configuration
- `composition` - Composition rules and guidelines
- `character_name` - Character identifier for consistency
- `style_constraints` - Enable style consistency enforcement
- `maintain_character` - Enable character consistency
- `upscale` - Enable professional upscaling
- `upscale_preset` - Upscaling quality preset

### StylePersistenceManager

Manages style consistency across image sequences.

#### Class: StylePersistenceManager

**Methods:**

##### `load_style_bible(style_data: Dict[str, Any]) -> None`
Load style configuration from dictionary.
- **Parameters:**
  - `style_data` - Style bible configuration

##### `update_style(updates: Dict[str, Any]) -> None`
Update current style with new parameters.
- **Parameters:**
  - `updates` - Style updates to apply

##### `detect_style_drift(image_path: Path) -> Dict[str, Any]`
Detect style drift in generated image.
- **Parameters:**
  - `image_path` - Path to image to analyze
- **Returns:** Drift analysis result

##### `calculate_style_similarity(style1: Dict[str, Any], style2: Dict[str, Any]) -> float`
Calculate similarity between two style definitions.
- **Parameters:**
  - `style1` - First style definition
  - `style2` - Second style definition
- **Returns:** Similarity score (0.0 to 1.0)

### CharacterConsistencyManager

Manages character consistency across images.

#### Class: CharacterConsistencyManager

**Methods:**

##### `async create_character_profile(name: str, reference_images: List[Path], description: str) -> str`
Create a new character profile for consistency tracking.
- **Parameters:**
  - `name` - Character name/identifier
  - `reference_images` - List of reference image paths
  - `description` - Character description
- **Returns:** Character profile ID

##### `async validate_consistency(image_path: Path, character_id: str) -> Tuple[bool, float]`
Validate character consistency in generated image.
- **Parameters:**
  - `image_path` - Path to generated image
  - `character_id` - Character profile ID
- **Returns:** Tuple of (is_consistent, confidence_score)

##### `async find_similar_characters(character_id: str) -> List[Dict[str, Any]]`
Find characters similar to the specified character.
- **Parameters:**
  - `character_id` - Character profile ID
- **Returns:** List of similar characters with similarity scores

---

## Video Assembly APIs

### VideoAssembler

Professional video assembly and post-production engine.

#### Class: VideoAssembler

**Constructor:**
```python
VideoAssembler(base_dir: Path, config: Dict[str, Any] = None)
```

**Methods:**

##### `async assemble_video(job: AssemblyJob) -> Dict[str, Any]`
Assemble a complete video from components.
- **Parameters:**
  - `job` - Assembly job configuration
- **Returns:** Assembly result dictionary

##### `add_transition(transition_type: str, start_time: float, end_time: float, parameters: Dict[str, Any] = None) -> str`
Add a transition between clips.
- **Parameters:**
  - `transition_type` - Type of transition ("cut", "crossfade", "whip_pan")
  - `start_time` - Transition start time
  - `end_time` - Transition end time
  - `parameters` - Transition-specific parameters
- **Returns:** Transition ID

##### `apply_color_grade(stage: int, corrections: Dict[str, Any]) -> bool`
Apply color grading corrections.
- **Parameters:**
  - `stage` - Grading stage (1-4)
  - `corrections` - Color correction parameters
- **Returns:** `True` if successful

##### `add_motion_effect(effect_type: str, start_time: float, end_time: float, parameters: Dict[str, Any]) -> str`
Add motion effects and camera movements.
- **Parameters:**
  - `effect_type` - Type of motion effect
  - `start_time` - Effect start time
  - `end_time` - Effect end time
  - `parameters` - Effect parameters
- **Returns:** Effect ID

### AssemblyJob

Configuration for video assembly jobs.

#### Class: AssemblyJob

**Constructor:**
```python
AssemblyJob(
    job_id: str,
    audio_path: Path,
    output_path: Path,
    image_sequence: List[Dict[str, Any]],
    quality_mode: str = "high",
    assembly_mode: str = "offline"
)
```

**Attributes:**
- `job_id` - Unique job identifier
- `audio_path` - Path to audio file
- `output_path` - Path for output video
- `image_sequence` - List of image sequence definitions
- `quality_mode` - Quality preset ("draft", "web", "high", "broadcast", "cinema")
- `assembly_mode` - Assembly mode ("offline", "realtime")

### ColorGrader

Professional 4-stage color grading system.

#### Class: ColorGrader

**Methods:**

##### `set_primary_correction(exposure: float, contrast: float, brightness: float, saturation: float) -> None`
Set primary color corrections.
- **Parameters:**
  - `exposure` - Exposure adjustment (-2.0 to 2.0)
  - `contrast` - Contrast adjustment (0-100)
  - `brightness` - Brightness adjustment (0-100)
  - `saturation` - Saturation adjustment (0-100)

##### `apply_lut(lut_path: Path, strength: float = 1.0) -> bool`
Apply a Look-Up Table for creative grading.
- **Parameters:**
  - `lut_path` - Path to LUT file
  - `strength` - LUT effect strength (0.0 to 1.0)
- **Returns:** `True` if LUT applied successfully

##### `add_finishing_effect(effect_name: str, parameters: Dict[str, Any]) -> bool`
Add finishing effects (film grain, vignette, etc.).
- **Parameters:**
  - `effect_name` - Name of finishing effect
  - `parameters` - Effect parameters
- **Returns:** `True` if effect added successfully

---

## Audio Processing APIs

### AudioAnalyzer

Advanced audio analysis for video synchronization.

#### Class: AudioAnalyzer

**Methods:**

##### `async analyze_audio(audio_path: Path) -> AudioAnalysis`
Perform comprehensive audio analysis.
- **Parameters:**
  - `audio_path` - Path to audio file
- **Returns:** AudioAnalysis object with detailed metrics

##### `detect_beats(audio_path: Path, sensitivity: float = 0.5) -> List[BeatMarker]`
Detect beat positions in audio.
- **Parameters:**
  - `audio_path` - Path to audio file
  - `sensitivity` - Beat detection sensitivity
- **Returns:** List of beat markers with timestamps

##### `analyze_sections(audio_path: Path) -> List[AudioSection]`
Identify musical sections (verse, chorus, bridge, etc.).
- **Parameters:**
  - `audio_path` - Path to audio file
- **Returns:** List of audio sections with types and boundaries

##### `generate_spectral_analysis(audio_path: Path) -> SpectralFeatures`
Generate spectral feature analysis for video timing.
- **Parameters:**
  - `audio_path` - Path to audio file
- **Returns:** SpectralFeatures object with frequency analysis

### AudioAnalysis

Container for audio analysis results.

#### Class: AudioAnalysis

**Attributes:**
- `duration` - Total audio duration in seconds
- `tempo` - Estimated beats per minute
- `key` - Musical key
- `time_signature` - Time signature (e.g., "4/4")
- `sections` - List of musical sections
- `beats` - List of beat markers
- `spectral_features` - Spectral analysis data

---

## Asset Management APIs

### AssetManager

Complete asset organization and metadata management.

#### Class: AssetManager

**Constructor:**
```python
AssetManager(base_dir: Path, config: Dict[str, Any] = None)
```

**Methods:**

##### `create_project(name: str, description: str = "", tags: List[str] = None) -> Project`
Create a new project for organizing assets.
- **Parameters:**
  - `name` - Project name
  - `description` - Project description
  - `tags` - Project tags
- **Returns:** Project object

##### `store_asset(asset_data: Dict[str, Any], project_id: str = None) -> Path`
Store an asset with automatic organization.
- **Parameters:**
  - `asset_data` - Asset data dictionary with content and metadata
  - `project_id` - Optional project ID for organization
- **Returns:** Path to stored asset

##### `search_assets(query: str = "", category: str = "", tags: List[str] = None, **filters) -> List[Path]`
Search for assets using various criteria.
- **Parameters:**
  - `query` - Text search query
  - `category` - Asset category filter
  - `tags` - Tag filter list
  - **filters** - Additional custom filters
- **Returns:** List of matching asset paths

##### `get_asset_metadata(asset_path: Path) -> Dict[str, Any]`
Retrieve metadata for a specific asset.
- **Parameters:**
  - `asset_path` - Path to asset
- **Returns:** Asset metadata dictionary

##### `update_asset_metadata(asset_path: Path, updates: Dict[str, Any]) -> bool`
Update asset metadata.
- **Parameters:**
  - `asset_path` - Path to asset
  - `updates` - Metadata updates to apply
- **Returns:** `True` if update successful

##### `find_duplicates() -> List[List[Path]]`
Find duplicate assets in the system.
- **Returns:** List of duplicate asset groups

##### `get_storage_statistics() -> Dict[str, Any]`
Get comprehensive storage usage statistics.
- **Returns:** Storage statistics dictionary

##### `cleanup_temp_files(max_age_hours: int = 24) -> int`
Clean up temporary files older than specified age.
- **Parameters:**
  - `max_age_hours` - Maximum age in hours
- **Returns:** Number of files cleaned up

---

## Quality Assurance APIs

### QualityValidator

Comprehensive quality validation and broadcast compliance checking.

#### Class: QualityValidator

**Methods:**

##### `validate_image(image_path: Path, standards: List[str] = None) -> QualityReport`
Validate image quality against specified standards.
- **Parameters:**
  - `image_path` - Path to image to validate
  - `standards` - List of standards to check against
- **Returns:** QualityReport with validation results

##### `validate_video(video_path: Path) -> QualityReport`
Validate video quality and broadcast compliance.
- **Parameters:**
  - `video_path` - Path to video to validate
- **Returns:** QualityReport with validation results

##### `check_broadcast_standards(content_path: Path) -> BroadcastComplianceReport`
Check content against broadcast television standards.
- **Parameters:**
  - `content_path` - Path to content to check
- **Returns:** BroadcastComplianceReport

##### `detect_artifacts(image_path: Path) -> List[Artifact]`
Detect various types of image artifacts.
- **Parameters:**
  - `image_path` - Path to image to analyze
- **Returns:** List of detected artifacts with severity levels

### QualityReport

Comprehensive quality assessment report.

#### Class: QualityReport

**Attributes:**
- `overall_score` - Overall quality score (0.0 to 1.0)
- `broadcast_compliant` - Boolean broadcast compliance status
- `issues` - List of identified quality issues
- `recommendations` - List of improvement recommendations
- `detailed_scores` - Dictionary of specific quality metrics

---

## Configuration APIs

### ConfigManager

Centralized configuration management system.

#### Class: ConfigManager

**Constructor:**
```python
ConfigManager(config_path: Path = None, default_config: Dict[str, Any] = None)
```

**Methods:**

##### `load_config(config_path: Path = None) -> Dict[str, Any]`
Load configuration from file or defaults.
- **Parameters:**
  - `config_path` - Optional path to config file
- **Returns:** Configuration dictionary

##### `save_config(config: Dict[str, Any], config_path: Path = None) -> bool`
Save configuration to file.
- **Parameters:**
  - `config` - Configuration to save
  - `config_path` - Optional path to save to
- **Returns:** `True` if save successful

##### `get(path: str, default: Any = None) -> Any`
Get configuration value using dot notation.
- **Parameters:**
  - `path` - Configuration path (e.g., "backends.imagen.priority")
  - `default` - Default value if path not found
- **Returns:** Configuration value

##### `set(path: str, value: Any) -> None`
Set configuration value using dot notation.
- **Parameters:**
  - `path` - Configuration path
  - `value` - Value to set

##### `update(updates: Dict[str, Any]) -> None`
Update multiple configuration values.
- **Parameters:**
  - `updates` - Dictionary of configuration updates

---

## Error Handling

APEX DIRECTOR uses a comprehensive error handling system with custom exception types.

### Exception Types

#### `APEXDirectorError`
Base exception for all APEX DIRECTOR errors.

#### `GenerationError`
Image generation related errors.
- **Attributes:**
  - `backend` - Backend that failed
  - `prompt` - Generation prompt
  - `error_code` - Specific error code

#### `AssemblyError`
Video assembly related errors.
- **Attributes:**
  - `job_id` - Assembly job ID
  - `component` - Component that failed
  - `stage` - Assembly stage where error occurred

#### `AssetError`
Asset management related errors.
- **Attributes:**
  - `asset_path` - Path to problematic asset
  - `operation` - Operation that failed
  - `details` - Additional error details

#### `ConfigurationError`
Configuration related errors.
- **Attributes:**
  - `config_path` - Configuration path
  - `validation_errors` - List of validation errors

#### `BackendError`
Backend service related errors.
- **Attributes:**
  - `backend_name` - Name of failed backend
  - `status_code` - HTTP status code if applicable
  - `response` - Backend response data

### Error Recovery

APEX DIRECTOR implements automatic error recovery mechanisms:

1. **Backend Failover**: Automatic switching to backup backends
2. **Job Retry**: Configurable retry logic with exponential backoff
3. **Checkpoint Recovery**: Restore from last successful checkpoint
4. **Graceful Degradation**: Continue with reduced functionality when possible

### Error Monitoring

All errors are logged and monitored through the system:

```python
# Enable comprehensive error logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Get error statistics
orchestrator = get_orchestrator()
stats = orchestrator.get_system_stats()
error_stats = stats.get('errors', {})

print(f"Total errors: {error_stats.get('total', 0)}")
print(f"Recovery rate: {error_stats.get('recovery_rate', 0):.2%}")
```

---

## Rate Limiting and Quotas

### API Rate Limits

- **Generation Requests**: 100 requests per minute per backend
- **Asset Operations**: 1000 operations per minute
- **System Queries**: 10000 queries per minute

### Quota Management

```python
# Check current quota usage
asset_manager = get_asset_manager()
quota_status = asset_manager.get_quota_status()

print(f"Storage used: {quota_status['storage_used_gb']:.2f} GB")
print(f"Storage limit: {quota_status['storage_limit_gb']:.2f} GB")
print(f"Requests remaining: {quota_status['requests_remaining']}")
```

---

## Performance Considerations

### Optimization Tips

1. **Batch Operations**: Use batch APIs for better performance
2. **Async Processing**: All I/O operations are async - use await properly
3. **Caching**: Enable caching for repeated operations
4. **Memory Management**: Monitor memory usage with large assets
5. **Parallel Processing**: Configure concurrent job limits appropriately

### Performance Monitoring

```python
# Get performance metrics
orchestrator = get_orchestrator()
performance = orchestrator.get_performance_metrics()

print(f"Jobs per second: {performance['throughput']['jobs_per_second']:.2f}")
print(f"Average response time: {performance['latency']['average_ms']:.1f}ms")
print(f"Memory usage: {performance['memory']['current_mb']:.1f}MB")
```

---

## Security and Authentication

### API Authentication

All API calls require proper authentication:

```python
# Initialize with API key
director = APEXDirector(
    api_key="your-api-key",
    base_dir="/path/to/data"
)

# Validate authentication
is_authenticated = director.validate_credentials()
if not is_authenticated:
    raise AuthenticationError("Invalid API credentials")
```

### Permission Levels

- **Read**: Access to query and retrieve data
- **Write**: Ability to create and modify data
- **Execute**: Permission to run jobs and processes
- **Admin**: Full system access and configuration

### Data Protection

- All assets are encrypted at rest
- Sensitive metadata is encrypted
- Secure transmission using TLS 1.3
- Audit logging for all operations

---

## Changelog and Versioning

### API Versioning

APEX DIRECTOR uses semantic versioning for API compatibility:

- **Major Version** (X.y.z): Breaking changes
- **Minor Version** (x.Y.z): New features, backward compatible
- **Patch Version** (x.y.Z): Bug fixes, backward compatible

### Current API Version

Current stable API version: **1.0**

### Deprecation Policy

- Features are deprecated 6 months before removal
- Deprecated features show warnings
- Migration guides provided for major changes

### Backward Compatibility

The API maintains backward compatibility within major versions:
- Existing code continues to work
- New parameters are optional
- Default behavior is preserved

---

## Support and Resources

### Documentation
- [User Guide](user_guide.md) - Getting started and basic usage
- [Developer Guide](developer_guide.md) - Advanced development
- [Configuration Reference](config_reference.md) - Configuration options
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions

### Community
- GitHub Repository: https://github.com/apex-director/core
- Discussion Forum: https://community.apex-director.com
- Discord Server: https://discord.gg/apex-director

### Professional Support
- Enterprise Support: support@apex-director.com
- Priority Support: Available for enterprise customers
- Training Services: Custom training and consulting available

### API Status and Uptime
- Status Dashboard: https://status.apex-director.com
- API Uptime: 99.9% SLA
- Status Updates: @apexdirector on Twitter

---

*Last Updated: October 29, 2025*
*API Version: 1.0*
