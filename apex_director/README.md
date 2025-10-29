# Apex Director - Professional Media Production System

A comprehensive professional-grade media production system featuring both cinematic image generation and broadcast-quality video assembly. Designed for visual storytelling, film production, and broadcast applications.

## 🎬 System Overview

This system provides two core capabilities:

1. **Cinematic Image Generation Pipeline** - Multi-backend image generation with style consistency
2. **Professional Video Assembly System** - Broadcast-quality video editing with frame-accurate timing

### Core Features

#### Image Generation
- **Multi-Backend Generation**: Unified interface for multiple generation backends
- **Advanced Prompt Engineering**: Cinematography-focused prompts with camera/lighting settings
- **Style Persistence**: JSON-based style consistency management across 40+ shots
- **Character Identity**: Face consistency using FaceID/IP-Adapter technology
- **Quality Scoring**: 4-criteria scoring system (CLIP aesthetic + composition + style + artifacts)
- **Professional Upscaling**: Real-ESRGAN 4x broadcast quality upscaling

#### Video Assembly
- **Beat-locked Cutting**: Frame-accurate cuts with ±1 frame precision
- **4-Stage Color Grading**: Professional color correction pipeline (Rec.709/Rec.2020)
- **Professional Transitions**: Cut, crossfade, whip pan, match dissolve effects
- **Motion Effects**: Ken Burns, parallax, dolly zoom, camera movements
- **Broadcast-Quality Export**: Exact FFmpeg specs for 1080p/4K with multiple codecs

## 🎬 Features

### Multi-Backend Generation
- **Unified Interface**: Single API for multiple generation backends
- **Supported Backends**: Nano Banana, Google Imagen, MiniMax, Stable Diffusion XL
- **Backend Selection**: Automatic or manual backend selection based on requirements
- **Fallback Support**: Graceful degradation when backends are unavailable

### Advanced Prompt Engineering
- **Cinematography-Focused**: Professional film and TV production prompts
- **Camera Settings**: Lens types, aperture, ISO, shutter speed
- **Lighting Setup**: Three-point lighting, mood-based lighting
- **Composition Rules**: Rule of thirds, leading lines, symmetry
- **Director Styles**: Wes Anderson, Christopher Nolan, Guillermo del Toro, David Fincher

### Style Persistence Engine
- **Style Bible**: JSON-based style consistency management
- **CLIP Embeddings**: Style drift detection and prevention
- **Scene Consistency**: Maintain visual consistency across 40+ shots
- **Color Palette Management**: Automatic color consistency tracking

### Character Identity System
- **FaceID/IP-Adapter**: Character consistency across shots
- **Character Profiles**: Comprehensive character database
- **Facial Feature Analysis**: Automatic face encoding and matching
- **Consistency Validation**: Verify character consistency in generated images

### Multi-Variant Selection
- **4-Criteria Scoring**: CLIP aesthetic + composition + style + artifacts
- **Objective Metrics**: Quantitative quality assessment
- **CLIP Aesthetic Scoring**: Semantic and aesthetic quality evaluation
- **Composition Analysis**: Rule of thirds, leading lines, symmetry
- **Artifact Detection**: Noise, compression, banding, blurring detection

### Professional Upscaling
- **Real-ESRGAN 4x**: Broadcast quality upscaling
- **Tile-Based Processing**: Handle large images efficiently
- **Quality Metrics**: Sharpness, detail preservation, edge quality
- **Multiple Presets**: Web optimized, high quality, broadcast quality

## 📁 Project Structure

```
apex_director/
├── images/                     # Image Generation Pipeline
│   ├── __init__.py
│   ├── generator.py            # Main image generation engine
│   ├── backend_interface.py    # Multi-backend abstraction
│   ├── prompt_engineer.py      # Advanced cinematography prompts
│   ├── style_persistence.py    # Style consistency system
│   ├── character_system.py     # Face consistency
│   ├── variant_selector.py     # Quality scoring and selection
│   └── upscaller.py           # Professional upscaling
├── video/                      # Professional Video Assembly
│   ├── __init__.py
│   ├── timeline.py            # Timeline construction and cutting
│   ├── transitions.py         # Professional transition effects
│   ├── color_grader.py        # 4-stage color grading pipeline
│   ├── motion.py              # Camera movements and motion effects
│   ├── exporter.py            # Broadcast-quality export engine
│   ├── assembler.py           # Main assembly controller
│   └── assembler_core.py      # Core assembly components
├── style_bible.json           # Style consistency configuration
├── examples.py                # Usage examples
└── README.md                  # This file
```

## 🎥 Professional Video Assembly System

### Timeline Construction
- **Beat-locked cutting** with ±1 frame accuracy using librosa audio analysis
- **Frame-perfect synchronization** for professional broadcast standards
- **Professional markers** for beats, cuts, effects, and color corrections
- **Edit Decision List (EDL)** generation for external editing systems
- **JSON timeline persistence** with complete metadata preservation

### Professional Transitions
- **Cut, Crossfade, Whip Pan, Match Dissolve** with broadcast-precision timing
- **Transition preview** system with real-time parameter adjustment
- **Multiple fade curves** (linear, ease-in/out, sine, gamma)
- **FFmpeg command generation** for direct professional encoding
- **Broadcast-standard validation** with comprehensive error checking

### 4-Stage Color Grading Pipeline
- **Stage 1**: Primary correction (exposure, white balance, contrast, gamma)
- **Stage 2**: Secondary correction (skin tone isolation, selective desaturation)
- **Stage 3**: Creative grade (LUT application, cinematic curves, teal & orange)
- **Stage 4**: Finishing (film grain, sharpening, vignette, chromatic aberration)
- **Professional color spaces** (Rec.709, Rec.2020, DCI-P3)
- **Automatic color balance** and histogram analysis

### Motion Effects & Camera Work
- **Ken Burns effect** (pan, zoom, combined) with smooth keyframe interpolation
- **Parallax effects** for pseudo-3D depth using layered analysis
- **Professional camera movements** (dolly zoom, trucking, booming, panning, tilting)
- **Advanced keyframe system** with multiple interpolation curves
- **Motion blur** and depth of field simulation

### Broadcast-Quality Export
- **Exact FFmpeg specifications** for 1080p/4K with broadcast compliance
- **Professional codecs** (H.264, H.265, ProRes, DNxHD, RAV1E)
- **Multi-pass encoding** with VBV buffer control and lookahead
- **Quality analysis** using ffprobe with broadcast standard validation
- **Multiple format support** (MP4, MOV, AVI, MKV) with metadata embedding

### Video Assembly Quick Start

```python
from apex_director.video import Timeline, VideoAssembler, AssemblyJob, AssemblyMode, QualityMode

# Create professional timeline
timeline = Timeline(frame_rate=29.97, resolution=(1920, 1080))

# Add clips with frame-accurate timing
clip = Clip(
    id="intro",
    source_path="input/intro.mp4",
    in_frame=0,
    out_frame=int(5 * 29.97),  # 5 seconds at 29.97fps
    in_time=0.0,
    out_time=5.0,
    duration=5.0,
    frame_rate=29.97,
    width=1920,
    height=1080
)
timeline.add_clip(clip)

# Add beat-locked markers for professional cuts
timeline.add_beat_markers("audio/track.mp3", BeatType.BEAT)

# Create professional assembly job
job = AssemblyJob(
    job_id="broadcast_video",
    timeline=timeline,
    output_path="output/broadcast.mp4",
    assembly_mode=AssemblyMode.OFFLINE,
    quality_mode=QualityMode.BROADCAST,
    validate_broadcast_standards=True,
    quality_analysis=True
)

# Assemble with frame-perfect precision
assembler = VideoAssembler()
result = assembler.assemble_video(job)

if result.success:
    print(f"Broadcast-quality video exported: {result.output_path}")
    print(f"Duration: {result.duration}s | Processing: {result.processing_time:.2f}s")
    print(f"Frame count: {result.frame_count}")
```

### Professional Color Grading

```python
from apex_director.video import ColorGrader, ColorCorrection, LUT

grader = ColorGrader(timeline)

# Stage 1: Primary correction
grader.primary_correction = ColorCorrection(
    exposure=0.2,
    contrast=15.0,
    brightness=5.0,
    saturation=8.0,
    temperature=5600.0,
    tint=0.0
)

# Stage 2: Secondary correction with skin tone isolation
grader.skin_tone_mask = SkinToneMask(
    enabled=True,
    y_min=0.1, y_max=0.9,
    cb_min=0.35, cb_max=0.65,
    cr_min=0.45, cr_max=0.65,
    softness=0.7
)

# Stage 3: Creative cinematic grade
grader.creative_lut = LUT(
    name="Cinematic Teal & Orange",
    type="3d",
    file_path="luts/cinematic.cube"
)

# Stage 4: Professional finishing
grader.finishing_effects["film_grain"].enabled = True
grader.finishing_effects["film_grain"].intensity = 0.1
grader.finishing_effects["vignette"].enabled = True
grader.finishing_effects["vignette"].amount = 0.3
grader.finishing_effects["sharpening"] = 0.4
```

### Motion Effects

```python
from apex_director.video import MotionEngine, MotionType, CameraMovement, Keyframe

motion_engine = MotionEngine(timeline)

# Ken Burns combined pan and zoom
ken_burns = CameraMovement(
    motion_type=MotionType.KEN_BURNS_COMBINED,
    start_time=0.0,
    duration=4.0
)
ken_burns.keyframes = [
    Keyframe(time=0.0, position_x=0.3, position_y=0.3, zoom=1.2, interpolation="ease_out"),
    Keyframe(time=2.0, position_x=0.7, position_y=0.7, zoom=1.0, interpolation="linear"),
    Keyframe(time=4.0, position_x=0.8, position_y=0.8, zoom=1.1, interpolation="ease_in")
]

# Dolly zoom effect for dramatic impact
dolly_zoom = CameraMovement(
    motion_type=MotionType.DOLLY_ZOOM,
    start_time=5.0,
    duration=3.0
)
dolly_zoom.parameters = {
    "zoom_factor": 1.5,
    "speed": 1.0
}

motion_engine.add_camera_movement(ken_burns)
motion_engine.add_camera_movement(dolly_zoom)
```

### Broadcast-Standard Export

```python
from apex_director.video import BroadcastExporter, ExportSettings, VideoCodec, VideoResolution

exporter = BroadcastExporter(timeline)

# Broadcast H.264 settings
settings = ExportSettings()
settings.video_codec = VideoCodec.H264
settings.resolution = VideoResolution.HD_1080P
settings.frame_rate = 29.97
settings.crf = 18  # High quality
settings.preset = "slow"
settings.two_pass = True
settings.color_space = ColorSpace.REC_709
settings.gop_size = 30
settings.b_frames = 3

# Export with professional encoding
result = exporter.export_video("broadcast_output.mp4", settings)

# Cinema 4K H.265 settings
cinema_settings = ExportSettings()
cinema_settings.video_codec = VideoCodec.H265
cinema_settings.resolution = VideoResolution.UHD_4K
cinema_settings.frame_rate = 24.0
cinema_settings.crf = 16  # Very high quality
cinema_settings.preset = "slow"
cinema_settings.bit_depth = 10
cinema_settings.color_space = ColorSpace.REC_2020

# Professional formats
exporter.export_pro_res("broadcast_output.mov", profile="422")
exporter.export_dnxhd("avid_output.mov", profile="220")
```

## 🚀 Quick Start

### Basic Usage

```python
from apex_director.images import quick_generate

# Quick generation with defaults
result = await quick_generate(
    prompt="A cinematic portrait of a detective in a noir setting",
    output_dir="output/detective_portrait"
)

print(f"Generated {len(result.selected_variants)} variants")
print(f"Quality score: {result.overall_quality_score:.3f}")
```

### Advanced Usage

```python
from apex_director.images import (
    CinematicImageGenerator,
    GenerationRequest,
    CameraSettings,
    LightingSetup,
    Composition
)

# Create generator
generator = CinematicImageGenerator()

# Create detailed request
request = GenerationRequest(
    prompt="A space marine in a futuristic battlefield",
    scene_id="battle_scene_01",
    character_name="commander_sarah",
    genre="sci_fi",
    director_style="christopher_nolan",
    camera_settings=CameraSettings(
        lens="35mm",
        aperture="f/2.8",
        iso="ISO 800"
    ),
    lighting_setup=LightingSetup(
        key_light="volumetric_lighting",
        fill_light="blue_accents",
        rim_light="practical_glow",
        mood="dramatic"
    ),
    composition=Composition(
        rule_of_thirds=True,
        leading_lines=True,
        depth_of_field="shallow"
    ),
    upscale=True,
    upscale_preset="broadcast_quality"
)

# Generate image
result = await generator.generate_single_image(request)
```

### Batch Generation

```python
from apex_director.images import batch_generate

# Generate multiple images
prompts = [
    "A steampunk inventor in his workshop",
    "A fantasy wizard casting a spell", 
    "A cyberpunk hacker in neon-lit alley"
]

results = await batch_generate(prompts, "batch_output")
```

## ⚙️ Configuration

### Style Bible (style_bible.json)

```json
{
  "project_name": "My Cinematic Project",
  "overall_style": {
    "visual_style": "cinematic realism with dramatic flair",
    "color_grading": "natural with subtle warmth",
    "lighting_style": "three-point lighting with practical accents"
  },
  "color_palette": {
    "primary_colors": ["#2C3E50", "#ECF0F1", "#E74C3C"],
    "skin_tones": ["#FDBCB4", "#F1C27D", "#E0AC69"]
  },
  "camera_profile": {
    "preferred_lenses": ["35mm", "50mm", "85mm"],
    "aperture_range": "f/2.8 to f/5.6"
  }
}
```

### Character Profiles

```python
from apex_director.images import CharacterConsistencyManager

# Create character manager
character_manager = CharacterConsistencyManager()

# Create character profile
character_manager.create_character_profile(
    name="commander_sarah",
    reference_images=["sarah_ref1.jpg", "sarah_ref2.jpg"],
    description="Female military commander, short brown hair, blue eyes",
    style_attributes={
        "clothing": "tactical gear, military uniform",
        "age": "late 30s",
        "build": "athletic, commanding presence"
    }
)
```

## 📊 Quality Scoring System

### 4-Criteria Scoring

The system evaluates each variant using four key criteria:

1. **CLIP Aesthetic Score** (35% weight)
   - Semantic consistency with prompt
   - Visual aesthetic quality
   - Color balance and composition

2. **Composition Score** (25% weight)
   - Rule of thirds compliance
   - Leading lines detection
   - Symmetry and balance
   - Depth and layering

3. **Style Consistency Score** (25% weight)
   - Color palette matching
   - Lighting style consistency
   - Texture and mood alignment

4. **Artifact Quality Score** (15% weight)
   - Noise detection
   - Compression artifacts
   - Color banding
   - Blur/sharpness quality

### Scoring Example

```python
# Get detailed scoring explanation
for variant in selected_variants:
    explanation = selector.get_scoring_explanation(variant)
    print(explanation)

# Output:
# Variant: variant_1
# Overall Score: 0.892
# 
# Individual Scores:
#   CLIP Aesthetic: 0.923
#   Composition: 0.845
#   Style Consistency: 0.867
#   Artifact Quality: 0.934
# 
# Selection Reason:
#   exceptional aesthetic quality; outstanding composition; 
#   good style consistency; clean technical quality; 
#   exceptional overall quality
```

## 🎭 Character Consistency

### Face Recognition and Matching

```python
# Validate character consistency
is_consistent, confidence, issues = character_manager.validate_character_consistency(
    generated_image,
    expected_character="commander_sarah",
    confidence_threshold=0.8
)

if not is_consistent:
    print(f"Character consistency issues: {issues}")
else:
    print(f"Character consistency confirmed: {confidence:.2%}")
```

### Style Drift Detection

```python
# Detect style drift
drift_score, drift_issues = style_manager.detect_style_drift(generated_image)

if drift_score > 0.2:  # 20% drift threshold
    print(f"Style drift detected: {drift_issues}")
    # Regenerate with adjusted prompts
```

## 🎬 Supported Genres and Styles

### Film Genres
- **Film Noir**: High contrast, dramatic shadows, chiaroscuro lighting
- **Sci-Fi**: Neon lights, volumetric lighting, futuristic color palettes
- **Horror**: Low-key lighting, harsh shadows, ominous atmosphere
- **Thriller**: High contrast, directional lighting, tense compositions
- **Romance**: Soft, warm lighting, golden hour aesthetics
- **Action**: Dynamic lighting, vibrant colors, wide action shots

### Director Styles
- **Wes Anderson**: Symmetrical composition, vibrant pastels, whimsical details
- **Christopher Nolan**: IMAX compositions, natural lighting, technical precision
- **Guillermo del Toro**: Gothic elements, rich colors, practical lighting
- **David Fincher**: Precise framing, cool tones, meticulously crafted

### Camera Settings
```python
CameraSettings(
    lens="50mm",           # 24mm, 35mm, 50mm, 85mm, 135mm
    aperture="f/2.8",      # Affects depth of field
    iso="ISO 800",         # Affects noise and grain
    shutter_speed="1/125"  # Affects motion blur
)
```

### Lighting Setups
```python
LightingSetup(
    key_light="soft_box",       # Main light source
    fill_light="reflector",     # Shadow fill
    rim_light="hair_light",     # Edge lighting
    background="neutral",       # Background treatment
    mood="cinematic"            # Overall mood
)
```

## 📈 Performance and Quality

### Processing Times (Estimated)
- **Generation**: 30-60 seconds per variant
- **Quality Scoring**: 5-10 seconds per image
- **Upscaling**: 10-30 seconds (depends on image size)
- **Style Consistency Check**: 2-5 seconds

### Quality Metrics
- **Resolution**: Up to 4K upscaling with Real-ESRGAN
- **Consistency**: CLIP-based style drift detection
- **Character Matching**: Face recognition with 80%+ confidence threshold
- **Artifact Detection**: Comprehensive technical quality assessment

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ for tile-based processing
- **Storage**: 10GB+ for models and temporary files
- **Python**: 3.8+ with CUDA support

## 🔧 Advanced Configuration

### Custom Upscaling Presets

```python
from apex_director.images import UpscaleSettings

# Create custom preset
custom_preset = UpscaleSettings(
    scale_factor=4,
    model_name="RealESRGAN_x4plus",
    face_enhance=True,
    denoise_strength=0.3,
    tile_size=512
)

# Use in generation
result = await generator.generate_single_image(request)
```

### Backend-Specific Optimizations

```python
# Prompt optimization for specific backends
optimized_prompt = prompt_engineer.optimize_prompt_for_backend(
    base_prompt, 
    backend_name="sdxl"
)
```

### Style Consistency Rules

```python
# Define scene-specific style
scene_style = SceneStyle(
    scene_id="battle_scene_01",
    style_elements=[
        StyleElement("lighting", "dramatic_volumetric", weight=1.0),
        StyleElement("colors", "battle_hues", weight=0.8)
    ],
    lighting_setup={"mood": "intense", "shadows": "strong"},
    color_palette=["#8B0000", "#2F4F4F", "#FFD700"]
)

style_manager.add_scene_style(scene_style)
```

## 🛠️ Troubleshooting

### Common Issues

1. **No variants generated**
   - Check backend API keys and connectivity
   - Verify prompt format and length
   - Check available disk space

2. **Poor quality scores**
   - Adjust prompt with more specific details
   - Try different backends
   - Increase generation steps

3. **Character inconsistency**
   - Verify reference images are clear
   - Check face encoding quality
   - Adjust confidence threshold

4. **Style drift**
   - Update style bible with current generation
   - Check consistency scoring parameters
   - Regenerate with style constraints

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
generator = CinematicImageGenerator()
```

## 📚 API Reference

### Main Classes

- `CinematicImageGenerator`: Main generation engine
- `GenerationRequest`: Complete generation configuration
- `GenerationResult`: Generated image results and metadata
- `BackendManager`: Multi-backend management
- `VariantSelector`: Quality scoring and selection
- `ProfessionalUpscaler`: Real-ESRGAN upscaling

### Key Methods

- `generate_single_image(request)`: Generate single image
- `generate_image_sequence(requests)`: Generate sequence with consistency
- `select_best_variants(variants, prompt)`: Score and select variants
- `validate_character_consistency(image, character)`: Check character consistency
- `upscale_image(image, preset)`: Professional upscaling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request with detailed description

## 🆘 Support

For support and questions:
- Check examples in `examples.py`
- Review documentation in source files
- Open GitHub issue for bugs or feature requests

---

**Built for professional cinematic image generation and visual storytelling.**