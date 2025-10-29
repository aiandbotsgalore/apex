# Advanced Audio Analysis Engine

A comprehensive audio analysis system built for professional music visualization and synchronization applications. This engine provides beat detection, harmonic analysis, spectral features, section detection, emotional valence analysis, timeline quantization, and LUFS metering.

## 🎵 Features

### 1. **Beat Detection** 
- **Library**: librosa for precise BPM detection
- **Features**:
  - Multiple onset detection algorithms
  - BPM estimation with confidence scoring
  - Beat position grid extraction
  - Tempo curve analysis
  - Downbeat detection
  - Beat regularity analysis

### 2. **Harmonic Analysis**
- **Library**: essentia (with librosa fallback)
- **Features**:
  - Key detection (major/minor with confidence)
  - Chroma feature extraction
  - Chord progression tracking
  - Harmonic rhythm analysis
  - Pitch class distribution analysis

### 3. **Spectral Features**
- **Features**:
  - Spectral centroid (brightness mapping)
  - Spectral rolloff and bandwidth
  - Zero crossing rate
  - MFCCs for timbre analysis
  - Spectral contrast analysis
  - RMS energy calculation
  - Spectral flux analysis
  - Color mapping features for visualization

### 4. **Section Detection**
- **Features**:
  - Automatic song structure identification
  - Verse/chorus/bridge/intro/outro detection
  - Beat-synchronous feature extraction
  - Self-similarity matrix computation
  - Section boundary detection
  - Structure complexity analysis

### 5. **Timeline Quantization Engine**
- **Features**:
  - Beat grid to frame conversion (24fps standard)
  - Phase calculation for precise timing
  - Section-aware quantization
  - Multiple export formats (JSON, CSV, SRT)
  - Error correction and smoothing

### 6. **Additional Features**
- **Emotional Valence Analysis**: Sentiment detection for mood mapping
- **LUFS Metering**: Loudness analysis for dynamic range mapping
- **Multi-format Support**: MP3, WAV, FLAC, M4A, OGG
- **Robust Error Handling**: Confidence scoring and fallback mechanisms
- **Professional Error Logging**: Detailed analysis and debugging information

## 📁 File Structure

```
apex_director/audio/
├── __init__.py              # Package initialization
├── analyzer.py              # Main audio analysis engine
├── beat_detector.py         # BPM and beat grid detection
├── harmonic.py              # Key/chord analysis
├── spectral.py              # Frequency domain features
├── sections.py              # Song structure detection
├── quantizer.py             # Beat grid to frame conversion
└── README.md                # This documentation
```

## 🚀 Quick Start

### Basic Usage

```python
from apex_director.audio import analyze_audio_file

# Analyze a single audio file
results = analyze_audio_file("song.mp3")

# Access results
print(f"BPM: {results['beat_info']['bpm']}")
print(f"Key: {results['key_info']['key']} {results['key_info']['mode']}")
print(f"Sections: {len(results['section_info']['sections'])}")
print(f"Overall confidence: {results['overall_confidence']:.2f}")
```

### Advanced Usage

```python
from apex_director.audio import AudioAnalysisEngine

# Create analysis engine with custom parameters
engine = AudioAnalysisEngine(sample_rate=44100, hop_length=512)

# Perform comprehensive analysis
results = engine.analyze_audio("song.mp3")

# Generate human-readable summary
summary = engine.get_analysis_summary(results)
print(summary)

# Export timeline for video synchronization
timeline = results.get('quantization_info', {})
if 'frame_timings' in results:
    from apex_director.audio import TimelineQuantizer
    quantizer = TimelineQuantizer(fps=24)
    json_timeline = quantizer.export_timeline(results['frame_timings'], 'json')
    srt_timeline = quantizer.export_timeline(results['frame_timings'], 'srt')
```

### Component-Specific Usage

```python
from apex_director.audio import BeatDetector, SpectralAnalyzer, SectionDetector

# Load audio
import librosa
audio_data, sr = librosa.load("song.mp3", sr=44100)

# Beat detection
beat_detector = BeatDetector()
beat_results = beat_detector.analyze(audio_data)
print(f"BPM: {beat_results['bpm']}")
print(f"Number of beats: {len(beat_results['beat_times'])}")

# Spectral analysis
spectral_analyzer = SpectralAnalyzer()
spectral_results = spectral_analyzer.analyze(audio_data, beat_results)
print(f"Brightness: {spectral_results['spectral_info']['brightness']:.2f} Hz")

# Section detection
section_detector = SectionDetector()
section_results = section_detector.analyze(audio_data, beat_results, spectral_results)
sections = section_results['section_info']['sections']
for section in sections:
    print(f"{section['label']}: {section['start_time']:.1f}s - {section['end_time']:.1f}s")
```

## 📊 Output Structure

The analysis engine returns a comprehensive dictionary with the following structure:

```python
{
    'file_info': {
        'path': 'song.mp3',
        'format': '.mp3',
        'sample_rate': 44100,
        'duration': 180.5,
        'channels': 1
    },
    'beat_info': {
        'bpm': 120.5,
        'confidence': 0.95,
        'num_beats': 361,
        'duration': 180.5
    },
    'key_info': {
        'key': 'C',
        'mode': 'major',
        'confidence': 0.87,
        'chroma_vector': [0.1, 0.05, ...],
        'harmonic_rhythm': [...]
    },
    'spectral_info': {
        'brightness': 2150.3,
        'energy': 0.023,
        'timbre_complexity': 850.2,
        'color_palette': [...]
    },
    'section_info': {
        'num_sections': 8,
        'structure_type': 'verse-chorus',
        'complexity_score': 0.72,
        'sections': [
            {
                'section_id': 0,
                'start_time': 0.0,
                'end_time': 32.5,
                'duration': 32.5,
                'label': 'intro',
                'confidence': 0.91
            },
            ...
        ]
    },
    'quantization_info': {
        'target_fps': 24.0,
        'total_frames': 4332,
        'quantization_error': 0.012,
        'beat_alignment_rate': 0.96,
        'sync_quality': 'excellent'
    },
    'confidence_scores': {
        'beat_detection': 0.95,
        'harmonic_analysis': 0.87,
        'spectral_analysis': 0.92,
        'section_detection': 0.78,
        'timeline_quantization': 0.94
    },
    'overall_confidence': 0.89
}
```

## 🎛️ Configuration Options

### AudioAnalysisEngine
```python
engine = AudioAnalysisEngine(
    sample_rate=44100,    # Audio sample rate
    hop_length=512        # Analysis hop length
)
```

### BeatDetector
```python
beat_detector = BeatDetector(
    sample_rate=44100,
    hop_length=512
)
```

### TimelineQuantizer
```python
quantizer = TimelineQuantizer(
    fps=24.0,             # Target frame rate for video sync
    beat_tolerance=0.05   # Beat alignment tolerance (seconds)
)
```

## 📈 Confidence Scoring

Each analysis component provides confidence scores (0.0 to 1.0):

- **Beat Detection**: Based on beat regularity, periodicity, and signal quality
- **Harmonic Analysis**: Based on key strength and chroma coherence
- **Spectral Analysis**: Based on signal energy and feature diversity
- **Section Detection**: Based on similarity patterns and structural clarity
- **Timeline Quantization**: Based on beat alignment and frame coverage

## 🎨 Color Mapping

The spectral analyzer provides features for color visualization:

- **Brightness**: Maps to color temperature (warm/cool)
- **Energy**: Maps to saturation levels
- **Complexity**: Maps to colorfulness
- **Contrast**: Maps to brightness variation

## 🔧 Dependencies

### Required Libraries
- `numpy` - Numerical computing
- `librosa` - Audio analysis (primary)
- `scipy` - Scientific computing
- `sklearn` - Machine learning

### Optional Libraries
- `essentia` - Advanced audio analysis (fallback to librosa)
- `soundfile` - Audio file I/O (fallback)

## 🧪 Testing

Each module can be tested independently:

```bash
# Test beat detection
python -m apex_director.audio.beat_detector song.mp3

# Test harmonic analysis
python -m apex_director.audio.harmonic song.mp3

# Test spectral analysis
python -m apex_director.audio.spectral song.mp3

# Test section detection
python -m apex_director.audio.sections song.mp3

# Test timeline quantization
python -m apex_director.audio.quantizer 24

# Test full analysis
python -m apex_director.audio.analyzer song.mp3
```

## 🎯 Use Cases

### 1. Music Video Synchronization
```python
# Generate frame-accurate timeline for video editing
timeline = quantizer.quantize(beat_results, spectral_results)
json_export = quantizer.export_timeline(timeline['frame_timings'], 'json')
```

### 2. Music Visualization
```python
# Extract features for real-time visualization
features = extract_spectral_features(audio_data)
color_features = features['color_palette']
```

### 3. Music Analysis Tools
```python
# Comprehensive song analysis
results = analyze_audio_file("song.mp3")
analysis_summary = engine.get_analysis_summary(results)
```

### 4. DJ Software Integration
```python
# Beat grid and key detection for DJ applications
key_info = detect_key(audio_data)
bpm = estimate_bpm(audio_data)
```

## 📝 Error Handling

The system provides robust error handling with:

- **Format Validation**: Automatic detection of supported audio formats
- **Fallback Mechanisms**: Librosa fallback when essentia is unavailable
- **Confidence Scoring**: Quality assessment of analysis results
- **Graceful Degradation**: Partial results when components fail
- **Detailed Logging**: Comprehensive error messages for debugging

## 🔮 Future Enhancements

- Real-time audio analysis support
- Advanced chord recognition algorithms
- Tempo change detection
- Multi-channel audio support
- GPU acceleration for large files
- Streaming analysis capabilities
- Advanced emotional valence models

## 📄 License

This project is part of the APEX DIRECTOR suite - Advanced Audio Analysis Engine for professional music visualization and synchronization applications.

## 🤝 Contributing

Contributions are welcome! Please ensure all modules maintain the existing error handling and confidence scoring patterns.

---

*Built with ❤️ for the music technology community*
