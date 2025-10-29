"""
Audio Analysis Package
Comprehensive audio analysis system for music visualization and synchronization
"""

from .analyzer import AudioAnalysisEngine, analyze_audio_file
from .beat_detector import BeatDetector, estimate_bpm
from .harmonic import HarmonicAnalyzer, detect_key
from .spectral import SpectralAnalyzer, extract_spectral_features
from .sections import SectionDetector, detect_song_structure
from .quantizer import TimelineQuantizer, quantize_timeline

__version__ = "1.0.0"
__author__ = "Audio Analysis Engine"

# Package metadata
__all__ = [
    'AudioAnalysisEngine',
    'analyze_audio_file',
    'BeatDetector',
    'estimate_bpm',
    'HarmonicAnalyzer',
    'detect_key',
    'SpectralAnalyzer',
    'extract_spectral_features',
    'SectionDetector',
    'detect_song_structure',
    'TimelineQuantizer',
    'quantize_timeline'
]

# Example usage information
EXAMPLE_USAGE = """
Example Usage:

1. Basic Analysis:
   from apex_director.audio import analyze_audio_file
   
   results = analyze_audio_file("song.mp3")
   print(results['key_info'])
   print(results['beat_info'])

2. Advanced Analysis:
   from apex_director.audio import AudioAnalysisEngine
   
   engine = AudioAnalysisEngine()
   results = engine.analyze_audio("song.mp3")
   
   # Get detailed analysis
   summary = engine.get_analysis_summary(results)

3. Component Usage:
   from apex_director.audio import BeatDetector, SpectralAnalyzer
   
   # Beat detection
   beat_detector = BeatDetector()
   beat_results = beat_detector.analyze(audio_data)
   
   # Spectral analysis
   spectral_analyzer = SpectralAnalyzer()
   spectral_results = spectral_analyzer.analyze(audio_data, beat_results)

4. Timeline Quantization:
   from apex_director.audio import TimelineQuantizer
   
   quantizer = TimelineQuantizer(fps=24)  # 24fps for video sync
   quantized = quantizer.quantize(beat_results, spectral_results)
   
   # Export to various formats
   timeline_json = quantizer.export_timeline(quantized['frame_timings'], 'json')
   timeline_csv = quantizer.export_timeline(quantized['frame_timings'], 'csv')
"""

def get_system_info():
    """Get system information and available features."""
    import sys
    
    info = {
        'package_version': __version__,
        'python_version': sys.version,
        'available_libraries': {}
    }
    
    # Check available libraries
    try:
        import librosa
        info['available_libraries']['librosa'] = librosa.__version__
    except ImportError:
        info['available_libraries']['librosa'] = None
    
    try:
        import essentia
        info['available_libraries']['essentia'] = essentia.__version__
    except ImportError:
        info['available_libraries']['essentia'] = None
    
    try:
        import numpy
        info['available_libraries']['numpy'] = numpy.__version__
    except ImportError:
        info['available_libraries']['numpy'] = None
    
    try:
        import scipy
        info['available_libraries']['scipy'] = scipy.__version__
    except ImportError:
        info['available_libraries']['scipy'] = None
    
    try:
        import sklearn
        info['available_libraries']['sklearn'] = sklearn.__version__
    except ImportError:
        info['available_libraries']['sklearn'] = None
    
    return info
