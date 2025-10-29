#!/usr/bin/env python3
"""
Test Script for Advanced Audio Analysis Engine
Demonstrates the comprehensive audio analysis capabilities
"""

import sys
import os
from pathlib import Path

# Add apex_director/audio to path
sys.path.insert(0, str(Path(__file__).parent / "apex_director" / "audio"))

# Import individual modules directly
from analyzer import AudioAnalysisEngine, analyze_audio_file
from beat_detector import BeatDetector
from harmonic import HarmonicAnalyzer
from spectral import SpectralAnalyzer
from sections import SectionDetector
from quantizer import TimelineQuantizer

# System info function (simplified)
def get_system_info():
    """Get system information and available features."""
    import sys
    import numpy
    
    info = {
        'package_version': "1.0.0",
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
        import numpy
        info['available_libraries']['numpy'] = numpy.__version__
    except ImportError:
        info['available_libraries']['numpy'] = None
    
    info['available_libraries']['scipy'] = 'available'
    info['available_libraries']['sklearn'] = 'available'
    
    return info

def test_basic_analysis(audio_file: str):
    """Test basic audio analysis functionality."""
    print("=" * 60)
    print("TESTING BASIC AUDIO ANALYSIS")
    print("=" * 60)
    
    try:
        # Test convenience function
        print("1. Testing analyze_audio_file() convenience function...")
        results = analyze_audio_file(audio_file)
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return False
        
        print("✅ Basic analysis completed successfully")
        print(f"   File: {results['file_info']['path']}")
        print(f"   Duration: {results['file_info']['duration']:.2f}s")
        print(f"   Overall confidence: {results['overall_confidence']:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Basic analysis failed: {str(e)}")
        return False

def test_individual_components(audio_file: str):
    """Test individual analysis components."""
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 60)
    
    # Load audio data for individual component testing
    try:
        import librosa
        audio_data, sr = librosa.load(audio_file, sr=44100, mono=True)
        print(f"✅ Loaded audio: {len(audio_data)} samples ({len(audio_data)/sr:.2f}s)")
    except Exception as e:
        print(f"❌ Failed to load audio: {str(e)}")
        return False
    
    # Test beat detection
    print("\n1. Testing BeatDetector...")
    try:
        beat_detector = BeatDetector()
        beat_results = beat_detector.analyze(audio_data)
        
        if beat_results.get('error'):
            print(f"❌ Beat detection error: {beat_results['error']}")
        else:
            print(f"✅ BPM: {beat_results.get('bpm', 0):.1f}")
            print(f"   Beats detected: {len(beat_results.get('beat_times', []))}")
            print(f"   Confidence: {beat_results.get('confidence_scores', {}).get('beat_detection', 0):.2f}")
            
    except Exception as e:
        print(f"❌ Beat detection failed: {str(e)}")
    
    # Test harmonic analysis
    print("\n2. Testing HarmonicAnalyzer...")
    try:
        harmonic_analyzer = HarmonicAnalyzer()
        harmonic_results = harmonic_analyzer.analyze(audio_data, beat_results)
        
        if harmonic_results.get('error'):
            print(f"❌ Harmonic analysis error: {harmonic_results['error']}")
        else:
            key_info = harmonic_results.get('key_info', {})
            print(f"✅ Key: {key_info.get('key', 'Unknown')} {key_info.get('mode', 'Unknown')}")
            print(f"   Confidence: {key_info.get('confidence', 0):.2f}")
            print(f"   Chroma vector: {len(key_info.get('chroma_vector', []))} dimensions")
            
    except Exception as e:
        print(f"❌ Harmonic analysis failed: {str(e)}")
    
    # Test spectral analysis
    print("\n3. Testing SpectralAnalyzer...")
    try:
        spectral_analyzer = SpectralAnalyzer()
        spectral_results = spectral_analyzer.analyze(audio_data, beat_results)
        
        if spectral_results.get('error'):
            print(f"❌ Spectral analysis error: {spectral_results['error']}")
        else:
            spectral_info = spectral_results.get('spectral_info', {})
            print(f"✅ Brightness: {spectral_info.get('brightness', 0):.2f} Hz")
            print(f"   Energy: {spectral_info.get('energy', 0):.4f}")
            print(f"   Timbre complexity: {spectral_info.get('timbre_complexity', 0):.2f}")
            print(f"   Color features: {len(spectral_info.get('color_palette', []))}")
            
    except Exception as e:
        print(f"❌ Spectral analysis failed: {str(e)}")
    
    # Test section detection
    print("\n4. Testing SectionDetector...")
    try:
        section_detector = SectionDetector()
        section_results = section_detector.analyze(audio_data, beat_results, spectral_results)
        
        if section_results.get('error'):
            print(f"❌ Section detection error: {section_results['error']}")
        else:
            section_info = section_results.get('section_info', {})
            print(f"✅ Sections detected: {section_info.get('num_sections', 0)}")
            print(f"   Structure type: {section_info.get('structure_type', 'unknown')}")
            print(f"   Complexity score: {section_info.get('complexity_score', 0):.2f}")
            
    except Exception as e:
        print(f"❌ Section detection failed: {str(e)}")
    
    # Test timeline quantization
    print("\n5. Testing TimelineQuantizer...")
    try:
        quantizer = TimelineQuantizer(fps=24)
        quantize_results = quantizer.quantize(beat_results, spectral_results)
        
        if quantize_results.get('error'):
            print(f"❌ Timeline quantization error: {quantize_results['error']}")
        else:
            quant_info = quantize_results.get('quantization_info', {})
            print(f"✅ Target FPS: {quant_info.get('target_fps', 0)}")
            print(f"   Total frames: {quant_info.get('total_frames', 0)}")
            print(f"   Sync quality: {quant_info.get('sync_quality', 'unknown')}")
            print(f"   Quantization error: {quant_info.get('quantization_error', 0):.4f}s")
            
    except Exception as e:
        print(f"❌ Timeline quantization failed: {str(e)}")
    
    return True

def test_full_analysis_engine(audio_file: str):
    """Test the full analysis engine."""
    print("\n" + "=" * 60)
    print("TESTING FULL ANALYSIS ENGINE")
    print("=" * 60)
    
    try:
        # Create engine with custom parameters
        engine = AudioAnalysisEngine(sample_rate=44100, hop_length=512)
        
        print("Running comprehensive analysis...")
        results = engine.analyze_audio(audio_file)
        
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            return False
        
        print("✅ Full analysis completed successfully")
        
        # Display detailed summary
        summary = engine.get_analysis_summary(results)
        print("\n" + "=" * 40)
        print("ANALYSIS SUMMARY")
        print("=" * 40)
        print(summary)
        
        # Display confidence scores
        confidence_scores = results.get('confidence_scores', {})
        print("\n" + "=" * 40)
        print("CONFIDENCE SCORES")
        print("=" * 40)
        for component, score in confidence_scores.items():
            print(f"{component}: {score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full analysis engine test failed: {str(e)}")
        return False

def test_system_info():
    """Test system information and dependencies."""
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    try:
        info = get_system_info()
        
        print(f"Package Version: {info['package_version']}")
        print(f"Python Version: {info['python_version']}")
        
        print("\nDependency Status:")
        for lib, version in info['available_libraries'].items():
            if version:
                print(f"✅ {lib}: {version}")
            else:
                print(f"❌ {lib}: Not available")
        
        return True
        
    except Exception as e:
        print(f"❌ System info test failed: {str(e)}")
        return False

def create_sample_timeline():
    """Create and test sample timeline generation."""
    print("\n" + "=" * 60)
    print("TESTING TIMELINE EXPORT")
    print("=" * 60)
    
    try:
        # Create sample beat data
        sample_beat_results = {
            'bpm': 120.0,
            'beat_times': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            'confidence_scores': {'beat_detection': 0.9}
        }
        
        quantizer = TimelineQuantizer(fps=24)
        results = quantizer.quantize(sample_beat_results, {})
        
        frame_timings = results.get('frame_timings', [])
        print(f"✅ Generated {len(frame_timings)} frame timings")
        
        # Test JSON export
        json_export = quantizer.export_timeline(frame_timings[:10], 'json')
        if json_export:
            print(f"✅ JSON export: {len(json_export)} characters")
        
        # Test CSV export
        csv_export = quantizer.export_timeline(frame_timings[:10], 'csv')
        if csv_export:
            print(f"✅ CSV export: {len(csv_export)} characters")
        
        # Test SRT export
        srt_export = quantizer.export_timeline(frame_timings[:10], 'srt')
        if srt_export:
            print(f"✅ SRT export: {len(srt_export)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Timeline export test failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🎵 ADVANCED AUDIO ANALYSIS ENGINE TEST SUITE")
    print("=" * 60)
    
    # Test system information first
    test_system_info()
    
    # Create sample timeline
    create_sample_timeline()
    
    # Test with audio file if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return 1
        
        print(f"Testing with audio file: {audio_file}")
        
        # Run all tests
        test_basic_analysis(audio_file)
        test_individual_components(audio_file)
        test_full_analysis_engine(audio_file)
        
    else:
        print("ℹ️  No audio file provided. Running system tests only.")
        print("   Usage: python test_audio_analysis.py <audio_file>")
        print("\n   Example: python test_audio_analysis.py song.mp3")
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
