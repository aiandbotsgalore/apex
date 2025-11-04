"""
Advanced Audio Analysis Engine
Main orchestration module for comprehensive audio analysis
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .beat_detector import BeatDetector
from .harmonic import HarmonicAnalyzer
from .spectral import SpectralAnalyzer
from .sections import SectionDetector
from .quantizer import TimelineQuantizer


class AudioAnalysisEngine:
    """A comprehensive audio analysis engine.

    This class orchestrates multiple analysis modules to perform a complete
    analysis of an audio file, including beat detection, harmonic analysis,
    spectral feature extraction, and more.

    Attributes:
        sample_rate: The sample rate to use for audio processing.
        hop_length: The hop length for feature extraction.
        beat_detector: An instance of the BeatDetector.
        harmonic_analyzer: An instance of the HarmonicAnalyzer.
        spectral_analyzer: An instance of the SpectralAnalyzer.
        section_detector: An instance of the SectionDetector.
        quantizer: An instance of the TimelineQuantizer.
    """
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        """Initializes the AudioAnalysisEngine.

        Args:
            sample_rate: The sample rate to use for audio processing.
            hop_length: The hop length for feature extraction.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Initialize analysis modules
        self.beat_detector = BeatDetector(sample_rate, hop_length)
        self.harmonic_analyzer = HarmonicAnalyzer(sample_rate, hop_length)
        self.spectral_analyzer = SpectralAnalyzer(sample_rate, hop_length)
        self.section_detector = SectionDetector(sample_rate, hop_length)
        self.quantizer = TimelineQuantizer(fps=24)  # 24fps target
        
    def analyze_audio(self, audio_path: str, audio_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Performs a comprehensive analysis of an audio file.

        Args:
            audio_path: The path to the audio file.
            audio_data: Optional pre-loaded audio data.

        Returns:
            A dictionary containing the full analysis results.
        """
        try:
            # Validate input
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Determine audio format and load
            audio_ext = Path(audio_path).suffix.lower()
            supported_formats = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
            
            if audio_ext not in supported_formats:
                raise ValueError(f"Unsupported audio format: {audio_ext}")
            
            # Load audio data if not provided
            if audio_data is None:
                audio_data = self._load_audio(audio_path)
            
            if audio_data is None or len(audio_data) == 0:
                raise ValueError("Failed to load audio data or audio is empty")
            
            # Initialize results dictionary
            results = {
                'file_info': {
                    'path': audio_path,
                    'format': audio_ext,
                    'sample_rate': self.sample_rate,
                    'duration': len(audio_data) / self.sample_rate,
                    'channels': 1 if len(audio_data.shape) == 1 else audio_data.shape[0]
                },
                'confidence_scores': {},
                'analysis_timestamp': '2025-10-29T09:52:12'
            }
            
            # Perform beat detection
            print("Analyzing beat structure...")
            beat_results = self.beat_detector.analyze(audio_data)
            results.update(beat_results)
            
            # Perform harmonic analysis
            print("Analyzing harmonic content...")
            harmonic_results = self.harmonic_analyzer.analyze(audio_data, beat_results)
            results.update(harmonic_results)
            
            # Perform spectral analysis
            print("Extracting spectral features...")
            spectral_results = self.spectral_analyzer.analyze(audio_data, beat_results)
            results.update(spectral_results)
            
            # Detect song sections
            print("Detecting song structure...")
            section_results = self.section_detector.analyze(audio_data, beat_results, spectral_results)
            results.update(section_results)
            
            # Quantize timeline
            print("Quantizing timeline...")
            quantize_results = self.quantizer.quantize(beat_results, spectral_results)
            results.update(quantize_results)
            
            # Calculate overall confidence score
            overall_confidence = self._calculate_overall_confidence(results)
            results['overall_confidence'] = overall_confidence
            
            print(f"Analysis complete! Overall confidence: {overall_confidence:.2f}")
            return results
            
        except Exception as e:
            error_msg = f"Audio analysis failed: {str(e)}"
            print(error_msg)
            return {
                'error': error_msg,
                'file_info': {'path': audio_path},
                'analysis_timestamp': '2025-10-29T09:52:12'
            }
    
    def _load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Loads an audio file and returns its data as a NumPy array.

        This method includes error handling and a fallback mechanism.

        Args:
            audio_path: The path to the audio file.

        Returns:
            A NumPy array containing the audio data, or None if loading fails.
        """
        try:
            import librosa
            import soundfile as sf
            
            # Try loading with librosa (handles multiple formats)
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            if len(audio_data) == 0:
                raise ValueError("Loaded audio is empty")
            
            print(f"Loaded audio: {len(audio_data)} samples at {sr}Hz ({len(audio_data)/sr:.2f}s)")
            return audio_data
            
        except ImportError:
            # Fallback to soundfile only
            try:
                audio_data, sr = sf.read(audio_path)
                if sr != self.sample_rate:
                    # Resample if needed
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
                return audio_data if len(audio_data.shape) == 1 else np.mean(audio_data, axis=0)
            except Exception as e:
                print(f"Failed to load audio with fallback: {str(e)}")
                return None
        except Exception as e:
            print(f"Failed to load audio: {str(e)}")
            return None
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """Calculates an overall confidence score for the analysis.

        This score is a weighted average of the confidence scores from the
        individual analysis modules.

        Args:
            results: The analysis results dictionary.

        Returns:
            An overall confidence score between 0.0 and 1.0.
        """
        confidence_scores = results.get('confidence_scores', {})
        
        if not confidence_scores:
            return 0.0
        
        # Weight different analysis components
        weights = {
            'beat_detection': 0.25,
            'harmonic_analysis': 0.25,
            'spectral_analysis': 0.20,
            'section_detection': 0.20,
            'timeline_quantization': 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for component, weight in weights.items():
            if component in confidence_scores:
                score = confidence_scores[component]
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def get_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Generates a human-readable summary of the analysis results.

        Args:
            results: The analysis results from the `analyze_audio` method.

        Returns:
            A formatted string containing the summary.
        """
        if 'error' in results:
            return f"Analysis failed: {results['error']}"
        
        summary_lines = []
        
        # File info
        file_info = results.get('file_info', {})
        summary_lines.append(f"Audio File: {file_info.get('path', 'Unknown')}")
        summary_lines.append(f"Duration: {file_info.get('duration', 0):.2f} seconds")
        summary_lines.append(f"Sample Rate: {file_info.get('sample_rate', 0)} Hz")
        summary_lines.append("")
        
        # Beat analysis
        beat_info = results.get('beat_info', {})
        if beat_info:
            summary_lines.append(f"BPM: {beat_info.get('bpm', 'Unknown')}")
            confidence = beat_info.get('confidence', 0)
            summary_lines.append(f"Beat Detection Confidence: {confidence:.2f}")
            summary_lines.append("")
        
        # Key analysis
        key_info = results.get('key_info', {})
        if key_info:
            summary_lines.append(f"Key: {key_info.get('key', 'Unknown')}")
            summary_lines.append(f"Key Confidence: {key_info.get('confidence', 0):.2f}")
            summary_lines.append("")
        
        # Section analysis
        section_info = results.get('section_info', {})
        if section_info:
            sections = section_info.get('sections', [])
            summary_lines.append(f"Detected {len(sections)} sections:")
            for section in sections:
                start_time = section.get('start_time', 0)
                end_time = section.get('end_time', 0)
                label = section.get('label', 'Unknown')
                summary_lines.append(f"  {label}: {start_time:.2f}s - {end_time:.2f}s")
            summary_lines.append("")
        
        # Overall confidence
        overall_conf = results.get('overall_confidence', 0)
        summary_lines.append(f"Overall Analysis Confidence: {overall_conf:.2f}")
        
        return "\n".join(summary_lines)


def analyze_audio_file(audio_path: str, **kwargs) -> Dict[str, Any]:
    """A convenience function for analyzing a single audio file.

    This function creates an `AudioAnalysisEngine` instance and runs the
    analysis.

    Args:
        audio_path: The path to the audio file.
        **kwargs: Additional keyword arguments to be passed to the
            `AudioAnalysisEngine`.

    Returns:
        A dictionary containing the analysis results.
    """
    engine = AudioAnalysisEngine(**kwargs)
    return engine.analyze_audio(audio_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Analyzing: {audio_file}")
        
        engine = AudioAnalysisEngine()
        results = engine.analyze_audio(audio_file)
        
        print("\n" + "="*50)
        print(engine.get_analysis_summary(results))
        print("="*50)
    else:
        print("Usage: python analyzer.py <audio_file>")
