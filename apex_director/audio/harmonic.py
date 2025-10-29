"""
Harmonic Analysis Module
Key detection and chord progression analysis using essentia
"""

import numpy as np
import warnings
from typing import Dict, Any, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Try to import essentia, fallback to librosa if not available
try:
    import essentia
    import essentia.standard as estd
    ESSENTIA_AVAILABLE = True
except ImportError:
    print("Essentia not available, using librosa fallback")
    ESSENTIA_AVAILABLE = False

import librosa


class HarmonicAnalyzer:
    """
    Harmonic analysis using essentia (preferred) or librosa (fallback).
    
    Features:
    - Key detection (major/minor with confidence)
    - Chord progression tracking
    - Harmonic rhythm analysis
    - Pitch class distribution
    - Chord complexity metrics
    """
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        """
        Initialize harmonic analyzer.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Analysis hop length
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.use_essentia = ESSENTIA_AVAILABLE
        
        # Define musical note to pitch class mapping
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.chroma_features = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
    def analyze(self, audio_data: np.ndarray, beat_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive harmonic analysis.
        
        Args:
            audio_data: Audio signal array
            beat_results: Results from beat detection
            
        Returns:
            Dictionary containing harmonic analysis results
        """
        try:
            # Basic validation
            if len(audio_data) == 0:
                raise ValueError("Audio data is empty")
            
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            print(f"Harmonic analysis on {len(audio_data)} samples ({len(audio_data)/self.sample_rate:.2f}s)")
            
            results = {}
            
            if self.use_essentia:
                results.update(self._essentia_analysis(audio_data, beat_results))
            else:
                results.update(self._librosa_analysis(audio_data, beat_results))
            
            # Add confidence score
            confidence = self._calculate_harmonic_confidence(results)
            results['confidence_scores']['harmonic_analysis'] = confidence
            
            # Add summary information
            results['key_info'] = {
                'key': results.get('detected_key', 'Unknown'),
                'mode': results.get('key_mode', 'Unknown'),
                'confidence': confidence,
                'chroma_vector': results.get('chroma_vector', [0] * 12),
                'harmonic_rhythm': results.get('harmonic_rhythm', [])
            }
            
            return results
            
        except Exception as e:
            print(f"Harmonic analysis error: {str(e)}")
            return {
                'error': str(e),
                'confidence_scores': {'harmonic_analysis': 0.0},
                'key_info': {
                    'key': 'Unknown',
                    'mode': 'Unknown',
                    'confidence': 0.0,
                    'chroma_vector': [0] * 12,
                    'harmonic_rhythm': []
                }
            }
    
    def _essentia_analysis(self, audio_data: np.ndarray, beat_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform harmonic analysis using Essentia.
        
        Args:
            audio_data: Audio signal
            beat_results: Beat detection results
            
        Returns:
            Essentia-based harmonic analysis results
        """
        try:
            results = {}
            
            # Convert to Essentia format
            audio_vector = estd.VectorInput(audio_data)
            
            # Key detection
            key_extractor = estd.KeyExtractor(frameSize=4096, hopSize=hop_length)
            key_data = key_extractor(audio_vector)
            
            if hasattr(key_data, 'key'):
                results['detected_key'] = key_data.key()
                results['key_mode'] = 'major' if key_data.scale() == 'major' else 'minor'
                results['key_strength'] = key_data.strength()
            else:
                # Fallback if key detection doesn't work as expected
                chroma = self._compute_chroma_librosa(audio_data)
                key_info = self._estimate_key_from_chroma(chroma)
                results.update(key_info)
            
            # Chroma features
            chroma_extractor = estd.ChromaFeatures(frameSize=4096, hopSize=hop_length)
            chroma_features = chroma_extractor(audio_vector)
            
            # Average chroma vector
            if hasattr(chroma_features, 'shape') and len(chroma_features.shape) > 1:
                avg_chroma = np.mean(chroma_features, axis=1)
            else:
                avg_chroma = np.array(chroma_features) if hasattr(chroma_features, '__len__') else np.zeros(12)
            
            results['chroma_vector'] = avg_chroma.tolist()[:12]  # Ensure 12 pitch classes
            results['chroma_features'] = chroma_features.tolist() if hasattr(chroma_features, 'tolist') else chroma_features
            
            # Harmonic rhythm analysis using beat times
            harmonic_rhythm = self._analyze_harmonic_rhythm(audio_data, beat_results)
            results['harmonic_rhythm'] = harmonic_rhythm
            
            # Chord progression estimation (simplified)
            chord_progression = self._estimate_chord_progression(audio_data, beat_results)
            results['chord_progression'] = chord_progression
            
            return results
            
        except Exception as e:
            print(f"Essentia analysis failed: {str(e)}")
            # Fallback to librosa analysis
            return self._librosa_analysis(audio_data, beat_results)
    
    def _librosa_analysis(self, audio_data: np.ndarray, beat_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform harmonic analysis using Librosa (fallback).
        
        Args:
            audio_data: Audio signal
            beat_results: Beat detection results
            
        Returns:
            Librosa-based harmonic analysis results
        """
        try:
            results = {}
            
            # Compute chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Average chroma vector
            avg_chroma = np.mean(chroma, axis=1)
            results['chroma_vector'] = avg_chroma.tolist()
            results['chroma_features'] = chroma.tolist()
            
            # Estimate key from chroma
            key_info = self._estimate_key_from_chroma(avg_chroma)
            results.update(key_info)
            
            # Harmonic rhythm analysis
            harmonic_rhythm = self._analyze_harmonic_rhythm(audio_data, beat_results)
            results['harmonic_rhythm'] = harmonic_rhythm
            
            # Chord progression estimation
            chord_progression = self._estimate_chord_progression(audio_data, beat_results)
            results['chord_progression'] = chord_progression
            
            return results
            
        except Exception as e:
            print(f"Librosa analysis failed: {str(e)}")
            return {
                'detected_key': 'Unknown',
                'key_mode': 'Unknown',
                'key_strength': 0.0,
                'chroma_vector': [0] * 12,
                'chroma_features': [],
                'harmonic_rhythm': [],
                'chord_progression': []
            }
    
    def _compute_chroma_librosa(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute chroma features using librosa.
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Average chroma vector
        """
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            return np.mean(chroma, axis=1)
        except Exception:
            return np.zeros(12)
    
    def _estimate_key_from_chroma(self, chroma_vector: np.ndarray) -> Dict[str, Any]:
        """
        Estimate musical key from chroma vector.
        
        Args:
            chroma_vector: 12-element chroma vector
            
        Returns:
            Key estimation results
        """
        try:
            # Normalize chroma vector
            chroma_norm = chroma_vector / (np.sum(chroma_vector) + 1e-8)
            
            # Define key profiles (simplified major and minor scales)
            major_profile = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
            minor_profile = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
            
            # Calculate correlation with each key
            major_scores = []
            minor_scores = []
            
            for root in range(12):
                # Rotate profiles for each root
                major_rotated = np.roll(major_profile, root)
                minor_rotated = np.roll(minor_profile, root)
                
                # Calculate correlation
                major_corr = np.corrcoef(chroma_norm, major_rotated)[0, 1]
                minor_corr = np.corrcoef(chroma_norm, minor_rotated)[0, 1]
                
                major_scores.append(major_corr if not np.isnan(major_corr) else 0)
                minor_scores.append(minor_corr if not np.isnan(minor_corr) else 0)
            
            # Find best matching key
            major_best_idx = np.argmax(major_scores)
            minor_best_idx = np.argmax(minor_scores)
            
            if major_scores[major_best_idx] > minor_scores[minor_best_idx]:
                detected_key = self.note_names[major_best_idx]
                key_mode = 'major'
                confidence = max(0, min(1, major_scores[major_best_idx]))
            else:
                detected_key = self.note_names[minor_best_idx]
                key_mode = 'minor'
                confidence = max(0, min(1, minor_scores[minor_best_idx]))
            
            return {
                'detected_key': detected_key,
                'key_mode': key_mode,
                'key_strength': confidence,
                'major_scores': major_scores,
                'minor_scores': minor_scores
            }
            
        except Exception as e:
            print(f"Key estimation failed: {str(e)}")
            return {
                'detected_key': 'Unknown',
                'key_mode': 'Unknown',
                'key_strength': 0.0,
                'major_scores': [0] * 12,
                'minor_scores': [0] * 12
            }
    
    def _analyze_harmonic_rhythm(self, audio_data: np.ndarray, beat_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze harmonic rhythm (rate of chord changes).
        
        Args:
            audio_data: Audio signal
            beat_results: Beat detection results
            
        Returns:
            List of harmonic events with timing
        """
        try:
            harmonic_events = []
            
            # Get beat times for segmentation
            beat_times = np.array(beat_results.get('beat_times', []))
            
            if len(beat_times) < 4:
                return harmonic_events
            
            # Segment audio by beats and analyze each segment
            num_segments = min(16, len(beat_times) // 2)  # Limit to 16 segments
            segment_length = len(beat_times) // num_segments
            
            for i in range(num_segments):
                start_beat_idx = i * segment_length
                end_beat_idx = min((i + 1) * segment_length, len(beat_times) - 1)
                
                start_time = beat_times[start_beat_idx]
                end_time = beat_times[end_beat_idx]
                
                # Convert time to sample indices
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                
                # Extract segment
                if start_sample < len(audio_data) and end_sample <= len(audio_data):
                    segment = audio_data[start_sample:end_sample]
                    
                    # Analyze segment for key/chord
                    if len(segment) > 0:
                        chroma = self._compute_chroma_librosa(segment)
                        chord_estimation = self._estimate_chord_from_chroma(chroma)
                        
                        harmonic_events.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'estimated_chord': chord_estimation['chord'],
                            'confidence': chord_estimation['confidence']
                        })
            
            return harmonic_events
            
        except Exception as e:
            print(f"Harmonic rhythm analysis failed: {str(e)}")
            return []
    
    def _estimate_chord_progression(self, audio_data: np.ndarray, beat_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Estimate chord progression throughout the song.
        
        Args:
            audio_data: Audio signal
            beat_results: Beat detection results
            
        Returns:
            List of chord changes with timing
        """
        try:
            # Use harmonic rhythm as a basis for chord progression
            harmonic_events = self._analyze_harmonic_rhythm(audio_data, beat_results)
            
            # Convert to chord progression format
            progression = []
            
            for i, event in enumerate(harmonic_events):
                progression.append({
                    'time': event['start_time'],
                    'chord': event['estimated_chord'],
                    'duration': event.get('end_time', event['start_time']) - event['start_time'],
                    'confidence': event.get('confidence', 0.0)
                })
            
            return progression
            
        except Exception as e:
            print(f"Chord progression estimation failed: {str(e)}")
            return []
    
    def _estimate_chord_from_chroma(self, chroma_vector: np.ndarray) -> Dict[str, Any]:
        """
        Estimate chord from chroma vector.
        
        Args:
            chroma_vector: 12-element chroma vector
            
        Returns:
            Chord estimation results
        """
        try:
            # Normalize chroma vector
            chroma_norm = chroma_vector / (np.sum(chroma_vector) + 1e-8)
            
            # Define basic triads
            triads = {
                'major': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                'minor': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            }
            
            best_chord = 'N'
            best_score = 0.0
            
            # Test each root and mode
            for root in range(12):
                for mode, triad in triads.items():
                    # Rotate triad for this root
                    rotated_triad = np.roll(triad, root)
                    
                    # Calculate similarity
                    score = np.dot(chroma_norm, rotated_triad)
                    
                    if score > best_score:
                        best_score = score
                        if mode == 'major':
                            best_chord = self.note_names[root]
                        else:
                            best_chord = f"{self.note_names[root]}m"
            
            return {
                'chord': best_chord,
                'confidence': min(1.0, best_score)
            }
            
        except Exception:
            return {
                'chord': 'N',
                'confidence': 0.0
            }
    
    def _calculate_harmonic_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate confidence score for harmonic analysis.
        
        Args:
            results: Harmonic analysis results
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []
        
        # Key strength
        key_strength = results.get('key_strength', 0.0)
        if key_strength > 0:
            confidence_factors.append(key_strength)
        
        # Chroma vector coherence
        chroma_vector = results.get('chroma_vector', [])
        if len(chroma_vector) == 12:
            # Calculate entropy of chroma distribution
            chroma_array = np.array(chroma_vector)
            if np.sum(chroma_array) > 0:
                normalized = chroma_array / np.sum(chroma_array)
                entropy = -np.sum(normalized * np.log2(normalized + 1e-8))
                # Lower entropy = higher confidence
                entropy_confidence = max(0.0, 1.0 - entropy / 4.0)  # Max entropy for uniform distribution
                confidence_factors.append(entropy_confidence)
        
        # Number of harmonic events detected
        num_events = len(results.get('harmonic_rhythm', []))
        if num_events > 0:
            # More events might indicate more complex harmonic content
            event_confidence = min(1.0, num_events / 8.0)
            confidence_factors.append(event_confidence)
        
        # Calculate average confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.0


def detect_key(audio_data: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]:
    """
    Convenience function to detect key from audio data.
    
    Args:
        audio_data: Audio signal array
        sample_rate: Audio sample rate
        
    Returns:
        Key detection results
    """
    analyzer = HarmonicAnalyzer(sample_rate=sample_rate)
    results = analyzer.analyze(audio_data, {})
    return results.get('key_info', {})


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=44100)
        
        # Analyze harmony
        analyzer = HarmonicAnalyzer()
        results = analyzer.analyze(y, {})
        
        key_info = results.get('key_info', {})
        print(f"Key: {key_info.get('key', 'Unknown')} {key_info.get('mode', 'Unknown')}")
        print(f"Confidence: {key_info.get('confidence', 0):.2f}")
        print(f"Chord progression: {len(results.get('chord_progression', []))} events")
    else:
        print("Usage: python harmonic.py <audio_file>")
