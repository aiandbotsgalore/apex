"""
Beat Detection Module
BPM detection and beat grid extraction using librosa
"""

import numpy as np
import librosa
import warnings
from typing import Dict, Any, List, Tuple, Optional
warnings.filterwarnings('ignore')


class BeatDetector:
    """
    Beat detection using multiple librosa algorithms for robust BPM and beat timing detection.
    
    Features:
    - Multiple onset detection algorithms
    - BPM estimation with confidence scoring
    - Beat position grid extraction
    - Tempo curve analysis
    - Downbeat detection
    """
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        """
        Initialize beat detector.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Analysis hop length
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
    def analyze(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive beat analysis.
        
        Args:
            audio_data: Audio signal array
            
        Returns:
            Dictionary containing beat analysis results
        """
        try:
            # Basic validation
            if len(audio_data) == 0:
                raise ValueError("Audio data is empty")
            
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            print(f"Beat analysis on {len(audio_data)} samples ({len(audio_data)/self.sample_rate:.2f}s)")
            
            results = {}
            
            # Method 1: Librosa beat tracking
            beat_results = self._librosa_beat_tracking(audio_data)
            results.update(beat_results)
            
            # Method 2: Onset envelope analysis
            onset_results = self._onset_analysis(audio_data)
            results.update(onset_results)
            
            # Method 3: Tempo estimation
            tempo_results = self._tempo_estimation(audio_data)
            results.update(tempo_results)
            
            # Method 4: Beat strength analysis
            strength_results = self._beat_strength_analysis(audio_data, onset_results)
            results.update(strength_results)
            
            # Method 5: Downbeat detection (if applicable)
            downbeat_results = self._downbeat_detection(audio_data, beat_results)
            results.update(downbeat_results)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(results)
            results['confidence_scores']['beat_detection'] = confidence
            
            # Add summary information
            results['beat_info'] = {
                'bpm': results.get('bpm', 0),
                'confidence': confidence,
                'num_beats': len(results.get('beat_times', [])),
                'duration': len(audio_data) / self.sample_rate
            }
            
            return results
            
        except Exception as e:
            print(f"Beat detection error: {str(e)}")
            return {
                'error': str(e),
                'confidence_scores': {'beat_detection': 0.0},
                'beat_info': {'bpm': 0, 'confidence': 0.0, 'num_beats': 0}
            }
    
    def _librosa_beat_tracking(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Use librosa's built-in beat tracking algorithm.
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Beat tracking results
        """
        try:
            # Calculate tempo and beat times
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio_data, 
                sr=self.sample_rate,
                hop_length=self.hop_length,
                units='time'
            )
            
            # Convert to numpy arrays
            beat_times = np.array(beat_frames) if not isinstance(beat_frames, np.ndarray) else beat_frames
            
            # Calculate beat intervals
            beat_intervals = np.diff(beat_times) if len(beat_times) > 1 else np.array([])
            
            # Calculate confidence based on regularity of beat intervals
            confidence = self._calculate_beat_regularity(beat_intervals)
            
            return {
                'bpm': float(tempo),
                'beat_times': beat_times.tolist(),
                'beat_frames': beat_frames.tolist() if hasattr(beat_frames, 'tolist') else list(beat_frames),
                'beat_intervals': beat_intervals.tolist(),
                'beat_regularity': confidence
            }
            
        except Exception as e:
            print(f"Librosa beat tracking failed: {str(e)}")
            return {
                'bpm': 0.0,
                'beat_times': [],
                'beat_frames': [],
                'beat_intervals': [],
                'beat_regularity': 0.0
            }
    
    def _onset_analysis(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze onset envelope for beat detection.
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Onset analysis results
        """
        try:
            # Compute onset strength envelope
            onset_frames = librosa.onset.onset_detect(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                units='time'
            )
            
            # Get onset strength with frames
            onset_strength = librosa.onset.onset_strength(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Calculate onset envelope
            onset_times = librosa.frames_to_time(
                np.arange(len(onset_strength)),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            return {
                'onset_times': np.array(onset_frames).tolist() if not isinstance(onset_frames, np.ndarray) else onset_frames.tolist(),
                'onset_strength': onset_strength.tolist(),
                'onset_envelope_times': onset_times.tolist()
            }
            
        except Exception as e:
            print(f"Onset analysis failed: {str(e)}")
            return {
                'onset_times': [],
                'onset_strength': [],
                'onset_envelope_times': []
            }
    
    def _tempo_estimation(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Estimate tempo using multiple methods.
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Tempo estimation results
        """
        try:
            # Method 1: CQT-based tempo estimation
            cqt_tempo, _ = librosa.beat.tempo(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                aggregate=np.median
            )
            
            # Method 2: Chroma-based tempo estimation
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            chroma_tempo, _ = librosa.beat.tempo(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                feature=librosa.feature.chroma_stft,
                aggregate=np.median
            )
            
            # Choose best estimate
            primary_tempo = float(cqt_tempo) if abs(float(cqt_tempo) - 120) > abs(float(chroma_tempo) - 120) else float(chroma_tempo)
            
            return {
                'cqt_tempo': float(cqt_tempo),
                'chroma_tempo': float(chroma_tempo),
                'estimated_tempo': primary_tempo
            }
            
        except Exception as e:
            print(f"Tempo estimation failed: {str(e)}")
            return {
                'cqt_tempo': 0.0,
                'chroma_tempo': 0.0,
                'estimated_tempo': 0.0
            }
    
    def _beat_strength_analysis(self, audio_data: np.ndarray, onset_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze beat strength and periodicity.
        
        Args:
            audio_data: Audio signal
            onset_results: Results from onset analysis
            
        Returns:
            Beat strength analysis results
        """
        try:
            # Compute beat strength using spectral contrast
            beat_strength = librosa.beat.beat_track(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                strength=np.ones
            )[1] if hasattr(librosa.beat.beat_track, '__call__') else None
            
            # Alternative: use onset envelope for strength
            onset_strength = onset_results.get('onset_strength', [])
            if onset_strength:
                # Calculate strength statistics
                strength_values = np.array(onset_strength)
                strength_mean = np.mean(strength_values)
                strength_std = np.std(strength_values)
                strength_max = np.max(strength_values)
                
                # Calculate periodicity using autocorrelation
                periodicity = self._calculate_periodicity(strength_values)
            else:
                strength_mean = strength_std = strength_max = 0.0
                periodicity = 0.0
            
            return {
                'beat_strength_mean': float(strength_mean),
                'beat_strength_std': float(strength_std),
                'beat_strength_max': float(strength_max),
                'beat_periodicity': float(periodicity)
            }
            
        except Exception as e:
            print(f"Beat strength analysis failed: {str(e)}")
            return {
                'beat_strength_mean': 0.0,
                'beat_strength_std': 0.0,
                'beat_strength_max': 0.0,
                'beat_periodicity': 0.0
            }
    
    def _downbeat_detection(self, audio_data: np.ndarray, beat_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect downbeats (first beat of each bar).
        
        Args:
            audio_data: Audio signal
            beat_results: Results from beat tracking
            
        Returns:
            Downbeat detection results
        """
        try:
            # Simple downbeat detection using spectral features
            # More sophisticated algorithms would require additional libraries
            
            beat_times = np.array(beat_results.get('beat_times', []))
            
            if len(beat_times) < 4:  # Need at least 4 beats for bar detection
                return {
                    'downbeat_times': [],
                    'time_signature': 'unknown'
                }
            
            # Estimate time signature by analyzing beat grouping
            # Look for patterns in beat intervals to identify bars
            beat_intervals = np.diff(beat_times)
            interval_std = np.std(beat_intervals)
            
            # Simple heuristic: assume 4/4 time if beat intervals are regular
            is_regular = interval_std < (np.mean(beat_intervals) * 0.1)
            
            if is_regular:
                time_signature = '4/4'
                # Assume first beat of each group of 4 is a downbeat
                num_bars = len(beat_times) // 4
                downbeat_indices = np.arange(0, min(num_bars * 4, len(beat_times)), 4)
                downbeat_times = beat_times[downbeat_indices]
            else:
                time_signature = 'unknown'
                downbeat_times = beat_times[::4] if len(beat_times) >= 4 else beat_times[:1]
            
            return {
                'downbeat_times': downbeat_times.tolist(),
                'time_signature': time_signature,
                'num_bars': int(len(beat_times) // 4) if is_regular else 0
            }
            
        except Exception as e:
            print(f"Downbeat detection failed: {str(e)}")
            return {
                'downbeat_times': [],
                'time_signature': 'unknown',
                'num_bars': 0
            }
    
    def _calculate_beat_regularity(self, beat_intervals: np.ndarray) -> float:
        """
        Calculate beat regularity score.
        
        Args:
            beat_intervals: Array of time intervals between beats
            
        Returns:
            Regularity score (0.0 to 1.0)
        """
        if len(beat_intervals) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_interval = np.mean(beat_intervals)
        std_interval = np.std(beat_intervals)
        
        if mean_interval == 0:
            return 0.0
        
        cv = std_interval / mean_interval
        
        # Convert to confidence score (lower CV = higher confidence)
        confidence = max(0.0, 1.0 - cv)
        
        return float(confidence)
    
    def _calculate_periodicity(self, strength_values: np.ndarray) -> float:
        """
        Calculate periodicity in strength values using autocorrelation.
        
        Args:
            strength_values: Beat strength values
            
        Returns:
            Periodicity score (0.0 to 1.0)
        """
        if len(strength_values) < 4:
            return 0.0
        
        # Compute autocorrelation
        autocorr = np.correlate(strength_values, strength_values, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Look for periodic peaks
        if len(autocorr) < 4:
            return 0.0
        
        # Find the highest peak after the zero lag
        peak_idx = np.argmax(autocorr[1:]) + 1
        peak_value = autocorr[peak_idx]
        
        return float(max(0.0, min(1.0, peak_value)))
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall confidence for beat detection.
        
        Args:
            results: Beat analysis results
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []
        
        # Regularity of beat intervals
        beat_regularity = results.get('beat_regularity', 0.0)
        if beat_regularity > 0:
            confidence_factors.append(beat_regularity)
        
        # Beat strength
        beat_periodicity = results.get('beat_periodicity', 0.0)
        if beat_periodicity > 0:
            confidence_factors.append(beat_periodicity)
        
        # Number of detected beats (more beats = more confidence, up to a point)
        num_beats = len(results.get('beat_times', []))
        if num_beats > 0:
            # Normalize by duration to account for song length
            duration = results.get('duration', 1.0)
            beat_density = min(1.0, num_beats / (duration * 2))  # Expect ~2 beats per second
            confidence_factors.append(beat_density)
        
        # Tempo confidence (reasonable tempo range)
        tempo = results.get('bpm', 0)
        if 60 <= tempo <= 200:
            confidence_factors.append(1.0)
        elif 40 <= tempo <= 240:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.0)
        
        # Calculate average confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.0


def estimate_bpm(audio_data: np.ndarray, sample_rate: int = 44100) -> float:
    """
    Convenience function to estimate BPM from audio data.
    
    Args:
        audio_data: Audio signal array
        sample_rate: Audio sample rate
        
    Returns:
        Estimated BPM
    """
    detector = BeatDetector(sample_rate=sample_rate)
    results = detector.analyze(audio_data)
    return results.get('bpm', 0.0)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        
        # Load audio
        import librosa
        y, sr = librosa.load(audio_file, sr=44100)
        
        # Analyze beats
        detector = BeatDetector()
        results = detector.analyze(y)
        
        print(f"BPM: {results.get('bpm', 'Unknown')}")
        print(f"Number of beats: {len(results.get('beat_times', []))}")
        print(f"Confidence: {results.get('confidence_scores', {}).get('beat_detection', 0):.2f}")
    else:
        print("Usage: python beat_detector.py <audio_file>")
