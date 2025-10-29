"""
Spectral Analysis Module
Frequency domain features for color mapping and timbre analysis
"""

import numpy as np
import librosa
import warnings
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal
from scipy.fft import fft, fftfreq
warnings.filterwarnings('ignore')


class SpectralAnalyzer:
    """
    Spectral analysis for frequency domain features.
    
    Features:
    - Spectral centroid (brightness)
    - Spectral rolloff
    - Spectral bandwidth
    - Zero crossing rate
    - MFCCs for timbre
    - Spectral contrast
    - RMS energy
    - Spectral flux
    """
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        """
        Initialize spectral analyzer.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Analysis hop length
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = 2048  # Default frame size
        
    def analyze(self, audio_data: np.ndarray, beat_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive spectral analysis.
        
        Args:
            audio_data: Audio signal array
            beat_results: Results from beat detection
            
        Returns:
            Dictionary containing spectral analysis results
        """
        try:
            # Basic validation
            if len(audio_data) == 0:
                raise ValueError("Audio data is empty")
            
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            print(f"Spectral analysis on {len(audio_data)} samples ({len(audio_data)/self.sample_rate:.2f}s)")
            
            results = {}
            
            # Compute STFT
            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Compute frequency bins
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            results['frequency_bins'] = freqs.tolist()
            
            # Spectral centroid (brightness)
            centroid_results = self._spectral_centroid(magnitude, freqs)
            results.update(centroid_results)
            
            # Spectral rolloff
            rolloff_results = self._spectral_rolloff(magnitude, freqs)
            results.update(rolloff_results)
            
            # Spectral bandwidth
            bandwidth_results = self._spectral_bandwidth(magnitude, freqs, centroid_results.get('centroid_times', []))
            results.update(bandwidth_results)
            
            # Spectral contrast
            contrast_results = self._spectral_contrast(magnitude, freqs)
            results.update(contrast_results)
            
            # Zero crossing rate
            zcr_results = self._zero_crossing_rate(audio_data)
            results.update(zcr_results)
            
            # MFCCs for timbre
            mfcc_results = self._mfcc_analysis(audio_data)
            results.update(mfcc_results)
            
            # RMS energy
            energy_results = self._energy_analysis(magnitude)
            results.update(energy_results)
            
            # Spectral flux
            flux_results = self._spectral_flux(magnitude)
            results.update(flux_results)
            
            # Spectral flatness
            flatness_results = self._spectral_flatness(magnitude, freqs)
            results.update(flatness_results)
            
            # Color mapping features
            color_features = self._extract_color_features(results)
            results['color_features'] = color_features
            
            # Calculate confidence score
            confidence = self._calculate_spectral_confidence(results)
            results['confidence_scores']['spectral_analysis'] = confidence
            
            # Add summary information
            results['spectral_info'] = {
                'brightness': results.get('avg_brightness', 0.0),
                'energy': results.get('avg_energy', 0.0),
                'timbre_complexity': results.get('avg_bandwidth', 0.0),
                'color_palette': color_features
            }
            
            return results
            
        except Exception as e:
            print(f"Spectral analysis error: {str(e)}")
            return {
                'error': str(e),
                'confidence_scores': {'spectral_analysis': 0.0},
                'spectral_info': {
                    'brightness': 0.0,
                    'energy': 0.0,
                    'timbre_complexity': 0.0,
                    'color_palette': []
                }
            }
    
    def _spectral_centroid(self, magnitude: np.ndarray, freqs: np.ndarray) -> Dict[str, Any]:
        """
        Calculate spectral centroid (brightness center of mass).
        
        Args:
            magnitude: STFT magnitude spectrum
            freqs: Frequency bins
            
        Returns:
            Spectral centroid results
        """
        try:
            centroids = []
            
            for frame in range(magnitude.shape[1]):
                # Get magnitude for this frame
                mag_frame = magnitude[:, frame]
                
                # Calculate weighted average of frequencies
                if np.sum(mag_frame) > 0:
                    centroid = np.sum(freqs * mag_frame) / np.sum(mag_frame)
                else:
                    centroid = 0.0
                
                centroids.append(centroid)
            
            centroid_times = librosa.frames_to_time(
                np.arange(len(centroids)),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            avg_centroid = np.mean(centroids)
            std_centroid = np.std(centroids)
            
            return {
                'spectral_centroid': centroids,
                'centroid_times': centroid_times.tolist(),
                'avg_brightness': float(avg_centroid),
                'brightness_variance': float(std_centroid)
            }
            
        except Exception as e:
            print(f"Spectral centroid calculation failed: {str(e)}")
            return {
                'spectral_centroid': [],
                'centroid_times': [],
                'avg_brightness': 0.0,
                'brightness_variance': 0.0
            }
    
    def _spectral_rolloff(self, magnitude: np.ndarray, freqs: np.ndarray) -> Dict[str, Any]:
        """
        Calculate spectral rolloff (frequency below which 85% of energy exists).
        
        Args:
            magnitude: STFT magnitude spectrum
            freqs: Frequency bins
            
        Returns:
            Spectral rolloff results
        """
        try:
            rolloffs = []
            rolloff_threshold = 0.85
            
            for frame in range(magnitude.shape[1]):
                mag_frame = magnitude[:, frame]
                
                # Calculate cumulative energy
                cumulative = np.cumsum(mag_frame)
                total_energy = cumulative[-1]
                
                if total_energy > 0:
                    # Find frequency below which threshold percentage of energy exists
                    threshold_idx = np.where(cumulative >= rolloff_threshold * total_energy)[0]
                    if len(threshold_idx) > 0:
                        rolloff_freq = freqs[threshold_idx[0]]
                    else:
                        rolloff_freq = freqs[-1]
                else:
                    rolloff_freq = 0.0
                
                rolloffs.append(rolloff_freq)
            
            avg_rolloff = np.mean(rolloffs)
            
            return {
                'spectral_rolloff': rolloffs,
                'avg_rolloff': float(avg_rolloff)
            }
            
        except Exception as e:
            print(f"Spectral rolloff calculation failed: {str(e)}")
            return {
                'spectral_rolloff': [],
                'avg_rolloff': 0.0
            }
    
    def _spectral_bandwidth(self, magnitude: np.ndarray, freqs: np.ndarray, 
                          centroid_times: List[float]) -> Dict[str, Any]:
        """
        Calculate spectral bandwidth (spread around centroid).
        
        Args:
            magnitude: STFT magnitude spectrum
            freqs: Frequency bins
            centroid_times: Times corresponding to centroids
            
        Returns:
            Spectral bandwidth results
        """
        try:
            bandwidths = []
            
            for frame in range(magnitude.shape[1]):
                mag_frame = magnitude[:, frame]
                
                # Calculate centroid for this frame
                if np.sum(mag_frame) > 0:
                    centroid = np.sum(freqs * mag_frame) / np.sum(mag_frame)
                    
                    # Calculate bandwidth (weighted standard deviation)
                    variance = np.sum(((freqs - centroid) ** 2) * mag_frame) / np.sum(mag_frame)
                    bandwidth = np.sqrt(variance)
                else:
                    bandwidth = 0.0
                
                bandwidths.append(bandwidth)
            
            avg_bandwidth = np.mean(bandwidths)
            
            return {
                'spectral_bandwidth': bandwidths,
                'avg_bandwidth': float(avg_bandwidth)
            }
            
        except Exception as e:
            print(f"Spectral bandwidth calculation failed: {str(e)}")
            return {
                'spectral_bandwidth': [],
                'avg_bandwidth': 0.0
            }
    
    def _spectral_contrast(self, magnitude: np.ndarray, freqs: np.ndarray) -> Dict[str, Any]:
        """
        Calculate spectral contrast (difference between peaks and valleys).
        
        Args:
            magnitude: STFT magnitude spectrum
            freqs: Frequency bins
            
        Returns:
            Spectral contrast results
        """
        try:
            # Use librosa's spectral contrast
            contrast = librosa.feature.spectral_contrast(
                S=np.abs(magnitude),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Average contrast across frequency bands
            avg_contrast = np.mean(contrast, axis=1)
            
            # Frequency bands for contrast
            bands = ['low', 'low-mid', 'mid', 'high-mid', 'high']
            
            return {
                'spectral_contrast': contrast.tolist(),
                'avg_contrast_by_band': {
                    band: float(contrast_value) 
                    for band, contrast_value in zip(bands, avg_contrast)
                }
            }
            
        except Exception as e:
            print(f"Spectral contrast calculation failed: {str(e)}")
            return {
                'spectral_contrast': [],
                'avg_contrast_by_band': {}
            }
    
    def _zero_crossing_rate(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Calculate zero crossing rate.
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Zero crossing rate results
        """
        try:
            zcr = librosa.feature.zero_crossing_rate(
                audio_data,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]
            
            zcr_times = librosa.frames_to_time(
                np.arange(len(zcr)),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            avg_zcr = np.mean(zcr)
            
            return {
                'zero_crossing_rate': zcr.tolist(),
                'zcr_times': zcr_times.tolist(),
                'avg_zcr': float(avg_zcr)
            }
            
        except Exception as e:
            print(f"Zero crossing rate calculation failed: {str(e)}")
            return {
                'zero_crossing_rate': [],
                'zcr_times': [],
                'avg_zcr': 0.0
            }
    
    def _mfcc_analysis(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Calculate MFCCs for timbre analysis.
        
        Args:
            audio_data: Audio signal
            
        Returns:
            MFCC analysis results
        """
        try:
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_mfcc=13
            )
            
            # Average MFCCs across time
            avg_mfccs = np.mean(mfccs, axis=1)
            
            # Calculate delta MFCCs
            delta_mfccs = librosa.feature.delta(mfccs)
            avg_delta_mfccs = np.mean(delta_mfccs, axis=1)
            
            return {
                'mfccs': mfccs.tolist(),
                'avg_mfccs': avg_mfccs.tolist(),
                'delta_mfccs': delta_mfccs.tolist(),
                'avg_delta_mfccs': avg_delta_mfccs.tolist()
            }
            
        except Exception as e:
            print(f"MFCC analysis failed: {str(e)}")
            return {
                'mfccs': [],
                'avg_mfccs': [],
                'delta_mfccs': [],
                'avg_delta_mfccs': []
            }
    
    def _energy_analysis(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """
        Calculate RMS energy.
        
        Args:
            magnitude: STFT magnitude spectrum
            
        Returns:
            Energy analysis results
        """
        try:
            # Calculate RMS energy from magnitude
            energy = np.sqrt(np.mean(magnitude ** 2, axis=0))
            
            energy_times = librosa.frames_to_time(
                np.arange(len(energy)),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            avg_energy = np.mean(energy)
            std_energy = np.std(energy)
            
            return {
                'rms_energy': energy.tolist(),
                'energy_times': energy_times.tolist(),
                'avg_energy': float(avg_energy),
                'energy_variance': float(std_energy)
            }
            
        except Exception as e:
            print(f"Energy analysis failed: {str(e)}")
            return {
                'rms_energy': [],
                'energy_times': [],
                'avg_energy': 0.0,
                'energy_variance': 0.0
            }
    
    def _spectral_flux(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """
        Calculate spectral flux (rate of change of magnitude).
        
        Args:
            magnitude: STFT magnitude spectrum
            
        Returns:
            Spectral flux results
        """
        try:
            # Calculate difference between consecutive frames
            flux = np.diff(magnitude, axis=1)
            
            # Take sum of absolute values
            flux = np.sum(flux ** 2, axis=0)
            
            flux_times = librosa.frames_to_time(
                np.arange(len(flux)),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            avg_flux = np.mean(flux)
            
            return {
                'spectral_flux': flux.tolist(),
                'flux_times': flux_times.tolist(),
                'avg_flux': float(avg_flux)
            }
            
        except Exception as e:
            print(f"Spectral flux calculation failed: {str(e)}")
            return {
                'spectral_flux': [],
                'flux_times': [],
                'avg_flux': 0.0
            }
    
    def _spectral_flatness(self, magnitude: np.ndarray, freqs: np.ndarray) -> Dict[str, Any]:
        """
        Calculate spectral flatness (tonality vs noise).
        
        Args:
            magnitude: STFT magnitude spectrum
            freqs: Frequency bins
            
        Returns:
            Spectral flatness results
        """
        try:
            # Calculate geometric mean / arithmetic mean for each frame
            flatness = []
            
            for frame in range(magnitude.shape[1]):
                mag_frame = magnitude[:, frame]
                
                # Avoid zeros for geometric mean
                mag_frame_safe = np.where(mag_frame > 0, mag_frame, 1e-10)
                
                geometric_mean = np.exp(np.mean(np.log(mag_frame_safe)))
                arithmetic_mean = np.mean(mag_frame)
                
                if arithmetic_mean > 0:
                    flatness_val = geometric_mean / arithmetic_mean
                else:
                    flatness_val = 0.0
                
                flatness.append(flatness_val)
            
            avg_flatness = np.mean(flatness)
            
            return {
                'spectral_flatness': flatness,
                'avg_flatness': float(avg_flatness)
            }
            
        except Exception as e:
            print(f"Spectral flatness calculation failed: {str(e)}")
            return {
                'spectral_flatness': [],
                'avg_flatness': 0.0
            }
    
    def _extract_color_features(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract features for color mapping.
        
        Args:
            results: Spectral analysis results
            
        Returns:
            List of color features for visualization
        """
        try:
            color_features = []
            
            # Brightness (spectral centroid)
            brightness = results.get('avg_brightness', 0.0)
            # Map to color temperature (warm vs cool)
            if brightness < 1000:
                color_temp = 'warm'
            elif brightness > 4000:
                color_temp = 'cool'
            else:
                color_temp = 'neutral'
            
            color_features.append({
                'feature': 'brightness',
                'value': brightness,
                'color_mapping': color_temp,
                'description': 'Frequency center of mass'
            })
            
            # Energy (RMS)
            energy = results.get('avg_energy', 0.0)
            # Map to saturation (high energy = high saturation)
            saturation = min(1.0, energy * 10)
            
            color_features.append({
                'feature': 'energy',
                'value': energy,
                'color_mapping': f"saturation_{saturation:.2f}",
                'description': 'Overall signal strength'
            })
            
            # Bandwidth (timbre complexity)
            bandwidth = results.get('avg_bandwidth', 0.0)
            # Map to colorfulness (complex = more colorful)
            colorfulness = min(1.0, bandwidth / 2000)
            
            color_features.append({
                'feature': 'complexity',
                'value': bandwidth,
                'color_mapping': f"colorfulness_{colorfulness:.2f}",
                'description': 'Timbral complexity'
            })
            
            # Spectral contrast
            contrast_data = results.get('avg_contrast_by_band', {})
            if contrast_data:
                high_contrast = contrast_data.get('high', 0.0)
                # Map to value (contrast = brightness variation)
                brightness_var = min(1.0, high_contrast)
                
                color_features.append({
                    'feature': 'contrast',
                    'value': high_contrast,
                    'color_mapping': f"value_{brightness_var:.2f}",
                    'description': 'Dynamic range'
                })
            
            return color_features
            
        except Exception as e:
            print(f"Color feature extraction failed: {str(e)}")
            return []
    
    def _calculate_spectral_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate confidence score for spectral analysis.
        
        Args:
            results: Spectral analysis results
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []
        
        # Signal quality based on energy
        avg_energy = results.get('avg_energy', 0.0)
        if avg_energy > 0.01:  # Non-silent signal
            confidence_factors.append(min(1.0, avg_energy * 100))
        
        # Spectral content diversity
        bandwidth = results.get('avg_bandwidth', 0.0)
        if bandwidth > 100:  # Non-monotonic signal
            confidence_factors.append(min(1.0, bandwidth / 1000))
        
        # Spectral flux (activity)
        avg_flux = results.get('avg_flux', 0.0)
        if avg_flux > 0:
            confidence_factors.append(min(1.0, avg_flux * 10))
        
        # Number of features computed successfully
        computed_features = 0
        total_features = 8  # centroid, rolloff, bandwidth, contrast, zcr, mfcc, energy, flux
        
        feature_checks = [
            results.get('spectral_centroid', []),
            results.get('spectral_rolloff', []),
            results.get('spectral_bandwidth', []),
            results.get('spectral_contrast', []),
            results.get('zero_crossing_rate', []),
            results.get('mfccs', []),
            results.get('rms_energy', []),
            results.get('spectral_flux', [])
        ]
        
        for feature in feature_checks:
            if feature:
                computed_features += 1
        
        feature_confidence = computed_features / total_features
        confidence_factors.append(feature_confidence)
        
        # Calculate average confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.0


def extract_spectral_features(audio_data: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]:
    """
    Convenience function to extract spectral features from audio data.
    
    Args:
        audio_data: Audio signal array
        sample_rate: Audio sample rate
        
    Returns:
        Spectral features dictionary
    """
    analyzer = SpectralAnalyzer(sample_rate=sample_rate)
    results = analyzer.analyze(audio_data, {})
    return results.get('spectral_info', {})


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=44100)
        
        # Analyze spectrum
        analyzer = SpectralAnalyzer()
        results = analyzer.analyze(y, {})
        
        spectral_info = results.get('spectral_info', {})
        print(f"Brightness: {spectral_info.get('brightness', 0):.2f} Hz")
        print(f"Energy: {spectral_info.get('energy', 0):.4f}")
        print(f"Timbre complexity: {spectral_info.get('timbre_complexity', 0):.2f}")
        print(f"Color features: {len(spectral_info.get('color_palette', []))}")
    else:
        print("Usage: python spectral.py <audio_file>")
