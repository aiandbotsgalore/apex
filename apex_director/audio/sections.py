"""
Section Detection Module
Automatic song structure detection (verse, chorus, bridge, etc.)
"""

import numpy as np
import librosa
import warnings
from typing import Dict, Any, List, Tuple, Optional
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')


class SectionDetector:
    """
    Song structure detection using audio similarity and segmentation.
    
    Features:
    - Beat-synchronous feature extraction
    - Similarity matrix computation
    - Automatic section boundary detection
    - Section labeling (verse, chorus, bridge)
    - Structure complexity analysis
    """
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        """
        Initialize section detector.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Analysis hop length
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Section types with typical characteristics
        self.section_types = {
            'intro': {'min_duration': 8, 'energy_level': 'low', 'complexity': 'low'},
            'verse': {'min_duration': 16, 'energy_level': 'medium', 'complexity': 'medium'},
            'pre_chorus': {'min_duration': 8, 'energy_level': 'increasing', 'complexity': 'medium'},
            'chorus': {'min_duration': 16, 'energy_level': 'high', 'complexity': 'high'},
            'post_chorus': {'min_duration': 8, 'energy_level': 'decreasing', 'complexity': 'medium'},
            'bridge': {'min_duration': 12, 'energy_level': 'variable', 'complexity': 'high'},
            'outro': {'min_duration': 8, 'energy_level': 'low', 'complexity': 'low'}
        }
        
    def analyze(self, audio_data: np.ndarray, beat_results: Dict[str, Any], 
               spectral_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive section detection analysis.
        
        Args:
            audio_data: Audio signal array
            beat_results: Results from beat detection
            spectral_results: Results from spectral analysis
            
        Returns:
            Dictionary containing section detection results
        """
        try:
            # Basic validation
            if len(audio_data) == 0:
                raise ValueError("Audio data is empty")
            
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            duration = len(audio_data) / self.sample_rate
            print(f"Section detection on {len(audio_data)} samples ({duration:.2f}s)")
            
            results = {}
            
            # Extract beat-synchronous features
            print("Extracting beat-synchronous features...")
            features = self._extract_beat_sync_features(audio_data, beat_results)
            results['beat_sync_features'] = features
            
            # Compute self-similarity matrix
            print("Computing similarity matrix...")
            similarity_matrix = self._compute_similarity_matrix(features)
            results['similarity_matrix'] = similarity_matrix
            
            # Detect section boundaries
            print("Detecting section boundaries...")
            boundaries = self._detect_boundaries(similarity_matrix, features, duration)
            results['boundaries'] = boundaries
            
            # Segment audio based on boundaries
            print("Segmenting audio structure...")
            sections = self._segment_sections(audio_data, boundaries, features, beat_results)
            results['sections'] = sections
            
            # Analyze section characteristics
            print("Analyzing section characteristics...")
            section_analysis = self._analyze_section_characteristics(sections, spectral_results)
            results['section_analysis'] = section_analysis
            
            # Label sections
            print("Labeling sections...")
            labeled_sections = self._label_sections(sections, section_analysis)
            results['labeled_sections'] = labeled_sections
            
            # Calculate confidence score
            confidence = self._calculate_section_confidence(results)
            results['confidence_scores']['section_detection'] = confidence
            
            # Add summary information
            results['section_info'] = {
                'num_sections': len(labeled_sections),
                'structure_type': self._identify_structure_type(labeled_sections),
                'complexity_score': self._calculate_structure_complexity(labeled_sections),
                'sections': labeled_sections
            }
            
            return results
            
        except Exception as e:
            print(f"Section detection error: {str(e)}")
            return {
                'error': str(e),
                'confidence_scores': {'section_detection': 0.0},
                'section_info': {
                    'num_sections': 0,
                    'structure_type': 'unknown',
                    'complexity_score': 0.0,
                    'sections': []
                }
            }
    
    def _extract_beat_sync_features(self, audio_data: np.ndarray, beat_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features synchronized with beats.
        
        Args:
            audio_data: Audio signal
            beat_results: Beat detection results
            
        Returns:
            Beat-synchronous features
        """
        try:
            beat_times = np.array(beat_results.get('beat_times', []))
            
            if len(beat_times) < 8:
                print("Not enough beats for beat-synchronous analysis")
                return {'features': [], 'beat_times': []}
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_mfcc=13
            )
            
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Energy
            rms = librosa.feature.rms(
                y=audio_data,
                hop_length=self.hop_length
            )
            
            # Convert to time
            chroma_times = librosa.frames_to_time(
                np.arange(chroma.shape[1]),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Create beat-synchronized feature matrix
            features = []
            feature_times = []
            
            for beat_time in beat_times:
                # Find closest frame to this beat
                chroma_idx = np.argmin(np.abs(chroma_times - beat_time))
                mfcc_idx = chroma_idx  # Same timing
                centroid_idx = chroma_idx
                rms_idx = chroma_idx
                
                # Extract features at this beat
                beat_features = []
                
                # Chroma features (12 dims)
                if chroma_idx < chroma.shape[1]:
                    beat_features.extend(chroma[:, chroma_idx])
                
                # MFCC features (13 dims)
                if mfcc_idx < mfcc.shape[1]:
                    beat_features.extend(mfcc[:, mfcc_idx])
                
                # Spectral centroid (1 dim)
                if centroid_idx < centroid.shape[1]:
                    beat_features.append(centroid[0, centroid_idx])
                
                # RMS energy (1 dim)
                if rms_idx < rms.shape[1]:
                    beat_features.append(rms[0, rms_idx])
                
                if beat_features:
                    features.append(beat_features)
                    feature_times.append(beat_time)
            
            return {
                'features': np.array(features) if features else np.array([]).reshape(0, 27),
                'feature_times': feature_times,
                'feature_names': ['chroma_0', 'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4',
                                 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9',
                                 'chroma_10', 'chroma_11', 'mfcc_0', 'mfcc_1', 'mfcc_2',
                                 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8',
                                 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'centroid', 'rms']
            }
            
        except Exception as e:
            print(f"Beat-synchronous feature extraction failed: {str(e)}")
            return {
                'features': np.array([]).reshape(0, 27),
                'feature_times': [],
                'feature_names': []
            }
    
    def _compute_similarity_matrix(self, feature_data: Dict[str, Any]) -> np.ndarray:
        """
        Compute self-similarity matrix for section detection.
        
        Args:
            feature_data: Beat-synchronous features
            
        Returns:
            Similarity matrix
        """
        try:
            features = feature_data.get('features', np.array([]))
            
            if len(features) < 8:
                return np.array([]).reshape(0, 0)
            
            # Normalize features
            features_norm = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
            
            # Compute cosine similarity
            similarity_matrix = np.zeros((len(features), len(features)))
            
            for i in range(len(features)):
                for j in range(len(features)):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Cosine similarity
                        dot_product = np.dot(features_norm[i], features_norm[j])
                        norm_product = np.linalg.norm(features_norm[i]) * np.linalg.norm(features_norm[j])
                        
                        if norm_product > 0:
                            similarity_matrix[i, j] = dot_product / norm_product
                        else:
                            similarity_matrix[i, j] = 0.0
            
            return similarity_matrix
            
        except Exception as e:
            print(f"Similarity matrix computation failed: {str(e)}")
            return np.array([]).reshape(0, 0)
    
    def _detect_boundaries(self, similarity_matrix: np.ndarray, feature_data: Dict[str, Any], 
                          duration: float) -> List[float]:
        """
        Detect section boundaries from similarity matrix.
        
        Args:
            similarity_matrix: Self-similarity matrix
            feature_data: Beat-synchronous features
            duration: Audio duration in seconds
            
        Returns:
            List of boundary times in seconds
        """
        try:
            if similarity_matrix.size == 0:
                return []
            
            # Simple approach: find places where similarity drops significantly
            n = similarity_matrix.shape[0]
            boundaries = [0.0]  # Start of song
            
            if n < 4:
                boundaries.append(duration)  # End of song
                return boundaries
            
            # Calculate diagonal lag-1 similarity (similarity between adjacent beats)
            diag_similarity = np.zeros(n - 1)
            for i in range(n - 1):
                diag_similarity[i] = similarity_matrix[i, i + 1]
            
            # Find drops in similarity
            mean_similarity = np.mean(diag_similarity)
            std_similarity = np.std(diag_similarity)
            threshold = mean_similarity - 0.5 * std_similarity
            
            # Find peaks in the negative gradient (sudden drops)
            gradient = np.diff(diag_similarity)
            boundary_candidates = []
            
            for i in range(1, len(gradient) - 1):
                if gradient[i] < threshold - mean_similarity:
                    if gradient[i] < gradient[i - 1] and gradient[i] < gradient[i + 1]:
                        boundary_candidates.append(i)
            
            # Convert to times
            feature_times = feature_data.get('feature_times', [])
            for candidate in boundary_candidates:
                if candidate < len(feature_times):
                    boundaries.append(feature_times[candidate])
            
            # Add end of song
            boundaries.append(duration)
            
            # Remove duplicates and sort
            boundaries = sorted(list(set(boundaries)))
            
            # Ensure reasonable spacing (minimum 8 seconds between sections)
            min_duration = 8.0
            filtered_boundaries = [boundaries[0]]
            
            for boundary in boundaries[1:]:
                if boundary - filtered_boundaries[-1] >= min_duration:
                    filtered_boundaries.append(boundary)
            
            return filtered_boundaries
            
        except Exception as e:
            print(f"Boundary detection failed: {str(e)}")
            return [0.0, duration]
    
    def _segment_sections(self, audio_data: np.ndarray, boundaries: List[float], 
                         features: Dict[str, Any], beat_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Segment audio into sections based on boundaries.
        
        Args:
            audio_data: Audio signal
            boundaries: Section boundary times
            features: Beat-synchronous features
            beat_results: Beat detection results
            
        Returns:
            List of sections with characteristics
        """
        try:
            sections = []
            
            for i in range(len(boundaries) - 1):
                start_time = boundaries[i]
                end_time = boundaries[i + 1]
                
                # Convert to sample indices
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                
                if start_sample >= len(audio_data) or end_sample > len(audio_data):
                    continue
                
                # Extract section audio
                section_audio = audio_data[start_sample:end_sample]
                
                # Calculate section features
                section_features = self._calculate_section_features(section_audio, features, start_time)
                
                section = {
                    'section_id': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'audio': section_audio,
                    'features': section_features
                }
                
                sections.append(section)
            
            return sections
            
        except Exception as e:
            print(f"Section segmentation failed: {str(e)}")
            return []
    
    def _calculate_section_features(self, section_audio: np.ndarray, features: Dict[str, Any], 
                                  start_time: float) -> Dict[str, Any]:
        """
        Calculate features for a specific section.
        
        Args:
            section_audio: Audio for the section
            features: Beat-synchronous features
            start_time: Section start time
            
        Returns:
            Section features dictionary
        """
        try:
            section_features = {}
            
            # Energy statistics
            rms_values = np.sqrt(np.mean(section_audio ** 2))
            section_features['rms_energy'] = float(rms_values)
            
            # Spectral characteristics
            if len(section_audio) > self.hop_length:
                # Spectral centroid for brightness
                centroid = librosa.feature.spectral_centroid(
                    y=section_audio,
                    sr=self.sample_rate,
                    hop_length=self.hop_length
                )
                section_features['avg_brightness'] = float(np.mean(centroid))
                
                # Zero crossing rate for noisiness
                zcr = librosa.feature.zero_crossing_rate(
                    section_audio,
                    hop_length=self.hop_length
                )
                section_features['avg_zcr'] = float(np.mean(zcr))
            
            # Beat-synchronous features for this section
            feature_times = features.get('feature_times', [])
            feature_matrix = features.get('features', np.array([]))
            
            section_indices = []
            for i, feat_time in enumerate(feature_times):
                if start_time <= feat_time <= (start_time + section_features.get('duration', 10)):
                    section_indices.append(i)
            
            if section_indices and len(feature_matrix) > 0:
                section_feature_matrix = feature_matrix[section_indices]
                
                # Feature statistics
                for feat_idx in range(min(27, feature_matrix.shape[1])):
                    if section_feature_matrix.shape[0] > 0:
                        feat_values = section_feature_matrix[:, feat_idx]
                        section_features[f'feat_{feat_idx}_mean'] = float(np.mean(feat_values))
                        section_features[f'feat_{feat_idx}_std'] = float(np.std(feat_values))
            
            return section_features
            
        except Exception as e:
            print(f"Section feature calculation failed: {str(e)}")
            return {}
    
    def _analyze_section_characteristics(self, sections: List[Dict[str, Any]], 
                                       spectral_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze characteristics of detected sections.
        
        Args:
            sections: List of sections
            spectral_results: Spectral analysis results
            
        Returns:
            Section analysis results
        """
        try:
            analysis = {
                'energy_levels': [],
                'brightness_levels': [],
                'complexity_scores': [],
                'repetition_patterns': []
            }
            
            for i, section in enumerate(sections):
                features = section.get('features', {})
                
                # Energy level
                energy = features.get('rms_energy', 0.0)
                analysis['energy_levels'].append(energy)
                
                # Brightness level
                brightness = features.get('avg_brightness', 0.0)
                analysis['brightness_levels'].append(brightness)
                
                # Complexity score (based on feature variance)
                complexity = 0.0
                for key in features:
                    if 'std' in key:
                        complexity += abs(features[key])
                analysis['complexity_scores'].append(complexity)
            
            # Identify repetition patterns
            analysis['repetition_patterns'] = self._find_repetition_patterns(sections)
            
            return analysis
            
        except Exception as e:
            print(f"Section characteristic analysis failed: {str(e)}")
            return {
                'energy_levels': [],
                'brightness_levels': [],
                'complexity_scores': [],
                'repetition_patterns': []
            }
    
    def _find_repetition_patterns(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find repetition patterns in sections.
        
        Args:
            sections: List of sections
            
        Returns:
            List of repetition patterns
        """
        try:
            patterns = []
            
            if len(sections) < 2:
                return patterns
            
            # Simple similarity-based repetition detection
            for i in range(len(sections) - 1):
                for j in range(i + 1, len(sections)):
                    similarity = self._calculate_section_similarity(
                        sections[i], sections[j]
                    )
                    
                    if similarity > 0.7:  # High similarity threshold
                        patterns.append({
                            'section_1': i,
                            'section_2': j,
                            'similarity': similarity,
                            'is_repetition': True
                        })
            
            return patterns
            
        except Exception as e:
            print(f"Repetition pattern detection failed: {str(e)}")
            return []
    
    def _calculate_section_similarity(self, section1: Dict[str, Any], section2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two sections.
        
        Args:
            section1: First section
            section2: Second section
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            features1 = section1.get('features', {})
            features2 = section2.get('features', {})
            
            if not features1 or not features2:
                return 0.0
            
            # Extract comparable features
            common_features = []
            for key in features1:
                if key in features2 and '_mean' in key:
                    common_features.append(key)
            
            if not common_features:
                return 0.0
            
            # Calculate feature similarity
            similarities = []
            for feature in common_features:
                val1 = features1[feature]
                val2 = features2[feature]
                
                # Normalize difference
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    diff = abs(val1 - val2) / max_val
                    similarity = 1.0 - min(diff, 1.0)
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _label_sections(self, sections: List[Dict[str, Any]], 
                       section_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Label sections based on their characteristics.
        
        Args:
            sections: List of sections
            section_analysis: Section analysis results
            
        Returns:
            List of labeled sections
        """
        try:
            labeled_sections = []
            
            energy_levels = section_analysis.get('energy_levels', [])
            complexity_scores = section_analysis.get('complexity_scores', [])
            
            for i, section in enumerate(sections):
                labeled_section = section.copy()
                
                # Get section characteristics
                energy = energy_levels[i] if i < len(energy_levels) else 0.0
                complexity = complexity_scores[i] if i < len(complexity_scores) else 0.0
                duration = section.get('duration', 0.0)
                
                # Label based on characteristics
                label = self._classify_section(energy, complexity, duration, i, len(sections))
                labeled_section['label'] = label
                labeled_section['confidence'] = self._calculate_section_confidence(
                    energy, complexity, duration
                )
                
                labeled_sections.append(labeled_section)
            
            return labeled_sections
            
        except Exception as e:
            print(f"Section labeling failed: {str(e)}")
            return sections
    
    def _classify_section(self, energy: float, complexity: float, duration: float, 
                         index: int, total_sections: int) -> str:
        """
        Classify section type based on characteristics.
        
        Args:
            energy: Section energy level
            complexity: Section complexity score
            duration: Section duration
            index: Section index
            total_sections: Total number of sections
            
        Returns:
            Section label
        """
        try:
            # First section is typically intro
            if index == 0:
                return 'intro' if energy < 0.01 else 'verse'
            
            # Last section is typically outro
            if index == total_sections - 1:
                return 'outro'
            
            # High energy and complexity = chorus
            if energy > 0.02 and complexity > 1.0:
                return 'chorus'
            
            # Low energy and complexity = verse
            if energy < 0.015 and complexity < 1.5:
                return 'verse'
            
            # Medium energy, short duration = pre/post chorus
            if 0.015 <= energy <= 0.02 and duration < 20:
                return 'pre_chorus' if index < total_sections // 2 else 'post_chorus'
            
            # High complexity, medium energy = bridge
            if complexity > 2.0 and 0.01 <= energy <= 0.025:
                return 'bridge'
            
            # Default to verse
            return 'verse'
            
        except Exception:
            return 'unknown'
    
    def _calculate_section_confidence(self, energy: float, complexity: float, duration: float) -> float:
        """
        Calculate confidence for section classification.
        
        Args:
            energy: Section energy
            complexity: Section complexity
            duration: Section duration
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.5  # Base confidence
        
        # Duration confidence
        if 12 <= duration <= 60:  # Reasonable section duration
            confidence += 0.2
        elif duration < 4:  # Too short
            confidence -= 0.2
        
        # Energy confidence
        if 0.005 <= energy <= 0.05:  # Reasonable energy range
            confidence += 0.2
        elif energy < 0.001:  # Too quiet
            confidence -= 0.2
        
        # Complexity confidence
        if 0.5 <= complexity <= 5.0:  # Reasonable complexity
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _identify_structure_type(self, sections: List[Dict[str, Any]]) -> str:
        """
        Identify overall song structure type.
        
        Args:
            sections: List of labeled sections
            
        Returns:
            Structure type string
        """
        try:
            if not sections:
                return 'unknown'
            
            labels = [section.get('label', 'unknown') for section in sections]
            
            # Count section types
            structure_counts = {}
            for label in labels:
                structure_counts[label] = structure_counts.get(label, 0) + 1
            
            # Identify common patterns
            if 'intro' in structure_counts and 'outro' in structure_counts:
                if structure_counts.get('chorus', 0) > structure_counts.get('verse', 0):
                    return 'verse-chorus'
                else:
                    return 'strophic'
            
            if structure_counts.get('bridge', 0) > 0:
                return 'with-bridge'
            
            if structure_counts.get('verse', 0) > 3:
                return 'complex-verse'
            
            return 'simple'
            
        except Exception:
            return 'unknown'
    
    def _calculate_structure_complexity(self, sections: List[Dict[str, Any]]) -> float:
        """
        Calculate overall structure complexity score.
        
        Args:
            sections: List of labeled sections
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        try:
            if len(sections) < 2:
                return 0.0
            
            # Factors contributing to complexity
            complexity_factors = []
            
            # Number of sections
            num_sections = len(sections)
            complexity_factors.append(min(1.0, num_sections / 10.0))
            
            # Diversity of section types
            labels = [section.get('label', 'unknown') for section in sections]
            unique_labels = len(set(labels))
            complexity_factors.append(min(1.0, unique_labels / 5.0))
            
            # Section duration variance
            durations = [section.get('duration', 0) for section in sections]
            if len(durations) > 1:
                duration_variance = np.var(durations) / (np.mean(durations) + 1e-8)
                complexity_factors.append(min(1.0, duration_variance / 10.0))
            
            return np.mean(complexity_factors)
            
        except Exception:
            return 0.0
    
    def _calculate_section_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall confidence for section detection.
        
        Args:
            results: Section detection results
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []
        
        # Number of sections detected
        num_sections = len(results.get('sections', []))
        if 3 <= num_sections <= 15:  # Reasonable range
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(max(0.0, 1.0 - abs(num_sections - 9) / 9))
        
        # Similarity matrix quality
        similarity_matrix = results.get('similarity_matrix', np.array([]))
        if similarity_matrix.size > 0:
            # Check for clear patterns
            diagonal_values = np.diag(similarity_matrix)
            if np.all(diagonal_values == 1.0):  # Self-similarity
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.5)
        
        # Section labeling confidence
        sections = results.get('labeled_sections', [])
        if sections:
            avg_confidence = np.mean([s.get('confidence', 0.0) for s in sections])
            confidence_factors.append(avg_confidence)
        
        # Structure type identification
        structure_type = results.get('section_info', {}).get('structure_type', 'unknown')
        if structure_type != 'unknown':
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Calculate average confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.0


def detect_song_structure(audio_data: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]:
    """
    Convenience function to detect song structure from audio data.
    
    Args:
        audio_data: Audio signal array
        sample_rate: Audio sample rate
        
    Returns:
        Song structure detection results
    """
    detector = SectionDetector(sample_rate=sample_rate)
    results = detector.analyze(audio_data, {}, {})
    return results.get('section_info', {})


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=44100)
        
        # Detect sections
        detector = SectionDetector()
        results = detector.analyze(y, {}, {})
        
        section_info = results.get('section_info', {})
        print(f"Sections detected: {section_info.get('num_sections', 0)}")
        print(f"Structure type: {section_info.get('structure_type', 'unknown')}")
        print(f"Complexity score: {section_info.get('complexity_score', 0):.2f}")
        
        sections = section_info.get('sections', [])
        for section in sections:
            label = section.get('label', 'unknown')
            start = section.get('start_time', 0)
            end = section.get('end_time', 0)
            print(f"  {label}: {start:.1f}s - {end:.1f}s")
    else:
        print("Usage: python sections.py <audio_file>")
