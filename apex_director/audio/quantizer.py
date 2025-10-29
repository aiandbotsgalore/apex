"""
Timeline Quantization Engine
Convert beat grids to frame-accurate timings for video synchronization
"""

import numpy as np
import librosa
import warnings
import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
warnings.filterwarnings('ignore')


@dataclass
class FrameTiming:
    """Frame timing information"""
    frame_number: int
    time_seconds: float
    beat_time: Optional[float] = None
    beat_phase: Optional[float] = None
    section_id: Optional[str] = None
    confidence: float = 1.0


@dataclass
class BeatGrid:
    """Beat grid information"""
    bpm: float
    beat_times: List[float]
    beat_phases: List[float]  # Phase within beat (0.0 to 1.0)
    time_signature: str = "4/4"
    confidence: float = 1.0


class TimelineQuantizer:
    """
    Timeline quantization for frame-accurate synchronization.
    
    Features:
    - Beat grid to frame conversion (configurable fps)
    - Phase calculation for precise timing
    - Section-aware quantization
    - Error correction and smoothing
    - Multiple output formats
    """
    
    def __init__(self, fps: float = 24.0, beat_tolerance: float = 0.05):
        """
        Initialize timeline quantizer.
        
        Args:
            fps: Target frame rate for quantization
            beat_tolerance: Tolerance for beat alignment (seconds)
        """
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.beat_tolerance = beat_tolerance
        
        # Standard time signatures and their beat patterns
        self.time_signatures = {
            '4/4': {'beats_per_measure': 4, 'beat_subdivision': 4},
            '3/4': {'beats_per_measure': 3, 'beat_subdivision': 3},
            '2/4': {'beats_per_measure': 2, 'beat_subdivision': 2},
            '6/8': {'beats_per_measure': 6, 'beat_subdivision': 8},
            '12/8': {'beats_per_measure': 12, 'beat_subdivision': 8}
        }
        
    def quantize(self, beat_results: Dict[str, Any], spectral_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quantize timeline to frame-accurate timing.
        
        Args:
            beat_results: Results from beat detection
            spectral_results: Results from spectral analysis
            
        Returns:
            Dictionary containing quantized timeline information
        """
        try:
            results = {}
            
            # Extract beat grid information
            beat_grid = self._extract_beat_grid(beat_results)
            results['beat_grid'] = beat_grid
            
            # Generate frame timings
            frame_timings = self._generate_frame_timings(beat_grid, spectral_results)
            results['frame_timings'] = frame_timings
            
            # Calculate synchronization metrics
            sync_metrics = self._calculate_sync_metrics(beat_grid, frame_timings)
            results['sync_metrics'] = sync_metrics
            
            # Generate quantized features
            quantized_features = self._quantize_features(spectral_results, frame_timings)
            results['quantized_features'] = quantized_features
            
            # Calculate confidence score
            confidence = self._calculate_quantization_confidence(results)
            results['confidence_scores']['timeline_quantization'] = confidence
            
            # Add summary information
            results['quantization_info'] = {
                'target_fps': self.fps,
                'total_frames': len(frame_timings),
                'quantization_error': sync_metrics.get('average_error', 0.0),
                'beat_alignment_rate': sync_metrics.get('alignment_rate', 0.0),
                'sync_quality': self._assess_sync_quality(sync_metrics)
            }
            
            return results
            
        except Exception as e:
            print(f"Timeline quantization error: {str(e)}")
            return {
                'error': str(e),
                'confidence_scores': {'timeline_quantization': 0.0},
                'quantization_info': {
                    'target_fps': self.fps,
                    'total_frames': 0,
                    'quantization_error': 1.0,
                    'beat_alignment_rate': 0.0,
                    'sync_quality': 'poor'
                }
            }
    
    def _extract_beat_grid(self, beat_results: Dict[str, Any]) -> BeatGrid:
        """
        Extract and validate beat grid information.
        
        Args:
            beat_results: Beat detection results
            
        Returns:
            BeatGrid object
        """
        try:
            # Extract basic beat information
            bpm = beat_results.get('bpm', 120.0)
            beat_times = beat_results.get('beat_times', [])
            downbeat_times = beat_results.get('downbeat_times', [])
            time_signature = beat_results.get('time_signature', '4/4')
            confidence = beat_results.get('confidence_scores', {}).get('beat_detection', 0.0)
            
            if not beat_times:
                # Generate beat times from BPM if not available
                beat_times = self._generate_beat_times(bpm, beat_results.get('duration', 10.0))
            
            # Calculate beat phases (position within each beat)
            beat_phases = self._calculate_beat_phases(beat_times)
            
            return BeatGrid(
                bpm=bpm,
                beat_times=beat_times,
                beat_phases=beat_phases,
                time_signature=time_signature,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"Beat grid extraction failed: {str(e)}")
            # Return default beat grid
            return BeatGrid(
                bpm=120.0,
                beat_times=[],
                beat_phases=[],
                time_signature='4/4',
                confidence=0.0
            )
    
    def _generate_beat_times(self, bpm: float, duration: float) -> List[float]:
        """
        Generate beat times from BPM and duration.
        
        Args:
            bpm: Beats per minute
            duration: Total duration in seconds
            
        Returns:
            List of beat times
        """
        try:
            beat_interval = 60.0 / bpm
            num_beats = int(duration / beat_interval) + 1
            beat_times = [i * beat_interval for i in range(num_beats)]
            
            # Filter beats within duration
            beat_times = [t for t in beat_times if t <= duration]
            
            return beat_times
            
        except Exception:
            return []
    
    def _calculate_beat_phases(self, beat_times: List[float]) -> List[float]:
        """
        Calculate phase (position within beat) for each beat.
        
        Args:
            beat_times: List of beat times
            
        Returns:
            List of beat phases (0.0 to 1.0)
        """
        try:
            if len(beat_times) < 2:
                return [0.0] * len(beat_times)
            
            phases = []
            for i, beat_time in enumerate(beat_times):
                if i == 0:
                    # First beat
                    phases.append(0.0)
                else:
                    # Calculate phase based on time since last beat
                    prev_beat_time = beat_times[i - 1]
                    beat_interval = beat_time - prev_beat_time
                    
                    # For now, assume phase is 0.0 (regular beat)
                    # In a more sophisticated implementation, this could analyze
                    # the actual timing variations
                    phases.append(0.0)
            
            return phases
            
        except Exception:
            return [0.0] * len(beat_times)
    
    def _generate_frame_timings(self, beat_grid: BeatGrid, spectral_results: Dict[str, Any]) -> List[FrameTiming]:
        """
        Generate frame timings based on beat grid and target fps.
        
        Args:
            beat_grid: Beat grid information
            spectral_results: Spectral analysis results
            
        Returns:
            List of FrameTiming objects
        """
        try:
            frame_timings = []
            
            # Estimate total duration from beat grid or use spectral data
            duration = self._estimate_duration(beat_grid, spectral_results)
            
            if duration <= 0:
                print("Cannot estimate duration for timeline quantization")
                return frame_timings
            
            # Generate frames
            total_frames = int(duration * self.fps)
            
            for frame_num in range(total_frames):
                frame_time = frame_num * self.frame_duration
                
                # Find nearest beat
                beat_info = self._find_nearest_beat(frame_time, beat_grid)
                
                # Determine section if available
                section_id = self._find_section_at_time(frame_time, spectral_results)
                
                # Calculate confidence
                confidence = self._calculate_frame_confidence(frame_time, beat_info, beat_grid)
                
                frame_timing = FrameTiming(
                    frame_number=frame_num,
                    time_seconds=frame_time,
                    beat_time=beat_info['beat_time'],
                    beat_phase=beat_info['beat_phase'],
                    section_id=section_id,
                    confidence=confidence
                )
                
                frame_timings.append(frame_timing)
            
            return frame_timings
            
        except Exception as e:
            print(f"Frame timing generation failed: {str(e)}")
            return []
    
    def _estimate_duration(self, beat_grid: BeatGrid, spectral_results: Dict[str, Any]) -> float:
        """
        Estimate total duration for quantization.
        
        Args:
            beat_grid: Beat grid information
            spectral_results: Spectral analysis results
            
        Returns:
            Duration in seconds
        """
        try:
            # Try to get duration from various sources
            
            # From spectral results
            if 'spectral_info' in spectral_results:
                return spectral_results['spectral_info'].get('duration', 0.0)
            
            # From beat grid
            if beat_grid.beat_times:
                # Estimate duration from last beat with some padding
                last_beat = beat_grid.beat_times[-1]
                avg_beat_interval = np.mean(np.diff(beat_grid.beat_times))
                return last_beat + avg_beat_interval * 2
            
            # Default fallback
            return 30.0  # 30 seconds default
            
        except Exception:
            return 30.0
    
    def _find_nearest_beat(self, frame_time: float, beat_grid: BeatGrid) -> Dict[str, Any]:
        """
        Find nearest beat to a given time.
        
        Args:
            frame_time: Time in seconds
            beat_grid: Beat grid information
            
        Returns:
            Dictionary with beat information
        """
        try:
            if not beat_grid.beat_times:
                return {'beat_time': 0.0, 'beat_phase': 0.0, 'distance': 1.0}
            
            # Find nearest beat
            beat_times = np.array(beat_grid.beat_times)
            beat_phases = np.array(beat_grid.beat_phases)
            
            distances = np.abs(beat_times - frame_time)
            nearest_idx = np.argmin(distances)
            
            nearest_beat_time = beat_times[nearest_idx]
            nearest_beat_phase = beat_phases[nearest_idx]
            distance = distances[nearest_idx]
            
            return {
                'beat_time': float(nearest_beat_time),
                'beat_phase': float(nearest_beat_phase),
                'distance': float(distance),
                'beat_index': int(nearest_idx)
            }
            
        except Exception:
            return {'beat_time': 0.0, 'beat_phase': 0.0, 'distance': 1.0}
    
    def _find_section_at_time(self, time: float, spectral_results: Dict[str, Any]) -> Optional[str]:
        """
        Find section ID at a specific time.
        
        Args:
            time: Time in seconds
            spectral_results: Spectral analysis results
            
        Returns:
            Section ID or None
        """
        try:
            # Check if section information is available in spectral results
            if 'section_info' in spectral_results:
                sections = spectral_results['section_info'].get('sections', [])
                
                for section in sections:
                    start_time = section.get('start_time', 0)
                    end_time = section.get('end_time', 0)
                    
                    if start_time <= time < end_time:
                        return section.get('label', 'unknown')
            
            return None
            
        except Exception:
            return None
    
    def _calculate_frame_confidence(self, frame_time: float, beat_info: Dict[str, Any], 
                                  beat_grid: BeatGrid) -> float:
        """
        Calculate confidence for frame timing.
        
        Args:
            frame_time: Frame time
            beat_info: Beat information
            beat_grid: Beat grid
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            confidence = 1.0
            
            # Distance from nearest beat
            distance = beat_info.get('distance', 1.0)
            if distance > self.beat_tolerance:
                confidence -= (distance - self.beat_tolerance) * 2
            
            # Beat grid confidence
            confidence *= beat_grid.confidence
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    def _calculate_sync_metrics(self, beat_grid: BeatGrid, frame_timings: List[FrameTiming]) -> Dict[str, Any]:
        """
        Calculate synchronization metrics.
        
        Args:
            beat_grid: Beat grid information
            frame_timings: List of frame timings
            
        Returns:
            Dictionary of synchronization metrics
        """
        try:
            if not beat_grid.beat_times or not frame_timings:
                return {
                    'average_error': 1.0,
                    'max_error': 1.0,
                    'alignment_rate': 0.0,
                    'beat_coverage': 0.0
                }
            
            # Calculate alignment errors
            alignment_errors = []
            for frame_timing in frame_timings:
                if frame_timing.beat_time is not None:
                    error = abs(frame_timing.time_seconds - frame_timing.beat_time)
                    alignment_errors.append(error)
            
            if not alignment_errors:
                return {
                    'average_error': 1.0,
                    'max_error': 1.0,
                    'alignment_rate': 0.0,
                    'beat_coverage': 0.0
                }
            
            # Calculate metrics
            avg_error = np.mean(alignment_errors)
            max_error = np.max(alignment_errors)
            
            # Alignment rate: percentage of frames within tolerance
            aligned_frames = sum(1 for error in alignment_errors if error <= self.beat_tolerance)
            alignment_rate = aligned_frames / len(alignment_errors)
            
            # Beat coverage: percentage of beats that have nearby frames
            beat_coverage = len(beat_grid.beat_times) / len(frame_timings) if frame_timings else 0.0
            beat_coverage = min(1.0, beat_coverage)
            
            return {
                'average_error': float(avg_error),
                'max_error': float(max_error),
                'alignment_rate': float(alignment_rate),
                'beat_coverage': float(beat_coverage)
            }
            
        except Exception as e:
            print(f"Sync metrics calculation failed: {str(e)}")
            return {
                'average_error': 1.0,
                'max_error': 1.0,
                'alignment_rate': 0.0,
                'beat_coverage': 0.0
            }
    
    def _quantize_features(self, spectral_results: Dict[str, Any], frame_timings: List[FrameTiming]) -> Dict[str, Any]:
        """
        Quantize spectral features to frame rate.
        
        Args:
            spectral_results: Spectral analysis results
            frame_timings: List of frame timings
            
        Returns:
            Dictionary of quantized features
        """
        try:
            quantized_features = {
                'frame_brightness': [],
                'frame_energy': [],
                'frame_color_values': []
            }
            
            # Extract spectral features
            spectral_info = spectral_results.get('spectral_info', {})
            brightness_times = spectral_info.get('brightness_times', [])
            brightness_values = spectral_info.get('brightness_values', [])
            energy_times = spectral_info.get('energy_times', [])
            energy_values = spectral_info.get('energy_values', [])
            
            # Interpolate features to frame times
            for frame_timing in frame_timings:
                frame_time = frame_timing.time_seconds
                
                # Brightness interpolation
                brightness = self._interpolate_feature(frame_time, brightness_times, brightness_values)
                quantized_features['frame_brightness'].append(brightness)
                
                # Energy interpolation
                energy = self._interpolate_feature(frame_time, energy_times, energy_values)
                quantized_features['frame_energy'].append(energy)
                
                # Color values (simplified)
                color_values = self._extract_color_values(frame_time, spectral_results)
                quantized_features['frame_color_values'].append(color_values)
            
            return quantized_features
            
        except Exception as e:
            print(f"Feature quantization failed: {str(e)}")
            return {
                'frame_brightness': [],
                'frame_energy': [],
                'frame_color_values': []
            }
    
    def _interpolate_feature(self, query_time: float, feature_times: List[float], 
                           feature_values: List[float]) -> float:
        """
        Interpolate feature value at query time.
        
        Args:
            query_time: Time to interpolate at
            feature_times: Feature time points
            feature_values: Feature values
            
        Returns:
            Interpolated feature value
        """
        try:
            if not feature_times or not feature_values or len(feature_times) != len(feature_values):
                return 0.0
            
            if query_time <= feature_times[0]:
                return feature_values[0]
            elif query_time >= feature_times[-1]:
                return feature_values[-1]
            
            # Find bracketing time points
            for i in range(len(feature_times) - 1):
                if feature_times[i] <= query_time <= feature_times[i + 1]:
                    # Linear interpolation
                    t0, t1 = feature_times[i], feature_times[i + 1]
                    v0, v1 = feature_values[i], feature_values[i + 1]
                    
                    if t1 == t0:
                        return v0
                    
                    alpha = (query_time - t0) / (t1 - t0)
                    return v0 + alpha * (v1 - v0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _extract_color_values(self, time: float, spectral_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract color values for visualization at specific time.
        
        Args:
            time: Time in seconds
            spectral_results: Spectral analysis results
            
        Returns:
            Dictionary of color values
        """
        try:
            # Extract color features
            color_features = spectral_results.get('color_features', [])
            
            # Map spectral features to RGB values
            brightness = 0.5
            saturation = 0.5
            value = 0.5
            
            for feature in color_features:
                feature_name = feature.get('feature', '')
                feature_value = feature.get('value', 0.0)
                
                if feature_name == 'brightness':
                    brightness = min(1.0, feature_value / 4000.0)  # Normalize
                elif feature_name == 'energy':
                    saturation = min(1.0, feature_value * 10.0)
                elif feature_name == 'complexity':
                    value = min(1.0, feature_value / 2000.0)
            
            return {
                'brightness': brightness,
                'saturation': saturation,
                'value': value
            }
            
        except Exception:
            return {'brightness': 0.5, 'saturation': 0.5, 'value': 0.5}
    
    def _assess_sync_quality(self, sync_metrics: Dict[str, Any]) -> str:
        """
        Assess overall synchronization quality.
        
        Args:
            sync_metrics: Synchronization metrics
            
        Returns:
            Quality assessment string
        """
        try:
            alignment_rate = sync_metrics.get('alignment_rate', 0.0)
            average_error = sync_metrics.get('average_error', 1.0)
            
            if alignment_rate > 0.9 and average_error < 0.02:
                return 'excellent'
            elif alignment_rate > 0.7 and average_error < 0.05:
                return 'good'
            elif alignment_rate > 0.5 and average_error < 0.1:
                return 'fair'
            else:
                return 'poor'
                
        except Exception:
            return 'poor'
    
    def _calculate_quantization_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate confidence score for timeline quantization.
        
        Args:
            results: Quantization results
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []
        
        # Sync metrics
        sync_metrics = results.get('sync_metrics', {})
        alignment_rate = sync_metrics.get('alignment_rate', 0.0)
        if alignment_rate > 0:
            confidence_factors.append(alignment_rate)
        
        # Beat grid quality
        beat_grid = results.get('beat_grid', None)
        if beat_grid and beat_grid.confidence > 0:
            confidence_factors.append(beat_grid.confidence)
        
        # Frame coverage
        frame_timings = results.get('frame_timings', [])
        if frame_timings:
            avg_frame_confidence = np.mean([f.confidence for f in frame_timings])
            confidence_factors.append(avg_frame_confidence)
        
        # Feature quantization success
        quantized_features = results.get('quantized_features', {})
        if quantized_features:
            # Check if features were successfully quantized
            brightness_values = quantized_features.get('frame_brightness', [])
            if brightness_values and np.std(brightness_values) > 0:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)
        
        # Calculate average confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.0
    
    def export_timeline(self, frame_timings: List[FrameTiming], format_type: str = 'json') -> str:
        """
        Export timeline to specified format.
        
        Args:
            frame_timings: List of frame timings
            format_type: Export format ('json', 'csv', 'srt')
            
        Returns:
            Exported timeline string
        """
        try:
            if format_type == 'json':
                return self._export_json(frame_timings)
            elif format_type == 'csv':
                return self._export_csv(frame_timings)
            elif format_type == 'srt':
                return self._export_srt(frame_timings)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            print(f"Timeline export failed: {str(e)}")
            return ""
    
    def _export_json(self, frame_timings: List[FrameTiming]) -> str:
        """Export to JSON format."""
        import json
        
        timeline_data = []
        for frame_timing in frame_timings:
            timeline_data.append({
                'frame': frame_timing.frame_number,
                'time': frame_timing.time_seconds,
                'beat_time': frame_timing.beat_time,
                'beat_phase': frame_timing.beat_phase,
                'section': frame_timing.section_id,
                'confidence': frame_timing.confidence
            })
        
        return json.dumps(timeline_data, indent=2)
    
    def _export_csv(self, frame_timings: List[FrameTiming]) -> str:
        """Export to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['frame', 'time', 'beat_time', 'beat_phase', 'section', 'confidence'])
        
        # Data
        for frame_timing in frame_timings:
            writer.writerow([
                frame_timing.frame_number,
                f"{frame_timing.time_seconds:.6f}",
                f"{frame_timing.beat_time:.6f}" if frame_timing.beat_time else "",
                f"{frame_timing.beat_phase:.6f}" if frame_timing.beat_phase else "",
                frame_timing.section_id or "",
                f"{frame_timing.confidence:.3f}"
            ])
        
        return output.getvalue()
    
    def _export_srt(self, frame_timings: List[FrameTiming]) -> str:
        """Export to SRT subtitle format."""
        import datetime
        
        srt_content = []
        
        for frame_timing in frame_timings:
            if frame_timing.section_id:
                # Convert time to SRT format
                time_delta = datetime.timedelta(seconds=frame_timing.time_seconds)
                
                start_time = time_delta
                end_time = time_delta + datetime.timedelta(seconds=self.frame_duration)
                
                # Format as SRT time (HH:MM:SS,mmm)
                start_str = self._format_srt_time(start_time)
                end_str = self._format_srt_time(end_time)
                
                srt_entry = f"{frame_timing.frame_number + 1}\n"
                srt_entry += f"{start_str} --> {end_str}\n"
                srt_entry += f"Beat: {frame_timing.beat_phase:.2f}, Section: {frame_timing.section_id}\n\n"
                
                srt_content.append(srt_entry)
        
        return "\n".join(srt_content)
    
    def _format_srt_time(self, time_delta: datetime.timedelta) -> str:
        """Format timedelta as SRT time."""
        total_seconds = time_delta.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def quantize_timeline(beat_results: Dict[str, Any], fps: float = 24.0) -> Dict[str, Any]:
    """
    Convenience function to quantize timeline.
    
    Args:
        beat_results: Beat detection results
        fps: Target frame rate
        
    Returns:
        Quantized timeline results
    """
    quantizer = TimelineQuantizer(fps=fps)
    return quantizer.quantize(beat_results, {})


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Test with generated beat data
        fps = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
        
        quantizer = TimelineQuantizer(fps=fps)
        
        # Example beat results
        test_beat_results = {
            'bpm': 120.0,
            'beat_times': [0.0, 0.5, 1.0, 1.5, 2.0],
            'confidence_scores': {'beat_detection': 0.9}
        }
        
        results = quantizer.quantize(test_beat_results, {})
        print(f"Quantized {len(results.get('frame_timings', []))} frames at {fps} fps")
        print(f"Sync quality: {results.get('quantization_info', {}).get('sync_quality', 'unknown')}")
    else:
        print("Usage: python quantizer.py <fps>")
