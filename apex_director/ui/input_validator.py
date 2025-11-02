"""
APEX DIRECTOR Input Validation and Processing Module
Validates and processes user inputs for music video generation workflow
"""

import os
import re
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum

from ..core.config import get_config

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Represents the result of an input validation check.

    Attributes:
        is_valid: Whether the validation check passed.
        severity: The severity of the validation result.
        message: A message describing the validation result.
        field_name: The name of the field that was validated.
        suggested_fix: A suggested fix for the validation error.
        details: A dictionary of additional details about the validation result.
    """
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field_name: Optional[str] = None
    suggested_fix: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ProcessedInput:
    """Represents validated and processed input data.

    Attributes:
        project_name: The name of the project.
        audio_file: The path to the audio file.
        concept_description: A description of the concept.
        visual_style: The visual style to be used.
        duration_seconds: The duration of the video in seconds.
        output_resolution: The output resolution of the video.
        frame_rate: The frame rate of the video.
        generation_parameters: A dictionary of generation parameters.
        metadata: A dictionary of metadata.
        validation_notes: A list of validation notes.
        raw_input: The raw input data.
    """
    project_name: str
    audio_file: str
    concept_description: str
    visual_style: str
    duration_seconds: float
    output_resolution: Tuple[int, int]
    frame_rate: int
    generation_parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_notes: List[str] = field(default_factory=list)
    raw_input: Dict[str, Any] = field(default_factory=dict)


class InputValidator:
    """A comprehensive system for validating and processing user inputs.

    This class provides functionality for:
    - Validating project input data against a set of rules
    - Processing validated input data into a standardized format
    - Generating validation summaries and suggested fixes
    """
    
    def __init__(self):
        """Initializes the InputValidator."""
        self.config = get_config()
        self.validation_rules = self._initialize_validation_rules()
        self.supported_formats = self._get_supported_formats()
        
        logger.info("Input Validator initialized")
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initializes the validation rules and constraints.

        Returns:
            A dictionary of validation rules.
        """
        return {
            'project_name': {
                'min_length': 3,
                'max_length': 100,
                'pattern': r'^[a-zA-Z0-9\s\-_]+$',
                'forbidden_chars': ['<', '>', ':', '"', '|', '?', '*']
            },
            'audio_file': {
                'supported_extensions': ['.mp3', '.wav', '.flac', '.m4a', '.ogg'],
                'max_file_size': 500 * 1024 * 1024,  # 500MB
                'min_duration': 10.0,  # 10 seconds
                'max_duration': 3600.0  # 1 hour
            },
            'concept_description': {
                'min_length': 10,
                'max_length': 2000,
                'required_keywords': []  # Optional required keywords
            },
            'visual_style': {
                'valid_styles': [
                    'cinematic', 'anime', 'realistic', 'artistic', 'futuristic',
                    'vintage', 'minimalist', 'abstract', 'surreal', 'horror',
                    'fantasy', 'sci-fi', 'documentary', 'commercial'
                ]
            },
            'duration_seconds': {
                'min_value': 10.0,
                'max_value': 3600.0
            },
            'output_resolution': {
                'valid_resolutions': [
                    (1920, 1080), (1280, 720), (3840, 2160),
                    (1024, 1024), (512, 512), (768, 768)
                ],
                'default_resolution': (1920, 1080)
            },
            'frame_rate': {
                'valid_rates': [24, 25, 30, 50, 60],
                'default_rate': 30
            }
        }
    
    def _get_supported_formats(self) -> Dict[str, List[str]]:
        """Gets the supported file formats.

        Returns:
            A dictionary of supported file formats.
        """
        return {
            'audio': ['.mp3', '.wav', '.flac', '.m4a', '.ogg'],
            'image': ['.png', '.jpg', '.jpeg', '.webp'],
            'video': ['.mp4', '.mov', '.avi', '.mkv'],
            'subtitle': ['.srt', '.vtt', '.ass']
        }
    
    async def validate_project_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        """Validates the complete project input data.

        Args:
            input_data: The raw input data dictionary.

        Returns:
            A tuple containing a boolean indicating whether the input is valid, and a
            list of validation results.
        """
        results = []
        
        try:
            # Validate required fields
            required_fields = ['project_name', 'audio_file', 'concept_description']
            for field_name in required_fields:
                field_result = self._validate_field_exists(input_data, field_name)
                results.append(field_result)
            
            # Validate each field if present
            for field_name, value in input_data.items():
                if field_name in self.validation_rules:
                    field_result = await self._validate_field(field_name, value)
                    results.append(field_result)
            
            # Cross-field validation
            cross_field_results = self._validate_cross_field_dependencies(input_data)
            results.extend(cross_field_results)
            
            # Check overall validity
            is_valid = all(result.is_valid for result in results if result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
            
            logger.info(f"Validation completed: {len(results)} results, valid={is_valid}")
            return is_valid, results
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, [ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation system error: {str(e)}"
            )]
    
    def _validate_field_exists(self, data: Dict[str, Any], field_name: str) -> ValidationResult:
        """Check if required field exists and is not empty"""
        if field_name not in data or data[field_name] is None or str(data[field_name]).strip() == "":
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Required field '{field_name}' is missing or empty",
                field_name=field_name,
                suggested_fix=f"Please provide a value for {field_name}"
            )
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message=f"Field '{field_name}' present",
            field_name=field_name
        )
    
    async def _validate_field(self, field_name: str, value: Any) -> ValidationResult:
        """Validate individual field based on its type and rules"""
        rules = self.validation_rules.get(field_name, {})
        
        try:
            if field_name == 'project_name':
                return self._validate_project_name(str(value), rules)
            elif field_name == 'audio_file':
                return await self._validate_audio_file(value, rules)
            elif field_name == 'concept_description':
                return self._validate_concept_description(str(value), rules)
            elif field_name == 'visual_style':
                return self._validate_visual_style(str(value), rules)
            elif field_name == 'duration_seconds':
                return self._validate_duration(float(value), rules)
            elif field_name == 'output_resolution':
                return self._validate_resolution(value, rules)
            elif field_name == 'frame_rate':
                return self._validate_frame_rate(int(value), rules)
            else:
                # Unknown field, but that's okay for extensibility
                return ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Field '{field_name}' validation skipped (no rules defined)"
                )
                
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Error validating field '{field_name}': {str(e)}",
                field_name=field_name,
                details={'error': str(e), 'value': str(value)}
            )
    
    def _validate_project_name(self, name: str, rules: Dict[str, Any]) -> ValidationResult:
        """Validate project name"""
        # Check length
        if len(name) < rules.get('min_length', 3):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Project name too short (min {rules.get('min_length')} characters)",
                field_name='project_name',
                suggested_fix=f"Add at least {rules.get('min_length') - len(name)} more characters"
            )
        
        if len(name) > rules.get('max_length', 100):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Project name too long (max {rules.get('max_length')} characters)",
                field_name='project_name',
                suggested_fix=f"Reduce name to {rules.get('max_length')} characters or less"
            )
        
        # Check for forbidden characters
        forbidden = rules.get('forbidden_chars', [])
        found_forbidden = [char for char in forbidden if char in name]
        if found_forbidden:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Project name contains forbidden characters: {found_forbidden}",
                field_name='project_name',
                suggested_fix=f"Remove forbidden characters: {found_forbidden}"
            )
        
        # Check pattern match
        pattern = rules.get('pattern')
        if pattern and not re.match(pattern, name):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Project name should only contain letters, numbers, spaces, hyphens, and underscores",
                field_name='project_name',
                suggested_fix="Use only alphanumeric characters, spaces, hyphens, and underscores"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Project name is valid"
        )
    
    async def _validate_audio_file(self, file_path: str, rules: Dict[str, Any]) -> ValidationResult:
        """Validate audio file path and properties"""
        if not isinstance(file_path, str):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Audio file path must be a string",
                field_name='audio_file'
            )
        
        # Check file extension
        file_path_obj = Path(file_path)
        extension = file_path_obj.suffix.lower()
        supported_extensions = rules.get('supported_extensions', [])
        
        if extension not in supported_extensions:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Unsupported audio format: {extension}",
                field_name='audio_file',
                suggested_fix=f"Use one of the supported formats: {', '.join(supported_extensions)}"
            )
        
        # Check if file exists
        if not os.path.exists(file_path):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Audio file not found: {file_path}",
                field_name='audio_file',
                suggested_fix="Check that the file path is correct and the file exists"
            )
        
        # Check file size
        file_size = os.path.getsize(file_path)
        max_size = rules.get('max_file_size', 500 * 1024 * 1024)
        
        if file_size > max_size:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Audio file too large: {file_size / (1024*1024):.1f}MB (max {max_size / (1024*1024):.1f}MB)",
                field_name='audio_file',
                suggested_fix="Compress the audio file or use a shorter version"
            )
        
        # Note: Audio duration validation would require audio processing library
        # For now, we'll just add a warning note
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Audio file validated (duration check may be needed)",
            details={'file_size': file_size, 'format': extension}
        )
    
    def _validate_concept_description(self, description: str, rules: Dict[str, Any]) -> ValidationResult:
        """Validate concept description"""
        # Check length
        if len(description) < rules.get('min_length', 10):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Concept description too short (min {rules.get('min_length')} characters)",
                field_name='concept_description',
                suggested_fix=f"Add more detail to your concept description (at least {rules.get('min_length')} characters)"
            )
        
        if len(description) > rules.get('max_length', 2000):
            return ValidationResult(
                is_valid=True,  # Soft warning
                severity=ValidationSeverity.WARNING,
                message=f"Concept description is quite long ({len(description)} characters)",
                field_name='concept_description',
                suggested_fix="Consider shortening the description for better processing"
            )
        
        # Check for required keywords
        required_keywords = rules.get('required_keywords', [])
        missing_keywords = [kw for kw in required_keywords if kw.lower() not in description.lower()]
        
        if missing_keywords:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Missing recommended keywords: {missing_keywords}",
                field_name='concept_description',
                suggested_fix=f"Consider adding keywords related to: {', '.join(missing_keywords)}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Concept description is valid"
        )
    
    def _validate_visual_style(self, style: str, rules: Dict[str, Any]) -> ValidationResult:
        """Validate visual style selection"""
        valid_styles = rules.get('valid_styles', [])
        
        if style.lower() not in [s.lower() for s in valid_styles]:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid visual style: '{style}'",
                field_name='visual_style',
                suggested_fix=f"Choose from valid styles: {', '.join(valid_styles)}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Visual style is valid"
        )
    
    def _validate_duration(self, duration: float, rules: Dict[str, Any]) -> ValidationResult:
        """Validate video duration"""
        min_duration = rules.get('min_value', 10.0)
        max_duration = rules.get('max_value', 3600.0)
        
        if duration < min_duration:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Duration too short: {duration}s (min {min_duration}s)",
                field_name='duration_seconds',
                suggested_fix="Increase the duration or check the audio file"
            )
        
        if duration > max_duration:
            return ValidationResult(
                is_valid=True,  # Soft warning for long videos
                severity=ValidationSeverity.WARNING,
                message=f"Duration is quite long: {duration / 60:.1f} minutes",
                field_name='duration_seconds',
                suggested_fix="Consider creating a shorter video or breaking it into segments"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Duration is valid"
        )
    
    def _validate_resolution(self, resolution: Union[str, tuple, list], rules: Dict[str, Any]) -> ValidationResult:
        """Validate output resolution"""
        valid_resolutions = rules.get('valid_resolutions', [])
        default_resolution = rules.get('default_resolution', (1920, 1080))
        
        # Parse resolution input
        if isinstance(resolution, str):
            try:
                width, height = map(int, resolution.split('x'))
                resolution = (width, height)
            except:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid resolution format: '{resolution}' (use WIDTHxHEIGHT format)",
                    field_name='output_resolution',
                    suggested_fix=f"Use format like {default_resolution[0]}x{default_resolution[1]}"
                )
        elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            resolution = (int(resolution[0]), int(resolution[1]))
        else:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid resolution: {resolution}",
                field_name='output_resolution'
            )
        
        # Check if resolution is in valid list
        if resolution not in valid_resolutions:
            # Check if it's a valid aspect ratio
            width, height = resolution
            aspect_ratio = width / height if height > 0 else 0
            valid_aspects = [r[0]/r[1] for r in valid_resolutions]
            
            close_aspect = any(abs(aspect_ratio - valid_aspect) < 0.1 for valid_aspect in valid_aspects)
            
            if close_aspect:
                return ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message=f"Non-standard resolution {width}x{height}",
                    field_name='output_resolution',
                    suggested_fix=f"Consider using standard resolution: {default_resolution}"
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid aspect ratio: {width}x{height}",
                    field_name='output_resolution',
                    suggested_fix="Use standard 16:9, 4:3, or square resolutions"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Resolution is valid"
        )
    
    def _validate_frame_rate(self, frame_rate: int, rules: Dict[str, Any]) -> ValidationResult:
        """Validate frame rate"""
        valid_rates = rules.get('valid_rates', [])
        
        if frame_rate not in valid_rates:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid frame rate: {frame_rate}",
                field_name='frame_rate',
                suggested_fix=f"Use one of the valid rates: {', '.join(map(str, valid_rates))}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Frame rate is valid"
        )
    
    def _validate_cross_field_dependencies(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate cross-field dependencies and consistency"""
        results = []
        
        # Check duration vs audio file consistency
        if 'audio_file' in data and 'duration_seconds' in data:
            # This would normally check actual audio duration
            # For now, just add informational note
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Duration should match audio file length for best results",
                details={'note': 'Automated audio duration check recommended'}
            ))
        
        # Check output format compatibility
        if 'output_resolution' in data and 'frame_rate' in data:
            resolution = data['output_resolution']
            frame_rate = data['frame_rate']
            
            # High resolution + high frame rate warning
            if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
                width, height = resolution
                if (width >= 1920 and height >= 1080 and frame_rate > 30):
                    results.append(ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        message="High resolution and high frame rate may require significant processing time",
                        field_name='output_resolution'
                    ))
        
        return results
    
    async def process_validated_input(self, input_data: Dict[str, Any]) -> ProcessedInput:
        """Processes validated input data into a standardized format.

        Args:
            input_data: The raw input data (assumed to be validated).

        Returns:
            A ProcessedInput object with standardized data.
        """
        try:
            # Extract and standardize fields
            project_name = str(input_data.get('project_name', '')).strip()
            audio_file = str(input_data.get('audio_file', '')).strip()
            concept_description = str(input_data.get('concept_description', '')).strip()
            visual_style = str(input_data.get('visual_style', 'cinematic')).strip().lower()
            
            # Handle resolution
            resolution_input = input_data.get('output_resolution', (1920, 1080))
            if isinstance(resolution_input, str):
                width, height = map(int, resolution_input.split('x'))
                output_resolution = (width, height)
            else:
                output_resolution = tuple(resolution_input)
            
            # Handle frame rate
            frame_rate = int(input_data.get('frame_rate', 30))
            
            # Handle duration
            duration_seconds = float(input_data.get('duration_seconds', 0))
            
            # Build generation parameters
            generation_parameters = {
                'visual_style': visual_style,
                'output_resolution': output_resolution,
                'frame_rate': frame_rate,
                'quality_level': input_data.get('quality_level', 3),
                'prompt_strength': input_data.get('prompt_strength', 7.5),
                'steps': input_data.get('steps', 20),
                'seed': input_data.get('seed'),
                'batch_size': input_data.get('batch_size', 1)
            }
            
            # Build metadata
            metadata = {
                'project_id': str(uuid.uuid4()),
                'created_at': datetime.utcnow(),
                'input_source': 'user_interface',
                'validation_version': '1.0.0',
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
            # Add any additional metadata from input
            for key, value in input_data.items():
                if key not in ['project_name', 'audio_file', 'concept_description', 'visual_style', 
                              'duration_seconds', 'output_resolution', 'frame_rate']:
                    metadata[f'user_{key}'] = value
            
            processed_input = ProcessedInput(
                project_name=project_name,
                audio_file=audio_file,
                concept_description=concept_description,
                visual_style=visual_style,
                duration_seconds=duration_seconds,
                output_resolution=output_resolution,
                frame_rate=frame_rate,
                generation_parameters=generation_parameters,
                metadata=metadata,
                raw_input=input_data.copy()
            )
            
            logger.info(f"Input processed successfully for project: {project_name}")
            return processed_input
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            raise ValueError(f"Failed to process input data: {str(e)}")
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Gets a comprehensive summary of the validation results.

        Args:
            results: A list of validation results.

        Returns:
            A dictionary containing a summary of the validation results.
        """
        if not results:
            return {'summary': 'No validation results', 'is_valid': False}
        
        summary = {
            'total_checks': len(results),
            'is_valid': all(r.is_valid for r in results if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]),
            'error_count': len([r for r in results if r.severity == ValidationSeverity.ERROR]),
            'warning_count': len([r for r in results if r.severity == ValidationSeverity.WARNING]),
            'info_count': len([r for r in results if r.severity == ValidationSeverity.INFO]),
            'critical_count': len([r for r in results if r.severity == ValidationSeverity.CRITICAL])
        }
        
        # Group results by severity
        grouped_results = {}
        for severity in ValidationSeverity:
            grouped_results[severity.value] = [
                {'message': r.message, 'field': r.field_name, 'suggestion': r.suggested_fix}
                for r in results if r.severity == severity
            ]
        summary['results'] = grouped_results
        
        return summary
    
    def suggest_fixes(self, results: List[ValidationResult]) -> List[str]:
        """Extracts suggested fixes from a list of validation results.

        Args:
            results: A list of validation results.

        Returns:
            A list of suggested fixes.
        """
        suggestions = []
        
        for result in results:
            if not result.is_valid and result.suggested_fix:
                if result.field_name:
                    suggestions.append(f"{result.field_name}: {result.suggested_fix}")
                else:
                    suggestions.append(result.suggested_fix)
        
        return suggestions
