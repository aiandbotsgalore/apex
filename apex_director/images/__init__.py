"""
Cinematic Image Generation Pipeline
Professional image generation system with multi-backend support
"""

from .generator import (
    CinematicImageGenerator,
    GenerationRequest,
    GenerationResult,
    quick_generate,
    batch_generate
)

from .backend_interface import (
    BackendManager,
    GenerationConfig,
    BackendInterface,
    NanoBananaBackend,
    ImagenBackend,
    MiniMaxBackend,
    SDXLBackend
)

from .prompt_engineer import (
    CinematographyPromptEngineer,
    CameraSettings,
    LightingSetup,
    Composition
)

from .style_persistence import (
    StyleBibleManager,
    StyleConsistencyValidator,
    StyleElement,
    CharacterReference,
    SceneStyle
)

from .character_system import (
    CharacterConsistencyManager,
    CharacterProfile,
    FaceMatch,
    FaceRecognitionSystem
)

from .variant_selector import (
    VariantSelector,
    QualityScores,
    VariantResult,
    CLIPScorer,
    CompositionAnalyzer,
    StyleConsistencyScorer,
    ArtifactDetector
)

from .upscaller import (
    ProfessionalUpscaler,
    UpscaleSettings,
    UpscaleResult,
    RealESRGANUpscaler,
    TileBasedUpscaler
)

__version__ = "1.0.0"
__author__ = "Apex Director Team"

__all__ = [
    # Main generator
    "CinematicImageGenerator",
    "GenerationRequest", 
    "GenerationResult",
    "quick_generate",
    "batch_generate",
    
    # Backend interface
    "BackendManager",
    "GenerationConfig",
    "BackendInterface",
    "NanoBananaBackend",
    "ImagenBackend", 
    "MiniMaxBackend",
    "SDXLBackend",
    
    # Prompt engineering
    "CinematographyPromptEngineer",
    "CameraSettings",
    "LightingSetup",
    "Composition",
    
    # Style persistence
    "StyleBibleManager",
    "StyleConsistencyValidator",
    "StyleElement",
    "CharacterReference",
    "SceneStyle",
    
    # Character system
    "CharacterConsistencyManager",
    "CharacterProfile",
    "FaceMatch",
    "FaceRecognitionSystem",
    
    # Variant selection
    "VariantSelector",
    "QualityScores",
    "VariantResult",
    "CLIPScorer",
    "CompositionAnalyzer",
    "StyleConsistencyScorer",
    "ArtifactDetector",
    
    # Upscaling
    "ProfessionalUpscaler",
    "UpscaleSettings",
    "UpscaleResult",
    "RealESRGANUpscaler",
    "TileBasedUpscaler"
]