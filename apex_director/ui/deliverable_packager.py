"""
APEX DIRECTOR Deliverable Packaging System
Packages and organizes final deliverables for music video projects
"""

import os
import json
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

from .treatment_generator import VisualTreatment
from .storyboard import Storyboard

logger = logging.getLogger(__name__)


class DeliverableType(Enum):
    """Types of deliverables"""
    FINAL_VIDEO = "final_video"
    PROXY_VIDEO = "proxy_video"
    AUDIO_TRACK = "audio_track"
    STORYBOARD_PDF = "storyboard_pdf"
    STORYBOARD_IMAGES = "storyboard_images"
    TREATMENT_DOCUMENT = "treatment_document"
    STYLE_BOARD = "style_board"
    THUMBNAILS = "thumbnails"
    RAW_FOOTAGE = "raw_footage"
    PROJECT_FILES = "project_files"
    ASSETS = "assets"
    DOCUMENTATION = "documentation"
    METADATA = "metadata"


class PackageFormat(Enum):
    """Package formats"""
    ZIP = "zip"
    FOLDER = "folder"
    TAR_GZ = "tar.gz"


@dataclass
class DeliverableFile:
    """Represents an individual deliverable file.

    Attributes:
        id: The unique identifier for the deliverable file.
        type: The type of deliverable.
        name: The name of the file.
        file_path: The path to the file.
        size_bytes: The size of the file in bytes.
        format: The format of the file.
        description: A description of the file.
        created_at: The timestamp when the file was created.
        checksum: The checksum of the file.
        metadata: A dictionary of additional metadata.
    """
    id: str
    type: DeliverableType
    name: str
    file_path: str
    size_bytes: int
    format: str
    description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the DeliverableFile to a dictionary.

        Returns:
            A dictionary representation of the DeliverableFile.
        """
        data = asdict(self)
        data['type'] = self.type.value
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class DeliverablePackage:
    """Represents a complete deliverable package.

    Attributes:
        id: The unique identifier for the package.
        project_name: The name of the project.
        version: The version of the package.
        deliverables: A list of deliverable files in the package.
        package_path: The path to the package.
        format: The format of the package.
        created_at: The timestamp when the package was created.
        total_size_bytes: The total size of the package in bytes.
        file_count: The number of files in the package.
        metadata: A dictionary of additional metadata.
        notes: Notes about the package.
    """
    id: str
    project_name: str
    version: str
    deliverables: List[DeliverableFile]
    package_path: str
    format: PackageFormat
    created_at: datetime = field(default_factory=datetime.utcnow)
    total_size_bytes: int = 0
    file_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the DeliverablePackage to a dictionary.

        Returns:
            A dictionary representation of the DeliverablePackage.
        """
        data = asdict(self)
        data['format'] = self.format.value
        data['deliverables'] = [deliverable.to_dict() for deliverable in self.deliverables]
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @property
    def size_mb(self) -> float:
        """Gets the package size in megabytes.

        Returns:
            The package size in megabytes.
        """
        return self.total_size_bytes / (1024 * 1024)
    
    @property
    def size_gb(self) -> float:
        """Gets the package size in gigabytes.

        Returns:
            The package size in gigabytes.
        """
        return self.total_size_bytes / (1024 * 1024 * 1024)


@dataclass
class PackageTemplate:
    """A template for creating standard deliverable packages.

    Attributes:
        name: The name of the template.
        description: A description of the template.
        deliverable_types: A list of deliverable types included in the template.
        file_structure: A dictionary defining the file structure of the package.
        naming_convention: The naming convention for the package.
        metadata_fields: A list of required metadata fields.
        required_files: A list of required files.
    """
    name: str
    description: str
    deliverable_types: List[DeliverableType]
    file_structure: Dict[str, Any]
    naming_convention: str
    metadata_fields: List[str]
    required_files: List[str]


class DeliverablePackager:
    """A comprehensive system for packaging and organizing deliverables.

    This class provides functionality for creating, managing, and validating
    deliverable packages for projects. It includes features such as:
    - Creating packages from templates
    - Adding custom files to packages
    - Generating package manifests and checksums
    - Creating package archives
    - Validating package integrity
    - Cleaning up package files
    """
    
    def __init__(self, base_output_dir: str = "deliverables"):
        """Initializes the DeliverablePackager.

        Args:
            base_output_dir: The base directory for outputting packages.
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.package_templates = self._initialize_templates()
        self.packages: Dict[str, DeliverablePackage] = {}
        
        # File type mappings
        self.file_extensions = {
            'video': ['.mp4', '.mov', '.avi', '.mkv', '.webm'],
            'audio': ['.mp3', '.wav', '.flac', '.m4a', '.ogg'],
            'image': ['.png', '.jpg', '.jpeg', '.webp', '.tiff'],
            'document': ['.pdf', '.doc', '.docx', '.txt'],
            'data': ['.json', '.csv', '.xml', '.yaml', '.yml']
        }
        
        logger.info(f"Deliverable Packager initialized with output directory: {self.base_output_dir}")
    
    def _initialize_templates(self) -> Dict[str, PackageTemplate]:
        """Initializes the standard package templates.

        Returns:
            A dictionary of standard package templates.
        """
        return {
            'client_delivery': PackageTemplate(
                name="Client Delivery Package",
                description="Complete package for client delivery",
                deliverable_types=[
                    DeliverableType.FINAL_VIDEO,
                    DeliverableType.PROXY_VIDEO,
                    DeliverableType.AUDIO_TRACK,
                    DeliverableType.STORYBOARD_PDF,
                    DeliverableType.DOCUMENTATION,
                    DeliverableType.METADATA
                ],
                file_structure={
                    "01_Final_Video/": {"pattern": "*.mp4"},
                    "02_Proxy_Video/": {"pattern": "*.mp4"},
                    "03_Audio/": {"pattern": "*.wav"},
                    "04_Storyboard/": {"pattern": "*.pdf"},
                    "05_Documentation/": {"pattern": "*"},
                    "06_Metadata/": {"pattern": "*.json"}
                },
                naming_convention="{project_name}_v{version}_{type}",
                metadata_fields=['project_name', 'version', 'client', 'delivery_date'],
                required_files=['final_video', 'proxy_video', 'storyboard']
            ),
            'archive_package': PackageTemplate(
                name="Archive Package",
                description="Complete archive for long-term storage",
                deliverable_types=[
                    DeliverableType.FINAL_VIDEO,
                    DeliverableType.RAW_FOOTAGE,
                    DeliverableType.PROJECT_FILES,
                    DeliverableType.ASSETS,
                    DeliverableType.STORYBOARD_IMAGES,
                    DeliverableType.TREATMENT_DOCUMENT,
                    DeliverableType.DOCUMENTATION,
                    DeliverableType.METADATA
                ],
                file_structure={
                    "01_Final_Video/": {"pattern": "*.mp4"},
                    "02_Raw_Footage/": {"pattern": "*"},
                    "03_Project_Files/": {"pattern": "*"},
                    "04_Assets/": {"pattern": "*"},
                    "05_Storyboards/": {"pattern": "*"},
                    "06_Treatment/": {"pattern": "*"},
                    "07_Documentation/": {"pattern": "*"},
                    "08_Metadata/": {"pattern": "*.json"}
                },
                naming_convention="{project_name}_archive_v{version}",
                metadata_fields=['project_name', 'version', 'archive_date', 'retention_period'],
                required_files=['final_video', 'raw_footage', 'project_files']
            ),
            'review_package': PackageTemplate(
                name="Review Package",
                description="Lightweight package for review and feedback",
                deliverable_types=[
                    DeliverableType.PROXY_VIDEO,
                    DeliverableType.STORYBOARD_PDF,
                    DeliverableType.TREATMENT_DOCUMENT,
                    DeliverableType.METADATA
                ],
                file_structure={
                    "01_Review_Video/": {"pattern": "*.mp4"},
                    "02_Storyboard/": {"pattern": "*.pdf"},
                    "03_Treatment/": {"pattern": "*.pdf"},
                    "04_Notes/": {"pattern": "*.md"}
                },
                naming_convention="{project_name}_review_v{version}",
                metadata_fields=['project_name', 'version', 'review_date'],
                required_files=['proxy_video', 'storyboard']
            )
        }
    
    async def create_package(self,
                           project_name: str,
                           treatment: Optional[VisualTreatment] = None,
                           storyboard: Optional[Storyboard] = None,
                           template_name: str = 'client_delivery',
                           custom_files: Optional[List[str]] = None,
                           output_name: Optional[str] = None) -> str:
        """Creates a deliverable package.

        Args:
            project_name: The name of the project.
            treatment: An optional visual treatment.
            storyboard: An optional storyboard.
            template_name: The name of the package template to use.
            custom_files: An optional list of additional file paths to include.
            output_name: An optional custom output package name.

        Returns:
            The ID of the newly created package.
        """
        try:
            logger.info(f"Creating package for project: {project_name}")
            
            if template_name not in self.package_templates:
                raise ValueError(f"Unknown template: {template_name}")
            
            template = self.package_templates[template_name]
            package_id = str(uuid.uuid4())
            
            # Generate package name
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_name:
                package_name = output_name
            else:
                package_name = template.naming_convention.format(
                    project_name=project_name.replace(' ', '_'),
                    version=version,
                    type='delivery'
                )
            
            # Create package directory
            package_dir = self.base_output_dir / package_name
            package_dir.mkdir(parents=True, exist_ok=True)
            
            # Create deliverable files
            deliverables = []
            total_size = 0
            
            # Process template deliverable types
            for deliverable_type in template.deliverable_types:
                files = await self._create_deliverable_files(
                    deliverable_type, project_name, treatment, storyboard, package_dir, template
                )
                for file_info in files:
                    deliverables.append(file_info)
                    total_size += file_info.size_bytes
            
            # Add custom files if provided
            if custom_files:
                for file_path in custom_files:
                    if os.path.exists(file_path):
                        file_info = self._create_custom_file_entry(file_path, package_dir)
                        if file_info:
                            deliverables.append(file_info)
                            total_size += file_info.size_bytes
            
            # Calculate total file count
            file_count = len(deliverables)
            
            # Create package metadata
            package = DeliverablePackage(
                id=package_id,
                project_name=project_name,
                version=version,
                deliverables=deliverables,
                package_path=str(package_dir),
                format=PackageFormat.FOLDER,
                total_size_bytes=total_size,
                file_count=file_count,
                metadata={
                    'template_name': template_name,
                    'treatment_id': treatment.id if treatment else None,
                    'storyboard_id': storyboard.id if storyboard else None,
                    'creation_info': {
                        'created_by': 'APEX DIRECTOR',
                        'created_at': datetime.utcnow().isoformat(),
                        'package_version': '1.0'
                    }
                }
            )
            
            # Save package manifest
            await self._save_package_manifest(package, package_dir)
            
            self.packages[package_id] = package
            
            logger.info(f"Package created: {package_name} ({file_count} files, {total_size / (1024*1024):.1f} MB)")
            return package_id
            
        except Exception as e:
            logger.error(f"Error creating package: {e}")
            raise ValueError(f"Failed to create deliverable package: {str(e)}")
    
    async def _create_deliverable_files(self,
                                      deliverable_type: DeliverableType,
                                      project_name: str,
                                      treatment: Optional[VisualTreatment],
                                      storyboard: Optional[Storyboard],
                                      package_dir: Path,
                                      template: PackageTemplate) -> List[DeliverableFile]:
        """Create files for a specific deliverable type"""
        files = []
        
        if deliverable_type == DeliverableType.TREATMENT_DOCUMENT:
            files.extend(await self._create_treatment_files(treatment, package_dir))
        elif deliverable_type == DeliverableType.STORYBOARD_PDF:
            files.extend(await self._create_storyboard_files(storyboard, package_dir))
        elif deliverable_type == DeliverableType.STORYBOARD_IMAGES:
            files.extend(await self._create_storyboard_images(storyboard, package_dir))
        elif deliverable_type == DeliverableType.METADATA:
            files.extend(await self._create_metadata_files(project_name, treatment, storyboard, package_dir))
        elif deliverable_type == DeliverableType.DOCUMENTATION:
            files.extend(await self._create_documentation_files(project_name, treatment, storyboard, package_dir))
        elif deliverable_type == DeliverableType.STYLE_BOARD:
            files.extend(await self._create_style_board_files(treatment, package_dir))
        elif deliverable_type == DeliverableType.THUMBNAILS:
            files.extend(await self._create_thumbnail_files(storyboard, package_dir))
        else:
            # For file types that need actual files (video, audio, etc.)
            # These would be created by the video assembly process
            files.extend(await self._create_placeholder_files(deliverable_type, project_name, package_dir))
        
        return files
    
    async def _create_treatment_files(self, 
                                    treatment: Optional[VisualTreatment], 
                                    package_dir: Path) -> List[DeliverableFile]:
        """Create treatment-related files"""
        files = []
        
        treatment_dir = package_dir / "06_Treatment"
        treatment_dir.mkdir(exist_ok=True)
        
        if treatment:
            # Save treatment as JSON
            treatment_file = treatment_dir / "visual_treatment.json"
            with open(treatment_file, 'w', encoding='utf-8') as f:
                json.dump(treatment.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Create treatment summary
            summary_file = treatment_dir / "treatment_summary.md"
            await self._create_treatment_summary(treatment, summary_file)
            
            # Create scene breakdown
            scenes_file = treatment_dir / "scene_breakdown.json"
            scenes_data = [scene.to_dict() for scene in treatment.scenes]
            with open(scenes_file, 'w', encoding='utf-8') as f:
                json.dump(scenes_data, f, indent=2, ensure_ascii=False)
            
            # Add files to list
            for file_path in [treatment_file, summary_file, scenes_file]:
                file_info = DeliverableFile(
                    id=str(uuid.uuid4()),
                    type=DeliverableType.TREATMENT_DOCUMENT,
                    name=file_path.name,
                    file_path=str(file_path),
                    size_bytes=file_path.stat().st_size,
                    format=file_path.suffix[1:],
                    description=f"Treatment document: {file_path.name}"
                )
                files.append(file_info)
        else:
            # Create placeholder if no treatment
            placeholder_file = treatment_dir / "treatment_not_available.txt"
            placeholder_file.write_text("Visual treatment not available for this package.")
            
            file_info = DeliverableFile(
                id=str(uuid.uuid4()),
                type=DeliverableType.TREATMENT_DOCUMENT,
                name="treatment_not_available.txt",
                file_path=str(placeholder_file),
                size_bytes=placeholder_file.stat().st_size,
                format="txt",
                description="Treatment document placeholder"
            )
            files.append(file_info)
        
        return files
    
    async def _create_storyboard_files(self, 
                                     storyboard: Optional[Storyboard], 
                                     package_dir: Path) -> List[DeliverableFile]:
        """Create storyboard files"""
        files = []
        
        storyboard_dir = package_dir / "04_Storyboard"
        storyboard_dir.mkdir(exist_ok=True)
        
        if storyboard:
            # Save storyboard as JSON
            storyboard_file = storyboard_dir / "storyboard.json"
            with open(storyboard_file, 'w', encoding='utf-8') as f:
                json.dump(storyboard.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Create PDF summary (simplified)
            pdf_summary_file = storyboard_dir / "storyboard_summary.txt"
            await self._create_storyboard_summary(storyboard, pdf_summary_file)
            
            # Create shot list
            shot_list_file = storyboard_dir / "shot_list.json"
            shot_data = []
            for scene_idx, scene in enumerate(storyboard.scenes):
                for shot in scene.shots:
                    shot_data.append({
                        'scene_number': scene_idx + 1,
                        'scene_type': scene.scene_type,
                        'shot_number': shot.shot_number,
                        'shot_type': shot.shot_type.value,
                        'description': shot.description,
                        'duration': shot.duration
                    })
            
            with open(shot_list_file, 'w', encoding='utf-8') as f:
                json.dump(shot_data, f, indent=2, ensure_ascii=False)
            
            for file_path in [storyboard_file, pdf_summary_file, shot_list_file]:
                file_info = DeliverableFile(
                    id=str(uuid.uuid4()),
                    type=DeliverableType.STORYBOARD_PDF,
                    name=file_path.name,
                    file_path=str(file_path),
                    size_bytes=file_path.stat().st_size,
                    format=file_path.suffix[1:],
                    description=f"Storyboard document: {file_path.name}"
                )
                files.append(file_info)
        else:
            # Create placeholder
            placeholder_file = storyboard_dir / "storyboard_not_available.txt"
            placeholder_file.write_text("Storyboard not available for this package.")
            
            file_info = DeliverableFile(
                id=str(uuid.uuid4()),
                type=DeliverableType.STORYBOARD_PDF,
                name="storyboard_not_available.txt",
                file_path=str(placeholder_file),
                size_bytes=placeholder_file.stat().st_size,
                format="txt",
                description="Storyboard document placeholder"
            )
            files.append(file_info)
        
        return files
    
    async def _create_storyboard_images(self, 
                                      storyboard: Optional[Storyboard], 
                                      package_dir: Path) -> List[DeliverableFile]:
        """Create storyboard image files"""
        # For now, create placeholder images
        images_dir = package_dir / "05_Storyboard_Images"
        images_dir.mkdir(exist_ok=True)
        
        placeholder_file = images_dir / "storyboard_images_not_available.txt"
        placeholder_file.write_text("Storyboard images would be generated here in a full implementation.")
        
        file_info = DeliverableFile(
            id=str(uuid.uuid4()),
            type=DeliverableType.STORYBOARD_IMAGES,
            name="storyboard_images_not_available.txt",
            file_path=str(placeholder_file),
            size_bytes=placeholder_file.stat().st_size,
            format="txt",
            description="Storyboard images placeholder"
        )
        
        return [file_info]
    
    async def _create_metadata_files(self, 
                                   project_name: str,
                                   treatment: Optional[VisualTreatment],
                                   storyboard: Optional[Storyboard],
                                   package_dir: Path) -> List[DeliverableFile]:
        """Create metadata files"""
        files = []
        
        metadata_dir = package_dir / "06_Metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Project metadata
        project_metadata = {
            'project_name': project_name,
            'created_at': datetime.utcnow().isoformat(),
            'package_version': '1.0',
            'generator': 'APEX DIRECTOR',
            'treatment': {
                'id': treatment.id if treatment else None,
                'type': treatment.treatment_type.value if treatment else None,
                'scenes': len(treatment.scenes) if treatment else 0
            },
            'storyboard': {
                'id': storyboard.id if storyboard else None,
                'scenes': len(storyboard.scenes) if storyboard else 0,
                'total_shots': sum(len(scene.shots) for scene in storyboard.scenes) if storyboard else 0
            }
        }
        
        metadata_file = metadata_dir / "project_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(project_metadata, f, indent=2, ensure_ascii=False)
        
        file_info = DeliverableFile(
            id=str(uuid.uuid4()),
            type=DeliverableType.METADATA,
            name="project_metadata.json",
            file_path=str(metadata_file),
            size_bytes=metadata_file.stat().st_size,
            format="json",
            description="Project metadata and information"
        )
        files.append(file_info)
        
        return files
    
    async def _create_documentation_files(self, 
                                        project_name: str,
                                        treatment: Optional[VisualTreatment],
                                        storyboard: Optional[Storyboard],
                                        package_dir: Path) -> List[DeliverableFile]:
        """Create documentation files"""
        files = []
        
        docs_dir = package_dir / "05_Documentation"
        docs_dir.mkdir(exist_ok=True)
        
        # Readme file
        readme_file = docs_dir / "README.md"
        readme_content = f"""# {project_name} - Music Video Project

## Overview
This package contains the deliverables for the music video project "{project_name}".

## Contents
- Final video
- Proxy video for review
- Audio track
- Storyboard
- Treatment document
- Project metadata

## Technical Specifications
{f"Treatment Type: {treatment.treatment_type.value}" if treatment else "Treatment: Not available"}
{f"Total Scenes: {len(treatment.scenes)}" if treatment else "Scenes: Not available"}
{f"Storyboard Scenes: {len(storyboard.scenes)}" if storyboard else "Storyboard: Not available"}

## Created by APEX DIRECTOR
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_file.write_text(readme_content)
        
        file_info = DeliverableFile(
            id=str(uuid.uuid4()),
            type=DeliverableType.DOCUMENTATION,
            name="README.md",
            file_path=str(readme_file),
            size_bytes=readme_file.stat().st_size,
            format="md",
            description="Project README and overview"
        )
        files.append(file_info)
        
        return files
    
    async def _create_style_board_files(self, 
                                      treatment: Optional[VisualTreatment], 
                                      package_dir: Path) -> List[DeliverableFile]:
        """Create style board files"""
        if not treatment:
            return []
        
        files = []
        
        style_dir = package_dir / "05_Style_Board"
        style_dir.mkdir(exist_ok=True)
        
        # Color palette file
        colors_file = style_dir / "color_palette.json"
        color_data = {
            'primary_colors': treatment.color_scheme.get('primary', []),
            'secondary_colors': treatment.color_scheme.get('secondary', []),
            'accent_colors': treatment.color_scheme.get('accent', []),
            'scheme_name': treatment.color_scheme.get('scheme_name', 'Custom'),
            'description': treatment.color_scheme.get('description', '')
        }
        
        with open(colors_file, 'w', encoding='utf-8') as f:
            json.dump(color_data, f, indent=2, ensure_ascii=False)
        
        # Style keywords file
        style_file = style_dir / "style_keywords.json"
        style_data = {
            'visual_style': treatment.style_keywords,
            'treatment_type': treatment.treatment_type.value,
            'mood_templates': [scene.mood for scene in treatment.scenes],
            'lighting_styles': list(set(scene.lighting_style for scene in treatment.scenes))
        }
        
        with open(style_file, 'w', encoding='utf-8') as f:
            json.dump(style_data, f, indent=2, ensure_ascii=False)
        
        for file_path in [colors_file, style_file]:
            file_info = DeliverableFile(
                id=str(uuid.uuid4()),
                type=DeliverableType.STYLE_BOARD,
                name=file_path.name,
                file_path=str(file_path),
                size_bytes=file_path.stat().st_size,
                format="json",
                description=f"Style board: {file_path.name}"
            )
            files.append(file_info)
        
        return files
    
    async def _create_thumbnail_files(self, 
                                    storyboard: Optional[Storyboard], 
                                    package_dir: Path) -> List[DeliverableFile]:
        """Create thumbnail files"""
        # Create placeholder thumbnails
        thumbs_dir = package_dir / "07_Thumbnails"
        thumbs_dir.mkdir(exist_ok=True)
        
        placeholder_file = thumbs_dir / "thumbnails_not_available.txt"
        placeholder_file.write_text("Video thumbnails would be extracted here in a full implementation.")
        
        file_info = DeliverableFile(
            id=str(uuid.uuid4()),
            type=DeliverableType.THUMBNAILS,
            name="thumbnails_not_available.txt",
            file_path=str(placeholder_file),
            size_bytes=placeholder_file.stat().st_size,
            format="txt",
            description="Video thumbnails placeholder"
        )
        
        return [file_info]
    
    async def _create_placeholder_files(self, 
                                      deliverable_type: DeliverableType, 
                                      project_name: str, 
                                      package_dir: Path) -> List[DeliverableFile]:
        """Create placeholder files for file types that need actual content"""
        files = []
        
        # Determine directory and extension based on type
        type_dirs = {
            DeliverableType.FINAL_VIDEO: ("01_Final_Video", "mp4"),
            DeliverableType.PROXY_VIDEO: ("02_Proxy_Video", "mp4"),
            DeliverableType.AUDIO_TRACK: ("03_Audio", "wav"),
            DeliverableType.RAW_FOOTAGE: ("02_Raw_Footage", "mov"),
            DeliverableType.PROJECT_FILES: ("03_Project_Files", "zip"),
            DeliverableType.ASSETS: ("04_Assets", "zip")
        }
        
        if deliverable_type in type_dirs:
            dir_name, extension = type_dirs[deliverable_type]
            output_dir = package_dir / dir_name
            output_dir.mkdir(exist_ok=True)
            
            placeholder_file = output_dir / f"{project_name}_{deliverable_type.value}.{extension}"
            placeholder_file.write_text(f"Placeholder for {deliverable_type.value} - actual file would be generated here")
            
            file_info = DeliverableFile(
                id=str(uuid.uuid4()),
                type=deliverable_type,
                name=placeholder_file.name,
                file_path=str(placeholder_file),
                size_bytes=placeholder_file.stat().st_size,
                format=extension,
                description=f"Placeholder file for {deliverable_type.value}"
            )
            files.append(file_info)
        
        return files
    
    def _create_custom_file_entry(self, file_path: str, package_dir: Path) -> Optional[DeliverableFile]:
        """Create deliverable entry for custom file"""
        if not os.path.exists(file_path):
            return None
        
        file_path_obj = Path(file_path)
        file_size = file_path_obj.stat().st_size
        file_extension = file_path_obj.suffix[1:] if file_path_obj.suffix else 'unknown'
        
        return DeliverableFile(
            id=str(uuid.uuid4()),
            type=DeliverableType.ASSETS,  # Default to assets
            name=file_path_obj.name,
            file_path=str(file_path),
            size_bytes=file_size,
            format=file_extension,
            description=f"Custom file: {file_path_obj.name}"
        )
    
    async def _save_package_manifest(self, package: DeliverablePackage, package_dir: Path):
        """Save package manifest file"""
        manifest_file = package_dir / "package_manifest.json"
        
        manifest_data = {
            'package_info': package.to_dict(),
            'file_structure': self._generate_file_structure(package_dir),
            'checksums': await self._generate_checksums(package_dir),
            'validation_info': {
                'created_by': 'APEX DIRECTOR',
                'created_at': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
        }
        
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    
    def _generate_file_structure(self, package_dir: Path) -> Dict[str, Any]:
        """Generate file structure representation"""
        structure = {}
        
        for item in package_dir.rglob('*'):
            if item.is_file():
                relative_path = item.relative_to(package_dir)
                structure[str(relative_path)] = {
                    'size': item.stat().st_size,
                    'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                }
        
        return structure
    
    async def _generate_checksums(self, package_dir: Path) -> Dict[str, str]:
        """Generate checksums for all files in package"""
        import hashlib
        
        checksums = {}
        
        for item in package_dir.rglob('*'):
            if item.is_file() and item.name != 'package_manifest.json':
                try:
                    hasher = hashlib.md5()
                    with open(item, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
                    checksums[str(item.relative_to(package_dir))] = hasher.hexdigest()
                except Exception as e:
                    logger.warning(f"Could not generate checksum for {item}: {e}")
        
        return checksums
    
    async def _create_treatment_summary(self, treatment: VisualTreatment, output_file: Path):
        """Create treatment summary markdown file"""
        summary = f"""# Visual Treatment Summary

## Project: {treatment.project_name}

## Overview
{treatment.overall_concept}

## Treatment Details
- **Type**: {treatment.treatment_type.value.replace('_', ' ').title()}
- **Visual Complexity**: {treatment.visual_complexity.value.title()}
- **Duration**: {treatment.audio_duration:.1f} seconds
- **Scenes**: {len(treatment.scenes)}

## Scene Breakdown
"""
        
        for i, scene in enumerate(treatment.scenes, 1):
            summary += f"""
### Scene {i}: {scene.scene_type.title()}
- **Duration**: {scene.duration:.1f}s
- **Mood**: {scene.mood}
- **Description**: {scene.description}
- **Color Palette**: {', '.join(scene.color_palette)}
- **Camera Movement**: {scene.camera_movement}
"""
        
        summary += f"""
## Style Keywords
{', '.join(treatment.style_keywords)}

## Color Scheme
- **Scheme**: {treatment.color_scheme.get('scheme_name', 'Custom')}
- **Primary**: {', '.join(treatment.color_scheme.get('primary', []))}
- **Secondary**: {', '.join(treatment.color_scheme.get('secondary', []))}
- **Accent**: {', '.join(treatment.color_scheme.get('accent', []))}

## Technical Specifications
- **Resolution**: {treatment.technical_specs.get('output_resolution', 'Unknown')}
- **Frame Rate**: {treatment.technical_specs.get('frame_rate', 'Unknown')} fps
- **Codec**: {treatment.technical_specs.get('codec', 'Unknown')}

## Notes
{treatment.notes}
"""
        
        output_file.write_text(summary)
    
    async def _create_storyboard_summary(self, storyboard: Storyboard, output_file: Path):
        """Create storyboard summary file"""
        summary = f"""# Storyboard Summary

## Project: {storyboard.project_name}

## Overview
- **Total Duration**: {storyboard.total_duration:.1f} seconds
- **Total Scenes**: {len(storyboard.scenes)}
- **Total Shots**: {sum(len(scene.shots) for scene in storyboard.scenes)}

## Scene Breakdown
"""
        
        total_shots = 0
        for i, scene in enumerate(storyboard.scenes, 1):
            total_shots += len(scene.shots)
            summary += f"""
### Scene {i}: {scene.scene_type.title()}
- **Duration**: {scene.duration:.1f}s
- **Shots**: {len(scene.shots)}
- **Overview**: {scene.scene_overview}
"""
        
        summary += f"""
## Shot Statistics
- **Average shots per scene**: {total_shots / len(storyboard.scenes):.1f}
- **Scene types**: {', '.join(set(scene.scene_type for scene in storyboard.scenes))}

## Production Notes
{storyboard.notes if hasattr(storyboard, 'notes') and storyboard.notes else 'No additional production notes.'}
"""
        
        output_file.write_text(summary)
    
    def create_package_archive(self, package_id: str, format: PackageFormat = PackageFormat.ZIP) -> bool:
        """Creates an archive of a package.

        Args:
            package_id: The ID of the package to archive.
            format: The format of the archive.

        Returns:
            True if the archive was successfully created, False otherwise.
        """
        package = self.packages.get(package_id)
        if not package:
            logger.error(f"Package {package_id} not found")
            return False
        
        package_path = Path(package.package_path)
        
        try:
            if format == PackageFormat.ZIP:
                archive_path = package_path.parent / f"{package_path.name}.zip"
                
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in package_path.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(package_path)
                            zipf.write(file_path, arcname)
                
                # Remove original folder
                shutil.rmtree(package_path)
                
                # Update package path
                package.package_path = str(archive_path)
                package.format = PackageFormat.ZIP
                
                logger.info(f"Created ZIP archive: {archive_path}")
                return True
            
            else:
                logger.warning(f"Archive format {format} not yet implemented")
                return False
                
        except Exception as e:
            logger.error(f"Error creating archive: {e}")
            return False
    
    def get_package_info(self, package_id: str) -> Optional[Dict[str, Any]]:
        """Gets information about a package.

        Args:
            package_id: The ID of the package.

        Returns:
            A dictionary of package information, or None if the package is not found.
        """
        package = self.packages.get(package_id)
        if not package:
            return None
        
        return {
            'id': package.id,
            'project_name': package.project_name,
            'version': package.version,
            'format': package.format.value,
            'file_count': package.file_count,
            'total_size_bytes': package.total_size_bytes,
            'total_size_mb': f"{package.size_mb:.1f} MB",
            'total_size_gb': f"{package.size_gb:.2f} GB",
            'package_path': package.package_path,
            'deliverable_types': [d.type.value for d in package.deliverables],
            'created_at': package.created_at.isoformat(),
            'metadata': package.metadata
        }
    
    def list_packages(self) -> List[Dict[str, Any]]:
        """Lists all packages.

        Returns:
            A list of package information dictionaries.
        """
        return [self.get_package_info(package_id) for package_id in self.packages.keys()]
    
    def validate_package(self, package_id: str) -> Dict[str, Any]]:
        """Validates the integrity of a package.

        Args:
            package_id: The ID of the package to validate.

        Returns:
            A dictionary of validation results.
        """
        package = self.packages.get(package_id)
        if not package:
            return {'valid': False, 'error': 'Package not found'}
        
        package_path = Path(package.package_path)
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_count': 0,
            'total_size': 0
        }
        
        # Check if package exists
        if not package_path.exists():
            validation_results['valid'] = False
            validation_results['errors'].append("Package path does not exist")
            return validation_results
        
        # Check files
        for deliverable in package.deliverables:
            if not os.path.exists(deliverable.file_path):
                validation_results['errors'].append(f"Missing file: {deliverable.name}")
                validation_results['valid'] = False
            else:
                validation_results['file_count'] += 1
                validation_results['total_size'] += deliverable.size_bytes
        
        # Check package manifest
        manifest_path = package_path / "package_manifest.json"
        if not manifest_path.exists():
            validation_results['warnings'].append("Package manifest missing")
        else:
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    if 'validation_info' not in manifest:
                        validation_results['warnings'].append("Invalid package manifest")
            except Exception as e:
                validation_results['errors'].append(f"Failed to read package manifest: {e}")
                validation_results['valid'] = False
        
        return validation_results
    
    def cleanup_package(self, package_id: str, archive_first: bool = True) -> bool:
        """Cleans up the files for a package.

        Args:
            package_id: The ID of the package to clean up.
            archive_first: Whether to archive the package before cleaning up.

        Returns:
            True if the package was successfully cleaned up, False otherwise.
        """
        package = self.packages.get(package_id)
        if not package:
            logger.error(f"Package {package_id} not found")
            return False
        
        try:
            # Create archive if requested and not already archived
            if archive_first and package.format != PackageFormat.ZIP:
                self.create_package_archive(package_id)
            
            # Remove package files
            package_path = Path(package.package_path)
            if package_path.exists():
                if package_path.is_file():  # Archive file
                    package_path.unlink()
                else:  # Folder
                    shutil.rmtree(package_path)
            
            # Remove from registry
            del self.packages[package_id]
            
            logger.info(f"Cleaned up package {package_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up package {package_id}: {e}")
            return False
