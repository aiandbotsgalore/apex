"""
Character Identity System
FaceID/IP-Adapter for character consistency across shots
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from PIL import Image
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class CharacterProfile:
    """Represents a complete character profile for consistency.

    Attributes:
        name: The name of the character.
        description: A text description of the character.
        reference_images: A list of paths to reference images for the
            character.
        face_encodings: A list of face encodings extracted from the
            reference images.
        facial_features: A dictionary of analyzed facial features.
        style_attributes: A dictionary of style attributes for the
            character.
        consistency_notes: Notes on maintaining consistency.
        generation_prompts: A list of prompts used to generate images of the
            character.
    """
    name: str
    description: str
    reference_images: List[Path]
    face_encodings: List[np.ndarray]
    facial_features: Dict[str, Any]
    style_attributes: Dict[str, str]
    consistency_notes: str
    generation_prompts: List[str]
    
    def __post_init__(self):
        if not isinstance(self.face_encodings, list):
            self.face_encodings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the CharacterProfile to a dictionary.

        Returns:
            A dictionary representation of the character profile.
        """
        return {
            "name": self.name,
            "description": self.description,
            "reference_images": [str(p) for p in self.reference_images],
            "face_encodings": [enc.tolist() for enc in self.face_encodings],
            "facial_features": self.facial_features,
            "style_attributes": self.style_attributes,
            "consistency_notes": self.consistency_notes,
            "generation_prompts": self.generation_prompts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterProfile':
        """Creates a CharacterProfile from a dictionary.

        Args:
            data: A dictionary containing character profile data.

        Returns:
            A CharacterProfile instance.
        """
        return cls(
            name=data["name"],
            description=data["description"],
            reference_images=[Path(p) for p in data["reference_images"]],
            face_encodings=[np.array(enc) for enc in data["face_encodings"]],
            facial_features=data["facial_features"],
            style_attributes=data["style_attributes"],
            consistency_notes=data["consistency_notes"],
            generation_prompts=data["generation_prompts"]
        )

@dataclass
class FaceMatch:
    """Represents the result of a face matching operation.

    Attributes:
        character_name: The name of the matched character.
        confidence: The confidence of the match (0.0 to 1.0).
        match_locations: A list of bounding boxes for the matched faces.
        similarity_score: The similarity score of the match.
    """
    character_name: str
    confidence: float
    match_locations: List[Tuple[int, int, int, int]]  # (top, right, bottom, left)
    similarity_score: float
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Checks if the match has a high confidence score.

        Args:
            threshold: The confidence threshold.

        Returns:
            True if the confidence is above the threshold, False otherwise.
        """
        return self.confidence >= threshold

class FaceRecognitionSystem:
    """A system for face recognition and matching."""
    
    def __init__(self):
        """Initializes the FaceRecognitionSystem."""
        self.face_detector_model = "hog"  # or "cnn" for better accuracy
        self.face_encoding_model = "large"  # or "small" for faster processing
        
    def extract_face_encodings(
        self, 
        image: Image.Image,
        face_locations: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[np.ndarray]:
        """Extracts face encodings from an image.

        Args:
            image: The image to process.
            face_locations: Optional pre-computed face locations.

        Returns:
            A list of face encodings.
        """
        try:
            # Convert PIL to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use face_recognition library
            if face_locations:
                encodings = face_recognition.face_encodings(
                    np.array(image), 
                    face_locations,
                    model=self.face_encoding_model
                )
            else:
                encodings = face_recognition.face_encodings(
                    np.array(image),
                    model=self.face_encoding_model
                )
            
            logger.info(f"Extracted {len(encodings)} face encodings")
            return encodings
            
        except Exception as e:
            logger.error(f"Failed to extract face encodings: {e}")
            return []
    
    def detect_faces(
        self, 
        image: Image.Image
    ) -> List[Tuple[int, int, int, int]]:
        """Detects faces in an image.

        Args:
            image: The image to process.

        Returns:
            A list of bounding boxes for the detected faces.
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            face_locations = face_recognition.face_locations(
                np.array(image),
                model=self.face_detector_model
            )
            
            logger.info(f"Detected {len(face_locations)} faces")
            return face_locations
            
        except Exception as e:
            logger.error(f"Failed to detect faces: {e}")
            return []
    
    def compare_faces(
        self,
        reference_encoding: np.ndarray,
        candidate_encoding: np.ndarray,
        tolerance: float = 0.6
    ) -> float:
        """Compares two face encodings and returns a similarity score.

        Args:
            reference_encoding: The reference face encoding.
            candidate_encoding: The candidate face encoding to compare.
            tolerance: The tolerance for the comparison.

        Returns:
            A similarity score between 0.0 and 1.0.
        """
        try:
            # Use face_recognition built-in comparison
            match = face_recognition.compare_faces(
                [reference_encoding], 
                candidate_encoding,
                tolerance=tolerance
            )[0]
            
            # Calculate distance for confidence score
            distance = face_recognition.face_distance(
                [reference_encoding], 
                candidate_encoding
            )[0]
            
            # Convert distance to confidence score (1 - distance, but normalized)
            confidence = max(0, 1 - distance * 2)  # Scale distance to 0-1 range
            
            return confidence if match else 0.0
            
        except Exception as e:
            logger.error(f"Failed to compare faces: {e}")
            return 0.0

class CharacterConsistencyManager:
    """Manages character profiles and consistency across image generations."""
    
    def __init__(self, storage_path: str = "character_profiles"):
        """Initializes the CharacterConsistencyManager.

        Args:
            storage_path: The path to store character profiles.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.characters: Dict[str, CharacterProfile] = {}
        self.face_recognition = FaceRecognitionSystem()
        self.load_all_characters()
    
    def load_all_characters(self):
        """Loads all character profiles from the storage directory."""
        for profile_file in self.storage_path.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                
                character = CharacterProfile.from_dict(data)
                self.characters[character.name] = character
                
                logger.info(f"Loaded character profile: {character.name}")
                
            except Exception as e:
                logger.error(f"Failed to load character {profile_file}: {e}")
    
    def save_character(self, character: CharacterProfile):
        """Saves a character profile to the storage directory.

        Args:
            character: The CharacterProfile to save.
        """
        try:
            profile_file = self.storage_path / f"{character.name.replace(' ', '_')}.json"
            
            with open(profile_file, 'w') as f:
                json.dump(character.to_dict(), f, indent=2)
            
            self.characters[character.name] = character
            logger.info(f"Saved character profile: {character.name}")
            
        except Exception as e:
            logger.error(f"Failed to save character {character.name}: {e}")
    
    def create_character_profile(
        self,
        name: str,
        reference_images: List[Path],
        description: str = "",
        style_attributes: Optional[Dict[str, str]] = None
    ) -> CharacterProfile:
        """Creates a new character profile from a set of reference images.

        Args:
            name: The name of the character.
            reference_images: A list of paths to reference images.
            description: A text description of the character.
            style_attributes: A dictionary of style attributes.

        Returns:
            The newly created CharacterProfile.
        """
        
        logger.info(f"Creating character profile for: {name}")
        
        # Extract face encodings from all reference images
        all_encodings = []
        for img_path in reference_images:
            try:
                image = Image.open(img_path)
                encodings = self.face_recognition.extract_face_encodings(image)
                all_encodings.extend(encodings)
            except Exception as e:
                logger.error(f"Failed to process reference image {img_path}: {e}")
        
        if not all_encodings:
            raise ValueError("No faces detected in reference images")
        
        # Create facial features dictionary
        facial_features = self._analyze_facial_features(all_encodings)
        
        # Create profile
        profile = CharacterProfile(
            name=name,
            description=description,
            reference_images=reference_images,
            face_encodings=all_encodings,
            facial_features=facial_features,
            style_attributes=style_attributes or {},
            consistency_notes="",
            generation_prompts=[]
        )
        
        self.save_character(profile)
        return profile
    
    def _analyze_facial_features(self, encodings: List[np.ndarray]) -> Dict[str, Any]:
        """Analyzes facial features from a list of face encodings.

        Note:
            This is currently a placeholder implementation.

        Args:
            encodings: A list of face encodings.

        Returns:
            A dictionary of analyzed facial features.
        """
        # Placeholder for detailed facial feature analysis
        # In reality, would extract specific features like eye color, skin tone, etc.
        
        return {
            "face_count": len(encodings),
            "encoding_quality": "high" if len(encodings) >= 3 else "medium",
            "feature_consistency": self._calculate_feature_consistency(encodings)
        }
    
    def _calculate_feature_consistency(self, encodings: List[np.ndarray]) -> float:
        """Calculates the consistency of facial features across reference
        images.

        Args:
            encodings: A list of face encodings.

        Returns:
            A consistency score between 0.0 and 1.0.
        """
        if len(encodings) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(encodings)):
            for j in range(i + 1, len(encodings)):
                similarity = cosine_similarity([encodings[i]], [encodings[j]])[0][0]
                similarities.append(similarity)
        
        return np.mean(similarities)
    
    def find_matching_characters(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.8
    ) -> List[FaceMatch]:
        """Finds matching characters in a generated image.

        Args:
            image: The image to search for characters in.
            confidence_threshold: The minimum confidence for a match.

        Returns:
            A list of FaceMatch objects for the detected characters.
        """
        
        logger.info("Searching for character matches in image")
        
        # Detect faces in the image
        face_locations = self.face_recognition.detect_faces(image)
        if not face_locations:
            logger.info("No faces detected in image")
            return []
        
        # Extract encodings for detected faces
        face_encodings = self.face_recognition.extract_face_encodings(
            image, face_locations
        )
        
        matches = []
        
        # Compare against each character's encodings
        for character_name, character in self.characters.items():
            for face_encoding in face_encodings:
                # Compare with character's reference encodings
                similarities = []
                for ref_encoding in character.face_encodings:
                    similarity = self.face_recognition.compare_faces(
                        ref_encoding, face_encoding
                    )
                    similarities.append(similarity)
                
                if similarities:
                    max_similarity = max(similarities)
                    if max_similarity >= confidence_threshold:
                        match = FaceMatch(
                            character_name=character_name,
                            confidence=max_similarity,
                            match_locations=face_locations,  # All detected faces
                            similarity_score=max_similarity
                        )
                        matches.append(match)
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Found {len(matches)} character matches")
        return matches
    
    def generate_character_prompt(
        self,
        character_name: str,
        additional_attributes: Optional[Dict[str, str]] = None
    ) -> str:
        """Generates a prompt with character consistency information.

        Args:
            character_name: The name of the character.
            additional_attributes: Optional additional attributes to include
                in the prompt.

        Returns:
            A string prompt for image generation.
        """
        
        if character_name not in self.characters:
            logger.warning(f"Character '{character_name}' not found")
            return ""
        
        character = self.characters[character_name]
        
        prompt_parts = []
        
        # Add character description
        if character.description:
            prompt_parts.append(character.description)
        
        # Add style attributes
        for key, value in character.style_attributes.items():
            prompt_parts.append(f"{key}: {value}")
        
        # Add consistency notes
        if character.consistency_notes:
            prompt_parts.append(character.consistency_notes)
        
        # Add generated prompts from previous sessions
        if character.generation_prompts:
            prompt_parts.extend(character.generation_prompts[-3:])  # Use last 3 prompts
        
        # Add additional attributes
        if additional_attributes:
            for key, value in additional_attributes.items():
                prompt_parts.append(f"{key}: {value}")
        
        character_prompt = ", ".join(prompt_parts)
        logger.info(f"Generated character prompt for {character_name}")
        
        return character_prompt
    
    def validate_character_consistency(
        self,
        generated_image: Image.Image,
        expected_character: str,
        confidence_threshold: float = 0.8
    ) -> Tuple[bool, float, List[str]]:
        """Validates that a generated image maintains character consistency.

        Args:
            generated_image: The image to validate.
            expected_character: The name of the expected character.
            confidence_threshold: The minimum confidence for a valid match.

        Returns:
            A tuple containing a boolean for consistency, the confidence
            score, and a list of any issues.
        """
        
        issues = []
        
        if expected_character not in self.characters:
            return False, 0.0, ["Character not found in database"]
        
        # Find matches in the generated image
        matches = self.find_matching_characters(
            generated_image, 
            confidence_threshold=0.6  # Lower threshold for validation
        )
        
        # Check for expected character
        character_matches = [
            match for match in matches 
            if match.character_name == expected_character
        ]
        
        if not character_matches:
            return False, 0.0, [f"Expected character '{expected_character}' not found"]
        
        # Get best match
        best_match = character_matches[0]
        
        if not best_match.is_high_confidence(confidence_threshold):
            issues.append(
                f"Character consistency confidence {best_match.confidence:.2f} "
                f"below threshold {confidence_threshold:.2f}"
            )
        
        is_consistent = len(issues) == 0 and best_match.confidence >= confidence_threshold
        
        return is_consistent, best_match.confidence, issues
    
    def get_character_statistics(self) -> Dict[str, Any]:
        """Gets statistics about the character database.

        Returns:
            A dictionary of statistics.
        """
        return {
            "total_characters": len(self.characters),
            "character_names": list(self.characters.keys()),
            "storage_path": str(self.storage_path),
            "avg_encodings_per_character": np.mean([
                len(char.face_encodings) for char in self.characters.values()
            ]) if self.characters else 0
        }
    
    def update_character_style(
        self,
        character_name: str,
        style_attributes: Dict[str, str],
        consistency_notes: str = ""
    ):
        """Updates a character's style attributes.

        Args:
            character_name: The name of the character to update.
            style_attributes: A dictionary of style attributes to update.
            consistency_notes: New consistency notes.
        """
        
        if character_name not in self.characters:
            logger.warning(f"Character '{character_name}' not found")
            return
        
        character = self.characters[character_name]
        character.style_attributes.update(style_attributes)
        character.consistency_notes = consistency_notes
        
        self.save_character(character)
        logger.info(f"Updated style attributes for character: {character_name}")
    
    def merge_character_profiles(
        self,
        primary_name: str,
        secondary_name: str,
        confidence_threshold: float = 0.9
    ) -> bool:
        """Merges two character profiles if they represent the same person.

        Args:
            primary_name: The name of the primary character profile.
            secondary_name: The name of the secondary character profile to
                merge.
            confidence_threshold: The minimum confidence for merging.

        Returns:
            True if the profiles were merged, False otherwise.
        """
        
        if primary_name not in self.characters or secondary_name not in self.characters:
            logger.error("One or both characters not found")
            return False
        
        primary = self.characters[primary_name]
        secondary = self.characters[secondary_name]
        
        # Compare face encodings to determine if they're the same person
        similarities = []
        for primary_encoding in primary.face_encodings:
            for secondary_encoding in secondary.face_encodings:
                similarity = self.face_recognition.compare_faces(
                    primary_encoding, secondary_encoding
                )
                similarities.append(similarity)
        
        if not similarities:
            logger.warning("No encodings to compare")
            return False
        
        max_similarity = max(similarities)
        
        if max_similarity >= confidence_threshold:
            # Merge profiles
            primary.reference_images.extend(secondary.reference_images)
            primary.face_encodings.extend(secondary.face_encodings)
            primary.style_attributes.update(secondary.style_attributes)
            
            # Combine descriptions
            if secondary.description:
                if primary.description:
                    primary.description += f"; {secondary.description}"
                else:
                    primary.description = secondary.description
            
            self.save_character(primary)
            
            # Remove secondary profile
            (self.storage_path / f"{secondary_name.replace(' ', '_')}.json").unlink()
            del self.characters[secondary_name]
            
            logger.info(f"Merged character profiles: {secondary_name} into {primary_name}")
            return True
        
        else:
            logger.info(f"Characters {primary_name} and {secondary_name} are likely different people")
            return False