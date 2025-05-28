#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Rhetoric Analyzer for Basira System

This module implements the Arabic Rhetoric Analyzer, which analyzes the rhetorical
structure and stylistic features of Arabic text. It identifies rhetorical devices,
stylistic patterns, and aesthetic features.

Author: Basira System Development Team
Version: 1.0.0
"""

import re
import os
import json
import logging
import sys
from typing import Dict, List, Tuple, Union, Optional, Any, Set
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import core components
try:
    from core.general_shape_equation import GeneralShapeEquation, EquationType, LearningMode
except ImportError as e:
    logging.warning(f"Could not import GeneralShapeEquation: {e}")
    # Define placeholder classes
    class EquationType:
        CREATIVE = "creative"
        LINGUISTIC = "linguistic"

    class LearningMode:
        SUPERVISED = "supervised"
        ADAPTIVE = "adaptive"

    class GeneralShapeEquation:
        def __init__(self, equation_type, learning_mode):
            self.equation_type = equation_type
            self.learning_mode = learning_mode

# Configure logging
logger = logging.getLogger('arabic_nlp.rhetoric.rhetoric_analyzer')


@dataclass
class RhetoricalDevice:
    """Represents a rhetorical device in Arabic text."""
    device_type: str  # Type of rhetorical device
    name_ar: str  # Arabic name of the device
    name_en: str  # English name of the device
    description: str  # Description of the device
    examples: List[str] = field(default_factory=list)  # Examples of the device


@dataclass
class RhetoricalFeature:
    """Represents a rhetorical feature found in text."""
    device_type: str  # Type of rhetorical device
    text: str  # Text containing the feature
    start_index: int  # Start index in the original text
    end_index: int  # End index in the original text
    confidence: float = 1.0  # Confidence score
    details: Dict[str, Any] = field(default_factory=dict)  # Additional details


@dataclass
class RhetoricalAnalysis:
    """Represents the rhetorical analysis of a text."""
    text: str  # Original text
    features: List[RhetoricalFeature]  # List of rhetorical features
    summary: Dict[str, int]  # Summary of rhetorical devices found
    aesthetic_score: float  # Overall aesthetic score
    style_profile: Dict[str, float]  # Style profile


class ArabicRhetoricAnalyzer:
    """
    Arabic Rhetoric Analyzer class for analyzing the rhetorical structure of Arabic text.

    This class implements various algorithms for analyzing the rhetorical structure and
    stylistic features of Arabic text, including identification of rhetorical devices,
    stylistic patterns, and aesthetic features.
    """

    def __init__(self,
                 devices_file: Optional[str] = None,
                 patterns_file: Optional[str] = None,
                 use_ml_models: bool = False,
                 ml_models_path: Optional[str] = None):
        """
        Initialize the Arabic Rhetoric Analyzer.

        Args:
            devices_file: Path to the rhetorical devices file (optional)
            patterns_file: Path to the patterns file (optional)
            use_ml_models: Whether to use machine learning models
            ml_models_path: Path to the machine learning models (optional)
        """
        self.logger = logging.getLogger('arabic_nlp.rhetoric.rhetoric_analyzer.main')

        # Initialize General Shape Equation for rhetorical analysis
        self.general_equation = GeneralShapeEquation(
            equation_type=EquationType.CREATIVE,
            learning_mode=LearningMode.ADAPTIVE
        )

        # Load rhetorical devices
        self.devices = self._load_devices(devices_file)

        # Load patterns
        self.patterns = self._load_patterns(patterns_file)

        # Set whether to use machine learning models
        self.use_ml_models = use_ml_models

        # Initialize machine learning models if needed
        self.ml_models = None
        if self.use_ml_models:
            self._initialize_ml_models(ml_models_path)

        self.logger.info(f"Arabic Rhetoric Analyzer initialized with {len(self.devices)} devices and {len(self.patterns)} patterns")

    def _load_devices(self, devices_file: Optional[str]) -> Dict[str, RhetoricalDevice]:
        """
        Load rhetorical devices from file or use default devices.

        Args:
            devices_file: Path to the devices file

        Returns:
            Dictionary of rhetorical devices
        """
        default_devices = {
            "jinas": RhetoricalDevice(
                device_type="badi",
                name_ar="جناس",
                name_en="Paronomasia",
                description="Similarity in sound between words with different meanings",
                examples=["جاء الجار بالجرار"]
            ),
            "tibaq": RhetoricalDevice(
                device_type="badi",
                name_ar="طباق",
                name_en="Antithesis",
                description="Juxtaposition of contrasting ideas",
                examples=["يحيي ويميت"]
            ),
            "muqabala": RhetoricalDevice(
                device_type="badi",
                name_ar="مقابلة",
                name_en="Counterbalance",
                description="Parallel arrangement of contrasting ideas",
                examples=["يضحك قليلاً ويبكي كثيراً"]
            ),
            "tashbih": RhetoricalDevice(
                device_type="bayan",
                name_ar="تشبيه",
                name_en="Simile",
                description="Explicit comparison using 'like' or 'as'",
                examples=["العلم كالنور"]
            ),
            "istiara": RhetoricalDevice(
                device_type="bayan",
                name_ar="استعارة",
                name_en="Metaphor",
                description="Implicit comparison without using 'like' or 'as'",
                examples=["نور العلم"]
            ),
            "kinaya": RhetoricalDevice(
                device_type="bayan",
                name_ar="كناية",
                name_en="Metonymy",
                description="Indirect reference to something",
                examples=["كثير الرماد"]
            ),
            "saj": RhetoricalDevice(
                device_type="badi",
                name_ar="سجع",
                name_en="Rhymed Prose",
                description="Prose with rhyming endings",
                examples=["من جد وجد، ومن زرع حصد"]
            )
        }

        if devices_file and os.path.exists(devices_file):
            try:
                with open(devices_file, 'r', encoding='utf-8') as f:
                    devices_data = json.load(f)

                devices = {}
                for key, data in devices_data.items():
                    devices[key] = RhetoricalDevice(
                        device_type=data["device_type"],
                        name_ar=data["name_ar"],
                        name_en=data["name_en"],
                        description=data["description"],
                        examples=data.get("examples", [])
                    )

                self.logger.info(f"Rhetorical devices loaded from {devices_file}")
                return devices
            except Exception as e:
                self.logger.error(f"Error loading rhetorical devices from {devices_file}: {e}")
                self.logger.info("Using default rhetorical devices")
        else:
            if devices_file:
                self.logger.warning(f"Rhetorical devices file {devices_file} not found, using default devices")
            else:
                self.logger.info("No rhetorical devices file provided, using default devices")

        return default_devices

    def _load_patterns(self, patterns_file: Optional[str]) -> Dict[str, List[str]]:
        """
        Load patterns from file or use default patterns.

        Args:
            patterns_file: Path to the patterns file

        Returns:
            Dictionary of patterns
        """
        default_patterns = {
            "jinas": [
                r"(\w+).*\b\1\w*\b",  # Simple pattern for jinas
                r"\b(\w{3,}).*\b\w*\1\w*\b"  # Pattern for partial jinas
            ],
            "tibaq": [
                r"\b(أبيض|بيضاء).*\b(أسود|سوداء)\b",  # white/black
                r"\b(يحيي|حياة).*\b(يميت|موت)\b",  # life/death
                r"\b(يضحك|ضحك).*\b(يبكي|بكاء)\b",  # laugh/cry
                r"\b(نور|ضوء).*\b(ظلام|ظلمة)\b"  # light/darkness
            ],
            "saj": [
                r"\b(\w+)[اةى]\b.*\b(\w+)[اةى]\b",  # Simple pattern for saj
                r"\b(\w+)[اةى]\b.*\b(\w+)[اةى]\b.*\b(\w+)[اةى]\b"  # Pattern for extended saj
            ]
        }

        if patterns_file and os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)
                self.logger.info(f"Patterns loaded from {patterns_file}")
                return patterns
            except Exception as e:
                self.logger.error(f"Error loading patterns from {patterns_file}: {e}")
                self.logger.info("Using default patterns")
        else:
            if patterns_file:
                self.logger.warning(f"Patterns file {patterns_file} not found, using default patterns")
            else:
                self.logger.info("No patterns file provided, using default patterns")

        return default_patterns

    def _initialize_ml_models(self, ml_models_path: Optional[str]) -> None:
        """
        Initialize machine learning models.

        Args:
            ml_models_path: Path to the machine learning models
        """
        # Placeholder for machine learning model initialization
        # In a real implementation, this would load trained models

        self.ml_models = {
            "device_classifier": None,
            "style_analyzer": None,
            "aesthetic_scorer": None
        }

        self.logger.info("Machine learning models initialized")

    def analyze(self, text: str) -> RhetoricalAnalysis:
        """
        Analyze the rhetorical structure of Arabic text.

        Args:
            text: Arabic text

        Returns:
            Rhetorical analysis of the text
        """
        # Find rhetorical features
        features = self._find_rhetorical_features(text)

        # Create summary
        summary = self._create_summary(features)

        # Calculate aesthetic score
        aesthetic_score = self._calculate_aesthetic_score(features, text)

        # Create style profile
        style_profile = self._create_style_profile(features, text)

        # Create rhetorical analysis
        analysis = RhetoricalAnalysis(
            text=text,
            features=features,
            summary=summary,
            aesthetic_score=aesthetic_score,
            style_profile=style_profile
        )

        return analysis

    def _find_rhetorical_features(self, text: str) -> List[RhetoricalFeature]:
        """
        Find rhetorical features in text.

        Args:
            text: Arabic text

        Returns:
            List of rhetorical features
        """
        features = []

        # Apply pattern-based approach
        pattern_features = self._apply_patterns(text)
        features.extend(pattern_features)

        # Apply machine learning approach if enabled
        if self.use_ml_models:
            ml_features = self._apply_ml_models(text)
            features.extend(ml_features)

        return features

    def _apply_patterns(self, text: str) -> List[RhetoricalFeature]:
        """
        Apply pattern-based approach to find rhetorical features.

        Args:
            text: Arabic text

        Returns:
            List of rhetorical features
        """
        features = []

        # Apply patterns for each device type
        for device_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.UNICODE):
                    feature = RhetoricalFeature(
                        device_type=device_type,
                        text=match.group(0),
                        start_index=match.start(),
                        end_index=match.end(),
                        confidence=0.8,  # Default confidence for pattern-based approach
                        details={
                            "pattern": pattern,
                            "groups": match.groups()
                        }
                    )
                    features.append(feature)

        return features

    def _apply_ml_models(self, text: str) -> List[RhetoricalFeature]:
        """
        Apply machine learning approach to find rhetorical features.

        Args:
            text: Arabic text

        Returns:
            List of rhetorical features
        """
        # Placeholder for machine learning approach
        # In a real implementation, this would use trained models

        return []

    def _create_summary(self, features: List[RhetoricalFeature]) -> Dict[str, int]:
        """
        Create a summary of rhetorical devices found.

        Args:
            features: List of rhetorical features

        Returns:
            Dictionary mapping device types to counts
        """
        summary = {}

        for feature in features:
            device_type = feature.device_type
            if device_type in summary:
                summary[device_type] += 1
            else:
                summary[device_type] = 1

        return summary

    def _calculate_aesthetic_score(self, features: List[RhetoricalFeature], text: str) -> float:
        """
        Calculate the overall aesthetic score.

        Args:
            features: List of rhetorical features
            text: Original text

        Returns:
            Aesthetic score (0.0 to 1.0)
        """
        # Placeholder for aesthetic score calculation
        # In a real implementation, this would use a more sophisticated approach

        if not features:
            return 0.0

        # Simple calculation based on number of features and text length
        text_length = len(text)
        feature_count = len(features)

        # Normalize by text length (features per 100 characters)
        normalized_count = feature_count / (text_length / 100)

        # Cap at 1.0
        score = min(normalized_count / 5.0, 1.0)

        return score

    def _create_style_profile(self, features: List[RhetoricalFeature], text: str) -> Dict[str, float]:
        """
        Create a style profile.

        Args:
            features: List of rhetorical features
            text: Original text

        Returns:
            Dictionary mapping style dimensions to scores
        """
        # Placeholder for style profile creation
        # In a real implementation, this would use a more sophisticated approach

        # Count features by category
        badi_count = sum(1 for feature in features if self.devices.get(feature.device_type, RhetoricalDevice("", "", "", "")).device_type == "badi")
        bayan_count = sum(1 for feature in features if self.devices.get(feature.device_type, RhetoricalDevice("", "", "", "")).device_type == "bayan")

        # Calculate style dimensions
        style_profile = {
            "ornate": min(badi_count / 5.0, 1.0),  # Based on badi devices
            "figurative": min(bayan_count / 5.0, 1.0),  # Based on bayan devices
            "rhythmic": min(sum(1 for feature in features if feature.device_type == "saj") / 3.0, 1.0),  # Based on saj
            "balanced": min(sum(1 for feature in features if feature.device_type in ["tibaq", "muqabala"]) / 3.0, 1.0)  # Based on tibaq and muqabala
        }

        return style_profile


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create Arabic Rhetoric Analyzer
    analyzer = ArabicRhetoricAnalyzer()

    # Analyze text
    text = "العلم نور يضيء طريق الحياة، والجهل ظلام يحجب نور البصيرة"
    analysis = analyzer.analyze(text)

    # Print results
    print("Text:", text)
    print("Rhetorical Features:")
    for feature in analysis.features:
        print(f"  {feature.device_type}: {feature.text}")
    print("Summary:", analysis.summary)
    print(f"Aesthetic Score: {analysis.aesthetic_score:.2f}")
    print("Style Profile:")
    for dimension, score in analysis.style_profile.items():
        print(f"  {dimension}: {score:.2f}")
