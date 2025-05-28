#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Database Integration Module for Basira System

This module provides the integration between the semantic letter database and
the mathematical core of the Basira System. It enables semantic properties to
influence equation evolution and expert/explorer interactions.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import sys
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, Set
import random
import logging
from enum import Enum
from dataclasses import dataclass, field

# Import from parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced.general_shape_equation import (
    GeneralShapeEquation, EquationType, LearningMode, EquationMetadata,
    GSEFactory
)
from enhanced.expert_explorer_interaction import (
    AdvancedExpertExplorerSystem, ExplorationStrategy, ExpertKnowledgeType,
    ExplorationResult, ExpertFeedback
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('semantic_integration')


class SemanticAxis(str, Enum):
    """Semantic axes for letter properties."""
    AUTHORITY_TENDERNESS = "authority_tenderness"
    WHOLENESS_EMPTINESS = "wholeness_emptiness"
    MOVEMENT_CONTAINMENT = "movement_containment"
    INWARD_OUTWARD = "inward_outward"
    MOVEMENT_RESISTANCE = "movement_resistance"
    CONNECTION_SEPARATION = "connection_separation"
    MULTIPLICITY_SINGULARITY = "multiplicity_singularity"
    UNDULATION_STABILITY = "undulation_stability"
    FLOW_RESISTANCE = "flow_resistance"
    CONSTRUCTION_DESTRUCTION = "construction_destruction"
    # Add more axes as needed


class SemanticCategory(str, Enum):
    """Categories of semantic properties."""
    PHONETIC = "phonetic"
    VISUAL = "visual"
    CONCEPTUAL = "conceptual"
    EMOTIONAL = "emotional"
    FUNCTIONAL = "functional"
    # Add more categories as needed


@dataclass
class SemanticProperty:
    """A semantic property with its attributes."""
    name: str
    description: str
    axis: Optional[SemanticAxis] = None
    category: Optional[SemanticCategory] = None
    polarity: float = 0.0  # -1.0 to 1.0, where 0 is neutral
    strength: float = 1.0  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)


@dataclass
class SemanticVector:
    """A vector representation of semantic properties."""
    dimensions: Dict[str, float] = field(default_factory=dict)
    
    def __add__(self, other):
        """Add two semantic vectors."""
        result = SemanticVector()
        # Add all dimensions from self
        for dim, value in self.dimensions.items():
            result.dimensions[dim] = value
        
        # Add or update dimensions from other
        for dim, value in other.dimensions.items():
            if dim in result.dimensions:
                result.dimensions[dim] += value
            else:
                result.dimensions[dim] = value
        
        return result
    
    def __mul__(self, scalar):
        """Multiply a semantic vector by a scalar."""
        result = SemanticVector()
        for dim, value in self.dimensions.items():
            result.dimensions[dim] = value * scalar
        
        return result
    
    def normalize(self):
        """Normalize the vector to have unit length."""
        squared_sum = sum(value ** 2 for value in self.dimensions.values())
        if squared_sum > 0:
            length = np.sqrt(squared_sum)
            for dim in self.dimensions:
                self.dimensions[dim] /= length
        
        return self
    
    def similarity(self, other):
        """Calculate cosine similarity with another vector."""
        # Get all dimensions
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        
        # Calculate dot product
        dot_product = 0.0
        for dim in all_dims:
            dot_product += self.dimensions.get(dim, 0.0) * other.dimensions.get(dim, 0.0)
        
        # Calculate magnitudes
        self_magnitude = np.sqrt(sum(value ** 2 for value in self.dimensions.values()))
        other_magnitude = np.sqrt(sum(value ** 2 for value in other.dimensions.values()))
        
        # Calculate similarity
        if self_magnitude > 0 and other_magnitude > 0:
            return dot_product / (self_magnitude * other_magnitude)
        else:
            return 0.0


class SemanticDatabaseManager:
    """
    Manager for the semantic letter database.
    
    This class provides methods for loading, querying, and manipulating
    the semantic database, as well as integrating it with the mathematical core.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize a SemanticDatabaseManager.
        
        Args:
            database_path: Optional path to the semantic database file
        """
        self.logger = logging.getLogger('semantic_integration.database')
        
        # Initialize database
        self.database = {}
        self.properties = {}
        self.axes = {}
        self.categories = {}
        
        # Load database if provided
        if database_path:
            self.load_database(database_path)
    
    def load_database(self, file_path: str) -> bool:
        """
        Load semantic database from a file.
        
        Args:
            file_path: Path to the semantic database file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
            
            # Extract semantic properties
            self._extract_properties()
            
            self.logger.info(f"Loaded semantic database with {len(self.database)} entries")
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading semantic database: {e}")
            return False
    
    def _extract_properties(self):
        """Extract semantic properties from the database."""
        # Reset properties
        self.properties = {}
        self.axes = {}
        self.categories = {}
        
        # Process each letter
        for letter, data in self.database.items():
            # Process phonetic properties
            if 'phonetic_properties' in data:
                for prop_name, prop_value in data['phonetic_properties'].items():
                    if prop_value:
                        self._add_property(
                            prop_name, 
                            prop_value, 
                            SemanticCategory.PHONETIC
                        )
            
            # Process visual form semantics
            if 'visual_form_semantics' in data:
                for prop_value in data['visual_form_semantics']:
                    if prop_value:
                        self._add_property(
                            f"visual_{letter}_{len(self.properties)}", 
                            prop_value, 
                            SemanticCategory.VISUAL
                        )
            
            # Process core semantic axes
            if 'core_semantic_axes' in data:
                for axis_name, axis_values in data['core_semantic_axes'].items():
                    # Add axis
                    if axis_name not in self.axes:
                        self.axes[axis_name] = {
                            'name': axis_name,
                            'letters': [],
                            'values': []
                        }
                    
                    self.axes[axis_name]['letters'].append(letter)
                    
                    # Add values
                    for value in axis_values:
                        if value:
                            self.axes[axis_name]['values'].append(value)
                            
                            # Add as property
                            self._add_property(
                                f"axis_{axis_name}_{len(self.properties)}",
                                value,
                                SemanticCategory.CONCEPTUAL,
                                axis_name
                            )
            
            # Process general connotations
            if 'general_connotations' in data:
                for prop_value in data['general_connotations']:
                    if prop_value:
                        self._add_property(
                            f"connotation_{letter}_{len(self.properties)}",
                            prop_value,
                            SemanticCategory.CONCEPTUAL
                        )
        
        self.logger.info(f"Extracted {len(self.properties)} semantic properties")
        self.logger.info(f"Identified {len(self.axes)} semantic axes")
        self.logger.info(f"Categorized into {len(self.categories)} categories")
    
    def _add_property(self, name: str, description: str, 
                    category: Optional[SemanticCategory] = None,
                    axis: Optional[str] = None):
        """
        Add a semantic property.
        
        Args:
            name: Property name
            description: Property description
            category: Optional semantic category
            axis: Optional semantic axis
        """
        # Create property
        property_id = f"{category.value if category else 'unknown'}_{name}"
        
        self.properties[property_id] = {
            'name': name,
            'description': description,
            'category': category.value if category else None,
            'axis': axis
        }
        
        # Add to category
        if category:
            if category.value not in self.categories:
                self.categories[category.value] = []
            
            self.categories[category.value].append(property_id)
    
    def get_letter_properties(self, letter: str) -> Dict[str, Any]:
        """
        Get properties for a specific letter.
        
        Args:
            letter: The letter to query
            
        Returns:
            Dictionary of letter properties
        """
        return self.database.get(letter, {})
    
    def get_letter_semantic_vector(self, letter: str) -> SemanticVector:
        """
        Get semantic vector for a specific letter.
        
        Args:
            letter: The letter to query
            
        Returns:
            SemanticVector for the letter
        """
        letter_data = self.get_letter_properties(letter)
        vector = SemanticVector()
        
        # Add phonetic dimensions
        if 'phonetic_properties' in letter_data:
            for prop, value in letter_data['phonetic_properties'].items():
                if value:
                    vector.dimensions[f"phonetic_{prop}"] = 1.0
        
        # Add visual dimensions
        if 'visual_form_semantics' in letter_data:
            for i, form in enumerate(letter_data['visual_form_semantics']):
                if form:
                    vector.dimensions[f"visual_{i}"] = 1.0
        
        # Add semantic axes dimensions
        if 'core_semantic_axes' in letter_data:
            for axis, values in letter_data['core_semantic_axes'].items():
                for i, value in enumerate(values):
                    if value:
                        # First value is positive pole, second is negative
                        polarity = 1.0 if i == 0 else -1.0
                        vector.dimensions[f"axis_{axis}"] = polarity
        
        # Add connotation dimensions
        if 'general_connotations' in letter_data:
            for i, connotation in enumerate(letter_data['general_connotations']):
                if connotation:
                    vector.dimensions[f"connotation_{i}"] = 1.0
        
        return vector
    
    def get_word_semantic_vector(self, word: str) -> SemanticVector:
        """
        Get semantic vector for a word.
        
        Args:
            word: The word to analyze
            
        Returns:
            SemanticVector for the word
        """
        # Convert to uppercase for consistency
        word = word.upper()
        
        # Initialize vector
        vector = SemanticVector()
        
        # Add vectors for each letter
        for letter in word:
            if letter in self.database:
                letter_vector = self.get_letter_semantic_vector(letter)
                vector = vector + letter_vector
        
        # Normalize
        return vector.normalize()
    
    def find_similar_words(self, target_vector: SemanticVector, 
                         word_list: List[str], 
                         top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find words with similar semantic vectors.
        
        Args:
            target_vector: Target semantic vector
            word_list: List of words to search
            top_n: Number of top results to return
            
        Returns:
            List of (word, similarity) tuples
        """
        similarities = []
        
        for word in word_list:
            word_vector = self.get_word_semantic_vector(word)
            similarity = target_vector.similarity(word_vector)
            similarities.append((word, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return similarities[:top_n]
    
    def get_semantic_axes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all semantic axes.
        
        Returns:
            Dictionary of semantic axes
        """
        return self.axes
    
    def get_semantic_categories(self) -> Dict[str, List[str]]:
        """
        Get all semantic categories.
        
        Returns:
            Dictionary of semantic categories
        """
        return self.categories
    
    def get_properties_by_category(self, category: SemanticCategory) -> List[Dict[str, Any]]:
        """
        Get properties by category.
        
        Args:
            category: The semantic category
            
        Returns:
            List of properties in the category
        """
        property_ids = self.categories.get(category.value, [])
        return [self.properties[pid] for pid in property_ids]
    
    def get_properties_by_axis(self, axis: str) -> List[Dict[str, Any]]:
        """
        Get properties by axis.
        
        Args:
            axis: The semantic axis
            
        Returns:
            List of properties on the axis
        """
        return [prop for prop_id, prop in self.properties.items() if prop['axis'] == axis]
    
    def export_database(self, file_path: str) -> bool:
        """
        Export the semantic database to a file.
        
        Args:
            file_path: Path to save the database
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.database, f, indent=2)
            
            self.logger.info(f"Exported semantic database to {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error exporting semantic database: {e}")
            return False


class SemanticEquationGenerator:
    """
    Generator for equations based on semantic properties.
    
    This class provides methods for generating and evolving equations
    based on semantic properties and concepts.
    """
    
    def __init__(self, semantic_db: SemanticDatabaseManager):
        """
        Initialize a SemanticEquationGenerator.
        
        Args:
            semantic_db: SemanticDatabaseManager instance
        """
        self.logger = logging.getLogger('semantic_integration.generator')
        self.semantic_db = semantic_db
        
        # Initialize shape templates
        self.shape_templates = {
            'circle': {
                'function': lambda cx, cy, r: GSEFactory.create_basic_shape('circle', cx=cx, cy=cy, radius=r),
                'parameters': ['cx', 'cy', 'radius'],
                'semantic_axes': [
                    SemanticAxis.WHOLENESS_EMPTINESS,
                    SemanticAxis.MOVEMENT_CONTAINMENT
                ]
            },
            'rectangle': {
                'function': lambda x, y, w, h: GSEFactory.create_basic_shape('rectangle', x=x, y=y, width=w, height=h),
                'parameters': ['x', 'y', 'width', 'height'],
                'semantic_axes': [
                    SemanticAxis.CONSTRUCTION_DESTRUCTION,
                    SemanticAxis.AUTHORITY_TENDERNESS
                ]
            },
            'ellipse': {
                'function': lambda cx, cy, rx, ry: GSEFactory.create_basic_shape('ellipse', cx=cx, cy=cy, rx=rx, ry=ry),
                'parameters': ['cx', 'cy', 'rx', 'ry'],
                'semantic_axes': [
                    SemanticAxis.WHOLENESS_EMPTINESS,
                    SemanticAxis.MOVEMENT_CONTAINMENT,
                    SemanticAxis.INWARD_OUTWARD
                ]
            },
            'wave': {
                'function': lambda a, f, p: self._create_wave_equation(a, f, p),
                'parameters': ['amplitude', 'frequency', 'phase'],
                'semantic_axes': [
                    SemanticAxis.UNDULATION_STABILITY,
                    SemanticAxis.FLOW_RESISTANCE,
                    SemanticAxis.MOVEMENT_RESISTANCE
                ]
            },
            'spiral': {
                'function': lambda a, b: self._create_spiral_equation(a, b),
                'parameters': ['a', 'b'],
                'semantic_axes': [
                    SemanticAxis.MOVEMENT_RESISTANCE,
                    SemanticAxis.FLOW_RESISTANCE,
                    SemanticAxis.MOVEMENT_CONTAINMENT
                ]
            }
        }
    
    def _create_wave_equation(self, amplitude: float, frequency: float, phase: float) -> GeneralShapeEquation:
        """
        Create a wave equation.
        
        Args:
            amplitude: Wave amplitude
            frequency: Wave frequency
            phase: Wave phase
            
        Returns:
            GeneralShapeEquation for a wave
        """
        # Create a new equation
        equation = GeneralShapeEquation()
        
        # Add wave component
        equation.add_component(
            'wave',
            f"{amplitude} * sin({frequency} * x + {phase})"
        )
        
        # Set metadata
        equation.metadata.description = f"Wave with amplitude {amplitude}, frequency {frequency}, phase {phase}"
        equation.metadata.complexity = 2.0
        
        return equation
    
    def _create_spiral_equation(self, a: float, b: float) -> GeneralShapeEquation:
        """
        Create a spiral equation.
        
        Args:
            a: Spiral parameter a
            b: Spiral parameter b
            
        Returns:
            GeneralShapeEquation for a spiral
        """
        # Create a new equation
        equation = GeneralShapeEquation()
        
        # Add spiral components
        equation.add_component(
            'spiral_x',
            f"{a} * cos(t) * exp({b} * t)"
        )
        
        equation.add_component(
            'spiral_y',
            f"{a} * sin(t) * exp({b} * t)"
        )
        
        # Set metadata
        equation.metadata.description = f"Spiral with parameters a={a}, b={b}"
        equation.metadata.complexity = 3.0
        
        return equation
    
    def generate_equation_from_letter(self, letter: str) -> GeneralShapeEquation:
        """
        Generate an equation based on a letter's semantic properties.
        
        Args:
            letter: The letter to base the equation on
            
        Returns:
            GeneralShapeEquation based on the letter
        """
        # Get letter properties
        letter_data = self.semantic_db.get_letter_properties(letter)
        
        if not letter_data:
            self.logger.warning(f"No data found for letter {letter}")
            return GSEFactory.create_basic_shape('circle')  # Default
        
        # Determine the most appropriate shape template
        shape_type = self._select_shape_for_letter(letter_data)
        
        # Generate parameters based on semantic properties
        params = self._generate_parameters_for_shape(shape_type, letter_data)
        
        # Create the equation
        template = self.shape_templates[shape_type]
        equation = template['function'](*params)
        
        # Add semantic links
        self._add_semantic_links_to_equation(equation, letter_data)
        
        # Update metadata
        equation.metadata.description = f"Equation based on letter {letter}"
        equation.metadata.tags.append(f"letter_{letter}")
        
        return equation
    
    def _select_shape_for_letter(self, letter_data: Dict[str, Any]) -> str:
        """
        Select the most appropriate shape for a letter.
        
        Args:
            letter_data: Letter properties
            
        Returns:
            Selected shape type
        """
        # Default shape
        default_shape = 'circle'
        
        # Check if visual form suggests a specific shape
        if 'visual_form_semantics' in letter_data:
            visual_forms = letter_data['visual_form_semantics']
            
            # Look for specific shapes in visual forms
            for form in visual_forms:
                form_lower = form.lower() if form else ""
                
                if 'circle' in form_lower or 'round' in form_lower:
                    return 'circle'
                elif 'rectangle' in form_lower or 'square' in form_lower:
                    return 'rectangle'
                elif 'ellipse' in form_lower or 'oval' in form_lower:
                    return 'ellipse'
                elif 'wave' in form_lower or 'undulat' in form_lower:
                    return 'wave'
                elif 'spiral' in form_lower or 'curl' in form_lower:
                    return 'spiral'
        
        # Check semantic axes
        if 'core_semantic_axes' in letter_data:
            axes = letter_data['core_semantic_axes'].keys()
            
            # Match axes to shapes
            for shape, template in self.shape_templates.items():
                semantic_axes = template.get('semantic_axes', [])
                
                # Convert string axes to SemanticAxis enum
                semantic_axes = [axis.value if isinstance(axis, SemanticAxis) else axis 
                               for axis in semantic_axes]
                
                # Check for matches
                if any(axis in semantic_axes for axis in axes):
                    return shape
        
        # Default
        return default_shape
    
    def _generate_parameters_for_shape(self, shape_type: str, 
                                     letter_data: Dict[str, Any]) -> List[float]:
        """
        Generate parameters for a shape based on letter properties.
        
        Args:
            shape_type: Type of shape
            letter_data: Letter properties
            
        Returns:
            List of parameter values
        """
        # Get template
        template = self.shape_templates.get(shape_type)
        if not template:
            return [0, 0, 1]  # Default parameters
        
        # Get parameter names
        param_names = template.get('parameters', [])
        
        # Generate values
        values = []
        
        for param in param_names:
            # Default value
            value = 1.0
            
            # Modify based on semantic properties
            if 'core_semantic_axes' in letter_data:
                for axis, axis_values in letter_data['core_semantic_axes'].items():
                    # Adjust value based on axis
                    value = self._adjust_parameter_by_axis(param, axis, axis_values, value)
            
            values.append(value)
        
        return values
    
    def _adjust_parameter_by_axis(self, param: str, axis: str, 
                                axis_values: List[str], 
                                current_value: float) -> float:
        """
        Adjust a parameter value based on a semantic axis.
        
        Args:
            param: Parameter name
            axis: Semantic axis
            axis_values: Values on the axis
            current_value: Current parameter value
            
        Returns:
            Adjusted parameter value
        """
        # Default adjustment factor
        adjustment = 1.0
        
        # Adjust based on specific parameter and axis combinations
        if param in ['radius', 'width', 'height', 'rx', 'ry', 'amplitude']:
            if axis == SemanticAxis.AUTHORITY_TENDERNESS.value:
                # Authority increases size, tenderness decreases
                adjustment = 1.5 if 'authority' in ' '.join(axis_values).lower() else 0.7
            
            elif axis == SemanticAxis.WHOLENESS_EMPTINESS.value:
                # Wholeness increases size, emptiness decreases
                adjustment = 1.3 if 'wholeness' in ' '.join(axis_values).lower() else 0.8
            
            elif axis == SemanticAxis.MULTIPLICITY_SINGULARITY.value:
                # Multiplicity increases size
                adjustment = 1.4 if 'multiplicity' in ' '.join(axis_values).lower() else 0.9
        
        elif param in ['frequency']:
            if axis == SemanticAxis.MOVEMENT_RESISTANCE.value:
                # Movement increases frequency, resistance decreases
                adjustment = 1.5 if 'movement' in ' '.join(axis_values).lower() else 0.6
            
            elif axis == SemanticAxis.FLOW_RESISTANCE.value:
                # Flow increases frequency, resistance decreases
                adjustment = 1.4 if 'flow' in ' '.join(axis_values).lower() else 0.7
        
        # Apply adjustment
        return current_value * adjustment
    
    def _add_semantic_links_to_equation(self, equation: GeneralShapeEquation, 
                                      letter_data: Dict[str, Any]):
        """
        Add semantic links to an equation based on letter properties.
        
        Args:
            equation: The equation to modify
            letter_data: Letter properties
        """
        # Add phonetic properties
        if 'phonetic_properties' in letter_data:
            for prop, value in letter_data['phonetic_properties'].items():
                if value:
                    equation.add_semantic_link(
                        f"phonetic_{prop}",
                        value,
                        0.8
                    )
        
        # Add visual form semantics
        if 'visual_form_semantics' in letter_data:
            for form in letter_data['visual_form_semantics']:
                if form:
                    equation.add_semantic_link(
                        "visual_form",
                        form,
                        0.9
                    )
        
        # Add semantic axes
        if 'core_semantic_axes' in letter_data:
            for axis, values in letter_data['core_semantic_axes'].items():
                for value in values:
                    if value:
                        equation.add_semantic_link(
                            f"axis_{axis}",
                            value,
                            1.0
                        )
        
        # Add general connotations
        if 'general_connotations' in letter_data:
            for connotation in letter_data['general_connotations']:
                if connotation:
                    equation.add_semantic_link(
                        "connotation",
                        connotation,
                        0.7
                    )
    
    def generate_equation_from_word(self, word: str) -> GeneralShapeEquation:
        """
        Generate an equation based on a word's semantic properties.
        
        Args:
            word: The word to base the equation on
            
        Returns:
            GeneralShapeEquation based on the word
        """
        # Convert to uppercase for consistency
        word = word.upper()
        
        # Generate equations for each letter
        letter_equations = []
        for letter in word:
            if letter in self.semantic_db.database:
                letter_equations.append(self.generate_equation_from_letter(letter))
        
        # If no valid letters, return default
        if not letter_equations:
            self.logger.warning(f"No valid letters found in word {word}")
            return GSEFactory.create_basic_shape('circle')
        
        # Combine equations
        combined_equation = self._combine_equations(letter_equations)
        
        # Update metadata
        combined_equation.metadata.description = f"Equation based on word {word}"
        combined_equation.metadata.tags.append(f"word_{word}")
        
        return combined_equation
    
    def _combine_equations(self, equations: List[GeneralShapeEquation]) -> GeneralShapeEquation:
        """
        Combine multiple equations into one.
        
        Args:
            equations: List of equations to combine
            
        Returns:
            Combined equation
        """
        if not equations:
            return GSEFactory.create_basic_shape('circle')
        
        # Start with the first equation
        combined = equations[0].clone()
        
        # Add components from other equations
        for i, eq in enumerate(equations[1:], 1):
            for comp_name, comp_expr in eq.symbolic_components.items():
                # Add with modified name to avoid conflicts
                combined.add_component(
                    f"{comp_name}_{i}",
                    comp_expr
                )
            
            # Merge semantic links
            for link in eq.get_semantic_links():
                combined.add_semantic_link(
                    link.get('concept', ''),
                    link.get('type', ''),
                    link.get('strength', 0.5)
                )
        
        # Update complexity
        combined.metadata.complexity = sum(eq.metadata.complexity for eq in equations) / len(equations)
        
        return combined
    
    def generate_equation_from_semantic_concept(self, concept: str) -> GeneralShapeEquation:
        """
        Generate an equation based on a semantic concept.
        
        Args:
            concept: The semantic concept
            
        Returns:
            GeneralShapeEquation based on the concept
        """
        # Find letters with this concept
        relevant_letters = []
        
        for letter, data in self.semantic_db.database.items():
            # Check in all semantic properties
            concept_lower = concept.lower()
            
            # Check in core semantic axes
            if 'core_semantic_axes' in data:
                for axis, values in data['core_semantic_axes'].items():
                    for value in values:
                        if value and concept_lower in value.lower():
                            relevant_letters.append(letter)
                            break
            
            # Check in general connotations
            if 'general_connotations' in data:
                for connotation in data['general_connotations']:
                    if connotation and concept_lower in connotation.lower():
                        relevant_letters.append(letter)
                        break
            
            # Check in visual form semantics
            if 'visual_form_semantics' in data:
                for form in data['visual_form_semantics']:
                    if form and concept_lower in form.lower():
                        relevant_letters.append(letter)
                        break
        
        # If no relevant letters, use a default shape
        if not relevant_letters:
            self.logger.warning(f"No letters found for concept {concept}")
            
            # Create a default equation
            equation = GSEFactory.create_basic_shape('circle')
            
            # Add the concept as a semantic link
            equation.add_semantic_link(
                "concept",
                concept,
                1.0
            )
            
            return equation
        
        # Generate equations for relevant letters
        letter_equations = [self.generate_equation_from_letter(letter) for letter in relevant_letters]
        
        # Combine equations
        combined_equation = self._combine_equations(letter_equations)
        
        # Update metadata
        combined_equation.metadata.description = f"Equation based on concept {concept}"
        combined_equation.metadata.tags.append(f"concept_{concept}")
        
        # Add the concept as a semantic link
        combined_equation.add_semantic_link(
            "concept",
            concept,
            1.0
        )
        
        return combined_equation


class SemanticGuidedExplorer:
    """
    Explorer that uses semantic guidance for equation evolution.
    
    This class extends the AdvancedExpertExplorerSystem with semantic guidance
    capabilities, allowing exploration to be guided by semantic concepts.
    """
    
    def __init__(self, semantic_db: SemanticDatabaseManager, 
                 equation_generator: SemanticEquationGenerator,
                 expert_explorer_system: Optional[AdvancedExpertExplorerSystem] = None):
        """
        Initialize a SemanticGuidedExplorer.
        
        Args:
            semantic_db: SemanticDatabaseManager instance
            equation_generator: SemanticEquationGenerator instance
            expert_explorer_system: Optional AdvancedExpertExplorerSystem instance
        """
        self.logger = logging.getLogger('semantic_integration.explorer')
        self.semantic_db = semantic_db
        self.equation_generator = equation_generator
        
        # Create or use expert-explorer system
        if expert_explorer_system:
            self.system = expert_explorer_system
        else:
            self.system = AdvancedExpertExplorerSystem()
        
        # Add semantic evaluation metrics
        self._add_semantic_evaluation_metrics()
    
    def _add_semantic_evaluation_metrics(self):
        """Add semantic evaluation metrics to the expert-explorer system."""
        # Add semantic alignment metric
        self.system.add_evaluation_metric(
            'semantic_alignment',
            self._evaluate_semantic_alignment,
            "Evaluate alignment with semantic concepts",
            0.3,
            ['semantics', 'meaning']
        )
        
        # Add semantic coherence metric
        self.system.add_evaluation_metric(
            'semantic_coherence',
            self._evaluate_semantic_coherence,
            "Evaluate coherence of semantic properties",
            0.2,
            ['semantics', 'coherence']
        )
    
    def _evaluate_semantic_alignment(self, equation: GeneralShapeEquation) -> float:
        """
        Evaluate how well an equation aligns with semantic concepts.
        
        Args:
            equation: The equation to evaluate
            
        Returns:
            Semantic alignment score (higher is better)
        """
        # Get semantic links from the equation
        equation_semantics = equation.get_semantic_links()
        
        # If no semantic links, return low score
        if not equation_semantics:
            return 0.2
        
        # Calculate alignment score
        total_strength = 0.0
        for link in equation_semantics:
            concept = link.get('concept', '')
            strength = link.get('strength', 0.0)
            
            # Check if concept exists in semantic database
            found = False
            for letter, data in self.semantic_db.database.items():
                # Check in all semantic properties
                concept_lower = concept.lower()
                
                # Check in core semantic axes
                if 'core_semantic_axes' in data:
                    for axis, values in data['core_semantic_axes'].items():
                        for value in values:
                            if value and concept_lower in value.lower():
                                found = True
                                break
                
                # Check in general connotations
                if not found and 'general_connotations' in data:
                    for connotation in data['general_connotations']:
                        if connotation and concept_lower in connotation.lower():
                            found = True
                            break
                
                # Check in visual form semantics
                if not found and 'visual_form_semantics' in data:
                    for form in data['visual_form_semantics']:
                        if form and concept_lower in form.lower():
                            found = True
                            break
                
                if found:
                    break
            
            if found:
                total_strength += strength
        
        # Normalize score
        if len(equation_semantics) > 0:
            return min(1.0, total_strength / len(equation_semantics))
        else:
            return 0.0
    
    def _evaluate_semantic_coherence(self, equation: GeneralShapeEquation) -> float:
        """
        Evaluate the coherence of semantic properties in an equation.
        
        Args:
            equation: The equation to evaluate
            
        Returns:
            Semantic coherence score (higher is better)
        """
        # Get semantic links from the equation
        equation_semantics = equation.get_semantic_links()
        
        # If no semantic links, return neutral score
        if not equation_semantics or len(equation_semantics) < 2:
            return 0.5
        
        # Calculate coherence score
        # (based on whether semantic concepts are related)
        coherence_pairs = 0
        total_pairs = 0
        
        for i, link1 in enumerate(equation_semantics):
            concept1 = link1.get('concept', '')
            
            for link2 in equation_semantics[i+1:]:
                concept2 = link2.get('concept', '')
                
                # Skip if either concept is empty
                if not concept1 or not concept2:
                    continue
                
                # Check if concepts are related
                if self._are_concepts_related(concept1, concept2):
                    coherence_pairs += 1
                
                total_pairs += 1
        
        # Calculate score
        if total_pairs > 0:
            return coherence_pairs / total_pairs
        else:
            return 0.5
    
    def _are_concepts_related(self, concept1: str, concept2: str) -> bool:
        """
        Check if two semantic concepts are related.
        
        Args:
            concept1: First concept
            concept2: Second concept
            
        Returns:
            True if concepts are related, False otherwise
        """
        # Convert to lowercase
        concept1_lower = concept1.lower()
        concept2_lower = concept2.lower()
        
        # Check if concepts are the same or substrings
        if concept1_lower == concept2_lower:
            return True
        
        if concept1_lower in concept2_lower or concept2_lower in concept1_lower:
            return True
        
        # Check if concepts appear together in any letter's properties
        for letter, data in self.semantic_db.database.items():
            # Check in core semantic axes
            if 'core_semantic_axes' in data:
                for axis, values in data['core_semantic_axes'].items():
                    values_text = ' '.join(values).lower()
                    if concept1_lower in values_text and concept2_lower in values_text:
                        return True
            
            # Check in general connotations
            if 'general_connotations' in data:
                connotations_text = ' '.join(data['general_connotations']).lower()
                if concept1_lower in connotations_text and concept2_lower in connotations_text:
                    return True
        
        # Not related
        return False
    
    def explore_with_semantic_guidance(self, target_concepts: List[str], 
                                     cycles: int = 5, 
                                     steps_per_cycle: int = 10) -> GeneralShapeEquation:
        """
        Explore equation space with semantic guidance.
        
        Args:
            target_concepts: List of target semantic concepts
            cycles: Number of interaction cycles
            steps_per_cycle: Steps per exploration cycle
            
        Returns:
            Best equation found during exploration
        """
        # Generate initial equation based on concepts
        if not self.system.explorer['current_equation']:
            # Generate equation from first concept
            initial_equation = self.equation_generator.generate_equation_from_semantic_concept(
                target_concepts[0]
            )
            
            # Add other concepts as semantic links
            for concept in target_concepts[1:]:
                initial_equation.add_semantic_link(
                    "concept",
                    concept,
                    1.0
                )
            
            # Set as initial equation
            self.system.set_initial_equation(initial_equation)
        
        # Run expert-explorer interaction with semantic guidance
        best_equation = self.system.expert_explorer_interaction(
            cycles=cycles,
            steps_per_cycle=steps_per_cycle,
            target_semantics=target_concepts
        )
        
        return best_equation
    
    def explore_with_letter(self, letter: str, 
                          cycles: int = 5, 
                          steps_per_cycle: int = 10) -> GeneralShapeEquation:
        """
        Explore equation space based on a letter.
        
        Args:
            letter: The letter to base exploration on
            cycles: Number of interaction cycles
            steps_per_cycle: Steps per exploration cycle
            
        Returns:
            Best equation found during exploration
        """
        # Convert to uppercase for consistency
        letter = letter.upper()
        
        # Get letter properties
        letter_data = self.semantic_db.get_letter_properties(letter)
        
        if not letter_data:
            self.logger.warning(f"No data found for letter {letter}")
            return None
        
        # Extract target concepts
        target_concepts = []
        
        # Add core semantic axes
        if 'core_semantic_axes' in letter_data:
            for axis, values in letter_data['core_semantic_axes'].items():
                for value in values:
                    if value:
                        target_concepts.append(value)
        
        # Add top general connotations
        if 'general_connotations' in letter_data:
            top_connotations = letter_data['general_connotations'][:5]
            for connotation in top_connotations:
                if connotation:
                    target_concepts.append(connotation)
        
        # Generate initial equation
        initial_equation = self.equation_generator.generate_equation_from_letter(letter)
        
        # Set as initial equation
        self.system.set_initial_equation(initial_equation)
        
        # Run exploration
        best_equation = self.explore_with_semantic_guidance(
            target_concepts=target_concepts,
            cycles=cycles,
            steps_per_cycle=steps_per_cycle
        )
        
        return best_equation
    
    def explore_with_word(self, word: str, 
                        cycles: int = 5, 
                        steps_per_cycle: int = 10) -> GeneralShapeEquation:
        """
        Explore equation space based on a word.
        
        Args:
            word: The word to base exploration on
            cycles: Number of interaction cycles
            steps_per_cycle: Steps per exploration cycle
            
        Returns:
            Best equation found during exploration
        """
        # Convert to uppercase for consistency
        word = word.upper()
        
        # Generate initial equation
        initial_equation = self.equation_generator.generate_equation_from_word(word)
        
        # Set as initial equation
        self.system.set_initial_equation(initial_equation)
        
        # Extract target concepts from letters
        target_concepts = []
        
        for letter in word:
            letter_data = self.semantic_db.get_letter_properties(letter)
            
            if letter_data:
                # Add top connotations
                if 'general_connotations' in letter_data:
                    top_connotations = letter_data['general_connotations'][:2]
                    for connotation in top_connotations:
                        if connotation and connotation not in target_concepts:
                            target_concepts.append(connotation)
        
        # Run exploration
        best_equation = self.explore_with_semantic_guidance(
            target_concepts=target_concepts,
            cycles=cycles,
            steps_per_cycle=steps_per_cycle
        )
        
        return best_equation
    
    def get_semantic_exploration_results(self, target: str, 
                                       is_letter: bool = False,
                                       is_word: bool = False,
                                       is_concept: bool = False,
                                       cycles: int = 5,
                                       steps_per_cycle: int = 10,
                                       visualize: bool = True,
                                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive results from semantic exploration.
        
        Args:
            target: The target letter, word, or concept
            is_letter: Whether the target is a letter
            is_word: Whether the target is a word
            is_concept: Whether the target is a concept
            cycles: Number of interaction cycles
            steps_per_cycle: Steps per exploration cycle
            visualize: Whether to generate visualizations
            save_path: Optional path to save visualizations
            
        Returns:
            Dictionary with exploration results
        """
        # Determine exploration type
        if is_letter:
            best_equation = self.explore_with_letter(
                target,
                cycles=cycles,
                steps_per_cycle=steps_per_cycle
            )
            exploration_type = 'letter'
        elif is_word:
            best_equation = self.explore_with_word(
                target,
                cycles=cycles,
                steps_per_cycle=steps_per_cycle
            )
            exploration_type = 'word'
        elif is_concept:
            # Generate initial equation
            initial_equation = self.equation_generator.generate_equation_from_semantic_concept(target)
            
            # Set as initial equation
            self.system.set_initial_equation(initial_equation)
            
            # Run exploration
            best_equation = self.explore_with_semantic_guidance(
                target_concepts=[target],
                cycles=cycles,
                steps_per_cycle=steps_per_cycle
            )
            exploration_type = 'concept'
        else:
            raise ValueError("Must specify one of is_letter, is_word, or is_concept")
        
        # Get results with visualization
        if visualize:
            results = self.system.expert_explorer_interaction_with_visualization(
                cycles=0,  # Don't run additional cycles
                visualize=True,
                save_path=save_path
            )
            
            results['best_equation'] = best_equation
        else:
            results = {
                'best_equation': best_equation,
                'best_score': self.system.expert_evaluate(best_equation),
                'cycles': cycles,
                'steps_per_cycle': steps_per_cycle
            }
        
        # Add semantic information
        results['target'] = target
        results['exploration_type'] = exploration_type
        
        # Add semantic links
        results['semantic_links'] = best_equation.get_semantic_links()
        
        # Add evaluation details
        evaluation = self.system.expert_evaluate_with_target(
            best_equation,
            [target] if is_concept else None
        )
        
        results['evaluation'] = evaluation
        
        return results


# Utility functions for testing and demonstration

def create_semantic_integration_system(database_path: str) -> Tuple[SemanticDatabaseManager, 
                                                                 SemanticEquationGenerator,
                                                                 SemanticGuidedExplorer]:
    """
    Create a complete semantic integration system.
    
    Args:
        database_path: Path to the semantic database file
        
    Returns:
        Tuple of (SemanticDatabaseManager, SemanticEquationGenerator, SemanticGuidedExplorer)
    """
    # Create database manager
    db_manager = SemanticDatabaseManager(database_path)
    
    # Create equation generator
    equation_generator = SemanticEquationGenerator(db_manager)
    
    # Create expert-explorer system
    expert_explorer = AdvancedExpertExplorerSystem()
    
    # Create semantic guided explorer
    semantic_explorer = SemanticGuidedExplorer(
        db_manager,
        equation_generator,
        expert_explorer
    )
    
    return db_manager, equation_generator, semantic_explorer


def run_semantic_demonstration(database_path: str, 
                             target: str,
                             is_letter: bool = False,
                             is_word: bool = False,
                             is_concept: bool = False) -> Dict[str, Any]:
    """
    Run a demonstration of semantic integration.
    
    Args:
        database_path: Path to the semantic database file
        target: The target letter, word, or concept
        is_letter: Whether the target is a letter
        is_word: Whether the target is a word
        is_concept: Whether the target is a concept
        
    Returns:
        Dictionary with demonstration results
    """
    # Create system
    db_manager, equation_generator, semantic_explorer = create_semantic_integration_system(database_path)
    
    # Run exploration
    results = semantic_explorer.get_semantic_exploration_results(
        target=target,
        is_letter=is_letter,
        is_word=is_word,
        is_concept=is_concept,
        cycles=3,
        steps_per_cycle=5,
        visualize=True
    )
    
    # Print summary
    print("\nSemantic Demonstration Summary:")
    print(f"Target: {target} ({results['exploration_type']})")
    print(f"Best equation score: {results['best_score']:.4f}")
    print(f"Best equation complexity: {results['best_equation'].metadata.complexity:.4f}")
    print(f"Best equation components: {len(results['best_equation'].symbolic_components)}")
    
    # Print semantic links
    print("\nSemantic Links:")
    for link in results['semantic_links']:
        print(f"  - {link.get('concept', '')}: {link.get('strength', 0.0):.2f}")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    database_path = "/home/ubuntu/english_letters_extracted.json"
    
    # Test with a letter
    letter_results = run_semantic_demonstration(
        database_path=database_path,
        target="A",
        is_letter=True
    )
    
    # Test with a word
    word_results = run_semantic_demonstration(
        database_path=database_path,
        target="WAVE",
        is_word=True
    )
    
    # Test with a concept
    concept_results = run_semantic_demonstration(
        database_path=database_path,
        target="Movement",
        is_concept=True
    )
