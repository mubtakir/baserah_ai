#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Equation Evolution Service (EEE) for Basira System

This module provides the core evolutionary mechanisms for evolving various types of equations
in the Basira System, with initial focus on evolving shape representations parsed by
AdvancedShapeEquationParser and stored in Thing objects.

Author: Basira System Development Team
Version: 1.0.0
"""

import random
import copy
import math
import statistics
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union, Type, Set
from collections import deque
import colorsys


class EvolutionaryOperator(ABC):
    """
    Abstract base class for all evolutionary operators.
    
    An evolutionary operator represents a single specific evolution operation
    (such as mutating a numeric parameter, adding a shape component, etc.)
    that can be applied to a target representation.
    """
    
    @abstractmethod
    def apply(self, target_representation: Any, context: Optional[Dict] = None) -> Any:
        """
        Apply the evolutionary operation to the target representation.
        
        Args:
            target_representation: The representation to evolve (e.g., parsed_shape_representation)
            context: Optional dictionary with contextual information to guide the evolution
            
        Returns:
            The evolved representation
        """
        pass
    
    @abstractmethod
    def get_operator_type(self) -> str:
        """
        Get the type of this evolutionary operator.
        
        Returns:
            A string identifying the operator type
        """
        pass


class NumericParameterMutator(EvolutionaryOperator):
    """
    Evolutionary operator that mutates numeric parameters in a shape representation.
    
    This operator can modify numeric values in params, style, or animation properties
    of a shape component.
    """
    
    def __init__(self, mutation_strength: float = 0.1, min_value: Optional[float] = None, 
                 max_value: Optional[float] = None):
        """
        Initialize a NumericParameterMutator.
        
        Args:
            mutation_strength: The base strength of mutations (0.0 to 1.0)
            min_value: Optional minimum value for the parameter
            max_value: Optional maximum value for the parameter
        """
        self.mutation_strength = mutation_strength
        self.min_value = min_value
        self.max_value = max_value
    
    def apply(self, target_representation: List[Dict[str, Any]], 
              context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Apply numeric parameter mutation to a shape representation.
        
        Args:
            target_representation: The shape representation to evolve
            context: Optional dictionary with contextual information
                May include:
                - 'mutation_scale': A factor to adjust mutation strength
                - 'target_params': List of specific parameters to target
                - 'component_index': Index of a specific component to mutate
            
        Returns:
            The evolved shape representation
        """
        # Make a deep copy to avoid modifying the original
        evolved_representation = copy.deepcopy(target_representation)
        
        # Extract context information
        mutation_scale = context.get('mutation_scale', 1.0) if context else 1.0
        target_params = context.get('target_params', None) if context else None
        component_index = context.get('component_index', None) if context else None
        
        # Adjust mutation strength based on scale
        effective_strength = self.mutation_strength * mutation_scale
        
        # Determine which components to mutate
        components_to_mutate = []
        if component_index is not None:
            # Mutate only the specified component if it exists
            if 0 <= component_index < len(evolved_representation):
                components_to_mutate = [component_index]
        else:
            # Mutate all components
            components_to_mutate = list(range(len(evolved_representation)))
        
        # Apply mutation to selected components
        for idx in components_to_mutate:
            component = evolved_representation[idx]
            
            # Mutate params
            if 'params' in component:
                self._mutate_numeric_dict(component['params'], effective_strength, target_params)
            
            # Mutate style properties
            if 'style' in component:
                self._mutate_numeric_dict(component['style'], effective_strength, target_params)
            
            # Mutate animation properties
            if 'animation' in component:
                self._mutate_animation(component['animation'], effective_strength, target_params)
        
        return evolved_representation
    
    def _mutate_numeric_dict(self, properties: Dict[str, Any], strength: float, 
                            target_params: Optional[List[str]] = None) -> None:
        """
        Mutate numeric values in a dictionary of properties.
        
        Args:
            properties: Dictionary of properties to mutate
            strength: Mutation strength
            target_params: Optional list of specific parameters to target
        """
        for key, value in properties.items():
            # Skip if we have target_params and this key is not in it
            if target_params and key not in target_params:
                continue
            
            # Only mutate numeric values
            if isinstance(value, (int, float)):
                # Determine mutation range based on value magnitude
                mutation_range = abs(value) * strength if value != 0 else strength
                
                # Apply mutation
                delta = random.uniform(-mutation_range, mutation_range)
                new_value = value + delta
                
                # Apply bounds if specified
                if self.min_value is not None:
                    new_value = max(new_value, self.min_value)
                if self.max_value is not None:
                    new_value = min(new_value, self.max_value)
                
                # Update the value
                properties[key] = new_value
    
    def _mutate_animation(self, animation: Dict[str, Any], strength: float,
                         target_params: Optional[List[str]] = None) -> None:
        """
        Mutate animation properties.
        
        Args:
            animation: Animation dictionary to mutate
            strength: Mutation strength
            target_params: Optional list of specific parameters to target
        """
        # Mutate keyframes if they exist
        if 'keyframes' in animation:
            for param_name, keyframes in animation['keyframes'].items():
                # Skip if we have target_params and this param is not in it
                if target_params and param_name not in target_params:
                    continue
                
                # Mutate each keyframe's value if it's numeric
                for keyframe in keyframes:
                    if 'value' in keyframe and isinstance(keyframe['value'], (int, float)):
                        mutation_range = abs(keyframe['value']) * strength if keyframe['value'] != 0 else strength
                        delta = random.uniform(-mutation_range, mutation_range)
                        keyframe['value'] += delta
                        
                        # Apply bounds if specified
                        if self.min_value is not None:
                            keyframe['value'] = max(keyframe['value'], self.min_value)
                        if self.max_value is not None:
                            keyframe['value'] = min(keyframe['value'], self.max_value)
    
    def get_operator_type(self) -> str:
        """
        Get the type of this evolutionary operator.
        
        Returns:
            The operator type as a string
        """
        return "NumericParameterMutator"


class ColorMutator(EvolutionaryOperator):
    """
    Evolutionary operator that mutates color values in a shape representation.
    
    This operator can modify color values in style or animation properties
    of a shape component.
    """
    
    def __init__(self, mutation_strength: float = 0.1):
        """
        Initialize a ColorMutator.
        
        Args:
            mutation_strength: The base strength of mutations (0.0 to 1.0)
        """
        self.mutation_strength = mutation_strength
    
    def apply(self, target_representation: List[Dict[str, Any]], 
              context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Apply color mutation to a shape representation.
        
        Args:
            target_representation: The shape representation to evolve
            context: Optional dictionary with contextual information
                May include:
                - 'mutation_scale': A factor to adjust mutation strength
                - 'component_index': Index of a specific component to mutate
                - 'color_properties': List of specific color properties to target
                  (e.g., ['color', 'fill_color'])
            
        Returns:
            The evolved shape representation
        """
        # Make a deep copy to avoid modifying the original
        evolved_representation = copy.deepcopy(target_representation)
        
        # Extract context information
        mutation_scale = context.get('mutation_scale', 1.0) if context else 1.0
        component_index = context.get('component_index', None) if context else None
        color_properties = context.get('color_properties', ['color', 'fill_color']) if context else ['color', 'fill_color']
        
        # Adjust mutation strength based on scale
        effective_strength = self.mutation_strength * mutation_scale
        
        # Determine which components to mutate
        components_to_mutate = []
        if component_index is not None:
            # Mutate only the specified component if it exists
            if 0 <= component_index < len(evolved_representation):
                components_to_mutate = [component_index]
        else:
            # Mutate all components
            components_to_mutate = list(range(len(evolved_representation)))
        
        # Apply mutation to selected components
        for idx in components_to_mutate:
            component = evolved_representation[idx]
            
            # Mutate style colors
            if 'style' in component:
                self._mutate_colors_in_dict(component['style'], effective_strength, color_properties)
            
            # Mutate animation colors
            if 'animation' in component and 'keyframes' in component['animation']:
                for param_name, keyframes in component['animation']['keyframes'].items():
                    if param_name in color_properties:
                        for keyframe in keyframes:
                            if 'value' in keyframe and isinstance(keyframe['value'], str) and keyframe['value'].startswith('#'):
                                keyframe['value'] = self._mutate_color(keyframe['value'], effective_strength)
        
        return evolved_representation
    
    def _mutate_colors_in_dict(self, properties: Dict[str, Any], strength: float, 
                              color_properties: List[str]) -> None:
        """
        Mutate color values in a dictionary of properties.
        
        Args:
            properties: Dictionary of properties to mutate
            strength: Mutation strength
            color_properties: List of property names that represent colors
        """
        for key, value in properties.items():
            if key in color_properties and isinstance(value, str) and value.startswith('#'):
                properties[key] = self._mutate_color(value, strength)
    
    def _mutate_color(self, color_hex: str, strength: float) -> str:
        """
        Mutate a color value in hex format.
        
        Args:
            color_hex: Color in hex format (e.g., '#FF0000')
            strength: Mutation strength
            
        Returns:
            Mutated color in hex format
        """
        # Convert hex to RGB
        color_hex = color_hex.lstrip('#')
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0
        
        # Convert RGB to HSV (easier to mutate)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        # Mutate HSV values
        h = (h + random.uniform(-strength, strength)) % 1.0  # Hue wraps around
        s = max(0.0, min(1.0, s + random.uniform(-strength, strength)))  # Saturation is clamped
        v = max(0.0, min(1.0, v + random.uniform(-strength, strength)))  # Value is clamped
        
        # Convert back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Convert RGB to hex
        r_hex = format(int(r * 255), '02x')
        g_hex = format(int(g * 255), '02x')
        b_hex = format(int(b * 255), '02x')
        
        return f'#{r_hex}{g_hex}{b_hex}'.upper()
    
    def get_operator_type(self) -> str:
        """
        Get the type of this evolutionary operator.
        
        Returns:
            The operator type as a string
        """
        return "ColorMutator"


class StylePropertyMutator(EvolutionaryOperator):
    """
    Evolutionary operator that mutates boolean or categorical style properties.
    
    This operator can toggle boolean properties (like fill) or select from
    categorical options (like line_cap).
    """
    
    def __init__(self, toggle_probability: float = 0.3):
        """
        Initialize a StylePropertyMutator.
        
        Args:
            toggle_probability: Probability of toggling a boolean property
        """
        self.toggle_probability = toggle_probability
        
        # Define categorical property options
        self.categorical_options = {
            'line_cap': ['butt', 'round', 'square'],
            'line_join': ['miter', 'round', 'bevel'],
            'text_align': ['left', 'center', 'right'],
            'text_baseline': ['top', 'middle', 'bottom', 'alphabetic', 'hanging'],
            'font_family': ['Arial', 'Helvetica', 'Times New Roman', 'Courier New', 'Verdana']
        }
    
    def apply(self, target_representation: List[Dict[str, Any]], 
              context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Apply style property mutation to a shape representation.
        
        Args:
            target_representation: The shape representation to evolve
            context: Optional dictionary with contextual information
                May include:
                - 'component_index': Index of a specific component to mutate
                - 'target_properties': List of specific properties to target
            
        Returns:
            The evolved shape representation
        """
        # Make a deep copy to avoid modifying the original
        evolved_representation = copy.deepcopy(target_representation)
        
        # Extract context information
        component_index = context.get('component_index', None) if context else None
        target_properties = context.get('target_properties', None) if context else None
        
        # Determine which components to mutate
        components_to_mutate = []
        if component_index is not None:
            # Mutate only the specified component if it exists
            if 0 <= component_index < len(evolved_representation):
                components_to_mutate = [component_index]
        else:
            # Mutate all components
            components_to_mutate = list(range(len(evolved_representation)))
        
        # Apply mutation to selected components
        for idx in components_to_mutate:
            component = evolved_representation[idx]
            
            # Mutate style properties
            if 'style' in component:
                self._mutate_style_properties(component['style'], target_properties)
        
        return evolved_representation
    
    def _mutate_style_properties(self, style: Dict[str, Any], 
                                target_properties: Optional[List[str]] = None) -> None:
        """
        Mutate style properties in a style dictionary.
        
        Args:
            style: Style dictionary to mutate
            target_properties: Optional list of specific properties to target
        """
        # Boolean properties that can be toggled
        boolean_properties = ['fill', 'stroke', 'visible', 'closed']
        
        # Process each property in the style dictionary
        for key, value in list(style.items()):  # Use list() to avoid dictionary size change during iteration
            # Skip if we have target_properties and this key is not in it
            if target_properties and key not in target_properties:
                continue
            
            # Toggle boolean properties
            if key in boolean_properties and isinstance(value, bool):
                if random.random() < self.toggle_probability:
                    style[key] = not value
            
            # Mutate categorical properties
            elif key in self.categorical_options and isinstance(value, str):
                options = self.categorical_options[key]
                if value in options:
                    # Choose a different option
                    other_options = [opt for opt in options if opt != value]
                    if other_options:
                        style[key] = random.choice(other_options)
    
    def get_operator_type(self) -> str:
        """
        Get the type of this evolutionary operator.
        
        Returns:
            The operator type as a string
        """
        return "StylePropertyMutator"


class AddComponentOperator(EvolutionaryOperator):
    """
    Evolutionary operator that adds a new shape component to a representation.
    
    This operator creates a new shape component with default or random parameters
    and adds it to the representation.
    """
    
    def __init__(self):
        """Initialize an AddComponentOperator."""
        # Define available shape types and their required parameters
        self.shape_types = {
            'circle': ['cx', 'cy', 'radius'],
            'rectangle': ['x', 'y', 'width', 'height'],
            'line': ['x1', 'y1', 'x2', 'y2'],
            'path': ['commands'],
            'ellipse': ['cx', 'cy', 'rx', 'ry'],
            'polygon': ['points'],
            'text': ['x', 'y', 'text']
        }
        
        # Define default style properties
        self.default_styles = {
            'color': '#000000',
            'fill_color': '#FFFFFF',
            'fill': True,
            'stroke': True,
            'line_width': 1.0,
            'opacity': 1.0
        }
    
    def apply(self, target_representation: List[Dict[str, Any]], 
              context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Apply component addition to a shape representation.
        
        Args:
            target_representation: The shape representation to evolve
            context: Optional dictionary with contextual information
                May include:
                - 'shape_type': Specific shape type to add
                - 'position': Suggested position for the new component
                - 'size': Suggested size for the new component
                - 'style': Style properties for the new component
            
        Returns:
            The evolved shape representation with a new component
        """
        # Make a deep copy to avoid modifying the original
        evolved_representation = copy.deepcopy(target_representation)
        
        # Extract context information
        shape_type = context.get('shape_type', None) if context else None
        position = context.get('position', None) if context else None
        size = context.get('size', None) if context else None
        style_override = context.get('style', {}) if context else {}
        
        # If no shape type is specified, choose one randomly
        if not shape_type:
            shape_type = random.choice(list(self.shape_types.keys()))
        
        # Create a new component
        new_component = self._create_component(shape_type, position, size, style_override)
        
        # Add the new component to the representation
        evolved_representation.append(new_component)
        
        return evolved_representation
    
    def _create_component(self, shape_type: str, position: Optional[Dict[str, float]], 
                         size: Optional[Dict[str, float]], style_override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new shape component.
        
        Args:
            shape_type: Type of shape to create
            position: Optional position information
            size: Optional size information
            style_override: Style properties to override defaults
            
        Returns:
            A new shape component dictionary
        """
        # Ensure shape_type is valid
        if shape_type not in self.shape_types:
            shape_type = random.choice(list(self.shape_types.keys()))
        
        # Get required parameters for this shape type
        required_params = self.shape_types[shape_type]
        
        # Create params dictionary
        params = {'shape_type': shape_type}
        
        # Set position-related parameters
        if position:
            if 'cx' in required_params:
                params['cx'] = position.get('x', random.uniform(50, 350))
                params['cy'] = position.get('y', random.uniform(50, 350))
            elif 'x' in required_params:
                params['x'] = position.get('x', random.uniform(50, 350))
                params['y'] = position.get('y', random.uniform(50, 350))
            elif 'x1' in required_params:
                params['x1'] = position.get('x', random.uniform(50, 350))
                params['y1'] = position.get('y', random.uniform(50, 350))
                params['x2'] = position.get('x', random.uniform(50, 350)) + random.uniform(20, 100)
                params['y2'] = position.get('y', random.uniform(50, 350)) + random.uniform(20, 100)
        else:
            # Generate random positions
            if 'cx' in required_params:
                params['cx'] = random.uniform(50, 350)
                params['cy'] = random.uniform(50, 350)
            elif 'x' in required_params:
                params['x'] = random.uniform(50, 350)
                params['y'] = random.uniform(50, 350)
            elif 'x1' in required_params:
                params['x1'] = random.uniform(50, 350)
                params['y1'] = random.uniform(50, 350)
                params['x2'] = random.uniform(50, 350)
                params['y2'] = random.uniform(50, 350)
        
        # Set size-related parameters
        if size:
            if 'radius' in required_params:
                params['radius'] = size.get('radius', random.uniform(10, 50))
            elif 'width' in required_params:
                params['width'] = size.get('width', random.uniform(20, 100))
                params['height'] = size.get('height', random.uniform(20, 100))
            elif 'rx' in required_params:
                params['rx'] = size.get('rx', random.uniform(10, 50))
                params['ry'] = size.get('ry', random.uniform(10, 50))
        else:
            # Generate random sizes
            if 'radius' in required_params:
                params['radius'] = random.uniform(10, 50)
            elif 'width' in required_params:
                params['width'] = random.uniform(20, 100)
                params['height'] = random.uniform(20, 100)
            elif 'rx' in required_params:
                params['rx'] = random.uniform(10, 50)
                params['ry'] = random.uniform(10, 50)
        
        # Handle special cases
        if shape_type == 'path':
            # Create a simple path (e.g., a triangle)
            x = params.get('x', random.uniform(50, 350))
            y = params.get('y', random.uniform(50, 350))
            size = params.get('width', random.uniform(20, 100))
            
            params['commands'] = [
                {'command': 'M', 'points': [x, y]},
                {'command': 'L', 'points': [x + size, y]},
                {'command': 'L', 'points': [x + size/2, y + size]},
                {'command': 'Z', 'points': []}
            ]
        
        elif shape_type == 'polygon':
            # Create a simple polygon (e.g., a triangle)
            x = params.get('x', random.uniform(50, 350))
            y = params.get('y', random.uniform(50, 350))
            size = params.get('width', random.uniform(20, 100))
            
            params['points'] = [
                [x, y],
                [x + size, y],
                [x + size/2, y + size]
            ]
        
        elif shape_type == 'text':
            params['text'] = 'Text'
        
        # Create style dictionary by combining defaults with overrides
        style = copy.deepcopy(self.default_styles)
        style.update(style_override)
        
        # Generate a random color if not specified
        if 'color' not in style_override:
            style['color'] = f'#{random.randint(0, 0xFFFFFF):06X}'
        
        if 'fill_color' not in style_override:
            style['fill_color'] = f'#{random.randint(0, 0xFFFFFF):06X}'
        
        # Create the component
        component = {
            'params': params,
            'style': style
        }
        
        return component
    
    def get_operator_type(self) -> str:
        """
        Get the type of this evolutionary operator.
        
        Returns:
            The operator type as a string
        """
        return "AddComponentOperator"


class RemoveComponentOperator(EvolutionaryOperator):
    """
    Evolutionary operator that removes a shape component from a representation.
    
    This operator selects and removes a component from the representation.
    """
    
    def apply(self, target_representation: List[Dict[str, Any]], 
              context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Apply component removal to a shape representation.
        
        Args:
            target_representation: The shape representation to evolve
            context: Optional dictionary with contextual information
                May include:
                - 'component_index': Index of a specific component to remove
                - 'importance_scores': List of importance scores for components
            
        Returns:
            The evolved shape representation with a component removed
        """
        # Make a deep copy to avoid modifying the original
        evolved_representation = copy.deepcopy(target_representation)
        
        # If there's only one or no components, return unchanged
        if len(evolved_representation) <= 1:
            return evolved_representation
        
        # Extract context information
        component_index = context.get('component_index', None) if context else None
        importance_scores = context.get('importance_scores', None) if context else None
        
        # Determine which component to remove
        if component_index is not None:
            # Remove the specified component if it exists
            if 0 <= component_index < len(evolved_representation):
                evolved_representation.pop(component_index)
        elif importance_scores:
            # Remove a component based on inverse probability of importance
            # (less important components are more likely to be removed)
            total_importance = sum(importance_scores)
            if total_importance > 0:
                # Normalize scores and invert them
                inverse_probs = [1.0 - (score / total_importance) for score in importance_scores]
                total_inverse = sum(inverse_probs)
                probs = [inv / total_inverse for inv in inverse_probs]
                
                # Choose a component to remove based on probabilities
                component_index = random.choices(range(len(evolved_representation)), weights=probs, k=1)[0]
                evolved_representation.pop(component_index)
            else:
                # If all importance scores are 0, remove a random component
                component_index = random.randrange(len(evolved_representation))
                evolved_representation.pop(component_index)
        else:
            # Remove a random component
            component_index = random.randrange(len(evolved_representation))
            evolved_representation.pop(component_index)
        
        return evolved_representation
    
    def get_operator_type(self) -> str:
        """
        Get the type of this evolutionary operator.
        
        Returns:
            The operator type as a string
        """
        return "RemoveComponentOperator"


class ChangeComponentTypeOperator(EvolutionaryOperator):
    """
    Evolutionary operator that changes the type of a shape component.
    
    This operator selects a component and changes its shape_type while
    attempting to preserve compatible parameters.
    """
    
    def __init__(self):
        """Initialize a ChangeComponentTypeOperator."""
        # Define parameter mappings between shape types
        self.parameter_mappings = {
            'circle': {
                'rectangle': lambda p: {'x': p['cx'] - p['radius'], 'y': p['cy'] - p['radius'], 
                                       'width': p['radius'] * 2, 'height': p['radius'] * 2},
                'ellipse': lambda p: {'cx': p['cx'], 'cy': p['cy'], 'rx': p['radius'], 'ry': p['radius']}
            },
            'rectangle': {
                'circle': lambda p: {'cx': p['x'] + p['width']/2, 'cy': p['y'] + p['height']/2, 
                                    'radius': min(p['width'], p['height'])/2},
                'ellipse': lambda p: {'cx': p['x'] + p['width']/2, 'cy': p['y'] + p['height']/2, 
                                     'rx': p['width']/2, 'ry': p['height']/2}
            },
            'ellipse': {
                'circle': lambda p: {'cx': p['cx'], 'cy': p['cy'], 'radius': min(p['rx'], p['ry'])},
                'rectangle': lambda p: {'x': p['cx'] - p['rx'], 'y': p['cy'] - p['ry'], 
                                       'width': p['rx'] * 2, 'height': p['ry'] * 2}
            }
        }
        
        # Define available shape types
        self.shape_types = ['circle', 'rectangle', 'ellipse', 'line', 'path', 'polygon', 'text']
    
    def apply(self, target_representation: List[Dict[str, Any]], 
              context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Apply component type change to a shape representation.
        
        Args:
            target_representation: The shape representation to evolve
            context: Optional dictionary with contextual information
                May include:
                - 'component_index': Index of a specific component to change
                - 'target_type': Specific shape type to change to
            
        Returns:
            The evolved shape representation with a changed component type
        """
        # Make a deep copy to avoid modifying the original
        evolved_representation = copy.deepcopy(target_representation)
        
        # If there are no components, return unchanged
        if not evolved_representation:
            return evolved_representation
        
        # Extract context information
        component_index = context.get('component_index', None) if context else None
        target_type = context.get('target_type', None) if context else None
        
        # Determine which component to change
        if component_index is None:
            component_index = random.randrange(len(evolved_representation))
        elif component_index >= len(evolved_representation):
            # If the specified index is out of range, choose a random one
            component_index = random.randrange(len(evolved_representation))
        
        # Get the component to change
        component = evolved_representation[component_index]
        
        # Get current shape type
        current_type = component['params'].get('shape_type', 'unknown')
        
        # Determine target shape type
        if not target_type or target_type == current_type:
            # Choose a random shape type different from the current one
            available_types = [t for t in self.shape_types if t != current_type]
            if not available_types:
                return evolved_representation  # No change possible
            target_type = random.choice(available_types)
        
        # Change the shape type
        self._change_component_type(component, current_type, target_type)
        
        return evolved_representation
    
    def _change_component_type(self, component: Dict[str, Any], 
                              current_type: str, target_type: str) -> None:
        """
        Change the type of a shape component.
        
        Args:
            component: The component to modify
            current_type: Current shape type
            target_type: Target shape type
        """
        # Update shape_type
        component['params']['shape_type'] = target_type
        
        # Try to map parameters using defined mappings
        if current_type in self.parameter_mappings and target_type in self.parameter_mappings[current_type]:
            mapping_func = self.parameter_mappings[current_type][target_type]
            new_params = mapping_func(component['params'])
            
            # Update params with mapped values
            component['params'].update(new_params)
            
            # Remove parameters that are not relevant to the new shape type
            keys_to_remove = []
            for key in component['params']:
                if key != 'shape_type' and key not in new_params:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del component['params'][key]
        
        else:
            # Handle cases without defined mappings
            # This is a simplified approach; in a real system, more sophisticated
            # parameter mapping would be needed
            
            # Get position information from current parameters
            x, y = self._extract_position(component['params'], current_type)
            
            # Create new parameters based on target type
            if target_type == 'circle':
                component['params'] = {
                    'shape_type': 'circle',
                    'cx': x,
                    'cy': y,
                    'radius': random.uniform(10, 50)
                }
            elif target_type == 'rectangle':
                component['params'] = {
                    'shape_type': 'rectangle',
                    'x': x - random.uniform(10, 50),
                    'y': y - random.uniform(10, 50),
                    'width': random.uniform(20, 100),
                    'height': random.uniform(20, 100)
                }
            elif target_type == 'ellipse':
                component['params'] = {
                    'shape_type': 'ellipse',
                    'cx': x,
                    'cy': y,
                    'rx': random.uniform(10, 50),
                    'ry': random.uniform(10, 50)
                }
            elif target_type == 'line':
                component['params'] = {
                    'shape_type': 'line',
                    'x1': x,
                    'y1': y,
                    'x2': x + random.uniform(20, 100),
                    'y2': y + random.uniform(20, 100)
                }
            elif target_type == 'path':
                size = random.uniform(20, 100)
                component['params'] = {
                    'shape_type': 'path',
                    'commands': [
                        {'command': 'M', 'points': [x, y]},
                        {'command': 'L', 'points': [x + size, y]},
                        {'command': 'L', 'points': [x + size/2, y + size]},
                        {'command': 'Z', 'points': []}
                    ]
                }
            elif target_type == 'polygon':
                size = random.uniform(20, 100)
                component['params'] = {
                    'shape_type': 'polygon',
                    'points': [
                        [x, y],
                        [x + size, y],
                        [x + size/2, y + size]
                    ]
                }
            elif target_type == 'text':
                component['params'] = {
                    'shape_type': 'text',
                    'x': x,
                    'y': y,
                    'text': 'Text'
                }
    
    def _extract_position(self, params: Dict[str, Any], shape_type: str) -> Tuple[float, float]:
        """
        Extract a representative position from shape parameters.
        
        Args:
            params: Shape parameters
            shape_type: Shape type
            
        Returns:
            A tuple (x, y) representing a position
        """
        if shape_type == 'circle' or shape_type == 'ellipse':
            return params.get('cx', 0), params.get('cy', 0)
        elif shape_type == 'rectangle':
            return params.get('x', 0) + params.get('width', 0)/2, params.get('y', 0) + params.get('height', 0)/2
        elif shape_type == 'line':
            return params.get('x1', 0), params.get('y1', 0)
        elif shape_type == 'path' and 'commands' in params and params['commands']:
            # Get the first point in the path
            first_command = params['commands'][0]
            if 'points' in first_command and len(first_command['points']) >= 2:
                return first_command['points'][0], first_command['points'][1]
        elif shape_type == 'polygon' and 'points' in params and params['points']:
            # Get the first point in the polygon
            first_point = params['points'][0]
            if len(first_point) >= 2:
                return first_point[0], first_point[1]
        elif shape_type == 'text':
            return params.get('x', 0), params.get('y', 0)
        
        # Default position if we can't determine one
        return 0, 0
    
    def get_operator_type(self) -> str:
        """
        Get the type of this evolutionary operator.
        
        Returns:
            The operator type as a string
        """
        return "ChangeComponentTypeOperator"


class EquationEvolutionEngine:
    """
    Main engine for evolving shape equations and other representations.
    
    This class provides functionality for applying evolutionary operations
    to shape representations based on fitness scores and other contextual
    information.
    """
    
    def __init__(self, mutation_power: float = 0.1, history_size: int = 10,
                cooldown_period: int = 5, add_component_threshold: float = 0.7,
                prune_component_threshold: float = 0.3,
                operator_probabilities: Optional[Dict[str, float]] = None):
        """
        Initialize an EquationEvolutionEngine.
        
        Args:
            mutation_power: Base power of mutations (0.0 to 1.0)
            history_size: Number of fitness scores to keep in history
            cooldown_period: Steps to wait before applying structural changes again
            add_component_threshold: Percentile threshold for adding components
            prune_component_threshold: Percentile threshold for removing components
            operator_probabilities: Dictionary mapping operator types to probabilities
        """
        self.mutation_power = mutation_power
        self.history_size = history_size
        self.cooldown_period = cooldown_period
        self.add_component_threshold = add_component_threshold
        self.prune_component_threshold = prune_component_threshold
        
        # Initialize performance history
        self.performance_history = deque(maxlen=history_size)
        
        # Initialize cooldown counters
        self.last_structural_change_step = -cooldown_period
        
        # Set up operator probabilities
        self.operator_probabilities = operator_probabilities or {
            "NumericParameterMutator": 0.5,
            "ColorMutator": 0.2,
            "StylePropertyMutator": 0.1,
            "AddComponentOperator": 0.1,
            "RemoveComponentOperator": 0.05,
            "ChangeComponentTypeOperator": 0.05
        }
        
        # Initialize operators
        self.operators = {
            "NumericParameterMutator": NumericParameterMutator(mutation_strength=mutation_power),
            "ColorMutator": ColorMutator(mutation_strength=mutation_power),
            "StylePropertyMutator": StylePropertyMutator(),
            "AddComponentOperator": AddComponentOperator(),
            "RemoveComponentOperator": RemoveComponentOperator(),
            "ChangeComponentTypeOperator": ChangeComponentTypeOperator()
        }
    
    def evolve_representation(self, current_representation: List[Dict[str, Any]], 
                             fitness_score: float, current_step: int,
                             evolution_context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Evolve a shape representation based on fitness score and context.
        
        Args:
            current_representation: The current shape representation to evolve
            fitness_score: The fitness score of the current representation
            current_step: The current step in the evolution process
            evolution_context: Optional dictionary with contextual information
            
        Returns:
            The evolved shape representation
        """
        # Update performance history
        self.performance_history.append(fitness_score)
        
        # Calculate percentile of current fitness
        percentile = self._calculate_percentile(fitness_score)
        
        # Determine if structural change is needed
        structural_change_allowed = (current_step - self.last_structural_change_step) >= self.cooldown_period
        
        # Create a copy of the representation to evolve
        evolved_representation = copy.deepcopy(current_representation)
        
        # Apply structural changes if needed and allowed
        if structural_change_allowed:
            if percentile > self.add_component_threshold:
                # High performance, try adding a component
                evolved_representation = self.operators["AddComponentOperator"].apply(
                    evolved_representation, evolution_context)
                self.last_structural_change_step = current_step
            elif percentile < self.prune_component_threshold and len(current_representation) > 1:
                # Low performance, try removing a component
                evolved_representation = self.operators["RemoveComponentOperator"].apply(
                    evolved_representation, evolution_context)
                self.last_structural_change_step = current_step
        
        # Apply non-structural mutations
        # Calculate dynamic mutation scale based on performance
        mutation_scale = self._dynamic_mutation_scale(percentile)
        
        # Create context for mutations
        mutation_context = evolution_context.copy() if evolution_context else {}
        mutation_context['mutation_scale'] = mutation_scale
        
        # Apply mutations to each component
        for component_idx in range(len(evolved_representation)):
            # Set component index in context
            mutation_context['component_index'] = component_idx
            
            # Decide which operators to apply
            operators_to_apply = self._select_operators(evolved_representation[component_idx])
            
            # Apply selected operators
            for operator_type in operators_to_apply:
                if operator_type in self.operators:
                    evolved_representation = self.operators[operator_type].apply(
                        evolved_representation, mutation_context)
        
        return evolved_representation
    
    def _calculate_percentile(self, current_fitness: float) -> float:
        """
        Calculate the percentile of the current fitness in the history.
        
        Args:
            current_fitness: The current fitness score
            
        Returns:
            The percentile (0.0 to 1.0) of the current fitness
        """
        if not self.performance_history:
            return 0.5  # Default to middle percentile if no history
        
        # Convert deque to list for calculations
        history_list = list(self.performance_history)
        
        # Count how many history items are less than current fitness
        count_less = sum(1 for x in history_list if x < current_fitness)
        
        # Calculate percentile
        percentile = count_less / len(history_list)
        
        return percentile
    
    def _dynamic_mutation_scale(self, percentile: float) -> float:
        """
        Calculate a dynamic mutation scale based on performance percentile.
        
        Higher percentiles (better performance) lead to smaller mutations.
        Lower percentiles (worse performance) lead to larger mutations.
        
        Args:
            percentile: The performance percentile (0.0 to 1.0)
            
        Returns:
            A mutation scale factor
        """
        # Invert percentile and add small constant to avoid zero
        inverse_percentile = 1.0 - percentile + 0.1
        
        # Apply non-linear scaling (e.g., square root for less aggressive scaling)
        scale = math.sqrt(inverse_percentile)
        
        # Ensure scale is within reasonable bounds
        scale = max(0.1, min(2.0, scale))
        
        return scale
    
    def _select_operators(self, component: Dict[str, Any]) -> List[str]:
        """
        Select which operators to apply to a component.
        
        Args:
            component: The component to evolve
            
        Returns:
            A list of operator types to apply
        """
        # Start with non-structural operators
        candidate_operators = ["NumericParameterMutator", "ColorMutator", "StylePropertyMutator"]
        
        # Select operators based on probabilities
        selected_operators = []
        for operator_type in candidate_operators:
            if random.random() < self.operator_probabilities.get(operator_type, 0.0):
                selected_operators.append(operator_type)
        
        # Ensure at least one operator is selected
        if not selected_operators:
            selected_operators = [random.choices(
                candidate_operators, 
                weights=[self.operator_probabilities.get(op, 0.1) for op in candidate_operators],
                k=1
            )[0]]
        
        return selected_operators
    
    def get_performance_history(self) -> List[float]:
        """
        Get the performance history.
        
        Returns:
            A list of recent fitness scores
        """
        return list(self.performance_history)
    
    def reset_history(self) -> None:
        """Reset the performance history and cooldown counters."""
        self.performance_history.clear()
        self.last_structural_change_step = -self.cooldown_period


# Example usage
if __name__ == "__main__":
    # Create an evolution engine
    engine = EquationEvolutionEngine(
        mutation_power=0.2,
        history_size=10,
        cooldown_period=3,
        add_component_threshold=0.7,
        prune_component_threshold=0.3
    )
    
    # Example shape representation (simplified)
    shape_representation = [
        {
            "params": {
                "shape_type": "circle",
                "cx": 100,
                "cy": 150,
                "radius": 50
            },
            "style": {
                "color": "#FF0000",
                "fill_color": "#FFCCCC",
                "fill": True,
                "stroke": True,
                "line_width": 2,
                "opacity": 0.8
            }
        },
        {
            "params": {
                "shape_type": "rectangle",
                "x": 200,
                "y": 100,
                "width": 80,
                "height": 60
            },
            "style": {
                "color": "#0000FF",
                "fill_color": "#CCCCFF",
                "fill": True,
                "stroke": True,
                "line_width": 1,
                "opacity": 1.0
            }
        }
    ]
    
    # Evolve the representation
    evolved_representation = engine.evolve_representation(
        current_representation=shape_representation,
        fitness_score=0.6,
        current_step=1,
        evolution_context={"goal": "increase_complexity"}
    )
    
    # Print the evolved representation
    print("Original representation had", len(shape_representation), "components")
    print("Evolved representation has", len(evolved_representation), "components")
    
    # Print details of the first component before and after evolution
    if shape_representation and evolved_representation:
        print("\nOriginal first component:")
        print(f"Type: {shape_representation[0]['params']['shape_type']}")
        print(f"Position: ({shape_representation[0]['params'].get('cx', 'N/A')}, {shape_representation[0]['params'].get('cy', 'N/A')})")
        print(f"Color: {shape_representation[0]['style']['color']}")
        
        print("\nEvolved first component:")
        print(f"Type: {evolved_representation[0]['params']['shape_type']}")
        print(f"Position: ({evolved_representation[0]['params'].get('cx', 'N/A')}, {evolved_representation[0]['params'].get('cy', 'N/A')})")
        print(f"Color: {evolved_representation[0]['style']['color']}")
