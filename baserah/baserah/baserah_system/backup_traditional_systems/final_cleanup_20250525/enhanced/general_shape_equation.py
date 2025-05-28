#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General Shape Equation Module for Basira System

This module extends the mathematical foundation of the Basira System by implementing
the General Shape Equation (GSE) framework, which provides a unified mathematical
representation for shapes, patterns, and their transformations. The GSE framework
is designed to be extensible and compatible with deep learning and reinforcement
learning approaches.

The module serves as a bridge between symbolic representations and learning-based
approaches, enabling the system to evolve equations through both rule-based and
learning-based methods.

Author: Basira System Development Team
Version: 1.0.0
"""

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, Set
from dataclasses import dataclass, field
import copy
import math
from enum import Enum
import json
import os
import sys

# Import from parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mathematical_foundation import SymbolicExpression, ShapeEquationMathAnalyzer


class EquationType(str, Enum):
    """Types of equations supported in the General Shape Equation framework."""
    SHAPE = "shape"  # Equations describing geometric shapes
    PATTERN = "pattern"  # Equations describing patterns or textures
    BEHAVIOR = "behavior"  # Equations describing dynamic behaviors
    TRANSFORMATION = "transformation"  # Equations describing transformations
    CONSTRAINT = "constraint"  # Equations describing constraints or boundaries
    COMPOSITE = "composite"  # Composite equations combining multiple types


class LearningMode(str, Enum):
    """Learning modes for equation evolution."""
    NONE = "none"  # No learning, pure symbolic evolution
    SUPERVISED = "supervised"  # Supervised learning with labeled examples
    REINFORCEMENT = "reinforcement"  # Reinforcement learning with rewards
    UNSUPERVISED = "unsupervised"  # Unsupervised learning for pattern discovery
    HYBRID = "hybrid"  # Hybrid approach combining multiple modes


@dataclass
class EquationMetadata:
    """Metadata for a General Shape Equation."""
    equation_id: str  # Unique identifier for the equation
    equation_type: EquationType  # Type of equation
    creation_time: str  # ISO format timestamp of creation
    last_modified: str  # ISO format timestamp of last modification
    version: int = 1  # Version number
    author: str = "Basira System"  # Author or creator
    description: Optional[str] = None  # Optional description
    tags: List[str] = field(default_factory=list)  # Tags for categorization
    confidence: float = 1.0  # Confidence level (0.0 to 1.0)
    complexity: float = 0.0  # Complexity score
    semantic_links: Dict[str, Any] = field(default_factory=dict)  # Links to semantic concepts
    custom_properties: Dict[str, Any] = field(default_factory=dict)  # Custom metadata properties


class GeneralShapeEquation:
    """
    General Shape Equation (GSE) class that provides a unified mathematical
    representation for shapes, patterns, and their transformations.
    
    The GSE framework extends the basic SymbolicExpression to support:
    1. Multiple equation components (shape, behavior, constraint, etc.)
    2. Integration with deep learning and reinforcement learning
    3. Semantic interpretation and evolution
    4. Expert/explorer interaction model
    """
    
    def __init__(self, 
                 symbolic_components: Optional[Dict[str, SymbolicExpression]] = None,
                 equation_type: EquationType = EquationType.SHAPE,
                 metadata: Optional[EquationMetadata] = None,
                 learning_mode: LearningMode = LearningMode.NONE):
        """
        Initialize a new General Shape Equation.
        
        Args:
            symbolic_components: Dictionary mapping component names to SymbolicExpression objects
            equation_type: Type of equation (shape, pattern, behavior, etc.)
            metadata: Metadata for the equation
            learning_mode: Learning mode for equation evolution
        """
        # Initialize symbolic components
        self.symbolic_components = symbolic_components or {}
        
        # Set equation type
        self.equation_type = equation_type if isinstance(equation_type, EquationType) else EquationType(equation_type)
        
        # Initialize metadata
        from datetime import datetime
        import uuid
        
        current_time = datetime.now().isoformat()
        self.metadata = metadata or EquationMetadata(
            equation_id=str(uuid.uuid4()),
            equation_type=self.equation_type,
            creation_time=current_time,
            last_modified=current_time
        )
        
        # Set learning mode
        self.learning_mode = learning_mode if isinstance(learning_mode, LearningMode) else LearningMode(learning_mode)
        
        # Initialize neural network components (if using learning modes)
        self.neural_components = {}
        if self.learning_mode != LearningMode.NONE:
            self._initialize_neural_components()
        
        # Initialize variables dictionary (union of all component variables)
        self.variables = self._collect_variables()
        
        # Initialize parameter space for exploration
        self.parameter_space = self._define_parameter_space()
        
        # Initialize history for tracking evolution
        self.evolution_history = []
        
        # Calculate initial complexity
        self._update_complexity()
    
    def _initialize_neural_components(self):
        """Initialize neural network components based on learning mode."""
        if self.learning_mode in [LearningMode.SUPERVISED, LearningMode.HYBRID]:
            # Simple feedforward network for supervised learning
            self.neural_components['supervised'] = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 5)
            )
        
        if self.learning_mode in [LearningMode.REINFORCEMENT, LearningMode.HYBRID]:
            # Simple policy network for reinforcement learning
            self.neural_components['policy'] = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 5)
            )
            
            # Simple value network for reinforcement learning
            self.neural_components['value'] = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 1)
            )
    
    def _collect_variables(self) -> Dict[str, sp.Symbol]:
        """
        Collect all variables from all symbolic components.
        
        Returns:
            Dictionary mapping variable names to SymPy symbols
        """
        variables = {}
        for component_name, component in self.symbolic_components.items():
            for var_name, var_symbol in component.variables.items():
                if var_name not in variables:
                    variables[var_name] = var_symbol
        return variables
    
    def _define_parameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Define the parameter space for exploration.
        
        Returns:
            Dictionary mapping parameter names to their properties
            (type, range, constraints, etc.)
        """
        parameter_space = {}
        
        # For each variable, define a default parameter space
        for var_name, var_symbol in self.variables.items():
            parameter_space[var_name] = {
                'type': 'continuous',
                'range': [-10.0, 10.0],  # Default range
                'distribution': 'uniform',
                'mutable': True,
                'semantic_meaning': None
            }
        
        return parameter_space
    
    def _update_complexity(self):
        """Update the complexity score of the equation."""
        # Calculate complexity based on symbolic components
        if self.symbolic_components:
            complexity_scores = [
                component.get_complexity_score() 
                for component in self.symbolic_components.values()
            ]
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            
            # Adjust for number of components
            component_factor = math.log(len(self.symbolic_components) + 1)
            
            # Adjust for neural components if present
            neural_factor = 1.0
            if self.neural_components:
                neural_factor = 1.5
            
            self.metadata.complexity = avg_complexity * component_factor * neural_factor
        else:
            self.metadata.complexity = 0.0
    
    def add_component(self, name: str, expression: Union[str, sp.Expr, SymbolicExpression]) -> None:
        """
        Add a new symbolic component to the equation.
        
        Args:
            name: Name of the component
            expression: The expression to add (string, SymPy expression, or SymbolicExpression)
        """
        # Convert to SymbolicExpression if needed
        if isinstance(expression, str):
            component = SymbolicExpression(expression_str=expression)
        elif isinstance(expression, sp.Expr):
            component = SymbolicExpression(sympy_obj=expression)
        elif isinstance(expression, SymbolicExpression):
            component = expression
        else:
            raise TypeError(f"Expression must be a string, SymPy expression, or SymbolicExpression, not {type(expression)}")
        
        # Add the component
        self.symbolic_components[name] = component
        
        # Update variables dictionary
        for var_name, var_symbol in component.variables.items():
            if var_name not in self.variables:
                self.variables[var_name] = var_symbol
                # Also add to parameter space
                self.parameter_space[var_name] = {
                    'type': 'continuous',
                    'range': [-10.0, 10.0],  # Default range
                    'distribution': 'uniform',
                    'mutable': True,
                    'semantic_meaning': None
                }
        
        # Update metadata
        from datetime import datetime
        self.metadata.last_modified = datetime.now().isoformat()
        self.metadata.version += 1
        
        # Update complexity
        self._update_complexity()
    
    def remove_component(self, name: str) -> bool:
        """
        Remove a symbolic component from the equation.
        
        Args:
            name: Name of the component to remove
            
        Returns:
            True if the component was removed, False if it wasn't found
        """
        if name in self.symbolic_components:
            # Remove the component
            del self.symbolic_components[name]
            
            # Update variables dictionary
            self.variables = self._collect_variables()
            
            # Update parameter space
            self.parameter_space = self._define_parameter_space()
            
            # Update metadata
            from datetime import datetime
            self.metadata.last_modified = datetime.now().isoformat()
            self.metadata.version += 1
            
            # Update complexity
            self._update_complexity()
            
            return True
        return False
    
    def get_component(self, name: str) -> Optional[SymbolicExpression]:
        """
        Get a symbolic component by name.
        
        Args:
            name: Name of the component
            
        Returns:
            The SymbolicExpression component, or None if not found
        """
        return self.symbolic_components.get(name)
    
    def evaluate(self, assignments: Dict[str, float]) -> Dict[str, Optional[float]]:
        """
        Evaluate all symbolic components with the given variable assignments.
        
        Args:
            assignments: Dictionary mapping variable names to values
            
        Returns:
            Dictionary mapping component names to their evaluated values
        """
        results = {}
        for name, component in self.symbolic_components.items():
            results[name] = component.evaluate(assignments)
        return results
    
    def simplify(self) -> 'GeneralShapeEquation':
        """
        Create a new GeneralShapeEquation with all components simplified.
        
        Returns:
            A new GeneralShapeEquation with simplified components
        """
        simplified_components = {}
        for name, component in self.symbolic_components.items():
            simplified_components[name] = component.simplify()
        
        # Create a new equation with the simplified components
        new_equation = GeneralShapeEquation(
            symbolic_components=simplified_components,
            equation_type=self.equation_type,
            metadata=copy.deepcopy(self.metadata),
            learning_mode=self.learning_mode
        )
        
        # Update metadata
        from datetime import datetime
        new_equation.metadata.last_modified = datetime.now().isoformat()
        new_equation.metadata.version = self.metadata.version + 1
        
        return new_equation
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the equation to a dictionary representation.
        
        Returns:
            Dictionary representation of the equation
        """
        result = {
            "equation_type": self.equation_type.value,
            "learning_mode": self.learning_mode.value,
            "metadata": {
                "equation_id": self.metadata.equation_id,
                "equation_type": self.metadata.equation_type.value,
                "creation_time": self.metadata.creation_time,
                "last_modified": self.metadata.last_modified,
                "version": self.metadata.version,
                "author": self.metadata.author,
                "complexity": self.metadata.complexity,
                "confidence": self.metadata.confidence
            },
            "symbolic_components": {}
        }
        
        # Add optional metadata fields if present
        if self.metadata.description:
            result["metadata"]["description"] = self.metadata.description
        
        if self.metadata.tags:
            result["metadata"]["tags"] = self.metadata.tags
        
        if self.metadata.semantic_links:
            result["metadata"]["semantic_links"] = self.metadata.semantic_links
        
        if self.metadata.custom_properties:
            result["metadata"]["custom_properties"] = self.metadata.custom_properties
        
        # Add symbolic components
        for name, component in self.symbolic_components.items():
            result["symbolic_components"][name] = {
                "expression": component.to_string(),
                "variables": list(component.variables.keys())
            }
        
        # Add parameter space
        result["parameter_space"] = self.parameter_space
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneralShapeEquation':
        """
        Create a GeneralShapeEquation from a dictionary representation.
        
        Args:
            data: Dictionary representation of the equation
            
        Returns:
            A new GeneralShapeEquation instance
        """
        # Extract metadata
        metadata_dict = data.get("metadata", {})
        metadata = EquationMetadata(
            equation_id=metadata_dict.get("equation_id", ""),
            equation_type=EquationType(metadata_dict.get("equation_type", EquationType.SHAPE.value)),
            creation_time=metadata_dict.get("creation_time", ""),
            last_modified=metadata_dict.get("last_modified", ""),
            version=metadata_dict.get("version", 1),
            author=metadata_dict.get("author", "Basira System"),
            description=metadata_dict.get("description"),
            tags=metadata_dict.get("tags", []),
            confidence=metadata_dict.get("confidence", 1.0),
            complexity=metadata_dict.get("complexity", 0.0),
            semantic_links=metadata_dict.get("semantic_links", {}),
            custom_properties=metadata_dict.get("custom_properties", {})
        )
        
        # Extract symbolic components
        symbolic_components = {}
        for name, component_data in data.get("symbolic_components", {}).items():
            expression_str = component_data.get("expression", "")
            symbolic_components[name] = SymbolicExpression(expression_str=expression_str)
        
        # Create the equation
        equation = cls(
            symbolic_components=symbolic_components,
            equation_type=EquationType(data.get("equation_type", EquationType.SHAPE.value)),
            metadata=metadata,
            learning_mode=LearningMode(data.get("learning_mode", LearningMode.NONE.value))
        )
        
        # Set parameter space if present
        if "parameter_space" in data:
            equation.parameter_space = data["parameter_space"]
        
        return equation
    
    def to_json(self) -> str:
        """
        Convert the equation to a JSON string.
        
        Returns:
            JSON string representation of the equation
        """
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'GeneralShapeEquation':
        """
        Create a GeneralShapeEquation from a JSON string.
        
        Args:
            json_str: JSON string representation of the equation
            
        Returns:
            A new GeneralShapeEquation instance
        """
        import json
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save the equation to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'GeneralShapeEquation':
        """
        Load a GeneralShapeEquation from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            A new GeneralShapeEquation instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        return cls.from_json(json_str)
    
    def clone(self) -> 'GeneralShapeEquation':
        """
        Create a deep copy of this equation.
        
        Returns:
            A new GeneralShapeEquation that is a deep copy of this one
        """
        # Create a new equation with deep copies of all components
        cloned_components = {}
        for name, component in self.symbolic_components.items():
            # Create a new SymbolicExpression with the same expression
            cloned_components[name] = SymbolicExpression(
                expression_str=component.to_string()
            )
        
        # Create a new equation
        cloned_equation = GeneralShapeEquation(
            symbolic_components=cloned_components,
            equation_type=self.equation_type,
            metadata=copy.deepcopy(self.metadata),
            learning_mode=self.learning_mode
        )
        
        # Copy parameter space
        cloned_equation.parameter_space = copy.deepcopy(self.parameter_space)
        
        # Update metadata
        from datetime import datetime
        cloned_equation.metadata.equation_id = str(uuid.uuid4())  # New ID
        cloned_equation.metadata.creation_time = datetime.now().isoformat()
        cloned_equation.metadata.last_modified = datetime.now().isoformat()
        
        return cloned_equation
    
    def mutate(self, mutation_strength: float = 0.1) -> 'GeneralShapeEquation':
        """
        Create a mutated version of this equation.
        
        Args:
            mutation_strength: Strength of the mutation (0.0 to 1.0)
            
        Returns:
            A new GeneralShapeEquation with mutations applied
        """
        # Clone the equation first
        mutated_equation = self.clone()
        
        # Apply mutations to the symbolic components
        # This is a placeholder for more sophisticated mutation logic
        for name, component in mutated_equation.symbolic_components.items():
            # For now, we'll just add a small random term to each component
            # In a real implementation, this would use more sophisticated
            # symbolic mutation techniques
            original_expr = component.sympy_expr
            
            # Get a random variable from the component
            if component.variables:
                var_name = list(component.variables.keys())[0]
                var_symbol = component.variables[var_name]
                
                # Add a small random term
                import random
                random_factor = random.uniform(-mutation_strength, mutation_strength)
                mutated_expr = original_expr + random_factor * var_symbol
                
                # Replace the component with the mutated version
                mutated_equation.symbolic_components[name] = SymbolicExpression(
                    sympy_obj=mutated_expr,
                    variables=component.variables
                )
        
        # Update metadata
        from datetime import datetime
        mutated_equation.metadata.last_modified = datetime.now().isoformat()
        mutated_equation.metadata.version += 1
        
        # Update complexity
        mutated_equation._update_complexity()
        
        return mutated_equation
    
    def crossover(self, other: 'GeneralShapeEquation') -> 'GeneralShapeEquation':
        """
        Create a new equation by crossing over this equation with another.
        
        Args:
            other: Another GeneralShapeEquation to cross over with
            
        Returns:
            A new GeneralShapeEquation resulting from the crossover
        """
        # Create a new equation with components from both parents
        crossover_components = {}
        
        # Get all component names from both parents
        all_component_names = set(self.symbolic_components.keys()) | set(other.symbolic_components.keys())
        
        # For each component, randomly choose from one parent or the other
        import random
        for name in all_component_names:
            if name in self.symbolic_components and name in other.symbolic_components:
                # If both parents have this component, randomly choose one
                if random.random() < 0.5:
                    parent_component = self.symbolic_components[name]
                else:
                    parent_component = other.symbolic_components[name]
                
                # Create a new component with the same expression
                crossover_components[name] = SymbolicExpression(
                    expression_str=parent_component.to_string()
                )
            elif name in self.symbolic_components:
                # Only this parent has the component
                parent_component = self.symbolic_components[name]
                crossover_components[name] = SymbolicExpression(
                    expression_str=parent_component.to_string()
                )
            else:
                # Only the other parent has the component
                parent_component = other.symbolic_components[name]
                crossover_components[name] = SymbolicExpression(
                    expression_str=parent_component.to_string()
                )
        
        # Create a new equation with the crossover components
        crossover_equation = GeneralShapeEquation(
            symbolic_components=crossover_components,
            equation_type=self.equation_type,  # Use this parent's type
            learning_mode=self.learning_mode  # Use this parent's learning mode
        )
        
        # Update metadata
        from datetime import datetime
        import uuid
        crossover_equation.metadata.equation_id = str(uuid.uuid4())  # New ID
        crossover_equation.metadata.creation_time = datetime.now().isoformat()
        crossover_equation.metadata.last_modified = datetime.now().isoformat()
        crossover_equation.metadata.version = 1
        crossover_equation.metadata.description = f"Crossover of equations {self.metadata.equation_id} and {other.metadata.equation_id}"
        
        # Update complexity
        crossover_equation._update_complexity()
        
        return crossover_equation
    
    def get_torch_model(self) -> Optional[nn.Module]:
        """
        Get a PyTorch model representation of this equation.
        
        This is a placeholder for more sophisticated neural network integration.
        
        Returns:
            A PyTorch model, or None if not applicable
        """
        if not self.neural_components:
            return None
        
        # For now, just return the first neural component
        for name, model in self.neural_components.items():
            return model
        
        return None
    
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Perform a single training step on the neural components.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            
        Returns:
            Loss value
        """
        if not self.neural_components or self.learning_mode == LearningMode.NONE:
            return 0.0
        
        # Simple training step for supervised learning
        if 'supervised' in self.neural_components and self.learning_mode in [LearningMode.SUPERVISED, LearningMode.HYBRID]:
            model = self.neural_components['supervised']
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            return loss.item()
        
        return 0.0
    
    def reinforcement_step(self, state: torch.Tensor, action: torch.Tensor, 
                          reward: float, next_state: torch.Tensor, done: bool) -> float:
        """
        Perform a single reinforcement learning step.
        
        Args:
            state: Current state tensor
            action: Action tensor
            reward: Reward value
            next_state: Next state tensor
            done: Whether the episode is done
            
        Returns:
            Loss value
        """
        if not self.neural_components or self.learning_mode not in [LearningMode.REINFORCEMENT, LearningMode.HYBRID]:
            return 0.0
        
        # This is a placeholder for more sophisticated RL training
        # In a real implementation, this would use proper RL algorithms
        
        # Simple value network update
        if 'value' in self.neural_components:
            value_net = self.neural_components['value']
            optimizer = torch.optim.Adam(value_net.parameters(), lr=0.001)
            
            # Predict current state value
            state_value = value_net(state)
            
            # Calculate target value (simple TD learning)
            with torch.no_grad():
                next_value = value_net(next_state) if not done else torch.tensor([0.0])
                target_value = reward + 0.99 * next_value  # 0.99 is the discount factor
            
            # Calculate loss and update
            criterion = nn.MSELoss()
            loss = criterion(state_value, target_value)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            return loss.item()
        
        return 0.0
    
    def add_semantic_link(self, concept_name: str, link_type: str, strength: float = 1.0) -> None:
        """
        Add a semantic link to the equation.
        
        Args:
            concept_name: Name of the semantic concept
            link_type: Type of semantic link
            strength: Strength of the link (0.0 to 1.0)
        """
        if 'semantic_links' not in self.metadata.semantic_links:
            self.metadata.semantic_links['semantic_links'] = []
        
        self.metadata.semantic_links['semantic_links'].append({
            'concept': concept_name,
            'type': link_type,
            'strength': strength
        })
        
        # Update metadata
        from datetime import datetime
        self.metadata.last_modified = datetime.now().isoformat()
    
    def get_semantic_links(self) -> List[Dict[str, Any]]:
        """
        Get all semantic links for this equation.
        
        Returns:
            List of semantic link dictionaries
        """
        return self.metadata.semantic_links.get('semantic_links', [])
    
    def __str__(self) -> str:
        """String representation of the equation."""
        components_str = ", ".join([
            f"{name}: {component.to_string()}"
            for name, component in self.symbolic_components.items()
        ])
        return f"GeneralShapeEquation({self.equation_type.value}, {components_str})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the equation."""
        return f"GeneralShapeEquation(id={self.metadata.equation_id}, type={self.equation_type.value}, components={len(self.symbolic_components)})"


class GSEFactory:
    """
    Factory class for creating GeneralShapeEquation instances from various sources.
    """
    
    @staticmethod
    def from_shape_equation_parser(parsed_shape: Dict[str, Any]) -> GeneralShapeEquation:
        """
        Create a GeneralShapeEquation from a parsed shape equation.
        
        Args:
            parsed_shape: Dictionary from AdvancedShapeEquationParser
            
        Returns:
            A new GeneralShapeEquation instance
        """
        # Extract shape type
        shape_type = parsed_shape.get('shape_type', 'unknown')
        
        # Create symbolic components
        symbolic_components = {}
        
        # Add basic shape parameters as components
        for param_name, param_value in parsed_shape.get('parameters', {}).items():
            # Convert to string representation
            if isinstance(param_value, (int, float)):
                expr_str = str(param_value)
            else:
                expr_str = f"'{param_value}'"  # Quote non-numeric values
            
            symbolic_components[f"param_{param_name}"] = SymbolicExpression(expression_str=expr_str)
        
        # Add style properties as components
        for style_name, style_value in parsed_shape.get('styles', {}).items():
            # Convert to string representation
            if isinstance(style_value, (int, float)):
                expr_str = str(style_value)
            elif isinstance(style_value, str) and style_value.startswith('#'):  # Color
                expr_str = f"'{style_value}'"
            else:
                expr_str = f"'{style_value}'"  # Quote non-numeric values
            
            symbolic_components[f"style_{style_name}"] = SymbolicExpression(expression_str=expr_str)
        
        # Add animations as components
        for anim_param, keyframes in parsed_shape.get('animations', {}).items():
            # For now, we'll just use the first and last keyframe to create a linear equation
            if keyframes and len(keyframes) >= 2:
                first_kf = keyframes[0]
                last_kf = keyframes[-1]
                
                # Extract times and values
                t0, v0 = first_kf
                t1, v1 = last_kf
                
                # Create a linear interpolation equation: v0 + (v1-v0)*(t-t0)/(t1-t0)
                if t1 != t0:
                    if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                        expr_str = f"{v0} + ({v1}-{v0})*(t-{t0})/({t1}-{t0})"
                        symbolic_components[f"anim_{anim_param}"] = SymbolicExpression(expression_str=expr_str)
        
        # Create metadata
        from datetime import datetime
        import uuid
        
        metadata = EquationMetadata(
            equation_id=str(uuid.uuid4()),
            equation_type=EquationType.SHAPE,
            creation_time=datetime.now().isoformat(),
            last_modified=datetime.now().isoformat(),
            description=f"Generated from {shape_type} shape equation"
        )
        
        # Create the equation
        return GeneralShapeEquation(
            symbolic_components=symbolic_components,
            equation_type=EquationType.SHAPE,
            metadata=metadata
        )
    
    @staticmethod
    def create_basic_shape(shape_type: str, **params) -> GeneralShapeEquation:
        """
        Create a basic shape equation.
        
        Args:
            shape_type: Type of shape (circle, rectangle, etc.)
            **params: Shape parameters
            
        Returns:
            A new GeneralShapeEquation instance
        """
        symbolic_components = {}
        
        if shape_type == 'circle':
            # Circle equation: (x-cx)^2 + (y-cy)^2 = r^2
            cx = params.get('cx', 0)
            cy = params.get('cy', 0)
            r = params.get('radius', 1)
            
            expr_str = f"(x-{cx})^2 + (y-{cy})^2 - {r}^2"
            symbolic_components['circle_equation'] = SymbolicExpression(expression_str=expr_str)
            
            # Add individual parameters
            symbolic_components['cx'] = SymbolicExpression(expression_str=str(cx))
            symbolic_components['cy'] = SymbolicExpression(expression_str=str(cy))
            symbolic_components['radius'] = SymbolicExpression(expression_str=str(r))
            
        elif shape_type == 'rectangle':
            # Rectangle equations: x_min <= x <= x_max, y_min <= y <= y_max
            x = params.get('x', 0)
            y = params.get('y', 0)
            width = params.get('width', 1)
            height = params.get('height', 1)
            
            x_min, x_max = x, x + width
            y_min, y_max = y, y + height
            
            symbolic_components['x_bound_lower'] = SymbolicExpression(expression_str=f"x - {x_min}")
            symbolic_components['x_bound_upper'] = SymbolicExpression(expression_str=f"{x_max} - x")
            symbolic_components['y_bound_lower'] = SymbolicExpression(expression_str=f"y - {y_min}")
            symbolic_components['y_bound_upper'] = SymbolicExpression(expression_str=f"{y_max} - y")
            
            # Add individual parameters
            symbolic_components['x'] = SymbolicExpression(expression_str=str(x))
            symbolic_components['y'] = SymbolicExpression(expression_str=str(y))
            symbolic_components['width'] = SymbolicExpression(expression_str=str(width))
            symbolic_components['height'] = SymbolicExpression(expression_str=str(height))
            
        elif shape_type == 'ellipse':
            # Ellipse equation: (x-cx)^2/rx^2 + (y-cy)^2/ry^2 = 1
            cx = params.get('cx', 0)
            cy = params.get('cy', 0)
            rx = params.get('rx', 1)
            ry = params.get('ry', 1)
            
            expr_str = f"(x-{cx})^2/{rx}^2 + (y-{cy})^2/{ry}^2 - 1"
            symbolic_components['ellipse_equation'] = SymbolicExpression(expression_str=expr_str)
            
            # Add individual parameters
            symbolic_components['cx'] = SymbolicExpression(expression_str=str(cx))
            symbolic_components['cy'] = SymbolicExpression(expression_str=str(cy))
            symbolic_components['rx'] = SymbolicExpression(expression_str=str(rx))
            symbolic_components['ry'] = SymbolicExpression(expression_str=str(ry))
            
        else:
            # For other shapes, just add the parameters as components
            for name, value in params.items():
                if isinstance(value, (int, float)):
                    expr_str = str(value)
                else:
                    expr_str = f"'{value}'"  # Quote non-numeric values
                
                symbolic_components[name] = SymbolicExpression(expression_str=expr_str)
        
        # Create metadata
        from datetime import datetime
        import uuid
        
        metadata = EquationMetadata(
            equation_id=str(uuid.uuid4()),
            equation_type=EquationType.SHAPE,
            creation_time=datetime.now().isoformat(),
            last_modified=datetime.now().isoformat(),
            description=f"Basic {shape_type} shape"
        )
        
        # Create the equation
        return GeneralShapeEquation(
            symbolic_components=symbolic_components,
            equation_type=EquationType.SHAPE,
            metadata=metadata
        )
    
    @staticmethod
    def create_composite(equations: List[GeneralShapeEquation], 
                        operation: str = 'union') -> GeneralShapeEquation:
        """
        Create a composite equation from multiple equations.
        
        Args:
            equations: List of GeneralShapeEquation instances
            operation: Operation to combine equations ('union', 'intersection', 'difference')
            
        Returns:
            A new GeneralShapeEquation instance
        """
        if not equations:
            raise ValueError("Must provide at least one equation")
        
        # Create a new equation with components from all input equations
        composite_components = {}
        
        # Track which equation each component came from
        component_sources = {}
        
        # Add components from each equation with prefixed names
        for i, eq in enumerate(equations):
            for name, component in eq.symbolic_components.items():
                composite_name = f"eq{i}_{name}"
                composite_components[composite_name] = SymbolicExpression(
                    expression_str=component.to_string()
                )
                component_sources[composite_name] = i
        
        # Add a component representing the composite operation
        if operation == 'union':
            # Union: min(eq0, eq1, ...)
            # For each equation, find a representative component
            rep_components = []
            for i, eq in enumerate(equations):
                # Try to find a main equation component
                main_component = None
                for name, component in eq.symbolic_components.items():
                    if 'equation' in name.lower():
                        main_component = component
                        break
                
                if main_component:
                    rep_components.append(f"eq{i}_{name}")
            
            if rep_components:
                # Create a min() expression for union
                expr_str = f"min({', '.join(rep_components)})"
                composite_components['composite_operation'] = SymbolicExpression(
                    expression_str=expr_str
                )
        
        # Create metadata
        from datetime import datetime
        import uuid
        
        metadata = EquationMetadata(
            equation_id=str(uuid.uuid4()),
            equation_type=EquationType.COMPOSITE,
            creation_time=datetime.now().isoformat(),
            last_modified=datetime.now().isoformat(),
            description=f"Composite equation ({operation} of {len(equations)} equations)"
        )
        
        # Create the equation
        return GeneralShapeEquation(
            symbolic_components=composite_components,
            equation_type=EquationType.COMPOSITE,
            metadata=metadata
        )


class DeepLearningAdapter:
    """
    Adapter class for integrating deep learning with GeneralShapeEquation.
    
    This class provides methods for training neural networks to approximate
    or enhance GeneralShapeEquation instances.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 20, output_dim: int = 1):
        """
        Initialize a DeepLearningAdapter.
        
        Args:
            input_dim: Input dimension (default: 2 for x,y coordinates)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (default: 1 for scalar output)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create a simple feedforward network
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
    
    def train_from_equation(self, equation: GeneralShapeEquation, 
                           num_samples: int = 1000, 
                           num_epochs: int = 100) -> List[float]:
        """
        Train the neural network to approximate a GeneralShapeEquation.
        
        Args:
            equation: The GeneralShapeEquation to approximate
            num_samples: Number of samples to generate
            num_epochs: Number of training epochs
            
        Returns:
            List of loss values during training
        """
        # Generate training data from the equation
        X, y = self._generate_samples(equation, num_samples)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, self.output_dim)
        
        # Train the model
        losses = []
        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            losses.append(loss.item())
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        return losses
    
    def _generate_samples(self, equation: GeneralShapeEquation, 
                         num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training samples from a GeneralShapeEquation.
        
        Args:
            equation: The GeneralShapeEquation to sample from
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        # For now, we'll generate random x,y coordinates and evaluate the equation
        X = np.random.uniform(-10, 10, (num_samples, self.input_dim))
        y = np.zeros((num_samples, self.output_dim))
        
        # Evaluate the equation for each sample
        for i in range(num_samples):
            # Create variable assignments
            assignments = {}
            for j in range(self.input_dim):
                var_name = ['x', 'y', 'z', 't'][j] if j < 4 else f'var{j}'
                assignments[var_name] = X[i, j]
            
            # Evaluate the equation
            results = equation.evaluate(assignments)
            
            # Use the first component's result as the target
            if results:
                first_result = next(iter(results.values()))
                if first_result is not None:
                    y[i, 0] = first_result
        
        return X, y
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained neural network.
        
        Args:
            X: Input array of shape (n_samples, input_dim)
            
        Returns:
            Predictions array of shape (n_samples, output_dim)
        """
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        
        return predictions
    
    def enhance_equation(self, equation: GeneralShapeEquation) -> GeneralShapeEquation:
        """
        Enhance a GeneralShapeEquation with the trained neural network.
        
        Args:
            equation: The GeneralShapeEquation to enhance
            
        Returns:
            A new GeneralShapeEquation with neural enhancement
        """
        # Clone the equation
        enhanced_equation = equation.clone()
        
        # Add the neural network to the equation
        enhanced_equation.neural_components['dl_adapter'] = self.model
        
        # Update metadata
        from datetime import datetime
        enhanced_equation.metadata.last_modified = datetime.now().isoformat()
        enhanced_equation.metadata.version += 1
        enhanced_equation.metadata.description = f"{enhanced_equation.metadata.description} (Neural enhanced)"
        
        # Set learning mode to hybrid
        enhanced_equation.learning_mode = LearningMode.HYBRID
        
        return enhanced_equation
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            file_path: Path to save the model
        """
        torch.save(self.model.state_dict(), file_path)
    
    def load_model(self, file_path: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            file_path: Path to the saved model
        """
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()


class ReinforcementLearningAdapter:
    """
    Adapter class for integrating reinforcement learning with GeneralShapeEquation.
    
    This class provides methods for training agents to evolve GeneralShapeEquation
    instances through reinforcement learning.
    """
    
    def __init__(self, state_dim: int = 10, action_dim: int = 5):
        """
        Initialize a ReinforcementLearningAdapter.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create policy and value networks
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=0.001)
        
        # Initialize memory buffer for experience replay
        self.memory = []
        self.max_memory_size = 10000
    
    def equation_to_state(self, equation: GeneralShapeEquation) -> torch.Tensor:
        """
        Convert a GeneralShapeEquation to a state tensor.
        
        Args:
            equation: The GeneralShapeEquation to convert
            
        Returns:
            State tensor
        """
        # This is a placeholder for a more sophisticated state representation
        # In a real implementation, this would extract meaningful features from the equation
        
        # For now, we'll just use some basic metadata and component counts
        state = [
            len(equation.symbolic_components),  # Number of components
            equation.metadata.complexity,  # Complexity score
            equation.metadata.version,  # Version number
            float(equation.equation_type.value == EquationType.SHAPE.value),  # Is it a shape equation?
            float(equation.equation_type.value == EquationType.PATTERN.value),  # Is it a pattern equation?
            float(equation.equation_type.value == EquationType.BEHAVIOR.value),  # Is it a behavior equation?
            float(equation.equation_type.value == EquationType.TRANSFORMATION.value),  # Is it a transformation equation?
            float(equation.equation_type.value == EquationType.CONSTRAINT.value),  # Is it a constraint equation?
            float(equation.equation_type.value == EquationType.COMPOSITE.value),  # Is it a composite equation?
            len(equation.variables)  # Number of variables
        ]
        
        # Pad or truncate to match state_dim
        if len(state) < self.state_dim:
            state.extend([0.0] * (self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        
        return torch.tensor(state, dtype=torch.float32)
    
    def select_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (action index, action probability)
        """
        with torch.no_grad():
            # Get action probabilities from policy network
            probs = self.policy_net(state)
            
            # Sample an action
            action = torch.multinomial(probs, 1).item()
            
            # Get the probability of the selected action
            prob = probs[action].item()
        
        return action, prob
    
    def apply_action(self, equation: GeneralShapeEquation, action: int) -> GeneralShapeEquation:
        """
        Apply an action to evolve the equation.
        
        Args:
            equation: The GeneralShapeEquation to evolve
            action: Action index
            
        Returns:
            Evolved GeneralShapeEquation
        """
        # This is a placeholder for more sophisticated action application
        # In a real implementation, this would have a wider range of possible actions
        
        # Define possible actions
        actions = [
            lambda eq: eq.mutate(0.1),  # Small mutation
            lambda eq: eq.mutate(0.3),  # Medium mutation
            lambda eq: eq.mutate(0.5),  # Large mutation
            lambda eq: eq.simplify(),  # Simplify
            lambda eq: eq.clone()  # Just clone (no change)
        ]
        
        # Apply the selected action
        action_idx = action % len(actions)
        return actions[action_idx](equation)
    
    def calculate_reward(self, original_eq: GeneralShapeEquation, 
                        evolved_eq: GeneralShapeEquation) -> float:
        """
        Calculate the reward for an evolution step.
        
        Args:
            original_eq: Original equation before evolution
            evolved_eq: Evolved equation after applying an action
            
        Returns:
            Reward value
        """
        # This is a placeholder for more sophisticated reward calculation
        # In a real implementation, this would consider multiple factors
        
        # For now, we'll reward:
        # 1. Reduced complexity (if it's simpler)
        # 2. Maintained or increased expressiveness (number of components)
        
        # Complexity change (reward if complexity decreased)
        complexity_change = original_eq.metadata.complexity - evolved_eq.metadata.complexity
        complexity_reward = 1.0 if complexity_change > 0 else -0.5
        
        # Component count change (reward if maintained or increased slightly)
        orig_components = len(original_eq.symbolic_components)
        evolved_components = len(evolved_eq.symbolic_components)
        
        if evolved_components >= orig_components:
            component_reward = 0.5
        else:
            component_reward = -0.5
        
        # Combine rewards
        total_reward = complexity_reward + component_reward
        
        return total_reward
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store an experience in the memory buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Add experience to memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Limit memory size
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
    
    def train_step(self, batch_size: int = 32, gamma: float = 0.99) -> Tuple[float, float]:
        """
        Perform a training step using experiences from memory.
        
        Args:
            batch_size: Number of experiences to sample
            gamma: Discount factor for future rewards
            
        Returns:
            Tuple of (policy loss, value loss)
        """
        if len(self.memory) < batch_size:
            return 0.0, 0.0
        
        # Sample a batch of experiences
        import random
        batch = random.sample(self.memory, batch_size)
        
        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Calculate returns (discounted future rewards)
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
            returns = rewards + gamma * next_values * (1 - dones)
        
        # Get current values
        values = self.value_net(states).squeeze()
        
        # Calculate advantage (returns - values)
        advantages = returns - values
        
        # Get action probabilities
        probs = self.policy_net(states)
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Calculate losses
        value_loss = torch.mean((returns - values) ** 2)
        policy_loss = -torch.mean(torch.log(action_probs) * advantages.detach())
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def train_on_equation(self, equation: GeneralShapeEquation, 
                         num_episodes: int = 100, 
                         steps_per_episode: int = 10) -> Tuple[List[float], GeneralShapeEquation]:
        """
        Train the RL agent to evolve a GeneralShapeEquation.
        
        Args:
            equation: The GeneralShapeEquation to evolve
            num_episodes: Number of training episodes
            steps_per_episode: Number of steps per episode
            
        Returns:
            Tuple of (rewards list, best evolved equation)
        """
        total_rewards = []
        best_reward = float('-inf')
        best_equation = equation.clone()
        
        for episode in range(num_episodes):
            # Start with a fresh copy of the original equation
            current_eq = equation.clone()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Convert equation to state
                state = self.equation_to_state(current_eq)
                
                # Select action
                action, _ = self.select_action(state)
                
                # Apply action to get next equation
                next_eq = self.apply_action(current_eq, action)
                
                # Calculate reward
                reward = self.calculate_reward(current_eq, next_eq)
                episode_reward += reward
                
                # Convert next equation to state
                next_state = self.equation_to_state(next_eq)
                
                # Store experience
                done = (step == steps_per_episode - 1)
                self.store_experience(state, action, reward, next_state, done)
                
                # Move to next state
                current_eq = next_eq
                
                # Train on a batch of experiences
                if len(self.memory) >= 32:
                    self.train_step(batch_size=32)
            
            # Record total reward for this episode
            total_rewards.append(episode_reward)
            
            # Update best equation if this episode produced a better one
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_equation = current_eq.clone()
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f'Episode [{episode+1}/{num_episodes}], Reward: {episode_reward:.4f}')
        
        return total_rewards, best_equation
    
    def enhance_equation(self, equation: GeneralShapeEquation) -> GeneralShapeEquation:
        """
        Enhance a GeneralShapeEquation with the trained RL agent.
        
        Args:
            equation: The GeneralShapeEquation to enhance
            
        Returns:
            Enhanced GeneralShapeEquation
        """
        # Clone the equation
        enhanced_equation = equation.clone()
        
        # Add the RL networks to the equation
        enhanced_equation.neural_components['rl_policy'] = self.policy_net
        enhanced_equation.neural_components['rl_value'] = self.value_net
        
        # Update metadata
        from datetime import datetime
        enhanced_equation.metadata.last_modified = datetime.now().isoformat()
        enhanced_equation.metadata.version += 1
        enhanced_equation.metadata.description = f"{enhanced_equation.metadata.description} (RL enhanced)"
        
        # Set learning mode to reinforcement
        enhanced_equation.learning_mode = LearningMode.REINFORCEMENT
        
        return enhanced_equation
    
    def save_models(self, policy_path: str, value_path: str) -> None:
        """
        Save the trained models to files.
        
        Args:
            policy_path: Path to save the policy network
            value_path: Path to save the value network
        """
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.value_net.state_dict(), value_path)
    
    def load_models(self, policy_path: str, value_path: str) -> None:
        """
        Load trained models from files.
        
        Args:
            policy_path: Path to the saved policy network
            value_path: Path to the saved value network
        """
        self.policy_net.load_state_dict(torch.load(policy_path))
        self.policy_net.eval()
        
        self.value_net.load_state_dict(torch.load(value_path))
        self.value_net.eval()


class ExpertExplorerSystem:
    """
    Implementation of the Expert/Explorer interaction model for evolving equations.
    
    This system consists of:
    1. Expert: Provides guidance and evaluates explorer's findings
    2. Explorer: Explores the equation space and brings back findings
    
    The interaction between these components allows for efficient exploration
    of the equation space and evolution of equations.
    """
    
    def __init__(self, initial_equation: Optional[GeneralShapeEquation] = None):
        """
        Initialize an ExpertExplorerSystem.
        
        Args:
            initial_equation: Optional initial equation to start with
        """
        # Initialize the expert component
        self.expert = {
            'knowledge_base': {},  # Store known good equations and patterns
            'heuristics': {},      # Store heuristic rules for guidance
            'evaluation_metrics': {}  # Store metrics for evaluating equations
        }
        
        # Initialize the explorer component
        self.explorer = {
            'current_equation': initial_equation,
            'exploration_history': [],
            'current_direction': None,
            'learning_rate': 0.1
        }
        
        # Initialize the RL adapter for exploration
        self.rl_adapter = ReinforcementLearningAdapter()
        
        # Initialize interaction history
        self.interaction_history = []
    
    def initialize_expert_knowledge(self):
        """Initialize the expert's knowledge base with basic patterns and heuristics."""
        # Add basic shape equations to knowledge base
        self.expert['knowledge_base']['circle'] = GSEFactory.create_basic_shape('circle', cx=0, cy=0, radius=1)
        self.expert['knowledge_base']['rectangle'] = GSEFactory.create_basic_shape('rectangle', x=0, y=0, width=1, height=1)
        self.expert['knowledge_base']['ellipse'] = GSEFactory.create_basic_shape('ellipse', cx=0, cy=0, rx=1, ry=0.5)
        
        # Add basic heuristics
        self.expert['heuristics']['simplify'] = lambda eq: eq.simplify()
        self.expert['heuristics']['mutate_small'] = lambda eq: eq.mutate(0.1)
        self.expert['heuristics']['mutate_medium'] = lambda eq: eq.mutate(0.3)
        self.expert['heuristics']['mutate_large'] = lambda eq: eq.mutate(0.5)
        
        # Add basic evaluation metrics
        self.expert['evaluation_metrics']['complexity'] = lambda eq: -eq.metadata.complexity  # Lower complexity is better
        self.expert['evaluation_metrics']['component_count'] = lambda eq: len(eq.symbolic_components)  # More components can be better
    
    def expert_evaluate(self, equation: GeneralShapeEquation) -> float:
        """
        Expert evaluation of an equation.
        
        Args:
            equation: The equation to evaluate
            
        Returns:
            Evaluation score (higher is better)
        """
        # Apply all evaluation metrics
        scores = []
        for metric_name, metric_func in self.expert['evaluation_metrics'].items():
            score = metric_func(equation)
            scores.append(score)
        
        # Combine scores (simple average for now)
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.0
    
    def expert_guide(self, equation: GeneralShapeEquation) -> List[Tuple[str, Callable]]:
        """
        Expert guidance for exploration.
        
        Args:
            equation: The current equation
            
        Returns:
            List of (heuristic name, heuristic function) tuples to try
        """
        # For now, just return all heuristics
        # In a more sophisticated implementation, this would select
        # heuristics based on the current equation's properties
        return list(self.expert['heuristics'].items())
    
    def explorer_explore(self, steps: int = 10) -> GeneralShapeEquation:
        """
        Explorer exploration of the equation space.
        
        Args:
            steps: Number of exploration steps
            
        Returns:
            Best equation found during exploration
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        # Get guidance from expert
        guidance = self.expert_guide(self.explorer['current_equation'])
        
        # Explore using the guidance
        best_equation = self.explorer['current_equation']
        best_score = self.expert_evaluate(best_equation)
        
        for _ in range(steps):
            # Try each heuristic
            for heuristic_name, heuristic_func in guidance:
                # Apply the heuristic
                new_equation = heuristic_func(self.explorer['current_equation'])
                
                # Evaluate the new equation
                new_score = self.expert_evaluate(new_equation)
                
                # Update best if better
                if new_score > best_score:
                    best_equation = new_equation
                    best_score = new_score
            
            # Move to the best equation found
            self.explorer['current_equation'] = best_equation
            
            # Record in exploration history
            self.explorer['exploration_history'].append({
                'equation': best_equation,
                'score': best_score
            })
        
        return best_equation
    
    def explorer_explore_rl(self, episodes: int = 10, steps_per_episode: int = 5) -> GeneralShapeEquation:
        """
        Explorer exploration using reinforcement learning.
        
        Args:
            episodes: Number of RL episodes
            steps_per_episode: Steps per episode
            
        Returns:
            Best equation found during exploration
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        # Train the RL adapter on the current equation
        rewards, best_equation = self.rl_adapter.train_on_equation(
            self.explorer['current_equation'],
            num_episodes=episodes,
            steps_per_episode=steps_per_episode
        )
        
        # Update the current equation
        self.explorer['current_equation'] = best_equation
        
        # Record in exploration history
        self.explorer['exploration_history'].append({
            'equation': best_equation,
            'score': self.expert_evaluate(best_equation),
            'method': 'rl',
            'rewards': rewards
        })
        
        return best_equation
    
    def expert_explorer_interaction(self, cycles: int = 5, 
                                  steps_per_cycle: int = 10) -> GeneralShapeEquation:
        """
        Full expert-explorer interaction cycle.
        
        Args:
            cycles: Number of interaction cycles
            steps_per_cycle: Steps per exploration cycle
            
        Returns:
            Best equation found during all cycles
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        best_overall_equation = self.explorer['current_equation']
        best_overall_score = self.expert_evaluate(best_overall_equation)
        
        for cycle in range(cycles):
            # Explorer explores
            if cycle % 2 == 0:
                # Use standard exploration
                best_cycle_equation = self.explorer_explore(steps=steps_per_cycle)
            else:
                # Use RL-based exploration
                best_cycle_equation = self.explorer_explore_rl(
                    episodes=steps_per_cycle // 2,
                    steps_per_episode=5
                )
            
            # Expert evaluates
            cycle_score = self.expert_evaluate(best_cycle_equation)
            
            # Record interaction
            self.interaction_history.append({
                'cycle': cycle,
                'equation': best_cycle_equation,
                'score': cycle_score
            })
            
            # Update best overall if better
            if cycle_score > best_overall_score:
                best_overall_equation = best_cycle_equation
                best_overall_score = cycle_score
            
            # Expert provides new guidance based on results
            # (This happens implicitly in the next cycle)
            
            # Print progress
            print(f'Cycle {cycle+1}/{cycles}, Score: {cycle_score:.4f}, Best: {best_overall_score:.4f}')
        
        return best_overall_equation
    
    def set_initial_equation(self, equation: GeneralShapeEquation) -> None:
        """
        Set the initial equation for exploration.
        
        Args:
            equation: The initial equation
        """
        self.explorer['current_equation'] = equation
        self.explorer['exploration_history'] = []
    
    def get_best_equation(self) -> Optional[GeneralShapeEquation]:
        """
        Get the best equation found so far.
        
        Returns:
            Best equation, or None if no exploration has been done
        """
        if not self.explorer['exploration_history']:
            return self.explorer['current_equation']
        
        # Find the entry with the highest score
        best_entry = max(self.explorer['exploration_history'], 
                         key=lambda entry: entry['score'])
        
        return best_entry['equation']
