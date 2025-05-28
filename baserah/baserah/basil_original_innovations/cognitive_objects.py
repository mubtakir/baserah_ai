#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognitive Object Manager for Basira System

This module provides the foundation for representing cognitive objects in the Basira System
according to AI-OOP principles. It defines the base Thing class from which all cognitive
objects inherit, and provides functionality for managing these objects and their properties.

Author: Basira System Development Team
Version: 1.0.0
"""

import uuid
import json
import copy
import datetime
from typing import Dict, List, Tuple, Union, Optional, Any, ClassVar, Type

# Import the shape equation parser from the parsing utilities
from basira_system.parsing_utilities.advanced_shape_equation_parser import AdvancedShapeEquationParser, ParseError


class PropertyType:
    """Constants for different property types in cognitive objects."""
    SEMANTIC = "semantic"
    LOGICAL = "logical"
    BEHAVIORAL = "behavioral"
    LINGUISTIC = "linguistic"
    META = "meta"
    ALL = [SEMANTIC, LOGICAL, BEHAVIORAL, LINGUISTIC, META]


class Thing:
    """
    Base class for all cognitive objects in the Basira System.
    
    This class implements the core AI-OOP principles, representing any entity (concrete or abstract)
    as an object with various properties including a dynamic shape equation representation.
    """
    
    # Class-level parser instance to avoid creating multiple instances
    _parser_instance: ClassVar[Optional[AdvancedShapeEquationParser]] = None
    
    @classmethod
    def get_parser(cls) -> AdvancedShapeEquationParser:
        """
        Get or create a shared parser instance.
        
        Returns:
            An instance of AdvancedShapeEquationParser
        """
        if cls._parser_instance is None:
            cls._parser_instance = AdvancedShapeEquationParser()
        return cls._parser_instance
    
    def __init__(self, name: str, shape_equation_str: Optional[str] = None, 
                 object_id: Optional[str] = None, **kwargs):
        """
        Initialize a new Thing object.
        
        Args:
            name: The name of the object or concept (e.g., "cat", "tree", "justice_concept")
            shape_equation_str: A string representing the dynamic shape equation in AdvancedShapeEngine format
            object_id: A unique identifier for the object (generated automatically if not provided)
            **kwargs: Additional properties to set during initialization
        """
        # Basic identification
        self.name = name
        self.object_id = object_id or self._generate_unique_id()
        
        # Shape representation
        self.raw_shape_equation_string = shape_equation_str
        self.parsed_shape_representation: Optional[List[Dict[str, Any]]] = None
        
        # Parse the shape equation if provided
        if shape_equation_str:
            self._parse_shape_equation(shape_equation_str)
        
        # Initialize property dictionaries
        self.semantic_properties: Dict[str, Any] = {}
        self.logical_properties: Dict[str, Any] = {}
        self.behavioral_equations: List[Any] = []
        self.linguistic_properties: Dict[str, Any] = {}
        self.meta_properties: Dict[str, Any] = {
            "creation_time": datetime.datetime.now().isoformat(),
            "last_modified": datetime.datetime.now().isoformat(),
            "confidence": 1.0,  # Default confidence level
            "source": "direct_creation"  # Default source
        }
        
        # Process additional properties from kwargs
        self._process_kwargs(kwargs)
    
    def _generate_unique_id(self) -> str:
        """
        Generate a unique identifier for this object.
        
        Returns:
            A string containing a UUID
        """
        return str(uuid.uuid4())
    
    def _parse_shape_equation(self, shape_equation_str: str) -> None:
        """
        Parse the shape equation string into a structured representation.
        
        Args:
            shape_equation_str: The shape equation string to parse
        """
        try:
            parser = self.get_parser()
            
            # Option 1: If we decide to modify AdvancedShapeEquationParser to handle multiple shapes
            # This would require modifying the parser to return a list of dictionaries
            # parsed_data = parser.parse(shape_equation_str)  # This would return a list of dictionaries
            # self.parsed_shape_representation = parsed_data
            
            # Option 2: For now, we'll use the current parser which handles one shape at a time
            # and manually split the equation string if it contains multiple shapes
            # This approach is less ideal but works with the current parser implementation
            
            # Split the equation string by common operators (+, &, |, -)
            import re
            shape_parts = re.split(r'\s*[\+\&\|\-]\s*', shape_equation_str)
            
            # Parse each part separately
            parsed_shapes = []
            for part in shape_parts:
                if part.strip():  # Skip empty parts
                    try:
                        parsed_data = parser.parse(part.strip())
                        parsed_shapes.append(parsed_data)
                    except ParseError as e:
                        print(f"Error parsing shape component '{part}' for object '{self.name}': {e}")
                        # Continue with other parts instead of failing completely
            
            self.parsed_shape_representation = parsed_shapes if parsed_shapes else None
            
        except ParseError as e:
            print(f"Error parsing shape equation for object '{self.name}': {e}")
            self.parsed_shape_representation = None
        except Exception as e:
            print(f"Unexpected error during shape parsing for '{self.name}': {e}")
            self.parsed_shape_representation = None
    
    def _process_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Process additional properties from kwargs and add them to the appropriate property dictionaries.
        
        Args:
            kwargs: Dictionary of additional properties
        """
        for key, value in kwargs.items():
            # Check if the key has a prefix indicating the property type
            if "__" in key:
                prefix, actual_key = key.split("__", 1)
                if prefix == PropertyType.SEMANTIC:
                    self.semantic_properties[actual_key] = value
                elif prefix == PropertyType.LOGICAL:
                    self.logical_properties[actual_key] = value
                elif prefix == PropertyType.LINGUISTIC:
                    self.linguistic_properties[actual_key] = value
                elif prefix == PropertyType.META:
                    self.meta_properties[actual_key] = value
                # Behavioral equations are handled separately
            else:
                # Default to semantic properties if no prefix is specified
                self.semantic_properties[key] = value
    
    # Basic getters and setters
    
    def get_id(self) -> str:
        """
        Get the unique identifier of this object.
        
        Returns:
            The object's unique identifier
        """
        return self.object_id
    
    def get_name(self) -> str:
        """
        Get the name of this object.
        
        Returns:
            The object's name
        """
        return self.name
    
    def set_name(self, new_name: str) -> None:
        """
        Set a new name for this object.
        
        Args:
            new_name: The new name to set
        """
        self.name = new_name
        self._update_modification_time()
    
    def get_shape_equation_string(self) -> Optional[str]:
        """
        Get the original shape equation string.
        
        Returns:
            The raw shape equation string, or None if not set
        """
        return self.raw_shape_equation_string
    
    def get_parsed_shape_representation(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get the parsed representation of the shape equation.
        
        Returns:
            A list of dictionaries representing the parsed shapes, or None if not available
        """
        return self.parsed_shape_representation
    
    def update_shape_equation(self, new_shape_equation_str: str) -> None:
        """
        Update the shape equation with a new string and parse it.
        
        Args:
            new_shape_equation_str: The new shape equation string
        """
        self.raw_shape_equation_string = new_shape_equation_str
        self._parse_shape_equation(new_shape_equation_str)
        self._update_modification_time()
    
    def update_shape_from_parsed_representation(self, new_parsed_representation: List[Dict[str, Any]]) -> None:
        """
        Update the shape representation from a parsed structure.
        
        This is useful when the shape has been modified by another component like EquationEvolutionService_EEE.
        
        Args:
            new_parsed_representation: The new parsed shape representation
        """
        self.parsed_shape_representation = copy.deepcopy(new_parsed_representation)
        # Optionally rebuild the raw string from the parsed representation
        # self.raw_shape_equation_string = self._rebuild_shape_string_from_parsed(new_parsed_representation)
        self._update_modification_time()
    
    def _rebuild_shape_string_from_parsed(self, parsed_representation: List[Dict[str, Any]]) -> str:
        """
        Rebuild a shape equation string from its parsed representation.
        
        This is a placeholder for future implementation. In a complete system, this would
        convert the parsed representation back to a string format.
        
        Args:
            parsed_representation: The parsed shape representation to convert
            
        Returns:
            A string representation of the shape equation
        """
        # This is a placeholder. In a real implementation, this would convert
        # the parsed representation back to a string format.
        return "# Placeholder: Rebuilding shape string not yet implemented"
    
    # Property management methods
    
    def add_property(self, property_type: str, key: str, value: Any) -> None:
        """
        Add or update a property of the specified type.
        
        Args:
            property_type: The type of property (semantic, logical, linguistic, meta)
            key: The property key
            value: The property value
        
        Raises:
            ValueError: If the property_type is not recognized
        """
        if property_type == PropertyType.SEMANTIC:
            self.semantic_properties[key] = value
        elif property_type == PropertyType.LOGICAL:
            self.logical_properties[key] = value
        elif property_type == PropertyType.LINGUISTIC:
            self.linguistic_properties[key] = value
        elif property_type == PropertyType.META:
            self.meta_properties[key] = value
        else:
            raise ValueError(f"Unknown property type: {property_type}")
        
        self._update_modification_time()
    
    def get_property(self, property_type: str, key: str) -> Optional[Any]:
        """
        Get a property value of the specified type.
        
        Args:
            property_type: The type of property (semantic, logical, linguistic, meta)
            key: The property key
            
        Returns:
            The property value, or None if not found
            
        Raises:
            ValueError: If the property_type is not recognized
        """
        if property_type == PropertyType.SEMANTIC:
            return self.semantic_properties.get(key)
        elif property_type == PropertyType.LOGICAL:
            return self.logical_properties.get(key)
        elif property_type == PropertyType.LINGUISTIC:
            return self.linguistic_properties.get(key)
        elif property_type == PropertyType.META:
            return self.meta_properties.get(key)
        else:
            raise ValueError(f"Unknown property type: {property_type}")
    
    def get_all_properties(self, property_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all properties of a specific type, or all properties if type is None.
        
        Args:
            property_type: The type of properties to get, or None for all properties
            
        Returns:
            A dictionary of property keys and values
            
        Raises:
            ValueError: If the property_type is not recognized
        """
        if property_type is None:
            # Return all properties in a structured dictionary
            return {
                PropertyType.SEMANTIC: self.semantic_properties,
                PropertyType.LOGICAL: self.logical_properties,
                PropertyType.BEHAVIORAL: self.behavioral_equations,
                PropertyType.LINGUISTIC: self.linguistic_properties,
                PropertyType.META: self.meta_properties
            }
        elif property_type == PropertyType.SEMANTIC:
            return self.semantic_properties
        elif property_type == PropertyType.LOGICAL:
            return self.logical_properties
        elif property_type == PropertyType.BEHAVIORAL:
            return self.behavioral_equations
        elif property_type == PropertyType.LINGUISTIC:
            return self.linguistic_properties
        elif property_type == PropertyType.META:
            return self.meta_properties
        else:
            raise ValueError(f"Unknown property type: {property_type}")
    
    def remove_property(self, property_type: str, key: str) -> bool:
        """
        Remove a property of the specified type.
        
        Args:
            property_type: The type of property (semantic, logical, linguistic, meta)
            key: The property key
            
        Returns:
            True if the property was removed, False if it didn't exist
            
        Raises:
            ValueError: If the property_type is not recognized
        """
        if property_type == PropertyType.SEMANTIC:
            if key in self.semantic_properties:
                del self.semantic_properties[key]
                self._update_modification_time()
                return True
        elif property_type == PropertyType.LOGICAL:
            if key in self.logical_properties:
                del self.logical_properties[key]
                self._update_modification_time()
                return True
        elif property_type == PropertyType.LINGUISTIC:
            if key in self.linguistic_properties:
                del self.linguistic_properties[key]
                self._update_modification_time()
                return True
        elif property_type == PropertyType.META:
            if key in self.meta_properties:
                del self.meta_properties[key]
                self._update_modification_time()
                return True
        else:
            raise ValueError(f"Unknown property type: {property_type}")
        
        return False
    
    # Behavioral equation methods
    
    def add_behavioral_equation(self, equation_object: Any) -> None:
        """
        Add a behavioral equation to this object.
        
        Args:
            equation_object: An object representing a behavioral equation
        """
        self.behavioral_equations.append(equation_object)
        self._update_modification_time()
    
    def get_behavioral_equations(self) -> List[Any]:
        """
        Get all behavioral equations for this object.
        
        Returns:
            A list of behavioral equation objects
        """
        return self.behavioral_equations
    
    def remove_behavioral_equation(self, index: int) -> bool:
        """
        Remove a behavioral equation by index.
        
        Args:
            index: The index of the equation to remove
            
        Returns:
            True if the equation was removed, False if the index was out of range
        """
        if 0 <= index < len(self.behavioral_equations):
            del self.behavioral_equations[index]
            self._update_modification_time()
            return True
        return False
    
    # Serialization methods
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this object to a dictionary suitable for serialization.
        
        Returns:
            A dictionary representation of this object
        """
        return {
            "object_id": self.object_id,
            "name": self.name,
            "type": self.__class__.__name__,
            "raw_shape_equation_string": self.raw_shape_equation_string,
            "parsed_shape_representation": self.parsed_shape_representation,
            "semantic_properties": self.semantic_properties,
            "logical_properties": self.logical_properties,
            "behavioral_equations": self.behavioral_equations,
            "linguistic_properties": self.linguistic_properties,
            "meta_properties": self.meta_properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Thing':
        """
        Create a Thing object from a dictionary.
        
        Args:
            data: A dictionary containing object data
            
        Returns:
            A new Thing object
        """
        # Extract the basic properties
        name = data.get("name", "Unnamed")
        object_id = data.get("object_id")
        shape_equation_str = data.get("raw_shape_equation_string")
        
        # Create a new instance
        obj = cls(name=name, shape_equation_str=shape_equation_str, object_id=object_id)
        
        # Set the parsed representation directly if available
        if "parsed_shape_representation" in data and data["parsed_shape_representation"] is not None:
            obj.parsed_shape_representation = data["parsed_shape_representation"]
        
        # Set the property dictionaries
        if "semantic_properties" in data:
            obj.semantic_properties = data["semantic_properties"]
        if "logical_properties" in data:
            obj.logical_properties = data["logical_properties"]
        if "behavioral_equations" in data:
            obj.behavioral_equations = data["behavioral_equations"]
        if "linguistic_properties" in data:
            obj.linguistic_properties = data["linguistic_properties"]
        if "meta_properties" in data:
            obj.meta_properties = data["meta_properties"]
        
        return obj
    
    def to_json(self) -> str:
        """
        Convert this object to a JSON string.
        
        Returns:
            A JSON string representation of this object
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Thing':
        """
        Create a Thing object from a JSON string.
        
        Args:
            json_str: A JSON string containing object data
            
        Returns:
            A new Thing object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def _update_modification_time(self) -> None:
        """Update the last_modified timestamp in meta_properties."""
        self.meta_properties["last_modified"] = datetime.datetime.now().isoformat()
    
    def __str__(self) -> str:
        """
        Get a string representation of this object.
        
        Returns:
            A string describing this object
        """
        return f"{self.__class__.__name__}(id={self.object_id}, name={self.name})"
    
    def __repr__(self) -> str:
        """
        Get a detailed string representation of this object.
        
        Returns:
            A detailed string representation of this object
        """
        return f"{self.__class__.__name__}(id={self.object_id}, name={self.name}, properties={len(self.semantic_properties)})"


class PhysicalObject(Thing):
    """
    A class representing physical objects in the world.
    
    Physical objects have additional properties related to their physical nature,
    such as mass, volume, and material.
    """
    
    def __init__(self, name: str, shape_equation_str: Optional[str] = None, 
                 object_id: Optional[str] = None, **kwargs):
        """
        Initialize a new PhysicalObject.
        
        Args:
            name: The name of the physical object
            shape_equation_str: A string representing the dynamic shape equation
            object_id: A unique identifier for the object
            **kwargs: Additional properties to set during initialization
        """
        # Initialize default physical properties
        physical_properties = {
            "mass": kwargs.pop("mass", None),
            "volume": kwargs.pop("volume", None),
            "material": kwargs.pop("material", None),
            "location": kwargs.pop("location", None),
            "dimensions": kwargs.pop("dimensions", None)
        }
        
        # Remove None values
        physical_properties = {k: v for k, v in physical_properties.items() if v is not None}
        
        # Add physical properties with semantic prefix
        prefixed_properties = {f"semantic__{k}": v for k, v in physical_properties.items()}
        kwargs.update(prefixed_properties)
        
        # Call the parent constructor
        super().__init__(name, shape_equation_str, object_id, **kwargs)
        
        # Set the object type in meta properties
        self.meta_properties["object_type"] = "physical"
    
    def get_mass(self) -> Optional[float]:
        """
        Get the mass of this physical object.
        
        Returns:
            The mass value, or None if not set
        """
        return self.semantic_properties.get("mass")
    
    def set_mass(self, mass: float) -> None:
        """
        Set the mass of this physical object.
        
        Args:
            mass: The mass value to set
        """
        self.semantic_properties["mass"] = mass
        self._update_modification_time()
    
    def get_volume(self) -> Optional[float]:
        """
        Get the volume of this physical object.
        
        Returns:
            The volume value, or None if not set
        """
        return self.semantic_properties.get("volume")
    
    def set_volume(self, volume: float) -> None:
        """
        Set the volume of this physical object.
        
        Args:
            volume: The volume value to set
        """
        self.semantic_properties["volume"] = volume
        self._update_modification_time()
    
    def get_material(self) -> Optional[str]:
        """
        Get the material of this physical object.
        
        Returns:
            The material description, or None if not set
        """
        return self.semantic_properties.get("material")
    
    def set_material(self, material: str) -> None:
        """
        Set the material of this physical object.
        
        Args:
            material: The material description to set
        """
        self.semantic_properties["material"] = material
        self._update_modification_time()
    
    def get_location(self) -> Optional[Any]:
        """
        Get the location of this physical object.
        
        Returns:
            The location value, or None if not set
        """
        return self.semantic_properties.get("location")
    
    def set_location(self, location: Any) -> None:
        """
        Set the location of this physical object.
        
        Args:
            location: The location value to set
        """
        self.semantic_properties["location"] = location
        self._update_modification_time()
    
    def get_dimensions(self) -> Optional[Any]:
        """
        Get the dimensions of this physical object.
        
        Returns:
            The dimensions value, or None if not set
        """
        return self.semantic_properties.get("dimensions")
    
    def set_dimensions(self, dimensions: Any) -> None:
        """
        Set the dimensions of this physical object.
        
        Args:
            dimensions: The dimensions value to set
        """
        self.semantic_properties["dimensions"] = dimensions
        self._update_modification_time()


class ConceptObject(Thing):
    """
    A class representing abstract concepts.
    
    Concept objects have additional properties related to their abstract nature,
    such as abstraction level and related concepts.
    """
    
    def __init__(self, name: str, shape_equation_str: Optional[str] = None, 
                 object_id: Optional[str] = None, **kwargs):
        """
        Initialize a new ConceptObject.
        
        Args:
            name: The name of the concept
            shape_equation_str: A string representing the dynamic shape equation
            object_id: A unique identifier for the object
            **kwargs: Additional properties to set during initialization
        """
        # Initialize default concept properties
        concept_properties = {
            "abstraction_level": kwargs.pop("abstraction_level", None),
            "related_concepts": kwargs.pop("related_concepts", None),
            "domain": kwargs.pop("domain", None),
            "definition": kwargs.pop("definition", None)
        }
        
        # Remove None values
        concept_properties = {k: v for k, v in concept_properties.items() if v is not None}
        
        # Add concept properties with semantic prefix
        prefixed_properties = {f"semantic__{k}": v for k, v in concept_properties.items()}
        kwargs.update(prefixed_properties)
        
        # Call the parent constructor
        super().__init__(name, shape_equation_str, object_id, **kwargs)
        
        # Set the object type in meta properties
        self.meta_properties["object_type"] = "concept"
    
    def get_abstraction_level(self) -> Optional[int]:
        """
        Get the abstraction level of this concept.
        
        Returns:
            The abstraction level, or None if not set
        """
        return self.semantic_properties.get("abstraction_level")
    
    def set_abstraction_level(self, level: int) -> None:
        """
        Set the abstraction level of this concept.
        
        Args:
            level: The abstraction level to set
        """
        self.semantic_properties["abstraction_level"] = level
        self._update_modification_time()
    
    def get_related_concepts(self) -> Optional[List[str]]:
        """
        Get the related concepts of this concept.
        
        Returns:
            A list of related concept names, or None if not set
        """
        return self.semantic_properties.get("related_concepts")
    
    def set_related_concepts(self, concepts: List[str]) -> None:
        """
        Set the related concepts of this concept.
        
        Args:
            concepts: A list of related concept names
        """
        self.semantic_properties["related_concepts"] = concepts
        self._update_modification_time()
    
    def add_related_concept(self, concept: str) -> None:
        """
        Add a related concept to this concept.
        
        Args:
            concept: The name of the related concept to add
        """
        if "related_concepts" not in self.semantic_properties:
            self.semantic_properties["related_concepts"] = []
        
        if concept not in self.semantic_properties["related_concepts"]:
            self.semantic_properties["related_concepts"].append(concept)
            self._update_modification_time()
    
    def get_domain(self) -> Optional[str]:
        """
        Get the domain of this concept.
        
        Returns:
            The domain name, or None if not set
        """
        return self.semantic_properties.get("domain")
    
    def set_domain(self, domain: str) -> None:
        """
        Set the domain of this concept.
        
        Args:
            domain: The domain name to set
        """
        self.semantic_properties["domain"] = domain
        self._update_modification_time()
    
    def get_definition(self) -> Optional[str]:
        """
        Get the definition of this concept.
        
        Returns:
            The definition text, or None if not set
        """
        return self.semantic_properties.get("definition")
    
    def set_definition(self, definition: str) -> None:
        """
        Set the definition of this concept.
        
        Args:
            definition: The definition text to set
        """
        self.semantic_properties["definition"] = definition
        self._update_modification_time()


class LivingBeing(PhysicalObject):
    """
    A class representing living beings.
    
    Living beings have additional properties related to their living nature,
    such as age, lifespan, and vital actions.
    """
    
    def __init__(self, name: str, shape_equation_str: Optional[str] = None, 
                 object_id: Optional[str] = None, **kwargs):
        """
        Initialize a new LivingBeing.
        
        Args:
            name: The name of the living being
            shape_equation_str: A string representing the dynamic shape equation
            object_id: A unique identifier for the object
            **kwargs: Additional properties to set during initialization
        """
        # Initialize default living being properties
        living_properties = {
            "age": kwargs.pop("age", None),
            "lifespan": kwargs.pop("lifespan", None),
            "species": kwargs.pop("species", None),
            "is_sentient": kwargs.pop("is_sentient", False),
            "vital_actions": kwargs.pop("vital_actions", None)
        }
        
        # Remove None values
        living_properties = {k: v for k, v in living_properties.items() if v is not None}
        
        # Add living being properties with semantic prefix
        prefixed_properties = {f"semantic__{k}": v for k, v in living_properties.items()}
        kwargs.update(prefixed_properties)
        
        # Call the parent constructor
        super().__init__(name, shape_equation_str, object_id, **kwargs)
        
        # Set the object type in meta properties
        self.meta_properties["object_type"] = "living_being"
    
    def get_age(self) -> Optional[float]:
        """
        Get the age of this living being.
        
        Returns:
            The age value, or None if not set
        """
        return self.semantic_properties.get("age")
    
    def set_age(self, age: float) -> None:
        """
        Set the age of this living being.
        
        Args:
            age: The age value to set
        """
        self.semantic_properties["age"] = age
        self._update_modification_time()
    
    def get_lifespan(self) -> Optional[float]:
        """
        Get the lifespan of this living being.
        
        Returns:
            The lifespan value, or None if not set
        """
        return self.semantic_properties.get("lifespan")
    
    def set_lifespan(self, lifespan: float) -> None:
        """
        Set the lifespan of this living being.
        
        Args:
            lifespan: The lifespan value to set
        """
        self.semantic_properties["lifespan"] = lifespan
        self._update_modification_time()
    
    def get_species(self) -> Optional[str]:
        """
        Get the species of this living being.
        
        Returns:
            The species name, or None if not set
        """
        return self.semantic_properties.get("species")
    
    def set_species(self, species: str) -> None:
        """
        Set the species of this living being.
        
        Args:
            species: The species name to set
        """
        self.semantic_properties["species"] = species
        self._update_modification_time()
    
    def is_sentient(self) -> bool:
        """
        Check if this living being is sentient.
        
        Returns:
            True if sentient, False otherwise
        """
        return self.semantic_properties.get("is_sentient", False)
    
    def set_sentient(self, sentient: bool) -> None:
        """
        Set whether this living being is sentient.
        
        Args:
            sentient: True if sentient, False otherwise
        """
        self.semantic_properties["is_sentient"] = sentient
        self._update_modification_time()
    
    def get_vital_actions(self) -> Optional[List[str]]:
        """
        Get the vital actions of this living being.
        
        Returns:
            A list of vital action names, or None if not set
        """
        return self.semantic_properties.get("vital_actions")
    
    def set_vital_actions(self, actions: List[str]) -> None:
        """
        Set the vital actions of this living being.
        
        Args:
            actions: A list of vital action names
        """
        self.semantic_properties["vital_actions"] = actions
        self._update_modification_time()
    
    def add_vital_action(self, action: str) -> None:
        """
        Add a vital action to this living being.
        
        Args:
            action: The name of the vital action to add
        """
        if "vital_actions" not in self.semantic_properties:
            self.semantic_properties["vital_actions"] = []
        
        if action not in self.semantic_properties["vital_actions"]:
            self.semantic_properties["vital_actions"].append(action)
            self._update_modification_time()


class Human(LivingBeing):
    """
    A class representing human beings.
    
    Humans have additional properties related to their human nature,
    such as occupation, education, and social relationships.
    """
    
    def __init__(self, name: str, shape_equation_str: Optional[str] = None, 
                 object_id: Optional[str] = None, **kwargs):
        """
        Initialize a new Human.
        
        Args:
            name: The name of the human
            shape_equation_str: A string representing the dynamic shape equation
            object_id: A unique identifier for the object
            **kwargs: Additional properties to set during initialization
        """
        # Initialize default human properties
        human_properties = {
            "occupation": kwargs.pop("occupation", None),
            "education": kwargs.pop("education", None),
            "social_relationships": kwargs.pop("social_relationships", None),
            "skills": kwargs.pop("skills", None),
            "personality_traits": kwargs.pop("personality_traits", None)
        }
        
        # Remove None values
        human_properties = {k: v for k, v in human_properties.items() if v is not None}
        
        # Add human properties with semantic prefix
        prefixed_properties = {f"semantic__{k}": v for k, v in human_properties.items()}
        kwargs.update(prefixed_properties)
        
        # Set default values for living being properties
        kwargs.setdefault("species", "Homo sapiens")
        kwargs.setdefault("is_sentient", True)
        
        # Call the parent constructor
        super().__init__(name, shape_equation_str, object_id, **kwargs)
        
        # Set the object type in meta properties
        self.meta_properties["object_type"] = "human"
    
    def get_occupation(self) -> Optional[str]:
        """
        Get the occupation of this human.
        
        Returns:
            The occupation description, or None if not set
        """
        return self.semantic_properties.get("occupation")
    
    def set_occupation(self, occupation: str) -> None:
        """
        Set the occupation of this human.
        
        Args:
            occupation: The occupation description to set
        """
        self.semantic_properties["occupation"] = occupation
        self._update_modification_time()
    
    def get_education(self) -> Optional[str]:
        """
        Get the education of this human.
        
        Returns:
            The education description, or None if not set
        """
        return self.semantic_properties.get("education")
    
    def set_education(self, education: str) -> None:
        """
        Set the education of this human.
        
        Args:
            education: The education description to set
        """
        self.semantic_properties["education"] = education
        self._update_modification_time()
    
    def get_social_relationships(self) -> Optional[Dict[str, List[str]]]:
        """
        Get the social relationships of this human.
        
        Returns:
            A dictionary mapping relationship types to lists of person names,
            or None if not set
        """
        return self.semantic_properties.get("social_relationships")
    
    def set_social_relationships(self, relationships: Dict[str, List[str]]) -> None:
        """
        Set the social relationships of this human.
        
        Args:
            relationships: A dictionary mapping relationship types to lists of person names
        """
        self.semantic_properties["social_relationships"] = relationships
        self._update_modification_time()
    
    def add_social_relationship(self, relationship_type: str, person: str) -> None:
        """
        Add a social relationship to this human.
        
        Args:
            relationship_type: The type of relationship (e.g., "friend", "family")
            person: The name of the person in this relationship
        """
        if "social_relationships" not in self.semantic_properties:
            self.semantic_properties["social_relationships"] = {}
        
        if relationship_type not in self.semantic_properties["social_relationships"]:
            self.semantic_properties["social_relationships"][relationship_type] = []
        
        if person not in self.semantic_properties["social_relationships"][relationship_type]:
            self.semantic_properties["social_relationships"][relationship_type].append(person)
            self._update_modification_time()
    
    def get_skills(self) -> Optional[List[str]]:
        """
        Get the skills of this human.
        
        Returns:
            A list of skill names, or None if not set
        """
        return self.semantic_properties.get("skills")
    
    def set_skills(self, skills: List[str]) -> None:
        """
        Set the skills of this human.
        
        Args:
            skills: A list of skill names
        """
        self.semantic_properties["skills"] = skills
        self._update_modification_time()
    
    def add_skill(self, skill: str) -> None:
        """
        Add a skill to this human.
        
        Args:
            skill: The name of the skill to add
        """
        if "skills" not in self.semantic_properties:
            self.semantic_properties["skills"] = []
        
        if skill not in self.semantic_properties["skills"]:
            self.semantic_properties["skills"].append(skill)
            self._update_modification_time()
    
    def get_personality_traits(self) -> Optional[List[str]]:
        """
        Get the personality traits of this human.
        
        Returns:
            A list of personality trait names, or None if not set
        """
        return self.semantic_properties.get("personality_traits")
    
    def set_personality_traits(self, traits: List[str]) -> None:
        """
        Set the personality traits of this human.
        
        Args:
            traits: A list of personality trait names
        """
        self.semantic_properties["personality_traits"] = traits
        self._update_modification_time()
    
    def add_personality_trait(self, trait: str) -> None:
        """
        Add a personality trait to this human.
        
        Args:
            trait: The name of the personality trait to add
        """
        if "personality_traits" not in self.semantic_properties:
            self.semantic_properties["personality_traits"] = []
        
        if trait not in self.semantic_properties["personality_traits"]:
            self.semantic_properties["personality_traits"].append(trait)
            self._update_modification_time()


class CognitiveObjectManager:
    """
    A manager class for creating and managing cognitive objects.
    
    This class provides functionality for creating, retrieving, and managing
    Thing objects and their derivatives.
    """
    
    def __init__(self):
        """Initialize a new CognitiveObjectManager."""
        self.objects: Dict[str, Thing] = {}
    
    def create_thing(self, name: str, shape_equation_str: Optional[str] = None, 
                     object_type: str = "thing", **kwargs) -> Thing:
        """
        Create a new Thing object or one of its derivatives.
        
        Args:
            name: The name of the object
            shape_equation_str: A string representing the dynamic shape equation
            object_type: The type of object to create ("thing", "physical", "concept", "living_being", "human")
            **kwargs: Additional properties to set during initialization
            
        Returns:
            The newly created object
            
        Raises:
            ValueError: If the object_type is not recognized
        """
        # Create the appropriate object based on the type
        if object_type.lower() == "thing":
            obj = Thing(name, shape_equation_str, **kwargs)
        elif object_type.lower() == "physical":
            obj = PhysicalObject(name, shape_equation_str, **kwargs)
        elif object_type.lower() == "concept":
            obj = ConceptObject(name, shape_equation_str, **kwargs)
        elif object_type.lower() == "living_being":
            obj = LivingBeing(name, shape_equation_str, **kwargs)
        elif object_type.lower() == "human":
            obj = Human(name, shape_equation_str, **kwargs)
        else:
            raise ValueError(f"Unknown object type: {object_type}")
        
        # Store the object
        self.objects[obj.get_id()] = obj
        
        return obj
    
    def get_thing_by_id(self, object_id: str) -> Optional[Thing]:
        """
        Get a Thing object by its ID.
        
        Args:
            object_id: The ID of the object to retrieve
            
        Returns:
            The object with the specified ID, or None if not found
        """
        return self.objects.get(object_id)
    
    def get_things_by_name(self, name: str) -> List[Thing]:
        """
        Get all Thing objects with the specified name.
        
        Args:
            name: The name to search for
            
        Returns:
            A list of objects with the specified name
        """
        return [obj for obj in self.objects.values() if obj.get_name() == name]
    
    def get_things_by_type(self, object_type: Type[Thing]) -> List[Thing]:
        """
        Get all Thing objects of the specified type.
        
        Args:
            object_type: The type of objects to retrieve
            
        Returns:
            A list of objects of the specified type
        """
        return [obj for obj in self.objects.values() if isinstance(obj, object_type)]
    
    def get_all_things(self) -> List[Thing]:
        """
        Get all Thing objects managed by this manager.
        
        Returns:
            A list of all objects
        """
        return list(self.objects.values())
    
    def remove_thing(self, object_id: str) -> bool:
        """
        Remove a Thing object by its ID.
        
        Args:
            object_id: The ID of the object to remove
            
        Returns:
            True if the object was removed, False if it wasn't found
        """
        if object_id in self.objects:
            del self.objects[object_id]
            return True
        return False
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save all objects to a JSON file.
        
        Args:
            file_path: The path to the file to save to
        """
        data = {obj_id: obj.to_dict() for obj_id, obj in self.objects.items()}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load objects from a JSON file.
        
        Args:
            file_path: The path to the file to load from
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for obj_id, obj_data in data.items():
            obj_type = obj_data.get("type", "Thing")
            
            if obj_type == "Thing":
                obj = Thing.from_dict(obj_data)
            elif obj_type == "PhysicalObject":
                obj = PhysicalObject.from_dict(obj_data)
            elif obj_type == "ConceptObject":
                obj = ConceptObject.from_dict(obj_data)
            elif obj_type == "LivingBeing":
                obj = LivingBeing.from_dict(obj_data)
            elif obj_type == "Human":
                obj = Human.from_dict(obj_data)
            else:
                # Default to Thing if type is not recognized
                obj = Thing.from_dict(obj_data)
            
            self.objects[obj.get_id()] = obj
    
    def clear(self) -> None:
        """Remove all objects from this manager."""
        self.objects.clear()


# Example usage
if __name__ == "__main__":
    # Create a manager
    manager = CognitiveObjectManager()
    
    # Create a simple Thing
    simple_thing = manager.create_thing(
        name="Simple Thing",
        shape_equation_str="Circle(x=100, y=150, radius=50) {color: #FF0000, stroke_width: 2, fill: true} @radius=[(0, 50), (1, 75), (2, 50)]"
    )
    
    # Create a physical object
    physical_obj = manager.create_thing(
        name="Red Ball",
        shape_equation_str="Circle(x=200, y=200, radius=30) {color: #FF0000, fill: true}",
        object_type="physical",
        mass=0.5,
        material="rubber"
    )
    
    # Create a concept
    justice_concept = manager.create_thing(
        name="Justice",
        shape_equation_str="Rectangle(x=100, y=100, width=100, height=100) {color: #0000FF, fill: false}",
        object_type="concept",
        abstraction_level=5,
        domain="ethics",
        definition="The quality of being fair and reasonable"
    )
    
    # Create a human
    human = manager.create_thing(
        name="John Doe",
        shape_equation_str="Circle(x=150, y=150, radius=25) {color: #000000, fill: true}",
        object_type="human",
        age=30,
        occupation="Engineer"
    )
    
    # Print some information
    print(f"Created {len(manager.get_all_things())} objects:")
    for obj in manager.get_all_things():
        print(f"- {obj}")
    
    # Get and modify an object
    john = manager.get_things_by_name("John Doe")[0]
    john.add_skill("Programming")
    john.add_skill("Mathematics")
    
    # Print John's skills
    print(f"\n{john.get_name()}'s skills: {john.get_skills()}")
    
    # Save to file
    manager.save_to_file("cognitive_objects_example.json")
    print("\nSaved objects to cognitive_objects_example.json")
