#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Understanding Component for General Shape Equation

This module implements the semantic understanding component for the General Shape Equation,
which adds support for understanding and representing meaning in language and symbols.

Author: Baserah System Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import copy
import re

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import from core module
try:
    from core.general_shape_equation import (
        GeneralShapeEquation,
        EquationType,
        LearningMode,
        SymbolicExpression
    )
except ImportError:
    logging.error("Failed to import from core.general_shape_equation")
    sys.exit(1)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch not available, some functionality will be limited")
    TORCH_AVAILABLE = False

# Configure logging
logger = logging.getLogger('core.semantic_understanding')


class SemanticRelationType(str, Enum):
    """Types of semantic relations."""
    IS_A = "is_a"                  # Hyponymy (is-a) relation
    PART_OF = "part_of"            # Meronymy (part-of) relation
    HAS_PROPERTY = "has_property"  # Property relation
    CAUSES = "causes"              # Causal relation
    PRECEDES = "precedes"          # Temporal precedence relation
    SIMILAR_TO = "similar_to"      # Similarity relation
    OPPOSITE_OF = "opposite_of"    # Antonymy (opposite) relation
    USED_FOR = "used_for"          # Functional relation
    LOCATED_IN = "located_in"      # Spatial relation
    DERIVED_FROM = "derived_from"  # Derivational relation
    SYMBOLIZES = "symbolizes"      # Symbolic relation
    ASSOCIATED_WITH = "associated_with"  # General association


@dataclass
class SemanticConcept:
    """A semantic concept in the semantic network."""
    concept_id: str
    name: str
    concept_type: str
    definition: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    symbolic_expression: Optional[SymbolicExpression] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticRelation:
    """A semantic relation between concepts in the semantic network."""
    relation_id: str
    source_id: str
    target_id: str
    relation_type: SemanticRelationType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticUnderstandingComponent:
    """
    Semantic Understanding Component for the General Shape Equation.
    
    This class extends the General Shape Equation with semantic understanding capabilities,
    including concept representation, semantic relations, and meaning extraction.
    """
    
    def __init__(self, equation: GeneralShapeEquation):
        """
        Initialize the semantic understanding component.
        
        Args:
            equation: The General Shape Equation to extend
        """
        self.equation = equation
        self.concepts = {}
        self.relations = {}
        self.concept_embeddings = {}
        self.letter_meanings = {}
        
        # Add semantic understanding components to the equation
        self._initialize_semantic_components()
        
        # Initialize semantic models if PyTorch is available
        if TORCH_AVAILABLE:
            self._initialize_semantic_models()
    
    def _initialize_semantic_components(self) -> None:
        """Initialize the semantic understanding components in the equation."""
        # Add basic semantic components
        self.equation.add_component("semantic_similarity", "cosine_similarity(embedding1, embedding2)")
        self.equation.add_component("semantic_distance", "1.0 - semantic_similarity")
        self.equation.add_component("semantic_coherence", "average_similarity(embeddings)")
        self.equation.add_component("semantic_relevance", "similarity(query_embedding, concept_embedding)")
        
        # Add letter meaning components
        self.equation.add_component("letter_meaning", "extract_meaning(letter, context)")
        self.equation.add_component("word_meaning", "combine_letter_meanings(letters)")
        
        # Add semantic network components
        self.equation.add_component("concept_activation", "sum(relation_weights * related_concept_activations)")
        self.equation.add_component("semantic_spreading", "propagate_activation(source_concept, activation_threshold)")
    
    def _initialize_semantic_models(self) -> None:
        """Initialize semantic models if PyTorch is available."""
        # Text encoder for generating embeddings
        class TextEncoder(nn.Module):
            def __init__(self, vocab_size=10000, embedding_dim=300, hidden_dim=512, output_dim=300):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(hidden_dim * 2, output_dim)
            
            def forward(self, x):
                embedded = self.embedding(x)
                output, (hidden, _) = self.encoder(embedded)
                # Concatenate the final hidden states from both directions
                hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
                return self.fc(hidden)
        
        # Letter meaning extractor
        class LetterMeaningExtractor(nn.Module):
            def __init__(self, input_dim=300, hidden_dim=256, output_dim=300):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)
        
        # Semantic relation classifier
        class RelationClassifier(nn.Module):
            def __init__(self, input_dim=600, hidden_dim=256, num_relations=12):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, num_relations)
                self.relu = nn.ReLU()
            
            def forward(self, x1, x2):
                # Concatenate the two concept embeddings
                x = torch.cat((x1, x2), dim=1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)
        
        # Initialize models with placeholder dimensions
        self.semantic_models = {
            "text_encoder": TextEncoder(),
            "letter_extractor": LetterMeaningExtractor(),
            "relation_classifier": RelationClassifier()
        }
        
        # Initialize optimizers
        self.optimizers = {
            "text_encoder": torch.optim.Adam(self.semantic_models["text_encoder"].parameters(), lr=0.001),
            "letter_extractor": torch.optim.Adam(self.semantic_models["letter_extractor"].parameters(), lr=0.001),
            "relation_classifier": torch.optim.Adam(self.semantic_models["relation_classifier"].parameters(), lr=0.001)
        }
    
    def add_concept(self, name: str, concept_type: str, definition: Optional[str] = None,
                   properties: Optional[Dict[str, Any]] = None, embedding: Optional[np.ndarray] = None,
                   symbolic_expression: Optional[str] = None) -> str:
        """
        Add a semantic concept to the network.
        
        Args:
            name: Name of the concept
            concept_type: Type of the concept
            definition: Definition of the concept (optional)
            properties: Properties of the concept (optional)
            embedding: Embedding vector of the concept (optional)
            symbolic_expression: Symbolic expression representing the concept (optional)
            
        Returns:
            ID of the created concept
        """
        concept_id = f"concept_{int(time.time())}_{len(self.concepts)}"
        
        # Create symbolic expression if provided
        symbolic_expr = None
        if symbolic_expression:
            symbolic_expr = SymbolicExpression(symbolic_expression)
        
        # Create concept
        concept = SemanticConcept(
            concept_id=concept_id,
            name=name,
            concept_type=concept_type,
            definition=definition,
            properties=properties or {},
            embedding=embedding,
            symbolic_expression=symbolic_expr
        )
        
        # Add concept to the network
        self.concepts[concept_id] = concept
        
        # Generate embedding if not provided and models are available
        if embedding is None and TORCH_AVAILABLE and "text_encoder" in self.semantic_models:
            self._generate_concept_embedding(concept)
        
        # Add concept to the equation
        self._add_concept_to_equation(concept)
        
        return concept_id
    
    def add_relation(self, source_id: str, target_id: str, relation_type: SemanticRelationType,
                    weight: float = 1.0, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a semantic relation between concepts.
        
        Args:
            source_id: ID of the source concept
            target_id: ID of the target concept
            relation_type: Type of the relation
            weight: Weight of the relation (0.0 to 1.0)
            properties: Properties of the relation (optional)
            
        Returns:
            ID of the created relation
        """
        if source_id not in self.concepts:
            raise ValueError(f"Source concept with ID {source_id} not found")
        
        if target_id not in self.concepts:
            raise ValueError(f"Target concept with ID {target_id} not found")
        
        relation_id = f"relation_{int(time.time())}_{len(self.relations)}"
        
        # Create relation
        relation = SemanticRelation(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            properties=properties or {}
        )
        
        # Add relation to the network
        self.relations[relation_id] = relation
        
        # Add relation to the equation
        self._add_relation_to_equation(relation)
        
        return relation_id
    
    def _generate_concept_embedding(self, concept: SemanticConcept) -> None:
        """
        Generate an embedding for a concept.
        
        Args:
            concept: The concept to generate an embedding for
        """
        if not TORCH_AVAILABLE or "text_encoder" not in self.semantic_models:
            return
        
        # This is a placeholder implementation
        # In a real implementation, this would use the text encoder to generate an embedding
        # based on the concept name, definition, and properties
        
        # For now, just generate a random embedding
        embedding_dim = 300  # Placeholder dimension
        concept.embedding = np.random.randn(embedding_dim)
    
    def _add_concept_to_equation(self, concept: SemanticConcept) -> None:
        """
        Add a concept to the equation.
        
        Args:
            concept: The concept to add
        """
        # Add concept as a component to the equation
        component_name = f"concept_{concept.name.lower().replace(' ', '_')}"
        
        if concept.symbolic_expression:
            # If the concept has a symbolic expression, use it
            self.equation.add_component(component_name, concept.symbolic_expression.expression_str)
        else:
            # Otherwise, create a simple representation
            self.equation.add_component(component_name, f"semantic_concept('{concept.name}')")
    
    def _add_relation_to_equation(self, relation: SemanticRelation) -> None:
        """
        Add a relation to the equation.
        
        Args:
            relation: The relation to add
        """
        # Add relation as a component to the equation
        source_concept = self.concepts[relation.source_id]
        target_concept = self.concepts[relation.target_id]
        
        component_name = f"relation_{source_concept.name.lower().replace(' ', '_')}_{relation.relation_type.value}_{target_concept.name.lower().replace(' ', '_')}"
        
        # Create a symbolic representation of the relation
        relation_expr = f"{relation.relation_type.value}('{source_concept.name}', '{target_concept.name}', {relation.weight})"
        
        self.equation.add_component(component_name, relation_expr)
    
    def get_concept(self, concept_id: str) -> Optional[SemanticConcept]:
        """
        Get a concept by ID.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            The concept or None if not found
        """
        return self.concepts.get(concept_id)
    
    def get_relation(self, relation_id: str) -> Optional[SemanticRelation]:
        """
        Get a relation by ID.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            The relation or None if not found
        """
        return self.relations.get(relation_id)
    
    def find_concepts_by_name(self, name: str, partial_match: bool = False) -> List[SemanticConcept]:
        """
        Find concepts by name.
        
        Args:
            name: Name to search for
            partial_match: Whether to allow partial matches
            
        Returns:
            List of matching concepts
        """
        if partial_match:
            return [c for c in self.concepts.values() if name.lower() in c.name.lower()]
        else:
            return [c for c in self.concepts.values() if c.name.lower() == name.lower()]
    
    def find_concepts_by_type(self, concept_type: str) -> List[SemanticConcept]:
        """
        Find concepts by type.
        
        Args:
            concept_type: Type to search for
            
        Returns:
            List of matching concepts
        """
        return [c for c in self.concepts.values() if c.concept_type.lower() == concept_type.lower()]
    
    def find_relations_by_type(self, relation_type: SemanticRelationType) -> List[SemanticRelation]:
        """
        Find relations by type.
        
        Args:
            relation_type: Type to search for
            
        Returns:
            List of matching relations
        """
        return [r for r in self.relations.values() if r.relation_type == relation_type]
    
    def get_related_concepts(self, concept_id: str, relation_type: Optional[SemanticRelationType] = None) -> List[SemanticConcept]:
        """
        Get concepts related to a given concept.
        
        Args:
            concept_id: ID of the concept
            relation_type: Type of relation to filter by (optional)
            
        Returns:
            List of related concepts
        """
        if concept_id not in self.concepts:
            return []
        
        related_concepts = []
        
        # Find relations where the concept is the source
        for relation in self.relations.values():
            if relation.source_id == concept_id:
                if relation_type is None or relation.relation_type == relation_type:
                    target_concept = self.concepts.get(relation.target_id)
                    if target_concept:
                        related_concepts.append(target_concept)
        
        # Find relations where the concept is the target
        for relation in self.relations.values():
            if relation.target_id == concept_id:
                if relation_type is None or relation.relation_type == relation_type:
                    source_concept = self.concepts.get(relation.source_id)
                    if source_concept:
                        related_concepts.append(source_concept)
        
        return related_concepts
    
    def compute_semantic_similarity(self, concept1_id: str, concept2_id: str) -> float:
        """
        Compute semantic similarity between two concepts.
        
        Args:
            concept1_id: ID of the first concept
            concept2_id: ID of the second concept
            
        Returns:
            Semantic similarity (0.0 to 1.0)
        """
        concept1 = self.concepts.get(concept1_id)
        concept2 = self.concepts.get(concept2_id)
        
        if not concept1 or not concept2:
            return 0.0
        
        # If both concepts have embeddings, compute cosine similarity
        if concept1.embedding is not None and concept2.embedding is not None:
            # Compute cosine similarity
            dot_product = np.dot(concept1.embedding, concept2.embedding)
            norm1 = np.linalg.norm(concept1.embedding)
            norm2 = np.linalg.norm(concept2.embedding)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        # If embeddings are not available, use a simple heuristic
        # based on shared relations
        related_to_concept1 = set(r.target_id for r in self.relations.values() if r.source_id == concept1_id)
        related_to_concept2 = set(r.target_id for r in self.relations.values() if r.source_id == concept2_id)
        
        # Compute Jaccard similarity
        intersection = len(related_to_concept1.intersection(related_to_concept2))
        union = len(related_to_concept1.union(related_to_concept2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def extract_letter_meaning(self, letter: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract the meaning of an Arabic letter.
        
        Args:
            letter: The Arabic letter
            context: Context for the letter (optional)
            
        Returns:
            Dictionary containing the meaning and properties of the letter
        """
        # Check if letter meaning is already cached
        if letter in self.letter_meanings:
            return self.letter_meanings[letter]
        
        # Placeholder implementation
        # In a real implementation, this would use more sophisticated methods
        # to extract the meaning of the letter based on its properties and context
        
        # Basic meanings for some Arabic letters (very simplified)
        basic_meanings = {
            'ا': {'meaning': 'الألف', 'properties': {'first_letter': True, 'symbolic': 'unity, beginning'}},
            'ب': {'meaning': 'الباء', 'properties': {'symbolic': 'house, container'}},
            'ت': {'meaning': 'التاء', 'properties': {'symbolic': 'completion, femininity'}},
            'ث': {'meaning': 'الثاء', 'properties': {'symbolic': 'abundance, multiplicity'}},
            'ج': {'meaning': 'الجيم', 'properties': {'symbolic': 'gathering, beauty'}},
            'ح': {'meaning': 'الحاء', 'properties': {'symbolic': 'life, boundary'}},
            'خ': {'meaning': 'الخاء', 'properties': {'symbolic': 'emptiness, void'}},
            'د': {'meaning': 'الدال', 'properties': {'symbolic': 'door, entrance'}},
            'ذ': {'meaning': 'الذال', 'properties': {'symbolic': 'possession, memory'}},
            'ر': {'meaning': 'الراء', 'properties': {'symbolic': 'head, leadership'}},
            'ز': {'meaning': 'الزاي', 'properties': {'symbolic': 'ornament, decoration'}},
            'س': {'meaning': 'السين', 'properties': {'symbolic': 'support, foundation'}},
            'ش': {'meaning': 'الشين', 'properties': {'symbolic': 'spread, dispersion'}},
            'ص': {'meaning': 'الصاد', 'properties': {'symbolic': 'clarity, purity'}},
            'ض': {'meaning': 'الضاد', 'properties': {'symbolic': 'necessity, obligation'}},
            'ط': {'meaning': 'الطاء', 'properties': {'symbolic': 'goodness, blessing'}},
            'ظ': {'meaning': 'الظاء', 'properties': {'symbolic': 'appearance, manifestation'}},
            'ع': {'meaning': 'العين', 'properties': {'symbolic': 'eye, source'}},
            'غ': {'meaning': 'الغين', 'properties': {'symbolic': 'hidden, unseen'}},
            'ف': {'meaning': 'الفاء', 'properties': {'symbolic': 'opening, revelation'}},
            'ق': {'meaning': 'القاف', 'properties': {'symbolic': 'power, strength'}},
            'ك': {'meaning': 'الكاف', 'properties': {'symbolic': 'palm, grasp'}},
            'ل': {'meaning': 'اللام', 'properties': {'symbolic': 'authority, direction'}},
            'م': {'meaning': 'الميم', 'properties': {'symbolic': 'water, fluidity'}},
            'ن': {'meaning': 'النون', 'properties': {'symbolic': 'fish, hidden potential'}},
            'ه': {'meaning': 'الهاء', 'properties': {'symbolic': 'essence, identity'}},
            'و': {'meaning': 'الواو', 'properties': {'symbolic': 'connection, conjunction'}},
            'ي': {'meaning': 'الياء', 'properties': {'symbolic': 'hand, ability'}}
        }
        
        # Get basic meaning or create a placeholder
        meaning = basic_meanings.get(letter, {'meaning': f'حرف {letter}', 'properties': {}})
        
        # Add context-specific meaning if context is provided
        if context:
            # Placeholder for context-specific meaning
            meaning['context'] = context
            meaning['context_meaning'] = f"Meaning of {letter} in context: {context}"
        
        # Cache the meaning
        self.letter_meanings[letter] = meaning
        
        return meaning
    
    def analyze_word_semantics(self, word: str) -> Dict[str, Any]:
        """
        Analyze the semantics of an Arabic word based on its letters.
        
        Args:
            word: The Arabic word to analyze
            
        Returns:
            Dictionary containing the semantic analysis of the word
        """
        # Extract meanings of individual letters
        letter_meanings = [self.extract_letter_meaning(letter, word) for letter in word]
        
        # Combine letter meanings (placeholder implementation)
        # In a real implementation, this would use more sophisticated methods
        combined_meaning = {
            'word': word,
            'letter_meanings': letter_meanings,
            'root_letters': self._extract_root_letters(word),
            'symbolic_meaning': self._combine_letter_symbolism(letter_meanings),
            'properties': {}
        }
        
        return combined_meaning
    
    def _extract_root_letters(self, word: str) -> List[str]:
        """
        Extract the root letters of an Arabic word.
        
        Args:
            word: The Arabic word
            
        Returns:
            List of root letters
        """
        # Placeholder implementation
        # In a real implementation, this would use more sophisticated methods
        # to extract the root letters based on morphological analysis
        
        # Simple heuristic: remove common prefixes and suffixes
        # and keep consonants (very simplified)
        prefixes = ['ال', 'و', 'ف', 'ب', 'ل', 'لل']
        suffixes = ['ة', 'ات', 'ون', 'ين', 'ان', 'ه', 'ها', 'هم', 'هن', 'ك', 'كم', 'كن', 'ي', 'نا']
        
        # Remove prefixes
        for prefix in prefixes:
            if word.startswith(prefix):
                word = word[len(prefix):]
                break
        
        # Remove suffixes
        for suffix in suffixes:
            if word.endswith(suffix):
                word = word[:-len(suffix)]
                break
        
        # Keep consonants (simplified)
        vowels = ['ا', 'و', 'ي', 'ى', 'ة']
        root_letters = [letter for letter in word if letter not in vowels]
        
        # Limit to 3 letters (common root pattern)
        if len(root_letters) > 3:
            root_letters = root_letters[:3]
        
        return root_letters
    
    def _combine_letter_symbolism(self, letter_meanings: List[Dict[str, Any]]) -> str:
        """
        Combine the symbolic meanings of letters.
        
        Args:
            letter_meanings: List of letter meanings
            
        Returns:
            Combined symbolic meaning
        """
        # Extract symbolic meanings
        symbolism = [m.get('properties', {}).get('symbolic', '') for m in letter_meanings]
        symbolism = [s for s in symbolism if s]
        
        if not symbolism:
            return "No symbolic meaning available"
        
        # Simple combination (placeholder)
        return " + ".join(symbolism)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the semantic understanding component to a dictionary.
        
        Returns:
            Dictionary representation of the semantic understanding component
        """
        return {
            "concepts": {concept_id: self._concept_to_dict(concept) for concept_id, concept in self.concepts.items()},
            "relations": {relation_id: self._relation_to_dict(relation) for relation_id, relation in self.relations.items()},
            "letter_meanings": self.letter_meanings
        }
    
    def _concept_to_dict(self, concept: SemanticConcept) -> Dict[str, Any]:
        """
        Convert a semantic concept to a dictionary.
        
        Args:
            concept: The semantic concept to convert
            
        Returns:
            Dictionary representation of the semantic concept
        """
        result = {
            "concept_id": concept.concept_id,
            "name": concept.name,
            "concept_type": concept.concept_type,
            "definition": concept.definition,
            "properties": concept.properties,
            "metadata": concept.metadata
        }
        
        # Convert embedding to list if present
        if concept.embedding is not None:
            result["embedding"] = concept.embedding.tolist()
        
        # Convert symbolic expression to string if present
        if concept.symbolic_expression is not None:
            result["symbolic_expression"] = concept.symbolic_expression.to_string()
        
        return result
    
    def _relation_to_dict(self, relation: SemanticRelation) -> Dict[str, Any]:
        """
        Convert a semantic relation to a dictionary.
        
        Args:
            relation: The semantic relation to convert
            
        Returns:
            Dictionary representation of the semantic relation
        """
        return {
            "relation_id": relation.relation_id,
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "relation_type": relation.relation_type.value,
            "weight": relation.weight,
            "properties": relation.properties,
            "metadata": relation.metadata
        }
    
    def to_json(self) -> str:
        """
        Convert the semantic understanding component to a JSON string.
        
        Returns:
            JSON string representation of the semantic understanding component
        """
        return json.dumps(self.to_dict(), indent=2)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a General Shape Equation
    equation = GeneralShapeEquation(
        equation_type=EquationType.SEMANTIC,
        learning_mode=LearningMode.HYBRID
    )
    
    # Create a semantic understanding component
    semantic = SemanticUnderstandingComponent(equation)
    
    # Add some concepts
    tree_id = semantic.add_concept("شجرة", "natural_object", "نبات خشبي معمر له ساق وأغصان")
    oak_id = semantic.add_concept("بلوط", "tree_species", "نوع من أنواع الأشجار")
    forest_id = semantic.add_concept("غابة", "ecosystem", "مساحة كبيرة من الأرض مغطاة بالأشجار")
    
    # Add relations
    semantic.add_relation(oak_id, tree_id, SemanticRelationType.IS_A)
    semantic.add_relation(tree_id, forest_id, SemanticRelationType.PART_OF)
    
    # Analyze word semantics
    word_analysis = semantic.analyze_word_semantics("علم")
    
    print("Word Analysis:")
    print(f"Word: {word_analysis['word']}")
    print(f"Root Letters: {word_analysis['root_letters']}")
    print(f"Symbolic Meaning: {word_analysis['symbolic_meaning']}")
    
    # Extract letter meaning
    alef_meaning = semantic.extract_letter_meaning('ا')
    
    print("\nLetter Meaning:")
    print(f"Letter: ا")
    print(f"Meaning: {alef_meaning['meaning']}")
    print(f"Properties: {alef_meaning['properties']}")
    
    # Convert to JSON
    json_str = semantic.to_json()
    print("\nJSON representation:", json_str)
