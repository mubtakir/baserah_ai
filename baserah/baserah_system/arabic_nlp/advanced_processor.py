#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Arabic NLP Processor for Baserah System

This module implements an advanced Arabic NLP processor that integrates
morphological analysis, syntactic parsing, semantic understanding, and
rhetorical analysis using the General Shape Equation.

Author: Baserah System Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import numpy as np
import time
import re
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import copy
import uuid

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
    from core.semantic_understanding import (
        SemanticUnderstandingComponent,
        SemanticConcept,
        SemanticRelation,
        SemanticRelationType
    )
except ImportError:
    logging.error("Failed to import from core modules")
    sys.exit(1)

# Try to import from Arabic NLP modules
try:
    from arabic_nlp.morphology.root_extractor import ArabicRootExtractor
    from arabic_nlp.syntax.syntax_analyzer import ArabicSyntaxAnalyzer, ArabicToken, SyntacticRelation
    from arabic_nlp.rhetoric.rhetoric_analyzer import ArabicRhetoricAnalyzer, RhetoricalFeature
except ImportError:
    logging.warning("Failed to import from Arabic NLP modules, using placeholder implementations")

    # Placeholder implementations
    class ArabicRootExtractor:
        def extract_roots(self, text):
            return [(word, word) for word in text.split()]

    class ArabicToken:
        def __init__(self, text, position=0, length=0, pos_tag=None):
            self.text = text
            self.position = position
            self.length = length
            self.pos_tag = pos_tag

    class SyntacticRelation:
        def __init__(self, relation_type, head_index, dependent_index):
            self.relation_type = relation_type
            self.head_index = head_index
            self.dependent_index = dependent_index

    class SyntaxAnalysis:
        def __init__(self, sentence, tokens, relations):
            self.sentence = sentence
            self.tokens = tokens
            self.relations = relations

    class ArabicSyntaxAnalyzer:
        def analyze(self, text):
            sentences = text.split('.')
            return [SyntaxAnalysis(s, [ArabicToken(w) for w in s.split()], []) for s in sentences if s.strip()]

    class RhetoricalFeature:
        def __init__(self, device_type, text, start_index=0, end_index=0, confidence=1.0):
            self.device_type = device_type
            self.text = text
            self.start_index = start_index
            self.end_index = end_index
            self.confidence = confidence

    class RhetoricAnalysis:
        def __init__(self, text, features):
            self.text = text
            self.features = features
            self.summary = {}
            self.aesthetic_score = 0.5
            self.style_profile = {}

    class ArabicRhetoricAnalyzer:
        def analyze(self, text):
            return RhetoricAnalysis(text, [])

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch not available, some functionality will be limited")
    TORCH_AVAILABLE = False

# Configure logging
logger = logging.getLogger('arabic_nlp.advanced_processor')


class AnalysisLevel(str, Enum):
    """Analysis levels for Arabic NLP processing."""
    MORPHOLOGICAL = "morphological"  # Morphological analysis (roots, stems, affixes)
    SYNTACTIC = "syntactic"        # Syntactic analysis (parsing, POS tagging)
    SEMANTIC = "semantic"          # Semantic analysis (meaning, concepts)
    RHETORICAL = "rhetorical"      # Rhetorical analysis (style, devices)
    PRAGMATIC = "pragmatic"        # Pragmatic analysis (context, intent)
    COMPREHENSIVE = "comprehensive"  # All levels of analysis


@dataclass
class ArabicWord:
    """Representation of an Arabic word with its analysis."""
    text: str
    position: int
    length: int
    root: Optional[str] = None
    stem: Optional[str] = None
    prefixes: List[str] = field(default_factory=list)
    suffixes: List[str] = field(default_factory=list)
    pos_tag: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    semantic_concepts: List[str] = field(default_factory=list)
    letter_meanings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    symbolic_meaning: Optional[str] = None


@dataclass
class ArabicSentence:
    """Representation of an Arabic sentence with its analysis."""
    text: str
    position: int
    length: int
    words: List[ArabicWord] = field(default_factory=list)
    syntactic_relations: List[SyntacticRelation] = field(default_factory=list)
    rhetorical_features: List[RhetoricalFeature] = field(default_factory=list)
    sentence_type: Optional[str] = None
    semantic_concepts: List[str] = field(default_factory=list)
    parse_tree: Optional[Dict[str, Any]] = None


@dataclass
class ArabicText:
    """Representation of an Arabic text with its analysis."""
    text: str
    sentences: List[ArabicSentence] = field(default_factory=list)
    global_concepts: List[str] = field(default_factory=list)
    rhetorical_features: List[RhetoricalFeature] = field(default_factory=list)
    aesthetic_score: float = 0.0
    style_profile: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArabicNLPProcessor:
    """
    Advanced Arabic NLP Processor for Baserah System.

    This class implements an advanced Arabic NLP processor that integrates
    morphological analysis, syntactic parsing, semantic understanding, and
    rhetorical analysis using the General Shape Equation.
    """

    def __init__(self):
        """Initialize the advanced Arabic NLP processor."""
        # Initialize General Shape Equation
        self.equation = GeneralShapeEquation(
            equation_type=EquationType.SEMANTIC,
            learning_mode=LearningMode.HYBRID
        )

        # Initialize semantic understanding component
        self.semantic = SemanticUnderstandingComponent(self.equation)

        # Initialize Arabic NLP components
        self.root_extractor = ArabicRootExtractor()
        self.syntax_analyzer = ArabicSyntaxAnalyzer()
        self.rhetoric_analyzer = ArabicRhetoricAnalyzer()

        # Initialize equation components
        self._initialize_equation_components()

        # Load Arabic letter meanings
        self._load_letter_meanings()

    def _initialize_equation_components(self) -> None:
        """Initialize the components of the General Shape Equation."""
        # Add components for morphological analysis
        self.equation.add_component("root_extraction", "extract_root(word)")
        self.equation.add_component("stemming", "extract_stem(word)")
        self.equation.add_component("affix_analysis", "analyze_affixes(word)")

        # Add components for syntactic analysis
        self.equation.add_component("pos_tagging", "tag_pos(word, context)")
        self.equation.add_component("dependency_parsing", "parse_dependencies(sentence)")
        self.equation.add_component("constituency_parsing", "parse_constituency(sentence)")

        # Add components for semantic analysis
        self.equation.add_component("word_meaning", "extract_word_meaning(word, context)")
        self.equation.add_component("sentence_meaning", "extract_sentence_meaning(sentence)")
        self.equation.add_component("semantic_coherence", "measure_coherence(text)")

        # Add components for rhetorical analysis
        self.equation.add_component("rhetorical_devices", "identify_devices(text)")
        self.equation.add_component("aesthetic_evaluation", "evaluate_aesthetics(text)")
        self.equation.add_component("style_analysis", "analyze_style(text)")

    def _load_letter_meanings(self) -> None:
        """Load meanings of Arabic letters."""
        # Basic meanings for Arabic letters (simplified)
        self.letter_meanings = {
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

    def process_text(self, text: str, level: AnalysisLevel = AnalysisLevel.COMPREHENSIVE) -> ArabicText:
        """
        Process Arabic text with comprehensive analysis.

        Args:
            text: Arabic text to process
            level: Level of analysis to perform

        Returns:
            Processed Arabic text with analysis
        """
        # Create Arabic text object
        arabic_text = ArabicText(text=text)

        # Split text into sentences
        sentences = self._split_sentences(text)

        # Process each sentence
        position = 0
        for sentence_text in sentences:
            if not sentence_text.strip():
                continue

            # Find position in original text
            sentence_position = text.find(sentence_text, position)
            if sentence_position == -1:
                sentence_position = position

            # Process sentence
            sentence = self._process_sentence(
                sentence_text,
                sentence_position,
                len(sentence_text),
                level
            )

            # Add sentence to text
            arabic_text.sentences.append(sentence)

            # Update position
            position = sentence_position + len(sentence_text)

        # Perform global analysis if comprehensive
        if level == AnalysisLevel.COMPREHENSIVE or level == AnalysisLevel.RHETORICAL:
            # Analyze rhetoric
            rhetoric_analysis = self.rhetoric_analyzer.analyze(text)

            # Add rhetorical features
            arabic_text.rhetorical_features = rhetoric_analysis.features
            arabic_text.aesthetic_score = rhetoric_analysis.aesthetic_score
            arabic_text.style_profile = rhetoric_analysis.style_profile

        # Extract global concepts if comprehensive or semantic
        if level == AnalysisLevel.COMPREHENSIVE or level == AnalysisLevel.SEMANTIC:
            # Extract global concepts
            global_concepts = self._extract_global_concepts(arabic_text)
            arabic_text.global_concepts = global_concepts

        return arabic_text

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting based on punctuation
        # In a real implementation, this would be more sophisticated
        sentences = re.split(r'[.!?؟،\n]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _process_sentence(self, text: str, position: int, length: int, level: AnalysisLevel) -> ArabicSentence:
        """
        Process an Arabic sentence.

        Args:
            text: Sentence text
            position: Position in the original text
            length: Length of the sentence
            level: Level of analysis to perform

        Returns:
            Processed Arabic sentence with analysis
        """
        # Create Arabic sentence object
        sentence = ArabicSentence(
            text=text,
            position=position,
            length=length
        )

        # Split sentence into words
        words = self._split_words(text)

        # Process each word
        word_position = 0
        for word_text in words:
            if not word_text.strip():
                continue

            # Find position in sentence
            word_position = text.find(word_text, word_position)
            if word_position == -1:
                word_position = 0

            # Process word
            word = self._process_word(
                word_text,
                position + word_position,
                len(word_text),
                level
            )

            # Add word to sentence
            sentence.words.append(word)

            # Update position
            word_position += len(word_text)

        # Perform syntactic analysis if needed
        if level == AnalysisLevel.COMPREHENSIVE or level == AnalysisLevel.SYNTACTIC:
            # Analyze syntax
            syntax_analyses = self.syntax_analyzer.analyze(text)

            if syntax_analyses:
                syntax_analysis = syntax_analyses[0]  # Take the first analysis

                # Add syntactic relations
                sentence.syntactic_relations = syntax_analysis.relations

                # Add parse tree if available
                if hasattr(syntax_analysis, 'parse_tree'):
                    sentence.parse_tree = syntax_analysis.parse_tree

                # Add sentence type if available
                if hasattr(syntax_analysis, 'sentence_type'):
                    sentence.sentence_type = syntax_analysis.sentence_type

                # Update POS tags in words
                for i, token in enumerate(syntax_analysis.tokens):
                    if i < len(sentence.words):
                        sentence.words[i].pos_tag = token.pos_tag

        # Perform semantic analysis if needed
        if level == AnalysisLevel.COMPREHENSIVE or level == AnalysisLevel.SEMANTIC:
            # Extract sentence concepts
            sentence_concepts = self._extract_sentence_concepts(sentence)
            sentence.semantic_concepts = sentence_concepts

        # Perform rhetorical analysis if needed
        if level == AnalysisLevel.COMPREHENSIVE or level == AnalysisLevel.RHETORICAL:
            # Analyze rhetoric for the sentence
            rhetoric_analysis = self.rhetoric_analyzer.analyze(text)

            # Add rhetorical features
            sentence.rhetorical_features = rhetoric_analysis.features

        return sentence

    def _split_words(self, text: str) -> List[str]:
        """
        Split sentence into words.

        Args:
            text: Sentence text

        Returns:
            List of words
        """
        # Simple word splitting based on whitespace
        # In a real implementation, this would be more sophisticated
        return text.split()

    def _process_word(self, text: str, position: int, length: int, level: AnalysisLevel) -> ArabicWord:
        """
        Process an Arabic word.

        Args:
            text: Word text
            position: Position in the original text
            length: Length of the word
            level: Level of analysis to perform

        Returns:
            Processed Arabic word with analysis
        """
        # Create Arabic word object
        word = ArabicWord(
            text=text,
            position=position,
            length=length
        )

        # Perform morphological analysis if needed
        if level == AnalysisLevel.COMPREHENSIVE or level == AnalysisLevel.MORPHOLOGICAL:
            # Extract root
            roots = self.root_extractor.extract_roots(text)
            if roots:
                word.root = roots[0][1]  # Take the first root

            # Extract stem (placeholder)
            word.stem = text

            # Extract affixes (placeholder)
            # In a real implementation, this would be more sophisticated
            prefixes = []
            suffixes = []

            # Check for common prefixes
            common_prefixes = ['ال', 'و', 'ف', 'ب', 'ل', 'لل']
            for prefix in common_prefixes:
                if text.startswith(prefix):
                    prefixes.append(prefix)
                    break

            # Check for common suffixes
            common_suffixes = ['ة', 'ات', 'ون', 'ين', 'ان', 'ه', 'ها', 'هم', 'هن', 'ك', 'كم', 'كن', 'ي', 'نا']
            for suffix in common_suffixes:
                if text.endswith(suffix):
                    suffixes.append(suffix)
                    break

            word.prefixes = prefixes
            word.suffixes = suffixes

        # Perform semantic analysis if needed
        if level == AnalysisLevel.COMPREHENSIVE or level == AnalysisLevel.SEMANTIC:
            # Extract letter meanings
            letter_meanings = {}
            for i, letter in enumerate(text):
                if letter in self.letter_meanings:
                    letter_meanings[letter] = self.letter_meanings[letter]

            word.letter_meanings = letter_meanings

            # Extract symbolic meaning
            word.symbolic_meaning = self._extract_symbolic_meaning(text, letter_meanings)

            # Extract semantic concepts
            word_concepts = self._extract_word_concepts(text)
            word.semantic_concepts = word_concepts

        return word

    def _extract_symbolic_meaning(self, word: str, letter_meanings: Dict[str, Dict[str, Any]]) -> str:
        """
        Extract symbolic meaning of a word based on its letters.

        Args:
            word: Word text
            letter_meanings: Dictionary of letter meanings

        Returns:
            Symbolic meaning of the word
        """
        # Extract symbolic meanings of letters
        symbolism = []
        for letter in word:
            if letter in letter_meanings:
                symbol = letter_meanings[letter].get('properties', {}).get('symbolic')
                if symbol:
                    symbolism.append(symbol)

        if not symbolism:
            return "No symbolic meaning available"

        # Combine symbolic meanings
        return " + ".join(symbolism)

    def _extract_word_concepts(self, word: str) -> List[str]:
        """
        Extract semantic concepts related to a word.

        Args:
            word: Word text

        Returns:
            List of semantic concept IDs
        """
        # Find concepts by name
        concepts = self.semantic.find_concepts_by_name(word, partial_match=True)

        # If no concepts found, create a new one
        if not concepts:
            concept_id = self.semantic.add_concept(
                name=word,
                concept_type="word",
                definition=f"Word: {word}"
            )
            return [concept_id]

        return [concept.concept_id for concept in concepts]

    def _extract_sentence_concepts(self, sentence: ArabicSentence) -> List[str]:
        """
        Extract semantic concepts related to a sentence.

        Args:
            sentence: Sentence object

        Returns:
            List of semantic concept IDs
        """
        # Collect word concepts
        word_concepts = []
        for word in sentence.words:
            word_concepts.extend(word.semantic_concepts)

        # Create a sentence concept
        concept_id = self.semantic.add_concept(
            name=sentence.text[:50] + "..." if len(sentence.text) > 50 else sentence.text,
            concept_type="sentence",
            definition=sentence.text
        )

        # Link sentence concept to word concepts
        for word_concept_id in word_concepts:
            self.semantic.add_relation(
                concept_id,
                word_concept_id,
                SemanticRelationType.HAS_PROPERTY
            )

        return [concept_id] + word_concepts

    def _extract_global_concepts(self, text: ArabicText) -> List[str]:
        """
        Extract global semantic concepts related to a text.

        Args:
            text: Text object

        Returns:
            List of semantic concept IDs
        """
        # Collect sentence concepts
        sentence_concepts = []
        for sentence in text.sentences:
            sentence_concepts.extend(sentence.semantic_concepts)

        # Create a text concept
        concept_id = self.semantic.add_concept(
            name=text.text[:50] + "..." if len(text.text) > 50 else text.text,
            concept_type="text",
            definition=text.text[:200] + "..." if len(text.text) > 200 else text.text
        )

        # Link text concept to sentence concepts
        for sentence_concept_id in sentence_concepts:
            self.semantic.add_relation(
                concept_id,
                sentence_concept_id,
                SemanticRelationType.HAS_PROPERTY
            )

        return [concept_id] + sentence_concepts

    def analyze_word_semantics(self, word: str) -> Dict[str, Any]:
        """
        Analyze the semantics of an Arabic word.

        Args:
            word: Arabic word to analyze

        Returns:
            Dictionary with semantic analysis
        """
        return self.semantic.analyze_word_semantics(word)

    def extract_letter_meaning(self, letter: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract the meaning of an Arabic letter.

        Args:
            letter: Arabic letter
            context: Context for the letter (optional)

        Returns:
            Dictionary with letter meaning
        """
        return self.semantic.extract_letter_meaning(letter, context)

    def to_dict(self, arabic_text: ArabicText) -> Dict[str, Any]:
        """
        Convert Arabic text analysis to a dictionary.

        Args:
            arabic_text: Analyzed Arabic text

        Returns:
            Dictionary representation of the analysis
        """
        return {
            "text": arabic_text.text,
            "sentences": [self._sentence_to_dict(sentence) for sentence in arabic_text.sentences],
            "global_concepts": arabic_text.global_concepts,
            "rhetorical_features": [self._rhetorical_feature_to_dict(feature) for feature in arabic_text.rhetorical_features],
            "aesthetic_score": arabic_text.aesthetic_score,
            "style_profile": arabic_text.style_profile,
            "metadata": arabic_text.metadata
        }

    def _sentence_to_dict(self, sentence: ArabicSentence) -> Dict[str, Any]:
        """
        Convert Arabic sentence analysis to a dictionary.

        Args:
            sentence: Analyzed Arabic sentence

        Returns:
            Dictionary representation of the analysis
        """
        return {
            "text": sentence.text,
            "position": sentence.position,
            "length": sentence.length,
            "words": [self._word_to_dict(word) for word in sentence.words],
            "syntactic_relations": [self._relation_to_dict(relation) for relation in sentence.syntactic_relations],
            "rhetorical_features": [self._rhetorical_feature_to_dict(feature) for feature in sentence.rhetorical_features],
            "sentence_type": sentence.sentence_type,
            "semantic_concepts": sentence.semantic_concepts,
            "parse_tree": sentence.parse_tree
        }

    def _word_to_dict(self, word: ArabicWord) -> Dict[str, Any]:
        """
        Convert Arabic word analysis to a dictionary.

        Args:
            word: Analyzed Arabic word

        Returns:
            Dictionary representation of the analysis
        """
        return {
            "text": word.text,
            "position": word.position,
            "length": word.length,
            "root": word.root,
            "stem": word.stem,
            "prefixes": word.prefixes,
            "suffixes": word.suffixes,
            "pos_tag": word.pos_tag,
            "features": word.features,
            "semantic_concepts": word.semantic_concepts,
            "letter_meanings": word.letter_meanings,
            "symbolic_meaning": word.symbolic_meaning
        }

    def _relation_to_dict(self, relation: SyntacticRelation) -> Dict[str, Any]:
        """
        Convert syntactic relation to a dictionary.

        Args:
            relation: Syntactic relation

        Returns:
            Dictionary representation of the relation
        """
        return {
            "relation_type": relation.relation_type,
            "head_index": relation.head_index,
            "dependent_index": relation.dependent_index
        }

    def _rhetorical_feature_to_dict(self, feature: RhetoricalFeature) -> Dict[str, Any]:
        """
        Convert rhetorical feature to a dictionary.

        Args:
            feature: Rhetorical feature

        Returns:
            Dictionary representation of the feature
        """
        return {
            "device_type": feature.device_type,
            "text": feature.text,
            "start_index": feature.start_index,
            "end_index": feature.end_index,
            "confidence": feature.confidence
        }

    def to_json(self, arabic_text: ArabicText) -> str:
        """
        Convert Arabic text analysis to a JSON string.

        Args:
            arabic_text: Analyzed Arabic text

        Returns:
            JSON string representation of the analysis
        """
        return json.dumps(self.to_dict(arabic_text), ensure_ascii=False, indent=2)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create processor
    processor = AdvancedArabicProcessor()

    # Process text
    text = "العلم نور يضيء طريق الحياة. الجهل ظلام يحجب نور البصيرة."
    result = processor.process_text(text)

    # Print analysis
    print("Text:", result.text)
    print("Number of sentences:", len(result.sentences))

    for i, sentence in enumerate(result.sentences):
        print(f"\nSentence {i+1}: {sentence.text}")
        print(f"Number of words: {len(sentence.words)}")

        for j, word in enumerate(sentence.words):
            print(f"  Word {j+1}: {word.text}")
            if word.root:
                print(f"    Root: {word.root}")
            if word.symbolic_meaning:
                print(f"    Symbolic meaning: {word.symbolic_meaning}")

    # Analyze word semantics
    word_analysis = processor.analyze_word_semantics("علم")

    print("\nWord Analysis:")
    print(f"Word: {word_analysis['word']}")
    print(f"Root Letters: {word_analysis['root_letters']}")
    print(f"Symbolic Meaning: {word_analysis['symbolic_meaning']}")

    # Convert to JSON
    json_str = processor.to_json(result)
    print("\nJSON representation (excerpt):", json_str[:200] + "...")
