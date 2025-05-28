#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Arabic Syntax Analyzer for Basira System

This module implements an advanced Arabic syntax analyzer that provides
comprehensive grammatical analysis including I'rab (إعراب), sentence structure,
and syntactic relationships.

Author: Basira System Development Team
Version: 2.0.0 (Advanced)
"""

import re
import json
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import core components
try:
    from core.general_shape_equation import GeneralShapeEquation, EquationType, LearningMode
except ImportError as e:
    logging.warning(f"Could not import GeneralShapeEquation: {e}")
    # Define placeholder classes
    class EquationType:
        LINGUISTIC = "linguistic"
        SEMANTIC = "semantic"

    class LearningMode:
        SUPERVISED = "supervised"
        ADAPTIVE = "adaptive"

    class GeneralShapeEquation:
        def __init__(self, equation_type, learning_mode):
            self.equation_type = equation_type
            self.learning_mode = learning_mode

# Configure logging
logger = logging.getLogger('arabic_nlp.syntax.advanced_syntax_analyzer')


class WordType(Enum):
    """Arabic word types"""
    NOUN = "اسم"
    VERB = "فعل"
    PARTICLE = "حرف"
    PRONOUN = "ضمير"
    ADJECTIVE = "صفة"
    ADVERB = "ظرف"
    NUMBER = "عدد"
    UNKNOWN = "غير_محدد"


class VerbTense(Enum):
    """Arabic verb tenses"""
    PAST = "ماضي"
    PRESENT = "مضارع"
    IMPERATIVE = "أمر"
    UNKNOWN = "غير_محدد"


class VerbVoice(Enum):
    """Arabic verb voices"""
    ACTIVE = "معلوم"
    PASSIVE = "مجهول"
    UNKNOWN = "غير_محدد"


class CaseMarking(Enum):
    """Arabic case markings (إعراب)"""
    NOMINATIVE = "مرفوع"      # الرفع
    ACCUSATIVE = "منصوب"      # النصب
    GENITIVE = "مجرور"        # الجر
    JUSSIVE = "مجزوم"         # الجزم
    INDECLINABLE = "مبني"     # البناء
    UNKNOWN = "غير_محدد"


class SyntacticFunction(Enum):
    """Arabic syntactic functions"""
    SUBJECT = "فاعل"
    OBJECT = "مفعول_به"
    PREDICATE = "خبر"
    TOPIC = "مبتدأ"
    ADJECTIVE_MOD = "نعت"
    ADVERBIAL = "حال"
    CIRCUMSTANTIAL = "ظرف"
    GENITIVE_CONSTRUCT = "مضاف_إليه"
    PREPOSITION_OBJECT = "مجرور"
    UNKNOWN = "غير_محدد"


@dataclass
class AdvancedArabicToken:
    """Advanced Arabic token with comprehensive grammatical information"""
    text: str
    position: int
    length: int

    # Basic classification
    word_type: WordType = WordType.UNKNOWN
    root: Optional[str] = None
    pattern: Optional[str] = None

    # Verb-specific features
    verb_tense: Optional[VerbTense] = None
    verb_voice: Optional[VerbVoice] = None
    verb_person: Optional[str] = None  # أول، ثاني، ثالث
    verb_number: Optional[str] = None  # مفرد، مثنى، جمع
    verb_gender: Optional[str] = None  # مذكر، مؤنث

    # Noun-specific features
    noun_type: Optional[str] = None    # جامد، مشتق
    noun_number: Optional[str] = None  # مفرد، مثنى، جمع
    noun_gender: Optional[str] = None  # مذكر، مؤنث
    noun_definiteness: Optional[str] = None  # معرفة، نكرة

    # Grammatical analysis (إعراب)
    case_marking: CaseMarking = CaseMarking.UNKNOWN
    case_sign: Optional[str] = None    # علامة الإعراب (ضمة، فتحة، كسرة، سكون)
    syntactic_function: SyntacticFunction = SyntacticFunction.UNKNOWN

    # Additional features
    features: Dict[str, any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class SentenceStructure:
    """Arabic sentence structure analysis"""
    sentence_type: str  # اسمية، فعلية، شرطية، استفهامية، إلخ
    main_components: Dict[str, int]  # مبتدأ، خبر، فاعل، مفعول، إلخ
    subordinate_clauses: List[Dict] = field(default_factory=list)
    coordination: List[Dict] = field(default_factory=list)

    # Rhetorical features
    rhetorical_devices: List[str] = field(default_factory=list)
    emphasis_markers: List[str] = field(default_factory=list)


class AdvancedArabicSyntaxAnalyzer:
    """
    Advanced Arabic Syntax Analyzer with comprehensive grammatical analysis
    """

    def __init__(self):
        """Initialize the advanced syntax analyzer"""
        self.logger = logging.getLogger('arabic_nlp.syntax.advanced_syntax_analyzer.main')

        # Initialize General Shape Equation for linguistic analysis
        self.general_equation = GeneralShapeEquation(
            equation_type=EquationType.LINGUISTIC,
            learning_mode=LearningMode.ADAPTIVE
        )

        # Load grammatical rules and patterns
        self.grammatical_rules = self._load_grammatical_rules()
        self.verb_patterns = self._load_verb_patterns()
        self.noun_patterns = self._load_noun_patterns()
        self.particle_patterns = self._load_particle_patterns()

        # Load I'rab rules
        self.irab_rules = self._load_irab_rules()

        # Sentence pattern recognition
        self.sentence_patterns = self._load_sentence_patterns()

        self.logger.info("Advanced Arabic Syntax Analyzer initialized with General Shape Equation")

    def _load_grammatical_rules(self) -> Dict:
        """Load comprehensive grammatical rules"""
        return {
            "word_identification": {
                # Verb identification patterns
                "verb_prefixes": ["ي", "ت", "ن", "أ"],
                "verb_suffixes": ["ت", "تم", "تن", "وا", "ون", "ين", "ان"],
                "verb_patterns": [
                    r"^[يتنأ].+",  # مضارع
                    r".+[تمنوا]$",  # ماضي مع ضمائر
                ],

                # Noun identification patterns
                "noun_prefixes": ["ال", "م"],
                "noun_suffixes": ["ة", "ات", "ان", "ون", "ين"],
                "definite_markers": ["ال"],

                # Particle patterns
                "prepositions": ["في", "على", "إلى", "من", "عن", "ب", "ل", "ك"],
                "conjunctions": ["و", "ف", "ثم", "أو", "لكن", "بل"],
                "interrogatives": ["ما", "من", "متى", "أين", "كيف", "لماذا", "هل", "أ"],
            },

            "case_assignment": {
                "nominative_contexts": [
                    "subject_of_verb",      # فاعل
                    "topic_of_nominal",     # مبتدأ
                    "predicate_of_nominal", # خبر
                    "substitute_subject"    # نائب فاعل
                ],
                "accusative_contexts": [
                    "direct_object",        # مفعول به
                    "adverbial",           # حال
                    "circumstantial",      # ظرف
                    "absolute_object"      # مفعول مطلق
                ],
                "genitive_contexts": [
                    "after_preposition",   # مجرور بحرف الجر
                    "genitive_construct",  # مضاف إليه
                    "oath_object"          # مقسم به
                ]
            }
        }

    def _load_verb_patterns(self) -> Dict:
        """Load verb morphological patterns"""
        return {
            "trilateral": {
                "past": {
                    "فعل": {"pattern": "فعل", "example": "كتب"},
                    "فعِل": {"pattern": "فعِل", "example": "علم"},
                    "فعُل": {"pattern": "فعُل", "example": "كرم"}
                },
                "present": {
                    "يفعل": {"pattern": "يفعل", "example": "يكتب"},
                    "يفعِل": {"pattern": "يفعِل", "example": "يعلم"},
                    "يفعُل": {"pattern": "يفعُل", "example": "يكرم"}
                }
            },
            "derived": {
                "أفعل": {"pattern": "أفعل", "meaning": "causative"},
                "فعّل": {"pattern": "فعّل", "meaning": "intensive"},
                "فاعل": {"pattern": "فاعل", "meaning": "reciprocal"},
                "انفعل": {"pattern": "انفعل", "meaning": "reflexive"},
                "افتعل": {"pattern": "افتعل", "meaning": "reflexive"},
                "استفعل": {"pattern": "استفعل", "meaning": "seeking"},
                "تفعّل": {"pattern": "تفعّل", "meaning": "reflexive_intensive"},
                "تفاعل": {"pattern": "تفاعل", "meaning": "mutual"}
            }
        }

    def _load_noun_patterns(self) -> Dict:
        """Load noun morphological patterns"""
        return {
            "agent_nouns": {
                "فاعل": {"pattern": "فاعل", "example": "كاتب"},
                "مفعِل": {"pattern": "مفعِل", "example": "مدرس"},
                "فعّال": {"pattern": "فعّال", "example": "نجار"}
            },
            "patient_nouns": {
                "مفعول": {"pattern": "مفعول", "example": "مكتوب"},
                "فعيل": {"pattern": "فعيل", "example": "قتيل"}
            },
            "instrument_nouns": {
                "مفعال": {"pattern": "مفعال", "example": "مفتاح"},
                "مفعل": {"pattern": "مفعل", "example": "مقص"},
                "مفعلة": {"pattern": "مفعلة", "example": "مكنسة"}
            },
            "place_nouns": {
                "مفعل": {"pattern": "مفعل", "example": "مكتب"},
                "مفعلة": {"pattern": "مفعلة", "example": "مدرسة"}
            }
        }

    def _load_particle_patterns(self) -> Dict:
        """Load particle patterns and functions"""
        return {
            "prepositions": {
                "في": {"meaning": "in", "case": "genitive"},
                "على": {"meaning": "on", "case": "genitive"},
                "إلى": {"meaning": "to", "case": "genitive"},
                "من": {"meaning": "from", "case": "genitive"},
                "عن": {"meaning": "about", "case": "genitive"},
                "ب": {"meaning": "with", "case": "genitive"},
                "ل": {"meaning": "for", "case": "genitive"},
                "ك": {"meaning": "like", "case": "genitive"}
            },
            "conjunctions": {
                "و": {"type": "coordinating", "meaning": "and"},
                "ف": {"type": "coordinating", "meaning": "then"},
                "ثم": {"type": "coordinating", "meaning": "then"},
                "أو": {"type": "coordinating", "meaning": "or"},
                "لكن": {"type": "adversative", "meaning": "but"},
                "بل": {"type": "corrective", "meaning": "rather"}
            },
            "subordinating": {
                "أن": {"type": "complementizer", "function": "verbal_noun"},
                "إن": {"type": "conditional", "function": "condition"},
                "لو": {"type": "conditional", "function": "hypothetical"},
                "كي": {"type": "purpose", "function": "purpose"},
                "لكي": {"type": "purpose", "function": "purpose"}
            }
        }

    def _load_irab_rules(self) -> Dict:
        """Load I'rab (grammatical analysis) rules"""
        return {
            "subject_rules": {
                "verbal_sentence": {
                    "position": "after_verb",
                    "case": "nominative",
                    "agreement": ["number", "gender"]
                },
                "nominal_sentence": {
                    "position": "sentence_initial",
                    "case": "nominative",
                    "function": "topic"
                }
            },
            "object_rules": {
                "direct_object": {
                    "position": "after_subject",
                    "case": "accusative",
                    "conditions": ["transitive_verb"]
                }
            },
            "predicate_rules": {
                "nominal_predicate": {
                    "position": "after_topic",
                    "case": "nominative",
                    "agreement": ["number", "gender"]
                }
            }
        }

    def _load_sentence_patterns(self) -> Dict:
        """Load Arabic sentence patterns"""
        return {
            "nominal": {
                "basic": ["مبتدأ", "خبر"],
                "extended": ["مبتدأ", "خبر", "نعت"],
                "with_kana": ["كان", "اسم_كان", "خبر_كان"]
            },
            "verbal": {
                "basic": ["فعل", "فاعل"],
                "transitive": ["فعل", "فاعل", "مفعول_به"],
                "ditransitive": ["فعل", "فاعل", "مفعول_به_أول", "مفعول_به_ثاني"]
            },
            "conditional": {
                "basic": ["أداة_شرط", "فعل_الشرط", "جواب_الشرط"],
                "nominal_condition": ["أداة_شرط", "مبتدأ", "خبر", "جواب_الشرط"]
            }
        }

    def analyze_advanced(self, text: str) -> List[Dict]:
        """
        Perform advanced syntactic analysis

        Args:
            text: Arabic text to analyze

        Returns:
            List of sentence analyses with comprehensive grammatical information
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        analyses = []
        for sentence in sentences:
            analysis = self._analyze_sentence_advanced(sentence)
            analyses.append(analysis)

        return analyses

    def _analyze_sentence_advanced(self, sentence: str) -> Dict:
        """Perform advanced analysis of a single sentence"""

        # Tokenize
        tokens = self._tokenize_advanced(sentence)

        # Morphological analysis
        tokens = self._morphological_analysis(tokens)

        # Syntactic analysis
        tokens = self._syntactic_analysis(tokens)

        # I'rab analysis
        tokens = self._irab_analysis(tokens)

        # Sentence structure analysis
        structure = self._analyze_sentence_structure(tokens)

        return {
            "sentence": sentence,
            "tokens": tokens,
            "structure": structure,
            "analysis_confidence": self._calculate_analysis_confidence(tokens)
        }

    def _tokenize_advanced(self, sentence: str) -> List[AdvancedArabicToken]:
        """Advanced tokenization with position tracking"""

        # Simple whitespace tokenization (can be enhanced)
        words = sentence.split()

        tokens = []
        position = 0

        for word in words:
            # Find actual position in sentence
            word_start = sentence.find(word, position)

            token = AdvancedArabicToken(
                text=word,
                position=word_start,
                length=len(word)
            )

            tokens.append(token)
            position = word_start + len(word)

        return tokens

    def _morphological_analysis(self, tokens: List[AdvancedArabicToken]) -> List[AdvancedArabicToken]:
        """Perform morphological analysis on tokens"""

        for token in tokens:
            # Identify word type
            token.word_type = self._identify_word_type(token.text)

            # Extract root (simplified)
            token.root = self._extract_root_simple(token.text)

            # Identify pattern
            token.pattern = self._identify_pattern(token.text, token.word_type)

            # Verb-specific analysis
            if token.word_type == WordType.VERB:
                self._analyze_verb_features(token)

            # Noun-specific analysis
            elif token.word_type == WordType.NOUN:
                self._analyze_noun_features(token)

        return tokens

    def _identify_word_type(self, word: str) -> WordType:
        """Identify the type of Arabic word"""

        # Check for verb patterns
        verb_prefixes = self.grammatical_rules["word_identification"]["verb_prefixes"]
        if any(word.startswith(prefix) for prefix in verb_prefixes):
            return WordType.VERB

        # Check for definite article (noun indicator)
        if word.startswith("ال"):
            return WordType.NOUN

        # Check for prepositions
        prepositions = self.grammatical_rules["word_identification"]["prepositions"]
        if word in prepositions:
            return WordType.PARTICLE

        # Check for conjunctions
        conjunctions = self.grammatical_rules["word_identification"]["conjunctions"]
        if word in conjunctions:
            return WordType.PARTICLE

        # Default to noun (most common in Arabic)
        return WordType.NOUN

    def _extract_root_simple(self, word: str) -> Optional[str]:
        """Simple root extraction (placeholder for advanced extractor)"""

        # Remove common prefixes and suffixes
        processed = word

        # Remove definite article
        if processed.startswith("ال"):
            processed = processed[2:]

        # Remove common suffixes
        suffixes = ["ة", "ات", "ان", "ون", "ين"]
        for suffix in suffixes:
            if processed.endswith(suffix):
                processed = processed[:-len(suffix)]
                break

        # Return if reasonable length
        if len(processed) >= 3:
            return processed

        return None

    def _identify_pattern(self, word: str, word_type: WordType) -> Optional[str]:
        """Identify morphological pattern"""

        if word_type == WordType.VERB:
            # Check verb patterns
            for category, patterns in self.verb_patterns.items():
                if isinstance(patterns, dict):
                    for pattern_name, pattern_data in patterns.items():
                        if self._matches_pattern(word, pattern_data.get("pattern", "")):
                            return pattern_name

        elif word_type == WordType.NOUN:
            # Check noun patterns
            for category, patterns in self.noun_patterns.items():
                for pattern_name, pattern_data in patterns.items():
                    if self._matches_pattern(word, pattern_data.get("pattern", "")):
                        return pattern_name

        return None

    def _matches_pattern(self, word: str, pattern: str) -> bool:
        """Check if word matches morphological pattern (simplified)"""
        # This is a very simplified pattern matching
        # In reality, this would be much more sophisticated
        return len(word) == len(pattern)

    def _analyze_verb_features(self, token: AdvancedArabicToken) -> None:
        """Analyze verb-specific features"""

        word = token.text

        # Determine tense
        if any(word.startswith(prefix) for prefix in ["ي", "ت", "ن", "أ"]):
            token.verb_tense = VerbTense.PRESENT
        elif word.endswith(("ت", "تم", "تن", "وا")):
            token.verb_tense = VerbTense.PAST
        else:
            token.verb_tense = VerbTense.PAST  # Default

        # Determine voice (simplified)
        token.verb_voice = VerbVoice.ACTIVE  # Default

        # Person, number, gender analysis would go here
        token.verb_person = "ثالث"  # Default
        token.verb_number = "مفرد"  # Default
        token.verb_gender = "مذكر"  # Default

    def _analyze_noun_features(self, token: AdvancedArabicToken) -> None:
        """Analyze noun-specific features"""

        word = token.text

        # Determine definiteness
        if word.startswith("ال"):
            token.noun_definiteness = "معرفة"
        else:
            token.noun_definiteness = "نكرة"

        # Determine number
        if word.endswith(("ان", "ين")):
            token.noun_number = "مثنى"
        elif word.endswith(("ون", "ات")):
            token.noun_number = "جمع"
        else:
            token.noun_number = "مفرد"

        # Determine gender (simplified)
        if word.endswith("ة"):
            token.noun_gender = "مؤنث"
        else:
            token.noun_gender = "مذكر"

    def _syntactic_analysis(self, tokens: List[AdvancedArabicToken]) -> List[AdvancedArabicToken]:
        """Perform syntactic analysis"""

        # Identify sentence type
        sentence_type = self._identify_sentence_type(tokens)

        # Assign syntactic functions based on sentence type
        if sentence_type == "فعلية":
            self._analyze_verbal_sentence(tokens)
        elif sentence_type == "اسمية":
            self._analyze_nominal_sentence(tokens)

        return tokens

    def _identify_sentence_type(self, tokens: List[AdvancedArabicToken]) -> str:
        """Identify Arabic sentence type"""

        if not tokens:
            return "غير_محدد"

        # Check if first word is a verb
        if tokens[0].word_type == WordType.VERB:
            return "فعلية"

        # Check if first word is a noun
        if tokens[0].word_type == WordType.NOUN:
            return "اسمية"

        return "غير_محدد"

    def _analyze_verbal_sentence(self, tokens: List[AdvancedArabicToken]) -> None:
        """Analyze verbal sentence structure"""

        verb_found = False
        subject_found = False

        for i, token in enumerate(tokens):
            if token.word_type == WordType.VERB and not verb_found:
                token.syntactic_function = SyntacticFunction.UNKNOWN  # Verb doesn't have syntactic function
                verb_found = True

            elif token.word_type == WordType.NOUN and verb_found and not subject_found:
                token.syntactic_function = SyntacticFunction.SUBJECT
                subject_found = True

            elif token.word_type == WordType.NOUN and subject_found:
                token.syntactic_function = SyntacticFunction.OBJECT

    def _analyze_nominal_sentence(self, tokens: List[AdvancedArabicToken]) -> None:
        """Analyze nominal sentence structure"""

        topic_found = False

        for i, token in enumerate(tokens):
            if token.word_type == WordType.NOUN and not topic_found:
                token.syntactic_function = SyntacticFunction.TOPIC
                topic_found = True

            elif token.word_type == WordType.NOUN and topic_found:
                token.syntactic_function = SyntacticFunction.PREDICATE
                break

    def _irab_analysis(self, tokens: List[AdvancedArabicToken]) -> List[AdvancedArabicToken]:
        """Perform I'rab (case marking) analysis"""

        for token in tokens:
            # Assign case based on syntactic function
            if token.syntactic_function == SyntacticFunction.SUBJECT:
                token.case_marking = CaseMarking.NOMINATIVE
                token.case_sign = "ضمة"

            elif token.syntactic_function == SyntacticFunction.OBJECT:
                token.case_marking = CaseMarking.ACCUSATIVE
                token.case_sign = "فتحة"

            elif token.syntactic_function == SyntacticFunction.TOPIC:
                token.case_marking = CaseMarking.NOMINATIVE
                token.case_sign = "ضمة"

            elif token.syntactic_function == SyntacticFunction.PREDICATE:
                token.case_marking = CaseMarking.NOMINATIVE
                token.case_sign = "ضمة"

            # Check for preposition objects
            if self._is_after_preposition(token, tokens):
                token.case_marking = CaseMarking.GENITIVE
                token.case_sign = "كسرة"
                token.syntactic_function = SyntacticFunction.PREPOSITION_OBJECT

        return tokens

    def _is_after_preposition(self, token: AdvancedArabicToken, tokens: List[AdvancedArabicToken]) -> bool:
        """Check if token comes after a preposition"""

        token_index = tokens.index(token)
        if token_index > 0:
            previous_token = tokens[token_index - 1]
            prepositions = self.grammatical_rules["word_identification"]["prepositions"]
            return previous_token.text in prepositions

        return False

    def _analyze_sentence_structure(self, tokens: List[AdvancedArabicToken]) -> SentenceStructure:
        """Analyze overall sentence structure"""

        # Identify sentence type
        sentence_type = self._identify_sentence_type(tokens)

        # Find main components
        main_components = {}
        for i, token in enumerate(tokens):
            if token.syntactic_function != SyntacticFunction.UNKNOWN:
                main_components[token.syntactic_function.value] = i

        return SentenceStructure(
            sentence_type=sentence_type,
            main_components=main_components
        )

    def _calculate_analysis_confidence(self, tokens: List[AdvancedArabicToken]) -> float:
        """Calculate confidence score for the analysis"""

        if not tokens:
            return 0.0

        # Count tokens with identified features
        identified_count = sum(1 for token in tokens if token.word_type != WordType.UNKNOWN)

        return identified_count / len(tokens)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""

        # Arabic sentence delimiters
        sentences = re.split(r'[.!?؟]', text)
        return [s.strip() for s in sentences if s.strip()]


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create advanced analyzer
    analyzer = AdvancedArabicSyntaxAnalyzer()

    # Test sentences
    test_sentences = [
        "العلم نور",
        "يقرأ الطالب الكتاب",
        "كتب المعلم الدرس على السبورة",
        "إن العلم نافع للإنسان"
    ]

    print("Advanced Arabic Syntax Analysis Results:")
    print("=" * 60)

    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        analyses = analyzer.analyze_advanced(sentence)

        for analysis in analyses:
            print(f"Sentence Type: {analysis['structure'].sentence_type}")
            print(f"Confidence: {analysis['analysis_confidence']:.2f}")
            print("Tokens:")

            for token in analysis['tokens']:
                print(f"  {token.text}:")
                print(f"    Type: {token.word_type.value}")
                print(f"    Function: {token.syntactic_function.value}")
                print(f"    Case: {token.case_marking.value}")
                if token.case_sign:
                    print(f"    Case Sign: {token.case_sign}")
                if token.root:
                    print(f"    Root: {token.root}")
                print()
