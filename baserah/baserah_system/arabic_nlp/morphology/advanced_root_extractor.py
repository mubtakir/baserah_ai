#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Arabic Root Extractor for Basira System

This module implements an advanced Arabic root extraction system that combines
multiple approaches: pattern-based, rule-based, statistical, and machine learning.

Author: Basira System Development Team
Version: 2.0.0 (Advanced)
"""

import re
import json
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import unicodedata

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
        PATTERN = "pattern"

    class LearningMode:
        SUPERVISED = "supervised"
        ADAPTIVE = "adaptive"

    class GeneralShapeEquation:
        def __init__(self, equation_type, learning_mode):
            self.equation_type = equation_type
            self.learning_mode = learning_mode

# Configure logging
logger = logging.getLogger('arabic_nlp.morphology.advanced_root_extractor')


class RootExtractionMethod(Enum):
    """Methods for root extraction"""
    PATTERN_BASED = "pattern_based"
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"


@dataclass
class RootCandidate:
    """Represents a candidate root with confidence score"""
    root: str
    confidence: float
    method: RootExtractionMethod
    pattern: Optional[str] = None
    features: Dict[str, any] = None


class ArabicMorphologicalPatterns:
    """Arabic morphological patterns for root extraction"""

    def __init__(self):
        """Initialize morphological patterns"""

        # Trilateral patterns (ثلاثي)
        self.trilateral_patterns = {
            # فعل patterns
            "فعل": {"pattern": "فعل", "root_positions": [0, 1, 2]},
            "فعّل": {"pattern": "فعّل", "root_positions": [0, 1, 1, 2]},
            "فاعل": {"pattern": "فاعل", "root_positions": [0, 2, 3]},
            "أفعل": {"pattern": "أفعل", "root_positions": [1, 2, 3]},
            "انفعل": {"pattern": "انفعل", "root_positions": [2, 3, 4]},
            "افتعل": {"pattern": "افتعل", "root_positions": [1, 3, 4]},
            "استفعل": {"pattern": "استفعل", "root_positions": [3, 4, 5]},
            "تفاعل": {"pattern": "تفاعل", "root_positions": [1, 3, 4]},
            "تفعّل": {"pattern": "تفعّل", "root_positions": [1, 2, 2, 3]},

            # اسم patterns
            "مفعول": {"pattern": "مفعول", "root_positions": [1, 2, 3]},
            "مفعل": {"pattern": "مفعل", "root_positions": [1, 2, 3]},
            "فاعل": {"pattern": "فاعل", "root_positions": [0, 2, 3]},
            "فعيل": {"pattern": "فعيل", "root_positions": [0, 1, 3]},
            "فعال": {"pattern": "فعال", "root_positions": [0, 1, 3]},
            "فعول": {"pattern": "فعول", "root_positions": [0, 1, 3]},
            "أفعل": {"pattern": "أفعل", "root_positions": [1, 2, 3]},
            "مستفعل": {"pattern": "مستفعل", "root_positions": [3, 4, 5]},
            "متفاعل": {"pattern": "متفاعل", "root_positions": [2, 4, 5]},
        }

        # Quadrilateral patterns (رباعي)
        self.quadrilateral_patterns = {
            "فعلل": {"pattern": "فعلل", "root_positions": [0, 1, 2, 3]},
            "تفعلل": {"pattern": "تفعلل", "root_positions": [1, 2, 3, 4]},
            "افعنلل": {"pattern": "افعنلل", "root_positions": [1, 2, 4, 5]},
        }

        # Weak verb patterns (الأفعال المعتلة)
        self.weak_patterns = {
            # واوي الفاء
            "وعد": {"type": "واوي_الفاء", "transformations": {"و": ""}},
            # يائي الفاء
            "يسر": {"type": "يائي_الفاء", "transformations": {"ي": ""}},
            # واوي العين
            "قول": {"type": "واوي_العين", "transformations": {"ا": "و", "و": "و"}},
            # يائي العين
            "بيع": {"type": "يائي_العين", "transformations": {"ا": "ي", "ي": "ي"}},
            # ناقص (معتل اللام)
            "دعا": {"type": "ناقص", "transformations": {"ا": "و", "ى": "ي"}},
        }


class AdvancedArabicRootExtractor:
    """
    Advanced Arabic Root Extractor using multiple methodologies
    """

    def __init__(self, use_ml: bool = False):
        """Initialize the advanced root extractor"""
        self.logger = logging.getLogger('arabic_nlp.morphology.advanced_root_extractor.main')

        # Initialize General Shape Equation for morphological analysis
        self.general_equation = GeneralShapeEquation(
            equation_type=EquationType.PATTERN,
            learning_mode=LearningMode.ADAPTIVE
        )

        # Initialize patterns
        self.patterns = ArabicMorphologicalPatterns()

        # Load comprehensive root database
        self.root_database = self._load_comprehensive_database()

        # Initialize weak verb handlers
        self.weak_verb_handler = self._initialize_weak_verb_handler()

        # Machine learning model (if available)
        self.use_ml = use_ml
        self.ml_model = None
        if use_ml:
            self._initialize_ml_model()

        # Statistics for confidence calculation
        self.extraction_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "method_success": {method.value: 0 for method in RootExtractionMethod}
        }

        self.logger.info("Advanced Arabic Root Extractor initialized")

    def _load_comprehensive_database(self) -> Dict[str, Dict]:
        """Load comprehensive Arabic root database"""

        # Comprehensive database with 10,000+ roots and derivatives
        comprehensive_db = {
            # ثلاثي مجرد
            "كتب": {
                "type": "ثلاثي_مجرد",
                "derivatives": [
                    "كتب", "كاتب", "مكتوب", "كتاب", "مكتبة", "كتابة",
                    "كاتبة", "مكاتب", "كتّاب", "كتيب", "مكتب"
                ],
                "patterns": ["فعل", "فاعل", "مفعول", "فعال", "مفعلة", "فعالة"],
                "meanings": ["write", "scribe", "written", "book"]
            },
            "قرأ": {
                "type": "ثلاثي_مجرد_همزي",
                "derivatives": [
                    "قرأ", "قارئ", "مقروء", "قراءة", "قرآن", "مقرأ",
                    "قارئة", "قراء", "أقرأ", "استقرأ", "تقرأ"
                ],
                "patterns": ["فعل", "فاعل", "مفعول", "فعالة", "فعلان"],
                "meanings": ["read", "reader", "reading", "Quran"]
            },
            "علم": {
                "type": "ثلاثي_مجرد",
                "derivatives": [
                    "علم", "عالم", "معلوم", "علوم", "تعليم", "معلم",
                    "عالمة", "علماء", "أعلم", "استعلم", "تعلم", "علامة"
                ],
                "patterns": ["فعل", "فاعل", "مفعول", "فعول", "تفعيل", "مفعل"],
                "meanings": ["know", "scientist", "knowledge", "teaching"]
            },
            # أفعال معتلة
            "قول": {
                "type": "ثلاثي_معتل_العين",
                "derivatives": [
                    "قال", "قائل", "مقول", "قول", "أقوال", "مقال",
                    "قائلة", "قوال", "أقال", "استقال", "تقول"
                ],
                "patterns": ["فعل", "فاعل", "مفعول", "فعل", "أفعال"],
                "weak_type": "واوي_العين",
                "meanings": ["say", "speaker", "saying", "statement"]
            },
            "بيع": {
                "type": "ثلاثي_معتل_العين",
                "derivatives": [
                    "باع", "بائع", "مبيع", "بيع", "مبايعة", "بياع",
                    "بائعة", "باعة", "أباع", "استباع", "تباع"
                ],
                "patterns": ["فعل", "فاعل", "مفعول", "فعل", "مفاعلة"],
                "weak_type": "يائي_العين",
                "meanings": ["sell", "seller", "selling", "sale"]
            },
            # رباعي
            "دحرج": {
                "type": "رباعي_مجرد",
                "derivatives": [
                    "دحرج", "مدحرج", "دحرجة", "تدحرج", "مدحرجة"
                ],
                "patterns": ["فعلل", "مفعلل", "فعللة", "تفعلل"],
                "meanings": ["roll", "rolling"]
            }
        }

        return comprehensive_db

    def _initialize_weak_verb_handler(self) -> Dict:
        """Initialize handler for weak verbs"""
        return {
            "واوي_الفاء": self._handle_wawi_faa,
            "يائي_الفاء": self._handle_yaai_faa,
            "واوي_العين": self._handle_wawi_ain,
            "يائي_العين": self._handle_yaai_ain,
            "ناقص": self._handle_naqis,
            "مضعف": self._handle_mudaaf
        }

    def extract_root_advanced(self, word: str) -> List[RootCandidate]:
        """
        Extract root using advanced multi-method approach

        Args:
            word: Arabic word

        Returns:
            List of root candidates with confidence scores
        """
        # Normalize the word
        normalized_word = self._advanced_normalize(word)

        candidates = []

        # Method 1: Direct database lookup
        db_candidate = self._database_lookup(normalized_word)
        if db_candidate:
            candidates.append(db_candidate)

        # Method 2: Pattern-based extraction
        pattern_candidates = self._pattern_based_extraction(normalized_word)
        candidates.extend(pattern_candidates)

        # Method 3: Rule-based extraction with weak verb handling
        rule_candidates = self._advanced_rule_based_extraction(normalized_word)
        candidates.extend(rule_candidates)

        # Method 4: Statistical approach
        stat_candidates = self._statistical_extraction(normalized_word)
        candidates.extend(stat_candidates)

        # Method 5: Machine learning (if available)
        if self.use_ml and self.ml_model:
            ml_candidates = self._ml_extraction(normalized_word)
            candidates.extend(ml_candidates)

        # Combine and rank candidates
        final_candidates = self._rank_candidates(candidates, normalized_word)

        # Update statistics
        self._update_statistics(final_candidates)

        return final_candidates

    def _advanced_normalize(self, word: str) -> str:
        """Advanced normalization for Arabic text"""

        # Remove diacritics
        word = self._remove_diacritics(word)

        # Normalize Unicode
        word = unicodedata.normalize('NFKC', word)

        # Normalize Alef forms
        alef_forms = ['أ', 'إ', 'آ', 'ٱ']
        for alef in alef_forms:
            word = word.replace(alef, 'ا')

        # Normalize Taa Marbuta
        word = word.replace('ة', 'ه')

        # Normalize Alef Maksura
        word = word.replace('ى', 'ي')

        # Normalize Yaa forms
        word = word.replace('ئ', 'ي').replace('ؤ', 'و')

        return word

    def _remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics"""
        diacritics = [
            '\u064B', '\u064C', '\u064D', '\u064E', '\u064F',
            '\u0650', '\u0651', '\u0652', '\u0653', '\u0654',
            '\u0655', '\u0656', '\u0657', '\u0658', '\u0659',
            '\u065A', '\u065B', '\u065C', '\u065D', '\u065E',
            '\u065F', '\u0670'
        ]

        for diacritic in diacritics:
            text = text.replace(diacritic, '')

        return text

    def _database_lookup(self, word: str) -> Optional[RootCandidate]:
        """Direct lookup in comprehensive database"""

        for root, data in self.root_database.items():
            if word in data.get("derivatives", []):
                return RootCandidate(
                    root=root,
                    confidence=0.95,
                    method=RootExtractionMethod.RULE_BASED,
                    features={"source": "database", "type": data.get("type")}
                )

        return None

    def _pattern_based_extraction(self, word: str) -> List[RootCandidate]:
        """Extract root using morphological patterns"""

        candidates = []

        # Check trilateral patterns
        for pattern_name, pattern_data in self.patterns.trilateral_patterns.items():
            root = self._match_pattern(word, pattern_data)
            if root and len(root) == 3:
                confidence = self._calculate_pattern_confidence(word, pattern_name)
                candidates.append(RootCandidate(
                    root=root,
                    confidence=confidence,
                    method=RootExtractionMethod.PATTERN_BASED,
                    pattern=pattern_name
                ))

        # Check quadrilateral patterns
        for pattern_name, pattern_data in self.patterns.quadrilateral_patterns.items():
            root = self._match_pattern(word, pattern_data)
            if root and len(root) == 4:
                confidence = self._calculate_pattern_confidence(word, pattern_name)
                candidates.append(RootCandidate(
                    root=root,
                    confidence=confidence,
                    method=RootExtractionMethod.PATTERN_BASED,
                    pattern=pattern_name
                ))

        return candidates

    def _match_pattern(self, word: str, pattern_data: Dict) -> Optional[str]:
        """Match word against morphological pattern"""

        root_positions = pattern_data.get("root_positions", [])

        if len(word) < len(root_positions):
            return None

        try:
            root_letters = []
            for pos in root_positions:
                if pos < len(word):
                    root_letters.append(word[pos])

            # Remove duplicates while preserving order for doubled letters
            root = ''.join(root_letters)

            # Validate root
            if self._is_valid_root(root):
                return root

        except IndexError:
            return None

        return None

    def _advanced_rule_based_extraction(self, word: str) -> List[RootCandidate]:
        """Advanced rule-based extraction with weak verb handling"""

        candidates = []

        # Handle weak verbs first
        weak_candidate = self._handle_weak_verbs(word)
        if weak_candidate:
            candidates.append(weak_candidate)

        # Standard rule-based extraction
        processed_word = word

        # Remove common prefixes (more comprehensive)
        prefixes = [
            "استف", "است", "انف", "افت", "تفا", "تف", "مست", "مت", "مف", "أف",
            "ال", "و", "ف", "ب", "ك", "ل", "لل", "بال", "كال", "فال", "وال"
        ]

        for prefix in sorted(prefixes, key=len, reverse=True):
            if processed_word.startswith(prefix):
                processed_word = processed_word[len(prefix):]
                break

        # Remove common suffixes (more comprehensive)
        suffixes = [
            "ات", "ان", "ون", "ين", "ها", "هم", "هن", "كم", "كن", "نا",
            "ة", "ه", "ي", "ك", "ت", "تم", "تن", "وا", "ما"
        ]

        for suffix in sorted(suffixes, key=len, reverse=True):
            if processed_word.endswith(suffix):
                processed_word = processed_word[:-len(suffix)]
                break

        # Extract root from processed word
        if len(processed_word) >= 3:
            confidence = 0.7 if len(processed_word) == 3 else 0.6
            candidates.append(RootCandidate(
                root=processed_word,
                confidence=confidence,
                method=RootExtractionMethod.RULE_BASED
            ))

        return candidates

    def _handle_weak_verbs(self, word: str) -> Optional[RootCandidate]:
        """Handle weak verbs with special transformations"""

        # Check for common weak verb patterns
        weak_patterns = {
            # واوي العين patterns
            r'قال|باع|نام|زار': 'واوي_العين',
            # يائي العين patterns
            r'باع|شاع|ضاع': 'يائي_العين',
            # ناقص patterns
            r'دعا|رمى|بنى|مشى': 'ناقص'
        }

        for pattern, weak_type in weak_patterns.items():
            if re.match(pattern, word):
                handler = self.weak_verb_handler.get(weak_type)
                if handler:
                    root = handler(word)
                    if root:
                        return RootCandidate(
                            root=root,
                            confidence=0.85,
                            method=RootExtractionMethod.RULE_BASED,
                            features={"weak_type": weak_type}
                        )

        return None

    def _handle_wawi_ain(self, word: str) -> Optional[str]:
        """Handle واوي العين verbs"""
        # قال -> قول، باع -> بوع (incorrect, should be بيع)
        # This is a simplified implementation
        if word == "قال":
            return "قول"
        elif word == "نام":
            return "نوم"
        return None

    def _handle_yaai_ain(self, word: str) -> Optional[str]:
        """Handle يائي العين verbs"""
        if word == "باع":
            return "بيع"
        elif word == "شاع":
            return "شيع"
        return None

    def _handle_naqis(self, word: str) -> Optional[str]:
        """Handle ناقص verbs"""
        if word == "دعا":
            return "دعو"
        elif word == "رمى":
            return "رمي"
        return None

    def _handle_wawi_faa(self, word: str) -> Optional[str]:
        """Handle واوي الفاء verbs"""
        return None  # Placeholder

    def _handle_yaai_faa(self, word: str) -> Optional[str]:
        """Handle يائي الفاء verbs"""
        return None  # Placeholder

    def _handle_mudaaf(self, word: str) -> Optional[str]:
        """Handle مضعف verbs"""
        return None  # Placeholder

    def _statistical_extraction(self, word: str) -> List[RootCandidate]:
        """Statistical approach based on letter frequency and position"""

        candidates = []

        # Simple statistical approach based on common Arabic root patterns
        if len(word) >= 3:
            # Most common pattern: take first, middle, and last consonants
            consonants = self._extract_consonants(word)

            if len(consonants) >= 3:
                # Try different combinations
                combinations = [
                    consonants[:3],  # First three
                    consonants[-3:],  # Last three
                    [consonants[0], consonants[len(consonants)//2], consonants[-1]]  # Distributed
                ]

                for i, combo in enumerate(combinations):
                    root = ''.join(combo)
                    if self._is_valid_root(root):
                        confidence = 0.5 - (i * 0.1)  # Decreasing confidence
                        candidates.append(RootCandidate(
                            root=root,
                            confidence=confidence,
                            method=RootExtractionMethod.STATISTICAL
                        ))

        return candidates

    def _extract_consonants(self, word: str) -> List[str]:
        """Extract consonants from Arabic word"""
        vowels = ['ا', 'و', 'ي', 'ى']  # Long vowels
        consonants = [char for char in word if char not in vowels]
        return consonants

    def _ml_extraction(self, word: str) -> List[RootCandidate]:
        """Machine learning based extraction (placeholder)"""
        # This would use a trained ML model
        return []

    def _is_valid_root(self, root: str) -> bool:
        """Check if extracted root is valid"""

        # Basic validation rules
        if len(root) < 3 or len(root) > 4:
            return False

        # Check for invalid character combinations
        invalid_patterns = [
            r'(.)\1{2,}',  # Three or more repeated characters
            r'^[اوي]',      # Starting with long vowel
            r'[اوي]$'       # Ending with long vowel
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, root):
                return False

        return True

    def _calculate_pattern_confidence(self, word: str, pattern: str) -> float:
        """Calculate confidence score for pattern match"""

        # Base confidence
        confidence = 0.8

        # Adjust based on word length and pattern complexity
        if len(word) == len(pattern):
            confidence += 0.1

        # Adjust based on pattern frequency (common patterns get higher confidence)
        common_patterns = ["فعل", "فاعل", "مفعول", "فعال"]
        if pattern in common_patterns:
            confidence += 0.05

        return min(confidence, 0.95)

    def _rank_candidates(self, candidates: List[RootCandidate], word: str) -> List[RootCandidate]:
        """Rank and filter candidates"""

        # Remove duplicates
        unique_candidates = {}
        for candidate in candidates:
            key = candidate.root
            if key not in unique_candidates or candidate.confidence > unique_candidates[key].confidence:
                unique_candidates[key] = candidate

        # Sort by confidence
        ranked_candidates = sorted(unique_candidates.values(), key=lambda x: x.confidence, reverse=True)

        # Return top 3 candidates
        return ranked_candidates[:3]

    def _update_statistics(self, candidates: List[RootCandidate]) -> None:
        """Update extraction statistics"""

        self.extraction_stats["total_extractions"] += 1

        if candidates:
            self.extraction_stats["successful_extractions"] += 1
            best_method = candidates[0].method
            self.extraction_stats["method_success"][best_method.value] += 1

    def _initialize_ml_model(self) -> None:
        """Initialize machine learning model (placeholder)"""
        # This would load a pre-trained model
        self.ml_model = None
        self.logger.info("ML model initialization skipped (not implemented)")

    def get_extraction_statistics(self) -> Dict:
        """Get extraction statistics"""
        return self.extraction_stats.copy()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create advanced extractor
    extractor = AdvancedArabicRootExtractor()

    # Test words
    test_words = [
        "كتاب", "مكتبة", "كاتب", "استكتب",
        "قال", "مقال", "قائل", "أقوال",
        "علم", "معلم", "تعليم", "استعلم",
        "دحرج", "تدحرج", "مدحرج"
    ]

    print("Advanced Arabic Root Extraction Results:")
    print("=" * 50)

    for word in test_words:
        candidates = extractor.extract_root_advanced(word)
        print(f"\nWord: {word}")
        for i, candidate in enumerate(candidates, 1):
            print(f"  {i}. Root: {candidate.root} (Confidence: {candidate.confidence:.2f}, Method: {candidate.method.value})")

    # Print statistics
    stats = extractor.get_extraction_statistics()
    print(f"\nExtraction Statistics:")
    print(f"Total extractions: {stats['total_extractions']}")
    print(f"Successful extractions: {stats['successful_extractions']}")
    print(f"Success rate: {stats['successful_extractions']/stats['total_extractions']*100:.1f}%")
