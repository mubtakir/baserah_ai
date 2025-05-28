#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Root Extractor for Basira System

This module implements the Arabic Root Extractor, which extracts the roots of Arabic words.
It uses a combination of rule-based and statistical approaches to extract the roots.

Author: Basira System Development Team
Version: 1.0.0
"""

import re
import os
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Set
from pathlib import Path

# Configure logging
logger = logging.getLogger('arabic_nlp.morphology.root_extractor')


class ArabicRootExtractor:
    """
    Arabic Root Extractor class for extracting roots from Arabic words.

    This class implements various algorithms for extracting the roots of Arabic words,
    including rule-based, pattern-based, and statistical approaches.
    """

    def __init__(self,
                 rules_file: Optional[str] = None,
                 patterns_file: Optional[str] = None,
                 exceptions_file: Optional[str] = None,
                 use_statistical: bool = True):
        """
        Initialize the Arabic Root Extractor.

        Args:
            rules_file: Path to the rules file (optional)
            patterns_file: Path to the patterns file (optional)
            exceptions_file: Path to the exceptions file (optional)
            use_statistical: Whether to use statistical approaches
        """
        self.logger = logging.getLogger('arabic_nlp.morphology.root_extractor.main')

        # Load rules
        self.rules = self._load_rules(rules_file)

        # Load patterns
        self.patterns = self._load_patterns(patterns_file)

        # Load exceptions
        self.exceptions = self._load_exceptions(exceptions_file)

        # Set whether to use statistical approaches
        self.use_statistical = use_statistical

        # Initialize statistical model if needed
        self.statistical_model = None
        if self.use_statistical:
            self._initialize_statistical_model()

        self.logger.info(f"Arabic Root Extractor initialized with {len(self.rules)} rules, {len(self.patterns)} patterns, and {len(self.exceptions)} exceptions")

    def _load_rules(self, rules_file: Optional[str]) -> Dict[str, List[str]]:
        """
        Load rules from file or use default rules.

        Args:
            rules_file: Path to the rules file

        Returns:
            Dictionary of rules
        """
        default_rules = {
            "prefix_removal": [
                "ال", "و", "ف", "ب", "ك", "ل", "لل"
            ],
            "suffix_removal": [
                "ة", "ات", "ان", "ون", "ين", "ه", "ها", "هم", "هن", "كم", "كن", "نا"
            ],
            "infix_patterns": [
                "ا", "و", "ي"
            ]
        }

        if rules_file and os.path.exists(rules_file):
            try:
                with open(rules_file, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                self.logger.info(f"Rules loaded from {rules_file}")
                return rules
            except Exception as e:
                self.logger.error(f"Error loading rules from {rules_file}: {e}")
                self.logger.info("Using default rules")
        else:
            if rules_file:
                self.logger.warning(f"Rules file {rules_file} not found, using default rules")
            else:
                self.logger.info("No rules file provided, using default rules")

        return default_rules

    def _load_patterns(self, patterns_file: Optional[str]) -> Dict[str, str]:
        """
        Load patterns from file or use default patterns.

        Args:
            patterns_file: Path to the patterns file

        Returns:
            Dictionary of patterns
        """
        default_patterns = {
            "فاعل": "فعل",
            "مفعول": "فعل",
            "فعّال": "فعل",
            "مفعّل": "فعل",
            "افتعل": "فعل",
            "استفعل": "فعل",
            "انفعل": "فعل",
            "تفاعل": "فعل",
            "تفعّل": "فعل"
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

    def _load_exceptions(self, exceptions_file: Optional[str]) -> Dict[str, str]:
        """
        Load exceptions from file or use default exceptions.

        Args:
            exceptions_file: Path to the exceptions file

        Returns:
            Dictionary of exceptions
        """
        default_exceptions = {
            "قال": "قول",
            "باع": "بيع",
            "كان": "كون",
            "ليس": "ليس",
            "أخذ": "أخذ",
            "رأى": "رأي",
            "جاء": "جيء"
        }

        # Try to load from the provided file
        if exceptions_file and os.path.exists(exceptions_file):
            try:
                with open(exceptions_file, 'r', encoding='utf-8') as f:
                    exceptions = json.load(f)
                self.logger.info(f"Exceptions loaded from {exceptions_file}")
                return exceptions
            except Exception as e:
                self.logger.error(f"Error loading exceptions from {exceptions_file}: {e}")
                self.logger.info("Using default exceptions")
        else:
            if exceptions_file:
                self.logger.warning(f"Exceptions file {exceptions_file} not found, trying to load from data directory")
            else:
                self.logger.info("No exceptions file provided, trying to load from data directory")

        # Try to load from the data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        roots_file = os.path.join(data_dir, 'arabic_roots.json')

        if os.path.exists(roots_file):
            try:
                with open(roots_file, 'r', encoding='utf-8') as f:
                    roots_data = json.load(f)

                # Create exceptions dictionary from roots data
                exceptions = {}
                for root, data in roots_data.items():
                    for derivative in data.get('derivatives', []):
                        exceptions[derivative] = root

                self.logger.info(f"Exceptions loaded from {roots_file}")
                return {**default_exceptions, **exceptions}  # Merge with default exceptions
            except Exception as e:
                self.logger.error(f"Error loading exceptions from {roots_file}: {e}")
                self.logger.info("Using default exceptions")

        return default_exceptions

    def _initialize_statistical_model(self) -> None:
        """Initialize the statistical model for root extraction."""
        # Placeholder for statistical model initialization
        self.statistical_model = {
            "trained": False,
            "accuracy": 0.0
        }

        self.logger.info("Statistical model initialized")

    def extract_roots(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract roots from Arabic text.

        Args:
            text: Arabic text

        Returns:
            List of tuples (word, root)
        """
        # Tokenize text
        words = self._tokenize(text)

        # Extract roots
        roots = []
        for word in words:
            root = self.extract_root(word)
            roots.append((word, root))

        return roots

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize Arabic text into words.

        Args:
            text: Arabic text

        Returns:
            List of words
        """
        # Simple tokenization by whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()

        return words

    def extract_root(self, word: str) -> str:
        """
        Extract the root of an Arabic word.

        Args:
            word: Arabic word

        Returns:
            Root of the word
        """
        # Normalize word (remove diacritics and normalize characters)
        normalized_word = self._normalize_word(word)

        # Check if word is in exceptions
        if normalized_word in self.exceptions:
            return self.exceptions[normalized_word]

        # Check if original word is in exceptions
        if word in self.exceptions:
            return self.exceptions[word]

        # Apply rule-based approach
        root = self._apply_rules(normalized_word)

        # Apply pattern-based approach if rule-based failed
        if not root or len(root) < 3:
            root = self._apply_patterns(normalized_word)

        # Apply statistical approach if enabled and other approaches failed
        if self.use_statistical and (not root or len(root) < 3):
            root = self._apply_statistical(normalized_word)

        # If all approaches failed, return the word itself
        if not root or len(root) < 3:
            return normalized_word

        return root

    def _normalize_word(self, word: str) -> str:
        """
        Normalize an Arabic word by removing diacritics and normalizing characters.

        Args:
            word: Arabic word

        Returns:
            Normalized word
        """
        # Remove diacritics (tashkeel)
        diacritics = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ']
        for diacritic in diacritics:
            word = word.replace(diacritic, '')

        # Normalize alef forms
        alef_forms = ['أ', 'إ', 'آ']
        for alef in alef_forms:
            word = word.replace(alef, 'ا')

        # Normalize taa marbuta to haa
        word = word.replace('ة', 'ه')

        # Normalize alef maksura to yaa
        word = word.replace('ى', 'ي')

        return word

    def _apply_rules(self, word: str) -> str:
        """
        Apply rule-based approach to extract the root.

        Args:
            word: Arabic word

        Returns:
            Root of the word
        """
        # Remove prefixes
        for prefix in sorted(self.rules["prefix_removal"], key=len, reverse=True):
            if word.startswith(prefix):
                word = word[len(prefix):]
                break

        # Remove suffixes
        for suffix in sorted(self.rules["suffix_removal"], key=len, reverse=True):
            if word.endswith(suffix):
                word = word[:-len(suffix)]
                break

        # Remove infixes
        for infix in self.rules["infix_patterns"]:
            if infix in word and len(word) > 3:
                word = word.replace(infix, "", 1)

        # Ensure the root is at least 3 characters
        if len(word) < 3:
            return ""

        return word

    def _apply_patterns(self, word: str) -> str:
        """
        Apply pattern-based approach to extract the root.

        Args:
            word: Arabic word

        Returns:
            Root of the word
        """
        # Placeholder for pattern-based approach
        # In a real implementation, this would match the word against known patterns
        # and extract the root based on the pattern

        # For now, just return the word if it's at least 3 characters
        if len(word) >= 3:
            return word

        return ""

    def _apply_statistical(self, word: str) -> str:
        """
        Apply statistical approach to extract the root.

        Args:
            word: Arabic word

        Returns:
            Root of the word
        """
        # Placeholder for statistical approach
        # In a real implementation, this would use a trained model to predict the root

        # For now, just return the word if it's at least 3 characters
        if len(word) >= 3:
            return word

        return ""


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create Arabic Root Extractor
    extractor = ArabicRootExtractor()

    # Extract roots from text
    text = "العلم نور يضيء طريق الحياة"
    roots = extractor.extract_roots(text)

    # Print results
    print("Text:", text)
    print("Roots:")
    for word, root in roots:
        print(f"  {word}: {root}")
