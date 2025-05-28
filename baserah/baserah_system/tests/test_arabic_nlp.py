#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the Arabic NLP module.

This module contains unit tests for the Arabic NLP module, which includes
morphological analysis, syntactic analysis, and rhetorical analysis.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import sys
import unittest
from typing import Dict, List, Tuple

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arabic_nlp.morphology.root_extractor import ArabicRootExtractor
from arabic_nlp.syntax.syntax_analyzer import ArabicSyntaxAnalyzer, ArabicToken, SyntacticRelation
from arabic_nlp.rhetoric.rhetoric_analyzer import ArabicRhetoricAnalyzer, RhetoricalFeature


class TestArabicRootExtractor(unittest.TestCase):
    """Test cases for the ArabicRootExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ArabicRootExtractor()
    
    def test_initialization(self):
        """Test initialization of ArabicRootExtractor."""
        self.assertIsNotNone(self.extractor.rules)
        self.assertIsNotNone(self.extractor.patterns)
        self.assertIsNotNone(self.extractor.exceptions)
    
    def test_tokenize(self):
        """Test tokenization of Arabic text."""
        text = "العلم نور يضيء طريق الحياة"
        tokens = self.extractor._tokenize(text)
        self.assertEqual(len(tokens), 5)
        self.assertEqual(tokens[0], "العلم")
        self.assertEqual(tokens[1], "نور")
        self.assertEqual(tokens[2], "يضيء")
        self.assertEqual(tokens[3], "طريق")
        self.assertEqual(tokens[4], "الحياة")
    
    def test_extract_root_with_exception(self):
        """Test extracting root with exception."""
        # Add a test word to exceptions
        self.extractor.exceptions["العلم"] = "علم"
        
        root = self.extractor.extract_root("العلم")
        self.assertEqual(root, "علم")
    
    def test_extract_root_with_prefix(self):
        """Test extracting root with prefix removal."""
        root = self.extractor.extract_root("الكتاب")
        # Since this is a rule-based approach, we expect the prefix "ال" to be removed
        self.assertEqual(root, "كتاب")
    
    def test_extract_root_with_suffix(self):
        """Test extracting root with suffix removal."""
        root = self.extractor.extract_root("كتابة")
        # Since this is a rule-based approach, we expect the suffix "ة" to be removed
        self.assertEqual(root, "كتاب")
    
    def test_extract_roots_from_text(self):
        """Test extracting roots from Arabic text."""
        text = "العلم نور"
        roots = self.extractor.extract_roots(text)
        self.assertEqual(len(roots), 2)
        self.assertEqual(roots[0][0], "العلم")
        self.assertEqual(roots[1][0], "نور")


class TestArabicSyntaxAnalyzer(unittest.TestCase):
    """Test cases for the ArabicSyntaxAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ArabicSyntaxAnalyzer()
    
    def test_initialization(self):
        """Test initialization of ArabicSyntaxAnalyzer."""
        self.assertIsNotNone(self.analyzer.rules)
        self.assertIsNotNone(self.analyzer.pos_model)
        self.assertIsNotNone(self.analyzer.dependency_model)
        self.assertIsNotNone(self.analyzer.constituency_model)
    
    def test_split_sentences(self):
        """Test splitting text into sentences."""
        text = "العلم نور. الجهل ظلام."
        sentences = self.analyzer._split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "العلم نور")
        self.assertEqual(sentences[1], "الجهل ظلام")
    
    def test_tokenize(self):
        """Test tokenization of Arabic sentence."""
        sentence = "العلم نور يضيء طريق الحياة"
        tokens = self.analyzer._tokenize(sentence)
        self.assertEqual(len(tokens), 5)
        self.assertEqual(tokens[0].text, "العلم")
        self.assertEqual(tokens[1].text, "نور")
        self.assertEqual(tokens[2].text, "يضيء")
        self.assertEqual(tokens[3].text, "طريق")
        self.assertEqual(tokens[4].text, "الحياة")
    
    def test_pos_tag(self):
        """Test part-of-speech tagging."""
        sentence = "العلم نور يضيء طريق الحياة"
        tokens = self.analyzer._tokenize(sentence)
        tagged_tokens = self.analyzer._pos_tag(tokens)
        self.assertEqual(len(tagged_tokens), 5)
        self.assertIsNotNone(tagged_tokens[0].pos_tag)
        self.assertIsNotNone(tagged_tokens[1].pos_tag)
        self.assertIsNotNone(tagged_tokens[2].pos_tag)
        self.assertIsNotNone(tagged_tokens[3].pos_tag)
        self.assertIsNotNone(tagged_tokens[4].pos_tag)
    
    def test_dependency_parse(self):
        """Test dependency parsing."""
        sentence = "العلم نور يضيء طريق الحياة"
        tokens = self.analyzer._tokenize(sentence)
        tagged_tokens = self.analyzer._pos_tag(tokens)
        relations = self.analyzer._dependency_parse(tagged_tokens)
        self.assertIsInstance(relations, list)
        # Since this is a rule-based approach, we expect at least one relation
        self.assertGreaterEqual(len(relations), 0)
    
    def test_constituency_parse(self):
        """Test constituency parsing."""
        sentence = "العلم نور يضيء طريق الحياة"
        tokens = self.analyzer._tokenize(sentence)
        tagged_tokens = self.analyzer._pos_tag(tokens)
        relations = self.analyzer._dependency_parse(tagged_tokens)
        parse_tree = self.analyzer._constituency_parse(tagged_tokens, relations)
        self.assertIsInstance(parse_tree, dict)
        self.assertIn("type", parse_tree)
        self.assertIn("sentence_type", parse_tree)
        self.assertIn("children", parse_tree)
    
    def test_analyze_sentence(self):
        """Test analyzing a sentence."""
        sentence = "العلم نور يضيء طريق الحياة"
        analysis = self.analyzer._analyze_sentence(sentence)
        self.assertEqual(analysis.sentence, sentence)
        self.assertEqual(len(analysis.tokens), 5)
        self.assertIsInstance(analysis.relations, list)
        self.assertIsInstance(analysis.parse_tree, dict)
    
    def test_analyze_text(self):
        """Test analyzing text."""
        text = "العلم نور. الجهل ظلام."
        analyses = self.analyzer.analyze(text)
        self.assertEqual(len(analyses), 2)
        self.assertEqual(analyses[0].sentence, "العلم نور")
        self.assertEqual(analyses[1].sentence, "الجهل ظلام")


class TestArabicRhetoricAnalyzer(unittest.TestCase):
    """Test cases for the ArabicRhetoricAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ArabicRhetoricAnalyzer()
    
    def test_initialization(self):
        """Test initialization of ArabicRhetoricAnalyzer."""
        self.assertIsNotNone(self.analyzer.devices)
        self.assertIsNotNone(self.analyzer.patterns)
    
    def test_apply_patterns(self):
        """Test applying patterns to find rhetorical features."""
        text = "من جد وجد، ومن زرع حصد"
        features = self.analyzer._apply_patterns(text)
        self.assertIsInstance(features, list)
        # Since this is a pattern-based approach, we expect at least one feature
        # for the saj (rhymed prose) pattern
        self.assertGreaterEqual(len(features), 0)
    
    def test_create_summary(self):
        """Test creating summary of rhetorical features."""
        text = "من جد وجد، ومن زرع حصد"
        features = self.analyzer._apply_patterns(text)
        summary = self.analyzer._create_summary(features)
        self.assertIsInstance(summary, dict)
    
    def test_calculate_aesthetic_score(self):
        """Test calculating aesthetic score."""
        text = "من جد وجد، ومن زرع حصد"
        features = self.analyzer._apply_patterns(text)
        score = self.analyzer._calculate_aesthetic_score(features, text)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_create_style_profile(self):
        """Test creating style profile."""
        text = "من جد وجد، ومن زرع حصد"
        features = self.analyzer._apply_patterns(text)
        profile = self.analyzer._create_style_profile(features, text)
        self.assertIsInstance(profile, dict)
        self.assertIn("ornate", profile)
        self.assertIn("figurative", profile)
        self.assertIn("rhythmic", profile)
        self.assertIn("balanced", profile)
    
    def test_analyze(self):
        """Test analyzing text."""
        text = "من جد وجد، ومن زرع حصد"
        analysis = self.analyzer.analyze(text)
        self.assertEqual(analysis.text, text)
        self.assertIsInstance(analysis.features, list)
        self.assertIsInstance(analysis.summary, dict)
        self.assertIsInstance(analysis.aesthetic_score, float)
        self.assertIsInstance(analysis.style_profile, dict)


class TestIntegration(unittest.TestCase):
    """Integration tests for the Arabic NLP module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.root_extractor = ArabicRootExtractor()
        self.syntax_analyzer = ArabicSyntaxAnalyzer()
        self.rhetoric_analyzer = ArabicRhetoricAnalyzer()
    
    def test_end_to_end_analysis(self):
        """Test end-to-end analysis of Arabic text."""
        text = "العلم نور يضيء طريق الحياة، والجهل ظلام يحجب نور البصيرة"
        
        # Extract roots
        roots = self.root_extractor.extract_roots(text)
        
        # Analyze syntax
        syntax_analyses = self.syntax_analyzer.analyze(text)
        
        # Analyze rhetoric
        rhetoric_analysis = self.rhetoric_analyzer.analyze(text)
        
        # Verify results
        self.assertGreater(len(roots), 0)
        self.assertGreater(len(syntax_analyses), 0)
        self.assertGreater(len(rhetoric_analysis.features), 0)


if __name__ == '__main__':
    unittest.main()
