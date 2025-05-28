#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for the Basira System.

This module contains integration tests for the Basira System, testing the
interaction between different components.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import sys
import unittest
import json
import numpy as np
from io import BytesIO
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import Basira System components
from mathematical_core.general_shape_equation import GeneralShapeEquation, EquationType, LearningMode
from arabic_nlp.morphology.root_extractor import ArabicRootExtractor
from arabic_nlp.syntax.syntax_analyzer import ArabicSyntaxAnalyzer
from arabic_nlp.rhetoric.rhetoric_analyzer import ArabicRhetoricAnalyzer
from creative_generation.image.image_generator import ImageGenerator, GenerationParameters, GenerationMode
from code_execution.code_executor import CodeExecutor, ProgrammingLanguage, ExecutionConfig


class TestMathematicalCoreIntegration(unittest.TestCase):
    """Integration tests for the Mathematical Core module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.equation = GeneralShapeEquation(
            equation_type=EquationType.COMPOSITE,
            learning_mode=LearningMode.HYBRID
        )
    
    def test_equation_with_variables(self):
        """Test equation with variables."""
        # Add components
        self.equation.add_component("circle", "(x-cx)^2 + (y-cy)^2 - r^2")
        self.equation.add_component("cx", "0")
        self.equation.add_component("cy", "0")
        self.equation.add_component("r", "5")
        
        # Evaluate with different variables
        result1 = self.equation.evaluate({"x": 0, "y": 0})
        result2 = self.equation.evaluate({"x": 5, "y": 0})
        
        # Check that results contain expected components
        self.assertIn("circle", result1)
        self.assertIn("cx", result1)
        self.assertIn("cy", result1)
        self.assertIn("r", result1)
        
        self.assertIn("circle", result2)
        self.assertIn("cx", result2)
        self.assertIn("cy", result2)
        self.assertIn("r", result2)
    
    def test_equation_serialization(self):
        """Test equation serialization and deserialization."""
        # Add components
        self.equation.add_component("circle", "(x-cx)^2 + (y-cy)^2 - r^2")
        self.equation.add_component("cx", "0")
        self.equation.add_component("cy", "0")
        self.equation.add_component("r", "5")
        
        # Serialize to JSON
        json_str = self.equation.to_json()
        
        # Deserialize from JSON
        equation_dict = json.loads(json_str)
        
        # Check that deserialized equation contains expected components
        self.assertIn("symbolic_components", equation_dict)
        self.assertIn("circle", equation_dict["symbolic_components"])
        self.assertIn("cx", equation_dict["symbolic_components"])
        self.assertIn("cy", equation_dict["symbolic_components"])
        self.assertIn("r", equation_dict["symbolic_components"])


class TestArabicNLPIntegration(unittest.TestCase):
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
        
        # Check that results are as expected
        self.assertGreater(len(roots), 0)
        self.assertGreater(len(syntax_analyses), 0)
        self.assertGreater(len(rhetoric_analysis.features), 0)
    
    def test_root_extraction_with_normalization(self):
        """Test root extraction with normalization."""
        # Test with diacritics
        word_with_diacritics = "العِلْمُ"
        root = self.root_extractor.extract_root(word_with_diacritics)
        
        # Test without diacritics
        word_without_diacritics = "العلم"
        root2 = self.root_extractor.extract_root(word_without_diacritics)
        
        # Check that both return the same root
        self.assertEqual(root, root2)


class TestCreativeGenerationIntegration(unittest.TestCase):
    """Integration tests for the Creative Generation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.image_generator = ImageGenerator()
    
    @unittest.skipIf(not PIL_AVAILABLE, "PIL not available")
    def test_text_to_image_generation(self):
        """Test text to image generation."""
        # Generate image from text
        parameters = GenerationParameters(
            mode=GenerationMode.TEXT_TO_IMAGE,
            width=256,
            height=256,
            seed=42
        )
        result = self.image_generator.generate_image("A beautiful landscape", parameters)
        
        # Check that result is as expected
        self.assertIsNotNone(result.image)
        self.assertGreater(result.generation_time, 0)
        
        # Check image dimensions
        if isinstance(result.image, np.ndarray):
            self.assertEqual(result.image.shape[0], 256)
            self.assertEqual(result.image.shape[1], 256)
    
    @unittest.skipIf(not PIL_AVAILABLE, "PIL not available")
    def test_arabic_text_to_image_generation(self):
        """Test Arabic text to image generation."""
        # Generate image from Arabic text
        parameters = GenerationParameters(
            mode=GenerationMode.ARABIC_TEXT_TO_IMAGE,
            width=256,
            height=256,
            seed=42
        )
        result = self.image_generator.generate_image("العلم نور", parameters)
        
        # Check that result is as expected
        self.assertIsNotNone(result.image)
        self.assertGreater(result.generation_time, 0)
        
        # Check image dimensions
        if isinstance(result.image, np.ndarray):
            self.assertEqual(result.image.shape[0], 256)
            self.assertEqual(result.image.shape[1], 256)
    
    @unittest.skipIf(not PIL_AVAILABLE, "PIL not available")
    def test_calligraphy_generation(self):
        """Test calligraphy generation."""
        # Generate calligraphy
        parameters = GenerationParameters(
            mode=GenerationMode.CALLIGRAPHY,
            width=256,
            height=256,
            seed=42
        )
        result = self.image_generator.generate_image("بسم الله الرحمن الرحيم", parameters)
        
        # Check that result is as expected
        self.assertIsNotNone(result.image)
        self.assertGreater(result.generation_time, 0)
        
        # Check image dimensions
        if isinstance(result.image, np.ndarray):
            self.assertEqual(result.image.shape[0], 256)
            self.assertEqual(result.image.shape[1], 256)
    
    @unittest.skipIf(not PIL_AVAILABLE, "PIL not available")
    def test_equation_to_image_generation(self):
        """Test equation to image generation."""
        # Generate image from equation
        parameters = GenerationParameters(
            mode=GenerationMode.EQUATION_TO_IMAGE,
            width=256,
            height=256,
            seed=42
        )
        result = self.image_generator.generate_image("(x-cx)^2 + (y-cy)^2 - r^2", parameters)
        
        # Check that result is as expected
        self.assertIsNotNone(result.image)
        self.assertGreater(result.generation_time, 0)
        
        # Check image dimensions
        if isinstance(result.image, np.ndarray):
            self.assertEqual(result.image.shape[0], 256)
            self.assertEqual(result.image.shape[1], 256)
    
    @unittest.skipIf(not PIL_AVAILABLE, "PIL not available")
    def test_image_saving(self):
        """Test image saving."""
        # Generate image from text
        parameters = GenerationParameters(
            mode=GenerationMode.TEXT_TO_IMAGE,
            width=256,
            height=256,
            seed=42
        )
        result = self.image_generator.generate_image("A beautiful landscape", parameters)
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            self.image_generator.save_image(result, tmp_path)
            
            # Check that file exists and is not empty
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
            
            # Check that file is a valid image
            if PIL_AVAILABLE:
                image = Image.open(tmp_path)
                self.assertEqual(image.size, (256, 256))
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestCodeExecutionIntegration(unittest.TestCase):
    """Integration tests for the Code Execution module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.code_executor = CodeExecutor()
    
    def test_python_code_execution(self):
        """Test Python code execution."""
        # Execute Python code
        code = """
print("Hello, world!")
for i in range(3):
    print(f"Number: {i}")
"""
        result = self.code_executor.execute(code, ProgrammingLanguage.PYTHON)
        
        # Check that result is as expected
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hello, world!", result.stdout)
        self.assertIn("Number: 0", result.stdout)
        self.assertIn("Number: 1", result.stdout)
        self.assertIn("Number: 2", result.stdout)
        self.assertEqual(result.stderr, "")
    
    def test_javascript_code_execution(self):
        """Test JavaScript code execution."""
        # Execute JavaScript code
        code = """
console.log("Hello, world!");
for (let i = 0; i < 3; i++) {
    console.log(`Number: ${i}`);
}
"""
        result = self.code_executor.execute(code, ProgrammingLanguage.JAVASCRIPT)
        
        # Check that result is as expected
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hello, world!", result.stdout)
        self.assertIn("Number: 0", result.stdout)
        self.assertIn("Number: 1", result.stdout)
        self.assertIn("Number: 2", result.stdout)
    
    def test_code_execution_with_input(self):
        """Test code execution with input."""
        # Execute Python code with input
        code = """
name = input("Enter your name: ")
print(f"Hello, {name}!")
"""
        result = self.code_executor.execute(code, ProgrammingLanguage.PYTHON, input_data="Basira")
        
        # Check that result is as expected
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hello, Basira!", result.stdout)
    
    def test_code_execution_with_timeout(self):
        """Test code execution with timeout."""
        # Execute Python code that takes too long
        code = """
import time
time.sleep(10)
print("This should not be printed")
"""
        config = ExecutionConfig(
            language=ProgrammingLanguage.PYTHON,
            timeout=0.5  # 0.5 seconds timeout
        )
        result = self.code_executor.execute(code, ProgrammingLanguage.PYTHON, config)
        
        # Check that result indicates timeout
        self.assertEqual(result.exit_code, -1)  # -1 indicates timeout
        self.assertIn("timed out", result.stderr)


if __name__ == '__main__':
    unittest.main()
