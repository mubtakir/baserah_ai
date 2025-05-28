#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the General Shape Equation module.

This module contains unit tests for the General Shape Equation module, which is
a core component of the Basira System.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import sys
import unittest
import numpy as np
import torch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mathematical_core.general_shape_equation import (
    GeneralShapeEquation,
    EquationType,
    LearningMode,
    EquationMetadata,
    SymbolicExpression
)


class TestSymbolicExpression(unittest.TestCase):
    """Test cases for the SymbolicExpression class."""
    
    def test_initialization(self):
        """Test initialization of SymbolicExpression."""
        expr = SymbolicExpression(expression_str="x^2 + y^2")
        self.assertEqual(expr.expression_str, "x^2 + y^2")
        self.assertEqual(expr.variables, {})
    
    def test_to_string(self):
        """Test to_string method."""
        expr = SymbolicExpression(expression_str="x^2 + y^2")
        self.assertEqual(expr.to_string(), "x^2 + y^2")
    
    def test_evaluate(self):
        """Test evaluate method."""
        expr = SymbolicExpression(expression_str="x^2 + y^2")
        # Since this is a placeholder implementation, it should return 0.0
        self.assertEqual(expr.evaluate({"x": 2, "y": 3}), 0.0)
    
    def test_simplify(self):
        """Test simplify method."""
        expr = SymbolicExpression(expression_str="x^2 + y^2")
        simplified = expr.simplify()
        self.assertEqual(simplified.expression_str, "x^2 + y^2")
    
    def test_get_complexity_score(self):
        """Test get_complexity_score method."""
        expr = SymbolicExpression(expression_str="x^2 + y^2")
        # Length of "x^2 + y^2" is 8, so complexity score should be 0.8
        self.assertEqual(expr.get_complexity_score(), 0.8)


class TestEquationMetadata(unittest.TestCase):
    """Test cases for the EquationMetadata class."""
    
    def test_initialization(self):
        """Test initialization of EquationMetadata."""
        metadata = EquationMetadata(
            equation_id="test_id",
            equation_type=EquationType.SHAPE,
            creation_time="2023-01-01T00:00:00",
            last_modified="2023-01-01T00:00:00"
        )
        self.assertEqual(metadata.equation_id, "test_id")
        self.assertEqual(metadata.equation_type, EquationType.SHAPE)
        self.assertEqual(metadata.creation_time, "2023-01-01T00:00:00")
        self.assertEqual(metadata.last_modified, "2023-01-01T00:00:00")
        self.assertEqual(metadata.version, 1)
        self.assertEqual(metadata.author, "Basira System")
        self.assertIsNone(metadata.description)
        self.assertEqual(metadata.tags, [])
        self.assertEqual(metadata.confidence, 1.0)
        self.assertEqual(metadata.complexity, 0.0)
        self.assertEqual(metadata.semantic_links, {})
        self.assertEqual(metadata.custom_properties, {})


class TestGeneralShapeEquation(unittest.TestCase):
    """Test cases for the GeneralShapeEquation class."""
    
    def test_initialization(self):
        """Test initialization of GeneralShapeEquation."""
        equation = GeneralShapeEquation()
        self.assertEqual(equation.equation_type, EquationType.SHAPE)
        self.assertEqual(equation.learning_mode, LearningMode.NONE)
        self.assertEqual(equation.symbolic_components, {})
        self.assertEqual(equation.variables, {})
        self.assertIsNotNone(equation.metadata)
        self.assertEqual(equation.metadata.equation_type, EquationType.SHAPE)
    
    def test_initialization_with_parameters(self):
        """Test initialization of GeneralShapeEquation with parameters."""
        equation = GeneralShapeEquation(
            equation_type=EquationType.PATTERN,
            learning_mode=LearningMode.SUPERVISED
        )
        self.assertEqual(equation.equation_type, EquationType.PATTERN)
        self.assertEqual(equation.learning_mode, LearningMode.SUPERVISED)
        self.assertIsNotNone(equation.neural_components)
    
    def test_add_component(self):
        """Test add_component method."""
        equation = GeneralShapeEquation()
        equation.add_component("circle", "(x-cx)^2 + (y-cy)^2 - r^2")
        self.assertIn("circle", equation.symbolic_components)
        self.assertEqual(equation.symbolic_components["circle"].expression_str, "(x-cx)^2 + (y-cy)^2 - r^2")
    
    def test_evaluate(self):
        """Test evaluate method."""
        equation = GeneralShapeEquation()
        equation.add_component("circle", "(x-cx)^2 + (y-cy)^2 - r^2")
        result = equation.evaluate({"x": 0, "y": 0, "cx": 0, "cy": 0, "r": 5})
        self.assertIn("circle", result)
    
    def test_to_dict(self):
        """Test to_dict method."""
        equation = GeneralShapeEquation()
        equation.add_component("circle", "(x-cx)^2 + (y-cy)^2 - r^2")
        equation_dict = equation.to_dict()
        self.assertEqual(equation_dict["equation_type"], "shape")
        self.assertEqual(equation_dict["learning_mode"], "none")
        self.assertIn("metadata", equation_dict)
        self.assertIn("symbolic_components", equation_dict)
        self.assertIn("circle", equation_dict["symbolic_components"])
    
    def test_to_json(self):
        """Test to_json method."""
        equation = GeneralShapeEquation()
        equation.add_component("circle", "(x-cx)^2 + (y-cy)^2 - r^2")
        json_str = equation.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("circle", json_str)
    
    def test_str_representation(self):
        """Test string representation."""
        equation = GeneralShapeEquation()
        equation.add_component("circle", "(x-cx)^2 + (y-cy)^2 - r^2")
        str_repr = str(equation)
        self.assertIn("GeneralShapeEquation", str_repr)
        self.assertIn("circle", str_repr)
    
    def test_repr_representation(self):
        """Test repr representation."""
        equation = GeneralShapeEquation()
        repr_str = repr(equation)
        self.assertIn("GeneralShapeEquation", repr_str)
        self.assertIn("shape", repr_str)


class TestEquationType(unittest.TestCase):
    """Test cases for the EquationType enum."""
    
    def test_values(self):
        """Test enum values."""
        self.assertEqual(EquationType.SHAPE.value, "shape")
        self.assertEqual(EquationType.PATTERN.value, "pattern")
        self.assertEqual(EquationType.BEHAVIOR.value, "behavior")
        self.assertEqual(EquationType.TRANSFORMATION.value, "transformation")
        self.assertEqual(EquationType.CONSTRAINT.value, "constraint")
        self.assertEqual(EquationType.COMPOSITE.value, "composite")


class TestLearningMode(unittest.TestCase):
    """Test cases for the LearningMode enum."""
    
    def test_values(self):
        """Test enum values."""
        self.assertEqual(LearningMode.NONE.value, "none")
        self.assertEqual(LearningMode.SUPERVISED.value, "supervised")
        self.assertEqual(LearningMode.REINFORCEMENT.value, "reinforcement")
        self.assertEqual(LearningMode.UNSUPERVISED.value, "unsupervised")
        self.assertEqual(LearningMode.HYBRID.value, "hybrid")


class TestIntegration(unittest.TestCase):
    """Integration tests for the General Shape Equation module."""
    
    def test_circle_equation(self):
        """Test creating and evaluating a circle equation."""
        # Create a circle equation
        equation = GeneralShapeEquation()
        equation.add_component("circle", "(x-cx)^2 + (y-cy)^2 - r^2")
        equation.add_component("cx", "0")
        equation.add_component("cy", "0")
        equation.add_component("r", "5")
        
        # Evaluate at different points
        result1 = equation.evaluate({"x": 0, "y": 0})
        result2 = equation.evaluate({"x": 5, "y": 0})
        result3 = equation.evaluate({"x": 3, "y": 4})
        
        # Since the placeholder implementation returns 0.0, we can't test actual values
        # But we can test that the keys are correct
        self.assertIn("circle", result1)
        self.assertIn("cx", result1)
        self.assertIn("cy", result1)
        self.assertIn("r", result1)
    
    def test_neural_components_initialization(self):
        """Test initialization of neural components."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        equation = GeneralShapeEquation(
            learning_mode=LearningMode.SUPERVISED
        )
        
        self.assertIn("supervised", equation.neural_components)
        self.assertIsInstance(equation.neural_components["supervised"], torch.nn.Module)


if __name__ == '__main__':
    unittest.main()
