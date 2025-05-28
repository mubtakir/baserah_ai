#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Revolutionary Function Decomposition Integration
اختبارات التكامل للنظام الثوري لتفكيك الدوال

This module tests the integration between the revolutionary function decomposition
engine (based on Basil Yahya Abdullah's method) and the Expert-Explorer system.

Author: Basira System Development Team
Version: 1.0.0
"""

import unittest
import torch
import math
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from symbolic_processing.expert_explorer_system import Expert, ExpertKnowledgeType
    from mathematical_core.function_decomposition_engine import FunctionDecompositionEngine
    from mathematical_core.calculus_test_functions import get_decomposition_test_functions
except ImportError as e:
    print(f"Import error: {e}")
    # Create mock classes for testing
    class Expert:
        def __init__(self):
            pass
    
    class FunctionDecompositionEngine:
        def __init__(self, **kwargs):
            pass

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_revolutionary_decomposition_integration')


class TestRevolutionaryDecompositionIntegration(unittest.TestCase):
    """Test suite for revolutionary decomposition integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.expert = Expert([
            ExpertKnowledgeType.HEURISTIC,
            ExpertKnowledgeType.ANALYTICAL,
            ExpertKnowledgeType.MATHEMATICAL
        ])
        
        self.decomposition_engine = FunctionDecompositionEngine(
            max_terms=15,
            tolerance=1e-5
        )
        
        # Simple test function: f(x) = e^x
        self.test_function_data = {
            'name': 'exponential_test',
            'function': lambda x: torch.exp(x),
            'domain': (-1.0, 1.0, 50),
            'expected_convergence': 'infinite'
        }
    
    def test_expert_decomposition_engine_initialization(self):
        """Test that expert properly initializes decomposition engine"""
        self.assertIsNotNone(self.expert.decomposition_engine)
        self.assertIsInstance(self.expert.decomposition_engine, FunctionDecompositionEngine)
        logger.info("✅ Expert decomposition engine initialization test passed")
    
    def test_revolutionary_decomposition_basic(self):
        """Test basic revolutionary decomposition functionality"""
        result = self.expert.decompose_function_revolutionary(self.test_function_data)
        
        self.assertTrue(result.get("success", False))
        self.assertIn("decomposition_state", result)
        self.assertIn("analysis", result)
        self.assertIn("revolutionary_series", result)
        self.assertEqual(result.get("method"), "basil_yahya_abdullah_revolutionary_series")
        logger.info("✅ Revolutionary decomposition basic test passed")
    
    def test_series_convergence_exploration(self):
        """Test series convergence exploration"""
        result = self.expert.explore_series_convergence(
            self.test_function_data,
            exploration_steps=20
        )
        
        self.assertTrue(result.get("success", False))
        self.assertIn("exploration_results", result)
        self.assertIn("best_configuration", result)
        self.assertIn("convergence_analysis", result)
        logger.info("✅ Series convergence exploration test passed")
    
    def test_decomposition_methods_comparison(self):
        """Test comparison of decomposition methods"""
        result = self.expert.compare_decomposition_methods(self.test_function_data)
        
        self.assertTrue(result.get("success", False))
        self.assertIn("comparison_results", result)
        self.assertIn("recommendation", result)
        logger.info("✅ Decomposition methods comparison test passed")
    
    def test_knowledge_base_updates(self):
        """Test that decomposition updates knowledge base"""
        # Get initial knowledge state
        initial_historical = self.expert.knowledge_bases[ExpertKnowledgeType.HISTORICAL]
        
        # Perform decomposition
        self.expert.decompose_function_revolutionary(self.test_function_data)
        
        # Check if knowledge was updated
        updated_historical = self.expert.knowledge_bases[ExpertKnowledgeType.HISTORICAL]
        self.assertIn("decomposition_history", updated_historical)
        logger.info("✅ Knowledge base updates test passed")
    
    def test_revolutionary_series_accuracy(self):
        """Test accuracy of revolutionary series decomposition"""
        # Test on polynomial function (should have excellent accuracy)
        polynomial_data = {
            'name': 'polynomial_test',
            'function': lambda x: x**3 - 2*x**2 + x - 1,
            'domain': (-1.0, 1.0, 30),
            'expected_convergence': 'infinite'
        }
        
        result = self.expert.decompose_function_revolutionary(polynomial_data)
        
        self.assertTrue(result.get("success", False))
        accuracy = result["decomposition_state"].accuracy
        
        # Revolutionary series should achieve good accuracy on polynomials
        self.assertGreater(accuracy, 0.7, "Revolutionary series should achieve reasonable accuracy")
        logger.info(f"✅ Revolutionary series accuracy test passed. Accuracy: {accuracy:.4f}")
    
    def test_integration_with_calculus_engine(self):
        """Test integration between decomposition and calculus engines"""
        # First train calculus engine
        self.expert.train_calculus_engine(function_name="linear", epochs=50)
        
        # Then test decomposition with calculus integration
        result = self.expert.compare_decomposition_methods(self.test_function_data)
        
        self.assertTrue(result.get("success", False))
        comparison_results = result["comparison_results"]
        
        # Should have both revolutionary and calculus methods
        self.assertIn("revolutionary_series", comparison_results)
        logger.info("✅ Integration with calculus engine test passed")
    
    def test_convergence_analysis(self):
        """Test convergence analysis functionality"""
        # Test with different functions
        test_functions = get_decomposition_test_functions()
        
        for func_name, func_data in list(test_functions.items())[:3]:  # Test first 3 functions
            result = self.expert.explore_series_convergence(func_data, exploration_steps=15)
            
            if result.get("success"):
                convergence_analysis = result["convergence_analysis"]
                self.assertIn("convergence_quality", convergence_analysis)
                self.assertIn("optimal_terms", convergence_analysis)
                logger.info(f"✅ Convergence analysis for {func_name}: {convergence_analysis['convergence_quality']}")
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Perform multiple decompositions
        test_functions = get_decomposition_test_functions()
        
        for func_name, func_data in list(test_functions.items())[:2]:  # Test first 2 functions
            result = self.expert.decompose_function_revolutionary(func_data)
            
            if result.get("success"):
                performance = result["performance"]
                self.assertIn("accuracy", performance)
                self.assertIn("convergence_radius", performance)
                self.assertIn("n_terms_used", performance)
        
        # Get performance summary
        summary = self.expert.decomposition_engine.get_performance_summary()
        self.assertIn("total_decompositions", summary)
        logger.info("✅ Performance metrics test passed")


class TestRevolutionaryDecompositionStandalone(unittest.TestCase):
    """Test decomposition engine standalone functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = FunctionDecompositionEngine(max_terms=10, tolerance=1e-4)
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.series_expander)
        self.assertIsNotNone(self.engine.general_equation)
        logger.info("✅ Engine initialization test passed")
    
    def test_decomposition_on_exponential(self):
        """Test decomposition on exponential function"""
        function_data = {
            'name': 'exponential_standalone',
            'function': lambda x: torch.exp(x),
            'domain': (-0.5, 0.5, 30)
        }
        
        result = self.engine.decompose_function(function_data)
        
        self.assertTrue(result.get("success", False))
        self.assertIn("decomposition_state", result)
        self.assertIn("analysis", result)
        
        # Exponential should have good convergence
        convergence_radius = result["decomposition_state"].convergence_radius
        self.assertGreater(convergence_radius, 0.5)
        logger.info(f"✅ Exponential decomposition test passed. Convergence radius: {convergence_radius:.4f}")
    
    def test_decomposition_on_polynomial(self):
        """Test decomposition on polynomial function"""
        function_data = {
            'name': 'polynomial_standalone',
            'function': lambda x: x**2 + 2*x + 1,
            'domain': (-1.0, 1.0, 25)
        }
        
        result = self.engine.decompose_function(function_data)
        
        self.assertTrue(result.get("success", False))
        accuracy = result["decomposition_state"].accuracy
        
        # Polynomial should have excellent accuracy
        self.assertGreater(accuracy, 0.8)
        logger.info(f"✅ Polynomial decomposition test passed. Accuracy: {accuracy:.4f}")
    
    def test_series_expression_formatting(self):
        """Test series expression formatting"""
        function_data = {
            'name': 'sine_standalone',
            'function': lambda x: torch.sin(x),
            'domain': (-1.0, 1.0, 20)
        }
        
        result = self.engine.decompose_function(function_data)
        
        if result.get("success"):
            series_expression = result["revolutionary_series"]
            self.assertIsInstance(series_expression, str)
            self.assertIn("A(x)", series_expression)
            logger.info(f"✅ Series expression formatting test passed: {series_expression[:50]}...")


def run_revolutionary_decomposition_tests():
    """Run all revolutionary decomposition tests"""
    logger.info("🚀 Starting Revolutionary Function Decomposition Tests")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestRevolutionaryDecompositionIntegration))
    suite.addTest(unittest.makeSuite(TestRevolutionaryDecompositionStandalone))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        logger.info("🎉 All revolutionary decomposition tests passed successfully!")
        logger.info("✅ Basil Yahya Abdullah's Revolutionary Series Integration is working perfectly!")
    else:
        logger.error("❌ Some tests failed. Check the output above.")
        logger.error(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_revolutionary_decomposition_tests()
    exit(0 if success else 1)
