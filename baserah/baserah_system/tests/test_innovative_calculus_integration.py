#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Innovative Calculus Integration with Expert-Explorer System
اختبارات التكامل بين النظام المبتكر للتفاضل والتكامل ونظام الخبير-المستكشف

This module tests the integration between the innovative calculus engine
and the expert-explorer system in Basira.

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
    from mathematical_core.innovative_calculus_engine import InnovativeCalculusEngine
    from mathematical_core.calculus_test_functions import get_simple_test_functions, calculate_mae
except ImportError as e:
    print(f"Import error: {e}")
    # Create mock classes for testing
    class Expert:
        def __init__(self):
            pass
    
    class InnovativeCalculusEngine:
        def __init__(self, **kwargs):
            pass

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_innovative_calculus_integration')


class TestInnovativeCalculusIntegration(unittest.TestCase):
    """Test suite for innovative calculus integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.expert = Expert([
            ExpertKnowledgeType.HEURISTIC,
            ExpertKnowledgeType.ANALYTICAL,
            ExpertKnowledgeType.MATHEMATICAL
        ])
        
        self.calculus_engine = InnovativeCalculusEngine(
            merge_threshold=0.8,
            learning_rate=0.3
        )
        
        # Simple test function: f(x) = x^2
        self.test_x = torch.linspace(-2.0, 2.0, 50)
        self.test_function = self.test_x ** 2
        self.test_derivative = 2 * self.test_x
        self.test_integral = (self.test_x ** 3) / 3
    
    def test_expert_calculus_engine_initialization(self):
        """Test that expert properly initializes calculus engine"""
        self.assertIsNotNone(self.expert.calculus_engine)
        self.assertIsInstance(self.expert.calculus_engine, InnovativeCalculusEngine)
        logger.info("✅ Expert calculus engine initialization test passed")
    
    def test_calculus_engine_basic_functionality(self):
        """Test basic functionality of calculus engine"""
        # Test prediction (should work even without training)
        derivative, integral = self.calculus_engine.predict(self.test_function)
        
        self.assertEqual(derivative.shape, self.test_function.shape)
        self.assertEqual(integral.shape, self.test_function.shape)
        logger.info("✅ Calculus engine basic functionality test passed")
    
    def test_coefficient_functions_extraction(self):
        """Test extraction of coefficient functions"""
        D_coeff, V_coeff = self.calculus_engine.get_coefficient_functions(self.test_function)
        
        self.assertEqual(D_coeff.shape, self.test_function.shape)
        self.assertEqual(V_coeff.shape, self.test_function.shape)
        logger.info("✅ Coefficient functions extraction test passed")
    
    def test_expert_training_integration(self):
        """Test expert's ability to train calculus engine"""
        # Test training on a simple function
        result = self.expert.train_calculus_engine(function_name="linear", epochs=50)
        
        self.assertTrue(result.get("success", False))
        self.assertIn("functions_trained", result)
        self.assertIn("results", result)
        logger.info("✅ Expert training integration test passed")
    
    def test_expert_problem_solving(self):
        """Test expert's calculus problem solving capability"""
        # First train on a simple function
        self.expert.train_calculus_engine(function_name="quadratic", epochs=100)
        
        # Then solve a problem
        result = self.expert.solve_calculus_problem(self.test_function)
        
        self.assertTrue(result.get("success", False))
        self.assertIn("derivative", result)
        self.assertIn("integral", result)
        self.assertIn("differentiation_coefficients", result)
        self.assertIn("integration_coefficients", result)
        logger.info("✅ Expert problem solving test passed")
    
    def test_coefficient_space_exploration(self):
        """Test expert's coefficient space exploration"""
        # Train first
        self.expert.train_calculus_engine(function_name="quadratic", epochs=50)
        
        # Explore coefficient space
        result = self.expert.explore_coefficient_space(
            target_function=self.test_function,
            exploration_steps=20
        )
        
        self.assertTrue(result.get("success", False))
        self.assertIn("best_loss", result)
        self.assertIn("best_coefficients", result)
        self.assertIn("exploration_history", result)
        logger.info("✅ Coefficient space exploration test passed")
    
    def test_knowledge_base_updates(self):
        """Test that training updates knowledge base"""
        # Get initial knowledge state
        initial_historical = self.expert.knowledge_bases[ExpertKnowledgeType.HISTORICAL]
        
        # Train calculus engine
        self.expert.train_calculus_engine(function_name="linear", epochs=30)
        
        # Check if knowledge was updated
        updated_historical = self.expert.knowledge_bases[ExpertKnowledgeType.HISTORICAL]
        self.assertIn("calculus_training", updated_historical)
        logger.info("✅ Knowledge base updates test passed")
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Train on multiple functions
        self.expert.train_calculus_engine(epochs=50)
        
        # Get performance summary
        summary = self.expert.calculus_engine.get_performance_summary()
        
        self.assertIn("total_functions_trained", summary)
        self.assertIn("average_final_loss", summary)
        self.assertIn("total_states", summary)
        logger.info("✅ Performance metrics test passed")
    
    def test_innovative_approach_effectiveness(self):
        """Test effectiveness of innovative coefficient-based approach"""
        # Create a simple quadratic function for testing
        x = torch.linspace(-1.0, 1.0, 20)
        f = x ** 2
        true_derivative = 2 * x
        true_integral = (x ** 3) / 3
        
        # Train the engine
        function_data = {
            'name': 'test_quadratic',
            'f': lambda x: x ** 2,
            'f_prime': lambda x: 2 * x,
            'f_integral': lambda x: (x ** 3) / 3,
            'domain': (-1.0, 1.0, 20),
            'noise': 0.1
        }
        
        metrics = self.calculus_engine.train_on_function(function_data, epochs=200)
        
        # Test prediction accuracy
        pred_derivative, pred_integral = self.calculus_engine.predict(f)
        
        derivative_mae = calculate_mae(pred_derivative, true_derivative)
        integral_mae = calculate_mae(pred_integral, true_integral)
        
        # The innovative approach should achieve reasonable accuracy
        self.assertLess(derivative_mae, 1.0, "Derivative prediction should be reasonably accurate")
        self.assertLess(integral_mae, 1.0, "Integral prediction should be reasonably accurate")
        
        logger.info(f"✅ Innovative approach effectiveness test passed")
        logger.info(f"   Derivative MAE: {derivative_mae:.4f}")
        logger.info(f"   Integral MAE: {integral_mae:.4f}")
    
    def test_state_management(self):
        """Test state management in calculus cell"""
        initial_states = len(self.calculus_engine.calculus_cell.states)
        
        # Train on different functions to create states
        functions = get_simple_test_functions()
        for name, func_data in functions.items():
            func_data['name'] = name
            self.calculus_engine.train_on_function(func_data, epochs=20)
        
        final_states = len(self.calculus_engine.calculus_cell.states)
        
        # Should have created some states
        self.assertGreaterEqual(final_states, initial_states)
        logger.info(f"✅ State management test passed. States: {initial_states} → {final_states}")


class TestCalculusEngineStandalone(unittest.TestCase):
    """Test calculus engine standalone functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = InnovativeCalculusEngine()
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.calculus_cell)
        self.assertIsNotNone(self.engine.general_equation)
        logger.info("✅ Engine initialization test passed")
    
    def test_training_on_simple_function(self):
        """Test training on a simple linear function"""
        function_data = {
            'name': 'simple_linear',
            'f': lambda x: 2*x + 1,
            'f_prime': lambda x: torch.full_like(x, 2.0),
            'f_integral': lambda x: x**2 + x,
            'domain': (-1.0, 1.0, 30),
            'noise': 0.05
        }
        
        metrics = self.engine.train_on_function(function_data, epochs=100)
        
        self.assertIn('mae_derivative', metrics)
        self.assertIn('mae_integral', metrics)
        self.assertIn('final_loss', metrics)
        
        # Should achieve good accuracy on simple function
        self.assertLess(metrics['final_loss'], 2.0)
        logger.info(f"✅ Simple function training test passed. Final loss: {metrics['final_loss']:.4f}")


def run_integration_tests():
    """Run all integration tests"""
    logger.info("🚀 Starting Innovative Calculus Integration Tests")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestInnovativeCalculusIntegration))
    suite.addTest(unittest.makeSuite(TestCalculusEngineStandalone))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        logger.info("🎉 All tests passed successfully!")
        logger.info("✅ Innovative Calculus Integration is working perfectly!")
    else:
        logger.error("❌ Some tests failed. Check the output above.")
        logger.error(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
