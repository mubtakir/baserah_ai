#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Integration Validation Module for Basira System

This module provides tests and validation utilities for the integrated Basira System,
ensuring that all components work together seamlessly.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, Set
import random
import logging
import time
from enum import Enum
from dataclasses import dataclass, field

# Import from parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced.general_shape_equation import (
    GeneralShapeEquation, EquationType, LearningMode, EquationMetadata,
    GSEFactory
)
from enhanced.learning_integration import (
    EnhancedDeepLearningAdapter, EnhancedReinforcementLearningAdapter,
    EnhancedExpertExplorerSystem, ShapeEquationDataset
)
from enhanced.expert_explorer_interaction import (
    AdvancedExpertExplorerSystem, ExplorationStrategy, ExpertKnowledgeType,
    ExplorationResult, ExpertFeedback
)
from enhanced.semantic_integration import (
    SemanticDatabaseManager, SemanticEquationGenerator, SemanticGuidedExplorer,
    SemanticVector, SemanticAxis, SemanticCategory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('system_validation')


class ValidationResult:
    """Result of a validation test."""
    
    def __init__(self, name: str, success: bool, details: Dict[str, Any] = None):
        """
        Initialize a ValidationResult.
        
        Args:
            name: Test name
            success: Whether the test succeeded
            details: Optional test details
        """
        self.name = name
        self.success = success
        self.details = details or {}
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    def __str__(self) -> str:
        """String representation of the validation result."""
        status = "PASSED" if self.success else "FAILED"
        return f"{self.name}: {status} ({self.timestamp})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'success': self.success,
            'details': self.details,
            'timestamp': self.timestamp
        }


class SystemValidator:
    """
    Validator for the integrated Basira System.
    
    This class provides methods for testing and validating the integration
    of all system components.
    """
    
    def __init__(self, database_path: str, output_dir: str = None):
        """
        Initialize a SystemValidator.
        
        Args:
            database_path: Path to the semantic database file
            output_dir: Optional directory for output files
        """
        self.logger = logging.getLogger('system_validation')
        self.database_path = database_path
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'validation_results')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.db_manager = None
        self.equation_generator = None
        self.expert_explorer = None
        self.semantic_explorer = None
        
        # Initialize validation results
        self.results = []
    
    def initialize_components(self) -> ValidationResult:
        """
        Initialize all system components.
        
        Returns:
            ValidationResult for the initialization
        """
        try:
            # Create database manager
            self.db_manager = SemanticDatabaseManager(self.database_path)
            
            # Create equation generator
            self.equation_generator = SemanticEquationGenerator(self.db_manager)
            
            # Create expert-explorer system
            self.expert_explorer = AdvancedExpertExplorerSystem()
            
            # Create semantic guided explorer
            self.semantic_explorer = SemanticGuidedExplorer(
                self.db_manager,
                self.equation_generator,
                self.expert_explorer
            )
            
            # Check if all components are initialized
            components_initialized = (
                self.db_manager is not None and
                self.equation_generator is not None and
                self.expert_explorer is not None and
                self.semantic_explorer is not None
            )
            
            result = ValidationResult(
                name="Component Initialization",
                success=components_initialized,
                details={
                    'database_manager': self.db_manager is not None,
                    'equation_generator': self.equation_generator is not None,
                    'expert_explorer': self.expert_explorer is not None,
                    'semantic_explorer': self.semantic_explorer is not None
                }
            )
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            
            result = ValidationResult(
                name="Component Initialization",
                success=False,
                details={'error': str(e)}
            )
            
            self.results.append(result)
            return result
    
    def validate_semantic_database(self) -> ValidationResult:
        """
        Validate the semantic database.
        
        Returns:
            ValidationResult for the database validation
        """
        try:
            # Check if database is loaded
            if not self.db_manager or not self.db_manager.database:
                raise ValueError("Semantic database not loaded")
            
            # Check if database has entries
            if len(self.db_manager.database) == 0:
                raise ValueError("Semantic database is empty")
            
            # Check if properties are extracted
            if len(self.db_manager.properties) == 0:
                raise ValueError("No properties extracted from database")
            
            # Check if axes are extracted
            if len(self.db_manager.axes) == 0:
                raise ValueError("No semantic axes extracted from database")
            
            # Check a few specific letters
            required_letters = ['A', 'O', 'I', 'R', 'N']
            missing_letters = [letter for letter in required_letters if letter not in self.db_manager.database]
            
            if missing_letters:
                raise ValueError(f"Missing required letters: {', '.join(missing_letters)}")
            
            # Check semantic vector generation
            letter_vector = self.db_manager.get_letter_semantic_vector('A')
            if not letter_vector or not letter_vector.dimensions:
                raise ValueError("Failed to generate semantic vector for letter A")
            
            word_vector = self.db_manager.get_word_semantic_vector('TEST')
            if not word_vector or not word_vector.dimensions:
                raise ValueError("Failed to generate semantic vector for word TEST")
            
            result = ValidationResult(
                name="Semantic Database Validation",
                success=True,
                details={
                    'database_size': len(self.db_manager.database),
                    'properties_count': len(self.db_manager.properties),
                    'axes_count': len(self.db_manager.axes),
                    'categories_count': len(self.db_manager.categories),
                    'letter_vector_dimensions': len(letter_vector.dimensions),
                    'word_vector_dimensions': len(word_vector.dimensions)
                }
            )
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Error validating semantic database: {e}")
            
            result = ValidationResult(
                name="Semantic Database Validation",
                success=False,
                details={'error': str(e)}
            )
            
            self.results.append(result)
            return result
    
    def validate_equation_generation(self) -> ValidationResult:
        """
        Validate equation generation from semantic properties.
        
        Returns:
            ValidationResult for the equation generation validation
        """
        try:
            # Check if equation generator is initialized
            if not self.equation_generator:
                raise ValueError("Equation generator not initialized")
            
            # Generate equations from letters
            letter_equations = {}
            for letter in ['A', 'O', 'R']:
                equation = self.equation_generator.generate_equation_from_letter(letter)
                
                if not equation:
                    raise ValueError(f"Failed to generate equation for letter {letter}")
                
                letter_equations[letter] = equation
            
            # Generate equation from a word
            word_equation = self.equation_generator.generate_equation_from_word('TEST')
            
            if not word_equation:
                raise ValueError("Failed to generate equation for word TEST")
            
            # Generate equation from a concept
            concept_equation = self.equation_generator.generate_equation_from_semantic_concept('Movement')
            
            if not concept_equation:
                raise ValueError("Failed to generate equation for concept Movement")
            
            # Check semantic links
            for letter, equation in letter_equations.items():
                semantic_links = equation.get_semantic_links()
                
                if not semantic_links:
                    raise ValueError(f"No semantic links in equation for letter {letter}")
            
            word_semantic_links = word_equation.get_semantic_links()
            if not word_semantic_links:
                raise ValueError("No semantic links in equation for word TEST")
            
            concept_semantic_links = concept_equation.get_semantic_links()
            if not concept_semantic_links:
                raise ValueError("No semantic links in equation for concept Movement")
            
            result = ValidationResult(
                name="Equation Generation Validation",
                success=True,
                details={
                    'letter_equations': {
                        letter: {
                            'components': len(eq.symbolic_components),
                            'semantic_links': len(eq.get_semantic_links())
                        }
                        for letter, eq in letter_equations.items()
                    },
                    'word_equation': {
                        'components': len(word_equation.symbolic_components),
                        'semantic_links': len(word_semantic_links)
                    },
                    'concept_equation': {
                        'components': len(concept_equation.symbolic_components),
                        'semantic_links': len(concept_semantic_links)
                    }
                }
            )
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Error validating equation generation: {e}")
            
            result = ValidationResult(
                name="Equation Generation Validation",
                success=False,
                details={'error': str(e)}
            )
            
            self.results.append(result)
            return result
    
    def validate_expert_explorer_system(self) -> ValidationResult:
        """
        Validate the expert-explorer system.
        
        Returns:
            ValidationResult for the expert-explorer validation
        """
        try:
            # Check if expert-explorer system is initialized
            if not self.expert_explorer:
                raise ValueError("Expert-explorer system not initialized")
            
            # Create a test equation
            test_equation = GSEFactory.create_basic_shape('circle', cx=0, cy=0, radius=1)
            
            # Set as initial equation
            self.expert_explorer.set_initial_equation(test_equation)
            
            # Test expert evaluation
            evaluation = self.expert_explorer.expert_evaluate(test_equation)
            
            if evaluation is None:
                raise ValueError("Expert evaluation failed")
            
            # Test expert guidance
            guidance = self.expert_explorer.expert_provide_guidance(test_equation)
            
            if not guidance:
                raise ValueError("Expert guidance failed")
            
            # Test explorer exploration (with minimal cycles)
            exploration_result = self.expert_explorer.explorer_explore(
                strategy=ExplorationStrategy.GUIDED,
                steps=2
            )
            
            if not exploration_result:
                raise ValueError("Explorer exploration failed")
            
            # Test expert-explorer interaction (with minimal cycles)
            interaction_result = self.expert_explorer.expert_explorer_interaction(
                cycles=1,
                steps_per_cycle=2
            )
            
            if not interaction_result:
                raise ValueError("Expert-explorer interaction failed")
            
            result = ValidationResult(
                name="Expert-Explorer System Validation",
                success=True,
                details={
                    'expert_evaluation': evaluation,
                    'guidance_count': len(guidance),
                    'exploration_score': exploration_result.score,
                    'exploration_strategy': exploration_result.strategy.value,
                    'interaction_result': {
                        'components': len(interaction_result.symbolic_components),
                        'complexity': interaction_result.metadata.complexity
                    }
                }
            )
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Error validating expert-explorer system: {e}")
            
            result = ValidationResult(
                name="Expert-Explorer System Validation",
                success=False,
                details={'error': str(e)}
            )
            
            self.results.append(result)
            return result
    
    def validate_semantic_guided_exploration(self) -> ValidationResult:
        """
        Validate semantic-guided exploration.
        
        Returns:
            ValidationResult for the semantic-guided exploration validation
        """
        try:
            # Check if semantic explorer is initialized
            if not self.semantic_explorer:
                raise ValueError("Semantic explorer not initialized")
            
            # Test exploration with a letter (with minimal cycles)
            letter_result = self.semantic_explorer.explore_with_letter(
                letter='A',
                cycles=1,
                steps_per_cycle=2
            )
            
            if not letter_result:
                raise ValueError("Exploration with letter failed")
            
            # Test exploration with a word (with minimal cycles)
            word_result = self.semantic_explorer.explore_with_word(
                word='TEST',
                cycles=1,
                steps_per_cycle=2
            )
            
            if not word_result:
                raise ValueError("Exploration with word failed")
            
            # Test exploration with a concept (with minimal cycles)
            concept_result = self.semantic_explorer.explore_with_semantic_guidance(
                target_concepts=['Movement'],
                cycles=1,
                steps_per_cycle=2
            )
            
            if not concept_result:
                raise ValueError("Exploration with concept failed")
            
            result = ValidationResult(
                name="Semantic-Guided Exploration Validation",
                success=True,
                details={
                    'letter_exploration': {
                        'components': len(letter_result.symbolic_components),
                        'semantic_links': len(letter_result.get_semantic_links()),
                        'complexity': letter_result.metadata.complexity
                    },
                    'word_exploration': {
                        'components': len(word_result.symbolic_components),
                        'semantic_links': len(word_result.get_semantic_links()),
                        'complexity': word_result.metadata.complexity
                    },
                    'concept_exploration': {
                        'components': len(concept_result.symbolic_components),
                        'semantic_links': len(concept_result.get_semantic_links()),
                        'complexity': concept_result.metadata.complexity
                    }
                }
            )
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Error validating semantic-guided exploration: {e}")
            
            result = ValidationResult(
                name="Semantic-Guided Exploration Validation",
                success=False,
                details={'error': str(e)}
            )
            
            self.results.append(result)
            return result
    
    def validate_learning_integration(self) -> ValidationResult:
        """
        Validate integration with learning components.
        
        Returns:
            ValidationResult for the learning integration validation
        """
        try:
            # Create test equation
            test_equation = GSEFactory.create_basic_shape('circle', cx=0, cy=0, radius=1)
            
            # Create deep learning adapter
            dl_adapter = EnhancedDeepLearningAdapter()
            
            # Create dataset (with minimal samples)
            dataset = ShapeEquationDataset([test_equation], samples_per_equation=10)
            
            # Train adapter (with minimal epochs)
            dl_adapter.train_on_dataset(dataset, num_epochs=2)
            
            # Enhance equation
            enhanced_equation = dl_adapter.enhance_equation_with_neural_correction(test_equation)
            
            if not enhanced_equation:
                raise ValueError("Deep learning enhancement failed")
            
            # Create reinforcement learning adapter
            rl_adapter = EnhancedReinforcementLearningAdapter()
            
            # Train adapter (with minimal episodes)
            rewards, best_equation = rl_adapter.train_on_equation(
                test_equation,
                num_episodes=2,
                steps_per_episode=2
            )
            
            if not best_equation:
                raise ValueError("Reinforcement learning training failed")
            
            # Test integration with expert-explorer system
            self.expert_explorer.set_initial_equation(test_equation)
            
            # Test deep learning exploration
            dl_exploration = self.expert_explorer.explorer_explore(
                strategy=ExplorationStrategy.DEEP_LEARNING,
                steps=2
            )
            
            if not dl_exploration:
                raise ValueError("Deep learning exploration failed")
            
            # Test reinforcement learning exploration
            rl_exploration = self.expert_explorer.explorer_explore(
                strategy=ExplorationStrategy.REINFORCEMENT_LEARNING,
                steps=2
            )
            
            if not rl_exploration:
                raise ValueError("Reinforcement learning exploration failed")
            
            result = ValidationResult(
                name="Learning Integration Validation",
                success=True,
                details={
                    'deep_learning': {
                        'enhanced_equation': {
                            'components': len(enhanced_equation.symbolic_components),
                            'neural_components': len(enhanced_equation.neural_components)
                        },
                        'exploration': {
                            'score': dl_exploration.score,
                            'strategy': dl_exploration.strategy.value
                        }
                    },
                    'reinforcement_learning': {
                        'best_equation': {
                            'components': len(best_equation.symbolic_components),
                            'complexity': best_equation.metadata.complexity
                        },
                        'exploration': {
                            'score': rl_exploration.score,
                            'strategy': rl_exploration.strategy.value
                        }
                    }
                }
            )
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Error validating learning integration: {e}")
            
            result = ValidationResult(
                name="Learning Integration Validation",
                success=False,
                details={'error': str(e)}
            )
            
            self.results.append(result)
            return result
    
    def validate_end_to_end_workflow(self) -> ValidationResult:
        """
        Validate the end-to-end workflow.
        
        Returns:
            ValidationResult for the end-to-end workflow validation
        """
        try:
            # Generate equation from a letter
            letter_equation = self.equation_generator.generate_equation_from_letter('A')
            
            # Set as initial equation
            self.semantic_explorer.system.set_initial_equation(letter_equation)
            
            # Run semantic-guided exploration (with minimal cycles)
            exploration_results = self.semantic_explorer.get_semantic_exploration_results(
                target='A',
                is_letter=True,
                cycles=1,
                steps_per_cycle=2,
                visualize=False
            )
            
            if not exploration_results or 'best_equation' not in exploration_results:
                raise ValueError("End-to-end workflow failed")
            
            # Check if semantic links are preserved
            best_equation = exploration_results['best_equation']
            semantic_links = best_equation.get_semantic_links()
            
            if not semantic_links:
                raise ValueError("Semantic links not preserved in end-to-end workflow")
            
            # Check if evaluation includes semantic metrics
            evaluation = exploration_results.get('evaluation', {})
            metric_scores = evaluation.get('metric_scores', {})
            
            if 'semantic_alignment' not in metric_scores:
                raise ValueError("Semantic alignment metric not included in evaluation")
            
            result = ValidationResult(
                name="End-to-End Workflow Validation",
                success=True,
                details={
                    'initial_equation': {
                        'components': len(letter_equation.symbolic_components),
                        'semantic_links': len(letter_equation.get_semantic_links())
                    },
                    'best_equation': {
                        'components': len(best_equation.symbolic_components),
                        'semantic_links': len(semantic_links),
                        'complexity': best_equation.metadata.complexity
                    },
                    'evaluation': {
                        'overall_score': evaluation.get('overall_score', 0.0),
                        'semantic_alignment': metric_scores.get('semantic_alignment', 0.0)
                    }
                }
            )
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Error validating end-to-end workflow: {e}")
            
            result = ValidationResult(
                name="End-to-End Workflow Validation",
                success=False,
                details={'error': str(e)}
            )
            
            self.results.append(result)
            return result
    
    def validate_performance(self) -> ValidationResult:
        """
        Validate system performance.
        
        Returns:
            ValidationResult for the performance validation
        """
        try:
            # Measure equation generation performance
            start_time = time.time()
            for letter in ['A', 'O', 'I', 'R', 'N']:
                self.equation_generator.generate_equation_from_letter(letter)
            letter_generation_time = (time.time() - start_time) / 5
            
            # Measure word equation generation performance
            start_time = time.time()
            self.equation_generator.generate_equation_from_word('TEST')
            word_generation_time = time.time() - start_time
            
            # Measure expert evaluation performance
            test_equation = GSEFactory.create_basic_shape('circle', cx=0, cy=0, radius=1)
            start_time = time.time()
            for _ in range(10):
                self.expert_explorer.expert_evaluate(test_equation)
            evaluation_time = (time.time() - start_time) / 10
            
            # Measure exploration performance (with minimal steps)
            self.expert_explorer.set_initial_equation(test_equation)
            start_time = time.time()
            self.expert_explorer.explorer_explore(steps=1)
            exploration_time = time.time() - start_time
            
            # Define performance thresholds
            thresholds = {
                'letter_generation': 0.5,  # seconds
                'word_generation': 1.0,    # seconds
                'evaluation': 0.1,         # seconds
                'exploration': 1.0         # seconds
            }
            
            # Check if performance meets thresholds
            performance_ok = (
                letter_generation_time <= thresholds['letter_generation'] and
                word_generation_time <= thresholds['word_generation'] and
                evaluation_time <= thresholds['evaluation'] and
                exploration_time <= thresholds['exploration']
            )
            
            result = ValidationResult(
                name="Performance Validation",
                success=performance_ok,
                details={
                    'letter_generation_time': letter_generation_time,
                    'word_generation_time': word_generation_time,
                    'evaluation_time': evaluation_time,
                    'exploration_time': exploration_time,
                    'thresholds': thresholds
                }
            )
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Error validating performance: {e}")
            
            result = ValidationResult(
                name="Performance Validation",
                success=False,
                details={'error': str(e)}
            )
            
            self.results.append(result)
            return result
    
    def run_all_validations(self) -> Dict[str, Any]:
        """
        Run all validation tests.
        
        Returns:
            Dictionary with validation results
        """
        # Initialize components
        self.initialize_components()
        
        # Run validations
        self.validate_semantic_database()
        self.validate_equation_generation()
        self.validate_expert_explorer_system()
        self.validate_semantic_guided_exploration()
        self.validate_learning_integration()
        self.validate_end_to_end_workflow()
        self.validate_performance()
        
        # Calculate overall success
        success_count = sum(1 for result in self.results if result.success)
        total_count = len(self.results)
        overall_success = success_count == total_count
        
        # Create summary
        summary = {
            'overall_success': overall_success,
            'success_count': success_count,
            'total_count': total_count,
            'success_rate': success_count / total_count if total_count > 0 else 0.0,
            'results': [result.to_dict() for result in self.results]
        }
        
        # Save summary to file
        summary_path = os.path.join(self.output_dir, 'validation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Validation summary saved to {summary_path}")
        
        return summary
    
    def generate_validation_report(self) -> str:
        """
        Generate a validation report.
        
        Returns:
            Path to the generated report
        """
        # Run validations if not already run
        if not self.results:
            self.run_all_validations()
        
        # Calculate overall success
        success_count = sum(1 for result in self.results if result.success)
        total_count = len(self.results)
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        # Generate report content
        report_content = f"""# Basira System Validation Report

## Summary

- **Overall Success**: {"Yes" if success_count == total_count else "No"}
- **Success Rate**: {success_rate:.2%} ({success_count}/{total_count})
- **Timestamp**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Validation Results

| Test | Status | Details |
|------|--------|---------|
"""
        
        # Add results to report
        for result in self.results:
            status = "✅ PASSED" if result.success else "❌ FAILED"
            details = result.details.get('error', '') if not result.success else ''
            report_content += f"| {result.name} | {status} | {details} |\n"
        
        # Add component details
        report_content += """
## Component Details

### Semantic Database
"""
        
        if self.db_manager and self.db_manager.database:
            db_result = next((r for r in self.results if r.name == "Semantic Database Validation"), None)
            if db_result and db_result.success:
                details = db_result.details
                report_content += f"""
- **Database Size**: {details.get('database_size', 0)} letters
- **Properties Count**: {details.get('properties_count', 0)}
- **Semantic Axes**: {details.get('axes_count', 0)}
- **Semantic Categories**: {details.get('categories_count', 0)}
"""
            else:
                report_content += "\nNo detailed information available.\n"
        else:
            report_content += "\nSemantic database not initialized.\n"
        
        report_content += """
### Equation Generation
"""
        
        eq_result = next((r for r in self.results if r.name == "Equation Generation Validation"), None)
        if eq_result and eq_result.success:
            details = eq_result.details
            letter_equations = details.get('letter_equations', {})
            word_equation = details.get('word_equation', {})
            concept_equation = details.get('concept_equation', {})
            
            report_content += f"""
- **Letter Equations**:
  - A: {letter_equations.get('A', {}).get('components', 0)} components, {letter_equations.get('A', {}).get('semantic_links', 0)} semantic links
  - O: {letter_equations.get('O', {}).get('components', 0)} components, {letter_equations.get('O', {}).get('semantic_links', 0)} semantic links
  - R: {letter_equations.get('R', {}).get('components', 0)} components, {letter_equations.get('R', {}).get('semantic_links', 0)} semantic links
- **Word Equation (TEST)**: {word_equation.get('components', 0)} components, {word_equation.get('semantic_links', 0)} semantic links
- **Concept Equation (Movement)**: {concept_equation.get('components', 0)} components, {concept_equation.get('semantic_links', 0)} semantic links
"""
        else:
            report_content += "\nNo detailed information available.\n"
        
        report_content += """
### Expert-Explorer System
"""
        
        ee_result = next((r for r in self.results if r.name == "Expert-Explorer System Validation"), None)
        if ee_result and ee_result.success:
            details = ee_result.details
            report_content += f"""
- **Expert Evaluation Score**: {details.get('expert_evaluation', 0.0):.4f}
- **Guidance Count**: {details.get('guidance_count', 0)}
- **Exploration Score**: {details.get('exploration_score', 0.0):.4f}
- **Exploration Strategy**: {details.get('exploration_strategy', '')}
"""
        else:
            report_content += "\nNo detailed information available.\n"
        
        report_content += """
### Semantic-Guided Exploration
"""
        
        sg_result = next((r for r in self.results if r.name == "Semantic-Guided Exploration Validation"), None)
        if sg_result and sg_result.success:
            details = sg_result.details
            letter_exploration = details.get('letter_exploration', {})
            word_exploration = details.get('word_exploration', {})
            concept_exploration = details.get('concept_exploration', {})
            
            report_content += f"""
- **Letter Exploration (A)**:
  - Components: {letter_exploration.get('components', 0)}
  - Semantic Links: {letter_exploration.get('semantic_links', 0)}
  - Complexity: {letter_exploration.get('complexity', 0.0):.4f}
- **Word Exploration (TEST)**:
  - Components: {word_exploration.get('components', 0)}
  - Semantic Links: {word_exploration.get('semantic_links', 0)}
  - Complexity: {word_exploration.get('complexity', 0.0):.4f}
- **Concept Exploration (Movement)**:
  - Components: {concept_exploration.get('components', 0)}
  - Semantic Links: {concept_exploration.get('semantic_links', 0)}
  - Complexity: {concept_exploration.get('complexity', 0.0):.4f}
"""
        else:
            report_content += "\nNo detailed information available.\n"
        
        report_content += """
### Learning Integration
"""
        
        li_result = next((r for r in self.results if r.name == "Learning Integration Validation"), None)
        if li_result and li_result.success:
            details = li_result.details
            dl = details.get('deep_learning', {})
            rl = details.get('reinforcement_learning', {})
            
            report_content += f"""
- **Deep Learning**:
  - Enhanced Equation Components: {dl.get('enhanced_equation', {}).get('components', 0)}
  - Neural Components: {dl.get('enhanced_equation', {}).get('neural_components', 0)}
  - Exploration Score: {dl.get('exploration', {}).get('score', 0.0):.4f}
- **Reinforcement Learning**:
  - Best Equation Components: {rl.get('best_equation', {}).get('components', 0)}
  - Best Equation Complexity: {rl.get('best_equation', {}).get('complexity', 0.0):.4f}
  - Exploration Score: {rl.get('exploration', {}).get('score', 0.0):.4f}
"""
        else:
            report_content += "\nNo detailed information available.\n"
        
        report_content += """
### End-to-End Workflow
"""
        
        e2e_result = next((r for r in self.results if r.name == "End-to-End Workflow Validation"), None)
        if e2e_result and e2e_result.success:
            details = e2e_result.details
            initial_eq = details.get('initial_equation', {})
            best_eq = details.get('best_equation', {})
            evaluation = details.get('evaluation', {})
            
            report_content += f"""
- **Initial Equation**:
  - Components: {initial_eq.get('components', 0)}
  - Semantic Links: {initial_eq.get('semantic_links', 0)}
- **Best Equation**:
  - Components: {best_eq.get('components', 0)}
  - Semantic Links: {best_eq.get('semantic_links', 0)}
  - Complexity: {best_eq.get('complexity', 0.0):.4f}
- **Evaluation**:
  - Overall Score: {evaluation.get('overall_score', 0.0):.4f}
  - Semantic Alignment: {evaluation.get('semantic_alignment', 0.0):.4f}
"""
        else:
            report_content += "\nNo detailed information available.\n"
        
        report_content += """
### Performance
"""
        
        perf_result = next((r for r in self.results if r.name == "Performance Validation"), None)
        if perf_result:
            details = perf_result.details
            thresholds = details.get('thresholds', {})
            
            report_content += f"""
- **Letter Generation Time**: {details.get('letter_generation_time', 0.0):.4f}s (Threshold: {thresholds.get('letter_generation', 0.0):.4f}s)
- **Word Generation Time**: {details.get('word_generation_time', 0.0):.4f}s (Threshold: {thresholds.get('word_generation', 0.0):.4f}s)
- **Evaluation Time**: {details.get('evaluation_time', 0.0):.4f}s (Threshold: {thresholds.get('evaluation', 0.0):.4f}s)
- **Exploration Time**: {details.get('exploration_time', 0.0):.4f}s (Threshold: {thresholds.get('exploration', 0.0):.4f}s)
"""
        else:
            report_content += "\nNo detailed information available.\n"
        
        # Add conclusion
        report_content += """
## Conclusion

"""
        if success_count == total_count:
            report_content += """
All validation tests have passed successfully. The Basira System is fully integrated and ready for use.
The system demonstrates robust integration between the mathematical core, learning layers, expert/explorer interaction model, and semantic database.
"""
        else:
            report_content += f"""
{success_count} out of {total_count} validation tests have passed. The following issues need to be addressed:

"""
            for result in self.results:
                if not result.success:
                    error = result.details.get('error', 'Unknown error')
                    report_content += f"- **{result.name}**: {error}\n"
        
        # Save report to file
        report_path = os.path.join(self.output_dir, 'validation_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Validation report saved to {report_path}")
        
        return report_path


def run_system_validation(database_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Run system validation and generate a report.
    
    Args:
        database_path: Path to the semantic database file
        output_dir: Optional directory for output files
        
    Returns:
        Dictionary with validation results
    """
    # Create validator
    validator = SystemValidator(database_path, output_dir)
    
    # Run validations
    summary = validator.run_all_validations()
    
    # Generate report
    report_path = validator.generate_validation_report()
    
    # Print summary
    print("\nSystem Validation Summary:")
    print(f"Overall Success: {'Yes' if summary['overall_success'] else 'No'}")
    print(f"Success Rate: {summary['success_rate']:.2%} ({summary['success_count']}/{summary['total_count']})")
    print(f"Report: {report_path}")
    
    return {
        'summary': summary,
        'report_path': report_path
    }


if __name__ == "__main__":
    # Run system validation
    database_path = "/home/ubuntu/english_letters_extracted.json"
    output_dir = "/home/ubuntu/validation_results"
    
    validation_results = run_system_validation(database_path, output_dir)
