#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-Explorer Interaction Module for Basira System

This module implements the Expert-Explorer interaction model for evolving equations
in the Basira System. It provides a framework for guided exploration of the equation space,
where the Expert component provides guidance and evaluation, while the Explorer component
explores the space and brings back findings.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, Set
import random
import copy
import json
import time
from enum import Enum
from dataclasses import dataclass, field
import logging

# Import from parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced.general_shape_equation import (
    GeneralShapeEquation, EquationType, LearningMode, EquationMetadata,
    GSEFactory, ExpertExplorerSystem
)
from enhanced.learning_integration import (
    EnhancedDeepLearningAdapter, EnhancedReinforcementLearningAdapter,
    EnhancedExpertExplorerSystem, ShapeEquationDataset
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('expert_explorer')


class ExplorationStrategy(str, Enum):
    """Exploration strategies for the Explorer component."""
    RANDOM = "random"  # Random exploration
    GUIDED = "guided"  # Guided by Expert heuristics
    DEEP_LEARNING = "deep_learning"  # Using deep learning
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # Using reinforcement learning
    HYBRID = "hybrid"  # Combination of multiple strategies
    SEMANTIC = "semantic"  # Guided by semantic concepts


class ExpertKnowledgeType(str, Enum):
    """Types of knowledge in the Expert's knowledge base."""
    EQUATION = "equation"  # Complete equation
    PATTERN = "pattern"  # Pattern or template
    HEURISTIC = "heuristic"  # Heuristic rule
    CONSTRAINT = "constraint"  # Constraint or boundary
    SEMANTIC = "semantic"  # Semantic concept


@dataclass
class ExplorationResult:
    """Result of an exploration cycle."""
    equation: GeneralShapeEquation  # The evolved equation
    score: float  # Evaluation score
    strategy: ExplorationStrategy  # Strategy used
    metrics: Dict[str, Any] = field(default_factory=dict)  # Additional metrics
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class ExpertFeedback:
    """Feedback from the Expert to the Explorer."""
    score: float  # Evaluation score
    guidance: List[Dict[str, Any]]  # Guidance for next exploration
    critique: Dict[str, Any] = field(default_factory=dict)  # Critique of current equation
    semantic_alignment: Dict[str, float] = field(default_factory=dict)  # Semantic alignment scores


class AdvancedExpertExplorerSystem:
    """
    Advanced implementation of the Expert-Explorer interaction model.
    
    This system consists of:
    1. Expert: Provides guidance, evaluates results, and maintains knowledge
    2. Explorer: Explores the equation space using various strategies
    3. Interaction Manager: Manages the interaction between Expert and Explorer
    
    The system supports multiple exploration strategies, semantic guidance,
    and integration with both symbolic and learning-based approaches.
    """
    
    def __init__(self, 
                 initial_equation: Optional[GeneralShapeEquation] = None,
                 semantic_db_path: Optional[str] = None,
                 log_level: int = logging.INFO):
        """
        Initialize an AdvancedExpertExplorerSystem.
        
        Args:
            initial_equation: Optional initial equation to start with
            semantic_db_path: Optional path to semantic database
            log_level: Logging level
        """
        # Configure logger
        self.logger = logging.getLogger('expert_explorer')
        self.logger.setLevel(log_level)
        
        # Initialize the Expert component
        self.expert = {
            'knowledge_base': {},  # Store known good equations and patterns
            'heuristics': {},      # Store heuristic rules for guidance
            'evaluation_metrics': {},  # Store metrics for evaluating equations
            'semantic_concepts': {},  # Store semantic concepts
            'feedback_history': []  # Store history of feedback
        }
        
        # Initialize the Explorer component
        self.explorer = {
            'current_equation': initial_equation,
            'exploration_history': [],
            'current_strategy': ExplorationStrategy.GUIDED,
            'learning_rate': 0.1,
            'strategy_performance': {
                strategy: {'success_rate': 0.0, 'avg_improvement': 0.0, 'count': 0}
                for strategy in ExplorationStrategy
            }
        }
        
        # Initialize the Interaction Manager
        self.interaction_manager = {
            'cycles': 0,
            'best_equation': initial_equation,
            'best_score': float('-inf') if initial_equation else 0.0,
            'interaction_history': [],
            'adaptive_strategy': True,  # Whether to adapt strategy based on performance
            'strategy_weights': {
                strategy: 1.0 for strategy in ExplorationStrategy
            }
        }
        
        # Initialize learning components
        self.dl_adapter = EnhancedDeepLearningAdapter()
        self.rl_adapter = EnhancedReinforcementLearningAdapter()
        
        # Load semantic database if provided
        self.semantic_db = {}
        if semantic_db_path:
            self.load_semantic_database(semantic_db_path)
        
        # Initialize basic knowledge and heuristics
        self._initialize_basic_knowledge()
    
    def _initialize_basic_knowledge(self):
        """Initialize the Expert's knowledge base with basic patterns and heuristics."""
        # Add basic shape equations to knowledge base
        self.expert['knowledge_base']['circle'] = {
            'type': ExpertKnowledgeType.EQUATION,
            'equation': GSEFactory.create_basic_shape('circle', cx=0, cy=0, radius=1),
            'description': "Basic circle centered at origin with radius 1",
            'tags': ['basic', 'circle', 'shape']
        }
        
        self.expert['knowledge_base']['rectangle'] = {
            'type': ExpertKnowledgeType.EQUATION,
            'equation': GSEFactory.create_basic_shape('rectangle', x=0, y=0, width=1, height=1),
            'description': "Basic rectangle at origin with width and height 1",
            'tags': ['basic', 'rectangle', 'shape']
        }
        
        self.expert['knowledge_base']['ellipse'] = {
            'type': ExpertKnowledgeType.EQUATION,
            'equation': GSEFactory.create_basic_shape('ellipse', cx=0, cy=0, rx=1, ry=0.5),
            'description': "Basic ellipse centered at origin",
            'tags': ['basic', 'ellipse', 'shape']
        }
        
        # Add basic heuristics
        self.expert['heuristics']['simplify'] = {
            'name': 'simplify',
            'description': "Simplify the equation",
            'function': lambda eq: eq.simplify(),
            'applicability': lambda eq: eq.metadata.complexity > 2.0,
            'tags': ['simplification', 'complexity']
        }
        
        self.expert['heuristics']['mutate_small'] = {
            'name': 'mutate_small',
            'description': "Apply small mutation",
            'function': lambda eq: eq.mutate(0.1),
            'applicability': lambda eq: True,  # Always applicable
            'tags': ['mutation', 'exploration']
        }
        
        self.expert['heuristics']['mutate_medium'] = {
            'name': 'mutate_medium',
            'description': "Apply medium mutation",
            'function': lambda eq: eq.mutate(0.3),
            'applicability': lambda eq: True,  # Always applicable
            'tags': ['mutation', 'exploration']
        }
        
        self.expert['heuristics']['mutate_large'] = {
            'name': 'mutate_large',
            'description': "Apply large mutation",
            'function': lambda eq: eq.mutate(0.5),
            'applicability': lambda eq: True,  # Always applicable
            'tags': ['mutation', 'exploration']
        }
        
        # Add basic evaluation metrics
        self.expert['evaluation_metrics']['complexity'] = {
            'name': 'complexity',
            'description': "Evaluate equation complexity (lower is better)",
            'function': lambda eq: -eq.metadata.complexity,  # Negative because lower complexity is better
            'weight': 0.3,
            'tags': ['complexity', 'simplicity']
        }
        
        self.expert['evaluation_metrics']['component_count'] = {
            'name': 'component_count',
            'description': "Evaluate number of components (moderate is better)",
            'function': lambda eq: self._score_component_count(len(eq.symbolic_components)),
            'weight': 0.2,
            'tags': ['structure', 'components']
        }
        
        self.expert['evaluation_metrics']['variable_usage'] = {
            'name': 'variable_usage',
            'description': "Evaluate variable usage (moderate is better)",
            'function': lambda eq: self._score_variable_count(len(eq.variables)),
            'weight': 0.2,
            'tags': ['variables', 'structure']
        }
        
        self.expert['evaluation_metrics']['semantic_alignment'] = {
            'name': 'semantic_alignment',
            'description': "Evaluate alignment with semantic concepts",
            'function': lambda eq: self._evaluate_semantic_alignment(eq),
            'weight': 0.3,
            'tags': ['semantics', 'meaning']
        }
    
    def _score_component_count(self, count: int) -> float:
        """
        Score the number of components in an equation.
        
        Args:
            count: Number of components
            
        Returns:
            Score (higher is better)
        """
        # Prefer equations with a moderate number of components (3-7)
        if 3 <= count <= 7:
            return 1.0
        elif count < 3:
            return 0.5 * count / 3
        else:  # count > 7
            return 0.5 * (1.0 - min(1.0, (count - 7) / 10))
    
    def _score_variable_count(self, count: int) -> float:
        """
        Score the number of variables in an equation.
        
        Args:
            count: Number of variables
            
        Returns:
            Score (higher is better)
        """
        # Prefer equations with 2-4 variables
        if 2 <= count <= 4:
            return 1.0
        elif count < 2:
            return 0.5 * count / 2
        else:  # count > 4
            return 0.5 * (1.0 - min(1.0, (count - 4) / 6))
    
    def _evaluate_semantic_alignment(self, equation: GeneralShapeEquation) -> float:
        """
        Evaluate how well an equation aligns with semantic concepts.
        
        Args:
            equation: The equation to evaluate
            
        Returns:
            Semantic alignment score (higher is better)
        """
        # If no semantic database, return neutral score
        if not self.semantic_db:
            return 0.5
        
        # Get semantic links from the equation
        equation_semantics = equation.get_semantic_links()
        
        # If no semantic links, return low score
        if not equation_semantics:
            return 0.2
        
        # Calculate alignment score
        total_strength = 0.0
        for link in equation_semantics:
            concept = link.get('concept', '')
            strength = link.get('strength', 0.0)
            
            # Check if concept exists in semantic database
            if concept in self.semantic_db:
                total_strength += strength
        
        # Normalize score
        if len(equation_semantics) > 0:
            return min(1.0, total_strength / len(equation_semantics))
        else:
            return 0.0
    
    def load_semantic_database(self, file_path: str) -> bool:
        """
        Load semantic database from a file.
        
        Args:
            file_path: Path to the semantic database file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.semantic_db = json.load(f)
            
            # Extract semantic concepts
            for key, data in self.semantic_db.items():
                if isinstance(data, dict):
                    # Add as semantic concept
                    self.expert['semantic_concepts'][key] = {
                        'name': key,
                        'properties': data,
                        'tags': data.get('tags', [])
                    }
            
            self.logger.info(f"Loaded semantic database with {len(self.semantic_db)} entries")
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading semantic database: {e}")
            return False
    
    def expert_evaluate(self, equation: GeneralShapeEquation) -> float:
        """
        Expert evaluation of an equation.
        
        Args:
            equation: The equation to evaluate
            
        Returns:
            Evaluation score (higher is better)
        """
        # Apply all evaluation metrics
        scores = {}
        total_weight = 0.0
        
        for metric_name, metric_data in self.expert['evaluation_metrics'].items():
            metric_func = metric_data['function']
            weight = metric_data.get('weight', 1.0)
            
            # Calculate score
            try:
                score = metric_func(equation)
                scores[metric_name] = score
                total_weight += weight
            except Exception as e:
                self.logger.warning(f"Error applying metric {metric_name}: {e}")
        
        # Calculate weighted average
        if total_weight > 0:
            weighted_score = 0.0
            for metric_name, score in scores.items():
                weight = self.expert['evaluation_metrics'][metric_name].get('weight', 1.0)
                weighted_score += score * weight / total_weight
            
            return weighted_score
        else:
            return 0.0
    
    def expert_evaluate_with_target(self, equation: GeneralShapeEquation, 
                                  target_semantics: List[str] = None) -> Dict[str, Any]:
        """
        Expert evaluation with target semantic concepts.
        
        Args:
            equation: The equation to evaluate
            target_semantics: Optional list of target semantic concepts
            
        Returns:
            Dictionary with evaluation results
        """
        # Get base evaluation score
        base_score = self.expert_evaluate(equation)
        
        # Calculate individual metric scores
        metric_scores = {}
        for metric_name, metric_data in self.expert['evaluation_metrics'].items():
            metric_func = metric_data['function']
            try:
                score = metric_func(equation)
                metric_scores[metric_name] = score
            except Exception as e:
                self.logger.warning(f"Error applying metric {metric_name}: {e}")
                metric_scores[metric_name] = 0.0
        
        # Calculate semantic alignment if target semantics provided
        semantic_alignment = {}
        if target_semantics:
            equation_semantics = equation.get_semantic_links()
            
            for target in target_semantics:
                alignment_score = 0.0
                
                for link in equation_semantics:
                    concept = link.get('concept', '')
                    strength = link.get('strength', 0.0)
                    
                    if concept == target:
                        alignment_score = strength
                        break
                
                semantic_alignment[target] = alignment_score
        
        return {
            'overall_score': base_score,
            'metric_scores': metric_scores,
            'semantic_alignment': semantic_alignment
        }
    
    def expert_provide_guidance(self, equation: GeneralShapeEquation) -> List[Dict[str, Any]]:
        """
        Expert guidance for exploration.
        
        Args:
            equation: The current equation
            
        Returns:
            List of guidance items (heuristics to try)
        """
        guidance = []
        
        # Filter heuristics by applicability
        for heuristic_name, heuristic_data in self.expert['heuristics'].items():
            applicability_func = heuristic_data.get('applicability', lambda eq: True)
            
            if applicability_func(equation):
                guidance.append({
                    'name': heuristic_name,
                    'description': heuristic_data.get('description', ''),
                    'tags': heuristic_data.get('tags', [])
                })
        
        return guidance
    
    def expert_provide_feedback(self, equation: GeneralShapeEquation, 
                              target_semantics: List[str] = None) -> ExpertFeedback:
        """
        Expert feedback on an equation.
        
        Args:
            equation: The equation to evaluate
            target_semantics: Optional list of target semantic concepts
            
        Returns:
            ExpertFeedback object
        """
        # Evaluate the equation
        evaluation = self.expert_evaluate_with_target(equation, target_semantics)
        
        # Provide guidance
        guidance = self.expert_provide_guidance(equation)
        
        # Generate critique
        critique = self._generate_critique(equation, evaluation)
        
        # Create feedback
        feedback = ExpertFeedback(
            score=evaluation['overall_score'],
            guidance=guidance,
            critique=critique,
            semantic_alignment=evaluation.get('semantic_alignment', {})
        )
        
        # Record feedback in history
        self.expert['feedback_history'].append({
            'equation_id': equation.metadata.equation_id,
            'feedback': feedback,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return feedback
    
    def _generate_critique(self, equation: GeneralShapeEquation, 
                         evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate critique of an equation.
        
        Args:
            equation: The equation to critique
            evaluation: Evaluation results
            
        Returns:
            Dictionary with critique information
        """
        critique = {
            'strengths': [],
            'weaknesses': [],
            'suggestions': []
        }
        
        # Analyze complexity
        complexity = equation.metadata.complexity
        if complexity < 2.0:
            critique['strengths'].append("Good simplicity")
        elif complexity > 5.0:
            critique['weaknesses'].append("High complexity")
            critique['suggestions'].append("Consider simplifying the equation")
        
        # Analyze component count
        component_count = len(equation.symbolic_components)
        if 3 <= component_count <= 7:
            critique['strengths'].append(f"Good number of components ({component_count})")
        elif component_count < 3:
            critique['weaknesses'].append(f"Too few components ({component_count})")
            critique['suggestions'].append("Consider adding more components")
        else:  # component_count > 7
            critique['weaknesses'].append(f"Too many components ({component_count})")
            critique['suggestions'].append("Consider reducing the number of components")
        
        # Analyze variable usage
        variable_count = len(equation.variables)
        if 2 <= variable_count <= 4:
            critique['strengths'].append(f"Good number of variables ({variable_count})")
        elif variable_count < 2:
            critique['weaknesses'].append(f"Too few variables ({variable_count})")
            critique['suggestions'].append("Consider using more variables")
        else:  # variable_count > 4
            critique['weaknesses'].append(f"Too many variables ({variable_count})")
            critique['suggestions'].append("Consider reducing the number of variables")
        
        # Analyze semantic alignment
        semantic_alignment = evaluation.get('semantic_alignment', {})
        if semantic_alignment:
            aligned_concepts = [concept for concept, score in semantic_alignment.items() if score > 0.5]
            if aligned_concepts:
                critique['strengths'].append(f"Good alignment with concepts: {', '.join(aligned_concepts)}")
            
            unaligned_concepts = [concept for concept, score in semantic_alignment.items() if score <= 0.5]
            if unaligned_concepts:
                critique['weaknesses'].append(f"Poor alignment with concepts: {', '.join(unaligned_concepts)}")
                critique['suggestions'].append("Consider adding semantic links to these concepts")
        
        return critique
    
    def explorer_select_strategy(self) -> ExplorationStrategy:
        """
        Select an exploration strategy based on performance history.
        
        Returns:
            Selected exploration strategy
        """
        if not self.interaction_manager['adaptive_strategy']:
            # Use current strategy if not adaptive
            return self.explorer['current_strategy']
        
        # Calculate strategy weights based on performance
        weights = {}
        for strategy, performance in self.explorer['strategy_performance'].items():
            # Skip strategies with no history
            if performance['count'] == 0:
                weights[strategy] = 1.0
                continue
            
            # Calculate weight based on success rate and average improvement
            success_weight = performance['success_rate'] * 0.7
            improvement_weight = performance['avg_improvement'] * 0.3
            weights[strategy] = success_weight + improvement_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {s: w / total_weight for s, w in weights.items()}
        else:
            normalized_weights = {s: 1.0 / len(weights) for s in weights}
        
        # Store updated weights
        self.interaction_manager['strategy_weights'] = normalized_weights
        
        # Select strategy using weighted random choice
        strategies = list(normalized_weights.keys())
        weights_list = [normalized_weights[s] for s in strategies]
        
        return random.choices(strategies, weights=weights_list, k=1)[0]
    
    def explorer_explore_random(self, steps: int = 10) -> ExplorationResult:
        """
        Random exploration of the equation space.
        
        Args:
            steps: Number of exploration steps
            
        Returns:
            ExplorationResult with the best equation found
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        best_equation = self.explorer['current_equation']
        best_score = self.expert_evaluate(best_equation)
        
        for _ in range(steps):
            # Apply random mutation
            mutation_strength = random.uniform(0.1, 0.5)
            new_equation = best_equation.mutate(mutation_strength)
            
            # Evaluate
            new_score = self.expert_evaluate(new_equation)
            
            # Update best if better
            if new_score > best_score:
                best_equation = new_equation
                best_score = new_score
        
        # Create result
        result = ExplorationResult(
            equation=best_equation,
            score=best_score,
            strategy=ExplorationStrategy.RANDOM,
            metrics={
                'steps': steps,
                'improvement': best_score - self.expert_evaluate(self.explorer['current_equation'])
            }
        )
        
        # Update exploration history
        self.explorer['exploration_history'].append({
            'result': result,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return result
    
    def explorer_explore_guided(self, steps: int = 10) -> ExplorationResult:
        """
        Guided exploration using Expert heuristics.
        
        Args:
            steps: Number of exploration steps
            
        Returns:
            ExplorationResult with the best equation found
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        best_equation = self.explorer['current_equation']
        best_score = self.expert_evaluate(best_equation)
        
        for _ in range(steps):
            # Get guidance from expert
            guidance = self.expert_provide_guidance(best_equation)
            
            # Try each heuristic
            for guidance_item in guidance:
                heuristic_name = guidance_item['name']
                heuristic_data = self.expert['heuristics'].get(heuristic_name)
                
                if heuristic_data and 'function' in heuristic_data:
                    # Apply the heuristic
                    heuristic_func = heuristic_data['function']
                    new_equation = heuristic_func(best_equation)
                    
                    # Evaluate
                    new_score = self.expert_evaluate(new_equation)
                    
                    # Update best if better
                    if new_score > best_score:
                        best_equation = new_equation
                        best_score = new_score
        
        # Create result
        result = ExplorationResult(
            equation=best_equation,
            score=best_score,
            strategy=ExplorationStrategy.GUIDED,
            metrics={
                'steps': steps,
                'improvement': best_score - self.expert_evaluate(self.explorer['current_equation'])
            }
        )
        
        # Update exploration history
        self.explorer['exploration_history'].append({
            'result': result,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return result
    
    def explorer_explore_deep_learning(self, steps: int = 10) -> ExplorationResult:
        """
        Exploration using deep learning.
        
        Args:
            steps: Number of exploration steps
            
        Returns:
            ExplorationResult with the best equation found
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        # Create dataset from current equation
        dataset = ShapeEquationDataset([self.explorer['current_equation']], samples_per_equation=1000)
        
        # Train the deep learning adapter
        self.dl_adapter.train_on_dataset(dataset, num_epochs=min(50, steps * 5))
        
        # Enhance the equation with neural correction
        enhanced_equation = self.dl_adapter.enhance_equation_with_neural_correction(
            self.explorer['current_equation']
        )
        
        # Evaluate the enhanced equation
        enhanced_score = self.expert_evaluate(enhanced_equation)
        
        # Create result
        result = ExplorationResult(
            equation=enhanced_equation,
            score=enhanced_score,
            strategy=ExplorationStrategy.DEEP_LEARNING,
            metrics={
                'steps': steps,
                'improvement': enhanced_score - self.expert_evaluate(self.explorer['current_equation'])
            }
        )
        
        # Update exploration history
        self.explorer['exploration_history'].append({
            'result': result,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return result
    
    def explorer_explore_reinforcement_learning(self, steps: int = 10) -> ExplorationResult:
        """
        Exploration using reinforcement learning.
        
        Args:
            steps: Number of exploration steps
            
        Returns:
            ExplorationResult with the best equation found
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        # Train the RL adapter
        rewards, best_equation = self.rl_adapter.train_on_equation(
            self.explorer['current_equation'],
            num_episodes=min(20, steps * 2),
            steps_per_episode=5
        )
        
        # Evaluate the best equation
        best_score = self.expert_evaluate(best_equation)
        
        # Create result
        result = ExplorationResult(
            equation=best_equation,
            score=best_score,
            strategy=ExplorationStrategy.REINFORCEMENT_LEARNING,
            metrics={
                'steps': steps,
                'improvement': best_score - self.expert_evaluate(self.explorer['current_equation']),
                'rewards': rewards
            }
        )
        
        # Update exploration history
        self.explorer['exploration_history'].append({
            'result': result,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return result
    
    def explorer_explore_semantic(self, steps: int = 10, 
                                target_semantics: List[str] = None) -> ExplorationResult:
        """
        Exploration guided by semantic concepts.
        
        Args:
            steps: Number of exploration steps
            target_semantics: List of target semantic concepts
            
        Returns:
            ExplorationResult with the best equation found
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        if not target_semantics:
            # If no target semantics provided, use random concepts from semantic database
            if self.semantic_db:
                target_semantics = random.sample(list(self.semantic_db.keys()), 
                                               min(3, len(self.semantic_db)))
            else:
                # Fall back to guided exploration if no semantic database
                return self.explorer_explore_guided(steps)
        
        best_equation = self.explorer['current_equation']
        
        # Evaluate with target semantics
        evaluation = self.expert_evaluate_with_target(best_equation, target_semantics)
        best_score = evaluation['overall_score']
        
        for _ in range(steps):
            # Get guidance from expert
            guidance = self.expert_provide_guidance(best_equation)
            
            # Try each heuristic
            for guidance_item in guidance:
                heuristic_name = guidance_item['name']
                heuristic_data = self.expert['heuristics'].get(heuristic_name)
                
                if heuristic_data and 'function' in heuristic_data:
                    # Apply the heuristic
                    heuristic_func = heuristic_data['function']
                    new_equation = heuristic_func(best_equation)
                    
                    # Add semantic links to target concepts
                    for concept in target_semantics:
                        new_equation.add_semantic_link(concept, 'target', 0.8)
                    
                    # Evaluate with target semantics
                    evaluation = self.expert_evaluate_with_target(new_equation, target_semantics)
                    new_score = evaluation['overall_score']
                    
                    # Update best if better
                    if new_score > best_score:
                        best_equation = new_equation
                        best_score = new_score
        
        # Create result
        result = ExplorationResult(
            equation=best_equation,
            score=best_score,
            strategy=ExplorationStrategy.SEMANTIC,
            metrics={
                'steps': steps,
                'improvement': best_score - self.expert_evaluate(self.explorer['current_equation']),
                'target_semantics': target_semantics,
                'semantic_alignment': evaluation.get('semantic_alignment', {})
            }
        )
        
        # Update exploration history
        self.explorer['exploration_history'].append({
            'result': result,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return result
    
    def explorer_explore_hybrid(self, steps: int = 10, 
                              target_semantics: List[str] = None) -> ExplorationResult:
        """
        Hybrid exploration using multiple strategies.
        
        Args:
            steps: Number of exploration steps
            target_semantics: Optional list of target semantic concepts
            
        Returns:
            ExplorationResult with the best equation found
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        # Allocate steps to different strategies
        step_allocation = {
            ExplorationStrategy.GUIDED: max(1, steps // 4),
            ExplorationStrategy.DEEP_LEARNING: max(1, steps // 4),
            ExplorationStrategy.REINFORCEMENT_LEARNING: max(1, steps // 4),
            ExplorationStrategy.SEMANTIC: max(1, steps // 4) if target_semantics else 0
        }
        
        # If no semantic exploration, redistribute steps
        if step_allocation[ExplorationStrategy.SEMANTIC] == 0:
            extra_steps = steps // 3
            step_allocation[ExplorationStrategy.GUIDED] += extra_steps
            step_allocation[ExplorationStrategy.DEEP_LEARNING] += extra_steps
            step_allocation[ExplorationStrategy.REINFORCEMENT_LEARNING] += extra_steps
        
        # Explore using each strategy
        results = []
        
        # Guided exploration
        guided_result = self.explorer_explore_guided(
            steps=step_allocation[ExplorationStrategy.GUIDED]
        )
        results.append(guided_result)
        
        # Deep learning exploration
        dl_result = self.explorer_explore_deep_learning(
            steps=step_allocation[ExplorationStrategy.DEEP_LEARNING]
        )
        results.append(dl_result)
        
        # Reinforcement learning exploration
        rl_result = self.explorer_explore_reinforcement_learning(
            steps=step_allocation[ExplorationStrategy.REINFORCEMENT_LEARNING]
        )
        results.append(rl_result)
        
        # Semantic exploration if applicable
        if step_allocation[ExplorationStrategy.SEMANTIC] > 0:
            semantic_result = self.explorer_explore_semantic(
                steps=step_allocation[ExplorationStrategy.SEMANTIC],
                target_semantics=target_semantics
            )
            results.append(semantic_result)
        
        # Find the best result
        best_result = max(results, key=lambda r: r.score)
        
        # Create hybrid result
        hybrid_result = ExplorationResult(
            equation=best_result.equation,
            score=best_result.score,
            strategy=ExplorationStrategy.HYBRID,
            metrics={
                'steps': steps,
                'improvement': best_result.score - self.expert_evaluate(self.explorer['current_equation']),
                'best_strategy': best_result.strategy,
                'strategy_scores': {r.strategy: r.score for r in results}
            }
        )
        
        # Update exploration history
        self.explorer['exploration_history'].append({
            'result': hybrid_result,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return hybrid_result
    
    def explorer_explore(self, strategy: Optional[ExplorationStrategy] = None, 
                       steps: int = 10, 
                       target_semantics: List[str] = None) -> ExplorationResult:
        """
        Explore the equation space using the specified strategy.
        
        Args:
            strategy: Exploration strategy to use (if None, selects automatically)
            steps: Number of exploration steps
            target_semantics: Optional list of target semantic concepts
            
        Returns:
            ExplorationResult with the best equation found
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        # Select strategy if not specified
        if strategy is None:
            strategy = self.explorer_select_strategy()
        
        # Set current strategy
        self.explorer['current_strategy'] = strategy
        
        # Explore using the selected strategy
        if strategy == ExplorationStrategy.RANDOM:
            result = self.explorer_explore_random(steps)
        elif strategy == ExplorationStrategy.GUIDED:
            result = self.explorer_explore_guided(steps)
        elif strategy == ExplorationStrategy.DEEP_LEARNING:
            result = self.explorer_explore_deep_learning(steps)
        elif strategy == ExplorationStrategy.REINFORCEMENT_LEARNING:
            result = self.explorer_explore_reinforcement_learning(steps)
        elif strategy == ExplorationStrategy.SEMANTIC:
            result = self.explorer_explore_semantic(steps, target_semantics)
        elif strategy == ExplorationStrategy.HYBRID:
            result = self.explorer_explore_hybrid(steps, target_semantics)
        else:
            raise ValueError(f"Unknown exploration strategy: {strategy}")
        
        # Update current equation
        self.explorer['current_equation'] = result.equation
        
        # Update strategy performance
        self._update_strategy_performance(strategy, result)
        
        return result
    
    def _update_strategy_performance(self, strategy: ExplorationStrategy, 
                                   result: ExplorationResult) -> None:
        """
        Update performance metrics for an exploration strategy.
        
        Args:
            strategy: The strategy used
            result: The exploration result
        """
        performance = self.explorer['strategy_performance'][strategy]
        
        # Get improvement
        improvement = result.metrics.get('improvement', 0.0)
        
        # Update count
        performance['count'] += 1
        
        # Update success rate (success = positive improvement)
        success = improvement > 0
        old_success_rate = performance['success_rate']
        performance['success_rate'] = (old_success_rate * (performance['count'] - 1) + (1.0 if success else 0.0)) / performance['count']
        
        # Update average improvement
        old_avg_improvement = performance['avg_improvement']
        performance['avg_improvement'] = (old_avg_improvement * (performance['count'] - 1) + improvement) / performance['count']
    
    def expert_explorer_interaction(self, cycles: int = 5, 
                                  steps_per_cycle: int = 10,
                                  target_semantics: List[str] = None) -> GeneralShapeEquation:
        """
        Run a full expert-explorer interaction cycle.
        
        Args:
            cycles: Number of interaction cycles
            steps_per_cycle: Steps per exploration cycle
            target_semantics: Optional list of target semantic concepts
            
        Returns:
            Best equation found during all cycles
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        # Initialize best equation
        best_overall_equation = self.explorer['current_equation']
        best_overall_score = self.expert_evaluate(best_overall_equation)
        
        # Update interaction manager
        self.interaction_manager['best_equation'] = best_overall_equation
        self.interaction_manager['best_score'] = best_overall_score
        
        # Run interaction cycles
        for cycle in range(cycles):
            self.logger.info(f"Starting interaction cycle {cycle+1}/{cycles}")
            
            # 1. Explorer explores
            exploration_result = self.explorer_explore(
                steps=steps_per_cycle,
                target_semantics=target_semantics
            )
            
            # 2. Expert evaluates and provides feedback
            feedback = self.expert_provide_feedback(
                exploration_result.equation,
                target_semantics
            )
            
            # 3. Record interaction
            interaction = {
                'cycle': cycle,
                'exploration_result': exploration_result,
                'feedback': feedback,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.interaction_manager['interaction_history'].append(interaction)
            self.interaction_manager['cycles'] += 1
            
            # 4. Update best overall if better
            if exploration_result.score > best_overall_score:
                best_overall_equation = exploration_result.equation
                best_overall_score = exploration_result.score
                
                # Update interaction manager
                self.interaction_manager['best_equation'] = best_overall_equation
                self.interaction_manager['best_score'] = best_overall_score
            
            # Log progress
            self.logger.info(f"Cycle {cycle+1}/{cycles}, Score: {exploration_result.score:.4f}, Best: {best_overall_score:.4f}")
        
        return best_overall_equation
    
    def expert_explorer_interaction_with_visualization(self, cycles: int = 5, 
                                                    steps_per_cycle: int = 10,
                                                    target_semantics: List[str] = None,
                                                    visualize: bool = True,
                                                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a full expert-explorer interaction cycle with visualization.
        
        Args:
            cycles: Number of interaction cycles
            steps_per_cycle: Steps per exploration cycle
            target_semantics: Optional list of target semantic concepts
            visualize: Whether to generate visualizations
            save_path: Optional path to save visualizations
            
        Returns:
            Dictionary with results and visualizations
        """
        # Run interaction
        best_equation = self.expert_explorer_interaction(
            cycles=cycles,
            steps_per_cycle=steps_per_cycle,
            target_semantics=target_semantics
        )
        
        results = {
            'best_equation': best_equation,
            'best_score': self.expert_evaluate(best_equation),
            'cycles': cycles,
            'steps_per_cycle': steps_per_cycle,
            'target_semantics': target_semantics,
            'visualizations': {}
        }
        
        # Generate visualizations if requested
        if visualize:
            # Extract data for visualization
            cycle_numbers = list(range(1, cycles + 1))
            scores = [interaction['exploration_result'].score 
                     for interaction in self.interaction_manager['interaction_history'][-cycles:]]
            
            strategies = [interaction['exploration_result'].strategy 
                         for interaction in self.interaction_manager['interaction_history'][-cycles:]]
            
            improvements = [interaction['exploration_result'].metrics.get('improvement', 0.0) 
                           for interaction in self.interaction_manager['interaction_history'][-cycles:]]
            
            # Create score progression visualization
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(cycle_numbers, scores, 'b-o', label='Score')
            ax1.set_title('Equation Score Progression')
            ax1.set_xlabel('Cycle')
            ax1.set_ylabel('Score')
            ax1.grid(True)
            
            # Add best score line
            ax1.axhline(y=results['best_score'], color='r', linestyle='--', label='Best Score')
            ax1.legend()
            
            results['visualizations']['score_progression'] = fig1
            
            # Create strategy performance visualization
            strategy_performance = {}
            for strategy in ExplorationStrategy:
                performance = self.explorer['strategy_performance'][strategy]
                if performance['count'] > 0:
                    strategy_performance[strategy] = {
                        'success_rate': performance['success_rate'],
                        'avg_improvement': performance['avg_improvement'],
                        'count': performance['count']
                    }
            
            if strategy_performance:
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                
                strategies_list = list(strategy_performance.keys())
                success_rates = [strategy_performance[s]['success_rate'] for s in strategies_list]
                avg_improvements = [strategy_performance[s]['avg_improvement'] for s in strategies_list]
                counts = [strategy_performance[s]['count'] for s in strategies_list]
                
                x = np.arange(len(strategies_list))
                width = 0.35
                
                ax2.bar(x - width/2, success_rates, width, label='Success Rate')
                ax2.bar(x + width/2, [max(0, imp) for imp in avg_improvements], width, label='Avg Improvement')
                
                ax2.set_title('Strategy Performance')
                ax2.set_xlabel('Strategy')
                ax2.set_ylabel('Performance Metric')
                ax2.set_xticks(x)
                ax2.set_xticklabels([s.value for s in strategies_list])
                ax2.legend()
                
                # Add count annotations
                for i, count in enumerate(counts):
                    ax2.annotate(f'n={count}', xy=(i, 0.05), ha='center')
                
                results['visualizations']['strategy_performance'] = fig2
            
            # Save visualizations if requested
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                
                if 'score_progression' in results['visualizations']:
                    fig1.savefig(os.path.join(save_path, 'score_progression.png'), dpi=300, bbox_inches='tight')
                
                if 'strategy_performance' in results['visualizations']:
                    fig2.savefig(os.path.join(save_path, 'strategy_performance.png'), dpi=300, bbox_inches='tight')
        
        return results
    
    def set_initial_equation(self, equation: GeneralShapeEquation) -> None:
        """
        Set the initial equation for exploration.
        
        Args:
            equation: The initial equation
        """
        self.explorer['current_equation'] = equation
        self.explorer['exploration_history'] = []
        
        # Reset interaction manager
        self.interaction_manager['best_equation'] = equation
        self.interaction_manager['best_score'] = self.expert_evaluate(equation)
    
    def get_best_equation(self) -> Optional[GeneralShapeEquation]:
        """
        Get the best equation found so far.
        
        Returns:
            Best equation, or None if no exploration has been done
        """
        return self.interaction_manager['best_equation']
    
    def add_to_expert_knowledge(self, name: str, equation: GeneralShapeEquation, 
                              knowledge_type: ExpertKnowledgeType = ExpertKnowledgeType.EQUATION,
                              description: str = "", tags: List[str] = None) -> None:
        """
        Add an equation to the Expert's knowledge base.
        
        Args:
            name: Name for the knowledge item
            equation: The equation to add
            knowledge_type: Type of knowledge
            description: Description of the knowledge item
            tags: Optional list of tags
        """
        self.expert['knowledge_base'][name] = {
            'type': knowledge_type,
            'equation': equation,
            'description': description,
            'tags': tags or []
        }
        
        self.logger.info(f"Added {knowledge_type.value} '{name}' to Expert knowledge base")
    
    def add_heuristic(self, name: str, function: Callable, 
                    description: str = "", 
                    applicability: Callable = None,
                    tags: List[str] = None) -> None:
        """
        Add a heuristic to the Expert's heuristics.
        
        Args:
            name: Name for the heuristic
            function: Function implementing the heuristic
            description: Description of the heuristic
            applicability: Function determining when the heuristic is applicable
            tags: Optional list of tags
        """
        self.expert['heuristics'][name] = {
            'name': name,
            'description': description,
            'function': function,
            'applicability': applicability or (lambda eq: True),
            'tags': tags or []
        }
        
        self.logger.info(f"Added heuristic '{name}' to Expert heuristics")
    
    def add_evaluation_metric(self, name: str, function: Callable, 
                            description: str = "", 
                            weight: float = 1.0,
                            tags: List[str] = None) -> None:
        """
        Add an evaluation metric to the Expert's evaluation metrics.
        
        Args:
            name: Name for the metric
            function: Function implementing the metric
            description: Description of the metric
            weight: Weight of the metric in overall evaluation
            tags: Optional list of tags
        """
        self.expert['evaluation_metrics'][name] = {
            'name': name,
            'description': description,
            'function': function,
            'weight': weight,
            'tags': tags or []
        }
        
        self.logger.info(f"Added evaluation metric '{name}' to Expert evaluation metrics")
    
    def save_system_state(self, file_path: str) -> None:
        """
        Save the current state of the expert-explorer system.
        
        Args:
            file_path: Path to save the system state
        """
        # Create a dictionary of the system state
        system_state = {
            'expert': {
                'knowledge_base': {
                    name: {
                        'type': item['type'].value,
                        'equation': item['equation'].to_dict(),
                        'description': item['description'],
                        'tags': item['tags']
                    }
                    for name, item in self.expert['knowledge_base'].items()
                },
                'semantic_concepts': self.expert['semantic_concepts'],
                'feedback_history': [
                    {
                        'equation_id': item['equation_id'],
                        'feedback': {
                            'score': item['feedback'].score,
                            'guidance': item['feedback'].guidance,
                            'critique': item['feedback'].critique,
                            'semantic_alignment': item['feedback'].semantic_alignment
                        },
                        'timestamp': item['timestamp']
                    }
                    for item in self.expert['feedback_history']
                ]
            },
            'explorer': {
                'current_equation': self.explorer['current_equation'].to_dict() if self.explorer['current_equation'] else None,
                'current_strategy': self.explorer['current_strategy'].value,
                'learning_rate': self.explorer['learning_rate'],
                'strategy_performance': {
                    strategy.value: performance
                    for strategy, performance in self.explorer['strategy_performance'].items()
                }
            },
            'interaction_manager': {
                'cycles': self.interaction_manager['cycles'],
                'best_equation': self.interaction_manager['best_equation'].to_dict() if self.interaction_manager['best_equation'] else None,
                'best_score': self.interaction_manager['best_score'],
                'adaptive_strategy': self.interaction_manager['adaptive_strategy'],
                'strategy_weights': {
                    strategy.value: weight
                    for strategy, weight in self.interaction_manager['strategy_weights'].items()
                }
            }
        }
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(system_state, f, indent=2)
        
        self.logger.info(f"Saved system state to {file_path}")
    
    def load_system_state(self, file_path: str) -> bool:
        """
        Load the state of the expert-explorer system from a file.
        
        Args:
            file_path: Path to the system state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load from file
            with open(file_path, 'r', encoding='utf-8') as f:
                system_state = json.load(f)
            
            # Restore expert knowledge base
            self.expert['knowledge_base'] = {}
            for name, item in system_state['expert']['knowledge_base'].items():
                self.expert['knowledge_base'][name] = {
                    'type': ExpertKnowledgeType(item['type']),
                    'equation': GeneralShapeEquation.from_dict(item['equation']),
                    'description': item['description'],
                    'tags': item['tags']
                }
            
            # Restore semantic concepts
            self.expert['semantic_concepts'] = system_state['expert'].get('semantic_concepts', {})
            
            # Restore feedback history
            self.expert['feedback_history'] = []
            for item in system_state['expert'].get('feedback_history', []):
                feedback = ExpertFeedback(
                    score=item['feedback']['score'],
                    guidance=item['feedback']['guidance'],
                    critique=item['feedback']['critique'],
                    semantic_alignment=item['feedback']['semantic_alignment']
                )
                
                self.expert['feedback_history'].append({
                    'equation_id': item['equation_id'],
                    'feedback': feedback,
                    'timestamp': item['timestamp']
                })
            
            # Restore explorer state
            if system_state['explorer'].get('current_equation'):
                self.explorer['current_equation'] = GeneralShapeEquation.from_dict(
                    system_state['explorer']['current_equation']
                )
            
            self.explorer['current_strategy'] = ExplorationStrategy(
                system_state['explorer']['current_strategy']
            )
            
            self.explorer['learning_rate'] = system_state['explorer']['learning_rate']
            
            # Restore strategy performance
            self.explorer['strategy_performance'] = {}
            for strategy_str, performance in system_state['explorer'].get('strategy_performance', {}).items():
                self.explorer['strategy_performance'][ExplorationStrategy(strategy_str)] = performance
            
            # Restore interaction manager
            self.interaction_manager['cycles'] = system_state['interaction_manager']['cycles']
            
            if system_state['interaction_manager'].get('best_equation'):
                self.interaction_manager['best_equation'] = GeneralShapeEquation.from_dict(
                    system_state['interaction_manager']['best_equation']
                )
            
            self.interaction_manager['best_score'] = system_state['interaction_manager']['best_score']
            self.interaction_manager['adaptive_strategy'] = system_state['interaction_manager']['adaptive_strategy']
            
            # Restore strategy weights
            self.interaction_manager['strategy_weights'] = {}
            for strategy_str, weight in system_state['interaction_manager'].get('strategy_weights', {}).items():
                self.interaction_manager['strategy_weights'][ExplorationStrategy(strategy_str)] = weight
            
            self.logger.info(f"Loaded system state from {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading system state: {e}")
            return False


# Utility functions for testing and demonstration

def create_test_system() -> AdvancedExpertExplorerSystem:
    """
    Create a test expert-explorer system with a basic equation.
    
    Returns:
        Initialized AdvancedExpertExplorerSystem
    """
    # Create a basic circle equation
    circle_equation = GSEFactory.create_basic_shape('circle', cx=0, cy=0, radius=1)
    
    # Create the system
    system = AdvancedExpertExplorerSystem(initial_equation=circle_equation)
    
    return system


def run_demonstration(cycles: int = 3, steps_per_cycle: int = 5) -> Dict[str, Any]:
    """
    Run a demonstration of the expert-explorer system.
    
    Args:
        cycles: Number of interaction cycles
        steps_per_cycle: Steps per exploration cycle
        
    Returns:
        Dictionary with demonstration results
    """
    # Create test system
    system = create_test_system()
    
    # Run interaction with visualization
    results = system.expert_explorer_interaction_with_visualization(
        cycles=cycles,
        steps_per_cycle=steps_per_cycle,
        visualize=True
    )
    
    # Print summary
    print("\nDemonstration Summary:")
    print(f"Ran {cycles} interaction cycles with {steps_per_cycle} steps per cycle")
    print(f"Best equation score: {results['best_score']:.4f}")
    print(f"Best equation complexity: {results['best_equation'].metadata.complexity:.4f}")
    print(f"Best equation components: {len(results['best_equation'].symbolic_components)}")
    print(f"Best equation variables: {len(results['best_equation'].variables)}")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    demo_results = run_demonstration(cycles=5, steps_per_cycle=10)
    
    # Show visualizations
    if 'visualizations' in demo_results:
        for name, fig in demo_results['visualizations'].items():
            plt.figure(fig.number)
            plt.show()
