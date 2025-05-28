#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Component for General Shape Equation

This module implements the reasoning component for the General Shape Equation,
which adds support for deep thinking, logical reasoning, and inference.

Author: Baserah System Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import from core module
try:
    from core.general_shape_equation import (
        GeneralShapeEquation,
        EquationType,
        LearningMode,
        SymbolicExpression
    )
except ImportError:
    logging.error("Failed to import from core.general_shape_equation")
    sys.exit(1)

# Configure logging
logger = logging.getLogger('core.reasoning_component')


class ReasoningType(str, Enum):
    """Types of reasoning supported by the reasoning component."""
    DEDUCTIVE = "deductive"        # Deductive reasoning (from general to specific)
    INDUCTIVE = "inductive"        # Inductive reasoning (from specific to general)
    ABDUCTIVE = "abductive"        # Abductive reasoning (inference to best explanation)
    ANALOGICAL = "analogical"      # Analogical reasoning (based on similarities)
    CAUSAL = "causal"              # Causal reasoning (cause and effect)
    COUNTERFACTUAL = "counterfactual"  # Counterfactual reasoning (what if)
    SPATIAL = "spatial"            # Spatial reasoning (about space and objects)
    TEMPORAL = "temporal"          # Temporal reasoning (about time and events)
    MATHEMATICAL = "mathematical"  # Mathematical reasoning (formal logic)
    COMMON_SENSE = "common_sense"  # Common sense reasoning


@dataclass
class InferenceStep:
    """A single step in a chain of reasoning."""
    step_id: str
    description: str
    premises: List[str]
    conclusion: str
    confidence: float = 1.0
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    symbolic_expression: Optional[SymbolicExpression] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """A chain of reasoning steps."""
    chain_id: str
    description: str
    steps: List[InferenceStep] = field(default_factory=list)
    final_conclusion: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningComponent:
    """
    Reasoning Component for the General Shape Equation.
    
    This class extends the General Shape Equation with reasoning capabilities,
    including deep thinking, logical reasoning, and inference.
    """
    
    def __init__(self, equation: GeneralShapeEquation):
        """
        Initialize the reasoning component.
        
        Args:
            equation: The General Shape Equation to extend
        """
        self.equation = equation
        self.reasoning_chains = {}
        self.knowledge_base = {}
        
        # Add reasoning components to the equation
        self._initialize_reasoning_components()
    
    def _initialize_reasoning_components(self) -> None:
        """Initialize the reasoning components in the equation."""
        # Add basic reasoning components
        self.equation.add_component("reasoning_mode", "deductive")
        self.equation.add_component("inference_confidence", "1.0")
        self.equation.add_component("chain_of_thought", "premises -> intermediate_steps -> conclusion")
        
        # Add components for different reasoning types
        self.equation.add_component("deductive_reasoning", "general_rule AND specific_case -> conclusion")
        self.equation.add_component("inductive_reasoning", "specific_cases -> general_rule")
        self.equation.add_component("abductive_reasoning", "observation AND explanation -> best_explanation")
        self.equation.add_component("analogical_reasoning", "source_domain AND target_domain AND mapping -> conclusion")
    
    def create_reasoning_chain(self, description: str) -> str:
        """
        Create a new reasoning chain.
        
        Args:
            description: Description of the reasoning chain
            
        Returns:
            ID of the created reasoning chain
        """
        chain_id = f"chain_{int(time.time())}_{len(self.reasoning_chains)}"
        
        chain = ReasoningChain(
            chain_id=chain_id,
            description=description
        )
        
        self.reasoning_chains[chain_id] = chain
        
        return chain_id
    
    def add_inference_step(self, chain_id: str, description: str, premises: List[str], 
                          conclusion: str, reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE,
                          confidence: float = 1.0) -> str:
        """
        Add an inference step to a reasoning chain.
        
        Args:
            chain_id: ID of the reasoning chain
            description: Description of the inference step
            premises: List of premises
            conclusion: Conclusion drawn from the premises
            reasoning_type: Type of reasoning used
            confidence: Confidence in the inference (0.0 to 1.0)
            
        Returns:
            ID of the created inference step
        """
        if chain_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain with ID {chain_id} not found")
        
        chain = self.reasoning_chains[chain_id]
        
        step_id = f"step_{len(chain.steps) + 1}"
        
        # Create symbolic expression for the inference
        symbolic_expr = self._create_symbolic_expression(premises, conclusion, reasoning_type)
        
        step = InferenceStep(
            step_id=step_id,
            description=description,
            premises=premises,
            conclusion=conclusion,
            confidence=confidence,
            reasoning_type=reasoning_type,
            symbolic_expression=symbolic_expr
        )
        
        chain.steps.append(step)
        
        # Update the chain's confidence as the product of all step confidences
        chain.confidence = np.prod([s.confidence for s in chain.steps])
        
        # If this is the first step, add the premises to the knowledge base
        if len(chain.steps) == 1:
            for premise in premises:
                self._add_to_knowledge_base(premise, "premise", confidence)
        
        # Add the conclusion to the knowledge base
        self._add_to_knowledge_base(conclusion, "conclusion", confidence)
        
        return step_id
    
    def _create_symbolic_expression(self, premises: List[str], conclusion: str, 
                                   reasoning_type: ReasoningType) -> SymbolicExpression:
        """
        Create a symbolic expression for an inference.
        
        Args:
            premises: List of premises
            conclusion: Conclusion drawn from the premises
            reasoning_type: Type of reasoning used
            
        Returns:
            Symbolic expression representing the inference
        """
        # Create a symbolic expression based on the reasoning type
        if reasoning_type == ReasoningType.DEDUCTIVE:
            # For deductive reasoning: P1 AND P2 AND ... AND Pn -> C
            expr_str = " AND ".join([f"({p})" for p in premises]) + f" -> ({conclusion})"
        
        elif reasoning_type == ReasoningType.INDUCTIVE:
            # For inductive reasoning: (P1 AND P2 AND ... AND Pn) -> PROBABLY C
            expr_str = "(" + " AND ".join([f"({p})" for p in premises]) + f") -> PROBABLY ({conclusion})"
        
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            # For abductive reasoning: C AND (C -> P1) AND (C -> P2) AND ... AND (C -> Pn) -> BEST_EXPLANATION(C)
            expr_str = f"({conclusion}) AND " + " AND ".join([f"(({conclusion}) -> ({p}))" for p in premises]) + f" -> BEST_EXPLANATION({conclusion})"
        
        elif reasoning_type == ReasoningType.ANALOGICAL:
            # For analogical reasoning: (SOURCE_DOMAIN AND TARGET_DOMAIN AND MAPPING) -> C
            # Simplified for now
            expr_str = f"ANALOGICAL({', '.join(premises)}) -> ({conclusion})"
        
        else:
            # Default expression
            expr_str = " AND ".join([f"({p})" for p in premises]) + f" -> ({conclusion})"
        
        return SymbolicExpression(expr_str)
    
    def _add_to_knowledge_base(self, statement: str, statement_type: str, confidence: float) -> None:
        """
        Add a statement to the knowledge base.
        
        Args:
            statement: The statement to add
            statement_type: Type of the statement (premise, conclusion, etc.)
            confidence: Confidence in the statement (0.0 to 1.0)
        """
        if statement in self.knowledge_base:
            # Update confidence if the new confidence is higher
            if confidence > self.knowledge_base[statement]["confidence"]:
                self.knowledge_base[statement]["confidence"] = confidence
                self.knowledge_base[statement]["types"].add(statement_type)
        else:
            self.knowledge_base[statement] = {
                "confidence": confidence,
                "types": {statement_type},
                "timestamp": time.time()
            }
    
    def finalize_reasoning_chain(self, chain_id: str, final_conclusion: str, confidence: Optional[float] = None) -> None:
        """
        Finalize a reasoning chain with a final conclusion.
        
        Args:
            chain_id: ID of the reasoning chain
            final_conclusion: Final conclusion of the reasoning chain
            confidence: Confidence in the final conclusion (optional)
        """
        if chain_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain with ID {chain_id} not found")
        
        chain = self.reasoning_chains[chain_id]
        
        chain.final_conclusion = final_conclusion
        
        if confidence is not None:
            chain.confidence = confidence
        
        # Add the final conclusion to the knowledge base
        self._add_to_knowledge_base(final_conclusion, "final_conclusion", chain.confidence)
    
    def evaluate_premise(self, premise: str, variable_values: Dict[str, Any] = None) -> float:
        """
        Evaluate a premise using the equation.
        
        Args:
            premise: The premise to evaluate
            variable_values: Dictionary mapping variable names to values (optional)
            
        Returns:
            Evaluation result (confidence or truth value)
        """
        # Add the premise as a component to the equation
        component_name = f"premise_{int(time.time())}"
        self.equation.add_component(component_name, premise)
        
        # Evaluate the component
        result = self.equation.evaluate(variable_values)
        
        # Remove the temporary component
        self.equation.remove_component(component_name)
        
        # Return the evaluation result
        if component_name in result:
            return result[component_name]
        else:
            return 0.0
    
    def chain_of_thought(self, question: str, variable_values: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform chain-of-thought reasoning to answer a question.
        
        Args:
            question: The question to answer
            variable_values: Dictionary mapping variable names to values (optional)
            
        Returns:
            Dictionary containing the reasoning chain and answer
        """
        # Create a new reasoning chain
        chain_id = self.create_reasoning_chain(f"Chain of thought for: {question}")
        
        # Extract relevant knowledge from the knowledge base
        relevant_knowledge = self._retrieve_relevant_knowledge(question)
        
        # Add initial premises from relevant knowledge
        premises = [k for k, v in relevant_knowledge.items() if "premise" in v["types"]]
        
        if not premises:
            # If no relevant premises found, create a default premise
            premises = [f"Question: {question}"]
        
        # Add the first inference step
        self.add_inference_step(
            chain_id=chain_id,
            description="Initial analysis",
            premises=premises,
            conclusion=f"Analyzing question: {question}",
            reasoning_type=ReasoningType.DEDUCTIVE,
            confidence=0.9
        )
        
        # Perform intermediate reasoning steps (placeholder)
        # In a real implementation, this would involve more sophisticated reasoning
        intermediate_conclusion = f"Intermediate analysis of {question}"
        
        self.add_inference_step(
            chain_id=chain_id,
            description="Intermediate reasoning",
            premises=[f"Analyzing question: {question}"],
            conclusion=intermediate_conclusion,
            reasoning_type=ReasoningType.DEDUCTIVE,
            confidence=0.8
        )
        
        # Formulate the final answer (placeholder)
        # In a real implementation, this would be derived from the reasoning chain
        answer = f"Answer to {question} based on chain-of-thought reasoning"
        
        self.add_inference_step(
            chain_id=chain_id,
            description="Final reasoning",
            premises=[intermediate_conclusion],
            conclusion=answer,
            reasoning_type=ReasoningType.DEDUCTIVE,
            confidence=0.7
        )
        
        # Finalize the reasoning chain
        self.finalize_reasoning_chain(chain_id, answer)
        
        # Return the reasoning chain and answer
        return {
            "question": question,
            "reasoning_chain": self.reasoning_chains[chain_id],
            "answer": answer,
            "confidence": self.reasoning_chains[chain_id].confidence
        }
    
    def _retrieve_relevant_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge from the knowledge base.
        
        Args:
            query: The query to retrieve knowledge for
            
        Returns:
            Dictionary of relevant knowledge
        """
        # Placeholder implementation
        # In a real implementation, this would use more sophisticated retrieval methods
        relevant_knowledge = {}
        
        # Simple keyword matching
        query_words = set(query.lower().split())
        
        for statement, info in self.knowledge_base.items():
            statement_words = set(statement.lower().split())
            
            # Check for word overlap
            if query_words.intersection(statement_words):
                relevant_knowledge[statement] = info
        
        return relevant_knowledge
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the reasoning component to a dictionary.
        
        Returns:
            Dictionary representation of the reasoning component
        """
        return {
            "reasoning_chains": {chain_id: self._chain_to_dict(chain) for chain_id, chain in self.reasoning_chains.items()},
            "knowledge_base": self.knowledge_base
        }
    
    def _chain_to_dict(self, chain: ReasoningChain) -> Dict[str, Any]:
        """
        Convert a reasoning chain to a dictionary.
        
        Args:
            chain: The reasoning chain to convert
            
        Returns:
            Dictionary representation of the reasoning chain
        """
        return {
            "chain_id": chain.chain_id,
            "description": chain.description,
            "steps": [self._step_to_dict(step) for step in chain.steps],
            "final_conclusion": chain.final_conclusion,
            "confidence": chain.confidence,
            "metadata": chain.metadata
        }
    
    def _step_to_dict(self, step: InferenceStep) -> Dict[str, Any]:
        """
        Convert an inference step to a dictionary.
        
        Args:
            step: The inference step to convert
            
        Returns:
            Dictionary representation of the inference step
        """
        return {
            "step_id": step.step_id,
            "description": step.description,
            "premises": step.premises,
            "conclusion": step.conclusion,
            "confidence": step.confidence,
            "reasoning_type": step.reasoning_type.value,
            "symbolic_expression": step.symbolic_expression.to_string() if step.symbolic_expression else None,
            "metadata": step.metadata
        }
    
    def to_json(self) -> str:
        """
        Convert the reasoning component to a JSON string.
        
        Returns:
            JSON string representation of the reasoning component
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], equation: GeneralShapeEquation) -> 'ReasoningComponent':
        """
        Create a reasoning component from a dictionary.
        
        Args:
            data: Dictionary representation of the reasoning component
            equation: The General Shape Equation to extend
            
        Returns:
            Reasoning component instance
        """
        component = cls(equation)
        
        # Load knowledge base
        component.knowledge_base = data.get("knowledge_base", {})
        
        # Load reasoning chains
        for chain_id, chain_data in data.get("reasoning_chains", {}).items():
            chain = ReasoningChain(
                chain_id=chain_data["chain_id"],
                description=chain_data["description"],
                final_conclusion=chain_data.get("final_conclusion"),
                confidence=chain_data.get("confidence", 1.0),
                metadata=chain_data.get("metadata", {})
            )
            
            # Load steps
            for step_data in chain_data.get("steps", []):
                step = InferenceStep(
                    step_id=step_data["step_id"],
                    description=step_data["description"],
                    premises=step_data["premises"],
                    conclusion=step_data["conclusion"],
                    confidence=step_data.get("confidence", 1.0),
                    reasoning_type=ReasoningType(step_data.get("reasoning_type", "deductive")),
                    symbolic_expression=SymbolicExpression(step_data.get("symbolic_expression", "")) if step_data.get("symbolic_expression") else None,
                    metadata=step_data.get("metadata", {})
                )
                
                chain.steps.append(step)
            
            component.reasoning_chains[chain_id] = chain
        
        return component
    
    @classmethod
    def from_json(cls, json_str: str, equation: GeneralShapeEquation) -> 'ReasoningComponent':
        """
        Create a reasoning component from a JSON string.
        
        Args:
            json_str: JSON string representation of the reasoning component
            equation: The General Shape Equation to extend
            
        Returns:
            Reasoning component instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data, equation)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a General Shape Equation
    equation = GeneralShapeEquation(
        equation_type=EquationType.REASONING,
        learning_mode=LearningMode.HYBRID
    )
    
    # Create a reasoning component
    reasoning = ReasoningComponent(equation)
    
    # Perform chain-of-thought reasoning
    result = reasoning.chain_of_thought("What is the relationship between energy and mass?")
    
    print("Question:", result["question"])
    print("Answer:", result["answer"])
    print("Confidence:", result["confidence"])
    print("Reasoning Chain:")
    for step in result["reasoning_chain"].steps:
        print(f"  Step {step.step_id}: {step.description}")
        print(f"    Premises: {step.premises}")
        print(f"    Conclusion: {step.conclusion}")
        print(f"    Confidence: {step.confidence}")
    
    # Convert to JSON and back
    json_str = reasoning.to_json()
    print("\nJSON representation:", json_str)
    
    reasoning2 = ReasoningComponent.from_json(json_str, equation)
    print("\nReconstructed reasoning component:", reasoning2.reasoning_chains.keys())
