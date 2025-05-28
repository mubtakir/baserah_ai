#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intrinsic Reward Component for General Shape Equation

This module implements the intrinsic reward component for the General Shape Equation,
which adds support for curiosity-driven exploration and intrinsic motivation.

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

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch not available, some functionality will be limited")
    TORCH_AVAILABLE = False

# Configure logging
logger = logging.getLogger('core.intrinsic_reward')


class RewardType(str, Enum):
    """Types of intrinsic rewards."""
    NOVELTY = "novelty"            # Reward for novel states or experiences
    SURPRISE = "surprise"          # Reward for surprising or unexpected outcomes
    CURIOSITY = "curiosity"        # Reward for satisfying curiosity
    COMPETENCE = "competence"      # Reward for improving skills or competence
    MASTERY = "mastery"            # Reward for mastering a task or domain
    EXPLORATION = "exploration"    # Reward for exploring the environment
    DISCOVERY = "discovery"        # Reward for discovering new knowledge
    CREATIVITY = "creativity"      # Reward for creative or innovative solutions
    AUTONOMY = "autonomy"          # Reward for autonomous decision-making
    COMPLEXITY = "complexity"      # Reward for handling complex situations


@dataclass
class RewardSignal:
    """A reward signal in the intrinsic reward system."""
    value: float
    reward_type: RewardType
    source: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardHistory:
    """History of reward signals for a particular entity."""
    entity_id: str
    signals: List[RewardSignal] = field(default_factory=list)
    total_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntrinsicRewardComponent:
    """
    Intrinsic Reward Component for the General Shape Equation.
    
    This class extends the General Shape Equation with intrinsic reward capabilities,
    including curiosity-driven exploration and intrinsic motivation.
    """
    
    def __init__(self, equation: GeneralShapeEquation):
        """
        Initialize the intrinsic reward component.
        
        Args:
            equation: The General Shape Equation to extend
        """
        self.equation = equation
        self.reward_history = {}
        self.state_memory = {}
        self.prediction_models = {}
        
        # Add intrinsic reward components to the equation
        self._initialize_reward_components()
        
        # Initialize prediction models if PyTorch is available
        if TORCH_AVAILABLE:
            self._initialize_prediction_models()
    
    def _initialize_reward_components(self) -> None:
        """Initialize the intrinsic reward components in the equation."""
        # Add basic reward components
        self.equation.add_component("novelty_reward", "1.0 - familiarity(state)")
        self.equation.add_component("surprise_reward", "abs(predicted_outcome - actual_outcome)")
        self.equation.add_component("curiosity_reward", "information_gain(state, action, next_state)")
        self.equation.add_component("exploration_reward", "exploration_bonus(state, action)")
        
        # Add combined reward component
        self.equation.add_component("intrinsic_reward", "w_novelty * novelty_reward + w_surprise * surprise_reward + w_curiosity * curiosity_reward + w_exploration * exploration_reward")
        
        # Add weight variables
        self.equation.set_variable("w_novelty", 0.25)
        self.equation.set_variable("w_surprise", 0.25)
        self.equation.set_variable("w_curiosity", 0.25)
        self.equation.set_variable("w_exploration", 0.25)
    
    def _initialize_prediction_models(self) -> None:
        """Initialize prediction models for intrinsic rewards."""
        # Forward model for predicting next state
        class ForwardModel(nn.Module):
            def __init__(self, state_dim=10, action_dim=5, hidden_dim=64):
                super().__init__()
                self.state_encoder = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU()
                )
                self.action_encoder = nn.Sequential(
                    nn.Linear(action_dim, hidden_dim),
                    nn.ReLU()
                )
                self.combined = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, state_dim)
                )
            
            def forward(self, state, action):
                state_enc = self.state_encoder(state)
                action_enc = self.action_encoder(action)
                combined = torch.cat([state_enc, action_enc], dim=1)
                next_state_pred = self.combined(combined)
                return next_state_pred
        
        # Inverse model for predicting action from states
        class InverseModel(nn.Module):
            def __init__(self, state_dim=10, action_dim=5, hidden_dim=64):
                super().__init__()
                self.state_encoder = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU()
                )
                self.combined = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim)
                )
            
            def forward(self, state, next_state):
                state_enc = self.state_encoder(state)
                next_state_enc = self.state_encoder(next_state)
                combined = torch.cat([state_enc, next_state_enc], dim=1)
                action_pred = self.combined(combined)
                return action_pred
        
        # Random network distillation for novelty
        class RandomNetworkDistillation(nn.Module):
            def __init__(self, state_dim=10, hidden_dim=64, output_dim=32):
                super().__init__()
                # Random target network (fixed)
                self.target = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
                # Predictor network (trained)
                self.predictor = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
                
                # Initialize target network and freeze it
                for param in self.target.parameters():
                    param.requires_grad = False
            
            def forward(self, state):
                target_output = self.target(state)
                predictor_output = self.predictor(state)
                return target_output, predictor_output
        
        # Initialize models
        state_dim = 10  # Placeholder, would be determined from the environment
        action_dim = 5  # Placeholder, would be determined from the environment
        
        self.prediction_models["forward"] = ForwardModel(state_dim, action_dim)
        self.prediction_models["inverse"] = InverseModel(state_dim, action_dim)
        self.prediction_models["rnd"] = RandomNetworkDistillation(state_dim)
        
        # Initialize optimizers
        self.optimizers = {
            "forward": torch.optim.Adam(self.prediction_models["forward"].parameters(), lr=0.001),
            "inverse": torch.optim.Adam(self.prediction_models["inverse"].parameters(), lr=0.001),
            "rnd": torch.optim.Adam(self.prediction_models["rnd"].predictor.parameters(), lr=0.001)
        }
    
    def compute_novelty_reward(self, state: Any, entity_id: str = "default") -> float:
        """
        Compute novelty reward for a state.
        
        Args:
            state: The state to compute novelty for
            entity_id: ID of the entity (agent, component, etc.)
            
        Returns:
            Novelty reward value
        """
        # Convert state to a hashable representation
        state_key = self._state_to_key(state)
        
        # Check if state is in memory
        if state_key in self.state_memory:
            # State is familiar, compute familiarity based on visit count
            visit_count = self.state_memory[state_key]["count"]
            familiarity = min(visit_count / 10.0, 1.0)  # Normalize to [0, 1]
            novelty = 1.0 - familiarity
            
            # Update visit count
            self.state_memory[state_key]["count"] += 1
            self.state_memory[state_key]["last_visit"] = time.time()
        else:
            # State is novel
            novelty = 1.0
            
            # Add state to memory
            self.state_memory[state_key] = {
                "count": 1,
                "first_visit": time.time(),
                "last_visit": time.time(),
                "entity_id": entity_id
            }
        
        # Create reward signal
        signal = RewardSignal(
            value=novelty,
            reward_type=RewardType.NOVELTY,
            source="novelty_detector",
            metadata={"state_key": state_key}
        )
        
        # Add signal to history
        self._add_signal_to_history(entity_id, signal)
        
        return novelty
    
    def compute_surprise_reward(self, predicted_outcome: Any, actual_outcome: Any, entity_id: str = "default") -> float:
        """
        Compute surprise reward based on prediction error.
        
        Args:
            predicted_outcome: Predicted outcome
            actual_outcome: Actual outcome
            entity_id: ID of the entity (agent, component, etc.)
            
        Returns:
            Surprise reward value
        """
        # Compute prediction error
        if isinstance(predicted_outcome, np.ndarray) and isinstance(actual_outcome, np.ndarray):
            # For numpy arrays, use mean squared error
            prediction_error = np.mean((predicted_outcome - actual_outcome) ** 2)
        elif isinstance(predicted_outcome, torch.Tensor) and isinstance(actual_outcome, torch.Tensor):
            # For PyTorch tensors, use mean squared error
            prediction_error = torch.mean((predicted_outcome - actual_outcome) ** 2).item()
        else:
            # For other types, use a simple difference
            try:
                prediction_error = abs(float(predicted_outcome) - float(actual_outcome))
            except (TypeError, ValueError):
                # If conversion to float fails, return a default value
                prediction_error = 0.5
        
        # Normalize prediction error to [0, 1]
        surprise = min(prediction_error, 1.0)
        
        # Create reward signal
        signal = RewardSignal(
            value=surprise,
            reward_type=RewardType.SURPRISE,
            source="surprise_detector",
            metadata={"prediction_error": prediction_error}
        )
        
        # Add signal to history
        self._add_signal_to_history(entity_id, signal)
        
        return surprise
    
    def compute_curiosity_reward(self, state: Any, action: Any, next_state: Any, entity_id: str = "default") -> float:
        """
        Compute curiosity reward based on information gain.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            entity_id: ID of the entity (agent, component, etc.)
            
        Returns:
            Curiosity reward value
        """
        if not TORCH_AVAILABLE or "forward" not in self.prediction_models:
            # If PyTorch or models not available, return a default value
            curiosity = 0.5
        else:
            # Convert inputs to tensors
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state)
            elif isinstance(state, torch.Tensor):
                state_tensor = state
            else:
                # Placeholder conversion
                state_tensor = torch.zeros(10)
            
            if isinstance(action, np.ndarray):
                action_tensor = torch.FloatTensor(action)
            elif isinstance(action, torch.Tensor):
                action_tensor = action
            else:
                # Placeholder conversion
                action_tensor = torch.zeros(5)
            
            if isinstance(next_state, np.ndarray):
                next_state_tensor = torch.FloatTensor(next_state)
            elif isinstance(next_state, torch.Tensor):
                next_state_tensor = next_state
            else:
                # Placeholder conversion
                next_state_tensor = torch.zeros(10)
            
            # Predict next state
            with torch.no_grad():
                predicted_next_state = self.prediction_models["forward"](state_tensor.unsqueeze(0), action_tensor.unsqueeze(0))
            
            # Compute prediction error
            prediction_error = torch.mean((predicted_next_state.squeeze(0) - next_state_tensor) ** 2).item()
            
            # Normalize prediction error to [0, 1]
            curiosity = min(prediction_error, 1.0)
            
            # Update forward model
            self._update_forward_model(state_tensor, action_tensor, next_state_tensor)
        
        # Create reward signal
        signal = RewardSignal(
            value=curiosity,
            reward_type=RewardType.CURIOSITY,
            source="curiosity_model",
            metadata={"prediction_error": curiosity}
        )
        
        # Add signal to history
        self._add_signal_to_history(entity_id, signal)
        
        return curiosity
    
    def compute_exploration_reward(self, state: Any, action: Any, entity_id: str = "default") -> float:
        """
        Compute exploration reward based on state-action visitation.
        
        Args:
            state: Current state
            action: Action taken
            entity_id: ID of the entity (agent, component, etc.)
            
        Returns:
            Exploration reward value
        """
        # Convert state-action pair to a hashable representation
        state_action_key = self._state_action_to_key(state, action)
        
        # Check if state-action pair is in memory
        if state_action_key in self.state_memory:
            # State-action pair is familiar, compute exploration bonus based on visit count
            visit_count = self.state_memory[state_action_key]["count"]
            exploration_bonus = 1.0 / np.sqrt(visit_count)  # 1/sqrt(N) bonus
            
            # Update visit count
            self.state_memory[state_action_key]["count"] += 1
            self.state_memory[state_action_key]["last_visit"] = time.time()
        else:
            # State-action pair is novel
            exploration_bonus = 1.0
            
            # Add state-action pair to memory
            self.state_memory[state_action_key] = {
                "count": 1,
                "first_visit": time.time(),
                "last_visit": time.time(),
                "entity_id": entity_id
            }
        
        # Create reward signal
        signal = RewardSignal(
            value=exploration_bonus,
            reward_type=RewardType.EXPLORATION,
            source="exploration_bonus",
            metadata={"state_action_key": state_action_key}
        )
        
        # Add signal to history
        self._add_signal_to_history(entity_id, signal)
        
        return exploration_bonus
    
    def compute_combined_reward(self, state: Any, action: Any, next_state: Any, 
                               predicted_outcome: Optional[Any] = None, entity_id: str = "default") -> float:
        """
        Compute combined intrinsic reward.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            predicted_outcome: Predicted outcome (optional)
            entity_id: ID of the entity (agent, component, etc.)
            
        Returns:
            Combined intrinsic reward value
        """
        # Compute individual rewards
        novelty_reward = self.compute_novelty_reward(state, entity_id)
        
        if predicted_outcome is not None:
            surprise_reward = self.compute_surprise_reward(predicted_outcome, next_state, entity_id)
        else:
            # If no predicted outcome provided, use a default value
            surprise_reward = 0.0
        
        curiosity_reward = self.compute_curiosity_reward(state, action, next_state, entity_id)
        exploration_reward = self.compute_exploration_reward(state, action, entity_id)
        
        # Get weights from equation variables
        w_novelty = self.equation.get_variable("w_novelty") or 0.25
        w_surprise = self.equation.get_variable("w_surprise") or 0.25
        w_curiosity = self.equation.get_variable("w_curiosity") or 0.25
        w_exploration = self.equation.get_variable("w_exploration") or 0.25
        
        # Compute combined reward
        combined_reward = (
            w_novelty * novelty_reward +
            w_surprise * surprise_reward +
            w_curiosity * curiosity_reward +
            w_exploration * exploration_reward
        )
        
        # Create reward signal
        signal = RewardSignal(
            value=combined_reward,
            reward_type=RewardType.CURIOSITY,  # Using CURIOSITY as a general type
            source="combined_intrinsic_reward",
            metadata={
                "novelty_reward": novelty_reward,
                "surprise_reward": surprise_reward,
                "curiosity_reward": curiosity_reward,
                "exploration_reward": exploration_reward,
                "w_novelty": w_novelty,
                "w_surprise": w_surprise,
                "w_curiosity": w_curiosity,
                "w_exploration": w_exploration
            }
        )
        
        # Add signal to history
        self._add_signal_to_history(entity_id, signal)
        
        return combined_reward
    
    def _state_to_key(self, state: Any) -> str:
        """
        Convert a state to a hashable key.
        
        Args:
            state: The state to convert
            
        Returns:
            Hashable key representing the state
        """
        if isinstance(state, np.ndarray):
            # For numpy arrays, convert to tuple
            return str(tuple(state.flatten()))
        elif isinstance(state, torch.Tensor):
            # For PyTorch tensors, convert to tuple
            return str(tuple(state.detach().cpu().numpy().flatten()))
        else:
            # For other types, use string representation
            return str(state)
    
    def _state_action_to_key(self, state: Any, action: Any) -> str:
        """
        Convert a state-action pair to a hashable key.
        
        Args:
            state: The state
            action: The action
            
        Returns:
            Hashable key representing the state-action pair
        """
        state_key = self._state_to_key(state)
        
        if isinstance(action, np.ndarray):
            # For numpy arrays, convert to tuple
            action_key = str(tuple(action.flatten()))
        elif isinstance(action, torch.Tensor):
            # For PyTorch tensors, convert to tuple
            action_key = str(tuple(action.detach().cpu().numpy().flatten()))
        else:
            # For other types, use string representation
            action_key = str(action)
        
        return f"{state_key}_{action_key}"
    
    def _add_signal_to_history(self, entity_id: str, signal: RewardSignal) -> None:
        """
        Add a reward signal to the history.
        
        Args:
            entity_id: ID of the entity
            signal: Reward signal to add
        """
        if entity_id not in self.reward_history:
            self.reward_history[entity_id] = RewardHistory(entity_id=entity_id)
        
        self.reward_history[entity_id].signals.append(signal)
        self.reward_history[entity_id].total_reward += signal.value
    
    def _update_forward_model(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> None:
        """
        Update the forward model.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
        """
        if not TORCH_AVAILABLE or "forward" not in self.prediction_models:
            return
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        
        # Predict next state
        predicted_next_state = self.prediction_models["forward"](state, action)
        
        # Compute loss
        loss = torch.mean((predicted_next_state - next_state) ** 2)
        
        # Update model
        self.optimizers["forward"].zero_grad()
        loss.backward()
        self.optimizers["forward"].step()
    
    def _update_inverse_model(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> None:
        """
        Update the inverse model.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
        """
        if not TORCH_AVAILABLE or "inverse" not in self.prediction_models:
            return
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        
        # Predict action
        predicted_action = self.prediction_models["inverse"](state, next_state)
        
        # Compute loss
        loss = torch.mean((predicted_action - action) ** 2)
        
        # Update model
        self.optimizers["inverse"].zero_grad()
        loss.backward()
        self.optimizers["inverse"].step()
    
    def _update_rnd_model(self, state: torch.Tensor) -> None:
        """
        Update the Random Network Distillation model.
        
        Args:
            state: Current state
        """
        if not TORCH_AVAILABLE or "rnd" not in self.prediction_models:
            return
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Get target and predictor outputs
        target_output, predictor_output = self.prediction_models["rnd"](state)
        
        # Compute loss
        loss = torch.mean((predictor_output - target_output.detach()) ** 2)
        
        # Update model
        self.optimizers["rnd"].zero_grad()
        loss.backward()
        self.optimizers["rnd"].step()
    
    def get_reward_history(self, entity_id: str) -> Optional[RewardHistory]:
        """
        Get the reward history for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Reward history or None if not found
        """
        return self.reward_history.get(entity_id)
    
    def get_total_reward(self, entity_id: str) -> float:
        """
        Get the total reward for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Total reward
        """
        if entity_id in self.reward_history:
            return self.reward_history[entity_id].total_reward
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the intrinsic reward component to a dictionary.
        
        Returns:
            Dictionary representation of the intrinsic reward component
        """
        return {
            "reward_history": {entity_id: self._history_to_dict(history) for entity_id, history in self.reward_history.items()},
            "state_memory": self.state_memory
        }
    
    def _history_to_dict(self, history: RewardHistory) -> Dict[str, Any]:
        """
        Convert a reward history to a dictionary.
        
        Args:
            history: The reward history to convert
            
        Returns:
            Dictionary representation of the reward history
        """
        return {
            "entity_id": history.entity_id,
            "signals": [self._signal_to_dict(signal) for signal in history.signals],
            "total_reward": history.total_reward,
            "metadata": history.metadata
        }
    
    def _signal_to_dict(self, signal: RewardSignal) -> Dict[str, Any]:
        """
        Convert a reward signal to a dictionary.
        
        Args:
            signal: The reward signal to convert
            
        Returns:
            Dictionary representation of the reward signal
        """
        return {
            "value": signal.value,
            "reward_type": signal.reward_type.value,
            "source": signal.source,
            "timestamp": signal.timestamp,
            "metadata": signal.metadata
        }
    
    def to_json(self) -> str:
        """
        Convert the intrinsic reward component to a JSON string.
        
        Returns:
            JSON string representation of the intrinsic reward component
        """
        return json.dumps(self.to_dict(), indent=2)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a General Shape Equation
    equation = GeneralShapeEquation(
        equation_type=EquationType.BEHAVIOR,
        learning_mode=LearningMode.REINFORCEMENT
    )
    
    # Create an intrinsic reward component
    intrinsic_reward = IntrinsicRewardComponent(equation)
    
    # Example state, action, and next state
    state = np.random.rand(10)
    action = np.random.rand(5)
    next_state = np.random.rand(10)
    
    # Compute combined intrinsic reward
    reward = intrinsic_reward.compute_combined_reward(state, action, next_state)
    
    print(f"Combined Intrinsic Reward: {reward}")
    
    # Get reward history
    history = intrinsic_reward.get_reward_history("default")
    
    print(f"Total Reward: {history.total_reward}")
    print(f"Number of Reward Signals: {len(history.signals)}")
    
    # Print individual reward signals
    for i, signal in enumerate(history.signals):
        print(f"Signal {i+1}: Type={signal.reward_type.value}, Value={signal.value}, Source={signal.source}")
