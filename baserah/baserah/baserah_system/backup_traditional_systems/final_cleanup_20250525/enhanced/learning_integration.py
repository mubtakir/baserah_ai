#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning Integration Module for Basira System

This module provides the integration layer between the General Shape Equation framework
and various machine learning approaches, including deep learning and reinforcement learning.
It extends the adapters defined in the general_shape_equation module with concrete
implementations and utilities for training, evaluation, and deployment.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, Set
import random
import copy
import json
import time
from enum import Enum
from dataclasses import dataclass, field

# Import from parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced.general_shape_equation import (
    GeneralShapeEquation, EquationType, LearningMode, EquationMetadata,
    DeepLearningAdapter, ReinforcementLearningAdapter, ExpertExplorerSystem,
    GSEFactory
)


class ShapeEquationDataset(Dataset):
    """
    Dataset class for training neural networks on shape equations.
    
    This dataset generates samples from GeneralShapeEquation instances
    for use in supervised learning.
    """
    
    def __init__(self, equations: List[GeneralShapeEquation], 
                 samples_per_equation: int = 1000,
                 input_dim: int = 2,
                 transform: Optional[Callable] = None):
        """
        Initialize a ShapeEquationDataset.
        
        Args:
            equations: List of GeneralShapeEquation instances to sample from
            samples_per_equation: Number of samples to generate per equation
            input_dim: Input dimension (default: 2 for x,y coordinates)
            transform: Optional transform to apply to samples
        """
        self.equations = equations
        self.samples_per_equation = samples_per_equation
        self.input_dim = input_dim
        self.transform = transform
        
        # Generate all samples at initialization
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Generate samples from all equations.
        
        Returns:
            List of (input, target, equation_index) tuples
        """
        all_samples = []
        
        for eq_idx, equation in enumerate(self.equations):
            # Generate random inputs
            X = np.random.uniform(-10, 10, (self.samples_per_equation, self.input_dim))
            y = np.zeros((self.samples_per_equation, 1))
            
            # Evaluate the equation for each sample
            for i in range(self.samples_per_equation):
                # Create variable assignments
                assignments = {}
                for j in range(self.input_dim):
                    var_name = ['x', 'y', 'z', 't'][j] if j < 4 else f'var{j}'
                    assignments[var_name] = X[i, j]
                
                # Evaluate the equation
                results = equation.evaluate(assignments)
                
                # Use the first component's result as the target
                if results:
                    first_result = next(iter(results.values()))
                    if first_result is not None:
                        y[i, 0] = first_result
            
            # Add samples to the list
            for i in range(self.samples_per_equation):
                all_samples.append((X[i], y[i], eq_idx))
        
        return all_samples
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input tensor, target tensor, equation index)
        """
        X, y, eq_idx = self.samples[idx]
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Apply transform if provided
        if self.transform:
            X_tensor, y_tensor = self.transform(X_tensor, y_tensor)
        
        return X_tensor, y_tensor, eq_idx


class EnhancedDeepLearningAdapter(DeepLearningAdapter):
    """
    Enhanced adapter for deep learning integration with GeneralShapeEquation.
    
    This class extends the basic DeepLearningAdapter with additional functionality
    for training, evaluation, visualization, and model selection.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 1,
                 learning_rate: float = 0.001, model_type: str = 'mlp'):
        """
        Initialize an EnhancedDeepLearningAdapter.
        
        Args:
            input_dim: Input dimension (default: 2 for x,y coordinates)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (default: 1 for scalar output)
            learning_rate: Learning rate for optimization
            model_type: Type of model to use ('mlp', 'cnn', 'transformer')
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.model_type = model_type
        
        # Create the model based on the specified type
        self.model = self._create_model()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Initialize training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': 0
        }
    
    def _create_model(self) -> nn.Module:
        """
        Create a neural network model based on the specified type.
        
        Returns:
            PyTorch neural network model
        """
        if self.model_type == 'mlp':
            return nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        elif self.model_type == 'cnn':
            # For CNN, we reshape the input to 2D grid
            # This assumes input_dim is a perfect square
            grid_size = int(np.sqrt(self.input_dim))
            if grid_size * grid_size != self.input_dim:
                raise ValueError(f"Input dimension {self.input_dim} must be a perfect square for CNN")
            
            return nn.Sequential(
                nn.Unflatten(1, (1, grid_size, grid_size)),  # Reshape to [batch, 1, grid, grid]
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * grid_size * grid_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        elif self.model_type == 'transformer':
            # Simple transformer-inspired model
            return nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                TransformerBlock(self.hidden_dim),
                TransformerBlock(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_on_dataset(self, dataset: ShapeEquationDataset, 
                        batch_size: int = 32, 
                        num_epochs: int = 100,
                        validation_split: float = 0.2,
                        early_stopping: bool = True,
                        patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the neural network on a dataset of shape equations.
        
        Args:
            dataset: ShapeEquationDataset to train on
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Dictionary of training history
        """
        # Split dataset into train and validation
        dataset_size = len(dataset)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets, _ in train_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['epochs'] = epoch + 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Check for early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
        
        return self.training_history
    
    def evaluate_on_equation(self, equation: GeneralShapeEquation, 
                           num_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate the model on a specific equation.
        
        Args:
            equation: GeneralShapeEquation to evaluate on
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Generate evaluation data
        X, y = self._generate_samples(equation, num_samples)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, self.output_dim)
        
        # Evaluate the model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # Calculate additional metrics
            mse = F.mse_loss(outputs, y_tensor).item()
            mae = F.l1_loss(outputs, y_tensor).item()
            
            # Calculate R^2 score
            y_mean = torch.mean(y_tensor)
            ss_tot = torch.sum((y_tensor - y_mean) ** 2)
            ss_res = torch.sum((y_tensor - outputs) ** 2)
            r2 = 1 - (ss_res / ss_tot).item()
        
        return {
            'loss': loss.item(),
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    def visualize_predictions(self, equation: GeneralShapeEquation, 
                             grid_size: int = 100,
                             x_range: Tuple[float, float] = (-5, 5),
                             y_range: Tuple[float, float] = (-5, 5),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize model predictions against true equation values.
        
        Args:
            equation: GeneralShapeEquation to visualize
            grid_size: Size of the visualization grid
            x_range: Range of x values
            y_range: Range of y values
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Create a grid of points
        x = np.linspace(x_range[0], x_range[1], grid_size)
        y = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Flatten the grid for evaluation
        points = np.column_stack((X.flatten(), Y.flatten()))
        
        # Evaluate the true equation
        true_values = np.zeros(points.shape[0])
        for i, point in enumerate(points):
            assignments = {'x': point[0], 'y': point[1]}
            results = equation.evaluate(assignments)
            if results:
                first_result = next(iter(results.values()))
                if first_result is not None:
                    true_values[i] = first_result
        
        # Reshape true values to grid
        true_grid = true_values.reshape(grid_size, grid_size)
        
        # Evaluate the model
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(points, dtype=torch.float32)
            outputs = self.model(inputs).numpy().flatten()
        
        # Reshape model outputs to grid
        pred_grid = outputs.reshape(grid_size, grid_size)
        
        # Calculate error
        error_grid = np.abs(true_grid - pred_grid)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot true values
        im1 = axes[0].imshow(true_grid, extent=[*x_range, *y_range], origin='lower', cmap='viridis')
        axes[0].set_title('True Equation Values')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot predicted values
        im2 = axes[1].imshow(pred_grid, extent=[*x_range, *y_range], origin='lower', cmap='viridis')
        axes[1].set_title('Model Predictions')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot error
        im3 = axes[2].imshow(error_grid, extent=[*x_range, *y_range], origin='lower', cmap='hot')
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def enhance_equation_with_neural_correction(self, equation: GeneralShapeEquation) -> GeneralShapeEquation:
        """
        Enhance an equation with neural correction.
        
        This creates a new equation that combines the original symbolic components
        with neural correction terms.
        
        Args:
            equation: The GeneralShapeEquation to enhance
            
        Returns:
            Enhanced GeneralShapeEquation
        """
        # Clone the equation
        enhanced_equation = equation.clone()
        
        # Add the neural network to the equation
        enhanced_equation.neural_components['dl_correction'] = self.model
        
        # Update metadata
        from datetime import datetime
        enhanced_equation.metadata.last_modified = datetime.now().isoformat()
        enhanced_equation.metadata.version += 1
        enhanced_equation.metadata.description = f"{enhanced_equation.metadata.description} (Neural corrected)"
        
        # Set learning mode to hybrid
        enhanced_equation.learning_mode = LearningMode.HYBRID
        
        # Add a special evaluation method that combines symbolic and neural evaluation
        def enhanced_evaluate(assignments: Dict[str, float]) -> Dict[str, Optional[float]]:
            # First, evaluate the symbolic components
            symbolic_results = equation.evaluate(assignments)
            
            # Then, evaluate the neural correction
            input_values = []
            for var_name in ['x', 'y', 'z', 't']:
                if var_name in assignments:
                    input_values.append(assignments[var_name])
            
            # Ensure we have enough input values
            while len(input_values) < self.input_dim:
                input_values.append(0.0)
            
            # Truncate if we have too many
            input_values = input_values[:self.input_dim]
            
            # Convert to tensor and get neural prediction
            input_tensor = torch.tensor(input_values, dtype=torch.float32)
            with torch.no_grad():
                neural_correction = self.model(input_tensor).item()
            
            # Combine results
            combined_results = {}
            for key, value in symbolic_results.items():
                if value is not None:
                    combined_results[key] = value
            
            # Add neural correction as a separate component
            combined_results['neural_correction'] = neural_correction
            
            # Add combined result
            if symbolic_results and next(iter(symbolic_results.values())) is not None:
                first_symbolic = next(iter(symbolic_results.values()))
                combined_results['combined'] = first_symbolic + neural_correction
            else:
                combined_results['combined'] = neural_correction
            
            return combined_results
        
        # Attach the enhanced evaluate method
        enhanced_equation.enhanced_evaluate = enhanced_evaluate
        
        return enhanced_equation
    
    def plot_training_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the training history.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, self.training_history['epochs'] + 1)
        ax.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        ax.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class TransformerBlock(nn.Module):
    """
    Simple transformer block for use in neural networks.
    
    This implements a basic transformer-style self-attention mechanism
    followed by a feed-forward network.
    """
    
    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None):
        """
        Initialize a TransformerBlock.
        
        Args:
            dim: Dimension of the input and output
            num_heads: Number of attention heads
            ff_dim: Dimension of the feed-forward network (defaults to 4*dim)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or 4 * dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(dim, num_heads)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape [batch_size, dim]
            
        Returns:
            Output tensor of shape [batch_size, dim]
        """
        # Reshape for attention if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add sequence dimension [1, batch, dim]
            x = x.transpose(0, 1)  # [batch, 1, dim]
        
        # Self-attention with residual connection and layer norm
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        
        # Feed-forward with residual connection and layer norm
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)
        
        # Reshape back if needed
        if x.shape[1] == 1:
            x = x.squeeze(1)  # Remove sequence dimension [batch, dim]
        
        return x


class EnhancedReinforcementLearningAdapter(ReinforcementLearningAdapter):
    """
    Enhanced adapter for reinforcement learning integration with GeneralShapeEquation.
    
    This class extends the basic ReinforcementLearningAdapter with additional functionality
    for advanced RL algorithms, visualization, and integration with the expert/explorer system.
    """
    
    def __init__(self, state_dim: int = 10, action_dim: int = 5, 
                 algorithm: str = 'ppo', gamma: float = 0.99,
                 learning_rate: float = 0.001):
        """
        Initialize an EnhancedReinforcementLearningAdapter.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            algorithm: RL algorithm to use ('ppo', 'dqn', 'a2c')
            gamma: Discount factor for future rewards
            learning_rate: Learning rate for optimization
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Initialize networks based on the algorithm
        self._initialize_networks()
        
        # Initialize memory buffer for experience replay
        self.memory = []
        self.max_memory_size = 10000
        
        # Initialize training history
        self.training_history = {
            'rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'episodes': 0
        }
    
    def _initialize_networks(self):
        """Initialize neural networks based on the selected algorithm."""
        if self.algorithm == 'ppo':
            # Policy network (actor)
            self.policy_net = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_dim),
                nn.Softmax(dim=-1)
            )
            
            # Value network (critic)
            self.value_net = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
            # Initialize optimizers
            self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
            self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
            
            # PPO-specific parameters
            self.clip_ratio = 0.2
            self.target_kl = 0.01
            self.old_policy = None
        
        elif self.algorithm == 'dqn':
            # Q-network
            self.q_net = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_dim)
            )
            
            # Target Q-network
            self.target_q_net = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_dim)
            )
            
            # Copy parameters from Q-network to target Q-network
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
            
            # DQN-specific parameters
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.target_update_freq = 10
        
        elif self.algorithm == 'a2c':
            # Combined actor-critic network
            self.ac_net = ActorCriticNetwork(self.state_dim, self.action_dim)
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=self.learning_rate)
            
            # A2C-specific parameters
            self.entropy_coef = 0.01
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def equation_to_state(self, equation: GeneralShapeEquation) -> torch.Tensor:
        """
        Convert a GeneralShapeEquation to a state tensor.
        
        Args:
            equation: The GeneralShapeEquation to convert
            
        Returns:
            State tensor
        """
        # Extract features from the equation
        features = [
            len(equation.symbolic_components),  # Number of components
            equation.metadata.complexity,  # Complexity score
            equation.metadata.version,  # Version number
            float(equation.equation_type.value == EquationType.SHAPE.value),  # Is it a shape equation?
            float(equation.equation_type.value == EquationType.PATTERN.value),  # Is it a pattern equation?
            float(equation.equation_type.value == EquationType.BEHAVIOR.value),  # Is it a behavior equation?
            float(equation.equation_type.value == EquationType.TRANSFORMATION.value),  # Is it a transformation equation?
            float(equation.equation_type.value == EquationType.CONSTRAINT.value),  # Is it a constraint equation?
            float(equation.equation_type.value == EquationType.COMPOSITE.value),  # Is it a composite equation?
            len(equation.variables)  # Number of variables
        ]
        
        # Add component-specific features
        for component_name in ['circle_equation', 'ellipse_equation', 'rectangle_equation']:
            features.append(float(component_name in equation.symbolic_components))
        
        # Add variable-specific features
        for var_name in ['x', 'y', 'z', 't']:
            features.append(float(var_name in equation.variables))
        
        # Pad or truncate to match state_dim
        if len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))
        elif len(features) > self.state_dim:
            features = features[:self.state_dim]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def select_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (action index, action probability or value)
        """
        if self.algorithm == 'ppo' or self.algorithm == 'a2c':
            with torch.no_grad():
                if self.algorithm == 'ppo':
                    # Get action probabilities from policy network
                    probs = self.policy_net(state)
                else:  # a2c
                    # Get action probabilities from actor-critic network
                    probs, _ = self.ac_net(state)
                
                # Sample an action
                action = torch.multinomial(probs, 1).item()
                
                # Get the probability of the selected action
                prob = probs[action].item()
            
            return action, prob
        
        elif self.algorithm == 'dqn':
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                # Random action
                action = random.randint(0, self.action_dim - 1)
                value = 0.0  # Placeholder
            else:
                with torch.no_grad():
                    # Greedy action
                    q_values = self.q_net(state)
                    action = torch.argmax(q_values).item()
                    value = q_values[action].item()
            
            return action, value
    
    def apply_action(self, equation: GeneralShapeEquation, action: int) -> GeneralShapeEquation:
        """
        Apply an action to evolve the equation.
        
        Args:
            equation: The GeneralShapeEquation to evolve
            action: Action index
            
        Returns:
            Evolved GeneralShapeEquation
        """
        # Define possible actions
        actions = [
            lambda eq: eq.mutate(0.1),  # Small mutation
            lambda eq: eq.mutate(0.3),  # Medium mutation
            lambda eq: eq.mutate(0.5),  # Large mutation
            lambda eq: eq.simplify(),  # Simplify
            lambda eq: self._add_random_component(eq),  # Add a random component
            lambda eq: self._remove_random_component(eq),  # Remove a random component
            lambda eq: self._modify_random_parameter(eq),  # Modify a random parameter
            lambda eq: eq.clone()  # Just clone (no change)
        ]
        
        # Apply the selected action
        action_idx = action % len(actions)
        return actions[action_idx](equation)
    
    def _add_random_component(self, equation: GeneralShapeEquation) -> GeneralShapeEquation:
        """
        Add a random component to the equation.
        
        Args:
            equation: The equation to modify
            
        Returns:
            Modified equation
        """
        # Clone the equation
        new_equation = equation.clone()
        
        # Create a simple random component
        import random
        var_names = list(equation.variables.keys()) or ['x', 'y']
        var_name = random.choice(var_names)
        
        # Generate a random expression
        coef = random.uniform(-1.0, 1.0)
        expr = f"{coef} * {var_name}"
        
        # Add the component
        component_name = f"random_component_{len(equation.symbolic_components)}"
        new_equation.add_component(component_name, expr)
        
        return new_equation
    
    def _remove_random_component(self, equation: GeneralShapeEquation) -> GeneralShapeEquation:
        """
        Remove a random component from the equation.
        
        Args:
            equation: The equation to modify
            
        Returns:
            Modified equation
        """
        # Clone the equation
        new_equation = equation.clone()
        
        # Get component names
        component_names = list(equation.symbolic_components.keys())
        
        # Don't remove if there's only one component
        if len(component_names) <= 1:
            return new_equation
        
        # Select a random component to remove
        import random
        component_to_remove = random.choice(component_names)
        
        # Remove the component
        new_equation.remove_component(component_to_remove)
        
        return new_equation
    
    def _modify_random_parameter(self, equation: GeneralShapeEquation) -> GeneralShapeEquation:
        """
        Modify a random parameter in the equation's parameter space.
        
        Args:
            equation: The equation to modify
            
        Returns:
            Modified equation
        """
        # Clone the equation
        new_equation = equation.clone()
        
        # Get parameter names
        param_names = list(equation.parameter_space.keys())
        
        # Return if no parameters
        if not param_names:
            return new_equation
        
        # Select a random parameter to modify
        import random
        param_to_modify = random.choice(param_names)
        
        # Get parameter properties
        param_props = equation.parameter_space[param_to_modify]
        
        # Modify the parameter range
        if 'range' in param_props:
            old_range = param_props['range']
            # Expand or contract the range by a random factor
            factor = random.uniform(0.8, 1.2)
            new_min = old_range[0] * factor
            new_max = old_range[1] * factor
            
            # Update the parameter space
            new_equation.parameter_space[param_to_modify]['range'] = [new_min, new_max]
        
        return new_equation
    
    def calculate_reward(self, original_eq: GeneralShapeEquation, 
                        evolved_eq: GeneralShapeEquation,
                        target_complexity: float = None) -> float:
        """
        Calculate the reward for an evolution step.
        
        Args:
            original_eq: Original equation before evolution
            evolved_eq: Evolved equation after applying an action
            target_complexity: Optional target complexity
            
        Returns:
            Reward value
        """
        # Initialize reward components
        complexity_reward = 0.0
        component_reward = 0.0
        variable_reward = 0.0
        
        # Complexity reward
        if target_complexity is not None:
            # Reward based on how close we are to the target complexity
            orig_diff = abs(original_eq.metadata.complexity - target_complexity)
            evolved_diff = abs(evolved_eq.metadata.complexity - target_complexity)
            
            # Reward if we're getting closer to the target
            complexity_reward = orig_diff - evolved_diff
        else:
            # Default: reward simpler equations
            complexity_change = original_eq.metadata.complexity - evolved_eq.metadata.complexity
            complexity_reward = 0.5 * complexity_change
        
        # Component count reward
        orig_components = len(original_eq.symbolic_components)
        evolved_components = len(evolved_eq.symbolic_components)
        
        # Reward if the number of components is appropriate
        if 3 <= evolved_components <= 7:
            component_reward = 0.5
        elif evolved_components < 3:
            component_reward = -0.2 * (3 - evolved_components)
        else:  # evolved_components > 7
            component_reward = -0.1 * (evolved_components - 7)
        
        # Variable usage reward
        orig_vars = len(original_eq.variables)
        evolved_vars = len(evolved_eq.variables)
        
        # Reward if using an appropriate number of variables
        if 2 <= evolved_vars <= 4:
            variable_reward = 0.3
        elif evolved_vars < 2:
            variable_reward = -0.1
        else:  # evolved_vars > 4
            variable_reward = -0.1 * (evolved_vars - 4)
        
        # Combine rewards
        total_reward = complexity_reward + component_reward + variable_reward
        
        return total_reward
    
    def train_ppo(self, batch_size: int = 32, epochs: int = 10) -> Tuple[float, float]:
        """
        Train using the PPO algorithm.
        
        Args:
            batch_size: Batch size for training
            epochs: Number of epochs to train on the current batch
            
        Returns:
            Tuple of (policy loss, value loss)
        """
        if len(self.memory) < batch_size:
            return 0.0, 0.0
        
        # Sample a batch of experiences
        import random
        batch = random.sample(self.memory, batch_size)
        
        # Unpack the batch
        states, actions, old_probs, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Calculate returns (discounted future rewards)
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
            returns = rewards + self.gamma * next_values * (1 - dones)
        
        # Train for multiple epochs
        policy_losses = []
        value_losses = []
        
        for _ in range(epochs):
            # Get current values
            values = self.value_net(states).squeeze()
            
            # Calculate advantage (returns - values)
            advantages = returns - values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Get action probabilities
            probs = self.policy_net(states)
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Calculate ratio of new and old probabilities
            ratio = action_probs / old_probs
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Calculate policy loss (negative because we want to maximize)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(values, returns)
            
            # Calculate total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            
            # Calculate KL divergence for early stopping
            with torch.no_grad():
                new_probs = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                kl = (old_probs * torch.log(old_probs / new_probs)).mean().item()
                if kl > self.target_kl:
                    break
        
        return sum(policy_losses) / len(policy_losses), sum(value_losses) / len(value_losses)
    
    def train_dqn(self, batch_size: int = 32) -> float:
        """
        Train using the DQN algorithm.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Loss value
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample a batch of experiences
        import random
        batch = random.sample(self.memory, batch_size)
        
        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Get current Q values
        current_q = self.q_net(states).gather(1, actions).squeeze()
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q = self.target_q_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Calculate loss
        loss = F.mse_loss(current_q, target_q)
        
        # Update Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train_a2c(self, states, actions, rewards, next_states, dones) -> Tuple[float, float]:
        """
        Train using the A2C algorithm.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            Tuple of (policy loss, value loss)
        """
        # Convert to tensors if not already
        if not isinstance(states, torch.Tensor):
            states = torch.stack(states)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones, dtype=torch.float32)
        
        # Get action probabilities and values
        probs, values = self.ac_net(states)
        
        # Calculate returns (discounted future rewards)
        with torch.no_grad():
            next_values = self.ac_net(next_states)[1].squeeze()
            returns = rewards + self.gamma * next_values * (1 - dones)
        
        # Calculate advantage (returns - values)
        advantages = returns - values.squeeze()
        
        # Get action log probabilities
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
        log_probs = torch.log(action_probs)
        
        # Calculate entropy (for exploration)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
        
        # Calculate policy loss (negative because we want to maximize)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Calculate value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Calculate total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def store_experience(self, state, action, action_prob, reward, next_state, done):
        """
        Store an experience in the memory buffer.
        
        Args:
            state: Current state
            action: Action taken
            action_prob: Probability or value of the action
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if self.algorithm == 'ppo':
            # For PPO, we need to store the old action probability
            self.memory.append((state, action, action_prob, reward, next_state, done))
        else:
            # For other algorithms
            self.memory.append((state, action, reward, next_state, done))
        
        # Limit memory size
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Perform a training step using experiences from memory.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary of loss values
        """
        if self.algorithm == 'ppo':
            policy_loss, value_loss = self.train_ppo(batch_size)
            return {'policy_loss': policy_loss, 'value_loss': value_loss}
        elif self.algorithm == 'dqn':
            loss = self.train_dqn(batch_size)
            return {'loss': loss}
        elif self.algorithm == 'a2c':
            if len(self.memory) < batch_size:
                return {'policy_loss': 0.0, 'value_loss': 0.0}
            
            # Sample a batch of experiences
            import random
            batch = random.sample(self.memory, batch_size)
            
            # Unpack the batch
            states, actions, rewards, next_states, dones = zip(*batch)
            
            policy_loss, value_loss = self.train_a2c(states, actions, rewards, next_states, dones)
            return {'policy_loss': policy_loss, 'value_loss': value_loss}
    
    def train_on_equation(self, equation: GeneralShapeEquation, 
                         num_episodes: int = 100, 
                         steps_per_episode: int = 10,
                         target_complexity: float = None) -> Tuple[List[float], GeneralShapeEquation]:
        """
        Train the RL agent to evolve a GeneralShapeEquation.
        
        Args:
            equation: The GeneralShapeEquation to evolve
            num_episodes: Number of training episodes
            steps_per_episode: Number of steps per episode
            target_complexity: Optional target complexity
            
        Returns:
            Tuple of (rewards list, best evolved equation)
        """
        total_rewards = []
        best_reward = float('-inf')
        best_equation = equation.clone()
        
        for episode in range(num_episodes):
            # Start with a fresh copy of the original equation
            current_eq = equation.clone()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Convert equation to state
                state = self.equation_to_state(current_eq)
                
                # Select action
                action, action_prob = self.select_action(state)
                
                # Apply action to get next equation
                next_eq = self.apply_action(current_eq, action)
                
                # Calculate reward
                reward = self.calculate_reward(current_eq, next_eq, target_complexity)
                episode_reward += reward
                
                # Convert next equation to state
                next_state = self.equation_to_state(next_eq)
                
                # Store experience
                done = (step == steps_per_episode - 1)
                
                if self.algorithm == 'ppo':
                    self.store_experience(state, action, action_prob, reward, next_state, done)
                else:
                    self.store_experience(state, action, reward, next_state, done)
                
                # Move to next state
                current_eq = next_eq
                
                # Train on a batch of experiences
                if len(self.memory) >= batch_size:
                    self.train_step(batch_size=32)
                
                # Update target network for DQN
                if self.algorithm == 'dqn' and step % self.target_update_freq == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())
            
            # Record total reward for this episode
            total_rewards.append(episode_reward)
            self.training_history['rewards'].append(episode_reward)
            
            # Update best equation if this episode produced a better one
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_equation = current_eq.clone()
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f'Episode [{episode+1}/{num_episodes}], Reward: {episode_reward:.4f}')
        
        # Update training history
        self.training_history['episodes'] += num_episodes
        
        return total_rewards, best_equation
    
    def plot_training_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the training history.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episodes = range(1, len(self.training_history['rewards']) + 1)
        ax.plot(episodes, self.training_history['rewards'], 'b-', label='Episode Rewards')
        
        # Add moving average
        window_size = min(10, len(episodes))
        if window_size > 0:
            moving_avg = np.convolve(self.training_history['rewards'], 
                                    np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size, len(episodes) + 1), moving_avg, 'r-', 
                   label=f'Moving Average ({window_size} episodes)')
        
        ax.set_title('Training Rewards')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward')
        ax.legend()
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ActorCriticNetwork(nn.Module):
    """
    Combined actor-critic network for A2C algorithm.
    
    This network outputs both action probabilities (actor) and
    state value estimates (critic).
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize an ActorCriticNetwork.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        
        # Actor (policy) layers
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) layers
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Tuple of (action probabilities, state value)
        """
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        
        return action_probs, state_value


class EnhancedExpertExplorerSystem(ExpertExplorerSystem):
    """
    Enhanced implementation of the Expert/Explorer interaction model.
    
    This class extends the basic ExpertExplorerSystem with additional functionality
    for visualization, analysis, and integration with the semantic database.
    """
    
    def __init__(self, initial_equation: Optional[GeneralShapeEquation] = None,
                 use_deep_learning: bool = True,
                 use_reinforcement_learning: bool = True):
        """
        Initialize an EnhancedExpertExplorerSystem.
        
        Args:
            initial_equation: Optional initial equation to start with
            use_deep_learning: Whether to use deep learning
            use_reinforcement_learning: Whether to use reinforcement learning
        """
        # Initialize the base system
        super().__init__(initial_equation)
        
        # Set learning flags
        self.use_deep_learning = use_deep_learning
        self.use_reinforcement_learning = use_reinforcement_learning
        
        # Initialize enhanced adapters
        if use_deep_learning:
            self.dl_adapter = EnhancedDeepLearningAdapter()
        
        if use_reinforcement_learning:
            self.rl_adapter = EnhancedReinforcementLearningAdapter()
        
        # Initialize semantic database connection
        self.semantic_db = {}
        
        # Initialize visualization history
        self.visualization_history = []
    
    def initialize_expert_knowledge_from_semantic_db(self, semantic_db_path: str) -> None:
        """
        Initialize the expert's knowledge base from a semantic database.
        
        Args:
            semantic_db_path: Path to the semantic database file
        """
        try:
            # Load semantic database
            with open(semantic_db_path, 'r', encoding='utf-8') as f:
                self.semantic_db = json.load(f)
            
            # Extract knowledge from semantic database
            for letter, data in self.semantic_db.items():
                if 'shape_properties' in data:
                    # Create a shape equation based on the semantic properties
                    shape_type = data.get('shape_properties', {}).get('basic_shape', 'unknown')
                    
                    if shape_type in ['circle', 'rectangle', 'ellipse']:
                        # Create a basic shape equation
                        equation = GSEFactory.create_basic_shape(shape_type)
                        
                        # Add to expert knowledge base
                        self.expert['knowledge_base'][f"{letter}_{shape_type}"] = equation
                        
                        # Add semantic links
                        for semantic_property, value in data.get('semantic_properties', {}).items():
                            equation.add_semantic_link(semantic_property, 'semantic_property', 1.0)
            
            print(f"Initialized expert knowledge from semantic database with {len(self.expert['knowledge_base'])} entries")
        
        except Exception as e:
            print(f"Error initializing expert knowledge from semantic database: {e}")
    
    def expert_evaluate_with_semantics(self, equation: GeneralShapeEquation, 
                                     target_semantics: List[str] = None) -> float:
        """
        Expert evaluation of an equation with semantic considerations.
        
        Args:
            equation: The equation to evaluate
            target_semantics: Optional list of target semantic concepts
            
        Returns:
            Evaluation score (higher is better)
        """
        # Get base evaluation score
        base_score = self.expert_evaluate(equation)
        
        # If no target semantics, return base score
        if not target_semantics:
            return base_score
        
        # Calculate semantic alignment score
        semantic_score = 0.0
        equation_semantics = equation.get_semantic_links()
        
        for link in equation_semantics:
            concept = link.get('concept', '')
            strength = link.get('strength', 0.0)
            
            if concept in target_semantics:
                semantic_score += strength
        
        # Combine scores (weighted average)
        combined_score = 0.7 * base_score + 0.3 * semantic_score
        
        return combined_score
    
    def explorer_explore_with_learning(self, steps: int = 10, 
                                     use_dl: bool = None, 
                                     use_rl: bool = None) -> GeneralShapeEquation:
        """
        Explorer exploration using learning-based approaches.
        
        Args:
            steps: Number of exploration steps
            use_dl: Whether to use deep learning (overrides instance setting)
            use_rl: Whether to use reinforcement learning (overrides instance setting)
            
        Returns:
            Best equation found during exploration
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        # Determine which learning approaches to use
        use_deep_learning = self.use_deep_learning if use_dl is None else use_dl
        use_reinforcement_learning = self.use_reinforcement_learning if use_rl is None else use_rl
        
        best_equation = self.explorer['current_equation']
        best_score = self.expert_evaluate(best_equation)
        
        # Deep learning exploration
        if use_deep_learning and hasattr(self, 'dl_adapter'):
            # Train the DL adapter on the current equation
            dl_dataset = ShapeEquationDataset([self.explorer['current_equation']])
            self.dl_adapter.train_on_dataset(dl_dataset, num_epochs=min(50, steps * 5))
            
            # Enhance the equation with neural correction
            dl_enhanced_equation = self.dl_adapter.enhance_equation_with_neural_correction(
                self.explorer['current_equation']
            )
            
            # Evaluate the enhanced equation
            dl_score = self.expert_evaluate(dl_enhanced_equation)
            
            # Update best if better
            if dl_score > best_score:
                best_equation = dl_enhanced_equation
                best_score = dl_score
        
        # Reinforcement learning exploration
        if use_reinforcement_learning and hasattr(self, 'rl_adapter'):
            # Train the RL adapter on the current equation
            rl_rewards, rl_enhanced_equation = self.rl_adapter.train_on_equation(
                self.explorer['current_equation'],
                num_episodes=min(20, steps * 2),
                steps_per_episode=5
            )
            
            # Evaluate the enhanced equation
            rl_score = self.expert_evaluate(rl_enhanced_equation)
            
            # Update best if better
            if rl_score > best_score:
                best_equation = rl_enhanced_equation
                best_score = rl_score
        
        # Update the current equation
        self.explorer['current_equation'] = best_equation
        
        # Record in exploration history
        self.explorer['exploration_history'].append({
            'equation': best_equation,
            'score': best_score,
            'method': 'learning',
            'dl_used': use_deep_learning,
            'rl_used': use_reinforcement_learning
        })
        
        return best_equation
    
    def visualize_equation_evolution(self, equations: List[GeneralShapeEquation], 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the evolution of equations.
        
        Args:
            equations: List of equations in evolution sequence
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        if not equations:
            raise ValueError("Must provide at least one equation")
        
        # Extract metrics for visualization
        complexities = [eq.metadata.complexity for eq in equations]
        component_counts = [len(eq.symbolic_components) for eq in equations]
        variable_counts = [len(eq.variables) for eq in equations]
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot complexity
        axes[0].plot(complexities, 'b-o')
        axes[0].set_title('Equation Complexity')
        axes[0].set_ylabel('Complexity')
        
        # Plot component count
        axes[1].plot(component_counts, 'r-o')
        axes[1].set_title('Number of Components')
        axes[1].set_ylabel('Components')
        
        # Plot variable count
        axes[2].plot(variable_counts, 'g-o')
        axes[2].set_title('Number of Variables')
        axes[2].set_ylabel('Variables')
        axes[2].set_xlabel('Evolution Step')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'evolution',
            'data': {
                'complexities': complexities,
                'component_counts': component_counts,
                'variable_counts': variable_counts
            },
            'figure': fig
        })
        
        return fig
    
    def expert_explorer_interaction_with_semantics(self, cycles: int = 5, 
                                                steps_per_cycle: int = 10,
                                                target_semantics: List[str] = None) -> GeneralShapeEquation:
        """
        Full expert-explorer interaction cycle with semantic guidance.
        
        Args:
            cycles: Number of interaction cycles
            steps_per_cycle: Steps per exploration cycle
            target_semantics: Optional list of target semantic concepts
            
        Returns:
            Best equation found during all cycles
        """
        if not self.explorer['current_equation']:
            raise ValueError("Explorer needs an initial equation")
        
        best_overall_equation = self.explorer['current_equation']
        best_overall_score = self.expert_evaluate_with_semantics(best_overall_equation, target_semantics)
        
        # Track evolution for visualization
        evolution_sequence = [best_overall_equation.clone()]
        
        for cycle in range(cycles):
            # Explorer explores
            if cycle % 3 == 0:
                # Use standard exploration
                best_cycle_equation = self.explorer_explore(steps=steps_per_cycle)
            elif cycle % 3 == 1:
                # Use learning-based exploration
                best_cycle_equation = self.explorer_explore_with_learning(steps=steps_per_cycle)
            else:
                # Use RL-based exploration
                best_cycle_equation = self.explorer_explore_rl(
                    episodes=steps_per_cycle // 2,
                    steps_per_episode=5
                )
            
            # Expert evaluates with semantic consideration
            cycle_score = self.expert_evaluate_with_semantics(best_cycle_equation, target_semantics)
            
            # Record interaction
            self.interaction_history.append({
                'cycle': cycle,
                'equation': best_cycle_equation,
                'score': cycle_score,
                'semantics': target_semantics
            })
            
            # Update best overall if better
            if cycle_score > best_overall_score:
                best_overall_equation = best_cycle_equation
                best_overall_score = cycle_score
            
            # Add to evolution sequence
            evolution_sequence.append(best_cycle_equation.clone())
            
            # Print progress
            print(f'Cycle {cycle+1}/{cycles}, Score: {cycle_score:.4f}, Best: {best_overall_score:.4f}')
        
        # Visualize the evolution
        self.visualize_equation_evolution(evolution_sequence)
        
        return best_overall_equation
    
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
                    name: eq.to_dict() for name, eq in self.expert['knowledge_base'].items()
                },
                'evaluation_metrics': {
                    name: str(func) for name, func in self.expert['evaluation_metrics'].items()
                }
            },
            'explorer': {
                'current_equation': self.explorer['current_equation'].to_dict() if self.explorer['current_equation'] else None,
                'exploration_history': [
                    {
                        'equation': entry['equation'].to_dict(),
                        'score': entry['score'],
                        'method': entry.get('method', 'standard')
                    }
                    for entry in self.explorer['exploration_history']
                ],
                'learning_rate': self.explorer['learning_rate']
            },
            'interaction_history': [
                {
                    'cycle': entry['cycle'],
                    'equation': entry['equation'].to_dict(),
                    'score': entry['score'],
                    'semantics': entry.get('semantics', [])
                }
                for entry in self.interaction_history
            ],
            'use_deep_learning': self.use_deep_learning,
            'use_reinforcement_learning': self.use_reinforcement_learning
        }
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(system_state, f, indent=2)
    
    def load_system_state(self, file_path: str) -> None:
        """
        Load the state of the expert-explorer system from a file.
        
        Args:
            file_path: Path to the system state file
        """
        # Load from file
        with open(file_path, 'r', encoding='utf-8') as f:
            system_state = json.load(f)
        
        # Restore expert knowledge base
        self.expert['knowledge_base'] = {}
        for name, eq_dict in system_state['expert']['knowledge_base'].items():
            self.expert['knowledge_base'][name] = GeneralShapeEquation.from_dict(eq_dict)
        
        # Restore explorer state
        if system_state['explorer']['current_equation']:
            self.explorer['current_equation'] = GeneralShapeEquation.from_dict(
                system_state['explorer']['current_equation']
            )
        
        self.explorer['exploration_history'] = []
        for entry in system_state['explorer']['exploration_history']:
            self.explorer['exploration_history'].append({
                'equation': GeneralShapeEquation.from_dict(entry['equation']),
                'score': entry['score'],
                'method': entry.get('method', 'standard')
            })
        
        self.explorer['learning_rate'] = system_state['explorer']['learning_rate']
        
        # Restore interaction history
        self.interaction_history = []
        for entry in system_state['interaction_history']:
            self.interaction_history.append({
                'cycle': entry['cycle'],
                'equation': GeneralShapeEquation.from_dict(entry['equation']),
                'score': entry['score'],
                'semantics': entry.get('semantics', [])
            })
        
        # Restore learning flags
        self.use_deep_learning = system_state.get('use_deep_learning', True)
        self.use_reinforcement_learning = system_state.get('use_reinforcement_learning', True)
        
        # Reinitialize adapters if needed
        if self.use_deep_learning:
            self.dl_adapter = EnhancedDeepLearningAdapter()
        
        if self.use_reinforcement_learning:
            self.rl_adapter = EnhancedReinforcementLearningAdapter()


# Utility functions for integration testing

def test_deep_learning_integration(equation: GeneralShapeEquation, 
                                 num_samples: int = 1000,
                                 num_epochs: int = 50) -> Dict[str, Any]:
    """
    Test the deep learning integration with a shape equation.
    
    Args:
        equation: The GeneralShapeEquation to test with
        num_samples: Number of samples to generate
        num_epochs: Number of training epochs
        
    Returns:
        Dictionary of test results
    """
    # Create a deep learning adapter
    dl_adapter = EnhancedDeepLearningAdapter()
    
    # Create a dataset
    dataset = ShapeEquationDataset([equation], samples_per_equation=num_samples)
    
    # Train the adapter
    training_history = dl_adapter.train_on_dataset(dataset, num_epochs=num_epochs)
    
    # Evaluate on the equation
    evaluation = dl_adapter.evaluate_on_equation(equation, num_samples=500)
    
    # Enhance the equation
    enhanced_equation = dl_adapter.enhance_equation_with_neural_correction(equation)
    
    # Create visualization
    fig = dl_adapter.visualize_predictions(equation)
    
    return {
        'adapter': dl_adapter,
        'training_history': training_history,
        'evaluation': evaluation,
        'enhanced_equation': enhanced_equation,
        'visualization': fig
    }


def test_reinforcement_learning_integration(equation: GeneralShapeEquation,
                                          num_episodes: int = 50,
                                          steps_per_episode: int = 10) -> Dict[str, Any]:
    """
    Test the reinforcement learning integration with a shape equation.
    
    Args:
        equation: The GeneralShapeEquation to test with
        num_episodes: Number of training episodes
        steps_per_episode: Steps per episode
        
    Returns:
        Dictionary of test results
    """
    # Create a reinforcement learning adapter
    rl_adapter = EnhancedReinforcementLearningAdapter()
    
    # Train the adapter
    rewards, best_equation = rl_adapter.train_on_equation(
        equation,
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode
    )
    
    # Create visualization
    fig = rl_adapter.plot_training_history()
    
    return {
        'adapter': rl_adapter,
        'rewards': rewards,
        'best_equation': best_equation,
        'visualization': fig
    }


def test_expert_explorer_integration(equation: GeneralShapeEquation,
                                   cycles: int = 5,
                                   steps_per_cycle: int = 10) -> Dict[str, Any]:
    """
    Test the expert-explorer integration with a shape equation.
    
    Args:
        equation: The GeneralShapeEquation to test with
        cycles: Number of interaction cycles
        steps_per_cycle: Steps per exploration cycle
        
    Returns:
        Dictionary of test results
    """
    # Create an expert-explorer system
    system = EnhancedExpertExplorerSystem(equation)
    
    # Initialize expert knowledge
    system.initialize_expert_knowledge()
    
    # Run expert-explorer interaction
    best_equation = system.expert_explorer_interaction_with_semantics(
        cycles=cycles,
        steps_per_cycle=steps_per_cycle
    )
    
    return {
        'system': system,
        'best_equation': best_equation,
        'interaction_history': system.interaction_history,
        'visualization_history': system.visualization_history
    }


if __name__ == "__main__":
    # Create a simple test equation
    test_equation = GSEFactory.create_basic_shape('circle', cx=0, cy=0, radius=1)
    
    # Test deep learning integration
    dl_results = test_deep_learning_integration(test_equation)
    
    # Test reinforcement learning integration
    rl_results = test_reinforcement_learning_integration(test_equation)
    
    # Test expert-explorer integration
    ee_results = test_expert_explorer_integration(test_equation)
    
    print("All integration tests completed successfully!")
