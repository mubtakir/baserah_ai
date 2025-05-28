#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revolutionary Expert/Explorer Agent for Baserah System - NO Traditional RL
وكيل الخبير/المستكشف الثوري لنظام بصيرة - بدون تعلم معزز تقليدي

Revolutionary replacement for traditional RL agent using:
- Expert/Explorer Systems instead of Traditional RL Algorithms
- Adaptive Equations instead of Neural Networks
- Basil's Methodology instead of Traditional Action Selection
- Revolutionary Wisdom instead of Traditional Rewards

استبدال ثوري لوكيل التعلم المعزز التقليدي باستخدام:
- أنظمة خبير/مستكشف بدلاً من خوارزميات التعلم المعزز التقليدية
- معادلات متكيفة بدلاً من الشبكات العصبية
- منهجية باسل بدلاً من اختيار الإجراءات التقليدي
- الحكمة الثورية بدلاً من المكافآت التقليدية

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 2.0.0 - Revolutionary Edition (NO Traditional RL/PyTorch)
Replaces: Traditional RL Agent
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
import uuid
import random
from collections import deque
import math

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import from revolutionary core modules
try:
    from mathematical_core.general_shape_equation import (
        GeneralShapeEquation,
        EquationType,
        LearningMode
    )
    REVOLUTIONARY_CORE_AVAILABLE = True
except ImportError:
    logging.warning("Revolutionary core modules not available, using placeholder implementation")
    REVOLUTIONARY_CORE_AVAILABLE = False

    # Define placeholder classes for revolutionary system
    class EquationType:
        BEHAVIOR = "behavior"
        MATHEMATICAL = "mathematical"
        REVOLUTIONARY = "revolutionary"

    class LearningMode:
        REINFORCEMENT = "reinforcement"
        ADAPTIVE = "adaptive"
        REVOLUTIONARY = "revolutionary"

# Configure logging
logger = logging.getLogger('learning.revolutionary.expert_explorer_agent')


class RevolutionaryDecisionStrategy(str, Enum):
    """استراتيجيات اتخاذ القرار الثوري - NO Traditional Action Selection"""
    EXPERT_GUIDED = "expert_guided"                    # موجه بالخبير
    EXPLORER_DRIVEN = "explorer_driven"                # مدفوع بالمستكشف
    BASIL_INTEGRATIVE = "basil_integrative"           # تكاملي باسل
    PHYSICS_INSPIRED = "physics_inspired"             # مستوحى من الفيزياء
    EQUATION_ADAPTIVE = "equation_adaptive"           # متكيف بالمعادلات
    WISDOM_BASED = "wisdom_based"                     # قائم على الحكمة
    REVOLUTIONARY_HYBRID = "revolutionary_hybrid"      # هجين ثوري


@dataclass
class RevolutionaryAgentConfig:
    """إعدادات الوكيل الثوري - NO Traditional RL Agent Config"""
    adaptation_rate: float = 0.01                     # معدل التكيف (بدلاً من learning_rate)
    wisdom_accumulation: float = 0.95                 # تراكم الحكمة (بدلاً من discount_factor)
    exploration_curiosity: float = 0.2                # فضول الاستكشاف (بدلاً من exploration_rate)
    curiosity_evolution: float = 0.99                 # تطور الفضول (بدلاً من exploration_decay)
    min_curiosity_level: float = 0.05                 # مستوى الفضول الأدنى
    evolution_batch_size: int = 32                    # حجم دفعة التطور (بدلاً من batch_size)
    wisdom_buffer_size: int = 5000                    # حجم مخزن الحكمة (بدلاً من buffer_size)
    evolution_frequency: int = 10                     # تكرار التطور (بدلاً من update_frequency)
    equation_evolution_frequency: int = 500           # تكرار تطور المعادلة
    decision_strategy: RevolutionaryDecisionStrategy = RevolutionaryDecisionStrategy.BASIL_INTEGRATIVE
    use_wisdom_signals: bool = True                   # استخدام إشارات الحكمة
    wisdom_signal_weight: float = 0.6                # وزن إشارة الحكمة
    use_adaptive_equations: bool = True               # استخدام المعادلات المتكيفة
    basil_methodology_weight: float = 0.4            # وزن منهجية باسل
    physics_thinking_weight: float = 0.3             # وزن التفكير الفيزيائي
    symbolic_evolution_weight: float = 0.3           # وزن التطور الرمزي
    equation_params: Dict[str, Any] = field(default_factory=dict)
    revolutionary_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RevolutionaryWisdomExperience:
    """تجربة الحكمة الثورية - NO Traditional RL Experience"""
    mathematical_situation: Any                       # الموقف الرياضي (بدلاً من state)
    expert_decision: Any                              # قرار الخبير (بدلاً من action)
    wisdom_gain: float                                # مكسب الحكمة (بدلاً من reward)
    evolved_situation: Any                            # الموقف المتطور (بدلاً من next_state)
    completion_status: bool                           # حالة الإكمال (بدلاً من done)
    wisdom_signal: float = 0.0                       # إشارة الحكمة (بدلاً من intrinsic_reward)
    basil_insights: Dict[str, Any] = field(default_factory=dict)
    equation_evolution: Dict[str, Any] = field(default_factory=dict)
    physics_principles: Dict[str, Any] = field(default_factory=dict)
    symbolic_transformations: Dict[str, Any] = field(default_factory=dict)
    expert_analysis: Dict[str, Any] = field(default_factory=dict)
    explorer_discoveries: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class RevolutionaryExpertExplorerAgent:
    """
    وكيل الخبير/المستكشف الثوري - NO Traditional RL Agent

    Revolutionary replacement for traditional RL agent using:
    - Expert/Explorer Systems instead of Traditional RL Algorithms
    - Adaptive Equations instead of Neural Networks
    - Basil's Methodology instead of Traditional Action Selection
    - Revolutionary Wisdom instead of Traditional Rewards
    """

    def __init__(self, config: RevolutionaryAgentConfig, situation_dimensions: int, decision_dimensions: int):
        """
        تهيئة الوكيل الثوري - NO Traditional RL Agent Initialization

        Args:
            config: إعدادات الوكيل الثوري
            situation_dimensions: أبعاد الموقف الرياضي
            decision_dimensions: أبعاد قرار الخبير
        """
        print("🌟" + "="*100 + "🌟")
        print("🚀 وكيل الخبير/المستكشف الثوري - بدون تعلم معزز تقليدي")
        print("⚡ أنظمة خبير/مستكشف + معادلات متكيفة + منهجية باسل")
        print("🧠 حكمة ثورية + تفكير فيزيائي بدلاً من الشبكات العصبية")
        print("🔬 بديل ثوري لوكيل التعلم المعزز التقليدي")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" + "="*100 + "🌟")

        self.config = config
        self.situation_dimensions = situation_dimensions
        self.decision_dimensions = decision_dimensions
        self.wisdom_buffer = deque(maxlen=config.wisdom_buffer_size)
        self.evolution_count = 0
        self.episode_count = 0
        self.total_wisdom_gain = 0.0
        self.total_wisdom_signals = 0.0

        # تهيئة المعادلة العامة للشكل الثورية
        if REVOLUTIONARY_CORE_AVAILABLE:
            self.revolutionary_equation = GeneralShapeEquation(
                equation_type=EquationType.MATHEMATICAL,
                learning_mode=LearningMode.ADAPTIVE
            )
        else:
            self.revolutionary_equation = None
            logger.warning("استخدام تطبيق المعادلة الثوري البديل")

        # تهيئة الأنظمة الثورية (بدلاً من الشبكات العصبية)
        self.expert_system = self._initialize_expert_system()
        self.explorer_system = self._initialize_explorer_system()
        self.adaptive_equations = self._initialize_adaptive_equations()
        self.basil_methodology_engine = self._initialize_basil_methodology()
        self.physics_thinking_engine = self._initialize_physics_thinking()
        self.wisdom_signal_processor = self._initialize_wisdom_signal_processor()

        # تهيئة مكونات المعادلة الثورية
        self._initialize_revolutionary_equation_components()

        print("✅ تم تهيئة الوكيل الثوري بنجاح!")
        print(f"🧠 نظام الخبير: نشط")
        print(f"🔍 نظام المستكشف: نشط")
        print(f"🧮 المعادلات المتكيفة: {len(self.adaptive_equations)}")
        print(f"🌟 محرك منهجية باسل: نشط")
        print(f"🔬 محرك التفكير الفيزيائي: نشط")
        print(f"💡 معالج إشارات الحكمة: نشط")

        # تهيئة متغيرات التتبع
        self.total_wisdom_accumulated = 0.0
        self.decision_history = []

    def _initialize_expert_system(self) -> Dict[str, Any]:
        """تهيئة نظام الخبير الثوري"""
        return {
            "basil_methodology_expert": {
                "wisdom_threshold": 0.8,
                "confidence_factor": 0.9,
                "decision_quality": 0.85
            },
            "physics_thinking_expert": {
                "resonance_factor": 0.75,
                "principle_application": 0.8,
                "coherence_measure": 0.9
            }
        }

    def _initialize_explorer_system(self) -> Dict[str, Any]:
        """تهيئة نظام المستكشف الثوري"""
        return {
            "curiosity_engine": {
                "novelty_detection": 0.8,
                "discovery_potential": 0.75,
                "exploration_depth": 0.7
            },
            "pattern_explorer": {
                "pattern_recognition": 0.85,
                "anomaly_detection": 0.8,
                "insight_generation": 0.9
            }
        }

    def _initialize_adaptive_equations(self) -> Dict[str, Any]:
        """تهيئة المعادلات المتكيفة الثورية"""
        return {
            "basil_decision_equation": {
                "wisdom_coefficient": 0.9,
                "insight_factor": 0.85,
                "methodology_weight": 1.0,
                "adaptive_threshold": 0.8
            },
            "physics_resonance_equation": {
                "resonance_frequency": 0.75,
                "harmonic_factor": 0.8,
                "coherence_measure": 0.9,
                "stability_coefficient": 0.85
            }
        }

    def _initialize_basil_methodology(self) -> Dict[str, Any]:
        """تهيئة محرك منهجية باسل الثوري"""
        return {
            "integrative_thinking": {
                "connection_depth": 0.9,
                "synthesis_quality": 0.85,
                "holistic_perspective": 0.9
            },
            "mathematical_reasoning": {
                "equation_mastery": 0.95,
                "symbolic_manipulation": 0.9,
                "logical_coherence": 0.85
            }
        }

    def _initialize_physics_thinking(self) -> Dict[str, Any]:
        """تهيئة محرك التفكير الفيزيائي الثوري"""
        return {
            "resonance_principles": {
                "frequency_matching": 0.8,
                "harmonic_analysis": 0.75,
                "stability_assessment": 0.85
            },
            "field_dynamics": {
                "field_strength": 0.8,
                "interaction_analysis": 0.75,
                "energy_conservation": 0.9
            }
        }

    def _initialize_wisdom_signal_processor(self) -> Dict[str, Any]:
        """تهيئة معالج إشارات الحكمة الثوري"""
        return {
            "wisdom_detection": {
                "signal_strength": 0.9,
                "noise_filtering": 0.85,
                "pattern_extraction": 0.8
            },
            "signal_amplification": {
                "amplification_factor": 0.8,
                "clarity_enhancement": 0.75,
                "coherence_boost": 0.9
            }
        }

    def make_revolutionary_decision(self, situation: np.ndarray) -> int:
        """
        اتخاذ قرار ثوري - NO Traditional RL Action Selection

        Args:
            situation: الموقف الحالي

        Returns:
            القرار الثوري
        """
        # الاستكشاف الثوري
        if random.random() < self.config.exploration_curiosity:
            return self._make_exploratory_decision(situation)

        # القرار الثوري الموجه
        return self._make_guided_revolutionary_decision(situation)

    def _make_exploratory_decision(self, situation: np.ndarray) -> int:
        """اتخاذ قرار استكشافي ثوري"""
        explorer_factors = self.explorer_system["curiosity_engine"]
        novelty_score = explorer_factors["novelty_detection"]
        discovery_potential = explorer_factors["discovery_potential"]

        exploration_value = novelty_score * discovery_potential
        decision_index = int(exploration_value * self.decision_dimensions) % self.decision_dimensions

        return decision_index

    def _make_guided_revolutionary_decision(self, situation: np.ndarray) -> int:
        """اتخاذ قرار ثوري موجه"""
        # تطبيق الاستراتيجية الثورية
        if self.config.decision_strategy == RevolutionaryDecisionStrategy.EXPERT_GUIDED:
            return self._apply_expert_guidance(situation)
        elif self.config.decision_strategy == RevolutionaryDecisionStrategy.BASIL_INTEGRATIVE:
            return self._apply_basil_integration(situation)
        elif self.config.decision_strategy == RevolutionaryDecisionStrategy.PHYSICS_INSPIRED:
            return self._apply_physics_inspiration(situation)
        else:
            return self._apply_basil_integration(situation)

    def _apply_expert_guidance(self, situation: np.ndarray) -> int:
        """تطبيق التوجيه الخبير"""
        expert_factors = self.expert_system["basil_methodology_expert"]
        wisdom_threshold = expert_factors["wisdom_threshold"]
        decision_quality = expert_factors["decision_quality"]

        decision_value = wisdom_threshold * decision_quality
        return int(decision_value * self.decision_dimensions) % self.decision_dimensions

    def _apply_basil_integration(self, situation: np.ndarray) -> int:
        """تطبيق التكامل الثوري لباسل"""
        basil_factors = self.basil_methodology_engine["integrative_thinking"]
        connection_depth = basil_factors["connection_depth"]
        synthesis_quality = basil_factors["synthesis_quality"]

        integration_value = (connection_depth + synthesis_quality) / 2.0
        return int(integration_value * self.decision_dimensions) % self.decision_dimensions

    def _apply_physics_inspiration(self, situation: np.ndarray) -> int:
        """تطبيق الإلهام الفيزيائي"""
        physics_factors = self.physics_thinking_engine["resonance_principles"]
        frequency_matching = physics_factors["frequency_matching"]
        stability_assessment = physics_factors["stability_assessment"]

        physics_value = frequency_matching * stability_assessment
        return int(physics_value * self.decision_dimensions) % self.decision_dimensions

    def learn_from_wisdom(self, situation: np.ndarray, decision: int, wisdom_gain: float,
                         evolved_situation: np.ndarray, completion_status: bool,
                         info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        التعلم من الحكمة - NO Traditional RL Learning

        Args:
            situation: الموقف الحالي
            decision: القرار المتخذ
            wisdom_gain: مكسب الحكمة
            evolved_situation: الموقف المتطور
            completion_status: حالة الإكمال
            info: معلومات إضافية

        Returns:
            إحصائيات التعلم
        """
        # حساب إشارة الحكمة
        wisdom_signal = 0.0
        if self.config.use_wisdom_signals:
            wisdom_signal = self._compute_wisdom_signal(situation, decision, evolved_situation)

        # دمج الحكمة
        combined_wisdom = wisdom_gain + self.config.wisdom_signal_weight * wisdom_signal

        # إنشاء تجربة حكمة ثورية
        experience = RevolutionaryWisdomExperience(
            mathematical_situation=situation,
            expert_decision=decision,
            wisdom_gain=wisdom_gain,
            evolved_situation=evolved_situation,
            completion_status=completion_status,
            wisdom_signal=wisdom_signal,
            expert_analysis=info or {}
        )

        # إضافة التجربة لمخزن الحكمة
        self.wisdom_buffer.append(experience)

        # تحديث عداد التطور والحكمة الإجمالية
        self.evolution_count += 1
        self.total_wisdom_gain += wisdom_gain
        self.total_wisdom_signals += wisdom_signal

        # تحديث عداد الحلقات إذا اكتملت
        if completion_status:
            self.episode_count += 1

            # تطور الفضول
            curiosity = self.config.exploration_curiosity
            curiosity = max(self.config.min_curiosity_level, curiosity * self.config.curiosity_evolution)
            self.config.exploration_curiosity = curiosity

        # حساب جودة القرار
        decision_quality = self._calculate_decision_quality(wisdom_gain, wisdom_signal)

        # تسجيل القرار
        self.decision_history.append({
            "decision": decision,
            "quality": decision_quality,
            "wisdom_gain": wisdom_gain,
            "wisdom_signal": wisdom_signal
        })

        # تطور النظام إذا حان الوقت
        evolution_info = {}
        if len(self.wisdom_buffer) >= self.config.evolution_batch_size and self.evolution_count % self.config.evolution_frequency == 0:
            evolution_info = self._evolve_revolutionary_systems()

        # إرجاع إحصائيات التعلم
        return {
            "evolution_step": self.evolution_count,
            "episode": self.episode_count,
            "wisdom_gain": wisdom_gain,
            "wisdom_signal": wisdom_signal,
            "combined_wisdom": combined_wisdom,
            "decision_quality": decision_quality,
            "exploration_curiosity": self.config.exploration_curiosity,
            "buffer_size": len(self.wisdom_buffer),
            **evolution_info
        }

    def _compute_wisdom_signal(self, situation: np.ndarray, decision: int, evolved_situation: np.ndarray) -> float:
        """حساب إشارة الحكمة"""
        # حساب التغيير في الموقف
        situation_change = np.linalg.norm(evolved_situation - situation)

        # تطبيق معالج إشارات الحكمة
        processor_factors = self.wisdom_signal_processor["wisdom_detection"]
        signal_strength = processor_factors["signal_strength"]

        # حساب إشارة الحكمة
        wisdom_signal = situation_change * signal_strength

        return min(1.0, wisdom_signal)  # تقييد القيمة

    def _calculate_decision_quality(self, wisdom_gain: float, wisdom_signal: float) -> float:
        """حساب جودة القرار"""
        # دمج مكسب الحكمة وإشارة الحكمة
        quality = (wisdom_gain + wisdom_signal) / 2.0

        # تطبيق عوامل الجودة من نظام الخبير
        expert_factors = self.expert_system["basil_methodology_expert"]
        quality_factor = expert_factors["decision_quality"]

        return min(1.0, quality * quality_factor)

    def _evolve_revolutionary_systems(self) -> Dict[str, Any]:
        """تطور الأنظمة الثورية"""
        # أخذ عينة من تجارب الحكمة
        batch = random.sample(self.wisdom_buffer, min(self.config.evolution_batch_size, len(self.wisdom_buffer)))

        # حساب متوسط مكسب الحكمة
        avg_wisdom = np.mean([exp.wisdom_gain for exp in batch])

        # تطور معاملات الأنظمة
        for system_name, system_params in self.adaptive_equations.items():
            for param_name, param_value in system_params.items():
                # تطور المعامل بناءً على الحكمة
                evolution_factor = 1.0 + (avg_wisdom - 0.5) * self.config.adaptation_rate
                new_value = param_value * evolution_factor

                # تقييد القيم
                new_value = max(0.1, min(1.0, new_value))
                self.adaptive_equations[system_name][param_name] = new_value

        return {
            "systems_evolved": True,
            "avg_wisdom": avg_wisdom,
            "evolution_factor": evolution_factor
        }

    def _initialize_equation_components(self) -> None:
        """Initialize the components of the General Shape Equation."""
        # Add basic components for state and action representation
        self.equation.add_component("state_value", "value_function(state)")
        self.equation.add_component("action_value", "q_function(state, action)")
        self.equation.add_component("policy", "policy_function(state)")

        # Add components for reinforcement learning
        self.equation.add_component("q_target", "reward + discount_factor * max_next_q_value * (1 - done)")
        self.equation.add_component("td_error", "q_target - q_value")
        self.equation.add_component("loss", "mean_squared_error(q_target, q_value)")

        # Add components for action selection
        self.equation.add_component("epsilon_greedy", "random() < epsilon ? random_action() : argmax(q_values)")
        self.equation.add_component("boltzmann", "softmax(q_values / temperature)")
        self.equation.add_component("ucb", "q_values + exploration_bonus * sqrt(log(total_steps) / (1 + action_counts))")

        # Add components for intrinsic motivation
        self.equation.add_component("combined_reward", "extrinsic_reward + intrinsic_reward_weight * intrinsic_reward")

        # Set variables
        self.equation.set_variable("discount_factor", self.config.discount_factor)
        self.equation.set_variable("epsilon", self.config.exploration_rate)
        self.equation.set_variable("temperature", 1.0)
        self.equation.set_variable("exploration_bonus", 2.0)
        self.equation.set_variable("intrinsic_reward_weight", self.config.intrinsic_reward_weight)

    def _initialize_models(self) -> None:
        """Initialize neural network models."""
        # Create Q-network
        class QNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim=128):
                super().__init__()
                self.fc1 = nn.Linear(state_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, action_dim)

            def forward(self, state):
                x = F.relu(self.fc1(state))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        # Initialize Q-network and target network
        self.model = QNetwork(self.state_dim, self.action_dim)
        self.target_model = QNetwork(self.state_dim, self.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action based on the current state.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        # Convert state to tensor if using PyTorch
        if TORCH_AVAILABLE and self.model is not None:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Select action based on strategy
        if self.config.action_selection == ActionSelectionStrategy.GREEDY:
            if TORCH_AVAILABLE and self.model is not None:
                with torch.no_grad():
                    q_values = self.model(state_tensor).squeeze(0).numpy()
                return np.argmax(q_values)
            else:
                return random.randint(0, self.action_dim - 1)

        elif self.config.action_selection == ActionSelectionStrategy.EPSILON_GREEDY:
            if random.random() < self.equation.get_variable("epsilon"):
                return random.randint(0, self.action_dim - 1)
            elif TORCH_AVAILABLE and self.model is not None:
                with torch.no_grad():
                    q_values = self.model(state_tensor).squeeze(0).numpy()
                return np.argmax(q_values)
            else:
                return random.randint(0, self.action_dim - 1)

        elif self.config.action_selection == ActionSelectionStrategy.BOLTZMANN:
            if TORCH_AVAILABLE and self.model is not None:
                with torch.no_grad():
                    q_values = self.model(state_tensor).squeeze(0).numpy()
                temperature = self.equation.get_variable("temperature") or 1.0
                exp_q = np.exp(q_values / temperature)
                probs = exp_q / np.sum(exp_q)
                return np.random.choice(self.action_dim, p=probs)
            else:
                return random.randint(0, self.action_dim - 1)

        elif self.config.action_selection == ActionSelectionStrategy.UCB:
            # UCB requires action counts, which we track in custom_params
            if "action_counts" not in self.config.custom_params:
                self.config.custom_params["action_counts"] = np.ones(self.action_dim)

            if TORCH_AVAILABLE and self.model is not None:
                with torch.no_grad():
                    q_values = self.model(state_tensor).squeeze(0).numpy()

                exploration_bonus = self.equation.get_variable("exploration_bonus") or 2.0
                total_steps = self.step_count + 1
                action_counts = self.config.custom_params["action_counts"]

                ucb_values = q_values + exploration_bonus * np.sqrt(np.log(total_steps) / action_counts)
                action = np.argmax(ucb_values)

                # Update action count
                self.config.custom_params["action_counts"][action] += 1

                return action
            else:
                return random.randint(0, self.action_dim - 1)

        elif self.config.action_selection == ActionSelectionStrategy.EQUATION_BASED:
            if self.config.use_equation:
                # Use the equation to select an action
                state_dict = {"state": state}
                result = self.equation.evaluate(state_dict)

                if "policy" in result:
                    # Convert policy result to an action
                    policy_value = result["policy"]
                    if isinstance(policy_value, (int, float)):
                        return int(policy_value) % self.action_dim

                # Fallback to epsilon-greedy
                if random.random() < self.equation.get_variable("epsilon"):
                    return random.randint(0, self.action_dim - 1)
                elif TORCH_AVAILABLE and self.model is not None:
                    with torch.no_grad():
                        q_values = self.model(state_tensor).squeeze(0).numpy()
                    return np.argmax(q_values)
                else:
                    return random.randint(0, self.action_dim - 1)
            else:
                # Fallback to epsilon-greedy
                return self.select_action_epsilon_greedy(state)

        elif self.config.action_selection == ActionSelectionStrategy.HYBRID:
            # Combine multiple strategies
            strategies = [
                ActionSelectionStrategy.EPSILON_GREEDY,
                ActionSelectionStrategy.BOLTZMANN,
                ActionSelectionStrategy.UCB
            ]

            # Select a strategy based on current episode
            strategy_index = self.episode_count % len(strategies)
            selected_strategy = strategies[strategy_index]

            # Temporarily change the strategy and select action
            original_strategy = self.config.action_selection
            self.config.action_selection = selected_strategy
            action = self.select_action(state)
            self.config.action_selection = original_strategy

            return action

        else:
            # Default to epsilon-greedy
            return self.select_action_epsilon_greedy(state)

    def select_action_epsilon_greedy(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy strategy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        if random.random() < self.equation.get_variable("epsilon"):
            return random.randint(0, self.action_dim - 1)
        elif TORCH_AVAILABLE and self.model is not None:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(0).numpy()
            return np.argmax(q_values)
        else:
            return random.randint(0, self.action_dim - 1)

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Learn from a single experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            info: Additional information

        Returns:
            Dictionary with learning statistics
        """
        # Compute intrinsic reward if enabled
        intrinsic_reward = 0.0
        if self.config.use_intrinsic_rewards:
            intrinsic_reward = self.compute_intrinsic_reward(state, action, next_state)

        # Combine rewards
        combined_reward = reward + self.config.intrinsic_reward_weight * intrinsic_reward

        # Create experience
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            intrinsic_reward=intrinsic_reward,
            info=info or {}
        )

        # Add experience to buffer
        self.experience_buffer.append(experience)

        # Update step count and total rewards
        self.step_count += 1
        self.total_reward += reward
        self.total_intrinsic_reward += intrinsic_reward

        # Update episode count if episode is done
        if done:
            self.episode_count += 1

            # Decay exploration rate
            epsilon = self.equation.get_variable("epsilon") or self.config.exploration_rate
            epsilon = max(self.config.min_exploration_rate, epsilon * self.config.exploration_decay)
            self.equation.set_variable("epsilon", epsilon)

        # Check if it's time to update the model
        update_info = {}
        if len(self.experience_buffer) >= self.config.batch_size and self.step_count % self.config.update_frequency == 0:
            update_info = self._update_model()

        # Check if it's time to update the target model
        if TORCH_AVAILABLE and self.model is not None and self.target_model is not None and self.step_count % self.config.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            update_info["target_updated"] = True

        # Return learning statistics
        return {
            "step": self.step_count,
            "episode": self.episode_count,
            "reward": reward,
            "intrinsic_reward": intrinsic_reward,
            "combined_reward": combined_reward,
            "epsilon": self.equation.get_variable("epsilon"),
            "buffer_size": len(self.experience_buffer),
            **update_info
        }

    def compute_intrinsic_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        Compute intrinsic reward for a state-action-next_state transition.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Intrinsic reward value
        """
        # Convert action to one-hot encoding for the intrinsic reward component
        action_one_hot = np.zeros(self.action_dim)
        action_one_hot[action] = 1.0

        # Compute combined intrinsic reward
        intrinsic_reward = self.intrinsic_reward.compute_combined_reward(state, action_one_hot, next_state)

        return intrinsic_reward

    def _update_model(self) -> Dict[str, Any]:
        """
        Update the model based on experiences in the buffer.

        Returns:
            Dictionary with update statistics
        """
        if not TORCH_AVAILABLE or self.model is None or self.target_model is None:
            return {"updated": False, "reason": "PyTorch or models not available"}

        # Sample batch from buffer
        batch = random.sample(self.experience_buffer, self.config.batch_size)

        # Convert batch to tensors
        states = torch.FloatTensor([exp.state for exp in batch])
        actions = torch.LongTensor([exp.action for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.FloatTensor([exp.next_state for exp in batch])
        dones = torch.FloatTensor([exp.done for exp in batch])
        intrinsic_rewards = torch.FloatTensor([exp.intrinsic_reward for exp in batch])

        # Compute combined rewards
        combined_rewards = rewards + self.config.intrinsic_reward_weight * intrinsic_rewards

        # Compute Q values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]

        # Compute target Q values
        target_q_values = combined_rewards + self.config.discount_factor * next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "updated": True,
            "loss": loss.item(),
            "mean_q": q_values.mean().item(),
            "mean_target_q": target_q_values.mean().item()
        }

    def save(self, directory: str) -> None:
        """
        Save the agent to a directory.

        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)

        # Save configuration
        with open(os.path.join(directory, "config.json"), 'w', encoding='utf-8') as f:
            config_dict = copy.deepcopy(vars(self.config))
            config_dict["action_selection"] = self.config.action_selection.value
            json.dump(config_dict, f, indent=2)

        # Save model
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))

        # Save equation
        with open(os.path.join(directory, "equation.json"), 'w', encoding='utf-8') as f:
            f.write(self.equation.to_json())

        # Save statistics
        with open(os.path.join(directory, "stats.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "step_count": self.step_count,
                "episode_count": self.episode_count,
                "total_reward": self.total_reward,
                "total_intrinsic_reward": self.total_intrinsic_reward
            }, f, indent=2)

    def load(self, directory: str) -> None:
        """
        Load the agent from a directory.

        Args:
            directory: Directory to load from
        """
        # Load configuration
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if key == "action_selection":
                        self.config.action_selection = ActionSelectionStrategy(value)
                    else:
                        setattr(self.config, key, value)

        # Load model
        if TORCH_AVAILABLE and self.model is not None:
            model_path = os.path.join(directory, "model.pt")
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                self.target_model.load_state_dict(self.model.state_dict())

        # Load equation
        equation_path = os.path.join(directory, "equation.json")
        if os.path.exists(equation_path):
            with open(equation_path, 'r', encoding='utf-8') as f:
                equation_json = f.read()
                self.equation = GeneralShapeEquation.from_json(equation_json)
                self.intrinsic_reward = IntrinsicRewardComponent(self.equation)

        # Load statistics
        stats_path = os.path.join(directory, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                self.step_count = stats.get("step_count", 0)
                self.episode_count = stats.get("episode_count", 0)
                self.total_reward = stats.get("total_reward", 0.0)
                self.total_intrinsic_reward = stats.get("total_intrinsic_reward", 0.0)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create agent configuration
    config = AgentConfig(
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=0.1,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        batch_size=64,
        buffer_size=10000,
        update_frequency=4,
        target_update_frequency=1000,
        action_selection=ActionSelectionStrategy.EPSILON_GREEDY,
        use_intrinsic_rewards=True,
        intrinsic_reward_weight=0.5,
        use_equation=True
    )

    # Create agent
    state_dim = 4  # Example: CartPole state dimension
    action_dim = 2  # Example: CartPole action dimension
    agent = InnovativeRLAgent(config, state_dim, action_dim)

    # Example learning step
    state = np.random.rand(state_dim)
    action = agent.select_action(state)
    reward = 1.0
    next_state = np.random.rand(state_dim)
    done = False

    learn_info = agent.learn(state, action, reward, next_state, done)

    print("Learning step info:", learn_info)

    # Save agent
    agent.save("agent_test")

    print("Agent saved to 'agent_test' directory")
