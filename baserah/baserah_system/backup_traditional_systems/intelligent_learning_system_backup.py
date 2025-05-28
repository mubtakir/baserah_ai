#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Adaptive Learning System for Basira

This module implements an advanced adaptive learning system that evolves
and improves based on interactions, embodying the Islamic principle of
continuous learning and self-improvement.

Author: Basira System Development Team
Version: 3.0.0 (Adaptive Intelligence)
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import pickle
import math

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.general_shape_equation import GeneralShapeEquation, EquationType, LearningMode
    from wisdom_engine.basira_wisdom_core import BasiraWisdomCore
    from wisdom_engine.deep_thinking_engine import DeepThinkingEngine
except ImportError as e:
    logging.warning(f"Some modules not available: {e}")

# Configure logging
logger = logging.getLogger('adaptive_learning.intelligent_learning_system')


class LearningStrategy(Enum):
    """Learning strategies inspired by Islamic educational principles"""
    GRADUAL = "تدريجي"           # Gradual learning (التدرج)
    REPETITIVE = "تكراري"       # Repetitive learning (التكرار)
    EXPERIENTIAL = "تجريبي"     # Experiential learning (التجربة)
    REFLECTIVE = "تأملي"        # Reflective learning (التأمل)
    COLLABORATIVE = "تشاركي"    # Collaborative learning (التشارك)
    INTUITIVE = "حدسي"          # Intuitive learning (الحدس)
    HOLISTIC = "شمولي"          # Holistic learning (الشمولية)


class KnowledgeType(Enum):
    """Types of knowledge in Islamic epistemology"""
    REVEALED = "منقول"           # Revealed knowledge (النقل)
    RATIONAL = "معقول"          # Rational knowledge (العقل)
    EXPERIENTIAL = "مكتسب"      # Acquired knowledge (التجربة)
    INTUITIVE = "كشفي"          # Intuitive knowledge (الكشف)
    PRACTICAL = "عملي"          # Practical knowledge (العمل)


@dataclass
class LearningPattern:
    """Represents a learning pattern discovered by the system"""
    pattern_id: str
    pattern_type: str
    description: str
    
    # Pattern characteristics
    frequency: int = 0
    success_rate: float = 0.0
    contexts: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Learning metrics
    effectiveness_score: float = 0.0
    adaptability_score: float = 0.0
    retention_rate: float = 0.0
    
    # Temporal aspects
    discovery_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    evolution_history: List[Dict] = field(default_factory=list)


@dataclass
class LearningExperience:
    """Represents a learning experience with outcomes"""
    experience_id: str
    user_query: str
    system_response: str
    
    # Context information
    learning_context: str
    knowledge_domain: str
    difficulty_level: str
    
    # Interaction data
    user_feedback: Optional[str] = None
    satisfaction_score: Optional[float] = None
    understanding_level: Optional[float] = None
    
    # Learning outcomes
    knowledge_gained: List[str] = field(default_factory=list)
    skills_developed: List[str] = field(default_factory=list)
    insights_generated: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = ""
    learning_strategy_used: Optional[LearningStrategy] = None


class IntelligentLearningSystem:
    """
    Advanced Adaptive Learning System that embodies Islamic principles
    of continuous learning, wisdom acquisition, and knowledge application
    """
    
    def __init__(self):
        """Initialize the Intelligent Learning System"""
        self.logger = logging.getLogger('adaptive_learning.intelligent_learning_system.main')
        
        # Initialize core components
        self.general_equation = GeneralShapeEquation(
            equation_type=EquationType.LEARNING,
            learning_mode=LearningMode.ADAPTIVE
        )
        
        # Initialize wisdom and thinking engines
        try:
            self.wisdom_core = BasiraWisdomCore()
            self.thinking_engine = DeepThinkingEngine()
        except:
            self.wisdom_core = None
            self.thinking_engine = None
            self.logger.warning("Wisdom and thinking engines not available")
        
        # Learning data structures
        self.learning_patterns = {}
        self.learning_experiences = []
        self.user_profiles = {}
        self.knowledge_graph = {}
        
        # Adaptive mechanisms
        self.adaptation_algorithms = self._initialize_adaptation_algorithms()
        self.learning_strategies = self._initialize_learning_strategies()
        self.feedback_processors = self._initialize_feedback_processors()
        
        # Performance metrics
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_learnings": 0,
            "pattern_discoveries": 0,
            "adaptation_events": 0,
            "user_satisfaction": 0.0
        }
        
        # Load existing learning data
        self._load_learning_data()
        
        self.logger.info("Intelligent Learning System initialized with adaptive capabilities")
    
    def _initialize_adaptation_algorithms(self) -> Dict[str, Any]:
        """Initialize adaptation algorithms"""
        
        return {
            "pattern_recognition": self._recognize_learning_patterns,
            "strategy_adaptation": self._adapt_learning_strategy,
            "content_personalization": self._personalize_content,
            "difficulty_adjustment": self._adjust_difficulty,
            "feedback_integration": self._integrate_feedback,
            "knowledge_evolution": self._evolve_knowledge_base,
            "performance_optimization": self._optimize_performance
        }
    
    def _initialize_learning_strategies(self) -> Dict[LearningStrategy, Dict]:
        """Initialize learning strategies based on Islamic educational principles"""
        
        return {
            LearningStrategy.GRADUAL: {
                "description": "التعلم التدريجي - البناء المرحلي للمعرفة",
                "method": self._gradual_learning,
                "principles": ["البناء على المعروف", "التدرج من البسيط للمعقد", "التأكد من الفهم"]
            },
            
            LearningStrategy.REPETITIVE: {
                "description": "التعلم التكراري - ترسيخ المعرفة بالتكرار",
                "method": self._repetitive_learning,
                "principles": ["التكرار للإتقان", "التنويع في التكرار", "الربط بالسياق"]
            },
            
            LearningStrategy.EXPERIENTIAL: {
                "description": "التعلم التجريبي - التعلم من التجربة والممارسة",
                "method": self._experiential_learning,
                "principles": ["التطبيق العملي", "التعلم من الأخطاء", "الربط بالواقع"]
            },
            
            LearningStrategy.REFLECTIVE: {
                "description": "التعلم التأملي - التفكر والتدبر في المعرفة",
                "method": self._reflective_learning,
                "principles": ["التأمل العميق", "استخراج الحكم", "الربط بالمعاني الكبرى"]
            },
            
            LearningStrategy.COLLABORATIVE: {
                "description": "التعلم التشاركي - التعلم من خلال التفاعل والحوار",
                "method": self._collaborative_learning,
                "principles": ["تبادل المعرفة", "الحوار البناء", "التعلم الجماعي"]
            }
        }
    
    def _initialize_feedback_processors(self) -> Dict[str, Any]:
        """Initialize feedback processing mechanisms"""
        
        return {
            "explicit_feedback": self._process_explicit_feedback,
            "implicit_feedback": self._process_implicit_feedback,
            "behavioral_feedback": self._process_behavioral_feedback,
            "performance_feedback": self._process_performance_feedback,
            "emotional_feedback": self._process_emotional_feedback
        }
    
    def adaptive_learn(self, user_query: str, user_id: str = "default", 
                      context: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform adaptive learning based on user query and context
        
        Args:
            user_query: User's question or learning request
            user_id: Unique identifier for the user
            context: Optional context information
            
        Returns:
            Adaptive learning response with personalized content
        """
        
        # Get or create user profile
        user_profile = self._get_user_profile(user_id)
        
        # Analyze the learning request
        learning_analysis = self._analyze_learning_request(user_query, context, user_profile)
        
        # Select optimal learning strategy
        optimal_strategy = self._select_learning_strategy(learning_analysis, user_profile)
        
        # Generate adaptive response
        adaptive_response = self._generate_adaptive_response(
            user_query, learning_analysis, optimal_strategy, user_profile
        )
        
        # Create learning experience record
        experience = LearningExperience(
            experience_id=self._generate_experience_id(),
            user_query=user_query,
            system_response=adaptive_response["content"],
            learning_context=context or "general",
            knowledge_domain=learning_analysis["domain"],
            difficulty_level=learning_analysis["difficulty"],
            learning_strategy_used=optimal_strategy
        )
        
        # Store experience
        self.learning_experiences.append(experience)
        
        # Update user profile
        self._update_user_profile(user_id, experience, learning_analysis)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Trigger adaptation if needed
        self._trigger_adaptation_if_needed()
        
        return {
            "response": adaptive_response,
            "learning_strategy": optimal_strategy.value,
            "personalization_level": adaptive_response.get("personalization_level", 0.5),
            "experience_id": experience.experience_id
        }
    
    def process_feedback(self, experience_id: str, feedback_data: Dict[str, Any]) -> None:
        """
        Process user feedback to improve learning
        
        Args:
            experience_id: ID of the learning experience
            feedback_data: Feedback information from user
        """
        
        # Find the experience
        experience = self._find_experience(experience_id)
        if not experience:
            self.logger.warning(f"Experience {experience_id} not found")
            return
        
        # Update experience with feedback
        experience.user_feedback = feedback_data.get("feedback_text")
        experience.satisfaction_score = feedback_data.get("satisfaction", 0.5)
        experience.understanding_level = feedback_data.get("understanding", 0.5)
        
        # Process different types of feedback
        for feedback_type, processor in self.feedback_processors.items():
            if feedback_type in feedback_data:
                processor(experience, feedback_data[feedback_type])
        
        # Learn from feedback
        self._learn_from_feedback(experience, feedback_data)
        
        # Update learning patterns
        self._update_learning_patterns(experience, feedback_data)
        
        self.logger.info(f"Processed feedback for experience {experience_id}")
    
    def discover_learning_patterns(self) -> List[LearningPattern]:
        """
        Discover new learning patterns from accumulated experiences
        
        Returns:
            List of newly discovered learning patterns
        """
        
        new_patterns = []
        
        # Analyze interaction sequences
        sequence_patterns = self._analyze_interaction_sequences()
        new_patterns.extend(sequence_patterns)
        
        # Analyze success patterns
        success_patterns = self._analyze_success_patterns()
        new_patterns.extend(success_patterns)
        
        # Analyze difficulty progression patterns
        difficulty_patterns = self._analyze_difficulty_patterns()
        new_patterns.extend(difficulty_patterns)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns()
        new_patterns.extend(temporal_patterns)
        
        # Store new patterns
        for pattern in new_patterns:
            self.learning_patterns[pattern.pattern_id] = pattern
        
        # Update metrics
        self.performance_metrics["pattern_discoveries"] += len(new_patterns)
        
        self.logger.info(f"Discovered {len(new_patterns)} new learning patterns")
        return new_patterns
    
    def _analyze_learning_request(self, query: str, context: Optional[str], 
                                 user_profile: Dict) -> Dict[str, Any]:
        """Analyze the learning request to understand user needs"""
        
        analysis = {
            "query": query,
            "domain": self._identify_knowledge_domain(query),
            "difficulty": self._assess_difficulty_level(query, user_profile),
            "learning_intent": self._identify_learning_intent(query),
            "cognitive_load": self._estimate_cognitive_load(query),
            "prior_knowledge": self._assess_prior_knowledge(query, user_profile),
            "learning_style": user_profile.get("preferred_learning_style", "mixed")
        }
        
        # Add context analysis if available
        if context:
            analysis["context_relevance"] = self._analyze_context_relevance(query, context)
            analysis["context_complexity"] = self._assess_context_complexity(context)
        
        return analysis
    
    def _select_learning_strategy(self, analysis: Dict, user_profile: Dict) -> LearningStrategy:
        """Select optimal learning strategy based on analysis and user profile"""
        
        # Get user's historical performance with different strategies
        strategy_performance = user_profile.get("strategy_performance", {})
        
        # Calculate strategy scores
        strategy_scores = {}
        
        for strategy in LearningStrategy:
            score = 0.0
            
            # Historical performance weight
            if strategy.value in strategy_performance:
                score += strategy_performance[strategy.value] * 0.4
            
            # Domain suitability
            domain_suitability = self._calculate_domain_suitability(strategy, analysis["domain"])
            score += domain_suitability * 0.3
            
            # Difficulty appropriateness
            difficulty_appropriateness = self._calculate_difficulty_appropriateness(
                strategy, analysis["difficulty"]
            )
            score += difficulty_appropriateness * 0.2
            
            # Learning style match
            style_match = self._calculate_style_match(strategy, analysis["learning_style"])
            score += style_match * 0.1
            
            strategy_scores[strategy] = score
        
        # Select strategy with highest score
        optimal_strategy = max(strategy_scores, key=strategy_scores.get)
        
        self.logger.info(f"Selected learning strategy: {optimal_strategy.value}")
        return optimal_strategy
    
    def _generate_adaptive_response(self, query: str, analysis: Dict, 
                                   strategy: LearningStrategy, user_profile: Dict) -> Dict[str, Any]:
        """Generate adaptive response based on selected strategy"""
        
        # Get strategy method
        strategy_info = self.learning_strategies[strategy]
        strategy_method = strategy_info["method"]
        
        # Generate base response
        base_response = self._generate_base_response(query, analysis)
        
        # Apply learning strategy
        adaptive_content = strategy_method(base_response, analysis, user_profile)
        
        # Personalize content
        personalized_content = self._personalize_content(adaptive_content, user_profile)
        
        # Add metacognitive elements
        metacognitive_elements = self._add_metacognitive_elements(
            personalized_content, analysis, strategy
        )
        
        return {
            "content": personalized_content,
            "metacognitive_guidance": metacognitive_elements,
            "strategy_explanation": strategy_info["description"],
            "learning_principles": strategy_info["principles"],
            "personalization_level": self._calculate_personalization_level(user_profile),
            "adaptive_features": self._identify_adaptive_features(personalized_content)
        }
    
    def _gradual_learning(self, content: str, analysis: Dict, user_profile: Dict) -> str:
        """Apply gradual learning strategy"""
        
        # Break content into progressive steps
        steps = self._break_into_steps(content, analysis["difficulty"])
        
        # Add scaffolding
        scaffolded_content = self._add_scaffolding(steps, user_profile)
        
        # Include progress indicators
        progress_content = self._add_progress_indicators(scaffolded_content)
        
        return f"سنتعلم هذا تدريجياً خطوة بخطوة:\n\n{progress_content}\n\nتذكر: التدرج في التعلم يضمن الفهم العميق والثبات."
    
    def _reflective_learning(self, content: str, analysis: Dict, user_profile: Dict) -> str:
        """Apply reflective learning strategy"""
        
        # Add reflection prompts
        reflection_prompts = self._generate_reflection_prompts(content, analysis)
        
        # Include contemplative questions
        contemplative_questions = self._generate_contemplative_questions(content)
        
        # Add wisdom connections
        wisdom_connections = self._find_wisdom_connections(content)
        
        reflective_content = f"{content}\n\n🤔 للتأمل والتفكر:\n{reflection_prompts}\n\n❓ أسئلة للتدبر:\n{contemplative_questions}"
        
        if wisdom_connections:
            reflective_content += f"\n\n💎 روابط الحكمة:\n{wisdom_connections}"
        
        return reflective_content
    
    def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get or create user profile"""
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "creation_date": datetime.now().isoformat(),
                "total_interactions": 0,
                "preferred_learning_style": "mixed",
                "knowledge_domains": {},
                "strategy_performance": {},
                "learning_history": [],
                "strengths": [],
                "areas_for_improvement": [],
                "learning_goals": [],
                "cultural_preferences": "arabic_islamic"
            }
        
        return self.user_profiles[user_id]
    
    def _update_user_profile(self, user_id: str, experience: LearningExperience, 
                           analysis: Dict) -> None:
        """Update user profile based on learning experience"""
        
        profile = self.user_profiles[user_id]
        
        # Update interaction count
        profile["total_interactions"] += 1
        
        # Update knowledge domains
        domain = analysis["domain"]
        if domain not in profile["knowledge_domains"]:
            profile["knowledge_domains"][domain] = {"interactions": 0, "proficiency": 0.5}
        
        profile["knowledge_domains"][domain]["interactions"] += 1
        
        # Update learning history
        profile["learning_history"].append({
            "experience_id": experience.experience_id,
            "timestamp": experience.timestamp,
            "domain": domain,
            "difficulty": analysis["difficulty"],
            "strategy": experience.learning_strategy_used.value if experience.learning_strategy_used else None
        })
        
        # Keep only recent history (last 100 interactions)
        if len(profile["learning_history"]) > 100:
            profile["learning_history"] = profile["learning_history"][-100:]
    
    def _save_learning_data(self) -> None:
        """Save learning data to persistent storage"""
        
        try:
            data = {
                "learning_patterns": self.learning_patterns,
                "user_profiles": self.user_profiles,
                "performance_metrics": self.performance_metrics
            }
            
            # Create directory if it doesn't exist
            os.makedirs("data/learning", exist_ok=True)
            
            # Save to file
            with open("data/learning/learning_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info("Learning data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save learning data: {e}")
    
    def _load_learning_data(self) -> None:
        """Load learning data from persistent storage"""
        
        try:
            if os.path.exists("data/learning/learning_data.json"):
                with open("data/learning/learning_data.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self.learning_patterns = data.get("learning_patterns", {})
                self.user_profiles = data.get("user_profiles", {})
                self.performance_metrics = data.get("performance_metrics", self.performance_metrics)
                
                self.logger.info("Learning data loaded successfully")
            else:
                self.logger.info("No existing learning data found, starting fresh")
                
        except Exception as e:
            self.logger.error(f"Failed to load learning data: {e}")
    
    # Placeholder implementations for helper methods
    def _recognize_learning_patterns(self): pass
    def _adapt_learning_strategy(self): pass
    def _personalize_content(self, content, profile): return content
    def _adjust_difficulty(self): pass
    def _integrate_feedback(self): pass
    def _evolve_knowledge_base(self): pass
    def _optimize_performance(self): pass
    def _repetitive_learning(self, content, analysis, profile): return content
    def _experiential_learning(self, content, analysis, profile): return content
    def _collaborative_learning(self, content, analysis, profile): return content
    def _process_explicit_feedback(self, experience, feedback): pass
    def _process_implicit_feedback(self, experience, feedback): pass
    def _process_behavioral_feedback(self, experience, feedback): pass
    def _process_performance_feedback(self, experience, feedback): pass
    def _process_emotional_feedback(self, experience, feedback): pass
    def _generate_experience_id(self): return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    def _update_performance_metrics(self): self.performance_metrics["total_interactions"] += 1
    def _trigger_adaptation_if_needed(self): pass
    def _find_experience(self, exp_id): return None
    def _learn_from_feedback(self, experience, feedback): pass
    def _update_learning_patterns(self, experience, feedback): pass
    def _analyze_interaction_sequences(self): return []
    def _analyze_success_patterns(self): return []
    def _analyze_difficulty_patterns(self): return []
    def _analyze_temporal_patterns(self): return []
    def _identify_knowledge_domain(self, query): return "general"
    def _assess_difficulty_level(self, query, profile): return "medium"
    def _identify_learning_intent(self, query): return "understanding"
    def _estimate_cognitive_load(self, query): return 0.5
    def _assess_prior_knowledge(self, query, profile): return 0.5
    def _analyze_context_relevance(self, query, context): return 0.7
    def _assess_context_complexity(self, context): return 0.5
    def _calculate_domain_suitability(self, strategy, domain): return 0.7
    def _calculate_difficulty_appropriateness(self, strategy, difficulty): return 0.7
    def _calculate_style_match(self, strategy, style): return 0.7
    def _generate_base_response(self, query, analysis): return f"إجابة أساسية على: {query}"
    def _add_metacognitive_elements(self, content, analysis, strategy): return ["فكر في...", "تأمل في..."]
    def _calculate_personalization_level(self, profile): return 0.7
    def _identify_adaptive_features(self, content): return ["تخصيص المحتوى", "تكييف الصعوبة"]
    def _break_into_steps(self, content, difficulty): return [f"خطوة {i+1}: جزء من المحتوى" for i in range(3)]
    def _add_scaffolding(self, steps, profile): return "\n".join(steps)
    def _add_progress_indicators(self, content): return f"📊 التقدم: {content}"
    def _generate_reflection_prompts(self, content, analysis): return "• فكر في المعنى العميق\n• تأمل في التطبيقات"
    def _generate_contemplative_questions(self, content): return "• ما الحكمة من هذا؟\n• كيف يمكن تطبيقه؟"
    def _find_wisdom_connections(self, content): return "• ربط بالحكمة التراثية"


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Intelligent Learning System
    learning_system = IntelligentLearningSystem()
    
    # Test adaptive learning
    test_queries = [
        "كيف أتعلم بفعالية أكبر؟",
        "ما هي أفضل طرق حفظ القرآن؟",
        "كيف أطور مهاراتي في الرياضيات؟",
        "ما أهمية الصبر في التعلم؟"
    ]
    
    print("🧠 Intelligent Adaptive Learning System 🧠")
    print("=" * 60)
    
    for i, query in enumerate(test_queries):
        print(f"\n📚 Learning Request {i+1}: {query}")
        
        # Adaptive learning
        result = learning_system.adaptive_learn(query, f"user_{i+1}")
        
        print(f"🎯 Strategy: {result['learning_strategy']}")
        print(f"📊 Personalization: {result['personalization_level']:.2f}")
        print(f"💡 Response: {result['response']['content'][:200]}...")
        
        # Simulate feedback
        feedback = {
            "satisfaction": 0.8 + (i * 0.05),
            "understanding": 0.7 + (i * 0.1),
            "feedback_text": "مفيد جداً"
        }
        
        learning_system.process_feedback(result['experience_id'], feedback)
        print(f"✅ Feedback processed")
        
        print("-" * 40)
    
    # Discover patterns
    patterns = learning_system.discover_learning_patterns()
    print(f"\n🔍 Discovered {len(patterns)} learning patterns")
    
    # Save learning data
    learning_system._save_learning_data()
    print("💾 Learning data saved")
