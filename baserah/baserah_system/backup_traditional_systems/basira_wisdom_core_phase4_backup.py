#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basira Wisdom Engine - Core Module

This module implements the core wisdom engine that embodies the essence of Basira:
"Where ancient wisdom meets modern innovation" - حيث تلتقي الحكمة القديمة بالابتكار الحديث

Author: Basira System Development Team
Version: 3.0.0 (Revolutionary Wisdom)
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.general_shape_equation import GeneralShapeEquation, EquationType, LearningMode
    from arabic_intelligence.advanced_arabic_ai import AdvancedArabicAI
except ImportError as e:
    logging.warning(f"Some modules not available: {e}")

# Configure logging
logger = logging.getLogger('wisdom_engine.basira_wisdom_core')


class WisdomType(Enum):
    """Types of wisdom in Basira system"""
    QURANIC = "قرآني"           # Quranic wisdom
    PROPHETIC = "نبوي"         # Prophetic traditions
    PHILOSOPHICAL = "فلسفي"     # Philosophical wisdom
    LITERARY = "أدبي"          # Literary wisdom
    SCIENTIFIC = "علمي"        # Scientific wisdom
    EXPERIENTIAL = "تجريبي"    # Experiential wisdom
    INTUITIVE = "حدسي"         # Intuitive wisdom
    CULTURAL = "ثقافي"         # Cultural wisdom


class InsightLevel(Enum):
    """Levels of insight depth"""
    SURFACE = "سطحي"           # Surface level
    INTERMEDIATE = "متوسط"     # Intermediate level
    DEEP = "عميق"              # Deep level
    PROFOUND = "عميق_جداً"      # Profound level
    TRANSCENDENT = "متعالي"    # Transcendent level


@dataclass
class WisdomPearl:
    """A pearl of wisdom with rich contextual information"""
    id: str
    text: str
    source: str
    wisdom_type: WisdomType
    insight_level: InsightLevel
    
    # Contextual information
    historical_context: Optional[str] = None
    cultural_significance: Optional[str] = None
    practical_applications: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    
    # Semantic information
    key_themes: List[str] = field(default_factory=list)
    moral_lessons: List[str] = field(default_factory=list)
    metaphors: List[str] = field(default_factory=list)
    
    # Computational aspects
    semantic_vector: Optional[np.ndarray] = None
    relevance_score: float = 0.0
    wisdom_weight: float = 1.0
    
    # Metadata
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0


@dataclass
class InsightGeneration:
    """Generated insight with reasoning chain"""
    insight_text: str
    confidence: float
    reasoning_chain: List[str]
    supporting_wisdom: List[str]
    practical_implications: List[str]
    depth_level: InsightLevel
    generation_method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BasiraWisdomCore:
    """
    Core Wisdom Engine for Basira System
    
    This engine embodies the vision of Basira: combining ancient Arabic-Islamic wisdom
    with cutting-edge AI to generate profound insights and guidance.
    """
    
    def __init__(self):
        """Initialize the Basira Wisdom Core"""
        self.logger = logging.getLogger('wisdom_engine.basira_wisdom_core.main')
        
        # Initialize core components
        self.general_equation = GeneralShapeEquation(
            equation_type=EquationType.WISDOM,
            learning_mode=LearningMode.TRANSCENDENT
        )
        
        # Initialize Arabic AI
        try:
            self.arabic_ai = AdvancedArabicAI()
        except:
            self.arabic_ai = None
            self.logger.warning("Arabic AI not available")
        
        # Wisdom repositories
        self.wisdom_pearls = {}
        self.wisdom_networks = {}
        self.insight_cache = {}
        
        # Load wisdom databases
        self._load_quranic_wisdom()
        self._load_prophetic_wisdom()
        self._load_philosophical_wisdom()
        self._load_cultural_wisdom()
        
        # Initialize reasoning engines
        self.reasoning_engines = self._initialize_reasoning_engines()
        
        # Wisdom generation algorithms
        self.generation_algorithms = self._initialize_generation_algorithms()
        
        self.logger.info("Basira Wisdom Core initialized with profound capabilities")
    
    def _load_quranic_wisdom(self):
        """Load Quranic wisdom pearls"""
        
        quranic_pearls = [
            WisdomPearl(
                id="quran_001",
                text="وَمَا أُوتِيتُم مِّنَ الْعِلْمِ إِلَّا قَلِيلًا",
                source="القرآن الكريم - سورة الإسراء آية 85",
                wisdom_type=WisdomType.QURANIC,
                insight_level=InsightLevel.PROFOUND,
                historical_context="نزلت في سياق سؤال اليهود عن الروح",
                cultural_significance="تؤكد على تواضع الإنسان أمام علم الله اللامحدود",
                practical_applications=[
                    "التواضع في طلب العلم",
                    "الاعتراف بحدود المعرفة البشرية",
                    "التحفيز على البحث والاستكشاف"
                ],
                key_themes=["العلم", "التواضع", "حدود المعرفة", "عظمة الله"],
                moral_lessons=["التواضع العلمي", "الاعتراف بالجهل", "طلب المزيد من العلم"],
                metaphors=["العلم كقطرة من بحر", "المعرفة البشرية كشعاع من نور"]
            ),
            
            WisdomPearl(
                id="quran_002",
                text="وَفَوْقَ كُلِّ ذِي عِلْمٍ عَلِيمٌ",
                source="القرآن الكريم - سورة يوسف آية 76",
                wisdom_type=WisdomType.QURANIC,
                insight_level=InsightLevel.DEEP,
                cultural_significance="تؤكد على التسلسل الهرمي للمعرفة",
                practical_applications=[
                    "احترام أهل العلم",
                    "التواضع مهما بلغ العلم",
                    "السعي للتعلم من الآخرين"
                ],
                key_themes=["العلم", "التواضع", "التعلم المستمر"],
                moral_lessons=["لا تتكبر بعلمك", "تعلم من الجميع", "العلم لا حدود له"]
            ),
            
            WisdomPearl(
                id="quran_003",
                text="إِنَّ مَعَ الْعُسْرِ يُسْرًا",
                source="القرآن الكريم - سورة الشرح آية 6",
                wisdom_type=WisdomType.QURANIC,
                insight_level=InsightLevel.PROFOUND,
                cultural_significance="مبدأ أساسي في التفاؤل الإسلامي",
                practical_applications=[
                    "الصبر في المحن",
                    "التفاؤل في الصعوبات",
                    "الثقة في الفرج"
                ],
                key_themes=["الصبر", "التفاؤل", "الفرج", "الابتلاء"],
                moral_lessons=["لا تيأس", "الفرج قريب", "الصبر مفتاح الفرج"],
                metaphors=["العسر واليسر كالليل والنهار", "المحنة بوابة المنحة"]
            )
        ]
        
        for pearl in quranic_pearls:
            self.wisdom_pearls[pearl.id] = pearl
    
    def _load_prophetic_wisdom(self):
        """Load Prophetic wisdom pearls"""
        
        prophetic_pearls = [
            WisdomPearl(
                id="hadith_001",
                text="اطلبوا العلم من المهد إلى اللحد",
                source="الحديث الشريف",
                wisdom_type=WisdomType.PROPHETIC,
                insight_level=InsightLevel.DEEP,
                cultural_significance="يؤكد على أهمية التعلم مدى الحياة",
                practical_applications=[
                    "التعليم المستمر",
                    "التطوير الذاتي",
                    "عدم التوقف عن التعلم"
                ],
                key_themes=["التعلم", "الاستمرارية", "النمو"],
                moral_lessons=["التعلم لا يتوقف", "كل مرحلة لها علمها", "العلم رحلة حياة"]
            ),
            
            WisdomPearl(
                id="hadith_002",
                text="إنما الأعمال بالنيات",
                source="صحيح البخاري",
                wisdom_type=WisdomType.PROPHETIC,
                insight_level=InsightLevel.PROFOUND,
                cultural_significance="أساس في فقه الأعمال والأخلاق",
                practical_applications=[
                    "تصحيح النية قبل العمل",
                    "تقييم الأعمال بالمقاصد",
                    "تطهير القلب من الرياء"
                ],
                key_themes=["النية", "الإخلاص", "المقاصد", "القلب"],
                moral_lessons=["النية أساس العمل", "أخلص نيتك", "القلب محل النظر"]
            )
        ]
        
        for pearl in prophetic_pearls:
            self.wisdom_pearls[pearl.id] = pearl
    
    def _load_philosophical_wisdom(self):
        """Load philosophical wisdom pearls"""
        
        philosophical_pearls = [
            WisdomPearl(
                id="philosophy_001",
                text="أعرف نفسك",
                source="الحكمة اليونانية - سقراط",
                wisdom_type=WisdomType.PHILOSOPHICAL,
                insight_level=InsightLevel.DEEP,
                cultural_significance="أساس الفلسفة والتطوير الذاتي",
                practical_applications=[
                    "التأمل الذاتي",
                    "فهم نقاط القوة والضعف",
                    "التطوير الشخصي"
                ],
                key_themes=["المعرفة الذاتية", "التأمل", "الوعي"],
                moral_lessons=["ابدأ بنفسك", "المعرفة الذاتية أساس الحكمة", "تأمل في ذاتك"]
            ),
            
            WisdomPearl(
                id="philosophy_002",
                text="الحكمة ضالة المؤمن أنى وجدها فهو أحق بها",
                source="الإمام علي رضي الله عنه",
                wisdom_type=WisdomType.PHILOSOPHICAL,
                insight_level=InsightLevel.PROFOUND,
                cultural_significance="يؤكد على عالمية الحكمة وطلبها",
                practical_applications=[
                    "البحث عن الحكمة في كل مكان",
                    "عدم التعصب للمصادر",
                    "الانفتاح على التعلم"
                ],
                key_themes=["الحكمة", "البحث", "الانفتاح", "التعلم"],
                moral_lessons=["الحكمة لا وطن لها", "تعلم من الجميع", "لا تحتقر مصدر الحكمة"]
            )
        ]
        
        for pearl in philosophical_pearls:
            self.wisdom_pearls[pearl.id] = pearl
    
    def _load_cultural_wisdom(self):
        """Load cultural wisdom pearls"""
        
        cultural_pearls = [
            WisdomPearl(
                id="culture_001",
                text="العلم نور والجهل ظلام",
                source="الحكمة العربية",
                wisdom_type=WisdomType.CULTURAL,
                insight_level=InsightLevel.INTERMEDIATE,
                cultural_significance="تعبر عن قيمة العلم في الثقافة العربية",
                practical_applications=[
                    "تحفيز طلب العلم",
                    "محاربة الجهل",
                    "نشر المعرفة"
                ],
                key_themes=["العلم", "النور", "الجهل", "الظلام"],
                moral_lessons=["اطلب العلم", "انشر المعرفة", "حارب الجهل"],
                metaphors=["العلم كالنور", "الجهل كالظلام"]
            ),
            
            WisdomPearl(
                id="culture_002",
                text="من لم يذق مر التعلم ساعة تجرع ذل الجهل أبداً",
                source="الإمام الشافعي",
                wisdom_type=WisdomType.CULTURAL,
                insight_level=InsightLevel.DEEP,
                cultural_significance="يؤكد على أهمية الصبر في التعلم",
                practical_applications=[
                    "الصبر على مشقة التعلم",
                    "تحمل صعوبات الدراسة",
                    "المثابرة في طلب العلم"
                ],
                key_themes=["التعلم", "الصبر", "المشقة", "الجهل"],
                moral_lessons=["اصبر على التعلم", "المشقة مؤقتة", "الجهل ذل دائم"]
            )
        ]
        
        for pearl in cultural_pearls:
            self.wisdom_pearls[pearl.id] = pearl
    
    def _initialize_reasoning_engines(self) -> Dict[str, Any]:
        """Initialize various reasoning engines"""
        
        return {
            "analogical": self._analogical_reasoning,
            "metaphorical": self._metaphorical_reasoning,
            "contextual": self._contextual_reasoning,
            "causal": self._causal_reasoning,
            "temporal": self._temporal_reasoning,
            "ethical": self._ethical_reasoning,
            "spiritual": self._spiritual_reasoning,
            "practical": self._practical_reasoning
        }
    
    def _initialize_generation_algorithms(self) -> Dict[str, Any]:
        """Initialize wisdom generation algorithms"""
        
        return {
            "synthesis": self._wisdom_synthesis,
            "analogy": self._wisdom_by_analogy,
            "induction": self._wisdom_by_induction,
            "deduction": self._wisdom_by_deduction,
            "intuition": self._wisdom_by_intuition,
            "integration": self._wisdom_integration,
            "transcendence": self._wisdom_transcendence
        }
    
    def generate_insight(self, query: str, context: Optional[str] = None) -> InsightGeneration:
        """
        Generate profound insights based on query and context
        
        Args:
            query: The question or topic to generate insight about
            context: Optional context to guide the insight generation
            
        Returns:
            Generated insight with reasoning chain
        """
        
        # Analyze the query
        query_analysis = self._analyze_query(query)
        
        # Find relevant wisdom pearls
        relevant_pearls = self._find_relevant_wisdom(query, context)
        
        # Generate insight using multiple methods
        insights = []
        
        for method_name, method_func in self.generation_algorithms.items():
            try:
                insight = method_func(query, relevant_pearls, context)
                if insight:
                    insights.append(insight)
            except Exception as e:
                self.logger.warning(f"Method {method_name} failed: {e}")
        
        # Select best insight
        best_insight = self._select_best_insight(insights, query_analysis)
        
        # Enhance with reasoning chain
        enhanced_insight = self._enhance_with_reasoning(best_insight, relevant_pearls)
        
        return enhanced_insight
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to understand its nature and requirements"""
        
        analysis = {
            "query": query,
            "themes": [],
            "complexity": "medium",
            "domain": "general",
            "intent": "understanding",
            "emotional_tone": "neutral"
        }
        
        # Extract themes
        if self.arabic_ai:
            try:
                ai_analysis = self.arabic_ai.analyze_text_with_cultural_intelligence(query)
                analysis["themes"] = [concept["semantic_field"] for concept in ai_analysis["semantic_analysis"]["concepts"]]
                analysis["cultural_significance"] = ai_analysis["cultural_analysis"]["cultural_significance"]
            except:
                pass
        
        # Determine complexity
        word_count = len(query.split())
        if word_count > 20:
            analysis["complexity"] = "high"
        elif word_count < 5:
            analysis["complexity"] = "low"
        
        # Determine domain
        religious_keywords = ["الله", "دين", "إيمان", "صلاة", "قرآن"]
        scientific_keywords = ["علم", "تجربة", "نظرية", "بحث"]
        philosophical_keywords = ["حكمة", "فلسفة", "معنى", "وجود"]
        
        if any(keyword in query for keyword in religious_keywords):
            analysis["domain"] = "religious"
        elif any(keyword in query for keyword in scientific_keywords):
            analysis["domain"] = "scientific"
        elif any(keyword in query for keyword in philosophical_keywords):
            analysis["domain"] = "philosophical"
        
        return analysis
    
    def _find_relevant_wisdom(self, query: str, context: Optional[str] = None) -> List[WisdomPearl]:
        """Find wisdom pearls relevant to the query"""
        
        relevant_pearls = []
        query_words = set(query.split())
        
        for pearl in self.wisdom_pearls.values():
            relevance_score = 0.0
            
            # Check text similarity
            pearl_words = set(pearl.text.split())
            common_words = query_words.intersection(pearl_words)
            relevance_score += len(common_words) * 0.3
            
            # Check theme overlap
            for theme in pearl.key_themes:
                if theme in query:
                    relevance_score += 0.4
            
            # Check practical applications
            for application in pearl.practical_applications:
                if any(word in application for word in query_words):
                    relevance_score += 0.2
            
            # Context relevance
            if context:
                context_words = set(context.split())
                pearl_context_words = set(pearl.cultural_significance.split() if pearl.cultural_significance else [])
                context_overlap = context_words.intersection(pearl_context_words)
                relevance_score += len(context_overlap) * 0.1
            
            if relevance_score > 0.3:
                pearl.relevance_score = relevance_score
                relevant_pearls.append(pearl)
        
        # Sort by relevance and return top 5
        relevant_pearls.sort(key=lambda p: p.relevance_score, reverse=True)
        return relevant_pearls[:5]
    
    def _wisdom_synthesis(self, query: str, pearls: List[WisdomPearl], context: Optional[str]) -> Optional[InsightGeneration]:
        """Generate insight by synthesizing multiple wisdom pearls"""
        
        if len(pearls) < 2:
            return None
        
        # Combine themes from multiple pearls
        combined_themes = []
        combined_lessons = []
        
        for pearl in pearls[:3]:  # Use top 3 pearls
            combined_themes.extend(pearl.key_themes)
            combined_lessons.extend(pearl.moral_lessons)
        
        # Generate synthetic insight
        insight_text = f"من خلال تأمل الحكم المختلفة، نجد أن {query} يتطلب فهماً عميقاً لـ {', '.join(set(combined_themes[:3]))}. "
        insight_text += f"والدروس المستفادة تشمل: {', '.join(set(combined_lessons[:2]))}."
        
        reasoning_chain = [
            f"تم تحليل {len(pearls)} من لآلئ الحكمة",
            f"استخراج المواضيع المشتركة: {', '.join(set(combined_themes[:3]))}",
            f"دمج الدروس الأخلاقية: {', '.join(set(combined_lessons[:2]))}",
            "توليد رؤية شاملة من خلال التركيب"
        ]
        
        return InsightGeneration(
            insight_text=insight_text,
            confidence=0.8,
            reasoning_chain=reasoning_chain,
            supporting_wisdom=[pearl.text for pearl in pearls[:3]],
            practical_implications=list(set([app for pearl in pearls for app in pearl.practical_applications[:2]])),
            depth_level=InsightLevel.DEEP,
            generation_method="synthesis"
        )
    
    def _wisdom_by_analogy(self, query: str, pearls: List[WisdomPearl], context: Optional[str]) -> Optional[InsightGeneration]:
        """Generate insight using analogical reasoning"""
        
        if not pearls:
            return None
        
        best_pearl = pearls[0]
        
        # Create analogy
        insight_text = f"بالقياس على حكمة '{best_pearl.text}', يمكننا فهم {query} "
        insight_text += f"من خلال تطبيق نفس المبادئ: {', '.join(best_pearl.key_themes[:2])}."
        
        reasoning_chain = [
            f"اختيار الحكمة الأكثر صلة: {best_pearl.text}",
            f"استخراج المبادئ الأساسية: {', '.join(best_pearl.key_themes[:2])}",
            f"تطبيق القياس على السؤال: {query}",
            "توليد فهم جديد من خلال القياس"
        ]
        
        return InsightGeneration(
            insight_text=insight_text,
            confidence=0.7,
            reasoning_chain=reasoning_chain,
            supporting_wisdom=[best_pearl.text],
            practical_implications=best_pearl.practical_applications[:3],
            depth_level=InsightLevel.INTERMEDIATE,
            generation_method="analogy"
        )
    
    def _select_best_insight(self, insights: List[InsightGeneration], query_analysis: Dict) -> InsightGeneration:
        """Select the best insight from generated options"""
        
        if not insights:
            # Generate default insight
            return InsightGeneration(
                insight_text=f"هذا سؤال عميق يتطلب تأملاً أكثر في {query_analysis['query']}",
                confidence=0.5,
                reasoning_chain=["لم يتم العثور على حكمة مناسبة", "توليد رد افتراضي"],
                supporting_wisdom=[],
                practical_implications=["التأمل أكثر", "البحث عن مصادر إضافية"],
                depth_level=InsightLevel.SURFACE,
                generation_method="default"
            )
        
        # Score insights based on multiple criteria
        for insight in insights:
            score = 0.0
            score += insight.confidence * 0.4
            score += len(insight.supporting_wisdom) * 0.2
            score += len(insight.practical_implications) * 0.2
            score += insight.depth_level.value.count("عميق") * 0.2
            insight.total_score = score
        
        # Return highest scoring insight
        return max(insights, key=lambda i: getattr(i, 'total_score', 0))
    
    def _enhance_with_reasoning(self, insight: InsightGeneration, pearls: List[WisdomPearl]) -> InsightGeneration:
        """Enhance insight with detailed reasoning chain"""
        
        enhanced_reasoning = insight.reasoning_chain.copy()
        
        # Add wisdom source analysis
        if pearls:
            enhanced_reasoning.append(f"مصادر الحكمة المستخدمة: {len(pearls)} لؤلؤة")
            for pearl in pearls[:2]:
                enhanced_reasoning.append(f"- {pearl.source}: {pearl.wisdom_type.value}")
        
        # Add depth analysis
        enhanced_reasoning.append(f"مستوى العمق المحقق: {insight.depth_level.value}")
        
        # Add confidence justification
        if insight.confidence > 0.8:
            enhanced_reasoning.append("مستوى ثقة عالي بناءً على تطابق قوي مع الحكم التراثية")
        elif insight.confidence > 0.6:
            enhanced_reasoning.append("مستوى ثقة متوسط بناءً على تطابق جزئي مع المصادر")
        else:
            enhanced_reasoning.append("مستوى ثقة منخفض - يحتاج مزيد من التأمل")
        
        insight.reasoning_chain = enhanced_reasoning
        return insight
    
    # Placeholder methods for other reasoning engines
    def _analogical_reasoning(self, *args): return {}
    def _metaphorical_reasoning(self, *args): return {}
    def _contextual_reasoning(self, *args): return {}
    def _causal_reasoning(self, *args): return {}
    def _temporal_reasoning(self, *args): return {}
    def _ethical_reasoning(self, *args): return {}
    def _spiritual_reasoning(self, *args): return {}
    def _practical_reasoning(self, *args): return {}
    
    # Placeholder methods for other generation algorithms
    def _wisdom_by_induction(self, *args): return None
    def _wisdom_by_deduction(self, *args): return None
    def _wisdom_by_intuition(self, *args): return None
    def _wisdom_integration(self, *args): return None
    def _wisdom_transcendence(self, *args): return None


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Basira Wisdom Core
    wisdom_core = BasiraWisdomCore()
    
    # Test queries
    test_queries = [
        "ما هو معنى الحكمة الحقيقية؟",
        "كيف أتعامل مع الصعوبات في الحياة؟",
        "ما أهمية العلم في حياة الإنسان؟",
        "كيف أحقق التوازن بين العمل والحياة؟"
    ]
    
    print("🌟 Basira Wisdom Engine - Insight Generation 🌟")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        insight = wisdom_core.generate_insight(query)
        
        print(f"💡 Insight: {insight.insight_text}")
        print(f"🎯 Confidence: {insight.confidence:.2f}")
        print(f"📊 Depth: {insight.depth_level.value}")
        print(f"🔧 Method: {insight.generation_method}")
        
        if insight.supporting_wisdom:
            print("📚 Supporting Wisdom:")
            for wisdom in insight.supporting_wisdom[:2]:
                print(f"   • {wisdom}")
        
        if insight.practical_implications:
            print("🛠️ Practical Implications:")
            for impl in insight.practical_implications[:2]:
                print(f"   • {impl}")
        
        print("-" * 40)
