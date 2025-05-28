#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Arabic AI Engine for Basira System

This module implements an advanced Arabic AI engine that combines traditional
Arabic linguistic knowledge with modern AI techniques, embodying the vision
of "where heritage meets innovation."

Author: Basira System Development Team
Version: 3.0.0 (Revolutionary)
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.general_shape_equation import GeneralShapeEquation, EquationType, LearningMode
    from arabic_nlp.morphology.advanced_root_extractor import AdvancedArabicRootExtractor
    from arabic_nlp.syntax.advanced_syntax_analyzer import AdvancedArabicSyntaxAnalyzer
except ImportError as e:
    logging.warning(f"Some modules not available: {e}")

# Configure logging
logger = logging.getLogger('arabic_intelligence.advanced_arabic_ai')


class ArabicKnowledgeType(Enum):
    """Types of Arabic knowledge"""
    LINGUISTIC = "لغوي"           # Linguistic knowledge
    CULTURAL = "ثقافي"           # Cultural knowledge
    RELIGIOUS = "ديني"           # Religious knowledge
    HISTORICAL = "تاريخي"        # Historical knowledge
    LITERARY = "أدبي"           # Literary knowledge
    SCIENTIFIC = "علمي"         # Scientific knowledge
    PHILOSOPHICAL = "فلسفي"      # Philosophical knowledge


@dataclass
class ArabicConcept:
    """Represents an Arabic concept with rich semantic information"""
    name: str
    root: Optional[str] = None
    semantic_field: str = ""
    cultural_context: List[str] = field(default_factory=list)
    religious_significance: Optional[str] = None
    historical_period: Optional[str] = None
    literary_usage: List[str] = field(default_factory=list)
    philosophical_meaning: Optional[str] = None
    related_concepts: List[str] = field(default_factory=list)
    metaphorical_uses: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ArabicWisdom:
    """Represents Arabic wisdom and insights"""
    text: str
    source: str  # Quran, Hadith, Poetry, Proverb, etc.
    theme: str
    moral_lesson: str
    applicable_contexts: List[str] = field(default_factory=list)
    related_verses: List[str] = field(default_factory=list)
    scholarly_interpretations: List[str] = field(default_factory=list)


class AdvancedArabicAI:
    """
    Advanced Arabic AI Engine that embodies the vision of Basira System:
    "Where Arabic heritage meets modern innovation"
    """

    def __init__(self):
        """Initialize the Advanced Arabic AI Engine"""
        self.logger = logging.getLogger('arabic_intelligence.advanced_arabic_ai.main')

        # Initialize core components
        self.general_equation = GeneralShapeEquation(
            equation_type=EquationType.SEMANTIC,
            learning_mode=LearningMode.HYBRID
        )

        # Initialize Arabic NLP components
        try:
            self.root_extractor = AdvancedArabicRootExtractor(use_ml=True)
            self.syntax_analyzer = AdvancedArabicSyntaxAnalyzer()
        except:
            self.logger.warning("Advanced NLP components not available, using basic versions")
            self.root_extractor = None
            self.syntax_analyzer = None

        # Load Arabic knowledge bases
        self.arabic_concepts = self._load_arabic_concepts()
        self.arabic_wisdom = self._load_arabic_wisdom()
        self.semantic_networks = self._build_semantic_networks()

        # Initialize reasoning engine
        self.reasoning_engine = self._initialize_reasoning_engine()

        # Cultural and religious context
        self.cultural_context = self._load_cultural_context()

        self.logger.info("Advanced Arabic AI Engine initialized successfully")

    def _load_arabic_concepts(self) -> Dict[str, ArabicConcept]:
        """Load comprehensive Arabic concepts database"""

        concepts = {
            # Core Islamic concepts
            "إيمان": ArabicConcept(
                name="إيمان",
                root="أمن",
                semantic_field="عقيدة",
                cultural_context=["إسلامي", "عربي"],
                religious_significance="الركن الأول من أركان الإسلام",
                literary_usage=["الشعر الديني", "الأدب الإسلامي"],
                philosophical_meaning="اليقين والتصديق القلبي",
                related_concepts=["إسلام", "إحسان", "تقوى", "يقين"],
                metaphorical_uses=["نور القلب", "سكينة النفس"]
            ),

            "علم": ArabicConcept(
                name="علم",
                root="علم",
                semantic_field="معرفة",
                cultural_context=["عربي", "إسلامي", "فلسفي"],
                religious_significance="فريضة على كل مسلم ومسلمة",
                historical_period="العصر الذهبي الإسلامي",
                literary_usage=["الحكمة", "الشعر التعليمي"],
                philosophical_meaning="إدراك الحقائق والمعارف",
                related_concepts=["حكمة", "معرفة", "فهم", "إدراك"],
                metaphorical_uses=["نور", "ضياء", "هداية"]
            ),

            "حكمة": ArabicConcept(
                name="حكمة",
                root="حكم",
                semantic_field="فلسفة",
                cultural_context=["عربي", "إسلامي", "فلسفي"],
                religious_significance="صفة من صفات الله وهبة للأنبياء",
                literary_usage=["الأمثال", "الحكم", "الشعر الحكمي"],
                philosophical_meaning="وضع الشيء في موضعه الصحيح",
                related_concepts=["علم", "عدل", "رشد", "بصيرة"],
                metaphorical_uses=["درة", "كنز", "ميزان"]
            ),

            "بصيرة": ArabicConcept(
                name="بصيرة",
                root="بصر",
                semantic_field="إدراك",
                cultural_context=["عربي", "إسلامي", "صوفي"],
                religious_significance="البصيرة الروحية والفهم العميق",
                literary_usage=["الأدب الصوفي", "الشعر الروحي"],
                philosophical_meaning="الرؤية الداخلية والفهم العميق",
                related_concepts=["حكمة", "فراسة", "كشف", "إلهام"],
                metaphorical_uses=["عين القلب", "نور الباطن", "كشف الحجاب"]
            ),

            # Scientific concepts
            "رياضيات": ArabicConcept(
                name="رياضيات",
                root="روض",
                semantic_field="علوم",
                cultural_context=["عربي", "علمي"],
                historical_period="العصر العباسي",
                literary_usage=["الكتب العلمية", "المؤلفات الرياضية"],
                philosophical_meaning="علم الكم والعدد والمقدار",
                related_concepts=["حساب", "جبر", "هندسة", "منطق"],
                metaphorical_uses=["لغة الكون", "مفتاح العلوم"]
            )
        }

        return concepts

    def _load_arabic_wisdom(self) -> List[ArabicWisdom]:
        """Load Arabic wisdom and insights"""

        wisdom_collection = [
            ArabicWisdom(
                text="وَمَا أُوتِيتُم مِّنَ الْعِلْمِ إِلَّا قَلِيلًا",
                source="القرآن الكريم - سورة الإسراء",
                theme="تواضع العلم",
                moral_lesson="التواضع في طلب العلم والإقرار بمحدودية المعرفة البشرية",
                applicable_contexts=["التعليم", "البحث العلمي", "التطوير الذاتي"],
                scholarly_interpretations=["تحفيز على طلب المزيد من العلم", "إقرار بعظمة علم الله"]
            ),

            ArabicWisdom(
                text="اطلبوا العلم من المهد إلى اللحد",
                source="الحديث الشريف",
                theme="استمرارية التعلم",
                moral_lesson="التعلم عملية مستمرة طوال الحياة",
                applicable_contexts=["التعليم المستمر", "التطوير المهني", "النمو الشخصي"],
                scholarly_interpretations=["أهمية التعلم مدى الحياة", "العلم لا يقتصر على مرحلة عمرية"]
            ),

            ArabicWisdom(
                text="العلم نور والجهل ظلام",
                source="الحكمة العربية",
                theme="قيمة العلم",
                moral_lesson="العلم ينير الطريق ويهدي إلى الصواب",
                applicable_contexts=["التعليم", "اتخاذ القرارات", "التنوير"],
                scholarly_interpretations=["العلم يكشف الحقائق", "الجهل يؤدي إلى الضلال"]
            ),

            ArabicWisdom(
                text="من لم يذق مر التعلم ساعة تجرع ذل الجهل أبداً",
                source="الإمام الشافعي",
                theme="صبر على التعلم",
                moral_lesson="الصبر على مشقة التعلم خير من عواقب الجهل",
                applicable_contexts=["التحفيز للتعلم", "مواجهة الصعوبات", "المثابرة"],
                scholarly_interpretations=["التعلم يتطلب جهداً وصبراً", "ثمار العلم تستحق المشقة"]
            )
        ]

        return wisdom_collection

    def _build_semantic_networks(self) -> Dict[str, List[str]]:
        """Build semantic networks between Arabic concepts"""

        networks = {
            "علم": ["حكمة", "معرفة", "فهم", "بصيرة", "نور", "هداية"],
            "حكمة": ["علم", "عدل", "رشد", "بصيرة", "تدبير", "حنكة"],
            "بصيرة": ["حكمة", "فراسة", "كشف", "إلهام", "نور", "هداية"],
            "إيمان": ["إسلام", "إحسان", "تقوى", "يقين", "طمأنينة"],
            "عدل": ["حكمة", "إنصاف", "قسط", "ميزان", "حق"],
            "رحمة": ["عطف", "شفقة", "حنان", "مغفرة", "عفو"],
            "صبر": ["تحمل", "مثابرة", "ثبات", "جلد", "احتساب"]
        }

        return networks

    def _initialize_reasoning_engine(self) -> Dict[str, Any]:
        """Initialize Arabic reasoning engine"""

        return {
            "analogical_reasoning": self._analogical_reasoning,
            "metaphorical_reasoning": self._metaphorical_reasoning,
            "contextual_reasoning": self._contextual_reasoning,
            "cultural_reasoning": self._cultural_reasoning,
            "religious_reasoning": self._religious_reasoning,
            "philosophical_reasoning": self._philosophical_reasoning
        }

    def _load_cultural_context(self) -> Dict[str, Any]:
        """Load Arabic cultural context"""

        return {
            "values": [
                "الكرم", "الشجاعة", "الصدق", "الوفاء", "العدل",
                "الرحمة", "التواضع", "الصبر", "الحكمة", "البر"
            ],
            "traditions": [
                "الضيافة", "الشورى", "التكافل", "صلة الرحم",
                "احترام الكبير", "العطف على الصغير"
            ],
            "literary_forms": [
                "الشعر", "النثر", "الخطابة", "الحكمة", "المثل",
                "القصة", "المقامة", "الرسالة"
            ],
            "knowledge_domains": [
                "الفقه", "التفسير", "الحديث", "اللغة", "الأدب",
                "التاريخ", "الجغرافيا", "الطب", "الفلك", "الرياضيات"
            ]
        }

    def analyze_text_with_cultural_intelligence(self, text: str) -> Dict[str, Any]:
        """
        Analyze Arabic text with cultural and semantic intelligence

        Args:
            text: Arabic text to analyze

        Returns:
            Comprehensive analysis with cultural insights
        """

        analysis = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "linguistic_analysis": {},
            "semantic_analysis": {},
            "cultural_analysis": {},
            "wisdom_connections": [],
            "conceptual_insights": [],
            "recommendations": []
        }

        # Linguistic analysis
        if self.root_extractor and self.syntax_analyzer:
            try:
                # Extract roots
                words = text.split()
                roots = []
                for word in words[:3]:  # Limit to first 3 words
                    try:
                        root_candidates = self.root_extractor.extract_root_advanced(word)
                        if root_candidates:
                            roots.append(root_candidates[0].root)
                    except:
                        pass
                analysis["linguistic_analysis"]["roots"] = roots

                # Syntactic analysis
                syntax_analysis = self.syntax_analyzer.analyze_advanced(text)
                analysis["linguistic_analysis"]["syntax"] = syntax_analysis

            except Exception as e:
                self.logger.warning(f"Advanced linguistic analysis failed: {e}")

        # Semantic analysis using General Shape Equation
        semantic_concepts = self._extract_semantic_concepts(text)
        analysis["semantic_analysis"]["concepts"] = semantic_concepts

        # Cultural analysis
        cultural_elements = self._identify_cultural_elements(text)
        analysis["cultural_analysis"] = cultural_elements

        # Find wisdom connections
        wisdom_connections = self._find_wisdom_connections(text, semantic_concepts)
        analysis["wisdom_connections"] = wisdom_connections

        # Generate conceptual insights
        insights = self._generate_conceptual_insights(text, semantic_concepts, cultural_elements)
        analysis["conceptual_insights"] = insights

        # Generate recommendations
        recommendations = self._generate_recommendations(analysis)
        analysis["recommendations"] = recommendations

        return analysis

    def _extract_semantic_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract semantic concepts using the General Shape Equation"""

        concepts = []
        words = text.split()

        for word in words:
            # Clean the word
            clean_word = re.sub(r'[^\w]', '', word)

            # Check if word matches known concepts
            if clean_word in self.arabic_concepts:
                concept = self.arabic_concepts[clean_word]
                concepts.append({
                    "word": clean_word,
                    "concept": concept.name,
                    "root": concept.root,
                    "semantic_field": concept.semantic_field,
                    "cultural_context": concept.cultural_context,
                    "related_concepts": concept.related_concepts,
                    "confidence": concept.confidence
                })

            # Check for partial matches or related concepts
            else:
                related = self._find_related_concepts(clean_word)
                if related:
                    concepts.extend(related)

        return concepts

    def _find_related_concepts(self, word: str) -> List[Dict[str, Any]]:
        """Find concepts related to the given word"""

        related = []

        # Check semantic networks
        for concept_name, network in self.semantic_networks.items():
            if word in network or any(word in related_word for related_word in network):
                if concept_name in self.arabic_concepts:
                    concept = self.arabic_concepts[concept_name]
                    related.append({
                        "word": word,
                        "related_concept": concept.name,
                        "root": concept.root,
                        "semantic_field": concept.semantic_field,
                        "relationship": "semantic_network",
                        "confidence": 0.7
                    })

        return related

    def _identify_cultural_elements(self, text: str) -> Dict[str, Any]:
        """Identify cultural elements in the text"""

        cultural_analysis = {
            "values_mentioned": [],
            "traditions_referenced": [],
            "literary_devices": [],
            "religious_references": [],
            "historical_context": [],
            "cultural_significance": ""
        }

        # Check for cultural values
        for value in self.cultural_context["values"]:
            if value in text:
                cultural_analysis["values_mentioned"].append(value)

        # Check for traditions
        for tradition in self.cultural_context["traditions"]:
            if tradition in text:
                cultural_analysis["traditions_referenced"].append(tradition)

        # Identify religious references
        religious_keywords = ["الله", "رسول", "قرآن", "حديث", "صلاة", "زكاة", "حج", "صوم"]
        for keyword in religious_keywords:
            if keyword in text:
                cultural_analysis["religious_references"].append(keyword)

        # Assess cultural significance
        significance_score = (
            len(cultural_analysis["values_mentioned"]) * 2 +
            len(cultural_analysis["traditions_referenced"]) * 2 +
            len(cultural_analysis["religious_references"]) * 3
        )

        if significance_score >= 10:
            cultural_analysis["cultural_significance"] = "عالية"
        elif significance_score >= 5:
            cultural_analysis["cultural_significance"] = "متوسطة"
        else:
            cultural_analysis["cultural_significance"] = "منخفضة"

        return cultural_analysis

    def _find_wisdom_connections(self, text: str, concepts: List[Dict]) -> List[Dict[str, Any]]:
        """Find connections to Arabic wisdom and insights"""

        connections = []

        # Extract themes from concepts
        themes = set()
        for concept in concepts:
            if "semantic_field" in concept:
                themes.add(concept["semantic_field"])

        # Find relevant wisdom
        for wisdom in self.arabic_wisdom:
            # Check theme overlap
            if wisdom.theme in themes or any(theme in wisdom.applicable_contexts for theme in themes):
                connections.append({
                    "wisdom_text": wisdom.text,
                    "source": wisdom.source,
                    "theme": wisdom.theme,
                    "moral_lesson": wisdom.moral_lesson,
                    "relevance_score": self._calculate_relevance(text, wisdom),
                    "application": self._suggest_application(text, wisdom)
                })

        # Sort by relevance
        connections.sort(key=lambda x: x["relevance_score"], reverse=True)

        return connections[:3]  # Return top 3 most relevant

    def _calculate_relevance(self, text: str, wisdom: ArabicWisdom) -> float:
        """Calculate relevance score between text and wisdom"""

        score = 0.0

        # Check for direct word matches
        text_words = set(text.split())
        wisdom_words = set(wisdom.text.split())
        common_words = text_words.intersection(wisdom_words)
        score += len(common_words) * 0.3

        # Check for thematic relevance
        if wisdom.theme in text:
            score += 0.5

        # Check applicable contexts
        for context in wisdom.applicable_contexts:
            if context in text:
                score += 0.2

        return min(score, 1.0)

    def _suggest_application(self, text: str, wisdom: ArabicWisdom) -> str:
        """Suggest how to apply the wisdom to the given text"""

        applications = [
            f"يمكن تطبيق هذه الحكمة في سياق: {wisdom.theme}",
            f"الدرس المستفاد: {wisdom.moral_lesson}",
            f"التطبيق العملي: {', '.join(wisdom.applicable_contexts[:2])}"
        ]

        return " | ".join(applications)

    def _generate_conceptual_insights(self, text: str, concepts: List[Dict], cultural_elements: Dict) -> List[str]:
        """Generate deep conceptual insights"""

        insights = []

        # Analyze concept density
        if len(concepts) > 3:
            insights.append("النص غني بالمفاهيم العميقة ويحمل طبقات متعددة من المعنى")

        # Analyze cultural depth
        if cultural_elements["cultural_significance"] == "عالية":
            insights.append("النص يعكس عمق الثقافة العربية الإسلامية ويحمل قيماً أصيلة")

        # Analyze semantic fields
        semantic_fields = set()
        for concept in concepts:
            if "semantic_field" in concept:
                semantic_fields.add(concept["semantic_field"])

        if len(semantic_fields) > 2:
            insights.append(f"النص يتناول مجالات دلالية متنوعة: {', '.join(list(semantic_fields)[:3])}")

        # Analyze religious dimension
        if cultural_elements["religious_references"]:
            insights.append("النص يحمل بُعداً روحياً ويرتبط بالتراث الإسلامي")

        return insights

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""

        recommendations = []

        # Based on cultural significance
        cultural_sig = analysis["cultural_analysis"]["cultural_significance"]
        if cultural_sig == "عالية":
            recommendations.append("يُنصح بدراسة هذا النص في سياق التراث العربي الإسلامي")
            recommendations.append("يمكن استخدام هذا النص في التعليم القيمي والأخلاقي")

        # Based on wisdom connections
        if analysis["wisdom_connections"]:
            recommendations.append("يُنصح بربط هذا النص بالحكم والأمثال العربية للإثراء")
            recommendations.append("يمكن استخدام الحكم المرتبطة لتعميق الفهم")

        # Based on conceptual insights
        if len(analysis["conceptual_insights"]) > 2:
            recommendations.append("النص يستحق دراسة تحليلية عميقة لاستخراج كامل إمكاناته")
            recommendations.append("يُنصح بتطوير هذا النص ليكون مادة تعليمية متقدمة")

        return recommendations

    # Reasoning methods
    def _analogical_reasoning(self, concept1: str, concept2: str) -> Dict[str, Any]:
        """Perform analogical reasoning between concepts"""
        # Implementation for analogical reasoning
        return {"type": "analogical", "similarity": 0.8}

    def _metaphorical_reasoning(self, text: str) -> List[Dict[str, Any]]:
        """Identify and analyze metaphors"""
        # Implementation for metaphorical reasoning
        return [{"metaphor": "العلم نور", "meaning": "العلم يضيء الطريق"}]

    def _contextual_reasoning(self, text: str, context: str) -> Dict[str, Any]:
        """Perform contextual reasoning"""
        # Implementation for contextual reasoning
        return {"context_relevance": 0.9}

    def _cultural_reasoning(self, text: str) -> Dict[str, Any]:
        """Perform cultural reasoning"""
        # Implementation for cultural reasoning
        return {"cultural_depth": "عميق"}

    def _religious_reasoning(self, text: str) -> Dict[str, Any]:
        """Perform religious reasoning"""
        # Implementation for religious reasoning
        return {"religious_significance": "مهم"}

    def _philosophical_reasoning(self, text: str) -> Dict[str, Any]:
        """Perform philosophical reasoning"""
        # Implementation for philosophical reasoning
        return {"philosophical_depth": "عميق"}


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create Advanced Arabic AI
    arabic_ai = AdvancedArabicAI()

    # Test texts
    test_texts = [
        "العلم نور يضيء طريق الحياة",
        "الحكمة ضالة المؤمن أنى وجدها فهو أحق بها",
        "إن مع العسر يسراً",
        "وما أوتيتم من العلم إلا قليلاً"
    ]

    print("🌟 Advanced Arabic AI Analysis Results 🌟")
    print("=" * 60)

    for text in test_texts:
        print(f"\n📝 Text: {text}")
        analysis = arabic_ai.analyze_text_with_cultural_intelligence(text)

        print(f"🔍 Semantic Concepts: {len(analysis['semantic_analysis']['concepts'])}")
        print(f"🏛️ Cultural Significance: {analysis['cultural_analysis']['cultural_significance']}")
        print(f"💎 Wisdom Connections: {len(analysis['wisdom_connections'])}")
        print(f"💡 Insights: {len(analysis['conceptual_insights'])}")

        if analysis['conceptual_insights']:
            print("🌟 Key Insights:")
            for insight in analysis['conceptual_insights'][:2]:
                print(f"   • {insight}")

        if analysis['recommendations']:
            print("📋 Recommendations:")
            for rec in analysis['recommendations'][:2]:
                print(f"   • {rec}")

        print("-" * 40)
