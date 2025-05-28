#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام تفسير الأحلام المتقدم في بصيرة

هذا النظام يوفر تفسيراً شاملاً ومتطوراً للأحلام باستخدام:
- دلالات الحروف والكلمات
- قواعد البيانات التراثية
- الذكاء الاصطناعي والتعلم الآلي
- التفسير متعدد الطبقات والثقافات
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import re
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# استيراد مكونات بصيرة
from ..symbolic_processing.letter_semantics.semantic_analyzer import LetterSemanticAnalyzer
from ..core.ai_oop.base_thing import Thing
from ..learning_systems.reinforcement_learning.rl_system import ReinforcementLearningSystem

class InterpretationLayer(Enum):
    """طبقات التفسير المختلفة"""
    LITERAL = "literal"          # التفسير الحرفي
    SYMBOLIC = "symbolic"        # التفسير الرمزي
    PSYCHOLOGICAL = "psychological"  # التفسير النفسي
    SPIRITUAL = "spiritual"      # التفسير الروحي
    CULTURAL = "cultural"        # التفسير الثقافي

class DreamContext(Enum):
    """سياق الحلم"""
    PERSONAL = "personal"        # شخصي
    FAMILY = "family"           # عائلي
    WORK = "work"              # مهني
    SPIRITUAL = "spiritual"     # روحي
    HEALTH = "health"          # صحي

@dataclass
class DreamSymbol:
    """رمز من رموز الحلم"""
    symbol: str
    category: str
    traditional_meanings: List[str]
    modern_meanings: List[str]
    cultural_context: List[str]
    emotional_associations: List[str]
    frequency: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "category": self.category,
            "traditional_meanings": self.traditional_meanings,
            "modern_meanings": self.modern_meanings,
            "cultural_context": self.cultural_context,
            "emotional_associations": self.emotional_associations,
            "frequency": self.frequency
        }

@dataclass
class DreamInterpretation:
    """نتيجة تفسير الحلم"""
    dream_text: str
    symbols_found: List[DreamSymbol]
    interpretations_by_layer: Dict[InterpretationLayer, str]
    overall_interpretation: str
    confidence_score: float
    recommendations: List[str]
    warnings: List[str]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dream_text": self.dream_text,
            "symbols_found": [symbol.to_dict() for symbol in self.symbols_found],
            "interpretations_by_layer": {layer.value: interp for layer, interp in self.interpretations_by_layer.items()},
            "overall_interpretation": self.overall_interpretation,
            "confidence_score": self.confidence_score,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat()
        }

class DreamSymbolDatabase:
    """قاعدة بيانات رموز الأحلام"""

    def __init__(self):
        self.symbols = {}
        self.categories = set()
        self.logger = logging.getLogger("dream_symbol_db")
        self._initialize_traditional_symbols()

    def _initialize_traditional_symbols(self):
        """تهيئة الرموز التقليدية"""
        traditional_symbols = {
            "ماء": DreamSymbol(
                symbol="ماء",
                category="طبيعة",
                traditional_meanings=["حياة", "طهارة", "رزق", "علم"],
                modern_meanings=["تجديد", "عواطف", "تدفق الحياة"],
                cultural_context=["إسلامي", "عربي"],
                emotional_associations=["سكينة", "انتعاش", "تطهير"]
            ),
            "نار": DreamSymbol(
                symbol="نار",
                category="عناصر",
                traditional_meanings=["فتنة", "حرب", "سلطان", "غضب"],
                modern_meanings=["شغف", "تحول", "طاقة"],
                cultural_context=["إسلامي", "عربي"],
                emotional_associations=["خوف", "قوة", "تدمير", "تطهير"]
            ),
            "طيران": DreamSymbol(
                symbol="طيران",
                category="حركة",
                traditional_meanings=["سفر", "علو مقام", "تحرر"],
                modern_meanings=["حرية", "طموح", "تجاوز الحدود"],
                cultural_context=["عالمي"],
                emotional_associations=["حرية", "خفة", "سعادة"]
            ),
            "موت": DreamSymbol(
                symbol="موت",
                category="أحداث",
                traditional_meanings=["نهاية مرحلة", "تغيير", "توبة"],
                modern_meanings=["تحول", "بداية جديدة", "تخلص من القديم"],
                cultural_context=["عالمي"],
                emotional_associations=["خوف", "حزن", "قلق", "تحرر"]
            ),
            "بيت": DreamSymbol(
                symbol="بيت",
                category="مكان",
                traditional_meanings=["أمان", "عائلة", "استقرار"],
                modern_meanings=["هوية", "راحة", "خصوصية"],
                cultural_context=["عالمي"],
                emotional_associations=["أمان", "دفء", "انتماء"]
            ),
            "شمس": DreamSymbol(
                symbol="شمس",
                category="أجرام",
                traditional_meanings=["ملك", "سلطان", "عدل", "هداية"],
                modern_meanings=["وضوح", "طاقة", "حيوية", "إشراق"],
                cultural_context=["عالمي"],
                emotional_associations=["دفء", "أمل", "قوة", "إشراق"]
            ),
            "قمر": DreamSymbol(
                symbol="قمر",
                category="أجرام",
                traditional_meanings=["وزير", "عالم", "جمال", "هدوء"],
                modern_meanings=["حدس", "أنوثة", "دورات", "غموض"],
                cultural_context=["عالمي"],
                emotional_associations=["هدوء", "رومانسية", "سكينة", "تأمل"]
            ),
            "شجرة": DreamSymbol(
                symbol="شجرة",
                category="نباتات",
                traditional_meanings=["عمر", "نسل", "خير", "بركة"],
                modern_meanings=["نمو", "استقرار", "جذور", "تطور"],
                cultural_context=["عالمي"],
                emotional_associations=["استقرار", "نمو", "حياة", "ظل"]
            ),
            "طريق": DreamSymbol(
                symbol="طريق",
                category="مكان",
                traditional_meanings=["سفر", "هجرة", "تغيير", "مسار"],
                modern_meanings=["خيارات", "مستقبل", "رحلة", "قرارات"],
                cultural_context=["عالمي"],
                emotional_associations=["ترقب", "قلق", "أمل", "مغامرة"]
            ),
            "مطر": DreamSymbol(
                symbol="مطر",
                category="طقس",
                traditional_meanings=["رحمة", "رزق", "خصب", "بركة"],
                modern_meanings=["تجديد", "تطهير", "نمو", "انتعاش"],
                cultural_context=["عالمي"],
                emotional_associations=["فرح", "انتعاش", "تطهير", "بركة"]
            ),
            "ثعبان": DreamSymbol(
                symbol="ثعبان",
                category="حيوانات",
                traditional_meanings=["عدو", "خصم", "مكر", "خطر"],
                modern_meanings=["تحول", "شفاء", "حكمة", "قوة خفية"],
                cultural_context=["متنوع"],
                emotional_associations=["خوف", "حذر", "قوة", "غموض"]
            ),
            "أسد": DreamSymbol(
                symbol="أسد",
                category="حيوانات",
                traditional_meanings=["ملك", "قوة", "شجاعة", "سلطان"],
                modern_meanings=["قيادة", "شجاعة", "كبرياء", "قوة"],
                cultural_context=["عالمي"],
                emotional_associations=["قوة", "هيبة", "شجاعة", "كبرياء"]
            ),
            "طائر": DreamSymbol(
                symbol="طائر",
                category="حيوانات",
                traditional_meanings=["رسالة", "خبر", "روح", "حرية"],
                modern_meanings=["حرية", "رسائل", "أحلام", "طموح"],
                cultural_context=["عالمي"],
                emotional_associations=["حرية", "خفة", "جمال", "سلام"]
            ),
            "سمك": DreamSymbol(
                symbol="سمك",
                category="حيوانات",
                traditional_meanings=["رزق", "مال", "خير", "بركة"],
                modern_meanings=["وفرة", "خصوبة", "عمق", "لاوعي"],
                cultural_context=["عالمي"],
                emotional_associations=["وفرة", "سلام", "عمق", "هدوء"]
            ),
            "ذهب": DreamSymbol(
                symbol="ذهب",
                category="معادن",
                traditional_meanings=["مال", "زينة", "فتنة", "ثروة"],
                modern_meanings=["قيمة", "نجاح", "تقدير", "إنجاز"],
                cultural_context=["عالمي"],
                emotional_associations=["فخر", "نجاح", "قيمة", "جمال"]
            ),
            "فضة": DreamSymbol(
                symbol="فضة",
                category="معادن",
                traditional_meanings=["مال", "جمال", "طهارة", "بركة"],
                modern_meanings=["وضوح", "نقاء", "حدس", "أنوثة"],
                cultural_context=["عالمي"],
                emotional_associations=["نقاء", "جمال", "هدوء", "وضوح"]
            )
        }

        for symbol_name, symbol_obj in traditional_symbols.items():
            self.add_symbol(symbol_obj)

    def add_symbol(self, symbol: DreamSymbol) -> None:
        """إضافة رمز جديد"""
        self.symbols[symbol.symbol] = symbol
        self.categories.add(symbol.category)
        self.logger.info(f"تم إضافة رمز: {symbol.symbol}")

    def get_symbol(self, symbol_name: str) -> Optional[DreamSymbol]:
        """الحصول على رمز"""
        return self.symbols.get(symbol_name)

    def search_symbols(self, query: str) -> List[DreamSymbol]:
        """البحث عن رموز"""
        results = []
        query_lower = query.lower()

        for symbol in self.symbols.values():
            if (query_lower in symbol.symbol.lower() or
                any(query_lower in meaning.lower() for meaning in symbol.traditional_meanings) or
                any(query_lower in meaning.lower() for meaning in symbol.modern_meanings)):
                results.append(symbol)

        return results

    def get_symbols_by_category(self, category: str) -> List[DreamSymbol]:
        """الحصول على رموز حسب الفئة"""
        return [symbol for symbol in self.symbols.values() if symbol.category == category]

class AdvancedDreamInterpreter:
    """مفسر الأحلام المتقدم"""

    def __init__(self, semantic_analyzer: LetterSemanticAnalyzer = None):
        self.symbol_db = DreamSymbolDatabase()
        self.semantic_analyzer = semantic_analyzer
        self.rl_system = ReinforcementLearningSystem()
        self.interpretation_history = []
        self.user_feedback = {}
        self.logger = logging.getLogger("dream_interpreter")

        # تهيئة المحلل الدلالي إذا لم يتم توفيره
        if self.semantic_analyzer is None:
            from ..symbolic_processing.data.initial_letter_semantics_data2 import get_initial_letter_semantics_data
            semantics_data = get_initial_letter_semantics_data()
            self.semantic_analyzer = LetterSemanticAnalyzer(semantics_data)

    def interpret_dream(self, dream_text: str, context: Optional[Dict[str, Any]] = None) -> DreamInterpretation:
        """
        تفسير حلم شامل

        Args:
            dream_text: نص الحلم
            context: سياق إضافي (عمر الحالم، جنسه، حالته الاجتماعية، إلخ)

        Returns:
            تفسير شامل للحلم
        """
        self.logger.info(f"بدء تفسير حلم: {dream_text[:50]}...")

        context = context or {}

        # استخراج الرموز من النص
        symbols_found = self._extract_symbols(dream_text)

        # تفسير كل طبقة
        interpretations_by_layer = {}

        # التفسير الحرفي
        interpretations_by_layer[InterpretationLayer.LITERAL] = self._interpret_literal(symbols_found, context)

        # التفسير الرمزي
        interpretations_by_layer[InterpretationLayer.SYMBOLIC] = self._interpret_symbolic(symbols_found, context)

        # التفسير النفسي
        interpretations_by_layer[InterpretationLayer.PSYCHOLOGICAL] = self._interpret_psychological(symbols_found, context)

        # التفسير الروحي
        interpretations_by_layer[InterpretationLayer.SPIRITUAL] = self._interpret_spiritual(symbols_found, context)

        # التفسير الثقافي
        interpretations_by_layer[InterpretationLayer.CULTURAL] = self._interpret_cultural(symbols_found, context)

        # التفسير الشامل
        overall_interpretation = self._generate_overall_interpretation(interpretations_by_layer, symbols_found, context)

        # حساب درجة الثقة
        confidence_score = self._calculate_confidence(symbols_found, context)

        # توليد التوصيات والتحذيرات
        recommendations = self._generate_recommendations(symbols_found, interpretations_by_layer, context)
        warnings = self._generate_warnings(symbols_found, interpretations_by_layer, context)

        # إنشاء نتيجة التفسير
        interpretation = DreamInterpretation(
            dream_text=dream_text,
            symbols_found=symbols_found,
            interpretations_by_layer=interpretations_by_layer,
            overall_interpretation=overall_interpretation,
            confidence_score=confidence_score,
            recommendations=recommendations,
            warnings=warnings,
            timestamp=datetime.now()
        )

        # حفظ التفسير في التاريخ
        self.interpretation_history.append(interpretation)

        # تسجيل التجربة للتعلم المعزز
        state = f"dream_interpretation_{len(symbols_found)}_symbols"
        action = "interpret_dream"
        reward = confidence_score  # سيتم تحديثه بناءً على تقييم المستخدم
        next_state = f"interpretation_completed"

        self.rl_system.record_experience(state, action, reward, next_state)

        self.logger.info(f"تم تفسير الحلم بنجاح، درجة الثقة: {confidence_score:.2f}")

        return interpretation

    def _extract_symbols(self, dream_text: str) -> List[DreamSymbol]:
        """استخراج الرموز من نص الحلم"""
        symbols_found = []

        # تنظيف النص
        cleaned_text = re.sub(r'[^\w\s]', ' ', dream_text)
        words = cleaned_text.split()

        # البحث عن رموز معروفة
        for word in words:
            symbol = self.symbol_db.get_symbol(word)
            if symbol:
                symbols_found.append(symbol)
                symbol.frequency += 1

        # استخدام المحلل الدلالي لاستخراج معاني إضافية
        for word in words:
            if not any(s.symbol == word for s in symbols_found):
                semantic_analysis = self.semantic_analyzer.analyze_word(word, "ar")
                if semantic_analysis and semantic_analysis.get('combined_semantics'):
                    # إنشاء رمز مؤقت بناءً على التحليل الدلالي
                    temp_symbol = DreamSymbol(
                        symbol=word,
                        category="مستخرج_دلالياً",
                        traditional_meanings=semantic_analysis['combined_semantics'].get('primary_connotations', []),
                        modern_meanings=[],
                        cultural_context=["عربي"],
                        emotional_associations=[]
                    )
                    symbols_found.append(temp_symbol)

        return symbols_found

    def _interpret_literal(self, symbols: List[DreamSymbol], context: Dict[str, Any]) -> str:
        """التفسير الحرفي للرموز"""
        if not symbols:
            return "لا توجد رموز واضحة في الحلم للتفسير الحرفي."

        interpretation = "التفسير الحرفي: "
        for symbol in symbols:
            if symbol.traditional_meanings:
                interpretation += f"{symbol.symbol} يدل على {', '.join(symbol.traditional_meanings[:2])}. "

        return interpretation

    def _interpret_symbolic(self, symbols: List[DreamSymbol], context: Dict[str, Any]) -> str:
        """التفسير الرمزي"""
        # سيتم تطويره بشكل أكثر تفصيلاً
        return "التفسير الرمزي: الرموز في الحلم تشير إلى تحولات داخلية وخارجية في حياة الحالم."

    def _interpret_psychological(self, symbols: List[DreamSymbol], context: Dict[str, Any]) -> str:
        """التفسير النفسي"""
        # سيتم تطويره بشكل أكثر تفصيلاً
        return "التفسير النفسي: الحلم يعكس حالة نفسية معينة وصراعات داخلية."

    def _interpret_spiritual(self, symbols: List[DreamSymbol], context: Dict[str, Any]) -> str:
        """التفسير الروحي"""
        # سيتم تطويره بشكل أكثر تفصيلاً
        return "التفسير الروحي: الحلم قد يحمل رسالة روحية أو إرشاد إلهي."

    def _interpret_cultural(self, symbols: List[DreamSymbol], context: Dict[str, Any]) -> str:
        """التفسير الثقافي"""
        # سيتم تطويره بشكل أكثر تفصيلاً
        return "التفسير الثقافي: الرموز تحمل معاني ثقافية خاصة بالمجتمع العربي الإسلامي."

    def _generate_overall_interpretation(self, interpretations: Dict[InterpretationLayer, str],
                                       symbols: List[DreamSymbol], context: Dict[str, Any]) -> str:
        """توليد التفسير الشامل"""
        overall = "التفسير الشامل:\n\n"

        # دمج التفسيرات من الطبقات المختلفة
        for layer, interpretation in interpretations.items():
            overall += f"{layer.value.title()}: {interpretation}\n\n"

        # إضافة خلاصة
        overall += "الخلاصة: هذا الحلم يحمل رسائل متعددة الأبعاد تتطلب تأملاً وتفكيراً عميقاً."

        return overall

    def _calculate_confidence(self, symbols: List[DreamSymbol], context: Dict[str, Any]) -> float:
        """حساب درجة الثقة في التفسير"""
        if not symbols:
            return 0.1

        # حساب الثقة بناءً على عدد الرموز المعروفة
        known_symbols = len([s for s in symbols if s.category != "مستخرج_دلالياً"])
        total_symbols = len(symbols)

        base_confidence = known_symbols / total_symbols if total_symbols > 0 else 0

        # تعديل الثقة بناءً على السياق
        if context:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _generate_recommendations(self, symbols: List[DreamSymbol],
                                interpretations: Dict[InterpretationLayer, str],
                                context: Dict[str, Any]) -> List[str]:
        """توليد التوصيات"""
        recommendations = [
            "تأمل في معاني الحلم وربطها بواقعك",
            "استشر أهل العلم إذا كان الحلم يحمل رسالة مهمة",
            "لا تبني قرارات مصيرية على تفسير الأحلام وحده"
        ]

        return recommendations

    def _generate_warnings(self, symbols: List[DreamSymbol],
                         interpretations: Dict[InterpretationLayer, str],
                         context: Dict[str, Any]) -> List[str]:
        """توليد التحذيرات"""
        warnings = [
            "هذا التفسير اجتهادي وليس قطعياً",
            "الأحلام قد تكون أضغاث أحلام لا معنى لها",
            "لا تدع القلق من تفسير الحلم يؤثر على حياتك"
        ]

        return warnings

    def record_user_feedback(self, interpretation_id: int, feedback_score: float, comments: str = "") -> None:
        """تسجيل تقييم المستخدم للتفسير"""
        if 0 <= interpretation_id < len(self.interpretation_history):
            self.user_feedback[interpretation_id] = {
                "score": feedback_score,
                "comments": comments,
                "timestamp": datetime.now()
            }

            # تحديث نظام التعلم المعزز
            self.rl_system.record_experience(
                f"interpretation_{interpretation_id}",
                "user_feedback",
                feedback_score,
                "feedback_recorded"
            )

            self.logger.info(f"تم تسجيل تقييم المستخدم: {feedback_score}")

    def get_interpretation_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التفسيرات"""
        total_interpretations = len(self.interpretation_history)
        avg_confidence = sum(i.confidence_score for i in self.interpretation_history) / total_interpretations if total_interpretations > 0 else 0

        feedback_scores = [f["score"] for f in self.user_feedback.values()]
        avg_user_rating = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0

        return {
            "total_interpretations": total_interpretations,
            "average_confidence": avg_confidence,
            "average_user_rating": avg_user_rating,
            "total_symbols_in_db": len(self.symbol_db.symbols),
            "categories_count": len(self.symbol_db.categories)
        }
