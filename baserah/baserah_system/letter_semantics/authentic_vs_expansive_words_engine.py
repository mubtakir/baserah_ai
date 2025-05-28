#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authentic vs Expansive Words Engine - Distinguishing Original from Derived Words
محرك التمييز بين الكلمات الأصيلة والتوسعية - تمييز الكلمات الأصلية من المشتقة

Revolutionary system for distinguishing between authentic original words and expansive derived words:
- Identification of authentic ancient Arabic words that follow letter semantics rules
- Detection of expansive words derived through metaphor, cultural contact, and linguistic evolution
- Analysis of word etymology and historical development
- Semantic rule validation for authentic words
- Cultural and linguistic expansion pattern recognition

نظام ثوري للتمييز بين الكلمات الأصيلة القديمة والكلمات التوسعية المشتقة:
- تحديد الكلمات العربية الأصيلة القديمة التي تتبع قواعد دلالة الحروف
- اكتشاف الكلمات التوسعية المشتقة من خلال المجاز والاحتكاك الثقافي والتطور اللغوي
- تحليل أصل الكلمات وتطورها التاريخي
- التحقق من صحة القواعد الدلالية للكلمات الأصيلة
- التعرف على أنماط التوسع الثقافي واللغوي

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0 - Authentic Words Edition
Based on Basil's insight about authentic vs expansive words
"""

import numpy as np
import sys
import os
import json
import re
from typing import Dict, List, Any, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, Counter

# إضافة المسارات
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class WordType(str, Enum):
    """أنواع الكلمات"""
    AUTHENTIC_ANCIENT = "authentic_ancient"
    EXPANSIVE_METAPHORICAL = "expansive_metaphorical"
    EXPANSIVE_CULTURAL = "expansive_cultural"
    EXPANSIVE_BORROWED = "expansive_borrowed"
    EXPANSIVE_MODERN = "expansive_modern"
    UNKNOWN = "unknown"

class ExpansionMethod(str, Enum):
    """طرق التوسع اللغوي"""
    METAPHORICAL_EXTENSION = "metaphorical_extension"
    CULTURAL_CONTACT = "cultural_contact"
    SEMANTIC_SHIFT = "semantic_shift"
    BORROWING = "borrowing"
    ANALOGY = "analogy"
    MODERNIZATION = "modernization"

class AuthenticityLevel(str, Enum):
    """مستويات الأصالة"""
    HIGHLY_AUTHENTIC = "highly_authentic"
    MODERATELY_AUTHENTIC = "moderately_authentic"
    QUESTIONABLE = "questionable"
    LIKELY_EXPANSIVE = "likely_expansive"
    CLEARLY_EXPANSIVE = "clearly_expansive"

# محاكاة النظام المتكيف للتمييز بين الكلمات
class AuthenticWordEquation:
    def __init__(self, word_type: WordType, authenticity_level: AuthenticityLevel):
        self.word_type = word_type
        self.authenticity_level = authenticity_level
        self.analysis_cycles = 0
        self.semantic_rule_compliance = 0.8
        self.historical_evidence = 0.75
        self.etymological_clarity = 0.85
        self.cultural_expansion_detection = 0.9
        self.metaphorical_pattern_recognition = 0.7
        self.linguistic_contact_analysis = 0.8
        self.authentic_examples = []
        self.expansive_examples = []

    def evolve_with_authenticity_analysis(self, analysis_data, word_analysis):
        """التطور مع تحليل الأصالة"""
        self.analysis_cycles += 1

        if hasattr(analysis_data, 'expansion_method'):
            if analysis_data.expansion_method == ExpansionMethod.METAPHORICAL_EXTENSION:
                self.metaphorical_pattern_recognition += 0.08
                self.semantic_rule_compliance += 0.06
            elif analysis_data.expansion_method == ExpansionMethod.CULTURAL_CONTACT:
                self.cultural_expansion_detection += 0.09
                self.linguistic_contact_analysis += 0.07
            elif analysis_data.expansion_method == ExpansionMethod.SEMANTIC_SHIFT:
                self.etymological_clarity += 0.08
                self.historical_evidence += 0.06

    def get_authenticity_summary(self):
        """الحصول على ملخص الأصالة"""
        return {
            "word_type": self.word_type.value,
            "authenticity_level": self.authenticity_level.value,
            "analysis_cycles": self.analysis_cycles,
            "semantic_rule_compliance": self.semantic_rule_compliance,
            "historical_evidence": self.historical_evidence,
            "etymological_clarity": self.etymological_clarity,
            "cultural_expansion_detection": self.cultural_expansion_detection,
            "metaphorical_pattern_recognition": self.metaphorical_pattern_recognition,
            "linguistic_contact_analysis": self.linguistic_contact_analysis,
            "authentic_examples": self.authentic_examples,
            "expansive_examples": self.expansive_examples,
            "authenticity_excellence_index": self._calculate_authenticity_excellence()
        }

    def _calculate_authenticity_excellence(self) -> float:
        """حساب مؤشر تميز الأصالة"""
        return (
            self.semantic_rule_compliance * 0.25 +
            self.historical_evidence * 0.2 +
            self.etymological_clarity * 0.2 +
            self.cultural_expansion_detection * 0.15 +
            self.metaphorical_pattern_recognition * 0.1 +
            self.linguistic_contact_analysis * 0.1
        )

@dataclass
class WordAuthenticityRequest:
    """طلب تحليل أصالة الكلمات"""
    target_words: List[str]
    analysis_methods: List[ExpansionMethod]
    authenticity_criteria: List[str]
    objective: str
    check_semantic_rules: bool = True
    analyze_etymology: bool = True
    detect_cultural_expansion: bool = True
    identify_metaphorical_usage: bool = True
    historical_validation: bool = True

@dataclass
class WordAuthenticityResult:
    """نتيجة تحليل أصالة الكلمات"""
    success: bool
    word_classifications: Dict[str, WordType]
    authenticity_levels: Dict[str, AuthenticityLevel]
    expansion_patterns: Dict[str, Any]
    authentic_word_examples: List[Dict[str, Any]]
    expansive_word_examples: List[Dict[str, Any]]
    semantic_rule_validation: Dict[str, Any]
    cultural_expansion_analysis: Dict[str, Any]
    expert_authenticity_evolution: Dict[str, Any] = None
    equation_analysis: Dict[str, Any] = None
    authenticity_advancement: Dict[str, float] = None
    next_analysis_recommendations: List[str] = None

class AuthenticVsExpansiveWordsEngine:
    """محرك التمييز بين الكلمات الأصيلة والتوسعية"""

    def __init__(self):
        """تهيئة محرك التمييز بين الكلمات الأصيلة والتوسعية"""
        print("🌟" + "="*160 + "🌟")
        print("🔤 محرك التمييز بين الكلمات الأصيلة والتوسعية - نظام تحليل أصالة الكلمات العربية")
        print("📚 مبني على رؤية باسل حول الكلمات الأصيلة القديمة مقابل الكلمات التوسعية")
        print("⚡ تمييز الأصيل من التوسعي + تحليل المجاز + اكتشاف الاحتكاك الثقافي")
        print("🧠 التحقق من القواعد الدلالية + تحليل التطور التاريخي + أنماط التوسع")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" + "="*160 + "🌟")

        # إنشاء معادلات تحليل الأصالة
        self.authenticity_equations = self._initialize_authenticity_equations()

        # قاعدة بيانات الكلمات الأصيلة والتوسعية
        self.word_authenticity_database = self._initialize_authenticity_database()

        # أنماط التوسع اللغوي
        self.expansion_patterns = self._initialize_expansion_patterns()

        # قواعد المعرفة للأصالة
        self.authenticity_knowledge_bases = {
            "basil_authentic_word_principle": {
                "name": "مبدأ باسل للكلمات الأصيلة",
                "principle": "الكلمات الأصيلة القديمة تتبع قواعد دلالة الحروف بدقة",
                "authenticity_meaning": "القواعد الدلالية تنطبق على الكلمات الأصيلة وليس التوسعية"
            },
            "linguistic_expansion_laws": {
                "name": "قوانين التوسع اللغوي",
                "principle": "اللغة تتوسع بالمجاز والاحتكاك الثقافي والاستعارة",
                "authenticity_meaning": "التوسع اللغوي يخلق كلمات لا تتبع القواعد الأصلية"
            },
            "cultural_contact_wisdom": {
                "name": "حكمة الاحتكاك الثقافي",
                "principle": "احتكاك الشعوب يؤثر على المفردات ويخلق كلمات جديدة",
                "authenticity_meaning": "الكلمات المستعارة قد لا تتبع قواعد اللغة الأصلية"
            }
        }

        # تاريخ تحليل الأصالة
        self.authenticity_analysis_history = []
        self.authenticity_learning_database = {}

        # نظام التطور في تحليل الأصالة
        self.authenticity_evolution_engine = self._initialize_authenticity_evolution()

        print("🔤 تم إنشاء معادلات تحليل الأصالة:")
        for eq_name, equation in self.authenticity_equations.items():
            print(f"   ✅ {eq_name} - نوع: {equation.word_type.value} - مستوى: {equation.authenticity_level.value}")

        print("✅ تم تهيئة محرك التمييز بين الكلمات الأصيلة والتوسعية!")

    def _initialize_authenticity_equations(self) -> Dict[str, AuthenticWordEquation]:
        """تهيئة معادلات تحليل الأصالة"""
        equations = {}

        # معادلات للكلمات الأصيلة
        equations["authentic_ancient_analyzer"] = AuthenticWordEquation(
            WordType.AUTHENTIC_ANCIENT, AuthenticityLevel.HIGHLY_AUTHENTIC
        )

        # معادلات للكلمات التوسعية
        equations["metaphorical_expansion_detector"] = AuthenticWordEquation(
            WordType.EXPANSIVE_METAPHORICAL, AuthenticityLevel.CLEARLY_EXPANSIVE
        )

        equations["cultural_contact_analyzer"] = AuthenticWordEquation(
            WordType.EXPANSIVE_CULTURAL, AuthenticityLevel.LIKELY_EXPANSIVE
        )

        equations["borrowed_word_identifier"] = AuthenticWordEquation(
            WordType.EXPANSIVE_BORROWED, AuthenticityLevel.CLEARLY_EXPANSIVE
        )

        equations["modern_expansion_tracker"] = AuthenticWordEquation(
            WordType.EXPANSIVE_MODERN, AuthenticityLevel.CLEARLY_EXPANSIVE
        )

        equations["questionable_word_evaluator"] = AuthenticWordEquation(
            WordType.UNKNOWN, AuthenticityLevel.QUESTIONABLE
        )

        return equations

    def _initialize_authenticity_database(self) -> Dict[str, Dict[str, Any]]:
        """تهيئة قاعدة بيانات الأصالة"""
        return {
            # أمثلة الكلمات الأصيلة من باسل
            "طلب": {
                "word_type": WordType.AUTHENTIC_ANCIENT,
                "authenticity_level": AuthenticityLevel.HIGHLY_AUTHENTIC,
                "semantic_rule_compliance": 0.95,
                "letter_analysis": {
                    "ط": "الطرق والاستئذان",
                    "ل": "الالتفاف والإحاطة",
                    "ب": "الانتقال والتشبع"
                },
                "basil_validation": True,
                "historical_evidence": "كلمة أصيلة قديمة",
                "expansion_history": []
            },
            "سلب": {
                "word_type": WordType.AUTHENTIC_ANCIENT,
                "authenticity_level": AuthenticityLevel.HIGHLY_AUTHENTIC,
                "semantic_rule_compliance": 0.92,
                "letter_analysis": {
                    "س": "الانسياب والسلاسة",
                    "ل": "الالتفاف والإحاطة",
                    "ب": "الانتقال والتشبع"
                },
                "basil_validation": True,
                "historical_evidence": "كلمة أصيلة قديمة",
                "expansion_history": []
            },
            "نهب": {
                "word_type": WordType.AUTHENTIC_ANCIENT,
                "authenticity_level": AuthenticityLevel.HIGHLY_AUTHENTIC,
                "semantic_rule_compliance": 0.90,
                "letter_analysis": {
                    "ن": "التشكيل والتكوين",
                    "ه": "الهدوء والسكينة",
                    "ب": "الانتقال والتشبع"
                },
                "basil_validation": True,
                "historical_evidence": "كلمة أصيلة قديمة",
                "expansion_history": []
            },
            "حلب": {
                "word_type": WordType.AUTHENTIC_ANCIENT,
                "authenticity_level": AuthenticityLevel.HIGHLY_AUTHENTIC,
                "semantic_rule_compliance": 0.88,
                "letter_analysis": {
                    "ح": "الحياة والحيوية",
                    "ل": "الالتفاف والإحاطة",
                    "ب": "الانتقال والتشبع"
                },
                "basil_validation": True,
                "historical_evidence": "كلمة أصيلة قديمة",
                "expansion_history": []
            },
            # أمثلة الكلمات التوسعية
            "هيجان": {
                "word_type": WordType.EXPANSIVE_METAPHORICAL,
                "authenticity_level": AuthenticityLevel.CLEARLY_EXPANSIVE,
                "semantic_rule_compliance": 0.3,
                "original_meaning": "هيجان البحر",
                "expanded_meaning": "الإنسان الساخط والغاضب",
                "expansion_method": ExpansionMethod.METAPHORICAL_EXTENSION,
                "expansion_history": [
                    "الأصل: حركة البحر العنيفة",
                    "التوسع: نقل المعنى للإنسان الغاضب"
                ]
            }
        }

    def _initialize_expansion_patterns(self) -> Dict[str, Any]:
        """تهيئة أنماط التوسع"""
        return {
            "metaphorical_patterns": [
                "نقل صفات الطبيعة للإنسان (هيجان البحر → هيجان الإنسان)",
                "تشبيه الأفعال البشرية بالحيوانية",
                "استعارة خصائص الأشياء للمعاني المجردة"
            ],
            "cultural_contact_patterns": [
                "استعارة كلمات من لغات أخرى",
                "تأثر بمفردات الشعوب المجاورة",
                "دمج مصطلحات تجارية وثقافية"
            ],
            "semantic_shift_patterns": [
                "تطور المعنى من الحسي إلى المجرد",
                "توسع المعنى من الخاص إلى العام",
                "تغيير الدلالة بسبب الاستخدام"
            ]
        }

    def _initialize_authenticity_evolution(self) -> Dict[str, Any]:
        """تهيئة محرك تطور الأصالة"""
        return {
            "evolution_cycles": 0,
            "authentic_word_mastery": 0.0,
            "expansion_detection_accuracy": 0.0,
            "semantic_rule_validation": 0.0,
            "cultural_pattern_recognition": 0.0,
            "historical_analysis_capability": 0.0
        }

    def analyze_word_authenticity(self, request: WordAuthenticityRequest) -> WordAuthenticityResult:
        """تحليل أصالة الكلمات"""
        print(f"\n🔍 بدء تحليل أصالة الكلمات: {', '.join(request.target_words)}")
        start_time = datetime.now()

        # المرحلة 1: تحليل طلب الأصالة
        authenticity_analysis = self._analyze_authenticity_request(request)
        print(f"📊 تحليل الأصالة: {authenticity_analysis['complexity_level']}")

        # المرحلة 2: توليد التوجيه الخبير للأصالة
        authenticity_guidance = self._generate_authenticity_expert_guidance(request, authenticity_analysis)
        print(f"🎯 التوجيه: {authenticity_guidance.primary_method.value}")

        # المرحلة 3: تطوير معادلات الأصالة
        equation_analysis = self._evolve_authenticity_equations(authenticity_guidance, authenticity_analysis)
        print(f"⚡ تطوير المعادلات: {len(equation_analysis)} معادلة")

        # المرحلة 4: التحقق من القواعد الدلالية
        semantic_rule_validation = self._validate_semantic_rules(request, equation_analysis)

        # المرحلة 5: تحليل التأصيل التاريخي
        historical_analysis = self._analyze_historical_etymology(request, semantic_rule_validation)

        # المرحلة 6: اكتشاف أنماط التوسع
        expansion_pattern_detection = self._detect_expansion_patterns(request, historical_analysis)

        # المرحلة 7: تصنيف الكلمات
        word_classification = self._classify_words(request, expansion_pattern_detection)

        # المرحلة 8: تحديد مستويات الأصالة
        authenticity_levels = self._determine_authenticity_levels(request, word_classification)

        # المرحلة 9: تحليل التوسع الثقافي
        cultural_expansion_analysis = self._analyze_cultural_expansion(request, authenticity_levels)

        # المرحلة 10: جمع أمثلة الكلمات الأصيلة والتوسعية
        authentic_examples, expansive_examples = self._collect_word_examples(request, cultural_expansion_analysis)

        # المرحلة 11: التطور في تحليل الأصالة
        authenticity_advancement = self._advance_authenticity_intelligence(equation_analysis, word_classification)

        # المرحلة 12: توليد توصيات التحليل التالية
        next_recommendations = self._generate_authenticity_recommendations(word_classification, authenticity_advancement)

        # إنشاء النتيجة
        result = WordAuthenticityResult(
            success=True,
            word_classifications=word_classification["classifications"],
            authenticity_levels=authenticity_levels["levels"],
            expansion_patterns=expansion_pattern_detection,
            authentic_word_examples=authentic_examples,
            expansive_word_examples=expansive_examples,
            semantic_rule_validation=semantic_rule_validation,
            cultural_expansion_analysis=cultural_expansion_analysis,
            expert_authenticity_evolution=authenticity_guidance.__dict__,
            equation_analysis=equation_analysis,
            authenticity_advancement=authenticity_advancement,
            next_analysis_recommendations=next_recommendations
        )

        # حفظ في قاعدة تحليل الأصالة
        self._save_authenticity_analysis(request, result)

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ انتهى تحليل الأصالة في {total_time:.2f} ثانية")
        print(f"🔤 كلمات محللة: {len(result.word_classifications)}")
        print(f"📊 مستويات أصالة: {len(result.authenticity_levels)}")

        return result

    def _analyze_authenticity_request(self, request: WordAuthenticityRequest) -> Dict[str, Any]:
        """تحليل طلب الأصالة"""

        # تحليل تعقيد الكلمات
        word_complexity = len(request.target_words) * 8.0

        # تحليل طرق التحليل
        method_richness = len(request.analysis_methods) * 12.0

        # تحليل معايير الأصالة
        criteria_complexity = len(request.authenticity_criteria) * 6.0

        # تحليل التحقق من القواعد
        rule_checking_boost = 15.0 if request.check_semantic_rules else 5.0

        # تحليل التأصيل التاريخي
        etymology_boost = 12.0 if request.analyze_etymology else 4.0

        # تحليل اكتشاف التوسع الثقافي
        cultural_detection_boost = 10.0 if request.detect_cultural_expansion else 3.0

        total_authenticity_complexity = (
            word_complexity + method_richness + criteria_complexity +
            rule_checking_boost + etymology_boost + cultural_detection_boost
        )

        return {
            "word_complexity": word_complexity,
            "method_richness": method_richness,
            "criteria_complexity": criteria_complexity,
            "rule_checking_boost": rule_checking_boost,
            "etymology_boost": etymology_boost,
            "cultural_detection_boost": cultural_detection_boost,
            "total_authenticity_complexity": total_authenticity_complexity,
            "complexity_level": "تحليل أصالة متعالي معقد جداً" if total_authenticity_complexity > 90 else "تحليل أصالة متقدم معقد" if total_authenticity_complexity > 70 else "تحليل أصالة متوسط" if total_authenticity_complexity > 50 else "تحليل أصالة بسيط",
            "recommended_cycles": int(total_authenticity_complexity // 15) + 3,
            "semantic_rule_emphasis": 1.0 if request.check_semantic_rules else 0.3,
            "authenticity_focus": self._identify_authenticity_focus(request)
        }

    def _identify_authenticity_focus(self, request: WordAuthenticityRequest) -> List[str]:
        """تحديد التركيز في تحليل الأصالة"""
        focus_areas = []

        # تحليل طرق التحليل
        for method in request.analysis_methods:
            if method == ExpansionMethod.METAPHORICAL_EXTENSION:
                focus_areas.append("metaphorical_expansion_detection")
            elif method == ExpansionMethod.CULTURAL_CONTACT:
                focus_areas.append("cultural_contact_analysis")
            elif method == ExpansionMethod.SEMANTIC_SHIFT:
                focus_areas.append("semantic_shift_tracking")
            elif method == ExpansionMethod.BORROWING:
                focus_areas.append("borrowed_word_identification")
            elif method == ExpansionMethod.MODERNIZATION:
                focus_areas.append("modern_expansion_detection")

        # تحليل الميزات المطلوبة
        if request.check_semantic_rules:
            focus_areas.append("semantic_rule_validation")

        if request.analyze_etymology:
            focus_areas.append("etymological_analysis")

        if request.detect_cultural_expansion:
            focus_areas.append("cultural_expansion_detection")

        if request.identify_metaphorical_usage:
            focus_areas.append("metaphorical_usage_identification")

        if request.historical_validation:
            focus_areas.append("historical_validation")

        return focus_areas

    def _generate_authenticity_expert_guidance(self, request: WordAuthenticityRequest, analysis: Dict[str, Any]):
        """توليد التوجيه الخبير للأصالة"""

        # تحديد الطريقة الأساسية
        if "semantic_rule_validation" in analysis["authenticity_focus"]:
            primary_method = ExpansionMethod.SEMANTIC_SHIFT
            effectiveness = 0.95
        elif "metaphorical_expansion_detection" in analysis["authenticity_focus"]:
            primary_method = ExpansionMethod.METAPHORICAL_EXTENSION
            effectiveness = 0.9
        elif "cultural_contact_analysis" in analysis["authenticity_focus"]:
            primary_method = ExpansionMethod.CULTURAL_CONTACT
            effectiveness = 0.88
        elif "borrowed_word_identification" in analysis["authenticity_focus"]:
            primary_method = ExpansionMethod.BORROWING
            effectiveness = 0.85
        else:
            primary_method = ExpansionMethod.ANALOGY
            effectiveness = 0.8

        # استخدام فئة التوجيه للأصالة
        class AuthenticityGuidance:
            def __init__(self, primary_method, effectiveness, focus_areas, semantic_emphasis):
                self.primary_method = primary_method
                self.effectiveness = effectiveness
                self.focus_areas = focus_areas
                self.semantic_emphasis = semantic_emphasis
                self.basil_principle_application = analysis.get("semantic_rule_emphasis", 0.9)
                self.authenticity_quality_target = 0.95
                self.expansion_detection_precision = 0.92

        return AuthenticityGuidance(
            primary_method=primary_method,
            effectiveness=effectiveness,
            focus_areas=analysis["authenticity_focus"],
            semantic_emphasis=request.check_semantic_rules
        )

    def _evolve_authenticity_equations(self, guidance, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """تطوير معادلات الأصالة"""

        equation_analysis = {}

        # إنشاء تحليل وهمي للمعادلات
        class AuthenticityAnalysis:
            def __init__(self):
                self.semantic_rule_compliance = 0.85
                self.historical_evidence = 0.8
                self.etymological_clarity = 0.88
                self.cultural_expansion_detection = 0.9
                self.metaphorical_pattern_recognition = 0.75
                self.linguistic_contact_analysis = 0.82
                self.areas_for_improvement = guidance.focus_areas

        authenticity_analysis = AuthenticityAnalysis()

        # تطوير كل معادلة أصالة
        for eq_name, equation in self.authenticity_equations.items():
            print(f"   🔍 تطوير معادلة أصالة: {eq_name}")
            equation.evolve_with_authenticity_analysis(guidance, authenticity_analysis)
            equation_analysis[eq_name] = equation.get_authenticity_summary()

        return equation_analysis

    def _validate_semantic_rules(self, request: WordAuthenticityRequest, equations: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من القواعد الدلالية"""

        semantic_validation = {
            "word_validations": {},
            "rule_compliance_scores": {},
            "authentic_indicators": {},
            "expansive_indicators": {}
        }

        if request.check_semantic_rules:
            for word in request.target_words:
                if word in self.word_authenticity_database:
                    word_data = self.word_authenticity_database[word]

                    # التحقق من الامتثال للقواعد
                    compliance_score = word_data.get("semantic_rule_compliance", 0.5)

                    semantic_validation["word_validations"][word] = {
                        "complies_with_rules": compliance_score > 0.7,
                        "compliance_score": compliance_score,
                        "letter_analysis": word_data.get("letter_analysis", {}),
                        "basil_validation": word_data.get("basil_validation", False)
                    }

                    semantic_validation["rule_compliance_scores"][word] = compliance_score

                    # مؤشرات الأصالة
                    if compliance_score > 0.8:
                        semantic_validation["authentic_indicators"][word] = [
                            "امتثال عالي للقواعد الدلالية",
                            "تحليل حرفي متسق",
                            "مصادقة من منهجية باسل"
                        ]
                    else:
                        semantic_validation["expansive_indicators"][word] = [
                            "امتثال منخفض للقواعد الدلالية",
                            "احتمالية كونها كلمة توسعية",
                            "تحتاج لتحليل تاريخي إضافي"
                        ]
                else:
                    # كلمة غير موجودة في قاعدة البيانات
                    semantic_validation["word_validations"][word] = {
                        "complies_with_rules": False,
                        "compliance_score": 0.0,
                        "letter_analysis": {},
                        "basil_validation": False
                    }

                    semantic_validation["expansive_indicators"][word] = [
                        "كلمة غير موجودة في قاعدة البيانات الأصيلة",
                        "تحتاج لتحليل شامل لتحديد أصالتها"
                    ]

        return semantic_validation
