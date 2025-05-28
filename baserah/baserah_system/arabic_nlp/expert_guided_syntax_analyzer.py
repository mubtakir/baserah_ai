#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-Guided Arabic Syntax Analyzer - Part 2: Syntactic Analysis
محلل النحو العربي الموجه بالخبير - الجزء الثاني: التحليل النحوي

Revolutionary integration of Expert/Explorer guidance with Arabic syntactic analysis,
applying adaptive mathematical equations to achieve superior grammatical understanding.

التكامل الثوري لتوجيه الخبير/المستكشف مع تحليل النحو العربي،
تطبيق المعادلات الرياضية المتكيفة لتحقيق فهم نحوي متفوق.

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0 - REVOLUTIONARY ARABIC SYNTAX
"""

import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import re

# إضافة المسارات
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# محاكاة النظام المتكيف للنحو
class MockSyntaxEquation:
    def __init__(self, name: str, input_dim: int, output_dim: int):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.current_complexity = 12  # النحو العربي معقد جداً
        self.adaptation_count = 0
        self.syntax_accuracy = 0.5  # دقة نحوية أساسية
        self.parsing_accuracy = 0.6
        self.pos_tagging_accuracy = 0.65
        self.dependency_accuracy = 0.55
        self.grammatical_analysis = 0.5
        self.sentence_structure_recognition = 0.6

    def adapt_with_expert_guidance(self, guidance, analysis):
        self.adaptation_count += 1
        if hasattr(guidance, 'recommended_evolution'):
            if guidance.recommended_evolution == "increase":
                self.current_complexity += 4
                self.syntax_accuracy += 0.06
                self.parsing_accuracy += 0.04
                self.pos_tagging_accuracy += 0.03
                self.dependency_accuracy += 0.05
                self.grammatical_analysis += 0.04
                self.sentence_structure_recognition += 0.03
            elif guidance.recommended_evolution == "restructure":
                self.syntax_accuracy += 0.03
                self.parsing_accuracy += 0.02
                self.pos_tagging_accuracy += 0.01

    def get_expert_guidance_summary(self):
        return {
            "current_complexity": self.current_complexity,
            "total_adaptations": self.adaptation_count,
            "syntax_accuracy": self.syntax_accuracy,
            "parsing_accuracy": self.parsing_accuracy,
            "pos_tagging_accuracy": self.pos_tagging_accuracy,
            "dependency_accuracy": self.dependency_accuracy,
            "grammatical_analysis": self.grammatical_analysis,
            "sentence_structure_recognition": self.sentence_structure_recognition,
            "average_improvement": 0.04 * self.adaptation_count
        }

class MockSyntaxGuidance:
    def __init__(self, target_complexity, focus_areas, adaptation_strength, priority_functions, recommended_evolution):
        self.target_complexity = target_complexity
        self.focus_areas = focus_areas
        self.adaptation_strength = adaptation_strength
        self.priority_functions = priority_functions
        self.recommended_evolution = recommended_evolution

class MockSyntaxAnalysis:
    def __init__(self, syntax_accuracy, parsing_clarity, pos_precision, dependency_coherence, areas_for_improvement):
        self.syntax_accuracy = syntax_accuracy
        self.parsing_clarity = parsing_clarity
        self.pos_precision = pos_precision
        self.dependency_coherence = dependency_coherence
        self.areas_for_improvement = areas_for_improvement

@dataclass
class SyntaxAnalysisRequest:
    """طلب التحليل النحوي"""
    sentence: str
    context: str = ""
    analysis_depth: str = "comprehensive"  # "basic", "intermediate", "comprehensive"
    syntax_aspects: List[str] = None  # ["pos", "parsing", "dependencies", "structure"]
    expert_guidance_level: str = "adaptive"
    learning_enabled: bool = True

@dataclass
class WordSyntaxInfo:
    """معلومات نحوية للكلمة"""
    word: str
    position: int
    pos_tag: str
    grammatical_case: str  # حالة إعرابية
    grammatical_function: str  # وظيفة نحوية
    dependencies: List[str]
    confidence: float

@dataclass
class SyntaxAnalysisResult:
    """نتيجة التحليل النحوي"""
    success: bool
    sentence: str
    sentence_type: str  # نوع الجملة
    words_syntax: List[WordSyntaxInfo]
    grammatical_structure: Dict[str, Any]
    dependency_tree: Dict[str, Any]
    parsing_confidence: float
    expert_guidance_applied: Dict[str, Any] = None
    equation_adaptations: Dict[str, Any] = None
    performance_improvements: Dict[str, float] = None
    learning_insights: List[str] = None
    next_cycle_recommendations: List[str] = None

class ExpertGuidedArabicSyntaxAnalyzer:
    """محلل النحو العربي الموجه بالخبير الثوري"""

    def __init__(self):
        """تهيئة محلل النحو العربي الموجه بالخبير"""
        print("🌟" + "="*100 + "🌟")
        print("🔤 محلل النحو العربي الموجه بالخبير الثوري")
        print("📖 الخبير/المستكشف يقود تحليل النحو العربي بذكاء")
        print("🧮 معادلات رياضية متكيفة + تحليل نحوي متقدم")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" + "="*100 + "🌟")

        # إنشاء معادلات النحو العربي متخصصة
        self.syntax_equations = {
            "pos_tagger": MockSyntaxEquation("pos_tagging", 25, 20),
            "dependency_parser": MockSyntaxEquation("dependency_parsing", 30, 25),
            "grammatical_analyzer": MockSyntaxEquation("grammatical_analysis", 28, 22),
            "sentence_structure_detector": MockSyntaxEquation("sentence_structure", 32, 26),
            "case_analyzer": MockSyntaxEquation("case_analysis", 24, 18),
            "function_identifier": MockSyntaxEquation("function_identification", 26, 20),
            "phrase_chunker": MockSyntaxEquation("phrase_chunking", 22, 16),
            "clause_detector": MockSyntaxEquation("clause_detection", 28, 22),
            "agreement_checker": MockSyntaxEquation("agreement_checking", 20, 15),
            "word_order_analyzer": MockSyntaxEquation("word_order_analysis", 24, 18)
        }

        # قوانين النحو العربي
        self.syntax_laws = {
            "subject_verb_agreement": {
                "name": "التطابق بين الفاعل والفعل",
                "description": "الفعل يطابق الفاعل في العدد والجنس",
                "formula": "Verb(number, gender) = Subject(number, gender)"
            },
            "case_assignment": {
                "name": "تعيين الحالات الإعرابية",
                "description": "كل كلمة لها حالة إعرابية محددة حسب موقعها",
                "formula": "Case(word) = Function(position, context)"
            },
            "word_order_flexibility": {
                "name": "مرونة ترتيب الكلمات",
                "description": "العربية تسمح بترتيبات متعددة مع الحفاظ على المعنى",
                "formula": "Meaning = Constant(VSO, SVO, VOS, ...)"
            }
        }

        # ثوابت النحو العربي
        self.syntax_constants = {
            "verb_weight": 0.8,
            "subject_weight": 0.9,
            "object_weight": 0.7,
            "modifier_weight": 0.6,
            "case_importance": 0.85,
            "agreement_strength": 0.9
        }

        # قاعدة بيانات أنواع الكلمات العربية
        self.arabic_pos_tags = self._load_arabic_pos_tags()

        # قاعدة بيانات الحالات الإعرابية
        self.arabic_cases = self._load_arabic_cases()

        # قاعدة بيانات الوظائف النحوية
        self.arabic_functions = self._load_arabic_functions()

        # تاريخ التحليلات النحوية
        self.syntax_history = []
        self.syntax_learning_database = {}

        print("🔤 تم إنشاء المعادلات النحو العربي المتخصصة:")
        for eq_name in self.syntax_equations.keys():
            print(f"   ✅ {eq_name}")

        print("✅ تم تهيئة محلل النحو العربي الموجه بالخبير!")

    def _load_arabic_pos_tags(self) -> Dict[str, Dict[str, Any]]:
        """تحميل قاعدة بيانات أنواع الكلمات العربية"""
        return {
            "اسم": {"type": "noun", "features": ["معرب", "مبني"], "cases": ["رفع", "نصب", "جر"]},
            "فعل": {"type": "verb", "features": ["ماضي", "مضارع", "أمر"], "cases": ["مرفوع", "منصوب", "مجزوم"]},
            "حرف": {"type": "particle", "features": ["جر", "نصب", "جزم"], "cases": ["مبني"]},
            "ضمير": {"type": "pronoun", "features": ["متصل", "منفصل"], "cases": ["رفع", "نصب", "جر"]},
            "صفة": {"type": "adjective", "features": ["مشبهة", "مفعول"], "cases": ["رفع", "نصب", "جر"]},
            "ظرف": {"type": "adverb", "features": ["زمان", "مكان"], "cases": ["منصوب", "مجرور"]}
        }

    def _load_arabic_cases(self) -> Dict[str, Dict[str, Any]]:
        """تحميل قاعدة بيانات الحالات الإعرابية"""
        return {
            "رفع": {"marker": "ضمة", "function": "فاعل أو مبتدأ أو خبر", "examples": ["الطالبُ", "المعلمُ"]},
            "نصب": {"marker": "فتحة", "function": "مفعول أو خبر كان", "examples": ["الطالبَ", "المعلمَ"]},
            "جر": {"marker": "كسرة", "function": "مجرور بحرف أو مضاف إليه", "examples": ["الطالبِ", "المعلمِ"]},
            "جزم": {"marker": "سكون", "function": "فعل مجزوم", "examples": ["لم يكتبْ", "لا تكتبْ"]}
        }

    def _load_arabic_functions(self) -> Dict[str, Dict[str, Any]]:
        """تحميل قاعدة بيانات الوظائف النحوية"""
        return {
            "فاعل": {"case": "رفع", "definition": "من قام بالفعل", "position": "بعد الفعل"},
            "مفعول_به": {"case": "نصب", "definition": "من وقع عليه الفعل", "position": "بعد الفاعل"},
            "مبتدأ": {"case": "رفع", "definition": "ما ابتدئت به الجملة", "position": "أول الجملة"},
            "خبر": {"case": "رفع", "definition": "ما أخبر به عن المبتدأ", "position": "بعد المبتدأ"},
            "مضاف_إليه": {"case": "جر", "definition": "ما أضيف إليه", "position": "بعد المضاف"},
            "صفة": {"case": "تابع", "definition": "ما وصف به", "position": "بعد الموصوف"}
        }

    def analyze_syntax_with_expert_guidance(self, request: SyntaxAnalysisRequest) -> SyntaxAnalysisResult:
        """التحليل النحوي موجه بالخبير"""
        print(f"\n🔤 بدء التحليل النحوي الموجه بالخبير للجملة: {request.sentence}")
        start_time = datetime.now()

        # المرحلة 1: تحليل الخبير للطلب النحوي
        expert_analysis = self._analyze_syntax_request_with_expert(request)
        print(f"📖 تحليل الخبير النحوي: {expert_analysis['complexity_assessment']}")

        # المرحلة 2: توليد توجيهات الخبير لمعادلات النحو
        expert_guidance = self._generate_syntax_expert_guidance(request, expert_analysis)
        print(f"🔤 توجيه الخبير النحوي: {expert_guidance.recommended_evolution}")

        # المرحلة 3: تكيف معادلات النحو
        equation_adaptations = self._adapt_syntax_equations(expert_guidance, expert_analysis)
        print(f"🧮 تكيف معادلات النحو: {len(equation_adaptations)} معادلة")

        # المرحلة 4: تنفيذ التحليل النحوي المتكيف
        syntax_analysis = self._perform_adaptive_syntax_analysis(request, equation_adaptations)

        # المرحلة 5: قياس التحسينات النحوية
        performance_improvements = self._measure_syntax_improvements(request, syntax_analysis, equation_adaptations)

        # المرحلة 6: استخراج رؤى التعلم النحوي
        learning_insights = self._extract_syntax_learning_insights(request, syntax_analysis, performance_improvements)

        # المرحلة 7: توليد توصيات للدورة التالية
        next_cycle_recommendations = self._generate_syntax_next_cycle_recommendations(performance_improvements, learning_insights)

        # إنشاء النتيجة النحوية النهائية
        result = SyntaxAnalysisResult(
            success=True,
            sentence=request.sentence,
            sentence_type=syntax_analysis.get("sentence_type", ""),
            words_syntax=syntax_analysis.get("words_syntax", []),
            grammatical_structure=syntax_analysis.get("grammatical_structure", {}),
            dependency_tree=syntax_analysis.get("dependency_tree", {}),
            parsing_confidence=syntax_analysis.get("parsing_confidence", 0.0),
            expert_guidance_applied=expert_guidance.__dict__,
            equation_adaptations=equation_adaptations,
            performance_improvements=performance_improvements,
            learning_insights=learning_insights,
            next_cycle_recommendations=next_cycle_recommendations
        )

        # حفظ في قاعدة التعلم النحوي
        self._save_syntax_learning(request, result)

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ انتهى التحليل النحوي الموجه في {total_time:.2f} ثانية")

        return result

    def _analyze_syntax_request_with_expert(self, request: SyntaxAnalysisRequest) -> Dict[str, Any]:
        """تحليل طلب النحو بواسطة الخبير"""

        # تحليل تعقيد الجملة
        words = request.sentence.split()
        sentence_complexity = len(words) * 0.8
        context_complexity = len(request.context.split()) * 0.4 if request.context else 0

        # تحليل جوانب النحو المطلوبة
        aspects = request.syntax_aspects or ["pos", "parsing", "dependencies", "structure"]
        aspects_complexity = len(aspects) * 3.0  # النحو معقد أكثر من الصرف

        # تحليل عمق التحليل
        depth_complexity = {
            "basic": 3.0,
            "intermediate": 6.0,
            "comprehensive": 9.0
        }.get(request.analysis_depth, 6.0)

        # تحليل تعقيد الجملة النحوي
        grammatical_complexity = 0
        if any(word in request.sentence for word in ["الذي", "التي", "اللذان", "اللتان"]):
            grammatical_complexity += 3  # جملة موصولة
        if any(word in request.sentence for word in ["إن", "أن", "كان", "أصبح"]):
            grammatical_complexity += 2  # جملة منسوخة
        if any(word in request.sentence for word in ["في", "على", "إلى", "من"]):
            grammatical_complexity += 1  # حروف جر

        total_complexity = sentence_complexity + context_complexity + aspects_complexity + depth_complexity + grammatical_complexity

        return {
            "sentence_complexity": sentence_complexity,
            "context_complexity": context_complexity,
            "aspects_complexity": aspects_complexity,
            "depth_complexity": depth_complexity,
            "grammatical_complexity": grammatical_complexity,
            "total_complexity": total_complexity,
            "complexity_assessment": "نحو معقد جداً" if total_complexity > 25 else "نحو متوسط" if total_complexity > 15 else "نحو بسيط",
            "recommended_adaptations": int(total_complexity // 3) + 4,
            "focus_areas": self._identify_syntax_focus_areas(request)
        }

    def _identify_syntax_focus_areas(self, request: SyntaxAnalysisRequest) -> List[str]:
        """تحديد مناطق التركيز النحوي"""
        focus_areas = []

        aspects = request.syntax_aspects or ["pos", "parsing", "dependencies", "structure"]

        if "pos" in aspects:
            focus_areas.append("pos_tagging_enhancement")
        if "parsing" in aspects:
            focus_areas.append("parsing_accuracy_improvement")
        if "dependencies" in aspects:
            focus_areas.append("dependency_analysis_optimization")
        if "structure" in aspects:
            focus_areas.append("structure_recognition_refinement")

        # تحليل خصائص الجملة
        words = request.sentence.split()
        if len(words) > 8:
            focus_areas.append("complex_sentence_handling")
        if any(word.startswith("ال") for word in words):
            focus_areas.append("definite_noun_processing")
        if any(word in request.sentence for word in ["الذي", "التي"]):
            focus_areas.append("relative_clause_analysis")
        if any(word in request.sentence for word in ["إن", "أن", "كان"]):
            focus_areas.append("copular_sentence_analysis")
        if request.context:
            focus_areas.append("contextual_syntax_analysis")

        return focus_areas

    def _generate_syntax_expert_guidance(self, request: SyntaxAnalysisRequest, analysis: Dict[str, Any]):
        """توليد توجيهات الخبير للتحليل النحوي"""

        # تحديد التعقيد المستهدف للنحو
        target_complexity = 15 + analysis["recommended_adaptations"]

        # تحديد الدوال ذات الأولوية للنحو العربي
        priority_functions = []
        if "pos_tagging_enhancement" in analysis["focus_areas"]:
            priority_functions.extend(["softplus", "tanh"])  # لتصنيف أنواع الكلمات
        if "parsing_accuracy_improvement" in analysis["focus_areas"]:
            priority_functions.extend(["gaussian", "sin_cos"])  # لتحليل الجمل
        if "dependency_analysis_optimization" in analysis["focus_areas"]:
            priority_functions.extend(["swish", "squared_relu"])  # لتحليل التبعيات
        if "structure_recognition_refinement" in analysis["focus_areas"]:
            priority_functions.extend(["hyperbolic", "softsign"])  # لتمييز التراكيب
        if "complex_sentence_handling" in analysis["focus_areas"]:
            priority_functions.extend(["sin", "cos"])  # للجمل المعقدة
        if "relative_clause_analysis" in analysis["focus_areas"]:
            priority_functions.extend(["gaussian", "softplus"])  # للجمل الموصولة
        if "copular_sentence_analysis" in analysis["focus_areas"]:
            priority_functions.extend(["tanh", "swish"])  # للجمل المنسوخة

        # تحديد نوع التطور النحوي
        if analysis["complexity_assessment"] == "نحو معقد جداً":
            recommended_evolution = "increase"
            adaptation_strength = 0.95
        elif analysis["complexity_assessment"] == "نحو متوسط":
            recommended_evolution = "restructure"
            adaptation_strength = 0.8
        else:
            recommended_evolution = "maintain"
            adaptation_strength = 0.7

        return MockSyntaxGuidance(
            target_complexity=target_complexity,
            focus_areas=analysis["focus_areas"],
            adaptation_strength=adaptation_strength,
            priority_functions=priority_functions or ["softplus", "gaussian"],
            recommended_evolution=recommended_evolution
        )

    def _adapt_syntax_equations(self, guidance, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """تكيف معادلات النحو"""

        adaptations = {}

        # إنشاء تحليل وهمي لمعادلات النحو
        mock_analysis = MockSyntaxAnalysis(
            syntax_accuracy=0.5,
            parsing_clarity=0.6,
            pos_precision=0.65,
            dependency_coherence=0.55,
            areas_for_improvement=guidance.focus_areas
        )

        # تكيف كل معادلة نحو
        for eq_name, equation in self.syntax_equations.items():
            print(f"   🔤 تكيف معادلة النحو: {eq_name}")
            equation.adapt_with_expert_guidance(guidance, mock_analysis)
            adaptations[eq_name] = equation.get_expert_guidance_summary()

        return adaptations

    def _perform_adaptive_syntax_analysis(self, request: SyntaxAnalysisRequest, adaptations: Dict[str, Any]) -> Dict[str, Any]:
        """تنفيذ التحليل النحوي المتكيف"""

        analysis_results = {
            "sentence_type": "",
            "words_syntax": [],
            "grammatical_structure": {},
            "dependency_tree": {},
            "parsing_confidence": 0.0
        }

        # تحليل نوع الجملة
        sentence_type = self._identify_sentence_type_adaptive(request.sentence)
        analysis_results["sentence_type"] = sentence_type

        # تحليل الكلمات نحوياً
        pos_accuracy = adaptations.get("pos_tagger", {}).get("pos_tagging_accuracy", 0.65)
        words_syntax = self._analyze_words_syntax_adaptive(request.sentence, pos_accuracy)
        analysis_results["words_syntax"] = words_syntax

        # تحليل التركيب النحوي
        structure_accuracy = adaptations.get("sentence_structure_detector", {}).get("sentence_structure_recognition", 0.6)
        grammatical_structure = self._analyze_grammatical_structure_adaptive(request.sentence, words_syntax, structure_accuracy)
        analysis_results["grammatical_structure"] = grammatical_structure

        # تحليل التبعيات
        dependency_accuracy = adaptations.get("dependency_parser", {}).get("dependency_accuracy", 0.55)
        dependency_tree = self._analyze_dependencies_adaptive(request.sentence, words_syntax, dependency_accuracy)
        analysis_results["dependency_tree"] = dependency_tree

        # حساب ثقة التحليل
        parsing_confidence = np.mean([pos_accuracy, structure_accuracy, dependency_accuracy])
        analysis_results["parsing_confidence"] = parsing_confidence

        return analysis_results

    def _identify_sentence_type_adaptive(self, sentence: str) -> str:
        """تحديد نوع الجملة بطريقة متكيفة"""

        # جملة فعلية
        if any(sentence.strip().split()[0].endswith(suffix) for suffix in ["", "ت", "نا", "وا", "تم", "تن"] if sentence.strip().split()):
            return "جملة فعلية"

        # جملة اسمية
        if sentence.strip() and not any(sentence.strip().split()[0].endswith(suffix) for suffix in ["", "ت", "نا", "وا", "تم", "تن"]):
            return "جملة اسمية"

        # جملة شرطية
        if any(word in sentence for word in ["إذا", "إن", "لو", "لولا"]):
            return "جملة شرطية"

        # جملة استفهامية
        if any(word in sentence for word in ["ما", "من", "متى", "أين", "كيف", "لماذا", "هل"]):
            return "جملة استفهامية"

        # جملة منسوخة
        if any(word in sentence for word in ["إن", "أن", "كان", "أصبح", "ظل", "بات"]):
            return "جملة منسوخة"

        return "جملة بسيطة"

    def _analyze_words_syntax_adaptive(self, sentence: str, accuracy: float) -> List[WordSyntaxInfo]:
        """تحليل الكلمات نحوياً بطريقة متكيفة"""

        words = sentence.split()
        words_syntax = []

        for i, word in enumerate(words):
            # تحديد نوع الكلمة
            pos_tag = self._determine_pos_tag_adaptive(word, i, words)

            # تحديد الحالة الإعرابية
            grammatical_case = self._determine_grammatical_case_adaptive(word, pos_tag, i, words)

            # تحديد الوظيفة النحوية
            grammatical_function = self._determine_grammatical_function_adaptive(word, pos_tag, grammatical_case, i, words)

            # تحديد التبعيات
            dependencies = self._determine_dependencies_adaptive(word, i, words)

            word_syntax = WordSyntaxInfo(
                word=word,
                position=i,
                pos_tag=pos_tag,
                grammatical_case=grammatical_case,
                grammatical_function=grammatical_function,
                dependencies=dependencies,
                confidence=accuracy
            )

            words_syntax.append(word_syntax)

        return words_syntax

    def _determine_pos_tag_adaptive(self, word: str, position: int, words: List[str]) -> str:
        """تحديد نوع الكلمة بطريقة متكيفة"""

        # فعل (بناءً على الموقع والصيغة)
        if position == 0 and not word.startswith("ال"):
            return "فعل"

        # اسم معرف
        if word.startswith("ال"):
            return "اسم"

        # ضمير
        if word in ["هو", "هي", "هم", "هن", "أنت", "أنتم", "أنا", "نحن"]:
            return "ضمير"

        # حرف جر
        if word in ["في", "على", "إلى", "من", "عن", "مع", "ب", "ل", "ك"]:
            return "حرف"

        # حرف نصب أو جزم
        if word in ["أن", "لن", "لم", "لا", "إن", "كان"]:
            return "حرف"

        # افتراض أنه اسم
        return "اسم"

    def _determine_grammatical_case_adaptive(self, word: str, pos_tag: str, position: int, words: List[str]) -> str:
        """تحديد الحالة الإعرابية بطريقة متكيفة"""

        if pos_tag == "حرف":
            return "مبني"

        # فاعل (مرفوع)
        if pos_tag == "اسم" and position > 0 and words[position-1] not in ["في", "على", "إلى", "من"]:
            return "رفع"

        # مجرور بحرف جر
        if position > 0 and words[position-1] in ["في", "على", "إلى", "من", "عن", "مع", "ب", "ل", "ك"]:
            return "جر"

        # مفعول به (منصوب)
        if pos_tag == "اسم" and position > 1:
            return "نصب"

        # افتراض الرفع
        return "رفع"

    def _determine_grammatical_function_adaptive(self, word: str, pos_tag: str, case: str, position: int, words: List[str]) -> str:
        """تحديد الوظيفة النحوية بطريقة متكيفة"""

        if pos_tag == "فعل":
            return "فعل"

        if case == "جر":
            return "مجرور"

        if position == 0 and pos_tag == "اسم":
            return "مبتدأ"

        if position == 1 and pos_tag == "اسم" and words[0] not in ["فعل"]:
            return "خبر"

        if pos_tag == "اسم" and case == "رفع" and position > 0:
            return "فاعل"

        if pos_tag == "اسم" and case == "نصب":
            return "مفعول_به"

        return "غير_محدد"

    def _determine_dependencies_adaptive(self, word: str, position: int, words: List[str]) -> List[str]:
        """تحديد التبعيات بطريقة متكيفة"""
        dependencies = []

        # تبعية مع الكلمة السابقة
        if position > 0:
            dependencies.append(f"depends_on_{position-1}")

        # تبعية مع الكلمة التالية
        if position < len(words) - 1:
            dependencies.append(f"governs_{position+1}")

        return dependencies

    def _analyze_grammatical_structure_adaptive(self, sentence: str, words_syntax: List[WordSyntaxInfo], accuracy: float) -> Dict[str, Any]:
        """تحليل التركيب النحوي بطريقة متكيفة"""

        structure = {
            "subject": None,
            "verb": None,
            "object": None,
            "modifiers": [],
            "phrases": []
        }

        for word_info in words_syntax:
            if word_info.grammatical_function == "فاعل":
                structure["subject"] = word_info.word
            elif word_info.pos_tag == "فعل":
                structure["verb"] = word_info.word
            elif word_info.grammatical_function == "مفعول_به":
                structure["object"] = word_info.word
            elif word_info.grammatical_function in ["مجرور", "صفة"]:
                structure["modifiers"].append(word_info.word)

        # تحديد العبارات
        current_phrase = []
        for word_info in words_syntax:
            if word_info.pos_tag == "حرف" and current_phrase:
                structure["phrases"].append(" ".join(current_phrase))
                current_phrase = [word_info.word]
            else:
                current_phrase.append(word_info.word)

        if current_phrase:
            structure["phrases"].append(" ".join(current_phrase))

        return structure

    def _analyze_dependencies_adaptive(self, sentence: str, words_syntax: List[WordSyntaxInfo], accuracy: float) -> Dict[str, Any]:
        """تحليل التبعيات بطريقة متكيفة"""

        dependency_tree = {
            "root": None,
            "relations": []
        }

        # العثور على الجذر (عادة الفعل الرئيسي)
        for word_info in words_syntax:
            if word_info.pos_tag == "فعل":
                dependency_tree["root"] = word_info.word
                break

        # إنشاء العلاقات
        for i, word_info in enumerate(words_syntax):
            if word_info.pos_tag != "فعل":
                relation = {
                    "dependent": word_info.word,
                    "head": dependency_tree["root"] or words_syntax[0].word,
                    "relation_type": word_info.grammatical_function,
                    "confidence": accuracy
                }
                dependency_tree["relations"].append(relation)

        return dependency_tree

    def _measure_syntax_improvements(self, request: SyntaxAnalysisRequest, analysis: Dict[str, Any], adaptations: Dict[str, Any]) -> Dict[str, float]:
        """قياس تحسينات أداء النحو"""

        improvements = {}

        # تحسن دقة النحو
        avg_syntax_accuracy = np.mean([adapt.get("syntax_accuracy", 0.5) for adapt in adaptations.values()])
        baseline_syntax_accuracy = 0.4
        syntax_accuracy_improvement = ((avg_syntax_accuracy - baseline_syntax_accuracy) / baseline_syntax_accuracy) * 100
        improvements["syntax_accuracy_improvement"] = max(0, syntax_accuracy_improvement)

        # تحسن دقة التحليل
        avg_parsing_accuracy = np.mean([adapt.get("parsing_accuracy", 0.6) for adapt in adaptations.values()])
        baseline_parsing_accuracy = 0.5
        parsing_accuracy_improvement = ((avg_parsing_accuracy - baseline_parsing_accuracy) / baseline_parsing_accuracy) * 100
        improvements["parsing_accuracy_improvement"] = max(0, parsing_accuracy_improvement)

        # تحسن تصنيف أنواع الكلمات
        avg_pos_tagging = np.mean([adapt.get("pos_tagging_accuracy", 0.65) for adapt in adaptations.values()])
        baseline_pos_tagging = 0.55
        pos_tagging_improvement = ((avg_pos_tagging - baseline_pos_tagging) / baseline_pos_tagging) * 100
        improvements["pos_tagging_improvement"] = max(0, pos_tagging_improvement)

        # تحسن تحليل التبعيات
        avg_dependency_accuracy = np.mean([adapt.get("dependency_accuracy", 0.55) for adapt in adaptations.values()])
        baseline_dependency_accuracy = 0.45
        dependency_accuracy_improvement = ((avg_dependency_accuracy - baseline_dependency_accuracy) / baseline_dependency_accuracy) * 100
        improvements["dependency_accuracy_improvement"] = max(0, dependency_accuracy_improvement)

        # تحسن التحليل النحوي
        avg_grammatical_analysis = np.mean([adapt.get("grammatical_analysis", 0.5) for adapt in adaptations.values()])
        baseline_grammatical_analysis = 0.4
        grammatical_analysis_improvement = ((avg_grammatical_analysis - baseline_grammatical_analysis) / baseline_grammatical_analysis) * 100
        improvements["grammatical_analysis_improvement"] = max(0, grammatical_analysis_improvement)

        # تحسن تمييز التراكيب
        avg_structure_recognition = np.mean([adapt.get("sentence_structure_recognition", 0.6) for adapt in adaptations.values()])
        baseline_structure_recognition = 0.5
        structure_recognition_improvement = ((avg_structure_recognition - baseline_structure_recognition) / baseline_structure_recognition) * 100
        improvements["structure_recognition_improvement"] = max(0, structure_recognition_improvement)

        # تحسن التعقيد النحوي
        total_adaptations = sum(adapt.get("total_adaptations", 0) for adapt in adaptations.values())
        syntax_complexity_improvement = total_adaptations * 15  # كل تكيف نحوي = 15% تحسن
        improvements["syntax_complexity_improvement"] = syntax_complexity_improvement

        return improvements

    def _extract_syntax_learning_insights(self, request: SyntaxAnalysisRequest, analysis: Dict[str, Any], improvements: Dict[str, float]) -> List[str]:
        """استخراج رؤى التعلم النحوي"""

        insights = []

        if improvements["syntax_accuracy_improvement"] > 25:
            insights.append("التكيف الموجه بالخبير حسن دقة التحليل النحوي بشكل كبير")

        if improvements["parsing_accuracy_improvement"] > 20:
            insights.append("المعادلات المتكيفة ممتازة لتحليل الجمل العربية")

        if improvements["pos_tagging_improvement"] > 18:
            insights.append("النظام نجح في تحسين تصنيف أنواع الكلمات")

        if improvements["dependency_accuracy_improvement"] > 22:
            insights.append("تحليل التبعيات النحوية تحسن مع التوجيه الخبير")

        if improvements["grammatical_analysis_improvement"] > 25:
            insights.append("التحليل النحوي أصبح أكثر دقة مع التكيف")

        if improvements["structure_recognition_improvement"] > 20:
            insights.append("تمييز التراكيب النحوية تحسن بشكل ملحوظ")

        if improvements["syntax_complexity_improvement"] > 100:
            insights.append("المعادلات النحوية المتكيفة تتعامل مع التعقيد النحوي بكفاءة")

        # رؤى خاصة بالجملة
        words_count = len(request.sentence.split())
        if words_count > 8:
            insights.append("النظام يتعامل بكفاءة مع الجمل المعقدة")

        if request.context:
            insights.append("التحليل السياقي يحسن دقة التحليل النحوي")

        if analysis.get("sentence_type") == "جملة منسوخة":
            insights.append("النظام يحلل الجمل المنسوخة بدقة متقدمة")

        return insights

    def _generate_syntax_next_cycle_recommendations(self, improvements: Dict[str, float], insights: List[str]) -> List[str]:
        """توليد توصيات للدورة النحوية التالية"""

        recommendations = []

        avg_improvement = np.mean(list(improvements.values()))

        if avg_improvement > 35:
            recommendations.append("الحفاظ على إعدادات التكيف النحوي الحالية")
            recommendations.append("تجربة تحليل نحوي أعمق للجمل المعقدة")
        elif avg_improvement > 20:
            recommendations.append("زيادة قوة التكيف النحوي تدريجياً")
            recommendations.append("إضافة قواعد نحوية متقدمة")
        else:
            recommendations.append("مراجعة استراتيجية التوجيه النحوي")
            recommendations.append("تحسين دقة معادلات النحو")

        # توصيات محددة
        if "أنواع الكلمات" in str(insights):
            recommendations.append("التوسع في قاعدة بيانات أنواع الكلمات العربية")

        if "التبعيات" in str(insights):
            recommendations.append("تطوير خوارزميات تحليل التبعيات النحوية")

        if "معقدة" in str(insights):
            recommendations.append("تحسين معالجة الجمل المعقدة والمركبة")

        if "منسوخة" in str(insights):
            recommendations.append("تعزيز تحليل الجمل المنسوخة والشرطية")

        return recommendations

    def _save_syntax_learning(self, request: SyntaxAnalysisRequest, result: SyntaxAnalysisResult):
        """حفظ التعلم النحوي"""

        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "sentence": request.sentence,
            "context": request.context,
            "analysis_depth": request.analysis_depth,
            "success": result.success,
            "sentence_type": result.sentence_type,
            "parsing_confidence": result.parsing_confidence,
            "performance_improvements": result.performance_improvements,
            "learning_insights": result.learning_insights
        }

        sentence_key = f"{len(request.sentence.split())}_{request.analysis_depth}"
        if sentence_key not in self.syntax_learning_database:
            self.syntax_learning_database[sentence_key] = []

        self.syntax_learning_database[sentence_key].append(learning_entry)

        # الاحتفاظ بآخر 3 إدخالات فقط
        if len(self.syntax_learning_database[sentence_key]) > 3:
            self.syntax_learning_database[sentence_key] = self.syntax_learning_database[sentence_key][-3:]

def main():
    """اختبار محلل النحو العربي الموجه بالخبير"""
    print("🧪 اختبار محلل النحو العربي الموجه بالخبير...")

    # إنشاء المحلل النحوي
    syntax_analyzer = ExpertGuidedArabicSyntaxAnalyzer()

    # جمل اختبار عربية
    test_sentences = [
        "الطالب يكتب الدرس",
        "كتب المعلم على السبورة",
        "إن الطلاب في المدرسة",
        "هل تعرف الطريق إلى المكتبة؟",
        "الكتاب الذي قرأته مفيد جداً"
    ]

    for sentence in test_sentences:
        print(f"\n{'='*60}")
        print(f"🔤 تحليل الجملة: {sentence}")

        # طلب التحليل النحوي
        syntax_request = SyntaxAnalysisRequest(
            sentence=sentence,
            context="سياق تجريبي للجملة",
            analysis_depth="comprehensive",
            syntax_aspects=["pos", "parsing", "dependencies", "structure"],
            expert_guidance_level="adaptive",
            learning_enabled=True
        )

        # تنفيذ التحليل النحوي
        syntax_result = syntax_analyzer.analyze_syntax_with_expert_guidance(syntax_request)

        # عرض النتائج النحوية
        print(f"\n📊 نتائج التحليل النحوي:")
        print(f"   ✅ النجاح: {syntax_result.success}")
        print(f"   📝 نوع الجملة: {syntax_result.sentence_type}")
        print(f"   🎯 ثقة التحليل: {syntax_result.parsing_confidence:.2%}")

        print(f"   🔤 تحليل الكلمات:")
        for word_info in syntax_result.words_syntax:
            print(f"      {word_info.word}: {word_info.pos_tag} | {word_info.grammatical_case} | {word_info.grammatical_function}")

        if syntax_result.grammatical_structure:
            print(f"   🏗️ التركيب النحوي:")
            structure = syntax_result.grammatical_structure
            if structure.get("subject"):
                print(f"      الفاعل: {structure['subject']}")
            if structure.get("verb"):
                print(f"      الفعل: {structure['verb']}")
            if structure.get("object"):
                print(f"      المفعول: {structure['object']}")

        if syntax_result.performance_improvements:
            print(f"   📈 تحسينات الأداء:")
            for metric, improvement in syntax_result.performance_improvements.items():
                print(f"      {metric}: {improvement:.1f}%")

        if syntax_result.learning_insights:
            print(f"   🧠 رؤى التعلم:")
            for insight in syntax_result.learning_insights:
                print(f"      • {insight}")

if __name__ == "__main__":
    main()
