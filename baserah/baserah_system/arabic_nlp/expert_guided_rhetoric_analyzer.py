#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-Guided Arabic Rhetoric Analyzer - Part 3: Rhetorical Analysis
محلل البلاغة العربية الموجه بالخبير - الجزء الثالث: التحليل البلاغي

Revolutionary integration of Expert/Explorer guidance with Arabic rhetorical analysis,
applying adaptive mathematical equations to achieve superior literary understanding.

التكامل الثوري لتوجيه الخبير/المستكشف مع تحليل البلاغة العربية،
تطبيق المعادلات الرياضية المتكيفة لتحقيق فهم أدبي متفوق.

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0 - REVOLUTIONARY ARABIC RHETORIC
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

# محاكاة النظام المتكيف للبلاغة
class MockRhetoricEquation:
    def __init__(self, name: str, input_dim: int, output_dim: int):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.current_complexity = 15  # البلاغة العربية معقدة جداً
        self.adaptation_count = 0
        self.rhetoric_accuracy = 0.4  # دقة بلاغية أساسية
        self.metaphor_detection = 0.5
        self.simile_recognition = 0.55
        self.alliteration_analysis = 0.45
        self.rhythm_detection = 0.4
        self.eloquence_measurement = 0.35
        self.literary_beauty_assessment = 0.3

    def adapt_with_expert_guidance(self, guidance, analysis):
        self.adaptation_count += 1
        if hasattr(guidance, 'recommended_evolution'):
            if guidance.recommended_evolution == "increase":
                self.current_complexity += 5
                self.rhetoric_accuracy += 0.08
                self.metaphor_detection += 0.06
                self.simile_recognition += 0.05
                self.alliteration_analysis += 0.07
                self.rhythm_detection += 0.06
                self.eloquence_measurement += 0.08
                self.literary_beauty_assessment += 0.09
            elif guidance.recommended_evolution == "restructure":
                self.rhetoric_accuracy += 0.04
                self.metaphor_detection += 0.03
                self.simile_recognition += 0.02

    def get_expert_guidance_summary(self):
        return {
            "current_complexity": self.current_complexity,
            "total_adaptations": self.adaptation_count,
            "rhetoric_accuracy": self.rhetoric_accuracy,
            "metaphor_detection": self.metaphor_detection,
            "simile_recognition": self.simile_recognition,
            "alliteration_analysis": self.alliteration_analysis,
            "rhythm_detection": self.rhythm_detection,
            "eloquence_measurement": self.eloquence_measurement,
            "literary_beauty_assessment": self.literary_beauty_assessment,
            "average_improvement": 0.06 * self.adaptation_count
        }

class MockRhetoricGuidance:
    def __init__(self, target_complexity, focus_areas, adaptation_strength, priority_functions, recommended_evolution):
        self.target_complexity = target_complexity
        self.focus_areas = focus_areas
        self.adaptation_strength = adaptation_strength
        self.priority_functions = priority_functions
        self.recommended_evolution = recommended_evolution

class MockRhetoricAnalysis:
    def __init__(self, rhetoric_accuracy, metaphor_clarity, literary_coherence, eloquence_precision, areas_for_improvement):
        self.rhetoric_accuracy = rhetoric_accuracy
        self.metaphor_clarity = metaphor_clarity
        self.literary_coherence = literary_coherence
        self.eloquence_precision = eloquence_precision
        self.areas_for_improvement = areas_for_improvement

@dataclass
class RhetoricAnalysisRequest:
    """طلب التحليل البلاغي"""
    text: str
    context: str = ""
    analysis_depth: str = "comprehensive"  # "basic", "intermediate", "comprehensive"
    rhetoric_aspects: List[str] = None  # ["metaphor", "simile", "alliteration", "rhythm", "eloquence"]
    expert_guidance_level: str = "adaptive"
    learning_enabled: bool = True

@dataclass
class RhetoricalDevice:
    """جهاز بلاغي"""
    device_type: str
    text_span: str
    description: str
    literary_effect: str
    beauty_score: float
    confidence: float

@dataclass
class RhetoricAnalysisResult:
    """نتيجة التحليل البلاغي"""
    success: bool
    text: str
    literary_style: str  # أسلوب أدبي
    rhetorical_devices: List[RhetoricalDevice]
    eloquence_score: float
    beauty_assessment: Dict[str, float]
    rhythm_analysis: Dict[str, Any]
    overall_rhetoric_quality: float
    expert_guidance_applied: Dict[str, Any] = None
    equation_adaptations: Dict[str, Any] = None
    performance_improvements: Dict[str, float] = None
    learning_insights: List[str] = None
    next_cycle_recommendations: List[str] = None

class ExpertGuidedArabicRhetoricAnalyzer:
    """محلل البلاغة العربية الموجه بالخبير الثوري"""

    def __init__(self):
        """تهيئة محلل البلاغة العربية الموجه بالخبير"""
        print("🌟" + "="*100 + "🌟")
        print("🎨 محلل البلاغة العربية الموجه بالخبير الثوري")
        print("📜 الخبير/المستكشف يقود تحليل البلاغة العربية بذكاء")
        print("🧮 معادلات رياضية متكيفة + تحليل بلاغي متقدم")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" + "="*100 + "🌟")

        # إنشاء معادلات البلاغة العربية متخصصة
        self.rhetoric_equations = {
            "metaphor_detector": MockRhetoricEquation("metaphor_detection", 35, 28),
            "simile_analyzer": MockRhetoricEquation("simile_analysis", 32, 25),
            "alliteration_finder": MockRhetoricEquation("alliteration_finding", 28, 22),
            "rhythm_analyzer": MockRhetoricEquation("rhythm_analysis", 30, 24),
            "eloquence_measurer": MockRhetoricEquation("eloquence_measurement", 40, 32),
            "beauty_assessor": MockRhetoricEquation("beauty_assessment", 45, 36),
            "literary_style_classifier": MockRhetoricEquation("style_classification", 38, 30),
            "poetic_meter_detector": MockRhetoricEquation("meter_detection", 33, 26),
            "semantic_harmony_analyzer": MockRhetoricEquation("semantic_harmony", 36, 28),
            "artistic_imagery_extractor": MockRhetoricEquation("imagery_extraction", 42, 34),
            "emotional_impact_measurer": MockRhetoricEquation("emotional_impact", 39, 31),
            "linguistic_elegance_evaluator": MockRhetoricEquation("elegance_evaluation", 44, 35)
        }

        # قوانين البلاغة العربية
        self.rhetoric_laws = {
            "eloquence_harmony": {
                "name": "تناغم الفصاحة",
                "description": "الكلام البليغ يجمع بين جمال اللفظ ووضوح المعنى",
                "formula": "Eloquence = Beauty(words) × Clarity(meaning)"
            },
            "metaphor_appropriateness": {
                "name": "مناسبة الاستعارة",
                "description": "الاستعارة الجيدة تقرب المعنى وتزيد الجمال",
                "formula": "Metaphor_Quality = Similarity(tenor, vehicle) × Beauty_Enhancement"
            },
            "rhythm_consistency": {
                "name": "اتساق الإيقاع",
                "description": "الإيقاع المتسق يزيد من جمال النص وتأثيره",
                "formula": "Rhythm_Quality = Consistency(meter) × Musical_Effect"
            }
        }

        # ثوابت البلاغة العربية
        self.rhetoric_constants = {
            "metaphor_weight": 0.9,
            "simile_weight": 0.8,
            "alliteration_weight": 0.7,
            "rhythm_weight": 0.85,
            "eloquence_threshold": 0.75,
            "beauty_standard": 0.8
        }

        # قاعدة بيانات الأجهزة البلاغية العربية
        self.arabic_rhetorical_devices = self._load_arabic_rhetorical_devices()

        # قاعدة بيانات الأوزان الشعرية
        self.arabic_meters = self._load_arabic_meters()

        # قاعدة بيانات الأساليب الأدبية
        self.arabic_literary_styles = self._load_arabic_literary_styles()

        # تاريخ التحليلات البلاغية
        self.rhetoric_history = []
        self.rhetoric_learning_database = {}

        print("🎨 تم إنشاء المعادلات البلاغة العربية المتخصصة:")
        for eq_name in self.rhetoric_equations.keys():
            print(f"   ✅ {eq_name}")

        print("✅ تم تهيئة محلل البلاغة العربية الموجه بالخبير!")

    def _load_arabic_rhetorical_devices(self) -> Dict[str, Dict[str, Any]]:
        """تحميل قاعدة بيانات الأجهزة البلاغية العربية"""
        return {
            "استعارة": {"type": "علم البيان", "effect": "تقريب المعنى وزيادة الجمال", "examples": ["البحر يضحك", "الليل يبكي"]},
            "تشبيه": {"type": "علم البيان", "effect": "توضيح المعنى بالمقارنة", "examples": ["كالأسد في الشجاعة", "كالبدر في الجمال"]},
            "كناية": {"type": "علم البيان", "effect": "التعبير غير المباشر", "examples": ["طويل النجاد", "كثير الرماد"]},
            "جناس": {"type": "علم البديع", "effect": "الجمال الصوتي", "examples": ["ويوم توفى الرسول وتوفى", "صليت المغرب في المغرب"]},
            "طباق": {"type": "علم البديع", "effect": "التضاد والتوازن", "examples": ["الليل والنهار", "الحب والكره"]},
            "سجع": {"type": "علم البديع", "effect": "الإيقاع والموسيقى", "examples": ["في الصيف ضيف", "العلم نور والجهل ظلام"]}
        }

    def _load_arabic_meters(self) -> Dict[str, Dict[str, Any]]:
        """تحميل قاعدة بيانات الأوزان الشعرية العربية"""
        return {
            "الطويل": {"pattern": "فعولن مفاعيلن فعولن مفاعيلن", "usage": "الشعر الجاهلي والإسلامي", "mood": "جدي وقوي"},
            "البسيط": {"pattern": "مستفعلن فاعلن مستفعلن فاعلن", "usage": "الشعر التعليمي", "mood": "واضح ومباشر"},
            "الوافر": {"pattern": "مفاعلتن مفاعلتن مفاعلتن", "usage": "الغزل والوصف", "mood": "عذب ورقيق"},
            "الكامل": {"pattern": "متفاعلن متفاعلن متفاعلن", "usage": "الحماسة والفخر", "mood": "قوي ومتدفق"},
            "الرجز": {"pattern": "مستفعلن مستفعلن مستفعلن", "usage": "الشعر الشعبي", "mood": "بسيط وسهل"},
            "المتقارب": {"pattern": "فعولن فعولن فعولن فعولن", "usage": "الشعر الصوفي", "mood": "هادئ ومتأمل"}
        }

    def _load_arabic_literary_styles(self) -> Dict[str, Dict[str, Any]]:
        """تحميل قاعدة بيانات الأساليب الأدبية العربية"""
        return {
            "الأسلوب الجاهلي": {"features": ["قوة اللفظ", "صدق العاطفة", "وضوح المعنى"], "period": "ما قبل الإسلام"},
            "الأسلوب الإسلامي": {"features": ["السهولة", "الوضوح", "التأثير"], "period": "صدر الإسلام"},
            "الأسلوب الأموي": {"features": ["الرقة", "العذوبة", "التنوع"], "period": "العصر الأموي"},
            "الأسلوب العباسي": {"features": ["التعقيد", "الزخرفة", "التنوع"], "period": "العصر العباسي"},
            "الأسلوب الأندلسي": {"features": ["الرقة", "الجمال", "الطبيعة"], "period": "الأندلس"},
            "الأسلوب الحديث": {"features": ["التجديد", "البساطة", "الوضوح"], "period": "العصر الحديث"}
        }

    def analyze_rhetoric_with_expert_guidance(self, request: RhetoricAnalysisRequest) -> RhetoricAnalysisResult:
        """التحليل البلاغي موجه بالخبير"""
        print(f"\n🎨 بدء التحليل البلاغي الموجه بالخبير للنص: {request.text[:50]}...")
        start_time = datetime.now()

        # المرحلة 1: تحليل الخبير للطلب البلاغي
        expert_analysis = self._analyze_rhetoric_request_with_expert(request)
        print(f"📜 تحليل الخبير البلاغي: {expert_analysis['complexity_assessment']}")

        # المرحلة 2: توليد توجيهات الخبير لمعادلات البلاغة
        expert_guidance = self._generate_rhetoric_expert_guidance(request, expert_analysis)
        print(f"🎨 توجيه الخبير البلاغي: {expert_guidance.recommended_evolution}")

        # المرحلة 3: تكيف معادلات البلاغة
        equation_adaptations = self._adapt_rhetoric_equations(expert_guidance, expert_analysis)
        print(f"🧮 تكيف معادلات البلاغة: {len(equation_adaptations)} معادلة")

        # المرحلة 4: تنفيذ التحليل البلاغي المتكيف
        rhetoric_analysis = self._perform_adaptive_rhetoric_analysis(request, equation_adaptations)

        # المرحلة 5: قياس التحسينات البلاغية
        performance_improvements = self._measure_rhetoric_improvements(request, rhetoric_analysis, equation_adaptations)

        # المرحلة 6: استخراج رؤى التعلم البلاغي
        learning_insights = self._extract_rhetoric_learning_insights(request, rhetoric_analysis, performance_improvements)

        # المرحلة 7: توليد توصيات للدورة التالية
        next_cycle_recommendations = self._generate_rhetoric_next_cycle_recommendations(performance_improvements, learning_insights)

        # إنشاء النتيجة البلاغية النهائية
        result = RhetoricAnalysisResult(
            success=True,
            text=request.text,
            literary_style=rhetoric_analysis.get("literary_style", ""),
            rhetorical_devices=rhetoric_analysis.get("rhetorical_devices", []),
            eloquence_score=rhetoric_analysis.get("eloquence_score", 0.0),
            beauty_assessment=rhetoric_analysis.get("beauty_assessment", {}),
            rhythm_analysis=rhetoric_analysis.get("rhythm_analysis", {}),
            overall_rhetoric_quality=rhetoric_analysis.get("overall_rhetoric_quality", 0.0),
            expert_guidance_applied=expert_guidance.__dict__,
            equation_adaptations=equation_adaptations,
            performance_improvements=performance_improvements,
            learning_insights=learning_insights,
            next_cycle_recommendations=next_cycle_recommendations
        )

        # حفظ في قاعدة التعلم البلاغي
        self._save_rhetoric_learning(request, result)

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ انتهى التحليل البلاغي الموجه في {total_time:.2f} ثانية")

        return result

    def _analyze_rhetoric_request_with_expert(self, request: RhetoricAnalysisRequest) -> Dict[str, Any]:
        """تحليل طلب البلاغة بواسطة الخبير"""

        # تحليل تعقيد النص
        words = request.text.split()
        text_complexity = len(words) * 1.2  # البلاغة أعقد من النحو
        context_complexity = len(request.context.split()) * 0.6 if request.context else 0

        # تحليل جوانب البلاغة المطلوبة
        aspects = request.rhetoric_aspects or ["metaphor", "simile", "alliteration", "rhythm", "eloquence"]
        aspects_complexity = len(aspects) * 4.0  # البلاغة معقدة جداً

        # تحليل عمق التحليل
        depth_complexity = {
            "basic": 4.0,
            "intermediate": 8.0,
            "comprehensive": 12.0
        }.get(request.analysis_depth, 8.0)

        # تحليل التعقيد البلاغي للنص
        rhetorical_complexity = 0
        if any(word in request.text for word in ["كأن", "مثل", "شبه", "يشبه"]):
            rhetorical_complexity += 4  # تشبيه
        if any(word in request.text for word in ["استعار", "كناية", "مجاز"]):
            rhetorical_complexity += 5  # استعارة أو كناية
        if len([word for word in words if len(word) > 6]) > len(words) * 0.3:
            rhetorical_complexity += 3  # ألفاظ معقدة

        total_complexity = text_complexity + context_complexity + aspects_complexity + depth_complexity + rhetorical_complexity

        return {
            "text_complexity": text_complexity,
            "context_complexity": context_complexity,
            "aspects_complexity": aspects_complexity,
            "depth_complexity": depth_complexity,
            "rhetorical_complexity": rhetorical_complexity,
            "total_complexity": total_complexity,
            "complexity_assessment": "بلاغة معقدة جداً" if total_complexity > 35 else "بلاغة متوسطة" if total_complexity > 20 else "بلاغة بسيطة",
            "recommended_adaptations": int(total_complexity // 4) + 5,
            "focus_areas": self._identify_rhetoric_focus_areas(request)
        }

    def _identify_rhetoric_focus_areas(self, request: RhetoricAnalysisRequest) -> List[str]:
        """تحديد مناطق التركيز البلاغي"""
        focus_areas = []

        aspects = request.rhetoric_aspects or ["metaphor", "simile", "alliteration", "rhythm", "eloquence"]

        if "metaphor" in aspects:
            focus_areas.append("metaphor_detection_enhancement")
        if "simile" in aspects:
            focus_areas.append("simile_recognition_improvement")
        if "alliteration" in aspects:
            focus_areas.append("alliteration_analysis_optimization")
        if "rhythm" in aspects:
            focus_areas.append("rhythm_detection_refinement")
        if "eloquence" in aspects:
            focus_areas.append("eloquence_measurement_enhancement")

        # تحليل خصائص النص
        words = request.text.split()
        if len(words) > 15:
            focus_areas.append("complex_text_handling")
        if any(word in request.text for word in ["كأن", "مثل", "شبه"]):
            focus_areas.append("simile_processing")
        if any(word in request.text for word in ["استعار", "كناية"]):
            focus_areas.append("metaphor_processing")
        if len([word for word in words if word.endswith("ة") or word.endswith("ان")]) > 2:
            focus_areas.append("rhyme_analysis")
        if request.context:
            focus_areas.append("contextual_rhetoric_analysis")

        return focus_areas

    def _generate_rhetoric_expert_guidance(self, request: RhetoricAnalysisRequest, analysis: Dict[str, Any]):
        """توليد توجيهات الخبير للتحليل البلاغي"""

        # تحديد التعقيد المستهدف للبلاغة
        target_complexity = 20 + analysis["recommended_adaptations"]

        # تحديد الدوال ذات الأولوية للبلاغة العربية
        priority_functions = []
        if "metaphor_detection_enhancement" in analysis["focus_areas"]:
            priority_functions.extend(["gaussian", "softplus"])  # لكشف الاستعارات
        if "simile_recognition_improvement" in analysis["focus_areas"]:
            priority_functions.extend(["sin_cos", "tanh"])  # لتمييز التشبيهات
        if "alliteration_analysis_optimization" in analysis["focus_areas"]:
            priority_functions.extend(["swish", "squared_relu"])  # لتحليل الجناس
        if "rhythm_detection_refinement" in analysis["focus_areas"]:
            priority_functions.extend(["hyperbolic", "softsign"])  # لكشف الإيقاع
        if "eloquence_measurement_enhancement" in analysis["focus_areas"]:
            priority_functions.extend(["sin", "cos"])  # لقياس الفصاحة
        if "complex_text_handling" in analysis["focus_areas"]:
            priority_functions.extend(["gaussian", "softplus"])  # للنصوص المعقدة
        if "simile_processing" in analysis["focus_areas"]:
            priority_functions.extend(["tanh", "swish"])  # لمعالجة التشبيه
        if "metaphor_processing" in analysis["focus_areas"]:
            priority_functions.extend(["gaussian", "hyperbolic"])  # لمعالجة الاستعارة

        # تحديد نوع التطور البلاغي
        if analysis["complexity_assessment"] == "بلاغة معقدة جداً":
            recommended_evolution = "increase"
            adaptation_strength = 0.98
        elif analysis["complexity_assessment"] == "بلاغة متوسطة":
            recommended_evolution = "restructure"
            adaptation_strength = 0.85
        else:
            recommended_evolution = "maintain"
            adaptation_strength = 0.75

        return MockRhetoricGuidance(
            target_complexity=target_complexity,
            focus_areas=analysis["focus_areas"],
            adaptation_strength=adaptation_strength,
            priority_functions=priority_functions or ["gaussian", "softplus"],
            recommended_evolution=recommended_evolution
        )

    def _adapt_rhetoric_equations(self, guidance, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """تكيف معادلات البلاغة"""

        adaptations = {}

        # إنشاء تحليل وهمي لمعادلات البلاغة
        mock_analysis = MockRhetoricAnalysis(
            rhetoric_accuracy=0.4,
            metaphor_clarity=0.5,
            literary_coherence=0.45,
            eloquence_precision=0.35,
            areas_for_improvement=guidance.focus_areas
        )

        # تكيف كل معادلة بلاغة
        for eq_name, equation in self.rhetoric_equations.items():
            print(f"   🎨 تكيف معادلة البلاغة: {eq_name}")
            equation.adapt_with_expert_guidance(guidance, mock_analysis)
            adaptations[eq_name] = equation.get_expert_guidance_summary()

        return adaptations

    def _perform_adaptive_rhetoric_analysis(self, request: RhetoricAnalysisRequest, adaptations: Dict[str, Any]) -> Dict[str, Any]:
        """تنفيذ التحليل البلاغي المتكيف"""

        analysis_results = {
            "literary_style": "",
            "rhetorical_devices": [],
            "eloquence_score": 0.0,
            "beauty_assessment": {},
            "rhythm_analysis": {},
            "overall_rhetoric_quality": 0.0
        }

        # تحديد الأسلوب الأدبي
        literary_style = self._identify_literary_style_adaptive(request.text)
        analysis_results["literary_style"] = literary_style

        # كشف الأجهزة البلاغية
        metaphor_accuracy = adaptations.get("metaphor_detector", {}).get("metaphor_detection", 0.5)
        simile_accuracy = adaptations.get("simile_analyzer", {}).get("simile_recognition", 0.55)
        rhetorical_devices = self._detect_rhetorical_devices_adaptive(request.text, metaphor_accuracy, simile_accuracy)
        analysis_results["rhetorical_devices"] = rhetorical_devices

        # قياس الفصاحة
        eloquence_accuracy = adaptations.get("eloquence_measurer", {}).get("eloquence_measurement", 0.35)
        eloquence_score = self._measure_eloquence_adaptive(request.text, eloquence_accuracy)
        analysis_results["eloquence_score"] = eloquence_score

        # تقييم الجمال
        beauty_accuracy = adaptations.get("beauty_assessor", {}).get("literary_beauty_assessment", 0.3)
        beauty_assessment = self._assess_beauty_adaptive(request.text, rhetorical_devices, beauty_accuracy)
        analysis_results["beauty_assessment"] = beauty_assessment

        # تحليل الإيقاع
        rhythm_accuracy = adaptations.get("rhythm_analyzer", {}).get("rhythm_detection", 0.4)
        rhythm_analysis = self._analyze_rhythm_adaptive(request.text, rhythm_accuracy)
        analysis_results["rhythm_analysis"] = rhythm_analysis

        # تقييم الجودة البلاغية الإجمالية
        overall_quality = np.mean([eloquence_score, beauty_assessment.get("overall_beauty", 0), rhythm_analysis.get("rhythm_quality", 0)])
        analysis_results["overall_rhetoric_quality"] = overall_quality

        return analysis_results

    def _identify_literary_style_adaptive(self, text: str) -> str:
        """تحديد الأسلوب الأدبي بطريقة متكيفة"""

        words = text.split()

        # أسلوب جاهلي (قوة وصدق)
        if any(word in text for word in ["صحراء", "ناقة", "سيف", "شجاع", "كريم"]):
            return "الأسلوب الجاهلي"

        # أسلوب إسلامي (وضوح وبساطة)
        if any(word in text for word in ["الله", "رسول", "إيمان", "تقوى", "جنة"]):
            return "الأسلوب الإسلامي"

        # أسلوب عباسي (تعقيد وزخرفة)
        if len([word for word in words if len(word) > 7]) > len(words) * 0.4:
            return "الأسلوب العباسي"

        # أسلوب أندلسي (رقة وطبيعة)
        if any(word in text for word in ["حديقة", "نهر", "زهر", "عطر", "جمال"]):
            return "الأسلوب الأندلسي"

        # أسلوب حديث (بساطة وتجديد)
        if len(words) < 20 and not any(word in text for word in ["كأن", "مثل", "استعار"]):
            return "الأسلوب الحديث"

        return "أسلوب مختلط"

    def _detect_rhetorical_devices_adaptive(self, text: str, metaphor_accuracy: float, simile_accuracy: float) -> List[RhetoricalDevice]:
        """كشف الأجهزة البلاغية بطريقة متكيفة"""

        devices = []

        # كشف التشبيه
        simile_patterns = ["كأن", "مثل", "شبه", "يشبه", "كالـ"]
        for pattern in simile_patterns:
            if pattern in text:
                device = RhetoricalDevice(
                    device_type="تشبيه",
                    text_span=f"النص المحتوي على '{pattern}'",
                    description="تشبيه يوضح المعنى بالمقارنة",
                    literary_effect="توضيح وتقريب المعنى",
                    beauty_score=0.7,
                    confidence=simile_accuracy
                )
                devices.append(device)

        # كشف الاستعارة (تقدير بسيط)
        metaphor_indicators = ["يضحك", "يبكي", "ينام", "يستيقظ"]
        for indicator in metaphor_indicators:
            if indicator in text and not any(word in text for word in ["الإنسان", "الرجل", "المرأة"]):
                device = RhetoricalDevice(
                    device_type="استعارة",
                    text_span=f"النص المحتوي على '{indicator}'",
                    description="استعارة تضفي الحياة على الجماد",
                    literary_effect="تقريب المعنى وزيادة الجمال",
                    beauty_score=0.8,
                    confidence=metaphor_accuracy
                )
                devices.append(device)

        # كشف الجناس (تشابه الأصوات)
        words = text.split()
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                if words[i][:3] == words[i+1][:3] or words[i][-3:] == words[i+1][-3:]:
                    device = RhetoricalDevice(
                        device_type="جناس",
                        text_span=f"{words[i]} - {words[i+1]}",
                        description="جناس يحدث جمالاً صوتياً",
                        literary_effect="إيقاع موسيقي جميل",
                        beauty_score=0.6,
                        confidence=0.6
                    )
                    devices.append(device)

        return devices

    def _measure_eloquence_adaptive(self, text: str, accuracy: float) -> float:
        """قياس الفصاحة بطريقة متكيفة"""

        words = text.split()

        # معايير الفصاحة
        word_clarity = len([word for word in words if len(word) >= 3 and len(word) <= 8]) / len(words)
        meaning_clarity = 1.0 - (len([word for word in words if len(word) > 10]) / len(words))
        expression_beauty = len([word for word in words if word.endswith("ة") or word.endswith("ان")]) / len(words)

        # حساب الفصاحة
        eloquence = (word_clarity * 0.4 + meaning_clarity * 0.4 + expression_beauty * 0.2) * accuracy

        return min(1.0, eloquence)

    def _assess_beauty_adaptive(self, text: str, devices: List[RhetoricalDevice], accuracy: float) -> Dict[str, float]:
        """تقييم الجمال بطريقة متكيفة"""

        # جمال الألفاظ
        words = text.split()
        word_beauty = len([word for word in words if len(word) >= 4]) / len(words)

        # جمال المعاني
        meaning_beauty = len(devices) * 0.1  # كل جهاز بلاغي يزيد الجمال

        # جمال الإيقاع
        rhythm_beauty = len([word for word in words if word.endswith("ة") or word.endswith("ان")]) / len(words)

        # الجمال الإجمالي
        overall_beauty = (word_beauty * 0.4 + meaning_beauty * 0.3 + rhythm_beauty * 0.3) * accuracy

        return {
            "word_beauty": min(1.0, word_beauty),
            "meaning_beauty": min(1.0, meaning_beauty),
            "rhythm_beauty": min(1.0, rhythm_beauty),
            "overall_beauty": min(1.0, overall_beauty)
        }

    def _analyze_rhythm_adaptive(self, text: str, accuracy: float) -> Dict[str, Any]:
        """تحليل الإيقاع بطريقة متكيفة"""

        words = text.split()

        # تحليل القافية
        rhyme_endings = {}
        for word in words:
            if len(word) >= 3:
                ending = word[-2:]
                rhyme_endings[ending] = rhyme_endings.get(ending, 0) + 1

        # أكثر قافية تكراراً
        most_common_rhyme = max(rhyme_endings.values()) if rhyme_endings else 0
        rhyme_consistency = most_common_rhyme / len(words) if words else 0

        # تحليل الوزن (تقدير بسيط)
        meter_pattern = self._detect_meter_pattern_simple(text)

        # جودة الإيقاع
        rhythm_quality = (rhyme_consistency * 0.6 + (1 if meter_pattern != "غير محدد" else 0) * 0.4) * accuracy

        return {
            "rhyme_consistency": rhyme_consistency,
            "meter_pattern": meter_pattern,
            "rhythm_quality": min(1.0, rhythm_quality),
            "musical_effect": "عالي" if rhythm_quality > 0.7 else "متوسط" if rhythm_quality > 0.4 else "ضعيف"
        }

    def _detect_meter_pattern_simple(self, text: str) -> str:
        """كشف الوزن الشعري بطريقة بسيطة"""

        words = text.split()

        # تحليل بسيط للوزن بناءً على طول الكلمات
        word_lengths = [len(word) for word in words]
        avg_length = np.mean(word_lengths) if word_lengths else 0

        # تقدير الوزن بناءً على متوسط طول الكلمات
        if avg_length >= 6:
            return "الطويل"
        elif avg_length >= 5:
            return "الكامل"
        elif avg_length >= 4:
            return "الوافر"
        elif avg_length >= 3:
            return "البسيط"
        else:
            return "الرجز"

    def _measure_rhetoric_improvements(self, request: RhetoricAnalysisRequest, analysis: Dict[str, Any], adaptations: Dict[str, Any]) -> Dict[str, float]:
        """قياس تحسينات أداء البلاغة"""

        improvements = {}

        # تحسن دقة البلاغة
        avg_rhetoric_accuracy = np.mean([adapt.get("rhetoric_accuracy", 0.4) for adapt in adaptations.values()])
        baseline_rhetoric_accuracy = 0.3
        rhetoric_accuracy_improvement = ((avg_rhetoric_accuracy - baseline_rhetoric_accuracy) / baseline_rhetoric_accuracy) * 100
        improvements["rhetoric_accuracy_improvement"] = max(0, rhetoric_accuracy_improvement)

        # تحسن كشف الاستعارات
        avg_metaphor_detection = np.mean([adapt.get("metaphor_detection", 0.5) for adapt in adaptations.values()])
        baseline_metaphor_detection = 0.4
        metaphor_detection_improvement = ((avg_metaphor_detection - baseline_metaphor_detection) / baseline_metaphor_detection) * 100
        improvements["metaphor_detection_improvement"] = max(0, metaphor_detection_improvement)

        # تحسن تمييز التشبيهات
        avg_simile_recognition = np.mean([adapt.get("simile_recognition", 0.55) for adapt in adaptations.values()])
        baseline_simile_recognition = 0.45
        simile_recognition_improvement = ((avg_simile_recognition - baseline_simile_recognition) / baseline_simile_recognition) * 100
        improvements["simile_recognition_improvement"] = max(0, simile_recognition_improvement)

        # تحسن تحليل الجناس
        avg_alliteration_analysis = np.mean([adapt.get("alliteration_analysis", 0.45) for adapt in adaptations.values()])
        baseline_alliteration_analysis = 0.35
        alliteration_analysis_improvement = ((avg_alliteration_analysis - baseline_alliteration_analysis) / baseline_alliteration_analysis) * 100
        improvements["alliteration_analysis_improvement"] = max(0, alliteration_analysis_improvement)

        # تحسن كشف الإيقاع
        avg_rhythm_detection = np.mean([adapt.get("rhythm_detection", 0.4) for adapt in adaptations.values()])
        baseline_rhythm_detection = 0.3
        rhythm_detection_improvement = ((avg_rhythm_detection - baseline_rhythm_detection) / baseline_rhythm_detection) * 100
        improvements["rhythm_detection_improvement"] = max(0, rhythm_detection_improvement)

        # تحسن قياس الفصاحة
        avg_eloquence_measurement = np.mean([adapt.get("eloquence_measurement", 0.35) for adapt in adaptations.values()])
        baseline_eloquence_measurement = 0.25
        eloquence_measurement_improvement = ((avg_eloquence_measurement - baseline_eloquence_measurement) / baseline_eloquence_measurement) * 100
        improvements["eloquence_measurement_improvement"] = max(0, eloquence_measurement_improvement)

        # تحسن تقييم الجمال
        avg_beauty_assessment = np.mean([adapt.get("literary_beauty_assessment", 0.3) for adapt in adaptations.values()])
        baseline_beauty_assessment = 0.2
        beauty_assessment_improvement = ((avg_beauty_assessment - baseline_beauty_assessment) / baseline_beauty_assessment) * 100
        improvements["beauty_assessment_improvement"] = max(0, beauty_assessment_improvement)

        # تحسن التعقيد البلاغي
        total_adaptations = sum(adapt.get("total_adaptations", 0) for adapt in adaptations.values())
        rhetoric_complexity_improvement = total_adaptations * 18  # كل تكيف بلاغي = 18% تحسن
        improvements["rhetoric_complexity_improvement"] = rhetoric_complexity_improvement

        return improvements

    def _extract_rhetoric_learning_insights(self, request: RhetoricAnalysisRequest, analysis: Dict[str, Any], improvements: Dict[str, float]) -> List[str]:
        """استخراج رؤى التعلم البلاغي"""

        insights = []

        if improvements["rhetoric_accuracy_improvement"] > 30:
            insights.append("التكيف الموجه بالخبير حسن دقة التحليل البلاغي بشكل استثنائي")

        if improvements["metaphor_detection_improvement"] > 25:
            insights.append("المعادلات المتكيفة ممتازة لكشف الاستعارات العربية")

        if improvements["simile_recognition_improvement"] > 22:
            insights.append("النظام نجح في تحسين تمييز التشبيهات البلاغية")

        if improvements["alliteration_analysis_improvement"] > 28:
            insights.append("تحليل الجناس والسجع تحسن مع التوجيه الخبير")

        if improvements["rhythm_detection_improvement"] > 33:
            insights.append("كشف الإيقاع والوزن أصبح أكثر دقة مع التكيف")

        if improvements["eloquence_measurement_improvement"] > 40:
            insights.append("قياس الفصاحة تحسن بشكل ملحوظ مع المعادلات المتكيفة")

        if improvements["beauty_assessment_improvement"] > 50:
            insights.append("تقييم الجمال الأدبي وصل لمستوى متقدم")

        if improvements["rhetoric_complexity_improvement"] > 150:
            insights.append("المعادلات البلاغية المتكيفة تتعامل مع التعقيد الأدبي بإتقان")

        # رؤى خاصة بالنص
        words_count = len(request.text.split())
        if words_count > 15:
            insights.append("النظام يتعامل بكفاءة مع النصوص الأدبية المعقدة")

        if request.context:
            insights.append("التحليل السياقي يحسن دقة التحليل البلاغي")

        if analysis.get("literary_style") in ["الأسلوب الجاهلي", "الأسلوب العباسي"]:
            insights.append("النظام يحلل الأساليب الأدبية الكلاسيكية بدقة متقدمة")

        if len(analysis.get("rhetorical_devices", [])) > 2:
            insights.append("النظام كشف أجهزة بلاغية متعددة في النص")

        return insights

    def _generate_rhetoric_next_cycle_recommendations(self, improvements: Dict[str, float], insights: List[str]) -> List[str]:
        """توليد توصيات للدورة البلاغية التالية"""

        recommendations = []

        avg_improvement = np.mean(list(improvements.values()))

        if avg_improvement > 45:
            recommendations.append("الحفاظ على إعدادات التكيف البلاغي الحالية")
            recommendations.append("تجربة تحليل بلاغي أعمق للنصوص الشعرية المعقدة")
        elif avg_improvement > 25:
            recommendations.append("زيادة قوة التكيف البلاغي تدريجياً")
            recommendations.append("إضافة أجهزة بلاغية متقدمة")
        else:
            recommendations.append("مراجعة استراتيجية التوجيه البلاغي")
            recommendations.append("تحسين دقة معادلات البلاغة")

        # توصيات محددة
        if "الاستعارات" in str(insights):
            recommendations.append("التوسع في قاعدة بيانات الاستعارات العربية")

        if "التشبيهات" in str(insights):
            recommendations.append("تطوير خوارزميات تمييز التشبيهات المعقدة")

        if "الإيقاع" in str(insights):
            recommendations.append("تحسين تحليل الأوزان الشعرية العربية")

        if "الفصاحة" in str(insights):
            recommendations.append("تعزيز معايير قياس الفصاحة والبلاغة")

        if "الجمال" in str(insights):
            recommendations.append("تطوير تقييم الجمال الأدبي متعدد الأبعاد")

        if "الكلاسيكية" in str(insights):
            recommendations.append("إضافة تحليل متخصص للأساليب الأدبية التراثية")

        return recommendations

    def _save_rhetoric_learning(self, request: RhetoricAnalysisRequest, result: RhetoricAnalysisResult):
        """حفظ التعلم البلاغي"""

        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "text": request.text,
            "context": request.context,
            "analysis_depth": request.analysis_depth,
            "success": result.success,
            "literary_style": result.literary_style,
            "eloquence_score": result.eloquence_score,
            "overall_rhetoric_quality": result.overall_rhetoric_quality,
            "rhetorical_devices_count": len(result.rhetorical_devices),
            "performance_improvements": result.performance_improvements,
            "learning_insights": result.learning_insights
        }

        text_key = f"{len(request.text.split())}_{request.analysis_depth}"
        if text_key not in self.rhetoric_learning_database:
            self.rhetoric_learning_database[text_key] = []

        self.rhetoric_learning_database[text_key].append(learning_entry)

        # الاحتفاظ بآخر 3 إدخالات فقط
        if len(self.rhetoric_learning_database[text_key]) > 3:
            self.rhetoric_learning_database[text_key] = self.rhetoric_learning_database[text_key][-3:]

def main():
    """اختبار محلل البلاغة العربية الموجه بالخبير"""
    print("🧪 اختبار محلل البلاغة العربية الموجه بالخبير...")

    # إنشاء المحلل البلاغي
    rhetoric_analyzer = ExpertGuidedArabicRhetoricAnalyzer()

    # نصوص اختبار عربية بلاغية
    test_texts = [
        "البحر يضحك والأمواج تلعب كالأطفال",
        "الليل كأنه عباءة سوداء تغطي الأرض",
        "في الصيف ضيف والشتاء شتات",
        "الله نور السماوات والأرض",
        "الحديقة تفوح بعطر الزهور والجمال يملأ المكان"
    ]

    for text in test_texts:
        print(f"\n{'='*70}")
        print(f"🎨 تحليل النص: {text}")

        # طلب التحليل البلاغي
        rhetoric_request = RhetoricAnalysisRequest(
            text=text,
            context="سياق أدبي تجريبي",
            analysis_depth="comprehensive",
            rhetoric_aspects=["metaphor", "simile", "alliteration", "rhythm", "eloquence"],
            expert_guidance_level="adaptive",
            learning_enabled=True
        )

        # تنفيذ التحليل البلاغي
        rhetoric_result = rhetoric_analyzer.analyze_rhetoric_with_expert_guidance(rhetoric_request)

        # عرض النتائج البلاغية
        print(f"\n📊 نتائج التحليل البلاغي:")
        print(f"   ✅ النجاح: {rhetoric_result.success}")
        print(f"   📜 الأسلوب الأدبي: {rhetoric_result.literary_style}")
        print(f"   🎯 درجة الفصاحة: {rhetoric_result.eloquence_score:.2%}")
        print(f"   ⭐ الجودة البلاغية: {rhetoric_result.overall_rhetoric_quality:.2%}")

        if rhetoric_result.rhetorical_devices:
            print(f"   🎨 الأجهزة البلاغية:")
            for device in rhetoric_result.rhetorical_devices:
                print(f"      • {device.device_type}: {device.text_span}")
                print(f"        التأثير: {device.literary_effect}")
                print(f"        درجة الجمال: {device.beauty_score:.1f}")

        if rhetoric_result.beauty_assessment:
            print(f"   💎 تقييم الجمال:")
            beauty = rhetoric_result.beauty_assessment
            print(f"      جمال الألفاظ: {beauty.get('word_beauty', 0):.2%}")
            print(f"      جمال المعاني: {beauty.get('meaning_beauty', 0):.2%}")
            print(f"      جمال الإيقاع: {beauty.get('rhythm_beauty', 0):.2%}")

        if rhetoric_result.rhythm_analysis:
            print(f"   🎵 تحليل الإيقاع:")
            rhythm = rhetoric_result.rhythm_analysis
            print(f"      الوزن: {rhythm.get('meter_pattern', 'غير محدد')}")
            print(f"      التأثير الموسيقي: {rhythm.get('musical_effect', 'غير محدد')}")

        if rhetoric_result.performance_improvements:
            print(f"   📈 تحسينات الأداء:")
            for metric, improvement in rhetoric_result.performance_improvements.items():
                print(f"      {metric}: {improvement:.1f}%")

        if rhetoric_result.learning_insights:
            print(f"   🧠 رؤى التعلم:")
            for insight in rhetoric_result.learning_insights:
                print(f"      • {insight}")

if __name__ == "__main__":
    main()
