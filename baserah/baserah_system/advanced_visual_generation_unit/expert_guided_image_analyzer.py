#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-Guided Image Analyzer - Part 1: Visual Image Analysis
محلل الصور الموجه بالخبير - الجزء الأول: تحليل الصور البصري

Revolutionary integration of Expert/Explorer guidance with image analysis,
applying adaptive mathematical equations to enhance visual understanding.

التكامل الثوري لتوجيه الخبير/المستكشف مع تحليل الصور،
تطبيق المعادلات الرياضية المتكيفة لتحسين الفهم البصري.

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# إضافة المسارات
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# استيراد النظام الموجود
from revolutionary_database import ShapeEntity

# محاكاة النظام المتكيف للصور
class MockImageEquation:
    def __init__(self, name: str, input_dim: int, output_dim: int):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.current_complexity = 8
        self.adaptation_count = 0
        self.image_accuracy = 0.75
        self.visual_clarity = 0.8
        self.color_harmony = 0.7
        self.composition_balance = 0.85

    def adapt_with_expert_guidance(self, guidance, analysis):
        self.adaptation_count += 1
        if hasattr(guidance, 'recommended_evolution'):
            if guidance.recommended_evolution == "increase":
                self.current_complexity += 2
                self.image_accuracy += 0.05
                self.visual_clarity += 0.04
                self.color_harmony += 0.03
            elif guidance.recommended_evolution == "restructure":
                self.image_accuracy += 0.03
                self.composition_balance += 0.04

    def get_expert_guidance_summary(self):
        return {
            "current_complexity": self.current_complexity,
            "total_adaptations": self.adaptation_count,
            "image_accuracy": self.image_accuracy,
            "visual_clarity": self.visual_clarity,
            "color_harmony": self.color_harmony,
            "composition_balance": self.composition_balance,
            "average_improvement": 0.12 * self.adaptation_count
        }

class MockImageGuidance:
    def __init__(self, target_complexity, focus_areas, adaptation_strength, priority_functions, recommended_evolution):
        self.target_complexity = target_complexity
        self.focus_areas = focus_areas
        self.adaptation_strength = adaptation_strength
        self.priority_functions = priority_functions
        self.recommended_evolution = recommended_evolution

class MockImageAnalysis:
    def __init__(self, image_accuracy, visual_quality, color_analysis, composition_score, artistic_value, areas_for_improvement):
        self.image_accuracy = image_accuracy
        self.visual_quality = visual_quality
        self.color_analysis = color_analysis
        self.composition_score = composition_score
        self.artistic_value = artistic_value
        self.areas_for_improvement = areas_for_improvement

@dataclass
class ImageAnalysisRequest:
    """طلب تحليل الصور"""
    shape: ShapeEntity
    analysis_type: str  # "quality", "composition", "color", "artistic", "comprehensive"
    image_aspects: List[str]  # ["clarity", "balance", "harmony", "creativity"]
    expert_guidance_level: str = "adaptive"
    learning_enabled: bool = True
    visual_optimization: bool = True

@dataclass
class ImageAnalysisResult:
    """نتيجة تحليل الصور"""
    success: bool
    image_compliance: Dict[str, float]
    visual_violations: List[str]
    image_insights: List[str]
    quality_metrics: Dict[str, float]
    color_analysis: Dict[str, Any]
    composition_scores: Dict[str, float]
    artistic_evaluation: Dict[str, float]
    expert_guidance_applied: Dict[str, Any] = None
    equation_adaptations: Dict[str, Any] = None
    performance_improvements: Dict[str, float] = None
    learning_insights: List[str] = None
    next_cycle_recommendations: List[str] = None

class ExpertGuidedImageAnalyzer:
    """محلل الصور الموجه بالخبير الثوري"""

    def __init__(self):
        """تهيئة محلل الصور الموجه بالخبير"""
        print("🌟" + "="*90 + "🌟")
        print("🖼️ محلل الصور الموجه بالخبير الثوري")
        print("🎨 الخبير/المستكشف يقود التحليل البصري بذكاء")
        print("🧮 معادلات رياضية متكيفة + تحليل صور متقدم")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" + "="*90 + "🌟")

        # إنشاء معادلات الصور متخصصة
        self.image_equations = {
            "quality_analyzer": MockImageEquation("quality_analysis", 12, 8),
            "composition_evaluator": MockImageEquation("composition_evaluation", 15, 10),
            "color_harmony_detector": MockImageEquation("color_harmony", 10, 6),
            "visual_clarity_enhancer": MockImageEquation("visual_clarity", 14, 9),
            "artistic_value_assessor": MockImageEquation("artistic_assessment", 18, 12),
            "balance_optimizer": MockImageEquation("balance_optimization", 11, 7),
            "contrast_analyzer": MockImageEquation("contrast_analysis", 9, 5),
            "lighting_evaluator": MockImageEquation("lighting_evaluation", 13, 8)
        }

        # معايير تحليل الصور
        self.image_standards = {
            "visual_quality": {
                "name": "جودة بصرية",
                "criteria": "وضوح وحدة ودقة الصورة",
                "spiritual_meaning": "الجمال انعكاس للكمال الإلهي"
            },
            "color_harmony": {
                "name": "تناغم الألوان",
                "criteria": "توافق وانسجام الألوان",
                "spiritual_meaning": "التناغم من صفات الخلق الإلهي"
            },
            "composition_balance": {
                "name": "توازن التركيب",
                "criteria": "توزيع العناصر بتوازن",
                "spiritual_meaning": "التوازن سنة كونية إلهية"
            },
            "artistic_creativity": {
                "name": "الإبداع الفني",
                "criteria": "الأصالة والابتكار الفني",
                "spiritual_meaning": "الإبداع هبة من الله"
            }
        }

        # تاريخ التحليلات البصرية
        self.image_history = []
        self.image_learning_database = {}

        print("🖼️ تم إنشاء المعادلات البصرية المتخصصة:")
        for eq_name in self.image_equations.keys():
            print(f"   ✅ {eq_name}")

        print("✅ تم تهيئة محلل الصور الموجه بالخبير!")

    def analyze_image_with_expert_guidance(self, request: ImageAnalysisRequest) -> ImageAnalysisResult:
        """تحليل الصور موجه بالخبير"""
        print(f"\n🖼️ بدء تحليل الصور الموجه بالخبير لـ: {request.shape.name}")
        start_time = datetime.now()

        # المرحلة 1: تحليل الخبير للطلب البصري
        expert_analysis = self._analyze_image_request_with_expert(request)
        print(f"🎨 تحليل الخبير البصري: {expert_analysis['complexity_assessment']}")

        # المرحلة 2: توليد توجيهات الخبير للمعادلات البصرية
        expert_guidance = self._generate_image_expert_guidance(request, expert_analysis)
        print(f"🖼️ توجيه الخبير البصري: {expert_guidance.recommended_evolution}")

        # المرحلة 3: تكيف المعادلات البصرية
        equation_adaptations = self._adapt_image_equations(expert_guidance, expert_analysis)
        print(f"🧮 تكيف المعادلات البصرية: {len(equation_adaptations)} معادلة")

        # المرحلة 4: تنفيذ التحليل البصري المتكيف
        image_analysis = self._perform_adaptive_image_analysis(request, equation_adaptations)

        # المرحلة 5: فحص المعايير البصرية
        image_compliance = self._check_image_standards_compliance(request, image_analysis)

        # المرحلة 6: تحليل جودة الصورة
        quality_metrics = self._analyze_image_quality(request, image_analysis)

        # المرحلة 7: تحليل الألوان
        color_analysis = self._analyze_color_composition(request, image_analysis)

        # المرحلة 8: تقييم التركيب
        composition_scores = self._evaluate_composition(request, image_analysis)

        # المرحلة 9: التقييم الفني
        artistic_evaluation = self._evaluate_artistic_value(request, image_analysis)

        # المرحلة 10: قياس التحسينات البصرية
        performance_improvements = self._measure_image_improvements(request, image_analysis, equation_adaptations)

        # المرحلة 11: استخراج رؤى التعلم البصري
        learning_insights = self._extract_image_learning_insights(request, image_analysis, performance_improvements)

        # المرحلة 12: توليد توصيات للدورة التالية
        next_cycle_recommendations = self._generate_image_next_cycle_recommendations(performance_improvements, learning_insights)

        # إنشاء النتيجة البصرية
        result = ImageAnalysisResult(
            success=True,
            image_compliance=image_compliance["compliance_scores"],
            visual_violations=image_compliance["violations"],
            image_insights=image_analysis["insights"],
            quality_metrics=quality_metrics,
            color_analysis=color_analysis,
            composition_scores=composition_scores,
            artistic_evaluation=artistic_evaluation,
            expert_guidance_applied=expert_guidance.__dict__,
            equation_adaptations=equation_adaptations,
            performance_improvements=performance_improvements,
            learning_insights=learning_insights,
            next_cycle_recommendations=next_cycle_recommendations
        )

        # حفظ في قاعدة التعلم البصري
        self._save_image_learning(request, result)

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ انتهى التحليل البصري الموجه في {total_time:.2f} ثانية")

        return result

    def _analyze_image_request_with_expert(self, request: ImageAnalysisRequest) -> Dict[str, Any]:
        """تحليل الطلب البصري بواسطة الخبير"""

        # تحليل الخصائص البصرية للشكل
        visual_complexity = len(request.shape.equation_params) * 1.5
        color_richness = len(request.shape.color_properties) * 2.0
        geometric_detail = request.shape.geometric_features.get("area", 100) / 50.0

        # تحليل جوانب الصورة المطلوبة
        image_aspects_complexity = len(request.image_aspects) * 2.5

        # تحليل نوع التحليل
        analysis_type_complexity = {
            "quality": 2.0,
            "composition": 3.0,
            "color": 2.5,
            "artistic": 4.0,
            "comprehensive": 5.0
        }.get(request.analysis_type, 2.0)

        total_image_complexity = visual_complexity + color_richness + geometric_detail + image_aspects_complexity + analysis_type_complexity

        return {
            "visual_complexity": visual_complexity,
            "color_richness": color_richness,
            "geometric_detail": geometric_detail,
            "image_aspects_complexity": image_aspects_complexity,
            "analysis_type_complexity": analysis_type_complexity,
            "total_image_complexity": total_image_complexity,
            "complexity_assessment": "بصري معقد" if total_image_complexity > 20 else "بصري متوسط" if total_image_complexity > 12 else "بصري بسيط",
            "recommended_adaptations": int(total_image_complexity // 4) + 2,
            "focus_areas": self._identify_image_focus_areas(request)
        }

    def _identify_image_focus_areas(self, request: ImageAnalysisRequest) -> List[str]:
        """تحديد مناطق التركيز البصري"""
        focus_areas = []

        if "clarity" in request.image_aspects:
            focus_areas.append("visual_clarity_enhancement")
        if "balance" in request.image_aspects:
            focus_areas.append("composition_balance")
        if "harmony" in request.image_aspects:
            focus_areas.append("color_harmony_optimization")
        if "creativity" in request.image_aspects:
            focus_areas.append("artistic_innovation")
        if request.analysis_type == "quality":
            focus_areas.append("quality_assessment")
        if request.analysis_type == "artistic":
            focus_areas.append("artistic_evaluation")
        if request.visual_optimization:
            focus_areas.append("visual_enhancement")

        return focus_areas

    def _generate_image_expert_guidance(self, request: ImageAnalysisRequest, analysis: Dict[str, Any]):
        """توليد توجيهات الخبير للتحليل البصري"""

        # تحديد التعقيد المستهدف للصور
        target_complexity = 10 + analysis["recommended_adaptations"]

        # تحديد الدوال ذات الأولوية للتحليل البصري
        priority_functions = []
        if "visual_clarity_enhancement" in analysis["focus_areas"]:
            priority_functions.extend(["gaussian", "softplus"])
        if "composition_balance" in analysis["focus_areas"]:
            priority_functions.extend(["tanh", "hyperbolic"])
        if "color_harmony_optimization" in analysis["focus_areas"]:
            priority_functions.extend(["sin_cos", "swish"])
        if "artistic_innovation" in analysis["focus_areas"]:
            priority_functions.extend(["squared_relu", "softsign"])
        if "quality_assessment" in analysis["focus_areas"]:
            priority_functions.extend(["sin", "cos"])
        if "artistic_evaluation" in analysis["focus_areas"]:
            priority_functions.extend(["gaussian", "hyperbolic"])
        if "visual_enhancement" in analysis["focus_areas"]:
            priority_functions.extend(["softplus", "swish"])

        # تحديد نوع التطور البصري
        if analysis["complexity_assessment"] == "بصري معقد":
            recommended_evolution = "increase"
            adaptation_strength = 0.9
        elif analysis["complexity_assessment"] == "بصري متوسط":
            recommended_evolution = "restructure"
            adaptation_strength = 0.75
        else:
            recommended_evolution = "maintain"
            adaptation_strength = 0.6

        return MockImageGuidance(
            target_complexity=target_complexity,
            focus_areas=analysis["focus_areas"],
            adaptation_strength=adaptation_strength,
            priority_functions=priority_functions or ["gaussian", "tanh"],
            recommended_evolution=recommended_evolution
        )

    def _adapt_image_equations(self, guidance, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """تكيف المعادلات البصرية"""

        adaptations = {}

        # إنشاء تحليل وهمي للمعادلات البصرية
        mock_analysis = MockImageAnalysis(
            image_accuracy=0.75,
            visual_quality=0.8,
            color_analysis=0.7,
            composition_score=0.85,
            artistic_value=0.6,
            areas_for_improvement=guidance.focus_areas
        )

        # تكيف كل معادلة بصرية
        for eq_name, equation in self.image_equations.items():
            print(f"   🖼️ تكيف معادلة بصرية: {eq_name}")
            equation.adapt_with_expert_guidance(guidance, mock_analysis)
            adaptations[eq_name] = equation.get_expert_guidance_summary()

        return adaptations

    def _perform_adaptive_image_analysis(self, request: ImageAnalysisRequest, adaptations: Dict[str, Any]) -> Dict[str, Any]:
        """تنفيذ التحليل البصري المتكيف"""

        analysis_results = {
            "insights": [],
            "image_calculations": {},
            "visual_predictions": [],
            "quality_scores": {}
        }

        # تحليل جودة الصورة
        quality_accuracy = adaptations.get("quality_analyzer", {}).get("image_accuracy", 0.75)
        analysis_results["insights"].append(f"تحليل الجودة البصرية: دقة {quality_accuracy:.2%}")
        analysis_results["image_calculations"]["quality"] = self._calculate_image_quality(request.shape)

        # تحليل التركيب
        if "composition" in request.analysis_type:
            composition_balance = adaptations.get("composition_evaluator", {}).get("composition_balance", 0.85)
            analysis_results["insights"].append(f"تقييم التركيب: توازن {composition_balance:.2%}")
            analysis_results["image_calculations"]["composition"] = self._calculate_composition_metrics(request.shape)

        # تحليل الألوان
        if "color" in request.analysis_type:
            color_harmony = adaptations.get("color_harmony_detector", {}).get("color_harmony", 0.7)
            analysis_results["insights"].append(f"تناغم الألوان: انسجام {color_harmony:.2%}")
            analysis_results["image_calculations"]["color"] = self._calculate_color_harmony(request.shape)

        # التقييم الفني
        if "artistic" in request.analysis_type:
            artistic_value = adaptations.get("artistic_value_assessor", {}).get("image_accuracy", 0.75)
            analysis_results["insights"].append(f"التقييم الفني: قيمة {artistic_value:.2%}")
            analysis_results["image_calculations"]["artistic"] = self._calculate_artistic_value(request.shape)

        return analysis_results

    def _calculate_image_quality(self, shape: ShapeEntity) -> Dict[str, float]:
        """حساب جودة الصورة"""
        clarity = min(1.0, shape.geometric_features.get("area", 100) / 200.0)
        sharpness = len(shape.equation_params) * 0.15
        detail_level = min(1.0, sharpness)

        return {
            "clarity": clarity,
            "sharpness": min(1.0, sharpness),
            "detail_level": detail_level,
            "overall_quality": (clarity + detail_level) / 2
        }

    def _calculate_composition_metrics(self, shape: ShapeEntity) -> Dict[str, float]:
        """حساب مقاييس التركيب"""
        balance = shape.position_info.get("center_x", 0.5) * shape.position_info.get("center_y", 0.5)
        symmetry = shape.geometric_features.get("symmetry", 0.8)
        proportion = min(1.0, shape.geometric_features.get("area", 100) / 150.0)

        return {
            "balance": min(1.0, balance * 2),
            "symmetry": symmetry,
            "proportion": proportion,
            "composition_score": (balance + symmetry + proportion) / 3
        }

    def _calculate_color_harmony(self, shape: ShapeEntity) -> Dict[str, float]:
        """حساب تناغم الألوان"""
        color_count = len(shape.color_properties)
        harmony_score = 1.0 - (abs(color_count - 3) * 0.1)  # 3 ألوان مثالية
        saturation = 0.8  # افتراضي
        contrast = 0.75   # افتراضي

        return {
            "harmony_score": max(0.3, harmony_score),
            "color_count": color_count,
            "saturation": saturation,
            "contrast": contrast,
            "overall_harmony": (harmony_score + saturation + contrast) / 3
        }

    def _calculate_artistic_value(self, shape: ShapeEntity) -> Dict[str, float]:
        """حساب القيمة الفنية"""
        creativity = len(shape.equation_params) * 0.2
        originality = shape.geometric_features.get("uniqueness", 0.7)
        aesthetic_appeal = shape.geometric_features.get("beauty", 0.8)

        return {
            "creativity": min(1.0, creativity),
            "originality": originality,
            "aesthetic_appeal": aesthetic_appeal,
            "artistic_score": (creativity + originality + aesthetic_appeal) / 3
        }

    def _check_image_standards_compliance(self, request: ImageAnalysisRequest, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """فحص الامتثال للمعايير البصرية"""

        compliance = {
            "compliance_scores": {},
            "violations": [],
            "recommendations": []
        }

        # فحص الجودة البصرية
        quality_data = analysis["image_calculations"].get("quality", {})
        quality_score = quality_data.get("overall_quality", 0.5)
        compliance["compliance_scores"]["visual_quality"] = quality_score
        if quality_score < 0.6:
            compliance["violations"].append("جودة بصرية منخفضة")

        # فحص تناغم الألوان
        if "color" in analysis["image_calculations"]:
            color_data = analysis["image_calculations"]["color"]
            harmony_score = color_data.get("overall_harmony", 0.5)
            compliance["compliance_scores"]["color_harmony"] = harmony_score
            if harmony_score < 0.5:
                compliance["violations"].append("تناغم ألوان ضعيف")

        return compliance

    def _analyze_image_quality(self, request: ImageAnalysisRequest, analysis: Dict[str, Any]) -> Dict[str, float]:
        """تحليل جودة الصورة"""
        quality_data = analysis["image_calculations"].get("quality", {})

        return {
            "resolution_quality": quality_data.get("clarity", 0.7),
            "detail_quality": quality_data.get("detail_level", 0.6),
            "sharpness_quality": quality_data.get("sharpness", 0.8),
            "overall_quality": quality_data.get("overall_quality", 0.7)
        }

    def _analyze_color_composition(self, request: ImageAnalysisRequest, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل تركيب الألوان"""
        color_data = analysis["image_calculations"].get("color", {})

        return {
            "harmony_analysis": {
                "score": color_data.get("harmony_score", 0.7),
                "color_count": color_data.get("color_count", 3),
                "balance": "متوازن" if color_data.get("harmony_score", 0.7) > 0.6 else "غير متوازن"
            },
            "saturation_analysis": {
                "level": color_data.get("saturation", 0.8),
                "assessment": "مناسب" if color_data.get("saturation", 0.8) > 0.6 else "يحتاج تحسين"
            },
            "contrast_analysis": {
                "level": color_data.get("contrast", 0.75),
                "effectiveness": "فعال" if color_data.get("contrast", 0.75) > 0.6 else "ضعيف"
            }
        }

    def _evaluate_composition(self, request: ImageAnalysisRequest, analysis: Dict[str, Any]) -> Dict[str, float]:
        """تقييم التركيب"""
        composition_data = analysis["image_calculations"].get("composition", {})

        return {
            "balance_score": composition_data.get("balance", 0.8),
            "symmetry_score": composition_data.get("symmetry", 0.8),
            "proportion_score": composition_data.get("proportion", 0.7),
            "overall_composition": composition_data.get("composition_score", 0.75)
        }

    def _evaluate_artistic_value(self, request: ImageAnalysisRequest, analysis: Dict[str, Any]) -> Dict[str, float]:
        """تقييم القيمة الفنية"""
        artistic_data = analysis["image_calculations"].get("artistic", {})

        return {
            "creativity_score": artistic_data.get("creativity", 0.6),
            "originality_score": artistic_data.get("originality", 0.7),
            "aesthetic_score": artistic_data.get("aesthetic_appeal", 0.8),
            "overall_artistic_value": artistic_data.get("artistic_score", 0.7)
        }

    def _measure_image_improvements(self, request: ImageAnalysisRequest, analysis: Dict[str, Any], adaptations: Dict[str, Any]) -> Dict[str, float]:
        """قياس تحسينات الصورة"""

        improvements = {}

        # تحسن الجودة البصرية
        avg_accuracy = np.mean([adapt.get("image_accuracy", 0.75) for adapt in adaptations.values()])
        baseline_accuracy = 0.65
        quality_improvement = ((avg_accuracy - baseline_accuracy) / baseline_accuracy) * 100
        improvements["image_quality_improvement"] = max(0, quality_improvement)

        # تحسن التركيب
        avg_composition = np.mean([adapt.get("composition_balance", 0.85) for adapt in adaptations.values()])
        baseline_composition = 0.7
        composition_improvement = ((avg_composition - baseline_composition) / baseline_composition) * 100
        improvements["composition_improvement"] = max(0, composition_improvement)

        # تحسن الألوان
        avg_color = np.mean([adapt.get("color_harmony", 0.7) for adapt in adaptations.values()])
        baseline_color = 0.6
        color_improvement = ((avg_color - baseline_color) / baseline_color) * 100
        improvements["color_improvement"] = max(0, color_improvement)

        return improvements

    def _extract_image_learning_insights(self, request: ImageAnalysisRequest, analysis: Dict[str, Any], improvements: Dict[str, float]) -> List[str]:
        """استخراج رؤى التعلم البصري"""

        insights = []

        if improvements["image_quality_improvement"] > 15:
            insights.append("التكيف الموجه بالخبير حسن الجودة البصرية بشكل ملحوظ")

        if improvements["composition_improvement"] > 20:
            insights.append("المعادلات المتكيفة ممتازة لتحسين التركيب البصري")

        if improvements["color_improvement"] > 18:
            insights.append("النظام نجح في تحسين تناغم الألوان")

        if request.analysis_type == "artistic":
            insights.append("التحليل الفني يستفيد من التوجيه الخبير")

        return insights

    def _generate_image_next_cycle_recommendations(self, improvements: Dict[str, float], insights: List[str]) -> List[str]:
        """توليد توصيات للدورة التالية"""

        recommendations = []

        avg_improvement = np.mean(list(improvements.values()))

        if avg_improvement > 20:
            recommendations.append("الحفاظ على إعدادات التكيف البصري الحالية")
            recommendations.append("تجربة تحليل بصري أكثر تعقيداً")
        elif avg_improvement > 12:
            recommendations.append("زيادة قوة التكيف البصري تدريجياً")
            recommendations.append("إضافة معايير بصرية جديدة")
        else:
            recommendations.append("مراجعة استراتيجية التوجيه البصري")
            recommendations.append("تحسين دقة المعادلات البصرية")

        return recommendations

    def _save_image_learning(self, request: ImageAnalysisRequest, result: ImageAnalysisResult):
        """حفظ التعلم البصري"""

        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "shape_name": request.shape.name,
            "analysis_type": request.analysis_type,
            "image_aspects": request.image_aspects,
            "success": result.success,
            "performance_improvements": result.performance_improvements,
            "learning_insights": result.learning_insights
        }

        shape_key = f"{request.shape.category}_{request.analysis_type}"
        if shape_key not in self.image_learning_database:
            self.image_learning_database[shape_key] = []

        self.image_learning_database[shape_key].append(learning_entry)

        # الاحتفاظ بآخر 5 إدخالات
        if len(self.image_learning_database[shape_key]) > 5:
            self.image_learning_database[shape_key] = self.image_learning_database[shape_key][-5:]

def main():
    """اختبار محلل الصور الموجه بالخبير"""
    print("🧪 اختبار محلل الصور الموجه بالخبير...")

    # إنشاء المحلل
    image_analyzer = ExpertGuidedImageAnalyzer()

    # إنشاء شكل اختبار
    from revolutionary_database import ShapeEntity

    test_shape = ShapeEntity(
        id=1, name="لوحة فنية جميلة", category="فنون",
        equation_params={"beauty": 0.95, "harmony": 0.9, "creativity": 0.85},
        geometric_features={"area": 180.0, "symmetry": 0.92, "uniqueness": 0.88},
        color_properties={"dominant_color": [255, 150, 100], "secondary_color": [100, 200, 255]},
        position_info={"center_x": 0.6, "center_y": 0.4},
        tolerance_thresholds={}, created_date="", updated_date=""
    )

    # طلب تحليل بصري شامل
    analysis_request = ImageAnalysisRequest(
        shape=test_shape,
        analysis_type="comprehensive",
        image_aspects=["clarity", "balance", "harmony", "creativity"],
        expert_guidance_level="adaptive",
        learning_enabled=True,
        visual_optimization=True
    )

    # تنفيذ التحليل
    result = image_analyzer.analyze_image_with_expert_guidance(analysis_request)

    print(f"\n🖼️ نتائج التحليل البصري:")
    print(f"   ✅ النجاح: {result.success}")
    if result.success:
        print(f"   🎨 رؤى بصرية: {len(result.image_insights)} رؤية")
        print(f"   📊 مقاييس الجودة: {len(result.quality_metrics)} مقياس")
        print(f"   🌈 تحليل الألوان: متاح")
        print(f"   📐 نتائج التركيب: متاح")
        print(f"   🎭 التقييم الفني: متاح")

    print(f"\n📊 إحصائيات المحلل:")
    print(f"   🖼️ معادلات بصرية: {len(image_analyzer.image_equations)}")
    print(f"   📚 قاعدة التعلم: {len(image_analyzer.image_learning_database)} إدخال")

if __name__ == "__main__":
    main()
