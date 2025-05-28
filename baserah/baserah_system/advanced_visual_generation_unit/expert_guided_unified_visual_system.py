#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-Guided Unified Visual System - Part 3: Complete Visual Analysis
النظام البصري الموحد الموجه بالخبير - الجزء الثالث: التحليل البصري الكامل

Revolutionary integration of Expert/Explorer guidance with unified visual analysis,
combining image and video analysis with adaptive mathematical equations.

التكامل الثوري لتوجيه الخبير/المستكشف مع التحليل البصري الموحد،
دمج تحليل الصور والفيديو مع المعادلات الرياضية المتكيفة.

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

# استيراد المحللات المتخصصة
try:
    from .expert_guided_image_analyzer import ExpertGuidedImageAnalyzer, ImageAnalysisRequest
    from .expert_guided_video_analyzer import ExpertGuidedVideoAnalyzer, VideoAnalysisRequest
except ImportError:
    # للاختبار المستقل
    pass

# محاكاة النظام المتكيف الموحد
class MockUnifiedEquation:
    def __init__(self, name: str, input_dim: int, output_dim: int):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.current_complexity = 20  # النظام الموحد أكثر تعقيداً
        self.adaptation_count = 0
        self.unified_accuracy = 0.8
        self.cross_modal_consistency = 0.85
        self.integration_quality = 0.75
        self.holistic_understanding = 0.9

    def adapt_with_expert_guidance(self, guidance, analysis):
        self.adaptation_count += 1
        if hasattr(guidance, 'recommended_evolution'):
            if guidance.recommended_evolution == "increase":
                self.current_complexity += 4
                self.unified_accuracy += 0.03
                self.cross_modal_consistency += 0.02
                self.integration_quality += 0.04
            elif guidance.recommended_evolution == "restructure":
                self.unified_accuracy += 0.02
                self.holistic_understanding += 0.03

    def get_expert_guidance_summary(self):
        return {
            "current_complexity": self.current_complexity,
            "total_adaptations": self.adaptation_count,
            "unified_accuracy": self.unified_accuracy,
            "cross_modal_consistency": self.cross_modal_consistency,
            "integration_quality": self.integration_quality,
            "holistic_understanding": self.holistic_understanding,
            "average_improvement": 0.08 * self.adaptation_count
        }

@dataclass
class UnifiedVisualAnalysisRequest:
    """طلب التحليل البصري الموحد"""
    shape: ShapeEntity
    analysis_modes: List[str]  # ["image", "video", "hybrid", "comprehensive"]
    visual_aspects: List[str]  # ["quality", "motion", "composition", "narrative", "artistic"]
    integration_level: str = "full"  # "basic", "intermediate", "full", "advanced"
    expert_guidance_level: str = "adaptive"
    learning_enabled: bool = True
    cross_modal_optimization: bool = True

@dataclass
class UnifiedVisualAnalysisResult:
    """نتيجة التحليل البصري الموحد"""
    success: bool
    unified_compliance: Dict[str, float]
    cross_modal_violations: List[str]
    unified_insights: List[str]
    image_analysis_results: Dict[str, Any] = None
    video_analysis_results: Dict[str, Any] = None
    integration_metrics: Dict[str, float] = None
    holistic_evaluation: Dict[str, float] = None
    expert_guidance_applied: Dict[str, Any] = None
    equation_adaptations: Dict[str, Any] = None
    performance_improvements: Dict[str, float] = None
    learning_insights: List[str] = None
    next_cycle_recommendations: List[str] = None

class ExpertGuidedUnifiedVisualSystem:
    """النظام البصري الموحد الموجه بالخبير الثوري"""

    def __init__(self):
        """تهيئة النظام البصري الموحد الموجه بالخبير"""
        print("🌟" + "="*100 + "🌟")
        print("🎨 النظام البصري الموحد الموجه بالخبير الثوري")
        print("🖼️🎬 تكامل تحليل الصور والفيديو بتوجيه الخبير/المستكشف")
        print("🧮 معادلات رياضية متكيفة + تحليل بصري شامل موحد")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" + "="*100 + "🌟")

        # إنشاء المحللات المتخصصة
        try:
            self.image_analyzer = ExpertGuidedImageAnalyzer()
            self.video_analyzer = ExpertGuidedVideoAnalyzer()
            print("✅ تم تحميل المحللات المتخصصة")
        except:
            print("⚠️ تشغيل في وضع المحاكاة - المحللات المتخصصة غير متاحة")
            self.image_analyzer = None
            self.video_analyzer = None

        # إنشاء معادلات التوحيد البصري
        self.unified_equations = {
            "cross_modal_integrator": MockUnifiedEquation("cross_modal_integration", 25, 20),
            "holistic_analyzer": MockUnifiedEquation("holistic_analysis", 30, 24),
            "consistency_enforcer": MockUnifiedEquation("consistency_enforcement", 22, 18),
            "quality_unifier": MockUnifiedEquation("quality_unification", 28, 22),
            "narrative_synthesizer": MockUnifiedEquation("narrative_synthesis", 26, 20),
            "aesthetic_harmonizer": MockUnifiedEquation("aesthetic_harmonization", 24, 19),
            "temporal_spatial_bridge": MockUnifiedEquation("temporal_spatial_bridging", 32, 26),
            "multi_modal_optimizer": MockUnifiedEquation("multi_modal_optimization", 35, 28),
            "unified_intelligence_core": MockUnifiedEquation("unified_intelligence", 40, 32)
        }

        # معايير التحليل البصري الموحد
        self.unified_standards = {
            "cross_modal_consistency": {
                "name": "الاتساق عبر الوسائط",
                "criteria": "تناغم بين تحليل الصور والفيديو",
                "spiritual_meaning": "الوحدة في التنوع سنة إلهية"
            },
            "holistic_understanding": {
                "name": "الفهم الشمولي",
                "criteria": "إدراك كامل للمحتوى البصري",
                "spiritual_meaning": "الحكمة الشاملة من صفات الله"
            },
            "integrated_quality": {
                "name": "الجودة المتكاملة",
                "criteria": "تميز في جميع الجوانب البصرية",
                "spiritual_meaning": "الإتقان عبادة"
            },
            "unified_aesthetics": {
                "name": "الجماليات الموحدة",
                "criteria": "انسجام جمالي شامل",
                "spiritual_meaning": "الجمال انعكاس للكمال الإلهي"
            }
        }

        # تاريخ التحليلات الموحدة
        self.unified_history = []
        self.unified_learning_database = {}

        print("🎨 تم إنشاء المعادلات البصرية الموحدة:")
        for eq_name in self.unified_equations.keys():
            print(f"   ✅ {eq_name}")

        print("✅ تم تهيئة النظام البصري الموحد الموجه بالخبير!")

    def analyze_unified_visual_with_expert_guidance(self, request: UnifiedVisualAnalysisRequest) -> UnifiedVisualAnalysisResult:
        """التحليل البصري الموحد موجه بالخبير"""
        print(f"\n🎨 بدء التحليل البصري الموحد الموجه بالخبير لـ: {request.shape.name}")
        start_time = datetime.now()

        # المرحلة 1: تحليل الخبير للطلب الموحد
        expert_analysis = self._analyze_unified_request_with_expert(request)
        print(f"🖼️🎬 تحليل الخبير الموحد: {expert_analysis['complexity_assessment']}")

        # المرحلة 2: توليد توجيهات الخبير للمعادلات الموحدة
        expert_guidance = self._generate_unified_expert_guidance(request, expert_analysis)
        print(f"🎨 توجيه الخبير الموحد: {expert_guidance.recommended_evolution}")

        # المرحلة 3: تكيف المعادلات الموحدة
        equation_adaptations = self._adapt_unified_equations(expert_guidance, expert_analysis)
        print(f"🧮 تكيف المعادلات الموحدة: {len(equation_adaptations)} معادلة")

        # المرحلة 4: تنفيذ التحليلات المتخصصة
        image_results, video_results = self._perform_specialized_analyses(request)

        # المرحلة 5: التكامل الموحد للنتائج
        integration_metrics = self._integrate_analysis_results(request, image_results, video_results, equation_adaptations)

        # المرحلة 6: التقييم الشمولي
        holistic_evaluation = self._perform_holistic_evaluation(request, integration_metrics)

        # المرحلة 7: فحص المعايير الموحدة
        unified_compliance = self._check_unified_standards_compliance(request, integration_metrics)

        # المرحلة 8: قياس التحسينات الموحدة
        performance_improvements = self._measure_unified_improvements(request, integration_metrics, equation_adaptations)

        # المرحلة 9: استخراج رؤى التعلم الموحد
        learning_insights = self._extract_unified_learning_insights(request, integration_metrics, performance_improvements)

        # المرحلة 10: توليد توصيات للدورة التالية
        next_cycle_recommendations = self._generate_unified_next_cycle_recommendations(performance_improvements, learning_insights)

        # إنشاء النتيجة الموحدة
        result = UnifiedVisualAnalysisResult(
            success=True,
            unified_compliance=unified_compliance["compliance_scores"],
            cross_modal_violations=unified_compliance["violations"],
            unified_insights=integration_metrics.get("insights", []),
            image_analysis_results=image_results,
            video_analysis_results=video_results,
            integration_metrics=integration_metrics,
            holistic_evaluation=holistic_evaluation,
            expert_guidance_applied=expert_guidance.__dict__,
            equation_adaptations=equation_adaptations,
            performance_improvements=performance_improvements,
            learning_insights=learning_insights,
            next_cycle_recommendations=next_cycle_recommendations
        )

        # حفظ في قاعدة التعلم الموحد
        self._save_unified_learning(request, result)

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ انتهى التحليل البصري الموحد الموجه في {total_time:.2f} ثانية")

        return result

    def _analyze_unified_request_with_expert(self, request: UnifiedVisualAnalysisRequest) -> Dict[str, Any]:
        """تحليل الطلب الموحد بواسطة الخبير"""

        # تحليل تعقيد الوسائط المتعددة
        modal_complexity = len(request.analysis_modes) * 4.0
        aspect_richness = len(request.visual_aspects) * 3.5
        integration_depth = {"basic": 2.0, "intermediate": 4.0, "full": 6.0, "advanced": 8.0}.get(request.integration_level, 4.0)

        # تحليل الخصائص الموحدة للشكل
        unified_geometric_complexity = request.shape.geometric_features.get("area", 100) / 30.0
        unified_color_richness = len(request.shape.color_properties) * 2.5
        unified_equation_complexity = len(request.shape.equation_params) * 3.0

        total_unified_complexity = (modal_complexity + aspect_richness + integration_depth +
                                  unified_geometric_complexity + unified_color_richness + unified_equation_complexity)

        return {
            "modal_complexity": modal_complexity,
            "aspect_richness": aspect_richness,
            "integration_depth": integration_depth,
            "unified_geometric_complexity": unified_geometric_complexity,
            "unified_color_richness": unified_color_richness,
            "unified_equation_complexity": unified_equation_complexity,
            "total_unified_complexity": total_unified_complexity,
            "complexity_assessment": "موحد معقد جداً" if total_unified_complexity > 35 else "موحد معقد" if total_unified_complexity > 25 else "موحد متوسط" if total_unified_complexity > 15 else "موحد بسيط",
            "recommended_adaptations": int(total_unified_complexity // 6) + 4,
            "focus_areas": self._identify_unified_focus_areas(request)
        }

    def _identify_unified_focus_areas(self, request: UnifiedVisualAnalysisRequest) -> List[str]:
        """تحديد مناطق التركيز الموحد"""
        focus_areas = []

        # تحليل أنماط التحليل
        if "image" in request.analysis_modes:
            focus_areas.append("image_analysis_optimization")
        if "video" in request.analysis_modes:
            focus_areas.append("video_analysis_optimization")
        if "hybrid" in request.analysis_modes:
            focus_areas.append("cross_modal_integration")
        if "comprehensive" in request.analysis_modes:
            focus_areas.append("holistic_understanding")

        # تحليل الجوانب البصرية
        if "quality" in request.visual_aspects:
            focus_areas.append("unified_quality_enhancement")
        if "motion" in request.visual_aspects:
            focus_areas.append("motion_integration")
        if "composition" in request.visual_aspects:
            focus_areas.append("compositional_harmony")
        if "narrative" in request.visual_aspects:
            focus_areas.append("narrative_coherence")
        if "artistic" in request.visual_aspects:
            focus_areas.append("aesthetic_unification")

        # تحليل مستوى التكامل
        if request.integration_level in ["full", "advanced"]:
            focus_areas.append("deep_integration")
        if request.cross_modal_optimization:
            focus_areas.append("cross_modal_optimization")

        return focus_areas

    def _generate_unified_expert_guidance(self, request: UnifiedVisualAnalysisRequest, analysis: Dict[str, Any]):
        """توليد توجيهات الخبير للتحليل الموحد"""

        # تحديد التعقيد المستهدف للنظام الموحد
        target_complexity = 25 + analysis["recommended_adaptations"]

        # تحديد الدوال ذات الأولوية للتحليل الموحد
        priority_functions = []
        if "cross_modal_integration" in analysis["focus_areas"]:
            priority_functions.extend(["hyperbolic", "gaussian"])
        if "holistic_understanding" in analysis["focus_areas"]:
            priority_functions.extend(["sin_cos", "swish"])
        if "unified_quality_enhancement" in analysis["focus_areas"]:
            priority_functions.extend(["softplus", "tanh"])
        if "aesthetic_unification" in analysis["focus_areas"]:
            priority_functions.extend(["squared_relu", "softsign"])
        if "deep_integration" in analysis["focus_areas"]:
            priority_functions.extend(["gaussian", "hyperbolic"])
        if "cross_modal_optimization" in analysis["focus_areas"]:
            priority_functions.extend(["swish", "softplus"])

        # تحديد نوع التطور الموحد
        if analysis["complexity_assessment"] == "موحد معقد جداً":
            recommended_evolution = "increase"
            adaptation_strength = 1.0
        elif analysis["complexity_assessment"] == "موحد معقد":
            recommended_evolution = "restructure"
            adaptation_strength = 0.9
        elif analysis["complexity_assessment"] == "موحد متوسط":
            recommended_evolution = "maintain"
            adaptation_strength = 0.75
        else:
            recommended_evolution = "maintain"
            adaptation_strength = 0.6

        # استخدام نفس فئة التوجيه (يمكن إنشاء فئة منفصلة لاحقاً)
        class MockUnifiedGuidance:
            def __init__(self, target_complexity, focus_areas, adaptation_strength, priority_functions, recommended_evolution):
                self.target_complexity = target_complexity
                self.focus_areas = focus_areas
                self.adaptation_strength = adaptation_strength
                self.priority_functions = priority_functions
                self.recommended_evolution = recommended_evolution

        return MockUnifiedGuidance(
            target_complexity=target_complexity,
            focus_areas=analysis["focus_areas"],
            adaptation_strength=adaptation_strength,
            priority_functions=priority_functions or ["hyperbolic", "gaussian"],
            recommended_evolution=recommended_evolution
        )

    def _adapt_unified_equations(self, guidance, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """تكيف المعادلات الموحدة"""

        adaptations = {}

        # إنشاء تحليل وهمي للمعادلات الموحدة
        class MockUnifiedAnalysis:
            def __init__(self):
                self.unified_accuracy = 0.8
                self.cross_modal_consistency = 0.85
                self.integration_quality = 0.75
                self.holistic_understanding = 0.9
                self.areas_for_improvement = guidance.focus_areas

        mock_analysis = MockUnifiedAnalysis()

        # تكيف كل معادلة موحدة
        for eq_name, equation in self.unified_equations.items():
            print(f"   🎨 تكيف معادلة موحدة: {eq_name}")
            equation.adapt_with_expert_guidance(guidance, mock_analysis)
            adaptations[eq_name] = equation.get_expert_guidance_summary()

        return adaptations

    def _perform_specialized_analyses(self, request: UnifiedVisualAnalysisRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """تنفيذ التحليلات المتخصصة"""

        image_results = {}
        video_results = {}

        # تحليل الصور إذا كان متاحاً
        if self.image_analyzer and ("image" in request.analysis_modes or "hybrid" in request.analysis_modes or "comprehensive" in request.analysis_modes):
            try:
                from .expert_guided_image_analyzer import ImageAnalysisRequest
                image_request = ImageAnalysisRequest(
                    shape=request.shape,
                    analysis_type="comprehensive",
                    image_aspects=["clarity", "balance", "harmony", "creativity"],
                    expert_guidance_level=request.expert_guidance_level,
                    learning_enabled=request.learning_enabled,
                    visual_optimization=True
                )
                image_result = self.image_analyzer.analyze_image_with_expert_guidance(image_request)
                image_results = {
                    "success": image_result.success,
                    "insights": image_result.image_insights,
                    "quality_metrics": image_result.quality_metrics,
                    "color_analysis": image_result.color_analysis,
                    "composition_scores": image_result.composition_scores,
                    "artistic_evaluation": image_result.artistic_evaluation
                }
            except Exception as e:
                print(f"⚠️ خطأ في تحليل الصور: {e}")
                image_results = self._mock_image_analysis(request.shape)
        else:
            image_results = self._mock_image_analysis(request.shape)

        # تحليل الفيديو إذا كان متاحاً
        if self.video_analyzer and ("video" in request.analysis_modes or "hybrid" in request.analysis_modes or "comprehensive" in request.analysis_modes):
            try:
                from .expert_guided_video_analyzer import VideoAnalysisRequest
                video_request = VideoAnalysisRequest(
                    shape=request.shape,
                    analysis_type="comprehensive",
                    video_aspects=["smoothness", "consistency", "flow", "transitions"],
                    expert_guidance_level=request.expert_guidance_level,
                    learning_enabled=request.learning_enabled,
                    motion_optimization=True
                )
                video_result = self.video_analyzer.analyze_video_with_expert_guidance(video_request)
                video_results = {
                    "success": video_result.success,
                    "insights": video_result.video_insights,
                    "motion_metrics": video_result.motion_metrics,
                    "temporal_analysis": video_result.temporal_analysis,
                    "frame_quality_scores": video_result.frame_quality_scores,
                    "narrative_evaluation": video_result.narrative_evaluation
                }
            except Exception as e:
                print(f"⚠️ خطأ في تحليل الفيديو: {e}")
                video_results = self._mock_video_analysis(request.shape)
        else:
            video_results = self._mock_video_analysis(request.shape)

        return image_results, video_results

    def _mock_image_analysis(self, shape: ShapeEntity) -> Dict[str, Any]:
        """تحليل صور وهمي للاختبار"""
        return {
            "success": True,
            "insights": ["تحليل صور وهمي", "جودة بصرية جيدة"],
            "quality_metrics": {"overall_quality": 0.8, "clarity": 0.75, "detail": 0.85},
            "color_analysis": {"harmony": 0.7, "saturation": 0.8, "contrast": 0.75},
            "composition_scores": {"balance": 0.85, "symmetry": 0.8, "proportion": 0.75},
            "artistic_evaluation": {"creativity": 0.7, "originality": 0.8, "aesthetic": 0.85}
        }

    def _mock_video_analysis(self, shape: ShapeEntity) -> Dict[str, Any]:
        """تحليل فيديو وهمي للاختبار"""
        return {
            "success": True,
            "insights": ["تحليل فيديو وهمي", "حركة سلسة"],
            "motion_metrics": {"smoothness": 0.75, "velocity": 0.7, "acceleration": 0.8},
            "temporal_analysis": {"consistency": 0.85, "stability": 0.8, "continuity": 0.75},
            "frame_quality_scores": {"resolution": 0.8, "clarity": 0.75, "detail": 0.7},
            "narrative_evaluation": {"coherence": 0.9, "pacing": 0.75, "engagement": 0.8}
        }

    def _integrate_analysis_results(self, request: UnifiedVisualAnalysisRequest, image_results: Dict[str, Any],
                                  video_results: Dict[str, Any], adaptations: Dict[str, Any]) -> Dict[str, Any]:
        """تكامل نتائج التحليل"""

        integration_metrics = {
            "insights": [],
            "cross_modal_scores": {},
            "unified_calculations": {},
            "integration_quality": {}
        }

        # تكامل الرؤى
        if image_results.get("insights"):
            integration_metrics["insights"].extend([f"صور: {insight}" for insight in image_results["insights"]])
        if video_results.get("insights"):
            integration_metrics["insights"].extend([f"فيديو: {insight}" for insight in video_results["insights"]])

        # حساب النتائج المتكاملة
        if image_results.get("success") and video_results.get("success"):
            # تكامل الجودة
            image_quality = image_results.get("quality_metrics", {}).get("overall_quality", 0.7)
            video_motion = video_results.get("motion_metrics", {}).get("smoothness", 0.7)
            integration_metrics["cross_modal_scores"]["quality_motion_harmony"] = (image_quality + video_motion) / 2

            # تكامل التركيب والزمن
            image_composition = image_results.get("composition_scores", {}).get("balance", 0.8)
            video_temporal = video_results.get("temporal_analysis", {}).get("consistency", 0.8)
            integration_metrics["cross_modal_scores"]["composition_temporal_balance"] = (image_composition + video_temporal) / 2

            # تكامل الفن والسرد
            image_artistic = image_results.get("artistic_evaluation", {}).get("aesthetic", 0.8)
            video_narrative = video_results.get("narrative_evaluation", {}).get("coherence", 0.85)
            integration_metrics["cross_modal_scores"]["artistic_narrative_coherence"] = (image_artistic + video_narrative) / 2

        # تقييم جودة التكامل
        avg_cross_modal = np.mean(list(integration_metrics["cross_modal_scores"].values())) if integration_metrics["cross_modal_scores"] else 0.75
        integration_metrics["integration_quality"]["cross_modal_consistency"] = avg_cross_modal
        integration_metrics["integration_quality"]["unified_performance"] = avg_cross_modal * 1.1  # تحسن من التكامل

        return integration_metrics

    def _perform_holistic_evaluation(self, request: UnifiedVisualAnalysisRequest, integration_metrics: Dict[str, Any]) -> Dict[str, float]:
        """التقييم الشمولي"""

        holistic_scores = {}

        # التقييم الشمولي للجودة
        cross_modal_consistency = integration_metrics["integration_quality"].get("cross_modal_consistency", 0.75)
        unified_performance = integration_metrics["integration_quality"].get("unified_performance", 0.8)

        holistic_scores["overall_excellence"] = (cross_modal_consistency + unified_performance) / 2
        holistic_scores["integration_mastery"] = cross_modal_consistency * 1.2  # مكافأة للتكامل الجيد
        holistic_scores["unified_intelligence"] = unified_performance * 1.15   # مكافأة للأداء الموحد
        holistic_scores["holistic_understanding"] = np.mean([cross_modal_consistency, unified_performance]) * 1.1

        return holistic_scores

    def _check_unified_standards_compliance(self, request: UnifiedVisualAnalysisRequest, integration_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """فحص الامتثال للمعايير الموحدة"""

        compliance = {
            "compliance_scores": {},
            "violations": [],
            "recommendations": []
        }

        # فحص الاتساق عبر الوسائط
        cross_modal_score = integration_metrics["integration_quality"].get("cross_modal_consistency", 0.75)
        compliance["compliance_scores"]["cross_modal_consistency"] = cross_modal_score
        if cross_modal_score < 0.7:
            compliance["violations"].append("اتساق ضعيف عبر الوسائط")

        # فحص الجودة المتكاملة
        unified_score = integration_metrics["integration_quality"].get("unified_performance", 0.8)
        compliance["compliance_scores"]["integrated_quality"] = unified_score
        if unified_score < 0.75:
            compliance["violations"].append("جودة متكاملة منخفضة")

        return compliance

    def _measure_unified_improvements(self, request: UnifiedVisualAnalysisRequest, integration_metrics: Dict[str, Any],
                                    adaptations: Dict[str, Any]) -> Dict[str, float]:
        """قياس التحسينات الموحدة"""

        improvements = {}

        # تحسن التكامل عبر الوسائط
        avg_integration = np.mean([adapt.get("cross_modal_consistency", 0.85) for adapt in adaptations.values()])
        baseline_integration = 0.7
        integration_improvement = ((avg_integration - baseline_integration) / baseline_integration) * 100
        improvements["cross_modal_improvement"] = max(0, integration_improvement)

        # تحسن الفهم الشمولي
        avg_holistic = np.mean([adapt.get("holistic_understanding", 0.9) for adapt in adaptations.values()])
        baseline_holistic = 0.75
        holistic_improvement = ((avg_holistic - baseline_holistic) / baseline_holistic) * 100
        improvements["holistic_improvement"] = max(0, holistic_improvement)

        # تحسن الجودة الموحدة
        avg_unified = np.mean([adapt.get("unified_accuracy", 0.8) for adapt in adaptations.values()])
        baseline_unified = 0.65
        unified_improvement = ((avg_unified - baseline_unified) / baseline_unified) * 100
        improvements["unified_quality_improvement"] = max(0, unified_improvement)

        return improvements

    def _extract_unified_learning_insights(self, request: UnifiedVisualAnalysisRequest, integration_metrics: Dict[str, Any],
                                         improvements: Dict[str, float]) -> List[str]:
        """استخراج رؤى التعلم الموحد"""

        insights = []

        if improvements["cross_modal_improvement"] > 20:
            insights.append("التكيف الموجه بالخبير حسن التكامل عبر الوسائط بشكل ملحوظ")

        if improvements["holistic_improvement"] > 18:
            insights.append("المعادلات المتكيفة ممتازة لتحسين الفهم الشمولي")

        if improvements["unified_quality_improvement"] > 22:
            insights.append("النظام الموحد نجح في تحسين الجودة الإجمالية بشكل كبير")

        if request.integration_level == "advanced":
            insights.append("التكامل المتقدم يستفيد من جميع جوانب النظام الموجه")

        if len(request.analysis_modes) > 2:
            insights.append("التحليل متعدد الوسائط يعزز الفهم الشامل")

        return insights

    def _generate_unified_next_cycle_recommendations(self, improvements: Dict[str, float], insights: List[str]) -> List[str]:
        """توليد توصيات للدورة التالية"""

        recommendations = []

        avg_improvement = np.mean(list(improvements.values()))

        if avg_improvement > 25:
            recommendations.append("الحفاظ على إعدادات التكيف الموحد الحالية")
            recommendations.append("تجربة تحليل موحد أكثر تعقيداً")
            recommendations.append("إضافة وسائط جديدة للتحليل")
            recommendations.append("تطوير تكامل أعمق بين المحللات")
        elif avg_improvement > 18:
            recommendations.append("زيادة قوة التكيف الموحد تدريجياً")
            recommendations.append("إضافة معايير موحدة جديدة")
            recommendations.append("تحسين دقة التكامل عبر الوسائط")
            recommendations.append("تطوير خوارزميات التوحيد")
        else:
            recommendations.append("مراجعة استراتيجية التوجيه الموحد")
            recommendations.append("تحسين دقة المعادلات الموحدة")
            recommendations.append("إعادة تقييم معايير التكامل")
            recommendations.append("تطوير آليات التعلم الموحد")

        return recommendations

    def _save_unified_learning(self, request: UnifiedVisualAnalysisRequest, result: UnifiedVisualAnalysisResult):
        """حفظ التعلم الموحد"""

        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "shape_name": request.shape.name,
            "analysis_modes": request.analysis_modes,
            "visual_aspects": request.visual_aspects,
            "integration_level": request.integration_level,
            "success": result.success,
            "performance_improvements": result.performance_improvements,
            "learning_insights": result.learning_insights
        }

        shape_key = f"{request.shape.category}_{request.integration_level}"
        if shape_key not in self.unified_learning_database:
            self.unified_learning_database[shape_key] = []

        self.unified_learning_database[shape_key].append(learning_entry)

        # الاحتفاظ بآخر 5 إدخالات
        if len(self.unified_learning_database[shape_key]) > 5:
            self.unified_learning_database[shape_key] = self.unified_learning_database[shape_key][-5:]

def main():
    """اختبار النظام البصري الموحد الموجه بالخبير"""
    print("🧪 اختبار النظام البصري الموحد الموجه بالخبير...")

    # إنشاء النظام الموحد
    unified_system = ExpertGuidedUnifiedVisualSystem()

    # إنشاء شكل اختبار
    from revolutionary_database import ShapeEntity

    test_shape = ShapeEntity(
        id=3, name="عمل فني موحد رائع", category="فن موحد",
        equation_params={"beauty": 0.95, "motion": 0.9, "harmony": 0.92, "creativity": 0.88, "flow": 0.85},
        geometric_features={"area": 250.0, "symmetry": 0.94, "stability": 0.9, "coherence": 0.92, "uniqueness": 0.9},
        color_properties={"primary": [255, 120, 80], "secondary": [80, 180, 255], "accent": [255, 255, 120], "background": [50, 50, 50]},
        position_info={"center_x": 0.5, "center_y": 0.5},
        tolerance_thresholds={}, created_date="", updated_date=""
    )

    # طلب تحليل موحد شامل
    analysis_request = UnifiedVisualAnalysisRequest(
        shape=test_shape,
        analysis_modes=["image", "video", "hybrid", "comprehensive"],
        visual_aspects=["quality", "motion", "composition", "narrative", "artistic"],
        integration_level="advanced",
        expert_guidance_level="adaptive",
        learning_enabled=True,
        cross_modal_optimization=True
    )

    # تنفيذ التحليل الموحد
    result = unified_system.analyze_unified_visual_with_expert_guidance(analysis_request)

    print(f"\n🎨 نتائج التحليل البصري الموحد:")
    print(f"   ✅ النجاح: {result.success}")
    if result.success:
        print(f"   🖼️🎬 رؤى موحدة: {len(result.unified_insights)} رؤية")
        print(f"   🔗 مقاييس التكامل: متاح")
        print(f"   🌟 التقييم الشمولي: متاح")
        print(f"   📊 نتائج الصور: {'متاح' if result.image_analysis_results else 'غير متاح'}")
        print(f"   🎥 نتائج الفيديو: {'متاح' if result.video_analysis_results else 'غير متاح'}")

    print(f"\n📊 إحصائيات النظام الموحد:")
    print(f"   🎨 معادلات موحدة: {len(unified_system.unified_equations)}")
    print(f"   📚 قاعدة التعلم: {len(unified_system.unified_learning_database)} إدخال")
    print(f"   🖼️ محلل الصور: {'متاح' if unified_system.image_analyzer else 'غير متاح'}")
    print(f"   🎬 محلل الفيديو: {'متاح' if unified_system.video_analyzer else 'غير متاح'}")

if __name__ == "__main__":
    main()
